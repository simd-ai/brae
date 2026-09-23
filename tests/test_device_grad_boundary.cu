// gradBKernel's TYPE-CODE branches, against the host reference, one patch per branch.
//
// WHAT IS UNDER TEST. fvc::grad(U)'s BOUNDARY value is the extrapolated cell gradient with its normal
// row replaced: gc + n (x) (snGrad - n & gc) (gaussGrad.C:96-106, correctBoundaryConditions). The
// snGrad in that expression is THE PATCH'S OWN, and OpenFOAM overrides it per class -- zeroGradient
// returns zero, fixedGradient the prescribed gradient, fixedValue/calculated the base
// (value - internal)*deltaCoeffs, mixed the blend, and a COUPLED patch is skipped outright
// (`if (!coupled())`), keeping the cell gradient. device_divdevreff.cu's gradBKernel carries all of
// them as type codes, and until now only two of those branches had a fixture: symmetry
// (tests/test_device_grad_symmetry.cu) and the flux-conditional ones (waterChannel's `oneCorrector`
// profile). The rest were held only by whole-case gates, where a branch that is wrong by the
// difference between `rg` and zero can hide behind everything else in the step.
//
// THE ORACLE is the host's fvc::gradUBoundary, which tests/interfoam_leakage_vs_openfoam.sh and the
// naca gates hold to OpenFOAM.
//
// EVERY ARM CARRIES ITS OWN CONTROL, in the test rather than in a comment: for each patch the device's
// value is compared BOTH with the host's and with what the NEIGHBOURING branch would have produced.
// An arm passes only when it matches its own branch AND the other branch is visibly different, so a
// kernel that fell through to the wrong code cannot read green.
//
// MEASURED, device against host: fixedValue 4.5e-13 of 7.6e+03, fixedGradient 2.2e-16, zeroGradient
// 1.1e-16 -- and the controls 7.59e+03, 2.30, 2.30. BROKEN TWICE, out of tree (a defect copy of
// device_divdevreff.cu linked ahead of libbrae_core.a, so the tree was never touched):
//   * the type-0 branch made to return zero, i.e. fixedGradient falling to zeroGradient: that arm
//     reads 2.300e+00 and fails, the other two stay at 1e-13 and 1e-16;
//   * the type-1 branch made to return zero: fixedValue reads 7.590e+03 (1.000e+00 relative) and
//     fails alone.
// Each injection fails exactly the arm it breaks, which is what says the arms are separate.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_divdevreff.cuh"
#include "device_kepsilon.cuh"   // deviceGradU
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

namespace
{
int failures = 0;

void check(const char* what, bool ok)
{
    std::printf("  %s:   %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// the nine components of a tensor in the device's row-major order
void nine(const tensor& t, scalar out[9])
{
    out[0] = t.xx; out[1] = t.xy; out[2] = t.xz;
    out[3] = t.yx; out[4] = t.yy; out[5] = t.yz;
    out[6] = t.zx; out[7] = t.zy; out[8] = t.zz;
}
}   // namespace


int main(int argc, char** argv)
{
    std::printf("== gradBKernel: one patch per type-code branch, against the host reference ==\n");
    if (argc < 2)
    {
        std::printf("  SKIP: usage: %s <caseDir>\n", argv[0]);
        return 77;
    }
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev <= 0)
    {
        cudaGetLastError();
        std::printf("  SKIP: no CUDA device\n");
        return 77;
    }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // a field with every gradient component live, so no row of the correction is idle
    GeometricField<vector> U;
    U.internal.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& x = g.C()[c];
        U.internal[c] = vector{std::sin(0.7*x.x + 0.3*x.y) + 0.2*x.z,
                               std::cos(0.5*x.y - 0.4*x.z) + 0.1*x.x,
                               std::sin(0.6*x.z + 0.2*x.x) - 0.3*x.y};
    }

    // ONE BRANCH PER PATCH. The prescribed gradient is deliberately not small: `sn = rg` and the base
    // formula (ub - uc)*dc are the two branches most easily confused, and they only separate when the
    // gradient is not what the value difference would have given anyway.
    const vector fixedVal{0.37, -0.21, 0.58};
    const vector fixedGrad{1.7, -2.3, 0.9};
    std::string branchOf[64];
    std::size_t nPatch = fvp.size();
    for (std::size_t pi = 0; pi < nPatch; ++pi)
    {
        const FvPatch& q = fvp[pi];
        if (q.type == "empty")
        {
            U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
            branchOf[pi] = "empty";
        }
        else if (isCoupledInterfaceType(q.type))
        {
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
            branchOf[pi] = "coupled";
        }
        else if (branchOf[pi].empty() && pi % 3 == 0)
        {
            U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                q, /*uniform=*/true, fixedVal, std::vector<vector>{}));
            branchOf[pi] = "fixedValue";
        }
        else if (pi % 3 == 1)
        {
            U.boundary.push_back(std::make_unique<FixedGradientPatchField<vector>>(
                q, /*uniform=*/true, fixedGrad, std::vector<vector>{}));
            branchOf[pi] = "fixedGradient";
        }
        else
        {
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
            branchOf[pi] = "zeroGradient";
        }
    }
    U.evaluateBoundary();

    bool haveFV = false, haveFG = false, haveZG = false;
    for (std::size_t pi = 0; pi < nPatch; ++pi)
    {
        if (fvp[pi].size <= 0) continue;
        haveFV = haveFV || branchOf[pi] == "fixedValue";
        haveFG = haveFG || branchOf[pi] == "fixedGradient";
        haveZG = haveZG || branchOf[pi] == "zeroGradient";
    }
    check("the fixture has a fixedValue, a fixedGradient and a zeroGradient patch with faces",
          haveFV && haveFG && haveZG);

    const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, fvp);
    const std::vector<std::vector<tensor>> gbHost = fvc::gradUBoundary(U, gradC, m, g, fvp);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    std::vector<scalar> ux(static_cast<std::size_t>(nC)), uy(ux.size()), uz(ux.size());
    for (label c = 0; c < nC; ++c)
    {
        ux[c] = U.internal[c].x;
        uy[c] = U.internal[c].y;
        uz[c] = U.internal[c].z;
    }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dGradU, dGradB;
    deviceGradU(dm, dbU, dUx, dUy, dUz, dGradU);
    deviceBoundaryGradU(dm, dbU, dUx, dUy, dUz, dGradU, dGradB, nullptr);
    std::vector<scalar> gb;
    dGradB.copyTo(gb);
    const std::size_t nB = static_cast<std::size_t>(dm.nBndFaces);

    // per branch: the device against the host, and the device against the OTHER branch's formula
    struct Acc { scalar worst = 0, control = 0, scale = 0; std::size_t n = 0; };
    Acc fv, fg, zg;
    std::size_t bi = 0;
    for (std::size_t pi = 0; pi < nPatch; ++pi)
    {
        const FvPatch& q = fvp[pi];
        if (isCoupledInterfaceType(q.type)) continue;      // not in the device's boundary arrays
        for (label i = 0; i < q.size; ++i, ++bi)
        {
            if (q.type == "empty") continue;
            Acc* a = branchOf[pi] == "fixedValue"    ? &fv
                   : branchOf[pi] == "fixedGradient" ? &fg
                   : branchOf[pi] == "zeroGradient"  ? &zg : nullptr;
            if (!a) continue;
            const std::size_t k = static_cast<std::size_t>(i);
            scalar hq[9], cq[9];
            nine(gbHost[pi][k], hq);
            nine(gradC[static_cast<std::size_t>(q.faceCells[i])], cq);

            // the neighbouring branch: gc + n (x) (snOther - n & gc), with snOther the formula the
            // OTHER class would have used on this face
            const vector& Sf = g.Sf()[q.start + i];
            const scalar mag = std::sqrt(Sf.x*Sf.x + Sf.y*Sf.y + Sf.z*Sf.z);
            const vector n{Sf.x/mag, Sf.y/mag, Sf.z/mag};
            const vector uc = U.internal[static_cast<std::size_t>(q.faceCells[i])];
            const vector ub = U.boundary[pi]->value()[k];
            const scalar dc = q.deltaCoeffs[k];
            // THE NEIGHBOURING BRANCH, chosen so that it CAN differ. On a freshly evaluated
            // fixedGradient patch the base formula (value - internal)*deltaCoeffs reproduces the
            // gradient exactly -- evaluate() sets the value from it -- so controlling that branch
            // against the base formula is vacuous by construction, and it read 5.7e-14 when this
            // test first tried it. Its real neighbour in the kernel's `else` chain is the OTHER type
            // 0: snGrad = 0, the zeroGradient branch, which differs by the whole prescribed gradient.
            vector snOther;
            if (branchOf[pi] == "fixedGradient")
                snOther = vector{0, 0, 0};                                               // zeroGradient
            else
                snOther = fixedGrad;                                                     // a prescribed one
            const scalar nv[3] = {n.x, n.y, n.z};
            const scalar so[3] = {snOther.x, snOther.y, snOther.z};
            scalar other[9];
            for (int r = 0; r < 3; ++r)
                for (int cIdx = 0; cIdx < 3; ++cIdx)
                    other[r*3 + cIdx] = cq[r*3 + cIdx];
            scalar ngc[3];
            for (int j = 0; j < 3; ++j)
                ngc[j] = nv[0]*cq[0*3+j] + nv[1]*cq[1*3+j] + nv[2]*cq[2*3+j];
            for (int r = 0; r < 3; ++r)
                for (int j = 0; j < 3; ++j)
                    other[r*3 + j] += nv[r]*(so[j] - ngc[j]);

            for (int c9 = 0; c9 < 9; ++c9)
            {
                const scalar dev = gb[static_cast<std::size_t>(c9)*nB + bi];
                a->worst = std::fmax(a->worst, std::fabs(dev - hq[c9]));
                a->control = std::fmax(a->control, std::fabs(other[c9] - hq[c9]));
                a->scale = std::fmax(a->scale, std::fabs(hq[c9]));
            }
            ++a->n;
        }
    }

    const scalar bound = scalar(1e-12);
    struct Named { const char* name; Acc* a; };
    const Named arms[3] = {{"fixedValue  (type 1, (ub - uc)*deltaCoeffs)", &fv},
                           {"fixedGradient (type 0, snGrad = the prescribed gradient)", &fg},
                           {"zeroGradient  (type 0, snGrad = 0)", &zg}};
    for (const Named& nm : arms)
    {
        Acc& a = *nm.a;
        const scalar rel = a.scale > 0 ? a.worst/a.scale : a.worst;
        std::printf("  %-56s %zu faces, device vs host %.3e (rel %.3e), the other branch %.3e\n",
                    nm.name, a.n, (double)a.worst, (double)rel, (double)a.control);
        check(nm.name, a.n > 0 && a.worst < bound);
    }
    // the controls: each branch must be visibly different from its neighbour, or matching it proves
    // nothing. zeroGradient against a prescribed gradient is the widest, fixedValue against the base
    // formula the narrowest -- and all three are asserted, not printed.
    check("...and the fixedValue faces are NOT what a prescribed gradient would give",
          fv.control > scalar(1e-3)*std::fmax(fv.scale, scalar(1)));
    check("...the fixedGradient faces are NOT what snGrad = 0 would give",
          fg.control > scalar(1e-3)*std::fmax(fg.scale, scalar(1)));
    check("...and the zeroGradient faces are NOT what a prescribed gradient would give",
          zg.control > scalar(1e-3)*std::fmax(zg.scale, scalar(1)));

    std::printf("test_device_grad_boundary: %d failures\n", failures);
    return failures ? 1 : 0;
}
