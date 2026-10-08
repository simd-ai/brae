// The DEVICE boundary gradient of U on a SYMMETRY PLANE, against the host reference.
//
// WHAT IS UNDER TEST. fvc::grad(U) is built with extrapolatedCalculated patches, but fvPatchField::New
// puts a constraint patch's own type in their place, keyed on the MESH patch type
// (fvPatchFieldNew.C:57-64), and gaussGrad ends in correctBoundaryConditions (gaussGrad.C:106) -- so on
// a `symmetry` or `symmetryPlane` patch the gradient's value is the cell gradient averaged with its
// mirror image, (G + R G R^T)/2 with R = I - 2 nn (basicSymmetryFvPatchField::evaluate), before the
// normal correction. The host has carried this since RAS/damBreakLeakage needed it (fvc.cu,
// gradUBoundary: HbyA 3% out beside the baffle without it, against OpenFOAM's dumped HbyA); the device
// kernel (device_divdevreff.cu, gradBKernel) did not.
//
// THE ORACLE is the host's fvc::gradUBoundary, which tests/interfoam_leakage_vs_openfoam.sh holds to
// OpenFOAM. Two fixtures, one per branch: validation/windAroundBuildingsBox carries a `symmetry` patch
// (the face's own normal), validation/halfChannel a `symmetryPlane` (the patch's one normal).
// MEASURED: device against the host 2.4e-14 of 9.7 and 2.3e-13 of 1.2e+03; every other patch 3e-15.
// BROKEN ONCE, the mirror taken out of the kernel: 3.0e-01 of 9.7 and 1.0e-01 of 1.2e+03, both arms
// red. The in-test sanity arm is weaker than that and says so: it only asserts that the boundary
// gradient differs visibly from the raw cell gradient there, so the faces are not idle.
//
// NOT CLAIMED: a `slip` wall is NOT mirrored -- OpenFOAM keys the constraint on the mesh patch type and
// a slip wall is a `wall` -- and the last arm asserts the device leaves such faces alone.
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
#include "device_kepsilon.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf("  %s %s\n", ok ? "ok:  " : "FAIL:", what);
    if (!ok) ++failures;
}

int runCase(
    const std::string& caseDir,
    const std::string& wantType)
{
    std::printf("== %s (a `%s` patch) ==\n", caseDir.c_str(), wantType.c_str());
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // a smooth field with every component of the gradient live, so no term of R G R^T is idle
    GeometricField<vector> U;
    U.internal.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& x = g.C()[c];
        U.internal[c] = vector{std::sin(0.7*x.x + 0.3*x.y) + 0.2*x.z,
                               std::cos(0.5*x.y - 0.4*x.z) + 0.1*x.x,
                               std::sin(0.6*x.z + 0.2*x.x) - 0.3*x.y};
    }
    bool haveWanted = false;
    for (const FvPatch& q : fvp)
    {
        if (q.type == "empty")
        {
            U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        }
        else if (q.type == "symmetry" || q.type == "symmetryPlane")
        {
            U.boundary.push_back(std::make_unique<SymmetryPlanePatchField<vector>>(q));
            haveWanted = haveWanted || q.type == wantType;
        }
        else
        {
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
        }
    }
    check("the fixture carries the patch type this arm is for", haveWanted);
    U.evaluateBoundary();

    // THE ORACLE, and the control: the same correction on the UNMIRRORED cell gradient
    const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, fvp);
    const std::vector<std::vector<tensor>> gbHost = fvc::gradUBoundary(U, gradC, m, g, fvp);

    // THE DEVICE
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

    scalar worst = 0, scale = 0, control = 0, offPlane = 0;
    std::size_t bi = 0, nFaces = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& q = fvp[pi];
        if (isCoupledInterfaceType(q.type)) continue;
        const bool plane = (q.type == "symmetry" || q.type == "symmetryPlane");
        for (label i = 0; i < q.size; ++i, ++bi)
        {
            if (q.type == "empty") continue;
            const tensor& h = gbHost[pi][static_cast<std::size_t>(i)];
            const scalar hq[9] = {h.xx, h.xy, h.xz, h.yx, h.yy, h.yz, h.zx, h.zy, h.zz};
            // the unmirrored form: gc + n (x) (sn - n & gc), with the host's own snGrad
            const tensor& gc = gradC[static_cast<std::size_t>(q.faceCells[i])];
            const scalar cq[9] = {gc.xx, gc.xy, gc.xz, gc.yx, gc.yy, gc.yz, gc.zx, gc.zy, gc.zz};
            scalar dWorst = 0, cWorst = 0;
            for (int k = 0; k < 9; ++k)
            {
                dWorst = std::fmax(dWorst, std::fabs(gb[static_cast<std::size_t>(k)*nB + bi] - hq[k]));
                cWorst = std::fmax(cWorst, std::fabs(cq[k] - hq[k]));
                scale = std::fmax(scale, std::fabs(hq[k]));
            }
            if (plane)
            {
                worst = std::fmax(worst, dWorst);
                control = std::fmax(control, cWorst);
                ++nFaces;
            }
            else
            {
                offPlane = std::fmax(offPlane, dWorst);
            }
        }
    }
    std::printf("  %zu symmetry-plane faces: device against the host %.4e of %.4e\n", nFaces,
                (double)worst, (double)scale);
    std::printf("  the raw cell gradient against the host there:      %.4e   [the faces are not idle]\n",
                (double)control);
    std::printf("  every OTHER patch, device against the host:        %.4e\n", (double)offPlane);
    check("the device's boundary gradient on the symmetry plane is the host's", worst <= scalar(1e-13)*scale);
    check("...and the boundary treatment is a visible fraction of it there, so the faces are not idle",
          control >= scalar(1e-3)*scale);
    check("...and no other patch moved", offPlane <= scalar(1e-13)*scale);
    return 0;
}

}   // namespace

int main(int argc, char** argv)
{
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev <= 0)
    {
        cudaGetLastError();
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    const std::string root = argc > 1 ? argv[1] : "validation";
    runCase(root + "/windAroundBuildingsBox", "symmetry");
    runCase(root + "/halfChannel", "symmetryPlane");
    std::printf("test_device_grad_symmetry: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
