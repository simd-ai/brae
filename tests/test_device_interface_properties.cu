// Interface normal and curvature: device against the HOST reference, on a real mesh.
//
// The host side is itself gated against OpenFOAM's UNMODIFIED interfaceProperties to 2.5e-14 at four
// calculateK pass counts (tests/interfoam_curvature_vs_openfoam.sh), so agreement here is agreement
// with OpenFOAM one link along.
//
// WHY THIS ONE CANNOT BE BIT-FOR-BIT, unlike the mixture. K = -div(nHatf) is a REDUCTION: every cell
// sums its faces, and the device sums them in a different order from the host's serial loop. Floating
// point addition is not associative, so a difference is guaranteed and the only question is how big.
// The bound below is therefore in ULP of the ACCUMULATED value rather than of a single operation, and
// it is measured rather than assumed.
//
// The per-face part has no reduction and IS asserted tightly: nHatf differs only by the fused
// multiply-add the two compilers pick differently (see device_two_phase_mixture.cu for that story).
#include "box_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "interface_properties_cpp.cuh"
#include "device_interface_properties.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
double ulps(scalar a, scalar b)
{
    if (a == b) return 0.0;
    const double d = std::fabs((double)a - (double)b);
    const double m = std::fmax(std::fabs((double)a), std::fabs((double)b));
    return d / std::max(std::nextafter(m, 1e300) - m, 1e-300);
}
}   // namespace

int main()
{
    std::printf("== interface normal and curvature: device vs host ==\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    // A 3D box with a SPHERICAL interface -- the same fixture the host curvature gate uses, where K is
    // known to be 2/R, so a device error shows up against a value that means something.
    const label N = 16;
    const scalar h = scalar(12) / scalar(N);
    PrimitiveMesh m = boxtest::boxMesh(N, N, N, scalar(0), h, h, h);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    const vector centre{scalar(6), scalar(6), scalar(6)};
    const scalar R = scalar(3), width = scalar(1.2);
    GeometricField<scalar> a;
    a.internal.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::sqrt((C.x-centre.x)*(C.x-centre.x) + (C.y-centre.y)*(C.y-centre.y)
                                 + (C.z-centre.z)*(C.z-centre.z));
        a.internal[c] = scalar(0.5)*(scalar(1) - std::tanh((r - R)/width));
    }
    for (const FvPatch& q : fvp)
        a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    a.evaluateBoundary();

    // --- host ---
    brae::cpu::interfaceProps::InterfaceCoeffs ic;
    ic.cAlpha = scalar(1);
    SurfaceScalarField hNHatf;
    std::vector<scalar> hK;
    brae::cpu::interfaceProps::calculateK(a, ic, m, g, fvp, /*gradLeastSquares=*/false, hNHatf, hK);
    const scalar dN = brae::cpu::interfaceProps::deltaN(g.V());

    // --- device ---
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<scalar> ab;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& v = a.boundary[pi]->value();
        ab.insert(ab.end(), v.begin(), v.end());
    }
    DeviceBuffer<scalar> dAlpha(a.internal), dAb(ab), dgx(nC), dgy(nC), dgz(nC);
    deviceGaussGrad(dm, dAlpha, dAb, dgx, dgy, dgz);

    DeviceBuffer<scalar> dNHatfInt(nIf), dNHatfBnd(nBf), dK(nC);
    deviceInterfaceNormalFlux(dm, (int)nIf, dgx, dgy, dgz, dN, dNHatfInt);

    // the boundary normals, corrected on the host (no contact angle here, so this is the plain
    // gradAlphaf/(mag+deltaN) -- the split the header explains)
    std::vector<scalar> hgx(nC), hgy(nC), hgz(nC);
    {
        const std::vector<vector> gc = fvc::gaussGrad(a, m, g, fvp);
        for (label c = 0; c < nC; ++c) { hgx[c]=gc[c].x; hgy[c]=gc[c].y; hgz[c]=gc[c].z; }
    }
    std::vector<scalar> nbx, nby, nbz;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& q = fvp[pi];
        const std::vector<scalar> snA = a.boundary[pi]->snGrad(a.internal);
        for (label i = 0; i < q.size; ++i)
        {
            const label ci = q.faceCells[i];
            const vector& n = q.nf[i];
            const scalar nn = n.x*hgx[ci] + n.y*hgy[ci] + n.z*hgz[ci];
            const scalar d  = snA[i] - nn;
            const vector gb{hgx[ci] + n.x*d, hgy[ci] + n.y*d, hgz[ci] + n.z*d};
            const scalar mg = std::sqrt(gb.x*gb.x + gb.y*gb.y + gb.z*gb.z);
            const scalar s  = scalar(1)/(mg + dN);
            nbx.push_back(gb.x*s); nby.push_back(gb.y*s); nbz.push_back(gb.z*s);
        }
    }
    DeviceBuffer<scalar> dnbx(nbx), dnby(nby), dnbz(nbz);
    deviceInterfaceNormalFluxBoundary(dm, (int)nBf, (int)nIf, dnbx, dnby, dnbz, dNHatfBnd);
    deviceInterfaceCurvature(dm, dNHatfInt, dNHatfBnd, dK);
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }

    std::vector<scalar> nInt, Kd;
    dNHatfInt.copyTo(nInt);
    failures += brae::gatecheck::nonFinite("nInt", nInt);
    dK.copyTo(Kd);
    failures += brae::gatecheck::nonFinite("Kd", Kd);

    // --- nHatf: per face, no reduction, so only the FMA choice separates them ---
    {
        // ULP IS THE WRONG YARDSTICK FOR THIS FIELD, and the first version of this arm used it and
        // reported 1.8e+16 ULP. The value it fired on was device -1.08e-16 against host +1.08e-16 --
        // the same magnitude with opposite signs, both of them zero to round-off. nHatf is
        // gradAlphaf/(mag + deltaN) dotted with Sf, and away from the interface gradAlphaf vanishes
        // while deltaN does not, so most faces carry pure cancellation noise at 1e-16. A relative
        // metric on a near-zero value measures the noise and nothing else.
        //
        // The yardstick is therefore the FIELD's own scale: nHatf is a unit normal dotted with an area,
        // so its meaningful size is max|nHatf| over the mesh, and that is what the difference is
        // measured against.
        scalar worstAbs = 0, ref = 0;
        label iw = -1;
        int nDiff = 0;
        for (label f = 0; f < nIf; ++f)
        {
            ref = std::fmax(ref, std::fabs(hNHatf.internal[f]));
            const scalar e = std::fabs(nInt[f] - hNHatf.internal[f]);
            if (nInt[f] != hNHatf.internal[f]) ++nDiff;
            if (e > worstAbs) { worstAbs = e; iw = f; }
        }
        std::printf("  nHatf: %d of %ld faces differ; worst %.4e of a field scale %.4e "
                    "(relative %.3e)\n",
                    nDiff, (long)nIf, (double)worstAbs, (double)ref, (double)(worstAbs/ref));
        if (iw >= 0)
            std::printf("    at face %ld: device %.17g, host %.17g\n",
                        (long)iw, (double)nInt[iw], (double)hNHatf.internal[iw]);
        // MEASURED 2.8e-15 of the field scale. The worst face is a sign flip on a 7.9e-16 value --
        // cancellation noise where gradAlphaf vanishes and deltaN is all that is left -- so the bound
        // is set a few times above that rather than at round-off for a single operation.
        check("nHatf agrees with the host to 1e-14 of the field -- per face, so only the FMA differs",
              worstAbs < scalar(1e-14) * ref);
    }

    // --- K: a reduction, so the summation ORDER differs by construction ---
    {
        double worstU = 0; scalar worstAbs = 0, ref = 0;
        for (label c = 0; c < nC; ++c)
        {
            worstU  = std::max(worstU, ulps(Kd[c], hK[c]));
            worstAbs = std::fmax(worstAbs, std::fabs(Kd[c] - hK[c]));
            ref      = std::fmax(ref, std::fabs(hK[c]));
        }
        std::printf("  K: worst %.4e absolute of %.4e (relative %.3e), %.1f ULP\n",
                    (double)worstAbs, (double)ref, (double)(worstAbs/ref), worstU);
        // The bound is relative and loose enough for a reordered sum of ~6 terms per cell, and tight
        // enough that it is still four orders below the curvature it is measuring.
        check("K agrees with the host to 1e-12 relative -- a reordered reduction, not a formula",
              worstAbs < scalar(1e-12) * ref);
    }

    // --- and the physical check, so agreement is agreement about something ---
    {
        scalar sum = 0; label n = 0;
        for (label c = 0; c < nC; ++c)
        {
            const vector& C = g.C()[c];
            const scalar r = std::sqrt((C.x-centre.x)*(C.x-centre.x) + (C.y-centre.y)*(C.y-centre.y)
                                     + (C.z-centre.z)*(C.z-centre.z));
            if (std::fabs(r - R) < scalar(0.5)*h) { sum += Kd[c]; ++n; }
        }
        const scalar Kavg = n ? sum/scalar(n) : scalar(0);
        std::printf("  device K on the interface: %.4f, exact 2/R = %.4f\n",
                    (double)Kavg, (double)(scalar(2)/R));
        check("the DEVICE curvature is 2/R on a sphere, to better than 5%",
              std::fabs(Kavg - scalar(2)/R) < scalar(0.05)*(scalar(2)/R));
    }

    std::printf("test_device_interface_properties: %d failures\n", failures);
    return failures ? 1 : 0;
}
