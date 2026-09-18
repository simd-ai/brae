// MULES on the device, against the HOST limiter and against its own contract.
//
// TWO KINDS OF ARM, and the distinction is the point.
//
//   THE PROPERTY, asserted exactly: after the device limiter, alpha stays in [0,1]. That is what MULES
//   is FOR, it holds on any mesh and any flux, and it needs no host to compare against. A device
//   limiter that agreed with the host to 1e-15 and let alpha reach 1.0001 would be broken, and this
//   arm is the one that would say so.
//
//   THE COMPARISON, within a measured tolerance: lambda itself. It cannot be bit-identical -- each
//   cell's five gathers run in a different ORDER from the host's face loop and floating-point addition
//   is not associative -- and the difference then passes through a clamp01 and a min, which mostly
//   absorbs it but can flip a face sitting exactly on a bound. So the bound is on how many faces move
//   and by how much, measured.
//
// The host side is gated on boundedness against an unlimited control in tests/test_mules_cpp.cu, so
// agreement here is agreement with a limiter that has already been shown to limit.
#include "box_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "mules_cpp.cuh"
#include "device_mules.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
namespace mules = brae::cpu::MULES;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
}   // namespace

int main()
{
    std::printf("== MULES: device vs host ==\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    // A 2D box with a DIAGONAL flux and a circular blob -- the fixture the host gate found makes the
    // limiter iteration actually bite, rather than reach its fixed point on the first pass.
    const label N = 12;
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    const vector U{scalar(0.8), scalar(0.6), scalar(0)};
    SurfaceScalarField phi;
    phi.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const vector& S = g.Sf()[f];
        phi.internal[f] = U.x*S.x + U.y*S.y + U.z*S.z;
    }
    phi.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& q = fvp[pi];
        phi.boundary[pi].resize(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i)
        {
            const vector& S = g.Sf()[q.start + i];
            phi.boundary[pi][i] = U.x*S.x + U.y*S.y + U.z*S.z;
        }
    }

    GeometricField<scalar> a;
    a.internal.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::hypot(C.x - scalar(N)/2, C.y - scalar(N)/2);
        a.internal[c] = scalar(0.5)*(scalar(1) + std::tanh(scalar(N)/5 - r));
    }
    for (const FvPatch& q : fvp)
        a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    a.evaluateBoundary();

    // the high-order flux, central differencing
    SurfaceScalarField phiPsi;
    phiPsi.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const scalar w = g.weights()[f];
        phiPsi.internal[f] = phi.internal[f]
            * (w*a.internal[m.owner()[f]] + (scalar(1) - w)*a.internal[m.neighbour()[f]]);
    }
    phiPsi.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& av = a.boundary[pi]->value();
        phiPsi.boundary[pi].resize(av.size());
        for (std::size_t i = 0; i < av.size(); ++i)
            phiPsi.boundary[pi][i] = phi.boundary[pi][i]*av[i];
    }

    const scalar dt = scalar(0.2);
    mules::Controls ctl;
    ctl.nLimiterIter = 5;

    // --- host ---
    SurfaceScalarField hPhiBD;
    mules::boundedDonorFlux(phi, a, phiPsi, m, fvp, hPhiBD);
    SurfaceScalarField hCorr = phiPsi;
    for (label f = 0; f < nIf; ++f) hCorr.internal[f] -= hPhiBD.internal[f];
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (std::size_t i = 0; i < hCorr.boundary[pi].size(); ++i)
            hCorr.boundary[pi][i] -= hPhiBD.boundary[pi][i];
    mules::Limiter hLam;
    mules::Fields hf;
    mules::limiter(hLam, scalar(1)/dt, a, a.internal, hPhiBD, hCorr, hf, ctl, m, g, fvp);

    // --- device ---
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    auto flatten = [&](const std::vector<std::vector<scalar>>& b)
    {
        std::vector<scalar> v;
        for (const auto& p : b) v.insert(v.end(), p.begin(), p.end());
        return v;
    };
    std::vector<scalar> aBnd;
    std::vector<int>    fixes, flag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& v = a.boundary[pi]->value();
        aBnd.insert(aBnd.end(), v.begin(), v.end());
        const int fx = a.boundary[pi]->fixesValue() ? 1 : 0;
        const int fl = (fvp[pi].type == "empty") ? 1 : ((fvp[pi].type == "wedge") ? 2 : 0);
        for (label i = 0; i < fvp[pi].size; ++i) { fixes.push_back(fx); flag.push_back(fl); }
    }

    DeviceBuffer<scalar> dPhi(phi.internal), dPsi(a.internal), dPsiOld(a.internal),
                         dPsiBnd(aBnd), dPhiPsiBnd(flatten(phiPsi.boundary));
    DeviceBuffer<int>    dFixes(fixes), dFlag(flag);
    DeviceBuffer<scalar> dBDInt, dBDBnd;
    deviceMulesDonorFlux(dm, (int)nIf, (int)nBf, dPhi, dPsi, dPhiPsiBnd, dBDInt, dBDBnd);

    // phiCorr = phiPsi - phiBD, formed here so the comparison is of the LIMITER and not of the
    // subtraction.
    DeviceBuffer<scalar> dCorrInt(hCorr.internal), dCorrBnd(flatten(hCorr.boundary));

    DeviceMulesFields df;
    DeviceMulesControls dc;
    dc.nLimiterIter = ctl.nLimiterIter;
    DeviceBuffer<scalar> dLamInt, dLamBnd;
    deviceMulesLimiter(dm, (int)nIf, (int)nBf, scalar(1)/dt, dPsi, dPsiOld, dPsiBnd,
                       dFixes, dFlag, dBDInt, dBDBnd, dCorrInt, dCorrBnd, df, dc, dLamInt, dLamBnd);
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }

    std::vector<scalar> bd, lam;
    dBDInt.copyTo(bd);
    failures += brae::gatecheck::nonFinite("bd", bd);
    dLamInt.copyTo(lam);
    failures += brae::gatecheck::nonFinite("lam", lam);

    // --- the donor flux is per face, no reduction: it IS bit-identical ---
    {
        int nDiff = 0;
        for (label f = 0; f < nIf; ++f) if (bd[f] != hPhiBD.internal[f]) ++nDiff;
        check("the bounded donor flux is bit-identical to the host -- per face, no reduction",
              nDiff == 0);
    }

    // --- lambda: the gathers run in a different order, so this is a measured comparison ---
    {
        int nDiff = 0, nBig = 0;
        scalar worst = 0;
        label iw = -1;
        for (label f = 0; f < nIf; ++f)
        {
            const scalar e = std::fabs(lam[f] - hLam.internal[f]);
            if (lam[f] != hLam.internal[f]) ++nDiff;
            if (e > scalar(1e-9)) ++nBig;
            if (e > worst) { worst = e; iw = f; }
        }
        std::printf("  lambda: %d of %ld faces differ at all, %d by more than 1e-9; worst %.4e\n",
                    nDiff, (long)nIf, nBig, (double)worst);
        if (iw >= 0)
            std::printf("    at face %ld: device %.17g, host %.17g\n",
                        (long)iw, (double)lam[iw], (double)hLam.internal[iw]);
        check("lambda agrees with the host to 1e-9", worst < scalar(1e-9));
        check("...and lambda is in [0,1] on every face",
              *std::min_element(lam.begin(), lam.end()) >= scalar(0)
           && *std::max_element(lam.begin(), lam.end()) <= scalar(1));
    }

    // --- THE PROPERTY: apply the device limiter and check alpha stays bounded ---
    {
        SurfaceScalarField blended = hPhiBD;
        for (label f = 0; f < nIf; ++f) blended.internal[f] += lam[f]*hCorr.internal[f];
        // the boundary correction is identically zero, so the boundary is phiBD unchanged
        std::vector<scalar> psi;
        mules::Fields f0;
        mules::explicitSolve(scalar(1)/dt, psi, a.internal, blended, f0, m, g, fvp);

        scalar worstExc = 0, moved = 0;
        for (label c = 0; c < nC; ++c)
        {
            worstExc = std::fmax(worstExc, std::fmax(-psi[c], psi[c] - scalar(1)));
            moved    = std::fmax(moved, std::fabs(psi[c] - a.internal[c]));
        }
        std::printf("  applying the DEVICE limiter: worst excursion %.3e, largest change %.4f\n",
                    (double)worstExc, (double)moved);
        check("the DEVICE limiter keeps alpha in [0,1]", worstExc <= scalar(1e-14));
        check("...and it moved the field, so the bound is not satisfied by doing nothing",
              moved > scalar(1e-3));
    }

    std::printf("test_device_mules: %d failures\n", failures);
    return failures ? 1 : 0;
}
