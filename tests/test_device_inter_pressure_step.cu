// ONE PASS OF pEqn.H on the device, against the host's pressureCorrector.
//
// Every operator under it is separately gated -- deviceBuoyancyFlux, deviceRhoRAUf,
// deviceInterAddPhiHbyATerms, deviceInterAssemblePEqn, deviceInterPEqnFlux, deviceCorrectVelocity and
// deviceStaticPressure -- so what is under test here is the SEQUENCE, and pEqn.H's sequence carries
// three placements that nothing underneath can check for itself:
//
//   phig ENTERS phiHbyA ON BOTH SIDES, so a wall's buoyancy and surface tension reach the pressure
//   equation's SOURCE. Arm 3 drops the boundary half and measures p_rgh, not just the flux.
//
//   THE SAME phig IS REUSED in the velocity correction, as (phig - flux)/rAUf. It is not recomputed
//   there, and it is the flux BEFORE the division by rAUf.
//
//   p IS REBUILT from the solved p_rgh. Arm 4 checks it against p_rgh + rho*gh at the END, not at the
//   start, which is the difference between carrying p and rebuilding it.
//
// THE COMPARISON IS TO THE SOLVER'S TOLERANCE, not to round-off: both sides solve the same p_rgh
// system with different Krylov solvers, so they land on the same field to about the tolerance they
// were given, and no tighter. The fixture asks for 1e-13.
#include "box_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "pbicgstab.cuh"
#include "inter_peqn_cpp.cuh"
#include "inter_ueqn_cpp.cuh"
#include "device_inter_pressure_step.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
namespace ifm = brae::cpu::interFoam;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
scalar worstOf(const std::vector<scalar>& a, const std::vector<scalar>& b, scalar& sc)
{
    scalar w = 0;
    sc = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        w  = std::fmax(w, std::fabs(a[i] - b[i]));
        sc = std::fmax(sc, std::fabs(b[i]));
    }
    return w;
}
}   // namespace

int main()
{
    std::printf("== one pass of interFoam's pEqn: device vs host\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 8;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N, scalar(0), scalar(2), scalar(1), scalar(0.5));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const label nFaces = static_cast<label>(g.magSf().size());
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // a VoF state: rho jumps 1000x, rAU = dt/rho follows it
    const scalar dt = scalar(1e-3);
    std::vector<scalar> rho(static_cast<std::size_t>(nC)), rAU(static_cast<std::size_t>(nC)),
                        gh(static_cast<std::size_t>(nC)), p_rgh0(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        rho[c]    = (C.y < scalar(4)) ? scalar(1000) : scalar(1);
        rAU[c]    = dt / rho[c];
        gh[c]     = scalar(-9.81) * C.y;
        p_rgh0[c] = scalar(2)*std::sin(C.x) + C.z;
    }

    std::vector<vector> HbyA(static_cast<std::size_t>(nC));
    std::vector<scalar> hx(nC), hy(nC), hz(nC);
    for (label c = 0; c < nC; ++c)
    {
        HbyA[c] = vector{scalar(0.2)*g.C()[c].x, scalar(-0.1)*g.C()[c].y, scalar(0.05)};
        hx[c] = HbyA[c].x; hy[c] = HbyA[c].y; hz[c] = HbyA[c].z;
    }

    // the face fields, over the mesh's FULL face array
    std::vector<scalar> stf(nFaces), ghf(nFaces), snRho(nFaces), rAUfAll(nFaces);
    for (label f = 0; f < nFaces; ++f)
    {
        stf[f]   = scalar(0.07)*std::sin(scalar(f));
        ghf[f]   = scalar(-9.81)*g.Cf()[f].y;
        snRho[f] = scalar(999)*std::cos(scalar(0.3)*scalar(f));
    }
    for (label f = 0; f < nIf; ++f)
    {
        const scalar w = g.weights()[f];
        rAUfAll[f] = w*rAU[m.owner()[f]] + (scalar(1) - w)*rAU[m.neighbour()[f]];
    }
    {
        label off = nIf;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                rAUfAll[off++] = rAU[fvp[pi].faceCells[i]];    // fvc::interpolate at an uncoupled patch
    }

    // fvc::flux(HbyA): what the shared pressure predictor leaves behind
    std::vector<scalar> phiHInt(static_cast<std::size_t>(nIf)), phiHBnd;
    for (label f = 0; f < nIf; ++f)
    {
        const scalar w = g.weights()[f];
        const label o = m.owner()[f], n = m.neighbour()[f];
        const vector& S = g.Sf()[f];
        phiHInt[f] = (w*HbyA[o].x + (1-w)*HbyA[n].x)*S.x
                   + (w*HbyA[o].y + (1-w)*HbyA[n].y)*S.y
                   + (w*HbyA[o].z + (1-w)*HbyA[n].z)*S.z;
    }
    std::vector<std::vector<scalar>> phiHBndH(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label gf = fvp[pi].start + i;
            const vector& hb = HbyA[fvp[pi].faceCells[i]];
            const vector& S = g.Sf()[gf];
            const scalar v = hb.x*S.x + hb.y*S.y + hb.z*S.z;
            phiHBndH[pi].push_back(v);
            phiHBnd.push_back(v);
        }

    // ---- the HOST pass, written out as pEqn.H does it ---------------------------------------------
    // p_rgh's own field, with ONE fixedValue patch so the laplacian has boundary coefficients that are
    // not (0, 0) -- a mesh of pure zeroGradient makes the boundary half of everything vacuous.
    auto makePrgh = [&]()
    {
        auto f = std::make_unique<GeometricField<scalar>>();
        f->internal = p_rgh0;
        for (const FvPatch& q : fvp)
        {
            if (q.name == "outlet")
                f->boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(
                    q, true, scalar(0), std::vector<scalar>{}));
            else
                f->boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        }
        f->evaluateBoundary();
        return f;
    };

    auto hostPass = [&](bool dropPhigBoundary)
    {
        auto prgh = makePrgh();
        std::vector<scalar> phigAll(nFaces);
        for (label f = 0; f < nFaces; ++f)
            phigAll[f] = (stf[f] - ghf[f]*snRho[f]) * rAUfAll[f] * g.magSf()[f];

        SurfaceScalarField phiH;
        phiH.internal = phiHInt;
        phiH.boundary = phiHBndH;
        for (label f = 0; f < nIf; ++f) phiH.internal[f] += phigAll[f];
        if (!dropPhigBoundary)
        {
            label off = nIf;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                for (label i = 0; i < fvp[pi].size; ++i)
                    phiH.boundary[pi][i] += phigAll[off++];
        }

        SurfaceScalarField rAUfField;
        rAUfField.internal.assign(rAUfAll.begin(), rAUfAll.begin() + nIf);
        rAUfField.boundary.resize(fvp.size());
        {
            label off = nIf;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                for (label i = 0; i < fvp[pi].size; ++i)
                    rAUfField.boundary[pi].push_back(rAUfAll[off++]);
        }

        FvScalarMatrix pe = fvm::laplacian<scalar>(rAUfField, *prgh, m, g, fvp, false);
        const std::vector<scalar> dv = fvc::div(phiH, m, g, fvp);
        for (label c = 0; c < nC; ++c) pe.source[c] += dv[c]*g.V()[c];
        pbicgstab(pe, prgh->internal, m, fvp, scalar(1e-13), scalar(0), 2000);
        prgh->evaluateBoundary();

        const SurfaceScalarField pFlux = matrixFlux(pe, prgh->internal, m, fvp);
        std::vector<scalar> ffInt(static_cast<std::size_t>(nIf));
        std::vector<std::vector<scalar>> ffBnd(fvp.size()), rfBnd(fvp.size());
        for (label f = 0; f < nIf; ++f) ffInt[f] = phigAll[f] - pFlux.internal[f];
        {
            label off = nIf;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    ffBnd[pi].push_back(phigAll[off] - pFlux.boundary[pi][i]);
                    rfBnd[pi].push_back(rAUfAll[off]);
                    ++off;
                }
        }
        std::vector<vector> Ucorr;
        std::vector<scalar> rfInt(rAUfAll.begin(), rAUfAll.begin() + nIf);
        ifm::correctVelocity(HbyA, rAU, ffInt, rfInt, ffBnd, rfBnd, m, g, fvp, Ucorr);
        std::vector<scalar> pStat;
        ifm::staticPressure(prgh->internal, rho, gh, pStat);

        struct R { std::vector<scalar> prgh, ux, p; };
        R r;
        r.prgh = prgh->internal;
        r.p    = pStat;
        for (label c = 0; c < nC; ++c) r.ux.push_back(Ucorr[c].x);
        return r;
    };

    const auto host = hostPass(false);

    // ---- the DEVICE pass -------------------------------------------------------------------------
    DeviceBuffer<scalar> dStf(stf), dGhf(ghf), dSnRho(snRho), dMagSf(g.magSf()), dRAUfAll(rAUfAll);
    DeviceBuffer<scalar> dRho(rho), dGh(gh), dRAU(rAU), dHx(hx), dHy(hy), dHz(hz);

    DeviceInterPressureInput din;
    din.stf = &dStf;
    din.ghf = &dGhf;
    din.snGradRho = &dSnRho;
    din.magSf = &dMagSf;
    din.rAUfAll = &dRAUfAll;
    din.rho = &dRho;
    din.gh = &dGh;
    din.ddtCorrInt = nullptr;          // a start from rest: no old flux to correct against
    din.solve.tol = scalar(1e-13);
    din.solve.relTol = 0;
    din.solve.maxIter = 2000;

    // the host's per-patch dispatch, handed in as the hook
    auto prghForCoeffs = makePrgh();
    DeviceInterPressureHooks hooks;
    hooks.pressureCoeffs =
        [&](const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>&,
            const DeviceBuffer<scalar>& rAUfInt, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        std::vector<scalar> rf;
        rAUfInt.copyTo(rf);
        failures += brae::gatecheck::nonFinite("rf", rf);
        SurfaceScalarField rAUfField;
        rAUfField.internal = rf;
        rAUfField.boundary.resize(fvp.size());
        {
            label off = nIf;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                for (label i = 0; i < fvp[pi].size; ++i)
                    rAUfField.boundary[pi].push_back(rAUfAll[off++]);
        }
        FvScalarMatrix pe = fvm::laplacian<scalar>(rAUfField, *prghForCoeffs, m, g, fvp, false);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                i2.push_back(pe.internalCoeffs[pi][i]);
                b2.push_back(pe.boundaryCoeffs[pi][i]);
            }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };

    DeviceBuffer<scalar> dPhiHI(phiHInt), dPhiHB(phiHBnd), dPrgh(p_rgh0);
    DeviceBuffer<scalar> phiI, phiB, ux, uy, uz, pOut;
    const scalar resid = deviceInterPressureStep(dm, din, hooks, dRAU, dHx, dHy, dHz,
                                                 dPhiHI, dPhiHB, dPrgh, phiI, phiB,
                                                 ux, uy, uz, pOut);
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }

    std::vector<scalar> gPrgh, gUx, gP;
    dPrgh.copyTo(gPrgh);
    failures += brae::gatecheck::nonFinite("gPrgh", gPrgh);
    ux.copyTo(gUx);
    failures += brae::gatecheck::nonFinite("gUx", gUx);
    pOut.copyTo(gP);
    failures += brae::gatecheck::nonFinite("gP", gP);

    // ---- 1. p_rgh, U and p against the host -------------------------------------------------------
    {
        scalar s1 = 0, s2 = 0, s3 = 0;
        const scalar wp = worstOf(gPrgh, host.prgh, s1);
        const scalar wu = worstOf(gUx,  host.ux,   s2);
        const scalar wq = worstOf(gP,   host.p,    s3);
        std::printf("  solver residual %.2e;  p_rgh %.3e of %.3e, U.x %.3e of %.3e, p %.3e of %.3e\n",
                    (double)resid, (double)wp, (double)s1, (double)wu, (double)s2,
                    (double)wq, (double)s3);
        check("p_rgh matches the host to the solver tolerance", wp <= scalar(1e-9)*s1);
        check("...and so does the corrected velocity", wu <= scalar(1e-9)*std::fmax(s2, scalar(1e-30)));
        check("...and the rebuilt p", wq <= scalar(1e-9)*s3);
        check("the pressure solve converged", resid <= scalar(1e-13));
    }

    // ---- 2. the fields actually moved -------------------------------------------------------------
    {
        scalar moved = 0;
        for (label c = 0; c < nC; ++c) moved = std::fmax(moved, std::fabs(gPrgh[c] - p_rgh0[c]));
        std::printf("  p_rgh moved by %.3e from its initial value\n", (double)moved);
        check("the pass actually solved something", moved > scalar(1e-3));
    }

    // ---- 3. phig's BOUNDARY half changes p_rgh, not just the flux ---------------------------------
    // fvc::div(phiHbyA) sums the boundary faces, so dropping it changes the pressure equation's SOURCE
    // and therefore the solved field. This is the arm that would catch adding phig to the internal
    // faces alone -- which leaves every other number in this file looking reasonable.
    {
        const auto hostNoBnd = hostPass(true);
        scalar s = 0;
        const scalar d = worstOf(hostNoBnd.prgh, host.prgh, s);
        std::printf("  dropping phig's BOUNDARY half moves the SOLVED p_rgh by %.3e of %.3e\n",
                    (double)d, (double)s);
        check("the boundary half of phig reaches the solved pressure, not just the flux",
              d > scalar(0.01)*s);
    }

    // ---- 4. p is REBUILT from the solved p_rgh, not carried ---------------------------------------
    {
        scalar worstRebuild = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar want = gPrgh[c] + rho[c]*gh[c];
            worstRebuild = std::fmax(worstRebuild, std::fabs(gP[c] - want));
        }
        scalar fromOld = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar stale = p_rgh0[c] + rho[c]*gh[c];
            fromOld = std::fmax(fromOld, std::fabs(gP[c] - stale));
        }
        std::printf("  p against p_rgh_solved + rho*gh: %.3e;  against the INITIAL p_rgh: %.3e\n",
                    (double)worstRebuild, (double)fromOld);
        check("p is rebuilt from the SOLVED p_rgh", worstRebuild == scalar(0));
        check("...which is a different field from the one the pass started with",
              fromOld > scalar(1e-3));
    }

    std::printf("test_device_inter_pressure_step: %d failures\n", failures);
    return failures ? 1 : 0;
}
