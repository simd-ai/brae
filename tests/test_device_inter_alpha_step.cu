// ONE TIME STEP'S ALPHA HALF on the device, end to end, against the host solver.
//
// This is the first thing in the interFoam port a SOLVER LOOP could call rather than a gate, and every
// operator under it is separately landed and separately gated. What is under test here is the
// SEQUENCE, and interFoam's sequence has three mixture.correct() placements that nothing underneath can
// check for itself:
//
//   at the BOTTOM of every corrector                       alphaEqn.H:225
//   ONCE MORE inside the MULESCorr block, before them      alphaEqn.H:151-153
//   ONCE MORE AGAIN after the whole sub-cycle              alphaEqnSubCycle.H:36-38
//
// Each is a calculateK pass and the curvature is a FIXED POINT in those passes -- on capillaryRise it
// walks 7070.5 -> 8659.4 -> 9353.1 -> 9681.2, so one pass short is not a rounding difference. Missing
// the middle one put damBreak at 1.05e-07 against OpenFOAM where the full sequence gives 3.4346e-09.
// Arm 4 removes the last one and measures what it is worth.
//
// THE ORACLE IS THE HOST SOLVER'S OWN alphaEqnSubCycle DRIVING ITS OWN alphaEqnStep, which is the path
// that reaches 3.4346e-09 in alpha on damBreak against real OpenFOAM -- so agreement here is agreement
// with OpenFOAM two links along, and the links are each measured.
#include "box_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "interface_properties_cpp.cuh"
#include "two_phase_mixture_cpp.cuh"
#include "inter_solve_cpp.cuh"
#include "alpha_eqn_cpp.cuh"
#include "mules_cpp.cuh"
#include "fvm.cuh"
#include "device_inter_alpha_step.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
namespace ip  = brae::cpu::interfaceProps;
namespace ifm = brae::cpu::interFoam;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
std::vector<scalar> flatten(const std::vector<std::vector<scalar>>& b)
{
    std::vector<scalar> v;
    for (const auto& p : b) v.insert(v.end(), p.begin(), p.end());
    return v;
}
scalar worstOf(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar w = 0;
    for (std::size_t i = 0; i < a.size(); ++i) w = std::fmax(w, std::fabs(a[i] - b[i]));
    return w;
}
}   // namespace

int main()
{
    std::printf("== interFoam: a whole time step's alpha half, device vs host\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 24;
    const scalar h = scalar(1) / scalar(N);
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1, scalar(0), h, h, h);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    const scalar omega = scalar(2), cx = scalar(0.5), cy = scalar(0.5);
    auto vel = [&](const vector& P) { return vector{-omega*(P.y - cy), omega*(P.x - cx), scalar(0)}; };
    SurfaceScalarField phi;
    phi.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const vector u = vel(g.Cf()[f]), &S = g.Sf()[f];
        phi.internal[f] = u.x*S.x + u.y*S.y + u.z*S.z;
    }
    phi.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phi.boundary[pi].resize(static_cast<std::size_t>(fvp[pi].size));
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label gf = fvp[pi].start + i;
            const vector u = vel(g.Cf()[gf]), &S = g.Sf()[gf];
            phi.boundary[pi][i] = u.x*S.x + u.y*S.y + u.z*S.z;
        }
    }

    std::vector<scalar> a0(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::hypot(C.x - scalar(0.35), C.y - scalar(0.5));
        a0[c] = scalar(0.5)*(scalar(1) - std::tanh((r - scalar(0.15))/scalar(0.03)));
    }

    // damBreak's own phases and its own alpha controls
    const scalar rho1 = scalar(1000), rho2 = scalar(1);
    const scalar nu1 = scalar(1e-6), nu2 = scalar(1.48e-5);
    ip::InterfaceCoeffs ic;
    ic.cAlpha = scalar(1);
    const scalar totalDt = scalar(0.02);
    const int nSub = 3, nAlphaCorr = 2, nSteps = 6;

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<int> fixes, flag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        { fixes.push_back(0); flag.push_back(fvp[pi].type == "empty" ? 1 : 0); }
    DeviceBuffer<scalar> dPhiInt(phi.internal), dPhiBnd(flatten(phi.boundary));
    DeviceBuffer<int> dFixes(fixes), dFlag(flag);

    DeviceAlphaStepInput din;
    din.phiInt = &dPhiInt;
    din.phiBnd = &dPhiBnd;
    din.phiCNInt = &dPhiInt;
    din.phiCNBnd = &dPhiBnd;
    din.cAlpha = ic.cAlpha;
    din.rho1 = rho1;
    din.rho2 = rho2;
    din.deltaN = ip::deltaN(g.V());
    din.alphaScheme  = DeviceAlphaScheme::vanLeer;
    din.alpharScheme = DeviceAlphaScheme::linear;

    DeviceMulesControls dmc;
    dmc.nLimiterIter = 5;
    DevicePhaseProperties props{rho1, nu1, rho2, nu2};

    // the host scratch the hooks evaluate through -- it never sees the HOST RUN's fields, only what the
    // device last wrote
    GeometricField<scalar> work;
    work.internal = a0;
    for (const FvPatch& q : fvp)
        work.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    work.evaluateBoundary();
    auto patchValues = [&]()
    {
        std::vector<scalar> v;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& b = work.boundary[pi]->value();
            v.insert(v.end(), b.begin(), b.end());
        }
        return v;
    };

    DeviceInterAlphaHooks hooks;
    hooks.updateBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd, DeviceBuffer<scalar>& nBnd)
    {
        a.copyTo(work.internal);
        work.evaluateBoundary();
        aBnd.copyFrom(patchValues());
        SurfaceScalarField nHb;
        std::vector<scalar> Kb;
        ip::calculateK(work, ic, m, g, fvp, false, nHb, Kb);
        nBnd.copyFrom(flatten(nHb.boundary));
    };
    hooks.divCoeffs =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        a.copyTo(work.internal);
        work.evaluateBoundary();
        FvScalarMatrix M = fvm::div<scalar>(phi.internal, phi.boundary, work, m, fvp);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                i2.push_back(M.internalCoeffs[pi][i]);
                b2.push_back(M.boundaryCoeffs[pi][i]);
            }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };

    // ---- the DEVICE run --------------------------------------------------------------------------
    struct DevResult { std::vector<scalar> alpha, rho, rhoPhi; };
    auto runDevice = [&](int steps, bool mulesCorr, bool skipFinalMixture)
    {
        work.internal = a0;
        work.evaluateBoundary();
        SurfaceScalarField nH0;
        std::vector<scalar> K0;
        ip::calculateK(work, ic, m, g, fvp, false, nH0, K0);

        DeviceBuffer<scalar> alpha(a0), alphaOld(a0), aBnd(patchValues()), nBnd(flatten(nH0.boundary));
        DeviceBuffer<scalar> nHatf(nH0.internal), K(K0), rpInt, rpBnd, alpha2, rho, mu, nu;

        DeviceInterAlphaControls ctl;
        ctl.nAlphaSubCycles = nSub;
        ctl.nAlphaCorr      = nAlphaCorr;
        ctl.MULESCorr       = mulesCorr;
        ctl.preSolve.tol    = scalar(1e-12);
        ctl.preSolve.maxIter = 2000;

        for (int s = 0; s < steps; ++s)
        {
            std::vector<scalar> cur;
            alpha.copyTo(cur);
            failures += brae::gatecheck::nonFinite("cur", cur);
            alphaOld.copyFrom(cur);
            deviceInterAlphaStep(dm, alpha, alphaOld, totalDt, din, dmc, ctl, props, hooks,
                                 aBnd, nBnd, dFixes, dFlag, nHatf, K, rpInt, rpBnd,
                                 alpha2, rho, mu, nu);
            if (skipFinalMixture)
            {
                // arm 4's control: rebuild the mixture from the field at the START of the step, which
                // is what leaving out alphaEqnSubCycle.H:36-38 amounts to -- UEqn would then be built
                // on the density the step began with.
                DeviceBuffer<scalar> stale(cur);
                deviceMixtureCorrect(stale.data(), nC, props,
                                     alpha2.data(), rho.data(), mu.data(), nu.data());
            }
        }
        DevResult r;
        alpha.copyTo(r.alpha);
        rho.copyTo(r.rho);
        rpInt.copyTo(r.rhoPhi);
        return r;
    };

    // ---- the HOST run ----------------------------------------------------------------------------
    auto runHost = [&](int steps, bool mulesCorr)
    {
        GeometricField<scalar> a;
        a.internal = a0;
        for (const FvPatch& q : fvp)
            a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        a.evaluateBoundary();
        SurfaceScalarField nHatf, alphaPhi, rhoPhi;
        std::vector<scalar> K;
        ip::calculateK(a, ic, m, g, fvp, false, nHatf, K);

        brae::cpu::MULES::Controls hctl;
        hctl.nLimiterIter = dmc.nLimiterIter;
        ifm::AlphaStepInput hin;
        hin.phi = &phi;
        hin.phiCN = &phi;
        hin.cAlpha = ic.cAlpha;
        hin.nAlphaCorr = nAlphaCorr;
        hin.rho1 = rho1;
        hin.rho2 = rho2;
        hin.MULESCorr = mulesCorr;
        hin.tolAlpha = scalar(1e-12);
        hin.relTolAlpha = 0;
        hin.maxIterAlpha = 2000;
        hin.alphaScheme  = ifm::AlphaFluxScheme::vanLeer;
        hin.alpharScheme = ifm::AlphaFluxScheme::linear;

        ifm::AlphaEqnStep hstep =
            [&](const std::vector<scalar>& old, scalar dtSub, std::vector<scalar>& out,
                SurfaceScalarField& rp)
        {
            hin.deltaT = dtSub;
            ifm::alphaEqnStep(a, old, hin, ic, hctl, m, g, fvp, alphaPhi, rp, nHatf, K, nullptr);
            out = a.internal;
        };

        std::vector<scalar> cur = a0;
        for (int s = 0; s < steps; ++s)
        {
            const std::vector<scalar> old = cur;
            ifm::alphaEqnSubCycle(nSub, totalDt, cur, old, rhoPhi, hstep);
            a.internal = cur;
            a.evaluateBoundary();
        }
        // the host mixture, from the same final field. rho takes the RAW alpha (mu and nu take the
        // clamped one), so alpha2 here is 1 - alpha1 and not a clamped complement.
        brae::cpu::twoPhase::PhaseProperties hp;
        hp.rho1 = rho1; hp.nu1 = nu1; hp.rho2 = rho2; hp.nu2 = nu2;
        std::vector<scalar> a2(cur.size()), hrho;
        for (std::size_t i = 0; i < cur.size(); ++i) a2[i] = scalar(1) - cur[i];
        brae::cpu::twoPhase::mixtureRho(cur, a2, hp, hrho);
        return std::pair<std::vector<scalar>, std::vector<scalar>>{cur, hrho};
    };

    // ---- 1 & 2: both paths, device against host ---------------------------------------------------
    for (int path = 0; path < 2; ++path)
    {
        const bool mulesCorr = (path == 1);
        const DevResult d = runDevice(nSteps, mulesCorr, false);
        const auto hst = runHost(nSteps, mulesCorr);

        scalar exc = 0, moved = 0;
        for (label c = 0; c < nC; ++c)
        {
            exc   = std::fmax(exc, std::fmax(-d.alpha[c], d.alpha[c] - scalar(1)));
            moved = std::fmax(moved, std::fabs(d.alpha[c] - a0[c]));
        }
        const scalar dAlpha = worstOf(d.alpha, hst.first);
        const scalar dRho   = worstOf(d.rho, hst.second);
        std::printf("  %s, %d steps x %d sub-cycles x %d correctors: alpha %.4e, rho %.4e of %.0f, "
                    "excursion %.3e, moved %.4f\n",
                    mulesCorr ? "MULESCorr" : "explicit ", nSteps, nSub, nAlphaCorr,
                    (double)dAlpha, (double)dRho, (double)rho1, (double)exc, (double)moved);
        check(mulesCorr ? "the MULESCorr path matches the host over six whole steps"
                        : "the explicit path matches the host over six whole steps",
              dAlpha < scalar(1e-9));
        check("...and so does the mixture density it leaves for UEqn", dRho < scalar(1e-6));
        check("...having actually advected the interface", moved > scalar(0.5));
        if (!mulesCorr)
            check("...with alpha in [0,1] exactly on the explicit path", exc <= scalar(1e-14));
    }

    // ---- 3: the two paths must differ, or arm 2 is a copy of arm 1 -------------------------------
    {
        const DevResult e = runDevice(nSteps, false, false);
        const DevResult c = runDevice(nSteps, true,  false);
        const scalar d = worstOf(e.alpha, c.alpha);
        std::printf("  the two paths differ by %.4e\n", (double)d);
        check("MULESCorr and the explicit path give different fields, so both arms mean something",
              d > scalar(1e-3));
    }

    // ---- 4: mixture.correct() AFTER the sub-cycle (alphaEqnSubCycle.H:36-38) ---------------------
    // Skipping it leaves rho at the density the step STARTED with. alpha is untouched -- the error is
    // entirely in the field UEqn is about to be built on, which is why it needs its own arm and why no
    // interface gate could find it.
    {
        const DevResult good  = runDevice(nSteps, false, false);
        const DevResult stale = runDevice(nSteps, false, true);
        const scalar dRho   = worstOf(good.rho, stale.rho);
        const scalar dAlpha = worstOf(good.alpha, stale.alpha);
        std::printf("  rebuilding the mixture from the step's OLD alpha: rho %.4e of %.0f, "
                    "alpha %.3e\n", (double)dRho, (double)rho1, (double)dAlpha);
        check("the mixture after the sub-cycle is load-bearing -- UEqn would be built on the wrong rho",
              dRho > scalar(1));
        check("...and alpha cannot see it at all, so no interface gate would catch it",
              dAlpha == scalar(0));
    }

    std::printf("test_device_inter_alpha_step: %d failures\n", failures);
    return failures ? 1 : 0;
}
