// ONE TIME STEP'S ALPHA HALF on the device, end to end, against the host solver.
//
// This is the first thing in the interFoam port a SOLVER LOOP could call rather than a gate, and every
// operator under it is separately landed and separately gated. What is under test here is the
// SEQUENCE, and interFoam's sequence has three mixture.correct() placements that nothing underneath can
// check for itself:
//
//   at the BOTTOM of every corrector                       alphaEqn.H:225
//   ONCE MORE inside the MULESCorr block, before them      alphaEqn.H:151-154
//   ONCE MORE AGAIN after the whole sub-cycle              interFoam.C:154
//
// Each is a calculateK pass and at a contact angle the curvature is a FIXED POINT in those passes -- on
// capillaryRise it walks 7070.5 -> 8659.4 -> 9353.1 -> 9681.2, so one pass short is not a rounding
// difference. Missing the middle one put damBreak at 1.05e-07 against OpenFOAM where the full sequence
// gives 3.4346e-09. The last one repeats the last corrector's where no patch makes the count of passes
// matter: arm 4 shows what a mixture of the step's OLD alpha would cost (it overwrites the mixture; it
// removes no pass), and arm 5 leaves that last pass out and holds every field handed back to the bit.
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
#include <cstring>
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
    // the CONSTRUCTOR's deltaN, off a mesh that does not move here (InterfaceCoeffs::deltaN)
    ic.deltaN = ip::deltaN(g.V());
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

    // how often the boundary hook ran, and what was left of it where the pass after the sub-cycle is left out
    long hookCalls = 0;
    long mixtureCalls = 0;
    long leftOutCalls = 0;
    long refreshCalls = 0;
    DeviceInterAlphaHooks hooks;
    hooks.updateBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd, DeviceBuffer<scalar>& nBnd)
    {
        ++hookCalls;
        a.copyTo(work.internal);
        work.evaluateBoundary();
        aBnd.copyFrom(patchValues());
        SurfaceScalarField nHb;
        std::vector<scalar> Kb;
        ip::calculateK(work, ic, m, g, fvp, false, nHb, Kb);
        nBnd.copyFrom(flatten(nHb.boundary));
    };
    hooks.divCoeffs =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC,
            const DeviceBuffer<scalar>*)
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
    // everything the step hands back
    struct DevResult
    {
        std::vector<scalar> alpha;
        std::vector<scalar> rho;
        std::vector<scalar> rhoPhi;
        std::vector<scalar> rhoPhiBnd;
        std::vector<scalar> K;
        std::vector<scalar> nu;
        std::vector<scalar> mu;
        std::vector<scalar> alpha2;
        std::vector<scalar> nHatf;
        std::vector<scalar> alphaBnd;
        std::vector<scalar> nHatfBnd;
    };
    // the correctors a device run makes: the file's, but for the one run of arm 6 that takes one
    int nAlphaCorrNow = nAlphaCorr;
    auto runDevice = [&](
        int steps,
        bool mulesCorr,
        bool skipFinalMixture,
        bool leaveRepeatOut = false,
        bool relaxedHooks = false,
        bool driverHooks = false)
    {
        // the caller's two halves of "the pass after the sub-cycle repeats": the answer and what is left
        DeviceInterAlphaHooks hooksNow = hooks;
        if (leaveRepeatOut)
        {
            hooksNow.mixtureRepeatLeftOut = [&]()
            {
                ++leftOutCalls;
            };
        }
        // ...and the pair a relaxed corrector takes (arm 5): on these zero-gradient patches the relaxation's
        // assignment is the relaxed cells' values, and the pass after it must not evaluate
        // (and arm 6, the explicit path with the hooks the driver hands every step: the two come as a pair)
        if (relaxedHooks || driverHooks)
        {
            hooksNow.relaxBoundary = [&](
                const DeviceBuffer<scalar>&,
                const DeviceBuffer<scalar>& relaxed,
                DeviceBuffer<scalar>& aBnd)
            {
                relaxed.copyTo(work.internal);
                work.evaluateBoundary();
                aBnd.copyFrom(patchValues());
            };
            hooksNow.mixtureCorrect = [&](
                const DeviceBuffer<scalar>& a,
                DeviceBuffer<scalar>& aBnd,
                DeviceBuffer<scalar>& nBnd)
            {
                ++mixtureCalls;
                a.copyTo(work.internal);
                aBnd.copyFrom(patchValues());
                SurfaceScalarField nHb;
                std::vector<scalar> Kb;
                ip::calculateK(work, ic, m, g, fvp, false, nHb, Kb);
                nBnd.copyFrom(flatten(nHb.boundary));
            };
        }
        // ...and the evaluate that opens the limited explicit solve (arm 6), with the hook that hands the top of
        // a sub-step the patch values AS THEY STAND -- the driver gives both, and without the second the step
        // would take its top-of-sub-step values through refreshBoundary and the count would be of two things
        if (driverHooks)
        {
            hooksNow.storedBoundary = [&](
                DeviceBuffer<scalar>& aBnd)
            {
                aBnd.copyFrom(patchValues());
            };
            hooksNow.refreshBoundary = [&](
                const DeviceBuffer<scalar>& a,
                DeviceBuffer<scalar>& aBnd)
            {
                ++refreshCalls;
                a.copyTo(work.internal);
                work.evaluateBoundary();
                aBnd.copyFrom(patchValues());
            };
        }
        work.internal = a0;
        work.evaluateBoundary();
        SurfaceScalarField nH0;
        std::vector<scalar> K0;
        ip::calculateK(work, ic, m, g, fvp, false, nH0, K0);

        DeviceBuffer<scalar> alpha(a0), alphaOld(a0), aBnd(patchValues()), nBnd(flatten(nH0.boundary));
        DeviceBuffer<scalar> nHatf(nH0.internal), K(K0), rpInt, rpBnd, alpha2, rho, mu, nu;

        DeviceInterAlphaControls ctl;
        ctl.nAlphaSubCycles = nSub;
        ctl.nAlphaCorr = nAlphaCorrNow;
        ctl.MULESCorr = mulesCorr;
        ctl.preSolve.tol = scalar(1e-12);
        ctl.preSolve.maxIter = 2000;
        ctl.mixtureCorrectRepeats = leaveRepeatOut;

        for (int s = 0; s < steps; ++s)
        {
            std::vector<scalar> cur;
            alpha.copyTo(cur);
            failures += brae::gatecheck::nonFinite("cur", cur);
            alphaOld.copyFrom(cur);
            deviceInterAlphaStep(dm, alpha, alphaOld, totalDt, din, dmc, ctl, props, hooksNow,
                                 aBnd, nBnd, dFixes, dFlag, nHatf, K, rpInt, rpBnd,
                                 alpha2, rho, mu, nu);
            if (skipFinalMixture)
            {
                // arm 4's control: OVERWRITE the mixture with one of the field at the START of the step
                // -- UEqn would then be built on the density the step began with. (This removes no
                // pass: the last corrector's has already built the mixture from the new alpha.)
                DeviceBuffer<scalar> stale(cur);
                deviceMixtureCorrect(stale.data(), nC, props,
                                     alpha2.data(), rho.data(), mu.data(), nu.data());
            }
        }
        DevResult r;
        alpha.copyTo(r.alpha);
        rho.copyTo(r.rho);
        rpInt.copyTo(r.rhoPhi);
        rpBnd.copyTo(r.rhoPhiBnd);
        K.copyTo(r.K);
        nu.copyTo(r.nu);
        mu.copyTo(r.mu);
        alpha2.copyTo(r.alpha2);
        nHatf.copyTo(r.nHatf);
        aBnd.copyTo(r.alphaBnd);
        nBnd.copyTo(r.nHatfBnd);
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

    // 4: the mixture UEqn is built on must be the NEW alpha's
    // A mixture rebuilt from the alpha the step STARTED with leaves rho stale and alpha untouched -- the
    // error is entirely in the field UEqn is about to be built on, which is why it needs its own arm and why
    // no interface gate could find it. (What this arm breaks is the mixture, by overwriting it: it does not
    // remove a pass. Whether the pass AFTER the sub-cycle can go is arm 5's question.)
    {
        const DevResult good  = runDevice(nSteps, false, false);
        const DevResult stale = runDevice(nSteps, false, true);
        const scalar dRho   = worstOf(good.rho, stale.rho);
        const scalar dAlpha = worstOf(good.alpha, stale.alpha);
        std::printf("  rebuilding the mixture from the step's OLD alpha: rho %.4e of %.0f, "
                    "alpha %.3e\n", (double)dRho, (double)rho1, (double)dAlpha);
        check("a mixture of the step's old alpha is seen in rho -- UEqn would be built on the wrong density",
              dRho > scalar(1));
        check("...and alpha cannot see it at all, so no interface gate would catch it",
              dAlpha == scalar(0));
    }

    // 5: the pass after the sub-cycle LEFT OUT, where the caller says it repeats
    // interFoam.C:154's mixture.correct() repeats the one at the bottom of the last corrector (alphaEqn.H:225)
    // on the same alpha. With the caller's answer and its hook for what is left, the step makes one hook call
    // fewer a step and everything it hands back is the same to the bit; without them (every run above) the
    // pass is made. Both flavours of that pass: the explicit run here has NO mixtureCorrect hook, so its pass
    // takes the step's fallback, the hook that evaluates (updateBoundary) -- a form only a fixture has, the
    // driver always gives the hook (arm 6 runs the driver's form); a MULESCorr step whose second corrector
    // relaxes ends on the hook that does not evaluate (mixtureCorrect). WHAT THIS ARM HOLDS IS THE STEP AND ITS
    // KERNELS: the patches here are zero-gradient, so the host half -- whether a real case's hook repeats -- is
    // the caller's answer and is not under test.
    const auto sameBits = [](
        const std::vector<scalar>& a,
        const std::vector<scalar>& b)
    {
        return a.size() == b.size() && !a.empty()
            && std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0;
    };
    // the eleven fields the step hands back, to the bit
    const auto sameResult = [&](
        const DevResult& a,
        const DevResult& b)
    {
        return sameBits(a.alpha, b.alpha) && sameBits(a.rho, b.rho) && sameBits(a.rhoPhi, b.rhoPhi)
            && sameBits(a.rhoPhiBnd, b.rhoPhiBnd) && sameBits(a.K, b.K) && sameBits(a.nu, b.nu)
            && sameBits(a.mu, b.mu) && sameBits(a.alpha2, b.alpha2) && sameBits(a.nHatf, b.nHatf)
            && sameBits(a.alphaBnd, b.alphaBnd) && sameBits(a.nHatfBnd, b.nHatfBnd);
    };
    for (const bool relaxed : {false, true})
    {
        const long hooksBefore = hookCalls;
        const long mixBefore = mixtureCalls;
        const DevResult made = runDevice(nSteps, relaxed, false, false, relaxed);
        const long madeHooks = hookCalls - hooksBefore;
        const long madeMix = mixtureCalls - mixBefore;
        const long leftBefore = leftOutCalls;
        const DevResult left = runDevice(nSteps, relaxed, false, true, relaxed);
        const long leftHooks = hookCalls - hooksBefore - madeHooks;
        const long leftMix = mixtureCalls - mixBefore - madeMix;
        const long leftOver = leftOutCalls - leftBefore;
        const bool same = sameResult(made, left);
        std::printf("  the pass after the sub-cycle left out (%s): updateBoundary %ld calls against %ld, "
                    "mixtureCorrect %ld against %ld over %d steps, %ld calls of what is left; the eleven fields "
                    "handed back %s\n", relaxed ? "MULESCorr, relaxed" : "explicit", leftHooks, madeHooks,
                    leftMix, madeMix, nSteps, leftOver, same ? "bit for bit" : "DIFFER");
        // the flavour the last pass has decides which hook loses a call a step, and the other loses none
        const long fewerHooks = madeHooks - leftHooks;
        const long fewerMix = madeMix - leftMix;
        check(relaxed ? "left out, relaxed: one mixtureCorrect call fewer a step, updateBoundary's unchanged"
                      : "left out, explicit: one updateBoundary call fewer a step, one call of what is left",
              (relaxed ? (fewerMix == nSteps && fewerHooks == 0) : (fewerHooks == nSteps && fewerMix == 0))
              && leftOver == nSteps);
        check(relaxed ? "...and relaxed, the eleven fields handed back are the same to the bit"
                      : "...and the eleven fields handed back are the same to the bit",
              same);
    }

    // 6: the explicit path with the two hooks the DRIVER hands it, which no arm above has
    // (a) The pass after the sub-cycle takes the hook that evaluates NO patch (mixtureCorrect): interFoam.C:154's
    // mixture.correct() evaluates none but a contact angle's, inside correctContactAngle. It took updateBoundary
    // until 2026-10-06, and a contact angle with `limit gradient` does not repeat under a second evaluate
    // (capillaryRise, 2.1e-03 of OpenFOAM at forty steps; tests/interfoam_write/subcycle/
    // final_mixture_no_evaluate.sh). (b) The evaluate that opens the limited solve (MULESTemplates.C:168) is
    // made through refreshBoundary at every corrector, whatever the caller says of its hook (where it says the
    // hook repeats, the pass after the sub-cycle is left out and this evaluate is still made). WHAT THIS ARM
    // HOLDS IS THE CALLS -- tests/test_device_alpha_opening_evaluate.cu holds what the evaluate does to the
    // answer: the patches here are zero-gradient,
    // so neither moves a value and everything handed back must be the plain explicit run's to the bit. (The
    // plain run has no hook but updateBoundary: its calls are the correctors', one a sub-step for the values
    // at its top, and the pass after the sub-cycle.)
    // SHOWN RED by hand, 2026-10-06: BRAE_CONTROL_FINAL_MIXTURE_EVALUATES=1 fails the first check,
    // BRAE_CONTROL_MULES_OPENING_EVALUATE_SKIPPED=1 the second.
    {
        const long h0 = hookCalls;
        const DevResult plain = runDevice(nSteps, false, false);
        const long plainHooks = hookCalls - h0;
        const long h1 = hookCalls;
        const long m1 = mixtureCalls;
        const long r1 = refreshCalls;
        const DevResult every = runDevice(nSteps, false, false, false, false, true);
        const long everyHooks = hookCalls - h1;
        const long everyMix = mixtureCalls - m1;
        const long everyRefresh = refreshCalls - r1;
        const long m2 = mixtureCalls;
        const long r2 = refreshCalls;
        const long l2 = leftOutCalls;
        const DevResult once = runDevice(nSteps, false, false, true, false, true);
        const long onceMix = mixtureCalls - m2;
        const long onceRefresh = refreshCalls - r2;
        const long onceLeft = leftOutCalls - l2;
        // ...and MULESCorr with ONE corrector: its only corrector is the first and does not relax
        // (alphaEqn.H:196-204), so the pass after the sub-cycle took the evaluating hook there too. Its calls:
        // the pre-solve's pass and the corrector's through updateBoundary, the pass after the sub-cycle through
        // mixtureCorrect.
        nAlphaCorrNow = 1;
        const long h3 = hookCalls;
        const long m3 = mixtureCalls;
        (void)runDevice(nSteps, true, false, false, false, true);
        nAlphaCorrNow = nAlphaCorr;
        const long oneHooks = hookCalls - h3;
        const long oneMix = mixtureCalls - m3;
        const long correctors = static_cast<long>(nSteps)*nSub*nAlphaCorr;
        std::printf("  explicit, the driver's hooks, %d steps of %d sub-cycles and %d correctors: updateBoundary "
                    "%ld calls (%ld without them), mixtureCorrect %ld; the opening evaluate %ld calls, and %ld "
                    "where the caller says its hook repeats (mixtureCorrect %ld)\n", nSteps, nSub, nAlphaCorr,
                    everyHooks, plainHooks, everyMix, everyRefresh, onceRefresh, onceMix);
        std::printf("  MULESCorr with one corrector, the driver's hooks: updateBoundary %ld calls, mixtureCorrect "
                    "%ld\n", oneHooks, oneMix);
        check("explicit, the driver's hooks: the pass after the sub-cycle is mixtureCorrect's, one a step, and "
              "updateBoundary's calls are the correctors' alone",
              plainHooks == correctors + nSteps*nSub + nSteps && everyHooks == correctors
              && everyMix == nSteps && oneHooks == 2L*nSteps*nSub && oneMix == nSteps);
        check("...the opening evaluate is made at every corrector, also where the caller says its hook repeats "
              "(the pass after the sub-cycle left out there)",
              everyRefresh == correctors && onceRefresh == correctors && onceMix == 0 && onceLeft == nSteps);
        check("...and the eleven fields handed back are the plain explicit run's to the bit, both ways",
              sameResult(plain, every) && sameResult(plain, once));
    }

    std::printf("test_device_inter_alpha_step: %d failures\n", failures);
    return failures ? 1 : 0;
}
