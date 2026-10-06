// One time step's alpha half -- see device_inter_alpha_step.cuh for the three mixture.correct()
// placements and for what stays on the host.
#include "device_inter_alpha_step.cuh"
#include "device_alpha_subcycle.cuh"
#include "device_alpha_flux.cuh"
#include "device_interface_properties.cuh"
#include "device_blas.cuh"
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>

namespace brae {

void deviceInterAlphaStep(
    const DeviceMesh&                dm,
    DeviceBuffer<scalar>&            alpha1,
    const DeviceBuffer<scalar>&      alpha1Old,
    scalar                           totalDeltaT,
    const DeviceAlphaStepInput&      in,
    const DeviceMulesControls&       mulesCtl,
    const DeviceInterAlphaControls&  ctl,
    const DevicePhaseProperties&     props,
    const DeviceInterAlphaHooks&     hooks,
    DeviceBuffer<scalar>&            alpha1Bnd,
    DeviceBuffer<scalar>&            nHatfBnd,
    const DeviceBuffer<int>&         bndFixesValue,
    const DeviceBuffer<int>&         bndFlag,
    DeviceBuffer<scalar>&            nHatfInt,
    DeviceBuffer<scalar>&            K,
    DeviceBuffer<scalar>&            rhoPhiInt,
    DeviceBuffer<scalar>&            rhoPhiBnd,
    DeviceBuffer<scalar>&            alpha2,
    DeviceBuffer<scalar>&            rho,
    DeviceBuffer<scalar>&            mu,
    DeviceBuffer<scalar>&            nu)
{
    // `alphaApplyPrevCorr` IS A NO-OP WITHOUT `MULESCorr`, so the pair is what gates these checks.
    // alphaEqn.H APPLIES the previous correction only at :133, inside the `if (MULESCorr)` block opened
    // at :99, and STORES it only at :228 under `alphaApplyPrevCorr && MULESCorr`; without MULESCorr
    // talphaPhi1Corr0 is cleared every step and the switch does nothing. The host path already reads it
    // that way (alpha_eqn_cpp.cu:552, :731); this REFUSED instead, stopping on a case OpenFOAM runs.
    //
    // NOT an early `return` -- this is the top of deviceInterAlphaStep, and returning here would skip
    // the whole alpha equation.
    if (ctl.alphaApplyPrevCorr && ctl.MULESCorr)
    {
        if (!ctl.prevCorrInt || !ctl.prevCorrBnd)
        {
            throw std::runtime_error(
                "brae interFoam device alpha step: `alphaApplyPrevCorr yes` and no cache was handed "
                "in. The previous correction outlives this call, so the caller owns it -- see "
                "DeviceInterAlphaControls.");
        }
        if (!hooks.refreshBoundary)
        {
            throw std::runtime_error(
                "brae interFoam device alpha step: `alphaApplyPrevCorr yes` needs the refreshBoundary "
                "hook. The limiter reads alpha's patch values as the pre-solve left them, and "
                "updateBoundary would buy them with a curvature pass OpenFOAM does not take there.");
        }
    }
    if (static_cast<bool>(hooks.relaxBoundary) != static_cast<bool>(hooks.mixtureCorrect))
        throw std::runtime_error(
            "brae interFoam device alpha step: relaxBoundary and mixtureCorrect come as a pair. The "
            "first assigns the relaxed patch values and the second must not evaluate them away.");
    if (!hooks.updateBoundary)
        throw std::runtime_error(
            "brae interFoam device alpha step: the boundary hook is required. alpha's patch values are "
            "evaluated on the host -- see device_inter_alpha_step.cuh -- and running without it would "
            "advance the interior against a boundary frozen at the start of the time step.");
    if (ctl.MULESCorr && !hooks.divCoeffs)
        throw std::runtime_error(
            "brae interFoam device alpha step: MULESCorr needs the div matrix's boundary coefficients, "
            "which come from the host's per-patch valueInternalCoeffs. Without them the implicit "
            "pre-solve would run with a boundary of zeros and still converge.");
    if (ctl.nAlphaCorr < 1)
        throw std::runtime_error("brae interFoam device alpha step: nAlphaCorr must be at least 1.");

    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nBf = dm.nBndFaces;

    DeviceBuffer<scalar> alphaPhiInt, alphaPhiBnd, iC, bC;
    // nHatf on the pair, rewritten by every mixture.correct() below and read by the corrector's phir.
    // The caller's buffer when it keeps one (it must -- see DeviceInterAlphaControls::nHatfIf); the
    // local is only the no-pair case, where nothing reads it.
    DeviceBuffer<scalar> nHatfIfLocal;
    DeviceBuffer<scalar>& nHatfIfBuf = ctl.nHatfIf ? *ctl.nHatfIf : nHatfIfLocal;
    if (in.cyc && in.cyc->n > 0 && !ctl.nHatfIf)
    {
        throw std::runtime_error(
            "brae interFoam device alpha step: the mesh has a periodic pair and the caller kept no "
            "nHatf for it. The first corrector's phir reads the normal the LAST mixture.correct() "
            "left, which without MULESCorr is the previous time step's, so the buffer has to outlive "
            "the call -- as nHatfInt and nHatfBnd do.");
    }

    // mixture.correct() at the BOTTOM of a corrector: the interface normal from alpha's new field, then
    // the mixture properties from it. interfaceProperties reads mu and nu right after, which is why the
    // two are one call and not two.
    // whether the LAST corrector run relaxed, for the mixture.correct() after the whole sub-cycle
    bool lastRelaxed = false;
    // `patchesCurrent`: alpha1Bnd already holds what OpenFOAM's alpha1 carries (a relaxed corrector's
    // assignment), so the pass must not evaluate in front -- see DeviceInterAlphaHooks::relaxBoundary.
    // ...and which buffer the last pass ran on and in which flavour: the pass after the sub-cycle is left out
    // only where it would be that one again
    const DeviceBuffer<scalar>* lastPassOn = nullptr;
    bool lastPassCurrent = false;
    auto correctMixture = [&](
        const DeviceBuffer<scalar>& a,
        bool assignsAlpha2,
        bool patchesCurrent)
    {
        lastPassOn = &a;
        lastPassCurrent = patchesCurrent;
        if (patchesCurrent)
        {
            hooks.mixtureCorrect(a, alpha1Bnd, nHatfBnd);
        }
        else
        {
            hooks.updateBoundary(a, alpha1Bnd, nHatfBnd);
        }
        // alpha1Bnd is alpha1's patch as MULES's correctBoundaryConditions left it and BEFORE the
        // curvature pass below moves it -- which is the state `alpha2 = 1.0 - alpha1` reads.
        if (assignsAlpha2 && ctl.alpha2BndOut && nBf > 0)
        {
            ctl.alpha2BndOut->resize(static_cast<std::size_t>(nBf));
            deviceMixtureCorrect(alpha1Bnd.data(), nBf, props,
                                 ctl.alpha2BndOut->data(), nullptr, nullptr, nullptr);
        }
        // ...and nHatf ON THE PAIR from the same pass, which is where phir gets its normal there
        deviceInterfaceCorrect(dm, a, alpha1Bnd, nHatfBnd, in.deltaN,
                               in.nHatGradLeastSquares, in.nHatGradCellLimitK, nHatfInt, K,
                               in.cyc, in.cyc ? &nHatfIfBuf : nullptr);
        alpha2.resize(static_cast<std::size_t>(nC));
        rho.resize(static_cast<std::size_t>(nC));
        mu.resize(static_cast<std::size_t>(nC));
        nu.resize(static_cast<std::size_t>(nC));
        deviceMixtureCorrect(a.data(), nC, props,
                             alpha2.data(), rho.data(), mu.data(), nu.data());
    };
    // WHERE THE CURVATURE PASS MOVES ALPHA'S PATCH VALUES, THE NEXT CORRECTOR TAKES THE ONES IT LEFT. On a
    // contact angle correctContactAngle sets the wall gradient and evaluates (interfaceProperties.C:101-102),
    // AFTER the pass's own cell gradient and `alpha2 = 1 - alpha1` have read the values -- which is why the hooks
    // upload alpha1Bnd before it, and why correctMixture reads that upload. What comes next INSIDE the sub-step
    // reads the values the pass left: the next corrector builds alphaPhiUn on them (alphaEqn.H:164-176;
    // vanLeer's limiter takes fvc::grad(alpha1) on them). The device re-read the host's values only at the top
    // of a sub-step, so a second corrector of the same sub-step, or the first after the MULESCorr pre-solve's
    // pass, built its flux on the values from BEFORE the pass.
    // FOUND 2026-10-06 by a reviewer reading, CONFIRMED by a run: laminar/capillaryRise restaged to nAlphaCorr 2,
    // forty pinned steps, the host arm 4.9e-15 of OpenFOAM: the device arm alpha 3.1e-03, U 2.0e-03, p_rgh
    // 2.7e-02. No shipped tutorial has a contact angle with a second corrector or with MULESCorr, which is how
    // it stood.
    // NOT AFTER THE PASS THAT FOLLOWS THE SUB-CYCLE: what the step hands back is what the rest of the time step
    // reads, and that is the values from BEFORE that last pass (MEASURED: re-read there too, the first step of
    // the same row has alpha at 1.8e-16 and U 3.9e-04, p_rgh 9.3e-04 from OpenFOAM).
    //   BRAE_CONTROL_CONTACT_ANGLE_VALUES_BEFORE_PASS=1: a gate's CONTROL, deliberately wrong: not re-read
    auto valuesThePassLeft = [&]()
    {
        if (!ctl.curvaturePassMovesPatches || !hooks.storedBoundary) return;
        static const bool beforePass = std::getenv("BRAE_CONTROL_CONTACT_ANGLE_VALUES_BEFORE_PASS") != nullptr;
        static bool said = false;
        if (!said)
        {
            said = true;
            std::printf(beforePass
                ? "  *** CONTROL MODE: after a curvature pass the next corrector reads a contact angle's patch "
                  "values from before it. This run is deliberately wrong. ***\n"
                : "  mixture.correct(): the next corrector reads alpha's patch values as the curvature pass left "
                  "them (a contact angle's evaluate moves them)\n");
        }
        if (!beforePass)
        {
            hooks.storedBoundary(alpha1Bnd);
        }
    };

    // which sub-cycle this is, 1-based -- the wave conditions' clock
    int subCycle = 0;
    // ...and the volumes it runs on, refilled per sub-cycle on a mesh that moves
    DeviceBuffer<scalar> VscBuf, Vsc0Buf;
    // ...and phic, when the step's geometry changes under it (DeviceInterAlphaHooks::geometryUpdate)
    DeviceBuffer<scalar> phicIntPre, phicBndPre, phicIfPre;
    // ...and the pair's mass flux summed over the sub-cycles (see where it is formed, below)
    DeviceBuffer<scalar> rhoPhiIfSum;
    const bool pairRhoPhiLast = std::getenv("BRAE_CONTROL_DEVICE_PAIR_RHOPHI_LAST") != nullptr;
    if (pairRhoPhiLast)
    {
        std::printf("  *** CONTROL MODE: the pair's mass flux is the last sub-cycle's, not the time-weighted sum. "
                    "This run is deliberately wrong. ***\n");
    }
    const bool pairRhoPhiSummed = ctl.nAlphaSubCycles > 1 && ctl.rhoPhiIf && !pairRhoPhiLast;
    DeviceAlphaEqnStep step =
        [&](const DeviceBuffer<scalar>& subOld, scalar dtSub, DeviceBuffer<scalar>& alpha,
            DeviceBuffer<scalar>& rpInt, DeviceBuffer<scalar>& rpBnd)
    {
        ++subCycle;
        if (ctl.alphaSubCycleBndWrite && subCycle == ctl.nAlphaSubCycles)
        {
            deviceCopy(*ctl.alphaSubCycleBndWrite, alpha1Bnd);
        }
        DeviceAlphaStepInput li = in;
        // the volumes THIS sub-cycle runs on, before anything reads them
        if (hooks.subCycleVolumes)
        {
            hooks.subCycleVolumes(subCycle, VscBuf, Vsc0Buf);
            li.Vsc  = &VscBuf;
            li.Vsc0 = &Vsc0Buf;
        }
        li.nHatfIf = (in.cyc && in.cyc->n > 0) ? &nHatfIfBuf : nullptr;
        li.deltaT    = dtSub;
        li.MULESCorr = ctl.MULESCorr;

        // alpha1 COMES IN AS IT STANDS AND IS NOT RESET TO ITS OLD TIME -- the host reference's rule
        // (alpha_eqn_cpp.cu, alphaEqnStep), which alphaEqn.H has no assignment to contradict: the old
        // time enters through the pre-solve's ddt source and MULES' psi.oldTime(), and the CURRENT
        // alpha1 is the pre-solve's initial guess and the field alphaPhiUn is built from. With one outer
        // corrector the two are the same field, which is how a `cudaMemcpy(alpha, subOld)` here passed
        // every gate. With `nOuterCorrectors 2` OpenFOAM's second pass starts from the first pass's
        // result: MEASURED on RAS/damBreak with nOuterCorrectors 2, two steps, the second pass's first
        // p_rgh residual 1.2e-06 from OpenFOAM's (the host's 1e-12), k 6.7e-06 and U 2.0e-06 after
        // two steps -- and the same numbers under CrankNicolson, which is where it was found. In a
        // sub-cycle `alpha` already holds the previous sub-step's result (deviceAlphaEqnSubCycle
        // carries it), so nothing is copied on that path either.
        if (alpha.size() != static_cast<std::size_t>(nC))
            throw std::runtime_error(
                "brae interFoam device alpha step: alpha1 must arrive one value per cell; the step "
                "continues from it and does not reset it.");
        // THE STORED patch values, not an evaluate -- see DeviceInterAlphaHooks::storedBoundary.
        // MEASURED on RAS/mixerVesselAMI's first step, host against device with the evaluate here:
        // the outlet's 62 cells at 1 where the host's (and OpenFOAM's) rise to 1.095, alpha 8.6e-02.
        if (hooks.storedBoundary)
        {
            hooks.storedBoundary(alpha1Bnd);
        }
        // patch values only -- NOT a mixture.correct(); see DeviceInterAlphaHooks::refreshBoundary
        else if (hooks.refreshBoundary)
        {
            hooks.refreshBoundary(alpha, alpha1Bnd);
        }
        else
        {
            hooks.updateBoundary(alpha, alpha1Bnd, nHatfBnd);
        }

        // phic, ONCE and FIRST, and then the step's geometry change -- alphaEqn.H:59 and the lazy
        // cyclicACMI rescale that follows it inside the pre-solve. Only under a geometryUpdate hook:
        // without one the corrector forms phic itself, on geometry that does not change under it.
        if (hooks.geometryUpdate)
        {
            const int nIfP = dm.nInternalFaces;
            const int nBfP = dm.nBndFaces;
            deviceCompressionFlux(dm, nIfP, nBfP, *li.phiInt, li.cAlpha, phicIntPre, phicBndPre);
            li.phicIntPre = &phicIntPre;
            li.phicBndPre = &phicBndPre;
            if (li.cyc && li.cyc->n > 0)
            {
                deviceAlphaCyclicCompressionFlux(*li.cyc, li.cAlpha, phicIfPre);
                li.phicIfPre = &phicIfPre;
            }
            hooks.geometryUpdate();
        }

        if (ctl.MULESCorr)
        {
            // alphaEqn.H:103-155: the implicit upwind pre-solve, ONCE per sub-step, then a
            // mixture.correct() of its own before the correctors begin.
            // ...and the fvMatrix constructor's updateCoeffs, ahead of the assembly that reads it
            if (hooks.updateModelledBoundary)
            {
                hooks.updateModelledBoundary(subCycle, alpha, alpha1Bnd);
            }
            hooks.divCoeffs(alpha, iC, bC, (li.phiCNBnd != li.phiBnd) ? li.phiCNBnd : nullptr);
            DeviceSolverPerf pre;
            deviceAlphaPreSolve(dm, alpha, subOld, *li.phiCNInt, iC, bC, dtSub, ctl.preSolve,
                                alphaPhiInt, alphaPhiBnd, &pre, li.cyc, li.alphaPhiIf,
                                li.Vsc, li.Vsc0, li.rDeltaT);
            if (ctl.preSolveLog)
            {
                ctl.preSolveLog->push_back(pre);
            }
            if (ctl.preSolveAlphaOut)
            {
                deviceCopy(*ctl.preSolveAlphaOut, alpha);
            }
            if (ctl.preSolveAlphaPhiIfOut && li.alphaPhiIf)
            {
                deviceCopy(*ctl.preSolveAlphaPhiIfOut, *li.alphaPhiIf);
            }

            // talphaPhi1UD, the upwind flux the pre-solve left, kept for the cache at the bottom
            DeviceBuffer<scalar> upInt, upBnd;
            if (ctl.alphaApplyPrevCorr)
            {
                deviceCopy(upInt, alphaPhiInt);
                deviceCopy(upBnd, alphaPhiBnd);
            }

            // alphaEqn.H:133-147 -- "Applying the previous iteration compression flux". `.valid()` is
            // the cache having been filled, which the first alpha step of a run has not done.
            if (ctl.alphaApplyPrevCorr
             && static_cast<int>(ctl.prevCorrInt->size()) == nIf
             && static_cast<int>(ctl.prevCorrBnd->size()) == nBf)
            {
                // alpha1Eqn.solve() ends with correctBoundaryConditions(): patch values, no curvature
                hooks.refreshBoundary(alpha, alpha1Bnd);

                // MULES::correct(one, alpha1, alphaPhi10, talphaPhi1Corr0.ref(), one, zero). The flux
                // the outlet test reads is alphaPhi10 -- the ALPHA flux -- and the cached correction
                // is limited IN PLACE, so what is added below is the limited one.
                // the host's mf0 (alpha_eqn_cpp.cu, MULES::correctLimited): the moving mesh's volumes and the
                // local step. The volumes were MISSING here -- a moving mesh's prev-corr limited and solved
                // with the static V -- which tools/default_audit.py found once this site set a field at all.
                // No gated fixture witnesses it yet: the moving tutorials with alphaApplyPrevCorr are
                // floatingObject and DTCHullMoving.
                DeviceMulesFields mf0;
                mf0.Vsc = li.Vsc;
                mf0.Vsc0 = li.Vsc0;
                mf0.rDeltaT = li.rDeltaT ? li.rDeltaT->data() : nullptr;
                const scalar rDeltaT = scalar(1)/dtSub;
                deviceMulesLimitCorr(dm, nIf, nBf, rDeltaT, alpha, alpha1Bnd, bndFixesValue, bndFlag,
                                     alphaPhiBnd, *ctl.prevCorrInt, *ctl.prevCorrBnd, mf0, mulesCtl);
                deviceMulesCorrect(dm, rDeltaT, *ctl.prevCorrInt, *ctl.prevCorrBnd, mf0, alpha);

                // alphaPhi10 += talphaPhi1Corr0()
                deviceAxpy(scalar(1), *ctl.prevCorrInt, alphaPhiInt);
                if (nBf > 0)
                {
                    deviceAxpy(scalar(1), *ctl.prevCorrBnd, alphaPhiBnd);
                }
            }

            correctMixture(alpha, true, false);                 // alphaEqn.H:151-153
            valuesThePassLeft();

            // Cache the upwind-flux (alphaEqn.H:150), to be turned into the correction once the
            // correctors below have run. Held in the cache buffers themselves, as OpenFOAM holds it
            // in talphaPhi1Corr0.
            if (ctl.alphaApplyPrevCorr)
            {
                deviceCopy(*ctl.prevCorrInt, upInt);
                deviceCopy(*ctl.prevCorrBnd, upBnd);
            }
        }

        for (int aCorr = 0; aCorr < ctl.nAlphaCorr; ++aCorr)
        {
            li.aCorr = aCorr;
            DeviceAlphaBoundary db;
            db.alpha1     = &alpha1Bnd;
            db.nHatfBnd   = &nHatfBnd;
            db.fixesValue = &bndFixesValue;
            db.flag       = &bndFlag;
            if (hooks.updateModelledBoundary && !ctl.MULESCorr)
            {
                db.updateModelled = [&](const DeviceBuffer<scalar>& a)
                {
                    hooks.updateModelledBoundary(subCycle, a, alpha1Bnd);
                };
            }
            // THE EVALUATE THAT OPENS THE LIMITED EXPLICIT SOLVE, on a case with no wave patch too.
            // MULES::explicitSolve begins with psi.correctBoundaryConditions() (MULESTemplates.C:168), AFTER
            // the high-order flux has been built on the stored patch values and BEFORE the limiter, which
            // takes its extrema on a patch that fixes a value from the patch values (:338-347). The host's
            // MULES makes it (mules_cpp.cu); this loop made it only where a wave model supplies the values,
            // so everywhere else the limiter read the values the LAST evaluate left -- on the flux of the
            // step before. They differ where a flux-conditional patch (inletOutlet,
            // variableHeightFlowRate) has changed direction since and its cell does not hold the inlet
            // value, and on a contact angle, whose evaluate does not repeat.
            // MADE AT EVERY CORRECTOR, as OpenFOAM makes it. It was left out "where it repeats the evaluate
            // before it" for a day: no shipped explicit case has a second corrector, and a sub-cycle is a
            // time of its own (alphaEqnSubCycle.H:19-27), so the shortcut saved nothing a tutorial runs and
            // rested on no patch class reading the time.
            // HELD by tests/test_device_alpha_opening_evaluate.cu on a fixture built to see it (an
            // inletOutlet side under the interface whose flux reverses every step: the device step is the
            // host's to 4.4e-16 with it and 8.6e-04 from it without). NO SHIPPED CASE TELLS THE TWO APART. MEASURED
            // 2026-10-06, device arm against OpenFOAM at pinned solves, made and skipped: capillaryRise 40
            // steps 2.8e-14 and 5.2e-14 (what moves there is a fixed-gradient wall's value, which the
            // limiter does not read), stokesI 40 steps and weirOverflow 60 the same bytes, the small static
            // AMI fixture 5 steps 8.4e-13.
            // COST, MEASURED the same day: the hook is 0.1-0.3 ms a call (capillaryRise, weirOverflow,
            // nozzleFlow2D, eulerianInjection 0.2 at 420k cells), one call a sub-cycle on a shipped case.
            //   BRAE_CONTROL_MULES_OPENING_EVALUATE_SKIPPED=1: never made without a wave patch, as before
            //   BRAE_MULES_OPENING_EVALUATE_REPORT=1: how many patch values each one moved and by how much at
            //     most (said when either grows)
            else if (!ctl.MULESCorr && hooks.refreshBoundary)
            {
                static const bool skipped = std::getenv("BRAE_CONTROL_MULES_OPENING_EVALUATE_SKIPPED") != nullptr;
                static const bool report = std::getenv("BRAE_MULES_OPENING_EVALUATE_REPORT") != nullptr;
                static bool saidForm = false;
                if (!saidForm)
                {
                    saidForm = true;
                    std::printf(skipped
                        ? "  *** CONTROL MODE: explicit MULES does not evaluate alpha's patches at the top of the "
                          "limited solve, as before 2026-10-06 (BRAE_CONTROL_MULES_OPENING_EVALUATE_SKIPPED). Not "
                          "OpenFOAM's form (MULESTemplates.C:168). ***\n"
                        : "  explicit MULES: alpha's patches are evaluated at the top of the limited solve "
                          "(MULESTemplates.C:168)\n");
                }
                if (!skipped)
                {
                    db.updateModelled = [&](const DeviceBuffer<scalar>& a)
                    {
                        std::vector<scalar> before;
                        if (report)
                        {
                            alpha1Bnd.copyTo(before);
                        }
                        hooks.refreshBoundary(a, alpha1Bnd);
                        if (!report) return;
                        std::vector<scalar> after;
                        alpha1Bnd.copyTo(after);
                        long moved = 0;
                        scalar most = 0;
                        for (std::size_t k = 0; k < after.size() && k < before.size(); ++k)
                        {
                            if (std::memcmp(&after[k], &before[k], sizeof(scalar)) == 0) continue;
                            ++moved;
                            most = std::fmax(most, std::fabs(after[k] - before[k]));
                        }
                        static long calls = 0;
                        static long movedMost = -1;
                        static long movedTotal = 0;
                        static scalar mostSeen = 0;
                        ++calls;
                        movedTotal += moved;
                        if (moved > movedMost || most > mostSeen || calls == 1)
                        {
                            movedMost = moved > movedMost ? moved : movedMost;
                            mostSeen = std::fmax(mostSeen, most);
                            std::printf("  explicit MULES opening evaluate: %ld made, %ld patch values moved in "
                                        "all, at most %ld in one and by %.3e\n", calls, movedTotal, movedMost,
                                        (double)mostSeen);
                        }
                    };
                }
            }
            // every corrector but the first RELAXES on this path, and its boundary half is an
            // assignment -- see DeviceInterAlphaHooks::relaxBoundary
            const bool relaxes = ctl.MULESCorr && aCorr != 0 && static_cast<bool>(hooks.relaxBoundary);
            DeviceBuffer<scalar> postMules;
            db.alphaPostMules = relaxes ? &postMules : nullptr;
            deviceAlphaCorrector(dm, alpha, subOld, li, db, mulesCtl, nHatfInt,
                                 alphaPhiInt, alphaPhiBnd);
            if (relaxes)
            {
                hooks.relaxBoundary(postMules, alpha, alpha1Bnd);
            }
            correctMixture(alpha, true, relaxes);               // alphaEqn.H:223-225
            valuesThePassLeft();
            lastRelaxed = relaxes;
        }

        // alphaEqn.H:228-236: talphaPhi1Corr0 = alphaPhi10 - talphaPhi1Corr0, i.e. the compression the
        // correctors ended up applying on top of the upwind flux. The `else` clears it, which here is
        // the caller's buffers staying empty because nothing above ever filled them.
        if (ctl.alphaApplyPrevCorr)
        {
            DeviceBuffer<scalar> tInt, tBnd;
            deviceSubtractFaces(nIf, alphaPhiInt, *ctl.prevCorrInt, tInt);
            deviceSubtractFaces(nBf, alphaPhiBnd, *ctl.prevCorrBnd, tBnd);
            deviceCopy(*ctl.prevCorrInt, tInt);
            deviceCopy(*ctl.prevCorrBnd, tBnd);
        }

        if (ctl.rhoPhiFromPhi)
        {
            // alphaEqn.H:253-262, the branch a non-Euler ddt(rho,U) takes: un-blend the end-of-step
            // flux, then rhoPhi with phi beside rho2 -- the host driver's step1 (inter_driver_cpp.cu)
            // THE PAIR IS CARRIED HERE TOO. Its faces are in neither of the two arrays above, so the
            // un-blend and the mass flux both need the pair's own buffers -- alphaPhi10 on those
            // faces, its old level, and rhoPhi's slot. The host arm walks the coupled patch with
            // every other one (inter_driver_cpp.cu:606-617), so this is the same arithmetic on a
            // third array and not a third formula.
            const bool pair = li.cyc && li.cyc->n > 0;
            if (pair && !li.alphaPhiIf)
                throw std::runtime_error(
                    "brae interFoam device alpha step: CrankNicolson on a coupled pair needs the "
                    "pair's own alpha flux; the caller gave none.");
            if (ctl.cnCoeffUnblend < scalar(1))
            {
                DeviceBuffer<scalar> createdI, createdB, createdIf;
                if (!ctl.alphaPhiOldInt)
                {
                    deviceCopy(createdI, alphaPhiInt);
                    deviceCopy(createdB, alphaPhiBnd);
                    if (ctl.alphaPhiCreatedInt) deviceCopy(*ctl.alphaPhiCreatedInt, alphaPhiInt);
                    if (ctl.alphaPhiCreatedBnd) deviceCopy(*ctl.alphaPhiCreatedBnd, alphaPhiBnd);
                }
                const DeviceBuffer<scalar>& oldI = ctl.alphaPhiOldInt ? *ctl.alphaPhiOldInt : createdI;
                const DeviceBuffer<scalar>& oldB = ctl.alphaPhiOldBnd ? *ctl.alphaPhiOldBnd : createdB;
                deviceUnblendAlphaFlux(nIf, ctl.cnCoeffUnblend, oldI, alphaPhiInt);
                deviceUnblendAlphaFlux(nBf, ctl.cnCoeffUnblend, oldB, alphaPhiBnd);
                if (pair)
                {
                    // the pair's level is created by this same pass when it does not exist, exactly
                    // as the other two are: GeometricField::oldTime() on a field never asked for one
                    if (!ctl.alphaPhiOldIf)
                    {
                        deviceCopy(createdIf, *li.alphaPhiIf);
                        if (ctl.alphaPhiCreatedIf) deviceCopy(*ctl.alphaPhiCreatedIf, *li.alphaPhiIf);
                    }
                    const DeviceBuffer<scalar>& oldIf = ctl.alphaPhiOldIf ? *ctl.alphaPhiOldIf : createdIf;
                    deviceUnblendAlphaFlux(li.cyc->n, ctl.cnCoeffUnblend, oldIf, *li.alphaPhiIf);
                }
            }
            deviceMassFlux(nIf, alphaPhiInt, *li.phiInt, li.rho1, li.rho2, rpInt);
            deviceMassFlux(nBf, alphaPhiBnd, *li.phiBnd, li.rho1, li.rho2, rpBnd);
            // ...and the pair's rhoPhi, which this branch takes beside the RAW phi and not phiCN
            if (pair && ctl.rhoPhiIf)
            {
                deviceMassFlux(li.cyc->n, *li.alphaPhiIf, li.cyc->phi, li.rho1, li.rho2, *ctl.rhoPhiIf);
            }
            if (ctl.alphaPhiOutInt) deviceCopy(*ctl.alphaPhiOutInt, alphaPhiInt);
            if (ctl.alphaPhiOutBnd) deviceCopy(*ctl.alphaPhiOutBnd, alphaPhiBnd);
            if (pair && ctl.alphaPhiOutIf) deviceCopy(*ctl.alphaPhiOutIf, *li.alphaPhiIf);
            if (ctl.alphaPhiWriteInt) deviceCopy(*ctl.alphaPhiWriteInt, alphaPhiInt);
            if (ctl.alphaPhiWriteBnd) deviceCopy(*ctl.alphaPhiWriteBnd, alphaPhiBnd);
            if (pair && ctl.alphaPhiWriteIf) deviceCopy(*ctl.alphaPhiWriteIf, *li.alphaPhiIf);
            return;
        }
        // the Euler branch's alphaPhi10 as the sub-step leaves it, for a write (copies only)
        if (ctl.alphaPhiWriteInt) deviceCopy(*ctl.alphaPhiWriteInt, alphaPhiInt);
        if (ctl.alphaPhiWriteBnd) deviceCopy(*ctl.alphaPhiWriteBnd, alphaPhiBnd);
        if (li.alphaPhiIf && li.cyc && li.cyc->n > 0 && ctl.alphaPhiWriteIf)
        {
            deviceCopy(*ctl.alphaPhiWriteIf, *li.alphaPhiIf);
        }
        // rhoPhi = alphaPhi10*(rho1 - rho2) + phiCN*rho2, alphaEqn.H:248 -- built once per SUB-STEP,
        // from the flux the last corrector left. The sub-cycle then time-weights these.
        deviceMassFlux(nIf, alphaPhiInt, *li.phiCNInt, li.rho1, li.rho2, rpInt);
        deviceMassFlux(nBf, alphaPhiBnd, *li.phiCNBnd, li.rho1, li.rho2, rpBnd);
        // ...and on the pair, whose mass flux the momentum equation reads like any other face's
        if (li.cyc && li.cyc->n > 0 && ctl.rhoPhiIf && li.phiCNIf && li.alphaPhiIf)
        {
            deviceMassFlux(li.cyc->n, *li.alphaPhiIf, *li.phiCNIf, li.rho1, li.rho2, *ctl.rhoPhiIf);
            // ...TIME-WEIGHTED over the sub-cycles like every other face's (alphaEqnSubCycle.H:20-29,
            // rhoPhiSum += (runTime.deltaT()/totalDeltaT)*rhoPhi): deviceAlphaEqnSubCycle sums the
            // internal and boundary arrays it is handed, and the pair's is in neither.
            if (pairRhoPhiSummed)
            {
                if (rhoPhiIfSum.size() != ctl.rhoPhiIf->size())
                {
                    rhoPhiIfSum.resize(ctl.rhoPhiIf->size());
                    cudaCheck(cudaMemset(rhoPhiIfSum.data(), 0, rhoPhiIfSum.size()*sizeof(scalar)),
                              "rhoPhi sum, interface");
                }
                deviceAxpy(dtSub/totalDeltaT, *ctl.rhoPhiIf, rhoPhiIfSum);
            }
        }
    };

    deviceAlphaEqnSubCycle(ctl.nAlphaSubCycles, totalDeltaT, alpha1, alpha1Old,
                           rhoPhiInt, rhoPhiBnd, step);
    if (pairRhoPhiSummed && rhoPhiIfSum.size() == ctl.rhoPhiIf->size())
    {
        deviceCopy(*ctl.rhoPhiIf, rhoPhiIfSum);
    }

    // ...and mixture.correct() ONCE MORE after the whole sub-cycle (interFoam.C, after alphaEqnSubCycle.H).
    // The momentum equation must be built on the NEW density -- a mixture of the alpha the step started with
    // is wrong by a factor of 1000 at a water/air interface, and converges -- and what gives it that is the
    // pass at the bottom of the last corrector, which has already run on this alpha; this one is OpenFOAM's
    // second call of the same thing.
    //
    // THIS PASS DOES NOT EVALUATE ALPHA'S PATCHES. mixture.correct() is calcNu() and calculateK()
    // (immiscibleIncompressibleTwoPhaseMixture.H:78-82): the only patch it evaluates is a contact angle's,
    // inside correctContactAngle (interfaceProperties.C:102), and alphaEqnSubCycle.H evaluates nothing after
    // the sub-cycle -- so the patch values stand as the last corrector left them. It went through the hook
    // that evaluates first (hooks.updateBoundary) wherever the last corrector had not relaxed, on the reading
    // that an evaluate on unchanged cells changes nothing. On a contact angle with `limit gradient` it does:
    // the evaluate clamps the gradient against the patch's own value. MEASURED 2026-10-05 on
    // laminar/capillaryRise at pinned solves, the host arm 2.0e-15 from OpenFOAM throughout: the device arm
    // the same until the twelfth step, then alpha 1.3e-03 from it in one step and, at the fortieth, alpha
    // 2.1e-03, U 1.1e-03, p_rgh 1.9e-02; without the evaluate 2.0e-15, 4.8e-15 and 2.8e-14. The two-step and
    // five-step gates that held this case end before the clamp bites.
    //   BRAE_CONTROL_FINAL_MIXTURE_EVALUATES=1: a gate's CONTROL, deliberately wrong: the evaluate again
    //
    // THIS PASS REPEATS THE LAST CORRECTOR'S, AND IS LEFT OUT WHERE IT DOES. The last corrector ended with the
    // same hook, the same deviceInterfaceCorrect and the same deviceMixtureCorrect on this same alpha1 (the step
    // lambda's `alpha` IS alpha1), and nothing they read is written in between: the sub-cycle sums rhoPhi and
    // copies alpha1, no more. OpenFOAM makes both calls too (alphaEqn.H:225, then interFoam.C:154); the second
    // leaves every field as the first left it. (Its curvature divides by mesh.Vsc() inside a sub-cycle,
    // fvcSurfaceIntegrate.C:77 -- but the first of the two calls is in the LAST sub-cycle, where Vsc() is V()
    // itself unless the sub-cycle's time fraction has rounded below 1 - SMALL, fvMeshGeometry.C:262-281. Every
    // pass here divides by V, which is what OpenFOAM's second call does whatever that rounding.)
    // MEASURED 2026-10-05: one of the four alpha hook calls a step on laminar/waves/streamFunction (the hook
    // 4.2 ms a step), one of the three mixtureCorrect calls on RAS/electrostaticDeposition (3.6 ms each), plus
    // the device half each time.
    // KEPT where it is NOT a repeat or cannot be shown to be one:
    //   * the caller's answer for the hook (ctl.mixtureCorrectRepeats): a contact angle, whose gradient every
    //     curvature pass moves -- the NUMBER of passes is part of the answer there (the hooks' header) --, an
    //     alpha patch whose evaluate is not known to repeat, an alpha patch that names rhoPhi;
    //   * a coupled pair on the mesh: its interface sums are atomic, so two passes need not agree to the bit
    //     and the check below could not hold them;
    //   * a last pass on another buffer or of the other flavour, which no path here makes.
    //   BRAE_CONTROL_MIXTURE_REPEAT_KEPT=1: the pass is always made, as before
    //   BRAE_CONTROL_MIXTURE_REPEAT_CHECK=1: where it would be left out, what is left of the hook is run, as the
    //     default path runs it, then the pass is made anyway, and the eight buffers it writes and the HOST's
    //     state the hooks write (hooks.mixtureHostState) are compared with what stood before it, bitwise.
    //     `=host` compares the host's state alone.
    //   BRAE_CONTROL_MIXTURE_REPEAT_LEFT_OUT_ANYWAY=1: a gate's CONTROL, deliberately wrong: left out whatever
    //     the caller answered (a contact angle's pass is then missing)
    //   BRAE_CONTROL_MIXTURE_REPEAT_NOTHING_LEFT=1: another: what is left of the hook
    //     (hooks.mixtureRepeatLeftOut) is dropped too. Under the check it is seen where a patch names rhoPhi
    //     (the host's rhoPhi boundary is then the last sub-step's, not the sum). Without it, MEASURED on the W
    //     rows of damBreakPermeable, solitaryGrimshaw and stokesI, 0 of 24 written files differ: the U hook
    //     pushes the same again before anything reads it.
    static const bool repeatKept = std::getenv("BRAE_CONTROL_MIXTURE_REPEAT_KEPT") != nullptr;
    static const char* const repeatCheckEnv = std::getenv("BRAE_CONTROL_MIXTURE_REPEAT_CHECK");
    static const bool repeatCheck = repeatCheckEnv != nullptr;
    static const bool repeatCheckHostOnly = repeatCheck && std::string(repeatCheckEnv) == "host";
    static const bool repeatAnyway = std::getenv("BRAE_CONTROL_MIXTURE_REPEAT_LEFT_OUT_ANYWAY") != nullptr;
    static const bool repeatNothingLeft = std::getenv("BRAE_CONTROL_MIXTURE_REPEAT_NOTHING_LEFT") != nullptr;
    static const bool finalEvaluates = std::getenv("BRAE_CONTROL_FINAL_MIXTURE_EVALUATES") != nullptr;
    // the pass's flavour: without the evaluate wherever the caller has the hook for it (the driver always has)
    const bool finalCurrent = (!finalEvaluates && static_cast<bool>(hooks.mixtureCorrect)) ? true : lastRelaxed;
    // (said where the evaluating pass is MADE: on a case that leaves the pass out the switch does nothing)
    const auto sayFinalEvaluates = [&]()
    {
        static bool said = false;
        if (finalEvaluates && !finalCurrent && !said)
        {
            said = true;
            std::printf("  *** CONTROL MODE: the mixture.correct() after the alpha sub-cycle evaluates alpha's "
                        "patches first. This run is deliberately wrong. ***\n");
        }
    };
    // ...and whether it would repeat the last pass: without the evaluate it is that pass's mixture and
    // curvature again, whichever flavour that pass was; with it, only where that pass evaluated too
    const bool samePass = lastPassOn == &alpha1 && (finalCurrent || !lastPassCurrent);
    const bool pairOnMesh = in.cyc && in.cyc->n > 0;
    const bool callerSaysNo = !ctl.mixtureCorrectRepeats && !repeatAnyway;
    // the DECISION is these five; the text below is what the notice says of it
    const bool kept = repeatKept || !hooks.mixtureRepeatLeftOut || !samePass || pairOnMesh || callerSaysNo;
    std::string path = "left out, it repeats the last corrector's; BRAE_CONTROL_MIXTURE_REPEAT_KEPT=1 makes it";
    if (repeatKept)
    {
        path = "made (BRAE_CONTROL_MIXTURE_REPEAT_KEPT is set)";
    }
    else if (!hooks.mixtureRepeatLeftOut)
    {
        path = "made (the caller gave no hook for what is left of it)";
    }
    else if (!samePass)
    {
        path = "made (the last pass was on another buffer or of the other flavour)";
    }
    else if (pairOnMesh)
    {
        path = "made (a coupled pair's interface sums are atomic)";
    }
    else if (callerSaysNo)
    {
        path = "made (" + (ctl.mixtureRepeatKeptFor.empty() ? std::string("the caller says its hook does not repeat")
                                                           : ctl.mixtureRepeatKeptFor) + ")";
    }
    // said whenever it CHANGES, not once a process: a binary that calls the step with two answers says both
    static std::string pathSaid;
    if (path != pathSaid)
    {
        pathSaid = path;
        std::printf("  mixture.correct() after the alpha sub-cycle: %s\n", path.c_str());
        if (!kept && repeatAnyway && !ctl.mixtureCorrectRepeats)
        {
            std::printf("  *** CONTROL MODE: the mixture.correct() after the alpha sub-cycle is left out though "
                        "the hook does not repeat (%s). This run is deliberately wrong. ***\n",
                        ctl.mixtureRepeatKeptFor.c_str());
        }
        if (!kept && repeatNothingLeft)
        {
            std::printf("  *** CONTROL MODE: what is left of the alpha hook where that pass is left out is "
                        "dropped too. This run is deliberately wrong. ***\n");
        }
    }
    if (kept)
    {
        sayFinalEvaluates();
        correctMixture(alpha1, false, finalCurrent);
        return;
    }
    // what is left of the hook: the production path's, and under the check too, so that what the check takes
    // as `before` is the state the default run goes on with
    if (!repeatNothingLeft)
    {
        hooks.mixtureRepeatLeftOut();
    }
    if (!repeatCheck) return;
    // THE CHECK: what stands, the pass, what stands after it
    struct Held
    {
        const char* name;
        const DeviceBuffer<scalar>* buffer;
        std::vector<scalar> before;
    };
    Held held[] = {
        {"nHatf on the internal faces", &nHatfInt, {}},
        {"the curvature K", &K, {}},
        {"alpha2", &alpha2, {}},
        {"rho", &rho, {}},
        {"mu", &mu, {}},
        {"nu", &nu, {}},
        {"alpha1's patch values", &alpha1Bnd, {}},
        {"nHatf on the boundary", &nHatfBnd, {}},
    };
    for (Held& h : held)
    {
        h.buffer->copyTo(h.before);
    }
    const std::vector<scalar> hostBefore =
        hooks.mixtureHostState ? hooks.mixtureHostState() : std::vector<scalar>();
    sayFinalEvaluates();
    correctMixture(alpha1, false, finalCurrent);
    std::size_t values = 0;
    for (const Held& h : held)
    {
        if (repeatCheckHostOnly) break;
        std::vector<scalar> after;
        h.buffer->copyTo(after);
        values += after.size();
        if (after.size() == h.before.size()
         && (after.empty() || std::memcmp(after.data(), h.before.data(), after.size()*sizeof(scalar)) == 0)) continue;
        std::size_t at = 0;
        while (at < after.size() && at < h.before.size()
            && std::memcmp(&after[at], &h.before[at], sizeof(scalar)) == 0)
        {
            ++at;
        }
        char buf[360];
        std::snprintf(buf, sizeof(buf),
                      "brae interFoam device alpha step: BRAE_CONTROL_MIXTURE_REPEAT_CHECK: the mixture.correct() "
                      "after the sub-cycle does not repeat the last corrector's: %s, entry %zu of %zu, was %.17g "
                      "and is %.17g after the pass.", h.name, at, after.size(),
                      at < h.before.size() ? (double)h.before[at] : 0.0,
                      at < after.size() ? (double)after[at] : 0.0);
        throw std::runtime_error(buf);
    }
    const std::vector<scalar> hostAfter =
        hooks.mixtureHostState ? hooks.mixtureHostState() : std::vector<scalar>();
    if (hostAfter.size() != hostBefore.size()
     || (!hostAfter.empty()
      && std::memcmp(hostAfter.data(), hostBefore.data(), hostAfter.size()*sizeof(scalar)) != 0))
    {
        std::size_t at = 0;
        while (at < hostAfter.size() && at < hostBefore.size()
            && std::memcmp(&hostAfter[at], &hostBefore[at], sizeof(scalar)) == 0)
        {
            ++at;
        }
        char buf[360];
        std::snprintf(buf, sizeof(buf),
                      "brae interFoam device alpha step: BRAE_CONTROL_MIXTURE_REPEAT_CHECK: the mixture.correct() "
                      "after the sub-cycle does not repeat the last corrector's: the host's state, entry %zu of "
                      "%zu (%zu before), was %.17g and is %.17g after the pass.", at, hostAfter.size(),
                      hostBefore.size(), at < hostBefore.size() ? (double)hostBefore[at] : 0.0,
                      at < hostAfter.size() ? (double)hostAfter[at] : 0.0);
        throw std::runtime_error(buf);
    }
    // a check that passes says so: a count that grows, said at each of the first ten passes and then at 100,
    // 1000, ... -- a two-step row's count is its steps
    static long passes = 0;
    static long nextSaid = 100;
    ++passes;
    if (passes <= 10 || passes == nextSaid)
    {
        nextSaid = passes == nextSaid ? nextSaid*10 : nextSaid;
        std::printf("  mixture repeat check: %ld passes made where they would be left out, each leaving 8 buffers "
                    "(%zu values) and the host's state (%zu values) as they stood, bitwise\n", passes, values,
                    hostAfter.size());
    }
}

} // namespace brae
