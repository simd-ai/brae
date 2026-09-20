// One time step's alpha half -- see device_inter_alpha_step.cuh for the three mixture.correct()
// placements and for what stays on the host.
#include "device_inter_alpha_step.cuh"
#include "device_alpha_subcycle.cuh"
#include "device_alpha_flux.cuh"
#include "device_interface_properties.cuh"
#include "device_blas.cuh"
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
    if (ctl.alphaApplyPrevCorr)
    {
        if (!ctl.MULESCorr)
        {
            // alphaEqn.H:228 tests `alphaApplyPrevCorr && MULESCorr`; without MULESCorr the switch does
            // nothing in OpenFOAM either, and saying so beats a cache nobody reads
            throw std::runtime_error(
                "brae interFoam device alpha step: `alphaApplyPrevCorr yes` without `MULESCorr yes`. "
                "OpenFOAM only ever applies the previous correction inside the MULESCorr block.");
        }
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

    // mixture.correct() at the BOTTOM of a corrector: the interface normal from alpha's new field, then
    // the mixture properties from it. interfaceProperties reads mu and nu right after, which is why the
    // two are one call and not two.
    auto correctMixture = [&](
        const DeviceBuffer<scalar>& a,
        bool assignsAlpha2)
    {
        hooks.updateBoundary(a, alpha1Bnd, nHatfBnd);
        // alpha1Bnd is alpha1's patch as MULES's correctBoundaryConditions left it and BEFORE the
        // curvature pass below moves it -- which is the state `alpha2 = 1.0 - alpha1` reads.
        if (assignsAlpha2 && ctl.alpha2BndOut && nBf > 0)
        {
            ctl.alpha2BndOut->resize(static_cast<std::size_t>(nBf));
            deviceMixtureCorrect(alpha1Bnd.data(), nBf, props,
                                 ctl.alpha2BndOut->data(), nullptr, nullptr, nullptr);
        }
        deviceInterfaceCorrect(dm, a, alpha1Bnd, nHatfBnd, in.deltaN, nHatfInt, K);
        alpha2.resize(static_cast<std::size_t>(nC));
        rho.resize(static_cast<std::size_t>(nC));
        mu.resize(static_cast<std::size_t>(nC));
        nu.resize(static_cast<std::size_t>(nC));
        deviceMixtureCorrect(a.data(), nC, props,
                             alpha2.data(), rho.data(), mu.data(), nu.data());
    };

    // which sub-cycle this is, 1-based -- the wave conditions' clock
    int subCycle = 0;
    DeviceAlphaEqnStep step =
        [&](const DeviceBuffer<scalar>& subOld, scalar dtSub, DeviceBuffer<scalar>& alpha,
            DeviceBuffer<scalar>& rpInt, DeviceBuffer<scalar>& rpBnd)
    {
        ++subCycle;
        DeviceAlphaStepInput li = in;
        li.deltaT    = dtSub;
        li.MULESCorr = ctl.MULESCorr;

        // alphaEqn.H:97 -- alpha1 is reset to the sub-step's old time ONCE per sub-step, not once per
        // corrector. `alpha` is the sub-cycle's output buffer; it is written here rather than aliased
        // to alpha1, so this lambda never assumes the two are the same object.
        alpha.resize(static_cast<std::size_t>(nC));
        cudaMemcpy(alpha.data(), subOld.data(), sizeof(scalar)*nC, cudaMemcpyDeviceToDevice);
        // patch values only -- NOT a mixture.correct(); see DeviceInterAlphaHooks::refreshBoundary
        if (hooks.refreshBoundary)
        {
            hooks.refreshBoundary(alpha, alpha1Bnd);
        }
        else
        {
            hooks.updateBoundary(alpha, alpha1Bnd, nHatfBnd);
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
            hooks.divCoeffs(alpha, iC, bC);
            DeviceSolverPerf pre;
            deviceAlphaPreSolve(dm, alpha, subOld, *li.phiCNInt, iC, bC, dtSub, ctl.preSolve,
                                alphaPhiInt, alphaPhiBnd, &pre, li.cyc, li.alphaPhiIf);
            if (ctl.preSolveLog)
            {
                ctl.preSolveLog->push_back(pre);
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
                const DeviceMulesFields mf0;
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

            correctMixture(alpha, true);                        // alphaEqn.H:151-153

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
            deviceAlphaCorrector(dm, alpha, subOld, li, db, mulesCtl, nHatfInt,
                                 alphaPhiInt, alphaPhiBnd);
            correctMixture(alpha, true);                        // alphaEqn.H:223-225
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

        // rhoPhi = alphaPhi10*(rho1 - rho2) + phiCN*rho2, alphaEqn.H:248 -- built once per SUB-STEP,
        // from the flux the last corrector left. The sub-cycle then time-weights these.
        deviceMassFlux(nIf, alphaPhiInt, *li.phiCNInt, li.rho1, li.rho2, rpInt);
        deviceMassFlux(nBf, alphaPhiBnd, *li.phiCNBnd, li.rho1, li.rho2, rpBnd);
        // ...and on the pair, whose mass flux the momentum equation reads like any other face's
        if (li.cyc && li.cyc->n > 0 && ctl.rhoPhiIf && li.phiCNIf && li.alphaPhiIf)
        {
            deviceMassFlux(li.cyc->n, *li.alphaPhiIf, *li.phiCNIf, li.rho1, li.rho2, *ctl.rhoPhiIf);
        }
    };

    deviceAlphaEqnSubCycle(ctl.nAlphaSubCycles, totalDeltaT, alpha1, alpha1Old,
                           rhoPhiInt, rhoPhiBnd, step);

    // ...and mixture.correct() ONCE MORE after the whole sub-cycle (alphaEqnSubCycle.H:36-38), so that
    // the momentum equation is built on the NEW density. Skipping it builds UEqn on the density the
    // step started with, which at a water/air interface is wrong by a factor of 1000 and converges.
    correctMixture(alpha1, false);
}

} // namespace brae
