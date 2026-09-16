// One time step's alpha half -- see device_inter_alpha_step.cuh for the three mixture.correct()
// placements and for what stays on the host.
#include "device_inter_alpha_step.cuh"
#include "device_alpha_subcycle.cuh"
#include "device_alpha_flux.cuh"
#include "device_interface_properties.cuh"
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
    auto correctMixture = [&](const DeviceBuffer<scalar>& a)
    {
        hooks.updateBoundary(a, alpha1Bnd, nHatfBnd);
        deviceInterfaceCorrect(dm, a, alpha1Bnd, nHatfBnd, in.deltaN, nHatfInt, K);
        alpha2.resize(static_cast<std::size_t>(nC));
        rho.resize(static_cast<std::size_t>(nC));
        mu.resize(static_cast<std::size_t>(nC));
        nu.resize(static_cast<std::size_t>(nC));
        deviceMixtureCorrect(a.data(), nC, props,
                             alpha2.data(), rho.data(), mu.data(), nu.data());
    };

    DeviceAlphaEqnStep step =
        [&](const DeviceBuffer<scalar>& subOld, scalar dtSub, DeviceBuffer<scalar>& alpha,
            DeviceBuffer<scalar>& rpInt, DeviceBuffer<scalar>& rpBnd)
    {
        DeviceAlphaStepInput li = in;
        li.deltaT    = dtSub;
        li.MULESCorr = ctl.MULESCorr;

        // alphaEqn.H:97 -- alpha1 is reset to the sub-step's old time ONCE per sub-step, not once per
        // corrector. `alpha` is the sub-cycle's output buffer; it is written here rather than aliased
        // to alpha1, so this lambda never assumes the two are the same object.
        alpha.resize(static_cast<std::size_t>(nC));
        cudaMemcpy(alpha.data(), subOld.data(), sizeof(scalar)*nC, cudaMemcpyDeviceToDevice);
        hooks.updateBoundary(alpha, alpha1Bnd, nHatfBnd);

        if (ctl.MULESCorr)
        {
            // alphaEqn.H:103-155: the implicit upwind pre-solve, ONCE per sub-step, then a
            // mixture.correct() of its own before the correctors begin.
            hooks.divCoeffs(alpha, iC, bC);
            deviceAlphaPreSolve(dm, alpha, subOld, *li.phiCNInt, iC, bC, dtSub, ctl.preSolve,
                                alphaPhiInt, alphaPhiBnd);
            correctMixture(alpha);                              // alphaEqn.H:151-153
        }

        for (int aCorr = 0; aCorr < ctl.nAlphaCorr; ++aCorr)
        {
            li.aCorr = aCorr;
            DeviceAlphaBoundary db;
            db.alpha1     = &alpha1Bnd;
            db.nHatfBnd   = &nHatfBnd;
            db.fixesValue = &bndFixesValue;
            db.flag       = &bndFlag;
            deviceAlphaCorrector(dm, alpha, subOld, li, db, mulesCtl, nHatfInt,
                                 alphaPhiInt, alphaPhiBnd);
            correctMixture(alpha);                              // alphaEqn.H:225
        }

        // rhoPhi = alphaPhi10*(rho1 - rho2) + phiCN*rho2, alphaEqn.H:248 -- built once per SUB-STEP,
        // from the flux the last corrector left. The sub-cycle then time-weights these.
        deviceMassFlux(nIf, alphaPhiInt, *li.phiCNInt, li.rho1, li.rho2, rpInt);
        deviceMassFlux(nBf, alphaPhiBnd, *li.phiCNBnd, li.rho1, li.rho2, rpBnd);
    };

    deviceAlphaEqnSubCycle(ctl.nAlphaSubCycles, totalDeltaT, alpha1, alpha1Old,
                           rhoPhiInt, rhoPhiBnd, step);

    // ...and mixture.correct() ONCE MORE after the whole sub-cycle (alphaEqnSubCycle.H:36-38), so that
    // the momentum equation is built on the NEW density. Skipping it builds UEqn on the density the
    // step started with, which at a water/air interface is wrong by a factor of 1000 and converges.
    correctMixture(alpha1);
}

} // namespace brae
