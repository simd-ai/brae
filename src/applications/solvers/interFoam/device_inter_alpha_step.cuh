#pragma once
// ONE TIME STEP'S ALPHA HALF, end to end on the device: the sub-cycle, the correctors inside it, the
// MULESCorr pre-solve where the case asks for one, mixture.correct() at every stage OpenFOAM runs it,
// and rhoPhi out at the end for the momentum equation to be built on.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/VoF/alphaEqnSubCycle.H:1-45, alphaEqn.H:1-260,
//              applications/solvers/multiphase/interFoam/interFoam.C:96-104 (the call site)
//   host:      src/applications/solvers/interFoam/inter_solve_cpp.cu (alphaEqnSubCycle) and
//              alpha_eqn_cpp.cu (alphaEqnStep) -- the ORACLE, and the path that reaches 3.4346e-09 in
//              alpha on damBreak against real OpenFOAM.
//   tests:     tests/test_device_inter_alpha_step.cu
//
// This is the first function in the interFoam port that a SOLVER LOOP could call rather than a gate.
// Everything under it is separately landed and separately gated: deviceAlphaEqnSubCycle, the explicit
// and MULESCorr branches of deviceAlphaCorrector, deviceAlphaPreSolve, deviceMulesLimitCorr,
// deviceInterfaceCorrect and deviceMixtureCorrect. What this file owns is the SEQUENCE, and interFoam's
// sequence has three placements that are not obvious and that nothing underneath can check:
//
//   mixture.correct() runs at the BOTTOM of every corrector (alphaEqn.H:225), ONCE MORE inside the
//   MULESCorr block before the correctors begin (alphaEqn.H:151-153), and ONCE MORE AGAIN after the
//   whole sub-cycle (alphaEqnSubCycle.H:36-38) so that UEqn is built on the new density. Each is a
//   calculateK pass, and the curvature is a FIXED POINT in those passes -- on capillaryRise it walks
//   7070.5 -> 8659.4 -> 9353.1 -> 9681.2. Missing the middle one put damBreak at 1.05e-07 against
//   OpenFOAM where the full sequence gives 3.4346e-09.
//
// WHAT STAYS ON THE HOST, and it is the same line this port has drawn throughout: the BOUNDARY
// CONDITIONS. Evaluating alpha1's patch values is branchy dispatch over inletOutlet, zeroGradient,
// alphaContactAngle and empty; correctContactAngle touches two faces of eight hundred on capillaryRise
// and is stateful and iterative; and fvm::div's per-patch valueInternalCoeffs are the same kind of
// work. All three arrive through `hooks` below. Everything that scales with the CELL COUNT is here.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_mules.cuh"
#include "device_alpha_step.cuh"
#include "device_alpha_presolve.cuh"
#include "device_two_phase_mixture.cuh"
#include <functional>

namespace brae {

// The host's side of the step. Called exactly where OpenFOAM calls the things they stand for.
struct DeviceInterAlphaHooks
{
    // alpha1's patch values, and the boundary interface normal correctContactAngle has acted on, from
    // whatever the device last wrote. Called after every MULES solve -- MULESTemplates.C:181 ends the
    // solve with psi.correctBoundaryConditions(), and the next corrector's flux, gradient and limiter
    // all read the result.
    std::function<void(const DeviceBuffer<scalar>& alpha1,
                       DeviceBuffer<scalar>&       alpha1Bnd,
                       DeviceBuffer<scalar>&       nHatfBnd)> updateBoundary;

    // alpha1's patch values ALONE, with no curvature pass behind them. Optional; when absent the step
    // falls back to updateBoundary, which is only right on a case with no contact angle.
    //
    // The two are different calls because interfaceProperties::correct() has a SIDE EFFECT at a
    // contact-angle wall: it rewrites the patch's gradient (interfaceProperties.C:97), and the gradient
    // is a fixed-point iteration in its passes -- 7070.5, 8659.4, 9353.1, 9681.2 on capillaryRise. So
    // the NUMBER of passes is part of the answer. The sub-step's reset of alpha1 needs fresh patch
    // values and is not a mixture.correct(); routing it through updateBoundary gave the device five
    // curvature passes a step where OpenFOAM takes three, and put it 1.9% out in U after ONE step on
    // capillaryRise with every other term exact. damBreak has no contact angle and could not see it.
    std::function<void(
        const DeviceBuffer<scalar>& alpha1,
        DeviceBuffer<scalar>& alpha1Bnd)> refreshBoundary;

    // fvm::div(phiCN, alpha1)'s internalCoeffs and boundaryCoeffs, flattened in boundary-face order.
    // Only reached when MULESCorr is on; a case without it never needs them and passing none is fine.
    std::function<void(const DeviceBuffer<scalar>& alpha1,
                       DeviceBuffer<scalar>&       iC,
                       DeviceBuffer<scalar>&       bC)> divCoeffs;
};

struct DeviceInterAlphaControls
{
    int  nAlphaSubCycles = 1;       // alphaControls.H, get<label>
    int  nAlphaCorr      = 1;
    bool MULESCorr       = false;
    DeviceAlphaSolverControls preSolve;   // the case's fvSolution entry for alpha, MULESCorr only
    // OUT, optional: alpha2's PATCH values, assigned where alphaEqn.H assigns alpha2 (lines 151 and
    // 223) and NOT at the mixture.correct() after the sub-cycle, which leaves alpha2 alone. They are
    // one contact-angle pass older than alpha1's -- see deviceBoundaryRho.
    DeviceBuffer<scalar>* alpha2BndOut = nullptr;
};

// `alpha1` is advanced in place from `alpha1Old`, which is never written. `rho`, `mu` and `nu` come out
// as mixture.correct() left them after the sub-cycle, ready for UEqn. `nHatfInt` and `K` are carried in
// and out so the calculateK pass count matches OpenFOAM's across time steps.
void deviceInterAlphaStep(
    const DeviceMesh&                dm,
    DeviceBuffer<scalar>&            alpha1,
    const DeviceBuffer<scalar>&      alpha1Old,
    scalar                           totalDeltaT,
    const DeviceAlphaStepInput&      in,           // deltaT is set per sub-step and is ignored here
    const DeviceMulesControls&       mulesCtl,
    const DeviceInterAlphaControls&  ctl,
    const DevicePhaseProperties&     props,
    const DeviceInterAlphaHooks&     hooks,
    DeviceBuffer<scalar>&            alpha1Bnd,    // in: the starting boundary; out: the final one
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
    DeviceBuffer<scalar>&            nu);

} // namespace brae
