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
#include <vector>

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

    // waveAlpha: the patch values a wave MODEL supplies, at the clock of the sub-cycle it is called in
    // (1-based). A sub-cycle is its own time index to OpenFOAM, so the model updates in each -- see
    // inter_waves_cpp.cuh. Called where OpenFOAM's first updateCoeffs of the pass fires: ahead of the
    // pre-solve's assembly under MULESCorr, and otherwise between the high-order flux and the limiter
    // (DeviceAlphaBoundary::updateModelled). Null on a case with no such patch.
    std::function<void(
        int subCycle,
        const DeviceBuffer<scalar>& alpha1,
        DeviceBuffer<scalar>& alpha1Bnd)> updateModelledBoundary;

    // fvMesh::Vsc() and Vsc0() at the clock of the sub-cycle it is called in (1-based), filled into
    // the two buffers the step then hands MULES and the pre-solve. A sub-cycle is its own time index
    // to OpenFOAM, and Vsc interpolates between V0 and V by where in the step that index falls
    // (fvMeshGeometry.C) -- so the volumes change WITHIN a step, which is why this is a call and not
    // a pair of pointers. Null on a mesh that does not move, where Vsc is V and Vsc0 is V.
    std::function<void(
        int subCycle,
        DeviceBuffer<scalar>& Vsc,
        DeviceBuffer<scalar>& Vsc0)> subCycleVolumes;

    // THE STEP'S GEOMETRY CHANGE, where the host's alphaEqnStep has it (alpha_eqn_cpp.cu,
    // geometryUpdate): AFTER phic is formed and BEFORE the pre-solve assembles. It exists for a
    // cyclicACMI whose `scale` moves with time -- OpenFOAM rescales the pair's areas lazily, at the
    // pre-solve's first updateCoeffs -- and the callee both rescales and re-uploads whatever this step
    // reads from the mesh and the pair, IN PLACE (`dm` and the pair are held by reference here). It is
    // called at every sub-step and every outer corrector; landing once per TIME STEP is the callee's
    // job, as it is the host hook's. Null on every other case, and the step is then bit for bit what it
    // was: phic stays the corrector's own.
    std::function<void()> geometryUpdate;
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
    // appended to, one record per pre-solve; null = not kept
    std::vector<DeviceSolverPerf>* preSolveLog = nullptr;
    // ...and the field that pre-solve LEAVES, plus its flux on the pair, for a gate bisecting the
    // implicit half against the corrector. Null in every production run.
    DeviceBuffer<scalar>* preSolveAlphaOut = nullptr;
    DeviceBuffer<scalar>* preSolveAlphaPhiIfOut = nullptr;
    // `alphaApplyPrevCorr yes` (alphaEqn.H:133-150, :228-236): the compression flux the correctors
    // ENDED on is cached, and the next pre-solve applies it, limited, before its own correctors run.
    // The cache is talphaPhi1Corr0 and it outlives the call -- it carries across SUB-CYCLES as well as
    // time steps -- so the CALLER owns it: two buffers, empty until the first alpha step has filled
    // them, which is what OpenFOAM's `.valid()` tests. Setting the switch without handing them in is
    // refused; this step used to ignore the switch altogether, and on damBreak at dt 5e-3 that put the
    // device 1.07e-02 of alpha from OpenFOAM -- to five digits, OpenFOAM's own distance from itself
    // with the switch off.
    bool alphaApplyPrevCorr = false;
    DeviceBuffer<scalar>* prevCorrInt = nullptr;
    DeviceBuffer<scalar>* prevCorrBnd = nullptr;
    // THE PAIR's mass flux out, for the momentum equation. The pair itself, its flux and its alpha flux
    // travel in DeviceAlphaStepInput, which is what the corrector reads.
    DeviceBuffer<scalar>* rhoPhiIf = nullptr;
    // nHatf ON THE PAIR, and it is the CALLER's buffer because it has to outlive the call. The first
    // corrector's phir reads the normal the LAST mixture.correct() left (alphaEqn.H:162) -- which,
    // without MULESCorr, is the previous TIME STEP's, since nothing runs between the two. Held here
    // for the same reason nHatfInt and nHatfBnd are the driver's: a buffer local to one step is empty
    // when the first corrector of a `MULESCorr no` case asks for it, and that was a refusal.
    DeviceBuffer<scalar>* nHatfIf = nullptr;
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
