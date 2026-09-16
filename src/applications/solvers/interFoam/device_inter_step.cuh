#pragma once
// ONE WHOLE interFoam TIME STEP on the device: the alpha sub-cycle, mixture.correct(), the momentum
// equation, and the pressure corrector.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/interFoam/interFoam.C:88-140
//   host:      src/applications/solvers/interFoam/inter_solve_cpp.cu, runTimeStep -- the ORACLE, and
//              the path that reaches alpha 3.4346e-09, p_rgh 2.29e-06 and U 3.31e-06 relative on
//              damBreak against real OpenFOAM.
//   tests:     tests/test_device_inter_step.cu
//
// Everything under this is separately landed and separately gated. What this file owns is THE LOOP
// ORDER, and interFoam's order is not numerics -- it is which field each equation sees:
//
//   ALPHA ADVANCES FIRST, before the momentum predictor, and mixture.correct() runs between them. So
//   UEqn is built on the NEW density. The single-phase order -- momentum first -- converges perfectly
//   well, on last step's density, which at a water/air interface is wrong by a factor of 1000.
//
//   rhoPhi COMES OUT OF THE ALPHA EQUATION and is what fvm::div in UEqn is built on. It is the
//   MULES-limited MASS flux, not the volumetric phi, and not rho*phi rebuilt afterwards.
//
//   THE MOMENTUM MATRIX IS ASSEMBLED AND RELAXED EVEN WHEN momentumPredictor IS OFF, because pEqn
//   needs A() and H(). damBreak sets `momentumPredictor no`, so on the case this port is gated against
//   the predictor never solves and the matrix is still the whole basis of the pressure equation.
//
//   THE PRESSURE CORRECTOR IS THE LAST THING, and the fields it writes -- phi, U, p_rgh, p -- are what
//   the NEXT step's alpha equation advects with.
//
// WHAT STAYS ON THE HOST is what has stayed on the host throughout: the boundary conditions. alpha's
// patch values and its contact angle, fvm::div's per-patch coefficients for the alpha pre-solve, U's
// boundary for the momentum matrix, and p_rgh's for the pressure laplacian. All four arrive through
// `hooks`. Everything that scales with the CELL COUNT runs on the device.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_inter_alpha_step.cuh"
#include "device_inter_pressure_step.cuh"
#include "device_inter_ueqn.cuh"
#include "UEqn.cuh"
#include "pEqn.cuh"
#include <functional>

namespace brae {

struct DeviceInterStepHooks
{
    DeviceInterAlphaHooks    alpha;      // alpha's patch values + contact angle, and fvm::div's coeffs
    DeviceInterPressureHooks pressure;   // p_rgh's laplacian boundary coefficients, after constrainPressure

    // U's boundary, rebuilt from whatever the device last wrote. It is refreshed once per step rather
    // than per corrector because interFoam's PIMPLE loop evaluates it there.
    std::function<void(const DeviceBuffer<scalar>& Ux,
                       const DeviceBuffer<scalar>& Uy,
                       const DeviceBuffer<scalar>& Uz,
                       DeviceVectorBoundary&       dbU)> updateUBoundary;

    // The face fields that depend on the NEW alpha, over the mesh's FULL face array:
    // surfaceTensionForce() = interpolate(sigma*K)*snGrad(alpha1), and snGrad(rho). Both are rebuilt
    // AFTER the alpha step and BEFORE the momentum equation, because both equations read them and the
    // interface has just moved. nuEff is the mixture's, which mixture.correct() has just written.
    std::function<void(const DeviceBuffer<scalar>& alpha1,
                       const DeviceBuffer<scalar>& K,
                       const DeviceBuffer<scalar>& rho,
                       DeviceBuffer<scalar>&       stf,
                       DeviceBuffer<scalar>&       snGradRho,
                       DeviceBuffer<scalar>&       nuEffCell,
                       DeviceBuffer<scalar>&       nuEffBnd,
                       // snGrad(p_rgh) over the full face array, needed ONLY when the case runs a
                       // momentum predictor: it is explicit there and implicit in the pressure
                       // equation, and its boundary half is per-patch dispatch like everything else.
                       DeviceBuffer<scalar>&       snGradPrgh)> interfaceForces;
};

// Intermediates the step is willing to hand back, for a gate that has to localise a disagreement
// rather than only measure one. A whole step is a dozen operators and a final U that is 60% out cannot
// say which of them did it; these can. Null costs nothing. This is the of-instrument approach applied
// to brae's own code -- the same reason tools/dumpInterFoam exists for OpenFOAM's.
struct DeviceInterStepTaps
{
    DeviceBuffer<scalar> rAU;
    DeviceBuffer<scalar> HbyA[3];
    DeviceBuffer<scalar> phiHbyAInt;      // AFTER the two interFoam terms
    DeviceBuffer<scalar> UEqnDiag;        // relaxed, as A() takes it
    DeviceBuffer<scalar> UEqnSourceX;
};

struct DeviceInterStepControls
{
    DeviceInterAlphaControls  alpha;
    DeviceMulesControls       mules;
    // The alpha equation's per-step settings -- cAlpha, deltaN and the two flux schemes. The flux
    // pointers and deltaT in it are IGNORED and filled by the step, because the sub-cycle owns the
    // time step and the caller owns the flux.
    DeviceAlphaStepInput      alphaInput;

    DeviceAlphaSolverControls momentum;      // the case's fvSolution entry for U
    DeviceAlphaSolverControls pressure;      // ...and for p_rgh
    // pimple.correct() -- fvSolution's PIMPLE/nCorrectors. THE WHOLE OF pEqn.H REPEATS, not just the
    // solve: interFoam.C wraps `#include "pEqn.H"` in `while (pimple.correct())`, so rAU, HbyA,
    // phiHbyA, phig, the solve, U and phi are all rebuilt each pass, each from the U and phi the last
    // one left. damBreak asks for THREE and this port ran ONE, which is a real omission -- but it is
    // NOT the cause of the 60% gap in U that this file is under diagnosis for: MEASURED, going from
    // one corrector to three moved U from 1.6039e-01 to 1.5848e-01 of a field whose max is 2.65e-01,
    // about one per cent of the gap. Written here because the first version of this comment claimed
    // the opposite before the measurement came back.
    int    nCorrectors       = 1;
    bool   momentumPredictor = true;         // damBreak sets this OFF
    scalar relaxU            = 1;
    bool   relaxEquationU    = false;        // the case NAMES a factor -- see gpu::MomentumInput
    bool   needReference     = false;
    int    pRefCell          = 0;
    scalar pRefValue         = 0;
    // polyMesh::solutionD() -- which coordinate directions the vector equation is SOLVED in. On a 2-D
    // case the empty direction is knocked out, and fvMatrix::H()'s validComponents block skips it. This
    // port hardcoded all three as valid, which on damBreak -- empty front and back -- told H() to solve
    // a direction OpenFOAM never does.
    int    solutionD[3]      = {1, 1, 1};

    // constrainHbyA's per-face `assignable` mask, which the shared pressure predictor requires.
    // assignable() is NOT fixesValue(): slip and inletOutlet are non-assignable without fixing one,
    // and damBreak's atmosphere is pressureInletOutletVelocity.
    const DeviceBuffer<int>*  takeUAtBoundary = nullptr;
};

// Every field is in and out: this is a time step, and the next one starts from what this leaves.
// `alpha1Old`, `UOld` and `phiOld` are the previous time level and are not written.
void deviceInterStep(
    const DeviceMesh&                dm,
    scalar                           deltaT,
    const DeviceInterStepControls&   ctl,
    const DevicePhaseProperties&     props,
    const DeviceInterStepHooks&      hooks,
    // geometry that does not change
    const DeviceBuffer<scalar>&      gh,
    const DeviceBuffer<scalar>&      ghf,        // full face array
    const DeviceBuffer<scalar>&      magSf,      // full face array
    // the state
    DeviceBuffer<scalar>&            alpha1,
    const DeviceBuffer<scalar>&      alpha1Old,
    DeviceBuffer<scalar>&            UX,
    DeviceBuffer<scalar>&            UY,
    DeviceBuffer<scalar>&            UZ,
    const DeviceBuffer<scalar>&      UOldX,
    const DeviceBuffer<scalar>&      UOldY,
    const DeviceBuffer<scalar>&      UOldZ,
    DeviceBuffer<scalar>&            phiInt,
    DeviceBuffer<scalar>&            phiBnd,
    // phi.oldTime(), and U's per-face fixesValue mask. Both are fvc::ddtCorr's, which pEqn.H adds to
    // phiHbyA: phi and U are separate state, so after a step the stored flux and the flux you would
    // get by interpolating the stored velocity DO NOT agree, and that difference is real information
    // the pressure equation needs. A solver that drops it decouples pressure and velocity slowly.
    const DeviceBuffer<scalar>&      phiOldInt,
    const DeviceBuffer<scalar>&      phiOldBnd,
    const DeviceBuffer<int>&         bndUFixesValue,
    DeviceBuffer<scalar>&            p_rgh,
    DeviceBuffer<scalar>&            p,
    // carried across steps so the calculateK pass count matches OpenFOAM's
    DeviceBuffer<scalar>&            nHatfInt,
    DeviceBuffer<scalar>&            nHatfBnd,
    DeviceBuffer<scalar>&            alpha1Bnd,
    DeviceBuffer<scalar>&            K,
    const DeviceBuffer<int>&         bndAlphaFixesValue,
    const DeviceBuffer<int>&         bndAlphaFlag,
    DeviceVectorBoundary&            dbU,
    // what the step leaves for the next one to read
    DeviceBuffer<scalar>&            rho,
    DeviceBuffer<scalar>&            mu,
    DeviceBuffer<scalar>&            nu,
    DeviceBuffer<scalar>&            rhoPhiInt,
    DeviceBuffer<scalar>&            rhoPhiBnd,
    DeviceInterStepTaps*             taps = nullptr);

} // namespace brae
