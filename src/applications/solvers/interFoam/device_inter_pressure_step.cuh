#pragma once
// ONE PASS OF pEqn.H on the device -- rAU and HbyA, phiHbyA with interFoam's two extra terms, the
// p_rgh solve, then phi, U and p rebuilt from it.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/interFoam/pEqn.H:1-89
//   host:      src/applications/solvers/interFoam/inter_peqn_cpp.cu, pressureCorrector -- the ORACLE,
//              and the path that reaches 2.29e-06 relative in p_rgh and 3.31e-06 in U on damBreak
//              against real OpenFOAM.
//   tests:     tests/test_device_inter_pressure_step.cu
//
// ONE PASS, NOT THE NON-ORTHOGONAL LOOP, for the same reason deviceAlphaCorrector is one corrector:
// OpenFOAM re-evaluates p_rgh's boundary after every solve and the next pass's matrix and flux both
// read it, and a call that looped internally could not do that. The loop is the caller's.
//
// STAGES 1-4 ARE simpleFoam's pressurePredictor, REUSED. rAU = 1/A(), H(), HbyA, constrainHbyA and
// fvc::flux(HbyA) are the same operators, and A() takes the RELAXED diagonal with the boundary average
// added ON TOP of it -- OpenFOAM's relax() writes diag_ in place and A() is taken afterwards. What
// interFoam adds is stages 5 onward, and each is already landed and gated on its own.
//
// THE THREE THINGS THIS FILE OWNS ARE PLACEMENTS, not operators:
//
//   phig GOES INTO phiHbyA ON BOTH SIDES. fvc::div(phiHbyA) sums the boundary faces, so a wall's
//   buoyancy and surface tension reach the pressure equation's SOURCE through it -- measured at 9.4e+01
//   of 1.9e+02, half the source. On capillaryRise, where momentumPredictor is off, that is the only
//   route surface tension has into the solution at all.
//
//   THE SAME phig IS REUSED IN THE VELOCITY CORRECTION, (phig - flux)/rAUf, and it is the flux BEFORE
//   the division. Recomputing it there from a separately interpolated rAUf would be a different number.
//
//   p IS REBUILT FROM THE SOLVED p_rgh, not carried. p == p_rgh + rho*gh, and when p_rgh needs a
//   reference BOTH move: p is shifted and p_rgh is then rebuilt from the shifted p. Stopping after the
//   first leaves them disagreeing by a constant that surfaces in the next step's momentum source.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_inter_peqn.cuh"
#include "device_MRF.cuh"
#include "device_cyclic.cuh"
#include "device_alpha_presolve.cuh"   // DeviceAlphaSolverControls, the same shape of solver entry
#include "device_dilu.cuh"
#include "device_gamg_solver.cuh"
#include "device_pcg.cuh"   // DeviceSolverPerf
#include <functional>
#include <vector>

namespace brae {

struct DeviceInterPressureHooks
{
    // p_rgh's patch values, re-evaluated after each solve. The NEXT corrector's laplacian and its flux
    // both read them, exactly as OpenFOAM's p_rgh.correctBoundaryConditions() at the end of pEqn.H
    // leaves them for the next pass. Optional: a case with nCorrectors 1 never needs it.
    std::function<void(const DeviceBuffer<scalar>& p_rgh)> updateBoundary;

    // p_rgh's laplacian boundary coefficients, flattened in boundary-face order, AFTER
    // constrainPressure has set any fixedFluxPressure patch's gradient from phiHbyA. Branchy per-patch
    // dispatch over fixedFluxPressure, totalPressure and zeroGradient -- host work, as everywhere else.
    // `rAUfAll` is the FULL face array -- internal faces then the boundary patches -- and not just the
    // internal half. Both are needed: the laplacian's boundary coefficients scale with rAUf at the
    // patch, and constrainPressure divides by it per face. A hook given only the internal half has to
    // invent the boundary values, and standing in the first internal face's value for all of them put
    // U 60% out on damBreak.
    // `rAUCell` is rAU itself, which the hook needs for one thing rAUfAll cannot give it: a COUPLED
    // patch's rAUf. The device's face arrays have no coupled patch in them, and fvm::laplacian reads
    // gammaf on every patch, so the pair's is interpolated from the two cells there.
    // `cycJump` is an OUT parameter: p_rgh's JUMP on each coupled face, already signed, in the pair's
    // own face order -- and EMPTY on a pair without one. It belongs to this hook because a
    // porousBafflePressure recomputes its jump in updateCoeffs, from the flux and the viscosity at
    // THIS assembly, so it is as fresh as the coefficients beside it.
    std::function<void(const DeviceBuffer<scalar>& phiHbyAInt,
                       const DeviceBuffer<scalar>& phiHbyABnd,
                       const DeviceBuffer<scalar>& rAUfAll,
                       const DeviceBuffer<scalar>& rAUCell,
                       DeviceBuffer<scalar>&       iC,
                       DeviceBuffer<scalar>&       bC,
                       DeviceBuffer<scalar>&       cycJump)> pressureCoeffs;

    // p_rgh's STORED patch values, flattened in boundary-face order, as they stand after pressureCoeffs
    // -- what the host's gradOf(p_rgh) reads for the corrected laplacian's non-orthogonal correction
    // (inter_peqn_cpp.cu). Required when DeviceInterPressureInput::correctedLaplacian is set.
    std::function<void(DeviceBuffer<scalar>& bval)> boundaryValues;
};

struct DeviceInterPressureInput
{
    // THE MESH FLUX, over the mesh's FULL face array, on a moving mesh only (null otherwise). pEqn.H
    // ends with fvc::makeRelative(phi, U) on a mesh that moves -- phi -= meshPhi (fvcMeshPhi.C:76) --
    // so the flux the next alpha equation convects with is the one relative to the motion. The host
    // reference does it at inter_peqn_cpp.cu:980, after the velocity correction.
    const DeviceBuffer<scalar>* meshPhiAll = nullptr;

    // the face fields the buoyancy flux is built from, over the mesh's FULL face array
    const DeviceBuffer<scalar>* stf       = nullptr;   // surfaceTensionForce
    const DeviceBuffer<scalar>* ghf       = nullptr;
    const DeviceBuffer<scalar>* snGradRho = nullptr;
    const DeviceBuffer<scalar>* magSf     = nullptr;
    const DeviceBuffer<scalar>* rAUfAll   = nullptr;   // interpolate(rAU) over the full face array

    const DeviceBuffer<scalar>* rho = nullptr;         // cells, for interpolate(rho*rAU) and for p
    const DeviceBuffer<scalar>* gh  = nullptr;         // cells

    // fvc::ddtCorr(U, phi). Null on a start from rest, where there is no old flux to correct against.
    const DeviceBuffer<scalar>* ddtCorrInt = nullptr;
    // ...and its boundary half, with rho's patch values: live on any patch whose U fixes no value
    const DeviceBuffer<scalar>* ddtCorrBnd = nullptr;
    const DeviceBuffer<scalar>* rhoBndFace = nullptr;
    const DeviceBuffer<int>*    bndUFixesValue = nullptr;
    // the mesh's periodic pair: its laplacian coefficients go into the matrix and its off-diagonal into
    // every SpMV of the solve
    DeviceCyclic*               cyc = nullptr;
    // ...and phiHbyA ON THE PAIR. fvc::flux(HbyA) there is the two cells' HbyA interpolated and dotted
    // with Sf (deviceCyclicFlux), which gpu::pressurePredictor does not yet produce -- so a caller with
    // a pair and no such array is refused below rather than solved against a source that is missing the
    // pair's own flux.
    const DeviceBuffer<scalar>* phiHbyAIf = nullptr;
    // ...and the three fields phig is built from, ON THE PAIR. phig = (stf - ghf*snGrad(rho))*rAUf*magSf
    // is a whole-surfaceScalarField expression in pEqn.H:28-36, so a coupled patch has it like any
    // other; the device's face arrays exclude coupled patches, so they arrive here separately. rAUf is
    // NOT among them: it is fvc::interpolate(rAU) on that face, which the step builds from the pair's
    // own weights (deviceCyclicFaceValue).
    // fvc::ddtCorr ON THE PAIR, built by the caller beside the internal and boundary halves
    const DeviceBuffer<scalar>* ddtCorrIf   = nullptr;
    const DeviceBuffer<scalar>* stfIf       = nullptr;
    const DeviceBuffer<scalar>* ghfIf       = nullptr;
    const DeviceBuffer<scalar>* snGradRhoIf = nullptr;
    // MRFZoneList::makeRelative(phiHbyA), pEqn.H:19
    const std::vector<DeviceMRFZone>* mrf = nullptr;

    bool   needReference = false;
    int    pRefCell      = 0;
    scalar pRefValue     = 0;

    DeviceAlphaSolverControls solve;    // the case's fvSolution entry for p_rgh
    // `solver PCG; preconditioner DIC;` -- run OpenFOAM's own pair (deviceDICPCG) rather than the
    // BiCGStab this step grew up on. It decides where a relTol 0.05 solve STOPS, which on capillaryRise
    // was the whole of the device's 9.2e-04. `dic` is the level schedule: mesh-only, built once by the
    // caller with buildDeviceDilu, and required when pcgDIC is set.
    bool pcgDIC = false;
    DeviceDilu* dic = nullptr;
    // `solver GAMG;` -- OpenFOAM's own V-cycle on the device (device_gamg_solver.cuh), with that
    // entry's controls; null on an entry that names another solver. The hierarchy is the mesh's and
    // the caller owns it across steps. `dic` is required here too: it is the FINE level's schedule.
    const GamgControls* gamg = nullptr;
    DeviceGamgCache* gamgCache = nullptr;
    // the coarsest-level solve of every V-cycle, in order; null = not kept
    GamgSolveLog* gamgLog = nullptr;
    // appended to, one record per solve -- the solver's own initial/final residual and iteration
    // count, which is what a gate compares with OpenFOAM's "Solving for p_rgh" lines; null = not kept
    std::vector<DeviceSolverPerf>* solveLog = nullptr;

    // THE NON-ORTHOGONAL LOOP, as the host's pressureCorrector runs it (inter_peqn_cpp.cu):
    // nNonOrthogonalCorrectors + 1 assemblies and solves of p_rgh, each pass's correction from the
    // p_rgh the last one left. `solve`/`pcgDIC`/`gamg` above are the LAST pass's settings; every pass
    // before it takes the plain `p_rgh` entry below, because p_rgh.select(finalInnerIter()) (pEqn.H:50)
    // is Final only on the last non-orthogonal pass of the last corrector.
    int nNonOrthogonalCorrectors = 0;
    DeviceAlphaSolverControls solveInner;
    bool pcgDICInner = false;
    const GamgControls* gamgInner = nullptr;
    // the case's laplacianSchemes for the p_rgh laplacian: `corrected` and a `limited` coefficient
    // (0 = unlimited). Under `corrected` each pass adds the explicit correction from grad(p_rgh) --
    // Gauss linear, the only gradSchemes entry the device takes -- and keeps its face flux for
    // p_rghEqn.flux().
    bool correctedLaplacian = false;
    scalar snGradLimitCoeff = 0;
};

// `phiHbyAInt`/`phiHbyABnd` come in as fvc::flux(HbyA) -- what the shared pressure predictor left --
// and are advanced in place to phiHbyA + the two terms. `p_rgh` goes in as the previous value and comes
// out solved. `phi`, `U` and `p` are rebuilt from it.
//
// Returns the solver's final normalised residual, so a caller can refuse a pass that did not converge.
struct DevicePressureTaps
{
    DeviceBuffer<scalar> diag, upper, lower, source, iC, bC;
    // phig and rAUf ON THE PAIR, for comparing the two arms across a coupled face
    DeviceBuffer<scalar> phigIf, rAUfIf, ffIf, phiIf, phiHbyAIfPrePhig, cycJumpTap;
    // phiHbyA on the INTERNAL faces before phig, the same point the host's PressureTaps::phiHbyA is
    DeviceBuffer<scalar> nonOrthSource;       // the corrected laplacian's source correction
    DeviceBuffer<scalar> phigIntTap;
    DeviceBuffer<scalar> phigBndTap;
    DeviceBuffer<scalar> divPhiHbyA;   // fvc::div(phiHbyA), before the multiply by V          // phig on the internal faces, as the host taps it
    DeviceBuffer<scalar> phiHbyAIntPrePhig;
    DeviceBuffer<scalar> phiHbyABndPrePhig;
    std::vector<std::vector<scalar>> jumpHistory;   // one entry per assembly
};

scalar deviceInterPressureStep(
    const DeviceMesh&                  dm,
    const DeviceInterPressureInput&    in,
    const DeviceInterPressureHooks&    hooks,
    const DeviceBuffer<scalar>&        rAU,
    const DeviceBuffer<scalar>&        HbyAX,
    const DeviceBuffer<scalar>&        HbyAY,
    const DeviceBuffer<scalar>&        HbyAZ,
    DeviceBuffer<scalar>&              phiHbyAInt,
    DeviceBuffer<scalar>&              phiHbyABnd,
    DeviceBuffer<scalar>&              p_rgh,
    DeviceBuffer<scalar>&              phiInt,
    DeviceBuffer<scalar>&              phiBnd,
    DeviceBuffer<scalar>&              UX,
    DeviceBuffer<scalar>&              UY,
    DeviceBuffer<scalar>&              UZ,
    DeviceBuffer<scalar>&              p,
    DevicePressureTaps*                taps = nullptr);

} // namespace brae
