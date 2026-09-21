#pragma once
// One alpha CORRECTOR on the device -- the EXPLICIT MULES path.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/VoF/alphaEqn.H:157-226 (the corrector loop),
//              :208-220 (MULES::explicitSolve), :248 (rhoPhi)
//   host:      src/applications/solvers/interFoam/alpha_eqn_cpp.cu, alphaEqnStep -- the ORACLE, and
//              itself gated end to end against real OpenFOAM: damBreak's alpha to 3.4346e-09 over
//              five steps (tests/interfoam_dambreak_vs_openfoam.sh).
//   tests:     tests/test_device_alpha_step.cu
//
// This is the first piece of interFoam that RUNS A WHOLE STEP on the GPU rather than one operator.
// Everything it calls is already gated on its own: deviceCompressionFlux and deviceAlphaFaceFlux
// (bit-identical to the host), deviceInterfaceNormalFlux and deviceInterfaceCurvature (2.8e-15 /
// 3.8e-15), deviceMulesDonorFlux (bit-identical) and deviceMulesLimiter (3.7e-15). What this file adds
// is the ORDER -- and the order is the part of alphaEqn.H that carries the physics:
//
//   phic BEFORE phir, phir from the nHatf the PREVIOUS corrector left, alpha2 rebuilt from the CURRENT
//   alpha1, the nested flux composed with both minus signs, MULES limiting alphaPhiUn IN PLACE, and
//   mixture.correct() at the BOTTOM of the corrector, not the top.
//
// WHAT IS DELIBERATELY NOT HERE.
//
//   THE BOUNDARY CONDITIONS. alpha1's patch values are handed in. Evaluating them is branchy
//   per-patch dispatch over inletOutlet, zeroGradient, alphaContactAngle and empty, it touches a few
//   hundred faces of a mesh with tens of thousands of cells, and putting it on the device would move a
//   fact about boundary conditions into a kernel. Same judgement, same reason, as correctContactAngle
//   in device_interface_properties.cuh.
//
//   ...AND THAT IS WHY THIS IS ONE CORRECTOR AND NOT THE WHOLE nAlphaCorr LOOP. OpenFOAM re-evaluates
//   alpha's boundary at the END of every MULES solve (MULESTemplates.C:181) and the next corrector's
//   flux, gradient and limiter all read it. A call that looped internally could not do that, and would
//   silently run correctors 2..n against the boundary corrector 1 started from. MEASURED, on a rotating
//   blob at nAlphaCorr = 2: one corrector agreed with the host to 2.2e-16 and two to 5.7e-08, which is
//   the stale boundary and nothing else. So the loop is the caller's, and between calls the caller
//   evaluates alpha1's patch values and the boundary nHatf exactly as it does between time steps.
//
//   MULESCorr. damBreak and twelve other tutorials set it, and it needs a linear solve for the
//   implicit upwind pre-solve plus CMULES' own limiter. Both are real work and both are next; this
//   file is the explicit path, which is what the other thirty-one tutorials run.
//
// The step is NOT bit-identical to the host and cannot be: MULES' five face-to-cell budgets are
// gathers here and a face loop there, so they sum in a different order, and lambda inherits that.
// tests/test_device_alpha_step.cu therefore holds this to MULES' own contract exactly -- alpha in
// [0,1] -- and to the host within a measured tolerance.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include <functional>
#include "device_mesh.cuh"
#include "device_cyclic.cuh"
#include "device_mules.cuh"

namespace brae {

// The face interpolation for one of the two fluxes. `interfaceCompression` is absent on purpose: it is
// not a limiter variant but a different scheme entirely, the host runs it and the device driver refuses
// it by name -- a device enum that mapped it to vanLeer, or to linear as the driver's mapping did until
// the host ported it, would be the exact substitution this project keeps finding.
// The integer values matter: deviceAlphaCyclicFlux takes the scheme as an int, and a pair's face must
// take the same weight an internal face does.
enum class DeviceAlphaScheme
{
    linear  = 0,
    upwind  = 1,
    vanLeer = 2,
    interfaceCompression = 3
};

struct DeviceAlphaStepInput
{
    const DeviceBuffer<scalar>* phiInt    = nullptr;   // the volumetric flux
    const DeviceBuffer<scalar>* phiBnd    = nullptr;
    const DeviceBuffer<scalar>* phiCNInt  = nullptr;   // off-centred; == phi for Euler
    const DeviceBuffer<scalar>* phiCNBnd  = nullptr;

    // THE MESH'S PERIODIC PAIR, whose faces are in neither list above. `cyc->phi` holds the pair's
    // VOLUMETRIC flux; phiCNIf its off-centred twin; alphaPhiIf is the alpha flux there, in on the
    // MULESCorr path (the pre-solve's) and out on every path. Every kernel these reach is gated
    // against the host in tests/test_device_mules_cyclic_vs_host.cu and
    // tests/test_device_cyclic_laplacian_vs_host.cu.
    DeviceCyclic*               cyc        = nullptr;
    const DeviceBuffer<scalar>* phiCNIf    = nullptr;
    DeviceBuffer<scalar>*       alphaPhiIf = nullptr;
    // nHatf on the pair, from the same mixture.correct() that produced nHatfInt: phir is phic*nHatf,
    // and a coupled face is the one kind alphaEqn.H:79-89 leaves compressed.
    const DeviceBuffer<scalar>* nHatfIf = nullptr;

    scalar cAlpha     = 0;
    scalar deltaT     = 0;
    scalar rho1       = 0;
    scalar rho2       = 0;
    scalar deltaN     = 0;                             // 1e-8/cbrt(average(V)), interfaceProperties.C

    DeviceAlphaScheme alphaScheme  = DeviceAlphaScheme::vanLeer;
    DeviceAlphaScheme alpharScheme = DeviceAlphaScheme::linear;

    // THE SEMI-IMPLICIT PATH, `MULESCorr yes` -- 13 of the 44 shipped tutorials, damBreak among them
    // (nAlphaCorr 2, nLimiterIter 5). The caller runs deviceAlphaPreSolve ONCE per sub-cycle before
    // the first corrector and hands its flux in as alphaPhi10; each corrector then computes what the
    // high-order flux ADDS to that, limits the addition with CMULES and applies it. With MULESCorr
    // off, alphaPhi10 is an output and the corrector limits the whole flux instead.
    bool MULESCorr = false;

    // Which corrector this is. It selects the under-relaxation weight OpenFOAM applies from the SECOND
    // corrector onward (alphaEqn.H:195-205), and it is the caller's loop index rather than a scalar
    // because the 0.5 is OpenFOAM's number and not a tunable.
    int aCorr = 0;

    // A MOVING MESH: mesh.Vsc() and mesh.Vsc0() at this sub-cycle's clock -- the volumes MULES limits
    // and solves with and the pre-solve's fvm::ddt takes (fvMeshGeometry.C, MULESTemplates.C:248 and
    // :397-417, fvcSurfaceIntegrate.C:77). Null == a mesh that does not move; both or neither.
    const DeviceBuffer<scalar>* Vsc  = nullptr;
    const DeviceBuffer<scalar>* Vsc0 = nullptr;

    // phic FORMED BY THE CALLER, ahead of a geometry change inside the step. alphaEqn.H forms
    // phic = cAlpha*|phi/magSf| ONCE, at :59, before the pre-solve and before anything interpolates
    // across the mesh; with fixed geometry it is the same field wherever it is formed, which is why
    // this corrector forms it itself. A cyclicACMI whose scale moves is the exception: OpenFOAM
    // rescales the areas lazily, INSIDE the pre-solve, so phic is the one built on the areas the step
    // started with (the host's alphaEqnStep says what that is worth on the face that just opened: 41.7
    // for 4e-9). All three or none; null = form it here, which is every case but that one.
    const DeviceBuffer<scalar>* phicIntPre = nullptr;
    const DeviceBuffer<scalar>* phicBndPre = nullptr;
    const DeviceBuffer<scalar>* phicIfPre  = nullptr;
};

// alpha1's boundary, evaluated by the caller. `nHatf` is the interface flux the previous
// mixture.correct() left, contact angle already applied -- the step reads it at the TOP of the first
// corrector and rewrites it at the BOTTOM of every one, which is what makes the number of calculateK
// passes match OpenFOAM's. The curvature is a fixed point in those passes: on capillaryRise it walks
// 7070.5 -> 8659.4 -> 9353.1 -> 9681.2, so one pass too few is 18% low at the contact line.
struct DeviceAlphaBoundary
{
    const DeviceBuffer<scalar>* alpha1     = nullptr;  // patch values, boundary-face order
    const DeviceBuffer<scalar>* nHatfBnd   = nullptr;  // read only, like nHatfInt
    const DeviceBuffer<int>*    fixesValue = nullptr;  // 1 where the patch fixes a value
    const DeviceBuffer<int>*    flag       = nullptr;  // 0 ordinary, 1 empty, 2 wedge
    // alpha1's boundaryField().updateCoeffs(), for a condition whose value a MODEL supplies (waveAlpha).
    // On the explicit path it is the correctBoundaryConditions() that OPENS MULES::explicitSolve
    // (MULESTemplates.C:168): AFTER the high-order flux has been built on the values the last update
    // left, BEFORE the limiter. The call REWRITES the buffer `alpha1` above points at, and the
    // corrector then builds the bounded flux's boundary on the new values -- upwind's boundary flux is
    // phi_b*psi_b -- so the correction is no longer zero there. Null on a case with no such patch, and
    // then nothing below changes by a bit. See inter_waves_cpp.cuh for what was and was not measured.
    std::function<void(const DeviceBuffer<scalar>& alpha1)> updateModelled;
};

// ONE corrector of alphaEqn.H:157-220 -- the flux and the MULES solve, and NOT mixture.correct().
//
// `alpha1` is advanced in place; `alpha1Old` is the sub-step's starting value and is not written. The
// corrector does NOT reset alpha1 to it, and neither does OpenFOAM: alphaEqn.H resets alpha1 once per
// sub-cycle and each corrector continues from the last one's answer. `nHatfInt` is READ, not written --
// it is what the last mixture.correct() left.
//
// TWO THINGS THE CALLER DOES BETWEEN CALLS, both of which OpenFOAM does between correctors:
//
//   evaluate alpha1's boundary   -- MULES::explicitSolve ends with psi.correctBoundaryConditions()
//   deviceInterfaceCorrect       -- mixture.correct(), alphaEqn.H:225, which reads that NEW boundary.
//
// ...and on the MULESCorr path, ONCE before the loop, deviceAlphaPreSolve followed by a
// deviceInterfaceCorrect of its own -- alphaEqn.H:151-153. That extra mixture.correct() is inside the
// MULESCorr block and before the correctors, and it is a calculateK pass in its own right. Missing it
// put damBreak at 1.05e-07 against OpenFOAM where the full sequence gives 3.4346e-09, thirty times
// worse, and only damBreak could show it because it is the case that sets MULESCorr.
//
// The second was inside this call once. It read the PRE-solve boundary, which put nHatf 2.885e-04 out
// on a field whose largest value is 1.7e-03 -- see device_interface_properties.cuh for the measurement.
//
// rhoPhi is NOT computed here either: alphaEqn.H builds it once, from the flux the LAST corrector left,
// so a per-corrector call would build it n times and throw the first n-1 away. The caller makes it with
// deviceMassFlux (device_alpha_flux.cuh), which is what this file would have called anyway.
void deviceAlphaCorrector(
    const DeviceMesh&              dm,
    DeviceBuffer<scalar>&          alpha1,
    const DeviceBuffer<scalar>&    alpha1Old,
    const DeviceAlphaStepInput&    in,
    const DeviceAlphaBoundary&     bnd,
    const DeviceMulesControls&     mulesCtl,
    const DeviceBuffer<scalar>&    nHatfInt,       // read only: what the last mixture.correct() left
    // OUT with MULESCorr off -- the limited high-order flux. IN AND OUT with it on: in as the flux the
    // pre-solve (or the previous corrector) left, out with this corrector's limited correction added.
    DeviceBuffer<scalar>&          alphaPhi10Int,
    DeviceBuffer<scalar>&          alphaPhi10Bnd);

} // namespace brae
