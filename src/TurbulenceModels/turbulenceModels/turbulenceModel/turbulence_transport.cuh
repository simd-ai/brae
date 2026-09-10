#pragma once
// The convection + diffusion half of ONE transported turbulence scalar, on the device.
//
// provenance:
//   openfoam: fvm::div(phi, vf) - fvm::laplacian(DEff, vf), as every RAS model's correct() writes it
//   brae:     src/TurbulenceModels/turbulenceModels/turbulenceModel/turbulence_transport.cu
//   tests:    every gate covering the mirror device closures -- it is the same code they already ran
//
// This lived inside the kEpsilon device closure, in an anonymous namespace, taking a KEpsilonInput. It
// is not kEpsilon-specific: k, epsilon, omega, nuTilda and the Langtry-Menter pair all transport the
// same way and differ only in their DIFFUSIVITY and their SOURCE. Keeping one copy is what stops the
// closures drifting on the things that have bitten here before -- the `corrected` laplacian's explicit
// half, limitedLinear's weights, and the limiter's own gradient.
//
// What is NOT here is the diffusivity itself. kEpsilon's is nut/sigma + nu with a constant sigma; the
// SST's is a blended alpha(F1)*nut + nu. Each closure builds gammaFace/gammaBnd its own way and hands
// them in, so this function has no opinion about the model.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include <string>
#include "device_dilu.cuh"    // DeviceDilu -- the case's preconditioner for these solves
#include "pEqn.cuh"               // PressureMatrix -- the assembled scalar object, shared not redefined

namespace brae {
namespace gpu {
namespace turbulence {

// The case's schemes for this one field. Every member is read from the case, never defaulted into a
// substitution: a closure that leaves `limitedLinear` false when the case named it runs upwind under
// the case's own name, which is the defect this project keeps finding.
struct TransportScheme
{
    const DeviceBuffer<scalar>* phiInt = nullptr;   // the equation's own flux (compressibly, the MASS flux)
    const DeviceBuffer<scalar>* phiBnd = nullptr;

    // `Gauss limitedLinear <k>`: a WEIGHT change, so it replaces the upwind coefficients rather than
    // adding to the source. `limiterCoeff` is the RAW k the case wrote -- the conversion to
    // 2/max(k,SMALL) happens here, once, so no caller can hand the wrong currency.
    bool   limitedLinear   = false;
    scalar limiterCoeff    = 1.0;
    // cellLimited k of the case's grad(<field>), which limits the LIMITER's gradient
    // (LimitedScheme.C:56-59). Zero means unlimited.
    scalar limGradK        = 0.0;
    // The limiter gradient's SCHEME: leastSquares rather than Gauss linear (gradSchemes grad(<field>)).
    bool   limGradLeastSq  = false;

    // `Gauss linearUpwind <name>`: upwind's weights plus the explicit correction
    // fvc::surfaceIntegrate(faceFlux*correction) (gaussConvectionScheme.C:112-115), built over the
    // gradient the entry NAMES -- mesh.gradScheme(<name>), linearUpwind.C:61-68 -- which is Gauss linear
    // with `luGradK` its cellLimited coefficient (0 = unlimited); the driver refuses any other. The host
    // closures' divWithScheme is the oracle (stage H3.5); this is its device twin (H3.6).
    bool   linearUpwind    = false;
    scalar luGradK         = 0.0;

    // `corrected` is TWO changes and this makes both: the implicit coefficient takes
    // nonOrthDeltaCoeffs, and the non-orthogonal part enters as an explicit source. Implementing only
    // the implicit half moves the SOURCE while leaving the DIAGONAL exact, which no gate comparing D()
    // can see.
    bool   correctedLaplacian = false;
    // The field's OWN grad scheme, which correctedSnGrad's correction takes (correctedSnGrad.C:52-55).
    // A DIFFERENT lookup from limGradK above, even though both come from gradSchemes.
    scalar gradFieldLimitK    = 0.0;
    // `limited <psi> corrected`: caps the non-orthogonal correction per face. Zero => uncapped.
    scalar snGradLimitCoeff   = 0.0;
    // The field's PATCH VALUES as the gradients below must read them, when they are not what a live
    // evaluate of `db` gives. OpenFOAM's gradients read the patch field's STORED values -- those of its
    // last evaluate, plus whatever updateCoeffs assigned since (epsilonWallFunction's
    // `epf == epsilon0`) -- while `db` carries the coefficients updateCoeffs has JUST refreshed for this
    // assembly. Null = evaluate `db` live, which is what every caller did before stage H3.5.
    const DeviceBuffer<scalar>* bndValues = nullptr;
};

// The linear solve for ONE transported scalar: relax() -> fvOptions.constrain() -> setValues(wall), in
// OpenFOAM's order (kEpsilon.C:265-267), then the solver the CASE asked for. Every model's second half
// looks like this and differs only in whether it has a wall constraint.
struct SolveControls
{
    scalar tol      = 1e-10;
    scalar relTol   = 0.0;
    int    maxIter  = 1000;
    int    minIter  = 0;
    // The case's own smoothSolver, when it named one: OpenFOAM's sweep under OpenFOAM's stopping rule.
    int    nSweeps     = 1;
    bool   gsSymmetric = true;
    // ...otherwise BiCGStab with the preconditioner the shared policy resolved (turbPreconFor), and
    // the Neumann series' degree that policy derived from the case's relaxation factor.
    const DeviceDilu* precon = nullptr;
    int    polyDeg  = 0;
};

// relax -> constrain -> wall setValues -> solve, writing the initial residual out. `wallMask`/`wallVal`
// null means the field has no wall constraint (kEpsilon.C:286-288 has no boundaryManipulate for k --
// kqRWallFunction is zeroGradient, and constraining k the way epsilon is constrained is a different
// equation). `dumpPrefix` empty disables the stage dump.
void solveScalarEqn(
    PressureMatrix&             M,
    DeviceBuffer<scalar>&       field,
    const DeviceMesh&           dm,
    bool                        relaxEquation,
    scalar                      alpha,
    const DeviceBuffer<label>*  fvoMask,
    const DeviceBuffer<scalar>* fvoVal,
    const DeviceBuffer<label>*  wallMask,
    const DeviceBuffer<scalar>* wallVal,
    const SolveControls&        sv,
    scalar&                     residualOut,
    const std::string&          dumpPrefix,
    bool                        gs);

// bnd[f] = field[bndCell[f]] on every boundary face `wfMask` marks, and nothing elsewhere: the one
// assignment the epsilon and omega wall functions make to their own patches inside updateCoeffs
// (`epf == scalarField(epsilon0, faceCells)`, epsilonWallFunctionFvPatchScalarField.C:168-175; the
// omega twin at omegaWallFunctionFvPatchScalarField.C:167-174). Shared by both closures so the two
// cannot drift on which faces it touches.
void wallFacesTakeCell(
    const DeviceMesh&           dm,
    const DeviceBuffer<label>&  wfMask,
    const DeviceBuffer<scalar>& field,
    DeviceBuffer<scalar>&       bnd);

// M = fvm::div(phi, field) - fvm::laplacian(gamma, field), with M's source zeroed and its boundary
// coefficients set. The caller adds the model's reaction terms afterwards.
void assembleScalarTransport(
    PressureMatrix&             M,
    const DeviceMesh&           dm,
    const DeviceBoundary&       db,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& gammaFace,   // DEff interpolated to internal faces
    const DeviceBuffer<scalar>& gammaBnd,    // ...and built from the patch values on boundary faces
    const TransportScheme&      sc);

} // namespace turbulence
} // namespace gpu
} // namespace brae
