#pragma once
// kOmegaSST on the device, in the OF-MIRROR lineage.
//
// provenance:
//   openfoam: src/TurbulenceModels/turbulenceModels/Base/kOmegaSST/kOmegaSSTBase.C
//   brae:     src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST.cu
//   host twin: kOmegaSST_cpp.cu -- this file is its device projection and must agree with it, not with
//              the legacy device SST
//   tests:    tests/rho_sst_device_vs_openfoam.sh
//
// WHY THIS EXISTS WHEN src/cuda/device_komega_sst.cu ALREADY HAS AN SST. That one is the LEGACY
// lineage's whole-closure entry point (deviceKOmegaSSTCorrect). Its PHYSICS kernels are shared and this
// file calls them -- F1, F2, CDkOmega, the two reactions, the production limiters, nut -- but its
// MACHINERY is the legacy one, and the mirror differs from it in exactly the places the mirror kEpsilon
// closure was written to fix:
//
//   * Foam::bound is an area-weighted neighbour average for a cell that solved NEGATIVE, not a clamp
//     (bound.C:38-56). deviceBoundField takes an unweighted face-count mean and reads the CELL value on
//     boundary faces.
//   * nut on a `calculated` patch is Cmu-style field assignment from the patch's OWN k and omega, not
//     the owner cell's value.
//   * the nut wall-function FAMILY dispatches per face on the patch's own type; the legacy path carries
//     a single nutWall integer for the whole case.
//   * `Gauss limitedLinear` and its limiter gradient, which the mirror resolves and refuses per field.
//
// Wiring the legacy closure into the rho mirror would give its CUDA arm a different answer from its own
// HOST arm on all four. Measured on validation/rhoSST at 20 iterations, the two lineages sit the same
// distance from OpenFOAM (legacy k 3.54e-04, mirror host 4.13e-04) but 6.44e-04 from EACH OTHER -- so
// the legacy SST is not wrong, it is a different code, and an arm must agree with its own reference.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"      // DeviceWallData, deviceGradU, deviceGByNuFromGradU
#include "device_komega_sst.cuh"    // the shared SST physics kernels
#include "device_dilu.cuh"
#include "komega_sst_coeffs.cuh"
#include "pEqn.cuh"
#include <string>

namespace brae {
namespace gpu {
namespace kOmegaSSTRAS {

// The same shape as KEpsilonInput, field for field where the two models share a meaning, so the driver
// fills one or the other from the same host parse and the two arms cannot disagree about the case.
struct KOmegaSSTInput
{
    const DeviceBuffer<scalar>* phiInt      = nullptr;   // MASS flux, kg/s -- convection and `bounded`
    const DeviceBuffer<scalar>* phiBnd      = nullptr;
    // The VOLUMETRIC flux, for divU ONLY. OpenFOAM's (2/3)divU terms take fvc::div(phi) where phi is the
    // volumetric flux; feeding the mass-flux divergence makes that term wrong by a factor of rho, and
    // the two coincide only at constant density.
    const DeviceBuffer<scalar>* phiByRhoInt = nullptr;
    const DeviceBuffer<scalar>* phiByRhoBnd = nullptr;

    const DeviceBuffer<scalar>* rhoCell    = nullptr;    // null => the incompressible 1
    const DeviceBuffer<scalar>* rhoBndFace = nullptr;
    // fvm::ddt(alpha, rho, k|omega) (kOmegaSSTBase.C:572,602; EulerDdtScheme.C:365-398): rDeltaT*rho*V
    // on the diagonal, rDeltaT*rho.oldTime()*psi.oldTime()*V in the source; 0 under steadyState (an
    // empty matrix there). psi.oldTime() is the field at correct()'s entry, before the wall override;
    // rhoOldCell is rho.oldTime() as the caller resolves it (RhoStepInput::firstIteration), null ->
    // rhoCell, the host closure's rhoOldAt. Same contract as KEpsilonInput's.
    scalar                      rDeltaT    = 0.0;
    const DeviceBuffer<scalar>* rhoOldCell = nullptr;
    const DeviceBuffer<scalar>* nuCell     = nullptr;    // mu(T)/rho per cell.       REQUIRED here
    const DeviceBuffer<scalar>* nuBndFace  = nullptr;    // mu_b/rho_b per bnd face.  REQUIRED here
    const DeviceBuffer<scalar>* nuWallFace = nullptr;    // the same, in WALL-face order
    const DeviceBuffer<scalar>* nutBndFace = nullptr;
    // The STORED wall nut in WALL-face order -- the nut boundary as it ENTERED correct().
    // omegaWallFunctionFvPatchScalarField::calculate reads nutw from the nut patch field
    // (omegaWallFunctionFvPatchScalarField.C:199-200) and pairs it with the CURRENT nu_w at :335-340.
    // Recomputing a nutkWallFunction there instead injects d(nutw) = (dnutw/dnu)*dnu_w every
    // iteration: measured k 1.98e-06 vs OpenFOAM at iteration 1 on rhoSST with every other field
    // at 1e-13, confined to the 160 wall cells, sign-flipped between the hot and cold walls
    // (predicted ratio -9.47, measured -8.8). Null keeps the recomputation.
    const DeviceBuffer<scalar>* nutWallFace = nullptr;

    // The wall-function face set and each face's own coefficients, exactly as the kEpsilon closure
    // carries them -- omegaWallFunction reads the NUT patch's coefficients for its G0 the same way.
    const DeviceBuffer<label>*  wfBndMask    = nullptr;
    const DeviceBuffer<scalar>* wallYBndFace = nullptr;
    // Per boundary face: 1 where F1 is 1 by construction (a wall or empty patch). See rhoCreateFields.cuh.
    const DeviceBuffer<label>*  f1OneMask    = nullptr;

    // --- the flux-conditional and turbulent inlets, refreshed where OpenFOAM refreshes them ---
    // omega_.boundaryFieldRef().updateCoeffs() (kOmegaSSTBase.C:541) fires BEFORE the gradients, so
    // turbulentMixingLengthFrequencyInlet recomputes omega's refValue from k's CURRENT patch values
    // there; k's own turbulentIntensityKineticEnergyInlet refreshes later, inside the k equation's
    // fvMatrix constructor (:600), from U's patch values. Both are inletOutlet underneath, so the flux
    // switch has to resolve after the refValue moves. Left out, an inlet stays at the case file's
    // placeholder `value`: measured on rhoTI, k's inlet sat at 0.1 where OpenFOAM computes 9.63 and the
    // field was 7.5e-02 off at iteration 1. A null mask means the case has no such patch.
    // omegaLen is the mixing length per boundary face; kInt the turbulent intensity.
    const DeviceBuffer<label>*  turbInletOmegaMask = nullptr;
    const DeviceBuffer<scalar>* turbInletOmegaLen  = nullptr;
    const DeviceBuffer<label>*  turbInletKMask     = nullptr;
    const DeviceBuffer<scalar>* turbInletKInt      = nullptr;
    const DeviceBuffer<scalar>* nutWfCmu25Bnd  = nullptr;
    const DeviceBuffer<scalar>* nutWfKappaBnd  = nullptr;
    const DeviceBuffer<scalar>* nutWfEBnd      = nullptr;
    const DeviceBuffer<scalar>* nutWfYplLamBnd = nullptr;
    const DeviceBuffer<label>*  nutWfKindBnd   = nullptr;
    // Which boundary faces nut's own patch FILLS (a `calculated` nut), so correctNut writes those and
    // leaves a pinned fixedValue alone.
    const DeviceBuffer<label>*  nutCalcMask    = nullptr;
    // alphat's boundary: WHICH faces EddyDiffusivity::correctNut writes, and each one's Prt. A
    // compressible::alphatWallFunction face carries its OWN Prt_ (its .C:125), an assignable
    // `calculated` face takes the model's, and a fixedValue face is left alone -- so this is a mask and
    // a per-face factor, not one scalar. Writing rho_b*nut_b/Prt over the whole boundary instead drifts
    // the energy equation every iteration.
    const DeviceBuffer<label>*  alphatWallMask = nullptr;
    const DeviceBuffer<scalar>* alphatPrtFace  = nullptr;

    const DeviceBuffer<scalar>* Ux = nullptr;
    const DeviceBuffer<scalar>* Uy = nullptr;
    const DeviceBuffer<scalar>* Uz = nullptr;
    const DeviceBuffer<scalar>* yCell = nullptr;         // wall distance per CELL, for F1/F2

    // --- schemes and solver, per field, from the case ---
    bool   boundedK     = false;
    bool   boundedOmega = false;
    bool   limitedLinear = false;      // ONE flag for the pair, as the host closure carries
    scalar limiterCoeff  = 1.0;        // RAW k; the transport helper converts to 2/max(k,SMALL)
    scalar limGradK      = 0.0;        // the LIMITER's gradient limiter, from grad(<field>)
    bool   limGradLeastSq = false;     // ...and its SCHEME, when the case names leastSquares
    bool   linearUpwind  = false;      // `Gauss linearUpwind <name>` on the pair (TransportScheme)
    scalar luGradK       = 0.0;        // ...the cellLimited coefficient of the gradient it NAMES
    bool   correctedLaplacian = false;
    scalar snGradLimitCoeff   = 0.0;
    scalar gradULimitK        = 0.0;   // grad(U) cellLimited, for the production strain
    bool   relaxEquationOmega = false;
    scalar relaxOmega         = 1.0;
    bool   relaxEquationK     = false;
    scalar relaxK             = 1.0;
    scalar tol     = 1e-12;
    scalar relTol  = 0.0;
    int    maxIter = 2000;
    int    minIter = 0;
    const DeviceDilu* precon = nullptr;
    int    polyDeg = 1;
    bool   gsK = false, gsOmega = false, gsSymmetric = true;
    int    nSweepsKE = 1;

    // fvOptions constraints, resolved to per-cell masks by the caller.
    const DeviceBuffer<label>*  fvoOmegaMask = nullptr;
    const DeviceBuffer<scalar>* fvoOmegaVal  = nullptr;
    const DeviceBuffer<label>*  fvoKMask     = nullptr;
    const DeviceBuffer<scalar>* fvoKVal      = nullptr;

    KOmegaSSTCoeffs co;
    scalar Prt = 1.0;

    // --- refusals, the same set the kEpsilon closure carries ---
    bool        hasCoupledPatches      = false;
    bool        hasUnportedFvOption    = false;
    bool        hasNonUpwindDivScheme  = false;
    bool        hasNonWallTurbWallFunc = false;
    std::string fvOptionUnsupported;
    std::string divSchemeUnsupported;
};

// Residuals, in solve order, so a gate can name the first divergent equation.
struct KOmegaSSTResiduals
{
    scalar omega = 0.0;
    scalar k     = 0.0;
};

// One kOmegaSSTBase::correct(): production -> omega wall function -> CDkOmega/F1/F2 -> omega eqn ->
// solve -> bound -> k eqn -> solve -> bound -> correctNut(S2). Updates k, omega, nut and alphat.
void correct(
    DeviceBuffer<scalar>&       k,
    DeviceBuffer<scalar>&       omega,
    DeviceBuffer<scalar>&       nut,
    DeviceBuffer<scalar>&       nutBnd,
    DeviceBuffer<scalar>*       alphat,
    DeviceBuffer<scalar>*       alphatBnd,
    KOmegaSSTResiduals&         res,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    DeviceBoundary&             dbK,
    DeviceBoundary&             dbOmega,
    const DeviceWallData&       wall,
    const KOmegaSSTInput&       in);

} // namespace kOmegaSSTRAS
} // namespace gpu
} // namespace brae
