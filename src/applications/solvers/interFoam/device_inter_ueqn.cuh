#pragma once
// interFoam's momentum predictor on the device -- the two pieces that are NOT rhoSimpleFoam's.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/interFoam/UEqn.H:1-33
//              src/finiteVolume/finiteVolume/ddtSchemes/EulerDdtScheme/EulerDdtScheme.C:434-470
//   host:      src/applications/solvers/interFoam/inter_ueqn_cpp.cu -- the ORACLE, gated in
//              tests/test_inter_ueqn_cpp.cu and running end to end at 3.3e-06 relative in U on
//              damBreak against real OpenFOAM.
//   tests:     tests/test_device_inter_ueqn.cu
//
// WHAT IS HERE AND WHAT ALREADY EXISTED. fvm::div(rhoPhi, U) is the same operator rhoSimpleFoam's
// momentum equation uses (deviceDivUpwindCoeffs and friends), and divDevRhoReff(rho, U) is
// deviceDivDevReff given mu_eff = rho*nuEff. Neither is re-ported. What interFoam adds is exactly two
// things, and both are places where the obvious reading is wrong:
//
//   1. fvm::ddt(rho, U) CARRIES TWO DIFFERENT DENSITY FIELDS.
//
//          fvm.diag()   = rDeltaT*rho*V                        <- THIS time level
//          fvm.source() = rDeltaT*rho.oldTime()*U.oldTime()*V  <- the PREVIOUS one
//
//      On a weakly compressible transient those differ by a per-cent and reusing one for both looks
//      like a rounding choice. At a VoF interface they differ BY A FACTOR OF 1000: any cell the alpha
//      solve moved the interface through has rho_old = 1 and rho = 1000, or the reverse. Writing rho in
//      the source multiplies that cell's old momentum by a thousand, exactly at the interface, which is
//      the only place a VoF solution is decided. `rhoOld` is a separate argument for that reason and
//      the gate's control arm passes rho twice to measure it.
//
//   2. THE BODY FORCE IS A RECONSTRUCTED FACE FLUX, NOT A CELL GRADIENT.
//
//          fvc::reconstruct((surfaceTensionForce - ghf*snGrad(rho) - snGrad(p_rgh)) * magSf)
//
//      The face form is what makes gravity and surface tension balance p_rgh face by face, which is the
//      whole reason the p_rgh formulation exists. On a hydrostatic start every one of the three terms
//      is identically zero away from the interface -- p_rgh uniform, rho piecewise constant,
//      snGrad(alpha) zero -- so the bulk of each phase feels NOTHING from the momentum predictor. A
//      cell-gradient substitute, or rho*g added to the source, accelerates the whole water column and
//      still converges.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"

namespace brae {

// fvm::ddt(rho, U), Euler, fixed mesh. Adds INTO an existing diagonal and source -- the div and stress
// terms are assembled first, as they are on the host.
//
// `rho` and `rhoOld` are separate arguments on purpose; see note 1.
void deviceInterEulerDdtRhoU(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& rhoOld,
    const DeviceBuffer<scalar>& UOldX,
    const DeviceBuffer<scalar>& UOldY,
    const DeviceBuffer<scalar>& UOldZ,
    scalar                      deltaT,
    DeviceBuffer<scalar>&       diag,
    DeviceBuffer<scalar>&       srcX,
    DeviceBuffer<scalar>&       srcY,
    DeviceBuffer<scalar>&       srcZ);

// (surfaceTensionForce - ghf*snGrad(rho) - snGrad(p_rgh)) * magSf, per face.
//
// `magSf` is the MESH'S FULL face array -- internal faces first, then the boundary patches -- so the
// three face fields index straight through it and it need only be long enough. deviceCompressionFlux
// carries the same convention, and getting it wrong cost a damBreak run that failed on a length
// mismatch rather than on a wrong number.
void deviceMomentumSourceFlux(
    int                         n,
    const DeviceBuffer<scalar>& surfaceTensionForce,
    const DeviceBuffer<scalar>& ghf,
    const DeviceBuffer<scalar>& snGradRho,
    const DeviceBuffer<scalar>& snGradPrgh,
    const DeviceBuffer<scalar>& magSf,
    DeviceBuffer<scalar>&       out);

// mu_eff = rho*nuEff, which is what interFoam's divDevRhoReff(rho, U) is built on -- the rho-weighted
// overload of linearViscousStress (linearViscousStress.C:119-131), reached through
// incompressibleInterPhaseTransportModel.C:129. Given mu, the operator is the one brae already has.
//
// TWO THINGS WORTH NAMING.
//
//   rho*nuEff IS NOT THE MIXTURE'S mu. The mixture's rho takes the RAW alpha while nu's denominator
//   takes the CLAMPED one (two_phase_mixture_cpp.cuh), so wherever MULES has left alpha outside [0,1]
//   the two differ. OpenFOAM forms rho*nuEff, so this forms rho*nuEff.
//
//   THE FACE VALUE IS interpolate(rho*nuEff), THE PRODUCT INTERPOLATED ONCE -- fvm::laplacian takes a
//   cell field and interpolates it. Interpolating the two factors separately is a different field, and
//   across a VoF interface rho jumps by 1000 in one face, which is exactly where the two stop agreeing.
//   deviceRhoRAUf carries the same lesson for interpolate(rho*rAU).
//
// `muBnd` is the product of the PATCH values, not of the face cells': the stress at a wall is what the
// boundary condition says nu and rho are there.
void deviceInterMuEff(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& nuEff,
    const DeviceBuffer<scalar>& rhoBnd,
    const DeviceBuffer<scalar>& nuEffBnd,
    DeviceBuffer<scalar>&       muCell,
    DeviceBuffer<scalar>&       muFace,     // internal faces: interpolate(rho*nuEff)
    DeviceBuffer<scalar>&       muBnd);

// ---------------------------------------------------------------------------------------------------
// `solve(UEqn == fvc::reconstruct(faceForce))` -- UEqn.H:19-31.
//
// source += V*reconstruct(faceForce). THE SIGN IS A PLUS: fvMatrix::operator== is source += V*R, where
// rhoSimpleFoam's grad(p) term carries the minus inside R itself. Both conventions live in this tree
// and the wrong one here still converges.
//
// THE ORDER IS OpenFOAM's: the matrix is assembled and RELAXED first, and the face force is added to a
// COPY afterwards. Relaxing after adding it would relax the buoyancy and surface tension too, which
// OpenFOAM does not do -- and rAU and H() are taken from the relaxed matrix BEFORE the force, so the
// caller keeps the original.
void deviceAddMomentumPredictorSource(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& faceForceInt,
    const DeviceBuffer<scalar>& faceForceBnd,
    DeviceBuffer<scalar>&       srcX,
    DeviceBuffer<scalar>&       srcY,
    DeviceBuffer<scalar>&       srcZ);

} // namespace brae
