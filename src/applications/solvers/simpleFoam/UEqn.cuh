#pragma once
// CUDA implementation of simpleFoam's momentum predictor -- the device twin of UEqn_cpp.
//
// provenance:
//   openfoam:  applications/solvers/incompressible/simpleFoam/UEqn.H:1-24
//   reference: src/applications/solvers/simpleFoam/UEqn_cpp.cu   (validated against OpenFOAM's own dumps)
//   cuda:      src/applications/solvers/simpleFoam/UEqn.cu
//   tests:     tests/test_ueqn_cuda.cu   (field by field against the reference)
//
// THE CONTRACT WITH THE REFERENCE. This produces the SAME OBJECT the _cpp path produces, in the same
// decomposition: one shared scalar LDU (diag/upper/lower), a per-component source, and per-component
// boundary coefficients. It deliberately does NOT fold the boundary into the diagonal or fuse the stages,
// because a fused kernel can only be compared as a single number and the whole method here is to name the
// first divergent stage. Folding is a separate, later step (deviceFold), exactly as OpenFOAM keeps
// internalCoeffs separate until solve time.
//
// Sign conventions, taken from the reference rather than re-derived:
//   * fvm::div(phi,U) enters with +1;
//   * divDevReff contributes -fvm::laplacian(nuEff,U) to the matrix and -fvc::div(nuEff*dev2(T(grad U)))
//     to the source, so the Laplacian is SUBTRACTED from diag/upper/lower and from the boundary coeffs;
//   * the device divDevReff kernel returns the EXTENSIVE V*div(sigma), which is precisely what the
//     reference adds to `source`, so it is added directly with no volume factor of its own.
//
// nuEff is required at BOUNDARY FACES as well as at cells. That is not a convenience: on a wall with a nut
// wall function the face value differs from the owner cell's by the whole of nut_wall, and using the cell
// value under-predicts wall shear silently. The reference makes the boundary array a required argument for
// the same reason.
//
// REFUSED, not ignored -- identical to the reference: MRF and fvOptions.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_MRF.cuh"
#include "rotor_disk.cuh"
#include "actuation_disk.cuh"
#include "device_ldu.cuh"
#include "device_fvoptions.cuh"   // DevicePorosity + deviceFvoPorosityDiag/Source
#include "UEqn_cpp.cuh"   // cpu::DivScheme -- one enum shared by both paths

namespace brae {
namespace gpu {

// The device momentum matrix, in the reference's decomposition.
struct MomentumMatrix
{
    DeviceBuffer<scalar> diag, upper, lower;   // raw LDU, before the boundary fold
    DeviceBuffer<scalar> source[3];            // per component, extensive
    DeviceBuffer<scalar> iC[3], bC[3];         // boundary coefficients, flattened in bndCell order
    // After relax(): the relaxed diagonal and delta = relaxedDiag - rawDiag. `delta` is kept because the
    // reference adds (diag - D0)*psi to the source, and having it explicitly makes that step checkable.
    DeviceBuffer<scalar> relaxedDiag, delta;
    bool relaxed = false;

    DeviceLduView view(const DeviceMesh& dm) const
    {
        DeviceLduView A{};
        A.nCells = dm.nCells; A.nInternalFaces = dm.nInternalFaces;
        A.diag = relaxed ? relaxedDiag.data() : diag.data();
        A.upper = upper.data(); A.lower = lower.data();
        A.owner = dm.owner.data(); A.nei = dm.nei.data();
        A.ownerStart = dm.ownerStart.data();
        A.losort = dm.losort.data(); A.losortStart = dm.losortStart.data();
        return A;
    }
};

struct MomentumInput
{
    const DeviceBuffer<scalar>* phiInt = nullptr;        // internal face flux
    const DeviceBuffer<scalar>* phiBnd = nullptr;        // boundary face flux
    const DeviceBuffer<scalar>* nuEffCell = nullptr;     // nCells
    const DeviceBuffer<scalar>* nuEffFace = nullptr;     // internal faces (interpolated)
    const DeviceBuffer<scalar>* nuEffBndFace = nullptr;  // boundary faces -- the PATCH value, not the cell's
    scalar relaxU = 1.0;
    bool   bounded = false;   // `bounded Gauss <scheme>`: diag -= V*div(phi); see UEqn_cpp.cuh
    // `Gauss linearUpwind grad(U)`: the matrix stays pure upwind and the whole scheme is a deferred
    // source correction -- see UEqn_cpp.cuh. Unlike `bounded` it does NOT vanish at convergence.
    bool   linearUpwind = false;                 // kept: equivalent to scheme == linearUpwind
    // The `k` of `grad(U) cellLimited Gauss linear <k>` -- the gradient linearUpwind NAMES. 0 leaves
    // the plain Gauss gradient. The correction does not vanish at convergence, so an unlimited
    // gradient under a limited name is a different equation: measured on windAroundBuildings, it puts
    // the momentum residual 272x OpenFOAM's instead of 1.4x.
    scalar gradULimitK = 0.0;
    // The `k` of the gradSchemes `grad(U)` ENTRY -- a DIFFERENT lookup from the one above, which
    // resolves the word linearUpwind names. On validation/rotorDisk they differ (`linearUpwind
    // unlimited` beside `grad(U) $limited`), so one field cannot serve both. This one is what
    // fvc::grad(U) takes, and divDevReff's explicit dev2 term is built from it
    // (linearViscousStress.C:114); it was left at the parameter's default, so the dev2 term ran the
    // plain Gauss gradient on every case that named a limiter.
    scalar gradUSchemeLimitK = 0.0;
    // The div(phi,U) scheme, shared with the reference (cpu::DivScheme). Each scheme is weights only, a
    // deferred correction only, or both; the assembly branches on that, not on a name.
    cpu::DivScheme scheme = cpu::DivScheme::upwind;
    scalar         schemeCoeff = 1.0;            // the `k` of `limitedLinear k`
    // `corrected` laplacianSchemes: switches the implicit coefficient to nonOrthDeltaCoeffs AND adds the
    // explicit deferred correction. Both halves, as in the reference -- see UEqn_cpp.cuh.
    bool   correctedLaplacian = false;
    // `limited <k> corrected` (OF limitedSnGrad). 0 = uncapped, which is what `corrected` means.
    scalar snGradLimitCoeff = 0.0;
    bool   hasMRF = false;
    bool   hasFvOptions = false;   // an UNIMPLEMENTED option -> refuse
    // explicitPorositySource/DarcyForchheimer, evaluated on the device each iteration from the current U.
    // nuLaminar, not nuEff: DarcyForchheimer.C looks up the field NAMED "nu".
    const DevicePorosity* porosity = nullptr;
    // rotorDiskSource. OF addSup is `eqn -= force` with force PER VOLUME, and operator-= is
    // source += V*su, so the extensive source GAINS the raw force. See rotorDiskSource_cpp.cuh.
    const DeviceRotorDisk* rotor = nullptr;
    // actuationDiskSource. OF writes the thrust straight into eqn.source() with `+=`, so unlike
    // rotorDiskSource there is no fvMatrix operator between the model and the matrix.
    const DeviceActuationDisk* actuationDisk = nullptr;
    // MRF.DDt(U), UEqn.H:8 -- the zones already resolved against the mesh and uploaded.
    const std::vector<DeviceMRFZone>* mrf = nullptr;
    scalar nuLaminar = 0.0;
};

// UEqn.H steps 1-2: fvm::div(phi,U) + turbulence->divDevReff(U), then UEqn.relax().
// Throws (host-side) on MRF/fvOptions, matching the reference's refusal contract.
void assembleUEqn(
    MomentumMatrix&              M,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    const DeviceBuffer<scalar>&  Ux,
    const DeviceBuffer<scalar>&  Uy,
    const DeviceBuffer<scalar>&  Uz,
    const MomentumInput&         in);

// UEqn.H step 3: the right-hand side of solve(UEqn == -fvc::grad(p)); source -= grad(p)*V.
// Applied to a COPY of the matrix by the driver, because pEqn.H needs the original for A() and H().
void addPressureGradient(
    MomentumMatrix&              M,
    const DeviceMesh&            dm,
    const DeviceBuffer<scalar>&  gradPx,
    const DeviceBuffer<scalar>&  gradPy,
    const DeviceBuffer<scalar>&  gradPz);

} // namespace gpu
} // namespace brae
