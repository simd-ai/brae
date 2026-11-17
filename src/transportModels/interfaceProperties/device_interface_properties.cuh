#pragma once
// Interface normal and curvature on the device.
//
// provenance:
//   openfoam:  src/transportModels/interfaceProperties/interfaceProperties.C:107-165 (calculateK)
//   host:      src/transportModels/interfaceProperties/interface_properties_cpp.cu -- the ORACLE for
//              this file, and itself gated against OpenFOAM's UNMODIFIED class to 2.5e-14 at four
//              calculateK pass counts (tests/interfoam_curvature_vs_openfoam.sh).
//   tests:     tests/test_device_interface_properties.cu (device vs host, on damBreak's own mesh)
//
// WHAT IS ON THE DEVICE AND WHAT IS NOT, and the split is deliberate.
//
//   ON:  the per-face work -- interpolate the cell gradient to the face, divide by
//        (mag + deltaN), dot with Sf. That is every internal face and every boundary face, and it runs
//        four times a step (interfaceProperties' constructor, once per alpha corrector per sub-cycle,
//        and again between the sub-cycle and UEqn).
//
//   OFF in general: correctContactAngle and the evaluation of alpha's patches stay with the host hooks.
//   ON for one validated family (the DeviceContactAngleState below): constantAlphaContactAngle walls
//   beside zeroGradient and empty patches, under an unsmoothed Gauss linear nHat. Every other patch
//   type or scheme keeps the host path (the driver decides and refuses to the host, it never approximates).
//
// THE GRADIENT AND THE DIVERGENCE ARE NOT HERE EITHER: deviceGaussGrad and deviceDiv already exist and
// are gated on their own. This file is the piece between them that is interfaceProperties' own.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"

namespace brae {

// nHatf on the INTERNAL faces:
//
//     gradAlphaf = w*grad[own] + (1-w)*grad[nei]          fvc::interpolate
//     nHatfv     = gradAlphaf/(mag(gradAlphaf) + deltaN)  interfaceProperties.C:141
//     nHatf      = nHatfv & Sf                            interfaceProperties.C:150
//
// Fused into one launch: the intermediate face gradient is never written, which is three face-sized
// vector fields of traffic saved on a kernel that is entirely memory-bound.
//
// deltaN is 1e-8/cbrt(average(V)) and is a SCALAR for the whole mesh, not a field -- it is the same
// stabiliser everywhere, and passing it per face would invite someone to make it one.
void deviceInterfaceNormalFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& gx,          // CELL gradient of alpha, component-wise
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar                      deltaN,
    DeviceBuffer<scalar>&       nHatfInt);

// ...and on the BOUNDARY, from normals the caller has already corrected for a contact angle where
// there is one. `nbx/nby/nbz` are per boundary face in the mesh's boundary-face order.
void deviceInterfaceNormalFluxBoundary(
    const DeviceMesh&           dm,
    int                         nBoundaryFaces,
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& nbx,
    const DeviceBuffer<scalar>& nby,
    const DeviceBuffer<scalar>& nbz,
    DeviceBuffer<scalar>&       nHatfBnd);

// K = -fvc::div(nHatf). The MINUS is the sign convention for the whole solver: gradAlpha points into
// phase 1, so -div of the unit normal is POSITIVE for a phase-1 drop. Wrapped rather than left to the
// caller because `deviceDiv` then a negate is two places to get one sign right.
void deviceInterfaceCurvature(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& nHatfInt,
    const DeviceBuffer<scalar>& nHatfBnd,
    DeviceBuffer<scalar>&       K);

// interfaceProperties::correct()'s device half, in one call: grad(alpha1) -> nHatf -> K. It is a
// function rather than three call sites because deltaN and the minus sign in K = -div(nHatf) are each
// one place to go wrong, and alphaEqn.H invokes this once per corrector and once more between the
// sub-cycle and UEqn.
//
// IT IS A SEPARATE CALL FROM THE ALPHA CORRECTOR ON PURPOSE. OpenFOAM runs it at alphaEqn.H:225, AFTER
// MULES::explicitSolve has ended with psi.correctBoundaryConditions() (MULESTemplates.C:80) -- so the
// gradient it takes sees alpha's NEW patch values, not the ones the corrector started from. Folding it
// into deviceAlphaCorrector meant it read the pre-solve boundary; MEASURED on a rotating blob, that put
// nHatf 2.885e-04 out on a field of scale 1.7e-03, which is 17%, and carried 3.2e-09 into alpha by the
// end of the step. The caller evaluates the boundary between the two calls, which is what OpenFOAM does.
//
// ACROSS PROCESSOR FACES (a DeviceProcCtx is current and the mesh has a halo): the Gauss gradient takes the processor
// faces from the boundary-value slot (alpha1Bnd's processor segment, in OpenFOAM's face order), the gradient's halo is
// refreshed, the interface kernel forms nHatf there from both sides' gradients into nHatfBnd's processor segment, the
// curvature sums it through the boundary gather like any boundary face, and K's halo is refreshed for the readers
// of sigma*K. The cyclic pair's own terms (cyc/nHatfIf) are not used for a processor face.
void deviceInterfaceCorrect(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& alpha1,
    const DeviceBuffer<scalar>& alpha1Bnd,      // the patch values the host has just evaluated
    // contact angle already applied, where there is one. NOT const: across processor faces (DeviceProcCtx) the
    // call writes the interface faces' normal into the processor segment of this array before it takes the
    // divergence -- and alpha1Bnd's processor segment must hold w*own + (1-w)*neighbour of `alpha1` (the alpha hooks)
    DeviceBuffer<scalar>&       nHatfBnd,
    scalar                      deltaN,
    // the gradSchemes entry NAMED nHat (interfaceProperties.C:117), as a GradChoice's two numbers
    bool                        gradLeastSquares,
    scalar                      gradCellLimitK,
    DeviceBuffer<scalar>&       nHatfInt,
    DeviceBuffer<scalar>&       K,
    // THE PAIR: its faces carry an interface normal like any other, and the curvature's divergence sums
    // them. interfaceProperties.C:134-150 -- gradAlphaf is fvc::interpolate(grad(alpha)), which on a
    // coupled patch is the two CELLS' gradients weighted.
    DeviceCyclic*               cyc = nullptr,
    DeviceBuffer<scalar>*       nHatfIf = nullptr);

// nHatf on the faces of a PERIODIC PAIR alone, from a gradient the caller has already given the pair's
// own contribution. interfaceProperties.C:134-150, with fvc::interpolate's coupled branch as the face
// gradient. Gated against OpenFOAM's OWN written nHatf in tests/interfoam_cyclic_nhatf_vs_openfoam.sh.
void deviceInterfaceNormalFluxCyclic(
    DeviceCyclic&               cyc,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar                      deltaN,
    DeviceBuffer<scalar>&       nHatfIf);

// THE CONTACT-ANGLE BOUNDARY ON THE DEVICE, the host hooks' state and arithmetic transcribed:
//   `stored`   alpha1's patch values as the host field holds them -- what every evaluate reads (the
//              `limit gradient` clamp takes the PREVIOUS stored value, fv_patch_field.cuh) and writes;
//   `refGrad`  a contact angle's wall gradient, rewritten by every curvature pass and read by the next.
// Both persist across hooks and steps exactly as the host patch objects did. A hook hands the alpha step
// a COPY of `stored` taken where the host uploaded its values (before the curvature pass moves them).
// Every product, sum and quotient is a separately rounded operation (__dmul_rn and friends): the host
// computes these lists in plain doubles with no contraction, and a fused multiply-add here would move
// the wall gradient by an ulp that the next pass then feeds back. sqrt and division are correctly rounded
// on both sides. cos/acos are NOT guaranteed to round as glibc does; see deviceContactAngleNHat.
// kind: 0 zeroGradient, 1 constantAlphaContactAngle, 2 empty. limit: 0 none, 1 gradient, 2 zeroGradient,
// 3 alpha (contact faces only). Arrays are in the device mesh's boundary order (no coupled patch).
struct DeviceContactAngleState
{
    int n = 0;
    DeviceBuffer<label> kind, limit, faceCell;
    DeviceBuffer<scalar> cosTheta, deltaCoeffs, nfx, nfy, nfz;
    DeviceBuffer<scalar> refGrad, stored;
    // host copies the libm half of the rotation reads (see deviceContactAngleNHat)
    std::vector<label> hostKind;
    std::vector<scalar> hostTheta;
    // work: the stencil cells' gradient (per cell); a face's unit normal, |gradAlphaf|, a12 and b2
    DeviceBuffer<scalar> gx, gy, gz, hx, hy, hz, mag, a12, b2;
    scalar* hostA12 = nullptr;   // pinned
    scalar* hostB2 = nullptr;    // pinned
    int*    hostIdx = nullptr;   // pinned
    // the faces whose a12 moved since the last call, and a12 as that call left it (see deviceContactAngleNHat)
    DeviceBuffer<scalar> prevA12, changedVal;
    DeviceBuffer<int>    changedIdx, changedCount;
    bool a12Valid = false;
    // the device's own libm half (device_glibc_math.cuh): the face's contact angle, and which variant reproduces the
    // host's libm (-2 not yet tested, -1 none: the host loop)
    DeviceBuffer<scalar> theta;
    int b2Variant = -2;
    DeviceContactAngleState() = default;
    DeviceContactAngleState(const DeviceContactAngleState&) = delete;
    DeviceContactAngleState& operator=(const DeviceContactAngleState&) = delete;
    ~DeviceContactAngleState()
    {
        if (hostA12) cudaFreeHost(hostA12);
        if (hostB2) cudaFreeHost(hostB2);
        if (hostIdx) cudaFreeHost(hostIdx);
    }
};

// alpha1.evaluateBoundary(): zeroGradient and empty take their cell, a contact angle clamps its gradient
// against the stored value (limit gradient) and evaluates cell + gradient/deltaCoeffs. `empties` false
// leaves an empty face's entry as it stands (the host's evaluateBoundaryWhere(notEmptyPatch)).
void deviceContactAngleEvaluate(DeviceContactAngleState& s, const DeviceBuffer<scalar>& cells, bool empties);

// A relaxed corrector's boundary (relaxBoundary hook): the evaluate on the post-MULES cells, then
// 0.5*that + 0.5*the value before it on every patch (all three kinds assign through operator=). An empty
// face takes the relaxed cell, or with `emptiesAverage` the average as every other face does (the host's
// whole evaluate and assignment, where an empty patch's entries are not kept on the GPU).
void deviceContactAngleRelax(DeviceContactAngleState& s, const DeviceBuffer<scalar>& postMules,
                             const DeviceBuffer<scalar>& relaxed, bool emptiesAverage);

// calculateK's boundary half (interfaceProps::finishNHatBoundary + nHatBoundary): the stencil cells'
// gradient from their internal-face sums `acc` (nHatStencilInternalKernel's, three blocks of nSt) plus
// their boundary faces in patch order over `stored`, divided by V; then each boundary face's corrected
// gradient, unit normal, the contact angle's rotation, the wall gradient written back and evaluated, and
// nHatf = nHat & Sf. An empty face's normal is zero.
void deviceContactAngleNHat(const DeviceMesh& dm, DeviceContactAngleState& s,
                            const DeviceBuffer<scalar>& cells, const DeviceBuffer<label>& stCells,
                            const DeviceBuffer<scalar>& acc, int nSt, scalar deltaN,
                            DeviceBuffer<scalar>& nHatfBnd);

// updateMixtureBoundary on the patch values `aBnd` (alpha2 = 1 - alpha1 there): rho from the raw value,
// mu and nu from the clamped one.
void deviceContactAngleMixture(const DeviceBuffer<scalar>& aBnd, scalar rho1, scalar nu1, scalar rho2,
                               scalar nu2, DeviceBuffer<scalar>& rhoBnd, DeviceBuffer<scalar>& muBnd,
                               DeviceBuffer<scalar>& nuBnd);

// fvm::div's patch coefficients: flux*valueInternalCoeffs and (-flux)*valueBoundaryCoeffs.
void deviceContactAngleDivCoeffs(const DeviceContactAngleState& s, const DeviceBuffer<scalar>& flux,
                                 DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC);

// The surface-tension flux on the boundary faces: (sigma*K_cell)*(gradientInternalCoeffs*alpha_cell +
// gradientBoundaryCoeffs), written to `stfBnd[0..n)`.
void deviceContactAngleSurfaceTension(const DeviceContactAngleState& s, const DeviceBuffer<scalar>& cells,
                                      const DeviceBuffer<scalar>& K, scalar sigma, scalar* stfBnd);

} // namespace brae
