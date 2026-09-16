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
//   OFF: correctContactAngle. It touches only the faces of alphaContactAngle patches, and of those
//        only the ones the interface has actually reached -- TWO of eight hundred on capillaryRise.
//        It is also branchy (four `limit` modes), stateful (it writes alpha's own wall gradient) and
//        iterative (the curvature is a fixed point in it). Putting a rare, branchy, stateful thing on
//        the device to save 2 faces of work would be the wrong trade; the corrected normals are
//        handed in and the bulk is what runs here.
//
// THE GRADIENT AND THE DIVERGENCE ARE NOT HERE EITHER: deviceGaussGrad and deviceDiv already exist and
// are gated on their own. This file is the piece between them that is interfaceProperties' own.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"

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
// MULES::explicitSolve has ended with psi.correctBoundaryConditions() (MULESTemplates.C:181) -- so the
// gradient it takes sees alpha's NEW patch values, not the ones the corrector started from. Folding it
// into deviceAlphaCorrector meant it read the pre-solve boundary; MEASURED on a rotating blob, that put
// nHatf 2.885e-04 out on a field of scale 1.7e-03, which is 17%, and carried 3.2e-09 into alpha by the
// end of the step. The caller evaluates the boundary between the two calls, which is what OpenFOAM does.
void deviceInterfaceCorrect(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& alpha1,
    const DeviceBuffer<scalar>& alpha1Bnd,      // the patch values the host has just evaluated
    const DeviceBuffer<scalar>& nHatfBnd,       // contact angle already applied, where there is one
    scalar                      deltaN,
    DeviceBuffer<scalar>&       nHatfInt,
    DeviceBuffer<scalar>&       K);

} // namespace brae
