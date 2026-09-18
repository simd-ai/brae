#pragma once
// The VoF Courant number on the device -- alphaCourantNo.H, which every adjustable-dt interFoam case
// evaluates once a step.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/VoF/alphaCourantNo.H:34-54
//              src/transportModels/interfaceProperties/interfaceProperties.C:244-248 (nearInterface)
//              src/finiteVolume/finiteVolume/fvc/fvcSurfaceIntegrate.C (surfaceSum)
//   host:      src/finiteVolume/cfdTools/general/time_controls.cuh -- the ORACLE, whose header carries
//              the four things to get right; setDeltaTVoF stays there, on the host, because it is four
//              scalars of arithmetic on numbers this call has already reduced.
//   tests:     tests/test_device_alpha_courant.cu
//
// WHY A SECOND COURANT NUMBER AT ALL. The ordinary one is a global maximum over every cell, so it is
// set by whatever corner of the domain has the fastest flow -- usually far from the interface. A VoF
// interface has its own, much tighter limit, and MULES does not protect against advecting it more than
// a cell per step. The shipped tutorials almost always set maxAlphaCo at or below maxCo: 0.65/0.65 in
// twelve of them, 0.5/0.5 in ten.
//
// THREE THINGS THIS FILE HAS TO GET RIGHT, all of them measurable and none of them visible in a
// converged field:
//
//   nearInterface() IS A 0/1 MASK, NOT A WEIGHT: pos0(alpha1 - 0.01)*pos0(0.99 - alpha1). pos0 is 1 at
//   exactly zero, so the band is the CLOSED interval [0.01, 0.99]. A smooth weight, or pos instead of
//   pos0, changes which cells are counted at the edges of the band.
//
//   surfaceSum ADDS |phi| TO BOTH SIDES of every internal face and to the owner of every boundary face.
//   Missing the boundary half understates Co near inlets, which is exactly where the limiting cell
//   usually is.
//
//   THE DENOMINATOR IS NOT MASKED. meanAlphaCoNum is gSum(maskedPhi)/gSum(V) over the WHOLE mesh
//   volume, not over the interface cells' volume. It is a domain-average of an interface quantity and
//   reads small; the MAX is what limits the step. Masking the volume too would make the mean read like
//   an interface-local Courant number and would be a different diagnostic.
//
// WITH NO INTERFACE THE ANSWER IS ZERO, and that is correct: maxAlphaCo/(0 + SMALL) is astronomically
// large and setDeltaTVoF's min picks the ordinary-Courant branch. A case that has not yet developed an
// interface must not be throttled.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"

namespace brae {

struct DeviceCourantNumbers
{
    scalar CoNum     = 0;
    scalar meanCoNum = 0;
};

// surfaceSum(mag(phi)) per cell, masked to the interface band, reduced to the two numbers
// alphaCourantNo.H prints. `alpha1` null gives the ORDINARY Courant number -- the same formula with no
// mask -- so the two cannot drift apart, which is the reason the host shares courantNo() too.
DeviceCourantNumbers deviceAlphaCourantNo(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>* alpha1,
    scalar                      deltaT);

} // namespace brae
