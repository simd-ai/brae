#pragma once
// fvc::reconstruct on the device -- turning a face flux back into a cell vector.
//
// provenance:
//   openfoam:  src/finiteVolume/finiteVolume/fvc/fvcReconstruct.C:64-85
//              src/finiteVolume/finiteVolume/fvc/fvcSurfaceIntegrate.C:160-170 (surfaceSum)
//   host:      src/finiteVolume/finiteVolume/fvc_reconstruct_cpp.cuh -- the ORACLE, gated on the
//              IDENTITY reconstruct(V & Sf) == V in tests/test_fvc_reconstruct.cu.
//   tests:     tests/test_device_fvc_reconstruct.cu
//
//     SfHat = Sf/magSf
//     reconstruct(ssf) = inv(surfaceSum(SfHat (x) Sf)) & surfaceSum(SfHat*ssf)
//
// TWO THINGS THAT LOOK LIKE NOTATION AND ARE NOT, both carried over from the host header because the
// device is where they are easiest to get wrong -- the gather loop below reads exactly like deviceDiv's.
//
//   THE SIGN. surfaceSum adds to owner AND neighbour with the SAME sign, which is the opposite of every
//   divergence in this tree. Reusing the div pattern leaves the tensor symmetric and invertible and the
//   answer merely different. The gate's uniform-field arm is what separates them.
//
//   THE NORMALISATION. It is inv(surfaceSum(SfHat (x) Sf)), not inv(surfaceSum(Sf (x) Sf)). Dropping the
//   hat reweights each face by magSf, which is INVISIBLE on a cube -- every face has the same area -- and
//   wrong on anything else.
//
// NO ATOMICS: the per-cell tensor and vector are built by the same ownerStart/losort/bndCellStart gather
// deviceDiv and deviceGaussGrad use, so the summation order is deterministic and a run is reproducible.
//
// A PERIODIC PAIR IS IN surfaceSum LIKE ANY OTHER PATCH. fvcSurfaceIntegrate.C:168-180 walks
// mesh.boundary() whole -- there is no coupled branch and no sign flip -- so a cyclic face adds its own
// Sf and its own ssf to its own cell, once. The device's boundary arrays deliberately exclude coupled
// patches (device_mesh.cuh:41-44), so the pair arrives separately, through `cyc`. Leaving it out is not
// a small error: on validation/interFoamCyclic it put the Courant number at 2.68 by step two where
// OpenFOAM reads 0.05.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"

namespace brae {

// `ssfInt`/`ssfBnd` are the face flux; the result is the cell vector field, SoA. `cyc`/`ssfIf` carry a
// periodic pair's faces, and are null on a mesh without one.
void deviceReconstruct(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& ssfInt,
    const DeviceBuffer<scalar>& ssfBnd,
    DeviceBuffer<scalar>&       outX,
    DeviceBuffer<scalar>&       outY,
    DeviceBuffer<scalar>&       outZ,
    const DeviceCyclic*         cyc   = nullptr,
    const DeviceBuffer<scalar>* ssfIf = nullptr);

} // namespace brae
