#pragma once
// patchWave's FaceCellWave<wallPoint> on the device -- the host wave (patch_wave_cpp.cu), in parallel, to the bit.
//
// provenance:
//   openfoam:  src/meshTools/cellDist/patchWave/patchWave.C (setChangedFaces, getValues)
//              src/meshTools/algorithms/MeshWave/FaceCellWave.C (faceToCell, cellToFace, iterate)
//              src/meshTools/cellDist/wallPoint/wallPointI.H (update, equal, valid)
//   host:      src/meshTools/cellDist/patchWave/patch_wave_cpp.cu -- the ORACLE
//   tests:     tests/interfoam_write/identity/patch_wave_device.sh
//
// THE ORDER IS KEPT the way deviceSmooth keeps it (device_fvc_smooth.cuh): a half-sweep reads one side and
// writes the other, so each cell (or face) folds its own visits in the host's order, and the next changed list
// is the elements sorted by the visit of their first change. The payload here is a wallPoint -- the nearest
// seed face's centre found so far and the squared distance to it -- and the element's own centre enters the
// update.
//
// THE SQUARED DISTANCE IS FORMED AS THE HOST'S COMPILED WAVE FORMS IT: fma(dz, dz, fma(dx, dx, dy*dy)), read
// off the host object (fmul d2,d2,d2; fmadd d1,d1,d1,d2; fmadd d0,d0,d0,d1 at all three sites of
// patch_wave_cpp.cu.o, aarch64 GCC). Written out with explicit fma calls so the device's distances are the
// host's bits; the identity gate is what says they still are after a compiler change.
//
// MEASURED on waveMakerPiston refined to 896,000 cells, where displacementLaplacian's inverseDistance
// diffusivity asks for this distance every step: the host wave and its diffusivity are 804 ms a step.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "fv_geometry.cuh"
#include "primitive_mesh.cuh"
#include "primitive_patch_cpp.cuh"
#include <vector>

namespace brae {

// The mesh's addressing on the device and the wave's work arrays, kept across calls. The caller sets `built`
// to false whenever the addressing changes; the geometry (cell and face centres) goes up at every call.
struct DevicePatchWave
{
    bool built = false;
    // the sweeps the last call took, FaceCellWave::iterate's count: with the cell count it is the front's mean
    // width, which is what decides whether this wave or the host's is the faster (inter_driver_device.cu)
    label sweeps = 0;
    label nC = 0;
    label nF = 0;
    label nIf = 0;
    DeviceBuffer<label> own;
    DeviceBuffer<label> nei;
    DeviceBuffer<label> cellStart;
    DeviceBuffer<label> cellFaces;
    // centres, component by component: cells [0, nC), then faces [nC, nC + nF)
    DeviceBuffer<scalar> px;
    DeviceBuffer<scalar> py;
    DeviceBuffer<scalar> pz;
    // each element's wallPoint, cells then faces as the centres: origin and squared distance
    DeviceBuffer<scalar> ox;
    DeviceBuffer<scalar> oy;
    DeviceBuffer<scalar> oz;
    DeviceBuffer<scalar> dist;
    DeviceBuffer<label> facePos;
    DeviceBuffer<label> cellPos;
    DeviceBuffer<label> touched;
    DeviceBuffer<label> changedFaces;
    DeviceBuffer<label> changedCells;
    DeviceBuffer<label> candidates;
    DeviceBuffer<label> count;
    DeviceBuffer<label> offset;
    DeviceBuffer<label> slot;
    DeviceBuffer<label> counter;
    // what the thin front's one-launch sweeps leave for the host: see pwThinFrontKernel
    DeviceBuffer<label> status;
    DeviceBuffer<unsigned char> scratch;
};

// FaceCellWave<wallPoint>(mesh, seedFaces, their centres at distance 0).iterate(nCells + 1) on the device.
// `seedFaces` in the order patchWave seeds them; `cells` is primitiveMesh::cells() as one array. Returns the
// squared distance the wave leaves in every cell and on every BOUNDARY face (face nInternalFaces + i at i), the
// two things patchWave::getValues reads; an element the wave never reached keeps -GREAT.
// BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1 drops the neighbour side's visits in faceToCell -- the identity gate's
// control.
void devicePatchWave(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const CellFaces& cells,
    const std::vector<label>& seedFaces,
    DevicePatchWave& w,
    std::vector<scalar>& cellDistSqr,
    std::vector<scalar>& boundaryDistSqr);

} // namespace brae
