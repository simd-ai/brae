#pragma once
// fvc::smooth on the device -- the host wave, in parallel, to the bit.
//
// provenance:
//   openfoam:  src/finiteVolume/finiteVolume/fvc/fvcSmooth/fvcSmooth.C:43-129
//              src/meshTools/algorithms/MeshWave/FaceCellWave.C (setFaceInfo, faceToCell, cellToFace, iterate)
//   host:      src/finiteVolume/finiteVolume/fvc/fvcSmooth/fvc_smooth_cpp.cu -- the ORACLE; its header says
//              why the order is part of the result
//   tests:     tests/interfoam_write/identity/wave_device.sh
//
// WHY A PARALLEL FORM CAN KEEP THE ORDER. Each half-sweep of FaceCellWave reads one side and writes the other:
// faceToCell reads the faces' values and writes the cells', cellToFace the reverse. So within a half-sweep a
// cell's end value is a fold over the changed faces that touch it, in the order the host visits them, and no
// other cell takes part; the same holds for a face, which at most two changed cells touch. Every cell (or
// face) folds its own visits here, in that order. What remains is the next changed list, which the host
// builds by appending each element at its FIRST successful update: each element records the index of that
// visit, and the list is those elements sorted by it -- a compaction over the visit indices.
//
//   faceToCell  visit index 2*i + side: the i-th changed face, its owner (0) before its neighbour (1)
//   cellToFace  visit index offset[j] + k: the j-th changed cell's k-th face in mesh.cells() order
//
// MEASURED on RAS/DTCHull (845,536 cells, about 150 sweeps a call): the host wave is 430 ms a step.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include "primitive_patch_cpp.cuh"
#include <vector>

namespace brae {

// The mesh's addressing on the device and the wave's work arrays, kept across calls. The addressing is the
// mesh's alone (no geometry), so the caller sets `built` to false whenever the addressing changes, as it
// rebuilds CellFaces.
struct DeviceSmoothWave
{
    bool built = false;
    label nC = 0;
    label nF = 0;
    label nIf = 0;
    DeviceBuffer<label> own;
    DeviceBuffer<label> nei;
    DeviceBuffer<label> cellStart;
    DeviceBuffer<label> cellFaces;
    DeviceBuffer<scalar> cellVal;
    DeviceBuffer<scalar> faceVal;
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
    DeviceBuffer<unsigned char> scratch;
};

// cpu::fvc::smooth (fvc_smooth_cpp.cuh) on the device: the same cell values, bit for bit. `cells` is the
// mesh's primitiveMesh::cells() as one array, the list the host wave walks.
// BRAE_CONTROL_WAVE_REVERSED=1 folds each cell's faceToCell visits in the REVERSE of the host's order -- the
// identity gate's control, which has to show that the order changes the answer and that the gate sees it.
void deviceSmooth(
    std::vector<scalar>& field,
    scalar coeff,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const CellFaces& cells,
    DeviceSmoothWave& w);

} // namespace brae
