#pragma once
// fvc::smooth -- raise a positive cell field so that no cell is more than (1 + coeff) times smaller than
// a neighbour. Host reference.
//
// provenance:
//   openfoam:  src/finiteVolume/finiteVolume/fvc/fvcSmooth/fvcSmooth.C:43-129 (smooth: the seeds, the
//                  initial cell and face data, maxIter = nTotalCells, correctBoundaryConditions)
//              src/finiteVolume/finiteVolume/fvc/fvcSmooth/smoothDataI.H:31-62 (update: take the
//                  neighbour's value over `scale` when mine is unset or below VSMALL, or when the
//                  neighbour's exceeds (1 + tol)*scale*mine), :67-85 (-GREAT, valid > -SMALL), updateCell
//                  (scale = maxRatio), updateFace (scale = 1)
//              src/meshTools/algorithms/MeshWave/FaceCellWave.C (setFaceInfo, faceToCell, cellToFace,
//                  iterate, updateCell, updateFace), FaceCellWaveBase.C:42 (propagationTol_ = 0.01)
//   brae:      the same wave as smoothDelta's (les_delta_cpp.cu), which OpenFOAM keeps as its own class
//              with its own data type; so does this tree
//   tests:     tests/interfoam_dtchull_vs_openfoam.sh -- setRDeltaT.H's smoothing, against the rDeltaT
//              OpenFOAM writes
//
// THE ORDER IS PART OF THE RESULT. A cell or a face takes a new value only when it is more than
// propagationTol (1%) larger than what it holds, so a cell reached by a large value late keeps a slightly
// smaller one it took earlier; the answer is not max over neighbours of value/maxRatio^distance. This file
// keeps FaceCellWave's order: the seeded faces in face order, each face's owner before its neighbour,
// each changed cell's faces in mesh.cells() order (owned, then neighboured).
//
// NOT PORTED, refused: a mesh with a coupled patch. fvcSmooth.C:76-92 seeds every coupled face and
// FaceCellWave carries the wave across them (handleCyclicPatches, handleProcPatches), which is not here.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace fvc {

// fvcSmooth.C:43-129 on the CELL values of `field`. The boundary is the caller's: OpenFOAM ends with
// field.correctBoundaryConditions(), and what that gives depends on the field's patch types.
void smooth(
    std::vector<scalar>& field,
    scalar coeff,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches);

} // namespace fvc
} // namespace cpu
} // namespace brae
