#pragma once
// OpenFOAM's patchWave: the distance from every cell centre to the nearest face of a set of patches, as
// the meshWave wall-distance method computes it. The host reference.
//
// provenance:
//   openfoam: src/meshTools/cellDist/patchWave/patchWave.C:38-222 (setChangedFaces, getValues, correct)
//             src/meshTools/algorithms/MeshWave/FaceCellWave.C:104-200 (updateCell, updateFace),
//                 :307-330 (setFaceInfo), :1073-1270 (faceToCell, cellToFace, iterate)
//             src/meshTools/algorithms/MeshWave/FaceCellWaveBase.C:42 (propagationTol_ = 0.01)
//             src/meshTools/cellDist/wallPoint/wallPointI.H (update, valid, equal)
//             src/meshTools/cellDist/cellDistFuncs.C:42 (useCombinedWallPatch = true), :244-420
//                 (correctBoundaryCells)
//             src/meshTools/cellDist/cellDistFuncsTemplates.C (smallestDist, getPointNeighbours)
//
// THE WAVE IS NOT A NEAREST-FACE SEARCH. Each wall face seeds its own centre; the centre travels face
// to cell to face, and a cell takes a new one only when it is nearer by more than 1% of the squared
// distance it already has (wallPoint::update). Which centre a cell ends with therefore depends on the
// ORDER the wave visits it in -- cells() order, the order faces changed -- and this file keeps that
// order. correctBoundaryCells then replaces the value of every cell that touches the patches with the
// exact distance to the nearest wall face near it, which is where the distance is small enough to
// matter to anything dividing by it.
//
// NOT PORTED, refused where it would be met: coupled patches (cyclic, processor, AMI) -- the wave crosses
// them in handleCyclicPatches/handleProcPatches, which are not here; and a cell the wave never reaches,
// whose "distance" OpenFOAM reports as -GREAT.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

struct PatchWave
{
    // distance_: per cell
    std::vector<scalar> distance;
    // patchDistance_: per patch, per face, sqrt(distSqr) + SMALL
    std::vector<std::vector<scalar>> patchDistance;
    // nUnset_: cells and faces the wave never reached
    label nUnset = 0;
};

// patchWave(mesh, patchIDs, correctWalls): patchIDs in any order; the wave seeds them in the mesh's
// patch order and correctBoundaryCells walks them sorted, both of which are the same thing
PatchWave patchWave(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const std::vector<label>& patchIDs,
    bool correctWalls);

} // namespace brae
