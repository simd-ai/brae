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
//   tests:    tests/test_cell_wall_dist.cu (the wall distance this computes, against OpenFOAM's own) and
//             tests/interfoam_moving_vs_openfoam.sh, where it is recomputed after every mesh update.
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
#include <functional>
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
// `cells` is primitiveMesh::cells() (meshCells, primitive_patch_cpp.cuh) when the caller keeps it across calls;
// null builds it here. It is a function of the mesh's addressing alone, so the caller rebuilds it whenever the
// addressing changes. MEASURED on waveMakerPiston refined to 896,000 cells, where the motion solver asks for
// this distance every step: building the lists, one allocation a cell, was most of 1.25 s a step.
// THE WAVE ELSEWHERE. When `runner` is set patchWave hands it the seed faces, in the order it seeds them, and
// reads back the squared distance the wave leaves in every cell and on every boundary face (face
// nInternalFaces + i at i; an element never reached keeps -GREAT) -- the device loop's devicePatchWave
// (device_patch_wave.cuh), which is this file's FaceCellWave to the bit. Null runs the host wave, and so does
// a runner that returns false: it did not run the wave at this call (the device loop does that where the host's
// is the faster of the two, inter_driver_device.cu).
// `pointFaces` is primitiveMesh::pointFaces() (meshPointFaces) for the near-wall correction when the caller
// keeps it; null builds it there, an allocation a point at every call.
using PatchWaveRunner = std::function<bool(
    const PrimitiveMesh&,
    const FvGeometry&,
    const std::vector<label>&,
    std::vector<scalar>&,
    std::vector<scalar>&)>;
PatchWave patchWave(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const std::vector<label>& patchIDs,
    bool correctWalls,
    const std::vector<std::vector<label>>* cells = nullptr,
    const PatchWaveRunner* runner = nullptr,
    const std::vector<std::vector<label>>* pointFaces = nullptr);

} // namespace brae
