#pragma once
// OpenFOAM's pointPatchDist: the distance from every mesh POINT to the nearest point of a set of
// patches, as PointEdgeWave computes it -- and the scale field rigidBodyMeshMotion builds from it.
// The host reference.
//
// provenance:
//   openfoam: src/dynamicMesh/pointPatchDist/pointPatchDist.C:64-137 (correct)
//             src/dynamicMesh/pointPatchDist/externalPointEdgePointI.H:38-130 (the two update
//                 overloads), :216-270 (updatePoint, updateEdge), :273-280 (equal)
//             src/meshTools/algorithms/PointEdgeWave/PointEdgeWave.C:153-283 (updatePoint,
//                 updateEdge), :716-748 (setPointInfo), :754-840 (edgeToPoint, pointToEdge),
//                 :860-935 (iterate)
//             src/meshTools/algorithms/PointEdgeWave/PointEdgeWaveBase.C:40 (propagationTol_ = 0.01)
//             src/rigidBodyMeshMotion/rigidBodyMeshMotion/rigidBodyMeshMotion.C:171-206 (the scale)
//   brae:
//     reference: this header
//     tests:     tests/test_point_patch_dist_vs_openfoam.cu against tools/dumpPointPatchDist
//
// IT IS NOT A NEAREST-POINT SEARCH, and that is the whole reason this is a port and not a loop. Each
// patch point seeds ITSELF as an origin with distance zero; origins travel point to edge to point, and
// a point keeps the nearest origin any edge-connected neighbour hands it. It stops there: an origin
// that would be nearer to a FURTHER point is never carried past a point that already holds a smaller
// distance. MEASURED on the floatingObject mesh against OpenFOAM's own pointPatchDist: measuring to
// the `atmosphere` patch, 60 of the 13,461 points end at 0.905538513813742 where the exact nearest
// patch point is 0.9. A brute-force nearest patch point is a DIFFERENT answer, not a slower one.
//
// THE ORDER THE WAVE REACHES A POINT IN DOES NOT CHANGE WHERE IT ENDS, measured rather than assumed:
// the gate stays green with the edge numbering left in creation order and with the 1% propagation
// tolerance set to zero. The fixed point is the same; the order only decides how many sweeps reach it.
// mesh_edges_cpp still reproduces OpenFOAM's numbering, and says there that it is not discriminated.
//
// NOT PORTED, refused where it would be met: coupled patches (cyclic, processor), across which
// handleCyclicPatches and handleProcPatches carry the wave.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "mesh_edges_cpp.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

struct PointPatchDist
{
    // per mesh point; the wave's distance, not the exact nearest-patch-point distance
    std::vector<scalar> distance;
    // points the wave never reached (OpenFOAM leaves their value untouched and counts them)
    label nUnset = 0;
};

// pointPatchDist(pointMesh, patchIDs, points): the patches in any order; the seeds are laid down in
// the order the patch list gives them, which is the order OpenFOAM's patchSet iterates.
PointPatchDist pointPatchDist(
    const PrimitiveMesh&        m,
    const MeshEdges&            e,
    const std::vector<FvPatch>& patches,
    const std::vector<label>&   patchIDs);

// rigidBodyMeshMotion.C:178-205 -- 1 up to innerDistance, linearly down to 0 at outerDistance, then
// turned into a cosine. Both clamps are OpenFOAM's and both are needed: the ratio leaves [0, 1] on
// either side of the band, and the cosine of a clamped ratio is not the clamp of a cosine.
std::vector<scalar> rigidBodyMeshMotionScale(
    const std::vector<scalar>& pointDist,
    scalar                     innerDistance,
    scalar                     outerDistance);

} // namespace brae
