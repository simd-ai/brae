#pragma once
// OpenFOAM's primitiveMesh::edges() and pointEdges(): the mesh's edge list and, per point, the edges
// that meet there. The host reference.
//
// provenance:
//   openfoam: src/OpenFOAM/meshes/primitiveMesh/primitiveMeshEdges.C:41-78 (getEdge),
//                 :83-385 (calcEdges -- the two branches, the sort and the renumber)
//             src/OpenFOAM/meshes/primitiveMesh/primitiveMesh.C:218-244 (reset -> calcPointOrder,
//                 which decides which branch), :246-315 (calcPointOrder)
//   brae:
//     reference: this header
//     tests:     tests/test_point_patch_dist_vs_openfoam.cu, which compares a wave that walks these
//                edges against OpenFOAM's own pointPatchDist -- an order-dependent algorithm, so the
//                edge NUMBERING and the pointEdges ORDER are part of the answer, not book-keeping.
//
// THE NUMBERING IS OpenFOAM's, AND IT IS NOT DISCRIMINATED. PointEdgeWave visits changed edges in the
// order they were marked, so two edge orderings reach the same answer by different routes -- and on
// the one mesh this is gated against they reach the SAME answer: tests/point_patch_dist_vs_openfoam.sh
// stays green with the sort below replaced by the creation order. The sort is kept because it is what
// primitiveMesh::calcEdges does and a later caller (faceEdges, an edge-addressed field) would depend
// on it; nothing here claims a measurement for it.
//
// THE BRANCH THAT IS NOT PORTED. primitiveMesh::reset asks calcPointOrder whether the mesh's points
// are ORDERED -- every point used only by internal faces coming before every point on a boundary face.
// When they are, OpenFOAM sorts the edges into four blocks with the external edges last; when they are
// not, into one upper-triangular block. No mesh in this tree is ordered (blockMesh numbers points
// geometrically; subsetMesh and renumberMesh leave the point order alone; every fixture measured here
// reports nInternalPoints -1), so the four-block branch is REFUSED by name rather than transcribed and
// left with nothing to check it.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

struct MeshEdges
{
    // edges(): start < end for every edge
    std::vector<label> start;
    std::vector<label> end;
    // pointEdges(): per point, ascending edge labels (Foam::sort on the renumbered list)
    std::vector<std::vector<label>> pointEdges;
    // nInternalPoints_: -1 when calcPointOrder found the points unordered
    label nInternalPoints = -1;

    label nEdges() const { return static_cast<label>(start.size()); }
    // edge::centre(points)
    vector centre(label e, const std::vector<vector>& p) const
    {
        return (p[static_cast<std::size_t>(start[static_cast<std::size_t>(e)])]
              + p[static_cast<std::size_t>(end[static_cast<std::size_t>(e)])])*scalar(0.5);
    }
};

MeshEdges buildMeshEdges(const PrimitiveMesh& m);

} // namespace brae
