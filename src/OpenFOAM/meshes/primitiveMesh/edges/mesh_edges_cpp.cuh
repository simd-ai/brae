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
// BOTH BRANCHES ARE PORTED. primitiveMesh::reset asks calcPointOrder whether the mesh's points are
// ORDERED -- every point used only by internal faces coming before every point on a boundary face. When
// they are, OpenFOAM sorts the edges into four blocks with the external edges last; when they are not,
// into one upper-triangular block. The four-block branch USED TO BE REFUSED here on the stated grounds
// that no mesh in this tree is ordered -- and that premise was false: a 2D case's `empty` front and back
// patches put EVERY point on a boundary face, so calcPointOrder's second loop finds no unnumbered point
// and returns ordered TRUE with nInternalPoints 0. MEASURED: laminar/damBreak reports 0 of 4746 and
// LES/nozzleFlow2D 0 of 15276, so the refused branch was the one every 2D fixture in this tree takes.
// The old comment had confused "nInternalPoints is -1" (the unordered SENTINEL) with "there are no
// internal points" (0, which is ordered). See mesh_edges_cpp.cu:106-116.
#include "cf_types.cuh"
#include "compact_list_list.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

struct MeshEdges
{
    // edges(): start < end for every edge
    std::vector<label> start;
    std::vector<label> end;
    // pointEdges(): per point, ascending edge labels (Foam::sort on the renumbered list). Compact
    // (compact_list_list.cuh): one array of values and one of offsets, where it was a heap block a point.
    CompactListList pointEdges;
    // nInternalPoints_: -1 when calcPointOrder found the points unordered
    label nInternalPoints = -1;
    // EACH POINT'S EDGES TO ITS HIGHER NEIGHBOURS, the neighbours ascending: point p's are entries
    // upperStart[p] .. upperStart[p+1] of upperEnd (the neighbour) and upperEdge (the edge). What the bucket
    // build finds the edges through, kept because "the edge between these two points" is then a scan of three
    // entries side by side (compactFaceEdges). Empty where the edges were built by chains; not part of what
    // OpenFOAM's edges() or pointEdges() are, and not compared by any check.
    std::vector<label> upperStart;
    std::vector<label> upperEnd;
    std::vector<label> upperEdge;

    label nEdges() const { return static_cast<label>(start.size()); }
    // edge::centre(points)
    vector centre(label e, const std::vector<vector>& p) const
    {
        return (p[static_cast<std::size_t>(start[static_cast<std::size_t>(e)])]
              + p[static_cast<std::size_t>(end[static_cast<std::size_t>(e)])])*scalar(0.5);
    }
};

// edges() and pointEdges(). THE ORDER calcEdges LEAVES IS (low point, high point) -- within each of the four
// blocks of an ordered mesh, and over the whole list of an unordered one: its last loop walks the points
// ascending and each point's higher neighbours sorted ascending (primitiveMeshEdges.C:304-410). So the edges
// are found here by BUCKETS: every face's consecutive point pairs dropped in the low point's bucket, each bucket
// sorted by the high point and made unique. OpenFOAM's own way -- a chain a point, searched for every pair --
// is buildMeshEdgesByChains, kept as the oracle. MEASURED on damBreakWithObstacle (92k cells, 305k edges),
// 2026-10-05: 13.5 ms a build by chains.
//   BRAE_CONTROL_MESH_EDGES_CHAINS=1     by chains, as before
//   BRAE_CONTROL_MESH_EDGES_CHECK=1      by both, and the lists compared entry for entry
//   BRAE_CONTROL_MESH_EDGES_UNSORTED=1   a gate's CONTROL, deliberately wrong: a bucket is left in the order the
//                                        faces filled it
MeshEdges buildMeshEdges(const PrimitiveMesh& m);
MeshEdges buildMeshEdgesByChains(const PrimitiveMesh& m);

// ----------------------------------------------------------------------------------------------
// faceEdges(), edgeFaces() and cellEdges() -- the three primitiveMesh addressings
// hexRef8::setRefinement reads, and unit 5a of the dynamicRefineFvMesh port.
//
// provenance:
//   openfoam: src/OpenFOAM/meshes/primitiveMesh/primitiveMeshEdges.C:540-600 (faceEdges, cached),
//                 :601-632 (faceEdges, on demand), :639-670 (cellEdges, on demand),
//                 :672-676 (cellEdges(celli))
//             src/OpenFOAM/meshes/primitiveMesh/primitiveMeshEdgeFaces.C:35-70 (edgeFaces, cached --
//                 invertManyToMany over faceEdges, so ascending face index)
//   tests:    tests/mesh_edges_addressing_vs_openfoam.sh, against tools/dumpMeshEdges
//
// EACH OF THE THREE HAS TWO IMPLEMENTATIONS IN OpenFOAM -- a cached whole-mesh form and an on-demand one
// that is taken while the cache is unbuilt -- and they need not agree in ORDER. MEASURED with
// tools/dumpMeshEdges on laminar/damBreak's own blockMesh, asking the on-demand forms FIRST so the cache
// could not answer for them:
//     faceEdges   on-demand == cached
//     edgeFaces   on-demand == cached
//     cellEdges   on-demand != cached, differing already at cell 0
//
// THE cellEdges DISAGREEMENT IS A HASH ORDER. The on-demand form inserts each face's edges into a
// `labelHashSet` and then walks THE SET (primitiveMeshEdges.C:650-668), so its order is bucket order --
// a function of OpenFOAM's hash and the set's capacity, which brae cannot reproduce by construction and
// should not try to.
//
// IT DOES NOT MATTER, and that is read off the call sites rather than hoped for. All three are used by
// setRefinement, and only one of them is used in an order-dependent way:
//     cellEdges(celli)   hexRef8.C:3415-3433 -- the loop MARKS: `edgeMidPoint[edgeI] = 12345` under a
//                        condition on the edge's own two point levels. Idempotent, per edge, independent
//                        of the iteration. ANY order gives the same answer.
//     edgeFaces(edgeI)   :3888-3896 -- `affectedFace.set(eFaces)`, a bitSet set operation. ANY order.
//     faceEdges(facei)   :4062-4080 -- `fEdges[fp]` indexed BY FACE POSITION. THE ORDER IS THE ANSWER.
// So brae reproduces faceEdges' order exactly, reproduces edgeFaces' because it is deterministic and
// free, and produces cellEdges SORTED -- which is neither OpenFOAM's cached nor its on-demand order, and
// is correct because the only consumer cannot tell. Said here rather than left as a silent choice.

// faceEdges(): per face, per face POSITION -- `out[facei][fp]` is the edge between the face's vertex fp
// and its next vertex. Built as primitiveMeshEdges.C:553-575 builds it: scan `pointEdges[f[fp]]` for the
// edge whose other vertex is `f[fp+1]`.
std::vector<std::vector<label>> buildFaceEdges(
    const PrimitiveMesh& m,
    const MeshEdges&     me);

// edgeFaces(): per edge, the faces on it in ASCENDING FACE INDEX -- which is what
// `invertManyToMany(nEdges(), faceEdges())` gives, because it walks faces in increasing order.
std::vector<std::vector<label>> buildEdgeFaces(
    const PrimitiveMesh&                   m,
    const std::vector<std::vector<label>>& faceEdges);

// cellEdges(): per cell, the union of its faces' edges. SORTED -- see the note above on why that is not
// OpenFOAM's order and why the one consumer cannot tell. `cells` is the cell-face list (meshCells).
std::vector<std::vector<label>> buildCellEdges(
    const std::vector<std::vector<label>>& cells,
    const std::vector<std::vector<label>>& faceEdges);

// THE SAME THREE, COMPACT (compact_list_list.cuh): row for row what the three above give, the refinement's
// addressing check (BRAE_CONTROL_REFINE_ADDRESSING_CHECK) holding each to its twin at every use. faceEdges is
// shaped like the faces themselves -- a row a face, an entry a vertex -- so its offsets ARE the mesh's.
CompactListList compactFaceEdges(
    const PrimitiveMesh& m,
    const MeshEdges&     me);
CompactListList compactEdgeFaces(
    const PrimitiveMesh& m,
    LabelListListRef     faceEdges);
// (the edge count where the caller has it; found by a scan of faceEdges otherwise)
//   BRAE_CONTROL_CELL_EDGES_SORT_UNIQUE=1   every cell's faces' edges sorted and then made unique, as before
//   BRAE_CONTROL_CELL_EDGES_FIRST_FACE=1    a gate's CONTROL, deliberately wrong: an edge two of a cell's faces
//                                           share is taken twice
CompactListList compactCellEdges(
    LabelListListRef cells,
    LabelListListRef faceEdges,
    label            nEdges = -1);


} // namespace brae
