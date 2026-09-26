#pragma once
// OpenFOAM's hexRef8: the 8-way hex split, played into a polyTopoChange. Unit 5b of the
// dynamicRefineFvMesh port, and the last piece of its PRODUCING half.
//
// provenance:
//   openfoam: src/dynamicMesh/polyTopoChange/polyTopoChange/hexRef8.C
//               setRefinement (:3304-4315), and the private helpers it calls:
//               findMaxLevel (:658-672), countAnchors (:678-694), faceLevel (:801-826),
//               getFaceInfo (:104-129), addFace (:133-189), addInternalFace (:198-290),
//               modFace (:294-354), getAnchorCell (:540-593), storeMidPointInfo (:952-1178),
//               createInternalFaces (:1182-1461), walkFaceToMid (:1464-1509),
//               walkFaceFromMid (:1513-1563)
//   brae:
//     actions:   src/OpenFOAM/meshes/polyMesh/polyTopoChange/poly_topo_change_cpp.cuh (units 3, 4)
//     addressing: src/OpenFOAM/meshes/primitiveMesh/edges/mesh_edges_cpp.cuh (unit 5a)
//     tests:     tests/hex_ref8_vs_openfoam.sh, against tools/dumpHexRef8
//
// WHAT A SPLIT IS, and it is the same count on every hex: one point at the cell centre, one at each of
// the six face centres, one at each of the twelve edge midpoints -- 19 new points -- seven new cells
// beside the original, and thirty new faces (twelve internal to the split cell, and each of the six
// original faces becoming four, so eighteen more). MEASURED on laminar/damBreak's own blockMesh with
// tools/dumpHexRef8 refining cell 0: 2268 -> 2275 cells, 4746 -> 4765 points, 9176 -> 9206 faces.
//
// THIS UNIT IS SPLIT IN TWO, and the split is where the oracle is, not where the code divides neatly:
//   5b-1  sections 1-8, 10 and 12 -- the POINTS, the CELLS and the LEVELS. Gated on its own, because
//         setRefinement RETURNS cellAddedCells and leaves cellLevel()/pointLevel() public, so OpenFOAM
//         can be asked for all three without instrumenting anything.
//   5b-2  section 9 -- the FACES, which is storeMidPointInfo, createInternalFaces, walkFaceToMid,
//         walkFaceFromMid, addFace, addInternalFace and modFace (about 550 lines of the 911). Gated on
//         the mapPolyMesh and the mesh, which cannot be right until it exists.
// Section 11, the refinement history, is 5b-3.
//
// WHAT IS REFUSED, and it is the same list unit 4 carries plus one: a COUPLED patch. setRefinement
// synchronises edgeMidPoint, the edge mid POSITIONS, faceMidPoint and the boundary neighbour levels
// across coupled patches (syncTools::syncEdgeList at :3438, syncEdgePositions at :3496,
// syncBoundaryFaceList at :3626, and the newNeiLevel swap at :3590). In serial with no coupled patch
// every one of those is a max or an or over a single value, so brae skips them -- and refuses the case
// that would make them matter rather than skipping them silently.
#include "cf_types.cuh"
#include "mesh_edges_cpp.cuh"
#include "poly_topo_change_cpp.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace hexRef8 {

// the action surface units 3 and 4 built, which every point and cell below is played into
using polyTopoChange::TopoActions;

// hexRef8's own persistent state: the refinement level of every cell and every point. Both are
// labelIOList in OpenFOAM, read from constant/polyMesh READ_IF_PRESENT -- absent means a mesh that has
// never been refined, and every level is 0.
struct Levels
{
    std::vector<label> cellLevel;
    std::vector<label> pointLevel;
};

// The mesh addressing setRefinement reads, gathered once so the call does not take fourteen arguments.
// `cells`, `cellPoints`, `pointCells` and `cellEdges` are the derived addressing (unit 5a and
// primitive_patch_cpp.cuh); `edges` is MeshEdges.
struct MeshView
{
    const PrimitiveMesh*                   m = nullptr;
    const MeshEdges*                       edges = nullptr;
    const std::vector<std::vector<label>>* cells = nullptr;
    const std::vector<std::vector<label>>* cellPoints = nullptr;
    const std::vector<std::vector<label>>* pointCells = nullptr;
    const std::vector<std::vector<label>>* cellEdges = nullptr;
    const std::vector<vector>*             cellCentres = nullptr;
    const std::vector<vector>*             faceCentres = nullptr;
};

// findMaxLevel (:658-672): the face POSITION of the vertex with the highest point level -- the first
// such, since the test is strictly greater.
label findMaxLevel(
    const Levels&             lv,
    const std::vector<label>& f);

// countAnchors (:678-694): how many of the face's vertices are at or below `anchorLevel`.
label countAnchors(
    const Levels&             lv,
    const std::vector<label>& f,
    label                     anchorLevel);

// faceLevel (:801-826). A face with four vertices or fewer takes the level of its most-refined vertex;
// a bigger face is a face that has already been split, and its level is whichever of the owner's level
// or one above it has exactly four anchors -- or -1 when neither does, which means "do not split".
label faceLevel(
    const MeshView& v,
    const Levels&   lv,
    label           facei);

// What 5b-1 produces: the marks and the tables section 9 then reads, plus what OpenFOAM returns.
struct RefinementMarks
{
    // -1, or the label of the point added at the cell centre
    std::vector<label> cellMidPoint;
    // -1, or the label of the point added at the edge midpoint
    std::vector<label> edgeMidPoint;
    // -1, or the label of the point added at the face centre
    std::vector<label> faceMidPoint;
    // faceLevel per face, as it was BEFORE the split
    std::vector<label> faceAnchorLevel;
    // per refined cell, its eight corner points in ascending POINT order; empty for an unrefined cell
    std::vector<std::vector<label>> cellAnchorPoints;
    // per refined cell, the eight cells it becomes, with the ORIGINAL at element 0; empty otherwise
    std::vector<std::vector<label>> cellAddedCells;
    // ...and the levels the split leaves behind, which replace the Levels the caller passed in
    std::vector<label> newCellLevel;
    std::vector<label> newPointLevel;
};

// setRefinement's sections 1-8 and 10: every POINT and every CELL the split adds, played into `a`, and
// the tables section 9 needs. Returns OpenFOAM's own return value -- per entry of `cellsToRefine`, the
// eight cells that cell became (:4302-4311).
//
// `cellsToRefine` must already be 2:1 consistent: OpenFOAM's caller runs consistentRefinement first
// (dynamicRefineFvMesh.C:398), brae's does too, and it is gated by refine_candidates_vs_openfoam. This
// does not re-check it.
std::vector<std::vector<label>> setRefinementPointsAndCells(
    const MeshView&           v,
    const Levels&             lv,
    const std::vector<label>& cellsToRefine,
    const std::vector<std::string>& patchTypes,
    TopoActions&              a,
    RefinementMarks&          marks);

}   // namespace hexRef8
}   // namespace cpu
}   // namespace brae
