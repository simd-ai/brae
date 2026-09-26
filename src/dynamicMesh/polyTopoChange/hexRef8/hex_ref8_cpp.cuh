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
    // faceEdges(): per face, per face POSITION -- section 9 and createInternalFaces index it by fp, so
    // the order IS the answer (unit 5a's gate holds it exactly for that reason)
    const std::vector<std::vector<label>>* faceEdges = nullptr;
    const std::vector<std::vector<label>>* edgeFaces = nullptr;
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

// ----------------------------------------------------------------------------------------------
// UNIT 5b-2: SECTION 9 -- THE FACES. 1,224 lines of OpenFOAM: setRefinement's own :3852-4190 plus the
// helpers only it calls. A split hex's faces come in four kinds and OpenFOAM handles them in this order,
// marking each face done as it goes so no face is touched twice:
//   1. faces that GET SPLIT, into four -- one per anchor point of the face (:3908-4010)
//   2. faces that do NOT split but whose EDGES do, so they gain vertices (:4014-4142)
//   3. faces that do not split at all but whose OWNER or NEIGHBOUR becomes one of the new cells (:4146-4172)
//   4. the twelve NEW INTERNAL faces inside each split cell (:4176-4218, createInternalFaces)
// `affectedFace` is the bookkeeping: every face of a split cell, every face being split, and every face
// on a split edge. Case 3 is whatever is left set after 1 and 2 -- which is why the ORDER of the four
// matters and they are not independent passes.

// findLevel (:697-740): walk the face from `startFp`, forward or back, to the first vertex whose point
// level is exactly `wantedLevel`. Throws where OpenFOAM FatalErrors: a level BELOW the wanted one means
// the face is not the shape the caller assumed.
label findLevel(
    const MeshView&           v,
    const Levels&             lv,
    label                     facei,
    const std::vector<label>& f,
    label                     startFp,
    bool                      searchForward,
    label                     wantedLevel);

// findMinLevel (:745-759): the face POSITION of the vertex with the LOWEST point level -- the first such,
// since the test is strictly less. Cases 2 and 3 use it to pick the anchor that names the new owner.
label findMinLevel(
    const Levels&             lv,
    const std::vector<label>& f);

// getAnchorCell (:540-593): which of a split cell's eight children owns `pointi`. An unsplit cell answers
// with itself. A point that is not one of the eight is looked for among the FACE's vertices instead --
// which is how an already-refined face finds its child -- and failing that OpenFOAM FatalErrors.
label getAnchorCell(
    const MeshView&                        v,
    const std::vector<std::vector<label>>& cellAnchorPoints,
    const std::vector<std::vector<label>>& cellAddedCells,
    label                                  celli,
    label                                  facei,
    label                                  pointi);

// setRefinement's section 9. Plays every face action into `a`: the splits, the edge-split insertions, the
// owner/neighbour changes and the twelve internal faces per split cell. `marks` is 5b-1's output and is
// read only.
//
// WHAT IS NOT PORTED, and it is debug-only in OpenFOAM too: checkInternalOrientation and
// checkBoundaryOrientation (:359-470), which sit inside `if (debug)` at every call site and compare the
// new face's area vector against the owner-to-neighbour direction. They change nothing.
void setRefinementFaces(
    const MeshView&        v,
    const Levels&          lv,
    const RefinementMarks& marks,
    TopoActions&           a);

// ----------------------------------------------------------------------------------------------
// UNIT 6: hexRef8::updateMesh, and the REFINEMENT HISTORY's producing side.
//
// provenance:
//   openfoam: hexRef8.C:4346-4351 (the one-argument form dynamicRefineFvMesh calls), :4357-4543 (the
//               four-argument one it forwards to)
//             refinementHistory.C -- resize, storeSplit, allocateSplitCell, updateMesh
//             hexRef8.C:4274-4300 (setRefinement's section 11, which drives storeSplit)
//
// WHICH BRANCH updateMesh TAKES, and it is not the one the comment first suggests. Both halves ask
// `reverseCellMap.size() == cellLevel_.size()` and, when it holds, REORDER through the reverse map rather
// than gathering through `cellMap` -- with the reason in OpenFOAM's own words: "We cannot use cellMap
// since then cells created from cells would get cellLevel_ of cell they were created from."
//
// The equality holds for a hexRef8 refinement, and the arithmetic of why is worth writing down:
// polyTopoChange's `reverseCellMap_` is APPENDED TO by every addCell, so after setRefinement it is sized
// nOldCells + nAdded -- 2275 on the gate's one-cell arm, not 2268 -- and `cellLevel_` is the same size
// because section 10 extended it over the added cells. So the reorder branch is the live one and the
// gather branch is for a change that was not hexRef8's.
//
// THE HISTORY IS ACTIVE EVEN ON A MESH THAT HAS NEVER BEEN REFINED, which decides whether any of this can
// be gated: refinementHistory's constructor sets `visibleCells_` to the identity when there is no file to
// read, and then `active_ = returnReduceOr(visibleCells_.size())` -- true for any non-empty mesh. So
// section 11 and this updateMesh both run on the gate's own fixtures.

// splitCell8 (refinementHistory.H:110-135) and the three lists refinementHistory keeps. brae's reader
// already carried `parent` and `visibleCells`, which is all getSplitPoints needs; PRODUCING a history
// needs the other two.
struct History
{
    std::vector<label>                parent;         // splitCells_[i].parent_, -1 at the top level
    std::vector<std::vector<label>>   addedCells;     // splitCells_[i].addedCellsPtr_, empty when null
    std::vector<label>                visibleCells;   // per CURRENT cell: -1, or an index into parent
    std::vector<label>                freeSplitCells; // the free list, used BACK first (LIFO)
    bool                              active = false;
};

// refinementHistory's constructor with no file to read (refinementHistory.C:392-412): every cell visible
// and its own top-level split entry, and `active` true for any non-empty mesh.
History freshHistory(label nCells);

// refinementHistory::resize (:1043-1060): grow visibleCells, the new entries NOT visible.
void resizeHistory(
    History& h,
    label    size);

// refinementHistory::allocateSplitCell (:940-985). Reuses the free list from the BACK, else appends. The
// LIFO order is part of the answer: it decides the INDEX a new split cell gets, and the indices are what
// `parent` and `visibleCells` are written in terms of.
label allocateSplitCell(
    History& h,
    label    parent,
    label    i);

// refinementHistory::storeSplit (:1000-1038). The cell that was live becomes the PARENT of its eight
// children and stops being live -- and then becomes live again as `addedCells[0]`, which is why the
// original cell keeps its own index through a refinement.
void storeSplit(
    History&                  h,
    label                     celli,
    const std::vector<label>& addedCells);

// refinementHistory::updateMesh (:1063-1120): renumber visibleCells through the change. Only the LIVE
// cells need renumbering, and a cell whose splitCell already has children is an error OpenFOAM stops on.
void historyUpdateMesh(
    History&                  h,
    const std::vector<label>& reverseCellMap,
    label                     nNewCells);

// hexRef8::updateMesh's own work: the two level lists remapped through the change. `nNewCells` and
// `nNewPoints` are the mesh AFTER it.
void updateLevels(
    Levels&                   lv,
    const std::vector<label>& reverseCellMap,
    const std::vector<label>& reversePointMap,
    const std::vector<label>& cellMap,
    const std::vector<label>& pointMap,
    label                     nNewCells,
    label                     nNewPoints);

// setRefinement's section 11 (:4274-4300): extend the history over the new cells and record every split.
void storeRefinementHistory(
    History&                               h,
    const std::vector<std::vector<label>>& cellAddedCells,
    label                                  nCellsAfterSplit);

// ----------------------------------------------------------------------------------------------
// UNIT 6b: UNREFINEMENT. hexRef8::setUnrefinement is 198 lines, but almost all of its topology is
// delegated to a `removeFaces` engine -- `compatibleRemoves` (178 lines) and that class's own
// `setRefinement` (759) -- so the real dependency is about 1,150 lines. Split where the oracle is, and
// both of removeFaces' entry points are PUBLIC, which is what makes the middle piece gateable alone:
//   6b-1  setUnrefinement's own half: the cell LEVELS come down and the HISTORY is combined.
//         Oracle: cellLevel and the history after setUnrefinement, which hexRef8 exposes.
//   6b-2  removeFaces::compatibleRemoves -- which cells merge into which region and which faces go.
//         Oracle: its three outputs, dumped directly because the function is public.
//   6b-3  removeFaces::setRefinement (759). Oracle: the map and the mesh.
//
// AND THIS IS THE ARM THAT GATES UNIT 6's REMAPPING HALF. A refinement leaves reverseCellMap and
// reversePointMap the IDENTITY -- measured -- so `updateLevels` and `historyUpdateMesh` were carried but
// unwitnessed. An unrefinement REMOVES cells: on the gate's own fixture the maps have 2940 and 5787
// entries and neither is the identity, and cellsFromCellsMap carries 96 merge sets.

// refinementHistory::freeSplitCell (:1607-1648). Detaches the entry from its parent -- which is why the
// parent's addedCells gets a -1 rather than shrinking -- marks it free with parent -2, and pushes it onto
// the free list. That push is what allocateSplitCell later pops from the BACK.
void freeSplitCell(
    History& h,
    label    index);

// refinementHistory::combineCells (:1652-1673). The eight children go away and their PARENT becomes live
// again at the master cell. The parent's addedCells pointer is reset, not cleared entry by entry.
void combineCells(
    History&                  h,
    label                     masterCelli,
    const std::vector<label>& combinedCells);

// hexRef8::setUnrefinement's own half (:5780-5797): for every split point, the eight cells around it drop
// one refinement level and the history combines them into their master -- which is `min(pointCells)`.
// POINT levels are untouched, and OpenFOAM says why: the points "either get removed or stay at the same
// position".
//
// The topology -- which faces go and which cells merge -- is removeFaces' and is NOT here. This function
// therefore does not touch `a`; it is the level and history half alone, which is what 6b-1 gates.
void setUnrefinementLevels(
    const MeshView&           v,
    Levels&                   lv,
    History&                  h,
    const std::vector<label>& splitPointLabels);

}   // namespace hexRef8
}   // namespace cpu
}   // namespace brae
