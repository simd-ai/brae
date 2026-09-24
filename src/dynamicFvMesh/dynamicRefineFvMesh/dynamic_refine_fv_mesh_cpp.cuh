// dynamicRefineFvMesh's REFINEMENT CANDIDATE SELECTION: which cells an adaptive mesh would refine,
// given a field and a band. This is the one part of AMR that is pure arithmetic on a FIXED mesh --
// it reads the point-cell addressing and a cell field and returns a marker per cell, and it touches
// no topology, allocates no polyTopoChange and calls nothing on hexRef8. So it is portable, and
// gateable against real OpenFOAM, before any of the mesh-changing machinery exists.
//
// OF v2412: src/dynamicFvMesh/dynamicRefineFvMesh/dynamicRefineFvMesh.C
//   cellToPoint            :754-771
//   error                  :774-793
//   maxPointField          :718-733
//   maxCellField           :736-751
//   selectRefineCandidates :796-827
//
// WHAT IS NOT HERE, and why: `selectRefineCells` (:830-907) needs hexRef8's cellLevel and its
// consistentRefinement 2:1 closure, `calculateProtectedCells` (:52-163) needs cellLevel and a face
// sync, and `selectUnrefinePoints` (:910-1002) needs getSplitPoints off an active refinementHistory.
// Each is its own unit. `extendMarkedCells` (:1005-1036) is pure except for one syncFaceList, which
// is a no-op only on a mesh with no coupled or cyclic patches.
//
// THE ORDER IS THE CONTENT. `cellToPoint` accumulates in pointCells order from 0.0, so a different
// order is a different last digit. OpenFOAM's primitiveMesh::calcPointCells has THREE orders and
// picks by what the mesh has already computed (see primitive_patch_cpp.cuh) -- so the addressing is
// taken as an argument here and the caller, and the gate, own that choice.
#pragma once

#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {
namespace dynamicRefine {

// dynamicRefineFvMesh.C:754-771. UNWEIGHTED arithmetic mean of the cell values over each point's
// cells -- no volume, no distance, no inverse-distance weighting -- accumulated in pointCells order
// from `sum = 0.0` and divided ONCE by pointCells[pointi].size(). OpenFOAM does not guard a point
// with no cells; this does not either, and says so rather than returning a quiet zero.
std::vector<scalar> cellToPoint(
    const std::vector<scalar>&                  vFld,
    const std::vector<std::vector<label>>&      pointCells);

// dynamicRefineFvMesh.C:774-793. `err = min(fld - minLevel, maxLevel - fld)` -- the distance to the
// NEARER band edge: positive strictly inside, exactly 0 on an edge, negative outside. The result
// starts at the sentinel -1 and `err` is written back under `err >= 0` (NON-strict), so a value
// sitting exactly on an edge is written as 0 and not left at -1. Both differences are formed in
// OpenFOAM's operand order.
std::vector<scalar> error(
    const std::vector<scalar>& fld,
    scalar                     minLevel,
    scalar                     maxLevel);

// dynamicRefineFvMesh.C:718-733. Point field -> cell field: each cell takes the MAX over its points.
// The cell field starts at -GREAT (1.0e+15), so a cell no point names keeps it.
std::vector<scalar> maxPointField(
    const std::vector<scalar>&                  pFld,
    const std::vector<std::vector<label>>&      pointCells,
    label                                       nCells);

// dynamicRefineFvMesh.C:736-751. Cell field -> point field: each point takes the MAX over its cells,
// from the INTERNAL field only. Starts at -GREAT. This is the field selectUnrefinePoints tests, and
// it is deliberately NOT the average above.
std::vector<scalar> maxCellField(
    const std::vector<scalar>&                  vFld,
    const std::vector<std::vector<label>>&      pointCells);

// dynamicRefineFvMesh.C:796-827: maxPointField(error(cellToPoint(vFld), lower, upper)), marked where
// the result is STRICTLY > 0. The two strictnesses do not agree and that is the point: `error`
// writes a band edge as 0 (`>= 0`) and this rejects it (`> 0`), so the refinement band is OPEN. A
// cell is a candidate only if one of its points is strictly inside it.
//
// Marks IN PLACE and never clears, as OpenFOAM's bitSet does: the caller hands in a marker sized
// nCells, and bits already set stay set.
void selectRefineCandidates(
    scalar                                      lowerRefineLevel,
    scalar                                      upperRefineLevel,
    const std::vector<scalar>&                  vFld,
    const std::vector<std::vector<label>>&      pointCells,
    label                                       nCells,
    std::vector<char>&                          candidateCell);

// ---------------------------------------------------------------------------------------------
// UNIT 2: from the candidates to the cells that will actually be refined. Still no topology change --
// these read cellLevel and the face addressing and return a list of cell labels -- but they are
// hexRef8's, not dynamicRefineFvMesh's, and they are where the 2:1 constraint lives.
//
// OF v2412: src/dynamicMesh/polyTopoChange/polyTopoChange/hexRef8/hexRef8.C
//   faceConsistentRefinement :1568-1652
//   consistentRefinement     :2234-2281
//   src/dynamicFvMesh/dynamicRefineFvMesh/dynamicRefineFvMesh.C
//   selectRefineCells        :830-907

// hexRef8.C:1568-1652, ONE pass. `cellLevel[c] + refineCell[c]` is the level the cell would have
// AFTER the proposed refinement, so the test is on the levels that would result, not the current
// ones. maxSet true ADDS the coarser neighbour, maxSet false REMOVES the finer one -- the same
// function serves the refinement superset and the unrefinement subset, and which way it moves is the
// whole difference between them.
//
// Returns the number of cells it changed; the caller loops until that is zero.
//
// REFUSES a coupled patch: OpenFOAM swaps the boundary levels across it
// (syncTools::swapBoundaryFaceList, :1625) and brae has no swap here. On an uncoupled patch the swap
// leaves the owner's own level in place and both tests are false, so the boundary loop is a no-op --
// which is why a mesh with no coupled patches needs nothing.
label faceConsistentRefinement(
    bool                                   maxSet,
    const std::vector<label>&              cellLevel,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches,
    std::vector<char>&                     refineCell);

// hexRef8.C:2234-2281: faceConsistentRefinement to a fixed point, then the marker as an ASCENDING
// list of cell labels (bitSet::toc()).
//
// THE `+ 1` IN THE 2:1 TEST IS WHAT MAKES THIS LOOP TERMINATE, and there is no iteration cap here
// because OpenFOAM has none. `refineCell` is a 0/1 flag, so the closure can raise a cell's effective
// level by at most one: a 2:1 test is satisfiable and a 1:1 test is not, because two cells whose
// stored levels differ by 2 can never be brought level. MEASURED while fail-proofing this unit, by
// changing `> neiLevel + 1` to `> neiLevel`: the loop does not end and the gate hangs. Keep the cap
// out (OpenFOAM's shape) and keep the test as written.
std::vector<label> consistentRefinement(
    const std::vector<label>&              cellLevel,
    const std::vector<label>&              cellsToRefine,
    bool                                   maxSet,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches);

// dynamicRefineFvMesh.C:830-907. The budget is INTEGER: `(maxCells - nTotalCells)/7`, because each
// refined hex becomes eight. Two branches, and they are not the same selection:
//   * candidates fit the budget -> take every candidate below `maxRefinement`, in ascending cell
//     order, with no truncation at all;
//   * they do not -> take WHOLE LEVELS, coarsest first, and stop after the first level that carries
//     the total past the budget. The list is never cut mid-level, so it can overshoot. OpenFOAM's own
//     comment there is "Sort by error? For now just truncate."
// Then hexRef8's 2:1 closure with maxSet TRUE, which can ADD cells and so overshoot `maxCells` again.
//
// `protectedCell` is the RAW protected set (dynamicRefineFvMesh's `protectedCell_`), not the cascade:
// this calls calculateProtectedCells on it itself, as OpenFOAM does at :845. Empty means nothing is
// protected and is not an error.
std::vector<label> selectRefineCells(
    label                                  maxCells,
    label                                  maxRefinement,
    const std::vector<char>&               candidateCell,
    const std::vector<label>&              cellLevel,
    const std::vector<char>&               protectedCell,
    label                                  nTotalCells,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches);

// ---------------------------------------------------------------------------------------------
// UNIT 3: the cells refinement must not touch, and the buffer that keeps unrefinement away from the
// cells it will. Still a fixed mesh.
//
// OF v2412: src/dynamicFvMesh/dynamicRefineFvMesh/dynamicRefineFvMesh.C
//   extendMarkedCells        :1005-1036
//   checkEightAnchorPoints   :1039-1079
//   calculateProtectedCells  :52-163

// :1005-1036. ONE face-layer dilation: cell -> its own faces -> owner and neighbour of those faces.
// Cell-face-cell, NOT point or edge connectivity, and `nBufferLayers` of them is `nBufferLayers`
// calls. It marks IN PLACE and never clears.
//
// What it is FOR is easy to get backwards: it does not widen the refinement band. It widens the set
// of cells that block UNREFINEMENT (dynamicRefineFvMesh.C:1414-1419, then the marked-cell veto in
// selectUnrefinePoints), so it is hysteresis.
//
// REFUSES a coupled patch: OpenFOAM ORs the face marks across cyclic and processor pairs
// (syncTools::syncFaceList, :1018) and brae has no sync here.
void extendMarkedCells(
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches,
    const std::vector<std::vector<label>>& cells,
    std::vector<char>&                     markedCell);

// dynamicRefineFvMesh::init's own scan (:1110-1248), the thing that FILLS protectedCell_. Four passes,
// cumulative and monotone -- nothing here ever clears a bit -- and then one sentinel:
//
//   a  :1131-1150  a cell with MORE THAN 8 points at pointLevel <= its cellLevel. The guard
//                  `!protected[c]` is OUTER, wrapping the level test itself, and the increment comes
//                  BEFORE the `> 8`, so the counter reaches 9 on the point that protects the cell.
//   b  :1158-1217  a FACE with more than 4 points at max(ownLevel, neiLevel) protects BOTH its cells
//                  (only the owner on a boundary face). The `faceLevel` here is a local max, NOT
//                  hexRef8::faceLevel(), which is a different quantity with the same name.
//   c  :1220-1239  a cell with FEWER than 6 faces, or any face of fewer than 4 points -- pure
//                  topology, no levels read at all. This is the one a prism trips.
//   d  :1242       checkEightAnchorPoints, the only pass that catches the UNDER-8 cells.
//
// ...and then :1245-1248: if nothing was set, the marker is CLEARED TO SIZE ZERO, not left as an
// array of falses. That size is the sentinel `calculateProtectedCells` (:57) and three other sites
// read, so returning a zeroed nCells array here would take the wrong branch in all four.
//
// REFUSES a coupled patch: pass b swaps face levels and ORs the face marks across pairs (:1169,
// :1201), and those are NOT no-ops in serial when the mesh has cyclics.
std::vector<char> initProtectedCells(
    const std::vector<label>&              cellLevel,
    const std::vector<label>&              pointLevel,
    const std::vector<std::vector<label>>& pointCells,
    const std::vector<std::vector<label>>& cells,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches);

// :1039-1079. Counts each cell's ANCHOR points -- points whose level is at or below the cell's -- and
// protects every cell that does not have exactly eight. The two `if`s inside are ordered: the `== 8`
// protect test runs BEFORE the increment, so it is the NINTH anchor that protects a cell and the
// count then freezes at 8. The final pass is an index loop over every cell, not over the marked ones.
//
// Only ever sets; never clears.
void checkEightAnchorPoints(
    const std::vector<label>&              cellLevel,
    const std::vector<label>&              pointLevel,
    const std::vector<std::vector<label>>& pointCells,
    label                                  nCells,
    std::vector<char>&                     protectedCell);

// :52-163. The 2:1 cascade closure of the protected set: a protected cell cannot be refined, so any
// FINER neighbour of one cannot be either, and that propagates to a fixed point. Both level tests are
// STRICT and both compare the other side against the protected side.
//
// An EMPTY `protectedCell` returns an empty result rather than a cleared array of the mesh's size --
// OpenFOAM's `unrefineableCell.clear()` leaves a bitSet of size 0, and every `test()` on it reads
// false, which is what makes `selectRefineCells` exclude nothing.
//
// REFUSES a coupled patch: the boundary half needs OpenFOAM's level swap (:74) and its face sync
// (:125).
std::vector<char> calculateProtectedCells(
    const std::vector<char>&               protectedCell,
    const std::vector<label>&              cellLevel,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches);

// ---------------------------------------------------------------------------------------------
// UNIT 4: which points can be UNSPLIT. Still a fixed mesh -- none of this touches polyTopoChange,
// mapPolyMesh or hexRef8::setUnrefinement -- but it needs one thing the units above did not: the
// refinement HISTORY, read off disk.
//
// OF v2412: src/dynamicMesh/polyTopoChange/polyTopoChange/hexRef8/hexRef8.C
//   getSplitPoints           :5180-5318
//   consistentUnrefinement   :5383-5603
//   src/dynamicFvMesh/dynamicRefineFvMesh/dynamicRefineFvMesh.C
//   selectUnrefinePoints     :910-1002

// refinementHistory, as much of it as unit 4 reads: the parent of each split entry and the entry each
// current cell is visible through. `addedCellsPtr_` is parsed to advance the stream and discarded --
// nothing here reads it.
//
// THE ON-DISK FORM IS ALWAYS THE COMPACTED ONE: operator<< calls compact() before writing
// (refinementHistory.C:1739), so a reader never meets a freed entry (`parent_ == -2`) or the free
// list. A never-refined mesh writes `0()` for the splits and `N{-1}` for the visible cells, so the
// uniform list form has to be parsed too.
struct RefinementHistory
{
    std::vector<label> parent;          // splitCells_[i].parent_, -1 at the top level
    std::vector<label> visibleCells;    // per CURRENT cell: -1, or an index into `parent`
    bool active = false;

    // refinementHistory.H:302-313. The caller must have checked visibleCells[celli] != -1 first --
    // OpenFOAM fatals here otherwise, and getSplitPoints' short-circuit at :5228 is load-bearing.
    label parentIndex(label celli) const;
};

RefinementHistory readRefinementHistory(
    const std::string& path,
    label              nCells);

// hexRef8.C:5180-5318. A point can be unsplit when it is the centre of one top-level split: it must
// have EXACTLY eight cells, all of them visible children of the SAME parent at the same level, and it
// must not lie on a boundary face.
//
// On a never-refined mesh every cell is its own top-level entry, `parentIndex` is -1 for all of them,
// and this returns EMPTY -- not an error. `pointLevel` is not read here at all.
std::vector<label> getSplitPoints(
    const RefinementHistory&               history,
    const std::vector<label>&              cellLevel,
    const std::vector<std::vector<label>>& pointCells,
    const std::vector<std::vector<label>>& cellPoints,
    const PrimitiveMesh&                   m);

// hexRef8.C:5383-5603, maxSet FALSE only -- OpenFOAM's own maxSet=true half FATALS at entry (:5395),
// so the function can only ever SHRINK the set. That is the mirror of consistentRefinement's
// superset, and the asymmetry is OpenFOAM's, not a simplification here.
//
// The tests are on the levels that would result AFTER unrefinement (`cellLevel - unrefineCell`), and
// they are the opposite direction from the refinement closure: `ownLevel < neiLevel - 1`.
std::vector<label> consistentUnrefinement(
    const std::vector<label>&              pointsToUnrefine,
    bool                                   maxSet,
    const std::vector<label>&              cellLevel,
    const std::vector<std::vector<label>>& pointCells,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches);

// dynamicRefineFvMesh.C:910-1002. A split point survives when no cell of it is protected, the field
// at it is STRICTLY below `unrefineLevel`, and none of its cells is in `markedCell` -- which is the
// refinement candidate set after the buffer layers, so the buffer's whole job is to veto here.
//
// `pFld` is maxCellField(vFld), NOT the cell-to-point average: a point is unrefinable only if the
// LARGEST neighbouring cell value is below the level.
std::vector<label> selectUnrefinePoints(
    scalar                                 unrefineLevel,
    const std::vector<char>&               markedCell,
    const std::vector<scalar>&             pFld,
    const std::vector<label>&              splitPoints,
    const std::vector<char>&               protectedCell,
    const std::vector<label>&              cellLevel,
    const std::vector<std::vector<label>>& pointCells,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches);

}   // namespace dynamicRefine
}   // namespace brae
