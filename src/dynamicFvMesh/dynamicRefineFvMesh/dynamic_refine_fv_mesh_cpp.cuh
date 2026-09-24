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
// `protectedCell` is empty where no cell is protected, and OpenFOAM's bitSet reads out of range as
// false, so an empty list here means nothing is protected -- it is not an error.
std::vector<label> selectRefineCells(
    label                                  maxCells,
    label                                  maxRefinement,
    const std::vector<char>&               candidateCell,
    const std::vector<label>&              cellLevel,
    const std::vector<char>&               protectedCell,
    label                                  nTotalCells,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches);

}   // namespace dynamicRefine
}   // namespace brae
