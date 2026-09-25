#pragma once
// polyTopoChange's FACE AND CELL ORDERING -- the part of OpenFOAM's mesh-change machinery that decides
// the numbering of the mesh it produces. Unit 1 of dynamicRefineFvMesh's PRODUCING half.
//
// WHY THIS IS THE FIRST UNIT. brae already CONSUMES a mapPolyMesh bit-exactly: it reads one dumped from
// a real OpenFOAM run (tools/dumpRefineMap) and maps every field through it with zero difference. What
// is missing is producing one. And a produced map is only useful if it carries OpenFOAM's OWN
// numbering -- a correct-but-differently-ordered mesh is a different mesh, and every consuming unit
// keyed on cellMap/faceMap/reversePointMap would then be reading a map brae invented. The numbering is
// decided here, in two const functions of ~234 lines, and nowhere else.
//
// WHAT THEY ARE (meshTools/polyTopoChange/polyTopoChange.C:475 and :745):
//
//   makeCells(nActiveFaces, cellFaces, cellFaceOffsets)
//       A CSR face list per cell. Counts owner-faces then neighbour-faces, offsets by prefix sum, then
//       fills owner-faces for every face in ascending face index, THEN neighbour-faces in ascending
//       face index. So a cell's list is its owner-faces ascending followed by its neighbour-faces
//       ascending -- NOT a single ascending merge of the two.
//
//   getFaceOrder(nActiveFaces, cellFaces, cellFaceOffsets, oldToNew, patchSizes, patchStarts)
//       Walks cells in ascending index. For each, builds `nbr` over that cell's faces: the neighbouring
//       CELL where this cell is the master (celli < nbrCelli), and -1 for a retired face, an external
//       face, or a face whose other cell is the master. `sortedOrder` (stable, ascending --
//       List.C:474-487 is identity() then stableSort) then emits the non-(-1) entries, so internal
//       faces come out cell by cell in ascending cell index and, within a cell, in ascending
//       NEIGHBOUR-CELL index, each emitted once by its lower-numbered cell.
//       Then the boundary: patchStarts[0] = the first unassigned face, sizes counted from `region`,
//       starts by prefix sum, and each boundary face takes the next slot of its own patch -- so
//       within-patch order is ascending OLD face index. Retired faces map to themselves.
//
// THE ORACLE NEEDS NO INSTRUMENTATION, which is why this unit is first. Any mesh OpenFOAM has written
// is ALREADY in exactly this order, so feeding its own owner/neighbour/patch-id back in must return the
// IDENTITY permutation, with patchStarts equal to `constant/polyMesh/boundary`'s startFace values and
// patchSizes equal to its nFaces. MEASURED on a real 172,230-face mesh before a line of this was
// written: 0 violations of ascending-owner, 0 of within-cell-ascending-neighbour, 0 of
// owner < neighbour, and patchStarts[0] == nInternalFaces with the patches tiling exactly to nFaces.
//
// WHAT IS NOT HERE: the actions (addPoint/addFace/addCell), the point and cell compaction, the flip
// rule, zones, coupled-patch reordering, and the mapPolyMesh construction. This unit is the ordering
// alone, and it is const -- it mutates no mesh and is handed to no field, which is what lets it be
// gated before brae's mesh can change size at all.
#include "cf_types.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace polyTopoChange {

// The state the two functions read. In OpenFOAM these are polyTopoChange's own accumulating members
// (polyTopoChange.H:110-214); here they are handed in, because this unit produces no mesh and so has
// nothing to accumulate into. `cellMapSize` is `cellMap_.size()` -- the number of cells the change will
// have, which for a no-op round trip is the mesh's own nCells.
struct OrderInput
{
    label                     cellMapSize = 0;
    std::vector<label>        faceOwner;        // per face, the owner cell
    std::vector<label>        faceNeighbour;    // per face, the neighbour cell or -1 for a boundary face
    std::vector<label>        region;           // per face, the patch id or -1 for an internal face
    label                     nPatches = 0;
};

// polyTopoChange.C:475-556. `cellFaces` is sized 2*nActiveFaces then shrunk to the last offset, as
// OpenFOAM does -- the shrink is observable only in the size, but the two-pass fill is not.
void makeCells(
    const OrderInput&    in,
    label                nActiveFaces,
    std::vector<label>&  cellFaces,
    std::vector<label>&  cellFaceOffsets);

// ----------------------------------------------------------------------------------------------
// UNIT 2: the COMPACTION, on the path dynamicRefineFvMesh actually takes.
//
// `dynamicRefineFvMesh` calls `meshMod.changeMesh(*this, false)` (dynamicRefineFvMesh.C:462 and :585),
// taking the DEFAULTS -- so `orderCells == false` AND `orderPoints == false`
// (polyTopoChange.H:720-738). That settles two things:
//   * `getCellOrder`/`makeCellCells`, the Cuthill-McKee half, are NEVER REACHED on this path
//     (polyTopoChange.C:1190-1197 is the `if (orderCells)` branch). Porting them would be dead code.
//   * points and cells keep INSERTION ORDER minus the removed ones, and `nInternalPoints` is -1.
//
// And the cell renumber -- with it the FLIP -- runs only `if (orderCells || newCelli != cellMap_.size())`
// (polyTopoChange.C:1222), i.e. only when cells were actually REMOVED. So a pure REFINEMENT skips it
// entirely and an UNREFINEMENT does not: the two halves of AMR take different paths through this
// function, which is why the gate has an arm for each.
//
// REMOVAL IS ENCODED IN SENTINELS, not flags, and the predicates are transcribed rather than invented
// (polyTopoChangeI.H:32-52):
//     a point is removed when EVERY component is > 0.5*vector::max's
//     a face is removed when its vertex list is EMPTY
//     a cell is removed when its cellMap entry is -2
// vector::max's component, which the removed-point sentinel is tested against. OpenFOAM's is VGREAT,
// and on a double build that is 1.0e+300 (doubleScalar.H:60, scalar.H:128). The predicate compares
// against HALF of it, so the exact value matters only to within a factor of two -- but it is
// transcribed rather than guessed because `points_` is the only place a removed point is recorded.
constexpr scalar kVectorMaxComponent = 1.0e+300;

struct TopoState
{
    std::vector<vector>               points;          // removed = every component > 0.5*vector::max
    std::vector<std::vector<label>>   faces;           // removed = empty
    std::vector<label>                faceOwner;
    std::vector<label>                faceNeighbour;   // -1 on a boundary face
    std::vector<label>                region;          // patch id, -1 internal
    std::vector<label>                cellMap;         // removed = -2
    std::vector<label>                pointMap;
    std::vector<label>                faceMap;
    std::vector<char>                 flipFaceFlux;    // bitSet in OpenFOAM
    std::vector<label>                retiredPoints;   // sorted; OpenFOAM keeps a labelHashSet
    label                             nPatches = 0;
};

bool pointRemoved(const TopoState& s, label pointi);
bool faceRemoved (const TopoState& s, label facei);
bool cellRemoved (const TopoState& s, label celli);

// The result of the compaction, kept beside the state so a gate can see the maps themselves rather
// than only their effect.
struct CompactResult
{
    std::vector<label> localPointMap;   // old point -> new, -1 for removed/retired
    std::vector<label> localFaceMap;    // old face  -> new, -1 for removed
    std::vector<label> localCellMap;    // old cell  -> new, -1 for removed
    label              nActivePoints = 0;
    label              nActiveFaces  = 0;
    label              nActiveCells  = 0;
    label              nInternalPoints = -1;   // OpenFOAM's value when orderPoints == false
    bool               cellsRenumbered = false;  // whether the flip block ran at all
    label              nFlipped = 0;
};

// polyTopoChange.C:966-1290, the `orderCells == false, orderPoints == false` path only. Mutates `s`
// in place as OpenFOAM does: it renumbers owner/neighbour and flips faces whose neighbour became the
// lower cell.
CompactResult compactNoOrder(TopoState& s);

// polyTopoChange.C:745-888. Throws where OpenFOAM's FatalError fires: a face left unassigned, which
// OpenFOAM reports as "Did not determine new position for face ... This is usually caused by not
// specifying a patch for a boundary face."
void getFaceOrder(
    const OrderInput&          in,
    label                      nActiveFaces,
    const std::vector<label>&  cellFaces,
    const std::vector<label>&  cellFaceOffsets,
    std::vector<label>&        oldToNew,
    std::vector<label>&        patchSizes,
    std::vector<label>&        patchStarts);

}   // namespace polyTopoChange
}   // namespace cpu
}   // namespace brae
