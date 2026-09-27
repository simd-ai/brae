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
#include "foam_dict.cuh"
#include "fv_patch.cuh"
#include "hex_ref8_cpp.cuh"
#include "poly_topo_change_cpp.cuh"
#include "fv_patch_field_mapper.cuh"
#include "geometric_field.cuh"
#include "map_poly_mesh_cpp.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <utility>
#include <vector>

namespace brae {


namespace dynamicRefine {

// ----------------------------------------------------------------------------------------------
// THE dynamicRefineFvMeshCoeffs READER. Until this existed the whole decision side above was fed by
// hand from a test -- `refineInterval` and `dumpLevel` appeared NOWHERE in src/, and the rest only as
// C++ parameters -- so no case could drive it.
//
// WHERE THE ENTRIES LIVE. `dynamicRefineFvMesh.C:181` and `:1292` both read
// `IOdictionary(dynamicMeshDict).optionalSubDict(typeName + "Coeffs")`, i.e. the
// `dynamicRefineFvMeshCoeffs` sub-dictionary WHEN THERE IS ONE and the dictionary itself otherwise.
// All three shipped AMR tutorials are needed to see that: laminar/damBreakWithObstacle writes the
// entries FLAT, laminar/oscillatingBox writes them flat BESIDE a `solvers { VF { ... } }` sub-dict for
// its motion solver, and RAS/motorBike WRAPS them in `dynamicRefineFvMeshCoeffs`. Reading only the
// sub-dictionary would miss two of the three; reading only the top level would miss the third.
//
// WHAT IS MANDATORY AND WHAT IS NOT, from the reads themselves (:1295-1356):
//   refineInterval    get<label>        MANDATORY.  == 0 means "never refine"; < 0 is FATAL
//   maxCells          get<label>        MANDATORY.  <= 0 is FATAL
//   maxRefinement     get<label>        MANDATORY.  <= 0 is FATAL
//   field             get<word>         MANDATORY
//   lowerRefineLevel  get<scalar>       MANDATORY
//   upperRefineLevel  get<scalar>       MANDATORY
//   nBufferLayers     get<label>        MANDATORY
//   unrefineLevel     getOrDefault      OPTIONAL, default GREAT = 1.0e+15 (doubleScalar.H:58).
//                                       RAS/motorBike omits it, which is why the default is not a
//                                       detail: reading 0 there would unrefine the whole mesh.
//   correctFluxes     get<List<Pair<word>>>  MANDATORY (:184, read at CONSTRUCTION)
//   dumpLevel         readEntry<bool>   MANDATORY (:193, read at construction)
//
// A MANDATORY entry that is absent must THROW and name itself -- OpenFOAM's `get<>` does, and a default
// quietly standing in for the case's own setting is the defect this project keeps finding.
struct RefineControls
{
    label                                            refineInterval = 0;
    std::string                                      field;
    scalar                                           lowerRefineLevel = 0;
    scalar                                           upperRefineLevel = 0;
    scalar                                           unrefineLevel = scalar(1.0e+15);   // GREAT
    label                                            nBufferLayers = 0;
    label                                            maxRefinement = 0;
    label                                            maxCells = 0;
    std::vector<std::pair<std::string, std::string>> correctFluxes;   // (flux, velocity) pairs
    bool                                             dumpLevel = false;
};

// `dynamicMeshDict` is the parsed constant/dynamicMeshDict. Throws on a missing mandatory entry and on
// each of OpenFOAM's three FatalErrors.
RefineControls readRefineControls(const FoamDict& dynamicMeshDict);

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

// ---------------------------------------------------------------------------------------------
// UNIT 6: the OLD-TIME VOLUMES, after a mesh change.
//
// OF v2412: src/dynamicFvMesh/dynamicRefineFvMesh/dynamicRefineFvMesh.C:203-254
//
// The cell mapper has already MAPPED V0 by the time this runs -- `mapFields` chains to
// `fvMesh::mapFields` first (:200) -- so this starts from the mapped old volumes and then OVERWRITES
// them on the cells the change touched, with the cell's CURRENT volume:
//
//   * a cell whose old cell contributed EXACTLY EIGHT new cells was split, so every one of those eight
//     takes its own new V (:229-238). The count is over cells whose old cell still exists
//     (`reverseCellMap[oldCelli] >= 0`), which is what makes it a split rather than a renumber.
//   * a cell in `cellsFromCellsMap` was MERGED, and takes its own new V too (:240-251). OpenFOAM's own
//     comment there wonders whether the old volumes should be summed instead; they are not.
//
// WHY IT MATTERS, and why a gate on mapped cell VALUES cannot see it: V0 is the volume the old-time
// field was stored in, and `ddt` divides by the change between V0 and V. A split cell whose V0 is the
// PARENT's volume reads a volume change of eight to one that never happened, so the first ddt after a
// refinement is wrong by the split ratio while every mapped field is exactly right.
//
// Takes the NEW mesh's volumes as an argument rather than computing them: brae's own FvGeometry::V is
// validated against OpenFOAM's but is FP-conditioning-limited rather than bit-exact, and putting it
// inside this would make a defect here indistinguishable from that.
std::vector<scalar> correctOldVolumes(
    const MapPolyMesh&         mpm,
    const std::vector<scalar>& mappedV0,
    const std::vector<scalar>& V);

// ---------------------------------------------------------------------------------------------
// UNIT 6b: the FLUX corrections. There are THREE sites, not one, and a port that finds only the first
// is green on every refinement and silently wrong on every unrefinement.
//
// OF v2412: src/dynamicFvMesh/dynamicRefineFvMesh/dynamicRefineFvMesh.C
//   mapFields, modified/added faces  :257-422   -- FOUR write sites
//   mapFields, injected faces        :424-437   -> mapNewInternalFaces
//   unrefine's OWN correction        :610-689   -- keyed on faceToSplitPoint and reversePointMap
//   dynamicRefineFvMeshTemplates.C   :32-183    -- the hull average and the oriented round trip
//
// MEASURED on damBreakWithObstacle with `correctFluxes ((phi U))`: the mapFields correction overwrites
// 2,156 faces on the refine step and **ZERO** on the unrefine step, where the second site overwrites
// 158. The first site's own count of split faces reads 269 and 0.

// What the corrections read off the NEW mesh. Taken as arguments because brae's own geometry is not
// bit-exactly OpenFOAM's (see the cell mapper's note on oldCellVolumes) and a defect here must not be
// confused with that.
struct FluxMeshView
{
    label                           nInternalFaces = 0;
    std::vector<label>              patchStart;
    std::vector<label>              patchSize;
    std::vector<label>              owner;
    std::vector<label>              neighbour;
    std::vector<std::vector<label>> cells;      // each cell's faces, owner-then-neighbour ascending
};

// :268-291. OpenFOAM's `masterFaces` is a bitSet LOCAL to mapFields: it dies with the block and
// nothing on the mapPolyMesh or the mesh carries it, so it is derived here exactly as OpenFOAM derives
// it. Its count is cross-checkable -- OpenFOAM prints `Found <n> split faces` at :295 under its own
// debug switch, and the derivation reads 269 and 0 against that.
//
// A face that maps from an old face whose reverse map is NEGATIVE is a face removed during refinement,
// which OpenFOAM treats as impossible and aborts on (:280-284). That is kept as a refusal.
std::vector<char> masterFaces(
    const MapPolyMesh& mpm,
    label              nNewFaces);

// :345-420. FOUR write sites: an inflated/appended internal face, an internal face from a master face,
// the same two on the boundary, and then every master face -- which dispatches on whether the face is
// internal and converts back through the patch it belongs to. Returns how many faces it wrote, so a
// gate can refuse a silent no-op.
label correctFluxes(
    std::vector<scalar>&                    phi,
    std::vector<std::vector<scalar>>&       phiBnd,
    const std::vector<scalar>&              phiU,
    const std::vector<std::vector<scalar>>& phiUBnd,
    const MapPolyMesh&                      mpm,
    const std::vector<char>&                masterFace,
    const FluxMeshView&                     m);

// Templates:32-98. For every INJECTED internal face (faceMap == -1), the average of the already-mapped
// faces of its owner and neighbour cells -- the "hull". A face with no mapped hull face is left alone
// (`counter > 0`), which is not the same as being set to zero.
//
// The accumulation order is the cell's own face order: all owner faces ascending, then all neighbour
// faces ascending (primitiveMeshCells.C:82-97), NOT globally ascending.
template <typename T>
void mapNewInternalFacesFlat(
    std::vector<T>&                    sFld,
    const std::vector<std::vector<T>>& sBnd,
    const MapPolyMesh&                 mpm,
    const FluxMeshView&                m);

// Templates:158-176, the ORIENTED branch -- which `phi` takes and `Uf` does not, decided by the
// `oriented` entry in the field FILE and not by the field's name. Converts the flux to an intensive
// vector (`sFld*Sf/sqr(magSf)`), maps that, and converts back with `sFld = (fFld & Sf)`.
//
// THAT LAST STEP IS A WHOLE-FIELD ASSIGNMENT and it is not the identity in floating point: it moves
// 1,083 faces on the refine step and 1,278 on the unrefine step that no correction touched, by about
// one ulp. It also DISCARDS the value :362 wrote on every injected internal face -- so that one line
// of OpenFOAM is unobservable from outside, and a port may implement or omit it with no oracle able to
// tell. Said here rather than left as a silent choice.
void mapNewInternalFacesOriented(
    std::vector<scalar>&                    phi,
    std::vector<std::vector<scalar>>&       phiBnd,
    const std::vector<vector>&              Sf,
    const std::vector<std::vector<vector>>& SfBnd,
    const std::vector<scalar>&              magSf,
    const std::vector<std::vector<scalar>>& magSfBnd,
    const MapPolyMesh&                      mpm,
    const FluxMeshView&                     m);

// :610-689, the site only unrefinement has. `faceToSplitPoint` maps each OLD face around a split point
// to that point, and is built from the PRE-change mesh (:555-574); a face whose midpoint was removed
// (`reversePointMap[oldPointi] < 0`) and which still exists takes the interpolated flux. There is no
// `NaN` branch here, unlike the first site.
label correctFluxesUnrefine(
    std::vector<scalar>&                          phi,
    std::vector<std::vector<scalar>>&             phiBnd,
    const std::vector<scalar>&                    phiU,
    const std::vector<std::vector<scalar>>&       phiUBnd,
    const std::vector<std::pair<label, label>>&   faceToSplitPoint,
    const MapPolyMesh&                            mpm,
    const FluxMeshView&                           m);

// ----------------------------------------------------------------------------------------------
// UNIT 7: THE DRIVER. dynamicRefineFvMesh::updateTopology() and the refine()/unrefine() it calls --
// the sequence that turns the pieces above into one adaptive step.
//
// provenance:
//   openfoam: dynamicRefineFvMesh.C:1274-1466 (updateTopology), :442-535 (refine), :537-716 (unrefine)
//   oracle:   tools/dumpRefineUpdate -- a copy of OpenFOAM's own class with WRITES ONLY (of-instrument),
//             stepped over an ANALYTIC field so what is compared is the driver and not a solve
//   tests:    tests/refine_update_vs_openfoam.sh
//
// THE ORDER IS THE PORT, and it is not the order the pieces were written in:
//   1  the candidates, from the field alone (selectRefineCandidates)
//   2  IF the mesh is under maxCells, the cells to refine (selectRefineCells), then refine
//   3  refineCell is REBUILT THROUGH THE MAP -- and this is the subtle one: a cell is marked again if
//      it is NEW (cellMap < 0), if it is not its old cell's master (reverseCellMap[old] != new) or if
//      its old cell was marked. So every child of a refined cell stays marked, which is what stops the
//      unrefinement below undoing the refinement just done.
//   4  nBufferLayers passes of extendMarkedCells
//   5  the points to unrefine (selectUnrefinePoints), then unrefine
//   6  the history is COMPACTED every tenth iteration -- and the counter starts at 0, so the FIRST step
//      compacts (`(0 % 10) == 0`).
// A step can refine, unrefine, both or neither, and `hasChanged` is the or of the two.
//
// WHAT THIS UNIT DOES NOT DO. The FIELD MAPPING -- mapFields, the old-time volumes and the flux
// correction -- is unit 7b. It cannot change the topology (OpenFOAM maps fields after the mesh is
// already built), so the driver is gated without it, and the gate's field is analytic and recomputed on
// the new cell centres at every step for exactly that reason.
// ----------------------------------------------------------------------------------------------
// UNIT 7b: THE FIELD MAPPING. What happens to a field when the mesh under it changes.
//
// provenance:
//   openfoam: src/OpenFOAM/lnInclude/cellMapper.C:28-60 (the direct branch), :62-230 (the
//                 interpolative one, including the VOLUME weighting), and cellMapper.C's constructor
//                 (:233-260) for the `direct()` predicate
//             src/OpenFOAM/fields/Fields/Field/Field.C:372-399 (the direct map), :456-470 (the weighted
//                 one, accumulated from Zero in the addressing order)
//             src/finiteVolume/fvMesh/fvMesh.C:851-891 (V0's OWN mapping rule, which is not the
//                 mapper's), :1020-1024 (when V0 exists at all)
//   tests:    tests/refine_update_vs_openfoam.sh, on two passive fields and the old-time volumes
//
// THE MAPPER HAS TWO BRANCHES AND A REFINEMENT AND AN UNREFINEMENT TAKE DIFFERENT ONES:
//   direct         when every cellsFrom* map is EMPTY, which is what a refinement leaves (measured: 0
//                  entries on every refinement arm of the hexRef8 gate). Each new cell takes the value
//                  of `cellMap[celli]`, and an INSERTED cell (cellMap -1) takes cell 0's value, because
//                  cellMapper rewrites its addressing entry to 0 and records it as inserted.
//   interpolative  when a merge happened. Each merged cell's value is a weighted sum over the old cells
//                  it came from -- and the weights are the OLD CELL VOLUMES, normalised, whenever the
//                  map carries them, uniform 1/n only when it does not or when the volumes sum to
//                  nothing. MEASURED: every change's map on the gate's fixture DOES carry them.
//
// WHAT THE WEIGHTING IS WORTH ON A UNIFORM MESH, measured off OpenFOAM's own dumped volumes rather than
// argued: the eight cells of a merge set have volumes equal to within 1.2e-15 of each other, so the
// volume weights are 1/8 to round-off and the weighted mean and the plain mean differ by about 2e-16.
// Replacing the weights with uniform ones is caught only by the gate arm that demands EXACTNESS (the one
// fed OpenFOAM's own volumes, bound 0); the arm that uses brae's own V cannot see it. A graded mesh would
// separate them by more; this fixture does not, and that is said rather than left.
//
// AND V0 IS MAPPED BY A RULE OF ITS OWN, not the mapper's: gather through cellMap with 0 where there is
// no old cell, then for every old cell whose reverseCellMap is a MERGE marker (< -1, meaning
// -master-2) ADD its old volume into the master's new cell. So a merged cell's V0 is the SUM of its
// parts, while its mapped field value is their weighted MEAN. dynamicRefineFvMesh::mapFields then
// overwrites V0 with the cell's own new V on split and merged cells (correctOldVolumes, above).
struct CellMapping
{
    bool                             direct = false;
    std::vector<label>               directAddressing;   // cellMap, with every negative rewritten to 0
    std::vector<std::vector<label>>  addressing;         // the interpolative branch
    std::vector<std::vector<scalar>> weights;
    std::vector<label>               insertedCells;      // ascending, for the record
};

// cellMapper's constructor and calcAddressing, from brae's own change map. `oldCellVolumes` empty means
// the map carried none and the merge weights stay uniform.
CellMapping cellMapping(
    const cpu::polyTopoChange::TopoChangeMap& map,
    label                                     nNewCells,
    const std::vector<scalar>&                oldCellVolumes);

// Field<Type>::autoMap through that mapping. The weighted branch accumulates from ZERO in the
// addressing order, which is the order cellMapper wrote it in.
std::vector<scalar> mapCellField(const std::vector<scalar>& oldField, const CellMapping& cm);
std::vector<vector> mapCellField(const std::vector<vector>& oldField, const CellMapping& cm);

// fvMesh::mapFields' own V0 rule (:851-891). `V0` is the OLD mesh's old-time volumes.
std::vector<scalar> mapOldVolumes(
    const std::vector<scalar>&                V0,
    const cpu::polyTopoChange::TopoChangeMap& map,
    label                                     nNewCells);

// ----------------------------------------------------------------------------------------------
// UNIT 7b-2: THE SURFACE FIELD MAPPING. What happens to a FLUX when the mesh under it changes -- step
// one, before any of the corrections above.
//
// provenance:
//   openfoam: src/OpenFOAM/lnInclude/faceMapper.C:28-190 (both branches), :233-247 (the predicate)
//             src/finiteVolume/lnInclude/fvSurfaceMapper.C:32-120 (the slice to the internal faces and
//                 the boundary-source reset)
//             src/finiteVolume/lnInclude/fvPatchMapper.C:60-215 (the slice to one patch)
//             src/finiteVolume/lnInclude/MapFvSurfaceField.H:60-95 (the ORIENTED flip after the map)
//   tests:    tests/refine_update_vs_openfoam.sh, on an oriented surface field carried over three steps
//
// THE INTERNAL FIELD IS THE FACE MAPPING SLICED, AND THE SLICE IS NOT INNOCENT. fvSurfaceMapper takes
// faceMapper's addressing, cuts it to the new mesh's INTERNAL faces, and then resets any entry whose
// source was a BOUNDARY face to read face 0 -- because an internal face that came from a boundary face
// has no flux to inherit.
//
// AND THE TWO BRANCHES DISAGREE BY ONE. The direct branch resets on `addr > oldNInternal` and the
// interpolative one on `max(addr) >= oldNInternal` (fvSurfaceMapper.C:57 against :76) -- so a face whose
// source is EXACTLY the first old boundary face is reset by one branch and kept by the other. That is
// OpenFOAM's own asymmetry, transcribed rather than tidied, and it is why both branches are written out
// here instead of being folded into one.
//
// THEN AN ORIENTED FIELD IS NEGATED on every face in flipFaceFlux (MapFvSurfaceField.H:83-93), which is
// what a flux is and `Uf` is not -- the flag comes from the field FILE's `oriented` entry, not its name.
// The addressing itself is FvPatchFieldMapping, declared beside the patch fields that consume it
// (fv_patch_field_mapper.cuh) so that a patch field can be mapped without depending on the dynamic mesh.
// This name is kept because the three builders below are OpenFOAM's three MAPPERS, not one mapping.
using FaceMapping = FvPatchFieldMapping;

FaceMapping faceMapping(
    const cpu::polyTopoChange::TopoChangeMap& map,
    label                                     nNewFaces);

// fvSurfaceMapper: the same addressing cut to the internal faces, with every boundary source reset.
FaceMapping surfaceMapping(
    const FaceMapping& fm,
    label              nNewInternalFaces,
    label              nOldInternalFaces);

// fvPatchMapper: cut to one patch, with each source rebased onto the OLD patch and anything from
// elsewhere dropped -- a -1 on the direct branch (which Field::map then leaves ALONE, keeping whatever
// the resized field held) and a re-scaled weight set on the other.
FaceMapping patchMapping(
    const FaceMapping& fm,
    label              newPatchStart,
    label              newPatchSize,
    label              oldPatchStart,
    label              oldPatchSize);

// Field<Type>::autoMap through one of those, then the ORIENTED flip. `flipFaceFlux` is the map's own
// list of new faces whose sense reversed; only the entries below the field's size are flipped.
std::vector<scalar> mapSurfaceField(
    const std::vector<scalar>& oldField,
    const FaceMapping&         sm,
    bool                       oriented,
    const std::vector<label>&  flipFaceFlux);

// ...and the same for a surface VECTOR, which is what Uf is. An unoriented field takes no flip, and
// OpenFOAM only negates ORIENTED fields (MapFvSurfaceField.H:83) -- so the flag is carried rather than
// assumed from the type.
std::vector<vector> mapSurfaceField(
    const std::vector<vector>& oldField,
    const FaceMapping&         sm,
    bool                       oriented,
    const std::vector<label>&  flipFaceFlux);

struct RefineUpdateState
{
    PrimitiveMesh        m;
    std::vector<FvPatch> patches;
    cpu::hexRef8::Levels levels;
    cpu::hexRef8::History history;
    // dynamicRefineFvMesh's own protectedCell_, which SURVIVES a step and is renumbered by each change.
    // Empty means nothing is protected.
    std::vector<char>    protectedCell;
    label                nRefinementIterations = 0;
    // HOW MANY ZONES THE MESH CARRIES -- pointZones + faceZones + cellZones. It is here because brae's
    // PrimitiveMesh does not carry zones and changeMesh's refusal of them could therefore never fire:
    // changeInput hardcoded 0, so a case with a cellZone ran with the zone silently unrenumbered. The
    // caller that READ the zones is the only one that knows, so it says so here. resetZones
    // (polyTopoChange.C:1600-1968) is what would renumber them; until it is ported, a non-zero count is
    // refused by name inside changeMesh.
    label                nZones = 0;

    // UNIT 7b. Fields the driver carries through every change, as OpenFOAM's registry does: each one is
    // mapped at each change and nothing else is done to it.
    std::vector<std::vector<scalar>> cellScalars;
    std::vector<std::vector<vector>> cellVectors;
    // the old-time volumes. EMPTY until the first change, which is when OpenFOAM's own V0 comes into
    // existence (fvMesh.C:1020-1024: updateMesh only stores them when the CURRENT volumes already
    // exist), and from then on mapped and corrected at every change.
    std::vector<scalar>              V0;
    // The old cell volumes each of a step's two changes is weighted with. Left empty, brae uses its own
    // FvGeometry::V() of the pre-change mesh -- which is what the shipped path must do. The gate injects
    // OpenFOAM's instead, so that a mapper defect and brae's own volume round-off are SEPARABLE: brae's V
    // is validated against OpenFOAM's but is FP-conditioning-limited, not bit-exact
    // (tests/test_mesh_geometry.cu), and a weighted mean carries that round-off into every merged value.
    // Two entries because a step can refine AND unrefine, from two different meshes.
    std::vector<scalar>              injectedRefineOldV;
    std::vector<scalar>              injectedUnrefineOldV;

    // UNIT 7b-2. Surface fields carried the same way: an internal field and one list per patch, mapped at
    // every change, flipped where the map says a face's sense reversed, and then hull-averaged on the
    // injected internal faces.
    //
    // `oriented` IS THE WHOLE DIFFERENCE and it comes from the field FILE's own `oriented` entry, not
    // from its name: an oriented field (a flux) is negated on a flipped face and hull-averaged as an
    // INTENSIVE VECTOR (phi*Sf/sqr(magSf), averaged, then dotted back with Sf); an unoriented one (Uf, or
    // any other surface field) is negated nowhere and averaged as itself. OpenFOAM's own comment on the
    // oriented branch is "Untested."
    struct CarriedSurfaceField
    {
        std::vector<scalar>              field;
        std::vector<std::vector<scalar>> bnd;
        bool                             oriented = true;
    };
    std::vector<CarriedSurfaceField> surfaceScalars;

    // ...and the surface VECTORS, which on this solver is Uf -- the face velocity a moving or adaptive
    // mesh's ddtCorr reads. MEASURED on damBreakWithObstacle: OpenFOAM writes a Uf beside every time
    // directory of that case, because `correctPhi` defaults to mesh.dynamic() and a refining mesh IS
    // dynamic -- so a refine-only case needs Uf carried just as a moving one does.
    struct CarriedSurfaceVectorField
    {
        std::vector<vector>              field;
        std::vector<std::vector<vector>> bnd;
        bool                             oriented = false;
    };
    std::vector<CarriedSurfaceVectorField> surfaceVectors;

    // ...and, per carried surface field, whether the case's correctFluxes names a VELOCITY for it. Where
    // it does, the four write sites of mapFields' own correction run on it, and so does unrefine's second
    // one. Empty or "none" means the field is only mapped -- which is what every interFoam tutorial with
    // an adaptive mesh asks for, all three of them.
    // UNIT 8a. Whole fields -- cells AND patch fields -- carried through every change: the internal field
    // through the cell mapper, then each patch field's own autoMap. Held by POINTER because a
    // GeometricField owns its patch-field objects and is not copyable, and because the solver's fields
    // live in its own state rather than here.
    //
    // THE PATCH OBJECTS MUST BE UPDATED IN PLACE. Every patch field holds a `const FvPatch&` into the
    // driver's patch vector, so that vector is ASSIGNED rather than rebuilt, and the patch count is
    // checked: a change that added or removed a patch would leave every patch field pointing at another
    // patch's data, which is silent. hexRef8 never changes the patch count -- it splits faces within a
    // patch -- and that is checked rather than assumed.
    std::vector<GeometricField<scalar>*> carriedScalarFields;
    std::vector<GeometricField<vector>*> carriedVectorFields;

    std::vector<std::string>         surfaceScalarVelocity;
    // The interpolated flux the correction writes, on the NEW mesh, per change. Injected by the gate so
    // that brae's own surface interpolation is not in the way of the correction's logic; empty means the
    // correction is not run at all.
    // one per change, because a step's refine and its unrefine happen on different meshes
    std::vector<scalar>              injectedPhiURefine;
    std::vector<std::vector<scalar>> injectedPhiURefineBnd;
    std::vector<scalar>              injectedPhiUUnrefine;
    std::vector<std::vector<scalar>> injectedPhiUUnrefineBnd;
    // ...and the one the change being played reads, set by the driver at each site
    std::vector<scalar>              injectedPhiU;
    std::vector<std::vector<scalar>> injectedPhiUBnd;
};

struct RefineUpdateStep
{
    bool                          hasChanged = false;
    bool                          refined = false;
    bool                          unrefined = false;
    std::vector<label>            cellsToRefine;
    std::vector<label>            pointsToUnrefine;
    // refineCell as it stands after the map rebuild and after the buffer layers -- the two states
    // OpenFOAM's own bitSet passes through
    std::vector<char>             refineCellAfterMap;
    std::vector<char>             refineCellAfterBuffer;
    cpu::polyTopoChange::TopoChangeMap refineMap;
    cpu::polyTopoChange::TopoChangeMap unrefineMap;
    bool                          compacted = false;
};

// One update() step. `field` is the driving field on the CURRENT mesh, `timeIndex` OpenFOAM's own time
// index -- the step is a no-op at 0 and whenever timeIndex % refineInterval != 0, exactly as
// updateTopology's guard says. The state's mesh, levels, history and protectedCell are left as the step
// leaves them, so the caller loops.
RefineUpdateStep refineUpdate(
    RefineUpdateState&         s,
    const RefineControls&      c,
    const std::vector<scalar>& field,
    label                      timeIndex);

}   // namespace dynamicRefine
}   // namespace brae
