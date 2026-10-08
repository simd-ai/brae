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
#include "compact_list_list.cuh"
#include <cstddef>
#include <map>
#include <string>
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

// polyTopoChange's faces_ -- a DynamicList<face> there -- HELD FLAT: every face's vertices in one pool and a
// (start, size) a face. As std::vector<std::vector<label>> it was a heap block a face, ~280,000 on a 90,000-cell
// mesh: built from the mesh on the way in, permuted twice by the reorder, and copied face by face on the way
// out. MEASURED on damBreakWithObstacle, ms a step: the way in 20.9, the two reorders 11.6 + part of 15.0, the
// copy out 6.0 -- 57.5 for the round trip, of a 231 ms step.
//   an added face goes to the pool's end; a MODIFIED face's new list goes there too (it may be longer than the
//   one it replaces) and the old one is left behind; a removed face has size 0; the reorder moves the (start,
//   size) pairs and not one vertex; compact(n) gathers the first n faces, in order, into a CompactListList.
class DynamicFaceList
{
public:
    std::size_t size() const { return start_.size(); }
    LabelRow operator[](std::size_t facei) const
    {
        return LabelRow{pool_.data() + static_cast<std::size_t>(start_[facei]),
                        static_cast<std::size_t>(size_[facei])};
    }
    // a face's vertices, to change IN PLACE (face::flip, the renumbering): never past its size
    label* vertices(std::size_t facei) { return pool_.data() + static_cast<std::size_t>(start_[facei]); }
    void reserve(
        std::size_t nFaces,
        std::size_t nVertices)
    {
        start_.reserve(nFaces);
        size_.reserve(nFaces);
        pool_.reserve(nVertices);
    }
    void push_back(LabelRow f)
    {
        const label at = toPool(f);
        start_.push_back(at);
        size_.push_back(static_cast<label>(f.size()));
    }
    // n faces at once, from their compact form (n + 1 offsets from 0, and the vertices): what n push_backs in
    // order leave, in three copies
    void appendFlat(
        const std::vector<label>& offsets,
        const std::vector<label>& vertices,
        std::size_t               n)
    {
        const label base = static_cast<label>(pool_.size());
        const std::size_t nVerts = static_cast<std::size_t>(offsets[n]);
        pool_.insert(pool_.end(), vertices.begin(), vertices.begin() + static_cast<std::ptrdiff_t>(nVerts));
        for (std::size_t i = 0; i < n; ++i)
        {
            start_.push_back(base + offsets[i]);
            size_.push_back(offsets[i + 1] - offsets[i]);
        }
    }
    void set(
        std::size_t facei,
        LabelRow    f)
    {
        const label at = toPool(f);
        start_[facei] = at;
        size_[facei] = static_cast<label>(f.size());
    }
    void clear(std::size_t facei) { size_[facei] = 0; }
    // keep a face's first n vertices
    void shrink(
        std::size_t facei,
        std::size_t n)
    {
        size_[facei] = static_cast<label>(n);
    }
    // polyTopoChangeTemplates.C's `reorder(oldToNew, lst)` and the shrink its caller does: face i goes to
    // oldToNew[i] where that is not negative, a place nothing goes to keeps the face it had, the list is cut
    // to newSize. The pool is not touched.
    void reorder(
        const std::vector<label>& oldToNew,
        label                     newSize)
    {
        const std::vector<label> oldStart(start_);
        const std::vector<label> oldSize(size_);
        for (std::size_t i = 0; i < oldStart.size(); ++i)
        {
            const label n = oldToNew[i];
            if (n >= 0)
            {
                start_[static_cast<std::size_t>(n)] = oldStart[i];
                size_[static_cast<std::size_t>(n)] = oldSize[i];
            }
        }
        start_.resize(static_cast<std::size_t>(newSize));
        size_.resize(static_cast<std::size_t>(newSize));
    }
    // the first n faces, in order
    CompactListList compact(std::size_t n) const
    {
        std::size_t total = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            total += static_cast<std::size_t>(size_[i]);
        }
        CompactListList out;
        out.start(n, total);
        for (std::size_t i = 0; i < n; ++i)
        {
            out.appendRow((*this)[i]);
        }
        return out;
    }
    std::vector<std::vector<label>> unpack() const { return compact(size()).unpack(); }

private:
    // the list written at the pool's end; one that is a row of this pool is copied out first, since the
    // insert may move the pool
    label toPool(LabelRow f)
    {
        const label at = static_cast<label>(pool_.size());
        if (!pool_.empty() && f.begin() >= pool_.data() && f.begin() < pool_.data() + pool_.size())
        {
            const std::vector<label> copy(f.begin(), f.end());
            pool_.insert(pool_.end(), copy.begin(), copy.end());
        }
        else
        {
            pool_.insert(pool_.end(), f.begin(), f.end());
        }
        return at;
    }

    std::vector<label> start_;
    std::vector<label> size_;
    std::vector<label> pool_;
};

struct TopoState
{
    std::vector<vector>               points;          // removed = every component > 0.5*vector::max
    DynamicFaceList                   faces;           // removed = empty
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


// ----------------------------------------------------------------------------------------------
// UNIT 3: THE ACTION SURFACE -- what a topology change is written INTO.
//
// polyTopoChange.C:2785-3700. `hexRef8::setRefinement` never touches the mesh: it calls
// `meshMod.setAction(polyAddPoint(...))`, `polyAddCell(...)`, `polyAddFace(...)` and
// `polyModifyFace(...)`, and `setUnrefinement` adds the three removals. This is that surface, and the
// state it accumulates -- nothing here decides anything, which is why it has no oracle of its own.
//
// TWO SENTINEL CONVENTIONS, transcribed rather than unified, because OpenFOAM does not unify them:
//   * a removed POINT has `pointMap = -1` and its coordinate set to `point::max`;
//     a removed FACE has `faceMap = -1` and an EMPTY vertex list;
//     a removed CELL has `cellMap = **-2**` -- not -1. (polyTopoChange.C:3026, :3392, :3634.)
//   * a MERGE is recorded in the REVERSE map as `-master-2`, and a plain removal as -1. So
//     `reverseCellMap[celli] == -5` means "cell celli was merged into cell 3", and the inflation maps
//     below are built by decoding exactly that.
//
// `addPoint`/`addFace`/`addCell` return the index they appended at, which is what `setRefinement` keys
// its own tables on -- so the return values are part of the contract, not a convenience.
//
// THE FROM-MAPS ARE std::map HERE AND A HASH IN OpenFOAM (`Map<label>`). The VALUES are identical; the
// ITERATION ORDER is not, and it reaches the answer in exactly one place: the order of entries within
// `facesFromPointsMap` / `cellsFromPointsMap` and friends, which `calcFaceInflationMaps` builds by
// walking the map. The three arms of the unit-3+4 gate leave all four of those lists EMPTY -- only a
// real 8-way split fills them -- so this is UNWITNESSED here and is settled at unit 5, against
// OpenFOAM's own map for a real refinement step. Said here rather than left as a silent choice.
struct TopoActions
{
    // the accumulated state, in OpenFOAM's own member order
    TopoState state;
    // reverseXMap: the new index of each accumulated entry, or -master-2 where it was merged away
    std::vector<label> reversePointMap;
    std::vector<label> reverseFaceMap;
    std::vector<label> reverseCellMap;
    // faces and cells INFLATED from a lower-dimensional master: new index -> master index
    std::map<label, label> faceFromPoint;
    std::map<label, label> faceFromEdge;
    std::map<label, label> cellFromPoint;
    std::map<label, label> cellFromEdge;
    std::map<label, label> cellFromFace;
};

// polyTopoChange.C:2785-2811. `inCell` false RETIRES the point: it keeps its slot but is not part of
// the mesh, and the compaction drops it.
label addPoint(
    TopoActions&  a,
    const vector& pt,
    label         masterPointID,
    bool          inCell);

// :2864-2944, the single-zone form with `multiZone` false -- which is the only form hexRef8 reaches.
void modifyPoint(
    TopoActions&  a,
    label         pointi,
    const vector& pt,
    bool          inCell);

// :3026-3076. `mergePointi >= 0` records the merge in reversePointMap as -mergePointi-2.
void removePoint(
    TopoActions& a,
    label        pointi,
    label        mergePointi);

// :3078-3140. Exactly one of masterPointID / masterEdgeID / masterFaceID is expected to be >= 0, and
// OpenFOAM's own "need to specify a master" FatalError is COMMENTED OUT there (:3121-3126) -- a face
// with no master is allowed and gets faceMap -1, so this does not throw either.
label addFace(
    TopoActions&              a,
    LabelRow                  f,
    label                     own,
    label                     nei,
    label                     masterPointID,
    label                     masterEdgeID,
    label                     masterFaceID,
    bool                      flipFaceFlux,
    label                     patchID);

// :3236-3322, `multiZone` false. Note the argument list is NOT addFace's: polyModifyFace carries a
// `removeFromZone` flag between patchID and zoneID that polyAddFace has no equivalent of. Zones are
// refused at changeMesh, so neither is kept here -- but the ACTION RECORD the gate replays has the
// slot, because OpenFOAM's does.
void modifyFace(
    TopoActions&              a,
    LabelRow                  f,
    label                     facei,
    label                     own,
    label                     nei,
    bool                      flipFaceFlux,
    label                     patchID);

// :3392-3440.
void removeFace(
    TopoActions& a,
    label        facei,
    label        mergeFacei);

// :3443-3477.
label addCell(
    TopoActions& a,
    label        masterPointID,
    label        masterEdgeID,
    label        masterFaceID,
    label        masterCellID);

// :3634-3680. The removed cell's own cellFrom* entries are ERASED, so a cell inflated from a face and
// then removed leaves no trace in the inflation maps.
void removeCell(
    TopoActions& a,
    label        celli,
    label        mergeCelli);

// polyTopoChange.C's `addMesh` (:2400-2600) reduced to what a mesh with no zones needs: every point,
// cell and face of an existing mesh played in through the actions above, in OpenFOAM's order -- points
// first, then cells, then the internal faces, then each patch's faces. This is what
// `polyTopoChange meshMod(mesh)` does, and it is why the no-op round trip is the identity.
//
// `patchStarts` and `patchSizes` describe the boundary as `constant/polyMesh/boundary` does.
// `faces` in either form (LabelListListRef): a list of lists, a CompactListList, or a mesh's own faceOffsets and
// faceVerts
void addMesh(
    TopoActions&                           a,
    const std::vector<vector>&             points,
    LabelListListRef                       faces,
    const std::vector<label>&              faceOwner,
    const std::vector<label>&              faceNeighbour,
    label                                  nCells,
    const std::vector<label>&              patchStarts,
    const std::vector<label>&              patchSizes);


// ----------------------------------------------------------------------------------------------
// UNIT 4: changeMesh -- the accumulated actions turned into a MESH and a MAP.
//
// polyTopoChange.C:3775-4022, on the arguments `dynamicRefineFvMesh` passes: `changeMesh(*this, false)`
// (dynamicRefineFvMesh.C:462, :585), which is `patchMap` empty, `inflate` FALSE, `syncParallel` true,
// `orderCells` FALSE and `orderPoints` FALSE. Everything below is that path and no other.
//
// The order is OpenFOAM's compactAndReorder (:2143-2270) followed by :3830-4022:
//   1. compact                     -- units 2 and 4: the local maps, the array reorder, the flip
//   2. reorderCoupledFaces         -- REFUSED (see below)
//   3. calcFaceInflationMaps       -- facesFromFaces here; the from-point/edge halves REFUSED
//   4. calcCellInflationMaps       -- cellsFromCells here; likewise
//   5. resetPrimitives             -- the new mesh
//   6. resetZones                  -- REFUSED
//   7. the mapPolyMesh itself
//
// WHAT IS REFUSED, each because it cannot be witnessed by the fixture that gates this unit rather than
// because it is hard:
//   * a COUPLED patch. `reorderCoupledFaces` (:2012-2141) reorders a coupled patch's faces so the two
//     sides match after the change, and it is a PARALLEL exchange in the general case. No adaptive
//     interFoam tutorial has a coupled patch and brae runs serial, so a port here would be ungated.
//   * ZONES. `resetZones` (:1600-1968) is 369 lines of pointZone/faceZone/cellZone renumbering.
//     laminar/damBreakWithObstacle, oscillatingBox and RAS/motorBike have none.
//   * `inflate == true`. dynamicRefineFvMesh passes false at both call sites; the true branch moves the
//     new points onto the old ones and is a different mesh.
//   * `orderCells` / `orderPoints`. Both default false and dynamicRefineFvMesh takes the defaults, so
//     the Cuthill-McKee cell ordering and the internal/external point split are never reached -- unit 2
//     recorded that and this keeps it.
//   * INFLATION FROM A LOWER-DIMENSIONAL MASTER: a face added from a point or an edge, or a cell added
//     from a point, an edge or a face. `calcFaceInflationMaps` needs the OLD mesh's `pointFaces` and
//     `edgeFaces` to answer them (`selectFaces`), which this unit does not carry, and nothing produces
//     such an action until `hexRef8::setRefinement` exists. Refused by name so unit 5 has to lift it.
//
// THE MERGE SETS ARE THE ONE PIECE OF REAL ARITHMETIC HERE and they are `getMergeSets` (:1290-1370),
// used twice -- once on the faces and once on the cells. It decodes the `-master-2` convention: every
// old index whose reverse map is below -1 was merged into `-value-2`, the sets come out in ascending
// NEW index order, and within a set element 0 is the master's OLD label and the rest are the slaves in
// ascending old order. That asymmetry is OpenFOAM's and the field mapper depends on it: element 0 is
// what an interpolative map treats as the cell that was already there.

// objectMap: one new index and the old ones whose values make it
struct ObjectMap
{
    label              index = -1;
    std::vector<label> masterObjects;
};

// mapPolyMesh's content, in OpenFOAM's own names. A SUPERSET of brae's MapPolyMesh, which is shaped for
// the consuming side (cell and face mappers) and is derived from this rather than replaced by it -- the
// field mapping is already gated against OpenFOAM's own map and must not be disturbed by this unit.
struct TopoChangeMap
{
    label nOldPoints = 0;
    label nOldFaces = 0;
    label nOldCells = 0;
    std::vector<label> pointMap;          // new -> old, -1 where inflated
    std::vector<label> faceMap;
    std::vector<label> cellMap;
    std::vector<label> reversePointMap;   // old -> new, -master-2 where merged, -1 where removed
    std::vector<label> reverseFaceMap;
    std::vector<label> reverseCellMap;
    std::vector<label> flipFaceFlux;      // the NEW faces whose flux flipped, ascending
    std::vector<ObjectMap> pointsFromPoints;
    std::vector<ObjectMap> facesFromPoints;
    std::vector<ObjectMap> facesFromEdges;
    std::vector<ObjectMap> facesFromFaces;
    std::vector<ObjectMap> cellsFromPoints;
    std::vector<ObjectMap> cellsFromEdges;
    std::vector<ObjectMap> cellsFromFaces;
    std::vector<ObjectMap> cellsFromCells;
    std::vector<label> oldPatchStarts;
    std::vector<label> oldPatchSizes;     // DERIVED, as mapPolyMesh.C:156-173 derives it: the next
                                          // patch's start, and nOldFaces for the last
    std::vector<label> oldPatchNMeshPoints;
};

// The mesh the change produced, in brae's own shapes: `faceNeighbour` over the INTERNAL faces only, as
// PrimitiveMesh keeps it, and the patches as `constant/polyMesh/boundary` describes them.
struct ChangedMesh
{
    std::vector<vector>             points;
    // every face's vertex list, compact (compact_list_list.cuh): its offsets and values ARE a PrimitiveMesh's
    // faceOffsets and faceVerts
    CompactListList                 faces;
    std::vector<label>              faceOwner;       // nFaces
    std::vector<label>              faceNeighbour;   // nInternalFaces
    std::vector<label>              patchStarts;
    std::vector<label>              patchSizes;
    label                           nInternalFaces = 0;
    label                           nCells = 0;
};

// What the OLD mesh has to say: its patch description, and whether it carries anything this unit
// refuses. `patchTypes` is only read to refuse a coupled one; `nZones` likewise.
struct ChangeMeshInput
{
    label                    nOldPoints = 0;
    label                    nOldFaces = 0;
    label                    nOldCells = 0;
    std::vector<label>       oldPatchStarts;
    std::vector<label>       oldPatchSizes;
    std::vector<label>       oldPatchNMeshPoints;   // patch.meshPoints().size() per patch
    std::vector<std::string> patchTypes;
    label                    nZones = 0;            // pointZones + faceZones; cellZones are CARRIED
};

// polyTopoChange::changeMesh on the path above. Consumes `a` (OpenFOAM's members are invalid after it
// returns, and this mutates the state in the same way).
void changeMesh(
    TopoActions&           a,
    const ChangeMeshInput& in,
    ChangedMesh&           out,
    TopoChangeMap&         map);

}   // namespace polyTopoChange
}   // namespace cpu
}   // namespace brae
