// polyTopoChange's face and cell ordering. See the header for what these are and why they are first.
#include "poly_topo_change_cpp.cuh"
#include "foam_dict.cuh"   // isCoupledInterfaceType
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace polyTopoChange {

namespace {
const char* WHO = "brae polyTopoChange: ";
}

void makeCells(
    const OrderInput&    in,
    label                nActiveFaces,
    std::vector<label>&  cellFaces,
    std::vector<label>&  cellFaceOffsets)
{
    const label nCellMap = in.cellMapSize;
    cellFaces.assign(static_cast<std::size_t>(2*nActiveFaces), label(0));
    cellFaceOffsets.assign(static_cast<std::size_t>(nCellMap + 1), label(0));

    // 1. Count faces per cell. OWNER FIRST, over every active face, then NEIGHBOUR over every active
    //    face -- two separate passes, which is what makes a cell's list owner-faces-then-neighbour-faces
    //    rather than a single ascending merge.
    std::vector<label> nNbrs(static_cast<std::size_t>(nCellMap), label(0));
    for (label facei = 0; facei < nActiveFaces; ++facei)
    {
        const label own = in.faceOwner[static_cast<std::size_t>(facei)];
        if (own < 0)
        {
            // polyTopoChange.C:509-514. OpenFOAM builds a newPoints diagnostic here and never uses it
            // before exiting; the refusal is the behaviour.
            throw std::runtime_error(
                std::string(WHO) + "face " + std::to_string(facei) + " is active but its owner has been "
                "deleted. This is usually due to deleting cells without modifying exposed faces to be "
                "boundary faces.");
        }
        ++nNbrs[static_cast<std::size_t>(own)];
    }
    for (label facei = 0; facei < nActiveFaces; ++facei)
    {
        const label nei = in.faceNeighbour[static_cast<std::size_t>(facei)];
        if (nei >= 0) ++nNbrs[static_cast<std::size_t>(nei)];
    }

    // 2. Offsets
    cellFaceOffsets[0] = 0;
    for (label celli = 0; celli < nCellMap; ++celli)
    {
        cellFaceOffsets[static_cast<std::size_t>(celli + 1)] =
            cellFaceOffsets[static_cast<std::size_t>(celli)] + nNbrs[static_cast<std::size_t>(celli)];
    }

    // 3. Fill, reusing nNbrs as the per-cell running counter, as OpenFOAM does
    std::fill(nNbrs.begin(), nNbrs.end(), label(0));
    for (label facei = 0; facei < nActiveFaces; ++facei)
    {
        const label celli = in.faceOwner[static_cast<std::size_t>(facei)];
        cellFaces[static_cast<std::size_t>(cellFaceOffsets[static_cast<std::size_t>(celli)]
                                         + nNbrs[static_cast<std::size_t>(celli)]++)] = facei;
    }
    for (label facei = 0; facei < nActiveFaces; ++facei)
    {
        const label celli = in.faceNeighbour[static_cast<std::size_t>(facei)];
        if (celli >= 0)
        {
            cellFaces[static_cast<std::size_t>(cellFaceOffsets[static_cast<std::size_t>(celli)]
                                             + nNbrs[static_cast<std::size_t>(celli)]++)] = facei;
        }
    }

    // ...and the shrink to the last offset (polyTopoChange.C:555)
    cellFaces.resize(static_cast<std::size_t>(cellFaceOffsets[static_cast<std::size_t>(nCellMap)]));
}


void getFaceOrder(
    const OrderInput&          in,
    label                      nActiveFaces,
    const std::vector<label>&  cellFaces,
    const std::vector<label>&  cellFaceOffsets,
    std::vector<label>&        oldToNew,
    std::vector<label>&        patchSizes,
    std::vector<label>&        patchStarts)
{
    const label nFaces = static_cast<label>(in.faceOwner.size());
    oldToNew.assign(static_cast<std::size_t>(nFaces), label(-1));

    label newFacei = 0;
    std::vector<label> nbr;
    std::vector<label> order;

    for (label celli = 0; celli < in.cellMapSize; ++celli)
    {
        const label startOfCell = cellFaceOffsets[static_cast<std::size_t>(celli)];
        const label nCellFaces  = cellFaceOffsets[static_cast<std::size_t>(celli + 1)] - startOfCell;

        nbr.assign(static_cast<std::size_t>(nCellFaces), label(-1));
        for (label i = 0; i < nCellFaces; ++i)
        {
            const label facei = cellFaces[static_cast<std::size_t>(startOfCell + i)];
            label nbrCelli = in.faceNeighbour[static_cast<std::size_t>(facei)];

            if (facei >= nActiveFaces)
            {
                nbr[static_cast<std::size_t>(i)] = -1;              // retired
            }
            else if (nbrCelli != -1)
            {
                // the other cell, whichever side this face is stored from
                if (nbrCelli == celli) nbrCelli = in.faceOwner[static_cast<std::size_t>(facei)];
                // ...and only the LOWER-numbered cell emits it
                nbr[static_cast<std::size_t>(i)] = (celli < nbrCelli) ? nbrCelli : label(-1);
            }
            else
            {
                nbr[static_cast<std::size_t>(i)] = -1;              // external, done below
            }
        }

        // Foam::sortedOrder is identity() then stableSort (List.C:474-487) -- ASCENDING and STABLE. The
        // stability is load-bearing where one cell meets another across two faces: an unstable sort
        // would order that pair by whatever the implementation happened to do.
        order.resize(static_cast<std::size_t>(nCellFaces));
        std::iota(order.begin(), order.end(), label(0));
        std::stable_sort(order.begin(), order.end(),
                         [&nbr](label a, label b)
                         {
                             return nbr[static_cast<std::size_t>(a)] < nbr[static_cast<std::size_t>(b)];
                         });

        for (const label index : order)
        {
            if (nbr[static_cast<std::size_t>(index)] != -1)
            {
                oldToNew[static_cast<std::size_t>(cellFaces[static_cast<std::size_t>(startOfCell + index)])]
                    = newFacei++;
            }
        }
    }

    // The boundary, patch by patch, in patch face order
    patchStarts.assign(static_cast<std::size_t>(in.nPatches), label(0));
    patchSizes.assign(static_cast<std::size_t>(in.nPatches), label(0));

    if (in.nPatches > 0)
    {
        patchStarts[0] = newFacei;
        for (label facei = 0; facei < nActiveFaces; ++facei)
        {
            const label r = in.region[static_cast<std::size_t>(facei)];
            if (r >= 0) ++patchSizes[static_cast<std::size_t>(r)];
        }
        label facei = patchStarts[0];
        for (label patchi = 0; patchi < in.nPatches; ++patchi)
        {
            patchStarts[static_cast<std::size_t>(patchi)] = facei;
            facei += patchSizes[static_cast<std::size_t>(patchi)];
        }
    }

    // ...each boundary face takes the next slot of its OWN patch, so within a patch the order is
    // ascending OLD face index
    std::vector<label> workPatchStarts(patchStarts);
    for (label facei = 0; facei < nActiveFaces; ++facei)
    {
        const label r = in.region[static_cast<std::size_t>(facei)];
        if (r >= 0)
        {
            oldToNew[static_cast<std::size_t>(facei)] = workPatchStarts[static_cast<std::size_t>(r)]++;
        }
    }

    // Retired faces map to themselves
    for (label facei = nActiveFaces; facei < nFaces; ++facei)
    {
        oldToNew[static_cast<std::size_t>(facei)] = facei;
    }

    // polyTopoChange.C:868-886
    for (label facei = 0; facei < nFaces; ++facei)
    {
        if (oldToNew[static_cast<std::size_t>(facei)] == -1)
        {
            throw std::runtime_error(
                std::string(WHO) + "did not determine new position for face " + std::to_string(facei)
                + " owner " + std::to_string(in.faceOwner[static_cast<std::size_t>(facei)])
                + " neighbour " + std::to_string(in.faceNeighbour[static_cast<std::size_t>(facei)])
                + " region " + std::to_string(in.region[static_cast<std::size_t>(facei)])
                + ". This is usually caused by not specifying a patch for a boundary face.");
        }
    }
}


// ----------------------------------------------------------------------------------------------
// UNIT 2: the compaction, orderCells == false and orderPoints == false -- the path
// dynamicRefineFvMesh takes (dynamicRefineFvMesh.C:462, :585 call changeMesh(mesh, false)).

bool pointRemoved(
    const TopoState& s,
    label            pointi)
{
    // polyTopoChangeI.H:32-41 -- EVERY component, against half of vector::max, not a flag
    const vector& pt = s.points[static_cast<std::size_t>(pointi)];
    const scalar half = scalar(0.5)*kVectorMaxComponent;
    return pt.x > half && pt.y > half && pt.z > half;
}

bool faceRemoved(
    const TopoState& s,
    label            facei)
{
    return s.faces[static_cast<std::size_t>(facei)].empty();     // polyTopoChangeI.H:43-46
}

bool cellRemoved(
    const TopoState& s,
    label            celli)
{
    return s.cellMap[static_cast<std::size_t>(celli)] == label(-2);   // polyTopoChangeI.H:49-52
}


CompactResult compactNoOrder(
    TopoState& s)
{
    CompactResult r;
    const label nPoints = static_cast<label>(s.points.size());
    const label nFaces  = static_cast<label>(s.faces.size());
    const label nCells  = static_cast<label>(s.cellMap.size());

    // ---- points: insertion order minus removed and retired; nInternalPoints stays -1 -------------
    // polyTopoChange.C:983-996
    r.nInternalPoints = -1;
    r.localPointMap.assign(static_cast<std::size_t>(nPoints), label(-1));
    {
        label newPointi = 0;
        for (label pointi = 0; pointi < nPoints; ++pointi)
        {
            const bool retired = std::binary_search(s.retiredPoints.begin(), s.retiredPoints.end(), pointi);
            if (!pointRemoved(s, pointi) && !retired)
            {
                r.localPointMap[static_cast<std::size_t>(pointi)] = newPointi++;
            }
        }
        r.nActivePoints = newPointi;
    }

    // ---- faces, pass one: active faces take 0..nActiveFaces-1 in ascending old index --------------
    // An active face is one that is neither removed nor owned by a deleted cell.
    r.localFaceMap.assign(static_cast<std::size_t>(nFaces), label(-1));
    {
        label newFacei = 0;
        for (label facei = 0; facei < nFaces; ++facei)
        {
            if (!faceRemoved(s, facei) && s.faceOwner[static_cast<std::size_t>(facei)] >= 0)
            {
                r.localFaceMap[static_cast<std::size_t>(facei)] = newFacei++;
            }
        }
        r.nActiveFaces = newFacei;
    }

    // ---- cells: insertion order minus removed ------------------------------------------------------
    // polyTopoChange.C:1199-1213
    r.localCellMap.assign(static_cast<std::size_t>(nCells), label(-1));
    {
        label newCelli = 0;
        for (label celli = 0; celli < nCells; ++celli)
        {
            if (!cellRemoved(s, celli))
            {
                r.localCellMap[static_cast<std::size_t>(celli)] = newCelli++;
            }
        }
        r.nActiveCells = newCelli;
    }

    // ---- the renumber, AND THE FLIP, only when cells were actually removed ------------------------
    // polyTopoChange.C:1222 is `if (orderCells || (newCelli != cellMap_.size()))`. With orderCells
    // false this is "only if a cell went", so a pure REFINEMENT skips the whole block and an
    // UNREFINEMENT does not.
    r.cellsRenumbered = (r.nActiveCells != nCells);
    if (r.cellsRenumbered)
    {
        // polyTopoChange.C:1241-1283
        for (label facei = 0; facei < nFaces; ++facei)
        {
            const label own = s.faceOwner[static_cast<std::size_t>(facei)];
            const label nei = s.faceNeighbour[static_cast<std::size_t>(facei)];
            if (own >= 0)
            {
                s.faceOwner[static_cast<std::size_t>(facei)] =
                    r.localCellMap[static_cast<std::size_t>(own)];
                if (nei >= 0)
                {
                    s.faceNeighbour[static_cast<std::size_t>(facei)] =
                        r.localCellMap[static_cast<std::size_t>(nei)];
                    // ...and the face is reversed when the neighbour became the LOWER cell
                    if (s.faceNeighbour[static_cast<std::size_t>(facei)] >= 0
                     && s.faceNeighbour[static_cast<std::size_t>(facei)]
                      < s.faceOwner[static_cast<std::size_t>(facei)])
                    {
                        std::vector<label>& f = s.faces[static_cast<std::size_t>(facei)];
                        // face::flip -- keep the first vertex, reverse the rest (face.H)
                        if (f.size() > 1) std::reverse(f.begin() + 1, f.end());
                        std::swap(s.faceOwner[static_cast<std::size_t>(facei)],
                                  s.faceNeighbour[static_cast<std::size_t>(facei)]);
                        if (facei < static_cast<label>(s.flipFaceFlux.size()))
                        {
                            s.flipFaceFlux[static_cast<std::size_t>(facei)] =
                                s.flipFaceFlux[static_cast<std::size_t>(facei)] ? 0 : 1;
                        }
                        ++r.nFlipped;
                    }
                }
            }
            else if (nei >= 0)
            {
                s.faceNeighbour[static_cast<std::size_t>(facei)] =
                    r.localCellMap[static_cast<std::size_t>(nei)];
            }
        }
    }
    return r;
}


// ----------------------------------------------------------------------------------------------
// UNIT 3: THE ACTION SURFACE. See the header for the two sentinel conventions and for what the
// std::map choice does and does not reach.

label addPoint(
    TopoActions&  a,
    const vector& pt,
    label         masterPointID,
    bool          inCell)
{
    const label pointi = static_cast<label>(a.state.points.size());
    a.state.points.push_back(pt);
    a.state.pointMap.push_back(masterPointID);
    a.reversePointMap.push_back(pointi);
    if (!inCell)
    {
        // retiredPoints_ is a hash set in OpenFOAM and a SORTED vector here, because the compaction
        // binary-searches it. Appending keeps it sorted: the index just taken is the largest there is.
        a.state.retiredPoints.push_back(pointi);
    }
    return pointi;
}


void modifyPoint(
    TopoActions&  a,
    label         pointi,
    const vector& pt,
    bool          inCell)
{
    a.state.points[static_cast<std::size_t>(pointi)] = pt;
    // polyTopoChange.C:2895-2905: `inCell` false INSERTS into retiredPoints_, true ERASES from it, and
    // the point's own map entry is untouched either way.
    const auto it = std::lower_bound(a.state.retiredPoints.begin(), a.state.retiredPoints.end(), pointi);
    const bool retired = (it != a.state.retiredPoints.end() && *it == pointi);
    if (!inCell && !retired)
    {
        a.state.retiredPoints.insert(it, pointi);
    }
    else if (inCell && retired)
    {
        a.state.retiredPoints.erase(it);
    }
}


void removePoint(
    TopoActions& a,
    label        pointi,
    label        mergePointi)
{
    // :3057-3070. The coordinate is the sentinel the removal predicate tests, so it is set and not
    // merely flagged -- `pointRemoved` reads points, nothing else.
    a.state.points[static_cast<std::size_t>(pointi)] =
        vector{kVectorMaxComponent, kVectorMaxComponent, kVectorMaxComponent};
    a.state.pointMap[static_cast<std::size_t>(pointi)] = -1;
    a.reversePointMap[static_cast<std::size_t>(pointi)] =
        (mergePointi >= 0) ? (-mergePointi - 2) : label(-1);
    const auto it = std::lower_bound(a.state.retiredPoints.begin(), a.state.retiredPoints.end(), pointi);
    if (it != a.state.retiredPoints.end() && *it == pointi) a.state.retiredPoints.erase(it);
}


label addFace(
    TopoActions&              a,
    const std::vector<label>& f,
    label                     own,
    label                     nei,
    label                     masterPointID,
    label                     masterEdgeID,
    label                     masterFaceID,
    bool                      flipFaceFlux,
    label                     patchID)
{
    const label facei = static_cast<label>(a.state.faces.size());
    a.state.faces.push_back(f);
    a.state.region.push_back(patchID);
    a.state.faceOwner.push_back(own);
    a.state.faceNeighbour.push_back(nei);
    // :3105-3127, the four-way choice, in OpenFOAM's order: a POINT master wins over an EDGE master
    // wins over a FACE master, and only the face master reaches faceMap.
    if (masterPointID >= 0)
    {
        a.state.faceMap.push_back(-1);
        a.faceFromPoint[facei] = masterPointID;
    }
    else if (masterEdgeID >= 0)
    {
        a.state.faceMap.push_back(-1);
        a.faceFromEdge[facei] = masterEdgeID;
    }
    else if (masterFaceID >= 0)
    {
        a.state.faceMap.push_back(masterFaceID);
    }
    else
    {
        a.state.faceMap.push_back(-1);
    }
    a.reverseFaceMap.push_back(facei);
    a.state.flipFaceFlux.push_back(flipFaceFlux ? 1 : 0);
    return facei;
}


void modifyFace(
    TopoActions&              a,
    const std::vector<label>& f,
    label                     facei,
    label                     own,
    label                     nei,
    bool                      flipFaceFlux,
    label                     patchID)
{
    // :3255-3262. The face's MAP entry is not touched: a modified face is still the same old face.
    a.state.faces[static_cast<std::size_t>(facei)] = f;
    a.state.faceOwner[static_cast<std::size_t>(facei)] = own;
    a.state.faceNeighbour[static_cast<std::size_t>(facei)] = nei;
    a.state.region[static_cast<std::size_t>(facei)] = patchID;
    a.state.flipFaceFlux[static_cast<std::size_t>(facei)] = flipFaceFlux ? 1 : 0;
}


void removeFace(
    TopoActions& a,
    label        facei,
    label        mergeFacei)
{
    // :3419-3430. The EMPTY vertex list is the removal predicate, as the point sentinel is for points.
    a.state.faces[static_cast<std::size_t>(facei)].clear();
    a.state.region[static_cast<std::size_t>(facei)] = -1;
    a.state.faceOwner[static_cast<std::size_t>(facei)] = -1;
    a.state.faceNeighbour[static_cast<std::size_t>(facei)] = -1;
    a.state.faceMap[static_cast<std::size_t>(facei)] = -1;
    a.reverseFaceMap[static_cast<std::size_t>(facei)] =
        (mergeFacei >= 0) ? (-mergeFacei - 2) : label(-1);
    a.faceFromPoint.erase(facei);
    a.faceFromEdge.erase(facei);
}


label addCell(
    TopoActions& a,
    label        masterPointID,
    label        masterEdgeID,
    label        masterFaceID,
    label        masterCellID)
{
    const label celli = static_cast<label>(a.state.cellMap.size());
    // :3452-3476, the same four-way choice as addFace with a FACE master in the middle
    if (masterPointID >= 0)
    {
        a.state.cellMap.push_back(-1);
        a.cellFromPoint[celli] = masterPointID;
    }
    else if (masterEdgeID >= 0)
    {
        a.state.cellMap.push_back(-1);
        a.cellFromEdge[celli] = masterEdgeID;
    }
    else if (masterFaceID >= 0)
    {
        a.state.cellMap.push_back(-1);
        a.cellFromFace[celli] = masterFaceID;
    }
    else
    {
        a.state.cellMap.push_back(masterCellID);
    }
    a.reverseCellMap.push_back(celli);
    return celli;
}


void removeCell(
    TopoActions& a,
    label        celli,
    label        mergeCelli)
{
    // :3658-3676. THE SENTINEL IS -2, not -1: `cellRemoved` tests for exactly that, and a cell with
    // cellMap -1 is an INFLATED cell that is very much present.
    a.state.cellMap[static_cast<std::size_t>(celli)] = -2;
    a.reverseCellMap[static_cast<std::size_t>(celli)] =
        (mergeCelli >= 0) ? (-mergeCelli - 2) : label(-1);
    a.cellFromPoint.erase(celli);
    a.cellFromEdge.erase(celli);
    a.cellFromFace.erase(celli);
}


void addMesh(
    TopoActions&                           a,
    const std::vector<vector>&             points,
    const std::vector<std::vector<label>>& faces,
    const std::vector<label>&              faceOwner,
    const std::vector<label>&              faceNeighbour,
    label                                  nCells,
    const std::vector<label>&              patchStarts,
    const std::vector<label>&              patchSizes)
{
    // polyTopoChange.C's addMesh, in ITS order and not the mesh file's: every point, then every cell,
    // then the faces -- internal ones first in ascending index, then each patch's in turn. Because
    // every action APPENDS, that order IS the numbering the accumulated state carries, and it is why a
    // no-op round trip comes back as the identity rather than merely as an equivalent mesh.
    a.state.nPatches = static_cast<label>(patchStarts.size());
    for (const vector& p : points)
    {
        addPoint(a, p, static_cast<label>(&p - points.data()), /*inCell=*/true);
    }
    for (label celli = 0; celli < nCells; ++celli)
    {
        // a cell from a cell: no point, edge or face master
        addCell(a, -1, -1, -1, celli);
    }
    const label nInternal = patchStarts.empty()
                          ? static_cast<label>(faces.size())
                          : patchStarts.front();
    for (label facei = 0; facei < nInternal; ++facei)
    {
        addFace(a, faces[static_cast<std::size_t>(facei)],
                faceOwner[static_cast<std::size_t>(facei)],
                faceNeighbour[static_cast<std::size_t>(facei)],
                /*masterPointID=*/-1, /*masterEdgeID=*/-1, /*masterFaceID=*/facei,
                /*flipFaceFlux=*/false, /*patchID=*/-1);
    }
    for (std::size_t pi = 0; pi < patchStarts.size(); ++pi)
    {
        for (label i = 0; i < patchSizes[pi]; ++i)
        {
            const label facei = patchStarts[pi] + i;
            addFace(a, faces[static_cast<std::size_t>(facei)],
                    faceOwner[static_cast<std::size_t>(facei)],
                    /*nei=*/-1,
                    -1, -1, /*masterFaceID=*/facei,
                    false, /*patchID=*/static_cast<label>(pi));
        }
    }
}


// ----------------------------------------------------------------------------------------------
// UNIT 4: changeMesh. See the header for OpenFOAM's order, and for what is refused and why.

namespace {

constexpr const char* WHO4 = "brae polyTopoChange::changeMesh: ";

// polyTopoChangeTemplates.C: `reorder(oldToNew, lst)` -- a COPY, then `lst[oldToNew[i]] = oldLst[i]`
// wherever the new index is not negative. The list keeps its old size; the caller shrinks it.
template <typename T>
void reorderInPlace(
    const std::vector<label>& oldToNew,
    std::vector<T>&           lst,
    label                     newSize)
{
    std::vector<T> old(lst);
    for (std::size_t i = 0; i < old.size(); ++i)
    {
        const label n = oldToNew[i];
        if (n >= 0) lst[static_cast<std::size_t>(n)] = std::move(old[i]);
    }
    lst.resize(static_cast<std::size_t>(newSize));
}

// polyTopoChange.C:57-77. A reverse map carries two encodings and BOTH are renumbered: a plain index
// through oldToNew, and a merge (`-master-2`) by renumbering the master and re-encoding it.
void renumberReverseMap(
    const std::vector<label>& oldToNew,
    std::vector<label>&       elems)
{
    for (label& v : elems)
    {
        if (v >= 0)
        {
            v = oldToNew[static_cast<std::size_t>(v)];
        }
        else if (v < -1)
        {
            v = -oldToNew[static_cast<std::size_t>(-v - 2)] - 2;
        }
    }
}

// :103-120, renumberKey on a Map: the KEY moves, the value does not, and a key that maps to -1 goes.
void renumberKey(
    const std::vector<label>& oldToNew,
    std::map<label, label>&   m)
{
    std::map<label, label> out;
    for (const auto& kv : m)
    {
        const label n = oldToNew[static_cast<std::size_t>(kv.first)];
        if (n >= 0) out[n] = kv.second;
    }
    m.swap(out);
}

// :80-100, renumber on a set of labels: drop what maps to -1. brae keeps it sorted.
void renumberSorted(
    const std::vector<label>& oldToNew,
    std::vector<label>&       s)
{
    std::vector<label> out;
    out.reserve(s.size());
    for (const label v : s)
    {
        const label n = oldToNew[static_cast<std::size_t>(v)];
        if (n >= 0) out.push_back(n);
    }
    std::sort(out.begin(), out.end());
    s.swap(out);
}

// :123-140, renumberCompact on a face's vertex list: renumber and DROP the -1s in place.
void renumberCompactFace(
    const std::vector<label>& oldToNew,
    std::vector<label>&       f)
{
    std::size_t n = 0;
    for (const label v : f)
    {
        const label nv = oldToNew[static_cast<std::size_t>(v)];
        if (nv != -1) f[n++] = nv;
    }
    f.resize(n);
}

// polyTopoChange.C:1290-1370, `getMergeSets`, used once on the faces and once on the cells. Both
// arguments are POST-compaction: `reverseMap` is old -> new and `map` is new -> old.
std::vector<ObjectMap> getMergeSets(
    const std::vector<label>& reverseMap,
    const std::vector<label>& map)
{
    std::vector<label> nMerged(map.size(), 1);
    for (const label newIdx : reverseMap)
    {
        if (newIdx < -1) ++nMerged[static_cast<std::size_t>(-newIdx - 2)];
    }
    // the sets are numbered in ascending NEW index order
    std::vector<label> toSet(map.size(), label(-1));
    label nSets = 0;
    for (std::size_t i = 0; i < nMerged.size(); ++i)
    {
        if (nMerged[i] > 1) toSet[i] = nSets++;
    }
    std::vector<ObjectMap> sets(static_cast<std::size_t>(nSets));
    // ...and filled walking the OLD indices ascending, so element 0 is the master's own old label and
    // the slaves follow in ascending old order. The field mapper depends on element 0 being the master.
    for (std::size_t oldIdx = 0; oldIdx < reverseMap.size(); ++oldIdx)
    {
        const label newIdx = reverseMap[oldIdx];
        if (newIdx >= -1) continue;
        const label master = -newIdx - 2;
        ObjectMap& ms = sets[static_cast<std::size_t>(toSet[static_cast<std::size_t>(master)])];
        if (ms.masterObjects.empty())
        {
            ms.index = master;
            ms.masterObjects.resize(static_cast<std::size_t>(nMerged[static_cast<std::size_t>(master)]));
            ms.masterObjects[0] = map[static_cast<std::size_t>(master)];
            ms.masterObjects[1] = static_cast<label>(oldIdx);
            nMerged[static_cast<std::size_t>(master)] = 2;
        }
        else
        {
            ms.masterObjects[static_cast<std::size_t>(nMerged[static_cast<std::size_t>(master)]++)] =
                static_cast<label>(oldIdx);
        }
    }
    return sets;
}

// polyTopoChange.C:899-930, reorderCompactFaces: every per-FACE array moves together, and the two maps
// that point AT faces are renumbered rather than moved. Called TWICE by compact -- once to remove the
// holes and once to put the faces in upper-triangular and patch order -- which is why it is a function.
void reorderFaceArrays(
    TopoActions&              a,
    const std::vector<label>& oldToNew,
    label                     newSize)
{
    reorderInPlace(oldToNew, a.state.faces, newSize);
    reorderInPlace(oldToNew, a.state.region, newSize);
    reorderInPlace(oldToNew, a.state.faceOwner, newSize);
    reorderInPlace(oldToNew, a.state.faceNeighbour, newSize);
    reorderInPlace(oldToNew, a.state.faceMap, newSize);
    renumberReverseMap(oldToNew, a.reverseFaceMap);
    renumberKey(oldToNew, a.faceFromPoint);
    renumberKey(oldToNew, a.faceFromEdge);
    reorderInPlace(oldToNew, a.state.flipFaceFlux, newSize);
}

}   // namespace


void changeMesh(
    TopoActions&           a,
    const ChangeMeshInput& in,
    ChangedMesh&           out,
    TopoChangeMap&         map)
{
    // ---- the refusals, before anything is consumed -----------------------------------------------
    for (const std::string& t : in.patchTypes)
    {
        if (isCoupledInterfaceType(t) || t == "processor" || t == "processorCyclic")
            throw std::runtime_error(
                std::string(WHO4) + "patch type `" + t + "` is coupled. OpenFOAM reorders a coupled "
                "patch's faces after the change so the two sides still match (reorderCoupledFaces, "
                "polyTopoChange.C:2012-2141), which in the general case is a parallel exchange. No "
                "adaptive interFoam tutorial has a coupled patch, so a port here would be ungated.");
    }
    if (in.nZones != 0)
        throw std::runtime_error(
            std::string(WHO4) + "the mesh carries " + std::to_string(in.nZones) + " zone(s). "
            "resetZones (polyTopoChange.C:1600-1968) renumbers pointZones, faceZones and cellZones "
            "through the change; damBreakWithObstacle, oscillatingBox and motorBike have none, so it "
            "is refused rather than written blind.");
    if (!a.faceFromPoint.empty() || !a.faceFromEdge.empty())
        throw std::runtime_error(
            std::string(WHO4) + "a face was added from a POINT or an EDGE master. Answering that needs "
            "the OLD mesh's pointFaces/edgeFaces (selectFaces, polyTopoChange.C:1417-1476), which this "
            "unit does not carry. Nothing produces such an action until hexRef8::setRefinement does, "
            "and that is the unit that lifts this.");
    if (!a.cellFromPoint.empty() || !a.cellFromEdge.empty() || !a.cellFromFace.empty())
        throw std::runtime_error(
            std::string(WHO4) + "a cell was added from a POINT, EDGE or FACE master. Same as the face "
            "case above: selectCells needs the old mesh's addressing, and hexRef8::setRefinement is the "
            "unit that lifts it.");

    map.nOldPoints = in.nOldPoints;
    map.nOldFaces = in.nOldFaces;
    map.nOldCells = in.nOldCells;

    // ---- 1. compact: the local maps and the flip (unit 2), then the ARRAY REORDER ----------------
    const CompactResult r = compactNoOrder(a.state);

    // points: reorder the coordinates and the map, renumber the reverse map and the retired set, then
    // relabel every face's vertices through it (polyTopoChange.C:1111-1139)
    reorderInPlace(r.localPointMap, a.state.points, r.nActivePoints);
    reorderInPlace(r.localPointMap, a.state.pointMap, r.nActivePoints);
    renumberReverseMap(r.localPointMap, a.reversePointMap);
    renumberSorted(r.localPointMap, a.state.retiredPoints);
    for (std::size_t facei = 0; facei < a.state.faces.size(); ++facei)
    {
        std::vector<label>& f = a.state.faces[facei];
        renumberCompactFace(r.localPointMap, f);
        if (!faceRemoved(a.state, static_cast<label>(facei)) && f.size() < 3)
            throw std::runtime_error(
                std::string(WHO4) + "filtering removed points left face " + std::to_string(facei)
                + " with " + std::to_string(f.size()) + " vertices. OpenFOAM's own FatalError: "
                "\"Created illegal face\".");
    }

    // faces: OpenFOAM's localFaceMap has a SECOND pass that gives every RETIRED face -- not removed but
    // with no owner -- an index after the active ones (polyTopoChange.C:1150-1158), so the array keeps
    // them and only nActiveFaces_ excludes them. compactNoOrder stops at the active block because unit
    // 2's gate could not see the rest; the tail is added here. NOT WITNESSED by this unit's arms
    // either: nothing hexRef8 does retires a face, and the three scenarios produce none.
    std::vector<label> faceMapLocal(r.localFaceMap);
    label newFacei = r.nActiveFaces;
    for (std::size_t facei = 0; facei < a.state.faces.size(); ++facei)
    {
        if (!faceRemoved(a.state, static_cast<label>(facei))
         && a.state.faceOwner[facei] < 0
         && faceMapLocal[facei] < 0)
        {
            faceMapLocal[facei] = newFacei++;
        }
    }
    reorderFaceArrays(a, faceMapLocal, newFacei);

    // cells: the reorder of the map and the reverse map. compactNoOrder has already renumbered
    // owner/neighbour and done the flip, which is why that had to run before the face reorder.
    reorderInPlace(r.localCellMap, a.state.cellMap, r.nActiveCells);
    renumberReverseMap(r.localCellMap, a.reverseCellMap);
    renumberKey(r.localCellMap, a.cellFromPoint);
    renumberKey(r.localCellMap, a.cellFromEdge);
    renumberKey(r.localCellMap, a.cellFromFace);

    // ---- 1b. AND THE FACES GO INTO UPPER-TRIANGULAR AND PATCH ORDER -------------------------------
    // polyTopoChange.C:1288-1308, the LAST thing compact does, after the cell renumber and the flip:
    //     makeCells(nActiveFaces_, cellFaces, cellFaceOffsets);
    //     getFaceOrder(nActiveFaces_, ..., localFaceMap, patchSizes, patchStarts);
    //     reorderCompactFaces(localFaceMap.size(), localFaceMap);
    // Units 1 and 2 ported both functions and gated them on a real mesh, but compactNoOrder never
    // CALLED them -- it stopped at the hole removal. FOUND BY THIS UNIT'S OWN GATE: with the step
    // missing, `noop` passed (nothing moves when nothing is removed) while `mergeTwoCells` read
    // faceMap[0] = 1 where OpenFOAM has 2, neighbour[0] = 22 where OpenFOAM has 1, and face 0's
    // vertex list differed. A removal reshuffles which face is a cell's first upper-triangular
    // neighbour, and the ascending-index order the hole removal leaves is not that order.
    //
    // patchStarts and patchSizes come OUT of getFaceOrder; deriving them from `region` afterwards
    // would be a second implementation of the same fact, and it got empty patches wrong.
    {
        OrderInput oi;
        oi.cellMapSize = r.nActiveCells;
        oi.faceOwner = a.state.faceOwner;
        oi.faceNeighbour = a.state.faceNeighbour;
        oi.region = a.state.region;
        oi.nPatches = a.state.nPatches;
        std::vector<label> cellFaces, cellFaceOffsets;
        makeCells(oi, r.nActiveFaces, cellFaces, cellFaceOffsets);
        std::vector<label> oldToNew, patchSizes, patchStarts;
        getFaceOrder(oi, r.nActiveFaces, cellFaces, cellFaceOffsets, oldToNew, patchSizes, patchStarts);
        reorderFaceArrays(a, oldToNew, static_cast<label>(oldToNew.size()));
        out.patchStarts = patchStarts;
        out.patchSizes = patchSizes;
        out.nInternalFaces = patchStarts.empty() ? r.nActiveFaces : patchStarts.front();
    }

    // ---- 3, 4. the inflation maps. Only the MERGE halves are reachable here; the from-point and
    // from-edge halves were refused above, so they come out empty as OpenFOAM's would.
    // ...and the POINT merge set, which OpenFOAM builds with the very same function -- "For point only
    // point merging" (polyTopoChange.C:2211-2217). Three calls, one arithmetic.
    map.pointsFromPoints = getMergeSets(a.reversePointMap, a.state.pointMap);
    map.facesFromFaces = getMergeSets(a.reverseFaceMap, a.state.faceMap);
    map.cellsFromCells = getMergeSets(a.reverseCellMap, a.state.cellMap);

    // ---- 5. the mesh the change produced ---------------------------------------------------------
    out.points = a.state.points;
    out.faces.assign(a.state.faces.begin(), a.state.faces.begin() + r.nActiveFaces);
    out.faceOwner.assign(a.state.faceOwner.begin(), a.state.faceOwner.begin() + r.nActiveFaces);
    out.nCells = r.nActiveCells;
    out.faceNeighbour.assign(a.state.faceNeighbour.begin(),
                             a.state.faceNeighbour.begin() + out.nInternalFaces);

    // ---- 7. the map ------------------------------------------------------------------------------
    map.pointMap = a.state.pointMap;
    map.faceMap.assign(a.state.faceMap.begin(), a.state.faceMap.begin() + r.nActiveFaces);
    map.cellMap = a.state.cellMap;
    map.reversePointMap = a.reversePointMap;
    map.reverseFaceMap = a.reverseFaceMap;
    map.reverseCellMap = a.reverseCellMap;
    // HashSetOps::used(flipFaceFlux_) -- the NEW faces whose flag is set, which a hash set hands back
    // in no order and mapPolyMesh's reader sorts. Built ascending here.
    map.flipFaceFlux.clear();
    for (label f = 0; f < r.nActiveFaces; ++f)
    {
        if (a.state.flipFaceFlux[static_cast<std::size_t>(f)]) map.flipFaceFlux.push_back(f);
    }
    map.oldPatchStarts = in.oldPatchStarts;
    map.oldPatchNMeshPoints = in.oldPatchNMeshPoints;
    // mapPolyMesh.C:156-173 derives the sizes rather than storing them: each patch's start to the next,
    // and the LAST one to nOldFaces.
    map.oldPatchSizes.assign(in.oldPatchStarts.size(), label(0));
    for (std::size_t p = 0; p + 1 < in.oldPatchStarts.size(); ++p)
    {
        map.oldPatchSizes[p] = in.oldPatchStarts[p + 1] - in.oldPatchStarts[p];
    }
    if (!in.oldPatchStarts.empty())
    {
        map.oldPatchSizes.back() = map.nOldFaces - in.oldPatchStarts.back();
    }
}

}   // namespace polyTopoChange
}   // namespace cpu
}   // namespace brae
