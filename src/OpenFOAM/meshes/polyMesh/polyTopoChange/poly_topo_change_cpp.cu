// polyTopoChange's face and cell ordering. See the header for what these are and why they are first.
#include "poly_topo_change_cpp.cuh"
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

}   // namespace polyTopoChange
}   // namespace cpu
}   // namespace brae
