#include "dynamic_refine_fv_mesh_cpp.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {
namespace dynamicRefine {

namespace {

const char* const WHO = "brae dynamicRefineFvMesh: ";

// OF doubleScalar.H:58 -- GREAT is 1.0e+15, not VGREAT and not the largest double
constexpr scalar GREAT = scalar(1.0e+15);

}   // namespace


std::vector<scalar> cellToPoint(
    const std::vector<scalar>&             vFld,
    const std::vector<std::vector<label>>& pointCells)
{
    std::vector<scalar> pFld(pointCells.size());
    for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
    {
        const std::vector<label>& pCells = pointCells[pointi];
        if (pCells.empty())
        {
            // OpenFOAM divides by pCells.size() unguarded (dynamicRefineFvMesh.C:768) and would
            // produce a NaN here. A NaN that propagates into a refinement decision is a mesh nobody
            // can explain afterwards, so this names the point instead.
            throw std::runtime_error(
                std::string(WHO) + "point " + std::to_string(pointi) + " has no cells, so the "
                "cell-to-point average has nothing to divide by. OpenFOAM divides by zero here.");
        }
        scalar sum = scalar(0);
        for (const label celli : pCells)
        {
            sum += vFld[static_cast<std::size_t>(celli)];
        }
        pFld[pointi] = sum/static_cast<scalar>(pCells.size());
    }
    return pFld;
}


std::vector<scalar> error(
    const std::vector<scalar>& fld,
    scalar                     minLevel,
    scalar                     maxLevel)
{
    std::vector<scalar> c(fld.size(), scalar(-1));
    for (std::size_t i = 0; i < fld.size(); ++i)
    {
        const scalar err = std::fmin(fld[i] - minLevel, maxLevel - fld[i]);
        if (err >= scalar(0))
        {
            c[i] = err;
        }
    }
    return c;
}


std::vector<scalar> maxPointField(
    const std::vector<scalar>&             pFld,
    const std::vector<std::vector<label>>& pointCells,
    label                                  nCells)
{
    std::vector<scalar> vFld(static_cast<std::size_t>(nCells), -GREAT);
    for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
    {
        for (const label celli : pointCells[pointi])
        {
            scalar& v = vFld[static_cast<std::size_t>(celli)];
            v = std::fmax(v, pFld[pointi]);
        }
    }
    return vFld;
}


std::vector<scalar> maxCellField(
    const std::vector<scalar>&             vFld,
    const std::vector<std::vector<label>>& pointCells)
{
    std::vector<scalar> pFld(pointCells.size(), -GREAT);
    for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
    {
        for (const label celli : pointCells[pointi])
        {
            pFld[pointi] = std::fmax(pFld[pointi], vFld[static_cast<std::size_t>(celli)]);
        }
    }
    return pFld;
}


void selectRefineCandidates(
    scalar                                 lowerRefineLevel,
    scalar                                 upperRefineLevel,
    const std::vector<scalar>&             vFld,
    const std::vector<std::vector<label>>& pointCells,
    label                                  nCells,
    std::vector<char>&                     candidateCell)
{
    if (candidateCell.size() != static_cast<std::size_t>(nCells))
    {
        throw std::runtime_error(
            std::string(WHO) + "the candidate marker is " + std::to_string(candidateCell.size())
            + " long where the mesh has " + std::to_string(nCells) + " cells. OpenFOAM's bitSet is "
            "sized nCells() at the call site (dynamicRefineFvMesh.C:1358) and this function only "
            "ever sets bits -- it does not clear them, and it does not resize.");
    }
    const std::vector<scalar> cellError =
        maxPointField(error(cellToPoint(vFld, pointCells), lowerRefineLevel, upperRefineLevel),
                      pointCells, nCells);
    for (std::size_t celli = 0; celli < cellError.size(); ++celli)
    {
        if (cellError[celli] > scalar(0))
        {
            candidateCell[celli] = 1;
        }
    }
}

label faceConsistentRefinement(
    bool                        maxSet,
    const std::vector<label>&   cellLevel,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    std::vector<char>&          refineCell)
{
    label nChanged = 0;
    // hexRef8.C:1578-1610, internal faces
    const label nInternal = m.nInternalFaces();
    for (label facei = 0; facei < nInternal; ++facei)
    {
        const label own = m.owner()[static_cast<std::size_t>(facei)];
        const label nei = m.neighbour()[static_cast<std::size_t>(facei)];
        const label ownLevel = cellLevel[static_cast<std::size_t>(own)]
                             + refineCell[static_cast<std::size_t>(own)];
        const label neiLevel = cellLevel[static_cast<std::size_t>(nei)]
                             + refineCell[static_cast<std::size_t>(nei)];
        if (ownLevel > neiLevel + 1)
        {
            if (maxSet) refineCell[static_cast<std::size_t>(nei)] = 1;
            else        refineCell[static_cast<std::size_t>(own)] = 0;
            ++nChanged;
        }
        else if (neiLevel > ownLevel + 1)
        {
            if (maxSet) refineCell[static_cast<std::size_t>(own)] = 1;
            else        refineCell[static_cast<std::size_t>(nei)] = 0;
            ++nChanged;
        }
    }

    // hexRef8.C:1613-1649, coupled faces. The swap is what makes the boundary loop do anything: on an
    // uncoupled patch neiLevel keeps the OWNER's own level, both tests are false, and nothing moves.
    for (const FvPatch& q : patches)
    {
        if (!q.coupled) continue;
        throw std::runtime_error(
            std::string(WHO) + "the mesh carries the coupled patch `" + q.name + "`, and the 2:1 "
            "refinement closure swaps cell levels across it (hexRef8.C:1625, "
            "syncTools::swapBoundaryFaceList). brae has no swap here, so a cell on the far side of "
            "that pair would keep a level it does not have and the closure would be wrong in a "
            "direction nothing reports.");
    }
    return nChanged;
}


std::vector<label> consistentRefinement(
    const std::vector<label>&   cellLevel,
    const std::vector<label>&   cellsToRefine,
    bool                        maxSet,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches)
{
    // hexRef8.C:2246 -- bitSet(nCells, cellsToRefine)
    std::vector<char> refineCell(static_cast<std::size_t>(m.nCells()), 0);
    for (const label c : cellsToRefine)
    {
        if (c < 0 || c >= m.nCells())
        {
            throw std::runtime_error(
                std::string(WHO) + "a cell to refine is outside the mesh: " + std::to_string(c));
        }
        refineCell[static_cast<std::size_t>(c)] = 1;
    }
    // :2248-2270 -- to a fixed point
    while (faceConsistentRefinement(maxSet, cellLevel, m, patches, refineCell) != 0)
    {
    }
    // :2273 -- bitSet::toc(), the on bits ASCENDING
    std::vector<label> out;
    for (std::size_t c = 0; c < refineCell.size(); ++c)
    {
        if (refineCell[c]) out.push_back(static_cast<label>(c));
    }
    return out;
}


std::vector<label> selectRefineCells(
    label                       maxCells,
    label                       maxRefinement,
    const std::vector<char>&    candidateCell,
    const std::vector<label>&   cellLevel,
    const std::vector<char>&    protectedCell,
    label                       nTotalCells,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches)
{
    // dynamicRefineFvMesh.C:838. INTEGER division: each refined hex becomes eight, so seven extra.
    const label nTotToRefine = (maxCells - nTotalCells)/7;

    // :848-849. Serial, so the local count IS the global one.
    label nCandidates = 0;
    for (const char c : candidateCell) nCandidates += (c != 0);

    // :844-845. The protected set is CASCADED first, not used raw: a protected cell cannot be
    // refined, so nor can any finer neighbour of one, and that propagates. On an all-hex mesh the set
    // is empty and the cascade returns empty, which is why taking the raw set here was
    // indistinguishable until a mesh with a non-hex cell existed to tell them apart.
    const std::vector<char> unrefineableCell =
        calculateProtectedCells(protectedCell, cellLevel, m, patches);
    auto unrefineable = [&](std::size_t c)
    {
        return c < unrefineableCell.size() && unrefineableCell[c] != 0;
    };

    std::vector<label> candidates;
    if (nCandidates < nTotToRefine)
    {
        // :856-866 -- every candidate under the level cap, ascending, no truncation
        for (std::size_t c = 0; c < candidateCell.size(); ++c)
        {
            if (!candidateCell[c]) continue;
            if (!unrefineable(c) && cellLevel[c] < maxRefinement)
            {
                candidates.push_back(static_cast<label>(c));
            }
        }
    }
    else
    {
        // :871-889 -- WHOLE LEVELS, coarsest first, and the budget is tested only after a level is
        // finished, so the list can overshoot by up to one level's worth
        for (label level = 0; level < maxRefinement; ++level)
        {
            for (std::size_t c = 0; c < candidateCell.size(); ++c)
            {
                if (!candidateCell[c]) continue;
                if (!unrefineable(c) && cellLevel[c] == level)
                {
                    candidates.push_back(static_cast<label>(c));
                }
            }
            if (static_cast<label>(candidates.size()) > nTotToRefine)
            {
                break;
            }
        }
    }

    // :893-900 -- the 2:1 closure, maxSet TRUE, which ADDS cells
    return consistentRefinement(cellLevel, candidates, /*maxSet=*/true, m, patches);
}


namespace {

// The coupled-patch refusal the three functions below share. Each of them crosses the boundary with a
// syncTools call in OpenFOAM, and each would be wrong in a direction nothing reports without it.
void refuseCoupled(
    const std::vector<FvPatch>& patches,
    const char*                 what,
    const char*                 ofSite)
{
    for (const FvPatch& q : patches)
    {
        if (!q.coupled) continue;
        throw std::runtime_error(
            std::string(WHO) + what + " crosses the coupled patch `" + q.name + "`, where OpenFOAM "
            "synchronises across the pair (" + ofSite + "). brae has no sync here, so the far side "
            "would keep a value it does not have.");
    }
}

}   // namespace


void extendMarkedCells(
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches,
    const std::vector<std::vector<label>>& cells,
    std::vector<char>&                     markedCell)
{
    refuseCoupled(patches, "the buffer-layer dilation", "dynamicRefineFvMesh.C:1018");

    // :1011-1016 -- every face of every marked cell
    std::vector<char> markedFace(static_cast<std::size_t>(m.nFaces()), 0);
    for (std::size_t celli = 0; celli < markedCell.size(); ++celli)
    {
        if (!markedCell[celli]) continue;
        for (const label facei : cells[celli])
        {
            markedFace[static_cast<std::size_t>(facei)] = 1;
        }
    }

    // :1021-1035 -- internal faces mark BOTH sides, boundary faces only the owner. The reads above
    // all finished before these writes, so growing markedCell here cannot feed back into this pass.
    const label nInternal = m.nInternalFaces();
    for (label facei = 0; facei < nInternal; ++facei)
    {
        if (!markedFace[static_cast<std::size_t>(facei)]) continue;
        markedCell[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)])] = 1;
        markedCell[static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(facei)])] = 1;
    }
    for (label facei = nInternal; facei < m.nFaces(); ++facei)
    {
        if (!markedFace[static_cast<std::size_t>(facei)]) continue;
        markedCell[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)])] = 1;
    }
}


std::vector<char> initProtectedCells(
    const std::vector<label>&              cellLevel,
    const std::vector<label>&              pointLevel,
    const std::vector<std::vector<label>>& pointCells,
    const std::vector<std::vector<label>>& cells,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches)
{
    refuseCoupled(patches, "the protected-cell scan", "dynamicRefineFvMesh.C:1169 and :1201");

    const std::size_t nCells = static_cast<std::size_t>(m.nCells());
    std::vector<char> protectedCell(nCells, 0);                 // :1110

    // PASS a, :1131-1150. Note the shape: the `!protected` guard is OUTER, and the increment runs
    // BEFORE the `> 8` -- so the counter reaches 9. checkEightAnchorPoints below counts the same
    // thing with the guard inside and the test before the increment, and the two are NOT the same
    // function.
    {
        std::vector<label> nAnchors(nCells, 0);
        for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
        {
            for (const label celli : pointCells[pointi])
            {
                const std::size_t c = static_cast<std::size_t>(celli);
                if (protectedCell[c]) continue;
                if (pointLevel[pointi] <= cellLevel[c])
                {
                    ++nAnchors[c];
                    if (nAnchors[c] > 8)
                    {
                        protectedCell[c] = 1;
                    }
                }
            }
        }
    }

    // PASS b, :1158-1217
    {
        const label nInternal = m.nInternalFaces();
        const label nFaces = m.nFaces();
        // :1159-1168 -- sized nFaces, the boundary half seeded with the OWNER's level. The swap at
        // :1169 would overwrite the coupled ones; there are none here (refused above).
        std::vector<label> neiLevel(static_cast<std::size_t>(nFaces), 0);
        for (label facei = 0; facei < nInternal; ++facei)
        {
            neiLevel[static_cast<std::size_t>(facei)] =
                cellLevel[static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(facei)])];
        }
        for (label facei = nInternal; facei < nFaces; ++facei)
        {
            neiLevel[static_cast<std::size_t>(facei)] =
                cellLevel[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)])];
        }

        std::vector<char> protectedFace(static_cast<std::size_t>(nFaces), 0);
        for (label facei = 0; facei < nFaces; ++facei)
        {
            const label ownLevel =
                cellLevel[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)])];
            // :1176-1180 -- a LOCAL max, not hexRef8::faceLevel()
            const label faceLevel = std::max(ownLevel, neiLevel[static_cast<std::size_t>(facei)]);
            label nAnchors = 0;
            const label n = m.faceSize(facei);
            for (label k = 0; k < n; ++k)
            {
                const label pointi = m.faceVert(facei, k);
                if (pointLevel[static_cast<std::size_t>(pointi)] <= faceLevel)
                {
                    ++nAnchors;
                    if (nAnchors > 4)
                    {
                        protectedFace[static_cast<std::size_t>(facei)] = 1;
                        break;
                    }
                }
            }
        }

        // :1203-1217 -- an internal protected face protects BOTH its cells
        for (label facei = 0; facei < nInternal; ++facei)
        {
            if (!protectedFace[static_cast<std::size_t>(facei)]) continue;
            protectedCell[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)])] = 1;
            protectedCell[static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(facei)])] = 1;
        }
        for (label facei = nInternal; facei < nFaces; ++facei)
        {
            if (!protectedFace[static_cast<std::size_t>(facei)]) continue;
            protectedCell[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)])] = 1;
        }

        // PASS c, :1220-1239 -- pure topology, no levels. A prism trips BOTH branches: five faces,
        // two of them triangles.
        for (std::size_t celli = 0; celli < cells.size(); ++celli)
        {
            if (cells[celli].size() < 6)
            {
                protectedCell[celli] = 1;
            }
            else
            {
                for (const label cfacei : cells[celli])
                {
                    if (m.faceSize(cfacei) < 4)
                    {
                        protectedCell[celli] = 1;
                        break;
                    }
                }
            }
        }

        // PASS d, :1242
        checkEightAnchorPoints(cellLevel, pointLevel, pointCells, m.nCells(), protectedCell);
    }

    // :1245-1248 -- SIZE ZERO is the sentinel for "nothing is protected", not an array of falses
    for (const char c : protectedCell)
    {
        if (c) return protectedCell;
    }
    return {};
}


void checkEightAnchorPoints(
    const std::vector<label>&              cellLevel,
    const std::vector<label>&              pointLevel,
    const std::vector<std::vector<label>>& pointCells,
    label                                  nCells,
    std::vector<char>&                     protectedCell)
{
    if (protectedCell.size() != static_cast<std::size_t>(nCells))
    {
        throw std::runtime_error(
            std::string(WHO) + "checkEightAnchorPoints was handed a protected-cell marker of "
            + std::to_string(protectedCell.size()) + " where the mesh has " + std::to_string(nCells)
            + " cells. OpenFOAM's is setSize(nCells()) before the scan (dynamicRefineFvMesh.C:1110).");
    }
    std::vector<label> nAnchorPoints(static_cast<std::size_t>(nCells), 0);
    for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
    {
        for (const label celli : pointCells[pointi])
        {
            const std::size_t c = static_cast<std::size_t>(celli);
            if (pointLevel[pointi] <= cellLevel[c])
            {
                // :1058-1061 -- the NINTH anchor is what protects the cell, because this runs BEFORE
                // the increment below and the increment is then suppressed
                if (nAnchorPoints[c] == 8)
                {
                    protectedCell[c] = 1;
                }
                if (!protectedCell[c])
                {
                    ++nAnchorPoints[c];
                }
            }
        }
    }
    // :1072-1078 -- an INDEX loop over every cell, not over the marked ones: this is what catches the
    // cells with FEWER than eight anchors, which is every cell that is not a hex
    for (std::size_t c = 0; c < protectedCell.size(); ++c)
    {
        if (nAnchorPoints[c] != 8)
        {
            protectedCell[c] = 1;
        }
    }
}


std::vector<char> calculateProtectedCells(
    const std::vector<char>&    protectedCell,
    const std::vector<label>&   cellLevel,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches)
{
    // :57-61 -- an empty input means nothing is protected, and the OUTPUT is empty too, not a cleared
    // array of the mesh's size. Every test() on OpenFOAM's cleared bitSet reads false.
    bool any = false;
    for (const char c : protectedCell) any = any || (c != 0);
    if (!any)
    {
        return {};
    }
    refuseCoupled(patches, "the protected-cell cascade", "dynamicRefineFvMesh.C:74 and :125");

    std::vector<char> unrefineableCell = protectedCell;     // :65
    const label nInternal = m.nInternalFaces();
    const label nFaces = m.nFaces();

    while (true)
    {
        // :82-83 -- a zeroed nFaces-long marker each pass
        std::vector<char> seedFace(static_cast<std::size_t>(nFaces), 0);

        // :85-107 -- seed only where the NON-protected side is FINER than the protected one. Both
        // tests are strict, and each compares the other side's level against the protected side's.
        for (label facei = 0; facei < nInternal; ++facei)
        {
            const std::size_t own = static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)]);
            const std::size_t nei = static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(facei)]);
            if ((unrefineableCell[own] && cellLevel[nei] > cellLevel[own])
             || (unrefineableCell[nei] && cellLevel[own] > cellLevel[nei]))
            {
                seedFace[static_cast<std::size_t>(facei)] = 1;
            }
        }
        // :108-123 -- the boundary half needs the swapped neighbour level, which on an UNCOUPLED
        // patch is the owner's own, so the strict test is false and nothing is seeded. A coupled
        // patch was refused above.

        // :131-156
        bool hasExtended = false;
        auto setCell = [&](std::size_t c)
        {
            if (!unrefineableCell[c])
            {
                unrefineableCell[c] = 1;
                hasExtended = true;
            }
        };
        for (label facei = 0; facei < nInternal; ++facei)
        {
            if (!seedFace[static_cast<std::size_t>(facei)]) continue;
            setCell(static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)]));
            setCell(static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(facei)]));
        }
        for (label facei = nInternal; facei < nFaces; ++facei)
        {
            if (!seedFace[static_cast<std::size_t>(facei)]) continue;
            setCell(static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)]));
        }
        if (!hasExtended)
        {
            break;
        }
    }
    return unrefineableCell;
}


}   // namespace dynamicRefine
}   // namespace brae
