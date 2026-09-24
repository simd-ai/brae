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

    // :844-845. An empty protected list means nothing is protected: OpenFOAM's bitSet reads out of
    // range as false, and calculateProtectedCells clears its output when protectedCell_ is empty.
    auto unrefineable = [&](std::size_t c)
    {
        return c < protectedCell.size() && protectedCell[c] != 0;
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


}   // namespace dynamicRefine
}   // namespace brae
