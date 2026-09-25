#include "dynamic_refine_fv_mesh_cpp.cuh"

#include "foam_token_reader.cuh"
#include <algorithm>
#include <filesystem>
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


label RefinementHistory::parentIndex(label celli) const
{
    const label index = visibleCells[static_cast<std::size_t>(celli)];
    if (index < 0)
    {
        throw std::runtime_error(
            std::string(WHO) + "cell " + std::to_string(celli) + " is not visible, so it has no "
            "parent. OpenFOAM fatals here too (refinementHistory.H:308); the caller must test "
            "visibleCells != -1 first, as getSplitPoints does at hexRef8.C:5228.");
    }
    return parent[static_cast<std::size_t>(index)];
}


namespace {

// A labelList in one of OpenFOAM's three written forms: `N ( a b ... )`, `N { v }` (uniform) and the
// empty `0 ( )`. A pristine history writes `0()` for the splits and `N{-1}` for the visible cells, so
// a reader that only knows the bracket form cannot open a never-refined mesh.
std::vector<label> readLabelList(TokenStream& ts)
{
    const label n = ts.nextLabel();
    const std::string open = ts.next();
    std::vector<label> out;
    if (open == "{")
    {
        const label v = ts.nextLabel();
        ts.expect("}");
        out.assign(static_cast<std::size_t>(n), v);
        return out;
    }
    if (open != "(")
    {
        throw std::runtime_error(
            std::string(WHO) + "a label list of " + std::to_string(n) + " entries is followed by `"
            + open + "`, which is neither `(` nor `{`.");
    }
    out.resize(static_cast<std::size_t>(n));
    for (label& x : out) x = ts.nextLabel();
    ts.expect(")");
    return out;
}

}   // namespace


RefinementHistory readRefinementHistory(
    const std::string& path,
    label              nCells)
{
    RefinementHistory h;
    if (!std::filesystem::exists(path))
    {
        // hexRef8.C:1953-1965 -- the object is NO_READ and defaults to every cell its own top-level
        // entry, so a missing file is a mesh that has never been refined, not an error. active() is
        // TRUE in that state and getSplitPoints returns empty.
        h.parent.assign(static_cast<std::size_t>(nCells), label(-1));
        h.visibleCells.resize(static_cast<std::size_t>(nCells));
        for (label c = 0; c < nCells; ++c)
        {
            h.visibleCells[static_cast<std::size_t>(c)] = c;
        }
        h.active = (nCells > 0);
        return h;
    }

    TokenStream ts(path);
    // refinementHistory.C:1741-1744 -- splitCells_ then visibleCells_, the `//` markers being
    // comments the tokenizer drops
    const label nSplit = ts.nextLabel();
    ts.expect("(");
    h.parent.resize(static_cast<std::size_t>(nSplit));
    for (label i = 0; i < nSplit; ++i)
    {
        h.parent[static_cast<std::size_t>(i)] = ts.nextLabel();
        // the eight added cells, parsed to advance the stream and discarded: unit 4 never reads them
        (void)readLabelList(ts);
    }
    ts.expect(")");
    h.visibleCells = readLabelList(ts);
    h.active = !h.visibleCells.empty();

    if (h.active && static_cast<label>(h.visibleCells.size()) != nCells)
    {
        throw std::runtime_error(
            std::string(WHO) + path + " holds " + std::to_string(h.visibleCells.size())
            + " visible cells where the mesh has " + std::to_string(nCells)
            + ". OpenFOAM fatals on the same mismatch (hexRef8.C:1980-1988).");
    }
    return h;
}


std::vector<label> getSplitPoints(
    const RefinementHistory&               history,
    const std::vector<label>&              cellLevel,
    const std::vector<std::vector<label>>& pointCells,
    const std::vector<std::vector<label>>& cellPoints,
    const PrimitiveMesh&                   m)
{
    if (!history.active)
    {
        throw std::runtime_error(
            std::string(WHO) + "getSplitPoints needs a refinement history, and this one is not "
            "active. OpenFOAM aborts with \"Only call if constructed with history capability\" "
            "(hexRef8.C:5194-5199).");
    }
    const std::size_t nPoints = static_cast<std::size_t>(m.nPoints());

    // :5205-5206. -1 undetermined, -2 certainly not a split point, >= 0 the master cell.
    std::vector<label> splitMaster(nPoints, label(-1));
    std::vector<label> splitMasterLevel(nPoints, 0);

    // :5211-5219 -- a split point has EXACTLY eight cells
    for (std::size_t pointi = 0; pointi < nPoints; ++pointi)
    {
        if (pointCells[pointi].size() != 8)
        {
            splitMaster[pointi] = -2;
        }
    }

    // :5224-5275. The `visibleCells != -1` test short-circuits parentIndex, which fatals otherwise.
    for (std::size_t celli = 0; celli < history.visibleCells.size(); ++celli)
    {
        const bool refined = (history.visibleCells[celli] != -1)
                          && (history.parentIndex(static_cast<label>(celli)) >= 0);
        if (refined)
        {
            const label parentIndex = history.parentIndex(static_cast<label>(celli));
            for (const label pointi : cellPoints[celli])
            {
                const std::size_t p = static_cast<std::size_t>(pointi);
                const label masterCelli = splitMaster[p];
                if (masterCelli == -1)
                {
                    // first visit: store the parent AND the level, so a point shared by two
                    // refinement patterns at different levels is caught below
                    splitMaster[p] = parentIndex;
                    splitMasterLevel[p] = cellLevel[celli] - 1;
                }
                else if (masterCelli == -2)
                {
                }
                else if (masterCelli != parentIndex || splitMasterLevel[p] != cellLevel[celli] - 1)
                {
                    splitMaster[p] = -2;
                }
            }
        }
        else
        {
            for (const label pointi : cellPoints[celli])
            {
                splitMaster[static_cast<std::size_t>(pointi)] = -2;
            }
        }
    }

    // :5278-5291 -- nothing on a boundary face can be unsplit
    for (label facei = m.nInternalFaces(); facei < m.nFaces(); ++facei)
    {
        const label n = m.faceSize(facei);
        for (label k = 0; k < n; ++k)
        {
            splitMaster[static_cast<std::size_t>(m.faceVert(facei, k))] = -2;
        }
    }

    // :5298-5315 -- ascending
    std::vector<label> splitPoints;
    for (std::size_t pointi = 0; pointi < nPoints; ++pointi)
    {
        if (splitMaster[pointi] >= 0) splitPoints.push_back(static_cast<label>(pointi));
    }
    return splitPoints;
}


std::vector<label> consistentUnrefinement(
    const std::vector<label>&              pointsToUnrefine,
    bool                                   maxSet,
    const std::vector<label>&              cellLevel,
    const std::vector<std::vector<label>>& pointCells,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches)
{
    if (maxSet)
    {
        // :5395-5400 -- OpenFOAM's own "maxSet not implemented yet."
        throw std::runtime_error(
            std::string(WHO) + "consistentUnrefinement was asked for maxSet, which OpenFOAM itself "
            "aborts on (hexRef8.C:5395-5400, \"maxSet not implemented yet\"). This closure can only "
            "remove points.");
    }
    refuseCoupled(patches, "the unrefinement closure", "hexRef8.C:5503");

    const std::size_t nPoints = static_cast<std::size_t>(m.nPoints());
    const std::size_t nCells = static_cast<std::size_t>(m.nCells());
    std::vector<char> unrefinePoint(nPoints, 0);
    for (const label p : pointsToUnrefine)
    {
        unrefinePoint[static_cast<std::size_t>(p)] = 1;
    }

    while (true)
    {
        // :5415-5425 -- rebuilt from the points every pass, so it can only shrink across passes
        std::vector<char> unrefineCell(nCells, 0);
        for (std::size_t pointi = 0; pointi < nPoints; ++pointi)
        {
            if (!unrefinePoint[pointi]) continue;
            for (const label celli : pointCells[pointi])
            {
                unrefineCell[static_cast<std::size_t>(celli)] = 1;
            }
        }

        // :5435-5489. The levels AFTER unrefinement, and the test is the opposite direction from the
        // refinement closure's.
        label nChanged = 0;
        const label nInternal = m.nInternalFaces();
        for (label facei = 0; facei < nInternal; ++facei)
        {
            const std::size_t own = static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(facei)]);
            const std::size_t nei = static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(facei)]);
            const label ownLevel = cellLevel[own] - unrefineCell[own];
            const label neiLevel = cellLevel[nei] - unrefineCell[nei];
            if (ownLevel < neiLevel - 1)
            {
                if (!unrefineCell[own])
                {
                    throw std::runtime_error(
                        std::string(WHO) + "the unrefinement closure met a 2:1 conflict it cannot "
                        "resolve: cell " + std::to_string(own) + " is finer than its neighbour by "
                        "more than one level and is not marked. OpenFOAM aborts here too "
                        "(hexRef8.C:5461-5465).");
                }
                unrefineCell[own] = 0;
                ++nChanged;
            }
            else if (neiLevel < ownLevel - 1)
            {
                if (!unrefineCell[nei])
                {
                    throw std::runtime_error(
                        std::string(WHO) + "the unrefinement closure met a 2:1 conflict it cannot "
                        "resolve at cell " + std::to_string(nei) + " (hexRef8.C:5479-5483).");
                }
                unrefineCell[nei] = 0;
                ++nChanged;
            }
        }
        // :5492-5539 -- the boundary half needs the swapped level; without a coupled patch the swap
        // leaves the owner's own value and both tests are false.

        if (nChanged == 0)
        {
            break;
        }

        // :5561-5576 -- knock out any point one of whose cells can no longer be unrefined
        for (std::size_t pointi = 0; pointi < nPoints; ++pointi)
        {
            if (!unrefinePoint[pointi]) continue;
            for (const label celli : pointCells[pointi])
            {
                if (!unrefineCell[static_cast<std::size_t>(celli)])
                {
                    unrefinePoint[pointi] = 0;
                    break;
                }
            }
        }
    }

    std::vector<label> out;
    for (std::size_t pointi = 0; pointi < nPoints; ++pointi)
    {
        if (unrefinePoint[pointi]) out.push_back(static_cast<label>(pointi));
    }
    return out;
}


std::vector<label> selectUnrefinePoints(
    scalar                                 unrefineLevel,
    const std::vector<char>&               markedCell,
    const std::vector<scalar>&             pFld,
    const std::vector<label>&              splitPoints,
    const std::vector<char>&               protectedCell,
    const std::vector<label>&              cellLevel,
    const std::vector<std::vector<label>>& pointCells,
    const PrimitiveMesh&                   m,
    const std::vector<FvPatch>&            patches)
{
    const std::size_t nPoints = static_cast<std::size_t>(m.nPoints());

    // :926-956. Guarded by protectedCell_.size(), which is ZERO when nothing is protected -- the
    // sentinel initProtectedCells returns, not an array of falses.
    std::vector<char> protectedPoint(nPoints, 0);
    if (!protectedCell.empty())
    {
        for (std::size_t pointi = 0; pointi < nPoints; ++pointi)
        {
            for (const label celli : pointCells[pointi])
            {
                if (protectedCell[static_cast<std::size_t>(celli)])
                {
                    protectedPoint[pointi] = 1;
                    break;
                }
            }
        }
    }

    // :961-982. The field test is STRICT, and `markedCell` vetoes: a point any of whose cells is a
    // refinement candidate (after the buffer layers) is not unsplit.
    std::vector<label> newSplitPoints;
    for (const label pointi : splitPoints)
    {
        const std::size_t p = static_cast<std::size_t>(pointi);
        if (protectedPoint[p] || !(pFld[p] < unrefineLevel)) continue;
        bool hasMarked = false;
        for (const label celli : pointCells[p])
        {
            if (markedCell[static_cast<std::size_t>(celli)])
            {
                hasMarked = true;
                break;
            }
        }
        if (!hasMarked)
        {
            newSplitPoints.push_back(pointi);
        }
    }

    // :988-995
    return consistentUnrefinement(newSplitPoints, /*maxSet=*/false, cellLevel, pointCells, m, patches);
}


std::vector<scalar> correctOldVolumes(
    const MapPolyMesh&         mpm,
    const std::vector<scalar>& mappedV0,
    const std::vector<scalar>& V)
{
    const std::size_t nNew = mpm.cellMap.size();
    if (mappedV0.size() != nNew || V.size() != nNew)
    {
        throw std::runtime_error(
            std::string(WHO) + "the old-time volume correction was handed " + std::to_string(mappedV0.size())
            + " mapped old volumes and " + std::to_string(V.size()) + " new ones where the map spans "
            + std::to_string(nNew) + " cells.");
    }
    // The reverse map is sized for all old AND ADDED cells (mapPolyMesh.H:535), so it is longer than
    // the old mesh on a refinement -- 24,815 entries for 24,185 old cells on the measured step. Only
    // the old-cell range is indexed here, which is what OpenFOAM does at :217.
    if (static_cast<label>(mpm.reverseCellMap.size()) < mpm.nOldCells)
    {
        throw std::runtime_error(
            std::string(WHO) + "the correction needs the reverse cell map -- it is what tells a SPLIT "
            "from a renumber (dynamicRefineFvMesh.C:217) -- and the map carries only "
            + std::to_string(mpm.reverseCellMap.size()) + " entries for " + std::to_string(mpm.nOldCells)
            + " old cells.");
    }

    // :212-222. Count only the new cells whose old cell still exists, so a renumber does not look like
    // a split.
    std::vector<label> nSubCells(static_cast<std::size_t>(mpm.nOldCells), 0);
    for (std::size_t celli = 0; celli < nNew; ++celli)
    {
        const label oldCelli = mpm.cellMap[celli];
        if (oldCelli >= 0 && mpm.reverseCellMap[static_cast<std::size_t>(oldCelli)] >= 0)
        {
            ++nSubCells[static_cast<std::size_t>(oldCelli)];
        }
    }

    // :225 -- start from the MAPPED old volumes, not from the raw ones
    std::vector<scalar> corrected = mappedV0;

    // :229-238 -- a split cell takes its own new volume
    for (std::size_t celli = 0; celli < nNew; ++celli)
    {
        const label oldCelli = mpm.cellMap[celli];
        if (oldCelli >= 0 && nSubCells[static_cast<std::size_t>(oldCelli)] == 8)
        {
            corrected[celli] = V[celli];
        }
    }

    // :240-251 -- and so does a merged one. OpenFOAM does NOT sum the old volumes here, and its own
    // comment asks whether it should.
    for (const auto& s : mpm.cellsFromCells)
    {
        corrected[static_cast<std::size_t>(s.first)] = V[static_cast<std::size_t>(s.first)];
    }
    return corrected;
}


namespace {

// `tmpValue/counter` as OpenFOAM writes it: a division per component, not a multiply by the reciprocal
inline scalar divideByCount(scalar v, scalar n)
{
    return v/n;
}

inline vector divideByCount(const vector& v, scalar n)
{
    return vector{v.x/n, v.y/n, v.z/n};
}

// which patch a boundary face belongs to, and its index within it -- OpenFOAM's
// boundaryMesh().whichPatch(facei) followed by facei - patch.start()
bool locateBoundaryFace(
    const FluxMeshView& m,
    label               facei,
    std::size_t&        patchi,
    std::size_t&        i)
{
    for (std::size_t p = 0; p < m.patchStart.size(); ++p)
    {
        const label s = m.patchStart[p];
        const label n = m.patchSize[p];
        if (facei >= s && facei < s + n)
        {
            patchi = p;
            i = static_cast<std::size_t>(facei - s);
            return true;
        }
    }
    return false;
}

}   // namespace


std::vector<char> masterFaces(
    const MapPolyMesh& mpm,
    label              nNewFaces)
{
    std::vector<char> master(static_cast<std::size_t>(nNewFaces), 0);
    for (std::size_t facei = 0; facei < mpm.faceMap.size(); ++facei)
    {
        const label oldFacei = mpm.faceMap[facei];
        if (oldFacei < 0) continue;
        const label masterFacei = mpm.reverseFaceMap[static_cast<std::size_t>(oldFacei)];
        if (masterFacei < 0)
        {
            // :280-284 -- OpenFOAM aborts here: refinement must not remove faces
            throw std::runtime_error(
                std::string(WHO) + "face " + std::to_string(facei) + " maps from old face "
                + std::to_string(oldFacei) + ", which the reverse map says was REMOVED. OpenFOAM "
                "aborts on this (dynamicRefineFvMesh.C:280-284): refinement does not remove faces.");
        }
        if (masterFacei != static_cast<label>(facei))
        {
            master[static_cast<std::size_t>(masterFacei)] = 1;
        }
    }
    return master;
}


label correctFluxes(
    std::vector<scalar>&                    phi,
    std::vector<std::vector<scalar>>&       phiBnd,
    const std::vector<scalar>&              phiU,
    const std::vector<std::vector<scalar>>& phiUBnd,
    const MapPolyMesh&                      mpm,
    const std::vector<char>&                masterFace,
    const FluxMeshView&                     m)
{
    label nWritten = 0;
    std::vector<char> written(mpm.faceMap.size(), 0);
    auto note = [&](label facei)
    {
        if (!written[static_cast<std::size_t>(facei)])
        {
            written[static_cast<std::size_t>(facei)] = 1;
            ++nWritten;
        }
    };

    // :355-369 -- new INTERNAL faces
    for (label facei = 0; facei < m.nInternalFaces; ++facei)
    {
        const label oldFacei = mpm.faceMap[static_cast<std::size_t>(facei)];
        if (oldFacei == -1
         || mpm.reverseFaceMap[static_cast<std::size_t>(oldFacei)] != facei)
        {
            phi[static_cast<std::size_t>(facei)] = phiU[static_cast<std::size_t>(facei)];
            note(facei);
        }
    }

    // :372-399 -- and new BOUNDARY faces, walked patch by patch with a running face index
    for (std::size_t patchi = 0; patchi < phiBnd.size(); ++patchi)
    {
        label facei = m.patchStart[patchi];
        for (std::size_t i = 0; i < phiBnd[patchi].size(); ++i, ++facei)
        {
            const label oldFacei = mpm.faceMap[static_cast<std::size_t>(facei)];
            if (oldFacei == -1
             || mpm.reverseFaceMap[static_cast<std::size_t>(oldFacei)] != facei)
            {
                phiBnd[patchi][i] = phiUBnd[patchi][i];
                note(facei);
            }
        }
    }

    // :402-420 -- then every MASTER face, internal or boundary
    for (std::size_t facei = 0; facei < masterFace.size(); ++facei)
    {
        if (!masterFace[facei]) continue;
        const label f = static_cast<label>(facei);
        if (f < m.nInternalFaces)
        {
            phi[facei] = phiU[facei];
            note(f);
        }
        else
        {
            std::size_t patchi = 0, i = 0;
            if (!locateBoundaryFace(m, f, patchi, i))
            {
                throw std::runtime_error(
                    std::string(WHO) + "master face " + std::to_string(f) + " is neither internal nor "
                    "on any patch of the new mesh.");
            }
            phiBnd[patchi][i] = phiUBnd[patchi][i];
            note(f);
        }
    }
    return nWritten;
}


template <typename T>
void mapNewInternalFacesFlat(
    std::vector<T>&                    sFld,
    const std::vector<std::vector<T>>& sBnd,
    const MapPolyMesh&                 mpm,
    const FluxMeshView&                m)
{
    // Templates:42-53 -- one flat field over every face, internal then boundary in patch order
    std::vector<T> flat(mpm.faceMap.size(), T{});
    for (label facei = 0; facei < m.nInternalFaces; ++facei)
    {
        flat[static_cast<std::size_t>(facei)] = sFld[static_cast<std::size_t>(facei)];
    }
    for (std::size_t patchi = 0; patchi < sBnd.size(); ++patchi)
    {
        label facei = m.patchStart[patchi];
        for (const T& v : sBnd[patchi])
        {
            flat[static_cast<std::size_t>(facei++)] = v;
        }
    }

    // Templates:59-97 -- the hull of already-mapped faces, in the cell's OWN face order
    for (label facei = 0; facei < m.nInternalFaces; ++facei)
    {
        if (mpm.faceMap[static_cast<std::size_t>(facei)] != -1) continue;
        T tmpValue{};
        label counter = 0;
        for (const label side : {m.owner[static_cast<std::size_t>(facei)],
                                 m.neighbour[static_cast<std::size_t>(facei)]})
        {
            for (const label f : m.cells[static_cast<std::size_t>(side)])
            {
                if (mpm.faceMap[static_cast<std::size_t>(f)] != -1)
                {
                    tmpValue = tmpValue + flat[static_cast<std::size_t>(f)];
                    ++counter;
                }
            }
        }
        // :92-95 -- a face with NO mapped hull face is left alone, which is not the same as zeroed
        if (counter > 0)
        {
            // Templates:94 is `tmpValue/counter` -- a DIVISION. Multiplying by the reciprocal instead
            // rounds differently: measured at 1.08e-19 on 398 of 76,039 faces on the refine arm, which
            // was the whole of its residue.
            sFld[static_cast<std::size_t>(facei)] =
                divideByCount(tmpValue, static_cast<scalar>(counter));
        }
    }
}

template void mapNewInternalFacesFlat<scalar>(
    std::vector<scalar>&, const std::vector<std::vector<scalar>>&, const MapPolyMesh&,
    const FluxMeshView&);
template void mapNewInternalFacesFlat<vector>(
    std::vector<vector>&, const std::vector<std::vector<vector>>&, const MapPolyMesh&,
    const FluxMeshView&);


void mapNewInternalFacesOriented(
    std::vector<scalar>&                    phi,
    std::vector<std::vector<scalar>>&       phiBnd,
    const std::vector<vector>&              Sf,
    const std::vector<std::vector<vector>>& SfBnd,
    const std::vector<scalar>&              magSf,
    const std::vector<std::vector<scalar>>& magSfBnd,
    const MapPolyMesh&                      mpm,
    const FluxMeshView&                     m)
{
    // Templates:169 -- to intensive and non-oriented: fFld = sFld*Sf/sqr(magSf)
    std::vector<vector> fFld(static_cast<std::size_t>(m.nInternalFaces), vector{0, 0, 0});
    for (std::size_t i = 0; i < fFld.size(); ++i)
    {
        // Templates:169 is `sFld*Sf/sqr(magSf)`: the field expression multiplies FIRST and divides
        // second. Dividing first and then scaling rounds differently -- measured at 1.08e-19 on 398
        // of 76,039 faces, which is the whole of the refine arm's residue.
        const scalar d = magSf[i]*magSf[i];
        fFld[i] = (phi[i]*Sf[i])/d;
    }
    std::vector<std::vector<vector>> fBnd(SfBnd.size());
    for (std::size_t p = 0; p < SfBnd.size(); ++p)
    {
        fBnd[p].assign(SfBnd[p].size(), vector{0, 0, 0});
        for (std::size_t i = 0; i < SfBnd[p].size(); ++i)
        {
            const scalar d = magSfBnd[p][i]*magSfBnd[p][i];
            fBnd[p][i] = (phiBnd[p][i]*SfBnd[p][i])/d;
        }
    }

    // :172 -- map the intensive field
    mapNewInternalFacesFlat<vector>(fFld, fBnd, mpm, m);

    // :175 -- and back, `sFld = (fFld & Sf)`. A WHOLE-FIELD assignment: every face is rewritten, not
    // only the injected ones, and the round trip is not the identity in floating point.
    for (std::size_t i = 0; i < fFld.size(); ++i)
    {
        phi[i] = dot(fFld[i], Sf[i]);
    }
    for (std::size_t p = 0; p < fBnd.size(); ++p)
    {
        for (std::size_t i = 0; i < fBnd[p].size(); ++i)
        {
            phiBnd[p][i] = dot(fBnd[p][i], SfBnd[p][i]);
        }
    }
}


label correctFluxesUnrefine(
    std::vector<scalar>&                        phi,
    std::vector<std::vector<scalar>>&           phiBnd,
    const std::vector<scalar>&                  phiU,
    const std::vector<std::vector<scalar>>&     phiUBnd,
    const std::vector<std::pair<label, label>>& faceToSplitPoint,
    const MapPolyMesh&                          mpm,
    const FluxMeshView&                         m)
{
    label nWritten = 0;
    std::vector<char> written(mpm.faceMap.size(), 0);
    // :659-687. Each entry writes the same value to the same face, so the iteration order of
    // OpenFOAM's Map does not reach the answer.
    for (const auto& entry : faceToSplitPoint)
    {
        const label oldFacei = entry.first;
        const label oldPointi = entry.second;
        if (static_cast<std::size_t>(oldPointi) >= mpm.reversePointMap.size()) continue;
        if (mpm.reversePointMap[static_cast<std::size_t>(oldPointi)] >= 0) continue;
        const label facei = mpm.reverseFaceMap[static_cast<std::size_t>(oldFacei)];
        if (facei < 0) continue;
        if (written[static_cast<std::size_t>(facei)]) continue;
        written[static_cast<std::size_t>(facei)] = 1;
        ++nWritten;
        if (facei < m.nInternalFaces)
        {
            phi[static_cast<std::size_t>(facei)] = phiU[static_cast<std::size_t>(facei)];
        }
        else
        {
            std::size_t patchi = 0, i = 0;
            if (!locateBoundaryFace(m, facei, patchi, i))
            {
                throw std::runtime_error(
                    std::string(WHO) + "the unrefinement flux correction met face "
                    + std::to_string(facei) + ", which is neither internal nor on any patch.");
            }
            phiBnd[patchi][i] = phiUBnd[patchi][i];
        }
    }
    return nWritten;
}



// ----------------------------------------------------------------------------------------------
// The dynamicRefineFvMeshCoeffs reader. See the header for where each entry lives and which are
// mandatory; the refusals below are OpenFOAM's own three FatalErrors, kept word for word in substance.

RefineControls readRefineControls(
    const FoamDict& dynamicMeshDict)
{
    const char* WHO = "brae dynamicRefineFvMesh: ";
    // dynamicRefineFvMesh.C:181 and :1292 -- the sub-dictionary when there is one, else this dict
    const FoamDict& d = *dynamicMeshDict.optionalSubDict("dynamicRefineFvMeshCoeffs");

    auto mustLabel = [&](const char* key) -> label
    {
        if (!d.found(key))
            throw std::runtime_error(std::string(WHO) + "constant/dynamicMeshDict has no `" + key
                                     + "`. OpenFOAM reads it with get<label>, which stops when it is "
                                       "absent; defaulting it would run a different case.");
        return static_cast<label>(d.intOr(key, 0));
    };
    auto mustScalar = [&](const char* key) -> scalar
    {
        if (!d.found(key))
            throw std::runtime_error(std::string(WHO) + "constant/dynamicMeshDict has no `" + key
                                     + "`. OpenFOAM reads it with get<scalar>, which stops when it is "
                                       "absent; defaulting it would run a different case.");
        return d.scalarOr(key, scalar(0));
    };

    RefineControls c;
    c.refineInterval = mustLabel("refineInterval");
    // dynamicRefineFvMesh.C:1305-1311
    if (c.refineInterval < 0)
        throw std::runtime_error(std::string(WHO) + "illegal refineInterval "
                                 + std::to_string(c.refineInterval)
                                 + ". The refineInterval setting in the dynamicMeshDict should be >= 1.");

    c.maxCells = mustLabel("maxCells");
    // :1321-1328
    if (c.maxCells <= 0)
        throw std::runtime_error(std::string(WHO) + "illegal maximum number of cells "
                                 + std::to_string(c.maxCells)
                                 + ". The maxCells setting in the dynamicMeshDict should be > 0.");

    c.maxRefinement = mustLabel("maxRefinement");
    // :1330-1338
    if (c.maxRefinement <= 0)
        throw std::runtime_error(std::string(WHO) + "illegal maximum refinement level "
                                 + std::to_string(c.maxRefinement)
                                 + ". The maxRefinement setting in the dynamicMeshDict should be > 0.");

    if (!d.found("field"))
        throw std::runtime_error(std::string(WHO) + "constant/dynamicMeshDict has no `field`. It names "
                                 "the volScalarField the refinement criterion is taken from.");
    c.field = d.wordOr("field", std::string());

    c.lowerRefineLevel = mustScalar("lowerRefineLevel");
    c.upperRefineLevel = mustScalar("upperRefineLevel");
    // :1350-1354 -- the ONLY optional one, and its default is GREAT. RAS/motorBike omits it.
    c.unrefineLevel    = d.scalarOr("unrefineLevel", scalar(1.0e+15));
    c.nBufferLayers    = mustLabel("nBufferLayers");

    // :184-191 -- `List<Pair<word>>`, read at construction and inserted into a HashTable keyed on the
    // flux name. The tokens arrive with the parentheses stripped, so they are consumed in pairs.
    if (!d.found("correctFluxes"))
        throw std::runtime_error(std::string(WHO) + "constant/dynamicMeshDict has no `correctFluxes`. "
                                 "It is read with get<List<Pair<word>>> at construction, and it is what "
                                 "says which surface fields are corrected across a mesh change.");
    {
        const std::vector<std::string> toks = d.wordListOr("correctFluxes", {});
        if (toks.size() % 2 != 0)
            throw std::runtime_error(std::string(WHO) + "`correctFluxes` has " + std::to_string(toks.size())
                                     + " words, which is not a whole number of (flux velocity) pairs.");
        for (std::size_t i = 0; i + 1 < toks.size(); i += 2)
        {
            c.correctFluxes.emplace_back(toks[i], toks[i + 1]);
        }
    }

    // :193 -- readEntry, so mandatory
    if (!d.found("dumpLevel"))
        throw std::runtime_error(std::string(WHO) + "constant/dynamicMeshDict has no `dumpLevel`. "
                                 "OpenFOAM reads it with readEntry, which stops when it is absent.");
    {
        const std::string v = d.wordOr("dumpLevel", "false");
        c.dumpLevel = (v == "true" || v == "yes" || v == "on" || v == "1");
    }
    return c;
}

}   // namespace dynamicRefine
}   // namespace brae
