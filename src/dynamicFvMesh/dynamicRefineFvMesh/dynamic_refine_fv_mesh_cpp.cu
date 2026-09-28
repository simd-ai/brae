#include "dynamic_refine_fv_mesh_cpp.cuh"

#include "foam_token_reader.cuh"
#include "fv_geometry.cuh"
#include "mesh_cell_cells_cpp.cuh"
#include "mesh_edges_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include "remove_faces_cpp.cuh"
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
        // A PATCH WHOSE FIELD OpenFOAM HOLDS NOTHING FOR leaves its faces at the ZERO the array was
        // built with -- see FluxMeshView::patchHoldsNoValues for the measurement. They still count.
        if (patchi < m.patchHoldsNoValues.size() && m.patchHoldsNoValues[patchi]) continue;
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

    // A MISSING ENTRY IS STOPPED IN OpenFOAM'S WORDS, and in OpenFOAM's ORDER: its constructor's
    // readDict() reads correctFluxes and then dumpLevel (dynamicRefineFvMesh.C:184, :193), and update()
    // reads refineInterval at the first step (:1295) -- so a dictionary missing several names the one
    // OpenFOAM names. A bare `dynamicFvMesh dynamicRefineFvMesh;` stops on correctFluxes in both.
    auto notFound = [&](const char* key, const char* how) -> std::runtime_error
    {
        return std::runtime_error(std::string(WHO) + "Entry '" + key + "' not found in dictionary "
                                  "constant/dynamicMeshDict. OpenFOAM reads it with " + how
                                  + ", which stops when it is absent; defaulting it would run a different case.");
    };
    auto mustLabel = [&](const char* key) -> label
    {
        if (!d.found(key)) throw notFound(key, "get<label>");
        return static_cast<label>(d.intOr(key, 0));
    };
    auto mustScalar = [&](const char* key) -> scalar
    {
        if (!d.found(key)) throw notFound(key, "get<scalar>");
        return d.scalarOr(key, scalar(0));
    };

    RefineControls c;

    // :184-191 -- `List<Pair<word>>`, read at construction and inserted into a HashTable keyed on the
    // flux name. The tokens arrive with the parentheses stripped, so they are consumed in pairs.
    if (!d.found("correctFluxes")) throw notFound("correctFluxes", "get<List<Pair<word>>> at construction");
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

    // :193 -- readEntry into a bool, so mandatory, and a token that is not a Switch word stops as OpenFOAM's
    // bool read does (switchOr throws on one) rather than reading as false
    if (!d.found("dumpLevel")) throw notFound("dumpLevel", "readEntry at construction");
    c.dumpLevel = d.switchOr("dumpLevel", false);

    // :1295, at every update()
    c.refineInterval = mustLabel("refineInterval");
    // dynamicRefineFvMesh.C:1305-1311
    if (c.refineInterval < 0)
        throw std::runtime_error(std::string(WHO) + "illegal refineInterval "
                                 + std::to_string(c.refineInterval)
                                 + ". The refineInterval setting in the dynamicMeshDict should be >= 1.");

    // :1320-1355, READ HERE AND NOT WHERE OpenFOAM READS THEM. OpenFOAM reads these at the first step that
    // refines, so a case with `refineInterval 0`, or whose run ends before step refineInterval, never
    // reads them and may omit them; brae reads them at construction and refuses such a case instead. A
    // refusal where OpenFOAM runs, never a substitution -- and no shipped dictionary omits one.
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

    // the volScalarField the refinement criterion is taken from
    if (!d.found("field")) throw notFound("field", "get<word>");
    c.field = d.wordOr("field", std::string());

    c.lowerRefineLevel = mustScalar("lowerRefineLevel");
    c.upperRefineLevel = mustScalar("upperRefineLevel");
    // :1350-1354 -- the ONLY optional one, and its default is GREAT. RAS/motorBike omits it.
    c.unrefineLevel    = d.scalarOr("unrefineLevel", scalar(1.0e+15));
    c.nBufferLayers    = mustLabel("nBufferLayers");
    return c;
}

// ----------------------------------------------------------------------------------------------
// UNIT 7b: the field mapping. See the header.

CellMapping cellMapping(
    const cpu::polyTopoChange::TopoChangeMap& map,
    label                                     nNewCells,
    const std::vector<scalar>&                oldCellVolumes)
{
    CellMapping cm;
    // cellMapper's constructor (:233-249): direct unless SOMETHING was inflated or merged
    cm.direct = map.cellsFromPoints.empty() && map.cellsFromEdges.empty()
             && map.cellsFromFaces.empty() && map.cellsFromCells.empty();
    if (nNewCells == 0) cm.direct = true;

    if (cm.direct)
    {
        // :36-60. The addressing IS cellMap, with each negative rewritten to 0 and recorded as inserted.
        cm.directAddressing.assign(map.cellMap.begin(),
                                   map.cellMap.begin() + static_cast<std::size_t>(nNewCells));
        for (std::size_t i = 0; i < cm.directAddressing.size(); ++i)
        {
            if (cm.directAddressing[i] < 0)
            {
                cm.directAddressing[i] = 0;
                cm.insertedCells.push_back(static_cast<label>(i));
            }
        }
        return cm;
    }

    // :62-230, the interpolative branch.
    cm.addressing.resize(static_cast<std::size_t>(nNewCells));
    cm.weights.resize(static_cast<std::size_t>(nNewCells));
    const auto setAddrWeights = [&](const std::vector<cpu::polyTopoChange::ObjectMap>& maps)
    {
        for (const cpu::polyTopoChange::ObjectMap& m : maps)
        {
            if (m.masterObjects.empty()) continue;
            const std::size_t celli = static_cast<std::size_t>(m.index);
            if (!cm.addressing[celli].empty())
                throw std::runtime_error(
                    "brae cellMapper: cell " + std::to_string(m.index) + " is mapped twice. OpenFOAM "
                    "FatalErrors here too (cellMapper.C:124-131).");
            cm.addressing[celli] = m.masterObjects;
            cm.weights[celli].assign(m.masterObjects.size(),
                                     scalar(1)/static_cast<scalar>(m.masterObjects.size()));
        }
    };
    // the order is OpenFOAM's: points, edges, faces, then cells
    setAddrWeights(map.cellsFromPoints);
    setAddrWeights(map.cellsFromEdges);
    setAddrWeights(map.cellsFromFaces);
    setAddrWeights(map.cellsFromCells);

    // :150-195. VOLUME-WEIGHTED where the map carries the old volumes, and only for cellsFromCells: the
    // uniform weights set above are overwritten in place, and a zero total falls back to uniform.
    if (!oldCellVolumes.empty())
    {
        if (static_cast<label>(oldCellVolumes.size()) != map.nOldCells)
            throw std::runtime_error(
                "brae cellMapper: " + std::to_string(oldCellVolumes.size()) + " old cell volumes for a "
                "map with " + std::to_string(map.nOldCells) + " old cells. OpenFOAM FatalErrors here too "
                "(cellMapper.C:160-170).");
        for (const cpu::polyTopoChange::ObjectMap& m : map.cellsFromCells)
        {
            if (m.masterObjects.empty()) continue;
            std::vector<scalar>& w = cm.weights[static_cast<std::size_t>(m.index)];
            scalar sumV = 0;
            for (std::size_t ci = 0; ci < m.masterObjects.size(); ++ci)
            {
                w[ci] = oldCellVolumes[static_cast<std::size_t>(m.masterObjects[ci])];
                sumV += w[ci];
            }
            if (sumV > scalar(1e-300))      // OF: VSMALL
            {
                for (scalar& wi : w) wi /= sumV;
            }
            else
            {
                const scalar uniform = scalar(1)/static_cast<scalar>(m.masterObjects.size());
                for (scalar& wi : w) wi = uniform;
            }
        }
    }

    // :198-215. Then the cells that came from ONE cell, where nothing has been set yet.
    for (label celli = 0; celli < nNewCells; ++celli)
    {
        const label mapped = map.cellMap[static_cast<std::size_t>(celli)];
        if (mapped >= 0 && cm.addressing[static_cast<std::size_t>(celli)].empty())
        {
            cm.addressing[static_cast<std::size_t>(celli)] = {mapped};
            cm.weights[static_cast<std::size_t>(celli)] = {scalar(1)};
        }
    }
    // :218-260. Whatever is still empty is INSERTED and reads cell 0.
    for (label celli = 0; celli < nNewCells; ++celli)
    {
        if (cm.addressing[static_cast<std::size_t>(celli)].empty())
        {
            cm.addressing[static_cast<std::size_t>(celli)] = {label(0)};
            cm.weights[static_cast<std::size_t>(celli)] = {scalar(1)};
            cm.insertedCells.push_back(celli);
        }
    }
    return cm;
}

namespace {

template<class T>
std::vector<T> mapCellFieldT(
    const std::vector<T>& oldField,
    const CellMapping&    cm,
    const T&              zero)
{
    if (cm.direct)
    {
        // Field<Type>::map(mapF, mapAddressing) (:372-399): a negative entry leaves the value alone, and
        // cellMapper has already rewritten every negative to 0, so there are none here.
        std::vector<T> out(cm.directAddressing.size(), zero);
        if (!oldField.empty())
        {
            for (std::size_t i = 0; i < out.size(); ++i)
            {
                out[i] = oldField[static_cast<std::size_t>(cm.directAddressing[i])];
            }
        }
        return out;
    }
    // :456-470. From ZERO, in the addressing order.
    std::vector<T> out(cm.addressing.size(), zero);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        T v = zero;
        for (std::size_t j = 0; j < cm.addressing[i].size(); ++j)
        {
            v = v + cm.weights[i][j]*oldField[static_cast<std::size_t>(cm.addressing[i][j])];
        }
        out[i] = v;
    }
    return out;
}

}   // namespace

std::vector<scalar> mapCellField(
    const std::vector<scalar>& oldField,
    const CellMapping&         cm)
{
    return mapCellFieldT<scalar>(oldField, cm, scalar(0));
}

std::vector<vector> mapCellField(
    const std::vector<vector>& oldField,
    const CellMapping&         cm)
{
    return mapCellFieldT<vector>(oldField, cm, vector{0, 0, 0});
}

std::vector<scalar> mapOldVolumes(
    const std::vector<scalar>&                V0,
    const cpu::polyTopoChange::TopoChangeMap& map,
    label                                     nNewCells)
{
    // fvMesh.C:851-891. The gather is `cellMap[i] > -1`, so an inserted cell gets 0 and not cell 0's
    // volume -- which is NOT what the cell mapper does, and the difference is OpenFOAM's.
    std::vector<scalar> out(static_cast<std::size_t>(nNewCells), scalar(0));
    for (label celli = 0; celli < nNewCells; ++celli)
    {
        const label oldCelli = map.cellMap[static_cast<std::size_t>(celli)];
        if (oldCelli > -1) out[static_cast<std::size_t>(celli)] = V0[static_cast<std::size_t>(oldCelli)];
    }
    // ...and then every MERGED old cell's volume is ADDED into the master's new cell, so a merged cell's
    // V0 is the SUM of its parts where its mapped field value is their weighted mean.
    for (std::size_t oldCelli = 0; oldCelli < map.reverseCellMap.size(); ++oldCelli)
    {
        const label index = map.reverseCellMap[oldCelli];
        if (index < -1)
        {
            const label celli = -index - 2;
            out[static_cast<std::size_t>(celli)] += V0[oldCelli];
        }
    }
    return out;
}

// ----------------------------------------------------------------------------------------------
// UNIT 7b-2: the surface field mapping. See the header.

FaceMapping faceMapping(
    const cpu::polyTopoChange::TopoChangeMap& map,
    label                                     nNewFaces)
{
    FaceMapping fm;
    // faceMapper's constructor (:233-247): no volume weighting here, unlike the cell mapper
    fm.direct = map.facesFromPoints.empty() && map.facesFromEdges.empty() && map.facesFromFaces.empty();

    if (fm.direct)
    {
        fm.directAddressing.assign(map.faceMap.begin(),
                                   map.faceMap.begin() + static_cast<std::size_t>(nNewFaces));
        for (std::size_t i = 0; i < fm.directAddressing.size(); ++i)
        {
            if (fm.directAddressing[i] < 0)
            {
                fm.directAddressing[i] = 0;
                fm.insertedFaces.push_back(static_cast<label>(i));
            }
        }
        return fm;
    }

    fm.addressing.resize(static_cast<std::size_t>(nNewFaces));
    fm.weights.resize(static_cast<std::size_t>(nNewFaces));
    const auto setAddrWeights = [&](const std::vector<cpu::polyTopoChange::ObjectMap>& maps)
    {
        for (const cpu::polyTopoChange::ObjectMap& m : maps)
        {
            if (m.masterObjects.empty()) continue;
            const std::size_t facei = static_cast<std::size_t>(m.index);
            if (!fm.addressing[facei].empty())
                throw std::runtime_error(
                    "brae faceMapper: face " + std::to_string(m.index) + " is mapped twice. OpenFOAM "
                    "FatalErrors here too (faceMapper.C:118-126).");
            fm.addressing[facei] = m.masterObjects;
            fm.weights[facei].assign(m.masterObjects.size(),
                                     scalar(1)/static_cast<scalar>(m.masterObjects.size()));
        }
    };
    setAddrWeights(map.facesFromPoints);
    setAddrWeights(map.facesFromEdges);
    setAddrWeights(map.facesFromFaces);

    for (label facei = 0; facei < nNewFaces; ++facei)
    {
        const label mapped = map.faceMap[static_cast<std::size_t>(facei)];
        if (mapped >= 0 && fm.addressing[static_cast<std::size_t>(facei)].empty())
        {
            fm.addressing[static_cast<std::size_t>(facei)] = {mapped};
            fm.weights[static_cast<std::size_t>(facei)] = {scalar(1)};
        }
    }
    for (label facei = 0; facei < nNewFaces; ++facei)
    {
        if (fm.addressing[static_cast<std::size_t>(facei)].empty())
        {
            fm.addressing[static_cast<std::size_t>(facei)] = {label(0)};
            fm.weights[static_cast<std::size_t>(facei)] = {scalar(1)};
            fm.insertedFaces.push_back(facei);
        }
    }
    return fm;
}

FaceMapping surfaceMapping(
    const FaceMapping& fm,
    label              nNewInternalFaces,
    label              nOldInternalFaces)
{
    FaceMapping sm;
    sm.direct = fm.direct;
    const std::size_t n = static_cast<std::size_t>(nNewInternalFaces);
    if (sm.direct)
    {
        sm.directAddressing.assign(fm.directAddressing.begin(), fm.directAddressing.begin() + n);
        for (std::size_t facei = 0; facei < n; ++facei)
        {
            // :55-60, and the test is STRICTLY greater -- see the header on the asymmetry
            if (sm.directAddressing[facei] > nOldInternalFaces) sm.directAddressing[facei] = 0;
        }
    }
    else
    {
        sm.addressing.assign(fm.addressing.begin(), fm.addressing.begin() + n);
        sm.weights.assign(fm.weights.begin(), fm.weights.begin() + n);
        for (std::size_t facei = 0; facei < n; ++facei)
        {
            label mx = -1;
            for (const label a : sm.addressing[facei]) { if (a > mx) mx = a; }
            // :74-79, and this one is >=
            if (mx >= nOldInternalFaces)
            {
                sm.addressing[facei] = {label(0)};
                sm.weights[facei] = {scalar(1)};
            }
        }
    }
    for (const label facei : fm.insertedFaces)
    {
        if (facei < nNewInternalFaces) sm.insertedFaces.push_back(facei);
    }
    return sm;
}

FaceMapping patchMapping(
    const FaceMapping& fm,
    label              newPatchStart,
    label              newPatchSize,
    label              oldPatchStart,
    label              oldPatchSize)
{
    const label oldPatchEnd = oldPatchStart + oldPatchSize;
    FaceMapping pm;
    pm.direct = fm.direct;
    const std::size_t b = static_cast<std::size_t>(newPatchStart);
    const std::size_t n = static_cast<std::size_t>(newPatchSize);
    if (pm.direct)
    {
        pm.directAddressing.assign(fm.directAddressing.begin() + b, fm.directAddressing.begin() + b + n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const label a = pm.directAddressing[i];
            if (a >= oldPatchStart && a < oldPatchEnd)
            {
                pm.directAddressing[i] = a - oldPatchStart;
            }
            else
            {
                // OpenFOAM's own commented-out `= 0` is right above this line: it writes -1 instead, and
                // Field::map then leaves the value ALONE, so such a face keeps whatever the resized
                // field held. Transcribed as written.
                pm.directAddressing[i] = -1;
            }
        }
        return pm;
    }
    pm.addressing.assign(fm.addressing.begin() + b, fm.addressing.begin() + b + n);
    pm.weights.assign(fm.weights.begin() + b, fm.weights.begin() + b + n);
    for (std::size_t i = 0; i < n; ++i)
    {
        std::vector<label>& addr = pm.addressing[i];
        std::vector<scalar>& w = pm.weights[i];
        label mn = addr.empty() ? label(-1) : addr[0];
        label mx = mn;
        for (const label a : addr) { if (a < mn) mn = a; if (a > mx) mx = a; }
        if (mn >= oldPatchStart && mx < oldPatchEnd)
        {
            for (label& a : addr) a -= oldPatchStart;
            continue;
        }
        // :170-205. Keep only the sources inside this patch and RE-SCALE their weights.
        std::size_t nActive = 0;
        scalar sumWeight = 0;
        for (std::size_t j = 0; j < addr.size(); ++j)
        {
            if (addr[j] >= oldPatchStart && addr[j] < oldPatchEnd)
            {
                addr[nActive] = addr[j] - oldPatchStart;
                w[nActive] = w[j];
                sumWeight += w[j];
                ++nActive;
            }
        }
        addr.resize(nActive);
        w.resize(nActive);
        if (nActive)
        {
            for (scalar& wi : w) wi /= sumWeight;
        }
    }
    return pm;
}

namespace {

// one body for both element types: the addressing is the same and only the zero differs
template <typename T>
std::vector<T> mapSurfaceFieldT(
    const std::vector<T>&      oldField,
    const FaceMapping&         sm,
    bool                       oriented,
    const std::vector<label>&  flipFaceFlux,
    const T&                   zero)
{
    std::vector<T> out;
    if (sm.direct)
    {
        out.assign(sm.directAddressing.size(), zero);
        if (!oldField.empty())
        {
            for (std::size_t i = 0; i < out.size(); ++i)
            {
                const label a = sm.directAddressing[i];
                if (a >= 0) out[i] = oldField[static_cast<std::size_t>(a)];
            }
        }
    }
    else
    {
        out.assign(sm.addressing.size(), zero);
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            T v = zero;
            for (std::size_t j = 0; j < sm.addressing[i].size(); ++j)
            {
                v = v + sm.weights[i][j]*oldField[static_cast<std::size_t>(sm.addressing[i][j])];
            }
            out[i] = v;
        }
    }
    if (oriented)
    {
        for (const label facei : flipFaceFlux)
        {
            if (facei < static_cast<label>(out.size()))
            {
                out[static_cast<std::size_t>(facei)] = scalar(-1)*out[static_cast<std::size_t>(facei)];
            }
        }
    }
    return out;
}

}   // namespace

std::vector<vector> mapSurfaceField(
    const std::vector<vector>& oldField,
    const FaceMapping&         sm,
    bool                       oriented,
    const std::vector<label>&  flipFaceFlux)
{
    return mapSurfaceFieldT<vector>(oldField, sm, oriented, flipFaceFlux, vector{0, 0, 0});
}

std::vector<scalar> mapSurfaceField(
    const std::vector<scalar>& oldField,
    const FaceMapping&         sm,
    bool                       oriented,
    const std::vector<label>&  flipFaceFlux)
{
    std::vector<scalar> out;
    if (sm.direct)
    {
        out.assign(sm.directAddressing.size(), scalar(0));
        if (!oldField.empty())
        {
            for (std::size_t i = 0; i < out.size(); ++i)
            {
                // a NEGATIVE entry leaves the value alone (Field::map, :386-396) -- which for a patch
                // face out of its own patch means it keeps the zero this vector was sized with
                const label a = sm.directAddressing[i];
                if (a >= 0) out[i] = oldField[static_cast<std::size_t>(a)];
            }
        }
    }
    else
    {
        out.assign(sm.addressing.size(), scalar(0));
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            scalar v = 0;
            for (std::size_t j = 0; j < sm.addressing[i].size(); ++j)
            {
                v += sm.weights[i][j]*oldField[static_cast<std::size_t>(sm.addressing[i][j])];
            }
            out[i] = v;
        }
    }
    if (oriented)
    {
        // MapFvSurfaceField.H:83-93. Only the faces inside this field's own size.
        for (const label facei : flipFaceFlux)
        {
            if (facei < static_cast<label>(out.size())) out[static_cast<std::size_t>(facei)] *= scalar(-1);
        }
    }
    return out;
}

// ----------------------------------------------------------------------------------------------
// UNIT 7: the driver. See the header.

namespace {

// A ChangedMesh back into a PrimitiveMesh, keeping the old patches' NAMES and TYPES: changeMesh returns
// starts and sizes, and every walk below asks the patches for their type.
PrimitiveMesh rebuiltMesh(
    const PrimitiveMesh&                     old,
    const cpu::polyTopoChange::ChangedMesh&  out)
{
    std::vector<label> faceVerts;
    std::vector<label> faceOffsets;
    faceOffsets.reserve(out.faces.size() + 1);
    faceOffsets.push_back(0);
    for (const std::vector<label>& f : out.faces)
    {
        faceVerts.insert(faceVerts.end(), f.begin(), f.end());
        faceOffsets.push_back(static_cast<label>(faceVerts.size()));
    }
    std::vector<label> nbr(out.faceNeighbour.begin(),
                           out.faceNeighbour.begin() + static_cast<std::size_t>(out.nInternalFaces));
    std::vector<PatchInfo> patches = old.patches();
    for (std::size_t p = 0; p < patches.size(); ++p)
    {
        patches[p].start = out.patchStarts[p];
        patches[p].size = out.patchSizes[p];
    }
    PrimitiveMesh m;
    m.assign(out.points, std::move(faceVerts), std::move(faceOffsets), out.faceOwner, std::move(nbr),
             std::move(patches), out.nCells);
    return m;
}

// the addressing every stage of a step reads, rebuilt after each change
struct StepAddressing
{
    FvGeometry                      g;
    std::vector<std::vector<label>> cells;
    std::vector<std::vector<label>> pointCells;
    std::vector<std::vector<label>> cellPoints;
    MeshEdges                       edges;
    std::vector<std::vector<label>> faceEdges;
    std::vector<std::vector<label>> edgeFaces;
    std::vector<std::vector<label>> cellEdges;
    std::vector<std::vector<label>> cellCells;
    std::vector<std::vector<label>> pointFaces;
};

void buildAddressing(
    const PrimitiveMesh& m,
    StepAddressing&      a)
{
    a.g.build(m);
    a.cells = meshCells(m);
    // pointCells through the CELLS branch, which is the one dynamicRefineFvMesh meets -- measured, and
    // the three branches sum differently
    a.pointCells = pointCellsFromCells(m, a.cells);
    a.cellPoints = cellPointsFromCells(m, a.cells);
    a.edges = buildMeshEdges(m);
    a.faceEdges = buildFaceEdges(m, a.edges);
    a.edgeFaces = buildEdgeFaces(m, a.faceEdges);
    a.cellEdges = buildCellEdges(a.cells, a.faceEdges);
    a.cellCells = buildCellCells(m);
    a.pointFaces = meshPointFaces(m);
}

cpu::hexRef8::MeshView hexView(
    const PrimitiveMesh&  m,
    const StepAddressing& a)
{
    cpu::hexRef8::MeshView v;
    v.m = &m;
    v.edges = &a.edges;
    v.faceEdges = &a.faceEdges;
    v.edgeFaces = &a.edgeFaces;
    v.cells = &a.cells;
    v.cellPoints = &a.cellPoints;
    v.pointCells = &a.pointCells;
    v.cellEdges = &a.cellEdges;
    v.cellCentres = &a.g.C();
    v.faceCentres = &a.g.Cf();
    return v;
}

cpu::polyTopoChange::TopoActions actionsFromMesh(const PrimitiveMesh& m)
{
    std::vector<label> starts, sizes;
    for (const PatchInfo& p : m.patches()) { starts.push_back(p.start); sizes.push_back(p.size); }
    std::vector<std::vector<label>> faces(static_cast<std::size_t>(m.nFaces()));
    for (label f = 0; f < m.nFaces(); ++f)
    {
        faces[static_cast<std::size_t>(f)].assign(
            m.faceVerts().begin() + m.faceOffsets()[static_cast<std::size_t>(f)],
            m.faceVerts().begin() + m.faceOffsets()[static_cast<std::size_t>(f) + 1]);
    }
    std::vector<label> nbr(static_cast<std::size_t>(m.nFaces()), label(-1));
    for (label f = 0; f < m.nInternalFaces(); ++f) nbr[static_cast<std::size_t>(f)] = m.neighbour()[f];
    cpu::polyTopoChange::TopoActions a;
    cpu::polyTopoChange::addMesh(a, m.points(), faces, m.owner(), nbr, m.nCells(), starts, sizes);
    return a;
}

cpu::polyTopoChange::ChangeMeshInput changeInput(
    const PrimitiveMesh& m,
    label                nZones)
{
    cpu::polyTopoChange::ChangeMeshInput ci;
    ci.nOldPoints = static_cast<label>(m.points().size());
    ci.nOldFaces = m.nFaces();
    ci.nOldCells = m.nCells();
    for (const PatchInfo& p : m.patches())
    {
        ci.oldPatchStarts.push_back(p.start);
        ci.oldPatchSizes.push_back(p.size);
        ci.patchTypes.push_back(p.type);
    }
    ci.oldPatchNMeshPoints.assign(ci.oldPatchStarts.size(), label(0));
    // THE CALLER'S COUNT, not zero. This was hardcoded to 0, which made changeMesh's zone refusal
    // unreachable from every caller -- a refusal standing in front of a silent substitution, the class
    // this project keeps finding. A mesh with a cellZone, faceZone or pointZone now stops by name.
    ci.nZones = nZones;
    return ci;
}


// protectedCell_ renumbered through a change (:518-530 and :700-712, the same block twice): the new cell
// is protected when the cell it came from was. A new cell with cellMap -1 is NOT protected.
void renumberProtectedCells(
    std::vector<char>&        protectedCell,
    const std::vector<label>& cellMap,
    label                     nNewCells)
{
    if (protectedCell.empty()) return;
    std::vector<char> next(static_cast<std::size_t>(nNewCells), char(0));
    for (label celli = 0; celli < nNewCells; ++celli)
    {
        const label oldCelli = cellMap[static_cast<std::size_t>(celli)];
        if (oldCelli >= 0 && protectedCell[static_cast<std::size_t>(oldCelli)]) next[static_cast<std::size_t>(celli)] = 1;
    }
    protectedCell.swap(next);
}

// ONE BUILDER for the MapPolyMesh the corrections read, because there are three call sites and each used to
// fill a different subset -- so a field one site set was silently defaulted at another. tools/default_audit
// reported exactly that (5 unlisted omissions over 3 sites), and this is the fix the of-defaults skill
// prescribes: one builder, every field it can fill, filled.
//
// What it CANNOT fill is the faceMapper addressing (`faceMapperDirect`, `faceDirectAddressing`,
// `faceAddressing`, `faceWeights`) and the oracle's own notes (`phase`, `mapCarriedOldVolumes`,
// `openfoamSaysDirect`): those exist for the unit-6 gates, which take them from OpenFOAM. They stay at
// their defaults HERE, in one place, rather than at three.
MapPolyMesh mapPolyMeshFrom(
    const cpu::polyTopoChange::TopoChangeMap& map,
    const std::vector<scalar>&                oldCellVolumes)
{
    MapPolyMesh mpm;
    mpm.nOldCells = map.nOldCells;
    mpm.cellMap = map.cellMap;
    mpm.reverseCellMap = map.reverseCellMap;
    for (const cpu::polyTopoChange::ObjectMap& m : map.cellsFromCells)
    {
        mpm.cellsFromCells.emplace_back(m.index, m.masterObjects);
    }
    mpm.oldCellVolumes = oldCellVolumes;
    mpm.faceMap = map.faceMap;
    mpm.reverseFaceMap = map.reverseFaceMap;
    mpm.reversePointMap = map.reversePointMap;
    return mpm;
}

// fvMesh::updateMesh + dynamicRefineFvMesh::mapFields, on the state the driver carries: every cell field
// through the cell mapper, the old-time volumes through their own rule and then corrected.
void mapCarriedFields(
    RefineUpdateState&                        s,
    const cpu::polyTopoChange::TopoChangeMap& map,
    const std::vector<scalar>&                oldCellVolumes,
    label                                     nNewCells,
    const std::vector<scalar>&                newV,
    label                                     nOldInternalFaces)
{
    const CellMapping cm = cellMapping(map, nNewCells, oldCellVolumes);
    for (std::vector<scalar>& f : s.cellScalars) f = mapCellField(f, cm);
    for (std::vector<vector>& f : s.cellVectors) f = mapCellField(f, cm);
    // the whole fields' cell halves, mapped BEFORE their patch fields -- which is OpenFOAM's order
    // (MapGeometricFields maps the internal field, then the boundary) and load-bearing, because a patch
    // field's unmapped faces are filled from the internal field it reads here
    for (GeometricField<scalar>* f : s.carriedScalarFields) f->internal = mapCellField(f->internal, cm);
    for (GeometricField<vector>* f : s.carriedVectorFields) f->internal = mapCellField(f->internal, cm);

    // ...and every carried surface field, through the face mapper sliced to the internal faces and to
    // each patch. The new mesh is already in place when this runs, which is what gives the patch starts.
    // the face mapping serves BOTH the carried patch fields (unit 8a) and the carried surface fields
    // (7b-2), so it is built whenever there is either
    if (!s.surfaceScalars.empty() || !s.surfaceVectors.empty()
     || !s.carriedScalarFields.empty() || !s.carriedVectorFields.empty())
    {
        const FaceMapping fm = faceMapping(map, s.m.nFaces());
        const FaceMapping sm = surfaceMapping(fm, s.m.nInternalFaces(), nOldInternalFaces);
        const std::vector<PatchInfo>& patches = s.m.patches();
        std::vector<FaceMapping> pm;
        pm.reserve(patches.size());
        for (std::size_t p = 0; p < patches.size(); ++p)
        {
            pm.push_back(patchMapping(fm, patches[p].start, patches[p].size, map.oldPatchStarts[p],
                                      map.oldPatchSizes[p]));
        }

        // what the hull average reads off the NEW mesh, built once for all the fields
        const MapPolyMesh mpmF = mapPolyMeshFrom(map, oldCellVolumes);
        FluxMeshView fv;
        fv.nInternalFaces = s.m.nInternalFaces();
        for (const PatchInfo& pp : patches)
        {
            fv.patchStart.push_back(pp.start);
            fv.patchSize.push_back(pp.size);
            fv.patchHoldsNoValues.push_back(pp.type == "empty" ? char(1) : char(0));
        }
        fv.owner = s.m.owner();
        fv.neighbour = s.m.neighbour();
        fv.cells = meshCells(s.m);
        FvGeometry gNew;
        gNew.build(s.m);
        std::vector<std::vector<vector>> SfBnd(patches.size());
        std::vector<std::vector<scalar>> magSfBnd(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            SfBnd[pi].assign(gNew.Sf().begin() + patches[pi].start,
                             gNew.Sf().begin() + patches[pi].start + patches[pi].size);
            magSfBnd[pi].assign(gNew.magSf().begin() + patches[pi].start,
                                gNew.magSf().begin() + patches[pi].start + patches[pi].size);
        }

        // UNIT 8a: whole fields, cells and patch fields together. The cell half went through the cell
        // mapper above; here each patch field is mapped by its OWN autoMap, which is what fills an
        // unmapped face from the internal field and what maps the type's own state.
        for (GeometricField<scalar>* f : s.carriedScalarFields)
        {
            if (f->boundary.size() != patches.size())
                throw std::runtime_error(
                    "brae dynamicRefineFvMesh: a carried field has " + std::to_string(f->boundary.size())
                    + " patch fields and the mesh " + std::to_string(patches.size()) + " patches. A "
                    "change that alters the patch count would leave every patch field reading another "
                    "patch's data.");
            for (std::size_t p = 0; p < patches.size(); ++p)
            {
                if (!f->boundary[p]->autoMapComplete())
                    throw std::runtime_error(
                        "brae dynamicRefineFvMesh: patch `" + patches[p].name + "` of a carried scalar "
                        "field has no autoMap yet, so a mesh change would leave its per-face state stale "
                        "or short. Ported types are named in fv_patch_field.cuh; this one is not one.");
                f->boundary[p]->autoMap(pm[p], f->internal);
            }
        }
        for (GeometricField<vector>* f : s.carriedVectorFields)
        {
            if (f->boundary.size() != patches.size())
                throw std::runtime_error(
                    "brae dynamicRefineFvMesh: a carried vector field has "
                    + std::to_string(f->boundary.size()) + " patch fields and the mesh "
                    + std::to_string(patches.size()) + " patches.");
            for (std::size_t p = 0; p < patches.size(); ++p)
            {
                if (!f->boundary[p]->autoMapComplete())
                    throw std::runtime_error(
                        "brae dynamicRefineFvMesh: patch `" + patches[p].name + "` of a carried vector "
                        "field has no autoMap yet.");
                f->boundary[p]->autoMap(pm[p], f->internal);
            }
        }

        // STEP ONE for the surface fields: the addressing, and the flip on an oriented one.
        for (RefineUpdateState::CarriedSurfaceField& f : s.surfaceScalars)
        {
            f.field = mapSurfaceField(f.field, sm, f.oriented, map.flipFaceFlux);
            std::vector<std::vector<scalar>> bnd(patches.size());
            for (std::size_t p = 0; p < patches.size(); ++p)
            {
                // a patch field's own values are indexed within the patch, which is what fvPatchMapper
                // rebased onto; flipFaceFlux carries internal faces only, so nothing is flipped here
                bnd[p] = mapSurfaceField(f.bnd[p], pm[p], false, std::vector<label>());
            }
            f.bnd.swap(bnd);
        }

        // THEN THE PER-FLUX CORRECTION, on the fields the dictionary names a velocity for -- and only
        // those. It runs BEFORE the hull average, which is OpenFOAM's order (:345-420 then :424-437) and
        // not a detail: the correction writes injected internal faces too, and the hull average writes
        // over them.
        if (!s.injectedPhiU.empty())
        {
            const std::vector<char> mf = masterFaces(mpmF, s.m.nFaces());
            for (std::size_t k = 0; k < s.surfaceScalars.size(); ++k)
            {
                const std::string& U = (k < s.surfaceScalarVelocity.size())
                                     ? s.surfaceScalarVelocity[k] : std::string();
                if (U.empty() || U == "none") continue;
                correctFluxes(s.surfaceScalars[k].field, s.surfaceScalars[k].bnd, s.injectedPhiU,
                              s.injectedPhiUBnd, mpmF, mf, fv);
            }
        }

        // the surface VECTORS -- Uf -- through the same addressing, and then the FLAT hull average,
        // because an unoriented field is averaged as itself
        for (RefineUpdateState::CarriedSurfaceVectorField& f : s.surfaceVectors)
        {
            f.field = mapSurfaceField(f.field, sm, f.oriented, map.flipFaceFlux);
            std::vector<std::vector<vector>> bnd(patches.size());
            for (std::size_t p = 0; p < patches.size(); ++p)
            {
                bnd[p] = mapSurfaceField(f.bnd[p], pm[p], false, std::vector<label>());
            }
            f.bnd.swap(bnd);
            mapNewInternalFacesFlat(f.field, f.bnd, mpmF, fv);
        }

        // ...AND THE HULL AVERAGE LAST, over the injected internal faces, on EVERY surface field in the
        // registry and not only on the ones correctFluxes names: mapFields calls mapNewInternalFaces
        // outside the per-flux loop. MEASURED: without it brae wrote face 0's value (0) on new internal
        // face 6293 where OpenFOAM had 5232.5, the mean of the two old faces of its hull.
        for (RefineUpdateState::CarriedSurfaceField& f : s.surfaceScalars)
        {
            if (f.oriented)
            {
                mapNewInternalFacesOriented(f.field, f.bnd, gNew.Sf(), SfBnd, gNew.magSf(), magSfBnd,
                                            mpmF, fv);
            }
            else
            {
                mapNewInternalFacesFlat(f.field, f.bnd, mpmF, fv);
            }
        }
    }

    // V0 comes into existence at the FIRST change and not before -- fvMesh::updateMesh only stores old
    // volumes when the current ones already exist, and what it stores is the OLD mesh's volumes.
    if (s.V0.empty())
    {
        s.V0 = oldCellVolumes;
    }
    if (!s.V0.empty())
    {
        const MapPolyMesh mpm = mapPolyMeshFrom(map, oldCellVolumes);
        s.V0 = correctOldVolumes(mpm, mapOldVolumes(s.V0, map, nNewCells), newV);
    }
}


// THE PATCH OBJECTS ARE ASSIGNED ELEMENT BY ELEMENT, NOT REPLACED, and that is not a style choice: every
// patch field holds a `const FvPatch&` into this vector, and `patches = buildPatches(...)` is a MOVE
// assignment -- it steals the new buffer and frees the old one, so every one of those references dangles.
// MEASURED: with the vector moved, the first two changes of the gate's own fixture read freed memory and
// happened to work, and the third segfaulted in strlen on the patch's NAME. Assigning the elements keeps
// the storage, so the references stay valid and now see the new patch.
void updatePatchesInPlace(
    std::vector<FvPatch>& patches,
    std::vector<FvPatch>  fresh)
{
    if (fresh.size() != patches.size())
        throw std::runtime_error(
            "brae dynamicRefineFvMesh: the change left " + std::to_string(fresh.size()) + " patches where "
            "the mesh had " + std::to_string(patches.size()) + ". hexRef8 splits faces WITHIN a patch and "
            "never adds or removes one; a change that did would dangle every patch field's reference.");
    for (std::size_t p = 0; p < patches.size(); ++p)
    {
        patches[p] = std::move(fresh[p]);
    }
}

}   // namespace
// See the header: polyTopoChange's cell half of resetZones, reduced to one zone.
void renumberCellZones(
    std::map<std::string, std::vector<label>>& zones,
    const std::vector<label>&                  cellMap,
    label                                      nOldCells)
{
    for (std::pair<const std::string, std::vector<label>>& z : zones)
    {
        std::vector<char> inOldZone(static_cast<std::size_t>(nOldCells), char(0));
        for (const label c : z.second)
        {
            if (c >= 0 && c < nOldCells) inOldZone[static_cast<std::size_t>(c)] = char(1);
        }
        std::vector<label> out;
        out.reserve(z.second.size());
        // ASCENDING, which is the order OpenFOAM's walk produces and the order its stableSort keeps
        for (std::size_t nc = 0; nc < cellMap.size(); ++nc)
        {
            const label old = cellMap[nc];
            if (old < 0 || old >= nOldCells) continue;
            if (inOldZone[static_cast<std::size_t>(old)]) out.push_back(static_cast<label>(nc));
        }
        z.second.swap(out);
    }
}


RefineUpdateStep refineUpdate(
    RefineUpdateState&         s,
    const RefineControls&      c,
    const std::vector<scalar>& field,
    label                      timeIndex)
{
    RefineUpdateStep r;

    // :1295-1312. refineInterval 0 switches the whole thing off; a negative one is an error.
    if (c.refineInterval == 0) return r;
    if (c.refineInterval < 0)
        throw std::runtime_error(
            "brae dynamicRefineFvMesh::updateTopology: refineInterval " + std::to_string(c.refineInterval)
            + " is illegal; it must be >= 1. OpenFOAM FatalErrors here too (:1307-1312).");

    // :1320. NOT at time 0: there is no V0 yet because the mesh has not moved. OpenFOAM's own comment.
    if (!(timeIndex > 0 && (timeIndex % c.refineInterval) == 0)) return r;

    if (c.maxCells <= 0)
        throw std::runtime_error(
            "brae dynamicRefineFvMesh::updateTopology: maxCells " + std::to_string(c.maxCells)
            + " is illegal; it must be > 0. OpenFOAM FatalErrors here too (:1326-1332).");
    if (c.maxRefinement <= 0)
        throw std::runtime_error(
            "brae dynamicRefineFvMesh::updateTopology: maxRefinement " + std::to_string(c.maxRefinement)
            + " is illegal; it must be > 0. OpenFOAM FatalErrors here too (:1336-1342).");

    if (static_cast<label>(field.size()) != s.m.nCells())
        throw std::runtime_error(
            "brae dynamicRefineFvMesh::updateTopology: the driving field has "
            + std::to_string(field.size()) + " values and the mesh " + std::to_string(s.m.nCells())
            + " cells. The field must be the one on the CURRENT mesh.");

    StepAddressing a;
    buildAddressing(s.m, a);
    if (s.patches.empty()) s.patches = buildPatches(s.m, a.g);
    else updatePatchesInPlace(s.patches, buildPatches(s.m, a.g));

    // :1357-1367. A fresh marker every step, marked from the field alone.
    std::vector<char> refineCell(static_cast<std::size_t>(s.m.nCells()), char(0));
    selectRefineCandidates(c.lowerRefineLevel, c.upperRefineLevel, field, a.pointCells, s.m.nCells(),
                           refineCell);

    // the field as the step will see it after a refinement -- see below
    std::vector<scalar> stepField = field;

    // :1369-1416. Only if the mesh is still under maxCells.
    if (s.m.nCells() < c.maxCells)
    {
        r.cellsToRefine = selectRefineCells(c.maxCells, c.maxRefinement, refineCell, s.levels.cellLevel,
                                           s.protectedCell, s.m.nCells(), s.m, s.patches);
        if (!r.cellsToRefine.empty())
        {
            // ---- refine (:442-535) ----------------------------------------------------------------
            cpu::polyTopoChange::TopoActions act = actionsFromMesh(s.m);
            const cpu::hexRef8::MeshView v = hexView(s.m, a);
            std::vector<std::string> patchTypes;
            for (const PatchInfo& p : s.m.patches()) patchTypes.push_back(p.type);
            cpu::hexRef8::RefinementMarks marks;
            (void)cpu::hexRef8::setRefinementPointsAndCells(v, s.levels, r.cellsToRefine, patchTypes,
                                                            act, marks);
            cpu::hexRef8::setRefinementFaces(v, s.levels, marks, act);
            // marks.cellAddedCells, NOT the RETURN value: section 11 indexes cellAddedCells BY CELL
            // (hexRef8.C:4295), and what setRefinement returns is the COMPACTED per-requested-cell form
            // it builds afterwards. Passing the compacted one hands storeSplit the request INDEX as a
            // cell label -- which still agreed with OpenFOAM on a first refinement, because a fresh
            // history's visibleCells is the identity so any index reuses "its own" entry, and the
            // compaction then dropped the difference. It came apart on the SECOND refinement: 470
            // splits allocated 470 new parents where OpenFOAM allocated 64 and reused 406.
            cpu::hexRef8::storeRefinementHistory(s.history, marks.cellAddedCells,
                                                 static_cast<label>(marks.newCellLevel.size()));

            cpu::polyTopoChange::ChangedMesh out;
            cpu::polyTopoChange::changeMesh(act, changeInput(s.m, s.nZones), out, r.refineMap);

            // hexRef8::updateMesh -- the levels and the history through the change
            s.levels.cellLevel = marks.newCellLevel;
            s.levels.pointLevel = marks.newPointLevel;
            cpu::hexRef8::updateLevels(s.levels, r.refineMap.reverseCellMap, r.refineMap.reversePointMap,
                                       r.refineMap.cellMap, r.refineMap.pointMap, out.nCells,
                                       static_cast<label>(out.points.size()));
            cpu::hexRef8::historyUpdateMesh(s.history, r.refineMap.reverseCellMap, out.nCells);

            // THE DRIVING FIELD IS MAPPED THROUGH cellMap, and that is OpenFOAM's rule and not a
            // simplification: a refinement leaves every cellsFrom* map EMPTY (measured -- 0 entries on
            // every refinement arm of hex_ref8_vs_openfoam), so cellMapper takes its DIRECT branch
            // (cellMapper.C:36-60) and each new cell takes the value of the cell it came from. The
            // interpolative branch -- which is VOLUME-WEIGHTED over the masters when the map carries old
            // cell volumes -- is only reached by a merge, and that is unit 7b's.
            {
                std::vector<scalar> mapped(static_cast<std::size_t>(out.nCells), scalar(0));
                for (label celli = 0; celli < out.nCells; ++celli)
                {
                    const label oldCelli = r.refineMap.cellMap[static_cast<std::size_t>(celli)];
                    // an inserted cell (cellMap -1) takes cell 0's value, as cellMapper's own
                    // inserted-object handling does
                    mapped[static_cast<std::size_t>(celli)] =
                        stepField[static_cast<std::size_t>(oldCelli >= 0 ? oldCelli : 0)];
                }
                stepField.swap(mapped);
            }

            renumberProtectedCells(s.protectedCell, r.refineMap.cellMap, out.nCells);
            renumberCellZones(s.cellZones, r.refineMap.cellMap, r.refineMap.nOldCells);
            const std::vector<scalar> oldV =
                s.injectedRefineOldV.empty() ? a.g.V() : s.injectedRefineOldV;
            const label nOldInternalFaces = s.m.nInternalFaces();
            s.injectedPhiU = s.injectedPhiURefine;
            s.injectedPhiUBnd = s.injectedPhiURefineBnd;
            s.m = rebuiltMesh(s.m, out);
            buildAddressing(s.m, a);
            updatePatchesInPlace(s.patches, buildPatches(s.m, a.g));
            mapCarriedFields(s, r.refineMap, oldV, out.nCells, a.g.V(), nOldInternalFaces);

            // :1391-1411. refineCell REBUILT THROUGH THE MAP: a cell stays marked if it is new, if it is
            // not its old cell's master, or if its old cell was marked. That is what keeps every child of
            // a refined cell marked, so the unrefinement below cannot undo this refinement.
            {
                std::vector<char> next(static_cast<std::size_t>(out.nCells), char(0));
                for (label celli = 0; celli < out.nCells; ++celli)
                {
                    const label oldCelli = r.refineMap.cellMap[static_cast<std::size_t>(celli)];
                    if (oldCelli < 0
                     || r.refineMap.reverseCellMap[static_cast<std::size_t>(oldCelli)] != celli
                     || refineCell[static_cast<std::size_t>(oldCelli)])
                    {
                        next[static_cast<std::size_t>(celli)] = 1;
                    }
                }
                refineCell.swap(next);
            }
            r.refineCellAfterMap = refineCell;

            for (label i = 0; i < c.nBufferLayers; ++i)
            {
                extendMarkedCells(s.m, s.patches, a.cells, refineCell);
            }
            r.refineCellAfterBuffer = refineCell;

            r.refined = true;
            r.hasChanged = true;
        }
    }

    // :1419-1440. The unrefinement, which runs whether or not anything was refined.
    {
        RefinementHistory hv;
        hv.parent = s.history.parent;
        hv.visibleCells = s.history.visibleCells;
        hv.active = s.history.active;
        const std::vector<label> splitPoints =
            getSplitPoints(hv, s.levels.cellLevel, a.pointCells, a.cellPoints, s.m);
        const std::vector<scalar> pFld = maxCellField(stepField, a.pointCells);
        r.pointsToUnrefine = selectUnrefinePoints(c.unrefineLevel, refineCell, pFld, splitPoints,
                                                  s.protectedCell, s.levels.cellLevel, a.pointCells,
                                                  s.m, s.patches);
        if (!r.pointsToUnrefine.empty())
        {
            // ---- unrefine (:537-716) ---------------------------------------------------------------
            cpu::polyTopoChange::TopoActions act = actionsFromMesh(s.m);
            const cpu::hexRef8::MeshView v = hexView(s.m, a);
            cpu::hexRef8::setUnrefinementLevels(v, s.levels, s.history, r.pointsToUnrefine);

            // the faces around the split points, which is what removeFaces is asked to remove
            std::vector<char> seen(static_cast<std::size_t>(s.m.nFaces()), char(0));
            std::vector<label> splitFaces;
            for (const label pointi : r.pointsToUnrefine)
            {
                for (const label facei : a.pointFaces[static_cast<std::size_t>(pointi)])
                {
                    if (!seen[static_cast<std::size_t>(facei)])
                    {
                        seen[static_cast<std::size_t>(facei)] = 1;
                        splitFaces.push_back(facei);
                    }
                }
            }
            cpu::removeFaces::RemoveFacesView rv;
            rv.m = &s.m;
            rv.edges = &a.edges;
            rv.faceEdges = &a.faceEdges;
            rv.edgeFaces = &a.edgeFaces;
            rv.cells = &a.cells;
            rv.pointFaces = &a.pointFaces;
            rv.faceAreas = &a.g.Sf();
            std::vector<label> cellRegion, cellRegionMaster, facesToRemove;
            cpu::removeFaces::compatibleRemoves(s.m, a.cellCells, splitFaces, cellRegion,
                                                cellRegionMaster, facesToRemove);
            // hexRef8 builds its faceRemover with GREAT, so the feature-angle guard never runs
            // (hexRef8.C:1967)
            const cpu::removeFaces::RemoveFacesDecisions dec =
                cpu::removeFaces::setRefinementDecisions(rv, facesToRemove, cellRegion, cellRegionMaster,
                                                         scalar(1.0e+15));
            cpu::removeFaces::setRefinementActions(rv, dec, facesToRemove, cellRegion, cellRegionMaster,
                                                   act);

            cpu::polyTopoChange::ChangedMesh out;
            cpu::polyTopoChange::changeMesh(act, changeInput(s.m, s.nZones), out, r.unrefineMap);

            cpu::hexRef8::updateLevels(s.levels, r.unrefineMap.reverseCellMap,
                                       r.unrefineMap.reversePointMap, r.unrefineMap.cellMap,
                                       r.unrefineMap.pointMap, out.nCells,
                                       static_cast<label>(out.points.size()));
            cpu::hexRef8::historyUpdateMesh(s.history, r.unrefineMap.reverseCellMap, out.nCells);
            renumberProtectedCells(s.protectedCell, r.unrefineMap.cellMap, out.nCells);
            renumberCellZones(s.cellZones, r.unrefineMap.cellMap, r.unrefineMap.nOldCells);
            const std::vector<scalar> oldV =
                s.injectedUnrefineOldV.empty() ? a.g.V() : s.injectedUnrefineOldV;
            const label nOldInternalFaces = s.m.nInternalFaces();
            // unrefine's OWN flux correction is keyed on faceToSplitPoint, which is built from the
            // PRE-change mesh (:555-574): for every split point, every face at the other end of one of
            // its edges, paired with that other point. Built here, before the mesh is replaced.
            std::vector<std::pair<label, label>> faceToSplitPoint;
            if (!s.injectedPhiU.empty())
            {
                std::vector<char> seenFace(static_cast<std::size_t>(s.m.nFaces()), char(0));
                for (const label pointi : r.pointsToUnrefine)
                {
                    for (const label edgei : a.edges.pointEdges[static_cast<std::size_t>(pointi)])
                    {
                        const label otherPointi =
                            (a.edges.start[static_cast<std::size_t>(edgei)] == pointi)
                          ? a.edges.end[static_cast<std::size_t>(edgei)]
                          : a.edges.start[static_cast<std::size_t>(edgei)];
                        for (const label facei : a.pointFaces[static_cast<std::size_t>(otherPointi)])
                        {
                            // a Map<label>::insert keeps the FIRST value written for a key
                            if (seenFace[static_cast<std::size_t>(facei)]) continue;
                            seenFace[static_cast<std::size_t>(facei)] = 1;
                            faceToSplitPoint.emplace_back(facei, otherPointi);
                        }
                    }
                }
            }
            s.injectedPhiU = s.injectedPhiUUnrefine;
            s.injectedPhiUBnd = s.injectedPhiUUnrefineBnd;
            s.m = rebuiltMesh(s.m, out);
            buildAddressing(s.m, a);
            updatePatchesInPlace(s.patches, buildPatches(s.m, a.g));
            mapCarriedFields(s, r.unrefineMap, oldV, out.nCells, a.g.V(), nOldInternalFaces);
            // ...and then unrefine's second correction, which runs AFTER updateMesh and so after the
            // hull average (:610-689)
            if (!s.injectedPhiU.empty() && !faceToSplitPoint.empty())
            {
                const MapPolyMesh mpmU = mapPolyMeshFrom(r.unrefineMap, oldV);
                FluxMeshView fvU;
                fvU.nInternalFaces = s.m.nInternalFaces();
                for (const PatchInfo& pp : s.m.patches())
                {
                    fvU.patchStart.push_back(pp.start);
                    fvU.patchSize.push_back(pp.size);
                    fvU.patchHoldsNoValues.push_back(pp.type == "empty" ? char(1) : char(0));
                }
                fvU.owner = s.m.owner();
                fvU.neighbour = s.m.neighbour();
                fvU.cells = a.cells;
                for (std::size_t k = 0; k < s.surfaceScalars.size(); ++k)
                {
                    const std::string& U = (k < s.surfaceScalarVelocity.size())
                                         ? s.surfaceScalarVelocity[k] : std::string();
                    if (U.empty() || U == "none") continue;
                    correctFluxesUnrefine(s.surfaceScalars[k].field, s.surfaceScalars[k].bnd,
                                          s.injectedPhiU, s.injectedPhiUBnd, faceToSplitPoint, mpmU,
                                          fvU);
                }
            }

            r.unrefined = true;
            r.hasChanged = true;
        }
    }

    // :1443-1450. Every tenth iteration, and the counter starts at 0 -- so the FIRST step compacts.
    if ((s.nRefinementIterations % 10) == 0)
    {
        cpu::hexRef8::compactHistory(s.history);
        r.compacted = true;
    }
    ++s.nRefinementIterations;

    return r;
}

}   // namespace dynamicRefine
}   // namespace brae
