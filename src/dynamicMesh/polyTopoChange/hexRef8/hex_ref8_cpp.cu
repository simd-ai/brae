// hex_ref8_cpp.cu -- see the header for what a split is, how the unit is divided and what is refused.
#include "hex_ref8_cpp.cuh"

#include "foam_dict.cuh"   // isCoupledInterfaceType

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace hexRef8 {

using polyTopoChange::addPoint;
using polyTopoChange::addCell;

namespace {

constexpr const char* WHO = "brae hexRef8::setRefinement: ";

// DynamicList's `operator()(label)` GROWS the list and returns a reference, which is how OpenFOAM writes
// `newPointLevel(addedPointi) = ...` for a point that does not exist yet (hexRef8.C:3364 and friends).
// Transcribed rather than replaced by a pre-size, because the grown entries are zero-initialised and a
// later read of one that was never written would then read 0 in both codes.
label& at(
    std::vector<label>& l,
    label               i)
{
    if (i >= static_cast<label>(l.size())) l.resize(static_cast<std::size_t>(i) + 1, label(0));
    return l[static_cast<std::size_t>(i)];
}

// the face's vertex list, from brae's CSR store
std::vector<label> faceVerts(
    const PrimitiveMesh& m,
    label                facei)
{
    const label b = m.faceOffsets()[facei];
    const label e = m.faceOffsets()[facei + 1];
    return std::vector<label>(m.faceVerts().begin() + b, m.faceVerts().begin() + e);
}

}   // namespace


label findMaxLevel(
    const Levels&             lv,
    const std::vector<label>& f)
{
    // :658-672. STRICTLY greater, so the FIRST vertex at the maximum wins.
    label maxLevel = std::numeric_limits<label>::min();
    label maxFp = -1;
    for (std::size_t fp = 0; fp < f.size(); ++fp)
    {
        const label level = lv.pointLevel[static_cast<std::size_t>(f[fp])];
        if (level > maxLevel)
        {
            maxLevel = level;
            maxFp = static_cast<label>(fp);
        }
    }
    return maxFp;
}


label countAnchors(
    const Levels&             lv,
    const std::vector<label>& f,
    label                     anchorLevel)
{
    // :678-694
    label n = 0;
    for (const label pointi : f)
    {
        if (lv.pointLevel[static_cast<std::size_t>(pointi)] <= anchorLevel) ++n;
    }
    return n;
}


label faceLevel(
    const MeshView& v,
    const Levels&   lv,
    label           facei)
{
    // :801-826
    const std::vector<label> f = faceVerts(*v.m, facei);
    if (f.size() <= 4)
    {
        return lv.pointLevel[static_cast<std::size_t>(f[static_cast<std::size_t>(findMaxLevel(lv, f))])];
    }
    const label ownLevel = lv.cellLevel[static_cast<std::size_t>(v.m->owner()[facei])];
    if (countAnchors(lv, f, ownLevel) == 4) return ownLevel;
    if (countAnchors(lv, f, ownLevel + 1) == 4) return ownLevel + 1;
    return -1;
}


std::vector<std::vector<label>> setRefinementPointsAndCells(
    const MeshView&                 v,
    const Levels&                   lv,
    const std::vector<label>&       cellsToRefine,
    const std::vector<std::string>& patchTypes,
    TopoActions&                    a,
    RefinementMarks&                marks)
{
    if (!v.m || !v.edges || !v.cells || !v.cellPoints || !v.pointCells || !v.cellEdges
     || !v.cellCentres || !v.faceCentres)
        throw std::runtime_error(std::string(WHO) + "the mesh view is incomplete.");

    // THE REFUSAL, before anything is added. Every sync in setRefinement is a max or an or across a
    // coupled patch, and in serial with none they are identities -- so they are skipped, and the case
    // that would make them matter is refused instead of being run without them.
    for (const std::string& t : patchTypes)
    {
        if (isCoupledInterfaceType(t) || t == "processor" || t == "processorCyclic")
            throw std::runtime_error(
                std::string(WHO) + "patch type `" + t + "` is coupled. setRefinement synchronises "
                "edgeMidPoint (hexRef8.C:3438), the edge mid POSITIONS (:3496), faceMidPoint (:3626) and "
                "the boundary neighbour levels (:3590) across coupled patches. In serial with no coupled "
                "patch each is a max or an or over one value and brae skips them; with one they decide "
                "the answer. No adaptive interFoam tutorial has a coupled patch.");
    }

    const PrimitiveMesh& m = *v.m;
    const label nCells = m.nCells();
    const label nFaces = m.nFaces();
    const label nPoints = m.nPoints();
    const label nInternalFaces = m.nInternalFaces();
    const label nEdges = static_cast<label>(v.edges->start.size());

    // ---- 1. the levels, copied so the new entries can be appended (:3327-3341) -------------------
    marks.newCellLevel = lv.cellLevel;
    marks.newPointLevel = lv.pointLevel;

    // ---- 2. a point at the centre of every refined cell (:3347-3366) ------------------------------
    marks.cellMidPoint.assign(static_cast<std::size_t>(nCells), label(-1));
    for (const label celli : cellsToRefine)
    {
        // the master point is the FIRST vertex of the cell's FIRST face -- `mesh_.faces()[cells()[celli][0]][0]`
        const label firstFace = (*v.cells)[static_cast<std::size_t>(celli)][0];
        const label anchorPointi = m.faceVerts()[m.faceOffsets()[firstFace]];
        marks.cellMidPoint[static_cast<std::size_t>(celli)] =
            addPoint(a, (*v.cellCentres)[static_cast<std::size_t>(celli)], anchorPointi, /*inCell=*/true);
        at(marks.newPointLevel, marks.cellMidPoint[static_cast<std::size_t>(celli)]) =
            lv.cellLevel[static_cast<std::size_t>(celli)] + 1;
    }

    // ---- 3. which edges get split (:3394-3436) ----------------------------------------------------
    // An edge is split when a cell using it is split AND both its points are at or below that cell's
    // level. The loop MARKS, so the order cellEdges comes in is inert -- see unit 5a's gate header.
    marks.edgeMidPoint.assign(static_cast<std::size_t>(nEdges), label(-1));
    for (label celli = 0; celli < nCells; ++celli)
    {
        if (marks.cellMidPoint[static_cast<std::size_t>(celli)] < 0) continue;
        for (const label edgeI : (*v.cellEdges)[static_cast<std::size_t>(celli)])
        {
            const label e0 = v.edges->start[static_cast<std::size_t>(edgeI)];
            const label e1 = v.edges->end[static_cast<std::size_t>(edgeI)];
            if (lv.pointLevel[static_cast<std::size_t>(e0)] <= lv.cellLevel[static_cast<std::size_t>(celli)]
             && lv.pointLevel[static_cast<std::size_t>(e1)] <= lv.cellLevel[static_cast<std::size_t>(celli)])
            {
                marks.edgeMidPoint[static_cast<std::size_t>(edgeI)] = 12345;   // OpenFOAM's own marker
            }
        }
    }

    // ---- 4. a point at each split edge's midpoint (:3446-3524) ------------------------------------
    // TWO PHASES in OpenFOAM: the positions first and synced, then the points. The sync is the identity
    // here (refused above), but the two-phase shape is kept because the POSITION is the edge's own
    // centre computed once -- `edge::centre(points)`, which is the average of its two points and not a
    // difference, so it is the same number either way.
    for (label edgeI = 0; edgeI < nEdges; ++edgeI)
    {
        if (marks.edgeMidPoint[static_cast<std::size_t>(edgeI)] < 0) continue;
        const label e0 = v.edges->start[static_cast<std::size_t>(edgeI)];
        const label e1 = v.edges->end[static_cast<std::size_t>(edgeI)];
        const vector& p0 = m.points()[static_cast<std::size_t>(e0)];
        const vector& p1 = m.points()[static_cast<std::size_t>(e1)];
        // edge::centre is 0.5*(a + b) -- Foam::edge's own, not a midpoint by subtraction
        const vector mid{scalar(0.5)*(p0.x + p1.x), scalar(0.5)*(p0.y + p1.y), scalar(0.5)*(p0.z + p1.z)};
        // the master point is the edge's FIRST vertex (:3487)
        marks.edgeMidPoint[static_cast<std::size_t>(edgeI)] = addPoint(a, mid, e0, /*inCell=*/true);
        at(marks.newPointLevel, marks.edgeMidPoint[static_cast<std::size_t>(edgeI)]) =
            std::max(lv.pointLevel[static_cast<std::size_t>(e0)],
                     lv.pointLevel[static_cast<std::size_t>(e1)]) + 1;
    }

    // ---- 5. which faces get split (:3526-3624) ----------------------------------------------------
    marks.faceAnchorLevel.assign(static_cast<std::size_t>(nFaces), label(0));
    for (label facei = 0; facei < nFaces; ++facei)
    {
        marks.faceAnchorLevel[static_cast<std::size_t>(facei)] = faceLevel(v, lv, facei);
    }
    marks.faceMidPoint.assign(static_cast<std::size_t>(nFaces), label(-1));
    // internal faces: both cells are known here
    for (label facei = 0; facei < nInternalFaces; ++facei)
    {
        if (marks.faceAnchorLevel[static_cast<std::size_t>(facei)] < 0) continue;
        const label own = m.owner()[facei];
        const label nei = m.neighbour()[facei];
        const label newOwnLevel = lv.cellLevel[static_cast<std::size_t>(own)]
                                + (marks.cellMidPoint[static_cast<std::size_t>(own)] >= 0 ? 1 : 0);
        const label newNeiLevel = lv.cellLevel[static_cast<std::size_t>(nei)]
                                + (marks.cellMidPoint[static_cast<std::size_t>(nei)] >= 0 ? 1 : 0);
        if (newOwnLevel > marks.faceAnchorLevel[static_cast<std::size_t>(facei)]
         || newNeiLevel > marks.faceAnchorLevel[static_cast<std::size_t>(facei)])
        {
            marks.faceMidPoint[static_cast<std::size_t>(facei)] = 12345;
        }
    }
    // boundary faces: OpenFOAM builds newNeiLevel from the OWNER and swaps it across coupled patches
    // (:3582-3610). With no coupled patch the swap is the identity, so the neighbour level IS the
    // owner's -- which makes the two tests below the same test, and that is OpenFOAM's arithmetic here
    // rather than a simplification of it.
    for (label facei = nInternalFaces; facei < nFaces; ++facei)
    {
        if (marks.faceAnchorLevel[static_cast<std::size_t>(facei)] < 0) continue;
        const label own = m.owner()[facei];
        const label newOwnLevel = lv.cellLevel[static_cast<std::size_t>(own)]
                                + (marks.cellMidPoint[static_cast<std::size_t>(own)] >= 0 ? 1 : 0);
        const label newNeiLevel = newOwnLevel;
        if (newOwnLevel > marks.faceAnchorLevel[static_cast<std::size_t>(facei)]
         || newNeiLevel > marks.faceAnchorLevel[static_cast<std::size_t>(facei)])
        {
            marks.faceMidPoint[static_cast<std::size_t>(facei)] = 12345;
        }
    }

    // ---- 6. a point at each split face's centre (:3632-3686) --------------------------------------
    for (label facei = 0; facei < nFaces; ++facei)
    {
        if (marks.faceMidPoint[static_cast<std::size_t>(facei)] < 0) continue;
        // the master point is the face's FIRST vertex (:3677)
        const label master = m.faceVerts()[m.faceOffsets()[facei]];
        marks.faceMidPoint[static_cast<std::size_t>(facei)] =
            addPoint(a, (*v.faceCentres)[static_cast<std::size_t>(facei)], master, /*inCell=*/true);
        // ...and the mid point's level is one above the face's ANCHOR level, not its own max (:3683)
        at(marks.newPointLevel, marks.faceMidPoint[static_cast<std::size_t>(facei)]) =
            marks.faceAnchorLevel[static_cast<std::size_t>(facei)] + 1;
    }

    // ---- 7. the eight corner points of every refined cell (:3721-3766) ---------------------------
    // The outer loop is over POINTS ascending and the inner over that point's cells, so a cell's eight
    // anchors come out in ascending POINT order -- and the order pointCells comes in cannot change that.
    // (It matters elsewhere; see the note in primitive_patch_cpp.cuh on its three orders.)
    marks.cellAnchorPoints.assign(static_cast<std::size_t>(nCells), std::vector<label>());
    {
        std::vector<label> nAnchor(static_cast<std::size_t>(nCells), label(0));
        for (label celli = 0; celli < nCells; ++celli)
        {
            if (marks.cellMidPoint[static_cast<std::size_t>(celli)] >= 0)
            {
                marks.cellAnchorPoints[static_cast<std::size_t>(celli)].assign(8, label(-1));
            }
        }
        for (label pointi = 0; pointi < nPoints; ++pointi)
        {
            for (const label celli : (*v.pointCells)[static_cast<std::size_t>(pointi)])
            {
                if (marks.cellMidPoint[static_cast<std::size_t>(celli)] >= 0
                 && lv.pointLevel[static_cast<std::size_t>(pointi)]
                  <= lv.cellLevel[static_cast<std::size_t>(celli)])
                {
                    if (nAnchor[static_cast<std::size_t>(celli)] == 8)
                        throw std::runtime_error(
                            std::string(WHO) + "cell " + std::to_string(celli) + " has more than eight "
                            "points at or below its own level. OpenFOAM's own error: \"cell has more than "
                            "8 anchor points\" -- the cell is not a hex, or its levels are inconsistent.");
                    marks.cellAnchorPoints[static_cast<std::size_t>(celli)]
                        [static_cast<std::size_t>(nAnchor[static_cast<std::size_t>(celli)]++)] = pointi;
                }
            }
        }
        for (label celli = 0; celli < nCells; ++celli)
        {
            if (marks.cellMidPoint[static_cast<std::size_t>(celli)] >= 0
             && nAnchor[static_cast<std::size_t>(celli)] != 8)
                throw std::runtime_error(
                    std::string(WHO) + "cell " + std::to_string(celli) + " has "
                    + std::to_string(nAnchor[static_cast<std::size_t>(celli)]) + " anchor points, not 8. "
                    "OpenFOAM stops here too (:3757-3765): a cell to be split must be a hex whose eight "
                    "corners are at or below its own refinement level.");
        }
    }

    // ---- 8. the seven added cells, with the original at element 0 (:3810-3838) --------------------
    marks.cellAddedCells.assign(static_cast<std::size_t>(nCells), std::vector<label>());
    for (label celli = 0; celli < nCells; ++celli)
    {
        if (marks.cellAnchorPoints[static_cast<std::size_t>(celli)].size() != 8) continue;
        std::vector<label>& cAdded = marks.cellAddedCells[static_cast<std::size_t>(celli)];
        cAdded.assign(8, label(-1));
        cAdded[0] = celli;
        marks.newCellLevel[static_cast<std::size_t>(celli)] =
            lv.cellLevel[static_cast<std::size_t>(celli)] + 1;
        for (label i = 1; i < 8; ++i)
        {
            // a cell from a CELL: no point, edge or face master, so nothing lands in cellFrom* and
            // changeMesh's inflation refusal is not reached -- see the note in its header
            cAdded[static_cast<std::size_t>(i)] = addCell(a, -1, -1, -1, /*masterCellID=*/celli);
            at(marks.newCellLevel, cAdded[static_cast<std::size_t>(i)]) =
                lv.cellLevel[static_cast<std::size_t>(celli)] + 1;
        }
    }

    // ---- 12. the return value: per REQUESTED cell, its eight (:4302-4311) -------------------------
    std::vector<std::vector<label>> refinedCells(cellsToRefine.size());
    for (std::size_t i = 0; i < cellsToRefine.size(); ++i)
    {
        refinedCells[i] = marks.cellAddedCells[static_cast<std::size_t>(cellsToRefine[i])];
    }
    return refinedCells;
}

}   // namespace hexRef8
}   // namespace cpu
}   // namespace brae
