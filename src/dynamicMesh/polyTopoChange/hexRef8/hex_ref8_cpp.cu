// hex_ref8_cpp.cu -- see the header for what a split is, how the unit is divided and what is refused.
#include "hex_ref8_cpp.cuh"

#include "foam_dict.cuh"   // isCoupledInterfaceType

#include <algorithm>
#include <limits>
#include <map>
#include <utility>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace hexRef8 {

using polyTopoChange::addPoint;
using polyTopoChange::addCell;

namespace {

constexpr const char* WHO = "brae hexRef8::setRefinement: ";

// EVERY view pointer, named. This was a plain `if (!a || !b || ...) throw "incomplete"` over the eight
// the first stage needed, and unit 5b-2 added two more (faceEdges, edgeFaces) that it did not cover --
// so the second stage dereferenced a null and SEGFAULTED where a message would have named the field in
// one line. Found by gdb on the first run of the faces (of-debug: backtrace first).
void requireView(const MeshView& v)
{
    const std::pair<const void*, const char*> needed[] =
    {
        {static_cast<const void*>(v.m), "the mesh"},
        {static_cast<const void*>(v.edges), "edges()"},
        {static_cast<const void*>(v.faceEdges), "faceEdges()"},
        {static_cast<const void*>(v.edgeFaces), "edgeFaces()"},
        {static_cast<const void*>(v.cells), "cells()"},
        {static_cast<const void*>(v.cellPoints), "cellPoints()"},
        {static_cast<const void*>(v.pointCells), "pointCells()"},
        {static_cast<const void*>(v.cellEdges), "cellEdges()"},
        {static_cast<const void*>(v.cellCentres), "cellCentres()"},
        {static_cast<const void*>(v.faceCentres), "faceCentres()"},
    };
    for (const auto& n : needed)
    {
        if (!n.first)
            throw std::runtime_error(
                std::string(WHO) + "the mesh view has no " + n.second + ". Every field is required: a "
                "missing one is a caller that has not built that addressing.");
    }
}

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
    requireView(v);

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


// ----------------------------------------------------------------------------------------------
// UNIT 5b-2: SECTION 9, THE FACES. See the header for the four kinds and why their order matters.

namespace {

// face::fcIndex / rcIndex -- the next and previous position, wrapping
inline label fcIndex(label fp, std::size_t n) { return static_cast<label>((static_cast<std::size_t>(fp) + 1) % n); }
inline label rcIndex(label fp, std::size_t n) { return static_cast<label>((static_cast<std::size_t>(fp) + n - 1) % n); }

// face::flip() (face.H): the FIRST vertex stays and the rest reverse -- not a plain reverse. Every
// owner/neighbour swap below reorients the face this way, and getting it wrong inverts a flux.
void flipFace(std::vector<label>& f)
{
    if (f.size() > 2) std::reverse(f.begin() + 1, f.end());
}

std::vector<label> reverseFace(const std::vector<label>& f)
{
    std::vector<label> out(f);
    flipFace(out);
    return out;
}

// meshTools::findEdge(mesh, p0, p1): the edge between two points, or -1. Found through pointEdges, which
// is what OpenFOAM's own does.
label findEdge(
    const MeshEdges& me,
    label            p0,
    label            p1)
{
    for (const label edgei : me.pointEdges[static_cast<std::size_t>(p0)])
    {
        const label s = me.start[static_cast<std::size_t>(edgei)];
        const label t = me.end[static_cast<std::size_t>(edgei)];
        if ((s == p0 && t == p1) || (s == p1 && t == p0)) return edgei;
    }
    return -1;
}

// getFaceInfo (:104-129) reduced to what a zone-free mesh needs: the patch, or -1 inside. Zones are
// refused at changeMesh, so zoneID is -1 and zoneFlip false at every call -- which is why neither is
// carried here rather than being passed as constants through six functions.
label facePatch(
    const MeshView& v,
    label           facei)
{
    if (facei < v.m->nInternalFaces()) return -1;
    label acc = 0;
    for (std::size_t pi = 0; pi < v.m->patches().size(); ++pi)
    {
        const auto& p = v.m->patches()[pi];
        if (facei >= p.start && facei < p.start + p.size) return static_cast<label>(pi);
        ++acc;
    }
    (void)acc;
    throw std::runtime_error(
        std::string(WHO) + "boundary face " + std::to_string(facei) + " is in no patch.");
}

// hexRef8::addFace (:133-189). The face is added with the ORIGINAL as its master face, and owner and
// neighbour are ordered -- a boundary face (nei == -1) or own < nei goes as given, otherwise the pair is
// swapped AND the face reversed.
label addSplitFace(
    const MeshView&           v,
    TopoActions&              a,
    label                     facei,
    const std::vector<label>& newFace,
    label                     own,
    label                     nei)
{
    const label patchID = facePatch(v, facei);
    if (nei == -1 || own < nei)
    {
        return polyTopoChange::addFace(a, newFace, own, nei, -1, -1, /*masterFaceID=*/facei,
                                       /*flipFaceFlux=*/false, patchID);
    }
    return polyTopoChange::addFace(a, reverseFace(newFace), nei, own, -1, -1, facei, false, patchID);
}

// hexRef8::addInternalFace (:198-290). BOTH branches of OpenFOAM's are the same call -- its own comment
// on the boundary one is "For now create out of nothing", with the inflate-from-point alternative left
// commented out -- so the face carries NO master and is not mapped. That is what keeps changeMesh's
// inflation refusal out of reach on the refinement path.
label addNewInternalFace(
    TopoActions&              a,
    const std::vector<label>& newFace,
    label                     own,
    label                     nei)
{
    return polyTopoChange::addFace(a, newFace, own, nei, -1, -1, -1, false, -1);
}

// hexRef8::modFace (:294-354). Does NOTHING unless the owner, the neighbour or the vertex list actually
// changed -- so an untouched face produces no action at all, and the same owner/neighbour ordering rule
// as addFace applies when it does.
void modifySplitFace(
    const MeshView&           v,
    TopoActions&              a,
    label                     facei,
    const std::vector<label>& newFace,
    label                     own,
    label                     nei)
{
    const PrimitiveMesh& m = *v.m;
    const bool internal = (facei < m.nInternalFaces());
    const label b = m.faceOffsets()[facei];
    const label e = m.faceOffsets()[facei + 1];
    const bool sameVerts = (static_cast<label>(newFace.size()) == e - b)
        && std::equal(newFace.begin(), newFace.end(), m.faceVerts().begin() + b);
    if (own == m.owner()[facei]
     && (!internal || nei == m.neighbour()[facei])
     && sameVerts)
    {
        return;
    }
    const label patchID = facePatch(v, facei);
    if (nei == -1 || own < nei)
    {
        polyTopoChange::modifyFace(a, newFace, facei, own, nei, /*flipFaceFlux=*/false, patchID);
    }
    else
    {
        polyTopoChange::modifyFace(a, reverseFace(newFace), facei, nei, own, false, patchID);
    }
}

}   // namespace


label findLevel(
    const MeshView&           v,
    const Levels&             lv,
    label                     facei,
    const std::vector<label>& f,
    label                     startFp,
    bool                      searchForward,
    label                     wantedLevel)
{
    // :697-740
    label fp = startFp;
    for (std::size_t i = 0; i < f.size(); ++i)
    {
        const label pointi = f[static_cast<std::size_t>(fp)];
        if (lv.pointLevel[static_cast<std::size_t>(pointi)] < wantedLevel)
        {
            throw std::runtime_error(
                std::string(WHO) + "walking face " + std::to_string(facei) + " for level "
                + std::to_string(wantedLevel) + " met point " + std::to_string(pointi) + " at level "
                + std::to_string(lv.pointLevel[static_cast<std::size_t>(pointi)])
                + ", which is BELOW it. OpenFOAM FatalErrors here too (:712-724): the face is not the "
                "shape the caller assumed.");
        }
        if (lv.pointLevel[static_cast<std::size_t>(pointi)] == wantedLevel) return fp;
        fp = searchForward ? fcIndex(fp, f.size()) : rcIndex(fp, f.size());
    }
    throw std::runtime_error(
        std::string(WHO) + "face " + std::to_string(facei) + " has no point at level "
        + std::to_string(wantedLevel) + ". OpenFOAM FatalErrors here too (:737-739).");
}


label findMinLevel(
    const Levels&             lv,
    const std::vector<label>& f)
{
    // :745-759. STRICTLY less, so the FIRST vertex at the minimum wins.
    label minLevel = std::numeric_limits<label>::max();
    label minFp = -1;
    for (std::size_t fp = 0; fp < f.size(); ++fp)
    {
        const label level = lv.pointLevel[static_cast<std::size_t>(f[fp])];
        if (level < minLevel)
        {
            minLevel = level;
            minFp = static_cast<label>(fp);
        }
    }
    return minFp;
}


label getAnchorCell(
    const MeshView&                        v,
    const std::vector<std::vector<label>>& cellAnchorPoints,
    const std::vector<std::vector<label>>& cellAddedCells,
    label                                  celli,
    label                                  facei,
    label                                  pointi)
{
    // :540-593
    const std::vector<label>& anchors = cellAnchorPoints[static_cast<std::size_t>(celli)];
    if (anchors.empty()) return celli;                    // the cell was not split: it is its own child
    {
        const auto it = std::find(anchors.begin(), anchors.end(), pointi);
        if (it != anchors.end())
        {
            return cellAddedCells[static_cast<std::size_t>(celli)]
                   [static_cast<std::size_t>(it - anchors.begin())];
        }
    }
    // `pointi` is not one of the eight. An ALREADY-REFINED face reaches here, and OpenFOAM then looks for
    // any of the face's own vertices among the anchors instead (:557-567).
    const PrimitiveMesh& m = *v.m;
    for (label k = m.faceOffsets()[facei]; k < m.faceOffsets()[facei + 1]; ++k)
    {
        const auto it = std::find(anchors.begin(), anchors.end(), m.faceVerts()[k]);
        if (it != anchors.end())
        {
            return cellAddedCells[static_cast<std::size_t>(celli)]
                   [static_cast<std::size_t>(it - anchors.begin())];
        }
    }
    throw std::runtime_error(
        std::string(WHO) + "no anchor of split cell " + std::to_string(celli) + " is on face "
        + std::to_string(facei) + " (looking for point " + std::to_string(pointi)
        + "). OpenFOAM FatalErrors here too (:578-590).");
}


namespace {

// walkFaceToMid (:1464-1502). From an anchor at `startFp`, collect the one or two vertices up to where the
// face splits. Three exits: the next vertex is another anchor (the split point on the edge has already
// been appended), it is the mid level, or it is two levels up and the walk continues.
void walkFaceToMid(
    const MeshView&           v,
    const Levels&             lv,
    const std::vector<label>& edgeMidPoint,
    label                     cLevel,
    label                     facei,
    label                     startFp,
    std::vector<label>&       faceVerts)
{
    const PrimitiveMesh& m = *v.m;
    const label b = m.faceOffsets()[facei];
    const std::size_t n = static_cast<std::size_t>(m.faceOffsets()[facei + 1] - b);
    const std::vector<label>& fEdges = (*v.faceEdges)[static_cast<std::size_t>(facei)];
    label fp = startFp;
    while (true)
    {
        if (edgeMidPoint[static_cast<std::size_t>(fEdges[static_cast<std::size_t>(fp)])] >= 0)
        {
            faceVerts.push_back(edgeMidPoint[static_cast<std::size_t>(fEdges[static_cast<std::size_t>(fp)])]);
        }
        fp = fcIndex(fp, n);
        const label pointi = m.faceVerts()[b + fp];
        const label pl = lv.pointLevel[static_cast<std::size_t>(pointi)];
        if (pl <= cLevel) return;                       // next anchor
        if (pl == cLevel + 1) { faceVerts.push_back(pointi); return; }   // the mid
        if (pl == cLevel + 2) faceVerts.push_back(pointi);               // and keep going
    }
}

// walkFaceFromMid (:1513-1563). The same walk BACKWARD to the mid, then forward again from there
// collecting -- so the vertices come out in face order on the far side of the anchor.
void walkFaceFromMid(
    const MeshView&           v,
    const Levels&             lv,
    const std::vector<label>& edgeMidPoint,
    label                     cLevel,
    label                     facei,
    label                     startFp,
    std::vector<label>&       faceVerts)
{
    const PrimitiveMesh& m = *v.m;
    const label b = m.faceOffsets()[facei];
    const std::size_t n = static_cast<std::size_t>(m.faceOffsets()[facei + 1] - b);
    const std::vector<label>& fEdges = (*v.faceEdges)[static_cast<std::size_t>(facei)];
    label fp = rcIndex(startFp, n);
    while (true)
    {
        const label pl = lv.pointLevel[static_cast<std::size_t>(m.faceVerts()[b + fp])];
        if (pl <= cLevel) break;                                                   // anchor
        if (pl == cLevel + 1) { faceVerts.push_back(m.faceVerts()[b + fp]); break; }  // the mid
        // cLevel+2: keep walking back
        fp = rcIndex(fp, n);
    }
    while (true)
    {
        if (edgeMidPoint[static_cast<std::size_t>(fEdges[static_cast<std::size_t>(fp)])] >= 0)
        {
            faceVerts.push_back(edgeMidPoint[static_cast<std::size_t>(fEdges[static_cast<std::size_t>(fp)])]);
        }
        fp = fcIndex(fp, n);
        if (fp == startFp) break;
        faceVerts.push_back(m.faceVerts()[b + fp]);
    }
}

// insertEdgeSplit (:1568-1585). If the two points are both ORIGINAL points and the edge between them is
// being split, put the split point in. The `p < nPoints` guard is OpenFOAM's: a mid point added by this
// very refinement has no edge in the old mesh.
void insertEdgeSplit(
    const MeshView&           v,
    const std::vector<label>& edgeMidPoint,
    label                     p0,
    label                     p1,
    std::vector<label>&       verts)
{
    if (p0 < v.m->nPoints() && p1 < v.m->nPoints())
    {
        const label edgeI = findEdge(*v.edges, p0, p1);
        if (edgeI != -1 && edgeMidPoint[static_cast<std::size_t>(edgeI)] != -1)
        {
            verts.push_back(edgeMidPoint[static_cast<std::size_t>(edgeI)]);
        }
    }
}

// the two Map<edge> tables storeMidPointInfo accumulates into. std::map where OpenFOAM has a hash: only
// looked up and written by key, never iterated, so the order cannot reach the answer.
using MidEdgeMap = std::map<label, std::pair<label, label>>;

label otherVertex(const std::pair<label, label>& e, label v)
{
    return (e.first == v) ? e.second : e.first;
}

// storeMidPointInfo (:952-1178). ONE INTERNAL FACE PER EDGE between anchor points, and this is called
// from two to four times per such edge -- twice for two unrefined faces, up to four times for refined
// ones. Each call stores what it knows about the edge mid point: which anchor it sits between and which
// face mid points. THE CALL THAT COMPLETES THE PICTURE -- two anchors and two face mids, and which itself
// changed something -- builds the face. Every other call returns -1.
label storeMidPointInfo(
    const MeshView&                        v,
    const Levels&                          lv,
    const std::vector<std::vector<label>>& cellAnchorPoints,
    const std::vector<std::vector<label>>& cellAddedCells,
    const std::vector<label>&              cellMidPoint,
    const std::vector<label>&              edgeMidPoint,
    label                                  celli,
    label                                  facei,
    bool                                   faceOrder,
    label                                  edgeMidPointi,
    label                                  anchorPointi,
    label                                  faceMidPointi,
    MidEdgeMap&                            midPointToAnchors,
    MidEdgeMap&                            midPointToFaceMids,
    TopoActions&                           a)
{
    bool changed = false;
    bool haveTwoAnchors = false;
    {
        const auto it = midPointToAnchors.find(edgeMidPointi);
        if (it == midPointToAnchors.end())
        {
            // the FIRST insert does not count as a change: nothing is complete yet
            midPointToAnchors[edgeMidPointi] = {anchorPointi, label(-1)};
        }
        else
        {
            std::pair<label, label>& e = it->second;
            if (anchorPointi != e.first && e.second == -1)
            {
                e.second = anchorPointi;
                changed = true;
            }
            if (e.first != -1 && e.second != -1) haveTwoAnchors = true;
        }
    }
    bool haveTwoFaceMids = false;
    {
        const auto it = midPointToFaceMids.find(edgeMidPointi);
        if (it == midPointToFaceMids.end())
        {
            midPointToFaceMids[edgeMidPointi] = {faceMidPointi, label(-1)};
        }
        else
        {
            std::pair<label, label>& e = it->second;
            if (faceMidPointi != e.first && e.second == -1)
            {
                e.second = faceMidPointi;
                changed = true;
            }
            if (e.first != -1 && e.second != -1) haveTwoFaceMids = true;
        }
    }
    if (!(changed && haveTwoAnchors && haveTwoFaceMids)) return -1;

    const std::pair<label, label> anchors = midPointToAnchors[edgeMidPointi];
    const std::pair<label, label> faceMids = midPointToFaceMids[edgeMidPointi];
    const label otherFaceMidPointi = otherVertex(faceMids, faceMidPointi);

    // the face is built so that `anchorPointi`'s child is the OWNER, and the two edges between the edge
    // mid and the face mids may themselves be split -- but never between the cell mid and a face mid,
    // which is why insertEdgeSplit is called on those two pairs only (:1035-1078).
    std::vector<label> newFaceVerts;
    newFaceVerts.reserve(6);
    if (faceOrder == (v.m->owner()[facei] == celli))
    {
        newFaceVerts.push_back(faceMidPointi);
        insertEdgeSplit(v, edgeMidPoint, faceMidPointi, edgeMidPointi, newFaceVerts);
        newFaceVerts.push_back(edgeMidPointi);
        insertEdgeSplit(v, edgeMidPoint, edgeMidPointi, otherFaceMidPointi, newFaceVerts);
        newFaceVerts.push_back(otherFaceMidPointi);
        newFaceVerts.push_back(cellMidPoint[static_cast<std::size_t>(celli)]);
    }
    else
    {
        newFaceVerts.push_back(otherFaceMidPointi);
        insertEdgeSplit(v, edgeMidPoint, otherFaceMidPointi, edgeMidPointi, newFaceVerts);
        newFaceVerts.push_back(edgeMidPointi);
        insertEdgeSplit(v, edgeMidPoint, edgeMidPointi, faceMidPointi, newFaceVerts);
        newFaceVerts.push_back(faceMidPointi);
        newFaceVerts.push_back(cellMidPoint[static_cast<std::size_t>(celli)]);
    }

    const label anchorCell0 = getAnchorCell(v, cellAnchorPoints, cellAddedCells, celli, facei, anchorPointi);
    const label anchorCell1 = getAnchorCell(v, cellAnchorPoints, cellAddedCells, celli, facei,
                                            otherVertex(anchors, anchorPointi));
    label own = 0, nei = 0;
    if (anchorCell0 < anchorCell1)
    {
        own = anchorCell0;
        nei = anchorCell1;
    }
    else
    {
        own = anchorCell1;
        nei = anchorCell0;
        flipFace(newFaceVerts);
    }
    (void)lv;
    return addNewInternalFace(a, newFaceVerts, own, nei);
}

}   // namespace


namespace {

// createInternalFaces (:1182-1461). The TWELVE faces inside one split cell -- one per edge between two of
// its eight anchor points. It cannot build them directly, because those edges may themselves have been
// split; instead it walks each of the cell's faces, finds the cLevel+1 points and the anchors, and hands
// each (anchor, edge mid) pair to storeMidPointInfo, which builds a face once it has seen both sides.
void createInternalFaces(
    const MeshView&                        v,
    const Levels&                          lv,
    const std::vector<std::vector<label>>& cellAnchorPoints,
    const std::vector<std::vector<label>>& cellAddedCells,
    const std::vector<label>&              cellMidPoint,
    const std::vector<label>&              faceMidPoint,
    const std::vector<label>&              edgeMidPoint,
    label                                  celli,
    TopoActions&                           a)
{
    const PrimitiveMesh& m = *v.m;
    const std::vector<label>& cFaces = (*v.cells)[static_cast<std::size_t>(celli)];
    const label cLevel = lv.cellLevel[static_cast<std::size_t>(celli)];
    MidEdgeMap midPointToAnchors;
    MidEdgeMap midPointToFaceMids;
    label nFacesAdded = 0;

    for (const label facei : cFaces)
    {
        const label b = m.faceOffsets()[facei];
        const std::size_t n = static_cast<std::size_t>(m.faceOffsets()[facei + 1] - b);
        const std::vector<label> f(m.faceVerts().begin() + b, m.faceVerts().begin() + b + n);
        const std::vector<label>& fEdges = (*v.faceEdges)[static_cast<std::size_t>(facei)];

        // this cell's side of the face has either ONE anchor -- the other side was already split with
        // cLevel+1 and cLevel+2 points -- or FOUR, and nothing else is a hex (:1215-1277)
        label faceMidPointi = -1;
        const label nAnchors = countAnchors(lv, f, cLevel);
        if (nAnchors == 1)
        {
            label anchorFp = -1;
            for (std::size_t fp = 0; fp < n; ++fp)
            {
                if (lv.pointLevel[static_cast<std::size_t>(f[fp])] <= cLevel)
                {
                    anchorFp = static_cast<label>(fp);
                    break;
                }
            }
            // the face mid is the SECOND cLevel+1 point walking forward from the anchor
            const label edgeMid = findLevel(v, lv, facei, f, fcIndex(anchorFp, n), true, cLevel + 1);
            const label faceMid = findLevel(v, lv, facei, f, fcIndex(edgeMid, n), true, cLevel + 1);
            faceMidPointi = f[static_cast<std::size_t>(faceMid)];
        }
        else if (nAnchors == 4)
        {
            // no mid point on the face YET -- it is the one this refinement is about to add
            faceMidPointi = faceMidPoint[static_cast<std::size_t>(facei)];
        }
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "face " + std::to_string(facei) + " of split cell "
                + std::to_string(celli) + " has " + std::to_string(nAnchors) + " anchor points at level "
                + std::to_string(cLevel) + ", not 1 or 4. OpenFOAM FatalErrors here too (:1271-1277).");
        }

        // every anchor of this face, forward then backward, into storeMidPointInfo
        for (std::size_t fp0 = 0; fp0 < n; ++fp0)
        {
            const label point0 = f[fp0];
            if (lv.pointLevel[static_cast<std::size_t>(point0)] > cLevel) continue;

            // ---- forward: to the cLevel+1 point, or the split of this level's own edge
            label edgeMidPointi = -1;
            const label fp1 = fcIndex(static_cast<label>(fp0), n);
            if (lv.pointLevel[static_cast<std::size_t>(f[static_cast<std::size_t>(fp1)])] <= cLevel)
            {
                // two anchors in a row: the edge between them is the one being split
                edgeMidPointi = edgeMidPoint[static_cast<std::size_t>(fEdges[fp0])];
                if (edgeMidPointi == -1)
                    throw std::runtime_error(
                        std::string(WHO) + "the edge between two anchors of cell " + std::to_string(celli)
                        + " on face " + std::to_string(facei) + " was not split. OpenFOAM FatalErrors "
                        "here too (:1302-1315).");
            }
            else
            {
                const label edgeMid = findLevel(v, lv, facei, f, fp1, true, cLevel + 1);
                edgeMidPointi = f[static_cast<std::size_t>(edgeMid)];
            }
            if (storeMidPointInfo(v, lv, cellAnchorPoints, cellAddedCells, cellMidPoint, edgeMidPoint,
                                  celli, facei, /*faceOrder=*/true, edgeMidPointi, point0, faceMidPointi,
                                  midPointToAnchors, midPointToFaceMids, a) != -1)
            {
                if (++nFacesAdded == 12) break;
            }

            // ---- backward: the same, the other way round the anchor
            const label fpMin1 = rcIndex(static_cast<label>(fp0), n);
            if (lv.pointLevel[static_cast<std::size_t>(f[static_cast<std::size_t>(fpMin1)])] <= cLevel)
            {
                edgeMidPointi = edgeMidPoint[static_cast<std::size_t>(fEdges[static_cast<std::size_t>(fpMin1)])];
                if (edgeMidPointi == -1)
                    throw std::runtime_error(
                        std::string(WHO) + "the edge between two anchors of cell " + std::to_string(celli)
                        + " on face " + std::to_string(facei) + " was not split (walking back). OpenFOAM "
                        "FatalErrors here too (:1377-1390).");
            }
            else
            {
                const label edgeMid = findLevel(v, lv, facei, f, fpMin1, false, cLevel + 1);
                edgeMidPointi = f[static_cast<std::size_t>(edgeMid)];
            }
            if (storeMidPointInfo(v, lv, cellAnchorPoints, cellAddedCells, cellMidPoint, edgeMidPoint,
                                  celli, facei, /*faceOrder=*/false, edgeMidPointi, point0, faceMidPointi,
                                  midPointToAnchors, midPointToFaceMids, a) != -1)
            {
                if (++nFacesAdded == 12) break;
            }
        }
        if (nFacesAdded == 12) break;
    }
}

// getFaceNeighbours (:598-632): the owner's child and, on an internal face, the neighbour's child that
// own `pointi`. A boundary face has no neighbour.
void getFaceNeighbours(
    const MeshView&                        v,
    const std::vector<std::vector<label>>& cellAnchorPoints,
    const std::vector<std::vector<label>>& cellAddedCells,
    label                                  facei,
    label                                  pointi,
    label&                                 own,
    label&                                 nei)
{
    own = getAnchorCell(v, cellAnchorPoints, cellAddedCells, v.m->owner()[facei], facei, pointi);
    nei = (facei < v.m->nInternalFaces())
        ? getAnchorCell(v, cellAnchorPoints, cellAddedCells, v.m->neighbour()[facei], facei, pointi)
        : label(-1);
}

}   // namespace


void setRefinementFaces(
    const MeshView&        v,
    const Levels&          lv,
    const RefinementMarks& marks,
    TopoActions&           a)
{
    requireView(v);
    const PrimitiveMesh& m = *v.m;
    const label nFaces = m.nFaces();
    const label nCells = m.nCells();
    const std::vector<label>& cellMidPoint = marks.cellMidPoint;
    const std::vector<label>& edgeMidPoint = marks.edgeMidPoint;
    const std::vector<label>& faceMidPoint = marks.faceMidPoint;

    // ---- the bookkeeping (:3867-3896) -------------------------------------------------------------
    // every face of a split cell, every face being split, and every face on a split edge. Case 3 below
    // is whatever is still set after cases 1 and 2 clear their own, which is why the order matters.
    std::vector<char> affectedFace(static_cast<std::size_t>(nFaces), char(0));
    for (label celli = 0; celli < nCells; ++celli)
    {
        if (cellMidPoint[static_cast<std::size_t>(celli)] < 0) continue;
        for (const label facei : (*v.cells)[static_cast<std::size_t>(celli)])
        {
            affectedFace[static_cast<std::size_t>(facei)] = 1;
        }
    }
    for (label facei = 0; facei < nFaces; ++facei)
    {
        if (faceMidPoint[static_cast<std::size_t>(facei)] >= 0) affectedFace[static_cast<std::size_t>(facei)] = 1;
    }
    for (std::size_t edgeI = 0; edgeI < edgeMidPoint.size(); ++edgeI)
    {
        if (edgeMidPoint[edgeI] < 0) continue;
        for (const label facei : (*v.edgeFaces)[edgeI]) affectedFace[static_cast<std::size_t>(facei)] = 1;
    }

    // ---- 1. faces that GET SPLIT, into one per anchor (:3908-4010) --------------------------------
    for (label facei = 0; facei < nFaces; ++facei)
    {
        if (faceMidPoint[static_cast<std::size_t>(facei)] < 0
         || !affectedFace[static_cast<std::size_t>(facei)]) continue;
        const label b = m.faceOffsets()[facei];
        const std::size_t n = static_cast<std::size_t>(m.faceOffsets()[facei + 1] - b);
        const std::vector<label> f(m.faceVerts().begin() + b, m.faceVerts().begin() + b + n);
        const label anchorLevel = marks.faceAnchorLevel[static_cast<std::size_t>(facei)];
        // the ORIGINAL face is MODIFIED for the first anchor and three more are ADDED -- not four added
        bool modifiedFace = false;
        for (std::size_t fp = 0; fp < n; ++fp)
        {
            const label pointi = f[fp];
            if (lv.pointLevel[static_cast<std::size_t>(pointi)] > anchorLevel) continue;
            std::vector<label> faceVerts;
            faceVerts.reserve(6);
            faceVerts.push_back(pointi);
            walkFaceToMid(v, lv, edgeMidPoint, anchorLevel, facei, static_cast<label>(fp), faceVerts);
            faceVerts.push_back(faceMidPoint[static_cast<std::size_t>(facei)]);
            walkFaceFromMid(v, lv, edgeMidPoint, anchorLevel, facei, static_cast<label>(fp), faceVerts);
            label own = 0, nei = 0;
            getFaceNeighbours(v, marks.cellAnchorPoints, marks.cellAddedCells, facei, pointi, own, nei);
            if (!modifiedFace)
            {
                modifiedFace = true;
                modifySplitFace(v, a, facei, faceVerts, own, nei);
            }
            else
            {
                addSplitFace(v, a, facei, faceVerts, own, nei);
            }
        }
        affectedFace[static_cast<std::size_t>(facei)] = 0;
    }

    // ---- 2. faces that do NOT split but whose EDGES do (:4014-4142) -------------------------------
    // Walked per SPLIT EDGE and then over that edge's faces, which is OpenFOAM's order and not a walk
    // over faces -- so a face on two split edges is reached twice and the second visit finds it cleared.
    for (std::size_t edgeI = 0; edgeI < edgeMidPoint.size(); ++edgeI)
    {
        if (edgeMidPoint[edgeI] < 0) continue;
        for (const label facei : (*v.edgeFaces)[edgeI])
        {
            if (faceMidPoint[static_cast<std::size_t>(facei)] >= 0
             || !affectedFace[static_cast<std::size_t>(facei)]) continue;
            const label b = m.faceOffsets()[facei];
            const std::size_t n = static_cast<std::size_t>(m.faceOffsets()[facei + 1] - b);
            const std::vector<label> f(m.faceVerts().begin() + b, m.faceVerts().begin() + b + n);
            const std::vector<label>& fEdges = (*v.faceEdges)[static_cast<std::size_t>(facei)];
            std::vector<label> newFaceVerts;
            newFaceVerts.reserve(2*n);
            for (std::size_t fp = 0; fp < n; ++fp)
            {
                newFaceVerts.push_back(f[fp]);
                const label e = fEdges[fp];
                if (edgeMidPoint[static_cast<std::size_t>(e)] >= 0)
                {
                    newFaceVerts.push_back(edgeMidPoint[static_cast<std::size_t>(e)]);
                }
            }
            // the LOWEST-level point is an anchor of the neighbouring cells, so it names the new owner
            const label anchorFp = findMinLevel(lv, f);
            label own = 0, nei = 0;
            getFaceNeighbours(v, marks.cellAnchorPoints, marks.cellAddedCells, facei,
                              f[static_cast<std::size_t>(anchorFp)], own, nei);
            modifySplitFace(v, a, facei, newFaceVerts, own, nei);
            affectedFace[static_cast<std::size_t>(facei)] = 0;
        }
    }

    // ---- 3. faces that do not change shape but change OWNER or NEIGHBOUR (:4146-4172) -------------
    for (label facei = 0; facei < nFaces; ++facei)
    {
        if (!affectedFace[static_cast<std::size_t>(facei)]) continue;
        const label b = m.faceOffsets()[facei];
        const std::size_t n = static_cast<std::size_t>(m.faceOffsets()[facei + 1] - b);
        const std::vector<label> f(m.faceVerts().begin() + b, m.faceVerts().begin() + b + n);
        const label anchorFp = findMinLevel(lv, f);
        label own = 0, nei = 0;
        getFaceNeighbours(v, marks.cellAnchorPoints, marks.cellAddedCells, facei,
                          f[static_cast<std::size_t>(anchorFp)], own, nei);
        modifySplitFace(v, a, facei, f, own, nei);
        affectedFace[static_cast<std::size_t>(facei)] = 0;
    }

    // ---- 4. the twelve new internal faces of every split cell (:4176-4218) ------------------------
    for (label celli = 0; celli < nCells; ++celli)
    {
        if (cellMidPoint[static_cast<std::size_t>(celli)] < 0) continue;
        createInternalFaces(v, lv, marks.cellAnchorPoints, marks.cellAddedCells, cellMidPoint,
                            faceMidPoint, edgeMidPoint, celli, a);
    }
}


// ----------------------------------------------------------------------------------------------
// UNIT 6: hexRef8::updateMesh and the refinement history's producing side. See the header for which
// branch updateMesh takes and why the history is active even on an unrefined mesh.

namespace {

// hexRef8::reorder (:76-99): scatter `elems` through `map` into a list of `len`, filling the untouched
// entries with `null`. A target index at or beyond `len` is OpenFOAM's own FatalError.
void scatterThroughMap(
    const std::vector<label>& map,
    label                     len,
    label                     null,
    std::vector<label>&       elems)
{
    std::vector<label> out(static_cast<std::size_t>(len), null);
    for (std::size_t i = 0; i < elems.size(); ++i)
    {
        const label newI = map[i];
        if (newI >= len)
            throw std::runtime_error(
                std::string(WHO) + "remapping past the end: entry " + std::to_string(i) + " maps to "
                + std::to_string(newI) + " but the new list is " + std::to_string(len)
                + " long. OpenFOAM FatalErrors here too (hexRef8.C:86-93).");
        if (newI >= 0) out[static_cast<std::size_t>(newI)] = elems[i];
    }
    elems.swap(out);
}

}   // namespace


History freshHistory(label nCells)
{
    // refinementHistory.C:392-412
    History h;
    h.visibleCells.resize(static_cast<std::size_t>(nCells));
    h.parent.assign(static_cast<std::size_t>(nCells), label(-1));
    h.addedCells.assign(static_cast<std::size_t>(nCells), std::vector<label>());
    for (label c = 0; c < nCells; ++c) h.visibleCells[static_cast<std::size_t>(c)] = c;
    // active_ = returnReduceOr(visibleCells_.size()) -- true for any non-empty mesh, which is why an
    // unrefined mesh still carries a live history
    h.active = (nCells > 0);
    return h;
}


void resizeHistory(
    History& h,
    label    size)
{
    // :1043-1060 -- the ADDITIONAL entries are -1, i.e. not visible, and the existing ones are untouched
    const std::size_t oldSize = h.visibleCells.size();
    h.visibleCells.resize(static_cast<std::size_t>(size));
    for (std::size_t i = oldSize; i < h.visibleCells.size(); ++i) h.visibleCells[i] = -1;
}


label allocateSplitCell(
    History& h,
    label    parent,
    label    i)
{
    // :940-985. THE FREE LIST IS USED FROM THE BACK, and that is not a detail: it decides which index a
    // new split cell gets, and every `parent` and `visibleCells` entry is written in terms of indices.
    label index = -1;
    if (!h.freeSplitCells.empty())
    {
        index = h.freeSplitCells.back();
        h.freeSplitCells.pop_back();
        h.parent[static_cast<std::size_t>(index)] = parent;
        h.addedCells[static_cast<std::size_t>(index)].clear();
    }
    else
    {
        index = static_cast<label>(h.parent.size());
        h.parent.push_back(parent);
        h.addedCells.emplace_back();
    }
    if (parent >= 0)
    {
        std::vector<label>& pAdded = h.addedCells[static_cast<std::size_t>(parent)];
        if (pAdded.empty()) pAdded.assign(8, label(-1));   // FixedList<label,8>(-1) on first use
        pAdded[static_cast<std::size_t>(i)] = index;
    }
    return index;
}


void storeSplit(
    History&                  h,
    label                     celli,
    const std::vector<label>& addedCells)
{
    // :1000-1038
    label parentIndex = -1;
    if (h.visibleCells[static_cast<std::size_t>(celli)] != -1)
    {
        // it was live: its own split cell becomes the parent, and it stops being live -- then becomes
        // live again below as addedCells[0], which is how the original cell keeps its index
        parentIndex = h.visibleCells[static_cast<std::size_t>(celli)];
        h.visibleCells[static_cast<std::size_t>(celli)] = -1;
    }
    else
    {
        // a 0th-level entry, whose own parent is -1
        parentIndex = allocateSplitCell(h, -1, -1);
    }
    for (std::size_t i = 0; i < addedCells.size(); ++i)
    {
        h.visibleCells[static_cast<std::size_t>(addedCells[i])] =
            allocateSplitCell(h, parentIndex, static_cast<label>(i));
    }
}


void historyUpdateMesh(
    History&                  h,
    const std::vector<label>& reverseCellMap,
    label                     nNewCells)
{
    // :1063-1120. Only the LIVE cells are renumbered; a cell whose split entry already has children
    // being live is an inconsistency OpenFOAM stops on.
    if (!h.active) return;
    std::vector<label> newVisible(static_cast<std::size_t>(nNewCells), label(-1));
    for (std::size_t celli = 0; celli < h.visibleCells.size(); ++celli)
    {
        if (h.visibleCells[celli] == -1) continue;
        const label index = h.visibleCells[celli];
        if (!h.addedCells[static_cast<std::size_t>(index)].empty())
            throw std::runtime_error(
                std::string(WHO) + "live cell " + std::to_string(celli) + " has split entry "
                + std::to_string(index) + ", which already has children. OpenFOAM FatalErrors here too "
                "(refinementHistory.C:1080-1090).");
        const label newCelli = reverseCellMap[celli];
        if (newCelli >= 0) newVisible[static_cast<std::size_t>(newCelli)] = index;
    }
    h.visibleCells.swap(newVisible);
}


void updateLevels(
    Levels&                   lv,
    const std::vector<label>& reverseCellMap,
    const std::vector<label>& reversePointMap,
    const std::vector<label>& cellMap,
    const std::vector<label>& pointMap,
    label                     nNewCells,
    label                     nNewPoints)
{
    // :4370-4400 and :4455-4485, the two halves being the same shape. The REORDER branch is the one a
    // hexRef8 refinement takes -- see the header on why the sizes match -- and OpenFOAM's reason for
    // preferring it is in its own comment: gathering through cellMap would give a cell created from a
    // cell the level of the cell it was created from, which is a level too low.
    if (reverseCellMap.size() == lv.cellLevel.size())
    {
        scatterThroughMap(reverseCellMap, nNewCells, -1, lv.cellLevel);
    }
    else
    {
        std::vector<label> out(cellMap.size(), label(-1));
        for (std::size_t newCelli = 0; newCelli < cellMap.size(); ++newCelli)
        {
            const label oldCelli = cellMap[newCelli];
            out[newCelli] = (oldCelli == -1) ? label(-1)
                                             : lv.cellLevel[static_cast<std::size_t>(oldCelli)];
        }
        lv.cellLevel.swap(out);
    }
    if (reversePointMap.size() == lv.pointLevel.size())
    {
        scatterThroughMap(reversePointMap, nNewPoints, -1, lv.pointLevel);
    }
    else
    {
        std::vector<label> out(pointMap.size(), label(-1));
        for (std::size_t newPointi = 0; newPointi < pointMap.size(); ++newPointi)
        {
            const label oldPointi = pointMap[newPointi];
            out[newPointi] = (oldPointi == -1) ? label(-1)
                                               : lv.pointLevel[static_cast<std::size_t>(oldPointi)];
        }
        lv.pointLevel.swap(out);
    }
}


void storeRefinementHistory(
    History&                               h,
    const std::vector<std::vector<label>>& cellAddedCells,
    label                                  nCellsAfterSplit)
{
    // :4274-4300. The resize comes FIRST, over the cells the split added, because storeSplit writes
    // visibleCells at the added cells' own indices.
    if (!h.active) return;
    resizeHistory(h, nCellsAfterSplit);
    for (std::size_t celli = 0; celli < cellAddedCells.size(); ++celli)
    {
        if (cellAddedCells[celli].empty()) continue;
        storeSplit(h, static_cast<label>(celli), cellAddedCells[celli]);
    }
}


// ----------------------------------------------------------------------------------------------
// UNIT 6b-1: setUnrefinement's level and history half. See the header for the three-way split and for
// why this is the arm that finally gates unit 6's remapping.

void freeSplitCell(
    History& h,
    label    index)
{
    // :1607-1648
    const label parent = h.parent[static_cast<std::size_t>(index)];
    if (parent >= 0)
    {
        std::vector<label>& sub = h.addedCells[static_cast<std::size_t>(parent)];
        if (!sub.empty())
        {
            const auto it = std::find(sub.begin(), sub.end(), index);
            if (it == sub.end())
                throw std::runtime_error(
                    std::string(WHO) + "split cell " + std::to_string(index) + " is not among its "
                    "parent " + std::to_string(parent) + "'s children. OpenFOAM warns here too "
                    "(refinementHistory.C:1626-1635).");
            // a -1 IN PLACE, not an erase: the eight slots are positional and the position is the child
            // index storeSplit wrote it at
            *it = -1;
        }
    }
    h.parent[static_cast<std::size_t>(index)] = -2;      // the free marker, distinct from -1
    h.freeSplitCells.push_back(index);
}


void combineCells(
    History&                  h,
    label                     masterCelli,
    const std::vector<label>& combinedCells)
{
    // :1652-1673. The parent index is read BEFORE the children are freed, because freeing them rewrites
    // the parent's own addedCells.
    const label parentIndex =
        h.parent[static_cast<std::size_t>(h.visibleCells[static_cast<std::size_t>(masterCelli)])];
    for (const label celli : combinedCells)
    {
        freeSplitCell(h, h.visibleCells[static_cast<std::size_t>(celli)]);
        h.visibleCells[static_cast<std::size_t>(celli)] = -1;
    }
    // the parent's children pointer is RESET, which is an empty list here and not eight -1s
    h.addedCells[static_cast<std::size_t>(parentIndex)].clear();
    h.visibleCells[static_cast<std::size_t>(masterCelli)] = parentIndex;
}


void setUnrefinementLevels(
    const MeshView&           v,
    Levels&                   lv,
    History&                  h,
    const std::vector<label>& splitPointLabels)
{
    requireView(v);
    if (!h.active)
        throw std::runtime_error(
            std::string(WHO) + "setUnrefinement on a mesh with no active refinement history. OpenFOAM "
            "FatalErrors here too (hexRef8.C:5613-5620): without the history there is nothing recording "
            "which eight cells came from which parent.");
    for (const label pointi : splitPointLabels)
    {
        const std::vector<label>& pCells = (*v.pointCells)[static_cast<std::size_t>(pointi)];
        if (pCells.size() != 8)
            throw std::runtime_error(
                std::string(WHO) + "split point " + std::to_string(pointi) + " has "
                + std::to_string(pCells.size()) + " cells, not 8. OpenFOAM FatalErrors here too "
                "(:5712-5722): a point that can be unsplit is the centre of exactly one split.");
        const label masterCelli = *std::min_element(pCells.begin(), pCells.end());
        for (const label celli : pCells)
        {
            --lv.cellLevel[static_cast<std::size_t>(celli)];
        }
        combineCells(h, masterCelli, pCells);
    }
    // POINT LEVELS ARE UNTOUCHED, and that is OpenFOAM's own note at :5781-5783: the points "either get
    // removed or stay at the same position", so a surviving point's level is still the level it had.
}

// refinementHistory::compact. See the header.

namespace {

// markSplit (:1553-1585). Depth-first, parent before children, and each entry is appended the FIRST time
// it is reached -- which is what fixes the compacted numbering.
void markSplit(
    const History&            h,
    label                     index,
    std::vector<label>&       oldToNew,
    std::vector<label>&       newParent,
    std::vector<std::vector<label>>& newAdded,
    std::vector<label>&       newFromOld)
{
    if (oldToNew[static_cast<std::size_t>(index)] != -1) return;
    oldToNew[static_cast<std::size_t>(index)] = static_cast<label>(newParent.size());
    newParent.push_back(h.parent[static_cast<std::size_t>(index)]);
    newAdded.push_back(h.addedCells[static_cast<std::size_t>(index)]);
    newFromOld.push_back(index);
    const label parent = h.parent[static_cast<std::size_t>(index)];
    if (parent >= 0)
    {
        markSplit(h, parent, oldToNew, newParent, newAdded, newFromOld);
    }
    for (const label child : h.addedCells[static_cast<std::size_t>(index)])
    {
        if (child >= 0)
        {
            markSplit(h, child, oldToNew, newParent, newAdded, newFromOld);
        }
    }
}

}   // namespace

void compactHistory(History& h)
{
    const std::size_t nOld = h.parent.size();
    std::vector<label> oldToNew(nOld, label(-1));
    std::vector<label> newParent;
    std::vector<std::vector<label>> newAdded;
    std::vector<label> newFromOld;
    newParent.reserve(nOld);
    newAdded.reserve(nOld);

    // :1722-1742. From visibleCells, and only where the entry has a parent or children.
    for (const label index : h.visibleCells)
    {
        if (index < 0) continue;
        if (h.parent[static_cast<std::size_t>(index)] != -1
         || !h.addedCells[static_cast<std::size_t>(index)].empty())
        {
            markSplit(h, index, oldToNew, newParent, newAdded, newFromOld);
        }
    }
    // :1745-1766. Then from the split cells: a freed entry (-2) and a recombined one (no parent, no
    // children) are skipped -- either may already have been marked through someone else.
    for (std::size_t index = 0; index < nOld; ++index)
    {
        if (h.parent[index] == -2) continue;
        if (h.parent[index] == -1 && h.addedCells[index].empty()) continue;
        markSplit(h, static_cast<label>(index), oldToNew, newParent, newAdded, newFromOld);
    }

    // :1772-1792. Renumber the compacted entries' own parent and children through oldToNew.
    for (std::size_t i = 0; i < newParent.size(); ++i)
    {
        if (newParent[i] >= 0) newParent[i] = oldToNew[static_cast<std::size_t>(newParent[i])];
        for (label& child : newAdded[i])
        {
            if (child >= 0) child = oldToNew[static_cast<std::size_t>(child)];
        }
    }

    h.parent = std::move(newParent);
    h.addedCells = std::move(newAdded);
    h.freeSplitCells.clear();

    // :1825-1839. And visibleCells. oldToNew can be -1, which RESETS the entry -- OpenFOAM's own note.
    for (label& index : h.visibleCells)
    {
        if (index >= 0) index = oldToNew[static_cast<std::size_t>(index)];
    }
}

}   // namespace hexRef8
}   // namespace cpu
}   // namespace brae
