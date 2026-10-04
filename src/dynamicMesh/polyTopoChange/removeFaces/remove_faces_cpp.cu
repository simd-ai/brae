// removeFaces::compatibleRemoves. See remove_faces_cpp.cuh.
#include "remove_faces_cpp.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace removeFaces {

namespace {

constexpr const char* WHO = "brae removeFaces::compatibleRemoves: ";

// removeFaces::changeCellRegion (:56-78). OpenFOAM recurses over cellCells; this walks the same
// component with an explicit stack, which reaches the SAME SET of cells: the function assigns one
// fixed newRegion to the connected component of oldRegion that contains celli, so no traversal order
// can change which cells end up in it or what they are set to. The stack is used because the component
// is a merged cell region and OpenFOAM's own comment only hopes it is small ("Can be recursive since
// hopefully only small area of faces removed in one go").
void changeCellRegion(
    LabelListListRef                       cellCells,
    label                                  celli,
    label                                  oldRegion,
    label                                  newRegion,
    std::vector<label>&                    cellRegion)
{
    if (cellRegion[static_cast<std::size_t>(celli)] != oldRegion) return;
    std::vector<label> todo;
    todo.push_back(celli);
    while (!todo.empty())
    {
        const label c = todo.back();
        todo.pop_back();
        if (cellRegion[static_cast<std::size_t>(c)] != oldRegion) continue;
        cellRegion[static_cast<std::size_t>(c)] = newRegion;
        for (const label nbr : cellCells[static_cast<std::size_t>(c)])
        {
            if (cellRegion[static_cast<std::size_t>(nbr)] == oldRegion) todo.push_back(nbr);
        }
    }
}

}   // namespace

label compatibleRemoves(
    const PrimitiveMesh&                   m,
    LabelListListRef                       cellCells,
    const std::vector<label>&              facesToRemove,
    std::vector<label>&                    cellRegion,
    std::vector<label>&                    regionMaster,
    std::vector<label>&                    newFacesToRemove)
{
    if (cellCells.size() != static_cast<std::size_t>(m.nCells()))
        throw std::runtime_error(
            std::string(WHO) + "the cellCells addressing has "
            + std::to_string(cellCells.size()) + " rows, not the mesh's "
            + std::to_string(m.nCells()) + ". The region flood walks it, so a stale one would stop in "
            "the wrong place silently.");

    const std::vector<label>& faceOwner = m.owner();
    const std::vector<label>& faceNeighbour = m.neighbour();

    // :591-596
    cellRegion.assign(static_cast<std::size_t>(m.nCells()), label(-1));
    regionMaster.assign(static_cast<std::size_t>(m.nCells()), label(-1));

    label nRegions = 0;

    // :600-683. One pass over the requested faces, growing and merging regions.
    for (const label facei : facesToRemove)
    {
        if (facei >= m.nInternalFaces())
            throw std::runtime_error(
                std::string(WHO) + "face " + std::to_string(facei) + " is not internal (the mesh has "
                + std::to_string(m.nInternalFaces()) + " internal faces). OpenFOAM FatalErrors here "
                "too (:605-610): only an internal face has two cells to merge.");

        const label own = faceOwner[static_cast<std::size_t>(facei)];
        const label nei = faceNeighbour[static_cast<std::size_t>(facei)];

        const label region0 = cellRegion[static_cast<std::size_t>(own)];
        const label region1 = cellRegion[static_cast<std::size_t>(nei)];

        if (region0 == -1)
        {
            if (region1 == -1)
            {
                // a new region, and its master is the OWNER -- which is the lower-numbered of the two,
                // because that is what owning an internal face means
                cellRegion[static_cast<std::size_t>(own)] = nRegions;
                cellRegion[static_cast<std::size_t>(nei)] = nRegions;
                regionMaster[static_cast<std::size_t>(nRegions)] = own;
                ++nRegions;
            }
            else
            {
                cellRegion[static_cast<std::size_t>(own)] = region1;
                regionMaster[static_cast<std::size_t>(region1)] =
                    std::min(own, regionMaster[static_cast<std::size_t>(region1)]);
            }
        }
        else
        {
            if (region1 == -1)
            {
                // nei is higher numbered than own, so it cannot be below region0's master and the
                // master is left alone -- OpenFOAM says so in its own comment (:643-644)
                cellRegion[static_cast<std::size_t>(nei)] = region0;
            }
            else if (region0 != region1)
            {
                // both sides already have a region: keep the LOWER-numbered region and the lower master
                label freedRegion = -1;
                label keptRegion = -1;
                if (region0 < region1)
                {
                    changeCellRegion(cellCells, nei, region1, region0, cellRegion);
                    keptRegion = region0;
                    freedRegion = region1;
                }
                else if (region1 < region0)
                {
                    changeCellRegion(cellCells, own, region0, region1, cellRegion);
                    keptRegion = region1;
                    freedRegion = region0;
                }
                const label master0 = regionMaster[static_cast<std::size_t>(region0)];
                const label master1 = regionMaster[static_cast<std::size_t>(region1)];
                regionMaster[static_cast<std::size_t>(freedRegion)] = -1;
                regionMaster[static_cast<std::size_t>(keptRegion)] = std::min(master0, master1);
            }
        }
    }

    regionMaster.resize(static_cast<std::size_t>(nRegions));

    // :687-727. OpenFOAM's own two checks, and they are what make the incremental min() above sound
    // rather than hopeful: the master must be the lowest-numbered cell of its region, and a region of
    // one cell means a requested face did not merge anything.
    {
        std::vector<label> nCells(static_cast<std::size_t>(nRegions), label(0));
        for (std::size_t celli = 0; celli < cellRegion.size(); ++celli)
        {
            const label r = cellRegion[celli];
            if (r == -1) continue;
            ++nCells[static_cast<std::size_t>(r)];
            if (static_cast<label>(celli) < regionMaster[static_cast<std::size_t>(r)])
                throw std::runtime_error(
                    std::string(WHO) + "cell " + std::to_string(celli) + " is in region "
                    + std::to_string(r) + " whose master is "
                    + std::to_string(regionMaster[static_cast<std::size_t>(r)])
                    + ", so the master is not the lowest-numbered cell of the region. OpenFOAM "
                    "FatalErrors here too (:699-707).");
        }
        for (std::size_t region = 0; region < nCells.size(); ++region)
        {
            if (nCells[region] == 1)
                throw std::runtime_error(
                    std::string(WHO) + "region " + std::to_string(region) + " has only 1 cell in it. "
                    "OpenFOAM FatalErrors here too (:713-721).");
        }
    }

    // :731-737
    label nUsedRegions = 0;
    for (const label master : regionMaster)
    {
        if (master != -1) ++nUsedRegions;
    }

    // :740-755. THE RECOUNT, and it is a walk over the internal faces rather than over the input: any
    // internal face whose two cells ended in the same region has the same cell on both sides once the
    // merge is played, so it must go too. That also means the answer is in ASCENDING FACE ORDER and can
    // contain faces the caller never asked about.
    std::vector<label> allFacesToRemove;
    allFacesToRemove.reserve(facesToRemove.size());
    for (label facei = 0; facei < m.nInternalFaces(); ++facei)
    {
        const label own = faceOwner[static_cast<std::size_t>(facei)];
        const label nei = faceNeighbour[static_cast<std::size_t>(facei)];
        if (cellRegion[static_cast<std::size_t>(own)] != -1
            && cellRegion[static_cast<std::size_t>(own)] == cellRegion[static_cast<std::size_t>(nei)])
        {
            allFacesToRemove.push_back(facei);
        }
    }
    newFacesToRemove.swap(allFacesToRemove);

    return nUsedRegions;
}

// ----------------------------------------------------------------------------------------------
// setRefinement's decisions. See remove_faces_cpp.cuh.

namespace {

constexpr const char* WHOSET = "brae removeFaces::setRefinement: ";

void requireView(const RemoveFacesView& v)
{
    const std::pair<const void*, const char*> need[] = {
        {v.m,          "mesh"},
        {v.edges,      "edges (edges() and pointEdges())"},
        {v.faceEdges,  "faceEdges"},
        {v.edgeFaces,  "edgeFaces"},
        {v.cells,      "cells"},
        {v.pointFaces, "pointFaces"},
        {v.faceAreas,  "faceAreas"},
    };
    for (const auto& n : need)
    {
        if (!n.first)
            throw std::runtime_error(
                std::string(WHOSET) + "the mesh view has no " + n.second + ". Every field is required: a "
                "missing one is a null dereference inside a walk, which names nothing.");
    }
}

// which patch a boundary face belongs to -- polyBoundaryMesh::whichPatch
label whichPatch(
    const PrimitiveMesh& m,
    label                facei)
{
    const std::vector<PatchInfo>& patches = m.patches();
    for (std::size_t p = 0; p < patches.size(); ++p)
    {
        if (facei >= patches[p].start && facei < patches[p].start + patches[p].size)
            return static_cast<label>(p);
    }
    return -1;
}

// removeFaces::changeFaceRegion (:81-131). OpenFOAM recurses face-edge-face across edges that will be
// removed; this walks the same component with an explicit stack. It reaches the SAME SET and returns the
// same count: the region label written is fixed, and a face enters the component exactly once because
// the first thing done to it is to set its region. Gated by walking each edge list backwards, which is
// green (see tests/hex_ref8_vs_openfoam.sh).
label changeFaceRegion(
    const RemoveFacesView&    v,
    const std::vector<char>&  removedFace,
    const std::vector<label>& nFacesPerEdge,
    label                     startFacei,
    label                     newRegion,
    std::vector<label>&       faceRegion)
{
    label nChanged = 0;
    std::vector<label> todo;
    todo.push_back(startFacei);
    while (!todo.empty())
    {
        const label facei = todo.back();
        todo.pop_back();
        if (faceRegion[static_cast<std::size_t>(facei)] != -1) continue;
        if (removedFace[static_cast<std::size_t>(facei)]) continue;
        faceRegion[static_cast<std::size_t>(facei)] = newRegion;
        ++nChanged;
        for (const label edgeI : (*v.faceEdges)[static_cast<std::size_t>(facei)])
        {
            // an edge that will be removed, so the faces either side of it become one face
            if (nFacesPerEdge[static_cast<std::size_t>(edgeI)] >= 0
             && nFacesPerEdge[static_cast<std::size_t>(edgeI)] <= 2)
            {
                for (const label nbrFacei : (*v.edgeFaces)[static_cast<std::size_t>(edgeI)])
                {
                    if (faceRegion[static_cast<std::size_t>(nbrFacei)] == -1
                     && !removedFace[static_cast<std::size_t>(nbrFacei)])
                    {
                        todo.push_back(nbrFacei);
                    }
                }
            }
        }
    }
    return nChanged;
}

// removeFaces::getFacesAffected (:142-196). A mask, so the order the sets are walked in cannot matter.
std::vector<char> getFacesAffected(
    const RemoveFacesView&    v,
    const std::vector<label>& cellRegion,
    const std::vector<label>& cellRegionMaster,
    const std::vector<label>& facesToRemove,
    const std::vector<label>& edgesToRemove,
    const std::vector<label>& pointsToRemove)
{
    std::vector<char> affected(static_cast<std::size_t>(v.m->nFaces()), char(0));
    // the faces of every cell that is merged AWAY (the master keeps its own faces)
    for (std::size_t celli = 0; celli < cellRegion.size(); ++celli)
    {
        const label region = cellRegion[celli];
        if (region != -1
         && static_cast<label>(celli) != cellRegionMaster[static_cast<std::size_t>(region)])
        {
            for (const label facei : (*v.cells)[celli])
            {
                affected[static_cast<std::size_t>(facei)] = 1;
            }
        }
    }
    for (const label facei : facesToRemove) affected[static_cast<std::size_t>(facei)] = 1;
    for (const label edgei : edgesToRemove)
    {
        for (const label facei : (*v.edgeFaces)[static_cast<std::size_t>(edgei)])
        {
            affected[static_cast<std::size_t>(facei)] = 1;
        }
    }
    for (const label pointi : pointsToRemove)
    {
        for (const label facei : (*v.pointFaces)[static_cast<std::size_t>(pointi)])
        {
            affected[static_cast<std::size_t>(facei)] = 1;
        }
    }
    return affected;
}

// Foam::edge equality, which is UNORDERED (edgeI.H:499 -> Pair::compare, PairI.H): the pair (a,b) equals
// (b,a). The internal-internal filter below compares two REGION PAIRS with it, and an ordered comparison
// there would preserve edges OpenFOAM merges.
bool sameEdge(
    label a0,
    label a1,
    label b0,
    label b1)
{
    return (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0);
}

}   // namespace

RemoveFacesDecisions setRefinementDecisions(
    const RemoveFacesView&    v,
    const std::vector<label>& facesToRemove,
    const std::vector<label>& cellRegion,
    const std::vector<label>& cellRegionMaster,
    scalar                    minCos)
{
    requireView(v);
    const PrimitiveMesh& m = *v.m;
    const label nFaces = m.nFaces();
    const label nInternalFaces = m.nInternalFaces();
    const label nEdges = v.edges->nEdges();

    // THE REFUSAL, before anything is decided. setRefinement synchronises nFacesPerEdge with a max
    // (:1095-1101), nEdgesPerPoint with a max (:1312-1318) and checks the face regions agree across
    // every coupled patch (:1200-1255). In serial with no coupled patch each is over one value and is
    // skipped here; with one, they decide which edges and points go.
    for (const PatchInfo& p : m.patches())
    {
        if (p.type == "cyclic" || p.type == "cyclicAMI" || p.type == "cyclicACMI"
         || p.type == "cyclicPeriodicAMI" || p.type == "cyclicSlip"
         || p.type == "processor" || p.type == "processorCyclic")
            throw std::runtime_error(
                std::string(WHOSET) + "patch `" + p.name + "` is of coupled type `" + p.type
                + "`. removeFaces::setRefinement synchronises the edge and point usage counts across "
                "coupled patches and checks that both sides agree on every face region; brae skips all "
                "three because in serial with no coupled patch they are identities. No adaptive "
                "interFoam tutorial has a coupled patch.");
    }

    RemoveFacesDecisions out;

    // :779-790. Every face to remove must be internal.
    std::vector<char> removedFace(static_cast<std::size_t>(nFaces), char(0));
    for (const label facei : facesToRemove)
    {
        if (facei >= nInternalFaces)
            throw std::runtime_error(
                std::string(WHOSET) + "face to remove " + std::to_string(facei) + " is not internal (the "
                "mesh has " + std::to_string(nInternalFaces) + " internal faces). OpenFOAM FatalErrors "
                "here too (:786-791).");
        removedFace[static_cast<std::size_t>(facei)] = 1;
    }

    // ---- the edges ---------------------------------------------------------------------------------
    // :820-839. Count DOWN from edgeFaces().size()-1 at the first face of the edge that goes, then once
    // per further face: what is left is the number of faces that will still use the edge.
    std::vector<label> nFacesPerEdge(static_cast<std::size_t>(nEdges), label(-1));
    for (const label facei : facesToRemove)
    {
        for (const label edgeI : (*v.faceEdges)[static_cast<std::size_t>(facei)])
        {
            if (nFacesPerEdge[static_cast<std::size_t>(edgeI)] == -1)
            {
                nFacesPerEdge[static_cast<std::size_t>(edgeI)] =
                    static_cast<label>((*v.edgeFaces)[static_cast<std::size_t>(edgeI)].size()) - 1;
            }
            else
            {
                --nFacesPerEdge[static_cast<std::size_t>(edgeI)];
            }
        }
    }
    // :847-880. The edges no removed face uses: >2 faces means the edge stays whatever happens, exactly
    // 2 is LEFT AT -1 (OpenFOAM: "nFacesPerEdge already -1 so do nothing"), fewer is an error.
    for (label edgeI = 0; edgeI < nEdges; ++edgeI)
    {
        if (nFacesPerEdge[static_cast<std::size_t>(edgeI)] != -1) continue;
        const std::size_t nEFaces = (*v.edgeFaces)[static_cast<std::size_t>(edgeI)].size();
        if (nEFaces > 2)
        {
            nFacesPerEdge[static_cast<std::size_t>(edgeI)] = static_cast<label>(nEFaces);
        }
        else if (nEFaces < 2)
        {
            throw std::runtime_error(
                std::string(WHOSET) + "edge " + std::to_string(edgeI) + " has only "
                + std::to_string(nEFaces) + " faces using it. OpenFOAM FatalErrors here too (:867-878).");
        }
    }

    // :911-1032. The filter: an edge with exactly 2 remaining faces is a merge UNLESS one of three
    // things says otherwise, and each of the three puts the count back to 3 so the edge is preserved.
    for (label edgeI = 0; edgeI < nEdges; ++edgeI)
    {
        if (nFacesPerEdge[static_cast<std::size_t>(edgeI)] != 2) continue;
        label f0 = -1;
        label f1 = -1;
        for (const label facei : (*v.edgeFaces)[static_cast<std::size_t>(edgeI)])
        {
            if (removedFace[static_cast<std::size_t>(facei)]) continue;
            if (f0 == -1) { f0 = facei; }
            else          { f1 = facei; break; }
        }
        const bool b0 = (f0 >= nInternalFaces);
        const bool b1 = (f1 >= nInternalFaces);
        if (b0 && b1)
        {
            const label patch0 = whichPatch(m, f0);
            const label patch1 = whichPatch(m, f1);
            if (patch0 != patch1)
            {
                // never merge across a patch boundary. OpenFOAM warns and preserves (:986-994).
                nFacesPerEdge[static_cast<std::size_t>(edgeI)] = 3;
            }
            else if (minCos < scalar(1) && minCos > scalar(-1))
            {
                // the feature-angle guard, on the patch's own face NORMALS (area vector normalised)
                // polyPatch::faceNormals() is face::unitNormal -- the AREA normal normalised with
                // ROOTVSMALL as the guard (faceI.H:104-109), not a division by mag+SMALL
                const vector& a0 = (*v.faceAreas)[static_cast<std::size_t>(f0)];
                const vector& a1 = (*v.faceAreas)[static_cast<std::size_t>(f1)];
                const scalar m0 = mag(a0);
                const scalar m1 = mag(a1);
                const vector n0 = (m0 < scalar(1e-150)) ? vector{0, 0, 0} : a0/m0;
                const vector n1 = (m1 < scalar(1e-150)) ? vector{0, 0, 0} : a1/m1;
                if (fabs(dot(n0, n1)) < minCos)                 // OF: mag(n0 & n1)
                {
                    nFacesPerEdge[static_cast<std::size_t>(edgeI)] = 3;
                }
            }
        }
        else if (b0 != b1)
        {
            throw std::runtime_error(
                std::string(WHOSET) + "edge " + std::to_string(edgeI) + " would have one boundary face ("
                + std::to_string(b0 ? f0 : f1) + ") and one internal face (" + std::to_string(b0 ? f1 : f0)
                + ") using it. OpenFOAM FatalErrors here too (:1021-1037): the remove pattern is wrong.");
        }
        else
        {
            // both survivors are internal: merge only if they lie between the SAME pair of regions, as
            // an UNORDERED pair. Merging faces between different cell pairs would leave two faces
            // between one pair.
            const label o0 = cellRegion[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(f0)])];
            const label n0 =
                cellRegion[static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(f0)])];
            const label o1 = cellRegion[static_cast<std::size_t>(m.owner()[static_cast<std::size_t>(f1)])];
            const label n1 =
                cellRegion[static_cast<std::size_t>(m.neighbour()[static_cast<std::size_t>(f1)])];
            if (!sameEdge(o0, n0, o1, n1))
            {
                nFacesPerEdge[static_cast<std::size_t>(edgeI)] = 3;
            }
        }
    }

    // :1036-1052
    for (label edgeI = 0; edgeI < nEdges; ++edgeI)
    {
        if (nFacesPerEdge[static_cast<std::size_t>(edgeI)] == 1)
            throw std::runtime_error(
                std::string(WHOSET) + "edge " + std::to_string(edgeI) + " would get 1 face using it "
                "only. OpenFOAM FatalErrors here too (:1040-1051).");
    }

    // :1105-1121. 0 means the edge is gone, 2 means its two faces merge; both remove the edge.
    std::vector<label> edgesToRemove;
    for (label edgeI = 0; edgeI < nEdges; ++edgeI)
    {
        const label n = nFacesPerEdge[static_cast<std::size_t>(edgeI)];
        if (n == 0 || n == 2) edgesToRemove.push_back(edgeI);
    }

    // ---- the face regions -------------------------------------------------------------------------
    // :1177-1196. Walk face-edge-face across every edge that will be removed; each connected set of
    // faces becomes one region and will be merged into one face. A set of ONE face is marked -2: there
    // is nothing to merge, and the face is handled by the affected-face pass instead.
    std::vector<label> faceRegion(static_cast<std::size_t>(nFaces), label(-1));
    label nRegions = 0;
    label startFacei = 0;
    while (true)
    {
        for (; startFacei < nFaces; ++startFacei)
        {
            if (faceRegion[static_cast<std::size_t>(startFacei)] == -1
             && !removedFace[static_cast<std::size_t>(startFacei)])
            {
                break;
            }
        }
        if (startFacei == nFaces) break;
        const label nRegion =
            changeFaceRegion(v, removedFace, nFacesPerEdge, startFacei, nRegions, faceRegion);
        if (nRegion < 1)
            throw std::runtime_error(
                std::string(WHOSET) + "the face-region walk from face " + std::to_string(startFacei)
                + " changed nothing. OpenFOAM FatalErrors here too (:1183-1185).");
        if (nRegion == 1)
        {
            faceRegion[static_cast<std::size_t>(startFacei)] = -2;
        }
        else
        {
            ++nRegions;
        }
    }

    // ---- the points -------------------------------------------------------------------------------
    // :1277-1340. A point survives only as long as three or more unremoved edges meet at it: with two
    // it is a mid point of a merged edge and goes with it.
    std::vector<label> nEdgesPerPoint(static_cast<std::size_t>(m.nPoints()), label(0));
    for (std::size_t pointi = 0; pointi < nEdgesPerPoint.size(); ++pointi)
    {
        nEdgesPerPoint[pointi] = static_cast<label>(v.edges->pointEdges[pointi].size());
    }
    for (const label edgei : edgesToRemove)
    {
        --nEdgesPerPoint[static_cast<std::size_t>(v.edges->start[static_cast<std::size_t>(edgei)])];
        --nEdgesPerPoint[static_cast<std::size_t>(v.edges->end[static_cast<std::size_t>(edgei)])];
    }
    for (std::size_t pointi = 0; pointi < nEdgesPerPoint.size(); ++pointi)
    {
        if (nEdgesPerPoint[pointi] == 1)
            throw std::runtime_error(
                std::string(WHOSET) + "point " + std::to_string(pointi) + " would get 1 edge using it "
                "only. OpenFOAM FatalErrors here too (:1300-1308).");
    }
    std::vector<label> pointsToRemove;
    for (std::size_t pointi = 0; pointi < nEdgesPerPoint.size(); ++pointi)
    {
        if (nEdgesPerPoint[pointi] == 0 || nEdgesPerPoint[pointi] == 2)
        {
            pointsToRemove.push_back(static_cast<label>(pointi));
        }
    }

    // ---- what is touched at all -------------------------------------------------------------------
    std::vector<char> affectedFace =
        getFacesAffected(v, cellRegion, cellRegionMaster, facesToRemove, edgesToRemove, pointsToRemove);

    // :1427. invertOneToMany ignores every negative entry, so the -1s and the -2s drop out and each
    // region's faces come out in ASCENDING face order.
    std::vector<std::vector<label>> regionToFaces(static_cast<std::size_t>(nRegions));
    for (std::size_t facei = 0; facei < faceRegion.size(); ++facei)
    {
        const label r = faceRegion[facei];
        if (r >= 0) regionToFaces[static_cast<std::size_t>(r)].push_back(static_cast<label>(facei));
    }
    for (std::size_t r = 0; r < regionToFaces.size(); ++r)
    {
        if (regionToFaces[r].size() <= 1)
            throw std::runtime_error(
                std::string(WHOSET) + "face region " + std::to_string(r) + " contains "
                + std::to_string(regionToFaces[r].size()) + " faces. OpenFOAM FatalErrors here too "
                "(:1433-1439): a region of one face should have been marked -2.");
    }

    out.nFacesPerEdge = std::move(nFacesPerEdge);
    out.edgesToRemove = std::move(edgesToRemove);
    out.faceRegion = std::move(faceRegion);
    out.nFaceRegions = nRegions;
    out.pointsToRemove = std::move(pointsToRemove);
    out.affectedFace = std::move(affectedFace);
    out.regionToFaces = std::move(regionToFaces);
    return out;
}

// ----------------------------------------------------------------------------------------------
// setRefinement's actions. See remove_faces_cpp.cuh.

namespace {

// removeFaces::getFaceInfo (:416-441). Zones are refused at changeMesh, and a mesh with none answers
// zoneID -1 and zoneFlip false for every face -- which is what the actions carry.
label facePatch(
    const PrimitiveMesh& m,
    label                facei)
{
    return (facei < m.nInternalFaces()) ? label(-1) : whichPatch(m, facei);
}

// removeFaces::filterFace (:446-471). The face with every removed point dropped, in face order.
std::vector<label> filterFace(
    const PrimitiveMesh&     m,
    const std::vector<char>& removedPoint,
    label                    facei)
{
    const label n = m.faceSize(facei);
    const label off = m.faceOffsets()[static_cast<std::size_t>(facei)];
    std::vector<label> out;
    out.reserve(static_cast<std::size_t>(n));
    for (label i = 0; i < n; ++i)
    {
        const label v = m.faceVerts()[static_cast<std::size_t>(off + i)];
        if (!removedPoint[static_cast<std::size_t>(v)]) out.push_back(v);
    }
    return out;
}

// removeFaces::modFace (:475-560). A polyMesh face is owned by the LOWER of its two cells, so a face
// whose new owner is the higher one is written REVERSED with the pair swapped.
void modFace(
    polyTopoChange::TopoActions& a,
    const std::vector<label>&    f,
    label                        masterFaceID,
    label                        own,
    label                        nei,
    bool                         flipFaceFlux,
    label                        newPatchID)
{
    if (nei == -1 || own < nei)
    {
        polyTopoChange::modifyFace(a, f, masterFaceID, own, nei, flipFaceFlux, newPatchID);
    }
    else
    {
        // face::reverseFace: the first vertex stays and the rest run backwards
        std::vector<label> r(f.size());
        if (!f.empty())
        {
            r[0] = f[0];
            for (std::size_t i = 1; i < f.size(); ++i) r[i] = f[f.size() - i];
        }
        polyTopoChange::modifyFace(a, r, masterFaceID, nei, own, flipFaceFlux, newPatchID);
    }
}

// the cell a face's side becomes: its region's master where it has a region, itself otherwise
label mergedCell(
    const std::vector<label>& cellRegion,
    const std::vector<label>& cellRegionMaster,
    label                     celli)
{
    const label region = cellRegion[static_cast<std::size_t>(celli)];
    return (region == -1) ? celli : cellRegionMaster[static_cast<std::size_t>(region)];
}

}   // namespace

std::vector<MergeRecord> setRefinementActions(
    const RemoveFacesView&       v,
    const RemoveFacesDecisions&  dec,
    const std::vector<label>&    facesToRemove,
    const std::vector<label>&    cellRegion,
    const std::vector<label>&    cellRegionMaster,
    polyTopoChange::TopoActions& a)
{
    requireView(v);
    const PrimitiveMesh& m = *v.m;

    // affectedFace is CONSUMED as OpenFOAM consumes it: each pass clears the faces it has dealt with,
    // and the last pass picks up whatever is left. So it is a local copy, not the decisions' own.
    std::vector<char> affected = dec.affectedFace;
    std::vector<char> removedPoint(static_cast<std::size_t>(m.nPoints()), char(0));
    for (const label pointi : dec.pointsToRemove) removedPoint[static_cast<std::size_t>(pointi)] = 1;

    // :1385-1400. The split faces. OpenFOAM's own comment says the test is never false and is there to
    // be consistent with the passes below.
    for (const label facei : facesToRemove)
    {
        if (affected[static_cast<std::size_t>(facei)])
        {
            affected[static_cast<std::size_t>(facei)] = 0;
            polyTopoChange::removeFace(a, facei, -1);
        }
    }

    // :1404-1408
    for (const label pointi : dec.pointsToRemove)
    {
        polyTopoChange::removePoint(a, pointi, -1);
    }

    // :1411-1421. Every cell of a region except its master is removed INTO the master, which is what
    // makes the eight children one cell again.
    for (std::size_t celli = 0; celli < cellRegion.size(); ++celli)
    {
        const label region = cellRegion[celli];
        if (region == -1) continue;
        const label master = cellRegionMaster[static_cast<std::size_t>(region)];
        if (static_cast<label>(celli) != master)
        {
            polyTopoChange::removeCell(a, static_cast<label>(celli), master);
        }
    }

    // :1425-1470. One merge per face region, in region order.
    std::vector<MergeRecord> merges;
    merges.reserve(dec.regionToFaces.size());
    for (const std::vector<label>& rFaces : dec.regionToFaces)
    {
        // mergeFaces (:235-405). The patch is built over the region's faces IN THEIR OWN ORDER, which
        // is ascending face label, and every index below is an index into it.
        const PrimitivePatchAddressing fp = primitivePatch(m, rFaces);
        const PatchEdgeAddressing fpe = patchEdges(fp);
        const std::vector<std::vector<label>> loops = patchEdgeLoops(fpe);
        if (loops.size() != 1)
            throw std::runtime_error(
                std::string(WHOSET) + "the " + std::to_string(rFaces.size()) + " faces of a region do "
                "not have a single outside loop -- " + std::to_string(loops.size()) + " loops. OpenFOAM "
                "FatalErrors here too (mergeFaces, :259-267): they cannot be merged into one face.");
        const std::vector<label>& edgeLoop = loops[0];
        if (edgeLoop.size() < 2)
            throw std::runtime_error(
                std::string(WHOSET) + "a region's outside loop has " + std::to_string(edgeLoop.size())
                + " vertices. mergeFaces reads its first two to pick the master face.");

        // :273-307. The master is the face that uses the loop's first two vertices CONSECUTIVELY, and
        // the direction it uses them in decides whether the merged face is the loop or its reverse.
        label masterIndex = -1;
        bool reverseLoop = false;
        for (const label facei : fp.pointFaces[static_cast<std::size_t>(edgeLoop[0])])
        {
            const std::vector<label>& f = fp.localFaces[static_cast<std::size_t>(facei)];
            const auto it1 = std::find(f.begin(), f.end(), edgeLoop[1]);
            if (it1 == f.end()) continue;
            const auto it0 = std::find(f.begin(), f.end(), edgeLoop[0]);
            if (it0 == f.end()) continue;
            const std::size_t i0 = static_cast<std::size_t>(it0 - f.begin());
            const std::size_t i1 = static_cast<std::size_t>(it1 - f.begin());
            if (i1 == (i0 + 1) % f.size())              // face::fcIndex
            {
                masterIndex = facei;
                reverseLoop = false;
                break;
            }
            if (i1 == (i0 + f.size() - 1) % f.size())   // face::rcIndex
            {
                masterIndex = facei;
                reverseLoop = true;
                break;
            }
        }
        if (masterIndex == -1)
            throw std::runtime_error(
                std::string(WHOSET) + "no face of a merged region uses the first two vertices of its "
                "outside loop consecutively, so there is no master face. OpenFOAM FatalErrors here too "
                "(:310-316).");

        const label facei = rFaces[static_cast<std::size_t>(masterIndex)];
        const label own = mergedCell(cellRegion, cellRegionMaster, m.owner()[static_cast<std::size_t>(facei)]);
        const label patchID = facePatch(m, facei);
        label nei = -1;
        if (facei < m.nInternalFaces())
        {
            nei = mergedCell(cellRegion, cellRegionMaster,
                             m.neighbour()[static_cast<std::size_t>(facei)]);
        }

        // :348-380. The loop in MESH point labels with the removed points dropped, reversed if the
        // master ran the loop backwards.
        std::vector<label> mergedFace;
        mergedFace.reserve(edgeLoop.size());
        for (const label localPt : edgeLoop)
        {
            const label pointi = fp.meshPoints[static_cast<std::size_t>(localPt)];
            if (!removedPoint[static_cast<std::size_t>(pointi)]) mergedFace.push_back(pointi);
        }
        if (reverseLoop)
        {
            // Foam::reverse(face) reverses the WHOLE list, unlike face::reverseFace which keeps [0]
            std::reverse(mergedFace.begin(), mergedFace.end());
        }

        modFace(a, mergedFace, facei, own, nei, false, patchID);

        // :400-404. Every other face of the region is removed INTO the master.
        for (std::size_t i = 0; i < rFaces.size(); ++i)
        {
            if (static_cast<label>(i) != masterIndex)
            {
                polyTopoChange::removeFace(a, rFaces[i], facei);
            }
        }
        for (const label rf : rFaces) affected[static_cast<std::size_t>(rf)] = 0;

        MergeRecord rec;
        rec.masterFace = facei;
        rec.masterIndex = masterIndex;
        rec.reverseLoop = reverseLoop;
        rec.mergedFace = mergedFace;
        merges.push_back(std::move(rec));
    }

    // :1476-1519. Whatever is left: a face that keeps its identity but has lost points, or whose owner
    // or neighbour has been merged away.
    for (std::size_t facei = 0; facei < affected.size(); ++facei)
    {
        if (!affected[facei]) continue;
        affected[facei] = 0;
        const std::vector<label> f = filterFace(m, removedPoint, static_cast<label>(facei));
        const label own =
            mergedCell(cellRegion, cellRegionMaster, m.owner()[facei]);
        const label patchID = facePatch(m, static_cast<label>(facei));
        label nei = -1;
        if (static_cast<label>(facei) < m.nInternalFaces())
        {
            nei = mergedCell(cellRegion, cellRegionMaster, m.neighbour()[facei]);
        }
        modFace(a, f, static_cast<label>(facei), own, nei, false, patchID);
    }

    return merges;
}

} // namespace removeFaces
} // namespace cpu
} // namespace brae
