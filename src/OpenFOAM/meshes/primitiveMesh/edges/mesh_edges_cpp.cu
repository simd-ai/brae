#include "mesh_edges_cpp.cuh"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

// primitiveMeshEdges.C:41-78. Returns the edge between two points, creating it on first sight.
// The pointEdges entry is registered ONCE when a face repeats a vertex -- blockMesh does produce
// such a face, and registering it twice would put the same edge in the list twice.
label getEdge(
    std::vector<std::vector<label>>& pe,
    std::vector<label>&              esStart,
    std::vector<label>&              esEnd,
    label                            pointi,
    label                            nextPointi)
{
    for (const label edgei : pe[static_cast<std::size_t>(pointi)])
    {
        const std::size_t k = static_cast<std::size_t>(edgei);
        if (edgei < static_cast<label>(esStart.size())
         && (esStart[k] == nextPointi || esEnd[k] == nextPointi))
        {
            return edgei;
        }
    }
    const label edgei = static_cast<label>(esStart.size());
    pe[static_cast<std::size_t>(pointi)].push_back(edgei);
    if (nextPointi != pointi)
    {
        pe[static_cast<std::size_t>(nextPointi)].push_back(edgei);
    }
    esStart.push_back(std::min(pointi, nextPointi));
    esEnd.push_back(std::max(pointi, nextPointi));
    return edgei;
}


// primitiveMesh::calcPointOrder (primitiveMesh.C:246-315): are the points ordered -- every point used
// only by internal faces before every point on a boundary face? Only `ordered` and nInternalPoints are
// wanted here; the map it also builds is not used by calcEdges.
bool calcPointOrder(
    const PrimitiveMesh& m,
    label&               nInternalPoints)
{
    const label nPoints = m.nPoints();
    std::vector<label> oldToNew(static_cast<std::size_t>(nPoints), label(-1));

    label nBoundaryPoints = 0;
    for (label facei = m.nInternalFaces(); facei < m.nFaces(); ++facei)
    {
        for (label fp = 0; fp < m.faceSize(facei); ++fp)
        {
            const label pointi = m.faceVerts()[m.faceOffsets()[facei] + fp];
            if (oldToNew[static_cast<std::size_t>(pointi)] == -1)
            {
                oldToNew[static_cast<std::size_t>(pointi)] = nBoundaryPoints++;
            }
        }
    }
    nInternalPoints = nPoints - nBoundaryPoints;
    for (label p = 0; p < nPoints; ++p)
    {
        if (oldToNew[static_cast<std::size_t>(p)] != -1)
        {
            oldToNew[static_cast<std::size_t>(p)] += nInternalPoints;
        }
    }

    label internalPointi = 0;
    bool ordered = true;
    for (label facei = 0; facei < m.nInternalFaces(); ++facei)
    {
        for (label fp = 0; fp < m.faceSize(facei); ++fp)
        {
            const label pointi = m.faceVerts()[m.faceOffsets()[facei] + fp];
            if (oldToNew[static_cast<std::size_t>(pointi)] == -1)
            {
                if (pointi >= nInternalPoints)
                {
                    ordered = false;
                }
                oldToNew[static_cast<std::size_t>(pointi)] = internalPointi++;
            }
        }
    }
    return ordered;
}

}   // namespace


MeshEdges buildMeshEdges(const PrimitiveMesh& m)
{
    MeshEdges out;
    const label nPoints = m.nPoints();
    label nIntPts = -1;
    // WHICH BRANCH calcEdges TAKES. On a mesh whose points are ORDERED -- every point used only by
    // internal faces coming before every point on a boundary face -- OpenFOAM numbers the edges in FOUR
    // BLOCKS: both points internal, one internal, both on the boundary, then the external edges last
    // (primitiveMeshEdges.C:286-410). Otherwise it is one upper-triangular block.
    //
    // THIS BRANCH USED TO BE REFUSED HERE, on the stated grounds that "no mesh in this tree is ordered:
    // blockMesh numbers points geometrically ... and every fixture measured here reports nInternalPoints
    // -1". THAT WAS WRONG, and unit 5a's gate found it: `calcPointOrder` counts a point as a BOUNDARY
    // point if any boundary face uses it, and a 2D case's `empty` front and back patches cover the whole
    // domain -- so EVERY point is a boundary point, nBoundaryPoints == nPoints, nInternalPoints == 0, and
    // the second loop finds no unnumbered point at all, leaving `ordered` TRUE. MEASURED: laminar/damBreak
    // reports 0 of 4746 and LES/nozzleFlow2D 0 of 15276. The refused branch is the one every 2D fixture
    // in this tree takes, and the old comment had confused "nInternalPoints is -1" (the unordered
    // SENTINEL) with "there are no internal points" (0, which is ordered).
    const bool ordered = calcPointOrder(m, nIntPts);
    out.nInternalPoints = ordered ? nIntPts : label(-1);

    std::vector<std::vector<label>> pe(static_cast<std::size_t>(nPoints));
    std::vector<label> esStart;
    std::vector<label> esEnd;

    // Walk one face's consecutive point pairs, creating the edges it needs.
    auto walkFace = [&](label facei)
    {
        const label b = m.faceOffsets()[facei];
        const label n = m.faceSize(facei);
        for (label fp = 0; fp < n; ++fp)
        {
            getEdge(pe, esStart, esEnd,
                    m.faceVerts()[b + fp],
                    m.faceVerts()[b + ((fp + 1) % n)]);
        }
    };

    // THE CREATION ORDER IS PART OF THE ANSWER on the ordered branch, because the sort below asks
    // `edgeI < nExtEdges` to decide whether an edge is external -- and that is a PRE-SORT index. So
    // OpenFOAM does the BOUNDARY faces first (:182-205), which puts every external edge in 0..nExtEdges,
    // and only then the internal faces (:206-250), counting how many internal edges have 0 and 1
    // boundary points as it goes. The unordered branch needs none of that and walks faces in index
    // order, which is what this did for every mesh before.
    label nExtEdges = 0;
    label nInternal0Edges = 0;
    label nInt1Edges = 0;
    if (ordered)
    {
        for (label facei = m.nInternalFaces(); facei < m.nFaces(); ++facei)
        {
            walkFace(facei);
        }
        nExtEdges = static_cast<label>(esStart.size());
        for (label facei = 0; facei < m.nInternalFaces(); ++facei)
        {
            const label b = m.faceOffsets()[facei];
            const label n = m.faceSize(facei);
            for (label fp = 0; fp < n; ++fp)
            {
                const label pointi = m.faceVerts()[b + fp];
                const label nextPointi = m.faceVerts()[b + ((fp + 1) % n)];
                const std::size_t before = esStart.size();
                getEdge(pe, esStart, esEnd, pointi, nextPointi);
                if (esStart.size() > before)
                {
                    // a NEW internal edge: classify it by how many of its two points are internal
                    if (pointi < nIntPts)
                    {
                        if (nextPointi < nIntPts) ++nInternal0Edges;
                        else                      ++nInt1Edges;
                    }
                    else if (nextPointi < nIntPts)
                    {
                        ++nInt1Edges;
                    }
                    // else: an internal edge with BOTH points on the boundary, counted by neither
                }
            }
        }
    }
    else
    {
        for (label facei = 0; facei < m.nFaces(); ++facei)
        {
            walkFace(facei);
        }
    }

    const label nEdges = static_cast<label>(esStart.size());
    // :252-256
    const label nInternalEdges = nEdges - nExtEdges;
    const label nInternal1Edges = nInternal0Edges + nInt1Edges;

    // Like faces, sort the edges in order of increasing neighbouring point -- one
    // upper-triangular block, which is the branch an unordered mesh takes.
    std::vector<label> oldToNew(static_cast<std::size_t>(nEdges), label(-1));
    // FOUR RUNNING COUNTERS on the ordered branch (:291-302), one on the other. Their starts are what
    // put the blocks in OpenFOAM's order; the per-point upper-triangular walk below is shared.
    label internal0EdgeI = 0;
    label internal1EdgeI = nInternal0Edges;
    label internal2EdgeI = nInternal1Edges;
    label externalEdgeI = nInternalEdges;

    std::vector<label> nbrPoints;
    std::vector<label> order;
    for (label pointi = 0; pointi < nPoints; ++pointi)
    {
        const std::vector<label>& pEdges = pe[static_cast<std::size_t>(pointi)];
        nbrPoints.assign(pEdges.size(), label(-1));
        for (std::size_t i = 0; i < pEdges.size(); ++i)
        {
            const std::size_t k = static_cast<std::size_t>(pEdges[i]);
            const label nbr = (esStart[k] == pointi) ? esEnd[k] : esStart[k];
            nbrPoints[i] = (nbr < pointi) ? label(-1) : nbr;
        }
        // SortableList::sort() is a STABLE sort by value that keeps the original indices
        order.resize(nbrPoints.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](label a, label b)
                         {
                             return nbrPoints[static_cast<std::size_t>(a)]
                                  < nbrPoints[static_cast<std::size_t>(b)];
                         });
        for (std::size_t i = 0; i < order.size(); ++i)
        {
            const std::size_t oi = static_cast<std::size_t>(order[i]);
            if (nbrPoints[oi] == -1) continue;
            const label edgeI = pEdges[oi];
            if (!ordered)
            {
                oldToNew[static_cast<std::size_t>(edgeI)] = internal0EdgeI++;
                continue;
            }
            // :318-408. The EXTERNAL test comes first and is on the PRE-SORT index; only then does the
            // neighbour's own internal/boundary status choose between the two internal blocks. The two
            // halves of OpenFOAM's `pointi < nInternalPoints_` split are the SAME four lines except that
            // the boundary-point half FatalErrors on `nbrPointi < nInternalPoints_` -- which cannot
            // happen, because an edge from a boundary point to an internal point would have been found
            // from the internal point first, where it is the upper-triangular one. Kept as one branch
            // with that impossibility thrown rather than duplicated.
            if (edgeI < nExtEdges)
            {
                oldToNew[static_cast<std::size_t>(edgeI)] = externalEdgeI++;
            }
            else if (nbrPoints[oi] < nIntPts)
            {
                if (pointi >= nIntPts)
                    throw std::runtime_error(
                        "brae buildMeshEdges: internal edge " + std::to_string((long)edgeI)
                        + " runs from boundary point " + std::to_string((long)pointi)
                        + " to internal point " + std::to_string((long)nbrPoints[oi])
                        + ". OpenFOAM calls this \"Not possible!\" (primitiveMeshEdges.C:389-394): the "
                        "edge would have been reached from the internal point first.");
                oldToNew[static_cast<std::size_t>(edgeI)] = internal0EdgeI++;
            }
            else if (pointi < nIntPts)
            {
                oldToNew[static_cast<std::size_t>(edgeI)] = internal1EdgeI++;
            }
            else
            {
                oldToNew[static_cast<std::size_t>(edgeI)] = internal2EdgeI++;
            }
        }
    }

    out.start.assign(static_cast<std::size_t>(nEdges), label(0));
    out.end.assign(static_cast<std::size_t>(nEdges), label(0));
    for (label e = 0; e < nEdges; ++e)
    {
        const std::size_t k = static_cast<std::size_t>(e);
        const label n = oldToNew[k];
        if (n < 0)
        {
            throw std::runtime_error("brae buildMeshEdges: an edge was never renumbered.");
        }
        out.start[static_cast<std::size_t>(n)] = esStart[k];
        out.end[static_cast<std::size_t>(n)] = esEnd[k];
    }
    out.pointEdges.assign(static_cast<std::size_t>(nPoints), std::vector<label>());
    for (label p = 0; p < nPoints; ++p)
    {
        std::vector<label>& dst = out.pointEdges[static_cast<std::size_t>(p)];
        dst = pe[static_cast<std::size_t>(p)];
        for (label& e : dst) e = oldToNew[static_cast<std::size_t>(e)];
        std::sort(dst.begin(), dst.end());
    }
    return out;
}


// ----------------------------------------------------------------------------------------------
// UNIT 5a: faceEdges, edgeFaces, cellEdges. See the header for which of the three orders is the answer
// and which two are inert, both read off hexRef8's own call sites.

std::vector<std::vector<label>> buildFaceEdges(
    const PrimitiveMesh& m,
    const MeshEdges&     me)
{
    const label nFaces = m.nFaces();
    std::vector<std::vector<label>> out(static_cast<std::size_t>(nFaces));
    for (label facei = 0; facei < nFaces; ++facei)
    {
        const label b = m.faceOffsets()[facei];
        const label e = m.faceOffsets()[facei + 1];
        const label n = e - b;
        std::vector<label>& fe = out[static_cast<std::size_t>(facei)];
        fe.assign(static_cast<std::size_t>(n), label(-1));
        for (label fp = 0; fp < n; ++fp)
        {
            // f.fcIndex(fp): the next vertex, wrapping
            const label pointi = m.faceVerts()[b + fp];
            const label nextPointi = m.faceVerts()[b + ((fp + 1) % n)];
            // primitiveMeshEdges.C:562-574 -- scan the point's edges for the one whose OTHER vertex is
            // the next point, and take the FIRST such. There is exactly one on a valid mesh.
            for (const label edgei : me.pointEdges[static_cast<std::size_t>(pointi)])
            {
                const label s = me.start[static_cast<std::size_t>(edgei)];
                const label t = me.end[static_cast<std::size_t>(edgei)];
                const label other = (s == pointi) ? t : s;
                if (other == nextPointi)
                {
                    fe[static_cast<std::size_t>(fp)] = edgei;
                    break;
                }
            }
            if (fe[static_cast<std::size_t>(fp)] < 0)
                throw std::runtime_error(
                    "brae faceEdges: face " + std::to_string(facei) + " position "
                    + std::to_string(fp) + " has no edge between points " + std::to_string(pointi)
                    + " and " + std::to_string(nextPointi) + ". The edge list and the face list "
                    "disagree about the mesh.");
        }
    }
    return out;
}


std::vector<std::vector<label>> buildEdgeFaces(
    const PrimitiveMesh&                   m,
    const std::vector<std::vector<label>>& faceEdges)
{
    // invertManyToMany over faceEdges: walk the faces in INCREASING order and append, so every edge's
    // list comes out in ascending face index -- which is what the cached form gives and what the
    // on-demand form's comment relies on (primitiveMeshEdgeFaces.C:35-70, :72-110).
    std::size_t nEdges = 0;
    for (const auto& fe : faceEdges)
    {
        for (const label e : fe) nEdges = std::max(nEdges, static_cast<std::size_t>(e) + 1);
    }
    std::vector<std::vector<label>> out(nEdges);
    std::vector<label> count(nEdges, label(0));
    for (const auto& fe : faceEdges)
    {
        for (const label e : fe) ++count[static_cast<std::size_t>(e)];
    }
    for (std::size_t e = 0; e < nEdges; ++e) out[e].reserve(static_cast<std::size_t>(count[e]));
    for (std::size_t facei = 0; facei < faceEdges.size(); ++facei)
    {
        for (const label e : faceEdges[facei])
        {
            out[static_cast<std::size_t>(e)].push_back(static_cast<label>(facei));
        }
    }
    (void)m;
    return out;
}


std::vector<std::vector<label>> buildCellEdges(
    const std::vector<std::vector<label>>& cells,
    const std::vector<std::vector<label>>& faceEdges)
{
    std::vector<std::vector<label>> out(cells.size());
    for (std::size_t celli = 0; celli < cells.size(); ++celli)
    {
        std::vector<label>& ce = out[celli];
        for (const label facei : cells[celli])
        {
            const std::vector<label>& fe = faceEdges[static_cast<std::size_t>(facei)];
            ce.insert(ce.end(), fe.begin(), fe.end());
        }
        // SORTED and uniqued. OpenFOAM's on-demand form walks a labelHashSet instead, so its order is
        // bucket order; the only consumer MARKS per edge (hexRef8.C:3415-3433) and cannot tell. See the
        // header -- this is a deliberate departure with a measured reason, not a transcription.
        std::sort(ce.begin(), ce.end());
        ce.erase(std::unique(ce.begin(), ce.end()), ce.end());
    }
    return out;
}


} // namespace brae
