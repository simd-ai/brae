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
    // internal faces before every point on a boundary face -- OpenFOAM numbers the edges in four
    // blocks (both points internal, one internal, both on the boundary, then the external edges),
    // and the wave that walks them would visit them in that order. No mesh in this tree is ordered:
    // blockMesh numbers points geometrically, subsetMesh and renumberMesh leave the point order
    // alone, and every fixture measured here reports nInternalPoints -1. So that branch is refused
    // rather than transcribed and left untested -- an edge numbering nothing can check is a wrong
    // answer waiting for the first ordered mesh.
    if (calcPointOrder(m, nIntPts))
    {
        throw std::runtime_error(
            "brae buildMeshEdges: this mesh's points are ORDERED (" + std::to_string((long)nIntPts) +
            " of " + std::to_string((long)nPoints) + " used only by internal faces, and numbered "
            "first). OpenFOAM then sorts the edges into four blocks with the external edges last "
            "(primitiveMeshEdges.C:230-330), and PointEdgeWave's answer depends on that order. Only "
            "the unordered branch is ported.");
    }
    out.nInternalPoints = -1;

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

    for (label facei = 0; facei < m.nFaces(); ++facei)
    {
        walkFace(facei);
    }

    const label nEdges = static_cast<label>(esStart.size());

    // Like faces, sort the edges in order of increasing neighbouring point -- one
    // upper-triangular block, which is the branch an unordered mesh takes.
    std::vector<label> oldToNew(static_cast<std::size_t>(nEdges), label(-1));
    label internal0EdgeI = 0;

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
            oldToNew[static_cast<std::size_t>(pEdges[oi])] = internal0EdgeI++;
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

} // namespace brae
