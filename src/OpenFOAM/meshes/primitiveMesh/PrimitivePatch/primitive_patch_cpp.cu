#include "primitive_patch_cpp.cuh"
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace brae {

std::vector<std::vector<label>> meshCells(const PrimitiveMesh& m)
{
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<std::vector<label>> cellFaceAddr(static_cast<std::size_t>(m.nCells()));
    for (std::size_t facei = 0; facei < own.size(); ++facei)
    {
        cellFaceAddr[static_cast<std::size_t>(own[facei])].push_back(static_cast<label>(facei));
    }
    for (std::size_t facei = 0; facei < nei.size(); ++facei)
    {
        cellFaceAddr[static_cast<std::size_t>(nei[facei])].push_back(static_cast<label>(facei));
    }
    return cellFaceAddr;
}

std::vector<std::vector<label>> meshPointFaces(const PrimitiveMesh& m)
{
    std::vector<std::vector<label>> pf(static_cast<std::size_t>(m.nPoints()));
    for (label facei = 0; facei < m.nFaces(); ++facei)
    {
        for (label fp = 0; fp < m.faceSize(facei); ++fp)
        {
            pf[static_cast<std::size_t>(m.faceVert(facei, fp))].push_back(facei);
        }
    }
    return pf;
}

std::vector<std::vector<label>> pointCellsFromPointFaces(
    const PrimitiveMesh& m,
    const std::vector<std::vector<label>>& pointFaces)
{
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const label nIf = m.nInternalFaces();
    std::vector<std::vector<label>> pointCellAddr(pointFaces.size());
    std::vector<char> usedCells(static_cast<std::size_t>(m.nCells()), 0);
    for (std::size_t pointi = 0; pointi < pointFaces.size(); ++pointi)
    {
        std::vector<label>& currCells = pointCellAddr[pointi];
        for (const label facei : pointFaces[pointi])
        {
            // Owner cell - only allow one occurance
            const label o = own[static_cast<std::size_t>(facei)];
            if (!usedCells[static_cast<std::size_t>(o)])
            {
                usedCells[static_cast<std::size_t>(o)] = 1;
                currCells.push_back(o);
            }
            // Neighbour cell - only allow one occurance
            if (facei < nIf)
            {
                const label n = nei[static_cast<std::size_t>(facei)];
                if (!usedCells[static_cast<std::size_t>(n)])
                {
                    usedCells[static_cast<std::size_t>(n)] = 1;
                    currCells.push_back(n);
                }
            }
        }
        for (const label c : currCells)
        {
            usedCells[static_cast<std::size_t>(c)] = 0;
        }
    }
    return pointCellAddr;
}

std::vector<std::vector<label>> pointCellsFromCells(
    const PrimitiveMesh& m,
    const std::vector<std::vector<label>>& cells)
{
    // primitiveMeshPointCells.C:114-186. OpenFOAM counts first and fills second, with a `usedPoints`
    // marker cleared by walking the points it just set rather than by clearing the whole list. The
    // two passes exist to size each row exactly; the ORDER they produce is what matters here, and it
    // is ascending cell index because the outer loop is `for (celli = 0; celli < nCells; ++celli)`.
    const std::size_t nPoints = static_cast<std::size_t>(m.nPoints());
    std::vector<std::vector<label>> pointCells(nPoints);
    std::vector<char> usedPoints(nPoints, 0);
    std::vector<label> currPoints;
    for (std::size_t celli = 0; celli < cells.size(); ++celli)
    {
        for (const label p : currPoints)
        {
            usedPoints[static_cast<std::size_t>(p)] = 0;
        }
        currPoints.clear();
        for (const label facei : cells[celli])
        {
            const label n = m.faceSize(facei);
            for (label k = 0; k < n; ++k)
            {
                const label pointi = m.faceVert(facei, k);
                char& used = usedPoints[static_cast<std::size_t>(pointi)];
                if (!used)
                {
                    used = 1;
                    currPoints.push_back(pointi);
                    pointCells[static_cast<std::size_t>(pointi)].push_back(static_cast<label>(celli));
                }
            }
        }
    }
    return pointCells;
}


std::vector<std::vector<label>> cellPointsFromCells(
    const PrimitiveMesh& m,
    const std::vector<std::vector<label>>& cells)
{
    std::vector<std::vector<label>> cellPoints(cells.size());
    std::vector<char> usedPoints(static_cast<std::size_t>(m.nPoints()), 0);
    for (std::size_t celli = 0; celli < cells.size(); ++celli)
    {
        std::vector<label>& out = cellPoints[celli];
        for (const label facei : cells[celli])
        {
            const label n = m.faceSize(facei);
            for (label k = 0; k < n; ++k)
            {
                const label pointi = m.faceVert(facei, k);
                char& used = usedPoints[static_cast<std::size_t>(pointi)];
                if (!used)
                {
                    used = 1;
                    out.push_back(pointi);
                }
            }
        }
        for (const label p : out)
        {
            usedPoints[static_cast<std::size_t>(p)] = 0;
        }
    }
    return cellPoints;
}


PrimitivePatchAddressing primitivePatch(
    const PrimitiveMesh& m,
    const std::vector<label>& faces)
{
    PrimitivePatchAddressing pp;
    pp.faces = faces;
    std::unordered_map<label, label> markedPoints;
    markedPoints.reserve(4*faces.size());
    pp.localFaces.resize(faces.size());
    for (std::size_t i = 0; i < faces.size(); ++i)
    {
        const label f = faces[i];
        for (label fp = 0; fp < m.faceSize(f); ++fp)
        {
            const label pointi = m.faceVert(f, fp);
            const auto inserted = markedPoints.emplace(pointi, static_cast<label>(pp.meshPoints.size()));
            if (inserted.second)
            {
                pp.meshPoints.push_back(pointi);
            }
            pp.localFaces[i].push_back(inserted.first->second);
        }
    }
    pp.pointFaces.resize(pp.meshPoints.size());
    for (std::size_t i = 0; i < pp.localFaces.size(); ++i)
    {
        for (const label pointi : pp.localFaces[i])
        {
            pp.pointFaces[static_cast<std::size_t>(pointi)].push_back(static_cast<label>(i));
        }
    }
    return pp;
}

std::vector<label> faceRange(
    label start,
    label size)
{
    std::vector<label> f(static_cast<std::size_t>(size));
    for (label i = 0; i < size; ++i)
    {
        f[static_cast<std::size_t>(i)] = start + i;
    }
    return f;
}

} // namespace brae

// ----------------------------------------------------------------------------------------------
// The patch's edge addressing and its edge loops. See the header.

namespace brae {

PatchEdgeAddressing patchEdges(const PrimitivePatchAddressing& p)
{
    const std::size_t nFaces = p.localFaces.size();
    PatchEdgeAddressing out;
    out.faceEdges.resize(nFaces);
    // face::edges(): edge i of a face runs from f[i] to f[i+1], cyclically (faceI.H) -- so the edge
    // order is the face's own vertex order, which is what faceEdges is indexed by
    std::vector<std::vector<std::pair<label, label>>> faceIntoEdges(nFaces);
    for (std::size_t facei = 0; facei < nFaces; ++facei)
    {
        const std::vector<label>& f = p.localFaces[facei];
        faceIntoEdges[facei].resize(f.size());
        for (std::size_t i = 0; i < f.size(); ++i)
        {
            faceIntoEdges[facei][i] = {f[i], f[(i + 1) % f.size()]};
        }
        out.faceEdges[facei].assign(f.size(), label(-1));
    }

    const auto sameEdge = [](const std::pair<label, label>& a, const std::pair<label, label>& b)
    {
        return (a.first == b.first && a.second == b.second)
            || (a.first == b.second && a.second == b.first);
    };

    std::vector<std::pair<label, label>> edges;
    std::vector<std::vector<label>> edgeFaces;

    // :106-230. The INTERNAL edges: for each face, each of its not-yet-assigned edges is looked for
    // among the faces at its start point that have a HIGHER label, and the ones found are its
    // neighbours. Multiple connectivity is allowed, so a list per edge.
    for (std::size_t facei = 0; facei < nFaces; ++facei)
    {
        const std::vector<std::pair<label, label>>& curEdges = faceIntoEdges[facei];
        std::vector<std::vector<label>> neiFaces(curEdges.size());
        std::vector<std::vector<label>> edgeOfNeiFace(curEdges.size());
        label nNeighbours = 0;
        for (std::size_t edgeI = 0; edgeI < curEdges.size(); ++edgeI)
        {
            if (out.faceEdges[facei][edgeI] >= 0) continue;
            const std::pair<label, label>& e = curEdges[edgeI];
            bool found = false;
            for (const label curNei : p.pointFaces[static_cast<std::size_t>(e.first)])
            {
                // only the higher-numbered neighbour looks for the match, so each internal edge is
                // registered once, by its lower face
                if (curNei <= static_cast<label>(facei)) continue;
                const std::vector<std::pair<label, label>>& searchEdges =
                    faceIntoEdges[static_cast<std::size_t>(curNei)];
                for (std::size_t neiEdgeI = 0; neiEdgeI < searchEdges.size(); ++neiEdgeI)
                {
                    if (sameEdge(searchEdges[neiEdgeI], e))
                    {
                        found = true;
                        neiFaces[edgeI].push_back(curNei);
                        edgeOfNeiFace[edgeI].push_back(static_cast<label>(neiEdgeI));
                        // and keep searching: a multiply connected surface has more
                    }
                }
            }
            if (found) ++nNeighbours;
        }
        // :232-290. The face's internal edges are numbered in increasing order of their LOWEST
        // neighbour face, not in the face's own edge order.
        for (label neiSearch = 0; neiSearch < nNeighbours; ++neiSearch)
        {
            label nextNei = -1;
            label minNei = static_cast<label>(nFaces);
            for (std::size_t nfI = 0; nfI < neiFaces.size(); ++nfI)
            {
                if (!neiFaces[nfI].empty() && neiFaces[nfI][0] < minNei)
                {
                    nextNei = static_cast<label>(nfI);
                    minNei = neiFaces[nfI][0];
                }
            }
            if (nextNei < 0)
                throw std::runtime_error(
                    "brae PrimitivePatch::calcAddressing: internal edge insertion failed on face "
                    + std::to_string(facei) + ". OpenFOAM FatalErrors here too (:285-289).");
            const std::size_t ne = edges.size();
            edges.push_back(curEdges[static_cast<std::size_t>(nextNei)]);
            out.faceEdges[facei][static_cast<std::size_t>(nextNei)] = static_cast<label>(ne);
            std::vector<label>& cnf = neiFaces[static_cast<std::size_t>(nextNei)];
            std::vector<label>& eonf = edgeOfNeiFace[static_cast<std::size_t>(nextNei)];
            std::vector<label> curEf;
            curEf.reserve(cnf.size() + 1);
            curEf.push_back(static_cast<label>(facei));
            for (std::size_t i = 0; i < cnf.size(); ++i)
            {
                out.faceEdges[static_cast<std::size_t>(cnf[i])][static_cast<std::size_t>(eonf[i])] =
                    static_cast<label>(ne);
                curEf.push_back(cnf[i]);
            }
            edgeFaces.push_back(curEf);
            cnf.clear();
            eonf.clear();
        }
    }
    out.nInternalEdges = static_cast<label>(edges.size());

    // :296-312. Everything still unassigned is a boundary edge, in face-then-edge order.
    for (std::size_t facei = 0; facei < nFaces; ++facei)
    {
        for (std::size_t edgeI = 0; edgeI < out.faceEdges[facei].size(); ++edgeI)
        {
            if (out.faceEdges[facei][edgeI] >= 0) continue;
            const std::size_t ne = edges.size();
            edges.push_back(faceIntoEdges[facei][edgeI]);
            out.faceEdges[facei][edgeI] = static_cast<label>(ne);
            edgeFaces.push_back({static_cast<label>(facei)});
        }
    }

    out.start.resize(edges.size());
    out.end.resize(edges.size());
    for (std::size_t e = 0; e < edges.size(); ++e)
    {
        out.start[e] = edges[e].first;
        out.end[e] = edges[e].second;
    }
    out.edgeFaces = std::move(edgeFaces);

    // calcPointEdges: invertManyToMany over the edges, so each point's edges come out ASCENDING
    out.pointEdges.resize(p.meshPoints.size());
    for (std::size_t e = 0; e < out.start.size(); ++e)
    {
        out.pointEdges[static_cast<std::size_t>(out.start[e])].push_back(static_cast<label>(e));
        out.pointEdges[static_cast<std::size_t>(out.end[e])].push_back(static_cast<label>(e));
    }
    return out;
}

std::vector<std::vector<label>> patchEdgeLoops(const PatchEdgeAddressing& pe)
{
    // PrimitivePatchEdgeLoops.C:50-130. Walk point-edge-point over the BOUNDARY edges only, starting
    // from the first unvisited one and stepping to the first unvisited boundary edge at each vertex.
    const label nIntEdges = pe.nInternalEdges;
    const label nEdges = static_cast<label>(pe.start.size());
    const label nBdryEdges = nEdges - nIntEdges;
    std::vector<std::vector<label>> loops;
    if (nBdryEdges == 0) return loops;
    std::vector<char> unvisited(static_cast<std::size_t>(nBdryEdges), char(1));
    std::size_t searchFrom = 0;
    while (true)
    {
        // unvisited.find(true), which scans from the START each time -- but the scan can only ever
        // move forwards, so a cursor gives the same answer for less work
        while (searchFrom < unvisited.size() && !unvisited[searchFrom]) ++searchFrom;
        if (searchFrom == unvisited.size()) break;
        label currentEdgei = static_cast<label>(searchFrom) + nIntEdges;
        // the loop starts at the edge's FIRST vertex, which is what fixes the loop's direction
        label currentVerti = pe.start[static_cast<std::size_t>(currentEdgei)];
        std::vector<label> loop;
        do
        {
            loop.push_back(currentVerti);
            unvisited[static_cast<std::size_t>(currentEdgei - nIntEdges)] = 0;
            // edge::otherVertex
            currentVerti = (pe.start[static_cast<std::size_t>(currentEdgei)] == currentVerti)
                         ? pe.end[static_cast<std::size_t>(currentEdgei)]
                         : pe.start[static_cast<std::size_t>(currentEdgei)];
            currentEdgei = -1;
            for (const label edgei : pe.pointEdges[static_cast<std::size_t>(currentVerti)])
            {
                if (edgei >= nIntEdges && unvisited[static_cast<std::size_t>(edgei - nIntEdges)])
                {
                    currentEdgei = edgei;
                    break;
                }
            }
        }
        while (currentEdgei != -1);
        loops.push_back(std::move(loop));
    }
    return loops;
}

} // namespace brae
