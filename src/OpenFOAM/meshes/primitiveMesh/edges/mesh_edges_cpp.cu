#include "mesh_edges_cpp.cuh"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

// primitiveMeshEdges.C:41-78. Returns the edge between two points, creating it on first sight.
// The pointEdges entry is registered ONCE when a face repeats a vertex -- blockMesh does produce
// such a face, and registering it twice would put the same edge in the list twice.
// THE POINTS' EDGES WHILE THEY ARE BEING FOUND: a chain a point, walked in the order the edges were met -- what
// a growing list a point held, without a heap block a point (the DynamicList<label> of primitiveMeshEdges.C:
// 228 is one too). An entry is (edge, the point's next entry).
struct PointEdgeChains
{
    std::vector<label> head;
    std::vector<label> tail;
    std::vector<label> count;
    std::vector<label> edge;
    std::vector<label> next;

    explicit PointEdgeChains(std::size_t nPoints)
    :
        head(nPoints, label(-1)),
        tail(nPoints, label(-1)),
        count(nPoints, label(0))
    {}

    void add(
        label pointi,
        label edgei)
    {
        const label entry = static_cast<label>(edge.size());
        edge.push_back(edgei);
        next.push_back(label(-1));
        const std::size_t p = static_cast<std::size_t>(pointi);
        if (tail[p] < 0)
        {
            head[p] = entry;
        }
        else
        {
            next[static_cast<std::size_t>(tail[p])] = entry;
        }
        tail[p] = entry;
        ++count[p];
    }
};

label getEdge(
    PointEdgeChains&    pe,
    std::vector<label>& esStart,
    std::vector<label>& esEnd,
    label               pointi,
    label               nextPointi)
{
    for (label at = pe.head[static_cast<std::size_t>(pointi)]; at >= 0; at = pe.next[static_cast<std::size_t>(at)])
    {
        const label edgei = pe.edge[static_cast<std::size_t>(at)];
        const std::size_t k = static_cast<std::size_t>(edgei);
        if (edgei < static_cast<label>(esStart.size())
         && (esStart[k] == nextPointi || esEnd[k] == nextPointi))
        {
            return edgei;
        }
    }
    const label edgei = static_cast<label>(esStart.size());
    pe.add(pointi, edgei);
    if (nextPointi != pointi)
    {
        pe.add(nextPointi, edgei);
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


MeshEdges buildMeshEdgesByChains(const PrimitiveMesh& m)
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

    PointEdgeChains pe(static_cast<std::size_t>(nPoints));
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
    std::vector<label> pEdges;
    for (label pointi = 0; pointi < nPoints; ++pointi)
    {
        pEdges.clear();
        for (label at = pe.head[static_cast<std::size_t>(pointi)]; at >= 0;
             at = pe.next[static_cast<std::size_t>(at)])
        {
            pEdges.push_back(pe.edge[static_cast<std::size_t>(at)]);
        }
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
    out.pointEdges.start(static_cast<std::size_t>(nPoints), pe.edge.size());
    for (label p = 0; p < nPoints; ++p)
    {
        for (label at = pe.head[static_cast<std::size_t>(p)]; at >= 0; at = pe.next[static_cast<std::size_t>(at)])
        {
            out.pointEdges.append(oldToNew[static_cast<std::size_t>(pe.edge[static_cast<std::size_t>(at)])]);
        }
        std::sort(out.pointEdges.openRowBegin(), out.pointEdges.openRowEnd());
        out.pointEdges.endRow();
    }
    return out;
}


namespace {

// THE BUCKET BUILD (see the header). A bucket entry is the pair's high point and whether the pair came off a
// BOUNDARY face, packed as 2*high + flag so that one sort orders by the high point and a run of equal high
// points can be folded into one edge that is external if any of its pairs was.
MeshEdges buildMeshEdgesByBuckets(const PrimitiveMesh& m)
{
    static const bool unsorted = std::getenv("BRAE_CONTROL_MESH_EDGES_UNSORTED") != nullptr;
    MeshEdges out;
    const std::size_t nPoints = static_cast<std::size_t>(m.nPoints());
    label nIntPts = -1;
    const bool ordered = calcPointOrder(m, nIntPts);
    out.nInternalPoints = ordered ? nIntPts : label(-1);
    const std::vector<label>& fv = m.faceVerts();
    const std::vector<label>& fo = m.faceOffsets();
    const label nFaces = m.nFaces();
    const label nIf = m.nInternalFaces();

    // the buckets' sizes, then their entries
    std::vector<label> bucket(nPoints + 1, label(0));
    for (label facei = 0; facei < nFaces; ++facei)
    {
        const label b = fo[static_cast<std::size_t>(facei)];
        const label n = fo[static_cast<std::size_t>(facei) + 1] - b;
        for (label fp = 0; fp < n; ++fp)
        {
            const label p = fv[static_cast<std::size_t>(b + fp)];
            const label q = fv[static_cast<std::size_t>(b + ((fp + 1) % n))];
            ++bucket[static_cast<std::size_t>(std::min(p, q)) + 1];
        }
    }
    for (std::size_t p = 0; p < nPoints; ++p)
    {
        bucket[p + 1] += bucket[p];
    }
    std::vector<label> entry(static_cast<std::size_t>(bucket[nPoints]));
    {
        std::vector<label> at(bucket.begin(), bucket.end() - 1);
        for (label facei = 0; facei < nFaces; ++facei)
        {
            // only an ordered mesh tells its external edges apart (primitiveMeshEdges.C:182-205)
            const label flag = (ordered && facei >= nIf) ? label(1) : label(0);
            const label b = fo[static_cast<std::size_t>(facei)];
            const label n = fo[static_cast<std::size_t>(facei) + 1] - b;
            for (label fp = 0; fp < n; ++fp)
            {
                const label p = fv[static_cast<std::size_t>(b + fp)];
                const label q = fv[static_cast<std::size_t>(b + ((fp + 1) % n))];
                entry[static_cast<std::size_t>(at[static_cast<std::size_t>(std::min(p, q))]++)] =
                    2*std::max(p, q) + flag;
            }
        }
    }

    // each bucket sorted and folded: the point's edges to its higher neighbours, ascending
    out.upperStart.assign(nPoints + 1, label(0));
    out.upperEnd.reserve(entry.size()/3);
    std::vector<char> external;
    external.reserve(entry.size()/3);
    // (a bucket holds each edge once a face on it -- four times on a hexahedral mesh -- and a point has three
    // higher neighbours or so: each entry is PLACED in the point's short ascending list, or found there, which
    // is the sort and the fold in one pass. MEASURED: 7.5 ms a build with std::sort on every bucket.)
    for (std::size_t p = 0; p < nPoints; ++p)
    {
        const std::size_t first = out.upperEnd.size();
        const label* const e = entry.data() + static_cast<std::size_t>(bucket[p + 1]);
        for (const label* k = entry.data() + static_cast<std::size_t>(bucket[p]); k != e; ++k)
        {
            const label hi = *k/2;
            const char flag = static_cast<char>(*k & 1);
            // where it goes among the neighbours found so far: after every smaller one
            std::size_t at = out.upperEnd.size();
            if (!unsorted)
            {
                while (at > first && out.upperEnd[at - 1] > hi)
                {
                    --at;
                }
            }
            else
            {
                at = first;
                while (at < out.upperEnd.size() && out.upperEnd[at] != hi)
                {
                    ++at;
                }
                if (at < out.upperEnd.size()) ++at;
            }
            if (at > first && out.upperEnd[at - 1] == hi)
            {
                external[at - 1] = static_cast<char>(external[at - 1] | flag);
                continue;
            }
            out.upperEnd.insert(out.upperEnd.begin() + static_cast<std::ptrdiff_t>(at), hi);
            external.insert(external.begin() + static_cast<std::ptrdiff_t>(at), flag);
        }
        out.upperStart[p + 1] = static_cast<label>(out.upperEnd.size());
    }
    const std::size_t nEdges = out.upperEnd.size();

    // the four blocks of an ordered mesh (primitiveMeshEdges.C:252-302); one block otherwise
    label nExtEdges = 0;
    label nInternal0Edges = 0;
    label nInt1Edges = 0;
    if (ordered)
    {
        for (std::size_t p = 0; p < nPoints; ++p)
        {
            for (label k = out.upperStart[p]; k < out.upperStart[p + 1]; ++k)
            {
                if (external[static_cast<std::size_t>(k)])
                {
                    ++nExtEdges;
                    continue;
                }
                // an internal edge, by how many of its two points are internal (:206-250)
                const bool lowIn = static_cast<label>(p) < nIntPts;
                const bool highIn = out.upperEnd[static_cast<std::size_t>(k)] < nIntPts;
                if (lowIn && highIn)
                {
                    ++nInternal0Edges;
                }
                else if (lowIn || highIn)
                {
                    ++nInt1Edges;
                }
            }
        }
    }
    label internal0EdgeI = 0;
    label internal1EdgeI = nInternal0Edges;
    label internal2EdgeI = nInternal0Edges + nInt1Edges;
    label externalEdgeI = static_cast<label>(nEdges) - nExtEdges;

    // the points ascending, each one's higher neighbours ascending: the walk that numbers the edges (:304-410)
    out.upperEdge.assign(nEdges, label(-1));
    out.start.assign(nEdges, label(0));
    out.end.assign(nEdges, label(0));
    for (std::size_t p = 0; p < nPoints; ++p)
    {
        const label pointi = static_cast<label>(p);
        for (label k = out.upperStart[p]; k < out.upperStart[p + 1]; ++k)
        {
            const std::size_t kk = static_cast<std::size_t>(k);
            const label nbr = out.upperEnd[kk];
            label edgei = -1;
            if (!ordered)
            {
                edgei = internal0EdgeI++;
            }
            else if (external[kk])
            {
                edgei = externalEdgeI++;
            }
            else if (nbr < nIntPts)
            {
                if (pointi >= nIntPts)
                    throw std::runtime_error(
                        "brae buildMeshEdges: an internal edge runs from boundary point "
                        + std::to_string((long)pointi) + " to internal point " + std::to_string((long)nbr)
                        + ". OpenFOAM calls this \"Not possible!\" (primitiveMeshEdges.C:389-394): the "
                        "edge would have been reached from the internal point first.");
                edgei = internal0EdgeI++;
            }
            else if (pointi < nIntPts)
            {
                edgei = internal1EdgeI++;
            }
            else
            {
                edgei = internal2EdgeI++;
            }
            out.upperEdge[kk] = edgei;
            out.start[static_cast<std::size_t>(edgei)] = pointi;
            out.end[static_cast<std::size_t>(edgei)] = nbr;
        }
    }

    // pointEdges: the edges walked in LABEL order and dropped on their two points, so every point's come out
    // ascending -- what the sort of each point's renumbered list gives. An edge from a point to itself (a face
    // that repeats a vertex) is the point's once.
    std::vector<label> degree(nPoints, label(0));
    for (std::size_t e = 0; e < nEdges; ++e)
    {
        ++degree[static_cast<std::size_t>(out.start[e])];
        if (out.end[e] != out.start[e]) ++degree[static_cast<std::size_t>(out.end[e])];
    }
    std::vector<label> at = out.pointEdges.setSizes(degree);
    std::vector<label>& pe = out.pointEdges.values();
    for (std::size_t e = 0; e < nEdges; ++e)
    {
        pe[static_cast<std::size_t>(at[static_cast<std::size_t>(out.start[e])]++)] = static_cast<label>(e);
        if (out.end[e] != out.start[e])
        {
            pe[static_cast<std::size_t>(at[static_cast<std::size_t>(out.end[e])]++)] = static_cast<label>(e);
        }
    }
    return out;
}

}   // namespace


MeshEdges buildMeshEdges(const PrimitiveMesh& m)
{
    static const bool chains = std::getenv("BRAE_CONTROL_MESH_EDGES_CHAINS") != nullptr;
    static const bool check = std::getenv("BRAE_CONTROL_MESH_EDGES_CHECK") != nullptr;
    // the packed entry is 2*point + flag in a label
    if (chains || m.nPoints() > label(0x3fffffff)) return buildMeshEdgesByChains(m);
    MeshEdges out = buildMeshEdgesByBuckets(m);
    if (check)
    {
        const MeshEdges want = buildMeshEdgesByChains(m);
        const char* what = out.start != want.start || out.end != want.end ? "the edges"
                         : out.pointEdges != want.pointEdges ? "pointEdges"
                         : out.nInternalPoints != want.nInternalPoints ? "nInternalPoints" : nullptr;
        if (what)
        {
            throw std::runtime_error(
                std::string("brae buildMeshEdges: BRAE_CONTROL_MESH_EDGES_CHECK: ") + what
                + " from the buckets are not the ones the chains give (" + std::to_string(out.start.size())
                + " edges against " + std::to_string(want.start.size()) + ").");
        }
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


CompactListList compactFaceEdges(
    const PrimitiveMesh& m,
    const MeshEdges&     me)
{
    // buildFaceEdges' rows, into an array shaped like faceVerts: entry (facei, fp) is the edge from the face's
    // vertex fp to its next one
    CompactListList out;
    out.setOffsets(m.faceOffsets());
    std::vector<label>& v = out.values();
    std::fill(v.begin(), v.end(), label(-1));
    const label nFaces = m.nFaces();
    // THE EDGE BETWEEN TWO POINTS THROUGH THE LOW POINT'S HIGHER NEIGHBOURS, where the edges carry them (the
    // bucket build's upperStart/upperEnd/upperEdge): three entries side by side on average, where the scan of
    // pointEdges below reads each candidate's two ends from the edge list. There is one edge between two points,
    // so "the first whose other vertex is the next point" is that one either way.
    // BRAE_CONTROL_FACE_EDGES_POINT_SCAN=1 scans pointEdges, as before.
    // BRAE_CONTROL_FACE_EDGES_NEXT_ENTRY=1 is a gate's CONTROL, deliberately wrong: the entry after the one
    // found, where the point has one -- its edge to another neighbour.
    static const bool pointScan = std::getenv("BRAE_CONTROL_FACE_EDGES_POINT_SCAN") != nullptr;
    static const bool nextEntry = std::getenv("BRAE_CONTROL_FACE_EDGES_NEXT_ENTRY") != nullptr;
    const bool upper = !pointScan && me.upperStart.size() == static_cast<std::size_t>(m.nPoints()) + 1
                    && me.upperEdge.size() == me.start.size();
    for (label facei = 0; facei < nFaces; ++facei)
    {
        const label b = m.faceOffsets()[facei];
        const label n = m.faceOffsets()[facei + 1] - b;
        for (label fp = 0; fp < n; ++fp)
        {
            const label pointi = m.faceVerts()[b + fp];
            const label nextPointi = m.faceVerts()[b + ((fp + 1) % n)];
            label& found = v[static_cast<std::size_t>(b + fp)];
            if (upper)
            {
                const label lo = std::min(pointi, nextPointi);
                const label hi = std::max(pointi, nextPointi);
                const label kEnd = me.upperStart[static_cast<std::size_t>(lo) + 1];
                for (label k = me.upperStart[static_cast<std::size_t>(lo)]; k < kEnd; ++k)
                {
                    if (me.upperEnd[static_cast<std::size_t>(k)] == hi)
                    {
                        const label taken = (nextEntry && k + 1 < kEnd) ? k + 1 : k;
                        found = me.upperEdge[static_cast<std::size_t>(taken)];
                        break;
                    }
                }
            }
            else
            {
                for (const label edgei : me.pointEdges[static_cast<std::size_t>(pointi)])
                {
                    const label s = me.start[static_cast<std::size_t>(edgei)];
                    const label t = me.end[static_cast<std::size_t>(edgei)];
                    const label other = (s == pointi) ? t : s;
                    if (other == nextPointi)
                    {
                        found = edgei;
                        break;
                    }
                }
            }
            if (found < 0)
                throw std::runtime_error(
                    "brae faceEdges: face " + std::to_string(facei) + " position "
                    + std::to_string(fp) + " has no edge between points " + std::to_string(pointi)
                    + " and " + std::to_string(nextPointi) + ". The edge list and the face list "
                    "disagree about the mesh.");
        }
    }
    return out;
}


CompactListList compactEdgeFaces(
    const PrimitiveMesh& m,
    LabelListListRef     faceEdges)
{
    // buildEdgeFaces' rows: an edge's faces in ascending face index
    std::size_t nEdges = 0;
    for (std::size_t facei = 0; facei < faceEdges.size(); ++facei)
    {
        for (const label e : faceEdges[facei]) nEdges = std::max(nEdges, static_cast<std::size_t>(e) + 1);
    }
    std::vector<label> count(nEdges, label(0));
    for (std::size_t facei = 0; facei < faceEdges.size(); ++facei)
    {
        for (const label e : faceEdges[facei]) ++count[static_cast<std::size_t>(e)];
    }
    CompactListList out;
    std::vector<label> at = out.setSizes(count);
    std::vector<label>& v = out.values();
    for (std::size_t facei = 0; facei < faceEdges.size(); ++facei)
    {
        for (const label e : faceEdges[facei])
        {
            v[static_cast<std::size_t>(at[static_cast<std::size_t>(e)]++)] = static_cast<label>(facei);
        }
    }
    (void)m;
    return out;
}


CompactListList compactCellEdges(
    LabelListListRef cells,
    LabelListListRef faceEdges,
    label            nEdges)
{
    // buildCellEdges' rows: the union of a cell's faces' edges, SORTED (the note in the header says why)
    CompactListList out;
    out.start(cells.size(), 12*cells.size());
    // AN EDGE IS TAKEN ONCE A CELL BY A MARK, and the cell's dozen then sorted -- where all its faces' edges
    // (two dozen on a hexahedron, every edge twice) were sorted and the doubles then dropped. The same set in
    // the same order. MEASURED on damBreakWithObstacle (92k cells), 2026-10-05: 7.4 ms a build the old way.
    static const bool sortUnique = std::getenv("BRAE_CONTROL_CELL_EDGES_SORT_UNIQUE") != nullptr;
    static const bool firstFace = std::getenv("BRAE_CONTROL_CELL_EDGES_FIRST_FACE") != nullptr;
    if (!sortUnique)
    {
        if (nEdges < 0)
        {
            for (std::size_t facei = 0; facei < faceEdges.size(); ++facei)
            {
                for (const label e : faceEdges[facei]) nEdges = std::max(nEdges, e + 1);
            }
        }
        std::vector<label> markedBy(static_cast<std::size_t>(std::max(nEdges, label(0))), label(-1));
        for (std::size_t celli = 0; celli < cells.size(); ++celli)
        {
            for (const label facei : cells[celli])
            {
                for (const label e : faceEdges[static_cast<std::size_t>(facei)])
                {
                    label& mark = markedBy[static_cast<std::size_t>(e)];
                    if (mark == static_cast<label>(celli) && !firstFace) continue;
                    mark = static_cast<label>(celli);
                    out.append(e);
                }
            }
            std::sort(out.openRowBegin(), out.openRowEnd());
            out.endRow();
        }
        return out;
    }
    for (std::size_t celli = 0; celli < cells.size(); ++celli)
    {
        for (const label facei : cells[celli])
        {
            for (const label e : faceEdges[static_cast<std::size_t>(facei)])
            {
                out.append(e);
            }
        }
        std::sort(out.openRowBegin(), out.openRowEnd());
        out.truncateOpenRow(std::unique(out.openRowBegin(), out.openRowEnd()));
        out.endRow();
    }
    return out;
}


} // namespace brae
