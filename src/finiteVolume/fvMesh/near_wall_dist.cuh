#pragma once
// brae::nearWallDist, OpenFOAM turbulence.y(): the near-wall distance used by wall functions.
// Mirrors src/finiteVolume/fvMesh/wallDist/nearWallDist. For each wall-patch face with adjacent
// cell C, y = the smallest distance from C to the nearest point on any wall face that is a
// point-neighbour of the face, across ALL wall patches (the own face included). This differs from
// the per-face normal distance 1/deltaCoeffs at corners (e.g. the pitzDaily step), where a
// point-neighbour wall face is geometrically closer than the cell's own wall face.
//
// Distance to a face polygon follows OF face::nearestPoint: fan the polygon into triangles from
// its centre and take the min point-to-triangle distance. The triangle nearest point is exact
// (Ericson closest-point-on-triangle). The fan apex is OF's AREA-WEIGHTED centroid (face::centre),
// which matters on WARPED snappy faces (see pointToFaceDist), the earlier vertex-average apex gave
// too-small near-wall distances there, blowing up omegaWallFunction (omega_vis ~ 1/y^2).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <algorithm>
#include <thread>
#include <vector>
#include <unordered_map>
#include <utility>

namespace brae {

constexpr scalar nwdGreat = 1.0e15;   // OF VGREAT-style sentinel for the running minimum

// Closest point on triangle (a,b,c) to p (Ericson, Real-Time Collision Detection 5.1.5).
inline vector closestPointOnTriangle(
    const vector& p,
    const vector& a,
    const vector& b,
    const vector& c)
{
    const vector ab = b - a, ac = c - a, ap = p - a;
    const scalar d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;                              // vertex region a
    const vector bp = p - b;
    const scalar d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;                               // vertex region b
    const scalar vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + (d1 / (d1 - d3)) * ab;   // edge ab
    const vector cp = p - c;
    const scalar d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;                               // vertex region c
    const scalar vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + (d2 / (d2 - d6)) * ac;   // edge ac
    const scalar va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)                       // edge bc
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
    const scalar denom = 1.0 / (va + vb + vc);                        // interior
    return a + (vb * denom) * ab + (vc * denom) * ac;
}

// Distance from p to face polygon (CSR verts [v0,v1)), via OF centre-fan triangulation.
// The fan apex MUST be OF face::centre() = the AREA-WEIGHTED centroid (face.C::centre, sumAc/(3.sumA)), NOT the
// vertex average. They coincide on planar faces but DIVERGE on WARPED snappy faces, where the vertex-average fan
// has a triangle that tilts toward p and reports a too-small distance, which on omegaWallFunction (omega_vis ~ 1/y^2)
// blows omega up ~7x. With the area-weighted centroid cf's near-wall y is byte-identical to OF's meshWave (verified
// on motorBike cells 167636/175393). (OF: meshShapes/face/face.C::centre, faceIntersection.C::nearestPointClassify.)
inline scalar pointToFaceDist(
    const vector& p,
    const std::vector<vector>& pts,
    const std::vector<label>& fv,
    label v0,
    label v1)
{
    const label n = v1 - v0;
    vector cp{0.0, 0.0, 0.0};                                         // vertex average (the centroid's seed)
    for (label j = v0; j < v1; ++j)
        cp += pts[fv[j]];
    cp = (1.0 / n) * cp;
    scalar sumA = 0.0;
    vector sumAc{0.0, 0.0, 0.0};                   // OF face::centre area-weighted centroid
    for (label j = 0; j < n; ++j)
    {
        const vector& a = pts[fv[v0 + j]];
        const vector& b = pts[fv[v0 + (j + 1) % n]];
        const scalar ta = mag(cross(a - cp, b - cp));                 // 2x sub-triangle area
        sumA += ta;
        sumAc += ta * (a + b + cp);                       // ta x (3x sub-triangle centroid)
    }
    const vector ctr = (sumA > 1.0e-30) ? sumAc / (3.0 * sumA) : cp;
    scalar best = nwdGreat;
    for (label j = 0; j < n; ++j)
    {
        const vector& a = pts[fv[v0 + j]];
        const vector& b = pts[fv[v0 + (j + 1) % n]];
        const scalar d = mag(p - closestPointOnTriangle(p, ctr, a, b));
        if (d < best) best = d;
    }
    return best;
}

// Per-patch near-wall distance y (size patches.size(); non-wall patches left empty).
//
// THE SEARCH CROSSES PATCH BOUNDARIES. OF v2412's nearWallDist has two branches, chosen by
// cellDistFuncs::useCombinedWallPatch, which DEFAULTS TO TRUE: it gathers the faces of every wall patch
// into one uindirectPrimitivePatch and takes point-neighbours in that combined set. brae implemented the
// legacy per-patch branch, so a face whose closest wall neighbour lives on a DIFFERENT wall patch got its
// own (larger) distance instead.
//
// Where that bites is a corner cell shared by two wall patches, and a cyclicACMI makes one routinely: its
// non-overlap blockage is a wall patch coincident with the interface, so the cell at the END of the
// interface touches both the blockage and the ordinary side wall. Measured on
// pimpleFoam/RAS/oscillatingInletACMI2D, cell 3200 (first face of ACMI2_couple):
//
//     OF: y(ACMI2_blockage) = 5.20833e-03   <- the WALLS distance, not its own face's 1.25e-02
//     OF: y(walls)          = 5.20833e-03
//     brae (per-patch)      = 1.25e-02 and 5.20833e-03 respectively
//
// and cell 3280, one row in and touching only the blockage, agrees at 1.25e-02 in both. Through
// epsilon0 = invNw*Cmu^.75*k^1.5/(kappa*y) that made epsilon at those cells 0.71x and 0.90x OpenFOAM's
// while the median interface cell was already right to 1.7e-07.
inline std::vector<std::vector<scalar>> nearWallDist(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const std::vector<vector>& pts = m.points();
    const std::vector<label>&  fv  = m.faceVerts();
    const std::vector<label>&  fo  = m.faceOffsets();
    const std::vector<vector>& C   = g.C();

    // One combined wall patch: the global face index of every wall face, plus where it came from.
    std::vector<label> wallFace;                     // global face index
    std::vector<std::pair<std::size_t, label>> from; // (patch, local index)
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& wp = patches[pi];
        if (wp.type != "wall") continue;
        for (label i = 0; i < wp.size; ++i)
        {
            wallFace.push_back(wp.start + i);
            from.push_back({pi, i});
        }
    }

    std::vector<std::vector<scalar>> y(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (patches[pi].type == "wall" && patches[pi].size > 0)
            y[pi].assign(patches[pi].size, nwdGreat);
    if (wallFace.empty()) return y;

    // point -> combined-wall-face indices
    std::unordered_map<label, std::vector<label>> pointFaces;
    for (std::size_t w = 0; w < wallFace.size(); ++w)
    {
        const label f = wallFace[w];
        for (label j = fo[f]; j < fo[f + 1]; ++j)
            pointFaces[fv[j]].push_back(static_cast<label>(w));
    }

    for (std::size_t w = 0; w < wallFace.size(); ++w)
    {
        const std::size_t pi = from[w].first;
        const label i = from[w].second;
        const vector& Cc = C[patches[pi].faceCells[i]];
        const label f = wallFace[w];
        // candidates: this face + every wall face sharing a vertex, on ANY wall patch
        scalar best = pointToFaceDist(Cc, pts, fv, fo[f], fo[f + 1]);
        for (label j = fo[f]; j < fo[f + 1]; ++j)
            for (label nw : pointFaces[fv[j]])
            {
                if (static_cast<std::size_t>(nw) == w) continue;
                const label gf = wallFace[nw];
                const scalar d = pointToFaceDist(Cc, pts, fv, fo[gf], fo[gf + 1]);
                if (d < best) best = d;
            }
        y[pi][i] = best;
    }
    return y;
}

// nearWallDist WITH ITS NEIGHBOUR LISTS KEPT, ON THE HOST'S THREADS: what a mesh that moves calls at every move
// (refreshDeviceInterTurbulenceGeometry). nearWallDist above builds, at every call, the map from a point to the
// wall faces on it, and then measures each wall face's cell centre against the face and every wall face sharing
// a vertex -- a face sharing two vertices twice. A mesh that moves keeps its faces, so which wall faces
// neighbour which is kept here from one call to the next (NearWallKept), each neighbour once, and the distances
// are measured by the host's threads. The minimum of the same distances is the same number whatever the order
// and however often one is repeated, so y is nearWallDist's to the bit.
// `selfOnly` is a gate's CONTROL, deliberately wrong: a face is measured against itself alone.
struct NearWallKept
{
    std::vector<label> wallFace;     // the wall faces the lists were built from, and their vertices
    std::vector<label> verts;
    std::vector<label> start;        // wall face -> its candidates: itself, then each vertex neighbour once
    std::vector<label> cand;
};

inline std::vector<std::vector<scalar>> nearWallDistKept(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    NearWallKept& kept,
    unsigned nThreads,
    bool selfOnly = false)
{
    const std::vector<vector>& pts = m.points();
    const std::vector<label>& fv = m.faceVerts();
    const std::vector<label>& fo = m.faceOffsets();
    const std::vector<vector>& C = g.C();

    std::vector<label> wallFace;
    std::vector<std::pair<std::size_t, label>> from;
    std::vector<label> verts;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& wp = patches[pi];
        if (wp.type != "wall") continue;
        for (label i = 0; i < wp.size; ++i)
        {
            const label f = wp.start + i;
            wallFace.push_back(f);
            from.push_back({pi, i});
            verts.insert(verts.end(), fv.begin() + fo[f], fv.begin() + fo[f + 1]);
        }
    }
    std::vector<std::vector<scalar>> y(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "wall" && patches[pi].size > 0) y[pi].assign(patches[pi].size, nwdGreat);
    }
    if (wallFace.empty()) return y;

    if (kept.wallFace != wallFace || kept.verts != verts)
    {
        std::unordered_map<label, std::vector<label>> pointFaces;
        for (std::size_t w = 0; w < wallFace.size(); ++w)
        {
            const label f = wallFace[w];
            for (label j = fo[f]; j < fo[f + 1]; ++j)
            {
                pointFaces[fv[j]].push_back(static_cast<label>(w));
            }
        }
        kept.start.assign(1, label(0));
        kept.cand.clear();
        for (std::size_t w = 0; w < wallFace.size(); ++w)
        {
            const std::size_t first = kept.cand.size();
            kept.cand.push_back(static_cast<label>(w));
            const label f = wallFace[w];
            for (label j = fo[f]; j < fo[f + 1]; ++j)
            {
                for (const label nw : pointFaces[fv[j]])
                {
                    bool have = false;
                    for (std::size_t k = first; k < kept.cand.size(); ++k)
                    {
                        if (kept.cand[k] == nw) have = true;
                    }
                    if (!have) kept.cand.push_back(nw);
                }
            }
            kept.start.push_back(static_cast<label>(kept.cand.size()));
        }
        kept.wallFace = wallFace;
        kept.verts = verts;
    }

    const std::size_t nW = wallFace.size();
    const unsigned nT = static_cast<unsigned>(std::min<std::size_t>(std::max(1u, nThreads), nW));
    const auto range = [&](unsigned t)
    {
        for (std::size_t w = nW*t/nT; w < nW*(t + 1)/nT; ++w)
        {
            const vector& Cc = C[patches[from[w].first].faceCells[from[w].second]];
            scalar best = nwdGreat;
            const label last = selfOnly ? kept.start[w] + 1 : kept.start[w + 1];
            for (label k = kept.start[w]; k < last; ++k)
            {
                const label gf = wallFace[static_cast<std::size_t>(kept.cand[static_cast<std::size_t>(k)])];
                const scalar d = pointToFaceDist(Cc, pts, fv, fo[gf], fo[gf + 1]);
                if (d < best) best = d;
            }
            y[from[w].first][static_cast<std::size_t>(from[w].second)] = best;
        }
    };
    std::vector<std::thread> workers;
    for (unsigned t = 1; t < nT; ++t)
    {
        workers.emplace_back(range, t);
    }
    range(0);
    for (std::thread& t : workers)
    {
        t.join();
    }
    return y;
}

} // namespace brae
