#pragma once
// brae::cellWallDist, OpenFOAM wallDist::New(mesh).y(): the cell-wise wall-distance field (every cell),
// used by kOmegaSST F1/F2/F3 and Spalart-Allmaras dTilda.
//
// Ported byte-for-byte from OF's DEFAULT method, meshWave (wallDist.H "method meshWave" ->
// patchDistMethods::meshWave -> patchWave -> FaceCellWave<wallPoint>):
//   1. SEED   (patchWave::setChangedFaces, patchWave.C:64): every wall-patch face f gets
//             faceInfo[f] = wallPoint(faceCentre Cf[f], distSqr 0). These are the changedFaces.
//   2. WAVE   (FaceCellWave::faceToCell / cellToFace): the front propagates origin (a wall-face CENTRE)
//             cell<->face through the mesh graph. wallPoint::update (wallPointI.H): dist2 = magSqr(pt-origin);
//             accept iff first-visit, OR strictly closer by more than propagationTol_ (= 0.01, the OF default
//             "just to limit propagation of small changes"). Each cell converges to the nearest PROPAGATED
//             wall-face centre, NOT a global Euclidean min; the front is connectivity-dependent, exactly as OF.
//   3. y      (patchWave::getValues, patchWave.C:99): y[c] = sqrt(cellInfo[c].distSqr()).
//   4. correctWalls (default true, patchWave.C:203): boundary-adjacent cells are OVERWRITTEN with the EXACT
//             near-wall distance, correctBoundaryFaceCells (point-to-face-polygon for a cell's own wall faces)
//             and correctBoundaryPointCells (point-connected). brae::nearWallDist computes exactly that and is
//             validated machine-precision vs OF's correctWalls on the gated near-wall cells (the SST-relevant
//             region). Interior cells keep the wave's face-centre distance (OF's own meshWave overestimate);
//             there F1/F2 -> 0 so the value cannot move the model (ctest treats interior as informational).
//
// This is O(nCells * frontRevisits), the FaceCellWave front, NOT the O(nCells * nWallFaces) brute force it
// replaces; on complex geometries (motorBike: ~40k wall faces) that is the difference between a multi-minute
// startup and seconds. Static geometry -> computed ONCE at setup.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "near_wall_dist.cuh"   // closestPointOnTriangle, pointToFaceDist, nwdGreat (= OF correctWalls)
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace brae {

inline std::vector<scalar> cellWallDist(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    std::vector<vector>* wallOrigin = nullptr,   // optional: nearest wall-face centre per cell (IDDES wall-normal source)
    // optional: the nearest wall face's OUTWARD unit normal, transported by the same wave.
    //
    // This is OF's wallDist::n(), and its direction is not the obvious one: wallDist seeds the wave with
    // `nbf[patchi] == patches[patchi].nf()` (wallDist.C constructn) and carries that vector along with
    // the origin (patchDataWave<wallPointData<vector>>), so a cell inherits the nearest wall face's
    // normal pointing OUT of the domain -- away from the fluid, i.e. roughly opposite to (C - origin).
    // ZDES2020 takes max(grad(nuTilda) & n, 0) with this n, so the sign is load-bearing: get it backwards
    // and the shielding fires where it should be dormant.
    std::vector<vector>* wallNormal = nullptr)
{
    const std::vector<vector>& C   = g.C();
    const std::vector<vector>& Cf  = g.Cf();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const label nCells = m.nCells();
    const label nFaces = m.nFaces();
    const label nIntF  = m.nInternalFaces();

    std::vector<scalar> y(nCells, nwdGreat);                  // OF wallDist: y initialised to GREAT
    if (wallOrigin) *wallOrigin = C;                          // default: cell centre (degenerate -> IDDES hwn falls back to hmax)
    if (wallNormal) wallNormal->assign(nCells, vector{0, 0, 0});   // cells the wave never reaches: no wall direction

    // the wave seed: every wall-patch face (patchWave::setChangedFaces)
    std::vector<char> isWallFace(nFaces, 0);
    bool anyWall = false;
    for (const FvPatch& p : patches)
        if (p.type == "wall" && p.size > 0)
        {
            anyWall = true;
            for (label i = 0; i < p.size; ++i)
                isWallFace[p.start + i] = 1;
        }
    if (!anyWall) return y;                                   // no walls -> all GREAT (no SST near-wall branch)

    // cell -> faces adjacency (OF mesh_.cells()), CSR built from owner/neighbour
    std::vector<label> cfOff(nCells + 1, 0);
    for (label f = 0; f < nFaces; ++f)
        cfOff[own[f] + 1]++;
    for (label f = 0; f < nIntF;  ++f)
        cfOff[nei[f] + 1]++;
    for (label c = 0; c < nCells; ++c)
        cfOff[c + 1] += cfOff[c];
    std::vector<label> cfList(cfOff[nCells]);
    {
        std::vector<label> cur(cfOff.begin(), cfOff.end() - 1);
        for (label f = 0; f < nFaces; ++f)
            cfList[cur[own[f]]++] = f;
        for (label f = 0; f < nIntF;  ++f)
            cfList[cur[nei[f]]++] = f;
    }

    // FaceCellWave<wallPoint> front. origin = wall-face centre; distSqr = magSqr(pt - origin).
    const scalar tol = 0.01;                                  // OF FaceCellWave::propagationTol_
    std::vector<vector> faceOrg(nFaces), cellOrg(nCells);
    std::vector<vector> faceNrm(nFaces), cellNrm(nCells);     // the wall normal riding along with the origin
    std::vector<scalar> faceD2(nFaces, 0.0), cellD2(nCells, 0.0);
    std::vector<char> faceSet(nFaces, 0), cellSet(nCells, 0);
    std::vector<char> faceQ(nFaces, 0), cellQ(nCells, 0);     // already-in-changed-list flags (OF changedFace_/changedCell_)

    // wallPoint::update (wallPointI.H): first visit accepts any value; else accept iff strictly closer by > tol.
    auto update = [&](char& set, scalar& d2cur, vector& orgcur, vector& nrmcur,
                      const vector& pt, const vector& org, const vector& nrm) -> bool
    {
        const scalar d2 = magSqr(pt - org);
        if (!set)
        {
            d2cur = d2;
            orgcur = org;
            nrmcur = nrm;
            set = 1;
            return true;
        }
        const scalar diff = d2cur - d2;
        if (diff < 0) return false;                                              // already nearer
        if (diff < 1e-300 || (d2cur > 1e-300 && diff / d2cur < tol)) return false; // improvement too small
        d2cur = d2;
        orgcur = org;
        nrmcur = nrm;
        return true;
    };

    std::vector<label> changedFaces, changedCells;
    changedFaces.reserve(nFaces / 8 + 1);
    for (label f = 0; f < nFaces; ++f)
        if (isWallFace[f])
        {
            faceOrg[f] = Cf[f];
            {   // the seed datum: this wall face's OUTWARD unit normal (OF patch.nf())
                const vector& sf = g.Sf()[f];
                const scalar a = g.magSf()[f];
                faceNrm[f] = (a > scalar(0)) ? vector{sf.x/a, sf.y/a, sf.z/a} : vector{0, 0, 0};
            }
            faceD2[f] = 0.0;
            faceSet[f] = 1;
            faceQ[f] = 1;
            changedFaces.push_back(f);
        }

    while (!changedFaces.empty())
    {
        changedCells.clear();                                // faceToCell
        for (label f : changedFaces)
        {
            faceQ[f] = 0;
            const vector org = faceOrg[f];
            const vector nrm = faceNrm[f];
            const label o = own[f];
            if (update(cellSet[o], cellD2[o], cellOrg[o], cellNrm[o], C[o], org, nrm) && !cellQ[o])
            {
                cellQ[o] = 1;
                changedCells.push_back(o);
            }
            if (f < nIntF)
            {
                const label n = nei[f];
                if (update(cellSet[n], cellD2[n], cellOrg[n], cellNrm[n], C[n], org, nrm) && !cellQ[n])
                {
                    cellQ[n] = 1;
                    changedCells.push_back(n);
                }
            }
        }
        changedFaces.clear();                                // cellToFace
        for (label c : changedCells)
        {
            cellQ[c] = 0;
            const vector org = cellOrg[c];
            const vector nrm = cellNrm[c];
            for (label k = cfOff[c]; k < cfOff[c + 1]; ++k)
            {
                const label f = cfList[k];
                if (update(faceSet[f], faceD2[f], faceOrg[f], faceNrm[f], Cf[f], org, nrm) && !faceQ[f])
                {
                    faceQ[f] = 1;
                    changedFaces.push_back(f);
                }
            }
        }
    }
    for (label c = 0; c < nCells; ++c)
        if (cellSet[c]) y[c] = std::sqrt(cellD2[c]);

    // correctWalls = correctBoundaryFaceCells THEN correctBoundaryPointCells (cellDistFuncs.C), in that order:
    // a cell already corrected by its own wall face is NOT overwritten by the point pass.
    const std::vector<vector>& pts = m.points();
    const std::vector<label>&  fv  = m.faceVerts();
    const std::vector<label>&  fo  = m.faceOffsets();

    // correctWalls, as OpenFOAM ACTUALLY runs it. patchWave::correct() branches on a static switch:
    //
    //     if (cellDistFuncs::useCombinedWallPatch)                       // cellDistFuncs.C:42, DEFAULT TRUE
    //         correctBoundaryCells(patchIDs.sortedToc(), true, ...);     // "Correct across multiple patches"
    //     else
    //         correctBoundaryFaceCells(...); correctBoundaryPointCells(...);   // "Backwards compatible"
    //
    // The pair of functions the class is usually read for is the BACKWARDS-COMPATIBLE branch and is not
    // what runs. correctBoundaryCells builds ONE uindirectPrimitivePatch out of every wall patch's faces
    // and does both passes on that, so a point's face set spans patch boundaries. Porting the per-patch
    // branch instead measured 2847 cells off against 963 with brae then reading a median 1.239 of
    // OpenFOAM -- too LARGE, because a per-patch point sees only a fraction of the faces around it. On
    // motorBike, whose bike surface is split into 68 wall patches, that distinction is most of the story.
    //
    // Combined patch, then:
    //   faces      every wall patch's faces, in patch order then face order (the combined index)
    //   meshPoints first appearance walking those faces in order (PrimitivePatchMeshData.C), NOT sorted
    //   pass 1     for each combined face: owner cell takes smallestDist over getPointNeighbours(face),
    //              written UNCONDITIONALLY -- a cell owning several wall faces keeps the LAST, not the
    //              smallest -- and the cell is recorded in nearestFace
    //   pass 2     for each combined meshPoint in order, for each cell on it NOT already recorded:
    //              smallestDist over that point's combined faces, then the cell is LOCKED
    std::vector<label> wf;                                   // combined index -> global face
    std::vector<label> wfCell;                               // combined index -> owner cell
    for (const FvPatch& p : patches)
        if (p.type == "wall")
            for (label i = 0; i < p.size; ++i)
            {
                wf.push_back(p.start + i);
                wfCell.push_back(p.faceCells[i]);
            }

    if (!wf.empty())
    {
        // combined-patch point -> combined faces, plus meshPoints in first-appearance order
        std::unordered_map<label, std::vector<label>> ptF;
        std::vector<label> meshPoints;
        meshPoints.reserve(wf.size() * 4);
        for (std::size_t i = 0; i < wf.size(); ++i)
            for (label j = fo[wf[i]]; j < fo[wf[i] + 1]; ++j)
            {
                const label v = fv[j];
                auto it = ptF.find(v);
                if (it == ptF.end()) { ptF.emplace(v, std::vector<label>{(label)i}); meshPoints.push_back(v); }
                else it->second.push_back((label)i);
            }

        // point -> cells (OF mesh().pointCells()): a cell touches a point if one of its faces uses it.
        std::vector<std::vector<label>> pc(pts.size());
        for (label f = 0; f < nFaces; ++f)
        {
            const label c0 = own[f], c1 = (f < nIntF ? nei[f] : -1);
            for (label j = fo[f]; j < fo[f + 1]; ++j)
            {
                pc[fv[j]].push_back(c0);
                if (c1 >= 0) pc[fv[j]].push_back(c1);
            }
        }
        for (auto& v : pc) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); }

        std::vector<char> claimed(nCells, 0);
        auto distTo = [&](const vector& Cc, label ci)
        { const label gf = wf[ci]; return pointToFaceDist(Cc, pts, fv, fo[gf], fo[gf + 1]); };

        // pass 1 -- cells with a face on the wall
        for (std::size_t i = 0; i < wf.size(); ++i)
        {
            const label c = wfCell[i];
            const vector& Cc = C[c];
            scalar best = distTo(Cc, (label)i);              // getPointNeighbours "adds myself" first
            for (label j = fo[wf[i]]; j < fo[wf[i] + 1]; ++j)
                for (const label nb : ptF[fv[j]])
                    if (nb != (label)i) best = std::fmin(best, distTo(Cc, nb));
            y[c] = best;                                     // unconditional: last face wins, as OF does
            claimed[c] = 1;
        }

        // pass 2 -- cells with only a point on the wall; first meshPoint to reach one wins and locks it
        for (const label v : meshPoints)
        {
            const std::vector<label>& faces = ptF[v];
            for (const label c : pc[v])
            {
                if (claimed[c]) continue;
                scalar best = nwdGreat;
                for (const label nb : faces) best = std::fmin(best, distTo(C[c], nb));
                y[c] = best;
                claimed[c] = 1;
            }
        }
    }

    // the wave's nearest wall-face centre per reached cell -> the IDDES wall-normal direction (C - origin). Cells the
    // wave never reached keep the default (C, degenerate). Does not alter y (the correctWalls override above is intact).
    if (wallOrigin)
        for (label c = 0; c < nCells; ++c)
            if (cellSet[c]) (*wallOrigin)[c] = cellOrg[c];
    if (wallNormal)
        for (label c = 0; c < nCells; ++c)
            if (cellSet[c]) (*wallNormal)[c] = cellNrm[c];
    return y;
}

// OF wallDist `method exactDistance` (patchDistMethods::exact): the TRUE Euclidean distance from every
// cell centre to the nearest point on the WALL SURFACE, not the connectivity-propagated wave above.
//
// OF triangulates the wall patches into a distributedTriSurfaceMesh and calls findNearest on an octree.
// brae measures to the face POLYGON instead, via the same pointToFaceDist (OF face::nearestPoint's
// centre-fan) that nearWallDist uses. For a PLANAR face the two agree exactly -- any correct
// triangulation of a planar polygon covers the same point set, so the nearest point is the same. They can
// differ on a WARPED face, where the triangulations disagree about the surface between the vertices; that
// is a genuine, small difference and is noted rather than hidden.
//
// WHY A GRID. Brute force is O(nCells * nWallFaces) and the cases that ask for exactDistance are not
// small: pimpleFoam/LES/wallMountedHump is 2.3M cells. Wall faces are bucketed into a uniform grid, and
// each query expands in Chebyshev rings until the searched box is guaranteed to contain the answer --
// exact, not approximate, because the stopping bound is the distance from the query point to the OUTSIDE
// of the box already searched, and every face is bucketed into every grid cell its bounding box touches.
inline std::vector<scalar> exactCellWallDist(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    // optional: the outward unit normal of the wall face the nearest point was found on -- OF's
    // wallDist::n() for this method (`surf.getNormal(info, n)`, the surface normal AT the hit). A case
    // that writes `nRequired yes` under wallDist is asking for exactly this field; NACA4412 does.
    std::vector<vector>* wallNormal = nullptr)
{
    const std::vector<vector>& pts = m.points();
    const std::vector<label>&  fv  = m.faceVerts();
    const std::vector<label>&  fo  = m.faceOffsets();
    const std::vector<vector>& C   = g.C();
    const label nC = m.nCells();

    if (wallNormal) wallNormal->assign(nC, vector{0, 0, 0});
    std::vector<label> wallFace;
    for (const FvPatch& p : fvp)
        if (p.type == "wall")
            for (label i = 0; i < p.size; ++i) wallFace.push_back(p.start + i);

    std::vector<scalar> y(nC, 0.0);
    if (wallFace.empty()) return y;   // no walls -> OF leaves y at its initial value

    // per-face bounding box + the overall one
    const label nW = static_cast<label>(wallFace.size());
    std::vector<vector> bLo(nW), bHi(nW);
    vector lo{ nwdGreat,  nwdGreat,  nwdGreat};
    vector hi{-nwdGreat, -nwdGreat, -nwdGreat};
    for (label w = 0; w < nW; ++w)
    {
        const label f = wallFace[w];
        vector a{ nwdGreat,  nwdGreat,  nwdGreat};
        vector b{-nwdGreat, -nwdGreat, -nwdGreat};
        for (label j = fo[f]; j < fo[f + 1]; ++j)
        {
            const vector& q = pts[fv[j]];
            a.x = std::fmin(a.x, q.x); a.y = std::fmin(a.y, q.y); a.z = std::fmin(a.z, q.z);
            b.x = std::fmax(b.x, q.x); b.y = std::fmax(b.y, q.y); b.z = std::fmax(b.z, q.z);
        }
        bLo[w] = a; bHi[w] = b;
        lo.x = std::fmin(lo.x, a.x); lo.y = std::fmin(lo.y, a.y); lo.z = std::fmin(lo.z, a.z);
        hi.x = std::fmax(hi.x, b.x); hi.y = std::fmax(hi.y, b.y); hi.z = std::fmax(hi.z, b.z);
    }
    // pad so a degenerate (planar/2D) extent still has a positive cell size
    const scalar span = std::fmax(std::fmax(hi.x - lo.x, hi.y - lo.y), std::fmax(hi.z - lo.z, scalar(1e-30)));
    const scalar pad  = 1e-6 * span;
    lo.x -= pad; lo.y -= pad; lo.z -= pad;
    hi.x += pad; hi.y += pad; hi.z += pad;

    // ~1 face per grid cell, capped so the grid itself stays small
    const scalar target = std::cbrt(std::fmax((hi.x-lo.x)*(hi.y-lo.y)*(hi.z-lo.z), scalar(1e-300))
                                    / std::fmax(scalar(nW), scalar(1)));
    const scalar h = std::fmax(target, span * scalar(1e-4));
    // Size the grid by its TOTAL bucket count, not per axis. A per-axis cap alone lets a 3D mesh ask for
    // 512^3 buckets -- the offsets array is then over half a gigabyte and its prefix sum dominates the
    // run. wallMountedHump (2.3M cells) sat in set-up for minutes before this was bounded.
    auto dimsFor = [&](scalar hh, int& ax, int& ay, int& az)
    {
        auto d1 = [&](scalar ext) { return std::max(1, std::min(1024, (int)std::floor(ext / hh) + 1)); };
        ax = d1(hi.x - lo.x); ay = d1(hi.y - lo.y); az = d1(hi.z - lo.z);
    };
    int nx = 1, ny = 1, nz = 1;
    {
        scalar hh = h;
        const double cap = std::max(4.0*double(nW), 1.0e5);
        for (int it = 0; it < 24; ++it)
        {
            dimsFor(hh, nx, ny, nz);
            if (double(nx)*double(ny)*double(nz) <= cap) break;
            hh *= std::cbrt(double(nx)*double(ny)*double(nz)/cap)*1.05;
        }
    }
    const vector hs{ (hi.x-lo.x)/nx, (hi.y-lo.y)/ny, (hi.z-lo.z)/nz };
    auto gidx = [&](int i, int j, int k) { return (std::size_t)((k*ny + j)*(std::size_t)nx + i); };
    auto clampi = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };

    // CSR bucket: every face into every grid cell its bbox touches (count, then fill)
    std::vector<label> cnt((std::size_t)nx*ny*nz + 1, 0);
    auto range = [&](label w, int* i0, int* i1, int* j0, int* j1, int* k0, int* k1)
    {
        *i0 = clampi((int)std::floor((bLo[w].x - lo.x)/hs.x), nx);
        *i1 = clampi((int)std::floor((bHi[w].x - lo.x)/hs.x), nx);
        *j0 = clampi((int)std::floor((bLo[w].y - lo.y)/hs.y), ny);
        *j1 = clampi((int)std::floor((bHi[w].y - lo.y)/hs.y), ny);
        *k0 = clampi((int)std::floor((bLo[w].z - lo.z)/hs.z), nz);
        *k1 = clampi((int)std::floor((bHi[w].z - lo.z)/hs.z), nz);
    };
    for (label w = 0; w < nW; ++w)
    {
        int i0,i1,j0,j1,k0,k1; range(w,&i0,&i1,&j0,&j1,&k0,&k1);
        for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i)
            ++cnt[gidx(i,j,k) + 1];
    }
    for (std::size_t q = 1; q < cnt.size(); ++q) cnt[q] += cnt[q-1];
    std::vector<label> bucket(cnt.back());
    { std::vector<label> at(cnt.begin(), cnt.end() - 1);
      for (label w = 0; w < nW; ++w)
      {
          int i0,i1,j0,j1,k0,k1; range(w,&i0,&i1,&j0,&j1,&k0,&k1);
          for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i)
              bucket[at[gidx(i,j,k)]++] = w;
      } }

    for (label c = 0; c < nC; ++c)
    {
        const vector& p = C[c];
        const int ci = clampi((int)std::floor((p.x - lo.x)/hs.x), nx);
        const int cj = clampi((int)std::floor((p.y - lo.y)/hs.y), ny);
        const int ck = clampi((int)std::floor((p.z - lo.z)/hs.z), nz);
        scalar best = nwdGreat;
        label  bestFace = -1;
        const int rMax = std::max(nx, std::max(ny, nz));
        for (int r = 0; r <= rMax; ++r)
        {
            // the shell at Chebyshev radius r (r = 0 is the query's own cell)
            const int i0 = ci - r, i1 = ci + r, j0 = cj - r, j1 = cj + r, k0 = ck - r, k1 = ck + r;
            for (int k = k0; k <= k1; ++k)
            {
                if (k < 0 || k >= nz) continue;
                const bool kEdge = (k == k0 || k == k1);
                for (int j = j0; j <= j1; ++j)
                {
                    if (j < 0 || j >= ny) continue;
                    const bool jEdge = (j == j0 || j == j1);
                    const int step = (kEdge || jEdge) ? 1 : (i1 - i0 == 0 ? 1 : i1 - i0);
                    for (int i = i0; i <= i1; i += step)
                    {
                        if (i < 0 || i >= nx) continue;
                        const std::size_t gi = gidx(i,j,k);
                        for (label b = cnt[gi]; b < cnt[gi+1]; ++b)
                        {
                            const label f = wallFace[bucket[b]];
                            const scalar d = pointToFaceDist(p, pts, fv, fo[f], fo[f+1]);
                            if (d < best) { best = d; bestFace = f; }
                        }
                    }
                }
            }
            // STOP when the box already searched provably contains the nearest face. Any face not yet
            // seen lies outside that box, so it is at least this far away.
            const scalar bx = lo.x + (i0    )*hs.x, bX = lo.x + (i1 + 1)*hs.x;
            const scalar by = lo.y + (j0    )*hs.y, bY = lo.y + (j1 + 1)*hs.y;
            const scalar bz = lo.z + (k0    )*hs.z, bZ = lo.z + (k1 + 1)*hs.z;
            const scalar guard = std::fmin(std::fmin(std::fmin(p.x - bx, bX - p.x),
                                                     std::fmin(p.y - by, bY - p.y)),
                                           std::fmin(p.z - bz, bZ - p.z));
            if (best <= std::fmax(guard, scalar(0))) break;
        }
        y[c] = best;
        if (wallNormal && bestFace >= 0)
        {
            const scalar a = g.magSf()[bestFace];
            if (a > scalar(0))
            {
                const vector& sf = g.Sf()[bestFace];
                (*wallNormal)[c] = vector{sf.x/a, sf.y/a, sf.z/a};
            }
        }
    }
    return y;
}

} // namespace brae
