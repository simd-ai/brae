#pragma once
// brae::AMIInterface, the cyclicAMI (Arbitrary Mesh Interface) lduInterface. Unlike cyclic (1:1 face pairing),
// the two patches are NON-CONFORMING: each SOURCE face couples to MULTIPLE neighbour cells via an area-weighted
// interpolation. Mirrors OpenFOAM AMIInterpolation (faceAreaWeightAMI) + cyclicAMIPolyPatch / cyclicAMIFvPatchField.
//
// Coupling (OF cyclicAMIFvPatchField::updateInterfaceMatrix / patchNeighbourField):
//   pnf[srcFace] = sum_j srcWeights[srcFace][j] * transform(forwardT, psi[ tgtCell[srcFace][j] ])
//   result[ownCell[srcFace]] -= coeff[srcFace] * pnf[srcFace]
// i.e. the single cyclic (own,nbr) pair becomes a weighted STENCIL (own, list of (tgtCell, weight)).
//
// Weights (faceAreaWeightAMI): map the target faces to the source side via the transform, project both to the
// source patch's average plane, and clip each src/tgt polygon pair (Sutherland-Hodgman) -> overlap area.
//   srcWeights[i][j] = overlap_ij / srcMagSf[i]    (conformal=false normalisation; sum_j = coverage fraction)
//   srcWeightsSum[i] = sum_j overlap_ij / srcMagSf[i]
// (this header: host build + weights; device coupling is in device_ami.{cuh,cu}). Both sides are built (symmetric).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cf_types.cuh"
#include "interface/cyclic_interface.cuh"   // rotationTensor + transform conventions
#include <map>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>

namespace brae {

struct AMIInterface
{
    label patch = -1, nbrPatch = -1;            // this (source) patch and its paired (target) patch
    std::vector<label>  ownCell;                // owner cell of each SOURCE face (size = nSrc)
    std::vector<label>  srcOffset;              // CSR offsets into (nbrCell, weight), size nSrc+1
    std::vector<label>  nbrCell;                // target-face owner cell per stencil entry
    std::vector<scalar> weight;                 // normalised AMI weight per stencil entry
    std::vector<scalar> weightsSum;             // per src face: coverage = sum of weights (pre-lowWeight)
    std::vector<scalar> magSf;                  // per src face |Sf|
    std::vector<vector> Sf;                     // per src face area vector (out of ownCell)
    std::vector<scalar> deltaCoeffs;            // per src face 1/(nf & delta), delta to the AMI-interpolated nbr
    std::vector<scalar> weights;                // per src face interp weight: face = w*own + (1-w)*nbr_interp
    std::vector<vector> corrVec;                // per src face non-orth correction vector (nf - delta*deltaCoeffs)
    std::vector<vector> dOwn;                   // per src face Cf - C[own] (linearUpwind own-side reconstruction delta)
    std::vector<vector> dNbr;                   // per STENCIL ENTRY Cf_tgt - C[nbr] (UN-rotated, nbr-side reconstruction)
    // fvPatch::delta(): dOwn minus the AMI-interpolated, TRANSFORMED neighbour delta -- the vector
    // deltaCoeffs and corrVec are already built from. Stored so the TVD limiter at a coupled face reads
    // the same number instead of re-deriving the interpolation and the rotation a second time.
    std::vector<vector> delta;
    // cyclicACMI. The interface is only PARTIALLY coupled: it shares its faces with a coincident wall
    // (nonOverlapPatch), and the overlap fraction decides how the area splits between them. Partial
    // coverage is the DEFINING FEATURE here, not a defect -- so ACMI is exempt from the coverage
    // refusal below, and carries no transform (see `separation`).
    bool                acmi = false;
    std::vector<scalar> mask;                   // OF cyclicACMIPolyPatch.C:348, clamp(weightsSum, 0, 1)
    bool                translational = true;
    vector              separation{0,0,0};      // translational period (Cf_nbr - Cf_own at a matched point)
    tensor              forwardT{1,0,0,0,1,0,0,0,1};   // rotational nbr->own rotation (identity if translational)
};

namespace ami_detail {
struct vec2 { scalar x, y; };
inline scalar signedArea(const std::vector<vec2>& p)
{
    scalar a = 0;
    const int n = (int)p.size();
    for (int i = 0; i < n; ++i)
    {
        const vec2 &u = p[i], &v = p[(i+1)%n];
        a += u.x*v.y - v.x*u.y;
    }
    return 0.5*a;
}
inline void orientCCW(std::vector<vec2>& p) { if (signedArea(p) < 0) std::reverse(p.begin(), p.end()); }
inline scalar cross2(const vec2& a, const vec2& b, const vec2& c)
{
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}
// Sutherland-Hodgman: clip `subject` (CCW) by convex `clip` (CCW). Returns the clipped polygon (CCW).
inline std::vector<vec2> clipPoly(const std::vector<vec2>& subject, const std::vector<vec2>& clip)
{
    std::vector<vec2> out = subject;
    const int nc = (int)clip.size();
    for (int e = 0; e < nc && !out.empty(); ++e)
    {
        const vec2 A = clip[e], B = clip[(e+1)%nc];                 // clip edge A->B (inside = left, CCW)
        const std::vector<vec2> in = out;
        out.clear();
        const int ni = (int)in.size();
        for (int i = 0; i < ni; ++i)
        {
            const vec2 P = in[i], Q = in[(i+1)%ni];
            const scalar sp = cross2(A,B,P), sq = cross2(A,B,Q);
            if (sp >= 0) out.push_back(P);
            if ((sp >= 0) != (sq >= 0))                            // edge P->Q crosses the clip line
            {
                const scalar t = sp / (sp - sq);
                out.push_back({P.x + t*(Q.x-P.x), P.y + t*(Q.y-P.y)});
            }
        }
    }
    return out;
}
inline scalar overlapArea(const std::vector<vec2>& a, const std::vector<vec2>& b)
{
    const std::vector<vec2> c = clipPoly(a, b);
    return c.size() < 3 ? 0.0 : std::fabs(signedArea(c));
}
} // namespace ami_detail

// Build the cyclicAMI interfaces (both source+target sides) with faceAreaWeightAMI weights.
//
// WEIGHTS ARE NORMALISED BY g.rawMagSf(), NOT g.magSf(). They differ only on a cyclicACMI mesh, where
// the coupled area has already been scaled by the overlap mask: normalising the overlap by the scaled
// area would return 1 for every face and destroy the very mask it expresses. OF has the same separation
// for free -- its AMI is built on the primitivePatch,
// whose areas come straight from the point positions, while the SCALED areas live on the polyPatch
// ("using primitivePatch face areas since these are based on the raw point locations (not affected by
// ACMI scaling)", cyclicACMIPolyPatch.C:394).
//
// The flux areas stored on the interface (ai.Sf / ai.magSf) deliberately keep using g, i.e. the SCALED
// values -- those are the areas the interface actually transports through.
//
// NOTE ON NORMALISATION -- TWO STEPS, and cyclicACMI needs BOTH. OF normalises AMI weights two ways
// (AMIInterpolation.C:159-208), chosen by `conformal` = requireMatch:
//     requireMatch 1 (cyclicAMI) : denom = sum(overlap)  -> weights sum to exactly 1
//     requireMatch 0 (cyclicACMI): denom = face area     -> weights sum to the COVERAGE
// and the ACMI polyMesh boundary carries `requireMatch 0`, so the loop above divides by the face area
// exactly like OF. That is where the resemblance used to stop, and it was one call short: for ACMI, and
// only for ACMI, OF then RE-NORMALISES those weights back to 1 at the end of
// cyclicACMIPolyPatch::scalePatchFaceAreas, because by then the coupled Sf carries the coverage instead.
// See the block at the re-normalisation below for what keeping both costs.
//
// OF's printed weight sums are NOT evidence either way. They come from inside normaliseWeights, before
// the ACMI override: on oscillatingInletACMI2D at t = 0.292 the log says average 0.7578655102 over 40
// faces (30 covered / 1 blended / 9 uncovered = (30 + 0.3146)/40) while the weights the run then solves
// with are 1 on all 31 non-empty faces.
// UNIFORM-GRID BROAD PHASE over the target patch's face bounding boxes.
//
// The sweep this feeds used to test EVERY source face against EVERY target face. That is O(nSrc*nTgt),
// and on pimpleFoam/RAS/propeller -- 18496 against 18720 faces -- it is 3.5e8 polygon-overlap tests per
// direction, per direction pair, rebuilt every moving time step. The case never reached its first time
// step in ten minutes. OpenFOAM does not do this: it walks an advancing front from a seed face, which
// is O(n) and is why OF meshes the same interface in seconds.
//
// A uniform grid is chosen over the LBVH the GPU literature favours (Karras' Morton-code radix tree)
// for one reason: AMI patches are surfaces with near-uniform face sizes, which is the case a uniform
// grid handles as well as a hierarchy and with a fraction of the code. It is also the shape that ports
// to the device unchanged -- flat arrays, a CSR of cell -> faces, and per-source-face queries that are
// completely independent of each other. An LBVH is the better answer only if face sizes span orders of
// magnitude on one patch, which an AMI pair does not.
//
// It is a BROAD PHASE and nothing else: it discards pairs whose bounding boxes cannot touch, and every
// surviving pair goes through exactly the same projection and clip as before. The weights it produces
// are therefore bit-identical to the brute-force ones -- which is the gate it has to pass, because a
// search that MISSES a pair silently loses that face's coverage instead of failing.
namespace ami_detail {
struct FaceGrid
{
    vector lo{0,0,0}, hi{0,0,0};
    scalar inv = 1;                       // 1 / cell size
    label  nx = 1, ny = 1, nz = 1;
    scalar pad = 0;                       // query inflation: see the note in query()
    std::vector<label> off, idx;          // CSR: cell -> target-face list
    std::vector<vector> bbLo, bbHi;       // per target face, its own bounding box

    label cellOf(label ix, label iy, label iz) const { return (iz*ny + iy)*nx + ix; }
    label clampi(scalar v, scalar o, label n) const
    {
        const long k = (long)std::floor((v - o)*inv);
        return (label)std::min<long>(std::max<long>(k, 0), (long)n - 1);
    }

    void build(const std::vector<std::vector<vector>>& polys)
    {
        const std::size_t n = polys.size();
        bbLo.assign(n, vector{0,0,0});
        bbHi.assign(n, vector{0,0,0});
        if (!n) { off.assign(2, 0); return; }
        lo = vector{ 1e300, 1e300, 1e300};
        hi = vector{-1e300,-1e300,-1e300};
        scalar diagSum = 0;
        for (std::size_t j = 0; j < n; ++j)
        {
            vector a{ 1e300, 1e300, 1e300}, b{-1e300,-1e300,-1e300};
            for (const vector& v : polys[j])
            {
                a.x = std::min(a.x, v.x); a.y = std::min(a.y, v.y); a.z = std::min(a.z, v.z);
                b.x = std::max(b.x, v.x); b.y = std::max(b.y, v.y); b.z = std::max(b.z, v.z);
            }
            bbLo[j] = a; bbHi[j] = b;
            lo.x = std::min(lo.x, a.x); lo.y = std::min(lo.y, a.y); lo.z = std::min(lo.z, a.z);
            hi.x = std::max(hi.x, b.x); hi.y = std::max(hi.y, b.y); hi.z = std::max(hi.z, b.z);
            diagSum += mag(b - a);
        }
        // Cell size = the MEAN face bounding-box diagonal. Smaller cells shrink the candidate lists but
        // make a face span more of them; the mean face size is the standard balance and needs no tuning.
        const scalar cell = std::max(diagSum/(scalar)n, scalar(1e-300));
        inv = scalar(1)/cell;
        // A BARE AABB TEST IS NOT A VALID BROAD PHASE HERE, and finding out cost a run.
        //
        // The narrow phase projects each pair onto the plane perpendicular to their pair normal, which
        // DISCARDS any separation along that normal -- the same property that lets two box faces a box
        // length apart project onto each other perfectly. So two faces on a CURVED interface, sitting at
        // slightly different radii, overlap in projection while their 3-D boxes do not touch at all.
        // Rejecting those pairs is not conservative, it is wrong: on RAS/rotatingFanInRoom (a cylindrical
        // AMI) the un-padded grid dropped mean coverage to 0.5512 with 3312 of 6984 source faces under
        // 99% -- caught by the coverage refusal rather than by a wrong answer, which is the one piece of
        // luck in it.
        //
        // Padding by one mean face size restores them. It is a bound, not a fudge: on a well-formed AMI
        // the two patches are nominally coincident, so a normal offset above one face size means the
        // pair could not share area anyway.
        pad = cell;
        auto dim = [&](scalar l, scalar h)
        {
            const long k = (long)std::floor((h - l)*inv) + 1;
            return (label)std::min<long>(std::max<long>(k, 1), 512);   // cap: memory, not accuracy
        };
        nx = dim(lo.x, hi.x); ny = dim(lo.y, hi.y); nz = dim(lo.z, hi.z);
        // recompute inv so the capped dimensions still cover the box
        const scalar sx = (hi.x-lo.x)/(scalar)nx, sy = (hi.y-lo.y)/(scalar)ny, sz = (hi.z-lo.z)/(scalar)nz;
        inv = scalar(1)/std::max(std::max(std::max(sx, sy), sz), scalar(1e-300));
        nx = dim(lo.x, hi.x); ny = dim(lo.y, hi.y); nz = dim(lo.z, hi.z);

        const label nCell = nx*ny*nz;
        std::vector<label> count((std::size_t)nCell + 1, 0);
        auto span = [&](std::size_t j, label& x0, label& x1, label& y0, label& y1, label& z0, label& z1)
        {
            x0 = clampi(bbLo[j].x, lo.x, nx); x1 = clampi(bbHi[j].x, lo.x, nx);
            y0 = clampi(bbLo[j].y, lo.y, ny); y1 = clampi(bbHi[j].y, lo.y, ny);
            z0 = clampi(bbLo[j].z, lo.z, nz); z1 = clampi(bbHi[j].z, lo.z, nz);
        };
        for (std::size_t j = 0; j < n; ++j)
        {
            label x0,x1,y0,y1,z0,z1; span(j,x0,x1,y0,y1,z0,z1);
            for (label z=z0; z<=z1; ++z) for (label y=y0; y<=y1; ++y) for (label x=x0; x<=x1; ++x)
                ++count[(std::size_t)cellOf(x,y,z) + 1];
        }
        off.assign((std::size_t)nCell + 1, 0);
        for (label c = 0; c < nCell; ++c) off[(std::size_t)c+1] = off[(std::size_t)c] + count[(std::size_t)c+1];
        idx.assign((std::size_t)off[(std::size_t)nCell], 0);
        std::vector<label> cur(off.begin(), off.end() - 1);
        for (std::size_t j = 0; j < n; ++j)
        {
            label x0,x1,y0,y1,z0,z1; span(j,x0,x1,y0,y1,z0,z1);
            for (label z=z0; z<=z1; ++z) for (label y=y0; y<=y1; ++y) for (label x=x0; x<=x1; ++x)
                idx[(std::size_t)cur[(std::size_t)cellOf(x,y,z)]++] = (label)j;
        }
    }

    // Candidate target faces whose bounding box overlaps `a`..`b`. De-duplicated: a face spanning
    // several cells appears in each of them.
    void query(const vector& a, const vector& b, std::vector<label>& out, std::vector<char>& seen) const
    {
        out.clear();
        if (idx.empty()) return;
        const vector qa{a.x - pad, a.y - pad, a.z - pad};
        const vector qb{b.x + pad, b.y + pad, b.z + pad};
        const label x0 = clampi(qa.x, lo.x, nx), x1 = clampi(qb.x, lo.x, nx);
        const label y0 = clampi(qa.y, lo.y, ny), y1 = clampi(qb.y, lo.y, ny);
        const label z0 = clampi(qa.z, lo.z, nz), z1 = clampi(qb.z, lo.z, nz);
        for (label z=z0; z<=z1; ++z) for (label y=y0; y<=y1; ++y) for (label x=x0; x<=x1; ++x)
        {
            const label c = cellOf(x,y,z);
            for (label k = off[(std::size_t)c]; k < off[(std::size_t)c+1]; ++k)
            {
                const label j = idx[(std::size_t)k];
                if (seen[(std::size_t)j]) continue;
                // exact AABB test: the grid only bounds the search, it does not decide overlap
                if (bbHi[(std::size_t)j].x < qa.x || bbLo[(std::size_t)j].x > qb.x) continue;
                if (bbHi[(std::size_t)j].y < qa.y || bbLo[(std::size_t)j].y > qb.y) continue;
                if (bbHi[(std::size_t)j].z < qa.z || bbLo[(std::size_t)j].z > qb.z) continue;
                seen[(std::size_t)j] = 1;
                out.push_back(j);
            }
        }
        for (const label j : out) seen[(std::size_t)j] = 0;
    }
};
}   // namespace ami_detail


inline std::vector<AMIInterface> buildAMIInterfaces(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    using namespace ami_detail;
    // The image count of a periodic AMI is a property of the PAIR, not of a direction.
    //
    // Each direction runs its own adaptive tiling loop and would otherwise stop as soon as ITS coverage
    // closed, so ami1->ami2 could settle on 1 image while ami2->ami1 took 2 -- stencils of different
    // extent for the same interface. Memoising K on the pair makes the second direction reuse the
    // first's count exactly; the coverage refusal below still catches a genuine shortfall.
    //
    // OpenFOAM does not need this because it builds ONE AMI per pair, on the owner, and the neighbour
    // direction reuses that addressing transposed. brae builds both directions -- which is what lets it
    // treat every interface uniformly -- so the pair has to agree on K explicitly.
    //
    // NOTE: this is an addressing guard, not a symmetry fix. The pressure operator is not self-adjoint
    // across ANY non-conforming AMI, periodic or not, in brae and in OpenFOAM alike; that is handled by
    // selecting BiCGStab rather than PCG (device_simple_foam.cu, and test_interface_invariants_real).
    std::map<std::pair<label, label>, label> periodicImages;
    std::map<std::string, label> nameToIdx;
    for (label pi = 0; pi < (label)fvp.size(); ++pi) nameToIdx[fvp[pi].name] = pi;
    const std::vector<PatchInfo>& pinfo = m.patches();
    const std::vector<vector>& pts = m.points();
    const std::vector<label>& fv = m.faceVerts();
    const std::vector<label>& fo = m.faceOffsets();

    // face polygon (global face index) as a list of 3D vertices
    auto facePoly = [&](label gf)
    {
        std::vector<vector> P;
        for (label k = fo[gf]; k < fo[gf+1]; ++k) P.push_back(pts[fv[k]]);
        return P;
    };

    std::vector<AMIInterface> out;
    for (label pi = 0; pi < (label)fvp.size(); ++pi)
    {
        const bool isACMI = (fvp[pi].type == "cyclicACMI");
        const bool isPeriodic = (fvp[pi].type == "cyclicPeriodicAMI");
        if (fvp[pi].type != "cyclicAMI" && !isACMI && !isPeriodic) continue;
        AMIInterface ai;
        ai.acmi = isACMI;
        ai.patch = pi;
        const std::string nbrName = pinfo[pi].neighbourPatch;
        const auto it = nameToIdx.find(nbrName);
        if (it == nameToIdx.end()) throw std::runtime_error("cyclicAMI: neighbourPatch '" + nbrName + "' not found");
        ai.nbrPatch = it->second;
        ai.translational = (pinfo[pi].transform != "rotational");
        const FvPatch& S = fvp[pi];
        const FvPatch& T = fvp[ai.nbrPatch];

        // transform (nbr/target -> own/source); same convention as cyclic
        vector axis{0,0,1};
        vector ctr{0,0,0};
        if (!ai.translational)
        {
            axis = pinfo[pi].rotationAxis / mag(pinfo[pi].rotationAxis);
            ctr = pinfo[pi].rotationCentre;
            const vector po = g.Cf()[S.start] - ctr, pn = g.Cf()[T.start] - ctr;
            const vector perpO = po - dot(po,axis)*axis, perpN = pn - dot(pn,axis)*axis;
            const scalar angle = std::atan2(dot(cross(perpN, perpO), axis), dot(perpN, perpO));
            ai.forwardT = rotationTensor(axis, angle);
        }
        // translational period = mean(Cf_tgt) - mean(Cf_src). Robust for NON-conforming patches (faces are not
        // ordered to match, so a face-0 difference is wrong): = the period vector for translational periodicity, ~=0
        // for coincident non-conformal joins. (A dict separationVector would override this; not yet read.)
        // cyclicACMI carries NO transform: the two patches are CO-LOCATED (OF's blockMeshDict gives
        // them the same faces) and any offset between their centroids is the physical slide of the
        // moving zone, not a period. Inferring a period there is catastrophic and silent -- measured on
        // pimpleFoam/RAS/oscillatingInletACMI2D at t=0.5, where the inlet channel has slid 0.5 in y:
        //
        //     inferred separation = (0 -0.5 0)   -> coverage 0 uncovered, 0 blended, 40 covered
        //     separation = 0                     -> coverage 19 uncovered, 1 blended, 20 covered
        //     OF's own report at t=0.5           -> 19, 1, 20
        //
        // With the inferred period the patches are re-aligned, the mask is identically 1 for all time,
        // the blockage wall gets zero area, and the sliding channel NEVER CLOSES -- a case that runs
        // clean and is entirely wrong.
        // cyclicPeriodicAMI is excluded for the SAME reason cyclicACMI is, and the evidence looked the
        // same: its two sides are CO-LOCATED (a rotor-stator station, or a sliding inlet against the duct
        // it slides along) and any centroid offset between them is the sector mismatch the periodic
        // tiling exists to cover -- not a period to subtract. Inferring one re-aligns the patches, the
        // raw overlap then comes out fully covered, the tiling loop runs ZERO images because it has
        // nothing left to do, and the interface silently couples the wrong faces to each other. That is
        // exactly what the first run showed: `images:0 srcSum:1` on a patch pair that needs images.
        if (!ai.acmi && !isPeriodic)
        {
            vector cS{0,0,0}, cT{0,0,0};
            for (label i = 0; i < S.size; ++i) cS += g.Cf()[S.start+i];
            for (label j = 0; j < T.size; ++j) cT += g.Cf()[T.start+j];
            ai.separation = cT/(scalar)T.size - cS/(scalar)S.size;
        }
        // map a target-side point to the source side (so src & mapped-tgt overlap in the same frame)
        auto mapTtoS = [&](const vector& p) -> vector
        {
            if (ai.translational) return p - ai.separation;
            return dot(p - ctr, transpose(ai.forwardT)) + ctr;   // forwardT*(p-ctr)+ctr  (nbr->own rotation)
        };

        // average source-plane basis (e1,e2,n): n = mean source unit normal
        vector n{0,0,0};
        for (label i = 0; i < S.size; ++i) n += g.Sf()[S.start+i] / g.magSf()[S.start+i];
        n = n / std::fmax(mag(n), 1e-300);
        vector e1 = std::fabs(n.x) < 0.9 ? cross(n, vector{1,0,0}) : cross(n, vector{0,1,0});
        e1 = e1 / std::fmax(mag(e1), 1e-300);
        vector e2 = cross(n, e1);
        const vector orig = g.Cf()[S.start];
        auto proj = [&](const vector& p) -> vec2
        {
            const vector d = p - orig;
            return { dot(d,e1), dot(d,e2) };
        };

        // 3-D polygons, kept unprojected: OF picks the projection direction PER SOURCE/TARGET PAIR,
        // so there is no one plane to flatten onto up front.
        std::vector<std::vector<vector>> srcW(S.size), tgtW(T.size);
        for (label i = 0; i < S.size; ++i)
            for (const vector& v : facePoly(S.start+i)) srcW[i].push_back(v);
        for (label j = 0; j < T.size; ++j)
            for (const vector& v : facePoly(T.start+j)) tgtW[j].push_back(mapTtoS(v));

        // Unit normals on the source side (the target's mapped to it), for the pair normal below.
        auto unitN = [&](label f) {
            const vector& Sf = g.Sf()[f]; const scalar a = g.magSf()[f];
            return a > 0 ? vector{Sf.x/a, Sf.y/a, Sf.z/a} : vector{0,0,0};
        };
        std::vector<vector> nSrc(S.size), nTgt(T.size);
        for (label i = 0; i < S.size; ++i) nSrc[i] = unitN(S.start+i);
        // A normal maps like a DIRECTION, not a point: unchanged under translation, rotated under a
        // rotational transform (the same forwardT the delta uses below).
        for (label j = 0; j < T.size; ++j)
        {
            const vector nj = unitN(T.start+j);
            nTgt[j] = ai.translational ? nj : dot(nj, transpose(ai.forwardT));
        }

        // OF faceAreaWeightAMI::calcInterArea (faceAreaWeightAMI.C:402-410):
        //     n = -srcNormal (+/-) tgtNormal ;  project the pair along n/|n|
        // A projection direction per PAIR, not one average plane for the whole patch. That is what
        // makes a curved interface work: each pair is locally planar even when the patch is a
        // cylinder, whose average normal nearly cancels. brae used a single source-patch average
        // plane, which collapsed opposite sides of the cylinder onto each other and lost 55% of the
        // face coverage on pimpleFoam/RAS/rotatingFanInRoom.
        auto projectPair = [&](const std::vector<vector>& poly, const vector& e1, const vector& e2,
                               const vector& orig)
        {
            std::vector<vec2> out;
            out.reserve(poly.size());
            for (const vector& v : poly)
            {
                const vector d{v.x-orig.x, v.y-orig.y, v.z-orig.z};
                out.push_back({ dot(d,e1), dot(d,e2) });
            }
            orientCCW(out);
            return out;
        };

        // ---- cyclicPeriodicAMI: the transform that TILES this interface ----------------------------
        // OF cyclicPeriodicAMIPolyPatch::resetAMI. A periodic AMI's two sides need not span the same
        // sector -- oscillatingInletPeriodicAMI2D pairs 40 faces against 96, axialTurbine a guide-vane
        // passage against a runner passage -- so the overlap of the two patches AS THEY SIT covers only
        // part of each. OF closes the gap by applying the transform of a NAMED periodic patch to one
        // side repeatedly, accumulating the extra overlaps, until the weights sum to 1.
        //
        // Coupling a source face to the periodic IMAGE of a target face is exact, not an approximation:
        // the transform is a symmetry of the solution, so the value at the image IS the value at the
        // face. That is the whole basis of the patch type.
        //
        // The periodic patch is itself a coupled pair (`cyclic` in the 2D case, a rotational `cyclicAMI`
        // in the turbine), and its transform is read the same way this function reads its own.
        bool     perRot = false;
        vector   perSep{0,0,0};
        tensor   perT{1,0,0,0,1,0,0,0,1}, perTinv{1,0,0,0,1,0,0,0,1};
        vector   perCtr{0,0,0};
        label    perMaxIter = 0;
        scalar   perTol = scalar(1e-4);
        if (isPeriodic)
        {
            const std::string ppName = pinfo[pi].periodicPatch;
            const auto pit = nameToIdx.find(ppName);
            if (ppName.empty() || pit == nameToIdx.end())
                throw std::runtime_error(
                    "brae: cyclicPeriodicAMI patch '" + fvp[pi].name + "' names periodicPatch '" + ppName
                    + "', which is not a patch of this mesh. Without its transform the interface cannot be "
                      "tiled and the two sides would coupled only where they happen to overlap.");
            const label ppi = pit->second;
            const auto pnit = nameToIdx.find(pinfo[ppi].neighbourPatch);
            if (pnit == nameToIdx.end())
                throw std::runtime_error(
                    "brae: periodicPatch '" + ppName + "' has no neighbourPatch, so it defines no transform.");
            const FvPatch& P = fvp[ppi];
            const FvPatch& Q = fvp[pnit->second];
            perMaxIter = pinfo[pi].maxIter;
            perTol     = pinfo[pi].matchTolerance;
            perRot     = (pinfo[ppi].transform == "rotational");
            if (perRot)
            {
                const vector ax = pinfo[ppi].rotationAxis / mag(pinfo[ppi].rotationAxis);
                perCtr = pinfo[ppi].rotationCentre;
                // The pitch angle, from the two halves' MEAN azimuth about the axis. Face 0 of each half
                // would do only if the two were ordered to match; a cyclicAMI periodic patch is under no
                // such obligation, and here it is exactly the patch that is not.
                auto meanPerp = [&](const FvPatch& X)
                {
                    vector acc{0,0,0};
                    for (label i = 0; i < X.size; ++i)
                    {
                        const vector r = g.Cf()[X.start+i] - perCtr;
                        const vector perp = r - dot(r, ax)*ax;
                        const scalar mp = mag(perp);
                        if (mp > scalar(0)) acc += perp/mp;
                    }
                    return acc;
                };
                const vector u = meanPerp(P), v = meanPerp(Q);
                const scalar ang = std::atan2(dot(cross(u, v), ax), dot(u, v));
                perT    = rotationTensor(ax, ang);
                perTinv = rotationTensor(ax, -ang);
            }
            else
            {
                vector cP{0,0,0}, cQ{0,0,0};
                for (label i = 0; i < P.size; ++i) cP += g.Cf()[P.start+i];
                for (label j = 0; j < Q.size; ++j) cQ += g.Cf()[Q.start+j];
                perSep = cQ/(scalar)Q.size - cP/(scalar)P.size;
            }
        }
        // Move a point by k periods (k may be negative). Rotational transforms are about the periodic
        // patch's own axis and centre, which need not be the AMI's.
        auto periodShift = [&](const vector& p, label k) -> vector
        {
            vector q = p;
            for (label t = 0; t < k; ++t)
                q = perRot ? (dot(q - perCtr, transpose(perT)) + perCtr) : (q + perSep);
            for (label t = 0; t > k; --t)
                q = perRot ? (dot(q - perCtr, transpose(perTinv)) + perCtr) : (q - perSep);
            return q;
        };

        // faceAreaWeightAMI: per src face, overlap-area against every tgt face (brute force; OF uses an advancing
        // front, same result). weight = overlap/srcMagSf; weightsSum = coverage fraction.
        ai.ownCell = S.faceCells;
        ai.srcOffset.assign(S.size + 1, 0);
        std::vector<std::vector<std::pair<label,scalar>>> stencil(S.size);   // per src face: (tgtFace j, weight)
        std::vector<scalar> tgtCov(T.size, scalar(0));                       // per tgt face: coverage fraction
        // One pass of the overlap sweep against a given set of (already source-frame) target polygons.
        // ACCUMULATES into stencil/tgtCov, which is what makes the periodic tiling a repeat of the same
        // computation rather than a special case of it -- OF's AMIInterpolation::append concatenates the
        // per-image addressing and adds the weight sums, exactly this.
        // BRAE_AMI_BRUTE=1 restores the exhaustive sweep. It exists so the broad phase can be proved
        // to change nothing: the two must produce identical weights, and a search that quietly drops a
        // pair would otherwise show up only as a slightly-wrong answer somewhere downstream.
        static const bool bruteForceEnv = std::getenv("BRAE_AMI_BRUTE") != nullptr;
        // THE BROAD PHASE ASSUMES THE TWO PATCHES ARE NOMINALLY COINCIDENT, and enforces it rather than
        // hoping. Its padding tolerates a normal offset of about one face -- enough for the radius
        // difference on a curved interface, which is what a real AMI has. It cannot tolerate patches
        // that are genuinely far apart, because the narrow phase PROJECTS the separation away and would
        // still find them overlapping.
        //
        // A real cyclicAMI/ACMI pair is coincident by construction, so this holds. What does not hold is
        // a pair whose sides are a domain apart and rely entirely on the projection -- which is exactly
        // what the periodic-AMI unit fixture builds, and refusing to notice would have made the grid
        // silently drop every pair. So: measure the offset, and fall back to the exhaustive sweep when
        // the assumption fails. Correct always, fast whenever the geometry allows it.
        bool bruteForce = bruteForceEnv;
        if (!bruteForce)
        {
            vector cs{0,0,0}, ct{0,0,0};
            for (label i = 0; i < S.size; ++i) cs += g.Cf()[S.start+i];
            for (label j = 0; j < T.size; ++j) ct += g.Cf()[T.start+j];
            if (S.size && T.size)
            {
                const vector d = mapTtoS(ct/(scalar)T.size) - cs/(scalar)S.size;
                scalar meanFace = 0;
                for (label i = 0; i < S.size; ++i) meanFace += std::sqrt(g.rawMagSf(S.start+i));
                meanFace /= (scalar)S.size;
                // One face: the same tolerance the padding provides. Beyond it the grid cannot be
                // trusted to find the pair, so the exhaustive sweep takes over.
                if (mag(d) > meanFace) bruteForce = true;
            }
        }
        std::vector<label> cand;
        std::vector<char>  seen(static_cast<std::size_t>(T.size), 0);
        auto accumulate = [&](const std::vector<std::vector<vector>>& tgtPoly)
        {
        ami_detail::FaceGrid grid;
        if (!bruteForce) grid.build(tgtPoly);
        for (label i = 0; i < S.size; ++i)
        {
            const scalar srcArea = g.rawMagSf(S.start+i);   // RAW: see the note on the signature
            // Candidate target faces: those whose bounding box can touch this source face's. Everything
            // else cannot overlap by any amount, so skipping it is exact, not approximate.
            if (!bruteForce)
            {
                vector a{ 1e300, 1e300, 1e300}, b{-1e300,-1e300,-1e300};
                for (const vector& v : srcW[i])
                {
                    a.x = std::min(a.x, v.x); a.y = std::min(a.y, v.y); a.z = std::min(a.z, v.z);
                    b.x = std::max(b.x, v.x); b.y = std::max(b.y, v.y); b.z = std::max(b.z, v.z);
                }
                grid.query(a, b, cand, seen);
            }
            const label nCand = bruteForce ? T.size : (label)cand.size();
            for (label c = 0; c < nCand; ++c)
            {
                const label j = bruteForce ? c : cand[(std::size_t)c];
                // per-pair projection normal (OF: -nSrc + nTgt, reversed target subtracts)
                // OF: n = -srcNormal + tgtNormal. The two AMI patches face EACH OTHER, so the mapped
                // target normal opposes the source one and the sum is ~ -2*nSrc -- a well-defined
                // direction. It degenerates only if the two point the same way, which the magN guard
                // below rejects rather than projecting onto a near-zero direction.
                vector n{-nSrc[i].x + nTgt[j].x, -nSrc[i].y + nTgt[j].y, -nSrc[i].z + nTgt[j].z};
                const scalar magN = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
                if (magN <= 1e-150) continue;             // OF: ROOTVSMALL -> skip the pair
                n = vector{n.x/magN, n.y/magN, n.z/magN};

                // an orthonormal basis in the plane perpendicular to n
                const vector a = (std::fabs(n.x) < 0.9) ? vector{1,0,0} : vector{0,1,0};
                vector e1{a.y*n.z - a.z*n.y, a.z*n.x - a.x*n.z, a.x*n.y - a.y*n.x};
                const scalar m1 = std::sqrt(e1.x*e1.x + e1.y*e1.y + e1.z*e1.z);
                if (m1 <= 1e-150) continue;
                e1 = vector{e1.x/m1, e1.y/m1, e1.z/m1};
                const vector e2{n.y*e1.z - n.z*e1.y, n.z*e1.x - n.x*e1.z, n.x*e1.y - n.y*e1.x};

                const vector orig = srcW[i][0];
                const scalar ov = overlapArea(projectPair(srcW[i], e1, e2, orig),
                                              projectPair(tgtPoly[j], e1, e2, orig));
                if (ov > 1e-14 * srcArea)
                {
                    stencil[i].push_back({ j, ov / srcArea });
                    const scalar ta = g.rawMagSf(T.start+j);
                    if (ta > scalar(0)) tgtCov[j] += ov / ta;
                }
            }
        }
        };
        accumulate(tgtW);                       // the untransformed overlap: OF's first AMI.calculate()

        // ---- the tiling loop, OF cyclicPeriodicAMIPolyPatch::resetAMI ------------------------------
        // Relative offset k means "the source displaced by k periods relative to the target"; OF gets it
        // by transforming one side's POINTS and pairing against the other side's originals, alternating
        // which side moves. Here the source stays put and the target images move the other way, which is
        // the same relative geometry and needs only one polygon set rebuilt per image.
        //
        // OF's direction logic is reproduced rather than simplified: it starts outward, and FLIPS as soon
        // as a step stops paying (srcSumDiffNew < srcSumDiff, or no gain at all). That is what makes the
        // images come out as 0, +1, -1, +2, ... instead of marching off in one direction -- and on a
        // patch whose neighbour lies on the other side, marching the wrong way finds nothing at all and
        // burns every iteration before the loop gives up.
        if (isPeriodic && perMaxIter > 0)
        {
            auto meanCov = [&](const std::vector<std::vector<std::pair<label,scalar>>>& st)
            {
                scalar acc = 0;
                for (const auto& v : st) { scalar t = 0; for (const auto& e : v) t += e.second; acc += t; }
                return st.empty() ? scalar(1) : acc/(scalar)st.size();
            };
            auto meanTgt = [&]()
            {
                scalar acc = 0;
                for (const scalar c : tgtCov) acc += c;
                return tgtCov.empty() ? scalar(1) : acc/(scalar)tgtCov.size();
            };
            // THE IMAGE SET IS SYMMETRIC (+k AND -k together), which is NOT how OF walks it. OF builds
            // the AMI once, on the OWNER side only, and the neighbour direction reuses that same
            // addressing transposed -- so its one-directional search can stop the moment the owner is
            // covered. brae builds the two directions INDEPENDENTLY (that is what makes a conforming
            // cyclicAMI work without an owner/neighbour concept), and two independent adaptive searches
            // do not stop at the same place: measured here, ami1 closed after 2 images and ami2 after 1.
            //
            // Different image sets mean the two directions are not transposes, so the interface block of
            // the pressure matrix is ASYMMETRIC -- and brae solves an interface pressure system with
            // Jacobi-PCG, which requires symmetry. It did not merely lose accuracy: every solve ran to
            // its 50-iteration cap and the FINAL residual came out above the initial one (1.00 -> 5.14
            // on step 1). Continuity then never closed: contGlobal pinned at -0.33 for the whole run,
            // 0.1 of mass entering and 0.013 leaving.
            //
            // Adding images in +/- pairs costs nothing when one side of the pair does not overlap (it
            // contributes no stencil entries) and makes the two directions transposes by construction,
            // because image +k of the target against the source is image -k of the source against the
            // target.
            scalar srcSum = meanCov(stencil), tgtSum = meanTgt();
            const std::pair<label, label> pairKey(std::min(ai.patch, ai.nbrPatch),
                                                  std::max(ai.patch, ai.nbrPatch));
            const auto known = periodicImages.find(pairKey);
            const label fixedK = (known == periodicImages.end()) ? -1 : known->second;
            label  k = 0, iter = 0;
            while (iter < perMaxIter
                && (fixedK >= 0 ? (k < fixedK)
                                : ((scalar(1) - srcSum > perTol) || (scalar(1) - tgtSum > perTol))))
            {
                ++k;
                for (const label kk : { -k, k })
                {
                    std::vector<std::vector<vector>> shifted(T.size);
                    for (label j = 0; j < T.size; ++j)
                    {
                        shifted[j].reserve(tgtW[j].size());
                        for (const vector& v : tgtW[j]) shifted[j].push_back(periodShift(v, kk));
                    }
                    accumulate(shifted);
                }
                srcSum = meanCov(stencil);
                tgtSum = meanTgt();
                ++iter;
            }
            if (fixedK < 0) periodicImages[pairKey] = k;   // this direction sets the pair's image count
            if (std::getenv("BRAE_AMI_REPORT"))
                std::printf("AMI periodic: %s <-> %s  images:+/-%d  iters:%d  srcSum:%g tgtSum:%g\n",
                            S.name.c_str(), T.name.c_str(), (int)k, (int)iter,
                            (double)srcSum, (double)tgtSum);
        }

        for (label i = 0; i < S.size; ++i)
        {
            scalar s = 0;
            for (auto& e : stencil[i]) s += e.second;
            ai.weightsSum.push_back(s);
            ai.srcOffset[i+1] = ai.srcOffset[i] + (label)stencil[i].size();
        }
        // THE ACMI MASK -- OF cyclicACMIPolyPatch.C:348, srcMask_ = clamp(AMI.srcWeightsSum(), 0, 1).
        // The coverage fraction IS the mask: it decides how each face's area splits between the coupled
        // patch and its coincident nonOverlapPatch wall. Computed for every interface (it is just the
        // clamped coverage) but only meaningful, and only used, for ACMI. Taken from the coverage BEFORE
        // the ACMI re-normalisation below, which is the whole point of the mask.
        ai.mask.reserve(ai.weightsSum.size());
        for (const scalar w : ai.weightsSum)
            ai.mask.push_back(w < scalar(0) ? scalar(0) : (w > scalar(1) ? scalar(1) : w));

        // ACMI ONLY -- RE-NORMALISE THE WEIGHTS, in OF's own words (cyclicACMIPolyPatch.C:264):
        //
        //     "Re-normalise the weights since the effect of overlap is already accounted for in the area"
        //         for (scalar& w : wghts) { w /= sum; }
        //         sum = 1.0;
        //
        // and that is the LAST step of cyclicACMIPolyPatch::scalePatchFaceAreas, i.e. it happens only
        // once the coupled Sf has been multiplied by the mask. The two carry the SAME coverage, so a
        // partially covered face that keeps both applies it TWICE and transmits mask^2 of its flux.
        //
        // The note on this function ("brae divides by the face area and never renormalises, so it is on
        // OF's ACMI branch already") is right about AMIInterpolation::normaliseWeights and stops one
        // call too early: cyclicACMI overrides that branch immediately afterwards. So does the log
        // evidence -- OF prints sum(weights) from inside normaliseWeights, BEFORE the override, which is
        // why the printed average is the coverage on a run whose solved weights are 1.
        //
        // Only BLENDED faces move (0 < mask < 1). A fully covered face already sums to 1, and an
        // uncovered one has an empty stencil, which OF skips (`if (wghts.size())`) and so does this --
        // leaving its weightsSum at 0 rather than inventing a 1 for a face with nothing to read.
        //
        // BEFORE the geometry loop, not after it. The loop below AMI-interpolates the neighbour delta
        // with these same weights (OF cyclicAMIFvPatch::makeDeltaCoeffs, which runs on the already
        // re-normalised weights), so normalising afterwards would leave a blended face with a delta
        // short by its coverage -- deltaCoeffs 72.7 instead of 53.3 on the fixture below, and a face
        // interpolation weight of 0.09 instead of 0.33.
        //
        // Measured on pimpleFoam/RAS/oscillatingInletACMI2D (static mesh, 2 blended target faces of
        // 136): those two faces carried 0.118 of a full face's flux against OpenFOAM's 0.201, and the
        // resulting 3.7e-04 shortfall -- 0.9% of the 0.04 through the interface -- set up a linear
        // pressure ramp of 7 across the downstream duct and ~10% velocity error at the interface.
        if (ai.acmi)
        {
            for (label i = 0; i < S.size; ++i)
            {
                const scalar s = ai.weightsSum[i];
                if (stencil[i].empty() || !(s > scalar(0))) continue;
                for (auto& e : stencil[i]) e.second /= s;
                ai.weightsSum[i] = scalar(1);
            }
        }

        // per-source-face geometry: the neighbour delta is AMI-interpolated (OF cyclicAMIFvPatch::makeDeltaCoeffs).
        // transform of a nbr DELTA to the src side: identity (translational) or forwardT (rotational).
        auto rotDelta = [&](const vector& d) -> vector
        {
            return ai.translational ? d : dot(d, transpose(ai.forwardT));
        };
        for (label i = 0; i < S.size; ++i)
        {
            const vector Sfi = g.Sf()[S.start+i];
            const scalar msf = g.magSf()[S.start+i];
            const vector nf = Sfi/msf;
            const vector patchD = g.Cf()[S.start+i] - g.C()[S.faceCells[i]];
            vector nbrDint{0,0,0};
            scalar dni = 0;
            for (const auto& e : stencil[i])
            {
                const label j = e.first;
                const vector nbrD = g.Cf()[T.start+j] - g.C()[T.faceCells[j]];
                const vector nfn  = g.Sf()[T.start+j] / g.magSf()[T.start+j];     // neighbour-side unit normal
                nbrDint += e.second * rotDelta(nbrD);                             // Sum w . transform(nbrD)  (for delta)
                dni     += e.second * dot(nfn, nbrD);                             // Sum w . (nfn & nbrD)  (OF makeWeights)
            }
            const vector delta = patchD - nbrDint;
            const scalar di = dot(nf, patchD);
            ai.magSf.push_back(msf);
            ai.Sf.push_back(Sfi);
            ai.weights.push_back(std::fabs(di + dni) > 1e-300 ? dni/(di + dni) : 0.5);
            const scalar dc = 1.0 / std::fmax(dot(nf, delta), 0.05 * mag(delta));
            ai.deltaCoeffs.push_back(dc);
            ai.corrVec.push_back(nf - delta * dc);
            ai.dOwn.push_back(patchD);   // own-side linearUpwind reconstruction delta (Cf - C_own)
            ai.delta.push_back(delta);   // fvPatch::delta(), the TVD limiter's d
            for (const auto& e : stencil[i])
            {
                const label j = e.first;
                ai.nbrCell.push_back(T.faceCells[j]);
                ai.weight.push_back(e.second);
                ai.dNbr.push_back(g.Cf()[T.start+j] - g.C()[T.faceCells[j]]);   // nbr-side delta (Cf_tgt - C_nbr), un-rotated
            }
        }
        // COVERAGE CHECK -- for cyclicAMI ONLY.
        //
        // A cyclicAMI source face must be FULLY covered by target faces (weightsSum ~ 1) or it loses
        // that fraction of its flux across the interface. That is a mass sink: continuity grows every
        // step and the run diverges. Measured on pimpleFoam/RAS/rotatingFanInRoom, on a STATIC mesh,
        // back when brae projected both patches onto the source patch's average plane:
        //
        //     planarity |sum Sf|/sum|Sf| = 0.4206      (1.0 = planar)
        //     mean coverage weightsSum   = 0.4492      1481 of 14080 faces at ~0
        //
        // That projection has since been replaced by OF's per-source/target-pair one
        // (faceAreaWeightAMI.C:402-410), which took the same case to mean coverage 1.0001. The check
        // stays as the guard that would catch any future regression of the same kind.
        //
        // NOT APPLIED TO cyclicACMI. There, partial coverage is the entire point: a face that has slid
        // off its neighbour is SUPPOSED to read ~0 and hand its area to the nonOverlapPatch wall. OF's
        // own report on oscillatingInletACMI2D at t=0.5 is 19 uncovered, 1 blended, 20 covered -- mean
        // coverage ~0.5, which this check would refuse outright. Refusing a correct ACMI interface for
        // looking like a broken AMI one would be exactly backwards.
        //
        // REFUSED, not warned. brae was building unusable weights and running to a confident wrong
        // answer, which is the one outcome this codebase does not accept.
        if (!ai.acmi)
        {
            const std::size_t n = ai.weightsSum.size();
            if (n)
            {
                scalar sum = 0, lo = ai.weightsSum[0];
                std::size_t under = 0;
                for (const scalar w : ai.weightsSum)
                {
                    sum += w;
                    lo = std::min(lo, w);
                    if (w < scalar(0.99)) ++under;
                }
                const scalar mean = sum/static_cast<scalar>(n);
                // 0.99 would reject on round-off; 0.9 still catches the 0.45 failure by a wide margin
                // while leaving a genuinely conforming interface alone.
                if (mean < scalar(0.9))
                {
                    char buf[512];
                    std::snprintf(buf, sizeof(buf),
                        "brae: cyclicAMI '%s' has mean face coverage %.4f (min %.4f); %zu of %zu source "
                        "faces are less than 99%% covered, so that fraction of their flux is lost across "
                        "the interface -- a mass sink that grows the continuity error every step until the "
                        "run diverges. Refused rather than solved with a leaking interface. (A partially "
                        "overlapping interface is what cyclicACMI is for, and is not refused.)",
                        S.name.c_str(), (double)mean, (double)lo, under, n);
                    throw std::runtime_error(buf);
                }
            }
        }
        // The same report OF prints when it constructs an AMI ("AMI: Patch source sum(weights)
        // min:.. max:.. average:.."), so brae's coverage can be diffed against OpenFOAM's own log line
        // for line rather than inferred from a diverging continuity error. Env-gated: OF prints it
        // unconditionally, but brae's logs are already compared byte-wise by the test harness.
        if (std::getenv("BRAE_AMI_REPORT") && !ai.weightsSum.empty())
        {
            scalar lo = ai.weightsSum[0], hi = ai.weightsSum[0], sum = 0;
            for (const scalar w : ai.weightsSum) { lo = std::min(lo, w); hi = std::max(hi, w); sum += w; }
            std::fprintf(stderr,
                         "AMI: source:%s (%d faces) target:%s (%d faces) sum(weights) min:%g max:%g average:%g%s\n",
                         S.name.c_str(), (int)S.size, T.name.c_str(), (int)T.size,
                         (double)lo, (double)hi, (double)(sum/(scalar)ai.weightsSum.size()),
                         ai.acmi ? "  [ACMI]" : "");
        }
        out.push_back(std::move(ai));
    }
    return out;
}

} // namespace brae
