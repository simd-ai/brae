// Device AMG preconditioner: host agglomeration (static) + device Galerkin and V-cycle smoothing.
// Used by deviceAMGPCG. Feature flags live in one BRAE_AMG_* block at the top of the anonymous namespace.
#include "device_amg.cuh"
#include "device_blas.cuh"
#include "device_ldu.cuh"       // deviceParallelAmul (halo-coupled matvec) for the distributed whole-loop graph PCG
#include "device_halo.cuh"      // DeviceHalo + its DeviceReducer (on-stream NVSHMEM reduce)
#include "device_amg_detail.cuh"  // shared inline primitives: safeDiag/warpReduceSum/blockDot + env-flag readers + AMG constants
#include "device_amg_coarse.cuh"  // deviceCoarsePCG/JacobiSingleBlock (internal coarse solvers, defined in device_amg_coarse.cu)
#include "amg_kernels.cuh"        // mixed-precision V-cycle template kernels (zeroT/smoothT/residualT/gsColorT/restrictT/prolongT)
#include "device_amg_internal.cuh"// shared AMG-core infra: LduF/lduF/cast_ (FP32 stack), Coloring/greedyColor/gsSweep, gsScaleInvK
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace cg = cooperative_groups;

namespace brae {

// V-cycle entry points defined in device_amg_vcycle.cu (external linkage); the PCG drivers below call them.
void vcycleAt(int g, AMGData& amg, const DeviceLduView& Ag, const DeviceBuffer<scalar>& bg, DeviceBuffer<scalar>& xg);
void vcycleAtF(int g, AMGData& amg, const DeviceLduView& topoG, const LduF& Ag, const float* bg, float* xg);
void amgCastFP32(AMGData& amg, const DeviceLduView& A);

AMGGraphCache::~AMGGraphCache()
{
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
}

PCGGraphCache::~PCGGraphCache()
{
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
}

namespace {
// General Galerkin RAP scatter (BRAE_AMG_SA): A_c[dst] += w * A_fine[src] over a precomputed triple list.
// srcKind/dstKind: 0=diag 1=upper 2=lower. One thread per triple into the zeroed coarse LDU; the SA twin of galDiagK/galFaceK.
__global__
void rapScatterK(
    int nT,
    const label* __restrict__ srcKind,
    const label* __restrict__ srcIdx,
    const scalar* __restrict__ w,
    const label* __restrict__ dstKind,
    const label* __restrict__ dstIdx,
    const scalar* __restrict__ fineDiag,
    const scalar* __restrict__ fineUp,
    const scalar* __restrict__ fineLo,
    scalar* __restrict__ cDiag,
    scalar* __restrict__ cUp,
    scalar* __restrict__ cLo)
{
    const int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nT) return;
    const int sk = srcKind[t], si = srcIdx[t];
    const scalar src = (sk==0) ? fineDiag[si] : (sk==1) ? fineUp[si] : fineLo[si];
    const scalar v = w[t]*src;
    const int dk = dstKind[t], di = dstIdx[t];
    if (dk==0) atomicAdd(&cDiag[di], v);
    else if (dk==1) atomicAdd(&cUp[di], v);
    else atomicAdd(&cLo[di], v);
}
// Injection Galerkin (default path): coarse LDU from fine diag/upper/lower via faceRestrict/faceFlip.
//
// DETERMINISM. These used to SCATTER with atomicAdd -- cDiag[map[c]] += fineDiag[c] and
// cUp[fr[f]] += up[f] -- and the contention is heavy by construction: agglomeration exists precisely to
// map many fine entities onto one coarse entity. The summation order was therefore whatever order the
// blocks happened to finish in, so the COARSE OPERATOR itself differed bit-for-bit between two runs of
// the same binary on the same case. That changes the preconditioner, which changes the Krylov path,
// which stops the pressure solve at a different iterate. Measured on pitzDaily/kEpsilon: two identical
// 1-iteration runs already disagreed by 1.9e-8, growing to 1.6e-6 by iteration 2, 1.4e-3 by 5 and
// 3.6e-2 by 20 -- the SIMPLE loop amplifies the seed.
//
// The cure is to GATHER instead. The agglomeration is static for the life of the mesh (that is why the
// AMG hierarchy can be cached at all), so the inverse maps -- which fine cells feed a coarse cell, which
// fine faces feed a coarse face -- are built once on the host as CSR lists and reused by every solve.
// One thread per COARSE entity then sums its own contributions in ascending fine index, a fixed order,
// and WRITES the result rather than accumulating into it. That also removes the need to pre-zero the
// coarse arrays on this path.
//
// Cost: three label arrays per level, built once. No per-solve work is added.
__global__
void galDiagGatherK(
    int nCoarse,
    const label* __restrict__ cellStart,   // [nCoarse+1] into cellList
    const label* __restrict__ cellList,    // fine cells feeding this coarse cell, ascending
    const label* __restrict__ dfaceStart,  // [nCoarse+1] into dfaceList
    const label* __restrict__ dfaceList,   // fine faces INTERNAL to this agglomerate (faceRestrict < 0)
    const scalar* __restrict__ fineDiag,
    const scalar* __restrict__ up,
    const scalar* __restrict__ lo,
    scalar* __restrict__ cDiag)
{
    const int ci = blockIdx.x*blockDim.x + threadIdx.x;
    if (ci >= nCoarse) return;
    scalar s = 0.0;
    for (label k = cellStart[ci]; k < cellStart[ci+1]; ++k) s += fineDiag[cellList[k]];
    // A face interior to an agglomerate contributes both its off-diagonals to the coarse DIAGONAL.
    for (label k = dfaceStart[ci]; k < dfaceStart[ci+1]; ++k)
    {
        const label f = dfaceList[k];
        s += up[f] + lo[f];
    }
    cDiag[ci] = s;
}
__global__
void galFaceGatherK(
    int nCoarseFaces,
    const label* __restrict__ faceStart,   // [nCoarseFaces+1] into faceList/faceFlipList
    const label* __restrict__ faceList,    // fine faces feeding this coarse face, ascending
    const label* __restrict__ faceFlipList,
    const scalar* __restrict__ up,
    const scalar* __restrict__ lo,
    scalar* __restrict__ cUp,
    scalar* __restrict__ cLo)
{
    const int cf = blockIdx.x*blockDim.x + threadIdx.x;
    if (cf >= nCoarseFaces) return;
    scalar u = 0.0, l = 0.0;
    for (label k = faceStart[cf]; k < faceStart[cf+1]; ++k)
    {
        const label f = faceList[k];
        // flip means the fine face's owner/neighbour orientation is reversed relative to the coarse
        // face's, so upper and lower swap -- exactly as the scatter version did.
        if (!faceFlipList[k]) { u += up[f]; l += lo[f]; }
        else                  { u += lo[f]; l += up[f]; }
    }
    cUp[cf] = u;
    cLo[cf] = l;
}


} // namespace



namespace {

// Invert map/faceRestrict into the CSR gather lists the deterministic Galerkin kernels read.
//
// Counting sort, so each coarse entity's list comes out in ASCENDING fine index -- a fixed traversal
// order, which is the whole point. Runs once per level at hierarchy build; the AMG cache stores the
// agglomeration this is derived from, so a cached hierarchy rebuilds these for free.
template <class AgglomT>
void buildGalerkinGather(AMGLevel& L, const AgglomT& a, int nFine)
{
    const int nC = a.nCoarse, nCF = a.nCoarseFaces;
    const int nFaces = static_cast<int>(a.faceRestrict.size());

    // coarse cell <- fine cells
    std::vector<label> cs(nC + 1, 0), cl(nFine);
    for (int c = 0; c < nFine; ++c) ++cs[a.map[c] + 1];
    for (int i = 0; i < nC; ++i) cs[i+1] += cs[i];
    {
        std::vector<label> at(cs.begin(), cs.end() - 1);
        for (int c = 0; c < nFine; ++c) cl[at[a.map[c]]++] = c;
    }

    // coarse cell <- fine faces interior to the agglomerate (faceRestrict < 0), and
    // coarse face  <- fine faces (faceRestrict >= 0), with the flip carried alongside.
    std::vector<label> ds(nC + 1, 0), fs(nCF + 1, 0);
    for (int f = 0; f < nFaces; ++f)
    {
        const label cf = a.faceRestrict[f];
        if (cf >= 0) ++fs[cf + 1];
        else         ++ds[(-1 - cf) + 1];
    }
    for (int i = 0; i < nC;  ++i) ds[i+1] += ds[i];
    for (int i = 0; i < nCF; ++i) fs[i+1] += fs[i];

    std::vector<label> dl(ds[nC]), fl(fs[nCF]), ff(fs[nCF]);
    {
        std::vector<label> dat(ds.begin(), ds.end() - 1), fat(fs.begin(), fs.end() - 1);
        for (int f = 0; f < nFaces; ++f)
        {
            const label cf = a.faceRestrict[f];
            if (cf >= 0)
            {
                const label k = fat[cf]++;
                fl[k] = f;
                ff[k] = a.faceFlip[f];
            }
            else dl[dat[-1 - cf]++] = f;
        }
    }

    L.galCellStart.copyFrom(cs);   L.galCellList.copyFrom(cl);
    L.galDFaceStart.copyFrom(ds);  L.galDFaceList.copyFrom(dl);
    L.galFaceStart.copyFrom(fs);   L.galFaceList.copyFrom(fl);
    L.galFaceFlipList.copyFrom(ff);
}

// One pairwise agglomeration step (host): merge cells of a grid (owner/nei/faceWeights, nC cells) into a coarse
// grid. Returns the cell->coarse map, the coarse addressing (cOwn/cNei + gather starts), the face restriction,
// and the carried coarse face weights (sum of the agglomerated fine face weights) for the NEXT level.
struct Agglom
{
    int nCoarse = 0, nCoarseFaces = 0;
    std::vector<label> map, cOwn, cNei, cOS, cLS, cLosort, faceRestrict, faceFlip;
    std::vector<scalar> coarseFaceWeights;
};
Agglom agglomerate(
    const std::vector<label>& owner,
    const std::vector<label>& nei,
    const std::vector<scalar>& fw,
    int nC)
{
    const int nFaces = static_cast<int>(owner.size());
    std::vector<label> nNbr(nC, 0);
    for (int f = 0; f < nFaces; ++f)
    {
        nNbr[owner[f]]++;
        nNbr[nei[f]]++;
    }
    std::vector<label> off(nC+1, 0);
    for (int c = 0; c < nC; ++c)
        off[c+1]=off[c]+nNbr[c];
    std::vector<label> cellFaces(off[nC]);
    std::fill(nNbr.begin(), nNbr.end(), 0);
    for (int f = 0; f < nFaces; ++f)
    {
        cellFaces[off[owner[f]]+nNbr[owner[f]]++]=f;
        cellFaces[off[nei[f]]+nNbr[nei[f]]++]=f;
    }
    // Optional strength-of-connection filter (BRAE_AMG_SOC=beta; 0 disables). A face is strong when
    // fw[f] >= beta*sqrt(D[o]*D[m]), D = row-sum of fw; matching only strong faces gives semi-coarsening along
    // the strong direction on anisotropic meshes (sweet spot beta~0.05).
    static const scalar socBeta = [](){ const char* e = std::getenv("BRAE_AMG_SOC"); return e ? std::atof(e) : 0.0; }();
    std::vector<scalar> D;
    if (socBeta > 0.0)
    {
        D.assign(nC, 0.0);
        for (int f = 0; f < nFaces; ++f)
        {
            D[owner[f]] += fw[f];
            D[nei[f]] += fw[f];
        }
    }
    auto strongF = [&](label f){ return socBeta <= 0.0 || fw[f] >= socBeta*std::sqrt(D[owner[f]]*D[nei[f]]); };
    std::vector<label> map(nC, -1);
    label nCoarse = 0;
    for (int c = 0; c < nC; ++c)
    {
        if (map[c] >= 0) continue;
        int mf = -1;
        scalar mw = -1e300;
        for (label o = off[c]; o < off[c+1]; ++o)
        {
            const label f = cellFaces[o];
            if (map[owner[f]]<0 && map[nei[f]]<0 && strongF(f) && fw[f]>mw)
            {
                mf=f;
                mw=fw[f];
            }
        }
        if (mf >= 0)
        {
            map[owner[mf]]=nCoarse;
            map[nei[mf]]=nCoarse;
            ++nCoarse;
        }
        else
        {
            int cm=-1;
            scalar cmw=-1e300;
            for (label o=off[c]; o<off[c+1]; ++o)
            {
                const label f=cellFaces[o];
                if (fw[f]>cmw)
                {
                    cm=f;
                    cmw=fw[f];
                }
            }
            if (cm>=0) map[c]=std::max(map[owner[cm]], map[nei[cm]]);
        }
    }
    for (int c = 0; c < nC; ++c)
        if (map[c] < 0) map[c] = nCoarse++;
    // Coarse-face dedup: unique (min,max) coarse-cell pairs in owner-sorted order (the coarse SpMV requires it).
    // sort+unique on a flat vector rather than a std::map: same ordering, cache-friendly, lookups are a binary search.
    std::vector<std::pair<label,label>> pr;
    pr.reserve(nFaces);
    for (int f = 0; f < nFaces; ++f)
    {
        const label co=map[owner[f]], cn=map[nei[f]];
        if (co!=cn) pr.emplace_back(std::min(co,cn), std::max(co,cn));
    }
    std::sort(pr.begin(), pr.end());
    pr.erase(std::unique(pr.begin(), pr.end()), pr.end());
    const int nCF = static_cast<int>(pr.size());
    std::vector<label> cOwn(nCF), cNei(nCF), faceRestrict(nFaces), faceFlip(nFaces);
    for (int i = 0; i < nCF; ++i)
    {
        cOwn[i]=pr[i].first;
        cNei[i]=pr[i].second;
    }
    auto findCF = [&](label a, label b){ return static_cast<label>(std::lower_bound(pr.begin(), pr.end(), std::make_pair(a,b)) - pr.begin()); };
    std::vector<scalar> cfw(nCF, 0.0);
    for (int f = 0; f < nFaces; ++f)
    {
        const label co=map[owner[f]], cn=map[nei[f]];
        if (co==cn)
        {
            faceRestrict[f]=-1-co;
            faceFlip[f]=0;
        }
        else
        {
            const label cf=findCF(std::min(co,cn),std::max(co,cn));
            faceRestrict[f]=cf;
            faceFlip[f]=(co>cn)?1:0;
            cfw[cf]+=fw[f];
        }
    }
    std::vector<label> cOS(nCoarse+1,0), cLS(nCoarse+1,0), cLosort(nCF);
    for (int f=0;f<nCF;++f)
    {
        cOS[cOwn[f]+1]++;
        cLS[cNei[f]+1]++;
    }
    for (int c=0;c<nCoarse;++c)
    {
        cOS[c+1]+=cOS[c];
        cLS[c+1]+=cLS[c];
    }
    {
        std::vector<label> pos(cLS.begin(),cLS.end());
        for (int f=0;f<nCF;++f)
            cLosort[pos[cNei[f]]++]=f;
    }
    Agglom a;
    a.nCoarse=static_cast<int>(nCoarse);
    a.nCoarseFaces=nCF;
    a.map=std::move(map);
    a.cOwn=std::move(cOwn);
    a.cNei=std::move(cNei);
    a.cOS=std::move(cOS);
    a.cLS=std::move(cLS);
    a.cLosort=std::move(cLosort);
    a.faceRestrict=std::move(faceRestrict);
    a.faceFlip=std::move(faceFlip);
    a.coarseFaceWeights=std::move(cfw);
    return a;
}

} // namespace  (greedyColor needs EXTERNAL linkage -- the standalone GS solver in device_amg_gauss_seidel.cu reuses it)

// Greedy cell coloring (host): give each cell the smallest color unused by any of its face-neighbours, so same-color
// cells never share a face. Returns the cells reordered by color (CSR) + the per-color offsets. Typically ~4-6
// colors on a 2D mesh, ~8-12 in 3D. Feeds the multicolor Gauss-Seidel smoother (one parallel sweep per color).
Coloring greedyColor(
    const std::vector<label>& owner,
    const std::vector<label>& nei,
    int nC)
{
    const int nFaces = static_cast<int>(owner.size());
    std::vector<label> nNbr(nC,0);
    for (int f=0;f<nFaces;++f)
    {
        nNbr[owner[f]]++;
        nNbr[nei[f]]++;
    }
    std::vector<label> off(nC+1,0);
    for (int c=0;c<nC;++c)
        off[c+1]=off[c]+nNbr[c];
    std::vector<label> adj(off[nC]);
    std::fill(nNbr.begin(),nNbr.end(),0);
    for (int f=0;f<nFaces;++f)
    {
        adj[off[owner[f]]+nNbr[owner[f]]++]=nei[f];
        adj[off[nei[f]]+nNbr[nei[f]]++]=owner[f];
    }
    std::vector<label> color(nC,-1);
    int nColors=0;
    std::vector<char> used;
    for (int c=0;c<nC;++c)
    {
        used.assign(nColors+1,0);
        for (label o=off[c]; o<off[c+1]; ++o)
        {
            const label nb=adj[o];
            if (color[nb]>=0)
            {
                if (color[nb]>=(int)used.size()) used.resize(color[nb]+1,0);
                used[color[nb]]=1;
            }
        }
        int col=0;
        while (col<(int)used.size() && used[col])
            ++col;
        color[c]=col;
        if (col+1>nColors) nColors=col+1;
    }
    std::vector<label> start(nColors+1,0);
    for (int c=0;c<nC;++c)
        start[color[c]+1]++;
    for (int k=0;k<nColors;++k)
        start[k+1]+=start[k];
    std::vector<label> cells(nC);
    std::vector<label> pos(start.begin(),start.end());
    for (int c=0;c<nC;++c)
        cells[pos[color[c]]++]=c;
    return {nColors, std::move(cells), std::move(start)};
}

namespace {  // resume internal-linkage AMG-build helpers
// Smoothed-aggregation host build (BRAE_AMG_SA), a port of the reference GAMG smoothed-aggregation build.
// The smoothed prolongator P = (I - omega D^-1 A) P_tent and the general Galerkin A_c = P^T A P depend on the
// matrix VALUES, but the hierarchy is built once at setup from only the GRAPH + face weights (|Sf|). So P is shaped
// from a fixed geometric proxy, the graph-Laplacian (diag = row-sum of |Sf|, off-diag = -|Sf|), whose zero interior
// row sums let the smoothed P interpolate the constant near-null-space exactly. Each coarser proxy level is the RAP
// of the one above; P is then fixed, and only the coarse OPERATOR VALUES are re-evaluated from the real fine matrix
// each SIMPLE step via a precomputed RAP scatter (device, outside the captured V-cycle graph).
struct HostLdu
{
    int nCells = 0;
    std::vector<label>  lowerAddr, upperAddr;                  // owner / neighbour per face
    std::vector<scalar> diag, lower, upper;
    int numFaces() const { return static_cast<int>(upperAddr.size()); }
    void spmv(const std::vector<scalar>& x, std::vector<scalar>& y) const
    {
        y.assign(nCells, 0.0);
        for (int c = 0; c < nCells; ++c) y[c] = diag[c]*x[c];
        for (int f = 0; f < numFaces(); ++f)
        {
            y[lowerAddr[f]] += upper[f]*x[upperAddr[f]];
            y[upperAddr[f]] += lower[f]*x[lowerAddr[f]];
        }
    }
};
// Geometric graph-Laplacian proxy from (owner,nei,faceWeights), the fixed stand-in for the fine pressure matrix.
HostLdu proxyLaplacian(
    const std::vector<label>& owner,
    const std::vector<label>& nei,
    const std::vector<scalar>& fw,
    int n)
{
    HostLdu A;
    A.nCells = n;
    const int nF = static_cast<int>(owner.size());
    A.lowerAddr = owner;
    A.upperAddr = nei;
    A.upper.assign(nF, 0.0);
    A.lower.assign(nF, 0.0);
    A.diag.assign(n, 0.0);
    for (int f = 0; f < nF; ++f)
    {
        A.upper[f] = -fw[f];
        A.lower[f] = -fw[f];
        A.diag[owner[f]] += fw[f];
        A.diag[nei[f]] += fw[f];
    }
    for (int c = 0; c < n; ++c)
        if (!(A.diag[c] > 0.0)) A.diag[c] = 1.0;   // guard isolated cell
    return A;
}
// Spectral radius of D^-1 A (power iteration) for the smoother damping omega = 4/(3 lambda). Ports the reference
// estimateSpectralRadius.
scalar hostSpectralRadius(
    const HostLdu& A,
    int iters = 15)
{
    const int n = A.nCells;
    if (n == 0) return 1.0;
    std::vector<scalar> x(n), y;
    for (int i = 0; i < n; ++i)
        x[i] = 1.0 + 0.1*(i % 7);
    scalar lambda = 1.0;
    for (int it = 0; it < iters; ++it)
    {
        A.spmv(x, y);
        scalar nz = 0.0, nx = 0.0;
        std::vector<scalar> z(n);
        for (int i = 0; i < n; ++i)
        {
            z[i] = y[i]/A.diag[i];
            nz += z[i]*z[i];
            nx += x[i]*x[i];
        }
        if (nx <= 0.0 || nz <= 0.0) break;
        lambda = std::sqrt(nz/nx);
        const scalar inv = 1.0/std::sqrt(nz);
        for (int i = 0; i < n; ++i)
            x[i] = z[i]*inv;
    }
    return (lambda > 0.0) ? lambda : 1.0;
}
// Greedy compact aggregation (Vanek-Mandel-Brezina SA), a port of the reference aggregateCompact: grow each
// aggregate as a seed + its free neighbours (~3^d cells). Compact aggregates keep the smoothed A_c = P^T A P stencil
// bounded per level, unlike chained pairwise aggregates. beta = strength-of-connection filter (0 = off).
// Returns the tentative prolongator as a fine->coarse map + the coarse count.
struct CompactAgg
{
    std::vector<label> map;
    int nCoarse = 0;
};
CompactAgg aggregateCompact(
    const HostLdu& A,
    double beta)
{
    const int n = A.nCells, nF = A.numFaces();
    std::vector<int> off(n+1, 0);
    for (int f = 0; f < nF; ++f)
    {
        ++off[A.lowerAddr[f]+1];
        ++off[A.upperAddr[f]+1];
    }
    for (int c = 1; c <= n; ++c)
        off[c] += off[c-1];
    std::vector<int> adj(2*nF);
    std::vector<scalar> adjS(2*nF);
    std::vector<int> cur = off;
    for (int f = 0; f < nF; ++f)
    {
        const int o = A.lowerAddr[f], m = A.upperAddr[f];
        const scalar s = std::max(std::fabs(A.upper[f]), std::fabs(A.lower[f]));
        const int ko = cur[o]++;
        adj[ko] = m;
        adjS[ko] = s;
        const int km = cur[m]++;
        adj[km] = o;
        adjS[km] = s;
    }
    std::vector<unsigned char> strong(2*nF, 1);
    if (beta > 0.0)
        for (int c = 0; c < n; ++c)
            for (int k = off[c]; k < off[c+1]; ++k)
            {
                const int j = adj[k];
                const scalar thr = beta*std::sqrt(std::fabs(A.diag[c])*std::fabs(A.diag[j]));
                strong[k] = (adjS[k] >= thr) ? 1 : 0;
            }
    CompactAgg agg;
    agg.map.assign(n, -1);
    auto f2c = [&](int c) -> label& { return agg.map[c]; };
    for (int c = 0; c < n; ++c)                               // phase 1: seed + STRONG free neighbours
    {
        if (f2c(c) != -1) continue;
        bool allFree = true;
        for (int k = off[c]; k < off[c+1]; ++k)
            if (strong[k] && f2c(adj[k]) != -1)
            {
                allFree = false;
                break;
            }
        if (!allFree) continue;
        const int cc = agg.nCoarse++;
        f2c(c) = cc;
        for (int k = off[c]; k < off[c+1]; ++k)
            if (strong[k]) f2c(adj[k]) = cc;
    }
    for (int c = 0; c < n; ++c)                               // phase 2: attach to strongest STRONG aggregate
    {
        if (f2c(c) != -1) continue;
        int best = -1;
        scalar bestS = -1.0;
        for (int k = off[c]; k < off[c+1]; ++k)
        {
            if (!strong[k]) continue;
            const int g = f2c(adj[k]);
            if (g == -1) continue;
            if (adjS[k] > bestS)
            {
                bestS = adjS[k];
                best = g;
            }
        }
        if (best != -1) f2c(c) = best;
    }
    for (int c = 0; c < n; ++c)                               // phase 3: leftover seeds + STRONG free neighbours
    {
        if (f2c(c) != -1) continue;
        const int cc = agg.nCoarse++;
        f2c(c) = cc;
        for (int k = off[c]; k < off[c+1]; ++k)
            if (strong[k] && f2c(adj[k]) == -1) f2c(adj[k]) = cc;
    }
    return agg;
}
// Sparse smoothed prolongator P=(I-omega D^-1 A)P_tent (CSR by fine row). map = tentative 0/1 aggregation (each fine
// cell -> one coarse cell). Faithful to the reference buildSmoothedProlongator (filterBeta omitted).
struct HostP
{
    int nFine = 0, nCoarse = 0;
    std::vector<label> rowPtr, col;
    std::vector<scalar> val;
};
HostP buildSmoothedP(
    const HostLdu& A,
    const std::vector<label>& map,
    int nCoarse)
{
    const int nF = A.nCells, nFaces = A.numFaces();
    const scalar lambda = hostSpectralRadius(A), omega = 4.0/(3.0*lambda);
    std::vector<int> off(nF+1, 0);
    for (int f = 0; f < nFaces; ++f)
    {
        ++off[A.lowerAddr[f]+1];
        ++off[A.upperAddr[f]+1];
    }
    for (int c = 1; c <= nF; ++c)
        off[c] += off[c-1];
    std::vector<int> adjOther(2*nFaces);
    std::vector<scalar> adjA(2*nFaces);
    std::vector<int> cur = off;
    for (int f = 0; f < nFaces; ++f)
    {
        const int o = A.lowerAddr[f], nb = A.upperAddr[f];
        adjOther[cur[o]] = nb;
        adjA[cur[o]] = A.upper[f];
        ++cur[o];
        adjOther[cur[nb]] = o;
        adjA[cur[nb]] = A.lower[f];
        ++cur[nb];
    }
    HostP P;
    P.nFine = nF;
    P.nCoarse = nCoarse;
    P.rowPtr.assign(nF+1, 0);
    for (int f = 0; f < nF; ++f)
    {
        std::map<int, scalar> row;                            // coarse col -> value
        const scalar s = omega/A.diag[f];
        for (int k = off[f]; k < off[f+1]; ++k)
            row[map[adjOther[k]]] -= s*adjA[k];
        row[map[f]] += 1.0 - s*A.diag[f];
        for (const auto& e : row)
        {
            P.col.push_back(e.first);
            P.val.push_back(e.second);
        }
        P.rowPtr[f+1] = static_cast<label>(P.col.size());
    }
    return P;
}
// General Galerkin A_c = P^T A P. Returns the coarse LDU (proxy values, feeding the next level's aggregation) and
// the scatter recipe that re-evaluates A_c each step from the real fine matrix. The coarse-face structure is
// value-independent (srcKind/dstKind: 0=diag 1=upper 2=lower), so the recipe is emitted once against the proxy
// (outer products Ac[P.col[f][i]][P.col[g][j]] += P[f][i]*a*P[g][j]) and replayed on the real values.
struct RapRecipe
{
    std::vector<label> srcKind, srcIdx, dstKind, dstIdx;
    std::vector<scalar> w;
};
HostLdu coarsenRAPRecipe(
    const HostLdu& A,
    const HostP& P,
    RapRecipe& rec)
{
    const long long nC = P.nCoarse;
    std::unordered_map<long long, scalar> Ac;                 // pass 1: proxy values -> discover entries/faces
    auto addVal = [&](int f, int g, scalar a)
    {
        for (label kp = P.rowPtr[f]; kp < P.rowPtr[f+1]; ++kp)
        {
            const long long I = P.col[kp];
            const scalar pfia = P.val[kp]*a;
            if (pfia == 0.0) continue;
            for (label kq = P.rowPtr[g]; kq < P.rowPtr[g+1]; ++kq)
                Ac[I*nC + P.col[kq]] += pfia*P.val[kq];
        }
    };
    for (int f = 0; f < A.nCells; ++f)
        addVal(f, f, A.diag[f]);
    for (int f = 0; f < A.numFaces(); ++f)
    {
        const int o = A.lowerAddr[f], nb = A.upperAddr[f];
        addVal(o, nb, A.upper[f]);
        addVal(nb, o, A.lower[f]);
    }
    HostLdu C;
    C.nCells = static_cast<int>(nC);
    C.diag.assign(nC, 0.0);
    // Coarse faces MUST be sorted by (owner, neighbour): the device coarse SpMV's ownerStart is a plain prefix-sum
    // that requires faces owned by a cell to be contiguous (like the injection path's ordered pairIdx). An ORDERED
    // map keyed by lo*nC+hi gives exactly that ordering; assign sorted face indices, then fill values + emit recipe.
    std::map<long long, int> faceOf;
    for (const auto& e : Ac)
    {
        const int I = static_cast<int>(e.first/nC), J = static_cast<int>(e.first%nC);
        if (I == J)
        {
            C.diag[I] = e.second;
            continue;
        }
        const int lo = std::min(I,J), hi = std::max(I,J);
        faceOf.emplace(static_cast<long long>(lo)*nC + hi, 0);
    }
    const int nCF = static_cast<int>(faceOf.size());
    {
        int idx = 0;
        for (auto& kv : faceOf)
            kv.second = idx++;
    }   // sorted (lo,hi) -> face index
    C.lowerAddr.resize(nCF);
    C.upperAddr.resize(nCF);
    C.upper.assign(nCF, 0.0);
    C.lower.assign(nCF, 0.0);
    for (const auto& kv : faceOf)
    {
        C.lowerAddr[kv.second] = static_cast<int>(kv.first/nC);
        C.upperAddr[kv.second] = static_cast<int>(kv.first%nC);
    }
    for (const auto& e : Ac)
    {
        const int I = static_cast<int>(e.first/nC), J = static_cast<int>(e.first%nC);
        if (I == J) continue;
        const int lo = std::min(I,J), hi = std::max(I,J);
        const int cf = faceOf[static_cast<long long>(lo)*nC + hi];
        if (I < J) C.upper[cf] = e.second;
        else C.lower[cf] = e.second;
    }
    auto emit = [&](int f, int g, label sk, label sidx)       // pass 2: emit triples (faces now assigned)
    {
        for (label kp = P.rowPtr[f]; kp < P.rowPtr[f+1]; ++kp)
        {
            const int I = P.col[kp];
            const scalar pfi = P.val[kp];
            if (pfi == 0.0) continue;
            for (label kq = P.rowPtr[g]; kq < P.rowPtr[g+1]; ++kq)
            {
                const int J = P.col[kq];
                const scalar w = pfi*P.val[kq];
                if (w == 0.0) continue;
                label dk, di;
                if (I == J)
                {
                    dk = 0;
                    di = I;
                }
                else
                {
                    const int lo = std::min(I,J), hi = std::max(I,J);
                    dk = (I < J) ? 1 : 2;
                    di = faceOf[static_cast<long long>(lo)*nC + hi];
                }
                rec.srcKind.push_back(sk);
                rec.srcIdx.push_back(sidx);
                rec.w.push_back(w);
                rec.dstKind.push_back(dk);
                rec.dstIdx.push_back(di);
            }
        }
    };
    for (int f = 0; f < A.nCells; ++f)
        emit(f, f, 0, f);
    for (int f = 0; f < A.numFaces(); ++f)
    {
        const int o = A.lowerAddr[f], nb = A.upperAddr[f];
        emit(o, nb, 1, f);
        emit(nb, o, 2, f);
    }
    return C;
}
} // namespace

// Allocate the V-cycle scratch, persistent buffers, and graph caches that don't depend on the agglomeration values;
// shared by buildAMG (after building the hierarchy) and loadAMGCache (after deserializing it).
// The Galerkin gather lists, rebuilt from the agglomeration the CACHE stores. buildGalerkinGather's own
// comment has always said "the AMG cache stores the agglomeration this is derived from, so a cached
// hierarchy rebuilds these for free" -- and nothing did: loadAMGCache restored map/faceRestrict/faceFlip
// and left galCellStart..galFaceFlipList EMPTY, so the first Galerkin re-fill after a cache load read
// index 0 of a zero-length buffer. Measured: run 1 writes .brae_amgcache and runs clean, run 2 dies in
// galDiagGatherK with "Invalid __global__ read of size 4 bytes / Access to 0x180 is out of bounds",
// surfacing at the next cudaGetLastError as "amul: an illegal memory access was encountered".
//
// Rebuilt rather than serialised: these lists are a pure function of map, faceRestrict and faceFlip,
// all three of which the cache already holds, and calling the SAME builder the build path calls is the
// only form that cannot drift from it. It also leaves the cache FORMAT unchanged, so a file written
// before this fix loads correctly afterwards.
void rebuildGalerkinGather(
    AMGLevel& L,
    int       nFine)
{
    struct Shim
    {
        int                 nCoarse, nCoarseFaces;
        std::vector<label>  map, faceRestrict, faceFlip;
    } a;
    a.nCoarse      = L.nCoarse;
    a.nCoarseFaces = L.nCoarseFaces;
    a.map          = L.map.host();
    a.faceRestrict = L.faceRestrict.host();
    a.faceFlip     = L.faceFlip.host();
    if (static_cast<int>(a.map.size()) != nFine) return;   // shape mismatch: caller rebuilds cold
    buildGalerkinGather(L, a, nFine);
}

void finalizeAMG(
    AMGData& A,
    int nFine)
{
    const int G = A.nLevels();
    A.vAx.resize(G+1);
    A.vR.resize(G+1);
    A.vX.resize(G+1);
    A.vB.resize(G+1);
    A.vD.resize(G+1);
    A.vPc.resize(G+1);
    for (int g = 0; g <= G; ++g)
    {
        const int sz = (g==0) ? nFine : A.level[g-1].nCoarse;
        A.vAx[g].resize(sz);
        A.vR[g].resize(sz);
        A.vX[g].resize(sz);
        A.vB[g].resize(sz);
        A.vD[g].resize(sz);
        A.vPc[g].resize(sz);
    }
    A.sScNum.resize(1);
    A.sScDen.resize(1);
    A.sScAlpha.resize(1);
    A.sZrOld.resize(1);
    A.lambdaMax.assign(G+1, 1.0);
    A.spectrumReady = false;
    A.wA.resize(nFine);
    A.rA.resize(nFine);
    A.sWArA.resize(1);
    A.sWArAold.resize(1);
    A.sPap.resize(1);
    A.sAlpha.resize(1);
    A.sNegAlpha.resize(1);
    A.sBeta.resize(1);
    A.sResNorm.resize(1);
    A.gcache = std::make_unique<AMGGraphCache>();
    A.gcacheF = std::make_unique<AMGGraphCache>();
}

// Build the AMG hierarchy, or reload it from cacheDir/.brae_amgcache if a valid one is present (newer than the
// polyMesh/owner file -> mesh unchanged). writeCache=true persists it (the "partition" step / BRAE_MESH_CACHE).
AMGData buildOrLoadAMG(
    const std::vector<label>& fineOwner,
    const std::vector<label>& fineNei,
    const std::vector<scalar>& faceWeights,
    int nFine,
    const std::string& cacheDir,
    bool writeCache)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string amgPath = cacheDir + "/.brae_amgcache";
    const std::string ownerPath = cacheDir + "/owner";
    if (fs::exists(amgPath, ec) && fs::exists(ownerPath, ec)
        && fs::last_write_time(amgPath, ec) >= fs::last_write_time(ownerPath, ec))
    {
        AMGData A;
        if (loadAMGCache(amgPath, A)) return A;     // warm: reuse the cached hierarchy
    }
    AMGData A = buildAMG(fineOwner, fineNei, faceWeights, nFine);
    if (writeCache) writeAMGCache(A, amgPath);
    return A;
}

AMGData buildAMG(
    const std::vector<label>& fineOwner,
    const std::vector<label>& fineNei,
    const std::vector<scalar>& faceWeights,
    int nFine)
{
    // Keep coarsening until the coarsest grid is <= TARGET cells. Overridable (BRAE_AMG_TARGET)
    // so a tiny mesh can still be made to build a real hierarchy: the demo/teaching cases are
    // below the default target and would otherwise get zero levels (coarsest solve only).
    static const int TARGET = []()
    {
        const char* e = std::getenv("BRAE_AMG_TARGET");
        const int v = e ? std::atoi(e) : 64;
        return v > 0 ? v : 64;
    }();
    AMGData A;
    A.nFine = nFine;
    // Multicolor Gauss-Seidel smoother (BRAE_AMG_GS): color every smoothed grid once at build (host, static geometry).
    auto pushColoring = [&](const Coloring& c)
    {
        GridColoring gc;
        gc.nColors=c.nColors;
        gc.cells.copyFrom(c.cells);
        gc.start.copyFrom(c.start);
        gc.startH=c.start;
        A.coloring.push_back(std::move(gc));
    };
    const bool gs = useGS();
    A.gsSmooth = gs;
    const bool sa = useSA();
    A.saSmooth = sa;                    // smoothed aggregation (BRAE_AMG_SA): general RAP coarse op
    std::vector<label> owner = fineOwner, nei = fineNei;
    std::vector<scalar> fw = faceWeights;
    int n = nFine;
    HostLdu proxy;
    if (sa) proxy = proxyLaplacian(fineOwner, fineNei, faceWeights, nFine);   // level-0 geometric proxy
    // SA strength-of-connection filter (BRAE_AMG_SOC, 0 = OFF = every neighbour strong), feeds compact aggregation.
    static const double saSoc = [](){ const char* e = std::getenv("BRAE_AMG_SOC"); return e ? std::atof(e) : 0.0; }();
    if (gs) pushColoring(greedyColor(fineOwner, fineNei, nFine));   // coloring[0] = fine grid
    while (n > TARGET)
    {
        if (!sa)                                              // injection (default): face-based Galerkin
        {
            Agglom a = agglomerate(owner, nei, fw, n);
            if (a.nCoarse >= n || a.nCoarseFaces == 0) break;   // no further coarsening possible
            if (gs) pushColoring(greedyColor(a.cOwn, a.cNei, a.nCoarse));   // coloring[k+1] = coarse grid k+1
            AMGLevel L;
            L.nFine = n;
            L.nCoarse = a.nCoarse;
            L.nCoarseFaces = a.nCoarseFaces;
            L.map.copyFrom(a.map);
            L.cOwn.copyFrom(a.cOwn);
            L.cNei.copyFrom(a.cNei);
            L.cOwnerStart.copyFrom(a.cOS);
            L.cLosort.copyFrom(a.cLosort);
            L.cLosortStart.copyFrom(a.cLS);
            L.faceRestrict.copyFrom(a.faceRestrict);
            L.faceFlip.copyFrom(a.faceFlip);
            buildGalerkinGather(L, a, n);
            L.cDiag.resize(a.nCoarse);
            L.cUpper.resize(a.nCoarseFaces);
            L.cLower.resize(a.nCoarseFaces);
            A.level.push_back(std::move(L));
            owner = std::move(a.cOwn);
            nei = std::move(a.cNei);
            fw = std::move(a.coarseFaceWeights);
            n = a.nCoarse;
        }
        else                                               // SMOOTHED AGGREGATION: compact aggregate + smoothed P + RAP
        {
            CompactAgg a = aggregateCompact(proxy, saSoc);      // tentative prolongator (map) on the proxy graph
            if (a.nCoarse >= n || proxy.numFaces() == 0) break;
            HostP P = buildSmoothedP(proxy, a.map, a.nCoarse);
            RapRecipe rec;
            HostLdu C = coarsenRAPRecipe(proxy, P, rec);
            const int nCF = C.numFaces();
            if (nCF == 0) break;                                // coarse graph disconnected -> stop coarsening
            if (gs) pushColoring(greedyColor(C.lowerAddr, C.upperAddr, a.nCoarse));
            AMGLevel L;
            L.nFine = n;
            L.nCoarse = a.nCoarse;
            L.nCoarseFaces = nCF;
            L.map.copyFrom(a.map);                              // retained for uniformity (SA uses the sparse P, not map)
            L.cOwn.copyFrom(C.lowerAddr);
            L.cNei.copyFrom(C.upperAddr);   // coarse SpMV addressing from RAP faces (owner<nbr)
            std::vector<label> cOS(a.nCoarse+1, 0), cLS(a.nCoarse+1, 0), cLosort(nCF);
            for (int f = 0; f < nCF; ++f)
            {
                cOS[C.lowerAddr[f]+1]++;
                cLS[C.upperAddr[f]+1]++;
            }
            for (int c = 0; c < a.nCoarse; ++c)
            {
                cOS[c+1] += cOS[c];
                cLS[c+1] += cLS[c];
            }
            {
                std::vector<label> pos(cLS.begin(), cLS.end());
                for (int f = 0; f < nCF; ++f)
                    cLosort[pos[C.upperAddr[f]]++] = f;
            }
            L.cOwnerStart.copyFrom(cOS);
            L.cLosort.copyFrom(cLosort);
            L.cLosortStart.copyFrom(cLS);
            L.cDiag.resize(a.nCoarse);
            L.cUpper.resize(nCF);
            L.cLower.resize(nCF);
            L.Prow.copyFrom(P.rowPtr);
            L.Pcol.copyFrom(P.col);
            L.Pval.copyFrom(P.val);   // sparse smoothed prolongator
            L.nTriples = static_cast<int>(rec.w.size());        // RAP scatter recipe (re-evaluated each step)
            L.rapSrcKind.copyFrom(rec.srcKind);
            L.rapSrcIdx.copyFrom(rec.srcIdx);
            L.rapDstKind.copyFrom(rec.dstKind);
            L.rapDstIdx.copyFrom(rec.dstIdx);
            L.rapW.copyFrom(rec.w);
            A.level.push_back(std::move(L));
            std::vector<scalar> cfw(nCF);                        // next level's agglomeration weights = |coarse off-diag|
            for (int f = 0; f < nCF; ++f)
                cfw[f] = std::max(std::fabs(C.upper[f]), std::fabs(C.lower[f]));
            owner = C.lowerAddr;
            nei = C.upperAddr;
            fw = std::move(cfw);
            n = a.nCoarse;
            proxy = std::move(C);
        }
    }
    A.nCoarse = A.level.empty() ? nFine : A.level.front().nCoarse;
    A.nCoarseFaces = A.level.empty() ? 0 : A.level.front().nCoarseFaces;
    if (std::getenv("BRAE_AMG_DEBUG"))
    {
        std::printf("[AMG] hierarchy: %d levels  (TARGET=%d)\n  L0(fine)=%d cells", A.nLevels(), TARGET, nFine);
        for (int k = 0; k < A.nLevels(); ++k)
            std::printf(" -> L%d=%d(%df)", k+1, A.level[k].nCoarse, A.level[k].nCoarseFaces);
        const double ratio = A.nLevels() ? std::pow((double)nFine / A.level.back().nCoarse, 1.0/A.nLevels()) : 1.0;
        std::printf("\n  coarsest=%d cells  avg coarsening=%.2fx/level\n",
                    A.level.empty()? nFine : A.level.back().nCoarse, ratio);
    }
    finalizeAMG(A, nFine);                                      // V-cycle scratch + persistent buffers + graph caches
    return A;
}

void amgGalerkin(
    AMGData& A,
    const DeviceBuffer<scalar>& fineDiag,
    const DeviceBuffer<scalar>& fineUpper,
    const DeviceBuffer<scalar>& fineLower)
{
    for (int k = 0; k < A.nLevels(); ++k)                      // grid k matrix -> grid k+1 (Galerkin scatter)
    {
        AMGLevel& L = A.level[k];
        const scalar* fd;
        const scalar* fu;
        const scalar* fl;
        int nFaces;
        if (k == 0)
        {
            fd = fineDiag.data();
            fu = fineUpper.data();
            fl = fineLower.data();
            nFaces = static_cast<int>(fineUpper.size());
        }
        else
        {
            fd = A.level[k-1].cDiag.data();
            fu = A.level[k-1].cUpper.data();
            fl = A.level[k-1].cLower.data();
            nFaces = A.level[k-1].nCoarseFaces;
        }
        if (A.saSmooth)                                          // general Galerkin A_c = P^T A P (precomputed RAP recipe)
        {
            // The SA path still SCATTERS, so it is still order-dependent. It is opt-in (BRAE_AMG_SA) and
            // off by default; leaving it as it was keeps this change to one behaviour at a time. The
            // determinism gate asserts the DEFAULT path, and the SA path is listed as a known gap.
            zeroT<scalar><<<nBlocks(L.nCoarse),TPB>>>(L.nCoarse, L.cDiag.data());
            zeroT<scalar><<<nBlocks(L.nCoarseFaces),TPB>>>(L.nCoarseFaces, L.cUpper.data());
            zeroT<scalar><<<nBlocks(L.nCoarseFaces),TPB>>>(L.nCoarseFaces, L.cLower.data());
            rapScatterK<<<nBlocks(L.nTriples),TPB>>>(L.nTriples, L.rapSrcKind.data(), L.rapSrcIdx.data(), L.rapW.data(),
                L.rapDstKind.data(), L.rapDstIdx.data(), fd, fu, fl, L.cDiag.data(), L.cUpper.data(), L.cLower.data());
        }
        else       // injection Galerkin (default): fixed-order GATHER per coarse entity, no pre-zero needed
        {
            galDiagGatherK<<<nBlocks(L.nCoarse),TPB>>>(
                L.nCoarse, L.galCellStart.data(), L.galCellList.data(),
                L.galDFaceStart.data(), L.galDFaceList.data(), fd, fu, fl, L.cDiag.data());
            galFaceGatherK<<<nBlocks(L.nCoarseFaces),TPB>>>(
                L.nCoarseFaces, L.galFaceStart.data(), L.galFaceList.data(), L.galFaceFlipList.data(),
                fu, fl, L.cUpper.data(), L.cLower.data());
        }
    }
    cudaCheck(cudaGetLastError(), "galerkin");
    // The coarse operators just changed, so any Chebyshev spectrum estimate keyed to the old
    // operator is stale. Across SIMPLE steps the pressure matrix is re-weighted non-uniformly
    // (face fluxes / Ap evolve with the velocity field), which shifts the spectrum of D^-1 A,
    // not just its diagonal scale -- a frozen interval eventually fails to cover the top modes
    // and the Chebyshev smoother stops damping them. Re-estimate on the next solve. No cost
    // unless BRAE_CHEBYSHEV is on (ensureSpectrum early-returns otherwise).
    A.spectrumReady = false;
    static bool saDbgOnce = false;                             // report the SA coarse-operator health just once
    if (A.saSmooth && std::getenv("BRAE_AMG_DEBUG") && !saDbgOnce)   // diag sign + dominance + coarsest definiteness
    {
        saDbgOnce = true;
        cudaCheck(cudaDeviceSynchronize(), "sa dbg sync");
        for (int k = 0; k < A.nLevels(); ++k)
        {
            AMGLevel& L = A.level[k];
            std::vector<scalar> d = L.cDiag.host(), u = L.cUpper.host(), lo = L.cLower.host();
            std::vector<label> oo = L.cOwn.host(), nn = L.cNei.host();
            std::vector<scalar> rowoff(L.nCoarse, 0.0);
            for (int f = 0; f < L.nCoarseFaces; ++f)
            {
                rowoff[oo[f]] += std::fabs(u[f]);
                rowoff[nn[f]] += std::fabs(lo[f]);
            }
            scalar dmin = 1e300, dmax = -1e300, domMax = 0.0;
            int nNeg = 0, nNan = 0;
            for (int c = 0; c < L.nCoarse; ++c)
            {
                if (!std::isfinite(d[c]))
                {
                    ++nNan;
                    continue;
                }
                dmin = std::min(dmin, d[c]);
                dmax = std::max(dmax, d[c]);
                if (d[c] < 0) ++nNeg;
                if (std::fabs(d[c]) > 0) domMax = std::max(domMax, rowoff[c]/std::fabs(d[c]));
            }
            std::fprintf(stderr, "[SA] L%d nC=%d faces=%d diag[min=%.3e max=%.3e neg=%d nan=%d] maxOffDiagDom=%.3f\n",
                         k+1, L.nCoarse, L.nCoarseFaces, dmin, dmax, nNeg, nNan, domMax);
            if (k == A.nLevels()-1 && L.nCoarse <= 64 && nNan == 0)   // coarsest: dense LDL pivot signs (definiteness)
            {
                const int m = L.nCoarse;
                std::vector<scalar> M(m*m, 0.0);
                for (int c = 0; c < m; ++c)
                    M[c*m+c] = d[c];
                for (int f = 0; f < L.nCoarseFaces; ++f)
                {
                    M[oo[f]*m+nn[f]] = u[f];
                    M[nn[f]*m+oo[f]] = lo[f];
                }
                int npos = 0, nneg = 0, nzero = 0;                      // symmetric LDL^T (no pivoting), pivot signs
                std::vector<scalar> A2 = M;
                for (int i = 0; i < m; ++i)
                {
                    scalar piv = A2[i*m+i];
                    if (std::fabs(piv) < 1e-300)
                    {
                        ++nzero;
                        continue;
                    }
                    if (piv > 0) ++npos;
                    else ++nneg;
                    for (int j = i+1; j < m; ++j)
                    {
                        const scalar fct = A2[j*m+i]/piv;
                        for (int kk = i; kk < m; ++kk)
                            A2[j*m+kk] -= fct*A2[i*m+kk];
                    }
                }
                std::fprintf(stderr, "[SA]   coarsest LDL pivots: pos=%d neg=%d zero=%d  -> %s\n",
                             npos, nneg, nzero, (npos==0||nneg==0) ? "DEFINITE" : "INDEFINITE");
            }
        }
    }
}

} // namespace brae
