// Device AMG preconditioner: host agglomeration (static) + device Galerkin and V-cycle smoothing.
// Used by deviceAMGPCG. Feature flags live in one BRAE_AMG_* block at the top of the anonymous namespace.
#include "device_amg.cuh"
#include "device_blas.cuh"
#include "device_ldu.cuh"       // deviceParallelAmul (halo-coupled matvec) for the distributed whole-loop graph PCG
#include "device_halo.cuh"      // DeviceHalo + its DeviceReducer (on-stream NVSHMEM reduce)
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
constexpr int TPB = 256;
constexpr scalar OMEGA = 0.8;          // weighted-Jacobi relaxation
constexpr int NPRE = 1, NPOST = 1, NCOARSE = 500;
constexpr int CHEB_DEG_DEFAULT = 2;    // Chebyshev smoother degree (per pre/post smooth)
constexpr scalar CHEB_EIGRATIO = 30.0; // Chebyshev interval [upper/ratio, upper]
constexpr scalar CHEB_UPPER_SAFETY = 1.2; // interval top = safety * lambdaMax; > 1 to over-cover (under-cover diverges)

// Feature flags (env vars), documented here once. The two default-ON flags preserve accuracy and opt out with
// =0; the rest are experimental smoother/coarsening levers, off unless set.
//   BRAE_AMG_FP32   (on)  FP32 V-cycle preconditioner; outer Krylov + residual stay FP64
//   BRAE_PCG_DEVICE (on)  run the whole pressure PCG on-device (conditional-graph WHILE), see deviceAMGPCGGraph
//   BRAE_CHEBYSHEV, BRAE_CHEB_DEG    Chebyshev polynomial smoother (and its degree) vs weighted-Jacobi
//   BRAE_AMG_GS                      multicolor Gauss-Seidel smoother (off-diagonal coupling fixes anisotropy)
//   BRAE_AMG_TSGS, BRAE_TSGS_ORDER   OpenFOAM v2606 twoStageGaussSeidel polynomial smoother (order 0 == Jacobi)
//   BRAE_AMG_SA                      smoothed aggregation: smoothed prolongator + general Galerkin A_c = P^T A P
//   BRAE_AMG_SOC                     strength-of-connection filter beta for aggregation (0 = off)
//   BRAE_NCOARSE_CG                  coarsest-grid PCG iteration count
inline bool envFlag(
    const char* name,
    bool def)
{          // default-ON flag: def unless explicitly 0/false/off/no
    const char* e = std::getenv(name);
    if (!e || !*e) return def;
    auto ieq = [](const char* a, const char* b)
    {
        for (; *a && *b; ++a, ++b)
            if ((*a|0x20) != (*b|0x20)) return false;
        return *a == *b;
    };
    return !(ieq(e,"0") || ieq(e,"false") || ieq(e,"off") || ieq(e,"no"));
}
inline bool useChebyshev()
{
    static const bool b = std::getenv("BRAE_CHEBYSHEV") != nullptr;
    return b;
}
inline int  chebDeg()
{
    static const int d = [](){ const char* e = std::getenv("BRAE_CHEB_DEG"); return (e && std::atoi(e) > 0) ? std::atoi(e) : CHEB_DEG_DEFAULT; }();
    return d;
}
inline bool useGS()
{
    static bool g = (std::getenv("BRAE_AMG_GS") != nullptr);
    return g;
}
inline bool useTSGS()
{
    static bool t = (std::getenv("BRAE_AMG_TSGS") != nullptr);
    return t;
}
inline int  tsgsOrder()
{
    static const int o = [](){ const char* e = std::getenv("BRAE_TSGS_ORDER"); return (e && std::atoi(e) >= 0) ? std::atoi(e) : 1; }();
    return o;
}
inline bool useSA()
{
    static bool s = (std::getenv("BRAE_AMG_SA") != nullptr);
    return s;
}
inline bool useFP32()
{
    static bool b = envFlag("BRAE_AMG_FP32", true);
    return b;
}
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// V-cycle work kernels are templated on the value type: <scalar> is the FP64 path, <float> the mixed-precision
// Diagonal floor for the coarse-grid Jacobi/PCG divisions. The default injection Galerkin always
// yields a strictly positive coarse diagonal (a sum of positive fine diagonals), but the general
// RAP under smoothed aggregation (BRAE_AMG_SA) can produce a near-zero or negative coarse diag
// (the BRAE_AMG_DEBUG block reports exactly this). Dividing by it gives Inf/NaN, which propagates
// through prolongation into the outer residual and runs the solve to maxIter on garbage. This
// helper is a STRICT no-op for any diagonal of magnitude > 1e-300, so it never perturbs a
// well-formed operator; it only replaces a would-be division by (near) zero.
template <typename T>
__device__ __forceinline__
T safeDiag(T d)
{
    const T floor = T(1e-300);
    if (d > floor)  return d;
    if (d < -floor) return d;
    return (d < T(0)) ? -floor : floor;
}
// V-cycle / FP32 GS. T(OMEGA) and T(0) reproduce the FP64 and FP32 constants exactly, so each instantiation is
// byte-identical to the hand-written twin it replaces.
template <typename T>
__global__
void zeroT(
    int n,
    T* x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = T(0);
}
// weighted-Jacobi update: x += omega*(b - Ax)/diag
template <typename T>
__global__
void smoothT(
    int n,
    const T* __restrict__ b,
    const T* __restrict__ Ax,
    const T* __restrict__ diag,
    T* __restrict__ x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] += T(OMEGA)*(b[i]-Ax[i])/diag[i];
}
template <typename T>
__global__
void residualT(
    int n,
    const T* __restrict__ b,
    const T* __restrict__ Ax,
    T* __restrict__ r)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i]-Ax[i];
}
// One color of in-place Gauss-Seidel: cells in [lo,hi) share no face, so the writes never race and the
// off-diagonal reads pick up already-swept colors -> true GS, not Jacobi. Same LDU gather as the SpMV.
template <typename T>
__global__
void gsColorT(
    int lo,
    int hi,
    const label* __restrict__ cells,
    const T* __restrict__ b,
    const T* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const T* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const T* __restrict__ lower,
    T* __restrict__ x)
{
    const int idx = lo + blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= hi) return;
    const int c = cells[idx];
    T off = T(0);
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        off += upper[f] * x[nei[f]];
    for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const int f = losort[k];
        off += lower[f] * x[owner[f]];
    }
    x[c] = (b[c] - off) / diag[c];
}
// z = src/diag: applies the diagonal (Jacobi) preconditioner, used by the power iteration on D^-1 A.
__global__
void invDiagMulK(
    int n,
    const scalar* __restrict__ src,
    const scalar* __restrict__ diag,
    scalar* __restrict__ z)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) z[i] = src[i]/diag[i];
}
// Chebyshev polynomial smoother step: d = c1*d + c2*(b - Ax)/diag ; x += d.
__global__
void chebStepK(
    int n,
    const scalar* __restrict__ b,
    const scalar* __restrict__ Ax,
    const scalar* __restrict__ diag,
    scalar c1,
    scalar c2,
    scalar* __restrict__ d,
    scalar* __restrict__ x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = c1*d[i] + c2*(b[i]-Ax[i])/diag[i];
    x[i] += d[i];
}
template <typename T>
__global__
void restrictT(
    int nF,
    const label* __restrict__ map,
    const T* __restrict__ r,
    T* __restrict__ rc)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) atomicAdd(&rc[map[c]], r[c]);
}
template <typename T>
__global__
void prolongT(
    int nF,
    const label* __restrict__ map,
    const T* __restrict__ xc,
    T* __restrict__ x)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) x[c] += xc[map[c]];   // injection correction
}
// twoStageGaussSeidel kernels (BRAE_AMG_TSGS). upperMul = strictly-upper-triangle SpMV r = U z; the tsgs* kernels
// below are the fused Jacobi/correction updates of the polynomial expansion.
__global__
void upperMulK(
    int n,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ z,
    scalar* __restrict__ r)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        s += upper[f] * z[nei[f]];
    r[c] = s;
}
// lowerMul = strictly-lower-triangle SpMV r = L z, the transpose of upperMul for the symmetric matrix. Pre-smooth
// with U and post-smooth with L so the V-cycle preconditioner stays symmetric, as the AMG-PCG outer solver requires.
__global__
void lowerMulK(
    int n,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    const scalar* __restrict__ z,
    scalar* __restrict__ r)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    scalar s = 0.0;
    for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const int f = losort[k];
        s += lower[f] * z[owner[f]];
    }
    r[c] = s;
}
__global__
void tsgsZUpdateK(
    int n,
    const scalar* __restrict__ b,
    const scalar* __restrict__ Ax,
    const scalar* __restrict__ diag,
    scalar* __restrict__ z,
    scalar* __restrict__ x)
{   // z = (b-Ax)/D; x += z
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const scalar zi = (b[i]-Ax[i])/diag[i];
        z[i]=zi;
        x[i]+=zi;
    }
}
__global__
void tsgsCorrSaveK(
    int n,
    const scalar* __restrict__ r,
    const scalar* __restrict__ diag,
    scalar mult,
    scalar* __restrict__ z,
    scalar* __restrict__ x)
{   // z = r/D; x += mult*z
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const scalar zi = r[i]/diag[i];
        z[i]=zi;
        x[i]+=mult*zi;
    }
}
__global__
void tsgsCorrLastK(
    int n,
    const scalar* __restrict__ r,
    const scalar* __restrict__ diag,
    scalar mult,
    scalar* __restrict__ x)
{                            // last term: x += mult*r/D (no z save)
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] += mult*r[i]/diag[i];
}
__global__
void prolongToK(
    int nF,
    const label* __restrict__ map,
    const scalar* __restrict__ xc,
    scalar* __restrict__ pc)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) pc[c] = xc[map[c]];   // prolong INTO pc (not added yet)
}
// Smoothed-aggregation sparse prolongator apply (BRAE_AMG_SA): restrict = P^T, prolong = P.
// restrict: rc += P^T r  (each fine cell f scatters val*r[f] into its coarse columns, atomically).
__global__
void restrictSparseK(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const scalar* __restrict__ val,
    const scalar* __restrict__ r,
    scalar* __restrict__ rc)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    const scalar rf = r[f];
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        atomicAdd(&rc[col[k]], val[k]*rf);
}
// prolong (ADD): x[f] += sum_k P[f][k]*xc[col], the smoothed-P twin of prolongT.
__global__
void prolongSparseK(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const scalar* __restrict__ val,
    const scalar* __restrict__ xc,
    scalar* __restrict__ x)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    scalar s = 0.0;
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        s += val[k]*xc[col[k]];
    x[f] += s;
}
// prolong (SET): pc[f] = sum_k P[f][k]*xc[col], the corrScaling line-search path (writes pc, not added yet).
__global__
void prolongToSparseK(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const scalar* __restrict__ val,
    const scalar* __restrict__ xc,
    scalar* __restrict__ pc)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    scalar s = 0.0;
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        s += val[k]*xc[col[k]];
    pc[f] = s;
}
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
// Energy-minimising coarse-correction scale (OF GAMG): alpha = (r . c)/(c . Ac), guarded since c.Ac = ||c||_A^2 >= 0.
__global__
void scaleFactorK(
    const scalar* __restrict__ num,
    const scalar* __restrict__ den,
    scalar* __restrict__ out)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar d = *den;
        *out = (d > 1e-30) ? (*num)/d : 1.0;
    }
}
// Flexible-CG beta (Polak-Ribiere+): beta = max(0, (z.r_new - z.r_old)/rho_old), guarded division. Robust for the
// nonlinear scaled V-cycle preconditioner; reduces to Fletcher-Reeves when the precond is linear (z.r_old == 0).
__global__
void flexBetaK(
    const scalar* __restrict__ zrNew,
    const scalar* __restrict__ zrOld,
    const scalar* __restrict__ rhoOld,
    scalar* __restrict__ out)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar d=*rhoOld;
        *out = (d>1e-300 || d<-1e-300) ? fmax(0.0, (*zrNew - *zrOld)/d) : 0.0;
    }
}
// Injection Galerkin scatter (default path): coarse LDU from fine diag/upper/lower via faceRestrict/faceFlip.
__global__
void galDiagK(
    int nF,
    const label* __restrict__ map,
    const scalar* __restrict__ fineDiag,
    scalar* __restrict__ cDiag)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) atomicAdd(&cDiag[map[c]], fineDiag[c]);
}
__global__
void galFaceK(
    int nFaces,
    const label* __restrict__ fr,
    const label* __restrict__ flip,
    const scalar* __restrict__ up,
    const scalar* __restrict__ lo,
    scalar* __restrict__ cDiag,
    scalar* __restrict__ cUp,
    scalar* __restrict__ cLo)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nFaces) return;
    const int cf = fr[f];
    if (cf >= 0)
    {
        if (!flip[f])
        {
            atomicAdd(&cUp[cf], up[f]);
            atomicAdd(&cLo[cf], lo[f]);
        }
        else
        {
            atomicAdd(&cUp[cf], lo[f]);
            atomicAdd(&cLo[cf], up[f]);
        }
    }
    else atomicAdd(&cDiag[-1-cf], up[f]+lo[f]);
}

constexpr int CCL = 8;             // fused coarse solve: blocks per cluster (whole coarse level = one cluster)
constexpr int COARSE_FUSE_MAX = 4096;   // fused single-cluster cap (measured crossover ~6k)

// All nSweeps of coarse weighted-Jacobi in ONE cluster kernel. The coarse vector is held in distributed shared
// memory: block 'rank' owns cells [rank*cpb, rank*cpb+myCount); a cell's SpMV reads neighbour values from the
// owning sibling block via cluster.map_shared_rank. Ping-pong (rd/wr) gives proper Jacobi (all-old-x per sweep);
// cluster.sync() between sweeps makes every block's update visible. Same arithmetic/order as deviceAmul+smoothK.
__global__
void coarseJacobiFusedKernel(
    int nC,
    int cpb,
    int nSweeps,
    scalar omega,
    const scalar* __restrict__ rc,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    scalar* __restrict__ xc)
{
#if __CUDA_ARCH__ >= 900   // clusters+DSM exist only on sm_90+. Pre-Hopper compiles this as a no-op; it is never
                           // launched there because deviceCoarseFitsCluster() returns false without cluster support,
                           // so the coarsest-solve dispatch falls through to the global-memory Jacobi fallback.
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank();
    extern __shared__ scalar sh[];
    scalar* buf0 = sh;
    scalar* buf1 = sh + cpb;
    const int base = rank * cpb;
    const int myCount = max(0, min(cpb, nC - base));
    for (int i = threadIdx.x; i < cpb; i += blockDim.x)
    {
        buf0[i] = (i < myCount) ? xc[base + i] : 0.0;
        buf1[i] = 0.0;
    }
    __syncthreads();
    cluster.sync();                                          // initial guess visible cluster-wide
    scalar* rd = buf0;
    scalar* wr = buf1;
    for (int s = 0; s < nSweeps; ++s)
    {
        for (int i = threadIdx.x; i < myCount; i += blockDim.x)
        {
            const int c = base + i;
            scalar Ax = diag[c] * rd[i];
            for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)   // faces owned by c
            {
                const int g = nei[f];
                const scalar* rem = cluster.map_shared_rank(rd, g / cpb);
                Ax += upper[f] * rem[g % cpb];
            }
            for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)   // faces neighbouring c
            {
                const int f = losort[k], g = owner[f];
                const scalar* rem = cluster.map_shared_rank(rd, g / cpb);
                Ax += lower[f] * rem[g % cpb];
            }
            wr[i] = rd[i] + omega * (rc[c] - Ax) / safeDiag(diag[c]);
        }
        cluster.sync();                                       // all writes (wr) + reads (rd) done -> swap
        scalar* t = rd;
        rd = wr;
        wr = t;
    }
    for (int i = threadIdx.x; i < myCount; i += blockDim.x)
        xc[base + i] = rd[i];
#endif
}

// SINGLE-BLOCK coarsest Jacobi: the whole (tiny) coarse vector lives in shared memory, nSweeps of ping-pong Jacobi
// with cheap __syncthreads() between them (NOT cluster.sync, the latter's per-sweep cost dominates at the high
// sweep counts a well-solved coarsest needs). Same arithmetic as deviceAmul+smoothK. Used for nC <= SB_MAX.
constexpr int SB_MAX = 2048;       // single-block coarsest Jacobi cap (2*SB_MAX doubles shared = 32KB)
constexpr int SB_CG_MAX = 1024;    // single-block coarsest PCG cap ((5*nC+TPB) doubles shared <= 48KB at nC=1024)
constexpr int NCOARSE_CG = 16;     // coarsest PCG iterations. The coarsest grid is mesh-independent (AMG coarsens to
                                   // TARGET~64), so a fixed count suffices everywhere; override with BRAE_NCOARSE_CG.
__global__
void coarseJacobiSingleBlockKernel(
    int nC,
    int nSweeps,
    scalar omega,
    const scalar* __restrict__ rc,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    scalar* __restrict__ xc)
{
    extern __shared__ scalar sh[];
    scalar* rd = sh;
    scalar* wr = sh + nC;
    for (int i = threadIdx.x; i < nC; i += blockDim.x)
        rd[i] = xc[i];
    __syncthreads();
    for (int s = 0; s < nSweeps; ++s)
    {
        for (int c = threadIdx.x; c < nC; c += blockDim.x)
        {
            scalar Ax = diag[c] * rd[c];
            for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
                Ax += upper[f] * rd[nei[f]];
            for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
            {
                const int f = losort[k];
                Ax += lower[f] * rd[owner[f]];
            }
            wr[c] = rd[c] + omega * (rc[c] - Ax) / safeDiag(diag[c]);
        }
        __syncthreads();
        scalar* t = rd;
        rd = wr;
        wr = t;
    }
    for (int i = threadIdx.x; i < nC; i += blockDim.x)
        xc[i] = rd[i];
}

// Block-wide dot product a.b, warp-shuffle reduce: each warp reduces its lanes with __shfl_down (no barrier), then
// warp 0 reduces the per-warp partials. Three barriers per dot instead of ~log2(blockDim), which matters because the
// single-block coarsest CG runs many sequential dots on one SM. red[] holds the per-warp partials; all threads return the sum.
__device__ __forceinline__
scalar warpReduceSum(scalar v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__
scalar blockDot(
    const scalar* a,
    const scalar* b,
    int n,
    scalar* red)
{
    const int tid = threadIdx.x;
    scalar v = 0.0;
    for (int i = tid; i < n; i += blockDim.x)
        v += a[i]*b[i];
    v = warpReduceSum(v);                                        // intra-warp: no barrier
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();                                            // (1) per-warp partials visible
    if (warp == 0)                                            // warp 0 reduces the (<=32) partials
    {
        const int nW = (blockDim.x + 31) >> 5;
        scalar w = (lane < nW) ? red[lane] : 0.0;
        w = warpReduceSum(w);
        if (lane == 0) red[0] = w;
    }
    __syncthreads();                                            // (2) red[0] visible to all
    const scalar r = red[0];
    __syncthreads();                  // (3) red[] free to reuse next call
    return r;
}
// Single-block coarsest solve by Jacobi-preconditioned CG (z = r/diag). CG converges as O(sqrt(kappa)) vs Jacobi's
// O(kappa), so a handful of iterations solve the tiny coarsest as well as hundreds of Jacobi sweeps. The whole CG
// (vectors + dot reductions) lives in shared memory in one block -> no host sync, capturable in the V-cycle graph.
// Fixed iteration count (deterministic, graph-safe); alpha/beta guarded so an early-converged solve can't NaN.
__global__
void coarsePCGKernel(
    int nC,
    int nIters,
    const scalar* __restrict__ rc,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    scalar* __restrict__ xc)
{
    extern __shared__ scalar sh[];
    scalar* x = sh;
    scalar* r = sh + nC;
    scalar* p = sh + 2*nC;
    scalar* Ap = sh + 3*nC;
    scalar* z = sh + 4*nC;
    scalar* red = sh + 5*nC;                                     // TPB scratch for the block reduction
    const int tid = threadIdx.x;
    for (int i = tid; i < nC; i += blockDim.x)   // x0=0, M^-1 r
    {
        x[i]=0.0;
        r[i]=rc[i];
        z[i]=r[i]/safeDiag(diag[i]);
        p[i]=z[i];
    }
    __syncthreads();
    scalar rz = blockDot(r, z, nC, red);                         // r . z
    for (int it = 0; it < nIters; ++it)
    {
        for (int c = tid; c < nC; c += blockDim.x)   // Ap = A p
        {
            scalar a = diag[c]*p[c];
            for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
                a += upper[f]*p[nei[f]];
            for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
            {
                const int f = losort[k];
                a += lower[f]*p[owner[f]];
            }
            Ap[c] = a;
        }
        __syncthreads();
        const scalar pAp = blockDot(p, Ap, nC, red);
        const scalar alpha = (pAp > 1e-300 || pAp < -1e-300) ? rz/pAp : 0.0;
        for (int i = tid; i < nC; i += blockDim.x)
        {
            x[i]+=alpha*p[i];
            r[i]-=alpha*Ap[i];
            z[i]=r[i]/safeDiag(diag[i]);
        }
        __syncthreads();
        const scalar rznew = blockDot(r, z, nC, red);
        const scalar beta = (rz > 1e-300 || rz < -1e-300) ? rznew/rz : 0.0;
        rz = rznew;
        for (int i = tid; i < nC; i += blockDim.x)
            p[i] = z[i] + beta*p[i];
        __syncthreads();
    }
    for (int i = tid; i < nC; i += blockDim.x)
        xc[i] = x[i];
}
} // namespace

bool deviceCoarseFitsCluster(int nCoarse)
{
    static const bool clusterOK = []()
    {
        int v = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, dev);
        return v != 0;
    }();
    if (!clusterOK) return false;                                            // pre-Hopper: no cluster launch -> global-mem Jacobi fallback
    const int cpb = (nCoarse + CCL - 1) / CCL;
    return 2 * static_cast<std::size_t>(cpb) * sizeof(scalar) <= 99 * 1024;   // GB10 opt-in DSM/block
}

void deviceCoarseJacobiFused(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nSweeps)
{
    const int nC = cv.nCells, cpb = (nC + CCL - 1) / CCL;
    const std::size_t shBytes = 2 * static_cast<std::size_t>(cpb) * sizeof(scalar);
    // Opt into the &gt;48KB dynamic shared memory ONCE. This is a non-stream host runtime call, and
    // deviceCoarseJacobiFused is reachable from vcycleAt, which is stream-captured -- issuing it
    // during capture is at best ignored and at worst refuses the capture. The attribute is a
    // per-function property, so setting it a single time at first use is sufficient.
    static std::once_flag coarseShmemOptin;
    std::call_once(coarseShmemOptin, []()
    {
        cudaCheck(cudaFuncSetAttribute(coarseJacobiFusedKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024), "coarse shmem optin");
    });
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CCL);
    cfg.blockDim = dim3(TPB);
    cfg.dynamicSmemBytes = shBytes;
    cfg.stream = cudaStreamPerThread;
    cudaLaunchAttribute attr[1] = {};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CCL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaCheck(cudaLaunchKernelEx(&cfg, coarseJacobiFusedKernel, nC, cpb, nSweeps, OMEGA,
        rc.data(), cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower, xc.data()),
        "coarseJacobiFused");
    cudaCheck(cudaGetLastError(), "coarseJacobiFused launch");
}

// Single-block coarsest solve (nC <= SB_MAX): nSweeps of in-shared-memory ping-pong Jacobi, cheap __syncthreads.
void deviceCoarseJacobiSingleBlock(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nSweeps)
{
    const int nC = cv.nCells;
    const std::size_t shBytes = 2 * static_cast<std::size_t>(nC) * sizeof(scalar);
    coarseJacobiSingleBlockKernel<<<1, TPB, shBytes, cudaStreamPerThread>>>(nC, nSweeps, OMEGA,
        rc.data(), cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower, xc.data());
    cudaCheck(cudaGetLastError(), "coarseJacobiSingleBlock launch");
}

// Single-block coarsest solve (nC <= SB_CG_MAX) by Jacobi-preconditioned CG, nIters iterations, one launch.
// Right-size the block to the tiny coarsest n (one warp per 32 cells, capped at TPB) so idle warps don't pay the
// block-reduce path. blockDim stays a multiple of 32 (full-warp shuffle mask) and >= 32 (one warp minimum).
void deviceCoarsePCG(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nIters)
{
    const int nC = cv.nCells;
    const int bs = nC >= TPB ? TPB : ((nC + 31) / 32) * 32;     // warp-rounded, in [32, TPB]
    const std::size_t shBytes = (5 * static_cast<std::size_t>(nC) + 32) * sizeof(scalar);   // red[] needs <= bs/32 slots
    coarsePCGKernel<<<1, bs, shBytes, cudaStreamPerThread>>>(nC, nIters,
        rc.data(), cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower, xc.data());
    cudaCheck(cudaGetLastError(), "coarsePCG launch");
}

void deviceCoarseJacobiLoop(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nSweeps)
{
    const int nC = cv.nCells;
    DeviceBuffer<scalar> Axc(nC);
    for (int s = 0; s < nSweeps; ++s)
    {
        deviceAmul(cv, xc, Axc);
        smoothT<scalar><<<nBlocks(nC),TPB>>>(nC, rc.data(), Axc.data(), cv.diag, xc.data());
    }
}

namespace {
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

// Greedy cell coloring (host): give each cell the smallest color unused by any of its face-neighbours, so same-color
// cells never share a face. Returns the cells reordered by color (CSR) + the per-color offsets. Typically ~4-6
// colors on a 2D mesh, ~8-12 in 3D. Feeds the multicolor Gauss-Seidel smoother (one parallel sweep per color).
struct Coloring
{
    int nColors = 0;
    std::vector<label> cells, start;
};
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
static void finalizeAMG(
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

// AMG hierarchy cache. The agglomeration (greedy + sort per level) is the AMG-build cost and is static per mesh
// (only the matrix VALUES change each step, via Galerkin), so the static hierarchy STRUCTURE is serialized: a
// "partition" step builds it once and the run reloads it. cDiag/cUpper/cLower hold values (Galerkin re-fills them),
// so they are not serialized, only re-sized.
namespace {
constexpr unsigned AMG_CACHE_MAGIC = 0x43464131;          // "CFA1"
template<class T>
void wbuf(
    std::FILE* f,
    const DeviceBuffer<T>& b)
{
    std::vector<T> h = b.host();
    std::size_t n = h.size();
    std::fwrite(&n,sizeof(n),1,f);
    if (n) std::fwrite(h.data(),sizeof(T),n,f);
}
template<class T>
bool rbuf(
    std::FILE* f,
    DeviceBuffer<T>& b)
{
    std::size_t n;
    if (std::fread(&n,sizeof(n),1,f)!=1) return false;
    std::vector<T> h(n);
    if (n && std::fread(h.data(),sizeof(T),n,f)!=n) return false;
    b.copyFrom(h);
    return true;
}
}
void writeAMGCache(
    const AMGData& A,
    const std::string& path)
{
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    unsigned magic = AMG_CACHE_MAGIC;
    std::fwrite(&magic,sizeof(magic),1,f);
    int nFine = A.nFine, nLev = A.nLevels();
    char gs = A.gsSmooth, sa = A.saSmooth;
    std::fwrite(&nFine,sizeof(nFine),1,f);
    std::fwrite(&nLev,sizeof(nLev),1,f);
    std::fwrite(&gs,1,1,f);
    std::fwrite(&sa,1,1,f);
    for (const auto& L : A.level)
    {
        std::fwrite(&L.nFine,sizeof(int),1,f);
        std::fwrite(&L.nCoarse,sizeof(int),1,f);
        std::fwrite(&L.nCoarseFaces,sizeof(int),1,f);
        std::fwrite(&L.nTriples,sizeof(int),1,f);
        wbuf(f,L.map);
        wbuf(f,L.cOwn);
        wbuf(f,L.cNei);
        wbuf(f,L.cOwnerStart);
        wbuf(f,L.cLosort);
        wbuf(f,L.cLosortStart);
        wbuf(f,L.faceRestrict);
        wbuf(f,L.faceFlip);
        wbuf(f,L.Prow);
        wbuf(f,L.Pcol);
        wbuf(f,L.Pval);
        wbuf(f,L.rapSrcKind);
        wbuf(f,L.rapSrcIdx);
        wbuf(f,L.rapDstKind);
        wbuf(f,L.rapDstIdx);
        wbuf(f,L.rapW);
    }
    int nCol = static_cast<int>(A.coloring.size());
    std::fwrite(&nCol,sizeof(nCol),1,f);
    for (const auto& c : A.coloring)
    {
        std::fwrite(&c.nColors,sizeof(c.nColors),1,f);
        wbuf(f,c.cells);
        wbuf(f,c.start);
        std::size_t ns = c.startH.size();
        std::fwrite(&ns,sizeof(ns),1,f);
        if (ns) std::fwrite(c.startH.data(),sizeof(label),ns,f);
    }
    std::fwrite(&magic,sizeof(magic),1,f);                 // trailing sentinel (truncation/corruption check)
    std::fclose(f);
}
bool loadAMGCache(
    const std::string& path,
    AMGData& A)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto fail = [&]()
    {
        std::fclose(f);
        return false;
    };
    unsigned magic = 0;
    if (std::fread(&magic,sizeof(magic),1,f)!=1 || magic!=AMG_CACHE_MAGIC) return fail();
    int nFine=0, nLev=0;
    char gs=0, sa=0;
    if (std::fread(&nFine,sizeof(nFine),1,f)!=1 || std::fread(&nLev,sizeof(nLev),1,f)!=1
        || std::fread(&gs,1,1,f)!=1 || std::fread(&sa,1,1,f)!=1) return fail();
    if ((bool)gs != useGS() || (bool)sa != useSA()) return fail();   // smoother/aggregation mode (env) must match
    A = AMGData{};
    A.nFine = nFine;
    A.gsSmooth = gs;
    A.saSmooth = sa;
    A.level.resize(nLev);
    bool ok = true;
    for (auto& L : A.level)
    {
        ok = ok && std::fread(&L.nFine,sizeof(int),1,f)==1 && std::fread(&L.nCoarse,sizeof(int),1,f)==1
                && std::fread(&L.nCoarseFaces,sizeof(int),1,f)==1 && std::fread(&L.nTriples,sizeof(int),1,f)==1;
        ok = ok && rbuf(f,L.map) && rbuf(f,L.cOwn) && rbuf(f,L.cNei) && rbuf(f,L.cOwnerStart) && rbuf(f,L.cLosort)
                && rbuf(f,L.cLosortStart) && rbuf(f,L.faceRestrict) && rbuf(f,L.faceFlip)
                && rbuf(f,L.Prow) && rbuf(f,L.Pcol) && rbuf(f,L.Pval)
                && rbuf(f,L.rapSrcKind) && rbuf(f,L.rapSrcIdx) && rbuf(f,L.rapDstKind) && rbuf(f,L.rapDstIdx) && rbuf(f,L.rapW);
        if (!ok) break;
        L.cDiag.resize(L.nCoarse);
        L.cUpper.resize(L.nCoarseFaces);
        L.cLower.resize(L.nCoarseFaces);   // VALUES via Galerkin
    }
    int nCol = 0;
    ok = ok && std::fread(&nCol,sizeof(nCol),1,f)==1;
    if (ok) A.coloring.resize(nCol);
    for (int i = 0; ok && i < nCol; ++i)
    {
        auto& c = A.coloring[i];
        ok = ok && std::fread(&c.nColors,sizeof(c.nColors),1,f)==1 && rbuf(f,c.cells) && rbuf(f,c.start);
        std::size_t ns = 0;
        ok = ok && std::fread(&ns,sizeof(ns),1,f)==1;
        if (ok)
        {
            c.startH.resize(ns);
            ok = ok && (ns==0 || std::fread(c.startH.data(),sizeof(label),ns,f)==ns);
        }
    }
    unsigned tail = 0;
    ok = ok && std::fread(&tail,sizeof(tail),1,f)==1 && tail==AMG_CACHE_MAGIC;   // sentinel
    std::fclose(f);
    if (!ok) return false;
    A.nCoarse = A.level.empty() ? nFine : A.level.front().nCoarse;
    A.nCoarseFaces = A.level.empty() ? 0 : A.level.front().nCoarseFaces;
    finalizeAMG(A, nFine);
    return true;
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
        zeroT<scalar><<<nBlocks(L.nCoarse),TPB>>>(L.nCoarse, L.cDiag.data());
        zeroT<scalar><<<nBlocks(L.nCoarseFaces),TPB>>>(L.nCoarseFaces, L.cUpper.data());
        zeroT<scalar><<<nBlocks(L.nCoarseFaces),TPB>>>(L.nCoarseFaces, L.cLower.data());
        if (A.saSmooth)                                          // general Galerkin A_c = P^T A P (precomputed RAP recipe)
            rapScatterK<<<nBlocks(L.nTriples),TPB>>>(L.nTriples, L.rapSrcKind.data(), L.rapSrcIdx.data(), L.rapW.data(),
                L.rapDstKind.data(), L.rapDstIdx.data(), fd, fu, fl, L.cDiag.data(), L.cUpper.data(), L.cLower.data());
        else                                                  // injection Galerkin (default): face-restrict scatter
        {
            galDiagK<<<nBlocks(L.nFine),TPB>>>(L.nFine, L.map.data(), fd, L.cDiag.data());
            galFaceK<<<nBlocks(nFaces),TPB>>>(nFaces, L.faceRestrict.data(), L.faceFlip.data(), fu, fl,
                                              L.cDiag.data(), L.cUpper.data(), L.cLower.data());
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

// Power iteration on D^-1 A -> estimate of its largest eigenvalue (sets the Chebyshev smoothing interval). Uses
// host-scalar reductions (deviceDot), so it MUST run outside any captured graph, called once per matrix, gated on
// High-frequency seed for the power iteration. The CONSTANT vector is the Laplacian's
// near-null-space (the SMALLEST eigenvalue of D^-1 A), so starting there makes power iteration
// reach lambda_max only through round-off: in exact arithmetic a constant start is orthogonal to
// every non-constant mode and never leaves the null space. In few iterations it UNDER-estimates
// lambda_max badly, and an under-estimated Chebyshev interval [., 1.1*lambdaMax] leaves the true
// top modes ABOVE the interval, where the Chebyshev polynomial amplifies rather than damps them
// -> the smoother diverges (observed as a stall at ~1.6e-3 on a 1M-cell Laplacian). A period-7
// sawtooth injects broad spectral content, exactly as the SA path's hostSpectralRadius does.
__global__
void fillSeedK(
    int n,
    scalar* __restrict__ x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0 + 0.1*static_cast<scalar>(i % 7);
}
// amg.spectrumReady. x/ax/z are grid-g sized scratch (the V-cycle's vX/vAx/vR, free before the first cycle).
static scalar estimateLambdaMax(
    const DeviceLduView& A,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& ax,
    DeviceBuffer<scalar>& z)
{
    const int n = A.nCells;
    fillSeedK<<<nBlocks(n),TPB>>>(n, x.data());
    scalar lam = 1.0;
    // 25 iterations, not 12: power iteration converges to lambda_max FROM BELOW, and every ms of
    // under-estimate risks divergence in the Chebyshev smoother, while an over-estimate only mildly
    // slows convergence -> spend the extra iterations, they run once per matrix.
    for (int it = 0; it < 25; ++it)
    {
        const scalar nrm = std::sqrt(deviceDot(x, x));
        if (!(nrm > 0.0)) break;
        deviceScale(x, 1.0/nrm);                                  // x <- x/||x||
        deviceAmul(A, x, ax);                                     // ax = A x
        invDiagMulK<<<nBlocks(n),TPB>>>(n, ax.data(), A.diag, z.data());  // z = D^-1 A x
        lam = std::sqrt(deviceDot(z, z));                         // ||D^-1 A x|| (||x||=1) -> Rayleigh-quotient bound
        deviceCopy(x, z);
    }
    return lam;
}

// One-time Chebyshev spectrum estimate per matrix (host scalars -> must precede any graph capture). lambdaMax[g] for
// the SMOOTHED grids 0..nLevels-1; the coarsest grid stays a many-sweep Jacobi solve. No-op unless Chebyshev is
// selected (no power-iteration cost when lambdaMax is unused) or already computed for this matrix.
static void ensureSpectrum(
    AMGData& amg,
    const DeviceLduView& A)
{
    if (!useChebyshev() || amg.spectrumReady) return;
    amg.lambdaMax[0] = estimateLambdaMax(A, amg.vX[0], amg.vAx[0], amg.vR[0]);
    for (int g = 1; g < amg.nLevels(); ++g)
        amg.lambdaMax[g] = estimateLambdaMax(amg.level[g-1].coarseView(), amg.vX[g], amg.vAx[g], amg.vR[g]);
    amg.spectrumReady = true;
}

// Preconditioned Chebyshev (polynomial) smoother of degree `deg` on the interval [upper/CHEB_EIGRATIO, upper] of
// D^-1 A (upper = CHEB_UPPER_SAFETY*lambdaMax). Damps the high-frequency error far better per flop than weighted-Jacobi, this is
// the standard GPU-AMG smoother upgrade. x is the in/out guess (zero on pre-smooth, prolonged correction on post-);
// d is grid-sized scratch. All coefficients are host CONSTANTS (no device reductions) -> graph-capturable.
static void chebyshevSmooth(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& d,
    DeviceBuffer<scalar>& ax,
    scalar lambdaMax,
    int deg)
{
    const int n = A.nCells;
    // Bias the interval top ABOVE the power-iteration estimate: the estimate converges to
    // lambda_max from below, so the true top eigenvalues can sit just past it, exactly where the
    // Chebyshev polynomial amplifies. Over-covering the interval only mildly slows smoothing;
    // under-covering it diverges. 1.2 (not 1.1) buys margin for the residual under-estimate.
    const scalar upper = CHEB_UPPER_SAFETY*lambdaMax, lower = upper/CHEB_EIGRATIO;
    const scalar theta = 0.5*(upper+lower), delta = 0.5*(upper-lower), sigma = theta/delta;
    zeroT<scalar><<<nBlocks(n),TPB>>>(n, d.data());                       // avoid 0*Inf=NaN on the c1=0 first step
    deviceAmul(A, x, ax);                                        // step 0: d = (1/theta)(b-Ax)/D ; x += d
    chebStepK<<<nBlocks(n),TPB>>>(n, b.data(), ax.data(), A.diag, 0.0, 1.0/theta, d.data(), x.data());
    scalar rho = 1.0/sigma;
    for (int k = 1; k < deg; ++k)
    {
        const scalar rhoNew = 1.0/(2.0*sigma - rho);
        const scalar c1 = rho*rhoNew, c2 = 2.0*rhoNew/delta;
        deviceAmul(A, x, ax);
        chebStepK<<<nBlocks(n),TPB>>>(n, b.data(), ax.data(), A.diag, c1, c2, d.data(), x.data());
        rho = rhoNew;
    }
}

// Two-stage Gauss-Seidel smoother (BRAE_AMG_TSGS), OpenFOAM v2606 twoStageGaussSeidel (arXiv:2111.09512, "Scaled
// Smoothers for Navier-Stokes Pressure Projection"). A fully parallel polynomial smoother: per sweep,
//   r = A x ; z = (b - r)/D ; x += z ;                              (weighted-Jacobi stage, w baked into the /D update)
//   for k in 0..order-1: r = U z ; x += (-1)^(k+1) z_k ; z = r/D    (Neumann-series correction in the strictly-upper
//                                                                    triangle, alternating signs)
// No serial dependency and no coloring -> graph-capturable at a fixed order (all coefficients are host constants,
// mult = +-1). order 0 reduces to the weighted-Jacobi smoothT exactly. z/r are level-sized scratch (the V-cycle's
// vD/vAx). Symmetry: the AMG-PCG outer solver needs an SPD preconditioner, so the correction triangle switches
// between pre (forward=U) and post (backward=L=U^T) smooth to keep the V-cycle symmetric; a U-only pre+post is
// non-symmetric here and wrecks PCG convergence.
static void twoStageGSSmooth(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& z,
    DeviceBuffer<scalar>& r,
    int nSweeps,
    int order,
    bool forward)
{
    const int n = A.nCells;
    for (int s = 0; s < nSweeps; ++s)
    {
        deviceAmul(A, x, r);                                                                 // r = A x
        if (order == 0)   // == weighted Jacobi
        {
            smoothT<scalar><<<nBlocks(n),TPB>>>(n, b.data(), r.data(), A.diag, x.data());
            continue;
        }
        tsgsZUpdateK<<<nBlocks(n),TPB>>>(n, b.data(), r.data(), A.diag, z.data(), x.data());  // z=(b-Ax)/D; x+=z
        scalar mult = -1.0;
        for (int k = 0; k < order; ++k)
        {
            if (forward) upperMulK<<<nBlocks(n),TPB>>>(n, A.ownerStart, A.nei, A.upper, z.data(), r.data());              // r = U z (pre)
            else         lowerMulK<<<nBlocks(n),TPB>>>(n, A.losortStart, A.losort, A.owner, A.lower, z.data(), r.data()); // r = L z (post -> symmetric)
            if (k < order - 1) tsgsCorrSaveK<<<nBlocks(n),TPB>>>(n, r.data(), A.diag, mult, z.data(), x.data());
            else               tsgsCorrLastK<<<nBlocks(n),TPB>>>(n, r.data(), A.diag, mult, x.data());
            mult = -mult;
        }
    }
}

// One multicolor Gauss-Seidel sweep over grid g. forward = colors 0..nColors-1, backward = reverse. The V-cycle
// pre-smooths forward and post-smooths backward so the whole preconditioner is SYMMETRIC (symmetric GS), required
// by the plain-CG callers; the SIMPLE loop's flexible CG would tolerate either. One kernel launch per color, all
// bounds host-constant -> the sweep is still capturable into the V-cycle graph (same as the Jacobi/Chebyshev paths).
static void gsSweep(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    const GridColoring& gc,
    bool forward)
{
    for (int ci = 0; ci < gc.nColors; ++ci)
    {
        const int col = forward ? ci : (gc.nColors-1-ci);
        const int lo = gc.startH[col], hi = gc.startH[col+1];
        const int nc = hi - lo;
        if (nc <= 0) continue;
        gsColorT<scalar><<<nBlocks(nc),TPB>>>(lo, hi, gc.cells.data(), b.data(), A.diag,
            A.ownerStart, A.nei, A.upper, A.losortStart, A.losort, A.owner, A.lower, x.data());
    }
}

// Greedy cell coloring for the symGaussSeidel scalar solver, cached per mesh (keyed on A.owner, the graph is fixed
// per mesh; the matrix VALUES change every call, the coloring does not). Shared by the host-loop and device-graph paths.
static const GridColoring& gsColoringFor(const DeviceLduView& A)
{
    static std::map<const label*, GridColoring> colorCache;
    auto it = colorCache.find(A.owner);
    if (it == colorCache.end())
    {
        const int nF = A.nInternalFaces;
        std::vector<label> ownerH(nF), neiH(nF);
        cudaMemcpy(ownerH.data(), A.owner, nF*sizeof(label), cudaMemcpyDeviceToHost);
        cudaMemcpy(neiH.data(),   A.nei,   nF*sizeof(label), cudaMemcpyDeviceToHost);
        Coloring c = greedyColor(ownerH, neiH, A.nCells);
        GridColoring gc;
        gc.nColors=c.nColors;
        gc.cells.copyFrom(c.cells);
        gc.start.copyFrom(c.start);
        gc.startH=c.start;
        it = colorCache.emplace(A.owner, std::move(gc)).first;
    }
    return it->second;
}

// Device-resident symGaussSeidel (BRAE_GS_DEVICE). The host-loop symGaussSeidel below reads the residual norm to the
// host after every sweep (a blocking D2H) to decide {converged? stop}; on the turbulent k/epsilon path that read is
// the dominant host stall. This variant moves the decision onto the GPU via a conditional-graph WHILE node: the loop
// body (fwd+bwd sweep, residual recompute, normalize, stop-test) is captured once and replayed on-device while a tiny
// kernel drives the loop condition with cudaGraphSetConditional(). It is exact, not batched: the on-device test is the
// same per-sweep predicate (res<tol || res<relTol*init || iter>=maxIter), so the sweep count and psi are bit-identical
// to the host loop. The only host sync left is the one initial-residual read per solve, which SIMPLE needs anyway.
//
// Graph-replay invariants: the k/epsilon matrix is reassembled into per-call temporaries each outer iter, so the
// incoming (diag,upper,lower,b) pointers move; the graph instead references stable cache-owned buffers and the current
// matrix is copied in (async D2D) before each replay. The cache is keyed on psi (a persistent field buffer), so k and
// epsilon get independent graph instances. normFactor is a device-resident scalar; tol/relTol/maxIter are baked in at
// capture. WHILE nodes require CUDA >= 13; on older toolkits this path compiles out and the host loop below is used.
#if CUDART_VERSION >= 13000
#define BRAE_HAS_GS_DEVICE 1
__global__
void gsScaleInvK(
    scalar* a,
    const scalar* b)
{
    if (threadIdx.x==0 && blockIdx.x==0) *a = (*a) / (*b);
}
__global__
void gsSetCondK(
    cudaGraphConditionalHandle h,
    const scalar* res,
    scalar tol,
    const scalar* init,
    scalar relTol,
    int* iter,
    int maxIter)
{
    if (threadIdx.x || blockIdx.x) return;
    const int it = ++(*iter);
    const scalar r = *res;
    const bool stop = (r < tol) || (r < relTol * (*init)) || (it >= maxIter);
    cudaGraphSetConditional(h, stop ? 0u : 1u);
}
struct GSGraphCache
{
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphConditionalHandle handle{};
    const void* key = nullptr;
    DeviceBuffer<scalar> gsDiag, gsUpper, gsLower, gsB, Ax, r;        // stable, graph-referenced
    DeviceBuffer<scalar> gNormF, gInit, gRes;
    DeviceBuffer<int> gIter;
    ~GSGraphCache()
    {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};
static scalar deviceSymGaussSeidelGraph(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const GridColoring& gc = gsColoringFor(A);
    static auto& cache = *new std::map<const void*, GSGraphCache>();  // leaked (no static-dtor-after-context-teardown hazard)
    GSGraphCache& c = cache[psi.data()];
    const int nC = A.nCells, nF = A.nInternalFaces;
    c.gsDiag.resize(nC);
    c.gsUpper.resize(nF);
    c.gsLower.resize(nF);
    c.gsB.resize(nC);
    c.Ax.resize(nC);
    c.r.resize(nC);
    c.gNormF.resize(1);
    c.gInit.resize(1);
    c.gRes.resize(1);
    c.gIter.resize(1);
    // copy the current matrix + rhs into the stable graph-referenced buffers (async D2D, no host sync)
    cudaMemcpyAsync(c.gsDiag.data(),  A.diag,  nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gsUpper.data(), A.upper, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gsLower.data(), A.lower, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gsB.data(),     b.data(),nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gNormF.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice,   cudaStreamPerThread);
    DeviceLduView sA = A;
    sA.diag = c.gsDiag.data();
    sA.upper = c.gsUpper.data();
    sA.lower = c.gsLower.data();  // stable values + mesh topology
    // initial residual (also PRE-SIZES Ax/r so the capture below allocates nothing): r = b - A*psi
    deviceAmul(sA, psi, c.Ax);
    deviceCopy(c.r, c.gsB);
    deviceAxpy(-1.0, c.Ax, c.r);
    deviceSumMagInto(c.r, c.gInit.data());
    gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.gInit.data(), c.gNormF.data());     // gInit = sum|r| / normFactor
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.gInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "gs init D2H");
    cudaStreamSynchronize(cudaStreamPerThread);                                       // the ONE host sync / solve (OF initialResidual)
    if (initRes < tol)
    {
        if (std::getenv("BRAE_GS_DEBUG")) std::printf("    GS[dev] init=%.4e (skip)\n", initRes);
        return initRes;
    }
    cudaMemsetAsync(c.gIter.data(), 0, sizeof(int), cudaStreamPerThread);
    if (!c.exec || c.key != psi.data())                                            // (re)capture the WHILE-graph
    {
        if (c.exec)
        {
            cudaGraphExecDestroy(c.exec);
            c.exec  = nullptr;
        }
        if (c.graph)
        {
            cudaGraphDestroy(c.graph);
            c.graph = nullptr;
        }
        cudaCheck(cudaGraphCreate(&c.graph, 0), "gs graph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 1, cudaGraphCondAssignDefault), "gs cond handle");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(cudaGraphAddNode(&cnode, c.graph, nullptr, nullptr, 0, &cp), "gs cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "gs capture begin");
        gsSweep(sA, c.gsB, psi, gc, true);                                           // forward color sweep
        gsSweep(sA, c.gsB, psi, gc, false);                                          // reverse color sweep
        deviceAmul(sA, psi, c.Ax);
        deviceCopy(c.r, c.gsB);
        deviceAxpy(-1.0, c.Ax, c.r);  // r = b - A*psi
        deviceSumMagInto(c.r, c.gRes.data());
        gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.gRes.data(), c.gNormF.data());      // finalRes = sum|r| / normFactor
        gsSetCondK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.gRes.data(), tol, c.gInit.data(), relTol, c.gIter.data(), maxIter);
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "gs capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "gs graph instantiate");
        c.key = psi.data();
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "gs graph launch");       // replay: loop runs to convergence on-device
    if (std::getenv("BRAE_GS_DEBUG"))
    {
        cudaStreamSynchronize(cudaStreamPerThread);
        scalar fr;
        int ni;
        cudaMemcpy(&fr, c.gRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ni, c.gIter.data(), sizeof(int), cudaMemcpyDeviceToHost);
        std::printf("    GS[dev] init=%.4e final=%.4e iters=%d (relTol=%.2g)\n", initRes, fr, ni, relTol);
    }
    return initRes;
}
#endif // CUDART_VERSION >= 13000

// symGaussSeidel scalar solver, a port of OpenFOAM smoothSolver + symGaussSeidelSmoother. OF updates each cell
// psi[c] = (b[c] - sum_{j!=c} A[c][j]*psi[j]) / diag[c] (the gsColorT gather), forward sweep then reverse. Here the
// sweep is parallelised by greedy multicolouring (one color = non-adjacent cells, so writes never race and off-diagonal
// reads see already-swept colors == true GS), converging to the same solution as OF's lexicographic order. One symGS
// sweep = gsSweep(forward) + gsSweep(reverse); the outer loop mirrors smoothSolver::solve (initial residual + normFactor,
// then {smooth; recompute r; finalRes = sumMag(r)/normFactor} until finalRes<tol || finalRes<relTol*initRes || maxIter).
// The coloring depends only on the graph (owner/nei), fixed per mesh, so it is built once and cached. Robust on the
// stiff near-wall k/omega system where Jacobi-BiCGStab amplifies the y+~1 instability.
static scalar deviceSymGaussSeidelF32(
    const DeviceLduView&,
    const DeviceBuffer<scalar>&,
    DeviceBuffer<scalar>&,
    scalar,
    scalar,
    scalar,
    int);   // FP32 turbulence GS (defined below)
scalar deviceSymGaussSeidel(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    DeviceSolverPerf* perf)
{
    // BRAE_TURB_FP32: solve in FP32 (half the bytes on the BW-bound sweeps; loose turbulence tol >> FP32 eps).
    static const bool turbF32 = std::getenv("BRAE_TURB_FP32") != nullptr;
    if (turbF32)
    {
        scalar r = deviceSymGaussSeidelF32(A, b, psi, normFactor, tol, relTol, maxIter);
        if (perf) *perf = {r, r, 1};
        return r;
    }   // opt-in path: report init only
    // BRAE_TURB_JACOBI: weighted-Jacobi solve (fully parallel, no colour sync). Experiment only -- Jacobi needs ~2x
    // the sweeps -> ~2x the bandwidth on this BW-bound path, so it is slower than the graphed GS. Same stop test as GS.
    static const bool turbJac = std::getenv("BRAE_TURB_JACOBI") != nullptr;
    if (turbJac)
    {
        DeviceBuffer<scalar> Ax, r;
        deviceAmul(A, psi, Ax);
        deviceCopy(r, b);
        deviceAxpy(-1.0, Ax, r);
        const scalar initRes = deviceSumMag(r) / normFactor;
        if (initRes >= tol)
        {
            int iter = 0;
            scalar finalRes = initRes;
            for (; iter < maxIter; ++iter)
            {
                deviceAmul(A, psi, Ax);                                                // Ax = A psi
                smoothT<scalar><<<nBlocks(A.nCells),TPB>>>(A.nCells, b.data(), Ax.data(), A.diag, psi.data());   // psi += w*(b-Ax)/diag
                deviceAmul(A, psi, Ax);
                deviceCopy(r, b);
                deviceAxpy(-1.0, Ax, r);
                finalRes = deviceSumMag(r) / normFactor;
                if (finalRes < tol || finalRes < relTol*initRes) break;
            }
            if (std::getenv("BRAE_GS_DEBUG")) std::printf("    JAC init=%.4e final=%.4e iters=%d\n", initRes, finalRes, iter+1);
        }
        if (perf) *perf = {initRes, initRes, 1};   // opt-in path: report init only
        return initRes;
    }
    // BRAE_GS_DEVICE: run the convergence loop ON-DEVICE (conditional-graph WHILE node), same per-sweep stop decision,
    // zero per-sweep host D2H. EXACT (not batched): identical sweep count + bit-identical psi vs this host loop.
    static const bool useGraph = std::getenv("BRAE_GS_DEVICE") != nullptr;
#ifdef BRAE_HAS_GS_DEVICE
    if (useGraph)
    {
        scalar r = deviceSymGaussSeidelGraph(A, b, psi, normFactor, tol, relTol, maxIter);
        if (perf) *perf = {r, r, 1};
        return r;
    }   // opt-in path: report init only
#else
    if (useGraph)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            std::fprintf(stderr, "brae: BRAE_GS_DEVICE needs CUDA >= 13.0; falling back to host-loop GS\n");
        }
    }
#endif
    const GridColoring& gc = gsColoringFor(A);
    DeviceBuffer<scalar> Ax, r;
    deviceAmul(A, psi, Ax);
    deviceCopy(r, b);
    deviceAxpy(-1.0, Ax, r);   // r = b - A*psi
    const scalar initRes = deviceSumMag(r) / normFactor;
    if (initRes < tol)
    {
        if (std::getenv("BRAE_GS_DEBUG")) std::printf("    GS init=%.4e (skip)\n", initRes);
        if (perf) *perf = {initRes, initRes, 0};
        return initRes;
    }
    int iter = 0;
    scalar finalRes = initRes;
    for (; iter < maxIter; ++iter)
    {
        gsSweep(A, b, psi, gc, true);                                    // forward color sweep
        gsSweep(A, b, psi, gc, false);                                   // reverse color sweep
        deviceAmul(A, psi, Ax);
        deviceCopy(r, b);
        deviceAxpy(-1.0, Ax, r);
        finalRes = deviceSumMag(r) / normFactor;                        // exact per-sweep check (matches OF's loose smoothSolver stop)
        if (finalRes < tol || finalRes < relTol*initRes) break;
    }
    if (std::getenv("BRAE_GS_DEBUG")) std::printf("    GS init=%.4e final=%.4e iters=%d (relTol=%.2g)\n", initRes, finalRes, iter+1, relTol);
    if (perf) *perf = {initRes, finalRes, iter + 1};   // default path: full OF-style init/final/nIter
    return initRes;
}

// Recursive V-cycle at grid g: x_g <- M^-1 b_g (x_g overwritten). g==nLevels is the coarsest grid (an approximate
// solve; single-block or fused-cluster when small); above it: pre-smooth, restrict the residual to g+1, recurse,
// prolong the correction, post-smooth.
static void vcycleAt(
    int g,
    AMGData& amg,
    const DeviceLduView& Ag,
    const DeviceBuffer<scalar>& bg,
    DeviceBuffer<scalar>& xg)
{
    const int n = Ag.nCells;
    zeroT<scalar><<<nBlocks(n),TPB>>>(n, xg.data());
    if (g == amg.nLevels())                                    // coarsest: approximate solve
    {
        // BRAE_NCOARSE_CG overrides the coarsest PCG iteration count.
        static const int ncoarseCG = [](){ const char* e = std::getenv("BRAE_NCOARSE_CG"); return (e && std::atoi(e) > 0) ? std::atoi(e) : NCOARSE_CG; }();
        if (n <= SB_CG_MAX) deviceCoarsePCG(Ag, bg, xg, ncoarseCG);            // tiny coarsest: single-block Jacobi-PCG (cheap+accurate)
        else if (n <= SB_MAX) deviceCoarseJacobiSingleBlock(Ag, bg, xg, NCOARSE);   // larger: single-block many-sweep Jacobi
        else if (deviceCoarseFitsCluster(n) && n <= COARSE_FUSE_MAX) deviceCoarseJacobiFused(Ag, bg, xg, NCOARSE);
        else for (int s = 0; s < NCOARSE; ++s)
        {
            deviceAmul(Ag, xg, amg.vAx[g]);
            smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
        }
        return;
    }
    if (useChebyshev()) chebyshevSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], amg.lambdaMax[g], chebDeg());  // pre-smooth (x=0)
    else if (useTSGS()) twoStageGSSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], NPRE, tsgsOrder(), true);     // OF v2606 twoStageGaussSeidel (fwd)
    else if (amg.gsSmooth) for (int s = 0; s < NPRE; ++s) gsSweep(Ag, bg, xg, amg.coloring[g], true);    // forward GS
    else for (int s = 0; s < NPRE; ++s)
    {
        deviceAmul(Ag, xg, amg.vAx[g]);
        smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
    }
    deviceAmul(Ag, xg, amg.vAx[g]);
    residualT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), amg.vR[g].data());
    const int nc = amg.level[g].nCoarse;
    const AMGLevel& Lg = amg.level[g];
    zeroT<scalar><<<nBlocks(nc),TPB>>>(nc, amg.vB[g+1].data());
    if (amg.saSmooth)                                            // restrict rc = P^T r (sparse smoothed prolongator)
        restrictSparseK<<<nBlocks(n),TPB>>>(n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vR[g].data(), amg.vB[g+1].data());
    else
        restrictT<scalar><<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vR[g].data(), amg.vB[g+1].data());
    vcycleAt(g+1, amg, amg.level[g].coarseView(), amg.vB[g+1], amg.vX[g+1]);   // recurse to the next coarser grid
    if (amg.corrScaling)
    {
        // OF-GAMG-style scaled coarse correction: alpha = (r . A pc)/(A pc . A pc); xg += alpha*pc. The single
        // line-search per level fixes the magnitude of the prolongation. All scalars stay on the device
        // (deviceDotInto/scaleFactorK/AxpyDev) so it adds NO host sync and remains capturable into the V-cycle graph.
        if (amg.saSmooth) prolongToSparseK<<<nBlocks(n),TPB>>>(n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vX[g+1].data(), amg.vPc[g].data());
        else              prolongToK<<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vX[g+1].data(), amg.vPc[g].data());  // c = P*corr
        deviceAmul(Ag, amg.vPc[g], amg.vAx[g]);                                       // Ac
        deviceDotInto(amg.vR[g], amg.vPc[g], amg.sScNum.data());                      // r . c   (energy-min, OF GAMG)
        deviceDotInto(amg.vPc[g], amg.vAx[g], amg.sScDen.data());                     // c . Ac  (= ||c||_A^2 > 0)
        scaleFactorK<<<1,1>>>(amg.sScNum.data(), amg.sScDen.data(), amg.sScAlpha.data());
        deviceAxpyDev(amg.sScAlpha.data(), amg.vPc[g], xg);                           // xg += alpha * c
    }
    else if (amg.saSmooth)
        prolongSparseK<<<nBlocks(n),TPB>>>(n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vX[g+1].data(), xg.data());
    else
        prolongT<scalar><<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vX[g+1].data(), xg.data());
    if (useChebyshev()) chebyshevSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], amg.lambdaMax[g], chebDeg());  // post-smooth
    else if (useTSGS()) twoStageGSSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], NPOST, tsgsOrder(), false);  // twoStageGaussSeidel (bwd -> symmetric)
    else if (amg.gsSmooth) for (int s = 0; s < NPOST; ++s) gsSweep(Ag, bg, xg, amg.coloring[g], false);  // backward GS (symmetric V-cycle)
    else for (int s = 0; s < NPOST; ++s)
    {
        deviceAmul(Ag, xg, amg.vAx[g]);
        smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
    }
}

// Mixed-precision (FP32) V-cycle: a mirror of the default V-cycle (weighted-Jacobi + map restrict/prolong) with the
// matrix values and work vectors in FP32 (half the bytes on the BW-bound SpMV+smooth); the topology (labels) is
// shared with the FP64 path. The coarsest level casts back to FP64 and reuses the exact FP64 coarse solve. Only the
// default smoother/aggregation is supported here (SA/GS/Chebyshev/corrScaling stay FP64; the caller gates on those).
// It reuses the templated zeroT/smoothT/residualT/restrictT/prolongT<float>; only the casts and the FP32 SpMV (amulF)
// are FP32-specific.
// cast_<scalar,float> = FP64->FP32 (down), cast_<float,scalar> = FP32->FP64 (up).
template <class S, class D>
__global__
void cast_(
    int n,
    const S* __restrict__ s,
    D* __restrict__ d)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) d[i] = (D)s[i];
}
__global__
void amulFK(
    int nC,
    const float* __restrict__ diag,
    const float* __restrict__ upper,
    const float* __restrict__ lower,
    const label* __restrict__ nei,
    const label* __restrict__ owner,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const float* __restrict__ psi,
    float* __restrict__ Apsi)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    float s = diag[c]*psi[c];
    for (int f=ownerStart[c]; f<ownerStart[c+1]; ++f)
        s += upper[f]*psi[nei[f]];
    for (int k=losortStart[c]; k<losortStart[c+1]; ++k)
    {
        const int f=losort[k];
        s += lower[f]*psi[owner[f]];
    }
    Apsi[c]=s;
}
// FP32 matrix = shared (FP64-side) topology + FP32 value arrays.
struct LduF
{
    const float* diag;
    const float* upper;
    const float* lower;
    const label* nei;
    const label* owner;
    const label* ownerStart;
    const label* losort;
    const label* losortStart;
    int nCells;
    int nInternalFaces;
};
static LduF lduF(
    const DeviceLduView& t,
    const DeviceBuffer<float>& d,
    const DeviceBuffer<float>& u,
    const DeviceBuffer<float>& l)
{
    return { d.data(), u.data(), l.data(), t.nei, t.owner, t.ownerStart, t.losort, t.losortStart, t.nCells, t.nInternalFaces };
}
static void amulF(
    const LduF& A,
    const float* x,
    float* y)
{
    amulFK<<<nBlocks(A.nCells),TPB>>>(A.nCells, A.diag, A.upper, A.lower, A.nei, A.owner, A.ownerStart, A.losort, A.losortStart, x, y);
}
// Cast the (current) FP64 fine + coarse matrices to their FP32 mirrors. Allocates the mirrors + FP32 work vectors once.
static void amgCastFP32(
    AMGData& amg,
    const DeviceLduView& A)
{
    const int G = amg.nLevels();
    if (!amg.fp32Alloc)
    {
        amg.fDiag.resize(G+1);
        amg.fUpper.resize(G+1);
        amg.fLower.resize(G+1);
        amg.vAxF.resize(G+1);
        amg.vRF.resize(G+1);
        amg.vXF.resize(G+1);
        amg.vBF.resize(G+1);
        for (int g=0; g<=G; ++g)
        {
            const DeviceLduView v = (g==0) ? A : amg.level[g-1].coarseView();
            amg.fDiag[g].resize(v.nCells);
            amg.fUpper[g].resize(v.nInternalFaces);
            amg.fLower[g].resize(v.nInternalFaces);
            amg.vAxF[g].resize(v.nCells);
            amg.vRF[g].resize(v.nCells);
            amg.vXF[g].resize(v.nCells);
            amg.vBF[g].resize(v.nCells);
        }
        amg.fp32Alloc = true;
    }
    for (int g=0; g<=G; ++g)
    {
        const DeviceLduView v = (g==0) ? A : amg.level[g-1].coarseView();
        cast_<scalar,float><<<nBlocks(v.nCells),TPB>>>(v.nCells, v.diag, amg.fDiag[g].data());
        if (v.nInternalFaces>0)
        {
            cast_<scalar,float><<<nBlocks(v.nInternalFaces),TPB>>>(v.nInternalFaces, v.upper, amg.fUpper[g].data());
            cast_<scalar,float><<<nBlocks(v.nInternalFaces),TPB>>>(v.nInternalFaces, v.lower, amg.fLower[g].data());
        }
    }
}
// FP32 recursive V-cycle. topoG = grid g's FP64 view (topology + the FP64 coarsest solve); Ag = grid g's FP32 matrix.
static void vcycleAtF(
    int g,
    AMGData& amg,
    const DeviceLduView& topoG,
    const LduF& Ag,
    const float* bg,
    float* xg)
{
    const int n = Ag.nCells;
    zeroT<float><<<nBlocks(n),TPB>>>(n, xg);
    if (g == amg.nLevels())                                    // coarsest: cast to FP64, exact FP64 solve, cast back
    {
        cast_<float,scalar><<<nBlocks(n),TPB>>>(n, bg, amg.vB[g].data());
        static const int ncoarseCG = [](){ const char* e=std::getenv("BRAE_NCOARSE_CG"); return (e&&std::atoi(e)>0)?std::atoi(e):NCOARSE_CG; }();
        if (n <= SB_CG_MAX) deviceCoarsePCG(topoG, amg.vB[g], amg.vX[g], ncoarseCG);
        else
        {
            zeroT<scalar><<<nBlocks(n),TPB>>>(n, amg.vX[g].data());
            for (int s=0;s<NCOARSE;++s)
            {
                deviceAmul(topoG, amg.vX[g], amg.vAx[g]);
                smoothT<scalar><<<nBlocks(n),TPB>>>(n, amg.vB[g].data(), amg.vAx[g].data(), topoG.diag, amg.vX[g].data());
            }
        }
        cast_<scalar,float><<<nBlocks(n),TPB>>>(n, amg.vX[g].data(), xg);
        return;
    }
    for (int s=0; s<NPRE; ++s)
    {
        amulF(Ag, xg, amg.vAxF[g].data());
        smoothT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), Ag.diag, xg);
    }
    amulF(Ag, xg, amg.vAxF[g].data());
    residualT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), amg.vRF[g].data());
    const AMGLevel& Lg = amg.level[g];
    const int nc = Lg.nCoarse;
    zeroT<float><<<nBlocks(nc),TPB>>>(nc, amg.vBF[g+1].data());
    restrictT<float><<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vRF[g].data(), amg.vBF[g+1].data());
    const DeviceLduView topoC = Lg.coarseView();
    const LduF Ac = lduF(topoC, amg.fDiag[g+1], amg.fUpper[g+1], amg.fLower[g+1]);
    vcycleAtF(g+1, amg, topoC, Ac, amg.vBF[g+1].data(), amg.vXF[g+1].data());
    prolongT<float><<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vXF[g+1].data(), xg);
    for (int s=0; s<NPOST; ++s)
    {
        amulF(Ag, xg, amg.vAxF[g].data());
        smoothT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), Ag.diag, xg);
    }
}

// Turbulence FP32 Gauss-Seidel (BRAE_TURB_FP32). The k/eps/omega transport is solved to a loose tolerance, far above
// FP32 machine-epsilon, so running the GS in FP32 (half the matrix+vector bytes on the BW-bound sweeps) preserves the
// converged field. Matrix + b + field are cast to FP32 once per solve; FP32 multicolor GS sweeps; the residual norm
// is taken in FP64 for a clean convergence check; the field is cast back to FP64. The sweep is gsColorT<float>, the
// same templated smoother as the FP64 path (see gsSweep).
static void gsSweepF(
    const LduF& A,
    const float* b,
    const GridColoring& gc,
    float* x,
    bool forward)
{
    for (int ci = 0; ci < gc.nColors; ++ci)
    {
        const int col = forward ? ci : (gc.nColors-1-ci);
        const int lo = gc.startH[col], hi = gc.startH[col+1];
        const int nc = hi - lo;
        if (nc <= 0) continue;
        gsColorT<float><<<nBlocks(nc),TPB>>>(lo, hi, gc.cells.data(), b, A.diag, A.ownerStart, A.nei, A.upper,
                                       A.losortStart, A.losort, A.owner, A.lower, x);
    }
}
struct GSFP32Cache
{
    DeviceBuffer<float> dF, uF, lF, bF, xF, AxF, rF;
    DeviceBuffer<scalar> rD;
};
static scalar deviceSymGaussSeidelF32(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const GridColoring& gc = gsColoringFor(A);
    static auto& cache = *new std::map<const void*, GSFP32Cache>();
    GSFP32Cache& c = cache[psi.data()];
    const int nC = A.nCells, nF = A.nInternalFaces;
    c.dF.resize(nC);
    c.uF.resize(nF);
    c.lF.resize(nF);
    c.bF.resize(nC);
    c.xF.resize(nC);
    c.AxF.resize(nC);
    c.rF.resize(nC);
    c.rD.resize(nC);
    cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, A.diag, c.dF.data());
    if (nF > 0)
    {
        cast_<scalar,float><<<nBlocks(nF),TPB>>>(nF, A.upper, c.uF.data());
        cast_<scalar,float><<<nBlocks(nF),TPB>>>(nF, A.lower, c.lF.data());
    }
    cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, b.data(),   c.bF.data());
    cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, psi.data(), c.xF.data());
    const LduF sA{ c.dF.data(), c.uF.data(), c.lF.data(), A.nei, A.owner, A.ownerStart, A.losort, A.losortStart, nC, nF };
    auto residNorm = [&]() -> scalar       // r = b - A x (FP32), |r|_1 in FP64
    {
        amulF(sA, c.xF.data(), c.AxF.data());
        residualT<float><<<nBlocks(nC),TPB>>>(nC, c.bF.data(), c.AxF.data(), c.rF.data());
        cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, c.rF.data(), c.rD.data());
        return deviceSumMag(c.rD) / normFactor;
    };
    const scalar initRes = residNorm();
    if (initRes >= tol)
    {
        int iter = 0;
        scalar finalRes = initRes;
        for (; iter < maxIter; ++iter)
        {
            gsSweepF(sA, c.bF.data(), gc, c.xF.data(), true);
            gsSweepF(sA, c.bF.data(), gc, c.xF.data(), false);
            finalRes = residNorm();
            if (finalRes < tol || finalRes < relTol*initRes) break;
        }
        if (std::getenv("BRAE_GS_DEBUG")) std::printf("    GS[fp32] init=%.4e final=%.4e iters=%d\n", initRes, finalRes, iter+1);
    }
    cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, c.xF.data(), psi.data());         // FP32 field -> FP64
    return initRes;
}

#ifdef BRAE_HAS_GS_DEVICE
// Device-resident AMG-PCG (BRAE_PCG_DEVICE). Extends the on-device stop decision from the k/eps Gauss-Seidel to the
// pressure solve. The V-cycle is already graph-capturable and the Krylov scalars already live on-device; the only
// host-in-the-loop part left is the per-iteration residual read driving {converged? stop}. Here the steady-state PCG
// iteration (V-cycle precond -> dot -> beta -> p-update -> SpMV -> dot -> alpha -> axpy x -> axpy r -> residual ->
// stop-test) is captured once into a conditional WHILE graph and replayed on-device, with a 1-thread pcgSetCondK
// driving cudaGraphSetConditional() from the same per-iter predicate (res<tol || res<relTol*init || iter>=maxIter).
// So the whole pressure solve is one graph launch, zero per-iter D2H, exact (same iteration count and psi as the host
// PCG). Iteration 0 differs (p=w, no beta) so it runs explicitly first (one residual read for the rare converge-in-1
// early-out); the WHILE body is the uniform iter-1+ recurrence. The pressure matrix is stable across SIMPLE steps
// (the V-cycle graph already keys on A.diag), so no matrix copy is needed; the graph references A, psi, amg.rA/wA and
// the cache's persistent pA/Ax directly. Non-corrScaling only (flexible-CG falls back to the host loop); normFactor
// is a device-resident scalar.
__global__
void pcgSetCondK(
    cudaGraphConditionalHandle h,
    const scalar* res,
    scalar tol,
    const scalar* init,
    scalar relTol,
    int* iter,
    int maxIter)
{
    if (threadIdx.x || blockIdx.x) return;
    const int it = ++(*iter);
    const scalar fr = *res;          // res already normalized by normFactor
    const bool conv = (fr < tol) || (relTol > 0.0 && fr < relTol * (*init));
    cudaGraphSetConditional(h, (conv || it >= maxIter) ? 0u : 1u);
}
static DeviceSolverPerf deviceAMGPCGGraph(
    const DeviceLduView& A,
    AMGData& amg,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const int nC = A.nCells;
    amg.corrScaling = false;
    DeviceBuffer<scalar>& wA = amg.wA;
    DeviceBuffer<scalar>& rA = amg.rA;
    if (!amg.pcgCache) amg.pcgCache = std::make_unique<PCGGraphCache>();          // per-solver lifetime: dies with amg, so no stale reuse across solvers
    PCGGraphCache& c = *amg.pcgCache;
    c.pA.resize(nC);
    c.Ax.resize(nC);
    c.sNormF.resize(1);
    c.sInit.resize(1);
    c.sRes.resize(1);
    c.sIter.resize(1);
    cudaMemcpyAsync(c.sNormF.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice, cudaStreamPerThread);
    scalar* dWArA = amg.sWArA.data();
    scalar* dWArAold = amg.sWArAold.data();
    scalar* dPap  = amg.sPap.data();
    scalar* dAlpha   = amg.sAlpha.data();
    scalar* dNegAlpha = amg.sNegAlpha.data();
    scalar* dBeta = amg.sBeta.data();
    ensureSpectrum(amg, A);                                                       // one-time Chebyshev spectrum (pre-capture)
    // FP32 V-cycle inside the device-resident PCG: cast matrices once per solve; the WHILE body captures the FP32
    // vcycleAtF automatically (host-scalar-free). Outer Krylov + residual stay FP64 (accuracy preserved).
    const bool fp32 = useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev();
    if (fp32) amgCastFP32(amg, A);
    const LduF A0 = fp32 ? lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]) : LduF{};
    auto applyPrec = [&]()
    {
        if (fp32)
        {
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), wA.data());
        }
        else vcycleAt(0, amg, A, rA, wA);
    };
    // initial residual r = b - A psi  (also PRE-SIZES the V-cycle scratch + pA/Ax so the capture allocates nothing)
    deviceAmul(A, psi, c.Ax);
    deviceCopy(rA, b);
    deviceAxpy(-1.0, c.Ax, rA);
    DeviceSolverPerf perf;
    deviceSumMagInto(rA, c.sInit.data());
    gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sInit.data(), c.sNormF.data());
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.sInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg init D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    perf.initialResidual = initRes;
    perf.finalResidual = initRes;
    auto convergedHost = [&](scalar fr){ return (fr < tol) || (relTol > 0.0 && fr < relTol*initRes); };
    if (convergedHost(initRes))
    {
        perf.nIterations = 0;
        return perf;
    }
    // iteration 0 (explicit: p = w, no beta)
    applyPrec();                                         // wA = M^-1 rA
    deviceDotInto(wA, rA, dWArA);
    deviceCopy(c.pA, wA);
    deviceAmul(A, c.pA, wA);
    deviceDotInto(wA, c.pA, dPap);
    deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
    deviceAxpyDev(dAlpha, c.pA, psi);
    deviceAxpyDev(dNegAlpha, wA, rA);
    deviceSumMagInto(rA, c.sRes.data());
    gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sRes.data(), c.sNormF.data());
    scalar res1;
    cudaCheck(cudaMemcpyAsync(&res1, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg it0 D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    if (convergedHost(res1) || maxIter <= 1)
    {
        perf.finalResidual = res1;
        perf.nIterations = 1;
        return perf;
    }
    cudaMemsetAsync(c.sIter.data(), 0, sizeof(int), cudaStreamPerThread);   // WHILE-body counter (0 = iter-1)
    // WHILE body = steady-state iteration 1+ (captured once, replayed on-device)
    if (!c.exec || c.key != psi.data() || c.keyTol != tol || c.keyRelTol != relTol || c.keyMaxIter != maxIter)
    {
        if (c.exec)
        {
            cudaGraphExecDestroy(c.exec);
            c.exec  = nullptr;
        }
        if (c.graph)
        {
            cudaGraphDestroy(c.graph);
            c.graph = nullptr;
        }
        cudaCheck(cudaGraphCreate(&c.graph, 0), "pcg graph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 1, cudaGraphCondAssignDefault), "pcg cond handle");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(cudaGraphAddNode(&cnode, c.graph, nullptr, nullptr, 0, &cp), "pcg cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "pcg capture begin");
        deviceScalarCopy(dWArA, dWArAold);
        applyPrec();                                     // wA = M^-1 rA
        deviceDotInto(wA, rA, dWArA);
        deviceScalarDiv(dWArA, dWArAold, dBeta);          // Fletcher-Reeves beta
        deviceFusedScaleAxpy(c.pA, dBeta, wA);            // p = beta*p + w  [fused]
        deviceAmul(A, c.pA, wA);
        deviceDotInto(wA, c.pA, dPap);
        deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
        deviceAxpyDev(dAlpha, c.pA, psi);
        deviceAxpyDev(dNegAlpha, wA, rA);
        deviceSumMagInto(rA, c.sRes.data());
        gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sRes.data(), c.sNormF.data());   // normalized residual
        pcgSetCondK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.sRes.data(), tol, c.sInit.data(), relTol, c.sIter.data(), maxIter-1);   // maxIter-1: iteration 0 ran outside the loop
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "pcg capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "pcg graph instantiate");
        c.key = psi.data();
        c.keyTol = tol;
        c.keyRelTol = relTol;
        c.keyMaxIter = maxIter;
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "pcg graph launch");
    scalar finalRes;
    int whileIters;
    cudaCheck(cudaMemcpyAsync(&finalRes, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg final D2H");
    cudaCheck(cudaMemcpyAsync(&whileIters, c.sIter.data(), sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg iters D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    perf.finalResidual = finalRes;
    perf.nIterations = 1 + whileIters;
    static const bool dbgCyc = std::getenv("BRAE_AMG_CYCLES") != nullptr;
    if (dbgCyc) std::fprintf(stderr, "[AMG] p cycles=%d finalRes=%.3e (device)\n", perf.nIterations, perf.finalResidual);
    return perf;
}

// DISTRIBUTED whole-loop graph PCG: the exact structure of deviceAMGPCGGraph (the entire steady-state PCG WHILE
// body captured once into a conditional graph and replayed on-device -- no per-iteration host launches or syncs),
// but the matvec is deviceParallelAmul (halo-coupled) and every dot is globalised by an ON-STREAM NVSHMEM reduce
// through the halo's DeviceReducer, so the halo put/barrier and the collective are captured INSIDE the graph body.
// This is the 2x lever: the ~13ms/iter of GPU work stops being spread across ~hundreds of separate launches. It
// requires the on-stream NVSHMEM ops to be graph-capturable (NVSHMEM 3.6 has cuda_graph paths) and 1 PE/GPU (the
// MPG host-MPI reduce fallback is NOT capturable) -- so it is real-multi-GPU only. Persistent buffers (psi/matrix
// held across steps by the caller) make it capture ONCE per run; otherwise it re-instantiates per step (still
// correct). Falls back to the direct deviceParallelAMGPCG if the caller keeps buffers non-persistent.
DeviceSolverPerf deviceParallelAMGPCGGraph(
    const DeviceLduView& A,
    AMGData& amg,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const cudaStream_t strm = cudaStreamPerThread;
    const int nC = A.nCells;
    amg.corrScaling = false;
    DeviceBuffer<scalar>& wA = amg.wA;
    DeviceBuffer<scalar>& rA = amg.rA;
    if (!amg.pcgCache) amg.pcgCache = std::make_unique<PCGGraphCache>();
    PCGGraphCache& c = *amg.pcgCache;
    c.pA.resize(nC); c.Ax.resize(nC); c.sNormF.resize(1); c.sInit.resize(1); c.sRes.resize(1); c.sIter.resize(1);
    cudaMemcpyAsync(c.sNormF.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice, strm);
    scalar* dWArA = amg.sWArA.data();  scalar* dWArAold = amg.sWArAold.data();
    scalar* dPap  = amg.sPap.data();   scalar* dAlpha = amg.sAlpha.data();
    scalar* dNegAlpha = amg.sNegAlpha.data(); scalar* dBeta = amg.sBeta.data();
    ensureSpectrum(amg, A);
    const bool fp32 = useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev();
    if (fp32) amgCastFP32(amg, A);
    const LduF A0 = fp32 ? lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]) : LduF{};
    auto applyPrec = [&]()
    {
        if (fp32)
        {
            cast_<scalar,float><<<nBlocks(nC),TPB,0,strm>>>(nC, rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB,0,strm>>>(nC, amg.vXF[0].data(), wA.data());
        }
        else vcycleAt(0, amg, A, rA, wA);
    };
    DeviceReducer& R = halo.reducer();
    auto gdot = [&](const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, scalar* dOut)
    {   // *dOut = global sum of x.y (local dot -> on-stream NVSHMEM reduce -> copy)
        deviceDotInto(x, y, R.src()); R.sumReduce(1, strm); deviceScalarCopy(R.dst(), dOut);
    };
    auto gsum = [&](const DeviceBuffer<scalar>& x, scalar* dOut)   // *dOut = global |x|_1
    {
        deviceSumMagInto(x, R.src()); R.sumReduce(1, strm); deviceScalarCopy(R.dst(), dOut);
    };

    deviceParallelAmul(A, halo, ifaceCoeffs, psi, c.Ax);          // r = b - A psi
    deviceCopy(rA, b);
    deviceAxpy(-1.0, c.Ax, rA);
    DeviceSolverPerf perf;
    gsum(rA, c.sInit.data());
    gsScaleInvK<<<1,1,0,strm>>>(c.sInit.data(), c.sNormF.data());
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.sInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "pgraph init D2H");
    cudaStreamSynchronize(strm);
    perf.initialResidual = initRes; perf.finalResidual = initRes;
    auto convergedHost = [&](scalar fr){ return (fr < tol) || (relTol > 0.0 && fr < relTol*initRes); };
    if (convergedHost(initRes)) { perf.nIterations = 0; return perf; }

    applyPrec();                                                  // iteration 0 (explicit)
    gdot(wA, rA, dWArA);
    deviceCopy(c.pA, wA);
    deviceParallelAmul(A, halo, ifaceCoeffs, c.pA, wA);
    gdot(wA, c.pA, dPap);
    deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
    deviceAxpyDev(dAlpha, c.pA, psi);
    deviceAxpyDev(dNegAlpha, wA, rA);
    gsum(rA, c.sRes.data());
    gsScaleInvK<<<1,1,0,strm>>>(c.sRes.data(), c.sNormF.data());
    scalar res1;
    cudaCheck(cudaMemcpyAsync(&res1, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "pgraph it0 D2H");
    cudaStreamSynchronize(strm);
    if (convergedHost(res1) || maxIter <= 1) { perf.finalResidual = res1; perf.nIterations = 1; return perf; }
    cudaMemsetAsync(c.sIter.data(), 0, sizeof(int), strm);

    if (!c.exec || c.key != psi.data() || c.keyTol != tol || c.keyRelTol != relTol || c.keyMaxIter != maxIter)
    {
        if (c.exec)  { cudaGraphExecDestroy(c.exec);  c.exec  = nullptr; }
        if (c.graph) { cudaGraphDestroy(c.graph);     c.graph = nullptr; }
        cudaCheck(cudaGraphCreate(&c.graph, 0), "pgraph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 1, cudaGraphCondAssignDefault), "pgraph cond handle");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(cudaGraphAddNode(&cnode, c.graph, nullptr, nullptr, 0, &cp), "pgraph cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(strm, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "pgraph capture begin");
        deviceScalarCopy(dWArA, dWArAold);
        applyPrec();
        gdot(wA, rA, dWArA);
        deviceScalarDiv(dWArA, dWArAold, dBeta);
        deviceFusedScaleAxpy(c.pA, dBeta, wA);
        deviceParallelAmul(A, halo, ifaceCoeffs, c.pA, wA);
        gdot(wA, c.pA, dPap);
        deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
        deviceAxpyDev(dAlpha, c.pA, psi);
        deviceAxpyDev(dNegAlpha, wA, rA);
        gsum(rA, c.sRes.data());
        gsScaleInvK<<<1,1,0,strm>>>(c.sRes.data(), c.sNormF.data());
        pcgSetCondK<<<1,1,0,strm>>>(c.handle, c.sRes.data(), tol, c.sInit.data(), relTol, c.sIter.data(), maxIter-1);
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(strm, &tmp), "pgraph capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "pgraph instantiate");
        c.key = psi.data();
        c.keyTol = tol;
        c.keyRelTol = relTol;
        c.keyMaxIter = maxIter;
    }
    cudaCheck(cudaGraphLaunch(c.exec, strm), "pgraph launch");
    scalar finalRes; int whileIters;
    cudaCheck(cudaMemcpyAsync(&finalRes, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "pgraph final D2H");
    cudaCheck(cudaMemcpyAsync(&whileIters, c.sIter.data(), sizeof(int), cudaMemcpyDeviceToHost, strm), "pgraph iters D2H");
    cudaStreamSynchronize(strm);
    perf.finalResidual = finalRes;
    perf.nIterations = 1 + whileIters;
    return perf;
}
#endif // BRAE_HAS_GS_DEVICE

#ifndef BRAE_HAS_GS_DEVICE
// CUDA < 13: WHILE-node conditional graphs are unavailable, so the whole-loop graph cannot be built. Return a
// sentinel (nIterations < 0); the caller falls back to the direct (non-graph) distributed AMG-PCG.
DeviceSolverPerf deviceParallelAMGPCGGraph(
    const DeviceLduView& A, AMGData&, DeviceHalo&,
    const std::vector<DeviceBuffer<scalar>>&,
    const DeviceBuffer<scalar>&, DeviceBuffer<scalar>&,
    scalar, scalar, scalar, int)
{
    (void)A;
    DeviceSolverPerf p; p.nIterations = -1; return p;
}
#endif

DeviceSolverPerf deviceAMGPCG(
    const DeviceLduView& A,
    AMGData& amg,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    bool captureVcycle,
    int checkEvery,
    bool corrScaling)
{
#ifdef BRAE_HAS_GS_DEVICE
    {
        static const bool pcgDev = envFlag("BRAE_PCG_DEVICE", true);                 // device-resident pressure solve (default ON; opt out BRAE_PCG_DEVICE=0)
        if (pcgDev && !corrScaling) return deviceAMGPCGGraph(A, amg, b, psi, normFactor, tol, relTol, maxIter);
    }
#endif
    const int nC = A.nCells;
    amg.corrScaling = corrScaling;                             // seen by vcycleAt (scaled prolongation)
    DeviceBuffer<scalar>& wA = amg.wA;                          // persistent (fixed addr) so the captured graph stays valid
    DeviceBuffer<scalar>& rA = amg.rA;
    DeviceBuffer<scalar> pA(nC), Ax(nC), rOld(nC);             // rOld: previous residual for flexible-CG beta (corrScaling)
    deviceAmul(A, psi, Ax);
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);
    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA)/normFactor;
    perf.finalResidual = perf.initialResidual;
    auto converged = [&](scalar fr){ return (fr<tol) || (relTol>0.0 && fr<relTol*perf.initialResidual); };

    // One-time Chebyshev spectrum estimate (host scalars -> must precede any graph capture); no-op unless Chebyshev.
    ensureSpectrum(amg, A);

    // The V-cycle (wA = M^-1 rA) has no host-scalar dependencies and runs on fixed buffers (amg.rA/wA + amg's
    // persistent work buffers), so it is captured into a CUDA graph and replayed. The graph is cached in amg.gcache,
    // keyed on the fine-matrix pointer A.diag: captured once and replayed across all PCG iters and SIMPLE steps (the
    // matrix values change in-buffer each step; the graph reads them at replay), re-captured only when the key changes.
    // FP32 mixed precision (default smoother/aggregation only; SA/GS/Cheb/corrScaling stay FP64): the matrices are cast
    // once per solve, and each application casts rA -> FP32 in / FP32 -> wA out.
    const bool fp32 = useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev() && !corrScaling;
    if (fp32)
    {
        amgCastFP32(amg, A);
        static bool once=false;
        if(!once)
        {
            once=true;
            if(std::getenv("BRAE_AMG_CYCLES")) std::fprintf(stderr,"[AMG] FP32 V-cycle ENGAGED (levels=%d)\n", amg.nLevels());
        }
    }
    AMGGraphCache& gc = *amg.gcache;
    auto applyPrecond = [&]()
    {
        if (fp32)
        {
            const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);
            auto runF = [&]()                                                        // cast in -> FP32 V-cycle -> cast out
            {
                cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, rA.data(), amg.vBF[0].data());
                vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
                cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), wA.data());
            };
            if (!captureVcycle)
            {
                runF();
                return;
            }
            AMGGraphCache& gcf = *amg.gcacheF;                                          // graph the FP32 V-cycle (host-scalar-free)
            if (!gcf.exec || gcf.key != A.diag)
            {
                if (gcf.exec)
                {
                    cudaGraphExecDestroy(gcf.exec);
                    gcf.exec = nullptr;
                }
                if (gcf.graph)
                {
                    cudaGraphDestroy(gcf.graph);
                    gcf.graph = nullptr;
                }
                cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amgF capture begin");
                runF();
                cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gcf.graph), "amgF capture end");
                cudaCheck(cudaGraphInstantiate(&gcf.exec, gcf.graph, 0), "amgF graph instantiate");
                gcf.key = A.diag;
            }
            cudaCheck(cudaGraphLaunch(gcf.exec, cudaStreamPerThread), "amgF graph launch");
            return;
        }
        if (!captureVcycle)
        {
            vcycleAt(0, amg, A, rA, wA);
            return;
        }
        if (!gc.exec || gc.key != A.diag)
        {
            if (gc.exec)
            {
                cudaGraphExecDestroy(gc.exec);
                gc.exec = nullptr;
            }
            if (gc.graph)
            {
                cudaGraphDestroy(gc.graph);
                gc.graph = nullptr;
            }
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amg capture begin");
            vcycleAt(0, amg, A, rA, wA);
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gc.graph), "amg capture end");
            cudaCheck(cudaGraphInstantiate(&gc.exec, gc.graph, 0), "amg graph instantiate");
            gc.key = A.diag;
        }
        cudaCheck(cudaGraphLaunch(gc.exec, cudaStreamPerThread), "amg graph launch");
    };

    // Device-resident Krylov scalars: wArA / pAp / alpha / beta live on the device and feed the *Dev kernels by
    // pointer, so the host never blocks on the two dot-products driving the recurrence. The only per-iter host sync is
    // the residual-norm read for the convergence check (K=1 -> exact per-iter; checkEvery>1 batches it, trading
    // exactness for fewer syncs). The scalars are persistent AMGData members, allocated once.
    scalar* dWArA    = amg.sWArA.data();
    scalar* dWArAold = amg.sWArAold.data();
    scalar* dPap     = amg.sPap.data();
    scalar* dAlpha   = amg.sAlpha.data();
    scalar* dNegAlpha= amg.sNegAlpha.data();
    scalar* dBeta    = amg.sBeta.data();
    scalar* dResNorm = amg.sResNorm.data();
    const int K = (checkEvery > 1) ? checkEvery : 1;            // residual read cadence (1 = exact per-iter)
    int nIter = 0;
    if (!converged(perf.finalResidual))
    {
        do
        {
            if (nIter > 0) deviceScalarCopy(dWArA, dWArAold);
            applyPrecond();                                     // wA = M^-1 rA (recursive V-cycle)
            deviceDotInto(wA, rA, dWArA);
            if (nIter == 0) deviceCopy(pA, wA);                 // p = w
            else if (corrScaling)                             // flexible CG (Polak-Ribiere+): nonlinear precond
            {
                deviceDotInto(wA, rOld, amg.sZrOld.data());
                flexBetaK<<<1,1>>>(dWArA, amg.sZrOld.data(), dWArAold, dBeta);
                deviceFusedScaleAxpy(pA, dBeta, wA);
            }
            else
            {
                deviceScalarDiv(dWArA, dWArAold, dBeta);     // Fletcher-Reeves beta
                deviceFusedScaleAxpy(pA, dBeta, wA);
            }
            if (corrScaling) deviceCopy(rOld, rA);              // save r_k for iter k+1 (rA is about to be updated)
            deviceAmul(A, pA, wA);
            deviceDotInto(wA, pA, dPap);
            deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
            deviceAxpyDev(dAlpha, pA, psi);
            deviceAxpyDev(dNegAlpha, wA, rA);
            ++nIter;
            if (nIter % K == 0 || nIter >= maxIter)           // read |r|_1 only every K iters (K=1: exact per-iter)
            {
                deviceSumMagInto(rA, dResNorm);
                perf.finalResidual = deviceReadScalar(dResNorm)/normFactor;   // the only host sync
            }
        } while (nIter < maxIter && !converged(perf.finalResidual));
    }
    perf.nIterations = nIter;
    static const bool dbgCyc = std::getenv("BRAE_AMG_CYCLES") != nullptr;   // benchmark: AMG V-cycles per pressure solve
    if (dbgCyc) std::fprintf(stderr, "[AMG] p cycles=%d finalRes=%.3e\n", nIter, perf.finalResidual);
    return perf;
}

// Cast the fine + coarse matrices to their FP32 mirrors for this solve's V-cycles. Call ONCE per solve, AFTER
// amgGalerkin has updated the FP64 operators (the coarse matrices change every SIMPLE step). No-op unless FP32
// mixed precision applies (BRAE_AMG_FP32 default on; the SA/GS/Chebyshev paths stay FP64). After this,
// amgVCycleApply runs the FP32 V-cycle automatically -- the same mixed-precision path the single-GPU deviceAMGPCG
// uses, now available to the distributed Krylov (the pressure V-cycle's SpMV+smoother are bandwidth-bound, so
// FP32 halves their bytes ~= 2x, while the r->FP32/FP32->z casts and the FP64 coarsest solve keep it accurate).
void amgPrepareFP32(AMGData& amg, const DeviceLduView& A)
{
    if (useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev())
        amgCastFP32(amg, A);
}

// z = M^-1 r : ONE symmetric AMG V-cycle applied as a PRECONDITIONER, factored out of deviceAMGPCG so the
// DISTRIBUTED Krylov (device_pcg.cu) can precondition each rank's LOCAL block with AMG. The V-cycle is built on
// internal faces only, so it omits the processor-interface coupling -- exactly the block-Jacobi / additive-Schwarz
// design: the outer distributed matvec (deviceParallelAmul) supplies the interface, the local V-cycle need only
// approximate the local block. amg must be built (buildAMG) and current (amgGalerkin gives it this step's coarse
// operators). Runs the FP32 V-cycle when amgPrepareFP32 cast the matrices this solve, else the FP64 one.
void amgVCycleApply(AMGData& amg, const DeviceLduView& A,
                    const DeviceBuffer<scalar>& r, DeviceBuffer<scalar>& z, bool captureVcycle)
{
    ensureSpectrum(amg, A);                // one-time Chebyshev spectrum estimate (no-op unless the Chebyshev smoother is on)
    const bool fp32 = amg.fp32Alloc && useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev();
    const int nC = A.nCells;

    if (!captureVcycle)
    {
        // DIRECT launch (default): every V-cycle kernel is a fresh launch.
        if (fp32)
        {
            const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);         // grid-0 FP32 matrix view
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, r.data(), amg.vBF[0].data());   // r -> FP32
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());              // FP32 V-cycle
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), z.data());   // FP32 -> z (FP64)
        }
        else vcycleAt(0, amg, A, r, z);    // FP64 V-cycle
        return;
    }

    // GRAPH REPLAY: the V-cycle is host-scalar-free and runs on FIXED buffers (amg.rA/wA), so capture it once
    // (keyed on A.diag; the coarse VALUES change each step but the graph references the buffers, read at replay) and
    // replay -- removing the launch overhead of the V-cycle's many small kernels (13 levels x SpMV+smoother per PCG
    // iter, the dominant per-iteration launch cost). Copy r -> persistent amg.rA, replay, amg.wA -> z. Same
    // capture/replay the single-GPU deviceAMGPCG uses; now available to the distributed pressure solve.
    deviceCopy(amg.rA, r);
    if (fp32)
    {
        const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);
        AMGGraphCache& gcf = *amg.gcacheF;
        if (!gcf.exec || gcf.key != A.diag)
        {
            if (gcf.exec)  { cudaGraphExecDestroy(gcf.exec);  gcf.exec  = nullptr; }
            if (gcf.graph) { cudaGraphDestroy(gcf.graph);     gcf.graph = nullptr; }
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amgF capture begin");
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, amg.rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), amg.wA.data());
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gcf.graph), "amgF capture end");
            cudaCheck(cudaGraphInstantiate(&gcf.exec, gcf.graph, 0), "amgF graph instantiate");
            gcf.key = A.diag;
        }
        cudaCheck(cudaGraphLaunch(gcf.exec, cudaStreamPerThread), "amgF graph launch");
    }
    else
    {
        AMGGraphCache& gc = *amg.gcache;
        if (!gc.exec || gc.key != A.diag)
        {
            if (gc.exec)  { cudaGraphExecDestroy(gc.exec);  gc.exec  = nullptr; }
            if (gc.graph) { cudaGraphDestroy(gc.graph);     gc.graph = nullptr; }
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amg capture begin");
            vcycleAt(0, amg, A, amg.rA, amg.wA);
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gc.graph), "amg capture end");
            cudaCheck(cudaGraphInstantiate(&gc.exec, gc.graph, 0), "amg graph instantiate");
            gc.key = A.diag;
        }
        cudaCheck(cudaGraphLaunch(gc.exec, cudaStreamPerThread), "amg graph launch");
    }
    deviceCopy(z, amg.wA);
}

} // namespace brae
