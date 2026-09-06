#pragma once
// ---------------------------------------------------------------------------------------------------------------
// device_amg_detail.cuh -- shared inline primitives for the AMG hierarchy build + V-cycle solve.
//
// Extracted from device_amg.cu (the >3k-line implementation) so it can later be split into operation-group
// translation units WITHOUT duplicating these inline helpers or losing cross-TU inlining: every .cu that
// includes this header gets its own inlined copy of safeDiag/warpReduceSum/blockDot and the env-flag readers.
// The non-template __device__ helpers are 'static' (internal linkage per TU) so they stay inlinable and
// ODR-clean when several TUs include this; the env readers are 'inline' (one shared instance). This is an
// implementation detail -- deliberately NOT pulled into the public device_amg.cuh.
// ---------------------------------------------------------------------------------------------------------------
#include "cf_types.cuh"     // scalar / label
#include <cuda_runtime.h>   // __shfl_down_sync, __syncthreads
#include <cstdlib>          // std::getenv / std::atoi (env-flag readers)

// Device-resident (conditional-graph WHILE) solver variants require CUDA 13's stream-capture-to-graph + conditional
// nodes. Defined here (not inline in device_amg.cu) so it gates BOTH the device-resident GS solver and the
// device-resident AMG-PCG driver consistently across the (to-be-split) AMG-core translation units.
#if CUDART_VERSION >= 13000
#define BRAE_HAS_GS_DEVICE 1
#endif

namespace brae {

// finalizeAMG builds the V-cycle scratch + persistent PCG buffers + graph caches for a hierarchy. It is the one
// build-time helper shared by both the fresh-build path (buildAMG) and the cache-reload path (loadAMGCache, now in
// device_amg_cache.cu), so it is declared here (external linkage) and defined once in device_amg.cu.
struct AMGData;                              // fwd (full definition in device_amg.cuh)
void finalizeAMG(AMGData& A, int nFine);

// The per-level Galerkin gather lists, rebuilt from the agglomeration a cache load restores. Declared
// here because loadAMGCache lives in its own translation unit and must call it for every level.
void rebuildGalerkinGather(AMGLevel& L, int nFine);

constexpr int TPB = 256;
constexpr scalar OMEGA = 0.8;          // weighted-Jacobi relaxation
constexpr int NPRE = 1, NPOST = 1, NCOARSE = 500;
constexpr int CHEB_DEG_DEFAULT = 2;    // Chebyshev smoother degree (per pre/post smooth)
constexpr scalar CHEB_EIGRATIO = 30.0; // Chebyshev interval [upper/ratio, upper]
constexpr scalar CHEB_UPPER_SAFETY = 1.2; // interval top = safety * lambdaMax; > 1 to over-cover (under-cover diverges)

// Coarsest-grid solver caps (shared by the V-cycle dispatch in device_amg.cu and the solvers in device_amg_coarse.cu):
constexpr int CCL = 8;                   // cluster-fused coarse solve: blocks per cluster (whole coarse level = one cluster)
constexpr int COARSE_FUSE_MAX = 4096;    // cluster-fused single-cluster cap (measured crossover ~6k)
constexpr int SB_MAX = 2048;             // single-block coarsest Jacobi cap (2*SB_MAX doubles shared = 32KB)
constexpr int SB_CG_MAX = 1024;          // single-block coarsest PCG cap ((5*nC+TPB) doubles shared <= 48KB at nC=1024)
constexpr int NCOARSE_CG = 16;           // coarsest PCG iterations (dispatch default; override with BRAE_NCOARSE_CG).

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
// helper is a STRICT no-op for any diagonal of magnitude > the type floor (1e-300 for FP64,
// 1e-30 for FP32 -- 1e-300 underflows to 0.0f in float, which would defeat the guard entirely),
// so it never perturbs a well-formed operator; it only replaces a would-be division by (near) zero.
template <typename T>
static __device__ __forceinline__
T safeDiag(T d)
{
    const T floor = T(sizeof(T) == 4 ? 1e-30 : 1e-300);
    if (d > floor)  return d;
    if (d < -floor) return d;
    return (d < T(0)) ? -floor : floor;
}

// Block-wide dot product a.b, warp-shuffle reduce: each warp reduces its lanes with __shfl_down (no barrier), then
// warp 0 reduces the per-warp partials. Three barriers per dot instead of ~log2(blockDim), which matters because the
// single-block coarsest CG runs many sequential dots on one SM. red[] holds the per-warp partials; all threads return the sum.
static __device__ __forceinline__
scalar warpReduceSum(scalar v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
static __device__
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

} // namespace brae
