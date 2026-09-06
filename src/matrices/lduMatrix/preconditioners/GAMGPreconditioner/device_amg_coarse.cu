// Coarse-grid solvers for the AMG V-cycle: the single-launch coarsest-grid solves (cluster-fused Jacobi,
// single-block Jacobi, single-block Jacobi-PCG) + deviceCoarseFitsCluster. Split out of device_amg.cu; the
// kernels reuse the shared inline device helpers (safeDiag/blockDot/warpReduceSum) via device_amg_detail.cuh,
// so each keeps its own inlined copy (no cross-TU inlining loss). deviceCoarseJacobiLoop lives here too (it
// launches the V-cycle's smoothT<> from amg_kernels.cuh + deviceAmul).
#include "device_amg.cuh"          // public deviceCoarse* decls + DeviceLduView/DeviceBuffer
#include "device_amg_detail.cuh"   // safeDiag/blockDot/warpReduceSum + TPB/OMEGA/CCL/... constants
#include "amg_kernels.cuh"       // smoothT<> for deviceCoarseJacobiLoop
#include "device_amg_coarse.cuh"   // deviceCoarsePCG/JacobiSingleBlock internal decls (match defs here)
#include "device_ldu.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mutex>
#include <cstddef>

namespace cg = cooperative_groups;

namespace brae {

namespace {
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
} // anon

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


// Iterated coarsest solve: nSweeps of global weighted-Jacobi (deviceAmul + smoothT). Reuses the V-cycle's
// smoothT<> kernel (amg_kernels.cuh) and deviceAmul; kept out of the single-launch solvers above for that reason.
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

} // namespace brae
