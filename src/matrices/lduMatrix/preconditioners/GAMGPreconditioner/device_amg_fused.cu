// THE FUSED COARSE-LEVEL FP32 V-CYCLE.
//
// vcycleAtF launches ~7 kernels a grid (smoothFromZero, product, residual, restrict, prolong, product, smooth)
// plus the coarsest solve's three, each on a grid of a few thousand cells or fewer where the work is 1-2 us and a
// dependent launch costs 3-4. On a decomposed mesh with ~300k cells a rank the pressure solve is latency-bound and
// that tail is most of the block-local cycle. From the first grid of at most BRAE_AMG_FUSE_BELOW cells (1024) this
// kernel runs the REST of the cycle -- down, the coarsest solve, back up -- as ONE block of up to 1024 threads
// with __syncthreads where the launches were. 1024 is MEASURED on an RTX 5090 (tests/test_amg_fused_coarse.cu's
// sweep, captured cycle per call, us unfused / 1024 / 4096 / 16384: pitzDaily 65.6 / 49.2 / 50.0 / 79.8, TurbBig
// 112 / 69.7 / 72.0 / 104, sbMatched 111 / 98.4 / 119 / 209, duct3d 100 / 88.0 / 91.8 / 129): one block's throughput
// loses to the separate launches on grids of a few thousand cells.
//
// BIT-IDENTICAL TO THE UNFUSED CYCLE, by construction: every per-cell expression is amg_cell_ops.cuh's (and
// amg_coarse_dev.cuh's, device_amg_coarse.cu's inverse rows), the same functions the separate kernels call, over
// the same operands in the same order; a cell's value never depends on which thread computes it or on the block
// size. Where the cycle does anything else the fusion does not apply and vcycleAtF recurses as before: coupled
// pair, smoothed aggregation, correction scaling, a coarsest solve other than the inverse or the LU, a gate's
// control, a split-timing solve.
//
// THE DESCRIPTORS ARE THE KERNEL'S ARGUMENTS (FusedArgs), made at every call from the hierarchy's current
// buffers. There is no cached device array to invalidate: a captured graph bakes the pointers at capture exactly
// as it does for the unfused launches, so what the existing graph keys (addressingId, pair epoch, view, scratch
// epoch) guard for those launches guards this one, and nothing here can hold a pointer the pool has since
// handed to another buffer.
#include "device_amg_fused.cuh"
#include "device_amg_detail.cuh"
#include "device_amg_split.cuh"
#include "device_amg_coarse.cuh"
#include "amg_kernels.cuh"        // the cell expressions
#include "amg_coarse_dev.cuh"
#include <atomic>
#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>

namespace brae {

namespace {

// One grid's product over the cells of this thread, each handed to store(cell, value): the rows where the grid has
// them, else the faces. The arrays are written by this kernel (x, ax, r, b), so they are read through plain
// pointers, never the read-only path.
template <class Store>
__device__ __forceinline__
void fusedProduct(const FusedLevel& L, int tid, int nt, const float* psi, Store store)
{
    for (int i = tid; i < L.n; i += nt)
    {
        store(i, L.row != nullptr
                     ? cellSpmvCsrF(i, L.diag, L.row, L.col, L.val, psi)
                     : cellSpmvFaceF(i, L.diag, L.upper, L.lower, L.nei, L.owner, L.ownerStart, L.losort, L.losortStart, psi));
    }
}

// one smoothing sweep: A x into ax, then x += omega (b - ax)/diag; a barrier between (the sweep is Jacobi: every
// product reads the OLD x) and after
__device__ __forceinline__
void fusedSweep(const FusedLevel& L, int tid, int nt)
{
    fusedProduct(L, tid, nt, L.x, [&](int i, float ax) { L.ax[i] = ax; });
    __syncthreads();
    for (int i = tid; i < L.n; i += nt) L.x[i] = cellSmooth<float>(L.x[i], L.b[i], L.ax[i], L.diag[i]);
    __syncthreads();
}

__global__ __launch_bounds__(1024)
void amgFusedVcycleKernel(const __grid_constant__ FusedArgs a)
{
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    // DOWN: pre-smooth, residual, restrict. The restriction also starts the next grid's cycle -- its first sweep from
    // zero (or the zeroing), or the FP64 cast on the coarsest -- in the thread that made each coarse value: one
    // barrier fewer a grid, the same expressions on the same values
    for (int l = 0; l < a.nLev; ++l)
    {
        const FusedLevel& L = a.lev[l];
        const FusedLevel& C = a.lev[l + 1];
        if (l == 0 && a.nPre == 0)
        {
            for (int i = tid; i < L.n; i += nt) L.x[i] = 0.0f;                 // zeroT
            __syncthreads();
        }
        for (int s = 0; s < a.nPre; ++s)
        {
            if (s == 0)                                                       // x = 0: smoothFromZeroT
            {
                if (l == 0)
                {
                    for (int i = tid; i < L.n; i += nt) L.x[i] = cellSmoothFromZero<float>(L.b[i], L.diag[i]);
                    __syncthreads();
                }
                continue;
            }
            fusedSweep(L, tid, nt);
        }
        fusedProduct(L, tid, nt, L.x, [&](int i, float ax)                    // r = b - A x
        {
            L.ax[i] = ax;
            L.r[i] = cellResidual<float>(L.b[i], ax);
        });
        __syncthreads();
        const bool coarsestNext = l + 1 == a.nLev;
        for (int ci = tid; ci < L.nc; ci += nt)
        {
            const float rb = cellRestrict<float>(L.galStart, L.galList, L.r, ci);
            C.b[ci] = rb;
            if (coarsestNext) a.coarse.rcD[ci] = (scalar)rb;                  // cast_<float, scalar>
            else C.x[ci] = a.nPre > 0 ? cellSmoothFromZero<float>(rb, C.diag[ci]) : 0.0f;
        }
        __syncthreads();
    }

    // THE COARSEST GRID: up to FP64, the exact solve, back down to FP32
    {
        const FusedLevel& L = a.lev[a.nLev];
        const FusedCoarse& K = a.coarse;
        const int n = K.n;
        if (a.nLev == 0)
        {
            for (int i = tid; i < n; i += nt) K.rcD[i] = (scalar)L.b[i];      // cast_<float, scalar>
            __syncthreads();
        }
        if (K.inverse)
        {
            if (*K.unsafe != 0)                                               // the LU's substitutions, one warp
            {
                if (tid < 32)
                {
                    scalar yr[TRI_ROWS];
                    #pragma unroll
                    for (int s = 0; s < TRI_ROWS; ++s)
                    {
                        const int i = tid + 32*s;
                        yr[s] = (i < n) ? K.rcD[K.perm[i]] : 0.0;
                    }
                    warpSubstitute(n, K.fc, true, yr, tid);
                    #pragma unroll
                    for (int s = 0; s < TRI_ROWS; ++s)
                    {
                        const int i = tid + 32*s;
                        if (i < n) K.xcD[i] = yr[s];
                    }
                }
            }
            else
            {
                const int lane = tid & 31;
                for (int i = tid >> 5; i < n; i += (nt >> 5))                 // a warp a row
                {
                    const scalar s = inverseRowDot(K.inv, K.rcD, i, n, lane);
                    if (lane == 0) K.xcD[i] = s;
                }
            }
            __syncthreads();
        }
        else                                                                  // coarseLUSolveKernel's steps, y in xcD
        {
            scalar* y = K.xcD;
            const scalar* A = K.lu;
            for (int i = tid; i < n; i += nt) y[i] = K.rcD[i];
            __syncthreads();
            if (tid == 0)
            {
                for (int k = 0; k < n; ++k)
                {
                    const int r = K.piv[k];
                    if (r != k) { const scalar t = y[k]; y[k] = y[r]; y[r] = t; }
                }
            }
            __syncthreads();
            for (int k = 0; k < n; ++k)
            {
                const scalar yk = y[k];
                for (int i = k+1+tid; i < n; i += nt) y[i] -= A[i*n + k]*yk;
                __syncthreads();
            }
            for (int k = n-1; k >= 0; --k)
            {
                if (tid == 0) y[k] /= A[k*n + k];
                __syncthreads();
                const scalar yk = y[k];
                for (int i = tid; i < k; i += nt) y[i] -= A[i*n + k]*yk;
                __syncthreads();
            }
        }
        for (int i = tid; i < n; i += nt) L.x[i] = (float)K.xcD[i];           // cast_<scalar, float>
        __syncthreads();
    }

    // UP: prolong, post-smooth
    for (int l = a.nLev - 1; l >= 0; --l)
    {
        const FusedLevel& L = a.lev[l];
        const FusedLevel& C = a.lev[l + 1];
        for (int i = tid; i < L.n; i += nt) L.x[i] = cellProlong<float>(L.x[i], C.x[L.map[i]]);
        __syncthreads();
        for (int s = 0; s < a.nPost; ++s) fusedSweep(L, tid, nt);
    }
}

// what a gate's control or an instrument has changed in the cycle, so that the unfused kernels must run
bool controlsHoldTheCycle()
{
    static const bool zeroProduct = std::getenv("BRAE_CONTROL_AMG_ZERO_PRODUCT") != nullptr;
    static const bool staleStart = std::getenv("BRAE_CONTROL_AMG_FROM_ZERO_STALE") != nullptr;
    return zeroProduct || staleStart || amgSplit::state().active;
}

// BRAE_AMG_FUSE_BELOW, read at every call (a test switches it between cycles): the grid size at or below which a
// grid is fused; 0 or less = off
int fuseBelow()
{
    const char* e = std::getenv("BRAE_AMG_FUSE_BELOW");
    if (e == nullptr || *e == '\0') return 1024;
    return std::atoi(e);
}

} // anon

namespace
{
std::atomic<long long> nLaunches{0};
std::atomic<int> lastFirst{-1};
std::atomic<int> lastGrids{0};
}

FusedStats amgFusedStats()
{
    FusedStats s;
    s.launches = nLaunches.load();
    s.firstGrid = lastFirst.load();
    s.grids = lastGrids.load();
    return s;
}

void amgFusedVcycleLaunch(const FusedArgs& args, int threads)
{
    amgFusedVcycleKernel<<<1, threads, 0, cudaStreamPerThread>>>(args);
    cudaCheck(cudaGetLastError(), "fused V-cycle launch");
    ++nLaunches;
    lastGrids = args.nLev + 1;
}

bool amgFusedVcycleAt(
    int g,
    AMGData& amg,
    const LduF& Ag,
    const float* bg,
    float* xg)
{
    const int G = amg.nLevels();
    const int below = fuseBelow();
    if (below <= 0 || g > G) return false;
    auto cells = [&](int k) { return k == 0 ? amg.nFine : amg.level[k - 1].nCoarse; };
    // the first fused grid: the start of the run of grids at or below the threshold that reaches the coarsest
    int first = G;
    while (first > 0 && cells(first - 1) <= below) --first;
    if (cells(G) > below) return false;
    first = std::max(first, G - (FUSED_MAX_LEVELS - 1));
    if (g != first) return false;

    // the cycle the kernel reproduces: weighted Jacobi, injection, no pair, no scaling, the inverse or the LU
    if (amg.saSmooth || amg.corrScaling || amg.pair.n != 0 || controlsHoldTheCycle()) return false;
    if (amg.fDiag.size() != static_cast<std::size_t>(G + 1) || amg.vAxF.size() != static_cast<std::size_t>(G + 1)) return false;
    const int nCoarsest = cells(G);
    if (amg.coarseLUn != nCoarsest) return false;
    const CoarseDirect& d = amg.coarseDirect;
    const bool lu = d.kind == CoarseSolveKind::LU || d.n != nCoarsest;
    if (!lu && (d.kind != CoarseSolveKind::Inverse || coarseInverseControlActive())) return false;
    if (amg.vB[G].size() < static_cast<std::size_t>(nCoarsest) || amg.vX[G].size() < static_cast<std::size_t>(nCoarsest)) return false;

    FusedArgs a;
    a.nLev = G - first;
    a.nPre = nPreSweeps();
    a.nPost = nPostSweeps();
    int maxN = 0;
    for (int l = 0; l <= a.nLev; ++l)
    {
        const int k = first + l;
        FusedLevel& L = a.lev[l];
        L.n = cells(k);
        if (L.n <= 0) return false;
        maxN = std::max(maxN, L.n);
        const bool top = l == 0;
        L.b = top ? const_cast<float*>(bg) : amg.vBF[k].data();
        L.x = top ? xg : amg.vXF[k].data();
        L.ax = amg.vAxF[k].data();
        L.r = amg.vRF[k].data();
        const LduF A = top ? Ag : lduF(amg.level[k - 1].coarseView(), amg.fDiag[k], amg.fUpper[k], amg.fLower[k]);
        L.diag = A.diag;
        if (!amg.csrVal.empty() && amg.csrVal[k].size() > 0)
        {
            L.row = amg.csrRow[k].data();
            L.col = amg.csrCol[k].data();
            L.val = amg.csrVal[k].data();
        }
        else
        {
            L.upper = A.upper;
            L.lower = A.lower;
            L.nei = A.nei;
            L.owner = A.owner;
            L.ownerStart = A.ownerStart;
            L.losort = A.losort;
            L.losortStart = A.losortStart;
        }
        if (k < G)
        {
            const AMGLevel& Lk = amg.level[k];
            L.nc = Lk.nCoarse;
            L.galStart = Lk.galCellStart.data();
            L.galList = Lk.galCellList.data();
            L.map = Lk.map.data();
            if (L.galStart == nullptr || L.galList == nullptr || L.map == nullptr) return false;
        }
    }
    FusedCoarse& K = a.coarse;
    K.n = nCoarsest;
    K.inverse = !lu;
    K.lu = amg.coarseLU.data();
    K.piv = amg.coarsePiv.data();
    if (!lu)
    {
        K.inv = d.f2.data();
        K.fc = d.fc.data();
        K.perm = d.perm.data();
        K.unsafe = d.unsafe.data();
    }
    K.rcD = amg.vB[G].data();
    K.xcD = amg.vX[G].data();
    const int threads = std::min(1024, std::max(128, (maxN + 31) & ~31));
    amgFusedVcycleLaunch(a, threads);
    lastFirst = first;
    return true;
}

} // namespace brae
