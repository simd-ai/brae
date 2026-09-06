// symGaussSeidel by level scheduling. See device_sym_gauss_seidel.cuh for why exactness is the
// requirement and not a preference.
#include "device_sym_gauss_seidel.cuh"
#include "device_blas.cuh"
#include <algorithm>
#include <cstdlib>
#include <map>

namespace brae
{

namespace
{

constexpr int TPB_G = 128;
inline int nBlkG(int n) { return (n + TPB_G - 1) / TPB_G; }

// The same update, factored so the two walks cannot drift apart.
__device__ __forceinline__ void gsCellUpdate(
    label c,
    const label* __restrict__ owner, const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ lower,
    const scalar* __restrict__ diag, const scalar* __restrict__ b,
    scalar* __restrict__ psi)
{
    scalar psii = b[c];
    for (label k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const label f = losort[k];
        psii -= lower[f]*psi[owner[f]];
    }
    for (label f = ownerStart[c]; f < ownerStart[c+1]; ++f) psii -= upper[f]*psi[nei[f]];
    psi[c] = psii/diag[c];
}

// One half-sweep in ONE launch: a single block walks every level, its threads striding over the
// level's cells, with __syncthreads() as the level barrier. Correct because cells at one level share
// no face (so no thread reads what another writes inside a level) and the barrier orders the levels
// exactly as the per-level launches did. The thread-to-cell assignment differs from the per-level
// kernel; the per-cell arithmetic does not, so the numbers are bit-identical.
__global__ void gsSingleBlockK(
    const label* __restrict__ off, int nLevels,
    const label* __restrict__ cells,
    const label* __restrict__ owner, const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ lower,
    const scalar* __restrict__ diag, const scalar* __restrict__ b,
    scalar* __restrict__ psi)
{
    for (int L = 0; L < nLevels; ++L)
    {
        const label lo = off[L], hi = off[L+1];
        for (label i = lo + (label)threadIdx.x; i < hi; i += (label)blockDim.x)
            gsCellUpdate(cells[i], owner, nei, ownerStart, losort, losortStart, upper, lower, diag, b, psi);
        __syncthreads();
    }
}
constexpr int TPB_SINGLE = 1024;   // a level up to this wide is walked by one block in one pass

// One level of a symGaussSeidel half-sweep, as a gather:
//
//     psi[c] = ( b[c]
//              - sum_{f : nei[f] == c}   lower[f]*psi[owner[f]]     <- OF's bPrime, gathered
//              - sum_{f : owner[f] == c} upper[f]*psi[nei[f]]  ) / diag[c]
//
// The first sum runs losort[losortStart[c] .. losortStart[c+1]), which is increasing face index and so
// the order OF's ascending owner walk subtracts those terms in; the second runs the cell's own face run,
// also increasing, which is OF's inner loop verbatim. Same terms, same order, same rounding.
//
// The SAME kernel serves both halves: what makes one forward and the other reverse is only which cells
// have already been written when a level runs, and that is decided by the level ordering handed in.
__global__ void gsLevelK(
    const label* __restrict__ cells, int n,
    const label* __restrict__ owner, const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ lower,
    const scalar* __restrict__ diag, const scalar* __restrict__ b,
    scalar* __restrict__ psi)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    gsCellUpdate(cells[i], owner, nei, ownerStart, losort, losortStart, upper, lower, diag, b, psi);
}

// The fused kernels: thread i of a level covers (component k = i / n, cell j = i - k*n), so consecutive
// threads still take consecutive cells of one component (the same coalescing as the scalar kernels) and
// every (cell, component) pair is updated by exactly one thread with gsCellUpdate -- the same arithmetic
// on that component's own diag/b/psi and the shared upper/lower.
__global__ void gsLevelFusedK(
    const label* __restrict__ cells, int n,
    const label* __restrict__ owner, const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ lower,
    GSFusedOperands ops)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n*ops.nComp) return;
    const int k = i / n;
    const int j = i - k*n;
    if (ops.active && !ops.active[k]) return;
    gsCellUpdate(cells[j], owner, nei, ownerStart, losort, losortStart, upper, lower, ops.diag[k], ops.b[k], ops.psi[k]);
}

// One block PER COMPONENT, in one launch: block k walks component k's levels exactly as gsSingleBlockK
// walks a scalar's -- same threads-over-cells striding, same block-local barrier -- so the per-level work
// on one SM is unchanged while the components run concurrently on separate SMs and the launches are
// shared. Measured on the composed 209,825-cell flat plate (929 levels, widest 385): the first fused
// version put both components' cells through ONE block, and at that width a level is throughput-bound
// on its SM, so the outer iteration went 84 -> 92 ms/it; on T3A (465 levels, widest 80, latency-bound)
// the same version went 53 -> 40. A block per component keeps the second gain without the first loss.
__global__ void gsSingleBlockFusedK(
    const label* __restrict__ off, int nLevels,
    const label* __restrict__ cells,
    const label* __restrict__ owner, const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ lower,
    GSFusedOperands ops)
{
    const int k = blockIdx.x;
    if (k >= ops.nComp) return;
    if (ops.active && !ops.active[k]) return;
    const scalar* __restrict__ diag = ops.diag[k];
    const scalar* __restrict__ b    = ops.b[k];
    scalar* __restrict__ psi        = ops.psi[k];
    for (int L = 0; L < nLevels; ++L)
    {
        const label lo = off[L], hi = off[L+1];
        for (label i = lo + (label)threadIdx.x; i < hi; i += (label)blockDim.x)
            gsCellUpdate(cells[i], owner, nei, ownerStart, losort, losortStart, upper, lower, diag, b, psi);
        __syncthreads();
    }
}

// Group cells by DAG depth and flatten into (cells, offsets). `deps[c]` holds the cells c must follow,
// and every one of them comes earlier in the sweep, so one pass in sweep order fixes every level.
void schedule(const std::vector<std::vector<label>>& deps, label nCells,
              DeviceBuffer<label>& out, std::vector<int>& off)
{
    std::vector<int> level((std::size_t)nCells, 0);
    int maxLevel = 0;
    for (label c = 0; c < nCells; ++c)
    {
        int L = 0;
        for (const label d : deps[(std::size_t)c]) L = std::max(L, level[(std::size_t)d] + 1);
        level[(std::size_t)c] = L;
        maxLevel = std::max(maxLevel, L);
    }
    std::vector<int> count((std::size_t)maxLevel + 1, 0);
    for (const int L : level) ++count[(std::size_t)L];
    off.assign((std::size_t)maxLevel + 2, 0);
    for (int L = 0; L <= maxLevel; ++L) off[(std::size_t)L+1] = off[(std::size_t)L] + count[(std::size_t)L];
    std::vector<label> flat((std::size_t)nCells);
    std::vector<int> cursor(off.begin(), off.end() - 1);
    for (label c = 0; c < nCells; ++c) flat[(std::size_t)cursor[(std::size_t)level[(std::size_t)c]]++] = c;
    out.copyFrom(flat);
}

}   // namespace

DeviceGaussSeidelLevels buildDeviceGaussSeidelLevels(const std::vector<label>& owner,
                                                     const std::vector<label>& nei,
                                                     label nCells)
{
    DeviceGaussSeidelLevels lv;
    lv.nCells = (int)nCells;

    // Forward: cell nei[f] gathers owner[f], which is a lower-numbered cell, so this is a DAG in
    // increasing cell order. Reverse: cell owner[f] gathers nei[f], the same DAG under the reversed
    // numbering. nei.size() is the INTERNAL face count -- owner() spans the boundary faces too.
    std::vector<std::vector<label>> fwd((std::size_t)nCells), bwd((std::size_t)nCells);
    for (std::size_t f = 0; f < nei.size(); ++f)
    {
        fwd[(std::size_t)nei[f]].push_back(owner[f]);
        bwd[(std::size_t)owner[f]].push_back(nei[f]);
    }
    schedule(fwd, nCells, lv.fwdCells, lv.fwdOff);
    lv.fwdOffD.copyFrom(std::vector<label>(lv.fwdOff.begin(), lv.fwdOff.end()));
    {
        std::vector<std::vector<label>> rev((std::size_t)nCells);
        for (label c = 0; c < nCells; ++c)
            for (const label u : bwd[(std::size_t)c])
                rev[(std::size_t)(nCells - 1 - c)].push_back(nCells - 1 - u);
        DeviceBuffer<label> tmp;
        schedule(rev, nCells, tmp, lv.bwdOff);
        std::vector<label> h; tmp.copyTo(h);
        for (label& c : h) c = nCells - 1 - c;
        lv.bwdCells.copyFrom(h);
        lv.bwdOffD.copyFrom(std::vector<label>(lv.bwdOff.begin(), lv.bwdOff.end()));
    }
    for (const std::vector<int>* off : {&lv.fwdOff, &lv.bwdOff})
        for (std::size_t L = 0; L + 1 < off->size(); ++L)
            lv.maxLevelWidth = std::max(lv.maxLevelWidth, (*off)[L+1] - (*off)[L]);
    lv.valid = true;
    return lv;
}

const DeviceGaussSeidelLevels& gsLevelsFor(const DeviceLduView& A)
{
    // Keyed on the addressing, not on the coefficients: the levels depend only on owner/neighbour, so a
    // matrix whose values are reassembled every outer iteration reuses them. Leaked deliberately, as the
    // colouring cache is, so no static destructor runs after the CUDA context is torn down.
    static auto& cache = *new std::map<const label*, DeviceGaussSeidelLevels>();
    auto it = cache.find(A.owner);
    if (it == cache.end())
    {
        const int nF = A.nInternalFaces;
        std::vector<label> ownerH((std::size_t)nF), neiH((std::size_t)nF);
        cudaMemcpy(ownerH.data(), A.owner, (std::size_t)nF*sizeof(label), cudaMemcpyDeviceToHost);
        cudaMemcpy(neiH.data(),   A.nei,   (std::size_t)nF*sizeof(label), cudaMemcpyDeviceToHost);
        it = cache.emplace(A.owner, buildDeviceGaussSeidelLevels(ownerH, neiH, A.nCells)).first;
    }
    return it->second;
}

void deviceSymGaussSeidelSweepExact(const DeviceLduView& A,
                                    const DeviceBuffer<scalar>& b,
                                    DeviceBuffer<scalar>& psi,
                                    const DeviceGaussSeidelLevels& lv,
                                    bool symmetric)
{
    // BRAE_GS_PER_LEVEL=1 forces the per-level launches on a mesh that would take the single-block
    // walk. Measurement only -- tests/gs_ladder times both -- the numbers are identical either way.
    static const bool forcePerLevel = std::getenv("BRAE_GS_PER_LEVEL") != nullptr;
    const bool singleBlock = lv.maxLevelWidth <= TPB_SINGLE && !forcePerLevel;
    auto half = [&](const DeviceBuffer<label>& cells, const std::vector<int>& off,
                    const DeviceBuffer<label>& offD)
    {
        const int nLevels = (int)off.size() - 1;
        if (singleBlock)
        {
            gsSingleBlockK<<<1, TPB_SINGLE>>>(offD.data(), nLevels, cells.data(),
                                               A.owner, A.nei, A.ownerStart,
                                               A.losort, A.losortStart,
                                               A.upper, A.lower, A.diag, b.data(), psi.data());
            return;
        }
        for (int L = 0; L < nLevels; ++L)
        {
            const int lo = off[(std::size_t)L], hi = off[(std::size_t)L+1];
            if (hi <= lo) continue;
            gsLevelK<<<nBlkG(hi-lo), TPB_G>>>(cells.data() + lo, hi - lo,
                                              A.owner, A.nei, A.ownerStart,
                                              A.losort, A.losortStart,
                                              A.upper, A.lower, A.diag, b.data(), psi.data());
        }
    };
    half(lv.fwdCells, lv.fwdOff, lv.fwdOffD);
    // GaussSeidelSmoother.C stops here: its sweep loop is the ascending walk and nothing else.
    if (symmetric) half(lv.bwdCells, lv.bwdOff, lv.bwdOffD);
    cudaCheck(cudaGetLastError(), "GaussSeidel sweep");
}

void deviceSymGaussSeidelSweepExactFused(const DeviceLduView& A,
                                         const GSFusedOperands& ops,
                                         const DeviceGaussSeidelLevels& lv,
                                         bool symmetric)
{
    static const bool forcePerLevel = std::getenv("BRAE_GS_PER_LEVEL") != nullptr;
    const bool singleBlock = lv.maxLevelWidth <= TPB_SINGLE && !forcePerLevel;
    auto half = [&](const DeviceBuffer<label>& cells, const std::vector<int>& off,
                    const DeviceBuffer<label>& offD)
    {
        const int nLevels = (int)off.size() - 1;
        if (singleBlock)
        {
            gsSingleBlockFusedK<<<ops.nComp, TPB_SINGLE>>>(offD.data(), nLevels, cells.data(),
                                                    A.owner, A.nei, A.ownerStart,
                                                    A.losort, A.losortStart,
                                                    A.upper, A.lower, ops);
            return;
        }
        for (int L = 0; L < nLevels; ++L)
        {
            const int lo = off[(std::size_t)L], hi = off[(std::size_t)L+1];
            if (hi <= lo) continue;
            gsLevelFusedK<<<nBlkG((hi-lo)*ops.nComp), TPB_G>>>(cells.data() + lo, hi - lo,
                                                               A.owner, A.nei, A.ownerStart,
                                                               A.losort, A.losortStart,
                                                               A.upper, A.lower, ops);
        }
    };
    half(lv.fwdCells, lv.fwdOff, lv.fwdOffD);
    if (symmetric) half(lv.bwdCells, lv.bwdOff, lv.bwdOffD);
    cudaCheck(cudaGetLastError(), "GaussSeidel fused sweep");
}

}   // namespace brae
