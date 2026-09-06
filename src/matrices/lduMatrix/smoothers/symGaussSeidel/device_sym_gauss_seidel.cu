// symGaussSeidel by level scheduling. See device_sym_gauss_seidel.cuh for why exactness is the
// requirement and not a preference.
#include "device_sym_gauss_seidel.cuh"
#include "device_blas.cuh"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>

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

// ---- the level-ordered gather (item 60b) -----------------------------------------------------------
// The same update as gsCellUpdate -- the same terms subtracted in the same order, then the divide -- read
// through the CSR in walk order. One loop where gsCellUpdate had two, because the CSR already lists the
// lower terms first: the sequence of operations is unchanged.
__device__ __forceinline__ void gsCellUpdateLv(
    label i,
    const label* __restrict__ cells,
    const label* __restrict__ entStart, const label* __restrict__ entNbr,
    const scalar* __restrict__ coef,
    const scalar* __restrict__ lvDiag, const scalar* __restrict__ lvB,
    scalar* __restrict__ psi)
{
    const label c  = cells[i];
    const label e0 = entStart[i], e1 = entStart[i+1];
    scalar psii = lvB[i];
    for (label e = e0; e < e1; ++e) psii -= coef[e]*psi[entNbr[e]];
    psi[c] = psii/lvDiag[i];
}
__global__ void gsLevelLvK(
    int lo, int n,
    const label* __restrict__ cells,
    const label* __restrict__ entStart, const label* __restrict__ entNbr,
    const scalar* __restrict__ coef,
    const scalar* __restrict__ lvDiag, const scalar* __restrict__ lvB,
    scalar* __restrict__ psi)
{
    const int g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= n) return;
    gsCellUpdateLv((label)(lo + g), cells, entStart, entNbr, coef, lvDiag, lvB, psi);
}
// The single-block walk, SOFTWARE-PIPELINED. In the single-block walk a level is at most one cell per
// thread (no level outgrows the block), and everything a cell needs except psi -- its offsets, diagonal,
// source, index and its first GS_PF entries -- is constant for the whole sweep. So a thread loads the NEXT
// level's operands before the barrier that ends this one, and after the barrier only the psi gather's
// latency is exposed. The sequence of arithmetic per cell is gsCellUpdateLv's exactly (the prefetched
// entries are the first GS_PF in order, the rest follow from memory in order); only WHEN the constant
// operands are read changes, so the result is bit-identical.
constexpr int GS_PF = 4;      // entries prefetched per cell; kept small so two sets fit the 1024-thread register budget
struct GSLvCell
{
    bool has = false;
    label c = 0, e0 = 0, e1 = 0;
    scalar b = 0, d = 1;
    scalar pc[GS_PF];
    label  pn[GS_PF];
};
__device__ __forceinline__ void gsLvLoad(
    label lo, label hi,
    const label* __restrict__ cells,
    const label* __restrict__ entStart, const label* __restrict__ entNbr,
    const scalar* __restrict__ coef,
    const scalar* __restrict__ lvDiag, const scalar* __restrict__ lvB,
    GSLvCell& r)
{
    const label i = lo + (label)threadIdx.x;
    r.has = i < hi;
    if (!r.has) return;
    r.c  = cells[i];
    r.e0 = entStart[i];
    r.e1 = entStart[i+1];
    r.b  = lvB[i];
    r.d  = lvDiag[i];
    #pragma unroll
    for (int j = 0; j < GS_PF; ++j)
    {
        const label e = r.e0 + j;
        if (e < r.e1)
        {
            r.pc[j] = coef[e];
            r.pn[j] = entNbr[e];
        }
    }
}
__device__ __forceinline__ void gsLvApply(
    const GSLvCell& r,
    const label* __restrict__ entNbr,
    const scalar* __restrict__ coef,
    scalar* __restrict__ psi)
{
    if (!r.has) return;
    scalar psii = r.b;
    const label n = r.e1 - r.e0;
    #pragma unroll
    for (int j = 0; j < GS_PF; ++j)
        if (j < n) psii -= r.pc[j]*psi[r.pn[j]];
    for (label e = r.e0 + GS_PF; e < r.e1; ++e) psii -= coef[e]*psi[entNbr[e]];
    psi[r.c] = psii/r.d;
}
__device__ __forceinline__ void gsWalkLv(
    const label* __restrict__ off, int nLevels,
    const label* __restrict__ cells,
    const label* __restrict__ entStart, const label* __restrict__ entNbr,
    const scalar* __restrict__ coef,
    const scalar* __restrict__ lvDiag, const scalar* __restrict__ lvB,
    scalar* __restrict__ psi)
{
    GSLvCell cur, nxt;
    if (nLevels > 0) gsLvLoad(off[0], off[1], cells, entStart, entNbr, coef, lvDiag, lvB, cur);
    for (int L = 0; L < nLevels; ++L)
    {
        gsLvApply(cur, entNbr, coef, psi);
        if (L + 1 < nLevels) gsLvLoad(off[L+1], off[L+2], cells, entStart, entNbr, coef, lvDiag, lvB, nxt);
        else nxt.has = false;
        __syncthreads();
        cur = nxt;
    }
}
__global__ void __launch_bounds__(1024) gsSingleBlockLvK(
    const label* __restrict__ off, int nLevels,
    const label* __restrict__ cells,
    const label* __restrict__ entStart, const label* __restrict__ entNbr,
    const scalar* __restrict__ coef,
    const scalar* __restrict__ lvDiag, const scalar* __restrict__ lvB,
    scalar* __restrict__ psi)
{
    gsWalkLv(off, nLevels, cells, entStart, entNbr, coef, lvDiag, lvB, psi);
}
struct GSFusedLvHalf
{
    int nComp = 0;
    const scalar* lvDiag[GS_FUSED_MAX] = {};
    const scalar* lvB[GS_FUSED_MAX] = {};
    scalar*       psi[GS_FUSED_MAX] = {};
    const int*    active = nullptr;
};
__global__ void gsLevelFusedLvK(
    int lo, int n,
    const label* __restrict__ cells,
    const label* __restrict__ entStart, const label* __restrict__ entNbr,
    const scalar* __restrict__ coef,
    GSFusedLvHalf ops)
{
    const int g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= n*ops.nComp) return;
    const int k = g / n;
    const int j = g - k*n;
    if (ops.active && !ops.active[k]) return;
    gsCellUpdateLv((label)(lo + j), cells, entStart, entNbr, coef, ops.lvDiag[k], ops.lvB[k], ops.psi[k]);
}
__global__ void __launch_bounds__(1024) gsSingleBlockFusedLvK(
    const label* __restrict__ off, int nLevels,
    const label* __restrict__ cells,
    const label* __restrict__ entStart, const label* __restrict__ entNbr,
    const scalar* __restrict__ coef,
    GSFusedLvHalf ops)
{
    const int k = blockIdx.x;
    if (k >= ops.nComp) return;
    if (ops.active && !ops.active[k]) return;
    gsWalkLv(off, nLevels, cells, entStart, entNbr, coef, ops.lvDiag[k], ops.lvB[k], ops.psi[k]);
}
// The per-solve permutations into walk order.
__global__ void gsPermuteCoefK(
    int nEnt,
    const label* __restrict__ entSrc,
    const scalar* __restrict__ lower, const scalar* __restrict__ upper, int nF,
    scalar* __restrict__ coef)
{
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= nEnt) return;
    const label src = entSrc[e];
    coef[e] = (src < nF) ? lower[src] : upper[src - nF];
}
__global__ void gsPermuteCellK(
    int nC,
    const label* __restrict__ cells,
    const scalar* __restrict__ diag, const scalar* __restrict__ b,
    scalar* __restrict__ lvDiag, scalar* __restrict__ lvB)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nC) return;
    const label c = cells[i];
    lvDiag[i] = diag[c];
    lvB[i]    = b[c];
}
// The CSR for one walk order: replay gsCellUpdate's two loops per cell, in walk order.
void buildEntries(const std::vector<label>& seq,
                  const std::vector<label>& owner, const std::vector<label>& nei,
                  const std::vector<label>& losort, const std::vector<label>& losortStart,
                  const std::vector<label>& ownerStart,
                  DeviceBuffer<label>& entStart, DeviceBuffer<label>& entNbr, DeviceBuffer<label>& entSrc)
{
    const label nF = (label)nei.size();
    std::vector<label> start(seq.size() + 1, 0), nbr, src;
    nbr.reserve(2*(std::size_t)nF);
    src.reserve(2*(std::size_t)nF);
    for (std::size_t i = 0; i < seq.size(); ++i)
    {
        const label c = seq[i];
        for (label k = losortStart[(std::size_t)c]; k < losortStart[(std::size_t)c+1]; ++k)
        {
            const label f = losort[(std::size_t)k];
            nbr.push_back(owner[(std::size_t)f]);
            src.push_back(f);
        }
        for (label f = ownerStart[(std::size_t)c]; f < ownerStart[(std::size_t)c+1]; ++f)
        {
            nbr.push_back(nei[(std::size_t)f]);
            src.push_back(f + nF);
        }
        start[i+1] = (label)nbr.size();
    }
    entStart.copyFrom(start);
    entNbr.copyFrom(nbr);
    entSrc.copyFrom(src);
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
                                                     label nCells,
                                                     const std::vector<label>& losort,
                                                     const std::vector<label>& losortStart,
                                                     const std::vector<label>& ownerStart)
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
    // the level-ordered CSR, from the view's own addressing (item 60b)
    if (losort.size() == nei.size() && losortStart.size() == (std::size_t)nCells + 1 && ownerStart.size() == (std::size_t)nCells + 1)
    {
        std::vector<label> seq;
        lv.fwdCells.copyTo(seq);
        buildEntries(seq, owner, nei, losort, losortStart, ownerStart, lv.fwdEntStart, lv.fwdEntNbr, lv.fwdEntSrc);
        lv.bwdCells.copyTo(seq);
        buildEntries(seq, owner, nei, losort, losortStart, ownerStart, lv.bwdEntStart, lv.bwdEntNbr, lv.bwdEntSrc);
        lv.nEntries = 2*(int)nei.size();
    }
    lv.valid = true;
    return lv;
}

bool gsSingleBlockWalk(const DeviceGaussSeidelLevels& lv)
{
    // BRAE_GS_PER_LEVEL=1 forces the per-level launches on a mesh that would take the single-block
    // walk. Measurement only -- tests/gs_ladder times both -- the numbers are identical either way.
    static const bool forcePerLevel = std::getenv("BRAE_GS_PER_LEVEL") != nullptr;
    return lv.maxLevelWidth <= TPB_SINGLE && !forcePerLevel;
}

bool gsLevelGatherEnabled(const DeviceGaussSeidelLevels& lv)
{
    static const bool off = []()
    {
        const char* e = std::getenv("BRAE_GS_LEVEL_GATHER");
        return e && std::string(e) == "0";
    }();
    return lv.nEntries > 0 && !off;
}

void gsLevelCoefsRefresh(const DeviceLduView& A, const DeviceGaussSeidelLevels& lv, GSLevelCoefs& lc)
{
    if (!gsLevelGatherEnabled(lv)) return;      // the index gather reads none of this; do not pay for it
    const int nE = lv.nEntries, nF = A.nInternalFaces;
    lc.fwd.resize(nE);
    lc.bwd.resize(nE);
    if (nE <= 0) return;
    gsPermuteCoefK<<<nBlkG(nE), TPB_G>>>(nE, lv.fwdEntSrc.data(), A.lower, A.upper, nF, lc.fwd.data());
    gsPermuteCoefK<<<nBlkG(nE), TPB_G>>>(nE, lv.bwdEntSrc.data(), A.lower, A.upper, nF, lc.bwd.data());
}

void gsLevelCellsRefresh(const DeviceLduView& A, const DeviceBuffer<scalar>& b, const DeviceGaussSeidelLevels& lv, GSLevelCells& cc)
{
    if (!gsLevelGatherEnabled(lv)) return;
    const int nC = lv.nCells;
    for (auto* v : {&cc.fwdDiag, &cc.bwdDiag, &cc.fwdB, &cc.bwdB}) v->resize(nC);
    if (nC <= 0) return;
    gsPermuteCellK<<<nBlkG(nC), TPB_G>>>(nC, lv.fwdCells.data(), A.diag, b.data(), cc.fwdDiag.data(), cc.fwdB.data());
    gsPermuteCellK<<<nBlkG(nC), TPB_G>>>(nC, lv.bwdCells.data(), A.diag, b.data(), cc.bwdDiag.data(), cc.bwdB.data());
}

const DeviceGaussSeidelLevels& gsLevelsFor(const DeviceLduView& A)
{
    // Keyed on the addressing, not on the coefficients: the levels depend only on owner/neighbour, so a
    // matrix whose values are reassembled every outer iteration reuses them. Leaked deliberately, as the
    // colouring cache is, so no static destructor runs after the CUDA context is torn down.
    static auto& cache = *new std::map<const label*, DeviceGaussSeidelLevels>();
    auto it = cache.find(A.owner);
    // A recycled owner pointer (the device pool hands equal-sized blocks back) must not replay another
    // mesh's levels: the entry has to match the view's sizes, or it is rebuilt.
    if (it != cache.end() && (it->second.nCells != A.nCells || it->second.nEntries != 2*A.nInternalFaces))
    {
        cache.erase(it);
        it = cache.end();
    }
    if (it == cache.end())
    {
        const int nF = A.nInternalFaces, nC = A.nCells;
        std::vector<label> ownerH((std::size_t)nF), neiH((std::size_t)nF);
        cudaMemcpy(ownerH.data(), A.owner, (std::size_t)nF*sizeof(label), cudaMemcpyDeviceToHost);
        cudaMemcpy(neiH.data(),   A.nei,   (std::size_t)nF*sizeof(label), cudaMemcpyDeviceToHost);
        // the view's own losort/ownerStart, so the level-ordered CSR replays gsCellUpdate on the same arrays
        std::vector<label> losortH, losortStartH, ownerStartH;
        if (A.losort && A.losortStart && A.ownerStart)
        {
            losortH.resize((std::size_t)nF);
            losortStartH.resize((std::size_t)nC + 1);
            ownerStartH.resize((std::size_t)nC + 1);
            cudaMemcpy(losortH.data(),      A.losort,      (std::size_t)nF*sizeof(label),       cudaMemcpyDeviceToHost);
            cudaMemcpy(losortStartH.data(), A.losortStart, ((std::size_t)nC + 1)*sizeof(label), cudaMemcpyDeviceToHost);
            cudaMemcpy(ownerStartH.data(),  A.ownerStart,  ((std::size_t)nC + 1)*sizeof(label), cudaMemcpyDeviceToHost);
        }
        it = cache.emplace(A.owner, buildDeviceGaussSeidelLevels(ownerH, neiH, nC, losortH, losortStartH, ownerStartH)).first;
    }
    return it->second;
}

void deviceSymGaussSeidelSweepExact(const DeviceLduView& A,
                                    const DeviceBuffer<scalar>& b,
                                    DeviceBuffer<scalar>& psi,
                                    const DeviceGaussSeidelLevels& lv,
                                    bool symmetric,
                                    const GSLevelCoefs* lc,
                                    const GSLevelCells* cc)
{
    const bool singleBlock = gsSingleBlockWalk(lv);
    if (gsLevelGatherEnabled(lv))
    {
        // no operands handed in (tests/gs_ladder): refresh a scratch for this sweep. Per thread and
        // leaked, like the file's other caches: no destructor after the context is gone, no sharing
        // should two threads ever solve at once.
        static thread_local auto& lcS = *new GSLevelCoefs();
        static thread_local auto& ccS = *new GSLevelCells();
        if (!lc) { gsLevelCoefsRefresh(A, lv, lcS); lc = &lcS; }
        if (!cc) { gsLevelCellsRefresh(A, b, lv, ccS); cc = &ccS; }
        auto halfLv = [&](const DeviceBuffer<label>& cells, const std::vector<int>& off, const DeviceBuffer<label>& offD,
                          const DeviceBuffer<label>& entStart, const DeviceBuffer<label>& entNbr,
                          const DeviceBuffer<scalar>& coef, const DeviceBuffer<scalar>& lvDiag, const DeviceBuffer<scalar>& lvB)
        {
            const int nLevels = (int)off.size() - 1;
            if (singleBlock)
            {
                gsSingleBlockLvK<<<1, TPB_SINGLE>>>(offD.data(), nLevels, cells.data(), entStart.data(), entNbr.data(),
                                                     coef.data(), lvDiag.data(), lvB.data(), psi.data());
                return;
            }
            for (int L = 0; L < nLevels; ++L)
            {
                const int lo = off[(std::size_t)L], hi = off[(std::size_t)L+1];
                if (hi <= lo) continue;
                gsLevelLvK<<<nBlkG(hi-lo), TPB_G>>>(lo, hi - lo, cells.data(), entStart.data(), entNbr.data(),
                                                    coef.data(), lvDiag.data(), lvB.data(), psi.data());
            }
        };
        halfLv(lv.fwdCells, lv.fwdOff, lv.fwdOffD, lv.fwdEntStart, lv.fwdEntNbr, lc->fwd, cc->fwdDiag, cc->fwdB);
        if (symmetric) halfLv(lv.bwdCells, lv.bwdOff, lv.bwdOffD, lv.bwdEntStart, lv.bwdEntNbr, lc->bwd, cc->bwdDiag, cc->bwdB);
        cudaCheck(cudaGetLastError(), "GaussSeidel sweep (level gather)");
        return;
    }
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
                                         bool symmetric,
                                         const GSLevelCoefs* lc,
                                         const GSLevelCells* const* cc)
{
    const bool singleBlock = gsSingleBlockWalk(lv);
    if (gsLevelGatherEnabled(lv) && lc && cc)
    {
        auto halfLv = [&](const DeviceBuffer<label>& cells, const std::vector<int>& off, const DeviceBuffer<label>& offD,
                          const DeviceBuffer<label>& entStart, const DeviceBuffer<label>& entNbr,
                          const DeviceBuffer<scalar>& coef, bool fwd)
        {
            GSFusedLvHalf h;
            h.nComp = ops.nComp;
            h.active = ops.active;
            for (int k = 0; k < ops.nComp; ++k)
            {
                h.lvDiag[k] = fwd ? cc[k]->fwdDiag.data() : cc[k]->bwdDiag.data();
                h.lvB[k]    = fwd ? cc[k]->fwdB.data()    : cc[k]->bwdB.data();
                h.psi[k]    = ops.psi[k];
            }
            const int nLevels = (int)off.size() - 1;
            if (singleBlock)
            {
                gsSingleBlockFusedLvK<<<ops.nComp, TPB_SINGLE>>>(offD.data(), nLevels, cells.data(), entStart.data(), entNbr.data(), coef.data(), h);
                return;
            }
            for (int L = 0; L < nLevels; ++L)
            {
                const int lo = off[(std::size_t)L], hi = off[(std::size_t)L+1];
                if (hi <= lo) continue;
                gsLevelFusedLvK<<<nBlkG((hi-lo)*ops.nComp), TPB_G>>>(lo, hi - lo, cells.data(), entStart.data(), entNbr.data(), coef.data(), h);
            }
        };
        halfLv(lv.fwdCells, lv.fwdOff, lv.fwdOffD, lv.fwdEntStart, lv.fwdEntNbr, lc->fwd, true);
        if (symmetric) halfLv(lv.bwdCells, lv.bwdOff, lv.bwdOffD, lv.bwdEntStart, lv.bwdEntNbr, lc->bwd, false);
        cudaCheck(cudaGetLastError(), "GaussSeidel fused sweep (level gather)");
        return;
    }
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
