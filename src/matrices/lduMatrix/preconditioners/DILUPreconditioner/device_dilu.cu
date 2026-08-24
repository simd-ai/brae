// DILU on the device by level scheduling. See device_dilu.cuh for why exactness is the requirement.
#include "device_dilu.cuh"
#include <cstdlib>
#include <cstdio>
#include "device_blas.cuh"
#include "pcuda_compat.cuh"
#include <algorithm>

namespace brae {

namespace {

constexpr int TPB_D = 128;
inline int nBlk(int n) { return (n + TPB_D - 1) / TPB_D; }

// One level of calcReciprocalD, as a GATHER:
//     d[c] = diag[c] - sum_{f : nei[f] == c} upper[f]*lower[f]/d[owner[f]]
// The faces are exactly losort[losortStart[c] .. losortStart[c+1]), in increasing face index -- the order
// OF's sequential loop meets them in, so the summation order matches.
__device__ void diluDiagLevelK(
    const label* __restrict__ cells, int n,
    const label* __restrict__ owner, const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ lower,
    const scalar* __restrict__ diag, scalar* __restrict__ d)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label c = cells[i];
    scalar acc = diag[c];
    for (label k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const label f = losort[k];
        acc -= upper[f]*lower[f]/d[owner[f]];
    }
    d[c] = acc;
}

// ---------------------------------------------------------------------------------------------------
// THE SINGLE-BLOCK WALKS (item 70). A DILU apply was a launch per level per half-sweep -- 537 of them on
// validation/sbMatched -- and each carried a few microseconds of work behind a launch that costs about
// as much. Measured on that matrix (268 levels, widest 800, mean 418) with a benchmark on the real
// system: 2213 us per apply as launches, 1460 as one block per half-sweep. On rhoBox (79 levels, widest
// 20): 651 -> 156 us. A cooperative grid-barrier walk was ALSO measured and is not the answer: 268 grid
// barriers cost what 268 launches cost (1111 us against 1106), which is why this takes the same route
// the Gauss-Seidel walk took in item 60.
//
// Correct for the same reason that one does: cells at one level share no face, so no thread reads what
// another writes inside a level, and __syncthreads orders the levels exactly as the separate launches
// did. The per-cell arithmetic is the same statement in the same order, so the result is bit-identical;
// only the thread-to-cell assignment changes. BRAE_DILU_PER_LEVEL=1 forces the launches back, which is
// the identity gate's other arm.
//
// AND THE SWITCH IS ON THE MEAN LEVEL, not the widest, because the benchmark above lied about the case
// that matters. It timed the launches on a STREAM; in production the apply is captured in the BiCGStab
// conditional graph, where a launch is a graph node and costs a fraction of a stream launch. Measured
// end to end, warm (20-minus-10 iterations, two repeats):
//     sbMatched      112,000 cells, 268 levels, mean 418   single-block 307 ms/it, per-level 291  LOSS
//     pitzDailyTurb   12,225 cells, 261 levels, mean  47   single-block  17.4,     per-level 18.8  GAIN
// One block is one SM: it wins while a level's work fits comfortably there and loses when the level is
// wide enough that the other 47 SMs were doing useful work. The threshold is the mean width, and it is a
// PERFORMANCE switch only: both walks compute the same bits, which is what the gate holds.
//
// RE-MEASURED 2026-09-12 (FP-2), after the solver loops went into graphs and the readbacks into the
// mailbox, the four phases in ms per outer iteration with DILU on U, e, k and epsilon, both walks forced:
//     aerofoilNACA0012  16,000 cells, 121 levels, mean  132, widest  200   per-level 14.1  single  10.4  GAIN
//     squareBend x0.7   38,416 cells, 187 levels, mean  205, widest  392   per-level 36.0  single  22.0  GAIN
//     squareBend x1    112,000 cells, 268 levels, mean  417, widest  800   per-level 57.5  single  47.9  GAIN
//     squareBend x1.4  307,328 cells, 376 levels, mean  817, widest 1568   per-level 121.5 single 135.4  LOSS
//     squareBend x2    896,000 cells, 538 levels, mean 1665, widest 3200   per-level 257.4 single 396.1  LOSS
// The 2026-09-08 table above (sbMatched a LOSS at mean 418) no longer holds on the same mesh size: what
// the per-level walk pays per launch grew relative to what one block pays per level once every other
// gap in the loop was removed. The crossover sits between 417 and 817; the rule is 512.
constexpr int TPB_SINGLE_D = 1024;
constexpr int DILU_SINGLE_MEAN_MAX = 512;   // FP-2 measurement, see diluSingleBlock

#ifndef BRAE_ACPP
// __launch_bounds__ on a PCUDA __global__ kernel doesn't parse (the macro that expands __global__ into
// SSCP annotations doesn't compose with it), and PCUDA has no equivalent hint. No correctness loss: the
// per-level walk below is the fallback these kernels exist to beat, not a degraded path.
__global__ void __launch_bounds__(1024) diluDiagSingleK(
    const label* __restrict__ off, int nLevels, const label* __restrict__ cells,
    const label* __restrict__ owner, const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ lower,
    const scalar* __restrict__ diag, scalar* __restrict__ d)
{
    for (int L = 0; L < nLevels; ++L)
    {
        const label lo = off[L], hi = off[L+1];
        for (label i = lo + (label)threadIdx.x; i < hi; i += (label)blockDim.x)
        {
            const label c = cells[i];
            scalar acc = diag[c];
            for (label k = losortStart[c]; k < losortStart[c+1]; ++k)
            {
                const label f = losort[k];
                acc -= upper[f]*lower[f]/d[owner[f]];
            }
            d[c] = acc;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(1024) diluFwdSingleK(
    const label* __restrict__ off, int nLevels, const label* __restrict__ cells,
    const label* __restrict__ owner, const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ lower, const scalar* __restrict__ rD, scalar* __restrict__ w)
{
    for (int L = 0; L < nLevels; ++L)
    {
        const label lo = off[L], hi = off[L+1];
        for (label i = lo + (label)threadIdx.x; i < hi; i += (label)blockDim.x)
        {
            const label c = cells[i];
            const scalar rd = rD[c];
            scalar x = w[c];
            for (label k = losortStart[c]; k < losortStart[c+1]; ++k)
            {
                const label f = losort[k];
                x -= rd*lower[f]*w[owner[f]];
            }
            w[c] = x;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(1024) diluBwdSingleK(
    const label* __restrict__ off, int nLevels, const label* __restrict__ cells,
    const label* __restrict__ nei, const label* __restrict__ ownerStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ rD, scalar* __restrict__ w)
{
    for (int L = 0; L < nLevels; ++L)
    {
        const label lo = off[L], hi = off[L+1];
        for (label i = lo + (label)threadIdx.x; i < hi; i += (label)blockDim.x)
        {
            const label c = cells[i];
            const scalar rd = rD[c];
            scalar x = w[c];
            for (label f = ownerStart[c+1] - 1; f >= ownerStart[c]; --f) x -= rd*upper[f]*w[nei[f]];
            w[c] = x;
        }
        __syncthreads();
    }
}
#endif // !BRAE_ACPP

bool diluSingleBlock(const DeviceDilu& d)
{
#ifdef BRAE_ACPP
    (void)d;
    return false;   // the single-block kernels above are excluded from the ACPP build; always per-level
#else
    static const bool forceLevels = std::getenv("BRAE_DILU_PER_LEVEL") != nullptr;
    // BRAE_DILU_SINGLE=1 takes the single-block walk whatever the mean width says. It exists for the
    // identity gate, which has to compare the two walks on a mesh the switch would send per-level --
    // sbMatched is the one whose applies dominate, so it is the one identity matters most on.
    static const bool forceSingle = std::getenv("BRAE_DILU_SINGLE") != nullptr;
    const int levels = d.levels();
    // FP-2 (bench/rhoSimpleFoam/FASTPATH.md, 2026-09-12): the mean-width rule was 128, set from two
    // points (pitzDailyTurb mean 47 wins, sbMatched mean 418 loses) measured before the solver loops
    // went into graphs and the readbacks into the mailbox. Re-measured end to end, the single block now
    // wins at every width tried -- aerofoilNACA0012 (121 levels, mean 132): turbulence 5.8 -> 3.3 ms/it,
    // energy 3.2 -> 2.1; squareBend at 38k with DILU on every field (187 levels, mean 205): the four
    // phases 36 -> 22 ms/it; at 112k (268 levels, mean 417, widest 800): 57.5 -> 47.9 -- so the rule
    // is 512 and the `fits` guard is gone: the single-block kernels stride a level in blockDim
    // chunks, so a level wider than the block is a speed question, not a correctness one, and the
    // mean-width rule already answers it. BRAE_DILU_PER_LEVEL=1 restores the launches.
    const bool single = levels > 0 && !forceLevels
                     && (forceSingle || (d.nCells / levels) <= DILU_SINGLE_MEAN_MAX);
    static bool announced = false;
    if (!announced && levels > 0)
    {
        announced = true;
        std::printf("  DILU: %s walk (%d levels, widest %d, mean %d); BRAE_DILU_PER_LEVEL=1 / "
                    "BRAE_DILU_SINGLE=1 force either\n",
                    single ? "single-block" : "per-level", levels, d.maxLevelWidth, d.nCells / levels);
    }
    return single;
#endif // !BRAE_ACPP
}

__device__ void reciprocalK(scalar* __restrict__ d, scalar* __restrict__ rD, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) rD[i] = scalar(1)/d[i];
}

__device__ void diluInitK(const scalar* __restrict__ rD, const scalar* __restrict__ r,
                          scalar* __restrict__ w, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) w[i] = rD[i]*r[i];
}

// Forward sweep, one level, as a gather over the faces where this cell is the UPPER cell:
//     w[c] -= rD[c] * sum_{f : nei[f] == c} lower[f]*w[owner[f]]
__device__ void diluFwdLevelK(
    const label* __restrict__ cells, int n,
    const label* __restrict__ owner, const label* __restrict__ losort, const label* __restrict__ losortStart,
    const scalar* __restrict__ lower, const scalar* __restrict__ rD, scalar* __restrict__ w)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label c = cells[i];
    // SUBTRACT PER FACE, and keep rD inside each term. OF's loop is
    //     wA[u] -= rD[u]*lower[f]*wA[l]
    // once per face; hoisting rD out of a summed accumulator is the same number in exact arithmetic and
    // a different one in floating point -- it was off by exactly one ULP (5.55e-17) until this matched.
    const scalar rd = rD[c];
    scalar x = w[c];
    for (label k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const label f = losort[k];
        x -= rd*lower[f]*w[owner[f]];
    }
    w[c] = x;
}

// Backward sweep, one level, as a gather over the faces where this cell is the LOWER cell:
//     w[c] -= rD[c] * sum_{f : owner[f] == c} upper[f]*w[nei[f]]
// OF walks the faces in DECREASING index, so the gather runs the owner's face run in reverse to keep the
// same summation order.
__device__ void diluBwdLevelK(
    const label* __restrict__ cells, int n,
    const label* __restrict__ nei, const label* __restrict__ ownerStart,
    const scalar* __restrict__ upper, const scalar* __restrict__ rD, scalar* __restrict__ w)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label c = cells[i];
    const scalar rd = rD[c];
    scalar x = w[c];
    for (label f = ownerStart[c+1] - 1; f >= ownerStart[c]; --f) x -= rd*upper[f]*w[nei[f]];
    w[c] = x;
}

// Group cells by DAG depth and flatten into (cells, offsets). `deps` gives, for each cell, the
// neighbours it must follow.
void schedule(const std::vector<std::vector<label>>& deps, label nCells,
              DeviceBuffer<label>& out, std::vector<int>& off)
{
    std::vector<int> level((std::size_t)nCells, 0);
    int maxLevel = 0;
    // deps[c] holds only cells that come EARLIER in the sweep, so a single pass in sweep order is enough
    // -- every dependency's level is already final when it is read.
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

DeviceDilu buildDeviceDilu(const std::vector<label>& owner, const std::vector<label>& nei, label nCells)
{
    DeviceDilu d;
    d.nCells = (int)nCells;
    // The INTERNAL face count is nei.size(): PrimitiveMesh::owner() spans every face, boundary included,
    // while neighbour() stops at the internal ones. Taking owner.size() here walked off the end of nei.
    d.nFaces = (int)nei.size();

    // forward: cell `nei[f]` follows `owner[f]`   (owner < nei, so this is a DAG in increasing cell order)
    std::vector<std::vector<label>> fwd((std::size_t)nCells), bwd((std::size_t)nCells);
    for (std::size_t f = 0; f < nei.size(); ++f)
    {
        fwd[(std::size_t)nei[f]].push_back(owner[f]);
        bwd[(std::size_t)owner[f]].push_back(nei[f]);
    }
    schedule(fwd, nCells, d.fwdCells, d.fwdOff);
    // backward runs in DECREASING cell order, so reverse the cell numbering, schedule, and map back.
    {
        std::vector<std::vector<label>> rev((std::size_t)nCells);
        for (label c = 0; c < nCells; ++c)
            for (const label u : bwd[(std::size_t)c])
                rev[(std::size_t)(nCells - 1 - c)].push_back(nCells - 1 - u);
        DeviceBuffer<label> tmp;
        schedule(rev, nCells, tmp, d.bwdOff);
        std::vector<label> h; tmp.copyTo(h);
        for (label& c : h) c = nCells - 1 - c;
        d.bwdCells.copyFrom(h);
    }
    d.rD.resize((int)nCells);
    d.work.resize((int)nCells);
    // the offsets on the device and the widest level: they pick the walk (item 70)
    d.fwdOffD.copyFrom(std::vector<label>(d.fwdOff.begin(), d.fwdOff.end()));
    d.bwdOffD.copyFrom(std::vector<label>(d.bwdOff.begin(), d.bwdOff.end()));
    for (const std::vector<int>* off : {&d.fwdOff, &d.bwdOff})
        for (std::size_t L = 0; L + 1 < off->size(); ++L)
            d.maxLevelWidth = std::max(d.maxLevelWidth, (*off)[L+1] - (*off)[L]);
    d.valid = true;
    return d;
}

void diluUpdate(const DeviceLduView& A, DeviceDilu& d)
{
    if (!d.valid) return;
    const label *owner = A.owner, *losort = A.losort, *losortStart = A.losortStart;
    const scalar *upper = A.upper, *lower = A.lower, *diag = A.diag;
    scalar* workd = d.work.data();
#ifndef BRAE_ACPP
    if (diluSingleBlock(d))                              // announces which walk, once per process
    {
        diluDiagSingleK<<<1, TPB_SINGLE_D>>>(d.fwdOffD.data(), d.levels(), d.fwdCells.data(),
                                             A.owner, A.losort, A.losortStart,
                                             A.upper, A.lower, A.diag, d.work.data());
        {
            scalar* rDd = d.rD.data();
            const int nC = d.nCells;
            pcudaParallelFor(nBlk(nC), TPB_D, [=] __device__ () { reciprocalK(workd, rDd, nC); });
        }
        cudaCheck(cudaGetLastError(), "dilu update");
        return;
    }
#endif // !BRAE_ACPP
    for (int L = 0; L + 1 < (int)d.fwdOff.size(); ++L)
    {
        const int b = d.fwdOff[(std::size_t)L], e = d.fwdOff[(std::size_t)L+1];
        if (e <= b) continue;
        const label* cells = d.fwdCells.data() + b;
        const int n = e - b;
        pcudaParallelFor(nBlk(n), TPB_D, [=] __device__ () {
            diluDiagLevelK(cells, n, owner, losort, losortStart, upper, lower, diag, workd);
        });
    }
    {
        scalar* rDd = d.rD.data();
        const int nC = d.nCells;
        pcudaParallelFor(nBlk(nC), TPB_D, [=] __device__ () { reciprocalK(workd, rDd, nC); });
    }
    cudaCheck(cudaGetLastError(), "dilu update");
}

void diluApply(const DeviceLduView& A, const DeviceDilu& d, const DeviceBuffer<scalar>& r,
               DeviceBuffer<scalar>& w)
{
    w.resize(d.nCells);
    const scalar* rDd = d.rD.data();
    scalar* wd = w.data();
    {
        const scalar* rd = r.data();
        const int nC = d.nCells;
        pcudaParallelFor(nBlk(nC), TPB_D, [=] __device__ () { diluInitK(rDd, rd, wd, nC); });
    }
#ifndef BRAE_ACPP
    if (diluSingleBlock(d))
    {
        diluFwdSingleK<<<1, TPB_SINGLE_D>>>(d.fwdOffD.data(), d.levels(), d.fwdCells.data(),
                                            A.owner, A.losort, A.losortStart,
                                            A.lower, d.rD.data(), w.data());
        diluBwdSingleK<<<1, TPB_SINGLE_D>>>(d.bwdOffD.data(), (int)d.bwdOff.size() - 1, d.bwdCells.data(),
                                            A.nei, A.ownerStart,
                                            A.upper, d.rD.data(), w.data());
        cudaCheck(cudaGetLastError(), "dilu apply");
        return;
    }
#endif // !BRAE_ACPP
    {
        const label *owner = A.owner, *losort = A.losort, *losortStart = A.losortStart;
        const scalar* lower = A.lower;
        for (int L = 0; L + 1 < (int)d.fwdOff.size(); ++L)
        {
            const int b = d.fwdOff[(std::size_t)L], e = d.fwdOff[(std::size_t)L+1];
            if (e <= b) continue;
            const label* cells = d.fwdCells.data() + b;
            const int n = e - b;
            pcudaParallelFor(nBlk(n), TPB_D, [=] __device__ () {
                diluFwdLevelK(cells, n, owner, losort, losortStart, lower, rDd, wd);
            });
        }
    }
    {
        const label *nei = A.nei, *ownerStart = A.ownerStart;
        const scalar* upper = A.upper;
        for (int L = 0; L + 1 < (int)d.bwdOff.size(); ++L)
        {
            const int b = d.bwdOff[(std::size_t)L], e = d.bwdOff[(std::size_t)L+1];
            if (e <= b) continue;
            const label* cells = d.bwdCells.data() + b;
            const int n = e - b;
            pcudaParallelFor(nBlk(n), TPB_D, [=] __device__ () {
                diluBwdLevelK(cells, n, nei, ownerStart, upper, rDd, wd);
            });
        }
    }
    cudaCheck(cudaGetLastError(), "dilu apply");
}

}   // namespace brae
