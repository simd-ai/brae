// DILU on the device by level scheduling. See device_dilu.cuh for why exactness is the requirement.
#include "device_dilu.cuh"
#include <cstdlib>
#include <cstdio>
#include "device_blas.cuh"
#include <algorithm>

namespace brae {

namespace {

constexpr int TPB_D = 128;
inline int nBlk(int n) { return (n + TPB_D - 1) / TPB_D; }

// One level of calcReciprocalD, as a GATHER:
//     d[c] = diag[c] - sum_{f : nei[f] == c} upper[f]*lower[f]/d[owner[f]]
// The faces are exactly losort[losortStart[c] .. losortStart[c+1]), in increasing face index -- the order
// OF's sequential loop meets them in, so the summation order matches.
__global__ void diluDiagLevelK(
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
// wide enough that the other 47 SMs were doing useful work. The threshold is the mean width, set at 128
// -- above pitzDailyTurb's 47 and T3A's 58, well below sbMatched's 418 -- and it is a PERFORMANCE
// switch only: both walks compute the same bits, which is what the gate holds.
constexpr int TPB_SINGLE_D = 1024;
constexpr int DILU_SINGLE_MEAN_MAX = 128;

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

bool diluSingleBlock(const DeviceDilu& d)
{
    static const bool forceLevels = std::getenv("BRAE_DILU_PER_LEVEL") != nullptr;
    // BRAE_DILU_SINGLE=1 takes the single-block walk whatever the mean width says. It exists for the
    // identity gate, which has to compare the two walks on a mesh the switch would send per-level --
    // sbMatched is the one whose applies dominate, so it is the one identity matters most on.
    static const bool forceSingle = std::getenv("BRAE_DILU_SINGLE") != nullptr;
    const int levels = d.levels();
    const bool fits  = d.maxLevelWidth > 0 && d.maxLevelWidth <= TPB_SINGLE_D && levels > 0;
    const bool single = fits && !forceLevels
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
}

__global__ void reciprocalK(scalar* __restrict__ d, scalar* __restrict__ rD, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) rD[i] = scalar(1)/d[i];
}

__global__ void diluInitK(const scalar* __restrict__ rD, const scalar* __restrict__ r,
                          scalar* __restrict__ w, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) w[i] = rD[i]*r[i];
}

// Forward sweep, one level, as a gather over the faces where this cell is the UPPER cell:
//     w[c] -= rD[c] * sum_{f : nei[f] == c} lower[f]*w[owner[f]]
__global__ void diluFwdLevelK(
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
__global__ void diluBwdLevelK(
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
    if (diluSingleBlock(d))                              // announces which walk, once per process
    {
        diluDiagSingleK<<<1, TPB_SINGLE_D>>>(d.fwdOffD.data(), d.levels(), d.fwdCells.data(),
                                             A.owner, A.losort, A.losortStart,
                                             A.upper, A.lower, A.diag, d.work.data());
        reciprocalK<<<nBlk(d.nCells), TPB_D>>>(d.work.data(), d.rD.data(), d.nCells);
        cudaCheck(cudaGetLastError(), "dilu update");
        return;
    }
    for (int L = 0; L + 1 < (int)d.fwdOff.size(); ++L)
    {
        const int b = d.fwdOff[(std::size_t)L], e = d.fwdOff[(std::size_t)L+1];
        if (e <= b) continue;
        diluDiagLevelK<<<nBlk(e-b), TPB_D>>>(d.fwdCells.data() + b, e - b,
                                             A.owner, A.losort, A.losortStart,
                                             A.upper, A.lower, A.diag, d.work.data());
    }
    reciprocalK<<<nBlk(d.nCells), TPB_D>>>(d.work.data(), d.rD.data(), d.nCells);
    cudaCheck(cudaGetLastError(), "dilu update");
}

void diluApply(const DeviceLduView& A, const DeviceDilu& d, const DeviceBuffer<scalar>& r,
               DeviceBuffer<scalar>& w)
{
    w.resize(d.nCells);
    diluInitK<<<nBlk(d.nCells), TPB_D>>>(d.rD.data(), r.data(), w.data(), d.nCells);
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
    for (int L = 0; L + 1 < (int)d.fwdOff.size(); ++L)
    {
        const int b = d.fwdOff[(std::size_t)L], e = d.fwdOff[(std::size_t)L+1];
        if (e <= b) continue;
        diluFwdLevelK<<<nBlk(e-b), TPB_D>>>(d.fwdCells.data() + b, e - b,
                                            A.owner, A.losort, A.losortStart,
                                            A.lower, d.rD.data(), w.data());
    }
    for (int L = 0; L + 1 < (int)d.bwdOff.size(); ++L)
    {
        const int b = d.bwdOff[(std::size_t)L], e = d.bwdOff[(std::size_t)L+1];
        if (e <= b) continue;
        diluBwdLevelK<<<nBlk(e-b), TPB_D>>>(d.bwdCells.data() + b, e - b,
                                            A.nei, A.ownerStart,
                                            A.upper, d.rD.data(), w.data());
    }
    cudaCheck(cudaGetLastError(), "dilu apply");
}

}   // namespace brae
