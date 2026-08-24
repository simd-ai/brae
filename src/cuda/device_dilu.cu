// DILU on the device by level scheduling. See device_dilu.cuh for why exactness is the requirement.
#include "device_dilu.cuh"
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
    d.valid = true;
    return d;
}

void diluUpdate(const DeviceLduView& A, DeviceDilu& d)
{
    if (!d.valid) return;
    const label *owner = A.owner, *losort = A.losort, *losortStart = A.losortStart;
    const scalar *upper = A.upper, *lower = A.lower, *diag = A.diag;
    scalar* workd = d.work.data();
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
