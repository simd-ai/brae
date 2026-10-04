#include "device_patch_wave.cuh"
#include "device_blas.cuh"
#include "inter_phase_time.cuh"
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

const char* const WHO = "brae devicePatchWave: ";

// SMALL, GREAT, VGREAT and FaceCellWaveBase::propagationTol_, as the host wave has them
constexpr scalar pwSmall = 1.0e-15;
constexpr scalar pwGreat = 1.0e15;
constexpr scalar pwVGreat = 1.0e300;
constexpr scalar pwTol = 0.01;

constexpr int pwTPB = 256;

int pwBlocks(label n)
{
    return static_cast<int>((n + pwTPB - 1)/pwTPB);
}

// a wallPoint: the origin and the squared distance to it
struct WallInfo
{
    scalar x;
    scalar y;
    scalar z;
    scalar d;
};

__device__
inline WallInfo pwLoad(
    const scalar* ox,
    const scalar* oy,
    const scalar* oz,
    const scalar* dist,
    label e)
{
    return WallInfo{ox[e], oy[e], oz[e], dist[e]};
}

// wallPoint::equal is operator==, which compares the origins only
__device__
inline bool pwEqual(
    const WallInfo& a,
    const WallInfo& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// wallPoint::update (wallPointI.H), as WallPoint::update of the host wave: the element at (px, py, pz) takes
// w2's origin if it is nearer by more than the tolerance. The squared distance is the host's compiled
// expression -- see the header.
__device__
inline bool pwUpdate(
    scalar px,
    scalar py,
    scalar pz,
    const WallInfo& w2,
    WallInfo& cur)
{
    const scalar dx = px - w2.x;
    const scalar dy = py - w2.y;
    const scalar dz = pz - w2.z;
    const scalar dist2 = fma(dz, dz, fma(dx, dx, dy*dy));
    if (!(cur.d > -pwSmall))
    {
        cur = WallInfo{w2.x, w2.y, w2.z, dist2};
        return true;
    }
    const scalar diff = cur.d - dist2;
    if (diff < 0)
    {
        return false;
    }
    if ((diff < pwSmall) || ((cur.d > pwSmall) && (diff/cur.d < pwTol)))
    {
        return false;
    }
    cur = WallInfo{w2.x, w2.y, w2.z, dist2};
    return true;
}

struct NonNegative
{
    __device__
    bool operator()(const label v) const
    {
        return v >= 0;
    }
};

// wallPoint(): origin point::max, distSqr -GREAT, on every cell and face
__global__
void pwInitKernel(
    label n,
    scalar* ox,
    scalar* oy,
    scalar* oz,
    scalar* dist)
{
    const label e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= n)
    {
        return;
    }
    ox[e] = pwVGreat;
    oy[e] = pwVGreat;
    oz[e] = pwVGreat;
    dist[e] = -pwGreat;
}

// patchWave::setChangedFaces: each seed face holds its own centre at distance 0
__global__
void pwSeedKernel(
    label nSeeds,
    const label* seeds,
    label nC,
    const scalar* px,
    const scalar* py,
    const scalar* pz,
    scalar* ox,
    scalar* oy,
    scalar* oz,
    scalar* dist)
{
    const label i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nSeeds)
    {
        return;
    }
    const label e = nC + seeds[i];
    ox[e] = px[e];
    oy[e] = py[e];
    oz[e] = pz[e];
    dist[e] = 0;
}

// one atomic a warp, as deviceSmooth lists its candidate cells
__device__
inline void pwPushCandidate(
    label c,
    bool want,
    label* candidates,
    label* counter)
{
    const unsigned active = __activemask();
    const unsigned ballot = __ballot_sync(active, want);
    if (ballot == 0)
    {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(ballot) - 1;
    label base = 0;
    if (lane == leader)
    {
        base = atomicAdd(counter, __popc(ballot));
    }
    base = __shfl_sync(active, base, leader);
    if (want)
    {
        candidates[base + __popc(ballot & ((1u << lane) - 1u))] = c;
    }
}

// faceToCell, first half: each changed face records its place in the list and names its cells, each cell once
__global__
void pwFaceMarkKernel(
    label nf,
    const label* changedFaces,
    label nIf,
    const label* own,
    const label* nei,
    label* facePos,
    label* touched,
    label* candidates,
    label* counter)
{
    const label i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nf)
    {
        return;
    }
    const label f = changedFaces[i];
    facePos[f] = i;
    pwPushCandidate(own[f], atomicExch(&touched[own[f]], 1) == 0, candidates, counter);
    const label n = (f < nIf) ? nei[f] : -1;
    pwPushCandidate(n, n >= 0 && atomicExch(&touched[n], 1) == 0, candidates, counter);
}

// faceToCell, second half: each cell a changed face touches takes those faces in the host's order -- by place
// in the changed list, the owner side before the neighbour's -- and records the visit of its first change
__global__
void pwFaceToCellKernel(
    const label* counter,
    const label* candidates,
    const label* cellStart,
    const label* cellFaces,
    const label* own,
    const label* facePos,
    label nC,
    const scalar* px,
    const scalar* py,
    const scalar* pz,
    bool ownerOnly,
    scalar* ox,
    scalar* oy,
    scalar* oz,
    scalar* dist,
    label* touched,
    label* slot)
{
    const label j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= *counter)
    {
        return;
    }
    const label c = candidates[j];
    touched[c] = 0;
    const label s = cellStart[c];
    const label e = cellStart[c + 1];
    WallInfo cur = pwLoad(ox, oy, oz, dist, c);
    label first = -1;
    // the next visit is the smallest index above the last one taken: a cell has a handful of faces
    label last = -1;
    for (;;)
    {
        label best = INT_MAX;
        label bestF = -1;
        for (label k = s; k < e; ++k)
        {
            const label f = cellFaces[k];
            const label p = facePos[f];
            if (p < 0)
            {
                continue;
            }
            const bool owned = own[f] == c;
            if (ownerOnly && !owned)
            {
                continue;
            }
            const label visit = 2*p + (owned ? 0 : 1);
            if (visit > last && visit < best)
            {
                best = visit;
                bestF = f;
            }
        }
        if (bestF < 0)
        {
            break;
        }
        last = best;
        const WallInfo nv = pwLoad(ox, oy, oz, dist, nC + bestF);
        if (pwEqual(cur, nv))
        {
            continue;
        }
        if (pwUpdate(px[c], py[c], pz[c], nv, cur) && first < 0)
        {
            first = best;
        }
    }
    ox[c] = cur.x;
    oy[c] = cur.y;
    oz[c] = cur.z;
    dist[c] = cur.d;
    if (first >= 0)
    {
        slot[first] = c;
    }
}

__global__
void pwFaceUnmarkKernel(
    label nf,
    const label* changedFaces,
    label* facePos)
{
    const label i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nf)
    {
        facePos[changedFaces[i]] = -1;
    }
}

// cellToFace, first half: each changed cell's place in the list and its number of faces; count[nc] is 0, so
// the exclusive scan's last entry is the number of visits
__global__
void pwCellMarkKernel(
    label nc,
    const label* changedCells,
    const label* cellStart,
    label* cellPos,
    label* count)
{
    const label j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j > nc)
    {
        return;
    }
    if (j == nc)
    {
        count[j] = 0;
        return;
    }
    const label c = changedCells[j];
    cellPos[c] = j;
    count[j] = cellStart[c + 1] - cellStart[c];
}

// cellToFace, second half: each face of a changed cell is folded once, by the EARLIER of its changed cells --
// that cell's wallPoint, then the other side's when it changed too, the order the host visits them in
__global__
void pwCellToFaceKernel(
    label nc,
    const label* changedCells,
    const label* cellStart,
    const label* cellFaces,
    label nIf,
    const label* own,
    const label* nei,
    const label* cellPos,
    const label* offset,
    label nC,
    const scalar* px,
    const scalar* py,
    const scalar* pz,
    scalar* ox,
    scalar* oy,
    scalar* oz,
    scalar* dist,
    label* slot)
{
    const label j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= nc)
    {
        return;
    }
    const label c = changedCells[j];
    const label s = cellStart[c];
    const label e = cellStart[c + 1];
    const WallInfo mine = pwLoad(ox, oy, oz, dist, c);
    for (label k = s; k < e; ++k)
    {
        const label f = cellFaces[k];
        const label o = (own[f] == c) ? (f < nIf ? nei[f] : -1) : own[f];
        const label po = (o >= 0) ? cellPos[o] : -1;
        if (po >= 0 && po < j)
        {
            continue;
        }
        const label ef = nC + f;
        WallInfo cur = pwLoad(ox, oy, oz, dist, ef);
        label first = -1;
        if (!pwEqual(cur, mine) && pwUpdate(px[ef], py[ef], pz[ef], mine, cur))
        {
            first = offset[j] + (k - s);
        }
        if (po > j)
        {
            const label so = cellStart[o];
            const label eo = cellStart[o + 1];
            label ko = so;
            while (ko < eo && cellFaces[ko] != f)
            {
                ++ko;
            }
            const WallInfo nv = pwLoad(ox, oy, oz, dist, o);
            if (!pwEqual(cur, nv) && pwUpdate(px[ef], py[ef], pz[ef], nv, cur) && first < 0)
            {
                first = offset[po] + (ko - so);
            }
        }
        ox[ef] = cur.x;
        oy[ef] = cur.y;
        oz[ef] = cur.z;
        dist[ef] = cur.d;
        if (first >= 0)
        {
            slot[first] = f;
        }
    }
}

__global__
void pwCellUnmarkKernel(
    label nc,
    const label* changedCells,
    label* cellPos)
{
    const label j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < nc)
    {
        cellPos[changedCells[j]] = -1;
    }
}

// one count back to the host through the solvers' mailbox, as deviceSmooth reads its counts
label pwReadLabel(const label* d)
{
    label h = 0;
    const DeviceReadValue v[1] =
    {
        {d, &h, true}
    };
    deviceReadValues(v, 1);
    return h;
}

// the non-negative entries of slot[0, n), in order, into `out`; their number
label pwCompactSlots(
    DevicePatchWave& w,
    label n,
    DeviceBuffer<label>& out)
{
    if (n == 0)
    {
        return 0;
    }
    std::size_t bytes = 0;
    cudaCheck(cub::DeviceSelect::If(nullptr, bytes, w.slot.data(), out.data(), w.counter.data(), n,
                                    NonNegative(), cudaStreamPerThread), WHO);
    if (w.scratch.size() < bytes)
    {
        w.scratch.resize(bytes);
    }
    cudaCheck(cub::DeviceSelect::If(w.scratch.data(), bytes, w.slot.data(), out.data(), w.counter.data(), n,
                                    NonNegative(), cudaStreamPerThread), WHO);
    return pwReadLabel(w.counter.data());
}

void pwBuild(
    const PrimitiveMesh& m,
    const CellFaces& cells,
    DevicePatchWave& w)
{
    w.nC = m.nCells();
    w.nF = m.nFaces();
    w.nIf = m.nInternalFaces();
    const std::size_t nC = static_cast<std::size_t>(w.nC);
    const std::size_t nF = static_cast<std::size_t>(w.nF);
    w.own = DeviceBuffer<label>(m.owner());
    w.nei = DeviceBuffer<label>(m.neighbour());
    w.cellStart = DeviceBuffer<label>(cells.start);
    w.cellFaces = DeviceBuffer<label>(cells.faces);
    w.px.resize(nC + nF);
    w.py.resize(nC + nF);
    w.pz.resize(nC + nF);
    w.ox.resize(nC + nF);
    w.oy.resize(nC + nF);
    w.oz.resize(nC + nF);
    w.dist.resize(nC + nF);
    w.facePos.resize(nF);
    w.cellPos.resize(nC);
    w.touched.resize(nC);
    w.changedFaces.resize(nF);
    w.changedCells.resize(nC);
    w.candidates.resize(nC);
    w.count.resize(nC + 1);
    w.offset.resize(nC + 1);
    // a half-sweep has at most 2*nf visits (faceToCell) or the changed cells' faces (cellToFace)
    w.slot.resize(2*nF);
    w.counter.resize(1);
    w.built = true;
}

} // namespace


void devicePatchWave(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const CellFaces& cells,
    const std::vector<label>& seedFaces,
    DevicePatchWave& w,
    std::vector<scalar>& cellDistSqr,
    std::vector<scalar>& boundaryDistSqr)
{
    if (static_cast<label>(cells.start.size()) != m.nCells() + 1)
    {
        throw std::runtime_error(std::string(WHO) + "the cell-to-face list handed in is not this mesh's.");
    }
    if (!w.built || w.nC != m.nCells() || w.nF != m.nFaces() || w.nIf != m.nInternalFaces())
    {
        pwBuild(m, cells, w);
    }
    interPhase::Nested timedWave("patchWave: the wave (device)");
    static const bool ownerOnly = std::getenv("BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY") != nullptr;
    const label nC = w.nC;
    const label nF = w.nF;
    const label nIf = w.nIf;
    const label nAll = nC + nF;
    // the centres as the mesh stands now: cells, then faces
    {
        const std::vector<vector>& C = g.C();
        const std::vector<vector>& Cf = g.Cf();
        std::vector<scalar> hx(static_cast<std::size_t>(nAll));
        std::vector<scalar> hy(static_cast<std::size_t>(nAll));
        std::vector<scalar> hz(static_cast<std::size_t>(nAll));
        for (label c = 0; c < nC; ++c)
        {
            hx[static_cast<std::size_t>(c)] = C[static_cast<std::size_t>(c)].x;
            hy[static_cast<std::size_t>(c)] = C[static_cast<std::size_t>(c)].y;
            hz[static_cast<std::size_t>(c)] = C[static_cast<std::size_t>(c)].z;
        }
        for (label f = 0; f < nF; ++f)
        {
            hx[static_cast<std::size_t>(nC + f)] = Cf[static_cast<std::size_t>(f)].x;
            hy[static_cast<std::size_t>(nC + f)] = Cf[static_cast<std::size_t>(f)].y;
            hz[static_cast<std::size_t>(nC + f)] = Cf[static_cast<std::size_t>(f)].z;
        }
        w.px.copyFrom(hx);
        w.py.copyFrom(hy);
        w.pz.copyFrom(hz);
    }
    cudaCheck(cudaMemsetAsync(w.facePos.data(), 0xFF, static_cast<std::size_t>(nF)*sizeof(label),
                              cudaStreamPerThread), WHO);
    cudaCheck(cudaMemsetAsync(w.cellPos.data(), 0xFF, static_cast<std::size_t>(nC)*sizeof(label),
                              cudaStreamPerThread), WHO);
    cudaCheck(cudaMemsetAsync(w.touched.data(), 0, static_cast<std::size_t>(nC)*sizeof(label),
                              cudaStreamPerThread), WHO);
    pwInitKernel<<<pwBlocks(nAll), pwTPB, 0, cudaStreamPerThread>>>(
        nAll,
        w.ox.data(),
        w.oy.data(),
        w.oz.data(),
        w.dist.data());
    label nf = static_cast<label>(seedFaces.size());
    if (nf > nF)
    {
        throw std::runtime_error(std::string(WHO) + "more seed faces than the mesh has faces.");
    }
    if (nf > 0)
    {
        cudaCheck(cudaMemcpy(w.changedFaces.data(), seedFaces.data(), seedFaces.size()*sizeof(label),
                             cudaMemcpyHostToDevice), WHO);
        pwSeedKernel<<<pwBlocks(nf), pwTPB, 0, cudaStreamPerThread>>>(
            nf,
            w.changedFaces.data(),
            nC,
            w.px.data(),
            w.py.data(),
            w.pz.data(),
            w.ox.data(),
            w.oy.data(),
            w.oz.data(),
            w.dist.data());
    }

    auto faceToCell = [&]()
    {
        if (nf == 0)
        {
            return label(0);
        }
        cudaCheck(cudaMemsetAsync(w.counter.data(), 0, sizeof(label), cudaStreamPerThread), WHO);
        cudaCheck(cudaMemsetAsync(w.slot.data(), 0xFF, 2*static_cast<std::size_t>(nf)*sizeof(label),
                                  cudaStreamPerThread), WHO);
        pwFaceMarkKernel<<<pwBlocks(nf), pwTPB, 0, cudaStreamPerThread>>>(
            nf,
            w.changedFaces.data(),
            nIf,
            w.own.data(),
            w.nei.data(),
            w.facePos.data(),
            w.touched.data(),
            w.candidates.data(),
            w.counter.data());
        // at most two cells a face; the kernel reads the true number from the counter
        const label maxCandidates = (2*nf < nC) ? 2*nf : nC;
        pwFaceToCellKernel<<<pwBlocks(maxCandidates), pwTPB, 0, cudaStreamPerThread>>>(
            w.counter.data(),
            w.candidates.data(),
            w.cellStart.data(),
            w.cellFaces.data(),
            w.own.data(),
            w.facePos.data(),
            nC,
            w.px.data(),
            w.py.data(),
            w.pz.data(),
            ownerOnly,
            w.ox.data(),
            w.oy.data(),
            w.oz.data(),
            w.dist.data(),
            w.touched.data(),
            w.slot.data());
        pwFaceUnmarkKernel<<<pwBlocks(nf), pwTPB, 0, cudaStreamPerThread>>>(
            nf,
            w.changedFaces.data(),
            w.facePos.data());
        const label nc = pwCompactSlots(w, 2*nf, w.changedCells);
        nf = 0;
        return nc;
    };
    auto cellToFace = [&](label nc)
    {
        pwCellMarkKernel<<<pwBlocks(nc + 1), pwTPB, 0, cudaStreamPerThread>>>(
            nc,
            w.changedCells.data(),
            w.cellStart.data(),
            w.cellPos.data(),
            w.count.data());
        std::size_t bytes = 0;
        cudaCheck(cub::DeviceScan::ExclusiveSum(nullptr, bytes, w.count.data(), w.offset.data(), nc + 1,
                                                cudaStreamPerThread), WHO);
        if (w.scratch.size() < bytes)
        {
            w.scratch.resize(bytes);
        }
        cudaCheck(cub::DeviceScan::ExclusiveSum(w.scratch.data(), bytes, w.count.data(), w.offset.data(), nc + 1,
                                                cudaStreamPerThread), WHO);
        const label nVisits = pwReadLabel(w.offset.data() + nc);
        cudaCheck(cudaMemsetAsync(w.slot.data(), 0xFF, static_cast<std::size_t>(nVisits)*sizeof(label),
                                  cudaStreamPerThread), WHO);
        pwCellToFaceKernel<<<pwBlocks(nc), pwTPB, 0, cudaStreamPerThread>>>(
            nc,
            w.changedCells.data(),
            w.cellStart.data(),
            w.cellFaces.data(),
            nIf,
            w.own.data(),
            w.nei.data(),
            w.cellPos.data(),
            w.offset.data(),
            nC,
            w.px.data(),
            w.py.data(),
            w.pz.data(),
            w.ox.data(),
            w.oy.data(),
            w.oz.data(),
            w.dist.data(),
            w.slot.data());
        pwCellUnmarkKernel<<<pwBlocks(nc), pwTPB, 0, cudaStreamPerThread>>>(
            nc,
            w.changedCells.data(),
            w.cellPos.data());
        nf = pwCompactSlots(w, nVisits, w.changedFaces);
        return nf;
    };
    // MeshWave<wallPoint>(mesh, changedFaces, faceDist, nTotalCells + 1): FaceCellWave::iterate
    const label maxIter = nC + 1;
    label iter = 0;
    for (; iter < maxIter; ++iter)
    {
        const label nCells = faceToCell();
        const label nFaces = nCells ? cellToFace(nCells) : 0;
        if (!nCells || !nFaces)
        {
            break;
        }
    }
    if (iter >= maxIter)
    {
        throw std::runtime_error(std::string(WHO) + "Maximum number of iterations reached. Increase maxIter.");
    }
    cellDistSqr.resize(static_cast<std::size_t>(nC));
    boundaryDistSqr.resize(static_cast<std::size_t>(nF - nIf));
    cudaCheck(cudaMemcpy(cellDistSqr.data(), w.dist.data(), cellDistSqr.size()*sizeof(scalar),
                         cudaMemcpyDeviceToHost), WHO);
    if (nF > nIf)
    {
        cudaCheck(cudaMemcpy(boundaryDistSqr.data(), w.dist.data() + nC + nIf,
                             boundaryDistSqr.size()*sizeof(scalar), cudaMemcpyDeviceToHost), WHO);
    }
}

} // namespace brae
