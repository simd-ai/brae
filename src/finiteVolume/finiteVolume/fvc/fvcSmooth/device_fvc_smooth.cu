#include "device_fvc_smooth.cuh"
#include "device_blas.cuh"
#include "inter_phase_time.cuh"
#include <climits>
#include <cstdlib>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

const char* const WHO = "brae deviceSmooth: ";

// smoothDataI.H and FaceCellWaveBase.C:42, as the host wave has them
constexpr scalar waveSmall = 1.0e-15;
constexpr scalar waveVSmall = 1.0e-300;
constexpr scalar waveGreat = 1.0e15;
constexpr scalar propagationTol = 0.01;

constexpr int waveTPB = 256;

int waveBlocks(label n)
{
    return static_cast<int>((n + waveTPB - 1)/waveTPB);
}

// SmoothData::update of the host wave (smoothDataI.H:31-62), in the same operations: no product here is
// followed by an add, so no contraction can make the device's arithmetic differ from the host's
__device__
inline bool waveUpdate(
    scalar& value,
    scalar svf,
    scalar scale)
{
    if (!(value > -waveSmall) || (value < waveVSmall))
    {
        value = svf/scale;
        return true;
    }
    if (svf > (1 + propagationTol)*scale*value)
    {
        value = svf/scale;
        return true;
    }
    return false;
}

struct NonNegative
{
    __device__
    bool operator()(const label v) const
    {
        return v >= 0;
    }
};

// fvcSmooth.C:58-74: every internal face across which one side is more than maxRatio times the other, seeded
// with the larger value; every other face unset (-GREAT). slot[f] names the seeded faces, for the compaction
// that lists them in face order.
__global__
void waveSeedKernel(
    label nF,
    label nIf,
    const label* own,
    const label* nei,
    const scalar* field,
    scalar maxRatio,
    scalar* faceVal,
    label* slot)
{
    const label f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF)
    {
        return;
    }
    scalar v = -waveGreat;
    if (f < nIf)
    {
        const scalar o = field[own[f]];
        const scalar n = field[nei[f]];
        label s = -1;
        if (o > maxRatio*n)
        {
            v = o;
            s = f;
        }
        else if (n > maxRatio*o)
        {
            v = n;
            s = f;
        }
        slot[f] = s;
    }
    faceVal[f] = v;
}

// one atomic a warp: every thread of a warp on the same counter serialised the marking kernel (MEASURED on
// RAS/DTCHull: 43 us a sweep for about 45,000 cells)
__device__
inline void pushCandidate(
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
void waveFaceMarkKernel(
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
    pushCandidate(own[f], atomicExch(&touched[own[f]], 1) == 0, candidates, counter);
    const label n = (f < nIf) ? nei[f] : -1;
    pushCandidate(n, n >= 0 && atomicExch(&touched[n], 1) == 0, candidates, counter);
}

// faceToCell, second half: each cell a changed face touches takes those faces in the host's order -- by
// place in the changed list, the owner side before the neighbour's -- and records the visit of its first
// change in `slot`. `reversed` is the identity gate's control.
__global__
void waveFaceToCellKernel(
    const label* counter,
    const label* candidates,
    const label* cellStart,
    const label* cellFaces,
    const label* own,
    const label* facePos,
    const scalar* faceVal,
    scalar maxRatio,
    bool reversed,
    scalar* cellVal,
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
    scalar cur = cellVal[c];
    label first = -1;
    // the cell's visits, gathered in one pass and kept sorted by insertion: a cell has a handful of changed
    // faces. A cell with more than the list holds takes the selection below instead.
    constexpr int maxVisits = 16;
    label visits[maxVisits];
    label visitFaces[maxVisits];
    int nVisits = 0;
    bool listed = !reversed;
    for (label k = s; k < e && listed; ++k)
    {
        const label f = cellFaces[k];
        const label p = facePos[f];
        if (p < 0)
        {
            continue;
        }
        if (nVisits == maxVisits)
        {
            listed = false;
            break;
        }
        const label visit = 2*p + (own[f] == c ? 0 : 1);
        int q = nVisits;
        while (q > 0 && visits[q - 1] > visit)
        {
            visits[q] = visits[q - 1];
            visitFaces[q] = visitFaces[q - 1];
            --q;
        }
        visits[q] = visit;
        visitFaces[q] = f;
        ++nVisits;
    }
    if (listed)
    {
        for (int q = 0; q < nVisits; ++q)
        {
            const scalar nv = faceVal[visitFaces[q]];
            if (cur == nv)
            {
                continue;
            }
            if (waveUpdate(cur, nv, maxRatio) && first < 0)
            {
                first = visits[q];
            }
        }
        cellVal[c] = cur;
        if (first >= 0)
        {
            slot[first] = c;
        }
        return;
    }
    // the next visit is the smallest index above the last one taken (the largest below, reversed)
    label last = reversed ? INT_MAX : -1;
    for (;;)
    {
        label best = reversed ? -1 : INT_MAX;
        label bestF = -1;
        for (label k = s; k < e; ++k)
        {
            const label f = cellFaces[k];
            const label p = facePos[f];
            if (p < 0)
            {
                continue;
            }
            const label visit = 2*p + (own[f] == c ? 0 : 1);
            const bool next = reversed ? (visit < last && visit > best) : (visit > last && visit < best);
            if (next)
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
        const scalar nv = faceVal[bestF];
        if (cur == nv)
        {
            continue;
        }
        if (waveUpdate(cur, nv, maxRatio) && first < 0)
        {
            first = best;
        }
    }
    cellVal[c] = cur;
    if (first >= 0)
    {
        slot[first] = c;
    }
}

__global__
void waveFaceUnmarkKernel(
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
void waveCellMarkKernel(
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
// that cell's value, then the other side's when it changed too, which is the order the host visits them in.
// The face's first change is recorded in `slot` at that visit's index.
__global__
void waveCellToFaceKernel(
    label nc,
    const label* changedCells,
    const label* cellStart,
    const label* cellFaces,
    label nIf,
    const label* own,
    const label* nei,
    const label* cellPos,
    const label* offset,
    const scalar* cellVal,
    scalar* faceVal,
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
    const scalar mine = cellVal[c];
    for (label k = s; k < e; ++k)
    {
        const label f = cellFaces[k];
        const label o = (own[f] == c) ? (f < nIf ? nei[f] : -1) : own[f];
        const label po = (o >= 0) ? cellPos[o] : -1;
        if (po >= 0 && po < j)
        {
            continue;
        }
        scalar cur = faceVal[f];
        label first = -1;
        if (!(cur == mine) && waveUpdate(cur, mine, scalar(1)))
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
            const scalar nv = cellVal[o];
            if (!(cur == nv) && waveUpdate(cur, nv, scalar(1)) && first < 0)
            {
                first = offset[po] + (ko - so);
            }
        }
        faceVal[f] = cur;
        if (first >= 0)
        {
            slot[first] = f;
        }
    }
}

__global__
void waveCellUnmarkKernel(
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

// one count back to the host through the solvers' mailbox (device_blas.cuh): the host spins on mapped memory.
// A stream synchronise left the GPU idle about 200 us after each of the three reads a sweep makes, pinned
// target or not -- MEASURED on RAS/DTCHull, 90 ms of a 114 ms call.
label readLabel(const label* d)
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
label compactSlots(
    DeviceSmoothWave& w,
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
    return readLabel(w.counter.data());
}

void build(
    const PrimitiveMesh& m,
    const CellFaces& cells,
    DeviceSmoothWave& w)
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
    w.cellVal.resize(nC);
    w.faceVal.resize(nF);
    w.facePos.resize(nF);
    w.cellPos.resize(nC);
    w.touched.resize(nC);
    w.changedFaces.resize(nF);
    w.changedCells.resize(nC);
    w.candidates.resize(nC);
    w.count.resize(nC + 1);
    w.offset.resize(nC + 1);
    // a half-sweep has at most 2*nf visits (faceToCell) or the changed cells' faces (cellToFace); both are at
    // most 2*nFaces
    w.slot.resize(2*nF);
    w.counter.resize(1);
    w.built = true;
}

} // namespace


void deviceSmooth(
    std::vector<scalar>& field,
    scalar coeff,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const CellFaces& cells,
    DeviceSmoothWave& w)
{
    // the host wave's refusal, word for word in substance: fvcSmooth.C:76-92 seeds the coupled faces
    for (const FvPatch& p : patches)
    {
        if (p.coupled || p.type == "cyclic" || p.type == "cyclicAMI" || p.type == "cyclicACMI"
         || p.type == "processor")
        {
            throw std::runtime_error(
                std::string(WHO) + "the mesh has the coupled patch '" + p.name + "'. fvcSmooth.C:76-92 "
                "seeds every coupled face and FaceCellWave carries the wave across them "
                "(handleCyclicPatches), which is not ported.");
        }
    }
    if (static_cast<label>(field.size()) != m.nCells())
    {
        throw std::runtime_error(std::string(WHO) + "the field is not the mesh's cells.");
    }
    if (static_cast<label>(cells.start.size()) != m.nCells() + 1)
    {
        throw std::runtime_error(std::string(WHO) + "the cell-to-face list handed in is not this mesh's.");
    }
    if (!w.built || w.nC != m.nCells() || w.nF != m.nFaces() || w.nIf != m.nInternalFaces())
    {
        build(m, cells, w);
    }
    interPhase::Nested timedWave("fvc::smooth: the wave (device)");
    static const bool reversed = std::getenv("BRAE_CONTROL_WAVE_REVERSED") != nullptr;
    const label nC = w.nC;
    const label nF = w.nF;
    const label nIf = w.nIf;
    // fvcSmooth.C:50 -- a scalar sum, 1 + coeff
    const scalar maxRatio = 1 + coeff;
    cudaCheck(cudaMemsetAsync(w.facePos.data(), 0xFF, static_cast<std::size_t>(nF)*sizeof(label),
                              cudaStreamPerThread), WHO);
    cudaCheck(cudaMemsetAsync(w.cellPos.data(), 0xFF, static_cast<std::size_t>(nC)*sizeof(label),
                              cudaStreamPerThread), WHO);
    cudaCheck(cudaMemsetAsync(w.touched.data(), 0, static_cast<std::size_t>(nC)*sizeof(label),
                              cudaStreamPerThread), WHO);
    w.cellVal.copyFrom(field);
    waveSeedKernel<<<waveBlocks(nF), waveTPB, 0, cudaStreamPerThread>>>(
        nF,
        nIf,
        w.own.data(),
        w.nei.data(),
        w.cellVal.data(),
        maxRatio,
        w.faceVal.data(),
        w.slot.data());
    label nf = compactSlots(w, nIf, w.changedFaces);

    auto faceToCell = [&]()
    {
        if (nf == 0)
        {
            return label(0);
        }
        cudaCheck(cudaMemsetAsync(w.counter.data(), 0, sizeof(label), cudaStreamPerThread), WHO);
        cudaCheck(cudaMemsetAsync(w.slot.data(), 0xFF, 2*static_cast<std::size_t>(nf)*sizeof(label),
                                  cudaStreamPerThread), WHO);
        waveFaceMarkKernel<<<waveBlocks(nf), waveTPB, 0, cudaStreamPerThread>>>(
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
        waveFaceToCellKernel<<<waveBlocks(maxCandidates), waveTPB, 0, cudaStreamPerThread>>>(
            w.counter.data(),
            w.candidates.data(),
            w.cellStart.data(),
            w.cellFaces.data(),
            w.own.data(),
            w.facePos.data(),
            w.faceVal.data(),
            maxRatio,
            reversed,
            w.cellVal.data(),
            w.touched.data(),
            w.slot.data());
        waveFaceUnmarkKernel<<<waveBlocks(nf), waveTPB, 0, cudaStreamPerThread>>>(
            nf,
            w.changedFaces.data(),
            w.facePos.data());
        const label nc = compactSlots(w, 2*nf, w.changedCells);
        nf = 0;
        return nc;
    };
    auto cellToFace = [&](label nc)
    {
        waveCellMarkKernel<<<waveBlocks(nc + 1), waveTPB, 0, cudaStreamPerThread>>>(
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
        const label nVisits = readLabel(w.offset.data() + nc);
        cudaCheck(cudaMemsetAsync(w.slot.data(), 0xFF, static_cast<std::size_t>(nVisits)*sizeof(label),
                                  cudaStreamPerThread), WHO);
        waveCellToFaceKernel<<<waveBlocks(nc), waveTPB, 0, cudaStreamPerThread>>>(
            nc,
            w.changedCells.data(),
            w.cellStart.data(),
            w.cellFaces.data(),
            nIf,
            w.own.data(),
            w.nei.data(),
            w.cellPos.data(),
            w.offset.data(),
            w.cellVal.data(),
            w.faceVal.data(),
            w.slot.data());
        waveCellUnmarkKernel<<<waveBlocks(nc), waveTPB, 0, cudaStreamPerThread>>>(
            nc,
            w.changedCells.data(),
            w.cellPos.data());
        nf = compactSlots(w, nVisits, w.changedFaces);
        return nf;
    };
    // FaceCellWave::iterate, maxIter = nTotalCells, as the host wave counts it
    const label maxIter = nC;
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
        throw std::runtime_error(
            std::string(WHO) + "the wave did not settle in nTotalCells sweeps; FaceCellWave stops with "
            "\"Maximum number of iterations reached\" there.");
    }
    w.cellVal.copyTo(field);
}

} // namespace brae
