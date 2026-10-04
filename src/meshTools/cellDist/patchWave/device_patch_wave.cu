#include "device_patch_wave.cuh"
#include "device_blas.cuh"
#include "inter_phase_time.cuh"
#include <optional>
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

// faceToCell on ONE cell: it takes its changed faces in the host's order -- by place in the changed list, the
// owner side before the neighbour's -- and records the visit of its first change. Shared by the launched
// half-sweep and by the thin front's one-launch sweeps, so the two cannot drift apart.
__device__
inline void pwFoldCell(
    label c,
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
    label* slot)
{
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

// faceToCell, second half: every cell a changed face touches, a cell a thread
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
    pwFoldCell(c, cellStart, cellFaces, own, facePos, nC, px, py, pz, ownerOnly, ox, oy, oz, dist, slot);
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

// cellToFace on ONE changed cell, the j-th of the list: each of its faces is folded once, by the EARLIER of its
// changed cells -- that cell's wallPoint, then the other side's when it changed too, the order the host visits
// them in. `offset` is the exclusive sum of the changed cells' face counts. Shared like pwFoldCell.
__device__
inline void pwFoldFaces(
    label j,
    label c,
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
    bool listBoundary,
    label* slot)
{
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
        // A CHANGED BOUNDARY FACE IS NOT LISTED. It has one cell, the one it was just set from, and the visit
        // back can change nothing: the cell either still holds that origin (equal, skipped) or has taken a
        // nearer one since, against which the face's is the cell's own old distance to the bit and is refused.
        // The host lists it and makes the visit; the squared distances are the same (the check against the
        // host's wave says so on every case). On a 2-D mesh the empty patches' faces were half the list.
        if (first >= 0 && (listBoundary || f < nIf))
        {
            slot[first] = f;
        }
    }
}

// cellToFace, second half: every changed cell, a cell a thread
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
    bool listBoundary,
    label* slot)
{
    const label j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= nc)
    {
        return;
    }
    pwFoldFaces(j, changedCells[j], cellStart, cellFaces, nIf, own, nei, cellPos, offset, nC, px, py, pz, ox, oy,
                oz, dist, listBoundary, slot);
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

// THE THIN FRONT: one thread block holds it, each thread a contiguous run of its entries. The block is 896
// threads, 28 warps: the kernel below takes 69 registers a thread and a block may ask for 65,536, so at 1,024
// threads the launch is refused ("too many resources requested for launch").
// A front of up to pwBlockCap entries is taken, two a thread at most, and NO MORE: one block runs on one of the
// device's multiprocessors, so past that size the launched half-sweeps, which spread over all of them, are the
// faster. MEASURED, us a sweep, one launch against launched: a 140-cell front (280 changed faces) 28 against 92;
// 280 cells 37; 560 cells (1,120 faces) 62 against 93; and on the 3-D waveMakerMultiPaddleFlap (448,000 cells,
// a front of some 2,000 cells) 328 against 125 when the block was allowed 7,168 entries.
constexpr int pwBlockT = 896;
constexpr int pwBlockCap = 2*pwBlockT;

// the block's prefix sum of one value a thread: inside a warp by shuffles, across the warps through `sW`, ONE
// __syncthreads. Returns the sum of the threads BEFORE this one; `total` is the block's. Every thread calls it.
// (A doubling scan through shared memory was ten barriers a sum and three sums a sweep: MEASURED on
// waveMakerFlap, 35 us a sweep, most of it barriers.)
__device__
inline label pwBlockPrefix(
    label v,
    label* sW,
    label& total)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    label x = v;
    for (int d = 1; d < 32; d <<= 1)
    {
        const label y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d)
        {
            x += y;
        }
    }
    if (lane == 31)
    {
        sW[warp] = x;
    }
    __syncthreads();
    label before = 0;
    total = 0;
    for (int i = 0; i < pwBlockT/32; ++i)
    {
        const label w = sW[i];
        if (i < warp)
        {
            before += w;
        }
        total += w;
    }
    return x - v + before;
}

// MANY SWEEPS IN ONE LAUNCH. While the front fits one thread block the whole wave loop runs here: the two
// half-sweeps above with their marks, folds, unmarks and compactions as phases of one block, a __syncthreads
// between them and the compactions by an in-block prefix sum -- the same folds (pwFoldCell, pwFoldFaces), the
// same visit indices, the same order of the next changed list. A thread takes the entries
// [t*k, t*k + k) of a list of n, k = ceil(n/896), so a list's order is the threads' order. It returns when the
// wave ends, when a front outgrows the block, or after maxSweeps, and says in `status` what the host does next:
//   status[0]  0 the wave has ended; 1 faceToCell on status[1] changed faces; 2 cellToFace on status[1] cells
//   status[2]  the full sweeps done here
// WHY: on a long tank the wave is one cell layer a sweep. MEASURED on waveMakerPiston refined to 896,000 cells
// (1,600 x 560): 1,600 sweeps of a 560-cell front, each half-sweep nine or so launches and a read back, 46 us
// of fixed cost for a few hundred cells' arithmetic -- 149 ms a step.
__global__
void pwThinFrontKernel(
    label nfStart,
    label maxSweeps,
    label capFaces,
    label capCells,
    const label* cellStart,
    const label* cellFaces,
    label nIf,
    const label* own,
    const label* nei,
    label nC,
    const scalar* px,
    const scalar* py,
    const scalar* pz,
    bool ownerOnly,
    bool listBoundary,
    scalar* ox,
    scalar* oy,
    scalar* oz,
    scalar* dist,
    label* facePos,
    label* cellPos,
    label* touched,
    label* slot,
    label* offset,
    label* changedFaces,
    label* changedCells,
    label* status)
{
    __shared__ label sW[pwBlockT/32];
    const label t = threadIdx.x;
    label nf = nfStart;
    label nc = 0;
    label sweeps = 0;
    label phase = 0;
    for (;;)
    {
        // faceToCell: each changed face records its place, then the cells they touch fold, each claimed once
        label k = (nf + pwBlockT - 1)/pwBlockT;
        label lo = (t*k < nf) ? t*k : nf;
        label hi = (lo + k < nf) ? lo + k : nf;
        for (label i = lo; i < hi; ++i)
        {
            facePos[changedFaces[i]] = i;
            slot[2*i] = -1;
            slot[2*i + 1] = -1;
        }
        __syncthreads();
        for (label i = lo; i < hi; ++i)
        {
            const label f = changedFaces[i];
            // the owner's cell, then the neighbour's
            for (int side = 0; side < 2; ++side)
            {
                const label cell = (side == 0) ? own[f] : (f < nIf ? nei[f] : -1);
                if (cell >= 0 && atomicExch(&touched[cell], 1) == 0)
                {
                    pwFoldCell(cell, cellStart, cellFaces, own, facePos, nC, px, py, pz, ownerOnly, ox, oy, oz,
                               dist, slot);
                }
            }
        }
        __syncthreads();
        // the next changed cells: the visits that changed a cell, in visit order (a face owns two)
        label kept = 0;
        for (label i = lo; i < hi; ++i)
        {
            const label f = changedFaces[i];
            facePos[f] = -1;
            touched[own[f]] = 0;
            if (f < nIf)
            {
                touched[nei[f]] = 0;
            }
            kept += (slot[2*i] >= 0 ? 1 : 0) + (slot[2*i + 1] >= 0 ? 1 : 0);
        }
        label at = pwBlockPrefix(kept, sW, nc);
        for (label i = lo; i < hi; ++i)
        {
            if (slot[2*i] >= 0)
            {
                changedCells[at++] = slot[2*i];
            }
            if (slot[2*i + 1] >= 0)
            {
                changedCells[at++] = slot[2*i + 1];
            }
        }
        __syncthreads();
        if (nc == 0)
        {
            phase = 0;
            break;
        }
        if (nc > capCells)
        {
            phase = 2;
            break;
        }
        // cellToFace: each changed cell records its place and its faces' offset, then folds its faces
        k = (nc + pwBlockT - 1)/pwBlockT;
        lo = (t*k < nc) ? t*k : nc;
        hi = (lo + k < nc) ? lo + k : nc;
        label faces = 0;
        for (label j = lo; j < hi; ++j)
        {
            const label c = changedCells[j];
            cellPos[c] = j;
            faces += cellStart[c + 1] - cellStart[c];
        }
        // the exclusive sum of the cells' face counts, an entry a cell and the total after the last
        label nVisits = 0;
        at = pwBlockPrefix(faces, sW, nVisits);
        for (label j = lo; j < hi; ++j)
        {
            const label c = changedCells[j];
            const label n = cellStart[c + 1] - cellStart[c];
            offset[j] = at;
            for (label v = at; v < at + n; ++v)
            {
                slot[v] = -1;
            }
            at += n;
        }
        if (t == 0)
        {
            offset[nc] = nVisits;
        }
        __syncthreads();
        for (label j = lo; j < hi; ++j)
        {
            pwFoldFaces(j, changedCells[j], cellStart, cellFaces, nIf, own, nei, cellPos, offset, nC, px, py, pz,
                        ox, oy, oz, dist, listBoundary, slot);
        }
        __syncthreads();
        // the next changed faces: the visits that changed a face, in visit order (a cell owns its faces' run)
        kept = 0;
        for (label j = lo; j < hi; ++j)
        {
            cellPos[changedCells[j]] = -1;
            for (label v = offset[j]; v < offset[j + 1]; ++v)
            {
                kept += (slot[v] >= 0 ? 1 : 0);
            }
        }
        at = pwBlockPrefix(kept, sW, nf);
        for (label j = lo; j < hi; ++j)
        {
            for (label v = offset[j]; v < offset[j + 1]; ++v)
            {
                if (slot[v] >= 0)
                {
                    changedFaces[at++] = slot[v];
                }
            }
        }
        __syncthreads();
        ++sweeps;
        if (nf == 0)
        {
            phase = 0;
            break;
        }
        if (nf > capFaces || sweeps >= maxSweeps)
        {
            phase = 1;
            break;
        }
    }
    if (t == 0)
    {
        status[0] = phase;
        status[1] = (phase == 2) ? nc : nf;
        status[2] = sweeps;
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
    w.status.resize(4);
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
    // BRAE_CONTROL_PATCH_WAVE_LIST_BOUNDARY=1 lists a changed boundary face as the host does (pwFoldFaces) -- the
    // identity gate's other arm
    static const bool listBoundary = std::getenv("BRAE_CONTROL_PATCH_WAVE_LIST_BOUNDARY") != nullptr;
    const label nC = w.nC;
    const label nF = w.nF;
    const label nIf = w.nIf;
    const label nAll = nC + nF;
    // the centres as the mesh stands now: cells, then faces
    std::optional<interPhase::Nested> part;
    part.emplace("wave: the centres to the device (host loop, three uploads)");
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
    part.emplace("wave: the initial state and the seeds");
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
            listBoundary,
            w.slot.data());
        pwCellUnmarkKernel<<<pwBlocks(nc), pwTPB, 0, cudaStreamPerThread>>>(
            nc,
            w.changedCells.data(),
            w.cellPos.data());
        nf = pwCompactSlots(w, nVisits, w.changedFaces);
        return nf;
    };
    // MeshWave<wallPoint>(mesh, changedFaces, faceDist, nTotalCells + 1): FaceCellWave::iterate
    part.reset();
    // A FRONT THAT FITS ONE THREAD BLOCK is swept inside one launch (pwThinFrontKernel) for as long as it fits;
    // a wider one takes the launched half-sweeps. BRAE_CONTROL_PATCH_WAVE_NO_BLOCK=1 launches every half-sweep,
    // as before -- the identity gate's other arm.
    static const bool noBlock = std::getenv("BRAE_CONTROL_PATCH_WAVE_NO_BLOCK") != nullptr;
    // BRAE_CONTROL_PATCH_WAVE_BLOCK_FACES / _CELLS shrink what the block takes, so that a small case hands the
    // wave back and forth between the two paths -- the identity gate's way of exercising every hand-over
    auto capOf = [](const char* name)
    {
        const char* e = std::getenv(name);
        const int v = e ? std::atoi(e) : pwBlockCap;
        return static_cast<label>((v > 0 && v < pwBlockCap) ? v : pwBlockCap);
    };
    static const label capFaces = capOf("BRAE_CONTROL_PATCH_WAVE_BLOCK_FACES");
    static const label capCells = capOf("BRAE_CONTROL_PATCH_WAVE_BLOCK_CELLS");
    const label maxIter = nC + 1;
    label iter = 0;
    bool ended = false;
    while (!ended && iter < maxIter)
    {
        if (!noBlock && nf > 0 && nf <= capFaces)
        {
            interPhase::Nested timedBlock("wave: thin-front sweeps inside one launch");
            static bool said = false;
            if (!said)
            {
                said = true;
                std::printf("  wall distance: a front of up to %d cells or faces is swept inside one launch; "
                            "BRAE_CONTROL_PATCH_WAVE_NO_BLOCK=1 launches every half-sweep\n", pwBlockCap);
            }
            pwThinFrontKernel<<<1, pwBlockT, 0, cudaStreamPerThread>>>(
                nf,
                maxIter - iter,
                capFaces,
                capCells,
                w.cellStart.data(),
                w.cellFaces.data(),
                nIf,
                w.own.data(),
                w.nei.data(),
                nC,
                w.px.data(),
                w.py.data(),
                w.pz.data(),
                ownerOnly,
                listBoundary,
                w.ox.data(),
                w.oy.data(),
                w.oz.data(),
                w.dist.data(),
                w.facePos.data(),
                w.cellPos.data(),
                w.touched.data(),
                w.slot.data(),
                w.offset.data(),
                w.changedFaces.data(),
                w.changedCells.data(),
                w.status.data());
            cudaCheck(cudaGetLastError(), WHO);
            label st[3] = {0, 0, 0};
            cudaCheck(cudaMemcpy(st, w.status.data(), 3*sizeof(label), cudaMemcpyDeviceToHost), WHO);
            iter += st[2];
            if (st[0] == 0)
            {
                ended = true;
                break;
            }
            if (st[0] == 2)
            {
                // the changed cells outgrew the block: this sweep's second half by launches
                const label nFaces = cellToFace(st[1]);
                ++iter;
                if (!nFaces)
                {
                    ended = true;
                    break;
                }
                continue;
            }
            nf = st[1];
            if (nf <= capFaces)
            {
                // the sweep limit, reached inside the block: the loop's own test stops it
                continue;
            }
        }
        label nCells = 0;
        {
            interPhase::Nested timedHalf("wave: faceToCell half-sweeps");
            nCells = faceToCell();
        }
        label nFaces = 0;
        if (nCells)
        {
            interPhase::Nested timedHalf("wave: cellToFace half-sweeps");
            nFaces = cellToFace(nCells);
        }
        if (!nCells || !nFaces)
        {
            ended = true;
            break;
        }
        ++iter;
    }
    if (!ended && iter >= maxIter)
    {
        throw std::runtime_error(std::string(WHO) + "Maximum number of iterations reached. Increase maxIter.");
    }
    part.emplace("wave: the squared distances down");
    cellDistSqr.resize(static_cast<std::size_t>(nC));
    boundaryDistSqr.resize(static_cast<std::size_t>(nF - nIf));
    cudaCheck(cudaMemcpy(cellDistSqr.data(), w.dist.data(), cellDistSqr.size()*sizeof(scalar),
                         cudaMemcpyDeviceToHost), WHO);
    if (nF > nIf)
    {
        cudaCheck(cudaMemcpy(boundaryDistSqr.data(), w.dist.data() + nC + nIf,
                             boundaryDistSqr.size()*sizeof(scalar), cudaMemcpyDeviceToHost), WHO);
    }
    part.reset();
}

} // namespace brae
