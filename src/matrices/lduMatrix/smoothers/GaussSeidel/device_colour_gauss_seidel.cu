// The colour-ordered Gauss-Seidel engine under OpenFOAM's smoothSolver loop, on a colour-major
// permutation of the cells. See the header for what is OpenFOAM's here (the loop, the stop rule, the
// per-cell update), what is not (the order), and the measurement behind the layout.
#include "device_colour_gauss_seidel.cuh"
#include "device_amg_internal.cuh"   // Coloring and greedyColor (the greedy colouring's host result)
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae
{

namespace
{

constexpr int TPB_COLOUR = 256;
constexpr int NC_MAX = COLOUR_GS_MAX_COMPONENTS;

inline int nBlk(int n) { return (n + TPB_COLOUR - 1)/TPB_COLOUR; }

// The natural-layout operands of the components being gathered, and their permuted scratch: slot j
// of both lists is the same component. b may be null (the layout diagnostic gathers no source).
struct ColourGSGather
{
    int nComp = 0;
    const scalar* diag[NC_MAX] = {};
    const scalar* b[NC_MAX] = {};
    const scalar* psi[NC_MAX] = {};
    scalar* diagP[NC_MAX] = {};
    scalar* bP[NC_MAX] = {};
    scalar* psiP[NC_MAX] = {};
};

// The components whose permuted field goes back to the natural layout.
struct ColourGSScatter
{
    int nComp = 0;
    const scalar* psiP[NC_MAX] = {};
    scalar* psi[NC_MAX] = {};
};

// What a colour launch does with a row's accumulator besides the update (the header's THE RESIDUAL
// FROM THE SWEEP): nothing more; also write the row's residual from its NEW value; or save the row's
// OLD value, write the residual from that, and then update it as usual.
enum ColourSweepMode
{
    SWEEP_ONLY = 0,
    SWEEP_RES_NEW = 1,
    SWEEP_RES_OLD_SPECULATIVE = 2
};

// The PERMUTED operands of ONE launch: only the components still sweeping, compacted on the host, so
// the kernels loop over nComp with no mask. slot[] (host side) maps a compacted index back to the
// caller's component. save[k] receives the old values a SWEEP_RES_OLD_SPECULATIVE launch overwrites,
// indexed from the launch's first row. part[k] is component k's row of the block partials
// (the header's THE REDUCTION IN THE LAUNCH), indexed by global block; r[k] is its residual vector,
// written only when writeR is set (null otherwise -- the production path stores no residual row).
struct ColourGSOperands
{
    int nComp = 0;
    int writeR = 0;
    const scalar* diag[NC_MAX] = {};
    const scalar* b[NC_MAX] = {};
    scalar* psi[NC_MAX] = {};
    scalar* r[NC_MAX] = {};
    scalar* save[NC_MAX] = {};
    scalar* part[NC_MAX] = {};
};

// The final sum of one pass: block j of residualSumKernel sums part[j] over every global block into
// *dst[j], the caller's slot of dRes.
struct ColourGSResidualSum
{
    int nComp = 0;
    const scalar* part[NC_MAX] = {};
    scalar* dst[NC_MAX] = {};
};

// The device normFactors a residual pass publishes alongside its sums: component k's device scalar,
// or null where its normFactor is a host value or the pass is not the initial one (the only pass
// that publishes them; the normFactor is fixed for the solve).
struct ColourGSNormFactors
{
    const scalar* p[NC_MAX] = {};
};

// Natural -> permuted for every component of the solve: one contiguous write per array, one strided
// read through cells[]. Paid once per solve, not per sweep.
template <int NC>
__global__
void gatherKernel(
    int nC,
    const label* __restrict__ cells,
    ColourGSGather g)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nC) return;
    const label c = cells[i];
#pragma unroll
    for (int k = 0; k < NC; ++k)
    {
        g.diagP[k][i] = g.diag[k][c];
        g.psiP[k][i] = g.psi[k][c];
        if (g.b[k])
        {
            g.bP[k][i] = g.b[k][c];
        }
    }
}

template <int NC>
__global__
void scatterKernel(
    int nC,
    const label* __restrict__ cells,
    ColourGSScatter s)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nC) return;
    const label c = cells[i];
#pragma unroll
    for (int k = 0; k < NC; ++k)
    {
        s.psi[k][c] = s.psiP[k][i];
    }
}

// The per-entry coefficient: upper[face] where the row's cell owns the face, lower[face] where it is
// the neighbour. The only per-solve cost that scales with faces (2 x nInternalFaces x 8 B written).
__global__
void coeffKernel(
    int nE,
    const label* __restrict__ face,
    const unsigned char* __restrict__ isUpper,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    scalar* __restrict__ coeff)
{
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= nE) return;
    const label f = face[e];
    coeff[e] = isUpper[e] ? upper[f] : lower[f];
}

// One row's residual from its accumulator acc = b - sum_j a_j psi_j (the row's entries in order)
// and its diagonal term: THE one expression the sweep (writing r from the accumulator it already
// holds) and the explicit residual pass both evaluate, so a residual the sweep produces and one the
// pass computes over the same state are the same bits row for row. nvcc contracts d*x into the
// subtraction (--fmad is on by default and nothing in the build turns it off); the identity does
// not rest on which way it goes, only on both sites going the same way, and
// tests/test_colour_gs_fused.cu arm (k1) holds the two vectors together with memcmp (with the
// colouring's writeResidualVector on, so the rows are stored as well as reduced).
__device__ __forceinline__
scalar rowResidual(
    scalar acc,
    scalar d,
    scalar x)
{
    return acc - d*x;
}

// sum|r| of one thread block, per component, into the block's slot of each component's partials
// (the header's THE REDUCTION IN THE LAUNCH): a fixed tree over the block's TPB_COLOUR threads in
// index order -- thread t adds thread t + s for s = 128, 64, ..., 1, the pairing sumMagKernel in
// reductions.cu uses -- so the partial is the same bits run to run; a thread past its launch's last
// row contributes the 0 its caller put in mag. Every thread of the block must reach this (it
// synchronises), including those past the last row.
template <int NC>
__device__ __forceinline__
void blockSumMagInto(
    const scalar* mag,
    const ColourGSOperands& ops,
    int g)
{
    __shared__ scalar sh[NC][TPB_COLOUR];
    const int t = threadIdx.x;
#pragma unroll
    for (int k = 0; k < NC; ++k)
    {
        sh[k][t] = mag[k];
    }
    __syncthreads();
    for (int s = TPB_COLOUR/2; s > 0; s >>= 1)
    {
        if (t < s)
        {
#pragma unroll
            for (int k = 0; k < NC; ++k)
            {
                sh[k][t] += sh[k][t + s];
            }
        }
        __syncthreads();
    }
    if (t == 0)
    {
#pragma unroll
        for (int k = 0; k < NC; ++k)
        {
            ops.part[k][g] = sh[k][0];
        }
    }
}

// One colour of the sweep, every active component from one read of the row. The update is
// GaussSeidelSmoother.C:143-171's: the source, minus the lower terms of the faces whose neighbour
// is this cell (its bPrime, gathered here in losort order because the sweep writes no bPrime),
// minus the upper terms of the faces it owns in face order (:157-160), divided by the diagonal
// (:163) with no guard, as OpenFOAM leaves it. The row's entries hold those terms in that order (see
// the header), so the accumulation is the same sequence the unit test's host reference (cellUpdate)
// performs. Cells of one colour share no face, so a thread never reads a psi another thread of the
// same launch writes.
//
// MODE (the header's THE RESIDUAL FROM THE SWEEP): SWEEP_RES_NEW also takes the row's residual
// acc - d*x_new, the round-off of its own update, which is the row's residual until its neighbours
// move again; SWEEP_RES_OLD_SPECULATIVE saves the row's old value, takes acc - d*x_old -- the
// residual the row had after the previous launch, its neighbours now being current -- and then
// updates it. The update itself is the same instruction sequence in every mode. In both residual
// modes the block reduces its rows' |r| into partial blockOff + blockIdx.x of each component
// (blockSumMagInto), and stores the row's r only when ops.writeR asks for it: the same r is
// reduced either way, so the flag cannot change what is reported.
template <int NC, int MODE>
__global__
void colourSweepKernel(
    int lo,
    int hi,
    int blockOff,
    const label* __restrict__ rowStart,
    const label* __restrict__ nbr,
    const scalar* __restrict__ coeff,
    ColourGSOperands ops)
{
    const int i = lo + blockIdx.x*blockDim.x + threadIdx.x;
    // 0 for a thread past the launch's last row: it takes no row but joins the block's reduction.
    scalar mag[NC];
#pragma unroll
    for (int k = 0; k < NC; ++k)
    {
        mag[k] = 0;
    }
    if (i < hi)
    {
        scalar acc[NC];
#pragma unroll
        for (int k = 0; k < NC; ++k)
        {
            acc[k] = ops.b[k][i];
        }
        const label e1 = rowStart[i + 1];
        for (label e = rowStart[i]; e < e1; ++e)
        {
            const label n = nbr[e];
            const scalar a = coeff[e];
#pragma unroll
            for (int k = 0; k < NC; ++k)
            {
                acc[k] -= a*ops.psi[k][n];
            }
        }
#pragma unroll
        for (int k = 0; k < NC; ++k)
        {
            const scalar d = ops.diag[k][i];
            if (MODE == SWEEP_RES_OLD_SPECULATIVE)
            {
                const scalar xOld = ops.psi[k][i];
                ops.save[k][i - lo] = xOld;
                const scalar r = rowResidual(acc[k], d, xOld);
                mag[k] = fabs(r);
                if (ops.writeR)
                {
                    ops.r[k][i] = r;
                }
            }
            const scalar xNew = acc[k]/d;
            ops.psi[k][i] = xNew;
            if (MODE == SWEEP_RES_NEW)
            {
                const scalar r = rowResidual(acc[k], d, xNew);
                mag[k] = fabs(r);
                if (ops.writeR)
                {
                    ops.r[k][i] = r;
                }
            }
        }
    }
    if constexpr (MODE != SWEEP_ONLY)
    {
        blockSumMagInto<NC>(mag, ops, blockOff + (int)blockIdx.x);
    }
}

// lduMatrix::residual (lduMatrixATmul.C:268-340) as a per-cell gather, every active component
// from one read of the row: the source, minus the lower term of each face whose neighbour is this
// cell and the upper term of each face it owns (:323-327), minus diag*psi (:315-318) -- OpenFOAM
// takes the diagonal term first; it is taken LAST here, through rowResidual, so that the residual
// is the same bits the sweep takes from its own accumulator (see rowResidual). Block g takes its
// rows from the colouring's block table, [blockLo[g], blockHi[g]) -- the colour launches'
// partition of the rows, colour by colour -- and reduces their |r| into partial g of each
// component, so a state's partials are the same numbers in the same slots whichever kernel wrote
// them (the header's THE REDUCTION IN THE LAUNCH); the rows are stored only when ops.writeR is set.
template <int NC>
__global__
void residualKernel(
    const label* __restrict__ blockLo,
    const label* __restrict__ blockHi,
    const label* __restrict__ rowStart,
    const label* __restrict__ nbr,
    const scalar* __restrict__ coeff,
    ColourGSOperands ops)
{
    const int g = blockIdx.x;
    const int i = blockLo[g] + (int)threadIdx.x;
    const int hi = blockHi[g];
    scalar mag[NC];
#pragma unroll
    for (int k = 0; k < NC; ++k)
    {
        mag[k] = 0;
    }
    if (i < hi)
    {
        scalar acc[NC];
#pragma unroll
        for (int k = 0; k < NC; ++k)
        {
            acc[k] = ops.b[k][i];
        }
        const label e1 = rowStart[i + 1];
        for (label e = rowStart[i]; e < e1; ++e)
        {
            const label n = nbr[e];
            const scalar a = coeff[e];
#pragma unroll
            for (int k = 0; k < NC; ++k)
            {
                acc[k] -= a*ops.psi[k][n];
            }
        }
#pragma unroll
        for (int k = 0; k < NC; ++k)
        {
            const scalar r = rowResidual(acc[k], ops.diag[k][i], ops.psi[k][i]);
            mag[k] = fabs(r);
            if (ops.writeR)
            {
                ops.r[k][i] = r;
            }
        }
    }
    blockSumMagInto<NC>(mag, ops, g);
}

// The final sum of a pass: block j sums component j's partials over every global block, in index
// order -- thread t accumulates blocks t, t + TPB_COLOUR, t + 2 TPB_COLOUR, ... ascending, then the
// same fixed tree over the threads as blockSumMagInto -- into the caller's slot of dRes. One
// launch per pass in place of the three two-stage deviceSumMagInto reductions it replaces (50 us
// of the 360 us pass on the 306k case, header).
__global__
void residualSumKernel(
    int nBlocks,
    ColourGSResidualSum f)
{
    __shared__ scalar sh[TPB_COLOUR];
    const int j = blockIdx.x;
    const int t = threadIdx.x;
    const scalar* part = f.part[j];
    scalar s = 0;
    for (int g = t; g < nBlocks; g += TPB_COLOUR)
    {
        s += part[g];
    }
    sh[t] = s;
    __syncthreads();
    for (int w = TPB_COLOUR/2; w > 0; w >>= 1)
    {
        if (t < w)
        {
            sh[t] += sh[t + w];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        *f.dst[j] = sh[0];
    }
}

// A stopped component's rows of the speculated colour back to the values that launch saved: its
// psi is then the state after its last counted sweep, as if the launch had never run.
__global__
void rollbackKernel(
    int lo,
    int n,
    const scalar* __restrict__ save,
    scalar* __restrict__ psi)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    psi[lo + i] = save[i];
}

// A psi through the layout, written straight back to the natural numbering (the diagnostic).
__global__
void layoutAmulKernel(
    int nC,
    const label* __restrict__ cells,
    const label* __restrict__ rowStart,
    const label* __restrict__ nbr,
    const scalar* __restrict__ coeff,
    const scalar* __restrict__ diagP,
    const scalar* __restrict__ psiP,
    scalar* __restrict__ Apsi)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nC) return;
    scalar s = diagP[i]*psiP[i];
    const label e1 = rowStart[i + 1];
    for (label e = rowStart[i]; e < e1; ++e)
    {
        s += coeff[e]*psiP[nbr[e]];
    }
    Apsi[cells[i]] = s;
}

// SolverPerformance::checkConvergence (SolverPerformance.C:62-88): strict `<` on both tests, and
// the relative test only when relTol exceeds small_ (1e-20, SolverPerformance.H:292).
inline bool ofConverged(
    scalar tol,
    scalar relTol,
    scalar finalRes,
    scalar initRes)
{
    return finalRes < tol || (relTol > 1e-20 && finalRes < relTol*initRes);
}

void refuse(const std::string& why)
{
    throw std::runtime_error("deviceColourGaussSeidelFused: " + why);
}

std::string ptrString(const void* p)
{
    char buf[32];
    std::snprintf(buf, sizeof buf, "%p", p);
    return buf;
}

// The scratch for nComp components, sized on first use; DeviceBuffer::resize is a no-op at the same
// size, so a warm colouring pays nothing here. The residual partials and sums are only needed by a
// solve that reports residuals (nSweeps >= 0), the residual vectors only when the colouring asks
// for them as well; the save buffers only by a solve whose residual comes from the sweep, and they
// hold one colour's rows -- the largest colour's. The block table (header) is built on the first
// call: colour k's rows in blocks of TPB_COLOUR from its first row, the last block cut at the
// colour's end, so nBlk(colour size) blocks per colour and colour k's first block at
// blockStartH[k] -- the same blocks the colour launches run.
void ensureScratch(
    const DeviceCellColouring& col,
    int nComp,
    bool withResidual,
    bool withSave)
{
    const std::size_t nC = (std::size_t)col.nCells;
    col.coeff.resize(2*(std::size_t)col.nInternalFaces);
    std::size_t nColourMax = 0;
    for (int k = 0; k < col.nColours; ++k)
    {
        const std::size_t n = (std::size_t)(col.startH[(std::size_t)k + 1] - col.startH[(std::size_t)k]);
        nColourMax = std::max(nColourMax, n);
    }
    if (!col.blocksBuilt)
    {
        std::vector<label> lo;
        std::vector<label> hi;
        col.blockStartH.assign(1, 0);
        for (int k = 0; k < col.nColours; ++k)
        {
            const label s = col.startH[(std::size_t)k];
            const label e = col.startH[(std::size_t)k + 1];
            for (label row = s; row < e; row += TPB_COLOUR)
            {
                lo.push_back(row);
                hi.push_back(std::min(row + TPB_COLOUR, e));
            }
            col.blockStartH.push_back((label)lo.size());
        }
        col.nBlocks = (int)lo.size();
        col.blockLo.copyFrom(lo);
        col.blockHi.copyFrom(hi);
        col.blocksBuilt = true;
    }
    for (int k = 0; k < nComp; ++k)
    {
        col.diagP[k].resize(nC);
        col.bP[k].resize(nC);
        col.psiP[k].resize(nC);
        if (withResidual && col.writeResidualVector)
        {
            col.rP[k].resize(nC);
        }
        if (withSave)
        {
            col.saveP[k].resize(nColourMax);
        }
    }
    if (withResidual)
    {
        // Sized for every component the engine can carry, whatever nComp this solve has, so the
        // addresses stay put across solves (a captured graph, the next step, will hold them).
        col.partP.resize((std::size_t)NC_MAX*(std::size_t)col.nBlocks);
        col.dRes.resize((std::size_t)NC_MAX);
    }
}

void launchGather(
    const DeviceCellColouring& col,
    const ColourGSGather& g)
{
    const int nb = nBlk(col.nCells);
    switch (g.nComp)
    {
        case 1:
            gatherKernel<1><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(col.nCells, col.cells.data(), g);
            break;
        case 2:
            gatherKernel<2><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(col.nCells, col.cells.data(), g);
            break;
        default:
            gatherKernel<3><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(col.nCells, col.cells.data(), g);
            break;
    }
    cudaCheck(cudaGetLastError(), "colourGaussSeidel gather");
}

void launchScatter(
    const DeviceCellColouring& col,
    const ColourGSScatter& s)
{
    if (s.nComp == 0) return;
    const int nb = nBlk(col.nCells);
    switch (s.nComp)
    {
        case 1:
            scatterKernel<1><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(col.nCells, col.cells.data(), s);
            break;
        case 2:
            scatterKernel<2><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(col.nCells, col.cells.data(), s);
            break;
        default:
            scatterKernel<3><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(col.nCells, col.cells.data(), s);
            break;
    }
    cudaCheck(cudaGetLastError(), "colourGaussSeidel scatter");
}

// The per-entry coefficients of A's upper/lower into coeff (2 x nInternalFaces long): the solver's
// scratch for a solve, the diagnostic's own for a layout check.
void launchCoeff(
    const DeviceLduView& A,
    const DeviceCellColouring& col,
    scalar* coeff)
{
    const int nE = 2*col.nInternalFaces;
    if (nE == 0) return;
    coeffKernel<<<nBlk(nE), TPB_COLOUR, 0, cudaStreamPerThread>>>(
        nE,
        col.face.data(),
        col.isUpper.data(),
        A.upper,
        A.lower,
        coeff);
    cudaCheck(cudaGetLastError(), "colourGaussSeidel coefficient gather");
}

// The colour launches of ONE sweep, appended to plan: the colours ascending; when symmetric, the
// colours descending afterwards; the no-op launches left out. Each colour is one contiguous block
// of the permuted numbering.
//
// lastColour is the colour of the most recent PLANNED launch of THIS solve (-1 before the first:
// the gather has just rewritten psi and nothing is current). A colour's launch reads its cells' own
// source, diagonal and coefficients and the psi of OTHER colours only; between two launches of one
// colour with no other colour launched in between, none of those has changed, so the second launch
// would recompute every cell from the same operands in the same order to the same bits. It is
// skipped. On K colours that is colour K-1 at the head of every descending pass (the ascending
// pass ended on it) and colour 0 at the head of an ascending pass that follows a descending one
// (which ended on it); on the 2-colour hex mesh half the launches of every symmetric sweep did no
// work. The state carries across residual passes untouched because a residual pass launches no
// colour and writes no psi, only r; and it holds when the operand list shrinks between launches,
// since a component dropped from it is frozen and one still on it was on the earlier launch's list
// too. A caller that changed psi by other means would reset it to -1; within a solve nothing does.
//
// Planning is separate from launching because the fused residual issues the head of a block's plan
// one pass early and the rest of it on the next pass; every planned launch is issued, in the
// planned order (a stopped component's rows are rolled back, but the launch itself ran), so the
// skip state can advance at planning time.
void planSweep(
    const DeviceCellColouring& col,
    bool symmetric,
    int& lastColour,
    std::vector<int>& plan)
{
    auto colour = [&](int k)
    {
        if (k == lastColour) return;
        if (col.startH[(std::size_t)k + 1] <= col.startH[(std::size_t)k]) return;
        lastColour = k;
        plan.push_back(k);
    };
    for (int k = 0; k < col.nColours; ++k)
    {
        colour(k);
    }
    if (!symmetric) return;
    for (int k = col.nColours - 1; k >= 0; --k)
    {
        colour(k);
    }
}

template <int NC>
void launchColourKernel(
    int mode,
    int lo,
    int hi,
    int blockOff,
    const DeviceCellColouring& col,
    const ColourGSOperands& ops)
{
    const int nb = nBlk(hi - lo);
    switch (mode)
    {
        case SWEEP_RES_NEW:
            colourSweepKernel<NC, SWEEP_RES_NEW><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(
                lo, hi, blockOff, col.rowStart.data(), col.nbr.data(), col.coeff.data(), ops);
            break;
        case SWEEP_RES_OLD_SPECULATIVE:
            colourSweepKernel<NC, SWEEP_RES_OLD_SPECULATIVE><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(
                lo, hi, blockOff, col.rowStart.data(), col.nbr.data(), col.coeff.data(), ops);
            break;
        default:
            colourSweepKernel<NC, SWEEP_ONLY><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(
                lo, hi, blockOff, col.rowStart.data(), col.nbr.data(), col.coeff.data(), ops);
            break;
    }
}

// One planned colour launch, in the given mode, for the compacted operands. The colour's blocks
// are its rows in TPB_COLOUR-row blocks from its first row (nBlk(hi - lo) of them), the block
// table's blocks for that colour, and its first one is global block blockStartH[k]: a residual
// mode writes partial blockStartH[k] + blockIdx.x.
void launchColour(
    const DeviceCellColouring& col,
    const ColourGSOperands& ops,
    int k,
    int mode)
{
    const int lo = col.startH[(std::size_t)k];
    const int hi = col.startH[(std::size_t)k + 1];
    const int blockOff = col.blockStartH[(std::size_t)k];
    switch (ops.nComp)
    {
        case 1:
            launchColourKernel<1>(mode, lo, hi, blockOff, col, ops);
            break;
        case 2:
            launchColourKernel<2>(mode, lo, hi, blockOff, col, ops);
            break;
        default:
            launchColourKernel<3>(mode, lo, hi, blockOff, col, ops);
            break;
    }
    cudaCheck(cudaGetLastError(), "colourGaussSeidel sweep");
}

// Component k's rows of colour c back from its save buffer (rollbackKernel).
void launchRollback(
    const DeviceCellColouring& col,
    int k,
    int c)
{
    const int lo = col.startH[(std::size_t)c];
    const int n = col.startH[(std::size_t)c + 1] - lo;
    if (n <= 0) return;
    rollbackKernel<<<nBlk(n), TPB_COLOUR, 0, cudaStreamPerThread>>>(
        lo,
        n,
        col.saveP[k].data(),
        col.psiP[k].data());
    cudaCheck(cudaGetLastError(), "colourGaussSeidel rollback");
}

// The per-pass wait. cudaStreamSynchronize came back ~300 us AFTER the pass's last kernel had finished
// on the 306k case (nsys 2026-09-08: the GPU done at 310 us, the sync returning at 601 us, the OS side
// parked in poll / sem_wait): once per pass that was a third of the momentum block. So the residual
// sums are PUBLISHED by a one-thread kernel straight into mapped pinned host memory behind a sequence
// number, ordered by a system-scope fence, and the host spins on the number: it sees the sums within
// microseconds of the kernel's end. The initial pass publishes the device normFactors the same way,
// so the solve drains the queue for nothing (the deviceReadScalar per component that preceded the
// first wait was a cudaMemcpy sync each). The spin is bounded: past two seconds it falls back to the
// stream sync and, if the number is still wrong, throws -- a wedge is reported, not waited on. A
// conditional WHILE graph would take the host out of the loop entirely; it is the next step (header).
struct ResidualMailbox
{
    scalar sum[COLOUR_GS_MAX_COMPONENTS];
    scalar nf[COLOUR_GS_MAX_COMPONENTS];
    unsigned long long seq;
};

__global__
void publishResidualsKernel(
    const scalar* __restrict__ dRes,
    int n,
    ColourGSNormFactors nf,
    ResidualMailbox* box,
    unsigned long long seq)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    for (int k = 0; k < n; ++k)
    {
        box->sum[k] = dRes[k];
        if (nf.p[k])
        {
            box->nf[k] = *nf.p[k];
        }
    }
    __threadfence_system();
    *reinterpret_cast<volatile unsigned long long*>(&box->seq) = seq;
}

ResidualMailbox* residualMailbox(ResidualMailbox** devPtr)
{
    static ResidualMailbox* box = nullptr;      // pinned + mapped, allocated once, never freed
    static ResidualMailbox* boxDev = nullptr;
    if (!box)
    {
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&box), sizeof(ResidualMailbox), cudaHostAllocMapped),
                  "colourGaussSeidel residual mailbox");
        box->seq = 0;
        cudaCheck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&boxDev), box, 0),
                  "colourGaussSeidel residual mailbox device pointer");
    }
    *devPtr = boxDev;
    return box;
}

void waitForSequence(
    const ResidualMailbox* box,
    unsigned long long seq)
{
    const volatile unsigned long long* p = reinterpret_cast<const volatile unsigned long long*>(&box->seq);
    const auto t0 = std::chrono::steady_clock::now();
    while (*p != seq)
    {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2))
        {
            cudaCheck(cudaStreamSynchronize(cudaStreamPerThread), "colourGaussSeidel mailbox fallback sync");
            if (*p != seq)
            {
                throw std::runtime_error("deviceColourGaussSeidelFused: the residual mailbox never carried this "
                                         "pass's sequence number, even after a stream sync");
            }
            break;
        }
    }
    // The sums were written before the number, on the device side; on this side the acquire fence
    // keeps the reads of sum[] and nf[] after the read of seq (the host is an aarch64 here).
    std::atomic_thread_fence(std::memory_order_acquire);
}

// The explicit residual pass for the active components: every block of the block table, so every
// row of the permuted layout, with the partials in the colour launches' slots. The initial
// residual's pass, and every pass on a colouring the sweep cannot carry the residual for
// (deviceColourGaussSeidelFusesResidual).
void launchResidual(
    const DeviceCellColouring& col,
    const ColourGSOperands& ops)
{
    const int nb = col.nBlocks;
    if (nb == 0) return;
    switch (ops.nComp)
    {
        case 1:
            residualKernel<1><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(
                col.blockLo.data(), col.blockHi.data(), col.rowStart.data(), col.nbr.data(), col.coeff.data(), ops);
            break;
        case 2:
            residualKernel<2><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(
                col.blockLo.data(), col.blockHi.data(), col.rowStart.data(), col.nbr.data(), col.coeff.data(), ops);
            break;
        default:
            residualKernel<3><<<nb, TPB_COLOUR, 0, cudaStreamPerThread>>>(
                col.blockLo.data(), col.blockHi.data(), col.rowStart.data(), col.nbr.data(), col.coeff.data(), ops);
            break;
    }
    cudaCheck(cudaGetLastError(), "colourGaussSeidel residual");
}

// Each active component's partials summed into its slot of dRes (the publish kernel reads the
// slots), whichever launches wrote them: one block per active component (residualSumKernel).
void launchResidualSum(
    const DeviceCellColouring& col,
    const ColourGSOperands& ops,
    const int* slot)
{
    ColourGSResidualSum f;
    f.nComp = ops.nComp;
    for (int j = 0; j < ops.nComp; ++j)
    {
        f.part[j] = ops.part[j];
        f.dst[j] = col.dRes.data() + slot[j];
    }
    residualSumKernel<<<ops.nComp, TPB_COLOUR, 0, cudaStreamPerThread>>>(col.nBlocks, f);
    cudaCheck(cudaGetLastError(), "colourGaussSeidel residual sum");
}

// Compact the active components' permuted operands into one list. The residual vector is handed
// over only when the colouring asks for it to be written; the partials row of component k is
// its NC_MAX-strided slice of partP (null before a residual-reporting solve sized it: a fixed-count
// solve never reads it).
ColourGSOperands compact(
    int nComp,
    const DeviceCellColouring& col,
    const int* active,
    int* slot)
{
    ColourGSOperands ops;
    ops.writeR = col.writeResidualVector ? 1 : 0;
    for (int k = 0; k < nComp; ++k)
    {
        if (!active[k]) continue;
        const int j = ops.nComp++;
        ops.diag[j] = col.diagP[k].data();
        ops.b[j] = col.bP[k].data();
        ops.psi[j] = col.psiP[k].data();
        ops.r[j] = ops.writeR ? col.rP[k].data() : nullptr;
        ops.save[j] = col.saveP[k].data();
        ops.part[j] = col.partP.size() ? col.partP.data() + (std::size_t)k*(std::size_t)col.nBlocks : nullptr;
        slot[j] = k;
    }
    return ops;
}

// The colouring's layout against the matrix it is asked to sweep.
void validateLayout(
    const DeviceCellColouring& col,
    const DeviceLduView& A)
{
    if (!col.valid)
    {
        refuse("the colouring was not built");
    }
    if (col.nCells != A.nCells)
    {
        refuse("colouring built for " + std::to_string(col.nCells) + " cells, matrix has " + std::to_string(A.nCells));
    }
    if (col.nInternalFaces != A.nInternalFaces)
    {
        refuse("colouring built for " + std::to_string(col.nInternalFaces) + " internal faces, matrix has " + std::to_string(A.nInternalFaces));
    }
    const std::size_t nC = (std::size_t)col.nCells;
    const std::size_t nE = 2*(std::size_t)col.nInternalFaces;
    const bool ranges = (int)col.startH.size() == col.nColours + 1
                     && !col.startH.empty() && col.startH.front() == 0 && col.startH.back() == col.nCells;
    const bool arrays = col.cells.size() == nC && col.newIndex.size() == nC && col.rowStart.size() == nC + 1
                     && col.nbr.size() == nE && col.face.size() == nE && col.isUpper.size() == nE;
    if (!ranges || !arrays)
    {
        refuse("colouring arrays are inconsistent with its nColours/nCells/nInternalFaces");
    }
    // The sizes cannot tell two meshes apart (a 5x7x9 box has a 9x7x5 box's 315 cells and 802
    // internal faces, joined differently), and a colouring swept over a mesh it was not built for is
    // a data race no residual can see. The owner and neighbour arrays live in the DeviceMesh for the
    // run, so their addresses name the mesh: recorded on the first sweep, required of every later one.
    if (!col.meshRecorded)
    {
        col.owner = A.owner;
        col.nei = A.nei;
        col.meshRecorded = true;
    }
    else if (col.owner != A.owner || col.nei != A.nei)
    {
        refuse("the colouring was first swept with the mesh whose owner/neighbour addressing is at "
               + ptrString(col.owner) + "/" + ptrString(col.nei) + "; this matrix's is at "
               + ptrString(A.owner) + "/" + ptrString(A.nei) + " -- a colouring belongs to one mesh");
    }
}

void validate(
    int nComp,
    const GSFusedComponent* comps,
    const DeviceCellColouring& col,
    int nSweeps,
    const DeviceSolverPerf* perf)
{
    if (nComp < 1 || nComp > NC_MAX)
    {
        refuse("nComp " + std::to_string(nComp) + " is outside [1, " + std::to_string(NC_MAX) + "]");
    }
    if (!comps)
    {
        refuse("no components");
    }
    if (!perf)
    {
        refuse("no perf array to report into");
    }
    if (nSweeps == 0)
    {
        refuse("nSweeps 0 would never advance nIterations (smoothSolver.C:178-209 loops forever)");
    }
    for (int k = 0; k < nComp; ++k)
    {
        const std::string who = "component " + std::to_string(k);
        if (!comps[k].A || !comps[k].b || !comps[k].psi)
        {
            refuse(who + " has a null matrix, source or field");
        }
        const DeviceLduView& A = *comps[k].A;
        if (A.nCyc != 0)
        {
            refuse(who + " carries a cyclic interface (" + std::to_string(A.nCyc) + " faces); the colour sweep applies no interfaces");
        }
        if (A.nAmi != 0)
        {
            refuse(who + " carries a cyclicAMI interface (" + std::to_string(A.nAmi) + " faces); the colour sweep applies no interfaces");
        }
        if ((int)comps[k].b->size() != A.nCells)
        {
            refuse(who + " source length " + std::to_string(comps[k].b->size()) + " is not nCells " + std::to_string(A.nCells));
        }
        if ((int)comps[k].psi->size() != A.nCells)
        {
            refuse(who + " field length " + std::to_string(comps[k].psi->size()) + " is not nCells " + std::to_string(A.nCells));
        }
        if (k == 0) continue;
        const DeviceLduView& A0 = *comps[0].A;
        const bool shared = A.upper == A0.upper && A.lower == A0.lower
                         && A.owner == A0.owner && A.nei == A0.nei
                         && A.ownerStart == A0.ownerStart && A.losort == A0.losort && A.losortStart == A0.losortStart
                         && A.nCells == A0.nCells && A.nInternalFaces == A0.nInternalFaces;
        if (!shared)
        {
            refuse(who + " does not share component 0's upper/lower/addressing; the fused sweep needs one matrix topology");
        }
    }
    validateLayout(col, *comps[0].A);
}

void refuseBuild(const std::string& why)
{
    throw std::runtime_error("buildDeviceCellColouring: " + why);
}

// The face lists every layout is built from: equal length, every cell in range, owner non-decreasing.
void checkFaces(
    const std::vector<label>& ownerInternal,
    const std::vector<label>& neiInternal,
    int nCells)
{
    if (ownerInternal.size() != neiInternal.size())
    {
        refuseBuild("owner has " + std::to_string(ownerInternal.size()) + " internal faces, neighbour has " + std::to_string(neiInternal.size()));
    }
    if (nCells < 0)
    {
        refuseBuild("negative nCells");
    }
    const int nF = (int)ownerInternal.size();
    for (int f = 0; f < nF; ++f)
    {
        const label o = ownerInternal[(std::size_t)f];
        const label n = neiInternal[(std::size_t)f];
        if (o < 0 || o >= nCells || n < 0 || n >= nCells)
        {
            refuseBuild("face " + std::to_string(f) + " joins cells " + std::to_string(o) + " and " + std::to_string(n)
                        + " outside [0, " + std::to_string(nCells) + ")");
        }
        // The ownerStart ranges every natural-layout kernel reads (device_mesh.cuh:137-146) are a
        // count-scan of owner, which only names the owned faces when owner is non-decreasing; a
        // layout built by sorting would then disagree with those kernels about which faces a cell
        // owns, so it is refused instead.
        if (f > 0 && o < ownerInternal[(std::size_t)f - 1])
        {
            refuseBuild("owner is not non-decreasing at face " + std::to_string(f) + "; the LDU addressing assumes upper-triangular face order");
        }
    }
}

}   // namespace

DeviceCellColouring buildDeviceCellColouringFromClasses(
    const std::vector<label>& ownerInternal,
    const std::vector<label>& neiInternal,
    int nCells,
    const std::vector<label>& cellsByColour,
    const std::vector<label>& colourStart)
{
    checkFaces(ownerInternal, neiInternal, nCells);
    const int nF = (int)ownerInternal.size();
    if (colourStart.empty() || colourStart.front() != 0 || colourStart.back() != nCells)
    {
        refuseBuild("colour ranges do not start at 0 and end at nCells " + std::to_string(nCells));
    }
    const int nColours = (int)colourStart.size() - 1;
    for (int k = 0; k < nColours; ++k)
    {
        if (colourStart[(std::size_t)k + 1] < colourStart[(std::size_t)k])
        {
            refuseBuild("colour range " + std::to_string(k) + " runs backwards");
        }
    }
    if ((int)cellsByColour.size() != nCells)
    {
        refuseBuild("the classes list " + std::to_string(cellsByColour.size()) + " cells, the mesh has " + std::to_string(nCells));
    }
    std::vector<label> colourOf((std::size_t)nCells, -1);
    for (int k = 0; k < nColours; ++k)
    {
        for (label i = colourStart[(std::size_t)k]; i < colourStart[(std::size_t)k + 1]; ++i)
        {
            const label c = cellsByColour[(std::size_t)i];
            if (c < 0 || c >= nCells)
            {
                refuseBuild("class " + std::to_string(k) + " lists cell " + std::to_string(c) + " outside [0, " + std::to_string(nCells) + ")");
            }
            if (colourOf[(std::size_t)c] >= 0)
            {
                refuseBuild("cell " + std::to_string(c) + " is listed twice (classes " + std::to_string(colourOf[(std::size_t)c]) + " and " + std::to_string(k) + ")");
            }
            colourOf[(std::size_t)c] = k;
        }
    }
    // nCells entries with no duplicate cover every cell, so no cell is left uncoloured here. The
    // fail-proof of the colouring itself: a face between two cells of one colour would make the
    // sweep a data race that no residual can see. Checked once here, not per solve (an O(nFaces)
    // host pass per solve is what the device path exists to avoid).
    for (int f = 0; f < nF; ++f)
    {
        const label o = ownerInternal[(std::size_t)f];
        const label n = neiInternal[(std::size_t)f];
        if (colourOf[(std::size_t)o] == colourOf[(std::size_t)n])
        {
            refuseBuild("face " + std::to_string(f) + " joins cells " + std::to_string(o) + " and " + std::to_string(n)
                        + " of one colour (" + std::to_string(colourOf[(std::size_t)o]) + "); not a colouring");
        }
    }

    // The permutation: each class in ascending original index (greedyColor emits them that way; a
    // caller's classes are sorted so the documented order holds whatever order they arrived in).
    std::vector<label> cells(cellsByColour);
    for (int k = 0; k < nColours; ++k)
    {
        std::sort(cells.begin() + colourStart[(std::size_t)k], cells.begin() + colourStart[(std::size_t)k + 1]);
    }
    std::vector<label> newIndex((std::size_t)nCells);
    for (label i = 0; i < nCells; ++i)
    {
        newIndex[(std::size_t)cells[(std::size_t)i]] = i;
    }

    // losort and ownerStart as the mesh builds them (device_mesh.cuh:136-151, lduAddressing.C:34-91):
    // a stable count-sort of the faces by neighbour, and the count-scan of the (sorted) owner.
    std::vector<label> losortStart((std::size_t)nCells + 1, 0), ownerStart((std::size_t)nCells + 1, 0);
    for (int f = 0; f < nF; ++f)
    {
        ++losortStart[(std::size_t)neiInternal[(std::size_t)f] + 1];
        ++ownerStart[(std::size_t)ownerInternal[(std::size_t)f] + 1];
    }
    for (int c = 0; c < nCells; ++c)
    {
        losortStart[(std::size_t)c + 1] += losortStart[(std::size_t)c];
        ownerStart[(std::size_t)c + 1] += ownerStart[(std::size_t)c];
    }
    std::vector<label> losort((std::size_t)nF);
    {
        std::vector<label> pos(losortStart);
        for (int f = 0; f < nF; ++f)
        {
            losort[(std::size_t)pos[(std::size_t)neiInternal[(std::size_t)f]]++] = f;
        }
    }

    // The rows over the new numbering. Per row: the neighboured faces in losort order, then the
    // owned faces in face order -- GaussSeidelSmoother's order and the host reference's, so the
    // summation order and every bit of the result are unchanged (see the header).
    std::vector<label> rowStart((std::size_t)nCells + 1, 0);
    for (label i = 0; i < nCells; ++i)
    {
        const label c = cells[(std::size_t)i];
        const label nLo = losortStart[(std::size_t)c + 1] - losortStart[(std::size_t)c];
        const label nOwn = ownerStart[(std::size_t)c + 1] - ownerStart[(std::size_t)c];
        rowStart[(std::size_t)i + 1] = rowStart[(std::size_t)i] + nLo + nOwn;
    }
    const std::size_t nE = 2*(std::size_t)nF;
    std::vector<label> nbr(nE), face(nE);
    std::vector<unsigned char> isUpper(nE);
    for (label i = 0; i < nCells; ++i)
    {
        const label c = cells[(std::size_t)i];
        std::size_t e = (std::size_t)rowStart[(std::size_t)i];
        for (label j = losortStart[(std::size_t)c]; j < losortStart[(std::size_t)c + 1]; ++j)
        {
            const label f = losort[(std::size_t)j];
            nbr[e] = newIndex[(std::size_t)ownerInternal[(std::size_t)f]];
            face[e] = f;
            isUpper[e] = 0;
            ++e;
        }
        for (label f = ownerStart[(std::size_t)c]; f < ownerStart[(std::size_t)c + 1]; ++f)
        {
            nbr[e] = newIndex[(std::size_t)neiInternal[(std::size_t)f]];
            face[e] = f;
            isUpper[e] = 1;
            ++e;
        }
    }

    DeviceCellColouring col;
    col.nColours = nColours;
    col.nCells = nCells;
    col.nInternalFaces = nF;
    col.cells.copyFrom(cells);
    col.newIndex = std::move(newIndex);
    col.startH = colourStart;
    col.rowStart.copyFrom(rowStart);
    col.nbr.copyFrom(nbr);
    col.face.copyFrom(face);
    col.isUpper.copyFrom(isUpper);
    col.valid = true;
    return col;
}

DeviceCellColouring buildDeviceCellColouring(
    const std::vector<label>& ownerInternal,
    const std::vector<label>& neiInternal,
    int nCells)
{
    // greedyColor indexes its adjacency by these lists unchecked, so the checks come first.
    checkFaces(ownerInternal, neiInternal, nCells);
    const Coloring c = greedyColor(ownerInternal, neiInternal, nCells);
    return buildDeviceCellColouringFromClasses(ownerInternal, neiInternal, nCells, c.cells, c.start);
}

bool deviceColourGaussSeidelFusesResidual(const DeviceCellColouring& colouring)
{
    if (colouring.nColours != 2 || colouring.startH.size() != 3) return false;
    return colouring.startH[1] > colouring.startH[0] && colouring.startH[2] > colouring.startH[1];
}

void deviceColourGaussSeidelFused(
    int nComp,
    const GSFusedComponent* comps,
    const DeviceCellColouring& colouring,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    int nSweeps,
    bool symmetric,
    DeviceSolverPerf* perf)
{
    validate(nComp, comps, colouring, nSweeps, perf);
    const DeviceLduView& A = *comps[0].A;
    const bool fused = nSweeps >= 0 && deviceColourGaussSeidelFusesResidual(colouring);
    ensureScratch(colouring, nComp, nSweeps >= 0, fused);

    // Natural -> permuted for every component, and the coefficients of this solve's matrix.
    ColourGSGather gather;
    gather.nComp = nComp;
    for (int k = 0; k < nComp; ++k)
    {
        gather.diag[k] = comps[k].A->diag;
        gather.b[k] = comps[k].b->data();
        gather.psi[k] = comps[k].psi->data();
        gather.diagP[k] = colouring.diagP[k].data();
        gather.bP[k] = colouring.bP[k].data();
        gather.psiP[k] = colouring.psiP[k].data();
    }
    launchGather(colouring, gather);
    launchCoeff(A, colouring, colouring.coeff.data());

    int active[NC_MAX];
    int swept[NC_MAX];
    int slot[NC_MAX];
    // The skip state of launchSweep: no colour is current on the freshly gathered psi.
    int lastColour = -1;
    for (int k = 0; k < nComp; ++k)
    {
        active[k] = 1;
        swept[k] = 0;
    }
    // Only a component that swept goes back: one that never entered the loop is left exactly as
    // the caller had it, not rewritten with a copy of itself.
    auto scatterSwept = [&]()
    {
        ColourGSScatter s;
        for (int k = 0; k < nComp; ++k)
        {
            if (!swept[k]) continue;
            const int j = s.nComp++;
            s.psiP[j] = colouring.psiP[k].data();
            s.psi[j] = comps[k].psi->data();
        }
        launchScatter(colouring, s);
    };

    // The launches of one block of nSweeps sweeps from the current skip state, in order.
    std::vector<int> plan;
    auto planBlock = [&](int n)
    {
        plan.clear();
        for (int s = 0; s < n; ++s)
        {
            planSweep(colouring, symmetric, lastColour, plan);
        }
    };

    // smoothSolver.C:95-119: a negative nSweeps is a fixed count of sweeps with no residual at all;
    // the report carries zero residuals and nIterations = -nSweeps. No normFactor is needed, so no
    // mailbox pass either.
    if (nSweeps < 0)
    {
        const ColourGSOperands ops = compact(nComp, colouring, active, slot);
        planBlock(-nSweeps);
        for (int c : plan)
        {
            launchColour(colouring, ops, c, SWEEP_ONLY);
        }
        for (int k = 0; k < nComp; ++k)
        {
            swept[k] = 1;
            perf[k] = {0, 0, -nSweeps};
        }
        scatterSwept();
        return;
    }

    ResidualMailbox* mailboxDev = nullptr;
    ResidualMailbox* mailbox = residualMailbox(&mailboxDev);
    static unsigned long long mailboxSeq = 0;
    scalar hRes[NC_MAX];
    // The sums of the active components' residual partials, whichever launches wrote them,
    // published with the normFactors asked for; then the wait.
    auto publishResiduals = [&](
        const ColourGSOperands& ops,
        const ColourGSNormFactors& nfPtrs)
    {
        launchResidualSum(colouring, ops, slot);
        const unsigned long long seq = ++mailboxSeq;
        publishResidualsKernel<<<1, 1, 0, cudaStreamPerThread>>>(colouring.dRes.data(), nComp, nfPtrs, mailboxDev, seq);
        cudaCheck(cudaGetLastError(), "colourGaussSeidel publish residuals");
        waitForSequence(mailbox, seq);
        for (int k = 0; k < nComp; ++k)
        {
            hRes[k] = reinterpret_cast<const volatile scalar*>(mailbox->sum)[k];
        }
    };

    // smoothSolver.C:127-150: the initial residual, which also seeds the final one. No sweep has run,
    // so it is the explicit pass on every colouring. The normFactor is fixed for the solve
    // (smoothSolver.C:127-135 computes it once); a device one rides this pass's mailbox publish, so
    // the solve waits once here and never drains the queue for it (each of the three
    // deviceReadScalar calls this replaced was a cudaMemcpy sync ahead of the mailbox's wait).
    ColourGSNormFactors nfPtrs;
    for (int k = 0; k < nComp; ++k)
    {
        nfPtrs.p[k] = comps[k].dNormFactor;
    }
    {
        const ColourGSOperands ops = compact(nComp, colouring, active, slot);
        launchResidual(colouring, ops);
        publishResiduals(ops, nfPtrs);
    }
    scalar nf[NC_MAX];
    for (int k = 0; k < nComp; ++k)
    {
        if (comps[k].dNormFactor)
        {
            nf[k] = reinterpret_cast<const volatile scalar*>(mailbox->nf)[k];
        }
        else
        {
            nf[k] = comps[k].normFactor;
        }
    }
    scalar init[NC_MAX];
    scalar fin[NC_MAX];
    int iter[NC_MAX];
    int nActive = 0;
    for (int k = 0; k < nComp; ++k)
    {
        init[k] = hRes[k]/nf[k];
        fin[k] = init[k];
        iter[k] = 0;
        // smoothSolver.C:159-165: enter the loop when minIter asks for sweeps or the initial
        // residual does not already pass.
        if (minIter > 0 || !ofConverged(tol, relTol, fin[k], init[k]))
        {
            active[k] = 1;
        }
        else
        {
            active[k] = 0;
        }
        nActive += active[k];
    }

    // smoothSolver.C:178-209. Every active component sweeps nSweeps times, then the residual of
    // that state decides each one's own continuation: nIterations grows by nSweeps BEFORE the
    // maxIter test, so it can overshoot maxIter when nSweeps > 1, exactly as OpenFOAM's does.
    //
    // Where the residual comes from (the header's THE RESIDUAL FROM THE SWEEP): on the explicit
    // path, a pass over every row after the block's launches. On the fused path, the block's last
    // launch writes its own colour's rows of r (SWEEP_RES_NEW) and the first launch of the NEXT
    // block, issued now as a speculation, writes the other colour's rows from the values it is about
    // to overwrite (SWEEP_RES_OLD_SPECULATIVE); `plan` then carries what that next block still owes,
    // and a component that stops has its rows of the speculated colour rolled back. The plan of a
    // block is made from the skip state before its head is issued, never remade after -- remaking it
    // from the head's colour would drop the wrong launches (the symmetric block after a speculated
    // colour 1 is [colour 1 skipped] colour 0; planned afresh from lastColour 1 it would be colour
    // 0, colour 1, colour 0).
    bool speculated = false;
    int specColour = -1;
    while (nActive > 0)
    {
        const ColourGSOperands ops = compact(nComp, colouring, active, slot);
        if (!fused)
        {
            planBlock(nSweeps);
            for (int c : plan)
            {
                launchColour(colouring, ops, c, SWEEP_ONLY);
            }
            launchResidual(colouring, ops);
        }
        else
        {
            if (!speculated)
            {
                planBlock(nSweeps);
            }
            const int nOwed = (int)plan.size();
            if (nOwed == 0)
            {
                refuse("internal: a block on two colours planned no launch");
            }
            for (int i = 0; i < nOwed; ++i)
            {
                launchColour(colouring, ops, plan[(std::size_t)i], i + 1 == nOwed ? SWEEP_RES_NEW : SWEEP_ONLY);
            }
            const int resColour = plan.back();
            // The next block's plan; its head is the speculation. On two nonempty colours a plan
            // has at least two launches and its head is the colour the last block did not end on
            // (ascending: it ended on colour 1, so colour 0 is not skipped; symmetric: it ended on
            // colour 0, and the plan is [colour 0 skipped] colour 1 ... colour 0), which is what
            // makes the two launches' rows of r together every row.
            planBlock(nSweeps);
            if (plan.size() < 2 || plan.front() == resColour)
            {
                refuse("internal: the speculative launch is not the other colour of a two-colour block");
            }
            specColour = plan.front();
            plan.erase(plan.begin());
            launchColour(colouring, ops, specColour, SWEEP_RES_OLD_SPECULATIVE);
            speculated = true;
        }
        publishResiduals(ops, ColourGSNormFactors());
        for (int k = 0; k < nComp; ++k)
        {
            if (!active[k]) continue;
            swept[k] = 1;
            fin[k] = hRes[k]/nf[k];
            iter[k] += nSweeps;
            const bool again = (iter[k] < maxIter && !ofConverged(tol, relTol, fin[k], init[k])) || iter[k] < minIter;
            if (again) continue;
            active[k] = 0;
            --nActive;
            if (fused)
            {
                launchRollback(colouring, k, specColour);
            }
        }
    }
    scatterSwept();
    for (int k = 0; k < nComp; ++k)
    {
        perf[k] = {init[k], fin[k], iter[k]};
    }
}

void deviceColourLayoutAmul(
    const DeviceLduView& A,
    const DeviceCellColouring& colouring,
    const DeviceBuffer<scalar>& psi,
    DeviceBuffer<scalar>& Apsi)
{
    validateLayout(colouring, A);
    if ((int)psi.size() != A.nCells)
    {
        refuse("deviceColourLayoutAmul: field length " + std::to_string(psi.size()) + " is not nCells " + std::to_string(A.nCells));
    }
    // Scratch of its own (header): the solver's is what a captured graph will hold the addresses
    // of, and a diagnostic must not rewrite it under a replay. No source is gathered, so no bP.
    DeviceBuffer<scalar> coeff(2*(std::size_t)A.nInternalFaces);
    DeviceBuffer<scalar> diagP((std::size_t)A.nCells);
    DeviceBuffer<scalar> psiP((std::size_t)A.nCells);
    ColourGSGather gather;
    gather.nComp = 1;
    gather.diag[0] = A.diag;
    gather.b[0] = nullptr;
    gather.psi[0] = psi.data();
    gather.diagP[0] = diagP.data();
    gather.bP[0] = nullptr;
    gather.psiP[0] = psiP.data();
    launchGather(colouring, gather);
    launchCoeff(A, colouring, coeff.data());
    Apsi.resize((std::size_t)A.nCells);
    layoutAmulKernel<<<nBlk(A.nCells), TPB_COLOUR, 0, cudaStreamPerThread>>>(
        A.nCells,
        colouring.cells.data(),
        colouring.rowStart.data(),
        colouring.nbr.data(),
        coeff.data(),
        diagP.data(),
        psiP.data(),
        Apsi.data());
    cudaCheck(cudaGetLastError(), "colourGaussSeidel layout Amul");
}

}   // namespace brae
