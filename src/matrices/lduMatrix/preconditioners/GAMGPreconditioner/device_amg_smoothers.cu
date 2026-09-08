// AMG V-cycle smoothers: the per-level smoothing kernels + apply functions the V-cycle calls each level.
//   * weighted-Jacobi is inline in the V-cycle (smoothT, amg_kernels.cuh); this file adds the upgrades --
//     Chebyshev polynomial (chebyshevSmooth), OpenFOAM v2606 twoStageGaussSeidel (twoStageGSSmooth), and one
//     multicolor Gauss-Seidel sweep (gsSweep) -- plus the power-iteration spectrum estimate the Chebyshev
//     interval needs (estimateLambdaMax/ensureSpectrum). The V-cycle (device_amg.cu) calls these via decls
//     in device_amg_internal.cuh; the standalone symGaussSeidel solver (device_amg.cu) reuses gsSweep too.
// Verbatim split of device_amg.cu -- no logic change apart from THE LAYOUT below.
//
// THE LAYOUT (BRAE_AMG_GS_PERM, default on). The multicolour Gauss-Seidel smoother needs the FEWEST
// V-cycles of any smoother here and was the SLOWEST in wall time. Measured BEFORE this change on
// squareBend's transonic pressure at 305,760 cells (GB10), per outer iteration and per V-cycle:
//
//                          ms/outer iter   V-cycles/solve   ms/cycle
//   weighted Jacobi            23.0             3.5           6.6
//   two-stage Gauss-Seidel     19.9             2.7           7.4
//   multicolour GS (indirect)  25.1             2.5          10.0
//
// It lost on APPLY COST alone, and for the reason the momentum solver hit first: gsColorT walks a
// cells[] indirection over the NATURAL cell numbering, so each colour launch touches every cache line
// of every per-cell and per-face array and uses only the fraction belonging to its colour. On the
// momentum solver the cure took a full sweep from 524 to 231 us at 306k, at the bandwidth roofline
// (src/matrices/lduMatrix/smoothers/GaussSeidel/device_colour_gauss_seidel.cuh, LAYOUT), and this is
// the same cure per AMG level:
//
//   * the PERMUTATION and the permuted addressing (rowStart/rowNbr/rowSrc on the GridColoring) are a
//     function of the level's GRAPH alone, so they are built once per grid per mesh. That makes them
//     structure, not values, so the AMG binary cache needs no format change: it stores the colouring
//     and the coarse addressing, and the layout is rebuilt from them on load exactly as the Galerkin
//     gather lists are (device_amg_cache.cu serialises neither and rebuilds both).
//   * the COEFFICIENTS change every outer iteration, because amgGalerkin recomputes cDiag/cUpper/cLower
//     per level, so coeff and diagP are gathered THERE -- once per outer iteration against the ~5 sweeps
//     per level an outer iteration runs (2 smooths per level per cycle x ~2.5 cycles per solve) and,
//     decisively, outside every graph capture. A "have I gathered this solve" host test evaluated inside
//     a captured V-cycle bakes its answer into the graph, and every replay would then smooth with the
//     coefficients of the iteration the graph was captured in.
//   * the SOURCE and the FIELD are gathered per sweep, because the V-cycle's natural-layout kernels
//     (zeroT, deviceAmul, the restriction, the prolongation) produce and consume them between one
//     smooth and the next. The scatter back is FUSED into the sweep (gsColorPermT writes the
//     natural-order field from the register it already holds), so a sweep costs one extra gather pass
//     over two vectors and no scatter pass at all.
//   * the SWEEP then runs each colour as one contiguous block, [startH[k], startH[k+1]), with no
//     indirection, accumulating the SAME operands in the SAME order as gsColorT -- so it produces the
//     same bits, which tests/test_gpu_amg.cu arm (a) holds it to. The division by safeDiag() is
//     gsColorT's and is left exactly as it is.
//
// WHAT IT COSTS IN MEMORY, per grid: rowNbr and rowSrc are 2 x nInternalFaces labels each, coeff the
// same count of scalars, and diagP/bP/psiP one scalar per cell -- about 37 MB on a 306k-cell hex fine
// grid and roughly twice that over the whole hierarchy. It is only paid under BRAE_AMG_GS, which also
// turns the FP32 mirrors off (amgPrepareFP32 skips a GS hierarchy), so it largely replaces them.
//
// AFTER is NOT measured here: this build runs the unit tests only. The number to read is the one the
// table above came from -- a 306k squareBend under
//     BRAE_AMG_GS=1 BRAE_PHASE_TIME=1            (and BRAE_AMG_GS_PERM=0 for the control)
// taking `p` from the "[phase] ... of which the linear solves" line and the `pIters` mean. The cycle
// count and pIters must NOT move (the same iterate, to the bit); only ms/cycle should.
#include "device_amg.cuh"          // AMGData / GridColoring
#include "device_amg_detail.cuh"   // safeDiag / OMEGA / CHEB_* / useChebyshev / useGSPermuted / nBlocks / TPB
#include "device_amg_internal.cuh" // matching decls for the smoother interface
#include "amg_kernels.cuh"         // smoothT / zeroT / gsColorT + the permuted-layout kernels
#include "device_ldu.cuh"          // DeviceLduView / deviceAmul
#include "device_blas.cuh"         // deviceDot / deviceScale / deviceCopy
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

namespace {
// z = src/diag: applies the diagonal (Jacobi) preconditioner, used by the power iteration on D^-1 A.
__global__
void invDiagMulK(
    int n,
    const scalar* __restrict__ src,
    const scalar* __restrict__ diag,
    scalar* __restrict__ z)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) z[i] = src[i]/diag[i];
}
// Chebyshev polynomial smoother step: d = c1*d + c2*(b - Ax)/diag ; x += d.
__global__
void chebStepK(
    int n,
    const scalar* __restrict__ b,
    const scalar* __restrict__ Ax,
    const scalar* __restrict__ diag,
    scalar c1,
    scalar c2,
    scalar* __restrict__ d,
    scalar* __restrict__ x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = c1*d[i] + c2*(b[i]-Ax[i])/diag[i];
    x[i] += d[i];
}
// twoStageGaussSeidel kernels (BRAE_AMG_TSGS). upperMul = strictly-upper-triangle SpMV r = U z; the tsgs* kernels
// below are the fused Jacobi/correction updates of the polynomial expansion.
__global__
void upperMulK(
    int n,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ z,
    scalar* __restrict__ r)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        s += upper[f] * z[nei[f]];
    r[c] = s;
}
// lowerMul = strictly-lower-triangle SpMV r = L z, the transpose of upperMul for the symmetric matrix. Pre-smooth
// with U and post-smooth with L so the V-cycle preconditioner stays symmetric, as the AMG-PCG outer solver requires.
__global__
void lowerMulK(
    int n,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    const scalar* __restrict__ z,
    scalar* __restrict__ r)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    scalar s = 0.0;
    for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const int f = losort[k];
        s += lower[f] * z[owner[f]];
    }
    r[c] = s;
}
__global__
void tsgsZUpdateK(
    int n,
    const scalar* __restrict__ b,
    const scalar* __restrict__ Ax,
    const scalar* __restrict__ diag,
    scalar* __restrict__ z,
    scalar* __restrict__ x)
{   // z = (b-Ax)/D; x += z
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const scalar zi = (b[i]-Ax[i])/diag[i];
        z[i]=zi;
        x[i]+=zi;
    }
}
__global__
void tsgsCorrSaveK(
    int n,
    const scalar* __restrict__ r,
    const scalar* __restrict__ diag,
    scalar mult,
    scalar* __restrict__ z,
    scalar* __restrict__ x)
{   // z = r/D; x += mult*z
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const scalar zi = r[i]/diag[i];
        z[i]=zi;
        x[i]+=mult*zi;
    }
}
__global__
void tsgsCorrLastK(
    int n,
    const scalar* __restrict__ r,
    const scalar* __restrict__ diag,
    scalar mult,
    scalar* __restrict__ x)
{                            // last term: x += mult*r/D (no z save)
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] += mult*r[i]/diag[i];
}

// High-frequency seed for the power iteration. The CONSTANT vector is the Laplacian's
// near-null-space (the SMALLEST eigenvalue of D^-1 A), so starting there makes power iteration
// reach lambda_max only through round-off: in exact arithmetic a constant start is orthogonal to
// every non-constant mode and never leaves the null space. In few iterations it UNDER-estimates
// lambda_max badly, and an under-estimated Chebyshev interval [., 1.1*lambdaMax] leaves the true
// top modes ABOVE the interval, where the Chebyshev polynomial amplifies rather than damps them
// -> the smoother diverges (observed as a stall at ~1.6e-3 on a 1M-cell Laplacian). A period-7
// sawtooth injects broad spectral content, exactly as the SA path's hostSpectralRadius does.
__global__
void fillSeedK(
    int n,
    scalar* __restrict__ x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0 + 0.1*static_cast<scalar>(i % 7);
}
} // anon

// amg.spectrumReady. x/ax/z are grid-g sized scratch (the V-cycle's vX/vAx/vR, free before the first cycle).
static scalar estimateLambdaMax(
    const DeviceLduView& A,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& ax,
    DeviceBuffer<scalar>& z)
{
    const int n = A.nCells;
    fillSeedK<<<nBlocks(n),TPB>>>(n, x.data());
    scalar lam = 1.0;
    // 25 iterations, not 12: power iteration converges to lambda_max FROM BELOW, and every ms of
    // under-estimate risks divergence in the Chebyshev smoother, while an over-estimate only mildly
    // slows convergence -> spend the extra iterations, they run once per matrix.
    for (int it = 0; it < 25; ++it)
    {
        const scalar nrm = std::sqrt(deviceDot(x, x));
        if (!(nrm > 0.0)) break;
        deviceScale(x, 1.0/nrm);                                  // x <- x/||x||
        deviceAmul(A, x, ax);                                     // ax = A x
        invDiagMulK<<<nBlocks(n),TPB>>>(n, ax.data(), A.diag, z.data());  // z = D^-1 A x
        lam = std::sqrt(deviceDot(z, z));                         // ||D^-1 A x|| (||x||=1) -> Rayleigh-quotient bound
        deviceCopy(x, z);
    }
    return lam;
}

// THE V-CYCLE'S ONE-TIME HOST-SIDE PREPARATION, which every driver already calls before it captures
// anything (device_amg_pcg.cu:472 ahead of the capture at :547, amgVCycleApply ahead of its own). Two
// things need that position, so both live here even though the name only says the first:
//   * the Chebyshev spectrum estimate below -- host scalars, uncapturable;
//   * the colour-major GS layout, whose build reads the level's addressing back to the host and is
//     therefore uncapturable too (THE LAYOUT at the top of this file). Building it here means the
//     lazy build in gsSweep is only ever a pair of pointer compares, and no capture path can fall
//     back to the indirection sweep and bake that into a graph.
// lambdaMax[g] is for the SMOOTHED grids 0..nLevels-1; the coarsest grid stays a many-sweep Jacobi
// solve. The spectrum half is a no-op unless Chebyshev is selected (no power-iteration cost when
// lambdaMax is unused) or it is already computed for this matrix.
void ensureSpectrum(
    AMGData& amg,
    const DeviceLduView& A)
{
    if (amg.gsSmooth && useGSPermuted())
    {
        const int G = amg.nLevels();
        for (int g = 0; g < G && g < static_cast<int>(amg.coloring.size()); ++g)
        {
            // Grid 0 is the fine matrix the caller handed in; grid g > 0 is level[g-1]'s coarse
            // operator, which amgGalerkin has already filled for this step (the build gathers the
            // coefficients once, and amgGalerkin refreshes them from the next step on).
            const DeviceLduView v = (g == 0) ? A : amg.level[g-1].coarseView();
            amgEnsurePermutedGSLayout(amg.coloring[g], v);
        }
    }
    if (!useChebyshev() || amg.spectrumReady) return;
    amg.lambdaMax[0] = estimateLambdaMax(A, amg.vX[0], amg.vAx[0], amg.vR[0]);
    for (int g = 1; g < amg.nLevels(); ++g)
        amg.lambdaMax[g] = estimateLambdaMax(amg.level[g-1].coarseView(), amg.vX[g], amg.vAx[g], amg.vR[g]);
    amg.spectrumReady = true;
}

// Preconditioned Chebyshev (polynomial) smoother of degree `deg` on the interval [upper/CHEB_EIGRATIO, upper] of
// D^-1 A (upper = CHEB_UPPER_SAFETY*lambdaMax). Damps the high-frequency error far better per flop than weighted-Jacobi, this is
// the standard GPU-AMG smoother upgrade. x is the in/out guess (zero on pre-smooth, prolonged correction on post-);
// d is grid-sized scratch. All coefficients are host CONSTANTS (no device reductions) -> graph-capturable.
void chebyshevSmooth(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& d,
    DeviceBuffer<scalar>& ax,
    scalar lambdaMax,
    int deg)
{
    const int n = A.nCells;
    // Bias the interval top ABOVE the power-iteration estimate: the estimate converges to
    // lambda_max from below, so the true top eigenvalues can sit just past it, exactly where the
    // Chebyshev polynomial amplifies. Over-covering the interval only mildly slows smoothing;
    // under-covering it diverges. 1.2 (not 1.1) buys margin for the residual under-estimate.
    const scalar upper = CHEB_UPPER_SAFETY*lambdaMax, lower = upper/CHEB_EIGRATIO;
    const scalar theta = 0.5*(upper+lower), delta = 0.5*(upper-lower), sigma = theta/delta;
    zeroT<scalar><<<nBlocks(n),TPB>>>(n, d.data());                       // avoid 0*Inf=NaN on the c1=0 first step
    deviceAmul(A, x, ax);                                        // step 0: d = (1/theta)(b-Ax)/D ; x += d
    chebStepK<<<nBlocks(n),TPB>>>(n, b.data(), ax.data(), A.diag, 0.0, 1.0/theta, d.data(), x.data());
    scalar rho = 1.0/sigma;
    for (int k = 1; k < deg; ++k)
    {
        const scalar rhoNew = 1.0/(2.0*sigma - rho);
        const scalar c1 = rho*rhoNew, c2 = 2.0*rhoNew/delta;
        deviceAmul(A, x, ax);
        chebStepK<<<nBlocks(n),TPB>>>(n, b.data(), ax.data(), A.diag, c1, c2, d.data(), x.data());
        rho = rhoNew;
    }
}

// Two-stage Gauss-Seidel smoother (BRAE_AMG_TSGS), OpenFOAM v2606 twoStageGaussSeidel (arXiv:2111.09512, "Scaled
// Smoothers for Navier-Stokes Pressure Projection"). A fully parallel polynomial smoother: per sweep,
//   r = A x ; z = (b - r)/D ; x += z ;                              (weighted-Jacobi stage, w baked into the /D update)
//   for k in 0..order-1: r = U z ; x += (-1)^(k+1) z_k ; z = r/D    (Neumann-series correction in the strictly-upper
//                                                                    triangle, alternating signs)
// No serial dependency and no coloring -> graph-capturable at a fixed order (all coefficients are host constants,
// mult = +-1). order 0 reduces to the weighted-Jacobi smoothT exactly. z/r are level-sized scratch (the V-cycle's
// vD/vAx). Symmetry: the AMG-PCG outer solver needs an SPD preconditioner, so the correction triangle switches
// between pre (forward=U) and post (backward=L=U^T) smooth to keep the V-cycle symmetric; a U-only pre+post is
// non-symmetric here and wrecks PCG convergence.
void twoStageGSSmooth(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& z,
    DeviceBuffer<scalar>& r,
    int nSweeps,
    int order,
    bool forward)
{
    const int n = A.nCells;
    for (int s = 0; s < nSweeps; ++s)
    {
        deviceAmul(A, x, r);                                                                 // r = A x
        if (order == 0)   // == weighted Jacobi
        {
            smoothT<scalar><<<nBlocks(n),TPB>>>(n, b.data(), r.data(), A.diag, x.data());
            continue;
        }
        tsgsZUpdateK<<<nBlocks(n),TPB>>>(n, b.data(), r.data(), A.diag, z.data(), x.data());  // z=(b-Ax)/D; x+=z
        scalar mult = -1.0;
        for (int k = 0; k < order; ++k)
        {
            if (forward) upperMulK<<<nBlocks(n),TPB>>>(n, A.ownerStart, A.nei, A.upper, z.data(), r.data());              // r = U z (pre)
            else         lowerMulK<<<nBlocks(n),TPB>>>(n, A.losortStart, A.losort, A.owner, A.lower, z.data(), r.data()); // r = L z (post -> symmetric)
            if (k < order - 1) tsgsCorrSaveK<<<nBlocks(n),TPB>>>(n, r.data(), A.diag, mult, z.data(), x.data());
            else               tsgsCorrLastK<<<nBlocks(n),TPB>>>(n, r.data(), A.diag, mult, x.data());
            mult = -mult;
        }
    }
}

// THE COLOUR-MAJOR PERMUTED LAYOUT: build, check, gather, sweep (see THE LAYOUT at the top of the file).
namespace {

void refuseLayout(const std::string& why)
{
    throw std::runtime_error("brae AMG multicolour Gauss-Seidel layout: " + why);
}

// A device array of the level's addressing, on the host. The layout is built from the matrix's OWN
// ownerStart/losort/losortStart rather than recomputed from owner/nei, so the entry order is the order
// gsColorT walks by construction, with nothing assumed about how those arrays were made.
template <typename T>
std::vector<T> readBack(
    const T* p,
    std::size_t n)
{
    std::vector<T> h(n);
    if (n)
    {
        cudaCheck(cudaMemcpy(h.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost), "amg GS layout readback");
    }
    return h;
}

// True while a stream capture is in progress on the stream every AMG kernel is launched into. The
// layout build reads memory back to the host, which a capture cannot record.
bool captureInProgress()
{
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    const cudaError_t e = cudaStreamIsCapturing(cudaStreamPerThread, &st);
    if (e != cudaSuccess)
    {
        cudaGetLastError();   // consume: the query must not leave an error for the next cudaCheck
        return true;          // unknown -> assume capturing, which only costs the slower sweep
    }
    return st != cudaStreamCaptureStatusNone;
}

// The layout's fail-proof, over host arrays. "" when the layout is sound, else what is wrong, naming
// the offending row or entry. Run by the builder before it uploads anything, and again by
// amgCheckPermutedGSLayout on what the device actually holds (which is how a corrupted layout is
// caught, tests/test_gpu_amg.cu arm (c)).
std::string checkLayoutHost(
    int nCells,
    int nFaces,
    int nColors,
    const std::vector<label>& startH,
    const std::vector<label>& cells,
    const std::vector<label>& rowStart,
    const std::vector<label>& rowNbr,
    const std::vector<label>& rowSrc,
    const std::vector<label>& ownerStart,
    const std::vector<label>& nei,
    const std::vector<label>& losortStart,
    const std::vector<label>& losort,
    const std::vector<label>& owner)
{
    const std::size_t nE = 2*static_cast<std::size_t>(nFaces);
    if (static_cast<int>(startH.size()) != nColors + 1)
    {
        return "the colour ranges are " + std::to_string(startH.size()) + " long, not nColours+1 = "
                   + std::to_string(nColors + 1);
    }
    if (startH.front() != 0 || startH.back() != nCells)
    {
        return "the colour ranges run [" + std::to_string(startH.front()) + ", " + std::to_string(startH.back())
                   + "), not [0, " + std::to_string(nCells) + ")";
    }
    for (int k = 0; k < nColors; ++k)
    {
        if (startH[static_cast<std::size_t>(k) + 1] < startH[static_cast<std::size_t>(k)])
        {
            return "colour range " + std::to_string(k) + " runs backwards";
        }
    }
    if (static_cast<int>(cells.size()) != nCells || static_cast<int>(rowStart.size()) != nCells + 1
        || rowNbr.size() != nE || rowSrc.size() != nE)
    {
        return "the layout arrays are not sized for " + std::to_string(nCells) + " cells and "
                   + std::to_string(nFaces) + " internal faces";
    }
    // cells[] is a bijection of [0, nCells): anything else and two rows would write one cell.
    std::vector<label> newIndex(static_cast<std::size_t>(nCells), -1);
    for (int i = 0; i < nCells; ++i)
    {
        const label c = cells[static_cast<std::size_t>(i)];
        if (c < 0 || c >= nCells)
        {
            return "cells[" + std::to_string(i) + "] is " + std::to_string(c) + ", outside [0, "
                       + std::to_string(nCells) + ")";
        }
        if (newIndex[static_cast<std::size_t>(c)] >= 0)
        {
            return "cell " + std::to_string(c) + " is at rows " + std::to_string(newIndex[static_cast<std::size_t>(c)])
                       + " and " + std::to_string(i) + "; cells[] is not a permutation";
        }
        newIndex[static_cast<std::size_t>(c)] = i;
    }
    std::vector<int> colourOf(static_cast<std::size_t>(nCells), -1);
    for (int k = 0; k < nColors; ++k)
    {
        for (label i = startH[static_cast<std::size_t>(k)]; i < startH[static_cast<std::size_t>(k) + 1]; ++i)
        {
            colourOf[static_cast<std::size_t>(i)] = k;
        }
    }
    if (rowStart.front() != 0 || rowStart.back() != static_cast<label>(nE))
    {
        return "the rows run [" + std::to_string(rowStart.front()) + ", " + std::to_string(rowStart.back())
                   + "), not [0, 2*nInternalFaces = " + std::to_string(nE) + ")";
    }
    for (int i = 0; i < nCells; ++i)
    {
        const label c = cells[static_cast<std::size_t>(i)];
        const label e0 = rowStart[static_cast<std::size_t>(i)];
        const label e1 = rowStart[static_cast<std::size_t>(i) + 1];
        if (e1 < e0)
        {
            return "row " + std::to_string(i) + " runs backwards";
        }
        label e = e0;
        // gsColorT's order: the faces the cell OWNS in face order, then the faces it NEIGHBOURS in
        // losort order. Every entry must name that face, that side and that neighbour's new index.
        for (label f = ownerStart[static_cast<std::size_t>(c)]; f < ownerStart[static_cast<std::size_t>(c) + 1]; ++f)
        {
            if (e >= e1)
            {
                return "row " + std::to_string(i) + " (cell " + std::to_string(c) + ") is shorter than the faces it owns";
            }
            if (rowSrc[static_cast<std::size_t>(e)] != f)
            {
                return "row " + std::to_string(i) + " entry " + std::to_string(e) + " sources "
                           + std::to_string(rowSrc[static_cast<std::size_t>(e)]) + ", not upper[" + std::to_string(f) + "]";
            }
            if (rowNbr[static_cast<std::size_t>(e)] != newIndex[static_cast<std::size_t>(nei[static_cast<std::size_t>(f)])])
            {
                return "row " + std::to_string(i) + " entry " + std::to_string(e) + " names row "
                           + std::to_string(rowNbr[static_cast<std::size_t>(e)]) + ", not the row of cell "
                           + std::to_string(nei[static_cast<std::size_t>(f)]);
            }
            ++e;
        }
        for (label k = losortStart[static_cast<std::size_t>(c)]; k < losortStart[static_cast<std::size_t>(c) + 1]; ++k)
        {
            const label f = losort[static_cast<std::size_t>(k)];
            if (e >= e1)
            {
                return "row " + std::to_string(i) + " (cell " + std::to_string(c) + ") is shorter than the faces it neighbours";
            }
            if (rowSrc[static_cast<std::size_t>(e)] != -1 - f)
            {
                return "row " + std::to_string(i) + " entry " + std::to_string(e) + " sources "
                           + std::to_string(rowSrc[static_cast<std::size_t>(e)]) + ", not lower[" + std::to_string(f) + "]";
            }
            if (rowNbr[static_cast<std::size_t>(e)] != newIndex[static_cast<std::size_t>(owner[static_cast<std::size_t>(f)])])
            {
                return "row " + std::to_string(i) + " entry " + std::to_string(e) + " names row "
                           + std::to_string(rowNbr[static_cast<std::size_t>(e)]) + ", not the row of cell "
                           + std::to_string(owner[static_cast<std::size_t>(f)]);
            }
            ++e;
        }
        if (e != e1)
        {
            return "row " + std::to_string(i) + " (cell " + std::to_string(c) + ") is longer than the faces it owns and neighbours";
        }
        // ...and the colouring itself: a face joining two cells of one colour makes the sweep a data
        // race that no residual can see, because both rows would be updated by the same launch.
        for (label q = e0; q < e1; ++q)
        {
            const label n = rowNbr[static_cast<std::size_t>(q)];
            if (n < 0 || n >= nCells)
            {
                return "row " + std::to_string(i) + " entry " + std::to_string(q) + " names row " + std::to_string(n)
                           + ", outside [0, " + std::to_string(nCells) + ")";
            }
            if (colourOf[static_cast<std::size_t>(n)] == colourOf[static_cast<std::size_t>(i)])
            {
                return "rows " + std::to_string(i) + " and " + std::to_string(n) + " share a face and colour "
                           + std::to_string(colourOf[static_cast<std::size_t>(i)]) + "; not a colouring";
            }
        }
    }
    return "";
}

// Grid g's permuted addressing from the matrix it will be swept with. Host work, once per grid per
// mesh: read the addressing back, lay each row's entries out in gsColorT's order over the colour-major
// numbering, check the result, upload. Throws on a layout the checker rejects -- a wrong permutation
// is a data race, and the residual of a preconditioner cannot see it.
void buildLayout(
    const GridColoring& gc,
    const DeviceLduView& A)
{
    const int nCells = A.nCells;
    const int nFaces = A.nInternalFaces;
    const std::vector<label> cells = gc.cells.host();
    const std::vector<label> ownerStart = readBack(A.ownerStart, static_cast<std::size_t>(nCells) + 1);
    const std::vector<label> losortStart = readBack(A.losortStart, static_cast<std::size_t>(nCells) + 1);
    const std::vector<label> losort = readBack(A.losort, static_cast<std::size_t>(nFaces));
    const std::vector<label> nei = readBack(A.nei, static_cast<std::size_t>(nFaces));
    const std::vector<label> owner = readBack(A.owner, static_cast<std::size_t>(nFaces));

    std::vector<label> newIndex(static_cast<std::size_t>(nCells), -1);
    for (int i = 0; i < nCells; ++i)
    {
        const label c = cells[static_cast<std::size_t>(i)];
        if (c < 0 || c >= nCells)
        {
            refuseLayout("the colouring lists cell " + std::to_string(c) + ", outside [0, " + std::to_string(nCells) + ")");
        }
        newIndex[static_cast<std::size_t>(c)] = i;
    }
    std::vector<label> rowStart(static_cast<std::size_t>(nCells) + 1, 0);
    for (int i = 0; i < nCells; ++i)
    {
        const label c = cells[static_cast<std::size_t>(i)];
        const label nOwn = ownerStart[static_cast<std::size_t>(c) + 1] - ownerStart[static_cast<std::size_t>(c)];
        const label nLo = losortStart[static_cast<std::size_t>(c) + 1] - losortStart[static_cast<std::size_t>(c)];
        rowStart[static_cast<std::size_t>(i) + 1] = rowStart[static_cast<std::size_t>(i)] + nOwn + nLo;
    }
    const std::size_t nE = 2*static_cast<std::size_t>(nFaces);
    std::vector<label> rowNbr(nE);
    std::vector<label> rowSrc(nE);
    for (int i = 0; i < nCells; ++i)
    {
        const label c = cells[static_cast<std::size_t>(i)];
        std::size_t e = static_cast<std::size_t>(rowStart[static_cast<std::size_t>(i)]);
        for (label f = ownerStart[static_cast<std::size_t>(c)]; f < ownerStart[static_cast<std::size_t>(c) + 1]; ++f)
        {
            rowNbr[e] = newIndex[static_cast<std::size_t>(nei[static_cast<std::size_t>(f)])];
            rowSrc[e] = f;                      // upper[f]: the row's cell owns face f
            ++e;
        }
        for (label k = losortStart[static_cast<std::size_t>(c)]; k < losortStart[static_cast<std::size_t>(c) + 1]; ++k)
        {
            const label f = losort[static_cast<std::size_t>(k)];
            rowNbr[e] = newIndex[static_cast<std::size_t>(owner[static_cast<std::size_t>(f)])];
            rowSrc[e] = -1 - f;                 // lower[f]: the row's cell is face f's neighbour
            ++e;
        }
    }
    const std::string bad = checkLayoutHost(nCells, nFaces, gc.nColors, gc.startH, cells, rowStart, rowNbr, rowSrc,
                                            ownerStart, nei, losortStart, losort, owner);
    if (!bad.empty())
    {
        refuseLayout("the layout just built is not sound: " + bad);
    }
    gc.rowStart.copyFrom(rowStart);
    gc.rowNbr.copyFrom(rowNbr);
    gc.rowSrc.copyFrom(rowSrc);
    gc.coeff.resize(nE);
    gc.diagP.resize(static_cast<std::size_t>(nCells));
    gc.bP.resize(static_cast<std::size_t>(nCells));
    gc.psiP.resize(static_cast<std::size_t>(nCells));
    gc.permCells = nCells;
    gc.permFaces = nFaces;
    gc.permOwner = A.owner;
    gc.permNei = A.nei;
    gc.permBuilt = true;
}

// The layout this colouring holds is the one this matrix needs.
bool layoutCurrent(
    const GridColoring& gc,
    const DeviceLduView& A)
{
    return gc.permBuilt
        && gc.permCells == A.nCells
        && gc.permFaces == A.nInternalFaces
        && gc.permOwner == A.owner
        && gc.permNei == A.nei;
}

} // namespace

bool amgEnsurePermutedGSLayout(
    const GridColoring& gc,
    const DeviceLduView& A)
{
    if (layoutCurrent(gc, A)) return true;
    // Sizes alone cannot tell two grids apart, but a colouring whose cell count is not the matrix's is
    // certainly not this grid's, and sweeping it is a data race no residual can see.
    if (static_cast<int>(gc.cells.size()) != A.nCells)
    {
        refuseLayout("the colouring holds " + std::to_string(gc.cells.size()) + " cells, the grid has "
                     + std::to_string(A.nCells));
    }
    if (gc.nColors <= 0 || static_cast<int>(gc.startH.size()) != gc.nColors + 1)
    {
        refuseLayout("the colouring has " + std::to_string(gc.nColors) + " colours and "
                     + std::to_string(gc.startH.size()) + " range bounds");
    }
    if (captureInProgress())
    {
        // The build reads the addressing back to the host, which a capture cannot record. Every driver
        // calls ensureSpectrum (which builds every grid's layout) before it captures, so this is not
        // reachable from the shipped paths; if some path ever does reach it, it degrades to the
        // indirection sweep -- the same numbers to the bit, at the old cost -- and says so once.
        static bool once = false;
        if (!once)
        {
            once = true;
            std::fprintf(stderr, "[AMG] the multicolour GS layout was first needed inside a graph capture, so this "
                                 "grid keeps the cells[] indirection sweep (same result, ~35%% slower per cycle). "
                                 "Call ensureSpectrum before capturing, or set BRAE_AMG_GS_PERM=0 to silence this.\n");
        }
        return false;
    }
    buildLayout(gc, A);
    // The coefficients of the matrix it was just built for. From here on amgGalerkin owns this gather,
    // once per outer iteration; this call covers the window before the first amgGalerkin that sees a
    // built layout.
    amgGatherPermutedGSCoeffs(gc, A.diag, A.upper, A.lower);
    return true;
}

void amgGatherPermutedGSCoeffs(
    const GridColoring& gc,
    const scalar* diag,
    const scalar* upper,
    const scalar* lower)
{
    if (!gc.permBuilt) return;
    const int nE = 2*gc.permFaces;
    if (nE > 0)
    {
        gsPermCoeffT<scalar><<<nBlocks(nE),TPB>>>(nE, gc.rowSrc.data(), upper, lower, gc.coeff.data());
    }
    gatherByIndexT<scalar><<<nBlocks(gc.permCells),TPB>>>(gc.permCells, gc.cells.data(), diag, gc.diagP.data());
    cudaCheck(cudaGetLastError(), "amg GS permuted coefficient gather");
}

std::string amgCheckPermutedGSLayout(
    const GridColoring& gc,
    const DeviceLduView& A)
{
    if (!layoutCurrent(gc, A)) return "the layout is not built for this grid";
    const std::size_t nC = static_cast<std::size_t>(A.nCells);
    const std::size_t nF = static_cast<std::size_t>(A.nInternalFaces);
    return checkLayoutHost(A.nCells, A.nInternalFaces, gc.nColors, gc.startH,
                           gc.cells.host(), gc.rowStart.host(), gc.rowNbr.host(), gc.rowSrc.host(),
                           readBack(A.ownerStart, nC + 1), readBack(A.nei, nF),
                           readBack(A.losortStart, nC + 1), readBack(A.losort, nF), readBack(A.owner, nF));
}

// One multicolor Gauss-Seidel sweep over grid g, through the cells[] indirection over the natural cell
// numbering. forward = colors 0..nColors-1, backward = reverse. The V-cycle pre-smooths forward and
// post-smooths backward so the whole preconditioner is SYMMETRIC (symmetric GS), required by the plain-CG
// callers; the SIMPLE loop's flexible CG would tolerate either. One kernel launch per color, all bounds
// host-constant -> the sweep is still capturable into the V-cycle graph (same as the Jacobi/Chebyshev paths).
void amgGSSweepIndirect(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    const GridColoring& gc,
    bool forward)
{
    for (int ci = 0; ci < gc.nColors; ++ci)
    {
        const int col = forward ? ci : (gc.nColors-1-ci);
        const int lo = gc.startH[col], hi = gc.startH[col+1];
        const int nc = hi - lo;
        if (nc <= 0) continue;
        gsColorT<scalar><<<nBlocks(nc),TPB>>>(lo, hi, gc.cells.data(), b.data(), A.diag,
            A.ownerStart, A.nei, A.upper, A.losortStart, A.losort, A.owner, A.lower, x.data());
    }
}

// The same sweep on the colour-major permuted layout (THE LAYOUT at the top of this file). One gather of
// the source and the field, then one launch per colour over a CONTIGUOUS row range with no indirection,
// each writing the natural-order field as it goes. Same operands, same order, same bits as the sweep
// above; all bounds are host constants, so it captures into the V-cycle graph exactly as that one does.
void amgGSSweepPermuted(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    const GridColoring& gc,
    bool forward)
{
    if (!layoutCurrent(gc, A))
    {
        refuseLayout("the permuted sweep was asked for a grid whose layout is not built");
    }
    const int n = A.nCells;
    gsPermGatherT<scalar><<<nBlocks(n),TPB>>>(n, gc.cells.data(), b.data(), x.data(), gc.bP.data(), gc.psiP.data());
    for (int ci = 0; ci < gc.nColors; ++ci)
    {
        const int col = forward ? ci : (gc.nColors-1-ci);
        const int lo = gc.startH[col], hi = gc.startH[col+1];
        const int nc = hi - lo;
        if (nc <= 0) continue;
        gsColorPermT<scalar><<<nBlocks(nc),TPB>>>(lo, hi, gc.rowStart.data(), gc.rowNbr.data(), gc.coeff.data(),
            gc.bP.data(), gc.diagP.data(), gc.cells.data(), gc.psiP.data(), x.data());
    }
    cudaCheck(cudaGetLastError(), "amg GS permuted sweep");
}

void amgPermutedLayoutAmul(
    const GridColoring& gc,
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& psi,
    DeviceBuffer<scalar>& Apsi)
{
    if (!layoutCurrent(gc, A))
    {
        refuseLayout("the layout matvec was asked for a grid whose layout is not built");
    }
    const int n = A.nCells;
    Apsi.resize(static_cast<std::size_t>(n));
    // The field into the permuted order; bP is written with a copy of it because the gather kernel
    // serves both arrays and this diagnostic has no source of its own to gather.
    gsPermGatherT<scalar><<<nBlocks(n),TPB>>>(n, gc.cells.data(), psi.data(), psi.data(), gc.bP.data(), gc.psiP.data());
    permLayoutAmulT<scalar><<<nBlocks(n),TPB>>>(n, gc.cells.data(), gc.rowStart.data(), gc.rowNbr.data(),
        gc.coeff.data(), gc.diagP.data(), gc.psiP.data(), Apsi.data());
    cudaCheck(cudaGetLastError(), "amg GS permuted layout matvec");
}

// The sweep the V-cycle calls: the permuted layout when it is available (the default), the cells[]
// indirection otherwise. They are the same numbers to the bit, so this is a layout choice and nothing
// downstream can tell which ran -- see BRAE_AMG_GS_PERM in device_amg_detail.cuh.
void gsSweep(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    const GridColoring& gc,
    bool forward)
{
    if (useGSPermuted() && amgEnsurePermutedGSLayout(gc, A))
    {
        amgGSSweepPermuted(A, b, x, gc, forward);
        return;
    }
    amgGSSweepIndirect(A, b, x, gc, forward);
}

} // namespace brae
