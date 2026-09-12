#pragma once
// ---------------------------------------------------------------------------------------------------------------
// amg_kernels.cuh -- the mixed-precision (FP32/FP64) V-cycle work kernels, templated on the value type T.
// Defined in a header because a __global__ template must be visible at every launch site; both device_amg.cu
// (V-cycle + smoothers) and device_amg_coarse.cu (deviceCoarseJacobiLoop) launch these, and each TU compiles
// the instantiations it uses (T = scalar for FP64, float for the FP32 preconditioner). Template instantiations
// have vague linkage, so the duplicates across TUs are merged at link -- no bloat, no ODR issue. safeDiag/OMEGA
// come from device_amg_detail.cuh (inlined per TU). The non-template single-precision K kernels stay in device_amg.cu.
// ---------------------------------------------------------------------------------------------------------------
#include "device_amg_detail.cuh"  // safeDiag<T>, OMEGA, and cf_types (scalar/label)
#include <cuda_runtime.h>

namespace brae {

// V-cycle / FP32 GS. T(OMEGA) and T(0) reproduce the FP64 and FP32 constants exactly, so each instantiation is
// byte-identical to the hand-written twin it replaces.
template <typename T>
__global__
void zeroT(
    int n,
    T* x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = T(0);
}
// weighted-Jacobi update: x += omega*(b - Ax)/diag
template <typename T>
__global__
void smoothT(
    int n,
    const T* __restrict__ b,
    const T* __restrict__ Ax,
    const T* __restrict__ diag,
    T* __restrict__ x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] += T(OMEGA)*(b[i]-Ax[i])/safeDiag(diag[i]);   // safeDiag: floor the (FP32) diagonal, never divide by ~0 -> no Inf/NaN preconditioner
}
template <typename T>
__global__
void residualT(
    int n,
    const T* __restrict__ b,
    const T* __restrict__ Ax,
    T* __restrict__ r)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i]-Ax[i];
}
// One color of in-place Gauss-Seidel: cells in [lo,hi) share no face, so the writes never race and the
// off-diagonal reads pick up already-swept colors -> true GS, not Jacobi. Same LDU gather as the SpMV.
template <typename T>
__global__
void gsColorT(
    int lo,
    int hi,
    const label* __restrict__ cells,
    const T* __restrict__ b,
    const T* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const T* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const T* __restrict__ lower,
    T* __restrict__ x)
{
    const int idx = lo + blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= hi) return;
    const int c = cells[idx];
    T off = T(0);
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        off += upper[f] * x[nei[f]];
    for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const int f = losort[k];
        off += lower[f] * x[owner[f]];
    }
    x[c] = (b[c] - off) / safeDiag(diag[c]);   // safeDiag: floor the (FP32) diagonal, never divide by ~0
}

// THE SAME SWEEP ON A COLOUR-MAJOR PERMUTED LAYOUT (device_amg_smoothers.cu, THE LAYOUT, has the
// measurement). gsColorT above reads its row through cells[] over the NATURAL numbering, so a colour
// launch touches every cache line of every per-cell and per-face array and uses only the fraction
// belonging to its colour. The four kernels below run the identical arithmetic over a numbering that
// puts each colour's rows CONTIGUOUS: one gather per Galerkin update (coefficients + diagonal), one
// gather per sweep (source + field), then one launch per colour with no indirection at all.

// dst[i] = src[idx[i]]. The permuted diagonal, gathered where the diagonal changes (amgGalerkin).
template <typename T>
__global__
void gatherByIndexT(
    int n,
    const label* __restrict__ idx,
    const T* __restrict__ src,
    T* __restrict__ dst)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

// The source and the field into the permuted order, one read of cells[] serving both. Paid per SWEEP,
// not per Galerkin: b and x are produced and consumed by the V-cycle's natural-layout kernels (zeroT,
// deviceAmul, the restriction and the prolongation) between one smooth and the next.
template <typename T>
__global__
void gsPermGatherT(
    int n,
    const label* __restrict__ cells,
    const T* __restrict__ b,
    const T* __restrict__ x,
    T* __restrict__ bP,
    T* __restrict__ xP)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label c = cells[i];
    bP[i] = b[c];
    xP[i] = x[c];
}

// The per-entry coefficient: upper[f] where the row's cell OWNS face f (src >= 0), lower[-1-src] where
// it is that face's neighbour. O(nCells + nFaces) per level, paid once per Galerkin update.
template <typename T>
__global__
void gsPermCoeffT(
    int nE,
    const label* __restrict__ src,
    const T* __restrict__ upper,
    const T* __restrict__ lower,
    T* __restrict__ coeff)
{
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= nE) return;
    const label s = src[e];
    coeff[e] = (s >= 0) ? upper[s] : lower[-1 - s];
}

// One colour of in-place Gauss-Seidel on the permuted layout. Rows [lo, hi) are one contiguous block,
// so the launch reads only its own colour's rowStart/b/diag/x and only its own entries.
//
// The arithmetic is gsColorT's, operand for operand: the row's entries are laid out in gsColorT's own
// accumulation sequence -- the owned faces' upper terms in face order, then the neighboured faces'
// lower terms in losort order -- accumulated into the same `off`, and the update is the same
// (b - off)/safeDiag(diag), with the division left unguarded-by-OpenFOAM's-standard exactly as
// gsColorT leaves it. So the two sweeps produce the same bits (tests/test_gpu_amg.cu arm (a)).
//
// xNat is the scatter back to the natural numbering, FUSED into the sweep rather than run as a pass of
// its own: a separate pass would re-read xP and cells[] for the same scattered stores. x (permuted) and
// xNat therefore hold the same value for every cell at every point of the sweep, which is why a
// colour's reads of x[nbr] -- always cells of OTHER colours -- see exactly what gsColorT's psi[nei[f]]
// would have seen.
template <typename T>
__global__
void gsColorPermT(
    int lo,
    int hi,
    const label* __restrict__ rowStart,
    const label* __restrict__ nbr,
    const T* __restrict__ coeff,
    const T* __restrict__ b,
    const T* __restrict__ diag,
    const label* __restrict__ cells,
    T* __restrict__ x,
    T* __restrict__ xNat)
{
    const int i = lo + blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= hi) return;
    T off = T(0);
    const label e1 = rowStart[i+1];
    for (label e = rowStart[i]; e < e1; ++e)
        off += coeff[e] * x[nbr[e]];
    const T xi = (b[i] - off) / safeDiag(diag[i]);   // safeDiag: gsColorT's guard, unchanged
    x[i] = xi;
    xNat[cells[i]] = xi;
}

// Apsi = A psi through the permuted layout, written straight back to the natural numbering. The
// diagonal term FIRST, then the row's entries in their layout order, which is amulKernel's order
// (device_spmv.cu:31-40) -- so on a sound layout this is deviceAmul's bits, and a disagreement is an
// addressing fault. The diagnostic behind test arm (b); the V-cycle never calls it.
template <typename T>
__global__
void permLayoutAmulT(
    int n,
    const label* __restrict__ cells,
    const label* __restrict__ rowStart,
    const label* __restrict__ nbr,
    const T* __restrict__ coeff,
    const T* __restrict__ diag,
    const T* __restrict__ x,
    T* __restrict__ Apsi)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    T s = diag[i] * x[i];
    const label e1 = rowStart[i+1];
    for (label e = rowStart[i]; e < e1; ++e)
        s += coeff[e] * x[nbr[e]];
    Apsi[cells[i]] = s;
}

// RESTRICTION, fine residual -> coarse right-hand side.
//
// This is the hottest nondeterminism site in the solver: it runs on every level of every V-cycle of every
// PCG iteration, and by construction many fine cells land on one coarse cell. The scatter form
//
//     if (c < nF) atomicAdd(&rc[map[c]], r[c]);
//
// therefore summed in whatever order the blocks finished, so the coarse right-hand side -- and with it the
// preconditioned direction, the Krylov path and the iterate the solve stops at -- differed between two
// runs of the same binary on the same case.
//
// The gather form sums each coarse cell's fine cells in ascending fine index from the CSR inverse of `map`
// (AMGLevel::galCellStart/galCellList, built once because agglomeration is static). One thread per COARSE
// cell, and it WRITES rather than accumulating, so the caller no longer has to pre-zero rc.
template <typename T>
__global__
void restrictGatherT(
    int nCoarse,
    const label* __restrict__ cellStart,
    const label* __restrict__ cellList,
    const T* __restrict__ r,
    T* __restrict__ rc)
{
    const int ci = blockIdx.x*blockDim.x + threadIdx.x;
    if (ci >= nCoarse) return;
    T s = T(0);
    for (label k = cellStart[ci]; k < cellStart[ci+1]; ++k) s += r[cellList[k]];
    rc[ci] = s;
}
template <typename T>
__global__
void prolongT(
    int nF,
    const label* __restrict__ map,
    const T* __restrict__ xc,
    T* __restrict__ x)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) x[c] += xc[map[c]];   // injection correction
}

} // namespace brae
