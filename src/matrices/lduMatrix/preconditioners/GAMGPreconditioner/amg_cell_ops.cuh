#pragma once
// ---------------------------------------------------------------------------------------------------------------
// amg_cell_ops.cuh -- the V-cycle's PER-ELEMENT expressions, written once.
//
// The cycle's kernels (amg_kernels.cuh's smoothT/residualT/restrictGatherT/prolongT..., device_amg_internal.cuh's
// amulFK/amulCsrFK) and the FUSED coarse-level kernel (device_amg_fused.cu, one block running every level below a
// size) must produce the same bits. They do so by calling THESE functions: one expression, one set of operands in
// one order, so the compiler contracts a multiply-add into an fma in all of them or in none. Nothing here is a
// launch; each takes the operands of ONE cell (or coarse cell) and returns what that cell's kernel thread wrote.
// ---------------------------------------------------------------------------------------------------------------
#include "device_amg_detail.cuh"   // safeDiag<T>, OMEGA, scalar/label

namespace brae {

// weighted Jacobi: x + omega*(b - Ax)/diag (safeDiag floors the diagonal: never a division by ~0)
template <typename T>
__device__ __forceinline__
T cellSmooth(T x, T b, T ax, T diag)
{
    return x + T(OMEGA)*(b - ax)/safeDiag(diag);
}

// the first sweep of a cycle, whose x and A*x are zero: cellSmooth's expression with the two zeros written in
// (they decide a zero's sign, so the compiler may not drop them)
template <typename T>
__device__ __forceinline__
T cellSmoothFromZero(T b, T diag)
{
    return T(0) + T(OMEGA)*(b - T(0))/safeDiag(diag);
}

template <typename T>
__device__ __forceinline__
T cellResidual(T b, T ax)
{
    return b - ax;
}

// one coarse cell's right-hand side: its fine cells' residuals in the list's (ascending fine index) order
template <typename T>
__device__ __forceinline__
T cellRestrict(const label* cellStart, const label* cellList, const T* r, int ci)
{
    T s = T(0);
    for (label k = cellStart[ci]; k < cellStart[ci+1]; ++k) s += r[cellList[k]];
    return s;
}

// injection correction
template <typename T>
__device__ __forceinline__
T cellProlong(T x, T xc)
{
    return x + xc;
}

// FP32 SpMV of cell c over the face arrays: the diagonal, the owner faces in ownerStart order, then the neighbour
// faces in losort order
__device__ __forceinline__
float cellSpmvFaceF(
    int c,
    const float* diag, const float* upper, const float* lower,
    const label* nei, const label* owner, const label* ownerStart,
    const label* losort, const label* losortStart,
    const float* psi)
{
    float s = diag[c]*psi[c];
    for (int f=ownerStart[c]; f<ownerStart[c+1]; ++f)
        s += upper[f]*psi[nei[f]];
    for (int k=losortStart[c]; k<losortStart[c+1]; ++k)
    {
        const int f=losort[k];
        s += lower[f]*psi[owner[f]];
    }
    return s;
}

// ...and over its contiguous row (AMGData::csrRow/csrCol/csrVal): the same terms in the same sequence
__device__ __forceinline__
float cellSpmvCsrF(
    int c,
    const float* diag, const label* row, const label* col, const float* val,
    const float* psi)
{
    float s = diag[c]*psi[c];
    for (label i = row[c]; i < row[c+1]; ++i)
        s += val[i]*psi[col[i]];
    return s;
}

// The coarsest solve's A^-1 r as a product, row i: each lane takes every 32nd column of the row, then a fixed
// shuffle tree. The whole warp calls it (full-mask shuffles) with the same i.
__device__ __forceinline__
scalar inverseRowDot(const scalar* inv, const scalar* rc, int i, int n, int lane)
{
    scalar s = 0.0;
    for (int j = lane; j < n; j += 32) s += inv[i*n + j]*rc[j];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    return s;
}

} // namespace brae
