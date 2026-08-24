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
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

// V-cycle / FP32 GS. T(OMEGA) and T(0) reproduce the FP64 and FP32 constants exactly, so each instantiation is
// byte-identical to the hand-written twin it replaces.
template <typename T>
__device__
void zeroT(
    int n,
    T* x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = T(0);
}
template <typename T>
inline void zeroTLaunch(int n, T* x)
{
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { zeroT<T>(n, x); });
}
// weighted-Jacobi update: x += omega*(b - Ax)/diag
template <typename T>
__device__
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
inline void smoothTLaunch(int n, const T* b, const T* Ax, const T* diag, T* x)
{
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { smoothT<T>(n, b, Ax, diag, x); });
}
template <typename T>
__device__
void residualT(
    int n,
    const T* __restrict__ b,
    const T* __restrict__ Ax,
    T* __restrict__ r)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i]-Ax[i];
}
template <typename T>
inline void residualTLaunch(int n, const T* b, const T* Ax, T* r)
{
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { residualT<T>(n, b, Ax, r); });
}
// One color of in-place Gauss-Seidel: cells in [lo,hi) share no face, so the writes never race and the
// off-diagonal reads pick up already-swept colors -> true GS, not Jacobi. Same LDU gather as the SpMV.
template <typename T>
__device__
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
template <typename T>
inline void gsColorTLaunch(int lo, int hi, const label* cells, const T* b, const T* diag,
    const label* ownerStart, const label* nei, const T* upper,
    const label* losortStart, const label* losort, const label* owner, const T* lower, T* x)
{
    const int n = hi - lo;
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
        gsColorT<T>(lo, hi, cells, b, diag, ownerStart, nei, upper, losortStart, losort, owner, lower, x); });
}

template <typename T>
__device__
void restrictT(
    int nF,
    const label* __restrict__ map,
    const T* __restrict__ r,
    T* __restrict__ rc)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) atomicAdd(&rc[map[c]], r[c]);
}
template <typename T>
inline void restrictTLaunch(int nF, const label* map, const T* r, T* rc)
{
    pcudaParallelFor(nBlocks(nF), TPB, [=] __device__ () { restrictT<T>(nF, map, r, rc); });
}
template <typename T>
__device__
void prolongT(
    int nF,
    const label* __restrict__ map,
    const T* __restrict__ xc,
    T* __restrict__ x)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) x[c] += xc[map[c]];   // injection correction
}
template <typename T>
inline void prolongTLaunch(int nF, const label* map, const T* xc, T* x)
{
    pcudaParallelFor(nBlocks(nF), TPB, [=] __device__ () { prolongT<T>(nF, map, xc, x); });
}

} // namespace brae
