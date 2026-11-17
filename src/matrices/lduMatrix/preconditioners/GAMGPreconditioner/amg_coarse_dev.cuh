#pragma once
// ---------------------------------------------------------------------------------------------------------------
// amg_coarse_dev.cuh -- the coarsest direct solve's one-warp substitutions, shared by device_amg_coarse.cu (the
// solve's own kernels) and device_amg_fused.cu (the fused coarse-level V-cycle, which runs the same solve inside
// its block): one definition, so both take the same bits.
// ---------------------------------------------------------------------------------------------------------------
#include "cf_types.cuh"

namespace brae {

// rows a lane holds: the warp kernels take n <= 32*TRI_ROWS
constexpr int TRI_ROWS = 4;

// row i lives in lane i % 32, slot i / 32 (a constant index into the slots, so they stay in registers)
__device__ __forceinline__
scalar warpRow(
    const scalar (&yr)[TRI_ROWS],
    int k)
{
    const int s = k >> 5;
    const scalar v = (s == 0) ? yr[0] : (s == 1) ? yr[1] : (s == 2) ? yr[2] : yr[3];
    return __shfl_sync(0xffffffffu, v, k & 31);
}

__device__ __forceinline__
void warpDivideRow(
    scalar (&yr)[TRI_ROWS],
    int k,
    int lane,
    scalar d)
{
    if ((k & 31) != lane) return;
    const int s = k >> 5;
    if (s == 0) yr[0] /= d;
    else if (s == 1) yr[1] /= d;
    else if (s == 2) yr[2] /= d;
    else yr[3] /= d;
}

// coarseLUSolveKernel's two substitutions, row for row and step for step (y_i -= F_ik y_k in k order, the
// backward step's division at its own row first), by one warp: the same arithmetic in the same order is the
// same bits. F is column-major. unitLower: L's diagonal is 1 and not stored (the LU); otherwise the forward
// step divides by F_kk first (Cholesky's L).
__device__ __forceinline__
void warpSubstitute(
    int n,
    const scalar* __restrict__ F,
    bool unitLower,
    scalar (&yr)[TRI_ROWS],
    int lane)
{
    for (int k = 0; k < n; ++k)
    {
        if (!unitLower) warpDivideRow(yr, k, lane, F[k*n + k]);
        const scalar yk = warpRow(yr, k);
        #pragma unroll
        for (int s = 0; s < TRI_ROWS; ++s)
        {
            const int i = lane + 32*s;
            if (i > k && i < n) yr[s] -= F[k*n + i]*yk;
        }
    }
    for (int k = n - 1; k >= 0; --k)
    {
        warpDivideRow(yr, k, lane, F[k*n + k]);
        const scalar yk = warpRow(yr, k);
        #pragma unroll
        for (int s = 0; s < TRI_ROWS; ++s)
        {
            const int i = lane + 32*s;
            if (i < k) yr[s] -= F[k*n + i]*yk;
        }
    }
}

} // namespace brae
