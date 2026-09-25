#pragma once
// ---------------------------------------------------------------------------------------------------------------
// device_amg_internal.cuh -- AMG-core INTERNAL shared infrastructure.
//
// The build / Galerkin / V-cycle / smoothers / PCG-driver code in device_amg.cu is a cohesive core that shares:
//   * the FP32 mixed-precision LDU view (LduF) + its builder (lduF) + the FP32<->FP64 cast kernel (cast_),
//   * the multicolor greedy coloring (Coloring / greedyColor) that feeds the Gauss-Seidel smoother,
//   * the one multicolor GS sweep (gsSweep), used by BOTH the V-cycle's GS smoother and the standalone solver,
//   * the residual-normalisation kernel (gsScaleInvK), used by the GS solver AND the AMG-PCG drivers.
// These are promoted here (verbatim, no logic change) so the eventual split into amg_{smoothers,vcycle,pcg}.cu is
// mechanical: each TU includes this instead of duplicating the FP32 stack or the coloring. Today device_amg.cu is
// still one TU and simply sources them from here. NOT part of the public device_amg.cuh.
// ---------------------------------------------------------------------------------------------------------------
#include "cf_types.cuh"       // scalar / label
#include "device_ldu.cuh"     // DeviceLduView
#include "device_buffer.cuh"  // DeviceBuffer
#include "device_amg_detail.cuh"  // nBlocks / TPB (for the inline FP32 matvec launch)
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace brae {

// ---- multicolor greedy coloring (graph-fixed per mesh; feeds the multicolor Gauss-Seidel smoother) ----
// Coloring struct lives here so the coloring builder and every GS smoother share one definition. greedyColor keeps
// its definition in device_amg.cu (external linkage) and is declared ONCE here for its three callers: device_amg.cu,
// device_amg_gauss_seidel.cu and the colour-ordered smoother in device_colour_gauss_seidel.cu. The local prototype
// device_amg_gauss_seidel.cu still carries is a duplicate of this one, which is legal.
struct Coloring
{
    int nColors = 0;
    std::vector<label> cells, start;
};

// Each cell takes the smallest colour unused by any of its face-neighbours (owner/nei are the internal faces, both
// indexing [0, nC) unchecked); the cells come back grouped by colour as a CSR, start[k] .. start[k+1] being colour k.
Coloring greedyColor(
    const std::vector<label>& owner,
    const std::vector<label>& nei,
    int nC);

// ---- V-cycle smoother interface (definitions in device_amg_smoothers.cu). The V-cycle (vcycleAt) and the PCG
//      drivers (ensureSpectrum) call these across TUs; the standalone symGaussSeidel solver also reuses gsSweep. ----
struct AMGData;   // fwd (full definition in the public device_amg.cuh)
void ensureSpectrum(AMGData& amg, const DeviceLduView& A);
void gsSweep(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& x,
             const GridColoring& gc, bool forward);
void chebyshevSmooth(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& x,
                     DeviceBuffer<scalar>& d, DeviceBuffer<scalar>& ax, scalar lambdaMax, int deg);
void twoStageGSSmooth(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& x,
                      DeviceBuffer<scalar>& z, DeviceBuffer<scalar>& r, int nSweeps, int order, bool forward);

// ---- FP32 mixed-precision LDU: shared FP64 topology + FP32 value arrays (used by the FP32 V-cycle, the FP32 GS
//      solver, and the mixed-precision PCG drivers) ----
struct LduF
{
    const float* diag;
    const float* upper;
    const float* lower;
    const label* nei;
    const label* owner;
    const label* ownerStart;
    const label* losort;
    const label* losortStart;
    int nCells;
    int nInternalFaces;
};
inline LduF lduF(const DeviceLduView& t, const DeviceBuffer<float>& d,
                 const DeviceBuffer<float>& u, const DeviceBuffer<float>& l)
{
    return { d.data(), u.data(), l.data(), t.nei, t.owner, t.ownerStart, t.losort, t.losortStart,
             t.nCells, t.nInternalFaces };
}
// cast_<scalar,float> = FP64->FP32 (down), cast_<float,scalar> = FP32->FP64 (up).
template <class S, class D>
__device__
void cast_(int n, const S* __restrict__ s, D* __restrict__ d)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) d[i] = (D)s[i];
}
template <class S, class D>
inline void castLaunch(int n, const S* s, D* d)
{
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { cast_<S,D>(n, s, d); });
}
template <class S, class D>
inline void castLaunch(int n, const S* s, D* d, cudaStream_t stream)
{
    pcudaParallelFor(nBlocks(n), TPB, size_t(0), stream, [=] __device__ () { cast_<S,D>(n, s, d); });
}
// FP32 SpMV: Apsi = A psi (shared FP64 topology + FP32 values). Used by the FP32 V-cycle and the FP32 GS solver.
static __device__
void amulFK(int nC, const float* __restrict__ diag, const float* __restrict__ upper, const float* __restrict__ lower,
           const label* __restrict__ nei, const label* __restrict__ owner, const label* __restrict__ ownerStart,
           const label* __restrict__ losort, const label* __restrict__ losortStart,
           const float* __restrict__ psi, float* __restrict__ Apsi)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    float s = diag[c]*psi[c];
    for (int f=ownerStart[c]; f<ownerStart[c+1]; ++f)
        s += upper[f]*psi[nei[f]];
    for (int k=losortStart[c]; k<losortStart[c+1]; ++k)
    {
        const int f=losort[k];
        s += lower[f]*psi[owner[f]];
    }
    Apsi[c]=s;
}
// FP-12 MEASURED THE LAUNCH SHAPE AND IT IS NOT THE CONSTRAINT. This SpMV is 451 launches and 2.60 of
// the pressure phase's 6.57 GPU ms on gasMixing/injectorPipe (74,650 cells, 11 levels, ~15 V-cycles per
// iteration), and it does not get cheaper as the levels shrink: 10.05 us at the finest grid, 5.4 on a
// 2,300-cell level, 4.2 below 1,000 cells -- where zeroT and residualT on the SAME grids take 0.8. At
// 256 threads a 1,000-cell level is four blocks, so sizing the block to the level (32/64/128 by cell
// count, bit-identical: same thread-per-cell arithmetic, same sum order) was tried and measured:
// 2.602 -> 2.533 ms on injectorPipe and nothing at all on squareBend (pEqn 8.6 -> 8.6-8.9, p solve 5.7
// -> 5.7-5.8). Reverted. The cost is the per-kernel floor times 121 kernels per V-cycle, and the lever
// that addresses that is fusing the coarse hierarchy into one kernel, the way device_dilu.cu walks its
// levels in a single block.
// FP-12: the SAME SpMV over a contiguous row. Each row holds the cell's owner faces in ownerStart order
// followed by its neighbour faces in losort order, so the sum is the one amulFK computes, term for term
// and in the same sequence -- identical bits. The point is the inner loop: one sequential read of
// (val, col) where the face form takes two indirections (nei[f] or losort[k] then owner[f]) into arrays
// that are not ordered for coalescing. Measured motive: on gasMixing/injectorPipe the face-form SpMV
// costs 4.2 us on a level under 1,000 cells where an elementwise kernel on the same grid costs 0.76.
static __global__
void amulCsrFK(int nC, const float* __restrict__ diag, const label* __restrict__ row,
               const label* __restrict__ col, const float* __restrict__ val,
               const float* __restrict__ psi, float* __restrict__ Apsi)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    float s = diag[c]*psi[c];
    for (label i = row[c]; i < row[c+1]; ++i)
        s += val[i]*psi[col[i]];
    Apsi[c]=s;
}
// Refill a grid's CSR values from its FP64 face arrays. Replaces the two cast_ launches that grid's
// upper and lower would otherwise need, so a CSR grid costs one launch per solve, not two.
static __global__
void csrGatherValsK(int nnz, const label* __restrict__ src, const scalar* __restrict__ upper,
                    const scalar* __restrict__ lower, float* __restrict__ val)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    const label s = src[i];
    val[i] = (s >= 0) ? (float)upper[s] : (float)lower[-s - 1];
}
inline void amulF(const LduF& A, const float* x, float* y)
{
    const int nC = A.nCells;
    const float* diag = A.diag; const float* upper = A.upper; const float* lower = A.lower;
    const label* nei = A.nei; const label* owner = A.owner; const label* ownerStart = A.ownerStart;
    const label* losort = A.losort; const label* losortStart = A.losortStart;
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
        amulFK(nC, diag, upper, lower, nei, owner, ownerStart, losort, losortStart, x, y); });
}
// finalRes = sumMag(r) / normFactor -- residual normalisation on-device; shared by the GS solver + AMG-PCG drivers.
static __device__
void gsScaleInvK(scalar* a, const scalar* b)
{
    if (threadIdx.x==0 && blockIdx.x==0) *a = (*a) / (*b);
}
inline void gsScaleInvLaunch(scalar* a, const scalar* b, cudaStream_t stream)
{
    pcudaParallelFor(dim3(1), dim3(1), size_t(0), stream, [=] __device__ () { gsScaleInvK(a, b); });
}

} // namespace brae
