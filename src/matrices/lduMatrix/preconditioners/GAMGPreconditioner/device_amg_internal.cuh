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
#include "amg_cell_ops.cuh"       // the cell expressions the FP32 SpMV kernels and the fused coarse kernel share
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
// The double cycle's row layout on its small grids (AMGData::csr64Row): built at the first call, its values gathered at every
// call. amgGalerkin's last act: the one point where the coarse matrices change and outside every captured cycle.
void amgGatherCsr64(AMGData& amg);
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
__global__
void cast_(int n, const S* __restrict__ s, D* __restrict__ d)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) d[i] = (D)s[i];
}
// FP32 SpMV: Apsi = A psi (shared FP64 topology + FP32 values). Used by the FP32 V-cycle and the FP32 GS solver.
static __global__
void amulFK(int nC, const float* __restrict__ diag, const float* __restrict__ upper, const float* __restrict__ lower,
           const label* __restrict__ nei, const label* __restrict__ owner, const label* __restrict__ ownerStart,
           const label* __restrict__ losort, const label* __restrict__ losortStart,
           const float* __restrict__ psi, float* __restrict__ Apsi)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    Apsi[c] = cellSpmvFaceF(c, diag, upper, lower, nei, owner, ownerStart, losort, losortStart, psi);
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
    Apsi[c] = cellSpmvCsrF(c, diag, row, col, val, psi);
}
// The double cycle's product on a grid with the row layout (AMGData::csr64Row): one contiguous row, where amulKernel merges two
// lists of faces through three chained reads an entry. Nsight Compute, one launch at 1 to 197 blocks (motorBike, 16 GH200s):
// amulKernel 13.5 to 19 us, amulCsrFK in single precision 5.9 to 9 us. The same sum as amulKernel, in its order, and the same
// rounding: its multiply and add stay apart (the if-converted merge compiles to DMUL then DADD), where a plain s += a*b here compiled
// to DFMA, one rounding fewer: 1e-14 from amulKernel's answers within 18 CG iterations on pitzDaily's pressure (tests/mgpu_sequence.sh).
static __global__
void amulCsrDK(int nC, const scalar* __restrict__ diag, const label* __restrict__ row,
               const label* __restrict__ col, const scalar* __restrict__ val,
               const scalar* __restrict__ psi, scalar* __restrict__ Apsi)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = diag[c]*psi[c];
    for (label i = row[c]; i < row[c+1]; ++i)
        s = __dmul_rn(val[i], psi[col[i]]) + s;
    Apsi[c]=s;
}
// Refill a double row layout's values from the grid's face arrays
static __global__
void csr64GatherK(int nnz, const label* __restrict__ src, const scalar* __restrict__ upper,
                  const scalar* __restrict__ lower, scalar* __restrict__ val)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    const label s = src[i];
    val[i] = (s >= 0) ? upper[s] : lower[-s - 1];
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
// ...and the same values through the grid's own addressing, so that the stored source list is not needed (csrSrc,
// AMGData). Row c is its owner faces in ownerStart order and then its neighbour faces in losort order (the layout
// buildCsrForGrid gives it), so upper[f] sits at row[owner[f]] + (f - ownerStart[owner[f]]) and the lower of the
// face at losort slot k at row[c] + (its owner count) + (k - losortStart[c]), c = nei[losort[k]]: one thread a
// slot, threads [0, nF) the upper values in face order and [nF, 2 nF) the lower ones in losort order. The same
// casts of the same values into the same slots. ONE THREAD A ROW, walking it, was measured first and cost the
// stored list's speed: on boxy5 at 4.94 M cells (H100, 200 iterations) 0.4088 and 0.4091 s/it against 0.4031 --
// its loads and stores stride a row apart across a warp, where these run along the face arrays.
static __global__
void csrGatherFacesK(
    int nF,
    const label* __restrict__ row,
    const label* __restrict__ owner,
    const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    float* __restrict__ val)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 2*nF) return;
    if (i < nF)
    {
        const label c = owner[i];
        val[row[c] + (i - ownerStart[c])] = (float)upper[i];
        return;
    }
    const label k = i - nF;
    const label f = losort[k];
    const label c = nei[f];
    val[row[c] + (ownerStart[c+1] - ownerStart[c]) + (k - losortStart[c])] = (float)lower[f];
}
inline void amulF(const LduF& A, const float* x, float* y)
{
    amulFK<<<nBlocks(A.nCells),TPB>>>(A.nCells, A.diag, A.upper, A.lower, A.nei, A.owner,
                                      A.ownerStart, A.losort, A.losortStart, x, y);
}
// finalRes = sumMag(r) / normFactor -- residual normalisation on-device; shared by the GS solver + AMG-PCG drivers.
static __global__
void gsScaleInvK(scalar* a, const scalar* b)
{
    if (threadIdx.x==0 && blockIdx.x==0) *a = (*a) / (*b);
}

} // namespace brae
