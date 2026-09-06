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
#include <cuda_runtime.h>
#include <vector>

namespace brae {

// ---- multicolor greedy coloring (graph-fixed per mesh; feeds the multicolor Gauss-Seidel smoother) ----
// Coloring struct lives here so the (to-be-split) coloring builder + GS smoother share one definition. greedyColor
// and gsSweep keep their definitions in device_amg.cu for now (only called within it); their declarations move here
// when gsColoringFor/the GS solver split into amg_smoothers.cu.
struct Coloring
{
    int nColors = 0;
    std::vector<label> cells, start;
};

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
