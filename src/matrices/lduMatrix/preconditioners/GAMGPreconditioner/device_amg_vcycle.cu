// AMG V-cycle: the recursive FP64 V-cycle (vcycleAt) + its mixed-precision FP32 mirror (amgCastFP32/vcycleAtF)
// + the public entry points (amgVCycleApply / amgPrepareFP32). Each level: pre-smooth, restrict residual, recurse,
// prolong+scale the correction, post-smooth; the coarsest level dispatches to the coarse solvers. Smoothers
// (ensureSpectrum/chebyshevSmooth/twoStageGSSmooth/gsSweep) + coarse solvers are called across TUs via the AMG
// headers. Verbatim split of device_amg.cu -- no logic change. The PCG drivers (device_amg.cu) call vcycleAt/
// vcycleAtF/amgCastFP32 (external linkage) via local decls there.
#include "device_amg.cuh"          // AMGData / AMGLevel / GridColoring / DeviceLduView / DeviceSolverPerf
#include "device_amg_detail.cuh"   // constants (OMEGA/NPRE/NPOST/CHEB_*/SB_*/NCOARSE_CG) + env flags + nBlocks/TPB
#include "device_amg_internal.cuh" // LduF/lduF/cast_/amulF, gsSweep + smoother decls (ensureSpectrum/cheb/tsGS)
#include "device_amg_coarse.cuh"   // deviceCoarsePCG / deviceCoarseJacobiSingleBlock
#include "amg_kernels.cuh"         // zeroT/smoothT/residualT/restrictT/prolongT<T>
#include "device_ldu.cuh"          // DeviceLduView / deviceAmul
#include "device_blas.cuh"         // deviceCopy / deviceDot / deviceReciprocalV
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace brae {

namespace {
__global__
void prolongToK(
    int nF,
    const label* __restrict__ map,
    const scalar* __restrict__ xc,
    scalar* __restrict__ pc)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < nF) pc[c] = xc[map[c]];   // prolong INTO pc (not added yet)
}
// Smoothed-aggregation sparse prolongator apply (BRAE_AMG_SA): restrict = P^T, prolong = P.
// restrict: rc += P^T r  (each fine cell f scatters val*r[f] into its coarse columns, atomically).
__global__
void restrictSparseK(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const scalar* __restrict__ val,
    const scalar* __restrict__ r,
    scalar* __restrict__ rc)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    const scalar rf = r[f];
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        atomicAdd(&rc[col[k]], val[k]*rf);
}
// prolong (ADD): x[f] += sum_k P[f][k]*xc[col], the smoothed-P twin of prolongT.
__global__
void prolongSparseK(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const scalar* __restrict__ val,
    const scalar* __restrict__ xc,
    scalar* __restrict__ x)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    scalar s = 0.0;
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        s += val[k]*xc[col[k]];
    x[f] += s;
}
// prolong (SET): pc[f] = sum_k P[f][k]*xc[col], the corrScaling line-search path (writes pc, not added yet).
__global__
void prolongToSparseK(
    int nF,
    const label* __restrict__ rowPtr,
    const label* __restrict__ col,
    const scalar* __restrict__ val,
    const scalar* __restrict__ xc,
    scalar* __restrict__ pc)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nF) return;
    scalar s = 0.0;
    for (int k = rowPtr[f]; k < rowPtr[f+1]; ++k)
        s += val[k]*xc[col[k]];
    pc[f] = s;
}

// Energy-minimising coarse-correction scale (OF GAMG): alpha = (r . c)/(c . Ac), guarded since c.Ac = ||c||_A^2 >= 0.
__global__
void scaleFactorK(
    const scalar* __restrict__ num,
    const scalar* __restrict__ den,
    scalar* __restrict__ out)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar d = *den;
        *out = (d > 1e-30) ? (*num)/d : 1.0;
    }
}
} // anon

// Recursive V-cycle at grid g: x_g <- M^-1 b_g (x_g overwritten). g==nLevels is the coarsest grid (an approximate
// solve; single-block or fused-cluster when small); above it: pre-smooth, restrict the residual to g+1, recurse,
// prolong the correction, post-smooth.
void vcycleAt(
    int g,
    AMGData& amg,
    const DeviceLduView& Ag,
    const DeviceBuffer<scalar>& bg,
    DeviceBuffer<scalar>& xg)
{
    const int n = Ag.nCells;
    zeroT<scalar><<<nBlocks(n),TPB>>>(n, xg.data());
    if (g == amg.nLevels())                                    // coarsest: approximate solve
    {
        // BRAE_NCOARSE_CG overrides the coarsest PCG iteration count.
        static const int ncoarseCG = [](){ const char* e = std::getenv("BRAE_NCOARSE_CG"); return (e && std::atoi(e) > 0) ? std::atoi(e) : NCOARSE_CG; }();
        if (n <= SB_CG_MAX) deviceCoarsePCG(Ag, bg, xg, ncoarseCG);            // tiny coarsest: single-block Jacobi-PCG (cheap+accurate)
        else if (n <= SB_MAX) deviceCoarseJacobiSingleBlock(Ag, bg, xg, NCOARSE);   // larger: single-block many-sweep Jacobi
        else if (deviceCoarseFitsCluster(n) && n <= COARSE_FUSE_MAX) deviceCoarseJacobiFused(Ag, bg, xg, NCOARSE);
        else for (int s = 0; s < NCOARSE; ++s)
        {
            deviceAmul(Ag, xg, amg.vAx[g]);
            smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
        }
        return;
    }
    if (useChebyshev()) chebyshevSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], amg.lambdaMax[g], chebDeg());  // pre-smooth (x=0)
    else if (useTSGS()) twoStageGSSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], NPRE, tsgsOrder(), true);     // OF v2606 twoStageGaussSeidel (fwd)
    else if (amg.gsSmooth) for (int s = 0; s < NPRE; ++s) gsSweep(Ag, bg, xg, amg.coloring[g], true);    // forward GS
    else for (int s = 0; s < NPRE; ++s)
    {
        deviceAmul(Ag, xg, amg.vAx[g]);
        smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
    }
    deviceAmul(Ag, xg, amg.vAx[g]);
    residualT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), amg.vR[g].data());
    const int nc = amg.level[g].nCoarse;
    const AMGLevel& Lg = amg.level[g];
    if (amg.saSmooth)                                            // restrict rc = P^T r (sparse smoothed prolongator)
    {
        // Sparse-prolongator restriction still scatters; the SA path is opt-in and remains nondeterministic.
        zeroT<scalar><<<nBlocks(nc),TPB>>>(nc, amg.vB[g+1].data());
        restrictSparseK<<<nBlocks(n),TPB>>>(n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vR[g].data(), amg.vB[g+1].data());
    }
    else                                          // fixed-order gather; writes rc, so no pre-zero needed
        restrictGatherT<scalar><<<nBlocks(nc),TPB>>>(
            nc, Lg.galCellStart.data(), Lg.galCellList.data(), amg.vR[g].data(), amg.vB[g+1].data());
    vcycleAt(g+1, amg, amg.level[g].coarseView(), amg.vB[g+1], amg.vX[g+1]);   // recurse to the next coarser grid
    if (amg.corrScaling)
    {
        // OF-GAMG-style scaled coarse correction: alpha = (r . A pc)/(A pc . A pc); xg += alpha*pc. The single
        // line-search per level fixes the magnitude of the prolongation. All scalars stay on the device
        // (deviceDotInto/scaleFactorK/AxpyDev) so it adds NO host sync and remains capturable into the V-cycle graph.
        if (amg.saSmooth) prolongToSparseK<<<nBlocks(n),TPB>>>(n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vX[g+1].data(), amg.vPc[g].data());
        else              prolongToK<<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vX[g+1].data(), amg.vPc[g].data());  // c = P*corr
        deviceAmul(Ag, amg.vPc[g], amg.vAx[g]);                                       // Ac
        deviceDotInto(amg.vR[g], amg.vPc[g], amg.sScNum.data());                      // r . c   (energy-min, OF GAMG)
        deviceDotInto(amg.vPc[g], amg.vAx[g], amg.sScDen.data());                     // c . Ac  (= ||c||_A^2 > 0)
        scaleFactorK<<<1,1>>>(amg.sScNum.data(), amg.sScDen.data(), amg.sScAlpha.data());
        deviceAxpyDev(amg.sScAlpha.data(), amg.vPc[g], xg);                           // xg += alpha * c
    }
    else if (amg.saSmooth)
        prolongSparseK<<<nBlocks(n),TPB>>>(n, Lg.Prow.data(), Lg.Pcol.data(), Lg.Pval.data(), amg.vX[g+1].data(), xg.data());
    else
        prolongT<scalar><<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vX[g+1].data(), xg.data());
    if (useChebyshev()) chebyshevSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], amg.lambdaMax[g], chebDeg());  // post-smooth
    else if (useTSGS()) twoStageGSSmooth(Ag, bg, xg, amg.vD[g], amg.vAx[g], NPOST, tsgsOrder(), false);  // twoStageGaussSeidel (bwd -> symmetric)
    else if (amg.gsSmooth) for (int s = 0; s < NPOST; ++s) gsSweep(Ag, bg, xg, amg.coloring[g], false);  // backward GS (symmetric V-cycle)
    else for (int s = 0; s < NPOST; ++s)
    {
        deviceAmul(Ag, xg, amg.vAx[g]);
        smoothT<scalar><<<nBlocks(n),TPB>>>(n, bg.data(), amg.vAx[g].data(), Ag.diag, xg.data());
    }
}

// Mixed-precision (FP32) V-cycle: a mirror of the default V-cycle (weighted-Jacobi + map restrict/prolong) with the
// matrix values and work vectors in FP32 (half the bytes on the BW-bound SpMV+smooth); the topology (labels) is
// shared with the FP64 path. The coarsest level casts back to FP64 and reuses the exact FP64 coarse solve. Only the
// default smoother/aggregation is supported here (SA/GS/Chebyshev/corrScaling stay FP64; the caller gates on those).
// It reuses the templated zeroT/smoothT/residualT/restrictT/prolongT<float>; only the casts and the FP32 SpMV (amulF)
// are FP32-specific.
// (amulFK/amulF -- the FP32 SpMV -- now live in device_amg_internal.cuh, shared with the FP32 GS solver.)
// Cast the (current) FP64 fine + coarse matrices to their FP32 mirrors. Allocates the mirrors + FP32 work vectors once.
void amgCastFP32(
    AMGData& amg,
    const DeviceLduView& A)
{
    const int G = amg.nLevels();
    if (!amg.fp32Alloc)
    {
        amg.fDiag.resize(G+1);
        amg.fUpper.resize(G+1);
        amg.fLower.resize(G+1);
        amg.vAxF.resize(G+1);
        amg.vRF.resize(G+1);
        amg.vXF.resize(G+1);
        amg.vBF.resize(G+1);
        for (int g=0; g<=G; ++g)
        {
            const DeviceLduView v = (g==0) ? A : amg.level[g-1].coarseView();
            amg.fDiag[g].resize(v.nCells);
            amg.fUpper[g].resize(v.nInternalFaces);
            amg.fLower[g].resize(v.nInternalFaces);
            amg.vAxF[g].resize(v.nCells);
            amg.vRF[g].resize(v.nCells);
            amg.vXF[g].resize(v.nCells);
            amg.vBF[g].resize(v.nCells);
        }
        amg.fp32Alloc = true;
    }
    for (int g=0; g<=G; ++g)
    {
        const DeviceLduView v = (g==0) ? A : amg.level[g-1].coarseView();
        cast_<scalar,float><<<nBlocks(v.nCells),TPB>>>(v.nCells, v.diag, amg.fDiag[g].data());
        if (v.nInternalFaces>0)
        {
            cast_<scalar,float><<<nBlocks(v.nInternalFaces),TPB>>>(v.nInternalFaces, v.upper, amg.fUpper[g].data());
            cast_<scalar,float><<<nBlocks(v.nInternalFaces),TPB>>>(v.nInternalFaces, v.lower, amg.fLower[g].data());
        }
    }
}
// FP32 recursive V-cycle. topoG = grid g's FP64 view (topology + the FP64 coarsest solve); Ag = grid g's FP32 matrix.
void vcycleAtF(
    int g,
    AMGData& amg,
    const DeviceLduView& topoG,
    const LduF& Ag,
    const float* bg,
    float* xg)
{
    const int n = Ag.nCells;
    zeroT<float><<<nBlocks(n),TPB>>>(n, xg);
    if (g == amg.nLevels())                                    // coarsest: cast to FP64, exact FP64 solve, cast back
    {
        cast_<float,scalar><<<nBlocks(n),TPB>>>(n, bg, amg.vB[g].data());
        static const int ncoarseCG = [](){ const char* e=std::getenv("BRAE_NCOARSE_CG"); return (e&&std::atoi(e)>0)?std::atoi(e):NCOARSE_CG; }();
        if (n <= SB_CG_MAX) deviceCoarsePCG(topoG, amg.vB[g], amg.vX[g], ncoarseCG);
        else
        {
            zeroT<scalar><<<nBlocks(n),TPB>>>(n, amg.vX[g].data());
            for (int s=0;s<NCOARSE;++s)
            {
                deviceAmul(topoG, amg.vX[g], amg.vAx[g]);
                smoothT<scalar><<<nBlocks(n),TPB>>>(n, amg.vB[g].data(), amg.vAx[g].data(), topoG.diag, amg.vX[g].data());
            }
        }
        cast_<scalar,float><<<nBlocks(n),TPB>>>(n, amg.vX[g].data(), xg);
        return;
    }
    for (int s=0; s<NPRE; ++s)
    {
        amulF(Ag, xg, amg.vAxF[g].data());
        smoothT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), Ag.diag, xg);
    }
    amulF(Ag, xg, amg.vAxF[g].data());
    residualT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), amg.vRF[g].data());
    const AMGLevel& Lg = amg.level[g];
    const int nc = Lg.nCoarse;
    restrictGatherT<float><<<nBlocks(nc),TPB>>>(
        nc, Lg.galCellStart.data(), Lg.galCellList.data(), amg.vRF[g].data(), amg.vBF[g+1].data());
    const DeviceLduView topoC = Lg.coarseView();
    const LduF Ac = lduF(topoC, amg.fDiag[g+1], amg.fUpper[g+1], amg.fLower[g+1]);
    vcycleAtF(g+1, amg, topoC, Ac, amg.vBF[g+1].data(), amg.vXF[g+1].data());
    prolongT<float><<<nBlocks(n),TPB>>>(n, Lg.map.data(), amg.vXF[g+1].data(), xg);
    for (int s=0; s<NPOST; ++s)
    {
        amulF(Ag, xg, amg.vAxF[g].data());
        smoothT<float><<<nBlocks(n),TPB>>>(n, bg, amg.vAxF[g].data(), Ag.diag, xg);
    }
}

// amgVCycleApply runs the FP32 V-cycle automatically -- the same mixed-precision path the single-GPU deviceAMGPCG
// uses, now available to the distributed Krylov (the pressure V-cycle's SpMV+smoother are bandwidth-bound, so
// FP32 halves their bytes ~= 2x, while the r->FP32/FP32->z casts and the FP64 coarsest solve keep it accurate).
void amgPrepareFP32(AMGData& amg, const DeviceLduView& A)
{
    if (useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev())
        amgCastFP32(amg, A);
}

// z = M^-1 r : ONE symmetric AMG V-cycle applied as a PRECONDITIONER, factored out of deviceAMGPCG so the
// DISTRIBUTED Krylov (device_pcg.cu) can precondition each rank's LOCAL block with AMG. The V-cycle is built on
// internal faces only, so it omits the processor-interface coupling -- exactly the block-Jacobi / additive-Schwarz
// design: the outer distributed matvec (deviceParallelAmul) supplies the interface, the local V-cycle need only
// approximate the local block. amg must be built (buildAMG) and current (amgGalerkin gives it this step's coarse
// operators). Runs the FP32 V-cycle when amgPrepareFP32 cast the matrices this solve, else the FP64 one.
void amgVCycleApply(AMGData& amg, const DeviceLduView& A,
                    const DeviceBuffer<scalar>& r, DeviceBuffer<scalar>& z, bool captureVcycle)
{
    ensureSpectrum(amg, A);                // one-time Chebyshev spectrum estimate (no-op unless the Chebyshev smoother is on)
    const bool fp32 = amg.fp32Alloc && useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev();
    const int nC = A.nCells;

    if (!captureVcycle)
    {
        // DIRECT launch (default): every V-cycle kernel is a fresh launch.
        if (fp32)
        {
            const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);         // grid-0 FP32 matrix view
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, r.data(), amg.vBF[0].data());   // r -> FP32
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());              // FP32 V-cycle
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), z.data());   // FP32 -> z (FP64)
        }
        else vcycleAt(0, amg, A, r, z);    // FP64 V-cycle
        return;
    }

    // GRAPH REPLAY: the V-cycle is host-scalar-free and runs on FIXED buffers (amg.rA/wA), so capture it once
    // (keyed on A.diag; the coarse VALUES change each step but the graph references the buffers, read at replay) and
    // replay -- removing the launch overhead of the V-cycle's many small kernels (13 levels x SpMV+smoother per PCG
    // iter, the dominant per-iteration launch cost). Copy r -> persistent amg.rA, replay, amg.wA -> z. Same
    // capture/replay the single-GPU deviceAMGPCG uses; now available to the distributed pressure solve.
    deviceCopy(amg.rA, r);
    if (fp32)
    {
        const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);
        AMGGraphCache& gcf = *amg.gcacheF;
        if (!gcf.exec || gcf.key != A.diag)
        {
            if (gcf.exec)  { cudaGraphExecDestroy(gcf.exec);  gcf.exec  = nullptr; }
            if (gcf.graph) { cudaGraphDestroy(gcf.graph);     gcf.graph = nullptr; }
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amgF capture begin");
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, amg.rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), amg.wA.data());
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gcf.graph), "amgF capture end");
            cudaCheck(cudaGraphInstantiate(&gcf.exec, gcf.graph, 0), "amgF graph instantiate");
            gcf.key = A.diag;
        }
        cudaCheck(cudaGraphLaunch(gcf.exec, cudaStreamPerThread), "amgF graph launch");
    }
    else
    {
        AMGGraphCache& gc = *amg.gcache;
        if (!gc.exec || gc.key != A.diag)
        {
            if (gc.exec)  { cudaGraphExecDestroy(gc.exec);  gc.exec  = nullptr; }
            if (gc.graph) { cudaGraphDestroy(gc.graph);     gc.graph = nullptr; }
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amg capture begin");
            vcycleAt(0, amg, A, amg.rA, amg.wA);
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gc.graph), "amg capture end");
            cudaCheck(cudaGraphInstantiate(&gc.exec, gc.graph, 0), "amg graph instantiate");
            gc.key = A.diag;
        }
        cudaCheck(cudaGraphLaunch(gc.exec, cudaStreamPerThread), "amg graph launch");
    }
    deviceCopy(z, amg.wA);
}

} // namespace brae
