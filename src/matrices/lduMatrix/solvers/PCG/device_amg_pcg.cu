// AMG-PCG SOLVER DRIVERS: the flexible-CG / conditional-graph Krylov loops that drive the AMG V-cycle preconditioner.
//   * deviceAMGPCG       -- the single-GPU AMG-PCG (host loop; dispatches to the device-resident graph when enabled).
//   * deviceAMGPCGGraph  -- the whole-loop conditional-graph PCG (BRAE_PCG_DEVICE): the entire steady-state PCG WHILE
//                           body captured once + replayed on-device (brae's ~1.85x-vs-SPUMA perf signature).
//   * deviceParallelAMGPCGGraph -- the distributed twin (halo-coupled matvec + on-stream NVSHMEM reduce in-graph).
// The V-cycle (amgVCycleApply / vcycleAt / vcycleAtF / amgCastFP32) + Galerkin re-coarsening (amgGalerkin) + the
// spectrum estimate (ensureSpectrum) are called across TUs. Verbatim split of device_amg.cu -- no logic change.
#include "device_amg.cuh"          // AMGData / DeviceSolverPerf / amgGalerkin / amgVCycleApply / deviceAMGPCG decls
#include "device_blas.cuh"          // dot / axpy / reductions
#include "device_ldu.cuh"           // DeviceLduView / deviceAmul / deviceParallelAmul (distributed matvec)
#include "device_halo.cuh"          // DeviceHalo + DeviceReducer (on-stream NVSHMEM reduce) for the distributed PCG
#include "device_amg_detail.cuh"    // BRAE_HAS_GS_DEVICE, gsScaleInvK-adjacent constants + env flags + nBlocks/TPB
#include "amg_kernels.cuh"          // zeroT<> etc. (if referenced)
#include "device_amg_internal.cuh"  // LduF/cast_/amulF, gsScaleInvK, ensureSpectrum decl
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace brae {

// V-cycle entry points (defined in device_amg_vcycle.cu, external linkage) that these PCG drivers call.
void vcycleAt(int g, AMGData& amg, const DeviceLduView& Ag, const DeviceBuffer<scalar>& bg, DeviceBuffer<scalar>& xg);
void vcycleAtF(int g, AMGData& amg, const DeviceLduView& topoG, const LduF& Ag, const float* bg, float* xg);
void amgCastFP32(AMGData& amg, const DeviceLduView& A);

namespace {
// Flexible-CG beta (Polak-Ribiere+): beta = max(0, (z.r_new - z.r_old)/rho_old), guarded division. Robust for the
// nonlinear scaled V-cycle preconditioner; reduces to Fletcher-Reeves when the precond is linear (z.r_old == 0).
__global__
void flexBetaK(
    const scalar* __restrict__ zrNew,
    const scalar* __restrict__ zrOld,
    const scalar* __restrict__ rhoOld,
    scalar* __restrict__ out)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar d=*rhoOld;
        *out = (d>1e-300 || d<-1e-300) ? fmax(0.0, (*zrNew - *zrOld)/d) : 0.0;
    }
}
} // anon

#ifdef BRAE_HAS_GS_DEVICE
// Device-resident AMG-PCG (BRAE_PCG_DEVICE). Extends the on-device stop decision from the k/eps Gauss-Seidel to the
// pressure solve. The V-cycle is already graph-capturable and the Krylov scalars already live on-device; the only
// host-in-the-loop part left is the per-iteration residual read driving {converged? stop}. Here the steady-state PCG
// iteration (V-cycle precond -> dot -> beta -> p-update -> SpMV -> dot -> alpha -> axpy x -> axpy r -> residual ->
// stop-test) is captured once into a conditional WHILE graph and replayed on-device, with a 1-thread pcgSetCondK
// driving cudaGraphSetConditional() from the same per-iter predicate (res<tol || res<relTol*init || iter>=maxIter).
// So the whole pressure solve is one graph launch, zero per-iter D2H, exact (same iteration count and psi as the host
// PCG). Iteration 0 differs (p=w, no beta) so it runs explicitly first (one residual read for the rare converge-in-1
// early-out); the WHILE body is the uniform iter-1+ recurrence. The pressure matrix is stable across SIMPLE steps
// (the V-cycle graph already keys on A.diag), so no matrix copy is needed; the graph references A, psi, amg.rA/wA and
// the cache's persistent pA/Ax directly. Non-corrScaling only (flexible-CG falls back to the host loop); normFactor
// is a device-resident scalar.
__global__
void pcgSetCondK(
    cudaGraphConditionalHandle h,
    const scalar* res,
    scalar tol,
    const scalar* init,
    scalar relTol,
    int* iter,
    int maxIter,
    int minIter)   // OF lduMatrix: keep iterating while nIter < minIter, whatever the residual says
{
    if (threadIdx.x || blockIdx.x) return;
    const int it = ++(*iter);
    const scalar fr = *res;          // res already normalized by normFactor
    const bool conv = (fr < tol) || (relTol > 0.0 && fr < relTol * (*init));
    const bool stop = (conv || it >= maxIter) && it >= minIter;
    cudaGraphSetConditional(h, stop ? 0u : 1u);
}
static DeviceSolverPerf deviceAMGPCGGraph(
    const DeviceLduView& A,
    AMGData& amg,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter)
{
    const int nC = A.nCells;
    amg.corrScaling = false;
    DeviceBuffer<scalar>& wA = amg.wA;
    DeviceBuffer<scalar>& rA = amg.rA;
    if (!amg.pcgCache) amg.pcgCache = std::make_unique<PCGGraphCache>();          // per-solver lifetime: dies with amg, so no stale reuse across solvers
    PCGGraphCache& c = *amg.pcgCache;
    c.pA.resize(nC);
    c.Ax.resize(nC);
    c.sNormF.resize(1);
    c.sInit.resize(1);
    c.sRes.resize(1);
    c.sIter.resize(1);
    cudaMemcpyAsync(c.sNormF.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice, cudaStreamPerThread);
    scalar* dWArA = amg.sWArA.data();
    scalar* dWArAold = amg.sWArAold.data();
    scalar* dPap  = amg.sPap.data();
    scalar* dAlpha   = amg.sAlpha.data();
    scalar* dNegAlpha = amg.sNegAlpha.data();
    scalar* dBeta = amg.sBeta.data();
    ensureSpectrum(amg, A);                                                       // one-time Chebyshev spectrum (pre-capture)
    // FP32 V-cycle inside the device-resident PCG: cast matrices once per solve; the WHILE body captures the FP32
    // vcycleAtF automatically (host-scalar-free). Outer Krylov + residual stay FP64 (accuracy preserved).
    const bool fp32 = useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev();
    if (fp32) amgCastFP32(amg, A);
    const LduF A0 = fp32 ? lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]) : LduF{};
    auto applyPrec = [&]()
    {
        if (fp32)
        {
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), wA.data());
        }
        else vcycleAt(0, amg, A, rA, wA);
    };
    // initial residual r = b - A psi  (also PRE-SIZES the V-cycle scratch + pA/Ax so the capture allocates nothing)
    deviceAmul(A, psi, c.Ax);
    deviceCopy(rA, b);
    deviceAxpy(-1.0, c.Ax, rA);
    DeviceSolverPerf perf;
    deviceSumMagInto(rA, c.sInit.data());
    gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sInit.data(), c.sNormF.data());
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.sInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg init D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    perf.initialResidual = initRes;
    perf.finalResidual = initRes;
    auto convergedHost = [&](scalar fr){ return (fr < tol) || (relTol > 0.0 && fr < relTol*initRes); };
    if (convergedHost(initRes) && minIter <= 0)
    {
        perf.nIterations = 0;
        return perf;
    }
    // iteration 0 (explicit: p = w, no beta)
    applyPrec();                                         // wA = M^-1 rA
    deviceDotInto(wA, rA, dWArA);
    deviceCopy(c.pA, wA);
    deviceAmul(A, c.pA, wA);
    deviceDotInto(wA, c.pA, dPap);
    deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
    deviceAxpyDev(dAlpha, c.pA, psi);
    deviceAxpyDev(dNegAlpha, wA, rA);
    deviceSumMagInto(rA, c.sRes.data());
    gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sRes.data(), c.sNormF.data());
    scalar res1;
    cudaCheck(cudaMemcpyAsync(&res1, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg it0 D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    if ((convergedHost(res1) || maxIter <= 1) && minIter <= 1)
    {
        perf.finalResidual = res1;
        perf.nIterations = 1;
        return perf;
    }
    cudaMemsetAsync(c.sIter.data(), 0, sizeof(int), cudaStreamPerThread);   // WHILE-body counter (0 = iter-1)
    // WHILE body = steady-state iteration 1+ (captured once, replayed on-device)
    if (!c.exec || c.key != psi.data() || c.keyTol != tol || c.keyRelTol != relTol || c.keyMaxIter != maxIter || c.keyMinIter != minIter
        || c.keyEpoch != deviceReductionScratchEpoch())          // the body captures reductions; the scratch may have been regrown
    {
        if (c.exec)
        {
            cudaGraphExecDestroy(c.exec);
            c.exec  = nullptr;
        }
        if (c.graph)
        {
            cudaGraphDestroy(c.graph);
            c.graph = nullptr;
        }
        cudaCheck(cudaGraphCreate(&c.graph, 0), "pcg graph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 1, cudaGraphCondAssignDefault), "pcg cond handle");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(cudaGraphAddNode(&cnode, c.graph, nullptr, nullptr, 0, &cp), "pcg cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "pcg capture begin");
        deviceScalarCopy(dWArA, dWArAold);
        applyPrec();                                     // wA = M^-1 rA
        deviceDotInto(wA, rA, dWArA);
        deviceScalarDiv(dWArA, dWArAold, dBeta);          // Fletcher-Reeves beta
        deviceFusedScaleAxpy(c.pA, dBeta, wA);            // p = beta*p + w  [fused]
        deviceAmul(A, c.pA, wA);
        deviceDotInto(wA, c.pA, dPap);
        deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
        deviceAxpyDev(dAlpha, c.pA, psi);
        deviceAxpyDev(dNegAlpha, wA, rA);
        deviceSumMagInto(rA, c.sRes.data());
        gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sRes.data(), c.sNormF.data());   // normalized residual
        pcgSetCondK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.sRes.data(), tol, c.sInit.data(), relTol, c.sIter.data(), maxIter-1, minIter-1);   // -1: iteration 0 ran outside the loop
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "pcg capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "pcg graph instantiate");
        c.key = psi.data();
        c.keyTol = tol;
        c.keyRelTol = relTol;
        c.keyMaxIter = maxIter;
        c.keyMinIter = minIter;
        c.keyEpoch = deviceReductionScratchEpoch();
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "pcg graph launch");
    scalar finalRes;
    int whileIters;
    cudaCheck(cudaMemcpyAsync(&finalRes, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg final D2H");
    cudaCheck(cudaMemcpyAsync(&whileIters, c.sIter.data(), sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg iters D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    perf.finalResidual = finalRes;
    perf.nIterations = 1 + whileIters;
    static const bool dbgCyc = std::getenv("BRAE_AMG_CYCLES") != nullptr;
    if (dbgCyc) std::fprintf(stderr, "[AMG] p cycles=%d finalRes=%.3e (device)\n", perf.nIterations, perf.finalResidual);
    return perf;
}

// DISTRIBUTED whole-loop graph PCG: the exact structure of deviceAMGPCGGraph (the entire steady-state PCG WHILE
// body captured once into a conditional graph and replayed on-device -- no per-iteration host launches or syncs),
// but the matvec is deviceParallelAmul (halo-coupled) and every dot is globalised by an ON-STREAM NVSHMEM reduce
// through the halo's DeviceReducer, so the halo put/barrier and the collective are captured INSIDE the graph body.
// This is the 2x lever: the ~13ms/iter of GPU work stops being spread across ~hundreds of separate launches. It
// requires the on-stream NVSHMEM ops to be graph-capturable (NVSHMEM 3.6 has cuda_graph paths) and 1 PE/GPU (the
// MPG host-MPI reduce fallback is NOT capturable) -- so it is real-multi-GPU only. Persistent buffers (psi/matrix
// held across steps by the caller) make it capture ONCE per run; otherwise it re-instantiates per step (still
// correct). Falls back to the direct deviceParallelAMGPCG if the caller keeps buffers non-persistent.
DeviceSolverPerf deviceParallelAMGPCGGraph(
    const DeviceLduView& A,
    AMGData& amg,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const cudaStream_t strm = cudaStreamPerThread;
    const int nC = A.nCells;
    amg.corrScaling = false;
    DeviceBuffer<scalar>& wA = amg.wA;
    DeviceBuffer<scalar>& rA = amg.rA;
    if (!amg.pcgCache) amg.pcgCache = std::make_unique<PCGGraphCache>();
    PCGGraphCache& c = *amg.pcgCache;
    c.pA.resize(nC); c.Ax.resize(nC); c.sNormF.resize(1); c.sInit.resize(1); c.sRes.resize(1); c.sIter.resize(1);
    cudaMemcpyAsync(c.sNormF.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice, strm);
    scalar* dWArA = amg.sWArA.data();  scalar* dWArAold = amg.sWArAold.data();
    scalar* dPap  = amg.sPap.data();   scalar* dAlpha = amg.sAlpha.data();
    scalar* dNegAlpha = amg.sNegAlpha.data(); scalar* dBeta = amg.sBeta.data();
    ensureSpectrum(amg, A);
    const bool fp32 = useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev();
    if (fp32) amgCastFP32(amg, A);
    const LduF A0 = fp32 ? lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]) : LduF{};
    auto applyPrec = [&]()
    {
        if (fp32)
        {
            cast_<scalar,float><<<nBlocks(nC),TPB,0,strm>>>(nC, rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB,0,strm>>>(nC, amg.vXF[0].data(), wA.data());
        }
        else vcycleAt(0, amg, A, rA, wA);
    };
    DeviceReducer& R = halo.reducer();
    auto gdot = [&](const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, scalar* dOut)
    {   // *dOut = global sum of x.y (local dot -> on-stream NVSHMEM reduce -> copy)
        deviceDotInto(x, y, R.src()); R.sumReduce(1, strm); deviceScalarCopy(R.dst(), dOut);
    };
    auto gsum = [&](const DeviceBuffer<scalar>& x, scalar* dOut)   // *dOut = global |x|_1
    {
        deviceSumMagInto(x, R.src()); R.sumReduce(1, strm); deviceScalarCopy(R.dst(), dOut);
    };

    deviceParallelAmul(A, halo, ifaceCoeffs, psi, c.Ax);          // r = b - A psi
    deviceCopy(rA, b);
    deviceAxpy(-1.0, c.Ax, rA);
    DeviceSolverPerf perf;
    gsum(rA, c.sInit.data());
    gsScaleInvK<<<1,1,0,strm>>>(c.sInit.data(), c.sNormF.data());
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.sInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "pgraph init D2H");
    cudaStreamSynchronize(strm);
    perf.initialResidual = initRes; perf.finalResidual = initRes;
    auto convergedHost = [&](scalar fr){ return (fr < tol) || (relTol > 0.0 && fr < relTol*initRes); };
    if (convergedHost(initRes)) { perf.nIterations = 0; return perf; }

    applyPrec();                                                  // iteration 0 (explicit)
    gdot(wA, rA, dWArA);
    deviceCopy(c.pA, wA);
    deviceParallelAmul(A, halo, ifaceCoeffs, c.pA, wA);
    gdot(wA, c.pA, dPap);
    deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
    deviceAxpyDev(dAlpha, c.pA, psi);
    deviceAxpyDev(dNegAlpha, wA, rA);
    gsum(rA, c.sRes.data());
    gsScaleInvK<<<1,1,0,strm>>>(c.sRes.data(), c.sNormF.data());
    scalar res1;
    cudaCheck(cudaMemcpyAsync(&res1, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "pgraph it0 D2H");
    cudaStreamSynchronize(strm);
    if (convergedHost(res1) || maxIter <= 1) { perf.finalResidual = res1; perf.nIterations = 1; return perf; }
    cudaMemsetAsync(c.sIter.data(), 0, sizeof(int), strm);

    if (!c.exec || c.key != psi.data() || c.keyTol != tol || c.keyRelTol != relTol || c.keyMaxIter != maxIter
        || c.keyEpoch != deviceReductionScratchEpoch())
    {
        if (c.exec)  { cudaGraphExecDestroy(c.exec);  c.exec  = nullptr; }
        if (c.graph) { cudaGraphDestroy(c.graph);     c.graph = nullptr; }
        cudaCheck(cudaGraphCreate(&c.graph, 0), "pgraph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 1, cudaGraphCondAssignDefault), "pgraph cond handle");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(cudaGraphAddNode(&cnode, c.graph, nullptr, nullptr, 0, &cp), "pgraph cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(strm, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "pgraph capture begin");
        deviceScalarCopy(dWArA, dWArAold);
        applyPrec();
        gdot(wA, rA, dWArA);
        deviceScalarDiv(dWArA, dWArAold, dBeta);
        deviceFusedScaleAxpy(c.pA, dBeta, wA);
        deviceParallelAmul(A, halo, ifaceCoeffs, c.pA, wA);
        gdot(wA, c.pA, dPap);
        deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
        deviceAxpyDev(dAlpha, c.pA, psi);
        deviceAxpyDev(dNegAlpha, wA, rA);
        gsum(rA, c.sRes.data());
        gsScaleInvK<<<1,1,0,strm>>>(c.sRes.data(), c.sNormF.data());
        // minIter 0: the DISTRIBUTED path does not carry the fvSolution minIter yet (the single-GPU
        // solvers do). Passing 0 keeps it exactly as it was rather than half-wiring it.
        pcgSetCondK<<<1,1,0,strm>>>(c.handle, c.sRes.data(), tol, c.sInit.data(), relTol, c.sIter.data(), maxIter-1, 0);
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(strm, &tmp), "pgraph capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "pgraph instantiate");
        c.key = psi.data();
        c.keyTol = tol;
        c.keyRelTol = relTol;
        c.keyMaxIter = maxIter;
        c.keyEpoch = deviceReductionScratchEpoch();
    }
    cudaCheck(cudaGraphLaunch(c.exec, strm), "pgraph launch");
    scalar finalRes; int whileIters;
    cudaCheck(cudaMemcpyAsync(&finalRes, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "pgraph final D2H");
    cudaCheck(cudaMemcpyAsync(&whileIters, c.sIter.data(), sizeof(int), cudaMemcpyDeviceToHost, strm), "pgraph iters D2H");
    cudaStreamSynchronize(strm);
    perf.finalResidual = finalRes;
    perf.nIterations = 1 + whileIters;
    return perf;
}
#endif // BRAE_HAS_GS_DEVICE

#ifndef BRAE_HAS_GS_DEVICE
// CUDA < 13: WHILE-node conditional graphs are unavailable, so the whole-loop graph cannot be built. Return a
// sentinel (nIterations < 0); the caller falls back to the direct (non-graph) distributed AMG-PCG.
DeviceSolverPerf deviceParallelAMGPCGGraph(
    const DeviceLduView& A, AMGData&, DeviceHalo&,
    const std::vector<DeviceBuffer<scalar>>&,
    const DeviceBuffer<scalar>&, DeviceBuffer<scalar>&,
    scalar, scalar, scalar, int)
{
    (void)A;
    DeviceSolverPerf p; p.nIterations = -1; return p;
}
#endif

DeviceSolverPerf deviceAMGPCG(
    const DeviceLduView& A,
    AMGData& amg,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    bool captureVcycle,
    int checkEvery,
    bool corrScaling,
    int minIter)
{
#ifdef BRAE_HAS_GS_DEVICE
    {
        static const bool pcgDev = envFlag("BRAE_PCG_DEVICE", true);                 // device-resident pressure solve (default ON; opt out BRAE_PCG_DEVICE=0)
        if (pcgDev && !corrScaling) return deviceAMGPCGGraph(A, amg, b, psi, normFactor, tol, relTol, maxIter, minIter);
    }
#endif
    const int nC = A.nCells;
    amg.corrScaling = corrScaling;                             // seen by vcycleAt (scaled prolongation)
    DeviceBuffer<scalar>& wA = amg.wA;                          // persistent (fixed addr) so the captured graph stays valid
    DeviceBuffer<scalar>& rA = amg.rA;
    DeviceBuffer<scalar> pA(nC), Ax(nC), rOld(nC);             // rOld: previous residual for flexible-CG beta (corrScaling)
    deviceAmul(A, psi, Ax);
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);
    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA)/normFactor;
    perf.finalResidual = perf.initialResidual;
    auto converged = [&](scalar fr){ return (fr<tol) || (relTol>0.0 && fr<relTol*perf.initialResidual); };

    // One-time Chebyshev spectrum estimate (host scalars -> must precede any graph capture); no-op unless Chebyshev.
    ensureSpectrum(amg, A);

    // The V-cycle (wA = M^-1 rA) has no host-scalar dependencies and runs on fixed buffers (amg.rA/wA + amg's
    // persistent work buffers), so it is captured into a CUDA graph and replayed. The graph is cached in amg.gcache,
    // keyed on the fine-matrix pointer A.diag: captured once and replayed across all PCG iters and SIMPLE steps (the
    // matrix values change in-buffer each step; the graph reads them at replay), re-captured only when the key changes.
    // FP32 mixed precision (default smoother/aggregation only; SA/GS/Cheb/corrScaling stay FP64): the matrices are cast
    // once per solve, and each application casts rA -> FP32 in / FP32 -> wA out.
    const bool fp32 = useFP32() && !amg.saSmooth && !amg.gsSmooth && !useChebyshev() && !corrScaling;
    if (fp32)
    {
        amgCastFP32(amg, A);
        static bool once=false;
        if(!once)
        {
            once=true;
            if(std::getenv("BRAE_AMG_CYCLES")) std::fprintf(stderr,"[AMG] FP32 V-cycle ENGAGED (levels=%d)\n", amg.nLevels());
        }
    }
    AMGGraphCache& gc = *amg.gcache;
    auto applyPrecond = [&]()
    {
        if (fp32)
        {
            const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);
            auto runF = [&]()                                                        // cast in -> FP32 V-cycle -> cast out
            {
                cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, rA.data(), amg.vBF[0].data());
                vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
                cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), wA.data());
            };
            if (!captureVcycle)
            {
                runF();
                return;
            }
            AMGGraphCache& gcf = *amg.gcacheF;                                          // graph the FP32 V-cycle (host-scalar-free)
            if (!gcf.exec || gcf.key != A.diag || gcf.keyEpoch != deviceReductionScratchEpoch())
            {
                if (gcf.exec)
                {
                    cudaGraphExecDestroy(gcf.exec);
                    gcf.exec = nullptr;
                }
                if (gcf.graph)
                {
                    cudaGraphDestroy(gcf.graph);
                    gcf.graph = nullptr;
                }
                cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amgF capture begin");
                runF();
                cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gcf.graph), "amgF capture end");
                cudaCheck(cudaGraphInstantiate(&gcf.exec, gcf.graph, 0), "amgF graph instantiate");
                gcf.key = A.diag; gcf.keyEpoch = deviceReductionScratchEpoch();
            }
            cudaCheck(cudaGraphLaunch(gcf.exec, cudaStreamPerThread), "amgF graph launch");
            return;
        }
        if (!captureVcycle)
        {
            vcycleAt(0, amg, A, rA, wA);
            return;
        }
        if (!gc.exec || gc.key != A.diag || gc.keyEpoch != deviceReductionScratchEpoch())
        {
            if (gc.exec)
            {
                cudaGraphExecDestroy(gc.exec);
                gc.exec = nullptr;
            }
            if (gc.graph)
            {
                cudaGraphDestroy(gc.graph);
                gc.graph = nullptr;
            }
            cudaCheck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "amg capture begin");
            vcycleAt(0, amg, A, rA, wA);
            cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &gc.graph), "amg capture end");
            cudaCheck(cudaGraphInstantiate(&gc.exec, gc.graph, 0), "amg graph instantiate");
            gc.key = A.diag; gc.keyEpoch = deviceReductionScratchEpoch();
        }
        cudaCheck(cudaGraphLaunch(gc.exec, cudaStreamPerThread), "amg graph launch");
    };

    // Device-resident Krylov scalars: wArA / pAp / alpha / beta live on the device and feed the *Dev kernels by
    // pointer, so the host never blocks on the two dot-products driving the recurrence. The only per-iter host sync is
    // the residual-norm read for the convergence check (K=1 -> exact per-iter; checkEvery>1 batches it, trading
    // exactness for fewer syncs). The scalars are persistent AMGData members, allocated once.
    scalar* dWArA    = amg.sWArA.data();
    scalar* dWArAold = amg.sWArAold.data();
    scalar* dPap     = amg.sPap.data();
    scalar* dAlpha   = amg.sAlpha.data();
    scalar* dNegAlpha= amg.sNegAlpha.data();
    scalar* dBeta    = amg.sBeta.data();
    scalar* dResNorm = amg.sResNorm.data();
    const int K = (checkEvery > 1) ? checkEvery : 1;            // residual read cadence (1 = exact per-iter)
    int nIter = 0;
    if (minIter > 0 || !converged(perf.finalResidual))
    {
        do
        {
            if (nIter > 0) deviceScalarCopy(dWArA, dWArAold);
            applyPrecond();                                     // wA = M^-1 rA (recursive V-cycle)
            deviceDotInto(wA, rA, dWArA);
            if (nIter == 0) deviceCopy(pA, wA);                 // p = w
            else if (corrScaling)                             // flexible CG (Polak-Ribiere+): nonlinear precond
            {
                deviceDotInto(wA, rOld, amg.sZrOld.data());
                flexBetaK<<<1,1>>>(dWArA, amg.sZrOld.data(), dWArAold, dBeta);
                deviceFusedScaleAxpy(pA, dBeta, wA);
            }
            else
            {
                deviceScalarDiv(dWArA, dWArAold, dBeta);     // Fletcher-Reeves beta
                deviceFusedScaleAxpy(pA, dBeta, wA);
            }
            if (corrScaling) deviceCopy(rOld, rA);              // save r_k for iter k+1 (rA is about to be updated)
            deviceAmul(A, pA, wA);
            deviceDotInto(wA, pA, dPap);
            deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
            deviceAxpyDev(dAlpha, pA, psi);
            deviceAxpyDev(dNegAlpha, wA, rA);
            ++nIter;
            if (nIter % K == 0 || nIter >= maxIter)           // read |r|_1 only every K iters (K=1: exact per-iter)
            {
                deviceSumMagInto(rA, dResNorm);
                perf.finalResidual = deviceReadScalar(dResNorm)/normFactor;   // the only host sync
            }
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
    }
    perf.nIterations = nIter;
    static const bool dbgCyc = std::getenv("BRAE_AMG_CYCLES") != nullptr;   // benchmark: AMG V-cycles per pressure solve
    if (dbgCyc) std::fprintf(stderr, "[AMG] p cycles=%d finalRes=%.3e\n", nIter, perf.finalResidual);
    return perf;
}

// Cast the fine + coarse matrices to their FP32 mirrors for this solve's V-cycles. Call ONCE per solve, AFTER
// amgGalerkin has updated the FP64 operators (the coarse matrices change every SIMPLE step). No-op unless FP32
// mixed precision applies (BRAE_AMG_FP32 default on; the SA/GS/Chebyshev paths stay FP64). After this,

} // namespace brae
