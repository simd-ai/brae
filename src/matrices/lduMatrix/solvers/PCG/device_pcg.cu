// cf GPU offload (G2): device-resident Jacobi-PCG. Mirrors brae::pcg's CG recurrence exactly, but the
// preconditioner is Jacobi (wA = rA/diag) and every vector op is a device kernel.
#include "device_pcg.cuh"
#include "device_pcg_detail.cuh"  // shared BiCGStab recurrence kernels + BiCGScalars scratch
#include "device_amg.cuh"   // AMGData + amgVCycleApply (the distributed AMG-PCG preconditioner)
#include "device_amg_detail.cuh"  // BRAE_HAS_GS_DEVICE (conditional graph nodes); needs device_amg.cuh first for AMGLevel
#include "device_blas.cuh"
#include "device_halo.cuh"
#include "device_reduce.cuh"   // DeviceReducer: on-stream NVSHMEM global reduction (replaces host MPI_Allreduce)
#include "cf_pstream.cuh"
#include <cuda_runtime.h>
#include <map>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <vector>


namespace brae {


DeviceSolverPerf deviceJacobiPCG(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter)
{
    const int nC = A.nCells;
    DeviceBuffer<scalar> wA(nC), rA(nC), pA(nC), Ax(nC);

    deviceAmul(A, psi, Ax);                                  // rA = b - A*psi
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);

    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    scalar wArA = 1e300, wArAold;
    int nIter = 0;
    // OF lduMatrix PCG: the loop is entered when minIter demands it even if the initial residual already
    // passes, and it keeps going until BOTH the convergence test and minIter are satisfied.
    if (minIter > 0 || !converged(perf.finalResidual))
    {
        do
        {
            wArAold = wArA;
            deviceJacobi(wA, rA, A.diag);                   // wA = M^-1 rA  (Jacobi)
            wArA = deviceDot(wA, rA);
            if (nIter == 0) deviceCopy(pA, wA);             // pA = wA
            else                                            // pA = wA + beta*pA
            {
                const scalar beta = (std::fabs(wArAold) > 1e-300) ? wArA / wArAold : 0.0;   // guard breakdown -> 0, not Inf/NaN
                deviceScale(pA, beta);
                deviceAxpy(1.0, wA, pA);
            }
            deviceAmul(A, pA, wA);                          // wA = A*pA
            const scalar wApA  = deviceDot(wA, pA);
            const scalar alpha = (std::fabs(wApA) > 1e-300) ? wArA / wApA : 0.0;   // guard pAp~0 breakdown -> 0, not Inf/NaN
            deviceAxpy(alpha, pA, psi);                     // psi += alpha*pA
            deviceAxpy(-alpha, wA, rA);                     // rA  -= alpha*wA
            perf.finalResidual = deviceSumMag(rA) / normFactor;
            ++nIter;
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
    }
    perf.nIterations = nIter;
    return perf;
}

void deviceNormFactorInto(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& b,
    const DeviceBuffer<scalar>& ones,
    DeviceBuffer<scalar>& dNorm)
{
    dNorm.resize(1);
    const int nC = A.nCells;
    DeviceBuffer<scalar> Apsi(nC), sumA(nC), tmp(nC), t(nC);
    // Device-resident: the 3 reductions (avgPsi, n1, n2) stay on the device; only the final normFactor is read to
    // the host (it scales the residual for the convergence check). Same kernels + same IEEE ops (the divide by nC,
    // the avg-multiply, and the (n1+n2)+1e-20 add are reproduced exactly) -> bit-identical, 3 D2H syncs -> 1.
    DeviceBuffer<scalar> dAvg(1), dN1(1), dN2(1);
    deviceAmul(A, psi, Apsi);                                // A*psi
    deviceAmul(A, ones, sumA);                               // sumA = rowSum(A) = A*1
    deviceDotInto(psi, ones, dAvg.data());                  // psi.ones
    deviceScalarDivConst(dAvg.data(), (scalar)nC, dAvg.data());   // avgPsi = gAverage(psi) = (psi.ones)/nC
    deviceCopy(tmp, sumA);
    deviceScaleDev(dAvg.data(), tmp);      // tmp = sumA*avg(psi)
    deviceCopy(t, Apsi);
    deviceAxpy(-1.0, tmp, t);
    deviceSumMagInto(t, dN1.data());   // n1 = |A*psi - tmp|
    deviceCopy(t, b);
    deviceAxpy(-1.0, tmp, t);
    deviceSumMagInto(t, dN2.data());   // n2 = |b - tmp|
    deviceScalarAdd2(dN1.data(), dN2.data(), 1e-20, dNorm.data());                    // n1 + n2 + 1e-20
}

scalar deviceNormFactor(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& b,
    const DeviceBuffer<scalar>& ones)
{
    DeviceBuffer<scalar> dNorm;
    deviceNormFactorInto(A, psi, b, ones, dNorm);
    return deviceReadScalar(dNorm.data());                                            // the only host sync
}

bool normFactorOnHost()
{
    static const bool on = std::getenv("BRAE_NORMFACTOR_HOST") != nullptr;
    return on;
}

void announceNormFactorMode()
{
    static bool announced = false;
    if (announced) return;
    announced = true;
    std::printf(normFactorOnHost() ? "  normFactor: read to the host per solve (BRAE_NORMFACTOR_HOST)\n"
                                   : "  normFactor: device-resident, never read by the host; BRAE_NORMFACTOR_HOST=1 restores the read\n");
}

#ifdef BRAE_HAS_GS_DEVICE
// ---------------------------------------------------------------------------------------------------
// OpenFOAM's PBiCGStab loop on the device. The host loop below is exact but reads the residual back TWICE
// per iteration at checkEvery 1 (|s| for the mid-iteration exit, |r| and the breakdown flag at the end),
// and each read blocks until the queued kernels drain. Measured with nsys (--cuda-graph-trace=node,
// 6-minus-3 iterations) on sbMatched, 112,000 cells, every field PBiCGStab at 1e-12: the host loop made
// 1,733 blocking reads and 95,265 kernel launches per outer iteration (402 + 285 ms of API time for
// 251 ms of kernels); this loop makes 28 reads and 7 graph launches. On pitzDailyTurb (12,225 cells,
// U at 1e-10 / relTol 0, 19-27 iterations per component) the outer iteration went from 45-48 to
// 13.4 ms warm (40-minus-20, two repeats each), with every solve line and every written field identical.
//
// This runs the SAME recurrence with the SAME two tests inside one conditional graph:
//   * iteration 0 stays host-driven, exactly the loop's nIter==0 path (two reads), so the body never
//     needs the `nIter > 0` branch;
//   * a WHILE node carries iterations >= 1; its body is the first half of an iteration, then an IF/ELSE
//     node: the IF body is the mid-iteration exit (psi += alpha*yA, count, stop), the ELSE body the
//     second half, the end-of-iteration residual, the breakdown flag and the loop test;
//   * the tests are the loop's own, verbatim, on device scalars: converged = r < tol || (relTol > 0 &&
//     r < relTol*init); mid exit needs nIter >= minIter; continue = bd == 0 && ((n < maxIter &&
//     !converged) || n < minIter), with n already incremented -- the host's `++nIter` then `while`.
// Same kernels, same order, same operands, so psi and the iteration count are bit-identical to the host
// loop; tests/bicg_device_loop_identity holds two cases' solve reports and written fields to that.
// Four host syncs per solve where the host loop paid 2*nIter+1.
namespace
{
__global__ void bicgNormK(const scalar* __restrict__ sum, const scalar* __restrict__ nf, scalar* __restrict__ out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = *sum / *nf;
}
__global__ void bicgMidCondK(cudaGraphConditionalHandle hIf, const scalar* __restrict__ sN, scalar tol,
                             const scalar* __restrict__ init, scalar relTol, const int* __restrict__ iter, int minIter)
{
    if (threadIdx.x || blockIdx.x) return;
    const scalar r = *sN;
    const bool conv = (r < tol) || (relTol > 0.0 && r < relTol * (*init));
    cudaGraphSetConditional(hIf, (conv && *iter >= minIter) ? 1u : 0u);
}
__global__ void bicgMidExitK(cudaGraphConditionalHandle hW, int* __restrict__ iter, const scalar* __restrict__ sN,
                             scalar* __restrict__ fin)
{
    if (threadIdx.x || blockIdx.x) return;
    ++(*iter);
    *fin = *sN;
    cudaGraphSetConditional(hW, 0u);
}
__global__ void bicgEndCondK(cudaGraphConditionalHandle hW, const scalar* __restrict__ rN, scalar tol,
                             const scalar* __restrict__ init, scalar relTol, int* __restrict__ iter, int maxIter,
                             int minIter, const scalar* __restrict__ bd, scalar* __restrict__ fin)
{
    if (threadIdx.x || blockIdx.x) return;
    const int n = ++(*iter);
    const scalar r = *rN;
    *fin = r;
    const bool conv = (r < tol) || (relTol > 0.0 && r < relTol * (*init));
    const bool cont = (*bd == 0.0) && ((n < maxIter && !conv) || n < minIter);
    cudaGraphSetConditional(hW, cont ? 1u : 0u);
}

struct BiCGGraphCache
{
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphConditionalHandle hW{}, hIf{};
    const void* key = nullptr;
    scalar tol = -1, relTol = -1;
    int maxIter = -1, minIter = -1;
    const void* precon = nullptr;
    // everything else the captured kernels bake in: the topology, the cell count (which sizes the
    // cache-owned vectors), the DILU factor's buffers and level count, and the reduction scratch epoch
    const void* owner = nullptr;
    int nC = -1, diluLevels = -1, scratchEpoch = -1;
    const void* diluRD = nullptr;
    DeviceBuffer<scalar> gDiag, gUpper, gLower, gB;                    // stable, graph-referenced
    DeviceBuffer<scalar> rA, rA0, pA, yA, AyA, sA, zA, tA, Ax;
    DeviceBuffer<scalar> gNormF, gInit, gSN, gRN, gFinal;
    DeviceBuffer<int>    gIter;
    ~BiCGGraphCache()
    {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};

// Returns false when this path does not apply (the caller then runs the host loop).
bool deviceJacobiBiCGStabGraph(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& psi,
                               const scalar* dNormFactor, scalar tol, scalar relTol, int maxIter, int minIter,
                               const DeviceDilu* precon, DeviceSolverPerf& perf)
{
    const int nC = A.nCells, nF = A.nInternalFaces;
    if (nC <= 0) return false;
    static auto& cache = *new std::map<const void*, BiCGGraphCache>();   // leaked: no static dtor after context teardown
    BiCGGraphCache& c = cache[psi.data()];
    for (auto* v : {&c.gDiag, &c.rA, &c.rA0, &c.pA, &c.yA, &c.AyA, &c.sA, &c.zA, &c.tA, &c.Ax, &c.gB}) v->resize(nC);
    c.gUpper.resize(nF); c.gLower.resize(nF);
    for (auto* v : {&c.gNormF, &c.gInit, &c.gSN, &c.gRN, &c.gFinal}) v->resize(1);
    c.gIter.resize(1);
    // the current matrix and rhs into the stable graph-referenced buffers (async D2D, no host sync)
    cudaMemcpyAsync(c.gDiag.data(),  A.diag,  nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gUpper.data(), A.upper, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gLower.data(), A.lower, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gB.data(),     b.data(),nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gNormF.data(), dNormFactor, sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    DeviceLduView sA = A;
    sA.diag = c.gDiag.data(); sA.upper = c.gUpper.data(); sA.lower = c.gLower.data();
    const bool useDilu = precon && precon->valid;
    if (useDilu) diluUpdate(sA, *const_cast<DeviceDilu*>(precon));
    auto applyPrecon = [&](DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& in)
    {
        if (useDilu) diluApply(sA, *precon, in, out);
        else         deviceJacobi(out, in, sA.diag);
    };
    BiCGScalars& s = bicgScalars();
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    // ---- the prologue: rA = b - A psi, the initial residual (sync 1 of 4) -----------------------------
    deviceAmul(sA, psi, c.Ax);
    deviceCopy(c.rA, c.gB);
    deviceAxpy(-1.0, c.Ax, c.rA);
    deviceCopy(c.rA0, c.rA);
    deviceSumMagInto(c.rA, s.rNorm.data());
    bicgNormK<<<1,1>>>(s.rNorm.data(), c.gNormF.data(), c.gInit.data());
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.gInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "bicg init D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    perf.initialResidual = initRes;
    perf.finalResidual   = initRes;
    perf.nIterations     = 0;
    cudaCheck(cudaMemsetAsync(s.bd.data(), 0, sizeof(scalar), cudaStreamPerThread), "bicg bd zero");
    if (!(minIter > 0 || !converged(initRes))) return true;

    // ---- iteration 0, host-driven: the loop's nIter == 0 path, verbatim (syncs 2 and 3 of 4) --------
    deviceDotInto(c.rA0, c.rA, s.rr.data());
    bicgRhoSingK<<<1,1>>>(s.rr.data(), s.bd.data());
    deviceCopy(c.pA, c.rA);
    applyPrecon(c.yA, c.pA);
    deviceAmul(sA, c.yA, c.AyA);
    deviceDotInto(c.rA0, c.AyA, s.r0Ay.data());
    bicgAlphaK<<<1,1>>>(s.rr.data(), s.r0Ay.data(), s.alpha.data(), s.negAlpha.data(), s.bd.data());
    deviceFusedSxpy(c.sA, c.rA, s.negAlpha.data(), c.AyA);
    deviceSumMagInto(c.sA, s.sNorm.data());
    bicgNormK<<<1,1>>>(s.sNorm.data(), c.gNormF.data(), c.gSN.data());              // |s|/nf on the device
    perf.finalResidual = deviceReadScalar(c.gSN.data());
    if (0 >= minIter && converged(perf.finalResidual))
    {
        deviceAxpyDev(s.alpha.data(), c.yA, psi);
        perf.nIterations = 1;
        return true;
    }
    applyPrecon(c.zA, c.sA);
    deviceAmul(sA, c.zA, c.tA);
    deviceDotInto(c.tA, c.tA, s.tt.data());
    deviceDotInto(c.tA, c.sA, s.ts.data());
    omegaK<<<1,1>>>(s.ts.data(), s.tt.data(), s.omega.data(), s.negOmega.data(), s.bd.data());
    deviceFusedAxpy2(psi, s.alpha.data(), c.yA, s.omega.data(), c.zA);
    deviceFusedSxpy(c.rA, c.sA, s.negOmega.data(), c.tA);
    deviceSumMagInto(c.rA, s.rNorm.data());
    bicgNormK<<<1,1>>>(s.rNorm.data(), c.gNormF.data(), c.gRN.data());              // |r|/nf on the device
    scalar rn, bdv;
    cudaCheck(cudaMemcpyAsync(&rn,  c.gRN.data(),  sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "bicg r0 D2H");
    cudaCheck(cudaMemcpyAsync(&bdv, s.bd.data(),    sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "bicg bd D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    perf.finalResidual = rn;
    int nIter = 1;
    if (bdv != 0.0 || !((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter))
    {
        perf.nIterations = nIter;
        return true;
    }

    // ---- iterations >= 1: the graph ------------------------------------------------------------------
    const scalar fin0 = perf.finalResidual;
    cudaMemcpyAsync(c.gIter.data(),  &nIter, sizeof(int),    cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gFinal.data(), &fin0,  sizeof(scalar), cudaMemcpyHostToDevice, cudaStreamPerThread);
    const void* diluRD   = useDilu ? (const void*)precon->rD.data() : nullptr;
    const int   diluLv   = useDilu ? precon->levels() : -1;
    const int   epoch    = deviceReductionScratchEpoch();
    const bool recapture = !c.exec || c.key != psi.data() || c.tol != tol || c.relTol != relTol
                        || c.maxIter != maxIter || c.minIter != minIter || c.precon != (useDilu ? (const void*)precon : nullptr)
                        || c.owner != (const void*)A.owner || c.nC != nC || c.diluRD != diluRD || c.diluLevels != diluLv
                        || c.scratchEpoch != epoch;
    if (recapture)
    {
        if (c.exec)  { cudaGraphExecDestroy(c.exec);  c.exec = nullptr; }
        if (c.graph) { cudaGraphDestroy(c.graph);     c.graph = nullptr; }
        cudaCheck(cudaGraphCreate(&c.graph, 0), "bicg graph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.hW, c.graph, 1, cudaGraphCondAssignDefault), "bicg while handle");
        cudaGraphNodeParams wp = {};
        wp.type = cudaGraphNodeTypeConditional;
        wp.conditional.handle = c.hW;
        wp.conditional.type = cudaGraphCondTypeWhile;
        wp.conditional.size = 1;
        cudaGraphNode_t wnode;
        cudaCheck(cudaGraphAddNode(&wnode, c.graph, nullptr, nullptr, 0, &wp), "bicg while node");
        cudaGraph_t body = wp.conditional.phGraph_out[0];
        // the IF/ELSE handle lives on the graph that will contain the IF node: the WHILE body
        cudaCheck(cudaGraphConditionalHandleCreate(&c.hIf, body, 0, cudaGraphCondAssignDefault), "bicg if handle");

        // segment A: the first half of an iteration, captured into the body
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "bicg capture A");
        deviceScalarCopy(s.rr.data(), s.rrOld.data());                                   // rA0rAold = rA0rA
        deviceDotInto(c.rA0, c.rA, s.rr.data());
        bicgRhoSingK<<<1,1>>>(s.rr.data(), s.bd.data());
        bicgBetaK<<<1,1>>>(s.rr.data(), s.rrOld.data(), s.alpha.data(), s.omega.data(), s.beta.data(), s.bd.data());
        deviceFusedBicgP(c.rA, c.pA, c.AyA, s.beta.data(), s.negOmega.data());
        applyPrecon(c.yA, c.pA);
        deviceAmul(sA, c.yA, c.AyA);
        deviceDotInto(c.rA0, c.AyA, s.r0Ay.data());
        bicgAlphaK<<<1,1>>>(s.rr.data(), s.r0Ay.data(), s.alpha.data(), s.negAlpha.data(), s.bd.data());
        deviceFusedSxpy(c.sA, c.rA, s.negAlpha.data(), c.AyA);
        deviceSumMagInto(c.sA, s.sNorm.data());
        bicgNormK<<<1,1>>>(s.sNorm.data(), c.gNormF.data(), c.gSN.data());
        bicgMidCondK<<<1,1>>>(c.hIf, c.gSN.data(), tol, c.gInit.data(), relTol, c.gIter.data(), minIter);
        // the IF node follows everything captured so far
        cudaStreamCaptureStatus st;
        const cudaGraphNode_t* deps = nullptr;
        size_t nDeps = 0;
        cudaCheck(cudaStreamGetCaptureInfo(cudaStreamPerThread, &st, nullptr, nullptr, &deps, nullptr, &nDeps), "bicg capture info");
        std::vector<cudaGraphNode_t> depv(deps, deps + nDeps);
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "bicg capture A end");
        cudaGraphNodeParams ip = {};
        ip.type = cudaGraphNodeTypeConditional;
        ip.conditional.handle = c.hIf;
        ip.conditional.type = cudaGraphCondTypeIf;
        ip.conditional.size = 2;                                                          // body[0] = if, body[1] = else
        cudaGraphNode_t inode;
        cudaCheck(cudaGraphAddNode(&inode, body, depv.data(), nullptr, depv.size(), &ip), "bicg if node");
        // IF: the mid-iteration exit
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, ip.conditional.phGraph_out[0], nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "bicg capture if");
        deviceAxpyDev(s.alpha.data(), c.yA, psi);
        bicgMidExitK<<<1,1>>>(c.hW, c.gIter.data(), c.gSN.data(), c.gFinal.data());
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "bicg capture if end");
        // ELSE: the second half, the end test, the loop decision
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, ip.conditional.phGraph_out[1], nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "bicg capture else");
        applyPrecon(c.zA, c.sA);
        deviceAmul(sA, c.zA, c.tA);
        deviceDotInto(c.tA, c.tA, s.tt.data());
        deviceDotInto(c.tA, c.sA, s.ts.data());
        omegaK<<<1,1>>>(s.ts.data(), s.tt.data(), s.omega.data(), s.negOmega.data(), s.bd.data());
        deviceFusedAxpy2(psi, s.alpha.data(), c.yA, s.omega.data(), c.zA);
        deviceFusedSxpy(c.rA, c.sA, s.negOmega.data(), c.tA);
        deviceSumMagInto(c.rA, s.rNorm.data());
        bicgNormK<<<1,1>>>(s.rNorm.data(), c.gNormF.data(), c.gRN.data());
        bicgEndCondK<<<1,1>>>(c.hW, c.gRN.data(), tol, c.gInit.data(), relTol, c.gIter.data(), maxIter, minIter, s.bd.data(), c.gFinal.data());
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "bicg capture else end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "bicg graph instantiate");
        c.key = psi.data(); c.tol = tol; c.relTol = relTol; c.maxIter = maxIter; c.minIter = minIter;
        c.precon = useDilu ? (const void*)precon : nullptr;
        c.owner = A.owner; c.nC = nC; c.diluRD = diluRD; c.diluLevels = diluLv; c.scratchEpoch = epoch;
    }
    static bool announced = false;
    if (!announced)
    {
        announced = true;
        std::printf("  BiCGStab: device loop (conditional graph, 4 host syncs per solve); BRAE_BICG_HOST_LOOP=1 restores the host loop\n");
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "bicg graph launch");
    // sync 4 of 4: the report
    scalar fr;
    int ni;
    cudaCheck(cudaMemcpyAsync(&fr, c.gFinal.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "bicg final D2H");
    cudaCheck(cudaMemcpyAsync(&ni, c.gIter.data(),  sizeof(int),    cudaMemcpyDeviceToHost, cudaStreamPerThread), "bicg iter D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    perf.finalResidual = fr;      // already |s|/nf or |r|/nf: bicgNormK divides on the device
    perf.nIterations   = ni;
    return true;
}
}   // namespace
#endif // BRAE_HAS_GS_DEVICE

DeviceSolverPerf deviceJacobiBiCGStab(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int checkEvery,
    int minIter,
    const DeviceDilu* precon)
{
    const int nC = A.nCells;
#ifdef BRAE_HAS_GS_DEVICE
    // THE DEFAULT at the exact per-iteration cadence: the same loop on the device (above), four host
    // syncs per solve where this loop pays 2*nIter+1. BRAE_BICG_HOST_LOOP=1 forces this loop
    // (tests/bicg_device_loop_identity holds the two byte-identical); a batched cadence (checkEvery > 1)
    // is a different stopping rule and keeps this loop. Dispatched before the DILU update below: the graph
    // path factorises on its own stable copy of the matrix, and doing both cost 1,608 extra level launches
    // per outer iteration on sbMatched.
    static const bool hostLoop = std::getenv("BRAE_BICG_HOST_LOOP") != nullptr;
    static bool announcedHost = false;
    if (hostLoop && !announcedHost)
    {
        announcedHost = true;
        std::printf("  BiCGStab: host loop (BRAE_BICG_HOST_LOOP set; 2 host syncs per iteration)\n");
    }
    if (checkEvery <= 1 && !hostLoop)
    {
        static thread_local auto& dNf = *new DeviceBuffer<scalar>(1);
        cudaMemcpyAsync(dNf.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice, cudaStreamPerThread);
        DeviceSolverPerf gp;
        if (deviceJacobiBiCGStabGraph(A, b, psi, dNf.data(), tol, relTol, maxIter, minIter, precon, gp)) return gp;
    }
#endif
    // rD depends on the matrix, which changes every solve (the momentum diagonal moves every outer
    // corrector), so the factorisation is rebuilt here rather than cached with the schedule.
    if (precon && precon->valid) diluUpdate(A, *const_cast<DeviceDilu*>(precon));
    auto applyPrecon = [&](DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& in)
    {
        if (precon && precon->valid) diluApply(A, *precon, in, out);
        else                         deviceJacobi(out, in, A.diag);
    };
    const int K = (checkEvery > 1) ? checkEvery : 1;             // convergence-read cadence (1 = exact per-iter)
    DeviceBuffer<scalar> rA(nC), rA0(nC), pA(nC), yA(nC), AyA(nC), sA(nC), zA(nC), tA(nC), Ax(nC);

    deviceAmul(A, psi, Ax);                                  // rA = b - A*psi
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);
    deviceCopy(rA0, rA);

    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    // DEVICE-RESIDENT BiCGStab: alpha/omega/beta and all dots live on the device (fed to the *Dev kernels by pointer).
    // Breakdown (OF checkSingularity: |rA0rA| or |omega| < VSMALL) is detected ON-DEVICE into `bd` and read only on the
    // K-cadence check (with |s|/|r|), so the host reads NOTHING per off-check iter (was 2 per-iter breakdown D2H reads).
    // For any non-breakdown solve nothing trips (rho/omega stay >> VSMALL until convergence breaks first) -> same kernels
    // + IEEE ops + branches -> bit-identical. On a true breakdown the guarded recurrence damps (no NaN) and the batched
    // bd read stops it within K iters (vs OF's immediate break -- equivalent result; breakdown is pathological & rare).
    BiCGScalars& s = bicgScalars();
    cudaCheck(cudaMemsetAsync(s.bd.data(), 0, sizeof(scalar), cudaStreamPerThread), "bicg bd zero");
    int nIter = 0;
    if (minIter > 0 || !converged(perf.finalResidual))
    {
        do
        {
            if (nIter > 0) deviceScalarCopy(s.rr.data(), s.rrOld.data());   // rA0rAold = rA0rA
            deviceDotInto(rA0, rA, s.rr.data());                  // rA0rA = rA0 . rA
            bicgRhoSingK<<<1,1>>>(s.rr.data(), s.bd.data());      // OF checkSingularity(mag(rA0rA)) -> flag (no host read)
            if (nIter == 0) deviceCopy(pA, rA);
            else
            {
                bicgBetaK<<<1,1>>>(s.rr.data(), s.rrOld.data(), s.alpha.data(), s.omega.data(), s.beta.data(), s.bd.data());
                deviceFusedBicgP(rA, pA, AyA, s.beta.data(), s.negOmega.data());            // pA = rA + beta*(pA - omega*AyA)  [fused 3->1]
            }
            applyPrecon(yA, pA);
            deviceAmul(A, yA, AyA);          // yA = M^-1 pA; AyA = A yA
            deviceDotInto(rA0, AyA, s.r0Ay.data());
            bicgAlphaK<<<1,1>>>(s.rr.data(), s.r0Ay.data(), s.alpha.data(), s.negAlpha.data(), s.bd.data());   // alpha = rA0rA/(rA0.AyA), guarded
            deviceFusedSxpy(sA, rA, s.negAlpha.data(), AyA);                            // sA = rA - alpha*AyA  [fused 2->1]
            const bool check = ((nIter + 1) % K == 0) || (nIter + 1 >= maxIter);        // read |s|/|r|/bd only on check iters
            if (check)                                          // mid-iter early-exit only when we read |s|
            {
                deviceSumMagInto(sA, s.sNorm.data());
                perf.finalResidual = deviceReadScalar(s.sNorm.data()) / normFactor;
                // OF guards this mid-iteration return with the minIter floor as well as the convergence
                // test (PBiCGStab.C:222-224, `solverPerf.nIterations() >= minIter_ && checkConvergence`).
                // Unguarded, a case asking `minIter 5` got 1: the do-while below honoured the floor but
                // this exit left before it ever came round. Measured on validation/simpleBoxIO with
                // `minIter 5` on U -- OpenFOAM 5 sweeps, brae 1.
                if (nIter >= minIter && converged(perf.finalResidual))
                {
                    deviceAxpyDev(s.alpha.data(), yA, psi);
                    ++nIter;
                    break;
                }
            }
            applyPrecon(zA, sA);
            deviceAmul(A, zA, tA);           // zA = M^-1 sA; tA = A zA
            deviceDotInto(tA, tA, s.tt.data());
            deviceDotInto(tA, sA, s.ts.data());
            omegaK<<<1,1>>>(s.ts.data(), s.tt.data(), s.omega.data(), s.negOmega.data(), s.bd.data());     // omega = tt>tiny ? ts/tt : 0 (+singularity flag)
            deviceFusedAxpy2(psi, s.alpha.data(), yA, s.omega.data(), zA);               // psi += alpha*yA + omega*zA  [fused 2->1]
            deviceFusedSxpy(rA, sA, s.negOmega.data(), tA);                             // rA = sA - omega*tA  [fused 2->1]
            if (check)
            {
                deviceSumMagInto(rA, s.rNorm.data());
                perf.finalResidual = deviceReadScalar(s.rNorm.data()) / normFactor;
                if (deviceReadScalar(s.bd.data()) != 0.0)   // OF break on singularity, batched to K
                {
                    ++nIter;
                    break;
                }
            }
            ++nIter;
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
    }
    perf.nIterations = nIter;
    return perf;
}

// ---- distributed (multi-GPU) Jacobi-PCG ------------------------------------------------------------------
// The device counterpart of host parallelPCG: same recurrence as deviceJacobiPCG, but A*x uses the
// interface-coupled deviceParallelAmul and every reduction is global (Pstream::allReduce, tier-1).


DeviceSolverPerf deviceJacobiBiCGStab(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    const scalar* dNormFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int checkEvery,
    int minIter,
    const DeviceDilu* precon)
{
    announceNormFactorMode();
#ifdef BRAE_HAS_GS_DEVICE
    static const bool hostLoop = std::getenv("BRAE_GS_HOST_LOOP") != nullptr || std::getenv("BRAE_BICG_HOST_LOOP") != nullptr;
    if (checkEvery <= 1 && !hostLoop && !normFactorOnHost())
    {
        DeviceSolverPerf gp;
        if (deviceJacobiBiCGStabGraph(A, b, psi, dNormFactor, tol, relTol, maxIter, minIter, precon, gp)) return gp;
    }
#endif
    // the host loop needs the number on the host: one read, on this path only
    return deviceJacobiBiCGStab(A, b, psi, deviceReadScalar(dNormFactor), tol, relTol, maxIter, checkEvery, minIter, precon);
}

} // namespace brae
