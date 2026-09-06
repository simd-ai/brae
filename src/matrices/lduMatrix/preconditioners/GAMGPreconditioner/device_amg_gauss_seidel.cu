// Standalone symGaussSeidel LINEAR SOLVER (OpenFOAM smoothSolver's stopping rule around a MULTICOLOUR
// Gauss-Seidel sweep -- not symGaussSeidelSmoother's index order; see device_amg.cuh): the FP64 host loop,
// the device-resident conditional-graph variant (BRAE_GS_DEVICE, CUDA>=13), and the FP32 twin (BRAE_TURB_FP32).
// Used for the momentum/turbulence GaussSeidel solver option -- it is NOT part of the AMG V-cycle (it only reuses
// the multicolor sweep gsSweep + the FP32 SpMV amulF, both from headers). Verbatim split of device_amg.cu -- no
// logic change. The greedy coloring builder greedyColor stays in the AMG core; declared locally below.
#include "device_amg.cuh"          // AMGData / GridColoring / DeviceSolverPerf / deviceSymGaussSeidel decl
#include "device_amg_detail.cuh"   // safeDiag / nBlocks / TPB / BRAE_HAS_GS_DEVICE
#include "device_amg_internal.cuh" // Coloring, gsSweep, gsScaleInvK, LduF/lduF/cast_, amulF
#include "device_sym_gauss_seidel.cuh"  // the level-scheduled sweep the default path runs
#include "amg_kernels.cuh"         // gsColorT<> (the multicolor GS kernel)
#include "device_ldu.cuh"          // DeviceLduView / deviceAmul
#include "device_blas.cuh"         // deviceCopy / deviceAxpy / deviceSumMagInto
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace brae {

// greedy multicolor coloring -- built by the AMG hierarchy (device_amg.cu); shared with the GS solver's cache.
Coloring greedyColor(const std::vector<label>& owner, const std::vector<label>& nei, int nC);

// Greedy cell coloring for the symGaussSeidel scalar solver, cached per mesh (keyed on A.owner, the graph is fixed
// per mesh; the matrix VALUES change every call, the coloring does not). Shared by the host-loop and device-graph paths.
static const GridColoring& gsColoringFor(const DeviceLduView& A)
{
    static std::map<const label*, GridColoring> colorCache;
    auto it = colorCache.find(A.owner);
    if (it == colorCache.end())
    {
        const int nF = A.nInternalFaces;
        std::vector<label> ownerH(nF), neiH(nF);
        cudaMemcpy(ownerH.data(), A.owner, nF*sizeof(label), cudaMemcpyDeviceToHost);
        cudaMemcpy(neiH.data(),   A.nei,   nF*sizeof(label), cudaMemcpyDeviceToHost);
        Coloring c = greedyColor(ownerH, neiH, A.nCells);
        GridColoring gc;
        gc.nColors=c.nColors;
        gc.cells.copyFrom(c.cells);
        gc.start.copyFrom(c.start);
        gc.startH=c.start;
        it = colorCache.emplace(A.owner, std::move(gc)).first;
    }
    return it->second;
}

// Device-resident symGaussSeidel (BRAE_GS_DEVICE). The host-loop symGaussSeidel below reads the residual norm to the
// host after every sweep (a blocking D2H) to decide {converged? stop}; on the turbulent k/epsilon path that read is
// the dominant host stall. This variant moves the decision onto the GPU via a conditional-graph WHILE node: the loop
// body (fwd+bwd sweep, residual recompute, normalize, stop-test) is captured once and replayed on-device while a tiny
// kernel drives the loop condition with cudaGraphSetConditional(). It is exact, not batched: the on-device test is the
// same per-sweep predicate (res<tol || res<relTol*init || iter>=maxIter), so the sweep count and psi are bit-identical
// to the host loop. The only host sync left is the one initial-residual read per solve, which SIMPLE needs anyway.
//
// Graph-replay invariants: the k/epsilon matrix is reassembled into per-call temporaries each outer iter, so the
// incoming (diag,upper,lower,b) pointers move; the graph instead references stable cache-owned buffers and the current
// matrix is copied in (async D2D) before each replay. The cache is keyed on psi (a persistent field buffer), so k and
// epsilon get independent graph instances. normFactor is a device-resident scalar; tol/relTol/maxIter are baked in at
// capture. WHILE nodes require CUDA >= 13; on older toolkits this path compiles out and the host loop below is used.
#if CUDART_VERSION >= 13000
// The host loop's stop test, run on the device: `sweeps += nSweeps` (smoothSolver.C:205 counts sweeps,
// not evaluations), then `(finalRes < tol || finalRes < relTol*initRes) && sweeps >= minIter`, or the
// cap. One thread; the handle is the WHILE node's condition.
__global__
void gsSetCondK(
    cudaGraphConditionalHandle h,
    const scalar* res,
    scalar tol,
    const scalar* init,
    scalar relTol,
    int* iter,
    int maxIter,
    int minIter,
    int sweepsPer)
{
    if (threadIdx.x || blockIdx.x) return;
    const int it = (*iter += sweepsPer);
    const scalar r = *res;
    const bool converged = ((r < tol) || (r < relTol * (*init))) && (it >= minIter);
    const bool stop = converged || (it >= maxIter);
    cudaGraphSetConditional(h, stop ? 0u : 1u);
}
struct GSGraphCache
{
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphConditionalHandle handle{};
    const void* key = nullptr;
    int scratchEpoch = -1;        // deviceReductionScratchEpoch() at capture
    const void* owner = nullptr;  // the mesh the baked levels, CSR and buffer sizes belong to
    int nC = -1, nF = -1;
    // tol/relTol/maxIter/minIter/nSweeps/symmetric are ARGUMENTS of the captured kernels, so a solve
    // that changes any of them on the same field (kFinal after k, a tightened arm in a gate) must
    // re-capture. Keying on psi alone replayed the previous solve's rule.
    scalar tol = -1, relTol = -1;
    int    maxIter = -1, minIter = -1, sweepsPer = -1;
    bool   symmetric = true;
    DeviceBuffer<scalar> gsDiag, gsUpper, gsLower, gsB, Ax, r;        // stable, graph-referenced
    GSLevelCoefs lc;                                                   // the level-ordered operands (item 60b),
    GSLevelCells cc;                                                   // refreshed per solve, baked by pointer
    DeviceBuffer<scalar> gNormF, gInit, gRes;
    DeviceBuffer<int> gIter;
    ~GSGraphCache()
    {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};
// OpenFOAM's smoothSolver loop, on the device: sweep, residual, stop test, all inside a conditional-graph
// WHILE node, replayed with one launch. The host loop below reads the residual back once per sweep and
// blocks on it; measured on T3A (item 55), those reads were 31 of the turbulence hook's ~43 ms per outer
// iteration against 11.5 ms of kernels -- the CPU and GPU never overlapped. This path syncs TWICE per
// solve: the initial residual (OpenFOAM's `Initial residual`, which the report needs first) and the
// final residual + count after the loop (item 52's report line). Same sweeps, same residual, same stop,
// so tests/gs_ladder and tests/solve_report_vs_openfoam hold it to OpenFOAM exactly as the host loop.
static void deviceSymGaussSeidelGraph(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    int sweepsPer,
    bool symmetric,
    DeviceSolverPerf& perf)
{
    const DeviceGaussSeidelLevels& lv = gsLevelsFor(A);
    static auto& cache = *new std::map<const void*, GSGraphCache>();  // leaked (no static-dtor-after-context-teardown hazard)
    GSGraphCache& c = cache[psi.data()];
    const int nC = A.nCells, nF = A.nInternalFaces;
    c.gsDiag.resize(nC);
    c.gsUpper.resize(nF);
    c.gsLower.resize(nF);
    c.gsB.resize(nC);
    c.Ax.resize(nC);
    c.r.resize(nC);
    c.gNormF.resize(1);
    c.gInit.resize(1);
    c.gRes.resize(1);
    c.gIter.resize(1);
    // copy the current matrix + rhs into the stable graph-referenced buffers (async D2D, no host sync)
    cudaMemcpyAsync(c.gsDiag.data(),  A.diag,  nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gsUpper.data(), A.upper, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gsLower.data(), A.lower, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gsB.data(),     b.data(),nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gNormF.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice,   cudaStreamPerThread);
    DeviceLduView sA = A;
    sA.diag = c.gsDiag.data();
    sA.upper = c.gsUpper.data();
    sA.lower = c.gsLower.data();  // stable values + mesh topology
    // this solve's operands in walk order (item 60b): permuted once here, read by every captured sweep
    gsLevelCoefsRefresh(sA, lv, c.lc);
    gsLevelCellsRefresh(sA, c.gsB, lv, c.cc);
    static bool announced = false;
    if (!announced)
    {
        announced = true;
        std::printf("  smoothSolver: device loop, %s gather (%d levels, widest %d: %s)\n",
                    gsLevelGatherEnabled(lv) ? "level-ordered" : "index", lv.levels(), lv.maxLevelWidth,
                    gsSingleBlockWalk(lv) ? "single-block" : "per-level launches");
    }
    // initial residual (also PRE-SIZES Ax/r and the reduction scratch, so the capture allocates nothing)
    deviceAmul(sA, psi, c.Ax);
    deviceCopy(c.r, c.gsB);
    deviceAxpy(-1.0, c.Ax, c.r);
    deviceSumMagInto(c.r, c.gInit.data());
    gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.gInit.data(), c.gNormF.data());     // gInit = sum|r| / normFactor
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.gInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "gs init D2H");
    cudaStreamSynchronize(cudaStreamPerThread);                                       // sync 1 of 2: OF initialResidual
    if (initRes < tol && minIter <= 0)
    {
        perf = {initRes, initRes, 0};
        return;
    }
    cudaMemsetAsync(c.gIter.data(), 0, sizeof(int), cudaStreamPerThread);
    // the reduction scratch is freed and regrown on demand; a graph holding the old pointer must rebuild
    const int epoch = deviceReductionScratchEpoch();
    const bool recapture = !c.exec || c.key != psi.data() || c.tol != tol || c.relTol != relTol
                        || c.maxIter != maxIter || c.minIter != minIter || c.sweepsPer != sweepsPer
                        || c.symmetric != symmetric || c.scratchEpoch != epoch
                        || c.owner != (const void*)A.owner || c.nC != nC || c.nF != nF;
    if (recapture)
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
        cudaCheck(cudaGraphCreate(&c.graph, 0), "gs graph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 1, cudaGraphCondAssignDefault), "gs cond handle");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(cudaGraphAddNode(&cnode, c.graph, nullptr, nullptr, 0, &cp), "gs cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "gs capture begin");
        // smoothSolver.C:186: nSweeps sweeps, THEN one residual evaluation
        for (int sw = 0; sw < sweepsPer; ++sw) deviceSymGaussSeidelSweepExact(sA, c.gsB, psi, lv, symmetric, &c.lc, &c.cc);
        deviceAmul(sA, psi, c.Ax);
        deviceCopy(c.r, c.gsB);
        deviceAxpy(-1.0, c.Ax, c.r);  // r = b - A*psi
        deviceSumMagInto(c.r, c.gRes.data());
        gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.gRes.data(), c.gNormF.data());      // finalRes = sum|r| / normFactor
        gsSetCondK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.gRes.data(), tol, c.gInit.data(), relTol,
                                                  c.gIter.data(), maxIter, minIter, sweepsPer);
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "gs capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "gs graph instantiate");
        c.key = psi.data(); c.scratchEpoch = epoch;
        c.owner = A.owner; c.nC = nC; c.nF = nF;
        c.tol = tol; c.relTol = relTol; c.maxIter = maxIter; c.minIter = minIter;
        c.sweepsPer = sweepsPer; c.symmetric = symmetric;
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "gs graph launch");       // replay: loop runs to its stop on-device
    // sync 2 of 2: the report. The host loop paid this once per sweep.
    scalar finalRes;
    int nIter;
    cudaCheck(cudaMemcpyAsync(&finalRes, c.gRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "gs final D2H");
    cudaCheck(cudaMemcpyAsync(&nIter,    c.gIter.data(), sizeof(int),   cudaMemcpyDeviceToHost, cudaStreamPerThread), "gs iter D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    perf = {initRes, finalRes, nIter};
    if (std::getenv("BRAE_GS_DEBUG"))
        std::printf("    GS[dev] init=%.4e final=%.4e sweeps=%d (relTol=%.2g nSweeps=%d)\n", initRes, finalRes, nIter, relTol, sweepsPer);
}

// ---------------------------------------------------------------------------------------------------
// The FUSED loop (item 60a): the components of one vector matrix in one WHILE graph. Per iteration of the
// body: nSweeps fused sweeps over the still-active components, then every component's residual
// (the same amul / copy / axpy / sumMag on its own diag, source and psi as the scalar graph runs), then
// ONE stop kernel that applies gsSetCondK's test to each active component -- counting its sweeps,
// recording its final residual, retiring it when it stops -- and keeps the loop alive while any remains.
// A component's sequence of sweeps, residual evaluations and its stopping point are therefore exactly
// what the scalar graph gives it; the only thing shared is the launch.
__global__
void gsFusedCondK(
    cudaGraphConditionalHandle h,
    int nComp,
    const scalar* res,
    scalar tol,
    const scalar* init,
    scalar relTol,
    int* iter,
    int maxIter,
    int minIter,
    int sweepsPer,
    int* active,
    scalar* fin)
{
    if (threadIdx.x || blockIdx.x) return;
    unsigned any = 0u;
    for (int k = 0; k < nComp; ++k)
    {
        if (!active[k]) continue;
        const int it = (iter[k] += sweepsPer);
        const scalar r = res[k];
        fin[k] = r;
        const bool converged = ((r < tol) || (r < relTol * init[k])) && (it >= minIter);
        const bool stop = converged || (it >= maxIter);
        if (stop) active[k] = 0;
        else      any = 1u;
    }
    cudaGraphSetConditional(h, any);
}
__global__
void gsScaleInvNK(int n, scalar* x, const scalar* nf)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / nf[i];
}
struct GSFusedGraphCache
{
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphConditionalHandle handle{};
    int nComp = 0;
    const void* psiKey[GS_FUSED_MAX] = {};
    const void* owner = nullptr;
    int nC = -1, nF = -1, scratchEpoch = -1;
    scalar tol = -1, relTol = -1;
    int    maxIter = -1, minIter = -1, sweepsPer = -1;
    bool   symmetric = true;
    DeviceBuffer<scalar> gsUpper, gsLower;                                     // shared, stable
    DeviceBuffer<scalar> gsDiag[GS_FUSED_MAX], gsB[GS_FUSED_MAX], Ax[GS_FUSED_MAX], r[GS_FUSED_MAX];
    GSLevelCoefs lc;                                                           // level-ordered operands (item 60b):
    GSLevelCells cc[GS_FUSED_MAX];                                             // coefficients shared, cells per component
    DeviceBuffer<scalar> gNormF, gInit, gRes, gFinal;                          // [nComp]
    DeviceBuffer<int>    gIter, gActive;                                       // [nComp]
    ~GSFusedGraphCache()
    {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};
static void deviceSymGaussSeidelGraphFused(
    int nComp,
    const GSFusedComponent* comps,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    int sweepsPer,
    bool symmetric,
    DeviceSolverPerf* perf)
{
    const DeviceLduView& A0 = *comps[0].A;
    const DeviceGaussSeidelLevels& lv = gsLevelsFor(A0);
    static auto& cache = *new std::map<const void*, GSFusedGraphCache>();
    GSFusedGraphCache& c = cache[comps[0].psi->data()];
    const int nC = A0.nCells, nF = A0.nInternalFaces;
    c.gsUpper.resize(nF);
    c.gsLower.resize(nF);
    for (auto* v : {&c.gNormF, &c.gInit, &c.gRes, &c.gFinal}) v->resize(nComp);
    c.gIter.resize(nComp);
    c.gActive.resize(nComp);
    cudaMemcpyAsync(c.gsUpper.data(), A0.upper, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gsLower.data(), A0.lower, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    scalar nfH[GS_FUSED_MAX];
    DeviceLduView sA[GS_FUSED_MAX];
    GSFusedOperands ops;
    ops.nComp = nComp;
    ops.active = c.gActive.data();
    for (int k = 0; k < nComp; ++k)
    {
        c.gsDiag[k].resize(nC);
        c.gsB[k].resize(nC);
        c.Ax[k].resize(nC);
        c.r[k].resize(nC);
        cudaMemcpyAsync(c.gsDiag[k].data(), comps[k].A->diag,   nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(c.gsB[k].data(),    comps[k].b->data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        nfH[k] = comps[k].normFactor;
        sA[k] = *comps[k].A;
        sA[k].diag  = c.gsDiag[k].data();
        sA[k].upper = c.gsUpper.data();
        sA[k].lower = c.gsLower.data();
        ops.diag[k] = c.gsDiag[k].data();
        ops.b[k]    = c.gsB[k].data();
        ops.psi[k]  = comps[k].psi->data();
    }
    cudaMemcpyAsync(c.gNormF.data(), nfH, nComp*sizeof(scalar), cudaMemcpyHostToDevice, cudaStreamPerThread);
    // this solve's operands in walk order (item 60b): the shared coefficients once, the cells per component
    gsLevelCoefsRefresh(sA[0], lv, c.lc);
    const GSLevelCells* ccp[GS_FUSED_MAX] = {};
    for (int k = 0; k < nComp; ++k)
    {
        gsLevelCellsRefresh(sA[k], c.gsB[k], lv, c.cc[k]);
        ccp[k] = &c.cc[k];
    }
    // every component's initial residual, one read (sync 1 of 2); this also pre-sizes the scratch
    for (int k = 0; k < nComp; ++k)
    {
        deviceAmul(sA[k], *comps[k].psi, c.Ax[k]);
        deviceCopy(c.r[k], c.gsB[k]);
        deviceAxpy(-1.0, c.Ax[k], c.r[k]);
        deviceSumMagInto(c.r[k], c.gInit.data() + k);
    }
    gsScaleInvNK<<<1, 32, 0, cudaStreamPerThread>>>(nComp, c.gInit.data(), c.gNormF.data());
    scalar initH[GS_FUSED_MAX];
    cudaCheck(cudaMemcpyAsync(initH, c.gInit.data(), nComp*sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "gs fused init D2H");
    cudaStreamSynchronize(cudaStreamPerThread);
    int activeH[GS_FUSED_MAX];
    int nActive = 0;
    for (int k = 0; k < nComp; ++k)
    {
        perf[k] = {initH[k], initH[k], 0};
        activeH[k] = (initH[k] < tol && minIter <= 0) ? 0 : 1;      // the scalar graph's early-out, per component
        nActive += activeH[k];
    }
    if (nActive == 0) return;
    cudaMemcpyAsync(c.gActive.data(), activeH, nComp*sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaMemsetAsync(c.gIter.data(), 0, nComp*sizeof(int), cudaStreamPerThread);
    cudaMemcpyAsync(c.gFinal.data(), c.gInit.data(), nComp*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    const int epoch = deviceReductionScratchEpoch();
    bool same = c.exec && c.nComp == nComp && c.tol == tol && c.relTol == relTol && c.maxIter == maxIter
             && c.minIter == minIter && c.sweepsPer == sweepsPer && c.symmetric == symmetric
             && c.scratchEpoch == epoch && c.owner == (const void*)A0.owner && c.nC == nC && c.nF == nF;
    for (int k = 0; k < nComp; ++k) same = same && c.psiKey[k] == (const void*)comps[k].psi->data();
    if (!same)
    {
        if (c.exec)
        {
            cudaGraphExecDestroy(c.exec);
            c.exec = nullptr;
        }
        if (c.graph)
        {
            cudaGraphDestroy(c.graph);
            c.graph = nullptr;
        }
        cudaCheck(cudaGraphCreate(&c.graph, 0), "gs fused graph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 1, cudaGraphCondAssignDefault), "gs fused cond handle");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(cudaGraphAddNode(&cnode, c.graph, nullptr, nullptr, 0, &cp), "gs fused cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "gs fused capture begin");
        for (int sw = 0; sw < sweepsPer; ++sw) deviceSymGaussSeidelSweepExactFused(sA[0], ops, lv, symmetric, &c.lc, ccp);
        for (int k = 0; k < nComp; ++k)
        {
            deviceAmul(sA[k], *comps[k].psi, c.Ax[k]);
            deviceCopy(c.r[k], c.gsB[k]);
            deviceAxpy(-1.0, c.Ax[k], c.r[k]);
            deviceSumMagInto(c.r[k], c.gRes.data() + k);
        }
        gsScaleInvNK<<<1, 32, 0, cudaStreamPerThread>>>(nComp, c.gRes.data(), c.gNormF.data());
        gsFusedCondK<<<1, 1, 0, cudaStreamPerThread>>>(c.handle, nComp, c.gRes.data(), tol, c.gInit.data(), relTol,
                                                       c.gIter.data(), maxIter, minIter, sweepsPer,
                                                       c.gActive.data(), c.gFinal.data());
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "gs fused capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "gs fused graph instantiate");
        c.nComp = nComp;
        for (int k = 0; k < nComp; ++k) c.psiKey[k] = comps[k].psi->data();
        c.owner = A0.owner; c.nC = nC; c.nF = nF; c.scratchEpoch = epoch;
        c.tol = tol; c.relTol = relTol; c.maxIter = maxIter; c.minIter = minIter;
        c.sweepsPer = sweepsPer; c.symmetric = symmetric;
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "gs fused graph launch");
    scalar finH[GS_FUSED_MAX];
    int iterH[GS_FUSED_MAX];
    cudaCheck(cudaMemcpyAsync(finH,  c.gFinal.data(), nComp*sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "gs fused final D2H");
    cudaCheck(cudaMemcpyAsync(iterH, c.gIter.data(),  nComp*sizeof(int),    cudaMemcpyDeviceToHost, cudaStreamPerThread), "gs fused iter D2H");
    cudaStreamSynchronize(cudaStreamPerThread);                                        // sync 2 of 2
    for (int k = 0; k < nComp; ++k)
        if (activeH[k]) perf[k] = {initH[k], finH[k], iterH[k]};
}
#endif // CUDART_VERSION >= 13000

void deviceSymGaussSeidelFused(
    int nComp,
    const GSFusedComponent* comps,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    int nSweeps,
    bool symmetric,
    DeviceSolverPerf* perf)
{
    // BRAE_GS_FUSED=0 restores the per-component walks (the identity gate's other arm). The scalar
    // path's own opt-ins and its host loop are honoured by falling back to it.
    static const bool fusedOff = []()
    {
        const char* e = std::getenv("BRAE_GS_FUSED");
        return e && std::string(e) == "0";
    }();
    bool shared = nComp >= 2 && nComp <= GS_FUSED_MAX && !fusedOff;
    for (int k = 1; k < nComp && shared; ++k)
        shared = comps[k].A->upper == comps[0].A->upper && comps[k].A->lower == comps[0].A->lower
              && comps[k].A->owner == comps[0].A->owner && comps[k].A->nCells == comps[0].A->nCells;
#ifdef BRAE_HAS_GS_DEVICE
    static const bool hostLoop = std::getenv("BRAE_GS_HOST_LOOP") != nullptr;
    static const bool optIn = std::getenv("BRAE_TURB_FP32") != nullptr || std::getenv("BRAE_TURB_JACOBI") != nullptr;
    if (shared && !hostLoop && !optIn)
    {
        static bool announced = false;
        if (!announced)
        {
            announced = true;
            // which walk the mesh takes (device_sym_gauss_seidel.cu: one block when no level outgrows it)
            const DeviceGaussSeidelLevels& lv = gsLevelsFor(*comps[0].A);
            std::printf("  symGaussSeidel: fused walk (%d components per level, %d levels, widest %d: %s, %s gather); BRAE_GS_FUSED=0 restores one walk per component\n",
                        nComp, lv.levels(), lv.maxLevelWidth, gsSingleBlockWalk(lv) ? "single-block" : "per-level launches",
                        gsLevelGatherEnabled(lv) ? "level-ordered" : "index");
        }
        deviceSymGaussSeidelGraphFused(nComp, comps, tol, relTol, maxIter, minIter, (nSweeps > 1) ? nSweeps : 1, symmetric, perf);
        return;
    }
#endif
    for (int k = 0; k < nComp; ++k)
        deviceSymGaussSeidel(*comps[k].A, *comps[k].b, *comps[k].psi, comps[k].normFactor, tol, relTol, maxIter,
                             &perf[k], minIter, nSweeps, symmetric);
}

// symGaussSeidel scalar solver. Each cell is updated psi[c] = (b[c] - sum_{j!=c} A[c][j]*psi[j]) / diag[c] (the
// gsColorT gather), forward then reverse, and the outer loop IS smoothSolver::solve (initial residual + normFactor,
// then {smooth; recompute r; finalRes = sumMag(r)/normFactor} until finalRes<tol || finalRes<relTol*initRes ||
// maxIter). What is NOT OpenFOAM's is the ORDER: symGaussSeidelSmoother.C walks cells in strict index order
// (:147 forward, :176 reverse) while this walks them in COLOUR order. That is the same algorithm under a
// permutation -- verified, a sequential transcription visiting cells in the colour permutation reproduces this
// sweep to 0.0 -- but Gauss-Seidel is order-dependent, so the ITERATE after n sweeps is not OpenFOAM's, and at
// the loose relTol a SIMPLE step asks for the two solvers stop in different places.
//
// The gap is not small. On validation/T3A (26820 cells; brae's greedy colouring gives 3 colours of
// 13390/13390/40, so the first colour is 49.93% of the mesh updated entirely from old values -- a Jacobi step)
// OpenFOAM cut Ux's residual 1.6186e-05 -> 6.940e-07 in ONE sweep against this solver's SEVEN to 1.278e-06,
// both stopping on the case's own relTol 0.1.
//
// The exact alternative is level scheduling -- device_dilu.cu already builds the identical DAG -- but T3A's
// schedule is 465 levels deep with a median of 74 cells per level, i.e. 930 kernel launches per sweep against
// the 6 here, so it is not run. One symGS sweep = gsSweep(forward) + gsSweep(reverse); the colouring depends
// only on the graph (owner/nei) and is built once per mesh and cached.
static scalar deviceSymGaussSeidelF32(
    const DeviceLduView&,
    const DeviceBuffer<scalar>&,
    DeviceBuffer<scalar>&,
    scalar,
    scalar,
    scalar,
    int);   // FP32 turbulence GS (defined below)
scalar deviceSymGaussSeidel(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    DeviceSolverPerf* perf,
    int minIter,
    int nSweeps,
    bool symmetric)
{
    // minIter (fvSolution solvers/<field>/minIter) is OF's floor on the sweep count: smoothSolver.C
    // enters the loop when minIter > 0 even if the initial residual already passes, and keeps sweeping
    // until nIterations >= minIter. The three opt-in fast paths below bake the stop test without it, so
    // a floor takes the host loop, which honours it exactly.
    const bool floored = minIter > 0;
    // nSweeps (fvSolution solvers/<field>/nSweeps, smoothSolver.C:78, default 1) is how many smoothing
    // sweeps run between residual EVALUATIONS: smoothSolver.C:186 smooths nSweeps_ times per pass of the
    // do-while and :205 then advances nIterations by nSweeps_, so the stop test is consulted only on a
    // multiple of it and the solve OVERSHOOTS its relTol by whatever the extra sweeps buy. Measured on
    // validation/simpleBoxIO (U smoothSolver/symGaussSeidel, tolerance 1e-10, relTol 0.1), first Ux
    // solve: OpenFOAM leaves at 9.04154904498e-04 with nSweeps 1 and at 1.40100300931e-06 with
    // nSweeps 2 -- 645x lower under the same relTol. brae read the entry nowhere, so validation/airFoil2D,
    // whose own fvSolution says `nSweeps 2` on U and nuTilda, ran one sweep per evaluation and reported
    // ODD counts where OpenFOAM's were even in all 600 of its solves.
    //
    // The three opt-in fast paths below bake a PER-SWEEP stop test, so a case asking for more than one
    // sweep takes the host loop.
    const int  sweepsPer = (nSweeps > 1) ? nSweeps : 1;
    const bool batched   = sweepsPer > 1;
    // BRAE_TURB_FP32: solve in FP32 (half the bytes on the BW-bound sweeps; loose turbulence tol >> FP32 eps).
    static const bool turbF32 = std::getenv("BRAE_TURB_FP32") != nullptr;
    if (turbF32 && !floored && !batched && symmetric)
    {
        scalar r = deviceSymGaussSeidelF32(A, b, psi, normFactor, tol, relTol, maxIter);
        if (perf) *perf = {r, r, 1};
        return r;
    }   // opt-in path: report init only
    // BRAE_TURB_JACOBI: weighted-Jacobi solve (fully parallel, no colour sync). Experiment only -- Jacobi needs ~2x
    // the sweeps -> ~2x the bandwidth on this BW-bound path, so it is slower than the graphed GS. Same stop test as GS.
    static const bool turbJac = std::getenv("BRAE_TURB_JACOBI") != nullptr;
    if (turbJac && !floored && !batched && symmetric)
    {
        DeviceBuffer<scalar> Ax, r;
        deviceAmul(A, psi, Ax);
        deviceCopy(r, b);
        deviceAxpy(-1.0, Ax, r);
        const scalar initRes = deviceSumMag(r) / normFactor;
        if (initRes >= tol)
        {
            int iter = 0;
            scalar finalRes = initRes;
            for (; iter < maxIter; ++iter)
            {
                deviceAmul(A, psi, Ax);                                                // Ax = A psi
                smoothT<scalar><<<nBlocks(A.nCells),TPB>>>(A.nCells, b.data(), Ax.data(), A.diag, psi.data());   // psi += w*(b-Ax)/diag
                deviceAmul(A, psi, Ax);
                deviceCopy(r, b);
                deviceAxpy(-1.0, Ax, r);
                finalRes = deviceSumMag(r) / normFactor;
                if (finalRes < tol || finalRes < relTol*initRes) break;
            }
            if (std::getenv("BRAE_GS_DEBUG")) std::printf("    JAC init=%.4e final=%.4e iters=%d\n", initRes, finalRes, iter+1);
        }
        if (perf) *perf = {initRes, initRes, 1};   // opt-in path: report init only
        return initRes;
    }
    // THE DEFAULT: the whole smoothSolver loop on the device (deviceSymGaussSeidelGraph above), which
    // syncs twice per solve where the host loop below syncs once per sweep. Same sweeps, same residual,
    // same stop -- tests/gs_device_loop_identity holds the two paths' logs byte-identical over 50 T3A
    // iterations. BRAE_GS_HOST_LOOP=1 forces the host loop (identity checks and measurement); a toolkit
    // without conditional graph nodes (CUDA < 13) takes it unconditionally.
#ifdef BRAE_HAS_GS_DEVICE
    static const bool hostLoop = std::getenv("BRAE_GS_HOST_LOOP") != nullptr;
    if (!hostLoop)
    {
        DeviceSolverPerf p;
        deviceSymGaussSeidelGraph(A, b, psi, normFactor, tol, relTol, maxIter, minIter, sweepsPer, symmetric, p);
        if (perf) *perf = p;
        return p.initialResidual;
    }
#endif
    const DeviceGaussSeidelLevels& lv = gsLevelsFor(A);
    // this solve's operands in walk order (item 60b), once here rather than once per sweep
    GSLevelCoefs lc;
    GSLevelCells cc;
    gsLevelCoefsRefresh(A, lv, lc);
    gsLevelCellsRefresh(A, b, lv, cc);
    DeviceBuffer<scalar> Ax, r;
    deviceAmul(A, psi, Ax);
    deviceCopy(r, b);
    deviceAxpy(-1.0, Ax, r);   // r = b - A*psi
    const scalar initRes = deviceSumMag(r) / normFactor;
    if (initRes < tol && !floored)
    {
        if (std::getenv("BRAE_GS_DEBUG")) std::printf("    GS init=%.4e (skip)\n", initRes);
        if (perf) *perf = {initRes, initRes, 0};
        return initRes;
    }
    // OpenFOAM counts SWEEPS, not residual evaluations -- `solverPerf.nIterations() += nSweeps_`
    // (smoothSolver.C:205) -- so every count it reports is a multiple of nSweeps, and maxIter bounds
    // that same accumulator rather than the number of evaluations.
    int sweeps = 0;
    scalar finalRes = initRes;
    while (sweeps < maxIter)
    {
        // OpenFOAM's own sweep, level-scheduled: forward then reverse in strict cell-index order.
        // This used to be two multicolour half-sweeps, which is a DIFFERENT smoother wearing the same
        // name -- see device_sym_gauss_seidel.cuh, and tests/gs_ladder for the 1.36x-to-6.88x it cost.
        for (int s = 0; s < sweepsPer; ++s) deviceSymGaussSeidelSweepExact(A, b, psi, lv, symmetric, &lc, &cc);
        sweeps += sweepsPer;
        deviceAmul(A, psi, Ax);
        deviceCopy(r, b);
        deviceAxpy(-1.0, Ax, r);
        finalRes = deviceSumMag(r) / normFactor;
        if ((finalRes < tol || finalRes < relTol*initRes) && sweeps >= minIter) break;
    }
    if (std::getenv("BRAE_GS_DEBUG")) std::printf("    GS init=%.4e final=%.4e sweeps=%d (relTol=%.2g nSweeps=%d)\n", initRes, finalRes, sweeps, relTol, sweepsPer);
    if (perf) *perf = {initRes, finalRes, sweeps};   // default path: full OF-style init/final/nIter
    return initRes;
}

// Turbulence FP32 Gauss-Seidel (BRAE_TURB_FP32). The k/eps/omega transport is solved to a loose tolerance, far above
// FP32 machine-epsilon, so running the GS in FP32 (half the matrix+vector bytes on the BW-bound sweeps) preserves the
// converged field. Matrix + b + field are cast to FP32 once per solve; FP32 multicolor GS sweeps; the residual norm
// is taken in FP64 for a clean convergence check; the field is cast back to FP64. The sweep is gsColorT<float>, the
// same templated smoother as the FP64 path (see gsSweep).
static void gsSweepF(
    const LduF& A,
    const float* b,
    const GridColoring& gc,
    float* x,
    bool forward)
{
    for (int ci = 0; ci < gc.nColors; ++ci)
    {
        const int col = forward ? ci : (gc.nColors-1-ci);
        const int lo = gc.startH[col], hi = gc.startH[col+1];
        const int nc = hi - lo;
        if (nc <= 0) continue;
        gsColorT<float><<<nBlocks(nc),TPB>>>(lo, hi, gc.cells.data(), b, A.diag, A.ownerStart, A.nei, A.upper,
                                       A.losortStart, A.losort, A.owner, A.lower, x);
    }
}
struct GSFP32Cache
{
    DeviceBuffer<float> dF, uF, lF, bF, xF, AxF, rF;
    DeviceBuffer<scalar> rD;
};
static scalar deviceSymGaussSeidelF32(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const GridColoring& gc = gsColoringFor(A);
    static auto& cache = *new std::map<const void*, GSFP32Cache>();
    GSFP32Cache& c = cache[psi.data()];
    const int nC = A.nCells, nF = A.nInternalFaces;
    c.dF.resize(nC);
    c.uF.resize(nF);
    c.lF.resize(nF);
    c.bF.resize(nC);
    c.xF.resize(nC);
    c.AxF.resize(nC);
    c.rF.resize(nC);
    c.rD.resize(nC);
    cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, A.diag, c.dF.data());
    if (nF > 0)
    {
        cast_<scalar,float><<<nBlocks(nF),TPB>>>(nF, A.upper, c.uF.data());
        cast_<scalar,float><<<nBlocks(nF),TPB>>>(nF, A.lower, c.lF.data());
    }
    cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, b.data(),   c.bF.data());
    cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, psi.data(), c.xF.data());
    const LduF sA{ c.dF.data(), c.uF.data(), c.lF.data(), A.nei, A.owner, A.ownerStart, A.losort, A.losortStart, nC, nF };
    auto residNorm = [&]() -> scalar       // r = b - A x (FP32), |r|_1 in FP64
    {
        amulF(sA, c.xF.data(), c.AxF.data());
        residualT<float><<<nBlocks(nC),TPB>>>(nC, c.bF.data(), c.AxF.data(), c.rF.data());
        cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, c.rF.data(), c.rD.data());
        return deviceSumMag(c.rD) / normFactor;
    };
    const scalar initRes = residNorm();
    if (initRes >= tol)
    {
        int iter = 0;
        scalar finalRes = initRes;
        for (; iter < maxIter; ++iter)
        {
            gsSweepF(sA, c.bF.data(), gc, c.xF.data(), true);
            gsSweepF(sA, c.bF.data(), gc, c.xF.data(), false);
            finalRes = residNorm();
            if (finalRes < tol || finalRes < relTol*initRes) break;
        }
        if (std::getenv("BRAE_GS_DEBUG")) std::printf("    GS[fp32] init=%.4e final=%.4e iters=%d\n", initRes, finalRes, iter+1);
    }
    cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, c.xF.data(), psi.data());         // FP32 field -> FP64
    return initRes;
}

} // namespace brae
