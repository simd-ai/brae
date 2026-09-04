// Standalone symGaussSeidel LINEAR SOLVER (OpenFOAM smoothSolver's stopping rule around a MULTICOLOUR
// Gauss-Seidel sweep -- not symGaussSeidelSmoother's index order; see device_amg.cuh): the FP64 host loop,
// the device-resident conditional-graph variant (BRAE_GS_DEVICE, CUDA>=13), and the FP32 twin (BRAE_TURB_FP32).
// Used for the momentum/turbulence GaussSeidel solver option -- it is NOT part of the AMG V-cycle (it only reuses
// the multicolor sweep gsSweep + the FP32 SpMV amulF, both from headers). Verbatim split of device_amg.cu -- no
// logic change. The greedy coloring builder greedyColor stays in the AMG core; declared locally below.
#include "device_amg.cuh"          // AMGData / GridColoring / DeviceSolverPerf / deviceSymGaussSeidel decl
#include "device_amg_detail.cuh"   // safeDiag / nBlocks / TPB / BRAE_HAS_GS_DEVICE
#include "device_amg_internal.cuh" // Coloring, gsSweep, gsScaleInvK, LduF/lduF/cast_, amulF
#include "amg_kernels.cuh"         // gsColorT<> (the multicolor GS kernel)
#include "device_ldu.cuh"          // DeviceLduView / deviceAmul
#include "device_blas.cuh"         // deviceCopy / deviceAxpy / deviceSumMagInto
#include <cuda_runtime.h>
#include <map>
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
__global__
void gsSetCondK(
    cudaGraphConditionalHandle h,
    const scalar* res,
    scalar tol,
    const scalar* init,
    scalar relTol,
    int* iter,
    int maxIter)
{
    if (threadIdx.x || blockIdx.x) return;
    const int it = ++(*iter);
    const scalar r = *res;
    const bool stop = (r < tol) || (r < relTol * (*init)) || (it >= maxIter);
    cudaGraphSetConditional(h, stop ? 0u : 1u);
}
struct GSGraphCache
{
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphConditionalHandle handle{};
    const void* key = nullptr;
    DeviceBuffer<scalar> gsDiag, gsUpper, gsLower, gsB, Ax, r;        // stable, graph-referenced
    DeviceBuffer<scalar> gNormF, gInit, gRes;
    DeviceBuffer<int> gIter;
    ~GSGraphCache()
    {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};
static scalar deviceSymGaussSeidelGraph(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const GridColoring& gc = gsColoringFor(A);
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
    // initial residual (also PRE-SIZES Ax/r so the capture below allocates nothing): r = b - A*psi
    deviceAmul(sA, psi, c.Ax);
    deviceCopy(c.r, c.gsB);
    deviceAxpy(-1.0, c.Ax, c.r);
    deviceSumMagInto(c.r, c.gInit.data());
    gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.gInit.data(), c.gNormF.data());     // gInit = sum|r| / normFactor
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.gInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "gs init D2H");
    cudaStreamSynchronize(cudaStreamPerThread);                                       // the ONE host sync / solve (OF initialResidual)
    if (initRes < tol)
    {
        if (std::getenv("BRAE_GS_DEBUG")) std::printf("    GS[dev] init=%.4e (skip)\n", initRes);
        return initRes;
    }
    cudaMemsetAsync(c.gIter.data(), 0, sizeof(int), cudaStreamPerThread);
    if (!c.exec || c.key != psi.data())                                            // (re)capture the WHILE-graph
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
        gsSweep(sA, c.gsB, psi, gc, true);                                           // forward color sweep
        gsSweep(sA, c.gsB, psi, gc, false);                                          // reverse color sweep
        deviceAmul(sA, psi, c.Ax);
        deviceCopy(c.r, c.gsB);
        deviceAxpy(-1.0, c.Ax, c.r);  // r = b - A*psi
        deviceSumMagInto(c.r, c.gRes.data());
        gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.gRes.data(), c.gNormF.data());      // finalRes = sum|r| / normFactor
        gsSetCondK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.gRes.data(), tol, c.gInit.data(), relTol, c.gIter.data(), maxIter);
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "gs capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "gs graph instantiate");
        c.key = psi.data();
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "gs graph launch");       // replay: loop runs to convergence on-device
    if (std::getenv("BRAE_GS_DEBUG"))
    {
        cudaStreamSynchronize(cudaStreamPerThread);
        scalar fr;
        int ni;
        cudaMemcpy(&fr, c.gRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ni, c.gIter.data(), sizeof(int), cudaMemcpyDeviceToHost);
        std::printf("    GS[dev] init=%.4e final=%.4e iters=%d (relTol=%.2g)\n", initRes, fr, ni, relTol);
    }
    return initRes;
}
#endif // CUDART_VERSION >= 13000

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
    int nSweeps)
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
    if (turbF32 && !floored && !batched)
    {
        scalar r = deviceSymGaussSeidelF32(A, b, psi, normFactor, tol, relTol, maxIter);
        if (perf) *perf = {r, r, 1};
        return r;
    }   // opt-in path: report init only
    // BRAE_TURB_JACOBI: weighted-Jacobi solve (fully parallel, no colour sync). Experiment only -- Jacobi needs ~2x
    // the sweeps -> ~2x the bandwidth on this BW-bound path, so it is slower than the graphed GS. Same stop test as GS.
    static const bool turbJac = std::getenv("BRAE_TURB_JACOBI") != nullptr;
    if (turbJac && !floored && !batched)
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
    // BRAE_GS_DEVICE: run the convergence loop ON-DEVICE (conditional-graph WHILE node), same per-sweep stop decision,
    // zero per-sweep host D2H. EXACT (not batched): identical sweep count + bit-identical psi vs this host loop.
    static const bool useGraph = std::getenv("BRAE_GS_DEVICE") != nullptr;
#ifdef BRAE_HAS_GS_DEVICE
    if (useGraph && !floored && !batched)
    {
        scalar r = deviceSymGaussSeidelGraph(A, b, psi, normFactor, tol, relTol, maxIter);
        if (perf) *perf = {r, r, 1};
        return r;
    }   // opt-in path: report init only
#else
    if (useGraph)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            std::fprintf(stderr, "brae: BRAE_GS_DEVICE needs CUDA >= 13.0; falling back to host-loop GS\n");
        }
    }
#endif
    const GridColoring& gc = gsColoringFor(A);
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
        for (int s = 0; s < sweepsPer; ++s)
        {
            gsSweep(A, b, psi, gc, true);                                // forward color sweep
            gsSweep(A, b, psi, gc, false);                               // reverse color sweep
        }
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
