// AMG-PCG SOLVER DRIVERS: the flexible-CG / conditional-graph Krylov loops that drive the AMG V-cycle preconditioner.
//   * deviceAMGPCG       -- the single-GPU AMG-PCG (host loop; dispatches to the device-resident graph when enabled).
//   * deviceAMGPCGGraph  -- the whole-loop conditional-graph PCG (BRAE_PCG_DEVICE): the entire steady-state PCG WHILE
//                           body captured once + replayed on-device (brae's ~1.85x-vs-SPUMA perf signature).
// The V-cycle (amgVCycleApply / vcycleAt / vcycleAtF / amgCastFP32) + Galerkin re-coarsening (amgGalerkin) + the
// spectrum estimate (ensureSpectrum) are called across TUs. Verbatim split of device_amg.cu -- no logic change.
#include "inter_phase_time.cuh"
#include "device_amg.cuh"          // AMGData / DeviceSolverPerf / amgGalerkin / amgVCycleApply / deviceAMGPCG decls
#include "device_blas.cuh"          // dot / axpy / reductions
#include "proc_hooks.cuh"          // ProcComm + the hooks' channel (decomposed solves)
#include "device_ldu.cuh"           // DeviceLduView / deviceAmul (the processor-interface hook is in deviceAmul)
#include "device_amg_detail.cuh"
#include "device_pcg.cuh"         // normFactorOnHost, deviceReadScalar    // BRAE_HAS_GS_DEVICE, gsScaleInvK-adjacent constants + env flags + nBlocks/TPB
#include "amg_kernels.cuh"          // zeroT<> etc. (if referenced)
#include "device_amg_internal.cuh"  // LduF/cast_/amulF, gsScaleInvK, ensureSpectrum decl
#include "device_amg_split.cuh"     // BRAE_AMG_PCG_SPLIT
#include <set>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
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
// So the whole pressure solve is ONE graph launch and ONE host read (item 72), exact (same iteration count and psi
// as the host PCG). It used to read three times: the initial residual, the end of iteration 0, and the report -- and
// each read drains a launch queue the driver runs about a millisecond ahead of the GPU (item 65's lesson). Iteration
// 0 differs from the rest only in the search direction (p = w, no beta), so it is no longer a separate host-driven
// step: it is the WHILE body's first execution, with pcgSearchDirK taking the p = w branch on the iteration counter,
// and the initial early-out is the WHILE handle's own start value, set on the device by pcgStartCondK. Both branches
// are the arithmetic the host path ran, statement for statement. The right-hand side and the fine matrix are copied
// into cache-owned buffers because a captured prologue bakes their pointers and the callers hand in fresh ones.
// Non-corrScaling only (flexible-CG falls back to the host loop); normFactor is a device-resident scalar.
// The search direction, with iteration 0's branch on the device. beta is scalarDivK's expression, guard
// included, so a replayed iteration is the host path's iteration to the bit; iteration 0 assigns p = w
// exactly as its deviceCopy did.
__global__
void pcgSearchDirK(
    scalar* __restrict__ p,
    const scalar* __restrict__ w,
    const scalar* __restrict__ num,     // wArA
    const scalar* __restrict__ den,     // wArAold
    const int* __restrict__ iter,
    int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (*iter == 0) { p[i] = w[i]; return; }
    const scalar d = *den;
    const scalar beta = (fabs(d) > scalar(1e-300)) ? (*num) / d : scalar(0);
    p[i] = __dmul_rn(beta, p[i]) + w[i];
}

// The loop's START value: the host used to read the initial residual and decide here. The test is the
// one it applied -- OF's `minIter > 0 || !converged(initialResidual)` -- and it also seeds the report so
// that a solve which runs no iteration at all reports its initial residual as its final one.
__global__
void pcgStartCondK(
    cudaGraphConditionalHandle h,
    const scalar* __restrict__ init,
    scalar tol,
    scalar relTol,
    int minIter,
    int* __restrict__ iter,
    scalar* __restrict__ res)
{
    if (threadIdx.x || blockIdx.x) return;
    *iter = 0;
    const scalar ir = *init;
    *res = ir;
    const bool conv = (ir < tol) || (relTol > 0.0 && ir < relTol * ir);
    cudaGraphSetConditional(h, (minIter > 0 || !conv) ? 1u : 0u);
}

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
// The serial graph's stopping controls, read from the device (PCGGraphCache::sCtl): tol, relTol, maxIter,
// minIter. pcgSetCtlK writes them before each launch; the two condition kernels below are pcgStartCondK and
// pcgSetCondK with the controls read instead of baked in.
__global__
void pcgSetCtlK(
    scalar* ctl,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter)
{
    if (threadIdx.x || blockIdx.x) return;
    ctl[0] = tol;
    ctl[1] = relTol;
    ctl[2] = maxIter;
    ctl[3] = minIter;
}

__global__
void pcgStartCondCtlK(
    cudaGraphConditionalHandle h,
    const scalar* __restrict__ init,
    const scalar* __restrict__ ctl,
    int* __restrict__ iter,
    scalar* __restrict__ res)
{
    if (threadIdx.x || blockIdx.x) return;
    const scalar tol = ctl[0];
    const scalar relTol = ctl[1];
    const int minIter = static_cast<int>(ctl[3]);
    *iter = 0;
    const scalar ir = *init;
    *res = ir;
    const bool conv = (ir < tol) || (relTol > 0.0 && ir < relTol * ir);
    cudaGraphSetConditional(h, (minIter > 0 || !conv) ? 1u : 0u);
}

__global__
void pcgSetCondCtlK(
    cudaGraphConditionalHandle h,
    const scalar* res,
    const scalar* init,
    const scalar* ctl,
    int* iter)
{
    if (threadIdx.x || blockIdx.x) return;
    const scalar tol = ctl[0];
    const scalar relTol = ctl[1];
    const int maxIter = static_cast<int>(ctl[2]);
    const int minIter = static_cast<int>(ctl[3]);
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
    const scalar* dNormFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter)
{
    const int nC = A.nCells;
    amg.corrScaling = false;
    DeviceBuffer<scalar>& wA = amg.wA;
    DeviceBuffer<scalar>& rA = amg.rA;
    // THE FINEST GRID'S ROWS (amgPrepareFineCsr64): made here, outside any capture; their values are filled in the captured prologue
    // below (amgFillFineCsr64) from the graph's own copies of the matrix, as a captured amulKernel reads them, so the graph's key
    // governs both forms alike
    amgPrepareFineCsr64(amg, A);
    if (!amg.pcgCache) amg.pcgCache = std::make_unique<PCGGraphCache>();          // per-solver lifetime: dies with amg, so no stale reuse across solvers
    PCGGraphCache& c = *amg.pcgCache;
    c.pA.resize(nC);
    c.Ax.resize(nC);
    c.sNormF.resize(1);
    c.sInit.resize(1);
    c.sRes.resize(1);
    c.sIter.resize(1);
    const int nF = A.nInternalFaces;
    // One off-diagonal handed, one copied: the second copy held the same values for the solver's life, a face
    // field in every phase (208 MiB of boxy5's footprint at 4.94 M cells, the BRAE_POOL_STATS class census).
    const bool oneOff = A.lower == A.upper;
    c.gB.resize(nC);
    c.gDiag.resize(nC);
    c.gUpper.resize(nF);
    // Grown when a view brings a lower of its own and never given back: a graph captured against gLower may still
    // exist when the next view shares its off-diagonals, and no captured graph may name a freed block. The key
    // below then recaptures that solve against gUpper.
    if (!oneOff)
    {
        c.gLower.resize(nF);
    }
    cudaMemcpyAsync(c.sNormF.data(), dNormFactor, sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    // this solve's right-hand side and fine matrix into the graph-referenced buffers (async D2D, no sync)
    cudaMemcpyAsync(c.gB.data(),     b.data(), (std::size_t)nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gDiag.data(),  A.diag,   (std::size_t)nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    if (nF > 0)
    {
        cudaMemcpyAsync(c.gUpper.data(), A.upper, (std::size_t)nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        if (!oneOff)
        {
            cudaMemcpyAsync(c.gLower.data(), A.lower, (std::size_t)nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        }
    }
    DeviceLduView sA = A;
    sA.diag  = c.gDiag.data();
    sA.upper = c.gUpper.data();
    sA.lower = oneOff ? c.gUpper.data() : c.gLower.data();
    // the pair's per-solve values into graph-owned buffers, as the matrix above (PCGGraphCache)
    if (A.nCyc > 0)
    {
        c.gCycCoeff.resize(static_cast<std::size_t>(A.nCyc));
        cudaMemcpyAsync(c.gCycCoeff.data(), A.cycCoeff, (std::size_t)A.nCyc*sizeof(scalar), cudaMemcpyDeviceToDevice,
                        cudaStreamPerThread);
        sA.cycCoeff = c.gCycCoeff.data();
        if (A.cycJump)
        {
            c.gCycJump.resize(static_cast<std::size_t>(A.nCyc));
            cudaMemcpyAsync(c.gCycJump.data(), A.cycJump, (std::size_t)A.nCyc*sizeof(scalar),
                            cudaMemcpyDeviceToDevice, cudaStreamPerThread);
            sA.cycJump = c.gCycJump.data();
        }
    }
    if (A.nAmi > 0)
    {
        c.gAmiIfc.resize(static_cast<std::size_t>(A.nAmi));
        cudaMemcpyAsync(c.gAmiIfc.data(), A.amiIfc, (std::size_t)A.nAmi*sizeof(scalar), cudaMemcpyDeviceToDevice,
                        cudaStreamPerThread);
        sA.amiIfc = c.gAmiIfc.data();
    }
    c.sCtl.resize(4);
    pcgSetCtlK<<<1,1,0,cudaStreamPerThread>>>(c.sCtl.data(), tol, relTol, maxIter, minIter);
    scalar* dWArA = amg.sWArA.data();
    scalar* dWArAold = amg.sWArAold.data();
    scalar* dPap  = amg.sPap.data();
    scalar* dAlpha   = amg.sAlpha.data();
    scalar* dNegAlpha = amg.sNegAlpha.data();
    ensureSpectrum(amg, A);                                                       // one-time Chebyshev spectrum (pre-capture)
    // FP32 V-cycle inside the device-resident PCG: cast matrices once per solve; the WHILE body captures the FP32
    // vcycleAtF automatically (host-scalar-free). Outer Krylov + residual stay FP64 (accuracy preserved).
    const bool fp32 = amgSinglePrecisionCycle(amg);
    if (fp32) amgCastFP32(amg, A);
    const LduF A0 = fp32 ? lduF(sA, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]) : LduF{};
    auto applyPrec = [&]()
    {
        if (fp32)
        {
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, sA, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), wA.data());
        }
        else vcycleAt(0, amg, sA, rA, wA);
    };
    DeviceSolverPerf perf;
    const int epoch = deviceReductionScratchEpoch();
    const bool samePair = c.keyNCyc == A.nCyc && c.keyNAmi == A.nAmi && c.keyJump == (A.cycJump != nullptr)
                       && c.keyCycOwn == (const void*)A.cycOwn && c.keyCycNbr == (const void*)A.cycNbr
                       && c.keyAmiOwn == (const void*)A.amiOwn && c.keyAmiOff == (const void*)A.amiOff
                       && c.keyAmiNbr == (const void*)A.amiNbr && c.keyAmiW == (const void*)A.amiW;
    if (!c.exec || c.key != psi.data() || c.keyEpoch != epoch
        || c.keyOwner != (const void*)A.owner || c.keyNC != nC || c.keyNF != nF
        || c.keyAddressingId != A.addressingId || !samePair || c.keyPairEpoch != amg.pair.epoch
        // BRAE_CONTROL_PCG_GRAPH_LOWER_UNKEYED=1, a gate's CONTROL, deliberately wrong: the lower is left out of the
        // key and a replay reads the other view's (tests/test_amg_one_off_diagonal.cu). Read only when the lower moved.
        || (c.keyLower != (const void*)sA.lower && !std::getenv("BRAE_CONTROL_PCG_GRAPH_LOWER_UNKEYED")))
    {
        interPhase::Nested timedCapture("AMG-PCG: graph capture");
        // PRE-SIZE everything the capture will touch (the V-cycle scratch, pA/Ax, the reduction
        // partials): a capture must allocate nothing. ON THE CAPTURE ONLY -- doing it per solve, as the
        // first version of this did, repeats the prologue and a whole V-cycle whose result is thrown
        // away, and cost more than the two reads it saved (T3A 15.0 -> 17.4 ms/iter, measured).
        // A*psi with `onField`: psi IS the solution field, the one product a jump cyclic applies its jump to
        // (device_pcg.cu's own initial residual). Without it a porousBafflePressure pair was solved with no jump:
        // RAS/damBreakPorousBaffle, phi 5.6e-03 from OpenFOAM.
        amgFillFineCsr64(amg, sA);
        deviceAmul(sA, psi, c.Ax, /*onField=*/true);
        deviceCopy(rA, c.gB);
        deviceAxpy(-1.0, c.Ax, rA);
        deviceSumMagInto(rA, c.sInit.data());
        applyPrec();
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
        // the loop's start value is set on the device by pcgStartCondK below, so the handle defaults to
        // "do not run" -- a solve whose initial residual already passes runs no iteration at all
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 0, cudaGraphCondAssignDefault), "pcg cond handle");
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, c.graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "pcg prologue capture");
        // the prologue: r = b - A psi, the initial residual, and the loop's start decision
        // A*psi with `onField`: psi IS the solution field, the one product a jump cyclic applies its jump to
        // (device_pcg.cu's own initial residual). Without it a porousBafflePressure pair was solved with no jump:
        // RAS/damBreakPorousBaffle, phi 5.6e-03 from OpenFOAM.
        amgFillFineCsr64(amg, sA);
        deviceAmul(sA, psi, c.Ax, /*onField=*/true);
        deviceCopy(rA, c.gB);
        deviceAxpy(-1.0, c.Ax, rA);
        deviceSumMagInto(rA, c.sInit.data());
        gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sInit.data(), c.sNormF.data());
        pcgStartCondCtlK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.sInit.data(), c.sCtl.data(), c.sIter.data(),
                                                        c.sRes.data());
        cudaStreamCaptureStatus st;
        const cudaGraphNode_t* deps = nullptr;
        std::size_t nDeps = 0;
        cudaCheck(braeStreamGetCaptureInfo(cudaStreamPerThread, &st, &deps, &nDeps), "pcg capture info");
        std::vector<cudaGraphNode_t> depv(deps, deps + nDeps);
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "pcg prologue capture end");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(braeGraphAddNode(&cnode, c.graph, depv.data(), depv.size(), &cp), "pcg cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "pcg capture begin");
        // ONE body for every iteration: iteration 0 takes pcgSearchDirK's p = w branch.
        deviceScalarCopy(dWArA, dWArAold);
        applyPrec();                                     // wA = M^-1 rA
        deviceDotInto(wA, rA, dWArA);
        pcgSearchDirK<<<nBlocks(nC),TPB,0,cudaStreamPerThread>>>(c.pA.data(), wA.data(), dWArA, dWArAold,
                                                                 c.sIter.data(), nC);
        amgFineAmul(amg, sA, c.pA, wA);
        deviceDotInto(wA, c.pA, dPap);
        deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
        deviceAxpyDev(dAlpha, c.pA, psi);
        deviceAxpyDev(dNegAlpha, wA, rA);
        deviceSumMagInto(rA, c.sRes.data());
        gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sRes.data(), c.sNormF.data());   // normalized residual
        pcgSetCondCtlK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.sRes.data(), c.sInit.data(), c.sCtl.data(),
                                                      c.sIter.data());
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "pcg capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "pcg graph instantiate");
        c.key = psi.data();
        c.keyNCyc = A.nCyc;
        c.keyNAmi = A.nAmi;
        c.keyJump = A.cycJump != nullptr;
        c.keyCycOwn = A.cycOwn;
        c.keyCycNbr = A.cycNbr;
        c.keyAmiOwn = A.amiOwn;
        c.keyAmiOff = A.amiOff;
        c.keyAmiNbr = A.amiNbr;
        c.keyAmiW = A.amiW;
        c.keyPairEpoch = amg.pair.epoch;
        c.keyEpoch = epoch;
        c.keyOwner = A.owner;
        c.keyAddressingId = A.addressingId;
        c.keyNC = nC;
        c.keyNF = nF;
        c.keyLower = sA.lower;
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "pcg graph launch");
    // the ONE read: the report OpenFOAM prints, after the whole solve has run on the device
    scalar hRes[2];
    int nIter;
    // The whole report in ONE publish and ONE wait, in the order the three copies had. The capture ended
    // above and the graph was LAUNCHED here, so this read is outside any capture (the mailbox refuses one
    // by name). BRAE_READ_SCALAR_SYNC=1 restores the three copies and the sync.
    const DeviceReadValue repV[3] =
    {
        {c.sInit.data(), &hRes[0], false},
        {c.sRes.data(), &hRes[1], false},
        {c.sIter.data(), &nIter, true}
    };
    deviceReadValues(repV, 3);
    perf.initialResidual = hRes[0];
    perf.finalResidual   = hRes[1];
    perf.nIterations     = nIter;
    static const bool dbgCyc = std::getenv("BRAE_AMG_CYCLES") != nullptr;
    if (dbgCyc) std::fprintf(stderr, "[AMG] p cycles=%d finalRes=%.3e (device)\n", perf.nIterations, perf.finalResidual);
    return perf;
}

#endif // BRAE_HAS_GS_DEVICE

// BRAE_AMG_PCG_SPLIT: a split solve's parts into the phase table -- a line a grid's part, the Krylov loop's
// own after them -- and, once a solve name, the hierarchy's shape, which is what the parts' sizes follow
static void chargeSplit(
    const DeviceLduView& A,
    const AMGData& amg,
    bool singlePrecisionCycle,
    int nIter)
{
    amgSplit::State& s = amgSplit::state();
    s.active = false;
    static const char* const partName[amgSplit::nParts] =
        {"pre-smooth", "residual", "restriction", "coarsest solve", "prolongation", "post-smooth"};
    static const char* const krylovName[amgSplit::nKrylov] =
        {"the first residual", "the matrices cast to single precision", "r and w cast (single-precision cycle)",
         "A p", "dots, p, psi and r", "the residual norm, read"};
    const int nGrids = amg.nLevels() + 1;
    static std::set<std::string> shown;
    if (shown.insert(s.name).second)
    {
        std::printf("  AMG-PCG split of %s (BRAE_AMG_PCG_SPLIT): the solve leaves the captured graph and waits "
                    "after every part -- an instrument, not a measured run.\n", s.name);
        std::printf("    its %d grids, %s aggregation, %s-precision cycle, cells / faces%s:\n", nGrids,
                    amg.saSmooth ? "smoothed" : "plain", singlePrecisionCycle ? "single" : "double",
                    amg.saSmooth ? " / prolongator entries to the next" : "");
        for (int g = 0; g < nGrids; ++g)
        {
            const int cells = g == 0 ? A.nCells : amg.level[g - 1].nCoarse;
            const int faces = g == 0 ? A.nInternalFaces : amg.level[g - 1].nCoarseFaces;
            if (amg.saSmooth && g < amg.nLevels())
            {
                std::printf("      grid %2d  %9d / %9d / %zu\n", g, cells, faces, amg.level[g].Pval.size());
            }
            else
            {
                std::printf("      grid %2d  %9d / %9d\n", g, cells, faces);
            }
        }
    }
    char label[160];
    double all = 0.0;
    for (int g = 0; g < nGrids && g < amgSplit::MAX_GRIDS; ++g)
    {
        for (int p = 0; p < amgSplit::nParts; ++p)
        {
            // the coarsest grid is solved and nothing else; every other grid is everything but solved
            if ((g == amg.nLevels()) != (p == amgSplit::coarsest)) continue;
            std::snprintf(label, sizeof(label), "%s split: grid %d %s", s.name, g, partName[p]);
            interPhase::charge(label, s.grid[g][p], nIter);
            all += s.grid[g][p];
        }
    }
    for (int k = 0; k < amgSplit::nKrylov; ++k)
    {
        // the two casts are the single-precision cycle's
        if (!singlePrecisionCycle && (k == amgSplit::matrixCast || k == amgSplit::vectorCasts)) continue;
        std::snprintf(label, sizeof(label), "%s split: Krylov, %s", s.name, krylovName[k]);
        interPhase::charge(label, s.krylov[k], k <= amgSplit::matrixCast ? 1 : nIter);
        all += s.krylov[k];
    }
    std::snprintf(label, sizeof(label), "%s split: ALL PARTS, waited for one by one", s.name);
    interPhase::charge(label, all, 1);
}

namespace {
std::vector<scalar>* g_pcgHistory = nullptr;
// BRAE_PROC_PCG_UNFUSED=1: a decomposed solve takes the plain loop below (3 collectives per iteration) where it
// would take the fused two-collective one. The two are bit-identical; the plain one is the fused one's oracle.
bool procPcgUnfused()
{
    return std::getenv("BRAE_PROC_PCG_UNFUSED") != nullptr;
}
}   // namespace

void amgPcgSetHistory(std::vector<scalar>* history)
{
    g_pcgHistory = history;
}

// AMG-PCG ON A DECOMPOSED MESH (A.proc.active()): OpenFOAM -parallel's PCG, with a per-rank block-local V-cycle as the
// preconditioner.
//   * the product A*p exchanges p's halo first (deviceAmul, through A.proc), the V-cycle runs on the block (vcycleAt
//     strips the hook), every dot and |r| is the global one (reductions.cu hook), so EVERY RANK TAKES THE SAME
//     DECISION from the same allreduced residual and therefore the same iteration count;
//   * normFactor is the caller's: the global one (deviceNormFactor on the same view).
// Host-driven: the decision is read once per iteration. This is the loop of every backend but NCCL and the oracle of
// deviceAMGPCGProcGraph below, which runs the same iteration as one CUDA graph (the collectives need a graph-capturable
// communicator: the NCCL backend's graph channel; MPI-host throws inside a capture, and on the loopback backend all
// ranks' bodies would have to be one joint graph, which a per-thread conditional node cannot be).
//
// FUSED: two collectives per iteration, not three. The last collective of an iteration carries sum|r| (the
// decision) together with the NEXT iteration's z.r, which needs the next preconditioner application and so a
// V-cycle that the converging iteration does not use (one V-cycle per solve thrown away). Every number is the one
// the plain loop (BRAE_PROC_PCG_UNFUSED) computes: the same local partials, folded in the same rank order.
static DeviceSolverPerf deviceAMGPCGProc(
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
    ProcComm& comm = *A.proc.comm;
    const int nC = A.nCells;
    amg.corrScaling = false;
    const DeviceLduView Aloc = deviceBlockLocal(A);
    DeviceBuffer<scalar>& wA = amg.wA;
    DeviceBuffer<scalar>& rA = amg.rA;
    DeviceBuffer<scalar> pA(static_cast<std::size_t>(nC), deviceProcHalo(A.proc));    // the operand of A*p: halo tail
    DeviceBuffer<scalar> Ax(nC), pair(2);
    deviceAmul(A, psi, Ax, /*onField=*/true);                                       // halo of psi, then A*psi
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);
    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA)/normFactor;                             // global
    perf.finalResidual = perf.initialResidual;
    auto converged = [&](scalar fr){ return (fr<tol) || (relTol>0.0 && fr<relTol*perf.initialResidual); };
    std::vector<scalar>* hist = g_pcgHistory;
    if (hist) hist->assign(1, perf.initialResidual);
    ensureSpectrum(amg, Aloc);
    const bool fp32 = amgSinglePrecisionCycle(amg);
    if (fp32) amgCastFP32(amg, Aloc);
    auto applyPrecond = [&]()
    {
        if (fp32)
        {
            const LduF A0 = lduF(Aloc, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, Aloc, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), wA.data());
        }
        else vcycleAt(0, amg, Aloc, rA, wA);
    };
    scalar* dWArA    = pair.data();                 // pair = { z.r, sum|r| }: one collective carries both
    scalar* dSumMag  = pair.data() + 1;
    scalar* dWArAold = amg.sWArAold.data();
    scalar* dPap     = amg.sPap.data();
    scalar* dAlpha   = amg.sAlpha.data();
    scalar* dNegAlpha= amg.sNegAlpha.data();
    scalar* dBeta    = amg.sBeta.data();
    int nIter = 0;
    if (minIter > 0 || !converged(perf.finalResidual))
    {
        applyPrecond();                                     // wA = M^-1 rA
        deviceDotInto(wA, rA, dWArA);                       // global z.r (the first iteration's)
        do
        {
            if (nIter == 0) deviceCopy(pA, wA);             // p = w
            else
            {
                deviceScalarDiv(dWArA, dWArAold, dBeta);    // Fletcher-Reeves beta
                deviceFusedScaleAxpy(pA, dBeta, wA);
            }
            deviceAmul(A, pA, wA);                          // halo of p, then A*p
            deviceDotInto(wA, pA, dPap);                    // collective 1: global p.Ap
            deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
            deviceAxpyDev(dAlpha, pA, psi);
            deviceAxpyDev(dNegAlpha, wA, rA);
            ++nIter;
            deviceScalarCopy(dWArA, dWArAold);
            applyPrecond();                                 // the NEXT iteration's z (unused if this one converges)
            {
                LocalReductions localOnly;
                deviceDotInto(wA, rA, dWArA);
                deviceSumMagInto(rA, dSumMag);
            }
            ++procHookCountsRef().reductions;
            comm.allReduce(pair.data(), 2, ReduceOp::Sum, cudaStreamPerThread, procChannel());   // collective 2
            perf.finalResidual = deviceReadScalar(dSumMag)/normFactor;
            if (hist) hist->push_back(perf.finalResidual);
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
    }
    perf.nIterations = nIter;
    return perf;
}

#ifdef BRAE_HAS_GS_DEVICE
// The decomposed solve's stop test, on the device: the iteration counter, the normalised global residual
// res = sum|r| / normFactor (the second element of the reduced pair, the division the host loop does), its history
// slot (test hook) and the WHILE condition. Every rank evaluates it on the same reduced numbers, so every rank
// leaves the loop at the same iteration.
__global__
void pcgProcStopK(
    cudaGraphConditionalHandle h,
    const scalar* __restrict__ pair,
    const scalar* __restrict__ normF,
    const scalar* __restrict__ init,
    const scalar* __restrict__ ctl,
    int* __restrict__ iter,
    scalar* __restrict__ resOut,
    scalar* __restrict__ hist,
    int histCap)
{
    if (threadIdx.x || blockIdx.x) return;
    const scalar tol = ctl[0];
    const scalar relTol = ctl[1];
    const int maxIter = static_cast<int>(ctl[2]);
    const int minIter = static_cast<int>(ctl[3]);
    const int it = ++(*iter);
    const scalar fr = pair[1] / (*normF);
    *resOut = fr;
    if (hist && it <= histCap) hist[it] = fr;
    const bool conv = (fr < tol) || (relTol > 0.0 && fr < relTol * (*init));
    const bool stop = (conv || it >= maxIter) && it >= minIter;
    cudaGraphSetConditional(h, stop ? 0u : 1u);
}

// AMG-PCG ON A DECOMPOSED MESH AS ONE CUDA GRAPH, on the NCCL backend (procGraphLoops): the fused two-collective
// iteration of deviceAMGPCGProc as the body of a conditional WHILE node. Every rank captures ITS OWN graph; the body
// holds the halo exchange of p, the local kernels, the block-local V-cycle and the two allgather-based deterministic
// reductions (graph communicator, device-resident results) and decides continue/stop from the reduced residual, so
// every rank runs the same count. Same statements as the host loop, in the same order, on the same reduced values:
// iteration count, residual history and solution are bit-identical to it (tests/test_dist_solvers.cu).
//   * no collective is issued outside the graph: the pre-capture warm-up is rank-local (LocalReductions, block view),
//     so ranks may disagree on WHETHER to re-capture without hanging each other;
//   * the zeroth precondition + global z.r sit in the prologue (the host loop runs them only when it enters the loop;
//     a solve that is already converged pays one speculative V-cycle and one collective);
//   * the host reads the report once (initial residual, final residual, count), as the single-GPU graph does.
static DeviceSolverPerf deviceAMGPCGProcGraph(
    const DeviceLduView& A,
    AMGData& amg,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    const scalar* dNormFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter)
{
    ProcComm& comm = *A.proc.comm;
    const int nC = A.nCells;
    amg.corrScaling = false;
    DeviceBuffer<scalar>& wA = amg.wA;
    DeviceBuffer<scalar>& rA = amg.rA;
    if (!amg.pcgCache) amg.pcgCache = std::make_unique<PCGGraphCache>();
    PCGGraphCache& c = *amg.pcgCache;
    c.pA.setTail(deviceProcHalo(A.proc));                                       // the operand of A*p: halo tail
    c.pA.resize(nC);
    c.Ax.resize(nC);
    c.sNormF.resize(1);
    c.sInit.resize(1);
    c.sRes.resize(1);
    c.sIter.resize(1);
    c.procPair.resize(2);
    const int nF = A.nInternalFaces;
    c.gB.resize(nC);
    c.gDiag.resize(nC);
    c.gUpper.resize(nF);
    c.gLower.resize(nF);
    cudaMemcpyAsync(c.sNormF.data(), dNormFactor, sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gB.data(),     b.data(), (std::size_t)nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(c.gDiag.data(),  A.diag,   (std::size_t)nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    if (nF > 0)
    {
        cudaMemcpyAsync(c.gUpper.data(), A.upper, (std::size_t)nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(c.gLower.data(), A.lower, (std::size_t)nF*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    }
    DeviceLduView sA = A;
    sA.diag  = c.gDiag.data();
    sA.upper = c.gUpper.data();
    sA.lower = c.gLower.data();
    if (A.nCyc > 0)
    {
        c.gCycCoeff.resize(static_cast<std::size_t>(A.nCyc));
        cudaMemcpyAsync(c.gCycCoeff.data(), A.cycCoeff, (std::size_t)A.nCyc*sizeof(scalar), cudaMemcpyDeviceToDevice,
                        cudaStreamPerThread);
        sA.cycCoeff = c.gCycCoeff.data();
        if (A.cycJump)
        {
            c.gCycJump.resize(static_cast<std::size_t>(A.nCyc));
            cudaMemcpyAsync(c.gCycJump.data(), A.cycJump, (std::size_t)A.nCyc*sizeof(scalar),
                            cudaMemcpyDeviceToDevice, cudaStreamPerThread);
            sA.cycJump = c.gCycJump.data();
        }
    }
    if (A.nAmi > 0)
    {
        c.gAmiIfc.resize(static_cast<std::size_t>(A.nAmi));
        cudaMemcpyAsync(c.gAmiIfc.data(), A.amiIfc, (std::size_t)A.nAmi*sizeof(scalar), cudaMemcpyDeviceToDevice,
                        cudaStreamPerThread);
        sA.amiIfc = c.gAmiIfc.data();
    }
    if (A.proc.nFaces > 0)                                                      // the interface coefficient, per solve
    {
        c.gProcCoeff.resize(static_cast<std::size_t>(A.proc.nFaces));
        cudaMemcpyAsync(c.gProcCoeff.data(), A.proc.coeff, (std::size_t)A.proc.nFaces*sizeof(scalar),
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        sA.proc.coeff = c.gProcCoeff.data();
    }
    const DeviceLduView Aloc = deviceBlockLocal(sA);
    const std::size_t histCap = g_pcgHistory ? static_cast<std::size_t>(std::max(maxIter, 1)) : 0;
    if (histCap) c.dHist.resize(histCap + 1);
    c.sCtl.resize(4);
    pcgSetCtlK<<<1,1,0,cudaStreamPerThread>>>(c.sCtl.data(), tol, relTol, maxIter, minIter);
    scalar* dWArA     = c.procPair.data();                // pair = { z.r, sum|r| }: one collective carries both
    scalar* dSumMag   = c.procPair.data() + 1;
    scalar* dWArAold  = amg.sWArAold.data();
    scalar* dPap      = amg.sPap.data();
    scalar* dAlpha    = amg.sAlpha.data();
    scalar* dNegAlpha = amg.sNegAlpha.data();
    ensureSpectrum(amg, Aloc);                                                  // host scalars: before any capture
    const bool fp32 = amgSinglePrecisionCycle(amg);
    if (fp32) amgCastFP32(amg, Aloc);
    const LduF A0 = fp32 ? lduF(Aloc, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]) : LduF{};
    auto applyPrec = [&]()
    {
        if (fp32)
        {
            cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, rA.data(), amg.vBF[0].data());
            vcycleAtF(0, amg, Aloc, A0, amg.vBF[0].data(), amg.vXF[0].data());
            cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), wA.data());
        }
        else vcycleAt(0, amg, Aloc, rA, wA);
    };
    const int epoch = deviceReductionScratchEpoch();
    const bool samePair = c.keyNCyc == A.nCyc && c.keyNAmi == A.nAmi && c.keyJump == (A.cycJump != nullptr)
                       && c.keyCycOwn == (const void*)A.cycOwn && c.keyCycNbr == (const void*)A.cycNbr
                       && c.keyAmiOwn == (const void*)A.amiOwn && c.keyAmiOff == (const void*)A.amiOff
                       && c.keyAmiNbr == (const void*)A.amiNbr && c.keyAmiW == (const void*)A.amiW;
    const bool sameProc = c.keyProcComm == (const void*)A.proc.comm && c.keyProcPlan == (const void*)A.proc.plan
                       && c.keyProcFaceCell == (const void*)A.proc.faceCell && c.keyProcNbrIdx == (const void*)A.proc.nbrIdx
                       && c.keyProcRank == (const void*)A.proc.rank && c.keyProcFaces == A.proc.nFaces;
    const char* recaptureWhy = !c.exec ? "first" : c.key != psi.data() ? "psi" : c.keyEpoch != epoch ? "scratchEpoch"
        : c.keyOwner != (const void*)A.owner ? "owner" : (c.keyNC != nC || c.keyNF != nF) ? "size"
        : c.keyAddressingId != A.addressingId ? "addressingId" : !samePair ? "pair" : c.keyPairEpoch != amg.pair.epoch ? "pairEpoch"
        : !sameProc ? "proc" : c.keyPA != (const void*)c.pA.data() ? "pA" : c.keyHistCap != static_cast<int>(histCap) ? "histCap" : nullptr;
    if (recaptureWhy)
    {
        static const bool log = std::getenv("BRAE_GRAPH_RECAPTURE_LOG") != nullptr;
        if (log) std::fprintf(stderr, "[recapture] PCG proc graph: %s (cache %p)\n", recaptureWhy, static_cast<void*>(&c));
        interPhase::Nested timedCapture("AMG-PCG: graph capture");
        {
            // PRE-SIZE what the capture touches (V-cycle scratch, the reduction partials): rank-local only, no halo,
            // no collective, so a rank that re-captures never waits for one that does not
            LocalReductions localOnly;
            deviceAmul(Aloc, psi, c.Ax, /*onField=*/true);
            deviceCopy(rA, c.gB);
            deviceAxpy(-1.0, c.Ax, rA);
            deviceSumMagInto(rA, c.sInit.data());
            deviceDotInto(wA, rA, dPap);
            applyPrec();
        }
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
        scalar* hist = histCap ? c.dHist.data() : nullptr;
        cudaCheck(cudaGraphCreate(&c.graph, 0), "pcg proc graph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 0, cudaGraphCondAssignDefault), "pcg proc cond handle");
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, c.graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "pcg proc prologue capture");
        {
            ProcChannelScope onGraph(CommChannel::Graph);
            // the prologue: r = b - A psi (halo of psi), the global initial residual, the loop's start decision,
            // then the first preconditioner application and the global z.r
            deviceAmul(sA, psi, c.Ax, /*onField=*/true);
            deviceCopy(rA, c.gB);
            deviceAxpy(-1.0, c.Ax, rA);
            deviceSumMagInto(rA, c.sInit.data());
            gsScaleInvK<<<1,1,0,cudaStreamPerThread>>>(c.sInit.data(), c.sNormF.data());
            pcgStartCondCtlK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.sInit.data(), c.sCtl.data(), c.sIter.data(),
                                                            c.sRes.data());
            if (hist) cudaMemcpyAsync(hist, c.sInit.data(), sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
            applyPrec();
            deviceDotInto(wA, rA, dWArA);                                       // global z.r (the first iteration's)
        }
        cudaStreamCaptureStatus st;
        const cudaGraphNode_t* deps = nullptr;
        std::size_t nDeps = 0;
        cudaCheck(braeStreamGetCaptureInfo(cudaStreamPerThread, &st, &deps, &nDeps), "pcg proc capture info");
        std::vector<cudaGraphNode_t> depv(deps, deps + nDeps);
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "pcg proc prologue capture end");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(braeGraphAddNode(&cnode, c.graph, depv.data(), depv.size(), &cp), "pcg proc cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "pcg proc capture begin");
        {
            ProcChannelScope onGraph(CommChannel::Graph);
            // ONE body for every iteration: iteration 0 takes pcgSearchDirK's p = w branch
            pcgSearchDirK<<<nBlocks(nC),TPB,0,cudaStreamPerThread>>>(c.pA.data(), wA.data(), dWArA, dWArAold,
                                                                     c.sIter.data(), nC);
            deviceAmul(sA, c.pA, wA);                                           // halo of p (graph communicator), then A*p
            deviceDotInto(wA, c.pA, dPap);                                      // collective 1: global p.Ap
            deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
            deviceAxpyDev(dAlpha, c.pA, psi);
            deviceAxpyDev(dNegAlpha, wA, rA);
            deviceScalarCopy(dWArA, dWArAold);
            applyPrec();                                                        // the NEXT iteration's z
            {
                LocalReductions localOnly;
                deviceDotInto(wA, rA, dWArA);
                deviceSumMagInto(rA, dSumMag);
            }
            comm.allReduce(c.procPair.data(), 2, ReduceOp::Sum, cudaStreamPerThread, CommChannel::Graph);   // collective 2
            pcgProcStopK<<<1,1,0,cudaStreamPerThread>>>(c.handle, c.procPair.data(), c.sNormF.data(), c.sInit.data(),
                                                        c.sCtl.data(), c.sIter.data(), c.sRes.data(), hist,
                                                        static_cast<int>(histCap));
        }
        cudaCheck(cudaStreamEndCapture(cudaStreamPerThread, &tmp), "pcg proc capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "pcg proc graph instantiate");
        c.key = psi.data();
        c.keyNCyc = A.nCyc;
        c.keyNAmi = A.nAmi;
        c.keyJump = A.cycJump != nullptr;
        c.keyCycOwn = A.cycOwn;
        c.keyCycNbr = A.cycNbr;
        c.keyAmiOwn = A.amiOwn;
        c.keyAmiOff = A.amiOff;
        c.keyAmiNbr = A.amiNbr;
        c.keyAmiW = A.amiW;
        c.keyPairEpoch = amg.pair.epoch;
        c.keyEpoch = epoch;
        c.keyOwner = A.owner;
        c.keyAddressingId = A.addressingId;
        c.keyNC = nC;
        c.keyNF = nF;
        c.keyProcComm = A.proc.comm;
        c.keyProcPlan = A.proc.plan;
        c.keyProcFaceCell = A.proc.faceCell;
        c.keyProcNbrIdx = A.proc.nbrIdx;
        c.keyProcRank = A.proc.rank;
        c.keyProcFaces = A.proc.nFaces;
        c.keyPA = c.pA.data();
        c.keyHistCap = static_cast<int>(histCap);
        if (c.registeredComm != (const void*)A.proc.comm)
        {
            if (c.registeredComm) procGraphUnregister(static_cast<const ProcComm*>(c.registeredComm), &c);
            procGraphRegister(A.proc.comm, [](void* p){ destroyPcgGraphs(static_cast<PCGGraphCache*>(p)); }, &c);
            c.registeredComm = A.proc.comm;
        }
    }
    cudaCheck(cudaGraphLaunch(c.exec, cudaStreamPerThread), "pcg proc graph launch");
    scalar hRes[2];
    int nIter;
    const DeviceReadValue repV[3] =
    {
        {c.sInit.data(), &hRes[0], false},
        {c.sRes.data(), &hRes[1], false},
        {c.sIter.data(), &nIter, true}
    };
    deviceReadValues(repV, 3);
    DeviceSolverPerf perf;
    perf.initialResidual = hRes[0];
    perf.finalResidual   = hRes[1];
    perf.nIterations     = nIter;
    if (histCap)
    {
        const std::size_t n = static_cast<std::size_t>(std::min<int>(nIter, static_cast<int>(histCap))) + 1;
        g_pcgHistory->resize(n);
        cudaCheck(cudaMemcpyAsync(g_pcgHistory->data(), c.dHist.data(), n*sizeof(scalar), cudaMemcpyDeviceToHost, cudaStreamPerThread), "pcg proc history");
        cudaCheck(cudaStreamSynchronize(cudaStreamPerThread), "pcg proc history sync");
    }
    return perf;
}

// Does this decomposed solve take the whole-loop graph? NCCL backend, the device-resident pressure solve not opted out
// (BRAE_PCG_DEVICE=0), BRAE_PROC_GRAPH not 0 (proc_hooks.cuh procGraphLoops).
static bool procPcgGraphWanted(const DeviceLduView& A)
{
    return procGraphLoops(*A.proc.comm) && envFlag("BRAE_PCG_DEVICE", true) && braeCondGraphsAvailable();
}
#endif // BRAE_HAS_GS_DEVICE

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
    // the matrix's coupled pair onto every grid of the hierarchy, this solve's coefficients (AMGPair)
    amgCouplePair(amg, A);
    // A decomposed mesh: the fused global-decision loop, unless asked for the plain one or the options only it has
    if (A.proc.active() && !procPcgUnfused() && !corrScaling && checkEvery <= 1)
    {
#ifdef BRAE_HAS_GS_DEVICE
        if (procPcgGraphWanted(A))                                  // the NCCL graph communicator: the whole loop in one graph
        {
            static thread_local auto& dNf = *new DeviceBuffer<scalar>(1);
            cudaMemcpyAsync(dNf.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice, cudaStreamPerThread);
            return deviceAMGPCGProcGraph(A, amg, b, psi, dNf.data(), tol, relTol, maxIter, minIter);
        }
#endif
        return deviceAMGPCGProc(A, amg, b, psi, normFactor, tol, relTol, maxIter, minIter);
    }
    // BRAE_AMG_PCG_SPLIT: this solve takes the plain loop below, uncaptured, the residual read at every
    // iteration as the graph reads it, with a lap after every part (device_amg_split.cuh)
    const bool split = amgSplit::wanted();
    if (split)
    {
        captureVcycle = false;
        checkEvery = 1;
    }
#ifdef BRAE_HAS_GS_DEVICE
    {
        static const bool pcgDev = envFlag("BRAE_PCG_DEVICE", true);                 // device-resident pressure solve (default ON; opt out BRAE_PCG_DEVICE=0)
        if (pcgDev && braeCondGraphsAvailable() && !corrScaling && !split && !A.proc.active())
        {
            static thread_local auto& dNf = *new DeviceBuffer<scalar>(1);
            cudaMemcpyAsync(dNf.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice, cudaStreamPerThread);
            return deviceAMGPCGGraph(A, amg, b, psi, dNf.data(), tol, relTol, maxIter, minIter);
        }
    }
#endif
    const int nC = A.nCells;
    amg.corrScaling = corrScaling;                             // seen by vcycleAt (scaled prolongation)
    // the finest grid's rows in the plain loop only when its V-cycle is launched: a captured cycle bakes the view it was handed, and
    // its gate (tests/interfoam_write/amg_pcg/captured_cycle_view.sh) holds that cycle's keying on amulKernel's terms
    if (captureVcycle) amg.csr64FineKey = nullptr;
    else amgGatherFineCsr64(amg, A);
    DeviceBuffer<scalar>& wA = amg.wA;                          // persistent (fixed addr) so the captured graph stays valid
    DeviceBuffer<scalar>& rA = amg.rA;
    DeviceBuffer<scalar> pA(static_cast<std::size_t>(nC), deviceProcHalo(A.proc)), Ax(nC), rOld(nC);   // rOld: previous residual for flexible-CG beta (corrScaling); pA: halo tail on a decomposed mesh
    if (split) amgSplit::begin();
    // A*psi with `onField`: psi IS the solution field, the one product a jump cyclic applies its jump to
    deviceAmul(A, psi, Ax, /*onField=*/true);
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);
    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA)/normFactor;
    perf.finalResidual = perf.initialResidual;
    auto converged = [&](scalar fr){ return (fr<tol) || (relTol>0.0 && fr<relTol*perf.initialResidual); };
    std::vector<scalar>* procHist = A.proc.active() ? g_pcgHistory : nullptr;       // the decomposed solve's test hook
    if (procHist) procHist->assign(1, perf.initialResidual);

    amgSplit::lap(amgSplit::prologue);
    // One-time Chebyshev spectrum estimate (host scalars -> must precede any graph capture); no-op unless Chebyshev.
    ensureSpectrum(amg, A);

    // The V-cycle (wA = M^-1 rA) has no host-scalar dependencies and runs on fixed buffers (amg.rA/wA + amg's
    // persistent work buffers), so it is captured into a CUDA graph and replayed. The graph is cached in amg.gcache,
    // keyed on the fine-matrix pointer A.diag: captured once and replayed across all PCG iters and SIMPLE steps (the
    // matrix values change in-buffer each step; the graph reads them at replay), re-captured only when the key changes.
    // FP32 mixed precision (the default smoother only; GS/Cheb/corrScaling stay FP64): the matrices are cast
    // once per solve, and each application casts rA -> FP32 in / FP32 -> wA out.
    const bool fp32 = amgSinglePrecisionCycle(amg) && !corrScaling;
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
    // A DECOMPOSED MESH'S V-CYCLE IS CAPTURED ON THE SINGLE-PRECISION PATH ONLY. Its matrices (LduF) carry no processor
    // interface, so the cycle is the block-local preconditioner with no halo exchange and no collective in it -- a
    // graph like the single mesh's. The FP64 cycle takes the DeviceLduView, whose processor product would put a halo
    // exchange inside the capture. MEASURED on the G4 case at np 8 (V100): about 100 kernels a cycle, ~30 cycles a step,
    // each launched from the host where np 1 replays one graph.
    if (A.proc.active() && !fp32) captureVcycle = false;
    amgSplit::lap(amgSplit::matrixCast);
    AMGGraphCache& gc = *amg.gcache;
    auto applyPrecond = [&]()
    {
        if (fp32)
        {
            const LduF A0 = lduF(A, amg.fDiag[0], amg.fUpper[0], amg.fLower[0]);
            auto runF = [&]()                                                        // cast in -> FP32 V-cycle -> cast out
            {
                cast_<scalar,float><<<nBlocks(nC),TPB>>>(nC, rA.data(), amg.vBF[0].data());
                amgSplit::lap(amgSplit::vectorCasts);
                vcycleAtF(0, amg, A, A0, amg.vBF[0].data(), amg.vXF[0].data());
                cast_<float,scalar><<<nBlocks(nC),TPB>>>(nC, amg.vXF[0].data(), wA.data());
                amgSplit::lap(amgSplit::vectorCasts);
            };
            if (!captureVcycle)
            {
                runF();
                return;
            }
            AMGGraphCache& gcf = *amg.gcacheF;                                          // graph the FP32 V-cycle (host-scalar-free)
            if (!gcf.exec || gcf.key != A.diag || gcf.keyEpoch != deviceReductionScratchEpoch()
             || gcf.keyPairEpoch != amg.pair.epoch || amgGraphViewMoved(gcf, A))
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
                gcf.keyPairEpoch = amg.pair.epoch;
                amgGraphViewStamp(gcf, A);
            }
            cudaCheck(cudaGraphLaunch(gcf.exec, cudaStreamPerThread), "amgF graph launch");
            return;
        }
        if (!captureVcycle)
        {
            vcycleAt(0, amg, A, rA, wA);
            return;
        }
        if (!gc.exec || gc.key != A.diag || gc.keyEpoch != deviceReductionScratchEpoch()
         || gc.keyPairEpoch != amg.pair.epoch || amgGraphViewMoved(gc, A))
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
            gc.keyPairEpoch = amg.pair.epoch;
            amgGraphViewStamp(gc, A);
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
            amgSplit::lap(amgSplit::updates);
            amgFineAmul(amg, A, pA, wA);
            amgSplit::lap(amgSplit::product);
            deviceDotInto(wA, pA, dPap);
            deviceScalarDivNeg(dWArA, dPap, dAlpha, dNegAlpha);
            deviceAxpyDev(dAlpha, pA, psi);
            deviceAxpyDev(dNegAlpha, wA, rA);
            amgSplit::lap(amgSplit::updates);
            ++nIter;
            if (nIter % K == 0 || nIter >= maxIter)           // read |r|_1 only every K iters (K=1: exact per-iter)
            {
                deviceSumMagInto(rA, dResNorm);
                perf.finalResidual = deviceReadScalar(dResNorm)/normFactor;   // the only host sync
                if (procHist) procHist->push_back(perf.finalResidual);
            }
            amgSplit::lap(amgSplit::residualRead);
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
    }
    perf.nIterations = nIter;
    if (split) chargeSplit(A, amg, fp32, nIter);
    static const bool dbgCyc = std::getenv("BRAE_AMG_CYCLES") != nullptr;   // benchmark: AMG V-cycles per pressure solve
    if (dbgCyc) std::fprintf(stderr, "[AMG] p cycles=%d finalRes=%.3e\n", nIter, perf.finalResidual);
    return perf;
}

// Cast the fine + coarse matrices to their FP32 mirrors for this solve's V-cycles. Call ONCE per solve, AFTER
// amgGalerkin has updated the FP64 operators (the coarse matrices change every SIMPLE step). No-op unless FP32
// mixed precision applies (BRAE_AMG_FP32 default on; the GS/Chebyshev paths stay FP64). After this,


DeviceSolverPerf deviceAMGPCG(
    const DeviceLduView& A,
    AMGData& amg,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    const scalar* dNormFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    bool captureVcycle,
    int checkEvery,
    bool corrScaling,
    int minIter)
{
    announceNormFactorMode();
#ifdef BRAE_HAS_GS_DEVICE
    // deviceAMGPCGGraph is DEFINED inside this guard (it needs conditional graph nodes, CUDA 12.4 and newer), so
    // the call has to sit inside it too. Without the guard this translation unit does not compile on a toolkit
    // older than 12.4 -- `identifier "deviceAMGPCGGraph" is undefined` -- which is the same defect as the
    // `neumannPrecon` one the GH200 found on 2026-09-13, in a different file. The sibling call at :472 was
    // guarded already.
    {
        static const bool pcgDev = std::getenv("BRAE_PCG_DEVICE") == nullptr || std::string(std::getenv("BRAE_PCG_DEVICE")) != "0";
        if (pcgDev && braeCondGraphsAvailable() && !corrScaling && !normFactorOnHost() && !amgSplit::wanted() && !A.proc.active())
        {
            amgCouplePair(amg, A);
            return deviceAMGPCGGraph(A, amg, b, psi, dNormFactor, tol, relTol, maxIter, minIter);
        }
        if (A.proc.active() && !corrScaling && checkEvery <= 1 && !procPcgUnfused() && procPcgGraphWanted(A))
        {
            amgCouplePair(amg, A);
            return deviceAMGPCGProcGraph(A, amg, b, psi, dNormFactor, tol, relTol, maxIter, minIter);
        }
    }
#endif
    return deviceAMGPCG(A, amg, b, psi, deviceReadScalar(dNormFactor), tol, relTol, maxIter, captureVcycle, checkEvery, corrScaling, minIter);
}

} // namespace brae
