#pragma once
// cf GPU offload (#6 + #7): an algebraic-multigrid PRECONDITIONER for the device Krylov solvers, the #1 perf
// lever vs Jacobi. RECURSIVE MULTI-LEVEL: the grid is pairwise-agglomerated repeatedly (host, static geometry,
// carrying face weights = sum of |Sf|) down to a tiny coarsest level, giving a level hierarchy. The V-cycle
// recurses level-to-level (weighted-Jacobi pre/post smoothing, Galerkin-coarsened operators, a many-sweep solve
// on the coarsest level, fused into one thread-block cluster when small enough, #7b). Each level's matrix is
// re-built by a device Galerkin scatter from the level above each time the fine matrix changes.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"   // DeviceSolverPerf
#include <vector>
#include <memory>
#include <string>

namespace brae {

// Cached CUDA graph of the V-cycle: captured once and replayed across solves while the fine-matrix buffer
// pointer (key) is unchanged. Owned via unique_ptr so AMGData stays movable and the graph is freed on destruct.
struct AMGGraphCache {
    cudaGraphExec_t exec = nullptr; cudaGraph_t graph = nullptr; const void* key = nullptr;
    ~AMGGraphCache();
};

// Cached CUDA conditional-graph of the device-resident PCG WHILE-body (#6, BRAE_PCG_DEVICE). Owned PER-SOLVER (in
// AMGData) so it is destroyed with the hierarchy. A process-global cache is unsafe: after one solver is freed, a
// later solver whose pressure buffer lands at the recycled address would hit the stale entry and replay a graph
// baked against freed buffers (illegal memory access). Holds the graph-referenced persistent work buffers.
struct PCGGraphCache {
    cudaGraphExec_t exec = nullptr; cudaGraph_t graph = nullptr;
    cudaGraphConditionalHandle handle{}; const void* key = nullptr;
    // tol/relTol/maxIter are baked into the captured pcgSetCondK node, so they are part of the
    // cache identity: replaying with a changed convergence control would silently keep the stale
    // bound. They stay constant per field in the SIMPLE loop (no extra re-capture there), so
    // including them in the key only guards the case where a caller reuses this cache with new
    // controls. Sentinels chosen so the first solve always misses.
    scalar keyTol = -1.0; scalar keyRelTol = -1.0; int keyMaxIter = -1;
    DeviceBuffer<scalar> pA, Ax, sNormF, sInit, sRes; DeviceBuffer<int> sIter;   // persistent (graph-referenced)
    ~PCGGraphCache();
};

// One agglomeration level: maps grid k (nFine cells) -> grid k+1 (nCoarse cells), and holds grid k+1's matrix.
struct AMGLevel {
    int nFine = 0, nCoarse = 0, nCoarseFaces = 0;
    DeviceBuffer<label>  map;                                   // grid-k cell -> grid-(k+1) cell
    DeviceBuffer<label>  cOwn, cNei, cOwnerStart, cLosort, cLosortStart;  // grid-(k+1) addressing (SpMV gather)
    DeviceBuffer<label>  faceRestrict, faceFlip;               // grid-k face -> grid-(k+1) face (>=0) / -1-coarseCell
    DeviceBuffer<scalar> cDiag, cUpper, cLower;                // grid-(k+1) matrix (rebuilt by Galerkin)
    // SMOOTHED AGGREGATION (BRAE_AMG_SA)
    // The tentative prolongator (= map) smoothed by one Jacobi step P=(I-omega D^-1 A)P_tent, stored sparse (CSR by
    // fine row). Built ONCE from the geometric (face-weight) proxy Laplacian; sparsity + values are fixed, so the
    // restrict/prolong kernels stay graph-capturable. The coarse matrix is then the GENERAL Galerkin A_c=P^T A P,
    // re-evaluated each step from the current fine matrix by the precomputed RAP scatter recipe below.
    DeviceBuffer<label>  Prow, Pcol;                           // CSR rowPtr [nFine+1], col [nnz] (coarse columns)
    DeviceBuffer<scalar> Pval;                                 // CSR values [nnz]
    // RAP scatter recipe: nTriples contributions A_c[dst] += w * A_fine[src]. src/dstKind: 0=diag 1=upper 2=lower.
    int nTriples = 0;
    DeviceBuffer<label>  rapSrcKind, rapSrcIdx, rapDstKind, rapDstIdx;
    DeviceBuffer<scalar> rapW;
    DeviceLduView coarseView() const {
        return {nCoarse, nCoarseFaces, cDiag.data(), cUpper.data(), cLower.data(), cOwn.data(), cNei.data(),
                cOwnerStart.data(), cLosort.data(), cLosortStart.data()};
    }
};

// Per-grid greedy cell coloring for the multicolor Gauss-Seidel smoother (BRAE_AMG_GS). Cells of one color share no
// face -> their GS updates are mutually independent (race-free) and read the latest neighbour values -> true GS.
// The standard cure for high-aspect-ratio anisotropy where point Jacobi/Chebyshev stall (see cf-airfoil-aero-test).
struct GridColoring {
    int nColors = 0;
    DeviceBuffer<label> cells;   // grid cells reordered by color (size = grid nCells)
    DeviceBuffer<label> start;   // color offsets into cells[] (size nColors+1, host-readable copy below)
    std::vector<label>  startH;  // host copy of start (the smoother loops colors on the host, launching per color)
};

struct AMGData {
    int nFine = 0;
    std::vector<AMGLevel> level;                                // level[k]: grid k -> grid k+1 (+ grid k+1's matrix)
    std::vector<GridColoring> coloring;                         // coloring[g] for smoothed grid g (0=fine .. nLevels-1); GS only
    std::vector<DeviceBuffer<scalar>> vAx, vR, vX, vB, vD, vPc; // V-cycle scratch per grid (vD = Chebyshev dir; vPc = prolonged corr.)
    std::vector<scalar> lambdaMax;                             // per-grid lambda_max(D^-1 A) for the Chebyshev smoother
    bool spectrumReady = false;                                // estimated once (D^-1 A is diagonal-scale invariant -> stable)
    bool gsSmooth = false;                                      // multicolor Gauss-Seidel smoother (BRAE_AMG_GS) instead of weighted-Jacobi
    bool saSmooth = false;                                      // smoothed aggregation (BRAE_AMG_SA): sparse smoothed P + general RAP coarse operator
    bool corrScaling = false;                                  // OF-GAMG coarse-correction scaling (nonlinear precond -> needs flexible CG)
    DeviceBuffer<scalar> sScNum, sScDen, sScAlpha, sZrOld;     // correction-scaling + flexible-CG scalars (device-resident, graph-safe)
    DeviceBuffer<scalar> wA, rA;                                // persistent V-cycle out/in (fixed addrs -> graph valid)
    DeviceBuffer<scalar> sWArA, sWArAold, sPap, sAlpha, sNegAlpha, sBeta, sResNorm;  // device-resident PCG scalars (1 each)
    // #7 MIXED PRECISION (BRAE_AMG_FP32): FP32 mirrors of every level matrix + work vectors so the BW-bound V-cycle
    // (SpMV+smoother) moves half the bytes. The outer Krylov + residual stay FP64 (accuracy preserved); the coarsest
    // solve casts back to FP64. Index g = grid g (0=fine, g>=1 = level[g-1].coarseView()).
    std::vector<DeviceBuffer<float>> fDiag, fUpper, fLower;     // FP32 matrices per grid
    std::vector<DeviceBuffer<float>> vAxF, vRF, vXF, vBF;       // FP32 V-cycle work vectors per grid
    bool fp32Alloc = false;
    std::unique_ptr<AMGGraphCache> gcache;                      // cached V-cycle graph (capture once, replay)
    std::unique_ptr<AMGGraphCache> gcacheF;                     // cached FP32 V-cycle graph (#7 mixed precision)
    std::unique_ptr<PCGGraphCache> pcgCache;                    // cached device-resident PCG WHILE-body graph (#6); per-solver lifetime
    int nCoarse = 0, nCoarseFaces = 0;                          // back-compat: the FIRST coarse level (level[0])
    int nLevels() const { return static_cast<int>(level.size()); }
    DeviceLduView coarseView() const { return level.front().coarseView(); }   // first coarse level (for #7b tests)
};

// Build the multi-level agglomeration hierarchy (host, once) from the fine internal addressing + face weights (|Sf|).
AMGData buildAMG(const std::vector<label>& fineOwner, const std::vector<label>& fineNei,
                 const std::vector<scalar>& faceWeights, int nFine);

// AMG hierarchy cache (the "partition" step): the agglomeration is static per mesh -> serialize the STRUCTURE so a
// warm run reloads it instead of re-agglomerating. Only the structure is cached (cDiag/cUpper/cLower VALUES are
// Galerkin-rebuilt each step). loadAMGCache returns false (caller rebuilds) on any mismatch/corruption/mode change.
void writeAMGCache(const AMGData& A, const std::string& path);
bool loadAMGCache(const std::string& path, AMGData& A);
// Build the hierarchy, or reload cacheDir/.brae_amgcache if valid (newer than cacheDir/owner). writeCache persists it.
AMGData buildOrLoadAMG(const std::vector<label>& fineOwner, const std::vector<label>& fineNei,
                       const std::vector<scalar>& faceWeights, int nFine, const std::string& cacheDir, bool writeCache);

// Galerkin: rebuild the coarse matrix coefficients from the current fine matrix (diag/upper/lower).
void amgGalerkin(AMGData& A, const DeviceBuffer<scalar>& fineDiag, const DeviceBuffer<scalar>& fineUpper,
                 const DeviceBuffer<scalar>& fineLower);

// AMG-preconditioned CG (the V-cycle replaces the Jacobi preconditioner). Same solution as deviceJacobiPCG.
// captureVcycle: record the (host-scalar-free) V-cycle into a CUDA graph and replay it each PCG iteration, same
// result, far less host launch overhead. WIN ONLY when captured once and replayed many times (a long solve with
// stable buffers): measured 1.44-1.54x on a standalone solve. In a SIMPLE loop the pressure buffers change every
// step, so per-solve re-instantiate costs more than it saves -> default OFF; the loop needs persistent buffers +
// a cached graph exec (cudaGraphExecUpdate) to benefit, which is the follow-up.
// checkEvery: read the residual norm to the host (the one per-iter host sync left after the device-resident
// scalars) only every K iterations instead of every iteration. K>1 cuts (K-1)/K of the PCG host syncs AND skips the
// residual reduction on non-check iters, at the cost of overshooting true convergence by < K iters (extra V-cycles,
// which only make the solution MORE converged). NOT bit-identical to K=1 (the iteration count changes) -> default 1
// so all direct-solver callers stay exactly per-iter; the SIMPLE loop opts into K>1 (validated vs OpenFOAM).
// corrScaling: enable OF-GAMG coarse-correction scaling in the V-cycle (a per-level line-search that fixes the
// unsmoothed-aggregation correction magnitude, the dominant cycle-count lever at scale). It makes the
// preconditioner NONLINEAR, so it switches the Krylov accelerator to flexible CG (Polak-Ribiere+ beta). Default
// false -> plain injection + standard CG (bit-identical to before); the SIMPLE loop opts in (validated vs OpenFOAM).
DeviceSolverPerf deviceAMGPCG(const DeviceLduView& Afine, AMGData& amg, const DeviceBuffer<scalar>& b,
                              DeviceBuffer<scalar>& psi, scalar normFactor, scalar tol, scalar relTol, int maxIter,
                              bool captureVcycle = false, int checkEvery = 1, bool corrScaling = false);

// z = M^-1 r : one symmetric AMG V-cycle applied as a PRECONDITIONER (the V-cycle factored out of deviceAMGPCG).
// Used by the distributed Krylov (deviceParallelAMGPCG) to precondition each rank's LOCAL block with AMG -- the
// V-cycle is internal-face only, so it omits the processor interface (block-Jacobi/additive-Schwarz: the outer
// distributed matvec supplies the interface coupling). amg must be built (buildAMG) and current (amgGalerkin).
// captureVcycle: replay the V-cycle from a cached CUDA graph (keyed on A.diag) instead of launching every kernel,
// removing the launch overhead of the V-cycle's many small kernels. Default false (direct launch).
void amgVCycleApply(AMGData& amg, const DeviceLduView& A,
                    const DeviceBuffer<scalar>& r, DeviceBuffer<scalar>& z, bool captureVcycle = false);

// Prepare the FP32 mixed-precision V-cycle for this solve: cast the (Galerkin-updated) matrices to FP32 mirrors.
// Call ONCE per solve before the amgVCycleApply loop; after it, amgVCycleApply runs FP32 automatically. No-op
// unless BRAE_AMG_FP32 (default on) and the default smoother/aggregation (SA/GS/Chebyshev stay FP64).
void amgPrepareFP32(AMGData& amg, const DeviceLduView& A);

class DeviceHalo;   // fwd (parallel/pstream/device_halo.cuh)

// DISTRIBUTED whole-loop graph PCG: the entire steady-state PCG WHILE body captured once into a conditional CUDA
// graph and replayed on-device (no per-iteration host launches), with deviceParallelAmul (halo) + on-stream
// NVSHMEM reduces INSIDE the captured body. Real-multi-GPU only (1 PE/GPU; the MPG host-MPI reduce fallback is
// not graph-capturable). Captures once per run only if the caller holds psi/matrix buffers persistent.
DeviceSolverPerf deviceParallelAMGPCGGraph(
    const DeviceLduView& A, AMGData& amg, DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& psi,
    scalar normFactor, scalar tol, scalar relTol, int maxIter);

// #7 (clusters+DSM): the coarse-level weighted-Jacobi solve (nSweeps of x += omega*(b-Ax)/diag), fused into a
// SINGLE thread-block-cluster kernel, the whole coarse vector lives in the cluster's distributed shared memory
// and all sweeps run with cluster.sync() between them (1 launch instead of 2*nSweeps). xc is the in/out guess.
// Returns false (does nothing) if the coarse level doesn't fit one cluster's DSM (caller uses the loop instead).
bool deviceCoarseFitsCluster(int nCoarse);
void deviceCoarseJacobiFused(const DeviceLduView& cv, const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc, int nSweeps);
// Reference: the unfused coarse Jacobi (deviceAmul + smooth, nSweeps launches each), for validation/timing.
void deviceCoarseJacobiLoop(const DeviceLduView& cv, const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc, int nSweeps);

// symGaussSeidel scalar solver (OpenFOAM smoothSolver + symGaussSeidelSmoother, byte-faithful via multicolor GS).
// For the stiff near-wall k/omega/epsilon transport on low-Re (y+~1) meshes where Jacobi-BiCGStab amplifies the
// near-wall instability. Coloring is built once per mesh (cached on A.owner), internal-face LDU only (no interface).
scalar deviceSymGaussSeidel(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& psi,
                            scalar normFactor, scalar tol, scalar relTol, int maxIter,
                            DeviceSolverPerf* perf = nullptr);   // returns the OF initialResidual; *perf (if given) gets init/final/nIter

} // namespace brae
