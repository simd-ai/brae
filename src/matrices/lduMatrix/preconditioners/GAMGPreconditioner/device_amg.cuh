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
    int keyEpoch = -1;   // deviceReductionScratchEpoch() at capture: the V-cycle captures reductions, whose scratch is regrown by freeing
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
    scalar keyTol = -1.0; scalar keyRelTol = -1.0; int keyMaxIter = -1; int keyMinIter = -1;
    int keyEpoch = -1;   // deviceReductionScratchEpoch() at capture (same hazard as AMGGraphCache)
    // The whole solve is captured now (item 72), so the graph also holds the MESH pointers and the sizes
    // it was built for; a different matrix on the same psi must rebuild rather than replay.
    const void* keyOwner = nullptr; int keyNC = -1; int keyNF = -1;
    DeviceBuffer<scalar> pA, Ax, sNormF, sInit, sRes; DeviceBuffer<int> sIter;   // persistent (graph-referenced)
    // ...and the right-hand side and the fine matrix, copied in per solve, because a captured prologue
    // bakes their pointers and the callers hand in fresh buffers each time.
    DeviceBuffer<scalar> gB, gDiag, gUpper, gLower;
    ~PCGGraphCache();
};

// One agglomeration level: maps grid k (nFine cells) -> grid k+1 (nCoarse cells), and holds grid k+1's matrix.
struct AMGLevel {
    int nFine = 0, nCoarse = 0, nCoarseFaces = 0;
    DeviceBuffer<label>  map;                                   // grid-k cell -> grid-(k+1) cell
    DeviceBuffer<label>  cOwn, cNei, cOwnerStart, cLosort, cLosortStart;  // grid-(k+1) addressing (SpMV gather)
    DeviceBuffer<label>  faceRestrict, faceFlip;               // grid-k face -> grid-(k+1) face (>=0) / -1-coarseCell
    DeviceBuffer<scalar> cDiag, cUpper, cLower;                // grid-(k+1) matrix (rebuilt by Galerkin)
    // DETERMINISTIC GALERKIN GATHER (see the note above galDiagGatherK in device_amg.cu).
    // The inverses of map/faceRestrict, as CSR lists over the COARSE entities. Agglomeration is static
    // for the life of the mesh, so these are built once on the host and reused by every solve; they let
    // the coarse operator be assembled by a fixed-order gather instead of an atomicAdd scatter, which is
    // what makes the whole solve reproducible run to run.
    DeviceBuffer<label>  galCellStart, galCellList;             // coarse cell <- fine cells
    DeviceBuffer<label>  galDFaceStart, galDFaceList;           // coarse cell <- fine faces interior to it
    DeviceBuffer<label>  galFaceStart, galFaceList, galFaceFlipList;  // coarse face <- fine faces
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

    // THE COLOUR-MAJOR PERMUTED LAYOUT of this grid (device_amg_smoothers.cu has the measurement).
    // gsColorT walks cells[] over the NATURAL cell numbering, so every colour launch touches every
    // cache line of every per-cell and per-face array and uses only the fraction belonging to its
    // colour. The layout below numbers the cells COLOUR-MAJOR -- colour k is rows
    // [startH[k], startH[k+1]) and cells[new] is the original index of the cell at that position --
    // and lays each row's entries out in gsColorT's own accumulation order, so a colour launch is one
    // contiguous block with no indirection.
    //
    // Lifetime. The permutation and rowStart/rowNbr/rowSrc are a function of this grid's GRAPH only,
    // so they are built once per grid per mesh (amgEnsurePermutedGSLayout) and survive the AMG binary
    // cache: the cache stores the colouring and the addressing, and this is derived from them exactly
    // as the Galerkin gather lists are, with no format change. coeff and diagP are VALUES: they are
    // re-gathered by amgGalerkin, which is where they change. bP/psiP are re-gathered by every sweep,
    // because the V-cycle's natural-layout kernels (zeroT, deviceAmul, restrict, prolong) produce and
    // consume b and x between one smooth and the next. Mutable because gsSweep takes the colouring by
    // const reference; none of it carries state between sweeps.
    mutable bool permBuilt = false;
    mutable int permCells = 0;
    mutable int permFaces = 0;
    mutable const void* permOwner = nullptr;   // the addressing the layout was built for (see above)
    mutable const void* permNei = nullptr;
    mutable DeviceBuffer<label> rowStart;      // nCells+1: row i's entries are [rowStart[i], rowStart[i+1])
    mutable DeviceBuffer<label> rowNbr;        // 2*nInternalFaces: the neighbour's NEW index
    // ...and where the entry's coefficient comes from: f >= 0 is upper[f] (the row's cell OWNS face f),
    // a negative v is lower[-1 - v] (the row's cell is that face's NEIGHBOUR). One array rather than a
    // face index plus a side byte, so the per-Galerkin gather reads 4 bytes per entry instead of 5.
    mutable DeviceBuffer<label> rowSrc;
    mutable DeviceBuffer<scalar> coeff;        // 2*nInternalFaces: gathered per Galerkin update
    mutable DeviceBuffer<scalar> diagP;        // nCells: gathered per Galerkin update
    mutable DeviceBuffer<scalar> bP, psiP;     // nCells: gathered per sweep
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
    // DIRECT COARSEST SOLVE (BRAE_AMG_COARSE_LU, on by default; device_amg_detail.cuh has the design).
    // The dense LU of the coarsest matrix, refreshed by amgGalerkin -- the one point where the coarse
    // coefficients change, and, as with the permuted GS coefficients, the one point outside every graph
    // capture. coarseLUn is the level size it was factorised for, and 0 when there is no factorisation
    // (level too big, flag off, or amgGalerkin not yet run): the V-cycle dispatch tests it against the
    // grid it is about to solve and falls through to the iterative coarsest solvers when they differ.
    DeviceBuffer<scalar> coarseLU;                              // n*n row-major, L (unit diagonal) and U in place
    DeviceBuffer<int>    coarsePiv;                             // n row interchanges, in factorisation order
    int coarseLUn = 0;
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

// THE COLOUR-MAJOR PERMUTED GAUSS-SEIDEL LAYOUT (GridColoring above; device_amg_smoothers.cu has the
// build, the sweep and the measurement). Public because tests/test_gpu_amg.cu holds the two sweeps
// together, checks the addressing against the level's own matvec, and corrupts a layout to prove the
// checker sees it; the V-cycle itself only ever calls gsSweep.

// Build grid g's permuted layout from the matrix it will be swept with, and gather its coefficients
// once, if it is not already current for that addressing. Cheap (two pointer compares) when warm.
// Throws, naming itself, when the colouring's sizes are not the matrix's -- a colouring swept over a
// grid it was not built for is a data race that no residual can see. Returns false, without building,
// when called with a stream capture in progress and no layout yet: the build reads the addressing back
// to the host, which a capture cannot record, so the caller falls back to the indirection sweep (the
// same bits, see below) rather than baking a half-built layout into a graph.
bool amgEnsurePermutedGSLayout(const GridColoring& gc, const DeviceLduView& A);

// The permuted coefficients (coeff) and diagonal (diagP) of one grid, from the matrix values that grid
// was just Galerkin-updated to. A no-op when the layout is not built. Called once per outer iteration
// from amgGalerkin -- the one point where the coefficients change, and the one point outside every
// graph capture: a "have I gathered this solve" host test evaluated INSIDE a captured V-cycle bakes its
// answer into the graph, and every replay would then smooth with the coefficients of the iteration the
// graph was captured in.
void amgGatherPermutedGSCoeffs(const GridColoring& gc, const scalar* diag, const scalar* upper, const scalar* lower);

// One multicolour Gauss-Seidel sweep, the two layouts. Same operands in the same order per row, so the
// same bits (test arm (a)); gsSweep picks between them on BRAE_AMG_GS_PERM. The permuted one refuses
// (throws) a layout that is not current for A.
void amgGSSweepPermuted(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& x,
                        const GridColoring& gc, bool forward);
void amgGSSweepIndirect(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& x,
                        const GridColoring& gc, bool forward);

// Apsi = A psi computed THROUGH the permuted layout (the gathered coefficients and row entries the
// sweep reads), written back to the natural numbering. Its per-row summation is amulKernel's
// (device_spmv.cu:31-40: the diagonal, then the owned faces' upper terms in face order, then the
// neighboured faces' lower terms in losort order), so it is the same bits as deviceAmul on a sound
// layout -- which is what makes it a test of the ADDRESSING. Throws as the sweep does. It writes the
// colouring's per-sweep scratch (bP/psiP), which every sweep rewrites anyway, so it must not be called
// between a sweep's gather and its colour launches -- a diagnostic, not a solver call.
void amgPermutedLayoutAmul(const GridColoring& gc, const DeviceLduView& A,
                           const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi);

// The layout's own fail-proof, run on the built layout read back from the device: cells[] is a
// bijection of [0, nCells), the colour ranges tile it, each row carries exactly the entries its cell
// owns and neighbours in gsColorT's order, every entry names the right face and side, and no entry
// names a cell of the row's own colour. Returns "" when the layout is sound, else what is wrong,
// naming the offending row or entry. The builder runs the same check on its host arrays before it
// uploads anything.
std::string amgCheckPermutedGSLayout(const GridColoring& gc, const DeviceLduView& A);

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
                              bool captureVcycle = false, int checkEvery = 1, bool corrScaling = false,
                              int minIter = 0);
// the same solve with the normFactor on the device (item 66)
DeviceSolverPerf deviceAMGPCG(const DeviceLduView& Afine, AMGData& amg, const DeviceBuffer<scalar>& b,
                              DeviceBuffer<scalar>& psi, const scalar* dNormFactor, scalar tol, scalar relTol, int maxIter,
                              bool captureVcycle = false, int checkEvery = 1, bool corrScaling = false,
                              int minIter = 0);


// z = M^-1 r : one symmetric AMG V-cycle applied as a PRECONDITIONER (the V-cycle factored out of deviceAMGPCG).
// Used by the distributed Krylov (deviceParallelAMGPCG) to precondition each rank's LOCAL block with AMG -- the
// V-cycle is internal-face only, so it omits the processor interface (block-Jacobi/additive-Schwarz: the outer
// distributed matvec supplies the interface coupling). amg must be built (buildAMG) and current (amgGalerkin).
// captureVcycle: replay the V-cycle from a cached CUDA graph (keyed on A.diag) instead of launching every kernel,
// removing the launch overhead of the V-cycle's many small kernels. Default false (direct launch).
void amgVCycleApply(AMGData& amg, const DeviceLduView& A,
                    const DeviceBuffer<scalar>& r, DeviceBuffer<scalar>& z, bool captureVcycle = false);

// THE ASYMMETRIC V-CYCLE (upper != lower: the transonic pressure equation's fvm::div(phid, p)).
//
// The HIERARCHY needs nothing: buildAMG agglomerates on the face weights alone and never sees the matrix,
// and amgGalerkin already takes OpenFOAM's asymmetric branch, producing both cUpper and cLower with the
// owner/neighbour flip (GAMGSolverAgglomerateMatrix.C:135-170's `if (fineMatrix.hasLower())` path). What is
// NOT valid for an asymmetric operator is the coarsest SOLVE -- deviceCoarsePCG is a conjugate gradient,
// whose step length presumes p.Ap is an A-norm -- and three of the V-cycle's options. So `asymmetric` is a
// SOLVE-TIME argument, not a property of the built hierarchy: the same cached hierarchy serves both.
//
// vcycleAt keeps its 5-argument symmetric entry point (the callers in device_amg.cu / device_amg_pcg.cu
// declare it themselves and are unchanged); this 6-argument overload is the one to call with
// asymmetric = true. It is an OVERLOAD rather than a defaulted parameter because those local declarations
// would otherwise make every existing 5-argument call ambiguous.
void vcycleAt(int g, AMGData& amg, const DeviceLduView& Ag, const DeviceBuffer<scalar>& bg,
              DeviceBuffer<scalar>& xg, bool asymmetric);

// The refusals the asymmetric V-cycle makes, one std::runtime_error per option, each naming itself and
// what to set instead. Exposed (rather than left inside vcycleAt) so a test can exercise each one; vcycleAt
// calls exactly this with (useChebyshev(), amg.corrScaling, amg.saSmooth). A no-op when nothing is set.
void amgRefuseAsymmetric(bool chebyshev, bool corrScaling, bool smoothedAggregation);

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

// symGaussSeidel scalar solver: OpenFOAM's smoothSolver STOPPING RULE (smoothSolver.C:135-209) around
// symGaussSeidelSmoother.C's own sweep, level-scheduled so it runs on the device without changing a
// single operation (device_sym_gauss_seidel.cuh). It USED to be a multicolour sweep, which visits the
// same cells in a different order and is therefore a different smoother: tests/gs_ladder measured it
// 1.36x behind OpenFOAM after one sweep and 6.88x after ten on T3A's own momentum system, and on that
// case the two orders stop on opposite sides of `relTol 0.1; maxIter 10`. There is no substitution here
// any more; the opt-in BRAE_TURB_FP32 / BRAE_TURB_JACOBI / BRAE_GS_DEVICE paths below still take the
// colour order and are experiments, not the solver the case asked for.
// For the stiff near-wall k/omega/epsilon transport on low-Re (y+~1) meshes where Jacobi-BiCGStab amplifies the
// near-wall instability. Coloring is built once per mesh (cached on A.owner), internal-face LDU only (no interface).
scalar deviceSymGaussSeidel(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& psi,
                            scalar normFactor, scalar tol, scalar relTol, int maxIter,
                            DeviceSolverPerf* perf = nullptr,
                            int minIter = 0,   // OF's floor on the sweep count; > 0 takes the host loop
                            // fvSolution solvers/<field>/nSweeps (smoothSolver.C:78, default 1): the
                            // number of smoothing sweeps between residual EVALUATIONS. > 1 takes the
                            // host loop, which evaluates where OpenFOAM evaluates.
                            int nSweeps = 1,
                            // WHICH OpenFOAM smoother: true = symGaussSeidel (ascending then
                            // descending), false = GaussSeidel (ascending ONLY, GaussSeidelSmoother.C
                            // has no reverse half). Read from the case's `smoother` entry; they are
                            // different solvers and a loose solve stops in different places.
                            bool symmetric = true);   // returns the OF initialResidual; *perf (if given) gets init/final/nIter
// The same solve with the normFactor on the device (item 66): the graph and host-smoother paths divide
// by it there and the host never reads it; the host loop and the opt-in paths read it once.
scalar deviceSymGaussSeidel(const DeviceLduView& A, const DeviceBuffer<scalar>& b, DeviceBuffer<scalar>& psi,
                            const scalar* dNormFactor, scalar tol, scalar relTol, int maxIter,
                            DeviceSolverPerf* perf = nullptr, int minIter = 0, int nSweeps = 1, bool symmetric = true);

// The components of ONE vector matrix, solved together (item 60a): their systems share topology, upper
// and lower; each has its own folded diagonal, source, normFactor, residual, sweep count and stop. One
// level walk per sweep updates every still-active component, so the per-level latency is paid once.
// Byte-identical to nComp calls of deviceSymGaussSeidel in order (tests/gs_fused_identity); the fallbacks
// (BRAE_GS_FUSED=0, BRAE_GS_HOST_LOOP, the FP32/Jacobi opt-ins, unshared coefficients, nComp 1) ARE those
// calls. perf[k] receives the k-th component's OpenFOAM report.
struct GSFusedComponent
{
    const DeviceLduView* A = nullptr;
    const DeviceBuffer<scalar>* b = nullptr;
    DeviceBuffer<scalar>* psi = nullptr;
    scalar normFactor = 1.0;
    const scalar* dNormFactor = nullptr;      // when set, the normFactor lives on the device and wins (item 66)
};
void deviceSymGaussSeidelFused(int nComp,
                               const GSFusedComponent* comps,
                               scalar tol,
                               scalar relTol,
                               int maxIter,
                               int minIter,
                               int nSweeps,
                               bool symmetric,
                               DeviceSolverPerf* perf);

} // namespace brae
