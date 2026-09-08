#pragma once
// GaussSeidel / symGaussSeidel as a MULTICOLOUR sweep under OpenFOAM's smoothSolver stopping rule, fused
// over the components of one vector matrix, on a colour-major permutation of the cells.
//
// WHAT IS OPENFOAM'S HERE AND WHAT IS NOT. The outer loop is smoothSolver::solve transcribed
// (src/OpenFOAM/matrices/lduMatrix/solvers/smoothSolver/smoothSolver.C:82-221): the nSweeps < 0 branch
// that sweeps a fixed count and reports no residual (:95-119), the normFactor and initial residual
// (:127-150), the pre-loop test `minIter > 0 || !checkConvergence` (:159-165), the do/while whose
// sweep count grows in multiples of nSweeps and is incremented BEFORE the maxIter test (:178-209),
// and SolverPerformance::checkConvergence's strict `<` with its `relTol > 1e-20` guard
// (src/OpenFOAM/matrices/LduMatrix/LduMatrix/SolverPerformance.C:62-88). The per-cell update is
// GaussSeidelSmoother.C:143-171's, with the division by the diagonal (:163) unguarded as OpenFOAM
// leaves it.
//
// The sweep ORDER is not OpenFOAM's. GaussSeidelSmoother walks cells in natural index order, which
// is a sequential recurrence; this walks them by colour: one launch per colour, ascending, and every
// cell of a colour updates at once because no two cells of one colour share a face. Gauss-Seidel is
// order-dependent, so this is a different smoother that stops in a different place on a loose relTol
// (device_sym_gauss_seidel.cuh has the measurements on T3A: 1.36x-6.88x behind after 1-10 sweeps,
// and the two orders on opposite sides of `relTol 0.1; maxIter 10`). It exists for the callers that
// announce that substitution; the level-scheduled sweep in device_sym_gauss_seidel.cuh is the one
// that reproduces OpenFOAM's order. tests/test_colour_gs_fused.cu arm (g) holds the two orders apart.
//
// `symmetric` adds a backward pass: the same colour update in DESCENDING colour order. That is the
// analogue of symGaussSeidelSmoother.C:174-198's descending walk (which re-uses the forward half's
// bPrime because a descending cell's owners are lower-numbered and not yet revisited); it is not a
// literal transcription of that bPrime re-use, since in colour order the forward pass has no bPrime
// to keep -- every colour gathers its off-diagonal terms afresh from the current psi. A colour
// launched twice with no other colour launched in between recomputes every cell from the same
// operands to the same bits (a cell reads only its own source and diagonal and the psi of OTHER
// colours), so the engine skips such a launch: colour K-1 at the head of every descending pass and
// colour 0 at the head of an ascending pass that follows one. On the 2-colour hex mesh that was
// half of every symmetric sweep's launches. The unit test's host reference performs every launch and
// its symmetric arm (b) still matches the device iterate.
//
// FUSION. The components of a vector matrix share upper, lower and the addressing; each has its own
// folded diagonal, source, normFactor, initial residual, sweep count and stop (fvMatrixSolve.C:148-240
// solves them one after the other with the diagonal re-folded per component at :169). One thread per
// cell reads the row's indices and coefficients ONCE and updates every still-active component from
// them, so the topology traffic is paid once per sweep rather than once per component; a component
// that has met its own stopping rule is dropped from the operand list of every later launch, so it is
// frozen exactly where OpenFOAM would have stopped it. The residual after each block of sweeps is
// computed per row on the permuted layout -- by the sweep launches themselves on two colours, by a
// separate pass (lduMatrix::residual, lduMatrixATmul.C:268-340, reading the row once for all
// active components) otherwise, see the next paragraph -- and its sum|r| is reduced inside those
// same launches in a fixed order (THE REDUCTION IN THE LAUNCH below), never through an atomicAdd
// (an atomicAdd sum was the hottest nondeterminism site in the AMG restriction).
//
// THE RESIDUAL FROM THE SWEEP (two colours). smoothSolver.C:189-201 takes the residual after each
// block of nSweeps sweeps as a full pass over the matrix (matrix_.residual, then gSumMag). On the
// 306k-cell case (2 colours of 152,880 rows, GB10, nSweeps 1) that pass was the largest single cost
// of a pass: the two colour launches 231 us together, the residual pass 240 us, the three
// reductions ~55 us. On two colours it is redundant. For a row c of the colour the block's LAST
// launch updated, r_c = b_c - sum_j a_cj x_j - d_c x_c is the round-off of its own update: the
// launch holds acc = b_c - sum_j a_cj x_j in a register and writes x_c = acc/d_c, so r_c =
// acc - d_c x_c (SWEEP_RES_NEW writes it). For a row of the OTHER colour, r_c = acc' - d_c x_c(old)
// with acc' = b_c - sum_j a_cj x_j over the just-updated neighbours, which is exactly what the NEXT
// launch of that colour computes before it overwrites x_c; so the first launch of the FOLLOWING
// block is issued straight after the block's last one, speculatively (SWEEP_RES_OLD_SPECULATIVE:
// it saves its rows' old values into the colouring's saveP, takes their r from them, then updates
// them as usual). The final sum, the mailbox publish and the wait then follow, over the block
// partials the two launches wrote. A component the rule says continues has simply had the first launch of its
// next block, which the schedule owed it anyway (the block's remaining launches follow on the next
// pass); one it says stops -- converged, maxIter, minIter met -- has its rows of the speculated
// colour rolled back from saveP by one small kernel, so its psi is the state after its last counted
// sweep. Per pass on two colours, GaussSeidel nSweeps 1: after the initial explicit residual,
// colour 0 and colour 1 (RES_NEW) then colour 0 (SPECULATIVE) on the first pass, then colour 1
// (RES_NEW) and colour 0 (SPECULATIVE) on every later one; symGaussSeidel: colour 0, colour 1,
// colour 0 (RES_NEW) then colour 1 (SPECULATIVE) on the first pass, then colour 0 (RES_NEW) and
// colour 1 (SPECULATIVE) on every later one (the skip rule above removes the colour a pass would
// begin on). So a pass costs the two colour launches, the final sum and the publish, and the
// solve as a whole one extra colour launch (the last speculation, rolled back) plus the saves;
// measured on the 306k case at 360 us before the reductions were folded in (next paragraph). The
// initial residual, before any sweep, and every colouring other than two nonempty colours (K > 2,
// or an empty colour) run the explicit pass, whose per-row arithmetic is the sweep's -- the source
// minus the row's entries in the sweep's order, then minus d_c x_c LAST, through one device
// function (rowResidual in the .cu) at both sites -- so a residual the fused launches produce and
// one the pass computes over the same state are the same floating-point numbers row for row.
// tests/test_colour_gs_fused.cu arm (k1) holds the two vectors together with memcmp (with the
// vector write turned on, see below), (k2) holds a stopped component's psi to a fixed-count
// solve's bit for bit, (k3) runs the same problem on a 3-class split of the same layout -- the
// explicit path -- to the same counts, psi and residual rows, and (k4) the symmetric schedule.
// deviceColourGaussSeidelFusesResidual below says which path a colouring takes.
//
// THE REDUCTION IN THE LAUNCH. Every launch that produces residual rows -- the RES_NEW and
// SPECULATIVE colour launches of a fused pass, and the explicit residual kernel -- reduces |r|
// over its thread block in shared memory, per component, by a fixed tree over the block's 256
// threads in index order (thread t adds thread t + s for s = 128, 64, ..., 1: the pairing
// sumMagKernel in reductions.cu uses), and thread 0 writes one partial per (component, global
// block) into the colouring's partP. The global block index is the colour's first block over the
// colour-major rows (blockStartH[k]) plus blockIdx.x, so the two colour launches of a pass write
// disjoint ranges of the one buffer and between them cover every row exactly once; the explicit
// kernel walks the SAME partition through a block table (blockLo/blockHi, one entry per block,
// colour by colour, built once in ensureScratch from the colour bounds), so its partials for a
// state are the fused launches' partials for that state, block for block. One final kernel per
// pass -- one block per active component -- sums that component's partials in index order
// (thread t accumulates blocks t, t + 256, t + 512, ... ascending, then the same tree over the
// threads) into dRes[k]; the mailbox publish and the host's wait are unchanged. No atomics
// anywhere, so a solve reports the same bits run to run. Measured before the fold on the 306k
// case (GB10, nsys, one pass, nSweeps 1): RES_NEW launch 130 us, SPECULATIVE launch 155 us, the
// three sumMag + finalSum pairs 50 us, publish 2 us, 360 us in all with no gap above 7 us. The
// fold removes the 50 us of reductions and the residual VECTOR writes (3 x 152,880 x 8 B per
// launch, ~13 us of each launch's ~36 MB at 277 GB/s) and adds one kernel over ~1,200 partials per
// component (a few us): an ESTIMATED ~70 us off the 360 us pass, not measured here (the integrator
// measures). The production path therefore writes no r vector at all: writeResidualVector on the
// colouring, default off, turns the row store back on for the unit test's bit-identity arms
// ((k1), (k3), (l1)); the flag changes no arithmetic -- the same r is reduced either way, the flag
// only decides whether it is also stored -- and arm (l2) holds the two settings to the same bits.
//
// LAYOUT. The sweep runs on a COLOUR-MAJOR PERMUTATION of the cells, not on the natural numbering.
// Measured on a 305,760-cell hex blockMesh (2 colours of 152,880 cells, 894,656 internal faces; nsys
// on a GB10 at 277 GB/s): with each colour class a stride-2 subset of the natural numbering, every
// per-cell array (ownerStart, losortStart, diag, b, psi) and every per-face range was touched with
// half of each cache line unused in each colour launch -- 147 MB per sweep of three components,
// which at 277 GB/s is exactly the 524 us measured (262 us per colour launch, 225 us per residual
// pass), twice the ~260 us bandwidth ideal (matrix arrays ~52 MB once, three psi gather/writes
// ~30 MB). So the colouring carries, built once per mesh on the host: cells[new], the original index
// of the cell at each position of the new numbering (colour 0's cells in ascending original index,
// then colour 1's, ...), and the rows as a CSR over NEW cells whose entries name the neighbour's new
// index, the original face and which side of that face the row is on. Each solve gathers diag, b and
// psi of every component into permuted scratch with one kernel, gathers the per-entry coefficient
// (upper[face] on the owner side, lower[face] on the neighbour side) with another -- 2 x
// nInternalFaces x 8 B, the only per-solve cost that scales with faces -- sweeps each colour as one
// contiguous block with no cells[] indirection, runs the residual pass on the permuted layout, and
// scatters psi back to the natural numbering once at the end. The scratch is owned by the colouring
// (mutable members, sized on first use) so it lives exactly as long as the layout it belongs to and
// needs no key: a cache keyed on a device pointer is handed a different matrix at the same address
// once the device pool recycles a block, and one keyed on the colouring's address has the same
// hazard one level up.
//
// SUMMATION ORDER. Each row's entries are laid out in the order GaussSeidelSmoother.C:143-171
// accumulates a cell's terms, which is the order the unit test's host reference (cellUpdate in
// tests/test_colour_gs_fused.cu) performs them: the source, minus the lower terms of the faces where
// the cell is the neighbour in losort order (a stable counting sort by neighbour, device_mesh.cuh:
// 136-151 as lduAddressing.C:34-91, so ascending face index), minus the upper terms of the faces it
// owns in face order. colourSweepKernel walks a row's entries in that sequence, so its floating-point
// sum is the same sequence of the same operands as the reference's; the test holds every arm's
// device iterate to the reference within 1e-13 of the field's magnitude and the printed difference
// has been 0.0 on this box (both sides contract a*b-c into FMA here; a host compiled without it
// would differ in the last bits). Arm (i) checks the entry order against the device's own
// losort/ownerStart rather than trusting the builder. What the permutation does change is the order
// sum|r| is REDUCED in: the reduction runs over the permuted rows, fixed run to run but a different
// rounding of the same sum than a natural-order reduction (a byte dump from the earlier natural-
// layout build, which reduced through deviceSumMagInto, showed the reported residuals differing by
// one ulp on some components, 0.7181024681042146 against ...47), so a stop decision that sat on an
// exact tolerance tie could fall the other way. The fold of the reduction into the launches (THE
// REDUCTION IN THE LAUNCH) is another fixed order again: its blocks follow the colour bounds where
// deviceSumMagInto's were 256-row blocks of the whole vector, so the reported sums can differ from
// that build's in the last bits, and two colourings of ONE layout that leave the same per-row
// residuals bit for bit (the unit test's 3-class split of the 2-colour box) can report sums that
// differ in the last bits too, because a colour cut in two is two block trees where it was one.
// The per-row residual numbers, the counts, the field, and the stop rule's minIter/maxIter/
// nSweeps/tolerance semantics are unchanged by the fold.
//
// A block-per-component variant (block row blockIdx.y took one component, so each block's psi reads
// stayed within one field but the row was read once per component) measured 30.7 ms per outer
// iteration against 20.7 ms for the fused layout on the same 305,760-cell case; it was removed.
//
// THE LOOP IS ON THE HOST in this version, with the stop decisions on the host, but no pass drains
// the queue: the per-component residual sums (and, on the initial pass, the per-component device
// normFactors) are published by a one-thread kernel into mapped pinned host memory behind a sequence
// number, and the host spins on the number (the .cu has the measurement: cudaStreamSynchronize came
// back ~300 us after the pass's last kernel had finished). One wait per pass, including the initial
// one; the three deviceReadScalar calls that used to fetch the normFactors ahead of it were each a
// cudaMemcpy sync of their own. What remains is the conditional-graph WHILE that
// device_amg_gauss_seidel.cu already runs for the level-scheduled sweep: it would take the host out
// of the loop entirely and let the sweep, residual and stop test replay as one captured body. The
// residual stays the FULL sum over every cell of every active component, because
// smoothSolver.C:189-201 computes it that way (matrix_.residual over the whole field, then gSumMag);
// on two colours the two launches that reduce it cover every cell between them.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_amg.cuh"   // GSFusedComponent, DeviceSolverPerf
#include <vector>

namespace brae
{

// The most components one fused solve carries (a vector has three); the scratch below is sized by it.
constexpr int COLOUR_GS_MAX_COMPONENTS = 3;

// The cells grouped by colour on a colour-major numbering: colour k is [startH[k], startH[k+1]) of the
// NEW numbering, cells[new] is the original index of the cell at that position (colour 0's cells in
// ascending original index, then colour 1's, ...) and newIndex[old] is its inverse. No kernel reads
// newIndex -- the rows below already carry the neighbours' new indices -- so it stays on the host for
// the unit test's structural check of the permutation. nCells and nInternalFaces record the mesh the
// colouring was built for, and the solver refuses a matrix whose sizes differ; sizes alone cannot
// tell two meshes apart (a 5x7x9 box has a 9x7x5 box's cells and internal faces, joined differently),
// so the first solve also records the device addresses of the matrix's owner and neighbour arrays --
// they live in the DeviceMesh for the run, unlike the pool-recycled coefficient blocks -- and every
// later matrix must carry the same ones, or it is refused by name. A colouring swept over a mesh it
// was not built for is a data race that no residual can see.
struct DeviceCellColouring
{
    int nColours = 0;
    int nCells = 0;
    int nInternalFaces = 0;
    bool valid = false;
    DeviceBuffer<label> cells;
    std::vector<label> newIndex;
    std::vector<label> startH;
    // The rows as a CSR over NEW cells: row i's entries are [rowStart[i], rowStart[i+1]); entry e
    // names the neighbour's NEW index (nbr), the ORIGINAL face the coefficient comes from (face) and
    // whether row i's cell owns that face (isUpper 1: the coefficient is upper[face]) or neighbours
    // it (isUpper 0: lower[face]). Per row the entries are in GaussSeidelSmoother's order -- the
    // neighboured faces in losort order, then the owned faces in face order -- so the summation
    // order, and with it every bit of the result, is the host reference's.
    DeviceBuffer<label> rowStart;
    DeviceBuffer<label> nbr;
    DeviceBuffer<label> face;
    DeviceBuffer<unsigned char> isUpper;
    // The mesh addressing the colouring was first swept with (see above). Mutable because the solver
    // records it through its const reference on the first solve.
    mutable bool meshRecorded = false;
    mutable const label* owner = nullptr;
    mutable const label* nei = nullptr;
    // Per-solve scratch on the permuted layout, sized on first use and kept across solves: the
    // per-entry coefficient, and per component the permuted diagonal, source and field, the old
    // values a speculative launch saves (one colour's rows), the per-block residual partials
    // (partP[k*nBlocks + g] is component k's sum|r| over block g) and the residual sums the
    // publish kernel reads. Mutable so the solver's const reference can size them; their contents
    // are rewritten by every solve and carry nothing between solves. rP is only sized and written
    // when writeResidualVector is on; after such a solve that reported residuals, rP[k] holds
    // component k's residual vector of its last pass (the unit test reads it).
    mutable DeviceBuffer<scalar> coeff;
    mutable DeviceBuffer<scalar> diagP[COLOUR_GS_MAX_COMPONENTS];
    mutable DeviceBuffer<scalar> bP[COLOUR_GS_MAX_COMPONENTS];
    mutable DeviceBuffer<scalar> psiP[COLOUR_GS_MAX_COMPONENTS];
    mutable DeviceBuffer<scalar> rP[COLOUR_GS_MAX_COMPONENTS];
    mutable DeviceBuffer<scalar> saveP[COLOUR_GS_MAX_COMPONENTS];
    mutable DeviceBuffer<scalar> partP;
    mutable DeviceBuffer<scalar> dRes;
    // Whether a residual-reporting solve also STORES each row's residual in rP (THE REDUCTION IN
    // THE LAUNCH): the production path does not -- the launches reduce |r| from registers through
    // shared memory and never write the vector -- and the unit test turns it on for its
    // bit-identity arms. Mutable, like the scratch, so the test can set it on the const colouring
    // the solver takes; it changes no arithmetic, only whether r is also written.
    mutable bool writeResidualVector = false;
    // The partition of the permuted rows into thread blocks, colour by colour: block g covers rows
    // [blockLo[g], blockHi[g]) with blockHi never past its colour's end, and colour k's blocks are
    // [blockStartH[k], blockStartH[k+1]). Built once, on the first solve, from the colour bounds
    // and the block size; every launch that writes residual partials indexes partP by this g, and
    // the explicit residual kernel takes its rows from the table so its blocks are the colour
    // launches' blocks.
    mutable bool blocksBuilt = false;
    mutable int nBlocks = 0;
    mutable std::vector<label> blockStartH;
    mutable DeviceBuffer<label> blockLo;
    mutable DeviceBuffer<label> blockHi;
};

// Whether a residual-reporting solve on this colouring takes its per-pass residuals from the sweep
// launches (THE RESIDUAL FROM THE SWEEP above): two colours, both with cells. Any other colouring
// runs the explicit residual pass after every block. The unit test prints it per arm.
bool deviceColourGaussSeidelFusesResidual(const DeviceCellColouring& colouring);

// The greedy colouring of the INTERNAL faces (ownerInternal and neiInternal of equal length; throws
// if they differ or index outside [0, nCells)), laid out colour-major by the function below.
DeviceCellColouring buildDeviceCellColouring(
    const std::vector<label>& ownerInternal,
    const std::vector<label>& neiInternal,
    int nCells);

// The colour-major layout of a GIVEN grouping: class k is cellsByColour[colourStart[k] ..
// colourStart[k+1]). Checks before building that the classes list every cell exactly once and that
// no face joins two cells of one class -- a colouring bug would otherwise turn the sweep into a data
// race that no residual can see -- and that owner is non-decreasing, since the ownerStart ranges the
// natural-layout kernels read assume the upper-triangular face order. Throws naming the offending
// cell or face. buildDeviceCellColouring calls this on greedyColor's classes; tests call it to lay
// out a colouring of their own choosing (a different colour order is a different, deterministic
// Gauss-Seidel iterate).
DeviceCellColouring buildDeviceCellColouringFromClasses(
    const std::vector<label>& ownerInternal,
    const std::vector<label>& neiInternal,
    int nCells,
    const std::vector<label>& cellsByColour,
    const std::vector<label>& colourStart);

// smoothSolver::solve, per component, around the colour sweep. Throws (naming itself) when:
// the components do not share upper/lower/addressing or their sizes; a view carries a cyclic or
// cyclicAMI interface (the sweep applies no interfaces); the colouring's sizes are not the matrix's;
// the colouring was first swept with a matrix whose owner/neighbour addressing lives elsewhere;
// nSweeps is 0 (smoothSolver.C:178-209 would loop forever, never advancing nIterations); or nComp is
// outside [1, 3]. normFactor: comps[k].dNormFactor when set -- fixed for the solve, it is published
// through the residual mailbox with the initial residual sums, so it costs no host sync of its own --
// else comps[k].normFactor. perf[k] receives component k's initial residual, final residual and
// nIterations; the nSweeps < 0 branch reports 0, 0, -nSweeps as OpenFOAM does. A component that
// never sweeps (its initial residual passes with minIter 0) has its psi left untouched.
void deviceColourGaussSeidelFused(
    int nComp,
    const GSFusedComponent* comps,
    const DeviceCellColouring& colouring,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    int nSweeps,
    bool symmetric,
    DeviceSolverPerf* perf);

// A psi computed through the colour-major layout (the same gathered coefficients and row entries the
// sweep reads) and scattered back to the natural numbering. The diagnostic behind
// tests/test_colour_gs_fused.cu arm (i), which holds it to deviceAmul on the natural layout; the
// solver itself never calls it. It gathers into scratch of its own, allocated per call, and touches
// none of the colouring's: a captured graph of the solver (the WHILE-graph next step) will hold the
// addresses of the colouring's scratch, and a diagnostic must not rewrite what a replay reads.
// Throws as deviceColourGaussSeidelFused does on a colouring that is not the matrix's.
void deviceColourLayoutAmul(
    const DeviceLduView& A,
    const DeviceCellColouring& colouring,
    const DeviceBuffer<scalar>& psi,
    DeviceBuffer<scalar>& Apsi);

}   // namespace brae
