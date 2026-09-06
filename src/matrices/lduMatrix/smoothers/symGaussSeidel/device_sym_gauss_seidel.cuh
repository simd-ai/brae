#pragma once
// symGaussSeidel: OpenFOAM's own smoother, EXACTLY, by level scheduling.
//
// WHY EXACTNESS IS THE REQUIREMENT HERE, and not a preference. brae used to run this as a MULTICOLOUR
// sweep, which visits the same cells in a different order. Gauss-Seidel is order-dependent, so that is a
// different smoother wearing the same name, and tests/gs_ladder measured the difference on OpenFOAM's own
// matrix: after n sweeps of T3A's first momentum system the index-order smoother leaves residual
// 4.13e-01 (n=1), 7.11e-02 (n=5), 8.59e-03 (n=10) where the colour order leaves 5.63e-01, 1.96e-01,
// 5.91e-02 -- 1.36x to 6.88x behind, and the gap GROWS with the sweep count.
//
// That is not only an efficiency gap. validation/T3A asks for `relTol 0.1` with `maxIter 10`, and the two
// orders land on opposite sides of it: OpenFOAM reaches relTol 0.1 in 4-5 sweeps and stops, the colour
// order needs 9-10 and so always takes the cap. Measured on that case with the pressure solve driven to
// 1e-12 in both codes, real simpleFoam v2412 CONVERGES at `relTol 0.1; maxIter 10` and DIVERGES when
// forced to take all ten sweeps (`relTol 0;`) -- Ux 2.997e-01 at iteration 400. T3A's SIMPLEC loop is
// marginally unstable and the smoother's STOPPING POINT is what damps it. A smoother that stops somewhere
// else is not a slower solve of the same problem, it is a different outer iteration.
//
// THE ALGORITHM IS SEQUENTIAL (symGaussSeidelSmoother.C:145-190):
//
//     bPrime = source
//     forward,  celli ascending:   psii  = bPrime[celli]
//                                  psii -= sum over faces OWNED by celli of upper[f]*psi[nei[f]]
//                                  psii /= diag[celli]
//                                  bPrime[nei[f]] -= lower[f]*psii   for those same faces
//                                  psi[celli] = psii
//     reverse,  celli descending:  psii  = bPrime[celli]             (the forward half's bPrime, not rebuilt)
//                                  psii -= sum over faces OWNED by celli of upper[f]*psi[nei[f]]
//                                  psii /= diag[celli]
//                                  psi[celli] = psii
//
// Both halves are recurrences. LEVEL SCHEDULING makes them parallel without changing a single operation:
//
//   * OF's face ordering has owner < neighbour and faces sorted by owner, so the forward recurrence is a
//     DAG over cells -- cell c follows exactly the owners of the faces whose neighbour is c. The reverse
//     recurrence is the same DAG over the reversed cell numbering.
//   * Cells at one DAG depth have no face between them, so a whole level runs in parallel.
//   * The scatter `bPrime[nei[f]] -= lower[f]*psii` is turned into the GATHER it is equivalent to:
//     bPrime[c] = source[c] - sum over faces whose NEIGHBOUR is c of lower[f]*psi[owner[f]], taken in
//     losort order, which is increasing face index -- the order OF's owner walk meets those faces in.
//     So the summation order is OpenFOAM's too, and the sweep is bit-identical rather than merely equal.
//   * The reverse half needs no separate bPrime array: when cell c is visited, every owner it gathers is
//     a LOWER-numbered cell, which the reverse walk has not reached yet and which therefore still holds
//     exactly the forward value OF's bPrime captured.
//
// THE COST, and how it is paid. Level scheduling as nLevels kernel launches per half-sweep was
// launch-bound: on validation/T3A (26820 cells, 53213 internal faces) the forward DAG has 465 levels
// averaging 57.7 cells -- every one under a single thread block -- so each of the ~930 launches per
// sweep did a few microseconds of work and the GPU sat idle between them. Measured end to end, 43.1 s
// for the 406 outer iterations the case takes to converge, against 21.3 s for 1000 iterations of the
// colour order that never converged: about 5x per outer iteration, with FEWER sweeps per solve.
//
// So when NO level is wider than one block, the whole half-sweep is ONE launch: a single block walks
// the levels in order with __syncthreads() between them. Every cell's gather is the same code as
// before, so the result is bit-identical -- tests/gs_ladder holds it to OpenFOAM at 2.8e-12 either way
// -- and the only thing that changed is that 465 launches became one barrier each. Meshes whose levels
// outgrow a block (a 1M-cell mesh has levels of thousands of cells) keep the per-level launches, where
// each launch carries enough work for the overhead not to dominate. A wrong-but-parallel smoother would
// still not be a cheaper version of either: it is a different solver, and on T3A the difference between
// converging and not.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include <vector>

namespace brae
{

struct DeviceGaussSeidelLevels
{
    int nCells = 0;
    bool valid = false;
    // cells ordered by DAG depth, with per-level [begin,end) held host-side because they drive the
    // launch loop. fwd = the ascending-index sweep, bwd = the descending one.
    DeviceBuffer<label> fwdCells, bwdCells;
    std::vector<int>    fwdOff, bwdOff;
    // ...and on the device too, for the single-block walk (the host copies drive the per-level loop).
    DeviceBuffer<label> fwdOffD, bwdOffD;
    // The widest level, which picks the execution strategy (see deviceSymGaussSeidelSweepExact).
    int maxLevelWidth = 0;

    // THE LEVEL-ORDERED GATHER (item 60b). For walk position i of each half, the cell's terms as a CSR in
    // EXACTLY gsCellUpdate's order -- its losort (lower) terms in losort order, then its owner (upper)
    // terms in face order -- built by replaying that gather's loops on the view's own losort/ownerStart.
    // entNbr is the cell whose psi the term multiplies; entSrc names the coefficient: f for lower[f],
    // f + nInternalFaces for upper[f]. Per level the walk then chases three dependent loads (its offsets,
    // its entries, psi) where the index gather chased five (the cell, its offsets, losort, the face's
    // coefficient and owner, psi), and the topology reads stream instead of scatter. Same terms, same
    // order, same arithmetic: bit-identical to the index gather (tests/gs_level_gather_identity).
    DeviceBuffer<label> fwdEntStart, fwdEntNbr, fwdEntSrc;
    DeviceBuffer<label> bwdEntStart, bwdEntNbr, bwdEntSrc;
    int nEntries = 0;                      // 2 * nInternalFaces when the CSR is built, 0 when not

    int levels() const { return (int)fwdOff.size() - 1; }
};

// The sweep's per-solve operands in walk order, refreshed once per solve (the values change every solve,
// the order never): the coefficients, shared by a vector matrix's components, and the per-component
// diagonal and source. The refreshes are plain kernels, capturable, and run once per solve.
struct GSLevelCoefs
{
    DeviceBuffer<scalar> fwd, bwd;
};
struct GSLevelCells
{
    DeviceBuffer<scalar> fwdDiag, bwdDiag, fwdB, bwdB;
};
void gsLevelCoefsRefresh(const DeviceLduView& A, const DeviceGaussSeidelLevels& lv, GSLevelCoefs& lc);
void gsLevelCellsRefresh(const DeviceLduView& A, const DeviceBuffer<scalar>& b, const DeviceGaussSeidelLevels& lv, GSLevelCells& cc);
// Whether the sweeps take the level-ordered gather (default when the CSR exists; BRAE_GS_LEVEL_GATHER=0
// restores the index gather for the identity gate).
bool gsLevelGatherEnabled(const DeviceGaussSeidelLevels& lv);
// Whether the sweeps walk every level in one block (no level outgrows it and BRAE_GS_PER_LEVEL is unset)
// or launch per level; the announces report this, so they say what actually ran.
bool gsSingleBlockWalk(const DeviceGaussSeidelLevels& lv);

// losort/losortStart/ownerStart are the VIEW's own (downloaded), so the CSR replays gsCellUpdate's loops
// on the same arrays it reads; pass them empty to build the levels without the CSR.
DeviceGaussSeidelLevels buildDeviceGaussSeidelLevels(const std::vector<label>& owner,
                                                     const std::vector<label>& nei,
                                                     label nCells,
                                                     const std::vector<label>& losort = {},
                                                     const std::vector<label>& losortStart = {},
                                                     const std::vector<label>& ownerStart = {});

// The level-scheduled sweeps, cached per matrix (keyed on A.owner, as the colouring was).
const DeviceGaussSeidelLevels& gsLevelsFor(const DeviceLduView& A);

// One sweep on psi in place. `b` is the folded source the linear solver was handed (fvMatrixSolve.C has
// already added boundaryCoeffs into it and internalCoeffs into the diagonal), exactly the `source` OF's
// smoother receives.
//
// `symmetric` picks WHICH of OpenFOAM's two Gauss-Seidel smoothers this is, and they are different
// solvers, not settings of one:
//   true   symGaussSeidelSmoother.C -- the ascending walk then the descending one.
//   false  GaussSeidelSmoother.C -- the ascending walk ONLY. Its sweep loop has no second half; read it
//          and there is simply no reverse pass. A case naming `GaussSeidel` that is answered with the
//          symmetric sweep gets about twice the smoothing per sweep it asked for, reaches its relTol in
//          fewer sweeps, and therefore stops somewhere else -- which on validation/T3A is the difference
//          between converging and limit-cycling (see the header note above).
// `lc`/`cc` are the level-ordered operands refreshed for THIS solve; null means the sweep refreshes a
// scratch of its own before walking (the host loop and tests/gs_ladder pay that per sweep).
void deviceSymGaussSeidelSweepExact(const DeviceLduView& A,
                                    const DeviceBuffer<scalar>& b,
                                    DeviceBuffer<scalar>& psi,
                                    const DeviceGaussSeidelLevels& lv,
                                    bool symmetric = true,
                                    const GSLevelCoefs* lc = nullptr,
                                    const GSLevelCells* cc = nullptr);

// THE FUSED WALK (item 60a). A vector fvMatrix's components share the topology and the upper/lower
// coefficients; only the folded diagonal (fvMatrix::addBoundaryDiag per component) and the source are per
// component, and OpenFOAM solves them one after another with nothing in between (fvMatrixSolve.C,
// solveSegregated). One component's sweep reads only its own psi, diag and source, so walking the levels
// ONCE and updating every component at each level performs the same per-cell arithmetic in the same
// level order as walking them once per component -- bit-identical -- while paying the per-level
// dependent-launch latency (measured ~2.7 us; 929 levels x 2 halves on the composed flat plate) once
// instead of once per component. `active[k]` (device) lets a component that has already met its own
// stopping rule sit out the remaining sweeps, which is what keeps each component's sweep count its own.
constexpr int GS_FUSED_MAX = 3;
struct GSFusedOperands
{
    int nComp = 0;
    const scalar* diag[GS_FUSED_MAX] = {};
    const scalar* b[GS_FUSED_MAX] = {};
    scalar*       psi[GS_FUSED_MAX] = {};
    const int*    active = nullptr;     // one flag per component; nullptr = all active
};
// `A` supplies the topology and the SHARED upper/lower; the per-component diag/b/psi come from `ops`.
// `cc[k]` is component k's level-ordered diagonal/source for this solve (all or none).
void deviceSymGaussSeidelSweepExactFused(const DeviceLduView& A,
                                         const GSFusedOperands& ops,
                                         const DeviceGaussSeidelLevels& lv,
                                         bool symmetric = true,
                                         const GSLevelCoefs* lc = nullptr,
                                         const GSLevelCells* const* cc = nullptr);

}   // namespace brae
