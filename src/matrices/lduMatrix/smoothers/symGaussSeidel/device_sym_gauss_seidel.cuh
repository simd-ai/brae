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
// THE COST IS REAL AND IS PAID DELIBERATELY: nLevels kernel launches per half-sweep, the same price
// device_dilu.cuh pays for the same reason. On validation/T3A (26820 cells, 53213 internal faces) the
// forward DAG has 465 levels averaging 57.7 cells -- every one of them under a single 128-thread block --
// so the sweep is launch-bound and the GPU is mostly idle inside it. Measured end to end on that case,
// 43.1 s for the 406 outer iterations this takes to converge against 21.3 s for the 1000 the colour
// order ran without converging: about 5x per outer iteration, and that is with FEWER sweeps per solve
// (the colour order needed 9-10 to reach the case's relTol 0.1 where this needs 4-5), so the gap is
// launch overhead rather than arithmetic.
//
// A wrong-but-parallel smoother is not a cheaper version of this; it is a different solver, and on T3A
// it is the difference between converging and not. The launches are worth attacking on their own terms
// -- one kernel with a grid-wide barrier per level, or a captured CUDA graph -- and neither changes an
// operation, so neither is a substitution.
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

    int levels() const { return (int)fwdOff.size() - 1; }
};

DeviceGaussSeidelLevels buildDeviceGaussSeidelLevels(const std::vector<label>& owner,
                                                     const std::vector<label>& nei,
                                                     label nCells);

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
void deviceSymGaussSeidelSweepExact(const DeviceLduView& A,
                                    const DeviceBuffer<scalar>& b,
                                    DeviceBuffer<scalar>& psi,
                                    const DeviceGaussSeidelLevels& lv,
                                    bool symmetric = true);

}   // namespace brae
