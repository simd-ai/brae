#pragma once
// brae::smoothSolver -- OpenFOAM's smoothSolver around its GaussSeidel and symGaussSeidel smoothers,
// the host reference.
//
// provenance:
//   openfoam:  src/OpenFOAM/matrices/lduMatrix/solvers/smoothSolver/smoothSolver.C:78 (nSweeps),
//                  :82-226 (solve: the fixed-sweep branch at :96, the do-while at :174-211)
//              src/OpenFOAM/matrices/lduMatrix/smoothers/GaussSeidel/GaussSeidelSmoother.C:116-176
//              src/OpenFOAM/matrices/lduMatrix/smoothers/symGaussSeidel/symGaussSeidelSmoother.C:116-198
//              src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixSolver.C:235-274 (normFactor)
//              src/OpenFOAM/matrices/LduMatrix/LduMatrix/SolverPerformance.C (checkConvergence)
//   device:    deviceSymGaussSeidel (device_amg.cuh), the same solver level-scheduled
//   tests:     tests/gs_ladder.cu holds gaussSeidelSmoothFolded against OpenFOAM's OWN residual after
//              exactly n sweeps, n = 1..10, on a real momentum system; tests/test_smooth_solver_cpp.cu
//              holds the solver loop against the device's; the interFoam gates hold the whole thing
//              against OpenFOAM's "smoothSolver: Solving for alpha.water" log lines.
//
// WHY A HOST COPY OF A SOLVER THE DEVICE ALREADY HAS. interFoam's MULESCorr pre-solve names
// `solver smoothSolver; smoother symGaussSeidel;` in every tutorial that sets MULESCorr, and the host
// path ran DILU-PBiCGStab instead, under a notice. On damBreak that was measured harmless (alpha 3.6e-14
// from OpenFOAM) because DILU on a near-triangular upwind matrix is as nearly exact as a Gauss-Seidel
// sweep -- measured on one case, not proven. The device path taught what the substitution costs when it
// is NOT harmless: Jacobi-BiCGStab at the case's own 1e-8 left alpha 3.3e-06 and U 1.2e-03 out.
//
// THREE THINGS TO GET RIGHT, none visible in a converged field:
//
//   THE REVERSE HALF DOES NOT DISTRIBUTE. The ascending walk gathers upper*psi[neighbour] over the
//   faces a cell OWNS, divides by the diagonal, and distributes lower*psi_c forward into bPrime. The
//   descending walk gathers the same faces off the bPrime the forward half LEFT, and does not
//   distribute again (symGaussSeidelSmoother.C:192, "these will not be revisited"). GaussSeidel is the
//   ascending walk alone; the two are different solvers and a loose solve stops in different places.
//
//   nIterations COUNTS SWEEPS, and the residual is evaluated once per nSweeps of them:
//   `(nIterations += nSweeps) < maxIter`. With nSweeps 2 and maxIter 5 the loop can run six sweeps.
//
//   A NEGATIVE nSweeps is a fixed sweep count with NO residual at all -- initial and final both stay 0
//   and nIterations reads -nSweeps.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include "pcg.cuh"   // SolverPerformance
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

// WHICH LINEAR SOLVER the case names for a field. Null, where a caller takes a pointer to one, keeps
// PBiCGStab -- what every caller ran before there was a choice. `smoothSolver` runs brae::smoothSolver
// below, OpenFOAM's own.
struct LinearSolverChoice
{
    bool smoothSolver = false;
    // OpenFOAM's PBiCG with DILU (pbicg.cuh); neither this nor smoothSolver means PBiCGStab
    bool pbicgDILU = false;
    // symGaussSeidel (true) or GaussSeidel (false)
    bool symmetric = true;
    int nSweeps = 1;
};

// ownerStartAddr: for cell c, the internal faces it owns are [ownStart[c], ownStart[c+1]). Throws if
// the internal faces are not in OpenFOAM's upper-triangular order (sorted by owner, owner < neighbour),
// because both smoothers are only the ones OpenFOAM runs on a mesh ordered that way.
std::vector<label> lduOwnerStart(
    const std::vector<label>& owner,
    const std::vector<label>& nei,
    label nCells);

// nSweeps sweeps of the smoother on a FOLDED system -- `diag` already carries the boundary
// internalCoeffs and `b` the boundary source, as fvMatrix::solve hands them to the solver.
void gaussSeidelSmoothFolded(
    const std::vector<label>& ownStart,
    const std::vector<label>& nei,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    int nSweeps,
    bool symmetric);

// ...with COUPLED PATCHES. Each sweep starts bPrime = source and then ADDS coeff*pnf over every coupled
// patch, pnf the cell on the other side AS IT STANDS at the start of the sweep (GaussSeidelSmoother.C:
// "the parallel boundary is treated as an effective jacobi interface", with the change of sign its
// comment explains). psi here is always the solution field, so a jump is always applied. `M` supplies
// the interface coefficients; `diag` is still the folded diagonal.
void gaussSeidelSmoothFolded(
    const std::vector<label>& ownStart,
    const std::vector<label>& nei,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    int nSweeps,
    bool symmetric,
    const FvScalarMatrix& M,
    const std::vector<FvPatch>& patches,
    const CoupledJumps* jumps);

// smoothSolver::solve. Folds the boundary like fvMatrix::solve and solves A*psi = b in place.
// `symmetric` selects symGaussSeidel (true) or GaussSeidel (false).
SolverPerformance smoothSolver(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    bool symmetric,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter = 0,
    int nSweeps = 1,
    const CoupledJumps* jumps = nullptr);

} // namespace brae
