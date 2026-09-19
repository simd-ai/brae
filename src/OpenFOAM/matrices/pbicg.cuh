#pragma once
// brae::pbicg -- OpenFOAM's PBiCG (preconditioned bi-conjugate gradient) with the DILU preconditioner,
// the host reference.
//
// provenance:
//   openfoam: src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C (solve)
//             src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/DILUPreconditioner.C
//                 (calcReciprocalD, precondition, preconditionT)
//             src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixATmul.C (Amul, Tmul, sumA)
//             src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixSolver.C (normFactor)
//   brae:     pbicgstab.cu carries the same completion, normFactor and DILU forward sweep
//   tests:    tests/interfoam_mangrove_vs_openfoam.sh -- k and epsilon on waves/mangroveInteraction,
//             every count and residual against OpenFOAM's log
//
// NOT PBiCGStab. The two share a preconditioner and nothing else: PBiCG carries a TRANSPOSE system
// (Tmul, preconditionT) beside the direct one, and at the same tolerance the two stop at different
// iterates. A case that names PBiCG runs this, or is refused.
//
// THE SWEEPS IN FACE ORDER. OpenFOAM's DILU walks the lower triangle in losortAddr order; for any one
// cell the subtractions it takes still arrive in increasing face order, and every value a subtraction
// reads is final by then, so face order gives the same bits (pbicgstab.cu relies on the same).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include "pcg.cuh"   // SolverPerformance
#include <vector>

namespace brae {

SolverPerformance pbicgDILU(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter = 0);

} // namespace brae
