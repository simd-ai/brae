#pragma once
// brae::pcg, preconditioned conjugate gradient with DIC preconditioner, transcribed from
// OpenFOAM lduMatrix PCG + DICPreconditioner + lduMatrix::solver::normFactor. Completes the
// matrix (boundary diag/source) like fvMatrix::solve, then solves A*psi = b in place.
// Symmetric matrices only (DIC). Serial (global reductions are local sums here).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

struct SolverPerformance
{
    scalar initialResidual = 0.0;
    scalar finalResidual   = 0.0;
    int    nIterations      = 0;
};

// COUPLED PATCHES (FvPatch::coupled). Their boundaryCoeffs are INTERFACE coefficients: fvMatrix::solve
// folds internalCoeffs into the diagonal and adds NOTHING to the source (addBoundarySource(source,
// false)); Amul ends with result[faceCell] -= coeff*pnf, sumA takes -coeff, and DIC never sees them.
// pnf is the cell on the other side -- LESS the patch's jump when, and only when, the operand is the
// solution field itself (jumpCyclicFvPatchFields.C, "only apply jump to original field"), which in PCG
// is the one Amul that builds the initial residual and the normalisation. `jumps`, per patch, is that
// already-signed jump; null, or a null entry, is a patch without one.
using CoupledJumps = std::vector<const std::vector<scalar>*>;

SolverPerformance pcg(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter = 0,
    const CoupledJumps* jumps = nullptr);

// A solver that holds no interface coefficients cannot run a matrix whose coupled patches carry them:
// it would solve the two sides of the pair as unconnected walls and converge.
inline void refuseCoupledPatches(
    const std::vector<FvPatch>& patches,
    const char* who)
{
    for (const FvPatch& fp : patches)
    {
        if (fp.coupled)
        {
            throw std::runtime_error(
                std::string("brae: ") + who + " does not carry the interface coefficients of the coupled "
                "patch '" + fp.name + "'. Across a cyclic the OF-mirror host path solves with PCG (DIC) and "
                "smoothSolver (GaussSeidel, symGaussSeidel) only.");
        }
    }
}

// result[faceCell] -= coeff*pnf over every coupled patch: lduMatrix::updateMatrixInterfaces as Amul calls
// it. `sign` +1 turns it into the residual's and Gauss-Seidel's form, which ADD. `isField` says the
// operand is the solution field, the one case in which a jump is subtracted from the neighbour cell.
inline void updateCoupledInterfaces(
    const FvScalarMatrix& M,
    const std::vector<FvPatch>& patches,
    const std::vector<scalar>& x,
    std::vector<scalar>& result,
    scalar sign,
    bool isField,
    const CoupledJumps* jumps)
{
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        if (!fp.coupled)
        {
            continue;
        }
        const std::vector<scalar>* jump =
            (isField && jumps && pi < jumps->size()) ? (*jumps)[pi] : nullptr;
        for (label i = 0; i < fp.size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(i);
            scalar pnf = x[static_cast<std::size_t>(fp.nbrFaceCells[k])];
            if (jump)
            {
                pnf -= (*jump)[k];
            }
            result[static_cast<std::size_t>(fp.faceCells[k])] += sign*(M.boundaryCoeffs[pi][k]*pnf);
        }
    }
}

} // namespace brae
