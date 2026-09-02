#pragma once
// brae::solveVector, segregated solve of a vector fvMatrix: solve each component with the
// scalar solver (PBiCGStab), as OpenFOAM fvMatrix<vector>::solve() does. The lduMatrix
// coefficients are shared; source / boundary coeffs are extracted per component.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include "geometric_field.cuh"
#include "pbicgstab.cuh"
#include <vector>

namespace brae {

inline SolverPerformance solveVector(
    const FvVectorMatrix& M,
    GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter = 0)
{
    const label nC = m.nCells();
    SolverPerformance perf;
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        FvScalarMatrix Mc;
        Mc.diag = M.diag;
        Mc.upper = M.upper;
        Mc.lower = M.lower;
        Mc.source.resize(nC);
        for (label c = 0; c < nC; ++c) Mc.source[c] = component(M.source[c], cmpt);
        Mc.internalCoeffs.resize(patches.size());
        Mc.boundaryCoeffs.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            Mc.internalCoeffs[pi].resize(patches[pi].size);
            Mc.boundaryCoeffs[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
            {
                Mc.internalCoeffs[pi][i] = component(M.internalCoeffs[pi][i], cmpt);
                Mc.boundaryCoeffs[pi][i] = component(M.boundaryCoeffs[pi][i], cmpt);
            }
        }
        std::vector<scalar> psi(nC);
        for (label c = 0; c < nC; ++c) psi[c] = component(U.internal[c], cmpt);
        const SolverPerformance p = pbicgstab(Mc, psi, m, patches, tolerance, relTol, maxIter, minIter);
        for (label c = 0; c < nC; ++c) setComponent(U.internal[c], cmpt, psi[c]);
        if (cmpt == 0) perf = p;
    }
    U.evaluateBoundary();   // correctBoundaryConditions
    return perf;
}

} // namespace brae
