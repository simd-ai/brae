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
#include "smooth_solver_cpp.cuh"
#include "solution_directions.cuh"
#include <vector>

namespace brae {

// The choice is LinearSolverChoice (smooth_solver_cpp.cuh). `smoothSolver` runs component by component,
// as fvMatrix<vector>::solveSegregated does.
using VectorLinearSolver = LinearSolverChoice;

// `solutionD` is fvMesh::validComponents<vector>() -- polyMesh::solutionD(): given, a component it
// knocks out (-1) is NOT solved, exactly fvMatrixSolve.C:162-164's `continue`, and its entry in
// `perfCmpt` stays default-constructed (initialResidual 0, SolverPerformance.H:117-121). Null solves
// all three, which is what the simpleFoam callers still ask for. `perfCmpt`, when given, receives the
// three per-component performances (fvMatrixSolve.C:235 `solverPerfVec.replace(cmpt, solverPerf)`);
// the RETURN stays component 0's, for the callers that predate it.
inline SolverPerformance solveVector(
    const FvVectorMatrix& M,
    GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter = 0,
    const SolutionDirections* solutionD = nullptr,
    SolverPerformance* perfCmpt = nullptr,
    const VectorLinearSolver* which = nullptr)
{
    const label nC = m.nCells();
    // fvMatrix<Type>::solveSegregated (fvMatrixSolve.C) on a COUPLED patch: addBoundarySource adds
    // cmptMultiply(boundaryCoeffs, patchNeighbourField()) -- the neighbour's whole VECTOR -- to the
    // source once, and each component's updateMatrixInterfaces(add = true) then takes back
    // boundaryCoeffs_c*pnf_c, the interface's own interpolate of that component. Untransformed the two
    // cancel but for round-off, which is OpenFOAM's: the sum is formed and then undone, not skipped.
    // The component solver carries the interface from there. PBiCGStab here does not.
    bool anyCoupled = false;
    for (const FvPatch& fp : patches)
    {
        anyCoupled = anyCoupled || fp.coupled;
    }
    if (anyCoupled && !(which && which->smoothSolver))
    {
        refuseCoupledPatches(patches, "the segregated vector solve with PBiCGStab");
    }
    std::vector<vector> source = M.source;
    if (anyCoupled)
    {
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& fp = patches[pi];
            if (!fp.coupled) continue;
            const std::vector<vector> pnf = U.boundary[pi]->patchNeighbourField(U.internal);
            for (label i = 0; i < fp.size; ++i)
            {
                const vector& bc = M.boundaryCoeffs[pi][i];
                const vector& n = pnf[static_cast<std::size_t>(i)];
                source[fp.faceCells[i]] += vector{bc.x*n.x, bc.y*n.y, bc.z*n.z};
            }
        }
    }
    SolverPerformance perf;
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        if (perfCmpt) perfCmpt[cmpt] = SolverPerformance();
        if (solutionD && !solutionD->valid(cmpt)) continue;
        FvScalarMatrix Mc;
        Mc.diag = M.diag;
        Mc.upper = M.upper;
        Mc.lower = M.lower;
        Mc.source.resize(nC);
        for (label c = 0; c < nC; ++c) Mc.source[c] = component(source[c], cmpt);
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
        for (std::size_t pi = 0; anyCoupled && pi < patches.size(); ++pi)
        {
            const FvPatch& fp = patches[pi];
            if (!fp.coupled) continue;
            for (label i = 0; i < fp.size; ++i)
            {
                Mc.source[fp.faceCells[i]] -= Mc.boundaryCoeffs[pi][i]*patchNeighbourValue(fp, i, psi);
            }
        }
        const SolverPerformance p = (which && which->smoothSolver)
            ? smoothSolver(Mc, psi, m, patches, which->symmetric, tolerance, relTol, maxIter, minIter,
                           which->nSweeps)
            : pbicgstab(Mc, psi, m, patches, tolerance, relTol, maxIter, minIter);
        for (label c = 0; c < nC; ++c) setComponent(U.internal[c], cmpt, psi[c]);
        if (perfCmpt) perfCmpt[cmpt] = p;
        if (cmpt == 0) perf = p;
    }
    U.evaluateBoundary();   // correctBoundaryConditions
    return perf;
}

} // namespace brae
