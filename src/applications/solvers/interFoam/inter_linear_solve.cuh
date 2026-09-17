#pragma once
// One fvSolution `solvers/<field>` entry, as interFoam's smoothSolver fields name it.
//
// provenance:
//   openfoam:  src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixSolver.C:195-205 (readControls:
//              tolerance 1e-6, relTol 0, maxIter 1000, minIter 0 when absent)
//              src/OpenFOAM/matrices/lduMatrix/solvers/smoothSolver/smoothSolver.C:78 (nSweeps 1)
//
// Hoisted out of InterFields, where it was the nested AlphaLinearSolve, because the turbulence block
// reads the same entry for k and epsilon and InterFields holds that block.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <string>

namespace brae {
namespace cpu {
namespace interFoam {

struct SmoothLinearSolve
{
    std::string solver;
    std::string smoother;
    scalar tol = 1e-6;
    scalar relTol = 0;
    int maxIter = 1000;
    // the floor on the iteration count: the loop continues while nIterations < minIter even when
    // converged (smoothSolver.C:174-211). RAS/damBreak names 1 for U, k and epsilon.
    int minIter = 0;
    int nSweeps = 1;
    bool gaussSeidel() const
    {
        return solver == "smoothSolver" && (smoother == "symGaussSeidel" || smoother == "GaussSeidel");
    }

    static SmoothLinearSolve read(const FoamDict& d)
    {
        SmoothLinearSolve s;
        s.solver = d.wordOr("solver", "");
        s.smoother = d.wordOr("smoother", "");
        s.tol = d.scalarOr("tolerance", scalar(1e-6));
        s.relTol = d.scalarOr("relTol", scalar(0));
        s.maxIter = static_cast<int>(d.scalarOr("maxIter", scalar(1000)));
        s.minIter = static_cast<int>(d.scalarOr("minIter", scalar(0)));
        s.nSweeps = static_cast<int>(d.scalarOr("nSweeps", scalar(1)));
        return s;
    }
};

} // namespace interFoam
} // namespace cpu
} // namespace brae
