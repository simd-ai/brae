#pragma once
// brae::gamgSolve -- OpenFOAM's GAMGSolver on a symmetric matrix, the host reference: the V-cycle,
// its smoothers, its correction scaling and its coarsest-level solve, on the hierarchy in
// pair_gamg_agglomeration_cpp.cuh.
//
// provenance:
//   openfoam:  src/OpenFOAM/matrices/lduMatrix/solvers/GAMG/GAMGSolver.C:50-90 (the defaults),
//                  :300-340 (the coarsest-level solver), :357-397 (readControls)
//              GAMGSolverSolve.C:36-167 (solve), :170-463 (Vcycle), :466-527 (initVcycle),
//                  :530-558 (the coarsest solver's dictionary), :561-690 (solveCoarsestLevel)
//              GAMGSolverAgglomerateMatrix.C:36-195 (the coarse matrices)
//              GAMGSolverScale.C:34-84 (scale)
//              GAMGAgglomerations/GAMGAgglomeration/GAMGAgglomerationTemplates.C:36-49, :131-176
//                  (restrictField, prolongField)
//              src/OpenFOAM/matrices/lduMatrix/smoothers/DIC/DICSmoother.C:72-127
//              smoothers/DICGaussSeidel/DICGaussSeidelSmoother.C (DIC's sweeps, then Gauss-Seidel's)
//              preconditioners/DICPreconditioner/DICPreconditioner.C:63-122
//              solvers/PCG/PCG.C:67-215
//              lduMatrix/lduMatrixATmul.C (Amul, sumA, residual), lduMatrixSolver.C:235-274
//   tests:     tests/test_inter_gamg_vs_openfoam.cu, tests/interfoam_gamg_vs_openfoam.sh
//
// WHY THIS EXISTS BESIDE brae::gamg. That one is a multigrid; this one is OpenFOAM's. Measured on
// the stokesI wave tank: OpenFOAM against ITSELF with nothing changed but p_rghFinal's solver -- GAMG
// or PCG, the same DIC, the same 1e-7 -- differs by 1.8e-02 of alpha after twenty steps, because the
// tank's active absorption feeds the water level back into the inlet velocity and where the last
// pressure solve stops is part of the answer at that tolerance. Eight of the twelve interFoam tutorials
// brae runs name GAMG for p_rghFinal, so agreeing with them AS SHIPPED needs the solver itself:
// the same hierarchy, the same smoother walking the same face order, the same stopping rule.
//
// WHAT A V-CYCLE IS HERE, since none of it is the textbook's:
//
//   NO PRE-SMOOTHING BY DEFAULT (nPreSweeps 0). The residual is restricted straight down, level to
//   level, by summation; nothing is solved on the way.
//
//   THE COARSEST LEVEL IS SOLVED BY PCG WITH DIC, to the GAMG's OWN tolerance and relTol, from zero.
//   Not a direct solve: with `tolerance 1e-7; relTol 0` it stops at 1e-7 of ITS normFactor, which
//   for a zero start is the sum of |source| -- two iterations on the wave tank's 12-cell level.
//
//   THE WAY UP IS INJECTION, THEN A SCALING, THEN THE SMOOTHER. Each level's correction is
//   OVERWRITTEN by the coarser level's, cell by cell through the restrict map; then scaled by
//   sf = (x.b)/(x.Ax) with a Jacobi step on what is left, x = sf*x + (b - sf*Ax)/D -- for a
//   symmetric matrix only, and NOT on the level directly above the coarsest, "because it evaluates
//   to 1"; then smoothed min(nPostSweeps + postSweepsLevelMultiplier*level, maxPostSweeps) times,
//   so 2, 3, 4, 4, ... from the first coarse level down. The finest level is scaled against the
//   residual, added to psi, and psi itself is smoothed nFinestSweeps times against the SOURCE.
//
//   THE SMOOTHER IS DIC AS AN ITERATION, not as a preconditioner: each sweep takes the true
//   residual, applies the incomplete-Cholesky solve to it, and adds the result to psi.
//
// NOT PORTED, refused where the entry is read: an asymmetric matrix (the smoothers and the coarsest
// solver differ), interpolateCorrection, directSolveCoarsest, a coarsestLevelCorr sub-dictionary,
// and any smoother but DIC, DICGaussSeidel, GaussSeidel and symGaussSeidel.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include "pair_gamg_agglomeration_cpp.cuh"
#include "pcg.cuh"   // SolverPerformance
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {

// GAMGSolver's controls with GAMGSolver.C's defaults; tolerance, relTol, maxIter and minIter are
// lduMatrix::solver's (lduMatrixSolver.C:195-205)
struct GamgControls
{
    std::string smoother;
    scalar tolerance = 1e-6;
    scalar relTol = 0;
    int maxIter = 1000;
    int minIter = 0;
    int nPreSweeps = 0;
    int preSweepsLevelMultiplier = 1;
    int maxPreSweeps = 4;
    int nPostSweeps = 2;
    int postSweepsLevelMultiplier = 1;
    int maxPostSweeps = 4;
    int nFinestSweeps = 2;
    // matrix.symmetric() unless the case says otherwise
    bool scaleCorrection = true;
    // GAMGAgglomeration's, read from the same dictionary
    label nCellsInCoarsestLevel = 10;
};

// THE HIERARCHY BELONGS TO THE MESH, NOT TO THE SOLVE. GAMGAgglomeration is a MeshObject: the first
// GAMG solve of a run builds it, from THAT entry's nCellsInCoarsestLevel, and every later solve on
// the mesh -- another field's, another entry's -- finds it there and reads no dictionary
// (GAMGAgglomeration.C:303-325). On a mesh that does not move it is never rebuilt. `forward` is
// pairGAMGAgglomeration's static direction flag, which outlives the object.
struct GamgAgglomerationCache
{
    bool built = false;
    bool forward = true;
    GamgAgglomeration agglomeration;

    const GamgAgglomeration& get(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        label nCellsInCoarsestLevel)
    {
        if (!built)
        {
            agglomeration = faceAreaPairGamgAgglomeration(m, g, nCellsInCoarsestLevel, forward);
            built = true;
        }
        return agglomeration;
    }
};

// A `preconditioner { preconditioner GAMG; ... }` sub-dictionary: GAMGSolver's controls read from IT --
// its own tolerance and relTol, which only the coarsest-level PCG uses, its smoother, its sweeps --
// and nVcycles, the V-cycles per application (GAMGPreconditioner.C:60, default 2).
struct GamgPreconditionerControls
{
    GamgControls gamg;
    int nVcycles = 2;
};

// True for the smoothers gamgSolve runs. The reader refuses the rest by name.
bool gamgSmootherPorted(const std::string& smoother);

// What OpenFOAM prints under `DebugSwitches { GAMG 1; }`: the coarsest-level solve of every V-cycle
struct GamgSolveLog
{
    std::vector<SolverPerformance> coarsest;
};

// THE COARSEST-LEVEL SOLVE ON ITS OWN: PCG with DIC on a bare symmetric LDU system, from the psi
// handed in, with every control but tolerance and relTol at lduMatrix::solver's default
// (GAMGSolverSolve.C:530-544). Public because the device V-cycle runs this level on the HOST: it has
// at least nCellsInCoarsestLevel cells -- ten by default, twelve on the wave tank -- which is a pair of
// short copies against dozens of kernel launches and a read-back per iteration.
SolverPerformance gamgCoarsestPcgDic(
    const GamgLduAddressing& addr,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    std::vector<scalar>& psi,
    const std::vector<scalar>& source,
    scalar tolerance,
    scalar relTol);

// GAMGSolver::solve. Folds the boundary like fvMatrix::solve and solves A*psi = b in place. Throws
// if M is not symmetric, or names a smoother gamgSmootherPorted() does not list.
SolverPerformance gamgSolve(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const GamgAgglomeration& agglomeration,
    const GamgControls& controls,
    GamgSolveLog* log = nullptr);

// PCG::scalarSolve with the GAMGPreconditioner: `solver PCG; preconditioner { preconditioner GAMG; ...}`,
// which six of the seven solid-body tutorials name for p_rghFinal and pcorr. Each application starts
// from zero and runs nVcycles V-cycles of the SAME cycle gamgSolve runs, on a GAMGSolver built on the
// PCG's own matrix with the sub-dictionary's controls; between cycles the residual is recomputed. The
// preconditioner is built only if the first residual does not already satisfy the tolerance, as
// PCG.C builds it. tolerance, relTol, maxIter and minIter are the PCG's.
SolverPerformance pcgGamgSolve(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const GamgAgglomeration& agglomeration,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter,
    const GamgPreconditionerControls& precond,
    GamgSolveLog* log = nullptr);

} // namespace brae
