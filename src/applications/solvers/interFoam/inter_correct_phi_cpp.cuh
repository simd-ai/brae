#pragma once
// interFoam's correctPhi -- CorrectPhi's pcorr equation, at the start of every case and after every
// mesh update under `correctPhi`. The host reference.
//
// provenance:
//   openfoam: src/finiteVolume/cfdTools/general/CorrectPhi/CorrectPhi.C:36-117 (the incompressible
//                 template)
//             src/finiteVolume/cfdTools/general/CorrectPhi/correctUphiBCs.C:34-66
//             applications/solvers/multiphase/interFoam/initCorrectPhi.H, correctPhi.H
//             applications/solvers/multiphase/interFoam/interFoam.C:130-148
//
// WHY IT LIVES HERE. OpenFOAM's CorrectPhi is general cfdTools, and the mirror's place for it is
// src/finiteVolume/cfdTools/general/CorrectPhi/. What it calls -- adjustPhi, makeRelative and
// makeAbsolute, the p_rgh solve's dispatch -- is still interFoam's own (inter_peqn_cpp.cuh), so it sits
// beside them until they move.
//
// WHAT ONE CALL DOES, in CorrectPhi.C's order:
//   1. correctUphiBCs, and only when the mesh is changing: every velocity patch that FIXES a value is
//      evaluated again -- a pressureInletOutletVelocity against the flux it has just been handed -- and
//      phi on it is set to U_b & Sf, overwriting what Sf & Uf gave there;
//   2. pcorr: zero, fixedValue where p_rgh fixes a value and zeroGradient elsewhere;
//   3. if pcorr then needs a reference -- a closed domain -- phi is made relative, adjustPhi balances
//      its adjustable outflow, and phi is made absolute again;
//   4. laplacian(rAUf, pcorr) == div(phi), pinned at cell 0 to 0 when it needs a reference, solved
//      nNonOrthogonalCorrectors + 1 times with pcorrFinal on the last pass;
//   5. on the last pass only, phi -= pcorrEqn.flux(), the non-orthogonal face correction included.
// The caller then makes phi relative (interFoam.C:141) and corrects the mixture on the moved mesh.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "gamg_solver_cpp.cuh"
#include "geometric_field.cuh"
#include "inter_case_cpp.cuh"
#include "inter_solve_record.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

struct CorrectPhiControls
{
    // solvers/pcorr, for every non-orthogonal pass but the last, and solvers/pcorrFinal for the last
    const InterFields::PressureLinearSolve* pcorr = nullptr;
    const InterFields::PressureLinearSolve* pcorrFinal = nullptr;
    // the mesh's GAMG hierarchy, which a GAMG pcorr solve shares with p_rgh's
    GamgAgglomerationCache* gamgCache = nullptr;
    // laplacianSchemes' default, for fvm::laplacian(rAUf, pcorr)
    bool correctedLaplacian = false;
    scalar snGradLimitCoeff = 0;
    // grad(pcorr)'s gradSchemes entry, which the laplacian's correction takes
    GradChoice gradPcorr;
    label nNonOrthogonalCorrectors = 0;
};

struct CorrectPhiInput
{
    // rAUf: interpolate(rAU) under correctPhi, or 1 on every face
    const SurfaceScalarField* rAUf = nullptr;
    // polyMesh::changing(): correctUphiBCs acts only then, so never at the start of a case
    bool meshChanging = false;
    // fvc::meshPhi, for makeRelative and makeAbsolute around adjustPhi; null when the mesh is not
    // moving, where both are no-ops
    const SurfaceScalarField* meshPhi = nullptr;
    // the alpha step's mass flux, for a velocity patch whose condition names `phi rhoPhi`
    const SurfaceScalarField* rhoPhi = nullptr;
    // one record per pcorr solve; null = not kept
    std::vector<LinearSolveRecord>* solveLog = nullptr;
};

// correctUphiBCs(U, phi)
void correctUphiBCs(
    GeometricField<vector>& U,
    SurfaceScalarField& phi,
    const SurfaceScalarField* rhoPhi,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

// CorrectPhi(U, phi, p_rgh, rAUf, geometricZeroField(), pimple)
void correctPhi(
    GeometricField<vector>& U,
    SurfaceScalarField& phi,
    const GeometricField<scalar>& p_rgh,
    const CorrectPhiInput& in,
    const CorrectPhiControls& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

// rAUf at the start of a case: EXACTLY 1 on every face. initCorrectPhi.H passes either the
// dimensionedScalar 1 or interpolate(rAU) of a field of 1, and OpenFOAM's linear interpolation,
// lambda*(P - N) + N, returns 1 there to the bit; brae's w*P + (1 - w)*N need not.
SurfaceScalarField unitFaceField(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches);

} // namespace interFoam
} // namespace cpu
} // namespace brae
