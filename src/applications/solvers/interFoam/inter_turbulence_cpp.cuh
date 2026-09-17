#pragma once
// interFoam's turbulence -- incompressibleInterPhaseTransportModel around kEpsilon, the host reference.
//
// provenance:
//   openfoam:  src/phaseSystemModels/twoPhaseInter/incompressibleInterPhaseTransportModel/
//                  incompressibleInterPhaseTransportModel.C:46-110 (the two lineages, and validate()
//                  in ONE of them), :117-131 (divDevRhoReff), :134-144 (correct)
//              src/TurbulenceModels/incompressible/incompressibleRhoTurbulenceModel.C:41-57 (the
//                  rho-weighted base: alphaRhoPhi and phi are two fields)
//              src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C:214-296 (correct)
//              src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress.C:107-133
//              applications/solvers/multiphase/interFoam/interFoam.C:169-172 (where correct() is called)
//   brae:      kEpsilon_cpp.cuh is the closure; this file only chooses what it is handed
//   tests:     tests/interfoam_ras_dambreak_vs_openfoam.sh against real OpenFOAM, both lineages
//
// THERE ARE TWO MODELS BEHIND ONE KEYWORD, and the case picks by a line most cases do not carry.
//
//   `density` absent or `uniform` (15 of the 17 turbulent tutorials):
//       incompressible::turbulenceModel::New(U, phi, mixture)  -- the ORDINARY single-phase model.
//       alpha = rho = 1, the equations convect with the volumetric phi and look up `div(phi,k)`,
//       and the constructor calls validate(), so nut is Cmu*k^2/epsilon before the first UEqn.
//
//   `density variable` (RAS/damBreak, laminar/damBreakPermeable):
//       phaseIncompressibleTurbulenceModel::New(rho, U, rhoPhi, phi, mixture) -- the same kEpsilon.C
//       with rho = the mixture density: the equations convect with rhoPhi and look up
//       `div(rhoPhi,k)`, every source carries rho, divU still comes from the volumetric phi -- and
//       validate() is NOT called, so the first UEqn runs on the nut the case FILE holds (uniform 0 in
//       RAS/damBreak) where the other lineage would already have 9e-03 from k = epsilon = 0.1.
//
// Both hand kEpsilon the MIXTURE's nu, a field (mu/rho with the clamped alpha), never a scalar.
// UEqn takes the stress as rho*nuEff in both: -fvc::div(rho*nuEff*dev2(T(grad U))) -
// fvm::laplacian(rho*nuEff, U), with nuEff = nut + nu.
//
// WHAT IS REFUSED, by name: every RASModel but kEpsilon, LES, `turbulence off`, a k/epsilon
// convection scheme other than `Gauss upwind` (what all 11 kEpsilon tutorials name), a nut wall
// function outside nutk/nutU/nutLowRe or on a patch that is not a `wall`, and a ddt scheme other than
// Euler. The device loop runs the same closure's device twin: device_inter_turbulence.cuh.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include "geometric_field.cuh"
#include "inter_linear_solve.cuh"
#include "inter_solve_record.cuh"
#include "kepsilon_coeffs.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

// fvMatrix::relax() for one equation name: `if (mesh.relaxEquation(name, coeff)) relax(coeff)`
// (fvMatrix.C:1252-1262), and relaxEquation is "the entry of that name, else `default`" (solution.C:
// 379-417). `on` false means OpenFOAM does not call relax at all -- not the same as a factor of 1,
// which still runs the diagonal-dominance fix.
struct EquationRelax
{
    bool on = false;
    scalar factor = 1;

    static EquationRelax read(
        const FoamDict* equations,
        const std::string& name);
};

struct InterTurbulence
{
    // simulationType RAS. False is laminar: no fields, nuEff = nu, correct() does nothing.
    bool on = false;
    // `density variable` -- see the header
    bool variableDensity = false;
    KEpsilonCoeffs coeffs;
    GeometricField<scalar> k;
    GeometricField<scalar> epsilon;
    GeometricField<scalar> nut;
    // per patch, a NutWall value; read where the dictionary TYPE still exists
    std::vector<int> nutWallKind;
    // THE Final ENTRIES, AND ONLY THOSE. fvMatrix::solve() and fvMatrix::relax() both select
    // <field>Final on the final outer corrector, and with turbOnFinalIterOnly (pimpleControl.C:51-52,
    // default true) that is the only corrector the closure runs on. `turbOnFinalIterOnly no` with
    // more than one outer corrector is refused: no tutorial sets it, and the second call inside a
    // time step needs k.oldTime(), which is not the field at entry.
    SmoothLinearSolve kSolveFinal;
    SmoothLinearSolve epsSolveFinal;
    EquationRelax kRelaxFinal;
    EquationRelax epsRelaxFinal;
};

// Reads constant/turbulenceProperties, the three fields and their solver, scheme and relaxation
// entries; refuses what is not ported. `laplacianCorrected`/`laplacianLimitCoeff` are the case's
// laplacianSchemes default, resolved once by the caller for every equation.
InterTurbulence readInterTurbulence(
    const std::string& caseDir,
    const std::string& startDir,
    const FoamDict& fvSolution,
    bool eulerDdt,
    bool laplacianCorrected,
    scalar laplacianLimitCoeff,
    const std::vector<FvPatch>& patches,
    label nCells);

// turbulence->validate(), which incompressibleInterPhaseTransportModel's constructor calls in the
// UNIFORM lineage only. Does nothing in the variable one, and nothing when laminar.
void validateInterTurbulence(
    InterTurbulence& t,
    const GeometricField<vector>& U,
    const std::vector<scalar>& nu,
    const std::vector<std::vector<scalar>>& nuBnd,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

// nuEff = nut + nu on cells and on every patch (eddyViscosity::nuEff). Laminar returns nu.
void interNuEff(
    const InterTurbulence& t,
    const std::vector<scalar>& nu,
    const std::vector<std::vector<scalar>>& nuBnd,
    std::vector<scalar>& nuEff,
    std::vector<std::vector<scalar>>& nuEffBnd);

struct InterTurbulenceStepInput
{
    const GeometricField<vector>* U = nullptr;
    // the volumetric flux, and the mass flux MULES left -- both, because the variable lineage
    // convects with one and takes divU from the other
    const SurfaceScalarField* phi = nullptr;
    const SurfaceScalarField* rhoPhi = nullptr;
    const std::vector<scalar>* rho = nullptr;
    const std::vector<std::vector<scalar>>* rhoBnd = nullptr;
    // rho.oldTime(): the density the time step STARTED on
    const std::vector<scalar>* rhoOld = nullptr;
    const std::vector<scalar>* nu = nullptr;
    const std::vector<std::vector<scalar>>* nuBnd = nullptr;
    scalar deltaT = 0;
    // every solve of the run, in order, for the solver-log gate
    std::vector<LinearSolveRecord>* epsilonLog = nullptr;
    std::vector<LinearSolveRecord>* kLog = nullptr;
};

// turbulence->correct(), interFoam.C:171.
void correctInterTurbulence(
    InterTurbulence& t,
    const InterTurbulenceStepInput& in,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

} // namespace interFoam
} // namespace cpu
} // namespace brae
