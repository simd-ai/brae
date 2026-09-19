#pragma once
// interFoam's turbulence -- incompressibleInterPhaseTransportModel around kEpsilon or kOmegaSST, the
// host reference.
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
//              src/TurbulenceModels/turbulenceModels/Base/kOmegaSST/kOmegaSSTBase.C:497-612 (correct),
//                  :117-126 (correctNut), :408-461 (decayControl)
//   brae:      kEpsilon_cpp.cuh and kOmegaSST_cpp.cuh are the closures; this file only chooses what
//              they are handed
//   tests:     tests/interfoam_ras_dambreak_vs_openfoam.sh against real OpenFOAM, both lineages;
//              tests/interfoam_waterchannel_vs_openfoam.sh, kOmegaSST on RAS/waterChannel
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
// kOmegaSST (RAS/waterChannel, and four tutorials that need more than the model) is the UNIFORM lineage
// only: the ordinary incompressible kOmegaSST, handed the mixture's nu as a field, the volumetric phi,
// and the CELL wall distance wallDist::New(mesh).y() for F1 and F2. The closure applies
// omegaWallFunction and nutkWallFunction on every patch whose MESH type is `wall`, so the reader holds
// the case to exactly that: each `wall` patch carries both, no other patch carries either.
//
// LES kEqn (LES/nozzleFlow2D) is the UNIFORM lineage only, with the cubeRootVol or smooth filter width --
// les_kEqn_cpp.cuh and les_delta_cpp.cuh carry the equation and the width. validate() is its correctNut,
// nut = Ck*sqrt(k)*delta, and nothing on its walls is a wall function.
//
// WHAT IS REFUSED, by name: every RASModel but kEpsilon and kOmegaSST, every LESModel but kEqn, `turbulence off`, a
// convection scheme on the turbulence scalars other than `Gauss upwind` (what every tutorial of either
// model names), a nut wall function outside nutk/nutU/nutLowRe (kEpsilon) or other than nutk
// (kOmegaSST) or on a patch that is not a `wall`, and a ddt scheme other than Euler. Under kOmegaSST
// also: `density variable` (no tutorial pairs them, so no gate would hold it), F3, decayControl, a
// wall-function blending other than the default binomial n = 2, wall-function coefficients other than
// the defaults, and a moving mesh (y is taken once). The device loop runs kEpsilon's device twin,
// device_inter_turbulence.cuh, and refuses kOmegaSST by name.
#include "fvOptions_cpp.cuh"
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include "geometric_field.cuh"
#include "inter_linear_solve.cuh"
#include "inter_solve_record.cuh"
#include "kepsilon_coeffs.cuh"
#include "komega_sst_coeffs.cuh"
#include "les_delta_cpp.cuh"
#include "les_kEqn_cpp.cuh"
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

enum class InterRasModel
{
    KEpsilon,
    KOmegaSST,
    // simulationType LES, LESModel kEqn -- the one LES model wired in (LES/nozzleFlow2D); the enum keeps
    // its RAS name because every consumer already switches on it
    KEqnLES
};

struct InterTurbulence
{
    // simulationType RAS. False is laminar: no fields, nuEff = nu, correct() does nothing.
    bool on = false;
    InterRasModel model = InterRasModel::KEpsilon;
    // `density variable` -- see the header
    bool variableDensity = false;
    KEpsilonCoeffs coeffs;
    GeometricField<scalar> k;
    // kEpsilon's second scalar; empty under kOmegaSST
    GeometricField<scalar> epsilon;
    // kOmegaSST's; empty under kEpsilon
    GeometricField<scalar> omega;
    KOmegaSSTCoeffs sstCoeffs;
    // wallDist::New(mesh).y(), the CELL wall distance F1 and F2 take -- not the near-wall face
    // distance the wall functions use. Taken once: kOmegaSST on a moving mesh is refused.
    std::vector<scalar> yCell;
    GeometricField<scalar> nut;
    // kEqn's: its coefficients, the filter width (constant on a mesh that does not move), and k's
    // solve on the final outer corrector
    LESkEqn::Coeffs lesCoeffs;
    LESdelta::Spec deltaSpec;
    std::vector<scalar> delta;
    // a gate's window into kEqn's stages at every correct(); null in a run
    LESkEqn::Taps* lesTaps = nullptr;
    // per patch, a NutWall value; read where the dictionary TYPE still exists
    std::vector<int> nutWallKind;
    // THE Final ENTRIES, AND ONLY THOSE. fvMatrix::solve() and fvMatrix::relax() both select
    // <field>Final on the final outer corrector, and with turbOnFinalIterOnly (pimpleControl.C:51-52,
    // default true) that is the only corrector the closure runs on. `turbOnFinalIterOnly no` with
    // more than one outer corrector is refused: no tutorial sets it, and the second call inside a
    // time step needs k.oldTime(), which is not the field at entry.
    SmoothLinearSolve kSolveFinal;
    SmoothLinearSolve epsSolveFinal;
    SmoothLinearSolve omegaSolveFinal;
    EquationRelax kRelaxFinal;
    EquationRelax epsRelaxFinal;
    EquationRelax omegaRelaxFinal;
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
    label nCells,
    // kOmegaSST's cell wall distance needs the mesh; null is a caller that can only run kEpsilon
    const PrimitiveMesh* mesh = nullptr,
    const FvGeometry* geometry = nullptr);

// turbulence->validate(), which incompressibleInterPhaseTransportModel's constructor calls in the
// UNIFORM lineage only. Does nothing in the variable one, and nothing when laminar.
void validateInterTurbulence(
    InterTurbulence& t,
    const GeometricField<vector>& U,
    const std::vector<scalar>& nu,
    const std::vector<std::vector<scalar>>& nuBnd,
    const SurfaceScalarField& phi,     // the flux nut's inletOutlet patches decide inflow by
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
    // A MOVING MESH: the old volumes and the mesh flux (see kEpsilonRef::Compressible). Null on a
    // static mesh.
    const std::vector<scalar>* V0 = nullptr;
    const SurfaceScalarField* meshPhi = nullptr;
    // every solve of the run, in order, for the solver-log gate
    // fvOptions(epsilon) and fvOptions(k), kEpsilon.C:258/279: the mangroves' turbulence source. Null
    // or empty is a case with none.
    const cpu::fvOptions::OptionList* fvOptions = nullptr;
    std::vector<LinearSolveRecord>* epsilonLog = nullptr;
    std::vector<LinearSolveRecord>* kLog = nullptr;
    // kOmegaSST's first solve, omega before k as kOmegaSSTBase.C:555-607 has them
    std::vector<LinearSolveRecord>* omegaLog = nullptr;
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
