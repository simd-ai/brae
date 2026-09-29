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
#include "inter_cn_restart.cuh"
#include "fvOptions_cpp.cuh"
#include "crank_nicolson_ddt_scheme_cpp.cuh"
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include "geometric_field.cuh"
#include "limitedSchemes_cpp.cuh"   // EqnDivScheme
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

// The closure's CrankNicolson state: the two ddt0 fields OpenFOAM keeps on the registry, and the
// old-old level of each field, which GeometricField::storeOldTimes rotates once per time index --
// `entry` is the field at this step's first correct() (its oldTime()), `oo` the previous step's.
struct InterTurbulenceCrankNicolson
{
    fv::CrankNicolsonDdt0<scalar> ddt0K;
    fv::CrankNicolsonDdt0<scalar> ddt0Eps;
    // the OLD-OLD level, which only CrankNicolson reads. The old-TIME level it used to sit beside is now
    // InterTurbulence::kOldStep, because every ddt scheme needs it -- see there.
    std::vector<scalar> kOO;
    std::vector<scalar> epsOO;
    // A RESTART: where OpenFOAM would look these two fields up (inter_cn_restart.cuh). The names are the
    // OPERANDS' -- "ddt0(rho,k)" under the variable lineage, "ddt0(k)" under the uniform one,
    // "ddt0(omega)" under kOmegaSST -- so they are known only once the model branch has run, which is
    // why the seed is done there and not at construction. Once per run.
    InterCnRestart restart;
    bool           restartSeeded = false;
};

// fvSolution's `cache { grad(U); }`: the grad(U) OpenFOAM keeps in the mesh registry. gradScheme::grad
// (gradScheme.C:120-160) forms and stores it at the first request, then returns it -- "Reusing" -- while
// U's eventNo is unchanged, and forms it again -- "Updating" -- at the first request after U changed.
// On RAS/DTCHull (solution's DebugSwitch, measured) kOmegaSST's validate forms it, UEqn's three sites
// reuse it at the first step, turbulence->correct updates it, and the next step's UEqn reuses THAT one.
// It is not the uncached answer: the fvMatrix constructor restores U's eventNo around its updateCoeffs
// (fvMatrix.C:394-397), so the atmosphere's pressureInletOutletVelocity, whose updateCoeffs ends in
// evaluate(), moves U's patch values under a gradient that is not refreshed. OpenFOAM against itself,
// cached against uncached: U 2.0e-06 at the first step, k 8.1e-03 and nut 2.6e-02 at the second.
struct GradUCache
{
    // the case names grad(U) in an active cache block
    bool on = false;
    // formed since U last changed -- OpenFOAM's `pgGrad->upToDate(vsf)`
    bool valid = false;
    std::vector<tensor> cells;
    // the boundary gaussGrad corrected when the gradient was formed (gaussGrad::correctBoundaryConditions,
    // against U's patch values THEN), which the dev2 term reads
    std::vector<std::vector<tensor>> bnd;
};

// Store a grad(U) just formed from U as it stands -- a no-op when the case caches nothing.
void storeGradU(
    GradUCache& cache,
    const std::vector<tensor>& gradU,
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

struct InterTurbulence
{
    // simulationType RAS. False is laminar: no fields, nuEff = nu, correct() does nothing.
    bool on = false;
    // `RAS { turbulence off; }` / `LES { turbulence off; }` -- a model that is CONSTRUCTED and
    // VALIDATED and then never corrected again. It is NOT laminar and it is NOT "keep the file's nut":
    //   * the constructor still bounds the two transported scalars (kEpsilon.C:182-183,
    //     kOmegaSSTBase.C:438-439, kEqn's bound(k_, kMin_))
    //   * validate() is NOT gated on turbulence_ (eddyViscosity.C:119-122 is `correctNut();` alone) and
    //     interFoam's uniform-density lineage calls it
    //     (incompressibleInterPhaseTransportModel.C:105), so nut is REBUILT from the bounded file
    //     fields by the model's own formula
    //   * only correct() returns early (kEpsilon.C:216-219, kOmegaSSTBase.C:502-505, kEqn.C:141-144),
    //     so k, the second scalar and nut all hold that one value for the whole run
    // `on` therefore stays TRUE: nuEff is still nut + nu, and the nut it adds is validate()'s.
    bool frozen = false;
    InterTurbulenceCrankNicolson cn;
    // psi.oldTime() FOR THE CLOSURE, kept per TIME INDEX and not per call.
    //
    // OpenFOAM's is the field at the step's FIRST non-const access: GeometricField::storeOldTimes() is
    // guarded on `timeIndex_ != time().timeIndex()` (GeometricField.C:904-917, and the const oldTime()
    // accessor calls it at :976), so the first access in a step stores the old level and every later one
    // in the same index is a no-op. With `turbOnFinalIterOnly no` and more than one outer corrector the
    // closure runs MORE THAN ONCE in a step, and corrector 2's fvm::ddt must still read the PREVIOUS
    // STEP's k -- not corrector 1's solved-and-bounded k.
    //
    // Every closure arm used to recapture it per CALL (`const std::vector<scalar> kOld = k.internal;`),
    // which is identical while the closure runs once per step and first order in dt wrong as soon as it
    // does not. These three were `cn.kEntry`/`cn.epsEntry`/`cn.timeIndex`, filled only when the scheme was
    // CrankNicolson; they are the snapshot every scheme needs, so they live here and are advanced
    // unconditionally.
    std::vector<scalar> kOldStep;
    std::vector<scalar> epsOldStep;
    label oldStepTimeIndex = -1;
    InterRasModel model = InterRasModel::KEpsilon;
    // `density variable` -- see the header
    bool variableDensity = false;
    // div(<flux>,k) and div(<flux>,<second field>): `Gauss upwind` or `Gauss limitedLinear <k>`, ONE
    // PER EQUATION. `fvm::div(phi, psi)` resolves the entry by the FIELD's name, so the two need not
    // agree and OpenFOAM assembles two different matrices; this carried a single pair and the reader
    // refused a mismatch. The host closures take k's positionally and the second through EqnDivScheme.
    // kOmegaSST also takes `Gauss linearUpwind <grad>` -- on the pair, which must agree (one flag and one
    // limiter coefficient in its closure) -- with luGradK the NAMED gradient's cellLimited coefficient;
    // the device closure refuses it (device_inter_turbulence.cu).
    cpu::EqnDivScheme kDiv;
    cpu::EqnDivScheme secondDiv;
    // grad(k) and grad(<second field>), ONE PER EQUATION. `fvc::grad(vf)` resolves `grad(<vf>)` by the
    // FIELD's name, so the two need not agree; the closures carried one pair of flags and the reader
    // refused a mismatch. Measured on RAS/angledDuct (44.5 deg non-orthogonal, `corrected` laplacian):
    // giving grad(epsilon) leastSquares where grad(k) keeps Gauss linear moves OpenFOAM's own epsilon
    // 6.8e-03 over 27,870 of 28,000 cells, and the mirror direction moves k 4.5e-02.
    cpu::EqnGradScheme kGrad;
    cpu::EqnGradScheme secondGrad;
    KEpsilonCoeffs coeffs;
    GeometricField<scalar> k;
    // kEpsilon's second scalar; empty under kOmegaSST
    GeometricField<scalar> epsilon;
    // kOmegaSST's; empty under kEpsilon
    GeometricField<scalar> omega;
    KOmegaSSTCoeffs sstCoeffs;
    // wallDist::New(mesh).y(), the CELL wall distance F1 and F2 take -- not the near-wall face
    // distance the wall functions use. Recomputed by moveInterTurbulence after every mesh motion, as
    // fvMesh::movePoints has wallDist::movePoints do; fvSchemes' `wallDist { updateInterval }` is kept
    // to refuse anything but 1 there.
    std::vector<scalar> yCell;
    label wallDistUpdateInterval = 1;
    // wallDist::movePoints's OWN LATCH (wallDist.C:193-221), and it is NOT a bare modulo. The interval
    // SETS `requireUpdate_`; the recompute then CLEARS it. So a step whose index the interval does not
    // divide keeps the stale distance, and the flag starts TRUE (the constructor's), which is why the
    // first move after start-up recomputes whatever the interval is. `<= 0` never sets it again, so y is
    // frozen at the start-up value for the whole run.
    bool wallDistRequireUpdate = true;
    // fvSchemes' `wallDist { correctWalls }`, default true (meshWavePatchDistMethod.C:59). False leaves
    // the wall-adjacent cells on the wave's face-CENTRE distance instead of the exact distance to the
    // face polygon (patchWave.C:203), which is what OpenFOAM does and what brae used to refuse.
    bool wallDistCorrectWalls = true;
    // THE PATCHES y IS MEASURED FROM. Empty: kOmegaSST's own wallDist, every `wall` patch. Non-empty: the
    // wallDist the motion solver's inverseDistance diffusivity registered first -- MeshObject::New finds
    // an object by its TYPE name alone (MeshObject.C), the motion solver is built before the turbulence
    // model (createDynamicFvMesh.H before createFields.H), and inverseDistanceDiffusivity::correct asks for
    // wallDist::New(mesh, meshWave, patchSet(<its patches>)). kOmegaSSTBase's wallDist::New(mesh) then
    // returns THAT object, and F1/F2 see the distance to the diffusivity's patches only. MEASURED on the
    // piston made kOmegaSST (tests/interfoam_moving_vs_openfoam.sh `pistonSST`): OpenFOAM's y is 0.01,
    // 0.03, 0.05 ... away from the paddle and 1.01 at x = 1, a cell whose bottom wall is 0.0025 below it.
    std::vector<label> wallDistPatchIDs;
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
    // BOTH SETS. fvMatrix::solve() selects `<field>Final` only when isFinalIteration()
    // (fvMatrix.C:1536-1542) and fvMatrix::relax() the same (:1249-1263), so correctors 1..N-1 of a step
    // use `solvers/k`. With `turbOnFinalIterOnly` at its default the closure only ever runs on the final
    // corrector and the non-Final SOLVER entries are never consulted -- which is why they are required
    // only when `!turbOnFinalIterOnly && nOuterCorrectors > 1` (solution::solverDict is FATAL when the
    // name is absent, solution.C:474-478). The non-Final RELAXATION entries are never required:
    // solution::relaxEquation returns false when neither the name nor `default` resolves and relax() is
    // then SKIPPED ENTIRELY, which is not the same as relaxing with 1.
    SmoothLinearSolve kSolve;
    SmoothLinearSolve epsSolve;
    SmoothLinearSolve omegaSolve;
    EquationRelax     kRelax;
    EquationRelax     epsRelax;
    EquationRelax     omegaRelax;
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
    // ...and WHICH delta coefficients: `uncorrected`/`limited 0` take nonOrthDeltaCoeffs with no
    // correction (uncorrectedSnGrad.H:113-119). Beside the flag it belongs to, not appended.
    bool laplacianNonOrth,
    scalar laplacianLimitCoeff,
    const std::vector<FvPatch>& patches,
    label nCells,
    // kOmegaSST's cell wall distance needs the mesh; null is a caller that can only run kEpsilon
    const PrimitiveMesh* mesh,
    const FvGeometry* geometry,
    // the patches of a wallDist the motion solver registered first (InterTurbulence::wallDistPatchIDs);
    // null or empty: kOmegaSST builds its own over the `wall` patches
    const std::vector<label>* sharedWallDistPatches,
    // PIMPLE's two facts the closure's reader needs: OpenFOAM looks the non-Final SOLVER entries up only
    // when the closure runs on a NON-final corrector, i.e. when `turbOnFinalIterOnly no` and there is more
    // than one. ONE place computes that rule so the three model branches cannot drift.
    label nOuterCorrectors,
    bool  turbOnFinalIterOnly);

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
    const std::vector<FvPatch>& patches,
    // kOmegaSST's validate forms grad(U) (kOmegaSSTBase.C:129-133), and a case caching it keeps that one
    GradUCache& gradUCache);

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
    // LOCALEULER: the per-cell rDeltaT setRDeltaT.H formed this step, which the closure's fvm::ddt takes in
    // the scalar's place (kOmegaSST::Compressible::rDeltaTCells). kOmegaSST only; the other closures
    // refuse it. Null == 1/deltaT.
    const std::vector<scalar>* rDeltaT = nullptr;
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
    // kOmegaSST::correct forms grad(U) (kOmegaSSTBase.C:522); a case caching it keeps that one. Null is
    // refused: every caller has the registry, caching or not.
    GradUCache* gradUCache = nullptr;
    // CrankNicolson: the scheme's clock and rho.oldTime().oldTime() (the `density variable` lineage
    // reads it; null in the other). The closure keeps its own ddt0 fields and old-old levels
    // (InterTurbulence::cn). Null runs the closure's fvm::ddt as Euler, which is what every other
    // scheme entry the reader admits is.
    const fv::CrankNicolsonClock* cn = nullptr;
    // THE STEP'S INDEX and WHETHER THIS IS THE FINAL OUTER CORRECTOR, both with unset sentinels: the
    // closure keys its old-time snapshot on the first and picks <field>Final against <field> by the
    // second (fvMatrix.C:1536-1542), and a default that stood in for either would be a silent
    // substitution of the case's own corrector. correctInterTurbulence throws when they are unset.
    label timeIndex = -1;
    int   finalIter = -1;
    const std::vector<scalar>* rhoOO = nullptr;
};

// turbulence->correct(), interFoam.C:171.
// The mesh has moved (fvMesh::movePoints): kOmegaSST's wall distance is an UpdateableMeshObject whose
// movePoints re-runs its method on the moved points (wallDist.C:193-221). Does nothing for the other
// closures, which build no wallDist.
void moveInterTurbulence(
    InterTurbulence&            t,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    // mesh_.time().timeIndex(), the index of the step being taken -- wallDist.C:198 tests it modulo the
    // interval. Threaded in rather than derived, because the closure has no clock of its own.
    label                       timeIndex);

// ...AND THE MESH'S TOPOLOGY CHANGED, which is not the same call. wallDist is an UpdateableMeshObject, so
// a change reaches wallDist::updateMesh (wallDist.C:224-234) and that FORCES its latch -- "Force update if
// performing topology change" -- before running movePoints' schedule. So this sets requireUpdate and then
// does the move path's work. The NEAR-WALL distance the wall functions use needs nothing on the host: it is
// a nearWallDist that turbulenceModel::correct() re-corrects whenever mesh_.changing()
// (turbulenceModel.C:94-100), and brae's host closures recompute it at every correct() unconditionally. The
// fields themselves (k, the second scalar, nut and their old times) are mapped by the AMR adapter, not here.
void updateMeshInterTurbulence(
    InterTurbulence&            t,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    label                       timeIndex);

// The registry names OpenFOAM gives the closure's two CrankNicolson ddt0 fields: the OPERANDS' names,
// so the model's and the density lineage's ("ddt0(rho,k)" / "ddt0(k)", "ddt0(rho,epsilon)" /
// "ddt0(epsilon)" / "ddt0(omega)", and nothing second under LES kEqn). ONE place, because a RESTART
// looks the fields up BY NAME in the start directory (inter_cn_restart.cuh) and the device arm must
// look up the same two -- three branches spelling their own names is how the two would drift.
void closureCnDdt0Names(
    const InterTurbulence& t,
    std::string& kName,
    std::string& secondName);

// The closure's OLD-TIME snapshots as a restart directory holds them: k_0 and epsilon_0/omega_0, read
// into the caller's vectors. Both or neither -- a directory with one is not a directory OpenFOAM wrote,
// and it throws saying so. False and nothing touched on a cold start. Shared by the two arms.
bool readClosureCnOldTime(
    const InterTurbulence& t,
    std::vector<scalar>& kOld,
    std::vector<scalar>& secondOld);

// ...the seed itself, once per run: the two ddt0 fields AND the old-time snapshot each field's own
// <field>_0 holds. Call it ABOVE advanceTurbulenceOldTime -- the first rotation moves the snapshot into
// the old-old level, so a seed after it arrives one step late.
void seedClosureCnRestart(
    InterTurbulence& t,
    const std::vector<FvPatch>& patches);

void correctInterTurbulence(
    InterTurbulence& t,
    const InterTurbulenceStepInput& in,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

} // namespace interFoam
} // namespace cpu
} // namespace brae
