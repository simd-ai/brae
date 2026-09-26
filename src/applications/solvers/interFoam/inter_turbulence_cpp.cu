// interFoam's turbulence, the host reference. See inter_turbulence_cpp.cuh for the two lineages.
#include "inter_turbulence_cpp.cuh"
#include <fstream>
#include "foam_field_reader.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "patch_wave_cpp.cuh"
#include "bound_cpp.cuh"
#include "cell_wall_dist.cuh"
#include "kEpsilon_cpp.cuh"
#include "kOmegaSST_cpp.cuh"
#include "near_wall_dist.cuh"
#include "nut_wall_function.cuh"
#include "patch_entry_lookup.cuh"
#include "scheme_parse.cuh"
#include "turbulence_setup.cuh"
#include <filesystem>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace interFoam {

EquationRelax EquationRelax::read(
    const FoamDict* equations,
    const std::string& name)
{
    EquationRelax r;
    if (!equations) return r;
    // FoamDict resolves a name the way OpenFOAM's dictionary does -- a literal key, else the last
    // regex key that matches -- so `".*" 1` answers for kFinal and `k 0.7` does not.
    if (equations->found(name))
    {
        r.on = true;
        r.factor = equations->scalarOr(name, scalar(1));
        return r;
    }
    if (equations->found("default"))
    {
        r.on = true;
        r.factor = equations->scalarOr("default", scalar(1));
    }
    return r;
}

namespace {

const char* const WHO = "brae interFoam: ";

GeometricField<scalar> readTurbulenceField(
    const std::string& startDir,
    const std::string& name,
    const std::vector<FvPatch>& patches,
    label nCells)
{
    const std::string path = startDir + "/" + name;
    if (!std::filesystem::exists(path))
        throw std::runtime_error(
            std::string(WHO) + "the case is RAS and " + path + " does not exist. OpenFOAM reads k, "
            "the model's second scalar and nut MUST_READ when the model is constructed and stops "
            "without them.");
    GeometricField<scalar> f = buildField<scalar>(readField<scalar>(path), patches, nCells);
    f.evaluateBoundary();
    return f;
}

// The solver entry fvMatrix::solve() selects on the final outer corrector.
// The solver entry for ONE name -- `k` or `kFinal`. `required` is false for the non-Final entries of a
// case whose closure only ever runs on the final corrector: OpenFOAM never looks them up there, so
// demanding them would refuse a case it runs.
SmoothLinearSolve readClosureSolve(
    const FoamDict& fvSolution,
    const std::string& entry,
    bool required,
    // the kEpsilon, kOmegaSST and LES kEqn closures all run PBiCG with DILU (pbicg.cuh) as well as
    // the smoothSolver; a caller that passes false still refuses it
    bool allowPBiCG = false)
{
    const std::string name = entry;
    const FoamDict* sv = fvSolution.subDict("solvers");
    const FoamDict* d = sv ? sv->subDict(name) : nullptr;
    if (!d && !required) return SmoothLinearSolve{};
    if (!d)
        throw std::runtime_error(
            std::string(WHO) + "fvSolution has no `solvers/" + name + "` entry. turbulence->correct() "
            "runs on the final outer corrector, where fvMatrix::solve() selects the Final entry, and "
            "OpenFOAM stops without it.");
    SmoothLinearSolve s = SmoothLinearSolve::read(*d);
    if (!s.gaussSeidel() && !(allowPBiCG && s.pbicgDILU()))
        throw std::runtime_error(
            std::string(WHO) + "`solvers/" + name + "` names `solver " + s.solver + "; smoother "
            + s.smoother + "; preconditioner " + s.preconditioner + ";`. brae's interFoam closure runs "
            "OpenFOAM's smoothSolver with the GaussSeidel or symGaussSeidel smoother -- what every "
            "turbulent tutorial names -- and, under kEpsilon, kOmegaSST and the LES kEqn, PBiCG with "
            "DILU; nothing else: a substituted solver at the same tolerance stops somewhere else.");
    return s;
}

// ...and the Final entry, which is always required: the closure runs on the final corrector whatever
// `turbOnFinalIterOnly` says.
SmoothLinearSolve readFinalSolve(
    const FoamDict& fvSolution,
    const std::string& field,
    bool allowPBiCG = false)
{
    return readClosureSolve(fvSolution, field + "Final", /*required=*/true, allowPBiCG);
}

// div(<flux>,<field>) for a RAS closure: `Gauss upwind` or `Gauss limitedLinear <k>`. Both arms of
// both closures carry the second and no interFoam tutorial names it, so it is gated by a staged
// profile (tests/interfoam_waterchannel_vs_openfoam.sh `limitedLinear`). `bounded` and `linearUpwind`
// stay refused: nothing holds either here. Both equations must name the SAME scheme -- the closures
// carry one flag for the pair, as `bounded` does.
//
// THE PROFILE GATES FIELDS AND NOT MATRIX COEFFICIENTS, and the reason is in the scheme. Where the
// two cells of a face hold the same value, `gradf = phiN - phiP` is EXACTLY zero, OpenFOAM takes
// NVDTVD's `mag(gradcf) >= 1000*mag(gradf)` branch, and `r = 2*1000*sign(gradcf)*sign(gradf) - 1`
// turns entirely on the SIGN of `gradcf` -- which is 1e-18 there, eighteen orders below the field's
// own scale. MEASURED on RAS/waterChannel: brae's omega and grad(omega) are OpenFOAM's to 7.3e-15
// and 5.7e-15, and the limiter still lands on the other side on 2 of 79,800 faces, each worth an
// O(1) coefficient. OpenFOAM's own answer there is arbitrary at the bit level. The FIELDS are not:
// host omega 7.2e-12, k 5.9e-11, U 4.2e-12.
void readClosureDivScheme(
    const std::string&  caseDir,
    const std::string&  field,
    const std::string&  fluxName,
    InterTurbulence&    t,
    bool                first)
{
    const FieldDivScheme fs = parseFieldDivScheme(caseDir, field, false, fluxName);
    if (fs.bounded || fs.linearUpwind)
        throw std::runtime_error(
            std::string(WHO) + "fvSchemes `div(" + fluxName + "," + field + ")` is neither `Gauss "
            "upwind` nor `Gauss limitedLinear <k>`, the two the closure carries.");
    // ONE PER EQUATION. `fvm::div(phi, psi)` resolves `div(<flux>,<psi>)` by the FIELD's name, so
    // `div(phi,k) Gauss upwind` beside `div(phi,epsilon) Gauss limitedLinear 1` is two different
    // matrices in OpenFOAM. This used to set one pair and refuse a mismatch.
    cpu::EqnDivScheme& d = first ? t.kDiv : t.secondDiv;
    d.limitedLinear = fs.limited;
    d.limiterCoeff  = fs.limited ? fs.coeff : scalar(1);
}

// grad(U), which the production GbyNu takes (kEpsilon.C:237, kOmegaSSTBase.C:520)
template <class Coeffs>
void readGradU(
    const std::string& caseDir,
    Coeffs& co)
{
    const FieldGradScheme gu = parseFieldGradScheme(caseDir, "U");
    if (!gu.unsupportedLimiter.empty() || !(gu.gaussLinear || gu.leastSquares))
        throw std::runtime_error(
            std::string(WHO) + "fvSchemes grad(U) resolves to `" + gu.raw + "`, which the closure's "
            "production does not compute (Gauss linear and leastSquares, optionally cellLimited).");
    co.gradULeastSq = gu.leastSquares;
    co.gradULimitK = gu.cellLimitK;
}

// grad(k) and grad(epsilon|omega), which the corrected laplacian's deferred correction takes -- and
// under kOmegaSST CDkOmega too (kOmegaSSTBase.C:548). The closure carries ONE setting for both fields.
template <class Coeffs>
void readGradK(
    const std::string& caseDir,
    const std::string& second,
    Coeffs& co,
    // K's too, filled here rather than re-derived: the DEVICE closure compares the two to decide whether
    // it can run, and a comparison against a default-constructed struct is not a comparison.
    cpu::EqnGradScheme& outK,
    cpu::EqnGradScheme& outSecond)
{
    const FieldGradScheme gk = parseFieldGradScheme(caseDir, "k");
    const FieldGradScheme ge = parseFieldGradScheme(caseDir, second);
    for (const FieldGradScheme* q : {&gk, &ge})
    {
        if (!q->unsupportedLimiter.empty() || !(q->gaussLinear || q->leastSquares))
            throw std::runtime_error(
                std::string(WHO) + "fvSchemes resolves a turbulence gradient to `" + q->raw
                + "`, which the closure does not compute.");
    }
    // ONE PER EQUATION, and `co` keeps K's -- the closures take K positionally, as they always have, and
    // the second field's through EqnGradScheme. This used to refuse a mismatch.
    co.gradKLeastSq = gk.leastSquares;
    co.gradKLimitK = gk.cellLimitK;
    outK.leastSquares      = gk.leastSquares;
    outK.cellLimitK        = gk.cellLimitK;
    outSecond.leastSquares = ge.leastSquares;
    outSecond.cellLimitK   = ge.cellLimitK;
}

// Which nut wall function each patch carries, read where the dictionary TYPE still exists, and held
// against epsilon's: the closure applies the nut wall function on exactly the patches whose epsilon
// is an epsilonWallFunction, so the two must name the same set.
std::vector<int> readNutWallKinds(
    const std::string& startDir,
    const GeometricField<scalar>& epsilon,
    const std::vector<FvPatch>& patches)
{
    std::vector<int> kind(patches.size(), -1);
    const FieldData<scalar> nutRaw = readField<scalar>(startDir + "/nut");
    for (const auto& b : nutRaw.boundary)
    {
        if (b.type.rfind("nut", 0) != 0 && b.type.rfind("atmNut", 0) != 0) continue;
        int k = -1;
        if (b.type == "nutkWallFunction")
        {
            k = static_cast<int>(NutWall::Nutk);
        }
        if (b.type == "nutUWallFunction")
        {
            k = static_cast<int>(NutWall::NutU);
        }
        if (b.type == "nutLowReWallFunction")
        {
            k = static_cast<int>(NutWall::LowRe);
        }
        if (k < 0)
            throw std::runtime_error(
                std::string(WHO) + "nut patch `" + b.name + "` carries `" + b.type + "`. The kEpsilon "
                "closure has nutkWallFunction, nutUWallFunction and nutLowReWallFunction; the rest of "
                "the family are different functions of different inputs and are not substituted.");
        for (const FvPatch* pp : patchesResolvingTo(nutRaw.boundary, b, patches))
        {
            // nutWallFunctionFvPatchScalarField::checkType (nutWallFunctionFvPatchScalarField.C:45-55):
            // "Invalid wall function specification ... must be wall", a FatalError at construction
            if (pp->type != "wall")
                throw std::runtime_error(
                    std::string(WHO) + "nut patch `" + pp->name + "` carries `" + b.type
                    + "` and its patch type is `" + pp->type + "`. A nut wall function's patch must be "
                    "a `wall`; OpenFOAM stops on this at construction (nutWallFunction checkType).");
            kind[static_cast<std::size_t>(pp - patches.data())] = k;
        }
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const bool epsWall = epsilon.boundary[pi]->isTurbulenceWallFunction();
        if (epsWall == (kind[pi] >= 0)) continue;
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patches[pi].name + "` carries "
            + (epsWall ? "an epsilonWallFunction but no nut wall function"
                       : "a nut wall function but no epsilonWallFunction")
            + ". The closure evaluates the two on the same patches; a case that splits them would get "
              "a wall viscosity OpenFOAM does not compute.");
    }
    return kind;
}

// kOmegaSST's walls. kOmegaSST_cpp applies omegaWallFunction (:466-520) and nutkWallFunction
// (correctNutField) on every patch whose MESH type is `wall`, with the model's kappa, E and CmuWall and
// OpenFOAM's default binomial n = 2 blender. The case is held to exactly that, patch by patch, here --
// the last place the dictionary types exist.
void requireSstWalls(
    const std::string& startDir,
    const KOmegaSSTCoeffs& co,
    const std::vector<FvPatch>& patches)
{
    const FieldData<scalar> nutRaw = readField<scalar>(startDir + "/nut");
    const FieldData<scalar> omegaRaw = readField<scalar>(startDir + "/omega");
    bool anyWall = false;
    for (const FvPatch& p : patches)
    {
        const PatchFieldData<scalar>* nb = findPatchEntry(nutRaw.boundary, p);
        const PatchFieldData<scalar>* ob = findPatchEntry(omegaRaw.boundary, p);
        const std::string nutType = nb ? nb->type : std::string();
        const std::string omegaType = ob ? ob->type : std::string();
        const bool wall = (p.type == "wall");
        anyWall = anyWall || wall;
        const bool nutIsWallFn = nutType.rfind("nut", 0) == 0 || nutType.rfind("atmNut", 0) == 0;
        if (nutIsWallFn && nutType != "nutkWallFunction")
            throw std::runtime_error(
                std::string(WHO) + "nut patch `" + p.name + "` carries `" + nutType + "`. The "
                "kOmegaSST closure has nutkWallFunction; the rest of the family are different "
                "functions of different inputs and are not substituted.");
        if (nutIsWallFn && !wall)
            throw std::runtime_error(
                std::string(WHO) + "nut patch `" + p.name + "` carries `" + nutType + "` and its patch "
                "type is `" + p.type + "`. A nut wall function's patch must be a `wall`; OpenFOAM stops "
                "on this at construction (nutWallFunction checkType).");
        if (wall && (nutType != "nutkWallFunction" || omegaType != "omegaWallFunction"))
            throw std::runtime_error(
                std::string(WHO) + "wall patch `" + p.name + "` carries nut `" + nutType + "` and omega `"
                + omegaType + "`. The kOmegaSST closure evaluates nutkWallFunction and "
                "omegaWallFunction on every `wall` patch; a wall without them would get values "
                "OpenFOAM does not compute there.");
        if (!wall && omegaType == "omegaWallFunction")
            throw std::runtime_error(
                std::string(WHO) + "omega patch `" + p.name + "` carries omegaWallFunction and its "
                "patch type is `" + p.type + "`; the closure applies it on `wall` patches only.");
        if (!wall) continue;
        for (const PatchFieldData<scalar>* b : {nb, ob})
        {
            // A PATCH'S OWN Cmu/kappa/E/beta1 ARE HONOURED NOW. OpenFOAM constructs
            // `wallFunctionCoefficients` from the PATCH dictionary in every wall function
            // (wallFunctionCoefficients.C:68-80), deriving yPlusLam from that patch's kappa and E, and
            // omegaWallFunction reads its own beta1 there (omegaWallFunctionFvPatchScalarField.C:408).
            // None of them is the model's. This refusal was OVER-STRICT: it stopped on a case OpenFOAM
            // runs, while the reader already parsed the values, the patch fields already carried them
            // and the kEpsilon closure already used them per patch.
            // THE TWO WALL FUNCTIONS HAVE DIFFERENT DEFAULTS, and this loop used to hold both to
            // omega's. nutkWallFunction is STEPWISE with n = 4
            // (nutkWallFunctionFvPatchScalarField.C:216); omegaWallFunction is BINOMIAL with n = 2
            // (omegaWallFunctionFvPatchScalarField.C:405). The tutorial files omit `blending`, so the
            // defaults applied and nothing showed -- but OpenFOAM WRITES the resolved word, so a restart
            // from its own output carries `blending stepwise` on nut, and checking that against omega's
            // rule made brae REFUSE A CASE OPENFOAM RUNS (found on a restart of
            // RAS/electrostaticDeposition at t = 0.001). brae computes the stepwise form for nut
            // (nut_wall_function.cuh:4, a hard switch at yPlusLam) and the binomial one for omega, so
            // each default IS what it implements.
            // `n` is STEPWISE's unused parameter -- OpenFOAM passes 4 and the switch never reads it --
            // so it is not checked on the nut entry.
            const bool isNut = (b == nb);
            const bool blendOther = !b->wfBlending.empty()
                                 && (isNut
                                     ? (b->wfBlending != "stepwise")
                                     : (b->wfBlending != "binomial"
                                        || (b->hasWfBlendN && b->wfBlendN != 2)));
            if (blendOther)
                throw std::runtime_error(
                    std::string(WHO) + (isNut ? "nut" : "omega") + " patch `" + p.name
                    + "` names `blending " + b->wfBlending + "`. brae blends "
                    + (isNut ? "nut's viscous and log values STEPWISE at yPlusLam, OpenFOAM's own "
                               "nutkWallFunction default (nutkWallFunctionFvPatchScalarField.C:216)"
                             : "omega's viscous and log values BINOMIALLY with n = 2, OpenFOAM's own "
                               "omegaWallFunction default (omegaWallFunctionFvPatchScalarField.C:405)")
                    + ", and nothing else.");
        }
    }
    if (!anyWall)
        throw std::runtime_error(
            std::string(WHO) + "the case is kOmegaSST and the mesh has no `wall` patch. F1 and F2 take "
            "the distance to the nearest wall, which does not exist here.");
}

} // namespace


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
    const PrimitiveMesh* mesh,
    const FvGeometry* geometry,
    const std::vector<label>* sharedWallDistPatches,
    // PIMPLE's two facts the closure's reader needs: OpenFOAM looks the non-Final SOLVER entries up only
    // when the closure runs on a NON-final corrector, i.e. when `turbOnFinalIterOnly no` and there is more
    // than one. ONE place computes that rule so the three model branches cannot drift.
    label nOuterCorrectors,
    bool  turbOnFinalIterOnly)
{
    InterTurbulence t;
    // See the parameters: OpenFOAM looks the non-Final solver entries up only on a NON-final corrector,
    // and solution::solverDict is FATAL when the name is absent (solution.C:474-478) -- so they are
    // required exactly there and nowhere else. ONE place computes the rule.
    const bool nonFinalRequired = (!turbOnFinalIterOnly && nOuterCorrectors > 1);
    const std::string path = caseDir + "/constant/momentumTransport";
    const std::string alt = caseDir + "/constant/turbulenceProperties";
    const std::string p = std::filesystem::exists(path) ? path
                        : (std::filesystem::exists(alt) ? alt : std::string());
    // no dictionary at all -> laminar
    if (p.empty()) return t;
    const std::string file = "constant/" + std::filesystem::path(p).filename().string();
    const FoamDict d = readDict(p);
    const std::string sim = d.wordOr("simulationType", "laminar");
    if (sim == "laminar") return t;
    if (sim != "RAS" && sim != "LES")
        throw std::runtime_error(
            std::string(WHO) + file + " asks for simulationType `" + sim + "`. brae's interFoam has "
            "laminar, RAS (kEpsilon, kOmegaSST) and LES (kEqn).");

    // incompressibleInterPhaseTransportModel.C:61-84 -- `variable`, `uniform`, or a FatalError
    if (d.found("density"))
    {
        const std::string method = d.wordOr("density", "");
        if (method != "variable" && method != "uniform")
            throw std::runtime_error(
                std::string(WHO) + file + " has `density " + method + "`. OpenFOAM accepts `variable` "
                "and `uniform` and stops on anything else (incompressibleInterPhaseTransportModel.C:"
                "75-81).");
        t.variableDensity = (method == "variable");
    }

    const FoamDict* rfAll = fvSolution.subDict("relaxationFactors");
    const FoamDict* eqAll = rfAll ? rfAll->subDict("equations") : nullptr;

    if (sim == "LES")
    {
        if (t.variableDensity)
            throw std::runtime_error(
                std::string(WHO) + file + " pairs `density variable` with LES. That is kEqn.C with rho the "
                "mixture density and rhoPhi its flux; no shipped tutorial pairs them. Refused rather than "
                "run ungated.");
        if (!mesh || !geometry)
            throw std::runtime_error(std::string(WHO) + "LES needs the mesh for its filter width.");
        const FoamDict* les = d.subDict("LES");
        const std::string lesModel = les ? les->wordOr("LESModel", "") : "";
        if (lesModel != "kEqn")
            throw std::runtime_error(
                std::string(WHO) + file + " asks for LESModel `" + lesModel + "`. kEqn is the LES model "
                "wired into interFoam (LES/nozzleFlow2D); Smagorinsky, WALE, dynamicKEqn and the rest "
                "are different models and are not substituted.");
        // LESModel.C:70 reads it as a Switch defaulting true. LESModel::correct() then calls
        // delta_().correct() BEFORE kEqn's gate (LESModel.C:251, kEqn.C:141), so a frozen LES model on a
        // MOVING mesh still recomputes its filter width every step -- brae recomputes it in
        // moveInterTurbulence, the same steps, and nothing reads it again once correct() is gated out.
        t.frozen = !les->switchOr("turbulence", true);
        if (!eulerDdt)
            throw std::runtime_error(
                std::string(WHO) + "kEqn takes fvm::ddt(k) through ddtSchemes (kEqn.C:162) and the "
                "closure carries Euler only.");
        t.model = InterRasModel::KEqnLES;
        // kEqn.C:88-96: Ck from coeffDict_ = optionalSubDict("kEqnCoeffs"); LESModel.C:81-100: Ce and
        // kMin from the LES dictionary ITSELF, not from the coefficients
        t.lesCoeffs.Ck = les->optionalSubDict("kEqnCoeffs")->scalarOr("Ck", t.lesCoeffs.Ck);
        t.lesCoeffs.Ce = les->scalarOr("Ce", t.lesCoeffs.Ce);
        t.lesCoeffs.kMin = les->scalarOr("kMin", t.lesCoeffs.kMin);
        t.lesCoeffs.correctedLaplacian = laplacianCorrected;
        t.lesCoeffs.nonOrthCoeffs = laplacianNonOrth;
        t.lesCoeffs.snGradLimitCoeff = laplacianLimitCoeff;
        t.deltaSpec = LESdelta::read(*les, file);
        {
            const FieldGradScheme gu = parseFieldGradScheme(caseDir, "U");
            const FieldGradScheme gk = parseFieldGradScheme(caseDir, "k");
            for (const FieldGradScheme* q : {&gu, &gk})
            {
                if (!q->gaussLinear || !q->unsupportedLimiter.empty() || q->cellLimitK > 0 || q->leastSquares)
                    throw std::runtime_error(
                        std::string(WHO) + "fvSchemes resolves a gradient kEqn takes to `" + q->raw
                        + "`; the LES closure computes plain `Gauss linear` only.");
            }
        }
        const FieldDivScheme fk = parseFieldDivScheme(caseDir, "k", false, "phi");
        if (fk.bounded || fk.linearUpwind)
            throw std::runtime_error(
                std::string(WHO) + "fvSchemes `div(phi,k)` is neither `Gauss upwind` nor `Gauss "
                "limitedLinear <k>`, the two the LES closure carries.");
        t.lesCoeffs.limitedLinear = fk.limited;
        t.lesCoeffs.limitedLinearCoeff = fk.coeff;

        t.k = readTurbulenceField(startDir, "k", patches, nCells);
        t.nut = readTurbulenceField(startDir, "nut", patches, nCells);
        {
            // nothing on a kEqn case is a wall function: nut is Ck*sqrt(k)*delta everywhere, and its
            // patches evaluate as their own types
            const FieldData<scalar> nutRaw = readField<scalar>(startDir + "/nut");
            for (const auto& b : nutRaw.boundary)
            {
                if (b.type.rfind("nut", 0) == 0 || b.type.rfind("atmNut", 0) == 0)
                    throw std::runtime_error(
                        std::string(WHO) + "nut patch `" + b.name + "` carries `" + b.type + "` under LES "
                        "kEqn. The wall functions are not wired into the LES closure.");
            }
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (t.k.boundary[pi]->fluxName() != "phi")
                throw std::runtime_error(
                    std::string(WHO) + "patch `" + patches[pi].name + "` of k names the flux `"
                    + t.k.boundary[pi]->fluxName() + "`; the LES closure hands its patches phi only.");
        }
        t.delta = LESdelta::compute(t.deltaSpec, *mesh, *geometry, patches);
        // kEqn's constructor: bound(k_, kMin_)
        bound(t.k, t.lesCoeffs.kMin, *mesh, *geometry, patches);
        t.kSolveFinal = readFinalSolve(fvSolution, "k", /*allowPBiCG=*/true);
        t.kRelaxFinal = EquationRelax::read(eqAll, "kFinal");
        t.kSolve = readClosureSolve(fvSolution, "k", nonFinalRequired, /*allowPBiCG=*/true);
        t.kRelax = EquationRelax::read(eqAll, "k");
        t.on = true;
        return t;
    }

    const FoamDict* ras = d.subDict("RAS");
    const std::string model = ras ? ras->wordOr("RASModel", "") : "";
    if (model != "kEpsilon" && model != "kOmegaSST")
        throw std::runtime_error(
            std::string(WHO) + file + " asks for RASModel `" + model + "`. kEpsilon and kOmegaSST are "
            "the models wired into interFoam (16 of the 17 turbulent tutorials name one of them); "
            "refusing rather than running it under another model's name.");
    t.model = (model == "kOmegaSST") ? InterRasModel::KOmegaSST : InterRasModel::KEpsilon;
    // RASModel.C:70, a Switch defaulting true. The open question the old refusal named is settled:
    // validate() is NOT gated on it (eddyViscosity.C:119-122), so the uniform lineage rebuilds nut from
    // the BOUNDED file fields and freezes there -- see InterTurbulence::frozen.
    t.frozen = !ras->switchOr("turbulence", true);
    // kEpsilon carries CrankNicolson too (InterTurbulenceStepInput::cn); kOmegaSST does not, and the
    // case reader has already said so by name where the two meet
    // EVERY CLOSURE TAKES CrankNicolson NOW -- kEpsilon always did; kOmegaSST's two equations and
    // kEqn's one take fvm::ddt through ddtSchemes (kOmegaSSTBase.C:572, :602; kEqn.C:172) and each
    // now carries the scheme's own term with the Euler line left inert. Gated on
    // validation/interFoamCyclic (`sstCN`, `lesCN`), both arms, against real OpenFOAM.
    t.coeffs.correctedLaplacian = laplacianCorrected;
    t.coeffs.nonOrthCoeffs = laplacianNonOrth;
    t.coeffs.snGradLimitCoeff = laplacianLimitCoeff;

    if (t.model == InterRasModel::KOmegaSST)
    {
        if (t.variableDensity)
            throw std::runtime_error(
                std::string(WHO) + file + " pairs `density variable` with kOmegaSST. That is "
                "kOmegaSSTBase.C with rho the mixture density and rhoPhi its flux; no shipped tutorial "
                "pairs them, so no gate would hold it against OpenFOAM. Refused rather than run "
                "ungated.");
        if (!mesh || !geometry)
            throw std::runtime_error(
                std::string(WHO) + "kOmegaSST needs the mesh for its wall distance and this caller "
                "handed readInterTurbulence none.");
        readKOmegaSSTCoeffs(ras, t.sstCoeffs);
        scalar epsilonMinUnused = 0;
        readTurbulenceMinima(ras, t.sstCoeffs.kMin, epsilonMinUnused, t.sstCoeffs.omegaMin);
        // kOmegaSSTBase.C:408-461. With it the two equations gain beta*sqr(omegaInf) and
        // betaStar*omegaInf*kInf and the closure here carries neither term.
        const FoamDict* sstDict = ras->optionalSubDict("kOmegaSSTCoeffs");
        // A REFUSAL A SPELLING COULD WALK PAST: this tested {yes,on,true} and let `decayControl 1;`
        // (and `any`, `t`, `y`) through, so the run silently omitted both decay terms.
        const bool decay = (sstDict ? sstDict : ras)->switchOr("decayControl", false);
        if (decay)
            throw std::runtime_error(
                std::string(WHO) + file + " sets `decayControl "
                + (sstDict ? sstDict : ras)->wordOr("decayControl", "yes") + "`. It adds "
                "beta*sqr(omegaInf) to the omega equation and betaStar*omegaInf*kInf to k's "
                "(kOmegaSSTBase.C:574, :594), and kOmegaSST_cpp carries neither.");
        if (t.sstCoeffs.F3)
            throw std::runtime_error(
                std::string(WHO) + file + " sets `F3 yes`. kOmegaSSTBase multiplies F23 by F3, which "
                "changes the eddy-viscosity limiter and the production limiter; not ported.");
        readGradU(caseDir, t.sstCoeffs);
        readGradK(caseDir, "omega", t.sstCoeffs, t.kGrad, t.secondGrad);
        readClosureDivScheme(caseDir, "k", "phi", t, /*first=*/true);
        readClosureDivScheme(caseDir, "omega", "phi", t, /*first=*/false);

        t.k = readTurbulenceField(startDir, "k", patches, nCells);
        t.omega = readTurbulenceField(startDir, "omega", patches, nCells);
        t.nut = readTurbulenceField(startDir, "nut", patches, nCells);
        // kOmegaSSTBase.C:438-439, THE CONSTRUCTOR's bound -- k first, omega second. It runs whether or
        // not `turbulence` is on, and validate() then builds the first momentum equation's nut from the
        // BOUNDED fields. interFoam's reader had it in the LES branch alone; the same hole in the
        // single-phase drivers is gated by tests/bound_at_construction_vs_openfoam.sh.
        bound(t.k, t.sstCoeffs.kMin, *mesh, *geometry, patches, "k");
        bound(t.omega, t.sstCoeffs.omegaMin, *mesh, *geometry, patches, "omega");
        requireSstWalls(startDir, t.sstCoeffs, patches);
        t.nutWallKind.assign(patches.size(), -1);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].type == "wall")
            {
                t.nutWallKind[pi] = static_cast<int>(NutWall::Nutk);
            }
            for (const std::string* name : {&t.k.boundary[pi]->fluxName(), &t.omega.boundary[pi]->fluxName()})
            {
                if (*name == "phi") continue;
                throw std::runtime_error(
                    std::string(WHO) + "patch `" + patches[pi].name + "` of k or omega names the flux `"
                    + *name + "` in its `phi` entry; the turbulence closure hands its patches phi only.");
            }
        }
        if (sharedWallDistPatches && !sharedWallDistPatches->empty())
        {
            // the motion solver's wallDist (InterTurbulence::wallDistPatchIDs): meshWave with correctWalls
            // over its patches, read from fvSchemes' `patchDist`, which the motion solver has checked
            t.wallDistPatchIDs = *sharedWallDistPatches;
            t.yCell = patchWave(*mesh, *geometry, patches, t.wallDistPatchIDs, true).distance;
        }
        else
        {
            // kOmegaSST's own: wallDist(mesh) with no default method, so fvSchemes' wallDist dictionary
            // MUST name one (patchDistMethod.C:64-76 reads `method` MUST_READ when the default is empty);
            // meshWave's correctWalls defaults true; updateInterval (default 1) matters on a moving mesh
            const FoamDict schemes = readDict(caseDir + "/system/fvSchemes");
            const FoamDict* wd = schemes.subDict("wallDist");
            const std::string method = wd ? wd->wordOr("method", "") : std::string();
            if (method.empty())
                throw std::runtime_error(
                    std::string(WHO) + "kOmegaSST needs fvSchemes' `wallDist { method ...; }`; OpenFOAM reads it "
                    "with no default and stops without it.");
            if (method != "meshWave")
                throw std::runtime_error(
                    std::string(WHO) + "fvSchemes names `wallDist { method " + method + "; }`. kOmegaSST's "
                    "wall distance is ported as meshWave (patchDistMethods/meshWave) only.");
            // ...and the same again: {false,no,off} let `correctWalls 0;` through, and brae then ran
            // its always-correcting meshWave against OpenFOAM's uncorrected one with no message.
            // meshWavePatchDistMethod.C:59, default true. `false` skips patchWave's wall-cell override
            // (patchWave.C:203), so those cells keep the wave's face-CENTRE distance. Refused until now.
            t.wallDistCorrectWalls = wd->switchOr("correctWalls", true);
            // wallDist.C:127 reads it as a LABEL (`getOrDefault<label>`), so `2.5` is an IO error in
            // OpenFOAM rather than a truncation to 2. scalarOr would have silently truncated it.
            t.wallDistUpdateInterval = static_cast<label>(wd->intOr("updateInterval", 1));
            {
                const std::string raw = wd->wordOr("updateInterval", "1");
                if (raw.find('.') != std::string::npos || raw.find('e') != std::string::npos
                                                       || raw.find('E') != std::string::npos)
                    throw std::runtime_error(
                        std::string(WHO) + "fvSchemes sets `wallDist { updateInterval " + raw
                        + "; }`. OpenFOAM reads it as a label (wallDist.C:127) and stops on a "
                          "non-integer; brae will not truncate it.");
            }
            t.yCell = cellWallDist(*mesh, *geometry, patches, nullptr, nullptr,
                                   t.wallDistCorrectWalls);
        }

        // ...and PBiCG for k and omega too: the SST closure runs it now, as kEpsilon's has since
        // waves/mangroveInteraction (kOmegaSST_cpp.cu, solveScalar)
        t.kSolveFinal = readFinalSolve(fvSolution, "k", /*allowPBiCG=*/true);
        t.omegaSolveFinal = readFinalSolve(fvSolution, "omega", /*allowPBiCG=*/true);
        t.kRelaxFinal = EquationRelax::read(eqAll, "kFinal");
        t.omegaRelaxFinal = EquationRelax::read(eqAll, "omegaFinal");
        t.kSolve = readClosureSolve(fvSolution, "k", nonFinalRequired, /*allowPBiCG=*/true);
        t.omegaSolve = readClosureSolve(fvSolution, "omega", nonFinalRequired, /*allowPBiCG=*/true);
        t.kRelax = EquationRelax::read(eqAll, "k");
        t.omegaRelax = EquationRelax::read(eqAll, "omega");
        t.on = true;
        return t;
    }

    // all six, as kEpsilon.C:199-204 reads them; optionalSubDict as RASModel.C:72
    if (const FoamDict* kec = ras->optionalSubDict("kEpsilonCoeffs"))
    {
        t.coeffs.Cmu = kec->scalarOr("Cmu", t.coeffs.Cmu);
        t.coeffs.C1 = kec->scalarOr("C1", t.coeffs.C1);
        t.coeffs.C2 = kec->scalarOr("C2", t.coeffs.C2);
        t.coeffs.C3 = kec->scalarOr("C3", t.coeffs.C3);
        t.coeffs.sigmaK = kec->scalarOr("sigmak", t.coeffs.sigmaK);
        t.coeffs.sigmaEps = kec->scalarOr("sigmaEps", t.coeffs.sigmaEps);
    }
    scalar omegaMinUnused = 0;
    readTurbulenceMinima(ras, t.coeffs.kMin, t.coeffs.epsilonMin, omegaMinUnused);
    readGradU(caseDir, t.coeffs);
    readGradK(caseDir, "epsilon", t.coeffs, t.kGrad, t.secondGrad);

    // THE KEY CARRIES THE FLUX'S NAME: div(rhoPhi,k) in the variable lineage, div(phi,k) in the other
    const std::string flux = t.variableDensity ? "rhoPhi" : "phi";
    readClosureDivScheme(caseDir, "k", flux, t, /*first=*/true);
    readClosureDivScheme(caseDir, "epsilon", flux, t, /*first=*/false);

    t.k = readTurbulenceField(startDir, "k", patches, nCells);
    t.epsilon = readTurbulenceField(startDir, "epsilon", patches, nCells);
    t.nut = readTurbulenceField(startDir, "nut", patches, nCells);
    // kEpsilon.C:182-183, the constructor's bound: k first, epsilon second. See the SST branch.
    // A REFUSAL RATHER THAN A GUARD: `if (mesh && geometry)` here would silently skip the bound for a
    // caller that passed neither, and frozenFloored is the only arm that could ever notice. The SST
    // branch already throws on the same condition (for its wall distance), so this matches it.
    if (!mesh || !geometry)
        throw std::runtime_error(
            std::string(WHO) + "kEpsilon needs the mesh for the constructor's bound(k)/bound(epsilon) "
            "and this caller handed readInterTurbulence none.");
    bound(t.k, t.coeffs.kMin, *mesh, *geometry, patches, "k");
    bound(t.epsilon, t.coeffs.epsilonMin, *mesh, *geometry, patches, "epsilon");
    t.nutWallKind = readNutWallKinds(startDir, t.epsilon, patches);
    // epsilonWallFunction's `lowReCorrection`, off the epsilon BC that names it
    // (epsilonWallFunctionFvPatchScalarField.C:373, :414). On a face with y+ < yPlusLam it switches
    // epsilon to the RESOLVED form 2*k*nu/y^2 and contributes NO wall production (:242, :338) -- a
    // different branch, not a scaling.
    //
    // NOTHING IN interFoam READ IT. The reader has parsed it into PatchFieldData::epsLowRe all along and
    // the single-phase drivers thread it (turbulence_setup.cuh:784-791), but no interFoam site did, so
    // `t.coeffs.epsLowRe` stayed false and kEpsilon_cpp.cu's low-Re branch could never fire: a case
    // asking for it got the high-Re log-law epsilon with no notice. The entry sits inside a
    // boundaryField patch dictionary, which the defaults audit does not track per key, which is why it
    // went unseen.
    {
        const FieldData<scalar> epsRaw = readField<scalar>(startDir + "/epsilon");
        for (const auto& pb : epsRaw.boundary)
        {
            if (pb.type == "epsilonWallFunction" && pb.epsLowRe)
            {
                t.coeffs.epsLowRe = true;
            }
        }
        // PER PATCH ON BOTH ARMS NOW, so a case that sets the switch on some walls and not others runs
        // two different branches as OpenFOAM does. `t.coeffs.epsLowRe` above stays as "any wall has it"
        // and is the FALLBACK only: the host closure reads the patch's own
        // WallFunctionCoeffs::lowRe (kEpsilon_cpp.cu) and the device kernel reads DeviceWallData::wfLowRe
        // per wall face, each falling back to this flag when a driver has not filled them.
    }
    // the closure tells k's and epsilon's flux-conditional patches the volumetric phi and nothing else
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (const std::string* name : {&t.k.boundary[pi]->fluxName(), &t.epsilon.boundary[pi]->fluxName()})
        {
            if (*name == "phi") continue;
            throw std::runtime_error(
                std::string(WHO) + "patch `" + patches[pi].name + "` of k or epsilon names the flux `"
                + *name + "` in its `phi` entry; the turbulence closure hands its patches phi only.");
        }
    }

    t.kSolveFinal = readFinalSolve(fvSolution, "k", /*allowPBiCG=*/true);
    t.epsSolveFinal = readFinalSolve(fvSolution, "epsilon", /*allowPBiCG=*/true);
    t.kRelaxFinal = EquationRelax::read(eqAll, "kFinal");
    t.epsRelaxFinal = EquationRelax::read(eqAll, "epsilonFinal");
    t.kSolve = readClosureSolve(fvSolution, "k", nonFinalRequired, /*allowPBiCG=*/true);
    t.epsSolve = readClosureSolve(fvSolution, "epsilon", nonFinalRequired, /*allowPBiCG=*/true);
    t.kRelax = EquationRelax::read(eqAll, "k");
    t.epsRelax = EquationRelax::read(eqAll, "epsilon");

    t.on = true;
    return t;
}


// ONE BUILDER for the second equation's solver setting, so the SST and kEpsilon branches cannot drift.
// The closure takes k's positionally and this for epsilon/omega, because fvMatrix::solve() looks the
// dictionary up by FIELD name and `kFinal` and `epsilonFinal` need not agree.
namespace {
// WHICH DICTIONARY THIS CORRECTOR USES. fvMatrix::solve() and fvMatrix::relax() both select
// `<field>Final` when isFinalIteration() and `<field>` otherwise (fvMatrix.C:1536-1542, :1249-1263), so
// the pick is the same for the solver and the relaxation, and it is made in ONE place for all three
// closures rather than three times over.
struct ClosurePick
{
    const SmoothLinearSolve* first  = nullptr;   // k's
    const SmoothLinearSolve* second = nullptr;   // epsilon's or omega's; null under LES kEqn, which solves k alone
    const EquationRelax*     relaxFirst  = nullptr;
    const EquationRelax*     relaxSecond = nullptr;
};

ClosurePick pickClosure(
    const InterTurbulence& t,
    bool                   finalIter)
{
    const bool sst = (t.model == InterRasModel::KOmegaSST);
    ClosurePick p;
    p.first      = finalIter ? &t.kSolveFinal : &t.kSolve;
    p.relaxFirst = finalIter ? &t.kRelaxFinal : &t.kRelax;
    if (t.model == InterRasModel::KEqnLES) return p;
    p.second      = sst ? (finalIter ? &t.omegaSolveFinal : &t.omegaSolve)
                        : (finalIter ? &t.epsSolveFinal   : &t.epsSolve);
    p.relaxSecond = sst ? (finalIter ? &t.omegaRelaxFinal : &t.omegaRelax)
                        : (finalIter ? &t.epsRelaxFinal   : &t.epsRelax);
    return p;
}

EqnSolveSetting secondEqnSolve(
    const SmoothLinearSolve& s)
{
    EqnSolveSetting e;
    e.which.pbicgDILU = s.pbicgDILU();
    e.which.smoothSolver = !e.which.pbicgDILU;
    e.which.symmetric = (s.smoother == "symGaussSeidel");
    e.which.nSweeps = s.nSweeps;
    e.tol = s.tol;
    e.relTol = s.relTol;
    e.maxIter = s.maxIter;
    e.minIter = s.minIter;
    return e;
}
}   // namespace


void validateInterTurbulence(
    InterTurbulence& t,
    const GeometricField<vector>& U,
    const std::vector<scalar>& nu,
    const std::vector<std::vector<scalar>>& nuBnd,
    const SurfaceScalarField& phi,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!t.on) return;
    // incompressibleInterPhaseTransportModel.C:99-109: validate() sits in the `else` branch, so the
    // variable lineage enters the first UEqn on the nut the case file holds.
    if (t.variableDensity) return;
    if (t.model == InterRasModel::KEqnLES)
    {
        // eddyViscosity::validate() -> kEqn::correctNut
        LESkEqn::correctNut(t.k, t.delta, t.lesCoeffs, t.nut);
        return;
    }
    if (t.model == InterRasModel::KOmegaSST)
    {
        // eddyViscosity::validate -> kOmegaSSTBase::correctNut() (kOmegaSSTBase.C:129-133), which
        // takes fvc::grad(U) by the case's own grad(U) scheme
        std::vector<tensor> gradU = t.sstCoeffs.gradULeastSq ? fvc::leastSquaresGrad(U, m, g, patches)
                                                             : fvc::gaussGrad(U, m, g, patches);
        if (t.sstCoeffs.gradULimitK > 0)
        {
            cpu::cellLimitGrad(gradU, U, t.sstCoeffs.gradULimitK, m, g, patches);
        }
        kOmegaSST::Compressible sstComp;
        sstComp.nu = &nu;
        sstComp.nuBnd = &nuBnd;
        sstComp.nutPhi = &phi;
        kOmegaSST::correctNutField(U, t.k, t.omega, t.nut, gradU, t.yCell, nearWallDist(m, g, patches),
                                   scalar(0), m, g, patches, t.sstCoeffs, &sstComp);
        return;
    }
    // the mixture's nu in BOTH lineages: a field, never the scalar the single-phase callers pass
    kEpsilonRef::Compressible comp;
    comp.nu = &nu;
    comp.nuBnd = &nuBnd;
    comp.nutPhi = &phi;
    kEpsilonRef::NutWallSelection sel;
    sel.kind = &t.nutWallKind;
    sel.U = &U;
    kEpsilonRef::correctNutField(t.k, t.epsilon, t.nut, nearWallDist(m, g, patches), scalar(0),
                                 patches, t.coeffs, &comp, &sel);
}


void interNuEff(
    const InterTurbulence& t,
    const std::vector<scalar>& nu,
    const std::vector<std::vector<scalar>>& nuBnd,
    std::vector<scalar>& nuEff,
    std::vector<std::vector<scalar>>& nuEffBnd)
{
    nuEff = nu;
    nuEffBnd = nuBnd;
    if (!t.on) return;
    // eddyViscosity::nuEff(): nut + nu, in that order, as a field -- so each patch is nut_b + nu_b
    for (std::size_t c = 0; c < nuEff.size(); ++c)
    {
        nuEff[c] = t.nut.internal[c] + nu[c];
    }
    for (std::size_t pi = 0; pi < nuEffBnd.size(); ++pi)
    {
        const std::vector<scalar>& nb = t.nut.boundary[pi]->value();
        for (std::size_t i = 0; i < nuEffBnd[pi].size(); ++i)
        {
            nuEffBnd[pi][i] = nb[i] + nuBnd[pi][i];
        }
    }
}


// GeometricField::storeOldTimes for the closure's fields under CrankNicolson: the old-old level is
// rotated ONCE per time index, and at the first step of a cold start oldTime().oldTime() is created
// as a copy of oldTime() -- the scheme does not read it that step. `second` is false under LES kEqn,
// which solves k alone. One function for all three closures so they cannot drift on when it happens.
namespace {
// storeOldTimes for the closure's transported scalars: ONCE per TIME INDEX, whatever the ddt scheme and
// however many outer correctors run. Keyed on the step index rather than on the CrankNicolson clock,
// because the old-TIME level is what every scheme reads and only the old-OLD level is CN's.
void advanceTurbulenceOldTime(
    InterTurbulence&                   t,
    label                              timeIndex,
    bool                               second)
{
    if (t.oldStepTimeIndex == timeIndex) return;
    // the old-old level rotates off the outgoing old-time level, as OpenFOAM's oldTime().oldTime() does;
    // at a cold start it is created as a copy of oldTime() and the scheme does not read it that step
    t.cn.kOO = t.kOldStep.empty() ? t.k.internal : t.kOldStep;
    t.kOldStep = t.k.internal;
    if (second)
    {
        const std::vector<scalar>& sec = (t.model == InterRasModel::KOmegaSST) ? t.omega.internal
                                                                               : t.epsilon.internal;
        t.cn.epsOO = t.epsOldStep.empty() ? sec : t.epsOldStep;
        t.epsOldStep = sec;
    }
    t.oldStepTimeIndex = timeIndex;
}
}

void moveInterTurbulence(
    InterTurbulence&            t,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    label                       timeIndex)
{
    if (!t.on) return;
    if (t.model == InterRasModel::KEqnLES)
    {
        // LESModel::correct() calls delta_().correct() FIRST (LESModel.C:251), and
        // cubeRootVolDelta::correct() recomputes the width whenever the mesh is changing
        // (cubeRootVolDelta.C:128-134). The width is (deltaCoeff*V)^(1/3) per cell, so on a mesh that
        // moves it changes at every update and the start-up value is a different filter.
        t.delta = LESdelta::compute(t.deltaSpec, m, g, patches);
        return;
    }
    if (t.model != InterRasModel::KOmegaSST) return;
    if (!t.wallDistPatchIDs.empty())
    {
        // UNCONDITIONAL ON PURPOSE. This is the wallDist object inverseDistanceDiffusivity registered,
        // whose dictionary is fvSchemes' `patchDist` (wallDist.C:96-102 builds it from
        // patchTypeName & "Dist", and that object's patchTypeName is "patch"), NOT `wallDist`. So the
        // `wallDist { updateInterval }` this function honours below is not its interval and must not be
        // applied to it -- the motion solver's own reader is where that one belongs.
        t.yCell = patchWave(m, g, patches, t.wallDistPatchIDs, true).distance;
        return;
    }
    // wallDist::movePoints's SCHEDULE, transcribed (wallDist.C:193-221) rather than reduced to a modulo:
    //
    //     if (updateInterval_ > 0 && (timeIndex % updateInterval_) == 0) requireUpdate_ = true;
    //     if (requireUpdate_ && pdm_->movePoints()) { requireUpdate_ = false; return pdm_->correct(y_); }
    //
    // `patchDistMethod::movePoints()` is the base `return true` for meshWave, so the second test is the
    // latch alone. Three consequences the bare modulo would get wrong: the flag starts TRUE, so the first
    // move after start-up recomputes whatever the interval is; a step the interval does not divide keeps
    // the STALE distance rather than recomputing a fresh one; and `updateInterval <= 0` never sets the
    // flag again, so y is frozen at the start-up value for the whole run instead of updating every step.
    // THE INTERVAL ITSELF IS STILL REFUSED, for want of a fixture and not for want of the code. The
    // schedule below is transcribed and exercised at interval 1 (where the latch is set every step), but
    // NO interFoam tutorial can gate a value other than 1: a rigid solid-body motion leaves every
    // wall-to-cell distance invariant, so testTubeMixer / sloshing* / cylinder / esd cannot witness it at
    // all, and the one DEFORMING case whose closure owns its wallDist -- RAS/floatingObject made
    // kOmegaSST -- has an INERT closure at those settings: MEASURED, OpenFOAM laminar against OpenFOAM
    // with kOmegaSST is U 9.34e-12, so the model moves that case by nothing and the interval can move it
    // by less. A profile was built and withdrawn on exactly that number. Under displacementLaplacian the
    // closure does not own the object at all (its dict is `patchDist`), which rules the rest out.
    if (t.wallDistUpdateInterval != 1)
        throw std::runtime_error(
            std::string(WHO) + "fvSchemes sets `wallDist { updateInterval "
            + std::to_string(t.wallDistUpdateInterval) + "; }` on a moving mesh. The schedule is ported "
            "(wallDist.C:193-221, the latch below) but no shipped tutorial can gate a value other than 1 "
            "-- see the comment above. Refused rather than run ungated.");
    if (t.wallDistUpdateInterval > 0 && (timeIndex % t.wallDistUpdateInterval) == 0)
    {
        t.wallDistRequireUpdate = true;
    }
    if (!t.wallDistRequireUpdate)
    {
        return;
    }
    t.wallDistRequireUpdate = false;
    t.yCell = cellWallDist(m, g, patches, nullptr, nullptr, t.wallDistCorrectWalls);
}


void correctInterTurbulence(
    InterTurbulence& t,
    const InterTurbulenceStepInput& in,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!t.on) return;
    // `turbulence off`: kEpsilon.C:216-219, kOmegaSSTBase.C:502-505 and kEqn.C:141-144 each open
    // correct() with `if (!this->turbulence_) { return; }`. Nothing else in the model is gated -- the
    // constructor bound and validate()'s correctNut have already run -- so this is the whole of it.
    if (t.frozen) return;
    if (!in.U || !in.phi || !in.rhoPhi || !in.rho || !in.rhoBnd || !in.rhoOld || !in.nu || !in.nuBnd)
        throw std::runtime_error(std::string(WHO) + "correctInterTurbulence needs every input field.");
    if (!(in.deltaT > 0))
        throw std::runtime_error(std::string(WHO) + "correctInterTurbulence needs a positive deltaT.");
    // THE SENTINELS, refused rather than defaulted: the old-time snapshot is keyed on the step index and
    // the solver dictionary is picked by the corrector, so a caller that supplied neither would silently
    // get step -1 and a non-final corrector.
    if (in.timeIndex < 0 || in.finalIter < 0)
        throw std::runtime_error(
            std::string(WHO) + "correctInterTurbulence needs the step's timeIndex and finalIter; the "
            "caller supplied " + std::to_string(in.timeIndex) + " and " + std::to_string(in.finalIter)
            + ". The closure keys psi.oldTime() on the first and selects <field>Final by the second.");
    // storeOldTimes, ONCE per step whatever the scheme and however many correctors run. LES kEqn solves k
    // alone, so it has no second scalar to rotate.
    advanceTurbulenceOldTime(t, in.timeIndex, /*second=*/t.model != InterRasModel::KEqnLES);

    if (t.model == InterRasModel::KEqnLES)
    {
        // WHICH DICTIONARY THIS CORRECTOR USES -- Final on the last outer corrector, the plain entry on the
        // others (fvMatrix.C:1536-1542 for solve(), :1249-1263 for relax()).
        const ClosurePick pk = pickClosure(t, in.finalIter != 0);
        const SmoothLinearSolve& ks = *pk.first;
        LESkEqn::Solve sv;
        // the case's own solver, as both RAS branches take it -- PBiCG with DILU where it names one
        sv.which.pbicgDILU = ks.pbicgDILU();
        sv.which.smoothSolver = !sv.which.pbicgDILU;
        sv.which.symmetric = (ks.smoother == "symGaussSeidel");
        sv.which.nSweeps = ks.nSweeps;
        sv.tol = ks.tol;
        sv.relTol = ks.relTol;
        sv.maxIter = ks.maxIter;
        sv.minIter = ks.minIter;
        sv.relaxOn = pk.relaxFirst->on;
        sv.relax = pk.relaxFirst->factor;
        // fvm::ddt(k) under CrankNicolson, with k's old-old level rotated once per time index. This
        // lineage is the uniform one, so there is no density at any level.
        if (in.cn)
        {
            t.cn.ddt0K.name = "ddt0(k)";
        }
        sv.kOld = &t.kOldStep;   // psi.oldTime(), per STEP -- see InterTurbulence::kOldStep
        const SolverPerformance p = LESkEqn::correct(*in.U, t.k, t.nut, *in.phi, *in.nu, *in.nuBnd, t.delta,
                                                     in.deltaT, t.lesCoeffs, sv, m, g, patches, t.lesTaps,
                                                     in.cn, in.cn ? &t.cn.ddt0K : nullptr,
                                                     in.cn ? &t.cn.kOO : nullptr,
                                                     // ...and a MOVING mesh's two terms, the pair
                                                     // the RAS closures already take
                                                     in.V0, in.meshPhi);
        if (in.kLog)
        {
            in.kLog->push_back({p.initialResidual, p.finalResidual, p.nIterations});
        }
        return;
    }
    if (t.model == InterRasModel::KOmegaSST)
    {
        // the ordinary incompressible kOmegaSST: alpha = rho = 1, the volumetric phi, the mixture's nu
        kOmegaSST::Compressible sstComp;
        sstComp.nu = in.nu;
        sstComp.nuBnd = in.nuBnd;
        sstComp.rDeltaT = scalar(1) / in.deltaT;
        // psi.oldTime(), per STEP -- see InterTurbulence::kOldStep
        sstComp.kOldIn = &t.kOldStep;
        sstComp.omegaOldIn = &t.epsOldStep;
        sstComp.nutPhi = in.phi;
        sstComp.V0 = in.V0;
        sstComp.meshPhi = in.meshPhi;
        // ...and fvm::ddt through ddtSchemes on BOTH equations under CrankNicolson (kOmegaSSTBase.C
        // :572 and :602). The second field is omega, and it uses the kEpsilon slots of the shared
        // CrankNicolson block, as every other per-model slot on this branch does.
        if (in.cn)
        {
            t.cn.ddt0K.name = "ddt0(k)";
            t.cn.ddt0Eps.name = "ddt0(omega)";
            sstComp.cn = in.cn;
            sstComp.cnDdt0K = &t.cn.ddt0K;
            sstComp.cnDdt0Omega = &t.cn.ddt0Eps;
            sstComp.kOO = &t.cn.kOO;
            sstComp.omegaOO = &t.cn.epsOO;
        }
        // WHICH DICTIONARY THIS CORRECTOR USES -- Final on the last outer corrector, the plain entry on the
        // others (fvMatrix.C:1536-1542 for solve(), :1249-1263 for relax()).
        const ClosurePick pk = pickClosure(t, in.finalIter != 0);
        const SmoothLinearSolve& ks = *pk.first;
        // omegaFinal NEED NOT MATCH kFinal. fvMatrix::solve() looks the solver dictionary up by FIELD
        // name, so OpenFOAM honours each; the closure takes k's positionally and omega's through
        // EqnSolveSetting. This used to refuse the pair outright.
        const EqnSolveSetting omegaSolve = secondEqnSolve(*pk.second);
        // THE SOLVER THE CASE NAMES. This said `smoothSolver = true` whatever fvSolution gave, so a
        // case naming PBiCG ran symGaussSeidel sweeps under PBiCG's tolerance -- the substitution the
        // kEpsilon branch below was fixed for, in the twin nobody looked at.
        LinearSolverChoice which;
        which.pbicgDILU = ks.pbicgDILU();
        which.smoothSolver = !which.pbicgDILU;
        which.symmetric = (ks.smoother == "symGaussSeidel");
        which.nSweeps = ks.nSweeps;
        kOmegaSST::SSTResiduals res;
        // THE CLOSURE'S ASSEMBLED SYSTEM, for the of-instrument comparison against
        // tools/dumpKOmegaSST's stage_sstOmD / OmSrc / OmDUpper / OmDLower. Writes only, gated on the
        // same variable the stage dump uses, and OFF unless it is set -- rhoSimpleFoam's SST site has
        // carried the same capture since its own port (rhoSimpleFoam_cpp.cu:1316).
        res.captureStages = (std::getenv("BRAE_SST_DUMP_DIR") != nullptr);
        kOmegaSST::correct(*in.U, t.k, t.omega, t.nut, *in.phi, t.yCell, scalar(0), m, g, patches,
                           pk.relaxSecond->factor, pk.relaxFirst->factor, ks.tol, ks.relTol, ks.maxIter,
                           t.sstCoeffs, &res, /*bounded=*/false, t.kDiv.limitedLinear,
                           t.kDiv.limiterCoeff, /*linearUpwind=*/false,
                           t.coeffs.correctedLaplacian, t.coeffs.snGradLimitCoeff, /*lm=*/nullptr,
                           &sstComp, ks.minIter, pk.relaxSecond->on, pk.relaxFirst->on, &which,
                           &omegaSolve, &t.secondDiv, &t.secondGrad, t.coeffs.nonOrthCoeffs);
        // The assembled systems are WRITTEN BY THE CLOSURE (kOmegaSST_cpp.cu), at the call its stage
        // dump latched. This site wrote them on every call instead, so the files held the LAST
        // closure call while every other column in the directory held the first.
        if (in.omegaLog)
        {
            in.omegaLog->push_back({res.omegaPerf.initialResidual, res.omegaPerf.finalResidual,
                                    res.omegaPerf.nIterations});
        }
        if (in.kLog)
        {
            in.kLog->push_back({res.kPerf.initialResidual, res.kPerf.finalResidual,
                                res.kPerf.nIterations});
        }
        return;
    }

    // the mixture's nu in BOTH lineages, as at validate()
    kEpsilonRef::Compressible comp;
    comp.nu = in.nu;
    comp.nuBnd = in.nuBnd;
    comp.rDeltaT = scalar(1) / in.deltaT;
    if (in.cn)
    {
        // k.oldTime().oldTime(): GeometricField::storeOldTimes rotates the levels once per time
        // index, on the first access of the step. At the first step of a cold start oldTime().oldTime()
        // is created as a copy of oldTime(), and the scheme does not read it that step.
        InterTurbulenceCrankNicolson& c = t.cn;
        c.ddt0K.name = t.variableDensity ? "ddt0(rho,k)" : "ddt0(k)";
        c.ddt0Eps.name = t.variableDensity ? "ddt0(rho,epsilon)" : "ddt0(epsilon)";
        comp.cn = in.cn;
        comp.cnDdt0K = &c.ddt0K;
        comp.cnDdt0Eps = &c.ddt0Eps;
        comp.kOO = &c.kOO;
        comp.epsOO = &c.epsOO;
        if (t.variableDensity)
        {
            if (!in.rhoOO)
                throw std::runtime_error(
                    std::string(WHO) + "CrankNicolson under `density variable` needs rho.oldTime().oldTime().");
            comp.rhoOO = in.rhoOO;
        }
    }
    comp.nutPhi = in.phi;
    comp.V0 = in.V0;
    comp.meshPhi = in.meshPhi;
    // The equation's own flux. In the variable lineage that is rhoPhi, while divU and every
    // flux-conditional patch still read the volumetric phi.
    const SurfaceScalarField* eqnFlux = in.phi;
    if (t.variableDensity)
    {
        comp.rho = in.rho;
        comp.rhoBnd = in.rhoBnd;
        comp.rhoOld = in.rhoOld;
        comp.phiByRho = in.phi;
        comp.bcPhi = in.phi;
        eqnFlux = in.rhoPhi;
    }
    // psi.oldTime(), per STEP -- BOTH lineages. These two were mis-indented INSIDE the
    // `if (t.variableDensity)` block above, so on the uniform lineage they stayed null and
    // kEpsilon_cpp.cu:387-388 fell back to the CURRENT field. Localised with tools/dumpKEpsilon against
    // OpenFOAM's own six numbers: with the per-call capture, corrector 2 of step 1 reads epsilon's initial
    // residual 0.00271633 where OpenFOAM reads 0.01414477 (relative 0.808) and k's 0.38632 where OpenFOAM
    // reads 0.03982068 (relative 8.702), with both final residuals and both iteration counts reproduced --
    // brae's reported 8.080e-01 and 8.702e+00 to three and four digits. The two sibling closures set these
    // unconditionally and the DEVICE arm was already correct, which is why only this lineage was wrong.
    comp.kOldIn = &t.kOldStep;
    comp.epsOldIn = &t.epsOldStep;

    kEpsilonRef::NutWallSelection sel;
    sel.kind = &t.nutWallKind;
    sel.U = in.U;

    // epsilonFinal NEED NOT MATCH kFinal -- see the SST branch. Every shipped tutorial writes the two as
    // one regex key, which is why the old refusal was never reached by a tutorial and why the profile
    // that gates this one is staged.
    // WHICH DICTIONARY THIS CORRECTOR USES -- Final on the last outer corrector, the plain entry on the
    // others (fvMatrix.C:1536-1542 for solve(), :1249-1263 for relax()).
    const ClosurePick pk = pickClosure(t, in.finalIter != 0);
    const SmoothLinearSolve& ks = *pk.first;
    const EqnSolveSetting epsSolve = secondEqnSolve(*pk.second);
    LinearSolverChoice which;
    which.pbicgDILU = ks.pbicgDILU();
    which.smoothSolver = !which.pbicgDILU;
    which.symmetric = (ks.smoother == "symGaussSeidel");
    which.nSweeps = ks.nSweeps;

    kEpsilonRef::KEResiduals res;
    // Instrument: BRAE_STAGE_DUMP_DIR=<dir> (+ BRAE_STAGE_DUMP_ITER=n, default 1) writes this
    // closure's as-solved epsilon and k systems, at ONE call, under the SAME names the DEVICE closure
    // writes through solveScalarEqn's dumpPrefix (kEpsilon.cu:1240) and the same names rhoSimpleFoam's
    // host driver uses -- so the two arms' columns, and OpenFOAM's own from tools/dumpKEpsilon, diff
    // directly. The latch is here and not in the closure because the capture already is
    // (KEResiduals::captureStages); writing on every call is what left the SST dump comparing step ten
    // against step one, so this one counts its calls.
    struct KeDump
    {
        std::string dir;
        bool on = false;
        void scalars(const char* name, const std::vector<scalar>& v) const
        {
            if (!on) return;
            std::ofstream o(dir + "/" + name);
            o.precision(17);
            for (const scalar x : v) o << x << "\n";
        }
    } kd;
    if (const char* dd = std::getenv("BRAE_STAGE_DUMP_DIR"))
    {
        static int calls = 0;
        const char* it = std::getenv("BRAE_STAGE_DUMP_ITER");
        if (++calls == (it && *it ? std::atoi(it) : 1))
        {
            std::error_code ec;
            std::filesystem::create_directories(dd, ec);
            kd.dir = dd;
            kd.on = !ec;
        }
    }
    res.captureStages = kd.on;
    kEpsilonRef::correct(*in.U, t.k, t.epsilon, t.nut, *eqnFlux, scalar(0), m, g, patches,
                         pk.relaxSecond->factor, pk.relaxFirst->factor, ks.tol, ks.relTol, ks.maxIter,
                         t.coeffs, &res, /*bounded=*/false, /*dropTerm=*/0, &comp, in.fvOptions,
                         pk.relaxSecond->on, pk.relaxFirst->on, /*constrainBeforeWall=*/true,
                         // the limiter's gradient limiter is `t.coeffs.gradKLimitK`, which the closure
                         // reads from the coeffs it was handed -- this site used to pass a literal 0
                         // here and limited nothing (1.9e-01 off OpenFOAM, gated on RAS/damBreak)
                         t.kDiv.limitedLinear, t.kDiv.limiterCoeff,
                         ks.minIter, &sel, /*linearUpwind=*/false, /*luGradK=*/scalar(0), &which,
                         &epsSolve, &t.secondDiv, &t.secondGrad);
    if (kd.on)
    {
        kd.scalars("epsD", res.epsD);     kd.scalars("epsSrc", res.epsSrc);
        kd.scalars("epsUpper", res.epsUpper); kd.scalars("epsLower", res.epsLower);
        kd.scalars("kD", res.kD);         kd.scalars("kSrc", res.kSrc);
        kd.scalars("kUpper", res.kUpper); kd.scalars("kLower", res.kLower);
    }
    if (in.epsilonLog)
    {
        in.epsilonLog->push_back({res.epsPerf.initialResidual, res.epsPerf.finalResidual,
                                  res.epsPerf.nIterations});
    }
    if (in.kLog)
    {
        in.kLog->push_back({res.kPerf.initialResidual, res.kPerf.finalResidual,
                            res.kPerf.nIterations});
    }
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
