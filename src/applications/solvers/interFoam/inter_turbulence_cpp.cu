// interFoam's turbulence, the host reference. See inter_turbulence_cpp.cuh for the two lineages.
#include "inter_turbulence_cpp.cuh"
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
SmoothLinearSolve readFinalSolve(
    const FoamDict& fvSolution,
    const std::string& field,
    // the kEpsilon closure also runs PBiCG with DILU (pbicg.cuh); the others do not
    bool allowPBiCG = false)
{
    const std::string name = field + "Final";
    const FoamDict* sv = fvSolution.subDict("solvers");
    const FoamDict* d = sv ? sv->subDict(name) : nullptr;
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
            "turbulent tutorial names -- and, under kEpsilon, PBiCG with DILU; nothing else: a "
            "substituted solver at the same tolerance stops somewhere else.");
    return s;
}

// div(<flux>,<field>): `Gauss upwind` and nothing else. The closures have limitedLinear and
// linearUpwind, but no interFoam case gates them here: all 11 kEpsilon tutorials name upwind, and so
// does waterChannel for k and omega.
void requireUpwind(
    const std::string& caseDir,
    const std::string& field,
    const std::string& fluxName)
{
    const FieldDivScheme fs = parseFieldDivScheme(caseDir, field, false, fluxName);
    if (fs.bounded || fs.limited || fs.linearUpwind)
        throw std::runtime_error(
            std::string(WHO) + "fvSchemes `div(" + fluxName + "," + field + ")` is not plain `Gauss "
            "upwind`. That is the one convection scheme the interFoam turbulence port is gated on; "
            "refusing rather than running an ungated one.");
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
    Coeffs& co)
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
    if (gk.leastSquares != ge.leastSquares || gk.cellLimitK != ge.cellLimitK)
        throw std::runtime_error(
            std::string(WHO) + "fvSchemes names different schemes for grad(k) (`" + gk.raw
            + "`) and grad(" + second + ") (`" + ge.raw + "`); the closure carries one for both.");
    co.gradKLeastSq = gk.leastSquares;
    co.gradKLimitK = gk.cellLimitK;
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
            const bool coeffsDiffer = (b->hasWfCmu && b->wfCmu != co.CmuWall)
                                   || (b->hasWfKappa && b->wfKappa != co.kappa)
                                   || (b->hasWfE && b->wfE != co.E);
            if (coeffsDiffer)
                throw std::runtime_error(
                    std::string(WHO) + "wall patch `" + p.name + "` names wall-function coefficients "
                    "other than Cmu 0.09, kappa 0.41, E 9.8. The kOmegaSST closure carries one set for "
                    "every wall and this reader does not thread a patch's own through.");
            const bool blendOther = !b->wfBlending.empty()
                                 && (b->wfBlending != "binomial" || (b->hasWfBlendN && b->wfBlendN != 2));
            if (blendOther)
                throw std::runtime_error(
                    std::string(WHO) + "wall patch `" + p.name + "` names `blending " + b->wfBlending
                    + "`. The kOmegaSST closure blends omega's viscous and log values binomially with "
                    "n = 2, OpenFOAM's default (omegaWallFunctionFvPatchScalarField.C:445), and "
                    "nothing else.");
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
    scalar laplacianLimitCoeff,
    const std::vector<FvPatch>& patches,
    label nCells,
    const PrimitiveMesh* mesh,
    const FvGeometry* geometry,
    const std::vector<label>* sharedWallDistPatches)
{
    InterTurbulence t;
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
        const std::string lsw = les->wordOr("turbulence", "on");
        if (lsw == "off" || lsw == "no" || lsw == "false")
            throw std::runtime_error(
                std::string(WHO) + file + " has `LES { turbulence off; }`, which no gate holds against "
                "OpenFOAM yet.");
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
        t.kSolveFinal = readFinalSolve(fvSolution, "k");
        t.kRelaxFinal = EquationRelax::read(eqAll, "kFinal");
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
    const std::string sw = ras->wordOr("turbulence", "on");
    if (sw == "off" || sw == "no" || sw == "false")
        throw std::runtime_error(
            std::string(WHO) + file + " has `RAS { turbulence off; }`. The frozen model keeps the nut "
            "its construction left -- validate()'s in one lineage, the case file's in the other -- "
            "and no gate holds that against OpenFOAM yet. Refused rather than run ungated.");
    // kEpsilon carries CrankNicolson too (InterTurbulenceStepInput::cn); kOmegaSST does not, and the
    // case reader has already said so by name where the two meet
    if (!eulerDdt && t.model != InterRasModel::KEpsilon)
        throw std::runtime_error(
            std::string(WHO) + "the closure's two equations take fvm::ddt through ddtSchemes "
            "(kOmegaSSTBase.C:558, :589) and this closure carries Euler only.");
    t.coeffs.correctedLaplacian = laplacianCorrected;
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
        const std::string decay = (sstDict ? sstDict : ras)->wordOr("decayControl", "no");
        if (decay == "yes" || decay == "on" || decay == "true")
            throw std::runtime_error(
                std::string(WHO) + file + " sets `decayControl " + decay + "`. It adds "
                "beta*sqr(omegaInf) to the omega equation and betaStar*omegaInf*kInf to k's "
                "(kOmegaSSTBase.C:574, :594), and kOmegaSST_cpp carries neither.");
        if (t.sstCoeffs.F3)
            throw std::runtime_error(
                std::string(WHO) + file + " sets `F3 yes`. kOmegaSSTBase multiplies F23 by F3, which "
                "changes the eddy-viscosity limiter and the production limiter; not ported.");
        readGradU(caseDir, t.sstCoeffs);
        readGradK(caseDir, "omega", t.sstCoeffs);
        requireUpwind(caseDir, "k", "phi");
        requireUpwind(caseDir, "omega", "phi");

        t.k = readTurbulenceField(startDir, "k", patches, nCells);
        t.omega = readTurbulenceField(startDir, "omega", patches, nCells);
        t.nut = readTurbulenceField(startDir, "nut", patches, nCells);
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
            const std::string cw = wd->wordOr("correctWalls", "true");
            if (cw == "false" || cw == "no" || cw == "off")
                throw std::runtime_error(
                    std::string(WHO) + "fvSchemes sets `wallDist { correctWalls " + cw + "; }`; brae's "
                    "meshWave always corrects the near-wall cells.");
            t.wallDistUpdateInterval = static_cast<label>(wd->scalarOr("updateInterval", 1));
            t.yCell = cellWallDist(*mesh, *geometry, patches);
        }

        t.kSolveFinal = readFinalSolve(fvSolution, "k");
        t.omegaSolveFinal = readFinalSolve(fvSolution, "omega");
        t.kRelaxFinal = EquationRelax::read(eqAll, "kFinal");
        t.omegaRelaxFinal = EquationRelax::read(eqAll, "omegaFinal");
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
    readGradK(caseDir, "epsilon", t.coeffs);

    // THE KEY CARRIES THE FLUX'S NAME: div(rhoPhi,k) in the variable lineage, div(phi,k) in the other
    const std::string flux = t.variableDensity ? "rhoPhi" : "phi";
    requireUpwind(caseDir, "k", flux);
    requireUpwind(caseDir, "epsilon", flux);

    t.k = readTurbulenceField(startDir, "k", patches, nCells);
    t.epsilon = readTurbulenceField(startDir, "epsilon", patches, nCells);
    t.nut = readTurbulenceField(startDir, "nut", patches, nCells);
    t.nutWallKind = readNutWallKinds(startDir, t.epsilon, patches);
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

    t.on = true;
    return t;
}


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


void moveInterTurbulence(
    InterTurbulence&            t,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    if (!t.on || t.model != InterRasModel::KOmegaSST) return;
    if (!t.wallDistPatchIDs.empty())
    {
        t.yCell = patchWave(m, g, patches, t.wallDistPatchIDs, true).distance;
        return;
    }
    // updateInterval N recomputes on every Nth time index only, and keeps the stale distance between
    if (t.wallDistUpdateInterval != 1)
        throw std::runtime_error(
            std::string(WHO) + "fvSchemes sets `wallDist { updateInterval "
            + std::to_string(t.wallDistUpdateInterval) + "; }` on a moving mesh; only 1, the default, "
            "is ported.");
    t.yCell = cellWallDist(m, g, patches);
}


void correctInterTurbulence(
    InterTurbulence& t,
    const InterTurbulenceStepInput& in,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!t.on) return;
    if (!in.U || !in.phi || !in.rhoPhi || !in.rho || !in.rhoBnd || !in.rhoOld || !in.nu || !in.nuBnd)
        throw std::runtime_error(std::string(WHO) + "correctInterTurbulence needs every input field.");
    if (!(in.deltaT > 0))
        throw std::runtime_error(std::string(WHO) + "correctInterTurbulence needs a positive deltaT.");

    if (t.model == InterRasModel::KEqnLES)
    {
        const SmoothLinearSolve& ks = t.kSolveFinal;
        LESkEqn::Solve sv;
        sv.which.smoothSolver = true;
        sv.which.symmetric = (ks.smoother == "symGaussSeidel");
        sv.which.nSweeps = ks.nSweeps;
        sv.tol = ks.tol;
        sv.relTol = ks.relTol;
        sv.maxIter = ks.maxIter;
        sv.minIter = ks.minIter;
        sv.relaxOn = t.kRelaxFinal.on;
        sv.relax = t.kRelaxFinal.factor;
        const SolverPerformance p = LESkEqn::correct(*in.U, t.k, t.nut, *in.phi, *in.nu, *in.nuBnd, t.delta,
                                                     in.deltaT, t.lesCoeffs, sv, m, g, patches, t.lesTaps);
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
        sstComp.nutPhi = in.phi;
        sstComp.V0 = in.V0;
        sstComp.meshPhi = in.meshPhi;
        const SmoothLinearSolve& ks = t.kSolveFinal;
        const SmoothLinearSolve& os = t.omegaSolveFinal;
        if (ks.smoother != os.smoother || ks.tol != os.tol || ks.relTol != os.relTol
         || ks.maxIter != os.maxIter || ks.minIter != os.minIter || ks.nSweeps != os.nSweeps)
            throw std::runtime_error(
                std::string(WHO) + "fvSolution gives kFinal and omegaFinal different solver settings; "
                "the closure takes one set for both equations.");
        LinearSolverChoice which;
        which.smoothSolver = true;
        which.symmetric = (ks.smoother == "symGaussSeidel");
        which.nSweeps = ks.nSweeps;
        kOmegaSST::SSTResiduals res;
        kOmegaSST::correct(*in.U, t.k, t.omega, t.nut, *in.phi, t.yCell, scalar(0), m, g, patches,
                           t.omegaRelaxFinal.factor, t.kRelaxFinal.factor, ks.tol, ks.relTol, ks.maxIter,
                           t.sstCoeffs, &res, /*bounded=*/false, /*limitedLinear=*/false,
                           /*limiterCoeff=*/scalar(1), /*linearUpwind=*/false,
                           t.coeffs.correctedLaplacian, t.coeffs.snGradLimitCoeff, /*lm=*/nullptr,
                           &sstComp, ks.minIter, t.omegaRelaxFinal.on, t.kRelaxFinal.on, &which);
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
        if (c.timeIndex != in.cn->timeIndex)
        {
            c.kOO = c.kEntry.empty() ? t.k.internal : c.kEntry;
            c.epsOO = c.epsEntry.empty() ? t.epsilon.internal : c.epsEntry;
            c.kEntry = t.k.internal;
            c.epsEntry = t.epsilon.internal;
            c.timeIndex = in.cn->timeIndex;
        }
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

    kEpsilonRef::NutWallSelection sel;
    sel.kind = &t.nutWallKind;
    sel.U = in.U;

    // ONE solver entry for both equations is what kEpsilonRef::correct takes; the two Final entries
    // are read separately and must agree. Every tutorial writes them as one regex key.
    const SmoothLinearSolve& ks = t.kSolveFinal;
    const SmoothLinearSolve& es = t.epsSolveFinal;
    if (ks.solver != es.solver || ks.preconditioner != es.preconditioner
     || ks.smoother != es.smoother || ks.tol != es.tol || ks.relTol != es.relTol
     || ks.maxIter != es.maxIter || ks.minIter != es.minIter || ks.nSweeps != es.nSweeps)
        throw std::runtime_error(
            std::string(WHO) + "fvSolution gives kFinal and epsilonFinal different solver settings; "
            "the closure takes one set for both equations.");
    LinearSolverChoice which;
    which.pbicgDILU = ks.pbicgDILU();
    which.smoothSolver = !which.pbicgDILU;
    which.symmetric = (ks.smoother == "symGaussSeidel");
    which.nSweeps = ks.nSweeps;

    kEpsilonRef::KEResiduals res;
    kEpsilonRef::correct(*in.U, t.k, t.epsilon, t.nut, *eqnFlux, scalar(0), m, g, patches,
                         t.epsRelaxFinal.factor, t.kRelaxFinal.factor, ks.tol, ks.relTol, ks.maxIter,
                         t.coeffs, &res, /*bounded=*/false, /*dropTerm=*/0, &comp, in.fvOptions,
                         t.epsRelaxFinal.on, t.kRelaxFinal.on, /*constrainBeforeWall=*/true,
                         /*limitedLinear=*/false, /*limiterCoeff=*/scalar(1), /*limGradK=*/scalar(0),
                         ks.minIter, &sel, /*linearUpwind=*/false, /*luGradK=*/scalar(0), &which);
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
