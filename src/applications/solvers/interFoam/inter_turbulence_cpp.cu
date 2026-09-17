// interFoam's turbulence, the host reference. See inter_turbulence_cpp.cuh for the two lineages.
#include "inter_turbulence_cpp.cuh"
#include "foam_field_reader.cuh"
#include "kEpsilon_cpp.cuh"
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
            std::string(WHO) + "the case is RAS kEpsilon and " + path + " does not exist. OpenFOAM "
            "reads k, epsilon and nut MUST_READ when the model is constructed and stops without them.");
    GeometricField<scalar> f = buildField<scalar>(readField<scalar>(path), patches, nCells);
    f.evaluateBoundary();
    return f;
}

// The solver entry fvMatrix::solve() selects on the final outer corrector.
SmoothLinearSolve readFinalSolve(
    const FoamDict& fvSolution,
    const std::string& field)
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
    if (!s.gaussSeidel())
        throw std::runtime_error(
            std::string(WHO) + "`solvers/" + name + "` names `solver " + s.solver + "; smoother "
            + s.smoother + ";`. brae's interFoam closure runs OpenFOAM's smoothSolver with the "
            "GaussSeidel or symGaussSeidel smoother -- what every kEpsilon tutorial names -- and "
            "nothing else: a substituted solver at the same tolerance stops somewhere else.");
    return s;
}

// div(<flux>,<field>): `Gauss upwind` and nothing else. The closure has limitedLinear and
// linearUpwind, but no interFoam case gates them here and all 11 kEpsilon tutorials name upwind.
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

// grad(U), which the production GbyNu takes (kEpsilon.C:237)
void readGradU(
    const std::string& caseDir,
    KEpsilonCoeffs& co)
{
    const FieldGradScheme gu = parseFieldGradScheme(caseDir, "U");
    if (!gu.unsupportedLimiter.empty() || !(gu.gaussLinear || gu.leastSquares))
        throw std::runtime_error(
            std::string(WHO) + "fvSchemes grad(U) resolves to `" + gu.raw + "`, which the kEpsilon "
            "production does not compute (Gauss linear and leastSquares, optionally cellLimited).");
    co.gradULeastSq = gu.leastSquares;
    co.gradULimitK = gu.cellLimitK;
}

// grad(k) and grad(epsilon), which the corrected laplacian's deferred correction takes. The closure
// carries ONE setting for both fields.
void readGradK(
    const std::string& caseDir,
    KEpsilonCoeffs& co)
{
    const FieldGradScheme gk = parseFieldGradScheme(caseDir, "k");
    const FieldGradScheme ge = parseFieldGradScheme(caseDir, "epsilon");
    for (const FieldGradScheme* q : {&gk, &ge})
    {
        if (!q->unsupportedLimiter.empty() || !(q->gaussLinear || q->leastSquares))
            throw std::runtime_error(
                std::string(WHO) + "fvSchemes resolves a turbulence gradient to `" + q->raw
                + "`, which the kEpsilon closure does not compute.");
    }
    if (gk.leastSquares != ge.leastSquares || gk.cellLimitK != ge.cellLimitK)
        throw std::runtime_error(
            std::string(WHO) + "fvSchemes names different schemes for grad(k) (`" + gk.raw
            + "`) and grad(epsilon) (`" + ge.raw + "`); the closure carries one for both.");
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

} // namespace


InterTurbulence readInterTurbulence(
    const std::string& caseDir,
    const std::string& startDir,
    const FoamDict& fvSolution,
    bool eulerDdt,
    bool laplacianCorrected,
    scalar laplacianLimitCoeff,
    const std::vector<FvPatch>& patches,
    label nCells)
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
    if (sim != "RAS")
        throw std::runtime_error(
            std::string(WHO) + file + " asks for simulationType `" + sim + "`. brae's interFoam has "
            "laminar and RAS kEpsilon; LES is not ported here.");

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

    const FoamDict* ras = d.subDict("RAS");
    const std::string model = ras ? ras->wordOr("RASModel", "") : "";
    if (model != "kEpsilon")
        throw std::runtime_error(
            std::string(WHO) + file + " asks for RASModel `" + model + "`. kEpsilon is the one model "
            "wired into interFoam (11 of the 17 turbulent tutorials); refusing rather than running it "
            "under another model's name.");
    const std::string sw = ras->wordOr("turbulence", "on");
    if (sw == "off" || sw == "no" || sw == "false")
        throw std::runtime_error(
            std::string(WHO) + file + " has `RAS { turbulence off; }`. The frozen model keeps the nut "
            "its construction left -- validate()'s in one lineage, the case file's in the other -- "
            "and no gate holds that against OpenFOAM yet. Refused rather than run ungated.");
    if (!eulerDdt)
        throw std::runtime_error(
            std::string(WHO) + "the k and epsilon equations take fvm::ddt through ddtSchemes "
            "(kEpsilon.C:254, :275) and the closure carries Euler only.");

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
    t.coeffs.correctedLaplacian = laplacianCorrected;
    t.coeffs.snGradLimitCoeff = laplacianLimitCoeff;
    readGradU(caseDir, t.coeffs);
    readGradK(caseDir, t.coeffs);

    // THE KEY CARRIES THE FLUX'S NAME: div(rhoPhi,k) in the variable lineage, div(phi,k) in the other
    const std::string flux = t.variableDensity ? "rhoPhi" : "phi";
    requireUpwind(caseDir, "k", flux);
    requireUpwind(caseDir, "epsilon", flux);

    t.k = readTurbulenceField(startDir, "k", patches, nCells);
    t.epsilon = readTurbulenceField(startDir, "epsilon", patches, nCells);
    t.nut = readTurbulenceField(startDir, "nut", patches, nCells);
    t.nutWallKind = readNutWallKinds(startDir, t.epsilon, patches);

    t.kSolveFinal = readFinalSolve(fvSolution, "k");
    t.epsSolveFinal = readFinalSolve(fvSolution, "epsilon");
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* eq = rf ? rf->subDict("equations") : nullptr;
    t.kRelaxFinal = EquationRelax::read(eq, "kFinal");
    t.epsRelaxFinal = EquationRelax::read(eq, "epsilonFinal");

    t.on = true;
    return t;
}


void validateInterTurbulence(
    InterTurbulence& t,
    const GeometricField<vector>& U,
    const std::vector<scalar>& nu,
    const std::vector<std::vector<scalar>>& nuBnd,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!t.on) return;
    // incompressibleInterPhaseTransportModel.C:99-109: validate() sits in the `else` branch, so the
    // variable lineage enters the first UEqn on the nut the case file holds.
    if (t.variableDensity) return;
    // the mixture's nu in BOTH lineages: a field, never the scalar the single-phase callers pass
    kEpsilonRef::Compressible comp;
    comp.nu = &nu;
    comp.nuBnd = &nuBnd;
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

    // the mixture's nu in BOTH lineages, as at validate()
    kEpsilonRef::Compressible comp;
    comp.nu = in.nu;
    comp.nuBnd = in.nuBnd;
    comp.rDeltaT = scalar(1) / in.deltaT;
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
    if (ks.smoother != es.smoother || ks.tol != es.tol || ks.relTol != es.relTol
     || ks.maxIter != es.maxIter || ks.minIter != es.minIter || ks.nSweeps != es.nSweeps)
        throw std::runtime_error(
            std::string(WHO) + "fvSolution gives kFinal and epsilonFinal different solver settings; "
            "the closure takes one set for both equations.");
    LinearSolverChoice which;
    which.smoothSolver = true;
    which.symmetric = (ks.smoother == "symGaussSeidel");
    which.nSweeps = ks.nSweeps;

    kEpsilonRef::KEResiduals res;
    kEpsilonRef::correct(*in.U, t.k, t.epsilon, t.nut, *eqnFlux, scalar(0), m, g, patches,
                         t.epsRelaxFinal.factor, t.kRelaxFinal.factor, ks.tol, ks.relTol, ks.maxIter,
                         t.coeffs, &res, /*bounded=*/false, /*dropTerm=*/0, &comp, /*fvOpts=*/nullptr,
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
