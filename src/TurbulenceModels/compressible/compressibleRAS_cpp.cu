// compressibleRAS_cpp.cu -- the case half of OpenFOAM's compressible RAS model (see the header).
#include "compressibleRAS_cpp.cuh"

#include "cell_wall_dist.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "fv_patch_field.cuh"
#include "near_wall_dist.cuh"
#include "scheme_parse.cuh"

#include <cmath>
#include <filesystem>
#include <regex>
#include <stdexcept>

namespace brae {
namespace compressibleRAS {

namespace {

[[noreturn]] void refuse(const std::string& why)
{
    throw std::runtime_error("brae compressible RAS: " + why);
}

bool isConstraint(const std::string& t)
{
    return t == "empty" || t == "symmetryPlane" || t == "symmetry" || t == "wedge" || t == "slip";
}

// fvSchemes entry text for `key` in `block`, as OpenFOAM resolves it (ofSchemeStatement, below)
std::string scheme(const std::string& caseDir, const char* block, const std::string& key)
{
    return ofSchemeStatement(caseDir, block, key);
}

// div(phi,<field>): the forms the closure runs -- Gauss upwind, Gauss limitedLinear <k>, Gauss linearUpwind
// <grad>, each optionally `bounded`; anything else refused by name
DivScheme divScheme(const std::string& caseDir, const std::string& key)
{
    const std::string s = scheme(caseDir, "divSchemes", key);
    std::smatch m;
    DivScheme d;
    std::string rest = s;
    if (rest.rfind("bounded ", 0) == 0)
    {
        d.bounded = true;
        rest = rest.substr(8);
    }
    if (rest == "Gauss upwind") return d;
    if (std::regex_match(rest, m, std::regex(R"(Gauss limitedLinear ([0-9.eE+-]+))")))
    {
        d.limitedLinear = true;
        d.limiterCoeff = std::stod(m[1]);
        // limitedLinear.H:67-76: k is read with no default and OpenFOAM stops outside [0, 1]; the coefficient was
        // taken here whatever it was
        if (d.limiterCoeff < 0 || d.limiterCoeff > 1)
            refuse("fvSchemes resolves " + key + " to `" + s + "`: limitedLinear's coefficient should be >= 0 and <= 1 "
                   "(limitedLinear.H:67-76, where OpenFOAM stops with the same FatalIOError).");
        return d;
    }
    if (std::regex_match(rest, m, std::regex(R"(Gauss linearUpwind (\S+))")))
    {
        d.linearUpwind = true;
        // THE GRADIENT IT NAMES. linearUpwind.H's constructor builds gradScheme::New(mesh, mesh.gradScheme(name)):
        // the gradSchemes entry of that name, else the last pattern matching it, else `default`. This accepted any
        // name and ran the correction on an unlimited Gauss gradient, so `linearUpwind limited` beside
        // `limited cellLimited Gauss linear 1` ran unlimited. The closure carries Gauss linear, optionally cellLimited
        // (KOmegaSSTInput::luGradK); any other gradient is refused by name.
        d.luGradName = m[1];
        const std::string g = scheme(caseDir, "gradSchemes", d.luGradName);
        std::smatch gm;
        if (g == "Gauss linear")
        {
            d.luGradK = 0.0;
        }
        else if (std::regex_match(g, gm, std::regex(R"(cellLimited Gauss linear ([0-9.eE+-]+))")))
        {
            d.luGradK = std::stod(gm[1]);
            if (d.luGradK < 0 || d.luGradK > 1)
                refuse("fvSchemes gradSchemes resolves " + d.luGradName + " (the gradient " + key + "'s linearUpwind "
                       "names) to `" + g + "`: cellLimited's coefficient should be >= 0 and <= 1 (cellLimitedGrad.C).");
        }
        else
        {
            refuse("fvSchemes gradSchemes resolves " + d.luGradName + ", the gradient " + key + "'s `Gauss linearUpwind "
                   + d.luGradName + "` names, to `" + g + "`. linearUpwind's correction is ported with `Gauss linear` "
                   "and `cellLimited Gauss linear <k>` gradients; give the entry one of those.");
        }
        return d;
    }
    refuse("fvSchemes resolves " + key + " to `" + s + "`; ported: Gauss upwind, Gauss limitedLinear <k>, Gauss "
           "linearUpwind <grad>, each optionally bounded.");
}

SolveCtl solveCtl(const FoamDict& fv, const std::string& field)
{
    const FoamDict* sol = fv.subDict("solvers");
    const FoamDict* e = sol ? sol->subDict(field) : nullptr;   // literal, else the last matching regex key
    if (!e) refuse("fvSolution solvers has no entry for " + field + " (OpenFOAM stops at its first solve).");
    if (e->wordOr("solver", "") != "smoothSolver")
        refuse("fvSolution solver for " + field + " is `" + e->wordOr("solver", "") + "`; ported: smoothSolver.");
    const std::string sm = e->wordOr("smoother", "");
    if (sm != "GaussSeidel" && sm != "symGaussSeidel")
        refuse("fvSolution smoother for " + field + " is `" + sm + "`; ported: GaussSeidel, symGaussSeidel.");
    SolveCtl c;
    c.tol = e->scalarOr("tolerance", 1e-6);
    c.relTol = e->scalarOr("relTol", 0);
    c.maxIter = e->intOr("maxIter", 1000);
    c.minIter = e->intOr("minIter", 0);
    c.nSweeps = e->intOr("nSweeps", 1);
    c.symmetric = (sm == "symGaussSeidel");
    return c;
}

std::vector<scalar> flat(GeometricField<scalar>& f, const std::vector<FvPatch>& patches)
{
    std::vector<scalar> out;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& v = f.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size; ++i)
            out.push_back(static_cast<std::size_t>(i) < v.size() ? v[static_cast<std::size_t>(i)] : scalar(0));
    }
    return out;
}

} // namespace

std::string ofSchemeStatement(const std::string& caseDir, const std::string& block, const std::string& key)
{
    std::string body = fvSchemesBlock(readFvSchemesText(caseDir), block);
    if (!body.empty() && body[0] == '{') body.erase(0, 1);
    std::string lit, pat, def;
    bool haveLit = false, havePat = false, haveDef = false;
    std::size_t a = 0;
    while (a < body.size())
    {
        const std::size_t e = body.find(';', a);
        const std::string st = body.substr(a, e == std::string::npos ? std::string::npos : e - a);
        a = (e == std::string::npos) ? body.size() : e + 1;
        std::string norm;   // the statement's words, single-spaced
        for (char ch : st)
        {
            const bool ws = (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n');
            if (ws) { if (!norm.empty() && norm.back() != ' ') norm += ' '; }
            else norm += ch;
        }
        while (!norm.empty() && norm.front() == ' ') norm.erase(0, 1);
        while (!norm.empty() && norm.back() == ' ') norm.pop_back();
        const std::size_t k = norm.find(' ');
        if (k == std::string::npos) continue;
        const std::string kw = norm.substr(0, k);
        const std::string val = norm.substr(k + 1);
        if (kw.size() > 1 && kw.front() == '"' && kw.back() == '"')
        {
            bool matches = false;
            try
            {
                matches = std::regex_match(key, compileFoamRegex(kw.substr(1, kw.size() - 2)));
            }
            catch (...)
            {
            }
            if (matches)
            {
                pat = val;   // the LAST matching pattern wins
                havePat = true;
            }
        }
        else if (kw == key)
        {
            lit = val;
            haveLit = true;
        }
        else if (kw == "default")
        {
            def = val;
            haveDef = true;
        }
    }
    if (haveLit) return lit;
    if (havePat) return pat;
    return haveDef ? def : std::string();
}

Case readCase(const std::string& caseDir, const std::string& startDir, const PrimitiveMesh& m, const FvGeometry& g,
              const std::vector<FvPatch>& patches, const GeometricField<vector>& U, bool lts,
              const FoamDict* fvSolution, const FoamDict* turbulenceProperties)
{
    Case c;
    std::string tpath = caseDir + "/constant/turbulenceProperties";
    if (!std::filesystem::exists(tpath)) tpath = caseDir + "/constant/momentumTransport";
    const FoamDict tpOwn = turbulenceProperties ? FoamDict{} : readDict(tpath);
    const FoamDict& tp = turbulenceProperties ? *turbulenceProperties : tpOwn;
    const std::string sim = tp.wordOr("simulationType", "");
    if (sim == "laminar")
    {
        const FoamDict* lam = tp.subDict("laminar");
        const std::string lm = lam ? lam->wordOr("model", lam->wordOr("laminarModel", "Stokes")) : "Stokes";
        if (lm != "Stokes")
            refuse("laminar model `" + lm + "`; ported: Stokes (laminarModel.C's default).");
        return c;
    }
    if (sim != "RAS")
        refuse("simulationType `" + sim + "`; ported: laminar and RAS (kOmegaSST). LES is not ported.");
    const FoamDict* ras = tp.subDict("RAS");
    if (!ras) refuse("simulationType RAS without a RAS dictionary.");
    // RASModel.C:140-144: getCompat<word>("model", {{"RASModel", -2006}})
    c.model = ras->found("model") ? ras->wordOr("model", "") : ras->wordOr("RASModel", "");
    if (c.model != "kOmegaSST")
        refuse("RAS model `" + c.model + "`; ported: kOmegaSST (milestone J).");
    if (!ras->switchOr("turbulence", true))
        refuse("RAS turbulence off: the model's correct() returns early and its nut stays the file's; not ported.");
    readKOmegaSSTCoeffs(ras, c.co);
    // decayControl (kOmegaSSTBase.C:449-468, setDecayControl): on, the omega equation gains beta*sqr(omegaInf) and the k
    // equation betaStar*omegaInf*kInf (:584, :609); off, OpenFOAM reads kInf and omegaInf and zeroes them, so the two
    // terms vanish. The closure carries neither term, and the switch was never read here: refused when on.
    {
        const FoamDict* kc = ras->optionalSubDict("kOmegaSSTCoeffs");
        if (kc->switchOr("decayControl", false))
            refuse("kOmegaSSTCoeffs sets `decayControl " + kc->wordOr("decayControl", "on") + "`. OpenFOAM then adds "
                   "beta*sqr(omegaInf) to the omega equation and betaStar*omegaInf*kInf to k's (kOmegaSSTBase.C:584, "
                   ":609); the closure carries neither term. Not ported yet; remove decayControl (its kInf and omegaInf "
                   "are zeroed when it is off).");
        kc->find("kInf");       // read and zeroed by OpenFOAM when decayControl is off (kOmegaSSTBase.C:463-467)
        kc->find("omegaInf");
    }
    // EddyDiffusivity::correctNut (EddyDiffusivity.C:36-38): Prt from the coeffDict, RASModel's
    // optionalSubDict(type + "Coeffs"), default 1.0
    const FoamDict* coeffs = ras->subDict("kOmegaSSTCoeffs");
    c.Prt = (coeffs ? coeffs : ras)->scalarOr("Prt", 1.0);
    c.on = true;

    // the fields, MUST_READ (kOmegaSSTBase.C's k_ and omega_, eddyViscosity's nut_, EddyDiffusivity's alphat_)
    const label nC = m.nCells();
    auto read = [&](const char* name) {
        const std::string p = startDir + "/" + name;
        if (!std::filesystem::exists(p) && !std::filesystem::exists(p + ".gz"))
            refuse(std::string("the start time directory has no ") + name + " (MUST_READ for the RAS model).");
        return readField<scalar>(p);
    };
    const FieldData<scalar> kd = read("k"), wd = read("omega"), nd = read("nut"), ad = read("alphat");
    // the patch types this port runs, field by field (OpenFOAM's own classes; the rest refused by name)
    auto check = [&](const FieldData<scalar>& fd, const char* name, const std::vector<std::string>& ok) {
        for (const FvPatch& p : patches)
        {
            const PatchFieldData<scalar>* d = findPatchEntry(fd, p);
            const std::string t = d ? d->type : (p.type == "empty" ? std::string("empty") : std::string());
            if (t.empty()) refuse(std::string(name) + " has no entry for patch '" + p.name + "'.");
            if (isConstraint(t) || isConstraint(p.type))
            {
                if (t != p.type && !(t == "slip" && p.type != "empty"))
                    refuse(std::string(name) + " patch '" + p.name + "' is `" + t + "` on a mesh patch of type `" + p.type
                           + "`.");
                continue;
            }
            bool hit = false;
            for (const std::string& o : ok) hit = hit || (t == o);
            if (!hit)
                refuse(std::string(name) + " patch '" + p.name + "' is `" + t + "`; ported here: " + [&] {
                    std::string s;
                    for (const std::string& o : ok) s += (s.empty() ? "" : ", ") + o;
                    return s;
                }() + ".");
            const bool wallFn = (t.find("WallFunction") != std::string::npos);
            if (wallFn && p.type != "wall")
                refuse(std::string(name) + " patch '" + p.name + "' is `" + t + "` on a mesh patch of type `" + p.type
                       + "` (OpenFOAM's wall functions require a wall patch: nutWallFunction::checkType).");
        }
    };
    check(kd, "k", {"fixedValue", "zeroGradient", "inletOutlet", "kqRWallFunction"});
    check(wd, "omega", {"fixedValue", "zeroGradient", "inletOutlet", "omegaWallFunction"});
    check(nd, "nut", {"calculated", "nutkWallFunction"});
    check(ad, "alphat", {"calculated", "compressible::alphatWallFunction"});
    c.k = buildField(kd, patches, nC);
    c.omega = buildField(wd, patches, nC);
    c.nut = buildField(nd, patches, nC);
    c.alphat = buildField(ad, patches, nC);
    // each patch as its dictionary constructor leaves it: fixedValue its value, zeroGradient (kqRWallFunction) the
    // cells', calculated and inletOutlet the file's value (brae's patch classes evaluate to exactly that before any
    // flux switch) -- the stored values the solver's fields are given too (rhoCentralCreateFields_cpp.cu)
    for (GeometricField<scalar>* f : {&c.k, &c.omega, &c.nut, &c.alphat})
    {
        for (auto& b : f->boundary) b->initEvaluate(f->internal);
        for (auto& b : f->boundary) b->evaluate(f->internal);
    }
    c.kB = flat(c.k, patches);
    c.omegaB = flat(c.omega, patches);
    c.nutB = flat(c.nut, patches);
    c.alphatB = flat(c.alphat, patches);

    // wall distance: wallDist::New(mesh).y() (meshWave) for F1/F2; nearWallDist for the wall functions' y
    {
        const FoamDict fs = readDict(caseDir + "/system/fvSchemes");
        const FoamDict* wdd = fs.subDict("wallDist");
        const std::string method = wdd ? wdd->wordOr("method", "meshWave") : std::string("meshWave");
        if (method != "meshWave")
            refuse("fvSchemes wallDist method `" + method + "`; ported: meshWave.");
        c.yCell = cellWallDist(m, g, patches);
        c.kDiv = divScheme(caseDir, "div(phi,k)");
        c.omegaDiv = divScheme(caseDir, "div(phi,omega)");
        if (c.kDiv.limitedLinear != c.omegaDiv.limitedLinear || c.kDiv.limiterCoeff != c.omegaDiv.limiterCoeff
            || c.kDiv.linearUpwind != c.omegaDiv.linearUpwind || c.kDiv.bounded != c.omegaDiv.bounded)
            refuse("div(phi,k) and div(phi,omega) name different schemes; the closure carries one for both.");
        if (lts && (c.kDiv.bounded))
            refuse("a bounded div(phi,k|omega) under localEuler is not ported.");
        for (const char* gk : {"grad(k)", "grad(omega)", "grad(U)"})
            if (scheme(caseDir, "gradSchemes", gk) != "Gauss linear")
                refuse(std::string("fvSchemes resolves ") + gk + " to `" + scheme(caseDir, "gradSchemes", gk) + "`; ported: "
                       "Gauss linear.");
        // THE NAMES OpenFOAM's COMPRESSIBLE kOmegaSST ASKS FOR. fvm::laplacian(alpha*rho*DkEff(F1), k_)
        // (kOmegaSSTBase.C:576, :603) names its term after the diffusivity, and with alpha the geometricOneField of a
        // compressible model alpha*rho is rho itself (geometricOneFieldI.H), so the keys are laplacian((rho*DkEff),k)
        // and laplacian((rho*DomegaEff),omega). These were checked under the incompressible names DkEff and DomegaEff,
        // so an explicit entry under the real key was never seen and `default` decided both.
        std::string lapK, lapW;
        for (const char* lk : {"laplacian((rho*DkEff),k)", "laplacian((rho*DomegaEff),omega)"})
        {
            const std::string s = scheme(caseDir, "laplacianSchemes", lk);
            if (s != "Gauss linear corrected" && s != "Gauss linear orthogonal")
                refuse(std::string("fvSchemes resolves ") + lk + " to `" + s + "`; ported: Gauss linear corrected and "
                       "Gauss linear orthogonal.");
            (lapK.empty() ? lapK : lapW) = s;
        }
        if (lapK != lapW)
            refuse("fvSchemes resolves laplacian((rho*DkEff),k) to `" + lapK + "` and laplacian((rho*DomegaEff),omega) to `"
                   + lapW + "`; the closure carries one laplacian scheme for both equations.");
        c.correctedLaplacian = (lapK == "Gauss linear corrected");
        // fvm::ddt(alpha, rho, k_) and (..., omega_) (kOmegaSSTBase.C:574, :601): with a geometricOneField alpha the
        // names are ddt(rho,k) and ddt(rho,omega) (fvmDdt.C). Never checked: the closure runs the solver's own scheme,
        // Euler or, under LTS, localEuler -- refused when either names another, as the solver's own terms are.
        for (const char* dk : {"ddt(rho,k)", "ddt(rho,omega)"})
        {
            const std::string s = scheme(caseDir, "ddtSchemes", dk);
            if (s != (lts ? "localEuler" : "Euler"))
                refuse(std::string("fvSchemes resolves ") + dk + " to `" + s + "`; the closure runs the solver's " +
                       (lts ? "localEuler" : "Euler") + " here and no other scheme is ported for it.");
        }
        // compressibleTurbulenceModel::phi() = phi_/fvc::interpolate(rho) (compressibleTurbulenceModel.C:61-70), the
        // flux the closure convects with: `interpolate(rho)`, linear on the device (deviceInterpolate); never checked
        if (scheme(caseDir, "interpolationSchemes", "interpolate(rho)") != "linear")
            refuse("fvSchemes resolves interpolate(rho) to `" + scheme(caseDir, "interpolationSchemes", "interpolate(rho)")
                   + "`; the turbulence model's volumetric flux phi/interpolate(rho) is ported with `linear`.");
    }
    {
        const FoamDict fvOwn = fvSolution ? FoamDict{} : readDict(caseDir + "/system/fvSolution");
        const FoamDict& fv = fvSolution ? *fvSolution : fvOwn;
        c.kSolve = solveCtl(fv, "k");
        c.omegaSolve = solveCtl(fv, "omega");
        // relaxationFactors, as solution::read and relaxEquation take them (solution.C:62-136, :379-418): the
        // `equations` sub-dictionary, or -- when neither `fields` nor `equations` is there -- the block itself (the
        // backward-compatible flat form); a factor for the name (literal, else the last matching pattern), else
        // `default`, else no relaxation at all. Never read here: both equations ran unrelaxed whatever the case said.
        const FoamDict* rf = fv.subDict("relaxationFactors");
        const FoamDict* eq = rf ? rf->subDict("equations") : nullptr;
        const FoamDict* fl = rf ? rf->subDict("fields") : nullptr;
        const FoamDict* eqns = (eq || fl) ? eq : rf;
        auto factor = [&](const char* name, bool& on, scalar& value) {
            if (!eqns) return;
            const std::vector<std::string>* v = eqns->find(name);
            const char* which = name;
            if (!v)
            {
                v = eqns->find("default");
                which = "default";
            }
            if (!v) return;
            // a Function1 of time (solution.C:385-392); `constant <v>` and a plain number are the constant ones
            const bool plain = v->size() == 1;
            const bool constant = v->size() == 2 && (*v)[0] == "constant";
            char* end = nullptr;
            const std::string& num = plain ? (*v)[0] : (constant ? (*v)[1] : std::string());
            const double x = num.empty() ? 0.0 : std::strtod(num.c_str(), &end);
            if (num.empty() || !end || *end != '\0')
                refuse(std::string("fvSolution relaxationFactors gives `") + which + "` (the factor for " + name
                       + ") a value that is not a constant; OpenFOAM evaluates it as a Function1 of time "
                       "(solution.C:385-392), and only a constant factor is ported here.");
            on = true;
            value = static_cast<scalar>(x);
        };
        factor("k", c.relaxK, c.relaxKFactor);
        factor("omega", c.relaxOmega, c.relaxOmegaFactor);
    }

    // the per-face masks, flat in patch order (no coupled patch reaches here)
    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, patches);
    c.wfPatch.assign(patches.size(), 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        c.wfPatch[pi] = c.omega.boundary[pi]->isTurbulenceWallFunction() && patches[pi].type == "wall" ? 1 : 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        const PatchFieldData<scalar>* nde = findPatchEntry(nd, p);
        const PatchFieldData<scalar>* ade = findPatchEntry(ad, p);
        const std::string nt = nde ? nde->type : p.type;
        const std::string at = ade ? ade->type : p.type;
        const bool wf = c.wfPatch[pi] != 0;
        // kOmegaSSTBase's F1 is 1 on wall and empty patches (keyed on the mesh patch type, as wallDist is)
        const bool f1One = (p.type == "wall" || p.type == "empty");
        const WallFunctionCoeffs& nc = c.nut.boundary[pi]->wallCoeffs();
        // EddyDiffusivity::correctNut is a field assignment: a calculated alphat takes rho*nut/Prt with the model's
        // Prt; compressible::alphatWallFunction its own Prt (alphatWallFunctionFvPatchScalarField.C:125)
        const bool alphatWF = (at == "compressible::alphatWallFunction");
        const scalar aPrt = alphatWF ? (ade->Prt > 0 ? ade->Prt : scalar(0.85)) : c.Prt;
        const bool alphatAssign = alphatWF || at == "calculated";
        for (label i = 0; i < p.size; ++i)
        {
            c.wfMask.push_back(wf ? 1 : 0);
            c.f1One.push_back(f1One ? 1 : 0);
            c.yBnd.push_back(wf ? yW[pi][static_cast<std::size_t>(i)] : scalar(0));
            c.nutCalc.push_back(nt == "calculated" ? 1 : 0);
            c.nutKind.push_back(0);   // NutWall::Nutk, the one ported
            c.nutCmu25.push_back(std::pow(nc.Cmu, 0.25));
            c.nutKappa.push_back(nc.kappa);
            c.nutE.push_back(nc.E);
            c.nutYplLam.push_back(nc.yPlusLam());
            c.alphatMask.push_back(alphatAssign ? 1 : 0);
            c.alphatPrt.push_back(aPrt);
            c.constraintFace.push_back(isConstraint(nt) || isConstraint(p.type) ? 1 : 0);
            c.bCell.push_back(p.faceCells[static_cast<std::size_t>(i)]);
        }
    }
    (void)U;
    return c;
}

} // namespace compressibleRAS
} // namespace brae
