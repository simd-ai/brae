// rhoCentralControls.cu -- the case: controls, mesh, and every start-up refusal. See the header.
#include "rhoCentralControls.cuh"

#include "foam_constants.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "geometric_field.cuh"
#include "scheme_parse.cuh"
#include "start_time.cuh"
#include "wedge_patch.cuh"
#include "device_mesh.cuh"   // the limiter selectors (kVanLeerTwoByk and its kin)

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace rhoCentral {

namespace {

constexpr scalar kSMALL = 1e-15;         // OpenFOAM's SMALL and ROOTVSMALL in double precision
constexpr scalar kROOTVSMALL = 1e-150;

[[noreturn]] void refuse(const std::string& what)
{
    throw std::runtime_error("brae rhoCentralFoam: " + what);
}

// The scheme statement OpenFOAM resolves for `key` in an fvSchemes block: the entry whose keyword IS key, else
// `default` (schemesLookup). Read from the block's text, statement by statement: a keyword like reconstruct(rho)
// carries parentheses, and the comparison is on the whole keyword, so reconstruct(rho) never answers for
// reconstruct(rhoU).
std::string schemeStatement(const std::string& caseDir, const std::string& block, const std::string& key)
{
    std::string body = fvSchemesBlock(readFvSchemesText(caseDir), block);
    if (!body.empty() && body[0] == '{') body.erase(0, 1);
    std::string hit, def;
    std::size_t a = 0;
    while (a < body.size())
    {
        const std::size_t e = body.find(';', a);
        std::string st = body.substr(a, e == std::string::npos ? std::string::npos : e - a);
        a = (e == std::string::npos) ? body.size() : e + 1;
        const std::size_t b = st.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        st = st.substr(b);
        const std::size_t k = st.find_first_of(" \t\r\n");
        if (k == std::string::npos) continue;
        std::string kw = st.substr(0, k);
        if (kw.size() > 1 && kw.front() == '"' && kw.back() == '"') kw = kw.substr(1, kw.size() - 2);
        std::string val = st.substr(k);
        const std::size_t vb = val.find_first_not_of(" \t\r\n");
        const std::size_t ve = val.find_last_not_of(" \t\r\n");
        val = (vb == std::string::npos) ? std::string() : val.substr(vb, ve - vb + 1);
        std::string norm;   // collapse runs of white space
        for (char ch : val)
        {
            const bool ws = (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n');
            if (ws) { if (!norm.empty() && norm.back() != ' ') norm += ' '; }
            else norm += ch;
        }
        if (kw == key) hit = norm;
        else if (kw == "default") def = norm;
    }
    return hit.empty() ? def : hit;
}

// interpolate(vf, dir, "reconstruct(...)") (directionInterpolate.H) through a limitedSurfaceInterpolationScheme: the
// limiter selector device_limiter.cuh's limiterOfR takes. Scalars (rho; T's rPsi, e, c) take the NVDTVD forms, rhoU the
// NVDVTVDV `V` forms (LimitedScheme.H, makeLimitedVSurfaceInterpolationScheme); `upwind` is upwind.H's pos0 weights,
// limiter 0, on either. A scalar-form limiter named for U is LimitedScheme's limitFuncs::magSqr instantiation -- the
// limiter built from magSqr(rhoU) and its gradient -- which is not ported and is refused by name, as is every other
// scheme; limitedLinear's coefficient outside [0, 1] stops OpenFOAM (limitedLinear.H) and is refused the same way.
scalar reconstructLimiter(const std::string& stmt, const std::string& key, bool vectorField)
{
    std::string name = stmt, arg;
    const std::size_t sp = stmt.find(' ');
    if (sp != std::string::npos)
    {
        name = stmt.substr(0, sp);
        arg = stmt.substr(sp + 1);
    }
    std::string base = name;
    const bool vForm = name.size() > 1 && name.back() == 'V';
    if (vForm) base = name.substr(0, name.size() - 1);
    const char* known[] = {"vanLeer", "vanAlbada", "Minmod", "MUSCL", "SuperBee", "limitedLinear"};
    bool isKnown = false;
    for (const char* k : known) isKnown = isKnown || base == k;
    if (name == "upwind")
    {
        if (!arg.empty()) refuse("interpolationSchemes " + key + " is `" + stmt + "`: `upwind` takes no coefficient.");
        return kUpwindTwoByk;
    }
    if (!isKnown)
        refuse("interpolationSchemes resolves " + key + " to `" + stmt + "`; this stage ports vanLeer, vanAlbada, "
               "Minmod, MUSCL, SuperBee and limitedLinear (and their V forms on U) and upwind "
               "(src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes). Refusing rather than reconstruct "
               "with another scheme.");
    if (vectorField && !vForm)
        refuse("interpolationSchemes resolves " + key + " to `" + stmt + "`: on a vector, OpenFOAM's scalar-form "
               "limiter is built from magSqr(rhoU) and its gradient (LimitedScheme.H, limitFuncs::magSqr), which is not "
               "ported; the `" + base + "V` form (NVDVTVDV) is.");
    if (!vectorField && vForm)
        refuse("interpolationSchemes resolves " + key + " to `" + stmt + "`: a V scheme is defined for vectors "
               "only (makeLimitedVSurfaceInterpolationScheme); OpenFOAM refuses it on a scalar too.");
    if (base == "limitedLinear")
    {
        char* end = nullptr;
        const double k = arg.empty() ? -1.0 : std::strtod(arg.c_str(), &end);
        if (arg.empty() || *end != '\0')
            refuse("interpolationSchemes " + key + " is `" + stmt + "`: limitedLinear needs its coefficient k.");
        if (k < 0 || k > 1)
            refuse("interpolationSchemes " + key + " is `" + stmt + "`: limitedLinear's coefficient " + arg
                   + " should be >= 0 and <= 1 (limitedLinear.H, where OpenFOAM stops with the same FatalIOError).");
        return 2.0/std::max(static_cast<scalar>(k), static_cast<scalar>(1e-15));   // twoByk_ = 2.0/max(k_, SMALL)
    }
    if (!arg.empty())
        refuse("interpolationSchemes " + key + " is `" + stmt + "`: `" + name + "` takes no coefficient.");
    if (base == "vanLeer") return kVanLeerTwoByk;
    if (base == "Minmod") return kMinmodTwoByk;
    if (base == "MUSCL") return kMUSCLTwoByk;
    if (base == "SuperBee") return kSuperBeeTwoByk;
    return 0.0;   // vanAlbada
}

// The constraint patch types this stage runs: the mesh patch's type IS the patch field's (fvPatchFieldNew.C:
// a constraint patch field is built only on its own patch type, and every field the solver derives -- rho, rhoU,
// rhoE, e -- takes the constraint type there, calculatedFvPatchField or not).
bool isTransformConstraint(const std::string& t) { return t == "symmetryPlane" || t == "symmetry" || t == "wedge"; }

// Every patch of one field must name a patch-field type this stage runs; anything else is refused here, before a
// field is built. The evaluation itself is brae's patch-field classes (fv_patch_field.cuh) through PatchFields.
// Returns the types, patch by patch.
template <typename T>
std::vector<std::string> checkPatchFieldTypes(const FieldData<T>& fd, const std::vector<FvPatch>& patches,
                                              const std::string& field)
{
    std::vector<std::string> types;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        const PatchFieldData<T>* d = findPatchEntry(fd, p);
        const std::string type = d ? d->type : (p.type == "empty" ? std::string("empty") : std::string());
        if (type.empty())
            refuse("no boundaryField entry for patch '" + p.name + "' in " + field + ".");
        if (type == "empty" || isTransformConstraint(type) || isTransformConstraint(p.type) || p.type == "empty")
        {
            // OpenFOAM: symmetryPlaneFvPatchField.C:44-58 (and symmetry's, wedge's, empty's) refuse a constraint
            // field on another patch type; GeometricBoundaryField refuses a non-constraint field on a constraint patch
            if (type != p.type)
                refuse(field + " patch '" + p.name + "' is `" + type + "` on a mesh patch of type `" + p.type
                       + "`; OpenFOAM requires the two to be the same for empty, symmetryPlane, symmetry and wedge.");
        }
        else if (type != "zeroGradient" && type != "fixedValue" && type != "slip" && type != "inletOutlet")
            refuse(field + " patch '" + p.name + "' is `" + type + "`. This stage runs zeroGradient, fixedValue, "
                   "slip, inletOutlet, symmetryPlane, symmetry, wedge and empty on U, p and T; totalPressure, "
                   "totalTemperature, waveTransmissive, partialSlip, outletInlet, maxwellSlipU, smoluchowskiJumpT and "
                   "the rest are not ported (applications/solvers/compressible/rhoCentralFoam/BCs, "
                   "src/finiteVolume/fields/fvPatchFields). Refusing rather than run the patch as something else.");
        if (type == "inletOutlet")
        {
            // inletOutletFvPatchField.C:59-83: refValue from the mandatory `inletValue`; the switch reads the
            // surfaceScalarField its `phi` entry names (default phi), the mass flux of rhoCentralFoam.C:195. Any other
            // name would look up another registered field (phiEp is one during the energy solve): not ported.
            if (!d->hasInletValue)
                refuse(field + " patch '" + p.name + "' is `inletOutlet` without `inletValue`; OpenFOAM requires it "
                       "(inletOutletFvPatchField.C, Field<Type>(\"inletValue\", dict, p.size())).");
            if (d->phiName != "phi")
                refuse(field + " patch '" + p.name + "' is `inletOutlet` on `phi " + d->phiName + "`. This stage "
                       "switches inletOutlet on rhoCentralFoam's mass flux `phi` (rhoCentralFoam.C:195) only.");
        }
        types.push_back(type);
    }
    return types;
}

// R = I - 2.0*sqr(nHat), as basicSymmetryFvPatchField.C:106-115 and symmetryPlaneFvPatchField.C:118-131 build it
// for (iF + transform(R, iF))/2.0: sqr(n) = n*n, 2.0 times each component, I minus that.
tensor mirrorTensor(const vector& n)
{
    return tensor{1.0 - 2.0*(n.x*n.x), -(2.0*(n.x*n.y)), -(2.0*(n.x*n.z)),
                  -(2.0*(n.y*n.x)), 1.0 - 2.0*(n.y*n.y), -(2.0*(n.y*n.z)),
                  -(2.0*(n.z*n.x)), -(2.0*(n.z*n.y)), 1.0 - 2.0*(n.z*n.z)};
}

// The start directory, as OpenFOAM would read it. Time::setControls (Time.C:160-183) takes startFrom latestTime's or
// firstTime's directory from findTimes: every directory whose NAME reads as a number, whatever it holds. The shared
// resolveStartTime (start_time.cuh) skips one without U, which other brae solvers rely on; here a newer (or, for
// firstTime, older) directory it skipped is refused by name, since OpenFOAM would pick it and stop reading its fields.
// Then p, T and U must all be in the start directory: OpenFOAM's MUST_READ stops on any missing one.
void checkStartDirectory(const std::string& caseDir, const FoamDict& controlDict, const StartTime& st)
{
    namespace fs = std::filesystem;
    const std::string startFrom = controlDict.wordOr("startFrom", "latestTime");
    if (startFrom == "latestTime" || startFrom == "firstTime")
    {
        std::error_code ec;
        bool any = false;
        double best = 0;
        std::string bestName;
        for (const auto& e : fs::directory_iterator(caseDir, ec))
        {
            if (!e.is_directory()) continue;
            const std::string nm = e.path().filename().string();
            char* end = nullptr;
            const double t = std::strtod(nm.c_str(), &end);
            if (end == nm.c_str() || *end != '\0') continue;   // constant, system, 0.orig: not a time
            if (!any || (startFrom == "latestTime" ? t > best : t < best))
            {
                best = t;
                bestName = nm;
            }
            any = true;
        }
        if (any && best != st.value)
        {
            std::string missing;
            for (const char* f : {"U", "p", "T"})
                if (!fs::exists(caseDir + "/" + bestName + "/" + f) && !fs::exists(caseDir + "/" + bestName + "/" + f + ".gz"))
                    missing += std::string(missing.empty() ? "" : ", ") + f;
            refuse("startFrom " + startFrom + " selects time directory " + bestName + " (Time::setControls takes it by "
                   "its name, Time.C:160-183), and it has no " + (missing.empty() ? std::string("readable fields")
                   : missing) + ". OpenFOAM stops reading it; brae refuses rather than start from " + st.name
                   + " instead.");
        }
    }
    std::string missing;
    for (const char* f : {"U", "p", "T"})
        if (!fs::exists(caseDir + "/" + st.name + "/" + f) && !fs::exists(caseDir + "/" + st.name + "/" + f + ".gz"))
            missing += std::string(missing.empty() ? "" : ", ") + f;
    if (!missing.empty())
        refuse("the start time directory " + st.name + " has no " + missing + "; rhoCentralFoam reads U, p and T "
               "(MUST_READ: createFields.H and the thermo), and OpenFOAM stops on a missing one.");
}

} // namespace


// ---- the viscous correctors' controls (milestone G) -----------------------------------------------------------------
// fvSolution: U's and e's solves (rhoCentralFoam.C:236-243, 274-277), smoothSolver with GaussSeidel or symGaussSeidel,
// the controls read with OpenFOAM's defaults. fvSchemes: what the viscous terms resolve to -- grad(U) (tauMC), div(tauMC),
// laplacian(muEff,U), laplacian(alphahe,e), snGrad(U) (sigmaDotU), interpolate(muEff) -- refused unless it is the
// statement ported. Anything else is refused by name before a field is read.
static void readViscousControls(const std::string& caseDir, Case& cs)
{
    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* solvers = fvSolution.subDict("solvers");
    auto solveControls = [&](const char* fld, SmoothSolveControls& c) {
        const FoamDict* sd = solvers ? solvers->subDict(fld) : nullptr;
        if (!sd)
            refuse(std::string("fvSolution solvers has no entry for ") + fld + ". With mu > 0 rhoCentralFoam solves "
                   + fld + " implicitly (rhoCentralFoam.C:" + (fld[0] == 'U' ? "236-243" : "274-277") + "); OpenFOAM "
                   "v2412 stops at its first " + fld + " solve (`Entry '" + fld + "' not found in dictionary "
                   "system/fvSolution/solvers`). The tutorials' entry is `h`, for an energy they do not solve.");
        const std::string solver = sd->wordOr("solver", "");
        if (solver != "smoothSolver")
            refuse(std::string("fvSolution solvers has `") + solver + "` for " + fld + "; this stage ports "
                   "smoothSolver (with GaussSeidel or symGaussSeidel), in OpenFOAM's sequential sweep order.");
        c.smoother = sd->wordOr("smoother", "");
        if (c.smoother != "GaussSeidel" && c.smoother != "symGaussSeidel")
            refuse(std::string("fvSolution smoother `") + c.smoother + "` for " + fld + "; this stage ports "
                   "GaussSeidel and symGaussSeidel.");
        c.symmetric = (c.smoother == "symGaussSeidel");
        c.tolerance = sd->scalarOr("tolerance", 1e-6);   // lduMatrixSolver.C readControls
        c.relTol = sd->scalarOr("relTol", 0);
        c.maxIter = static_cast<int>(sd->scalarOr("maxIter", 1000));
        c.minIter = static_cast<int>(sd->scalarOr("minIter", 0));
        c.nSweeps = static_cast<int>(sd->scalarOr("nSweeps", 1));   // smoothSolver.C readControls
        if (c.nSweeps < 1)
            refuse(std::string("fvSolution nSweeps ") + std::to_string(c.nSweeps) + " for " + fld + "; the fixed "
                   "sweep count without a residual (nSweeps < 0) is not ported.");
    };
    solveControls("U", cs.solveU);
    solveControls("e", cs.solveE);

    const FieldGradScheme gs = parseFieldGradScheme(caseDir, "U");
    if (!gs.gaussLinear || gs.leastSquares || gs.cellLimitK != 0 || !gs.unsupportedLimiter.empty())
        refuse("gradSchemes resolves grad(U) to `" + gs.raw + "`. tauMC = muEff*dev2(T(fvc::grad(U))) "
               "(rhoCentralFoam.C:218-219); this stage computes Gauss linear only.");
    const struct { const char* block; const char* key; const char* want; const char* what; } need[] = {
        {"divSchemes", "div(tauMC)", "Gauss linear", "fvc::div(tauMC), rhoCentralFoam.C:240"},
        {"laplacianSchemes", "laplacian(muEff,U)", "Gauss linear corrected", "fvm::laplacian(muEff, U), :239"},
        {"laplacianSchemes", "laplacian(alphahe,e)", "Gauss linear corrected", "fvm::laplacian(alphaEff, e), :275 (the field is thermo.alphahe(), heThermo.C:812)"},
        {"snGradSchemes", "snGrad(U)", "corrected", "fvc::snGrad(U) in sigmaDotU, :249"},
        {"interpolationSchemes", "interpolate(muEff)", "linear", "fvc::interpolate(muEff) in sigmaDotU, :249"},
    };
    for (const auto& n : need)
    {
        const std::string st = schemeStatement(caseDir, n.block, n.key);
        if (st != n.want)
            refuse(std::string(n.block) + " resolves " + n.key + " to `" + st + "` (" + n.what + "); this stage "
                   "ports `" + n.want + "` for the viscous terms.");
    }
}

Case readRhoCentralCase(const std::string& caseDir)
{
    Case cs;
    cs.caseDir = caseDir;

    // ---- controlDict: the clock (Time::setControls, readTimeControls.H) -----------------------------------------
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const std::string app = controlDict.wordOr("application", "rhoCentralFoam");
    if (app != "rhoCentralFoam")
        refuse("controlDict names application `" + app + "`; this executable runs rhoCentralFoam cases.");
    if (!(controlDict.scalarOr("deltaT", 0) > 0)) refuse("controlDict deltaT must be positive.");
    // purgeWrite: all but the last N written time directories removed after each write (Time::writeObject,
    // purge_write.cuh); the driver holds the FIFO
    cs.purgeWrite = controlDict.intOr("purgeWrite", 0);
    cs.timeControls = TimeControls::read(controlDict);   // createTimeControls.H: adjustTimeStep, maxCo, maxDeltaT
    cs.writePrecision = controlDict.intOr("writePrecision", 6);
    const StartTime st = resolveStartTime(caseDir, controlDict);
    checkStartDirectory(caseDir, controlDict, st);
    cs.startName = st.name;
    cs.startTime = static_cast<scalar>(st.value);
    try
    {
        cs.clock.emplace(controlDict, caseDir + "/" + cs.startName, cs.startTime);
    }
    catch (const std::exception& e)
    {
        refuse(e.what());
    }
    if (!cs.clock->run())
        refuse("endTime is not beyond the start time " + cs.startName + ": nothing to run.");

    // ---- static mesh only ----------------------------------------------------------------------------------------
    if (std::filesystem::exists(caseDir + "/constant/dynamicMeshDict"))
    {
        const FoamDict dm = readDict(caseDir + "/constant/dynamicMeshDict");
        const std::string t = dm.wordOr("dynamicFvMesh", "staticFvMesh");
        if (t != "staticFvMesh")
            refuse("constant/dynamicMeshDict selects `" + t + "`. Mesh motion (rhoCentralFoam.C:95 mesh.update(), "
                   "the meshPhi terms at :126-132 and :212-217) is not ported; this stage runs a static mesh.");
    }

    // ---- fvSchemes ------------------------------------------------------------------------------------------------
    const FoamDict fvSchemes = readDict(caseDir + "/system/fvSchemes");
    const std::string flux = fvSchemes.wordOr("fluxScheme", "Kurganov");   // readFluxScheme.H
    if (flux != "Kurganov" && flux != "Tadmor")
        refuse("fvSchemes fluxScheme `" + flux + "` is not a valid choice; options are Tadmor, Kurganov "
               "(readFluxScheme.H, where OpenFOAM stops with the same FatalError).");
    cs.tadmor = (flux == "Tadmor");

    const std::string ddt = schemeStatement(caseDir, "ddtSchemes", "ddt(rho)");
    if (ddt != "Euler")
        refuse("ddtSchemes resolves to `" + ddt + "`. This stage runs Euler only: localEuler (LTS, setRDeltaT.H) "
               "and the other ddt schemes are not ported.");

    const std::string rRho = schemeStatement(caseDir, "interpolationSchemes", "reconstruct(rho)");
    const std::string rU   = schemeStatement(caseDir, "interpolationSchemes", "reconstruct(U)");
    const std::string rT   = schemeStatement(caseDir, "interpolationSchemes", "reconstruct(T)");
    cs.limRho = reconstructLimiter(rRho, "reconstruct(rho)", false);
    cs.limT = reconstructLimiter(rT, "reconstruct(T)", false);   // it reconstructs rPsi, e and c
    cs.limU = reconstructLimiter(rU, "reconstruct(U)", true);    // it reconstructs rhoU
    cs.reconRho = rRho;
    cs.reconT = rT;
    cs.reconU = rU;
    // the limiters' gradients: LimitedScheme.C:56-59 builds them with fvc::grad of the reconstructed field itself
    for (const char* fld : {"rho", "rhoU", "rPsi", "e", "c"})
    {
        const FieldGradScheme gs = parseFieldGradScheme(caseDir, fld);
        if (!gs.gaussLinear || gs.leastSquares || gs.cellLimitK != 0 || !gs.unsupportedLimiter.empty())
        {
            // the statement without its `default` keyword, white space collapsed
            std::string raw;
            for (char ch : gs.raw)
            {
                const bool ws = (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n');
                if (ws) { if (!raw.empty() && raw.back() != ' ') raw += ' '; }
                else raw += ch;
            }
            const bool viaDefault = (raw.rfind("default ", 0) == 0);
            if (viaDefault) raw = raw.substr(8);
            refuse(std::string("gradSchemes resolves grad(") + fld + ") to `" + raw + "`"
                   + (viaDefault ? " (its default)" : "") + ". The vanLeer limiter reads that gradient "
                   "(LimitedScheme.C:56-59); this stage computes Gauss linear only.");
        }
    }

    // ---- fvSolution: the three diagonal solves --------------------------------------------------------------------
    {
        const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
        const FoamDict* solvers = fvSolution.subDict("solvers");
        for (const char* fld : {"rho", "rhoU", "rhoE"})
        {
            const FoamDict* sd = solvers ? solvers->subDict(fld) : nullptr;
            const std::string s = sd ? sd->wordOr("solver", "") : "";
            if (s != "diagonal")
                refuse(std::string("fvSolution solvers has `") + s + "` for " + fld + ". The equation is a "
                       "diagonal matrix (fvm::ddt + an explicit divergence) and this stage solves it exactly as "
                       "OpenFOAM's diagonalSolver does (x = source/diag); every v2412 tutorial names `diagonal`.");
        }
    }

    // ---- thermophysicalProperties ---------------------------------------------------------------------------------
    {
        const FoamDict tp = readDict(caseDir + "/constant/thermophysicalProperties");
        const FoamDict* tt = tp.subDict("thermoType");
        if (!tt) refuse("constant/thermophysicalProperties has no thermoType.");
        const struct { const char* key; const char* want; const char* why; } need[] = {
            {"type", "hePsiThermo", "rhoCentralFoam builds a psiThermo (createFields.H); hePsiThermo is the one ported"},
            {"mixture", "pureMixture", "multi-species mixtures are not ported"},
            {"transport", "const", "sutherland's mu(T) is not ported; the viscous correctors run with a constant mu"},
            {"thermo", "hConst", "janaf (biconic25-55Run35) and eConst are not ported"},
            {"equationOfState", "perfectGas", "the psi = 1/(R T) equation of state is the one ported"},
            {"specie", "specie", "the specie model"},
            {"energy", "sensibleInternalEnergy", "rhoCentralFoam transports e (createFields.H: e = thermo.he())"},
        };
        for (const auto& n : need)
        {
            const std::string v = tt->wordOr(n.key, "");
            if (v != n.want)
                refuse(std::string("thermoType ") + n.key + " is `" + v + "`; this stage ports `" + n.want + "` ("
                       + n.why + ").");
        }
        const FoamDict* mix = tp.subDict("mixture");
        const FoamDict* sp  = mix ? mix->subDict("specie") : nullptr;
        const FoamDict* th  = mix ? mix->subDict("thermodynamics") : nullptr;
        const FoamDict* tr  = mix ? mix->subDict("transport") : nullptr;
        if (!sp || !th || !tr) refuse("thermophysicalProperties mixture needs specie, thermodynamics and transport.");
        const scalar W = sp->scalarOr("molWeight", 0);
        if (!(W > 0)) refuse("mixture.specie.molWeight must be positive.");
        cs.gas.R     = foamRR()/W;   // specie::R() = RR/molWeight_ (specieI.H:87-90)
        cs.gas.Cp    = th->scalarOr("Cp", 0);
        cs.gas.Tref  = th->scalarOr("Tref", foamTstd());   // hConstThermo.C:40
        cs.gas.Hsref = th->scalarOr("Href", 0);            // hConstThermo.C:41 reads Hsref_ from `Href`
        if (!(cs.gas.Cp > cs.gas.R)) refuse("mixture Cp must exceed R (Cv = Cp - R must be positive).");
        if (!tr->found("mu") || !tr->found("Pr"))
            refuse("thermophysicalProperties mixture.transport needs mu and Pr (constTransport.C:37-38 reads both).");
        const scalar mu = tr->scalarOr("mu", -1);
        const scalar Pr = tr->scalarOr("Pr", 0);
        if (mu < 0) refuse("transport mu is " + std::to_string(mu) + ": a negative viscosity.");
        cs.gas.mu = mu;
        cs.gas.rPr = 1.0/Pr;   // constTransport.C:38, rPr_(1.0/Pr), read whatever mu is
        // createFieldRefs.H: inviscid unless max(thermo.mu()) > 0. Viscous: the implicit U and e correctors and
        // sigmaDotU (rhoCentralFoam.C:234-244, 246-256, 273-282), milestone G
        cs.viscous = mu > 0;
        if (cs.viscous && !(Pr > 0)) refuse("transport Pr is " + std::to_string(Pr) + "; alphah = mu/Pr needs Pr > 0.");
    }
    if (cs.viscous) readViscousControls(caseDir, cs);

    // ---- turbulence: laminar ----------------------------------------------------------------------------------------
    {
        std::string tpath = caseDir + "/constant/turbulenceProperties";
        if (!std::filesystem::exists(tpath)) tpath = caseDir + "/constant/momentumTransport";
        if (!std::filesystem::exists(tpath)) refuse("no constant/turbulenceProperties (or momentumTransport).");
        const FoamDict t = readDict(tpath);
        const std::string sim = t.wordOr("simulationType", "");
        if (sim != "laminar")
            refuse("simulationType is `" + sim + "`; this stage runs laminar (inviscid) only. Compressible "
                   "turbulence on the central path is a later step.");
        if (const FoamDict* lam = t.subDict("laminar"))
        {
            const std::string model = lam->wordOr("model", lam->wordOr("laminarModel", "Stokes"));
            if (model != "Stokes")
                refuse("laminar model `" + model + "` is not ported; Stokes (the default) is.");
        }
    }

    // ---- mesh -----------------------------------------------------------------------------------------------------
    cs.m.read(caseDir + "/constant/polyMesh");
    cs.g.build(cs.m);
    cs.patches = buildPatches(cs.m, cs.g);
    for (const FvPatch& p : cs.patches)
        if (p.type != "patch" && p.type != "wall" && p.type != "empty" && !isTransformConstraint(p.type))
            refuse("mesh patch '" + p.name + "' is of type `" + p.type + "`. This stage runs patch, wall, empty, "
                   "symmetryPlane, symmetry and wedge patches; cyclic, cyclicAMI and processor patches are not "
                   "ported.");
    cs.sd  = solutionDirections(cs.patches);
    cs.nC  = cs.m.nCells();
    cs.nIF = cs.m.nInternalFaces();
    for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
    {
        const FvPatch& p = cs.patches[pi];
        if (p.start != cs.nIF + cs.nB)
            refuse("patch '" + p.name + "' does not start where the previous one ended.");
        for (label i = 0; i < p.size; ++i)
        {
            cs.bCell.push_back(p.faceCells[i]);
            cs.bPatch.push_back(static_cast<int>(pi));
            cs.bEmpty.push_back(p.type == "empty" ? 1 : 0);
        }
        cs.nB += p.size;
    }

    // ---- patch-field types ---------------------------------------------------------------------------------------------
    const std::string dir = caseDir + "/" + cs.startName;
    const std::vector<std::string> Utypes = checkPatchFieldTypes(readField<vector>(dir + "/U"), cs.patches, "U");
    const std::vector<std::string> ptypes = checkPatchFieldTypes(readField<scalar>(dir + "/p"), cs.patches, "p");
    const std::vector<std::string> Ttypes = checkPatchFieldTypes(readField<scalar>(dir + "/T"), cs.patches, "T");

    // ---- the transform patches, face by face (rhoCentralPatchUpdate.cuh) -----------------------------------------------
    cs.bRhoConstraint.assign(cs.nB, 0);
    cs.bTseedOwn.assign(cs.nB, 0);
    cs.bUTransform.assign(cs.nB, transformNone);
    cs.bUR.assign(cs.nB, tensor{1, 0, 0, 0, 1, 0, 0, 0, 1});
    cs.bInletOutlet.assign(cs.nB, 0);
    label k0 = 0;
    for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
    {
        const FvPatch& p = cs.patches[pi];
        const std::string& Ut = Utypes[pi];
        // symmetryPlanePolyPatch::calcGeometry (symmetryPlanePolyPatch.C:48-89): ONE normal, the area-weighted
        // gSum(faceAreas).normalise(ROOTVSMALL), and a patch whose face normal leaves it by magSqr > SMALL refused
        vector nPlane{0, 0, 0};
        if (p.type == "symmetryPlane" && p.size)
        {
            const std::vector<vector>& Sf = cs.g.Sf();
            vector s{0, 0, 0};
            for (label i = 0; i < p.size; ++i) s = s + Sf[p.start + i];
            const scalar ms = mag(s);
            nPlane = (ms < kROOTVSMALL) ? vector{0, 0, 0} : vector{s.x/ms, s.y/ms, s.z/ms};
            for (label i = 0; i < p.size; ++i)
            {
                const vector& a = Sf[p.start + i];
                const scalar ma = mag(a);
                if (ma > kROOTVSMALL && magSqr(nPlane - vector{a.x/ma, a.y/ma, a.z/ma}) > kSMALL)
                    refuse("symmetryPlane patch '" + p.name + "' is not planar (symmetryPlanePolyPatch.C:68-87); "
                           "OpenFOAM refuses it too and asks for a `symmetry` patch.");
            }
        }
        tensor faceT{1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (p.type == "wedge" && p.size)
        {
            try
            {
                faceT = wedgeGeometry(p).faceT;   // wedgePolyPatch::calcGeometry, wedge_patch.cuh
            }
            catch (const std::exception& e)
            {
                refuse(e.what());
            }
        }
        const bool io[3] = {Ut == "inletOutlet", ptypes[pi] == "inletOutlet", Ttypes[pi] == "inletOutlet"};
        cs.ioU = cs.ioU || io[0];
        cs.ioP = cs.ioP || io[1];
        cs.ioT = cs.ioT || io[2];
        for (label i = 0; i < p.size; ++i)
        {
            const label k = k0 + i;
            cs.bInletOutlet[k] = (io[0] || io[1] || io[2]) ? 1 : 0;
            cs.bRhoConstraint[k] = isTransformConstraint(p.type) ? 1 : 0;
            cs.bTseedOwn[k] = (isTransformConstraint(Ttypes[pi]) || Ttypes[pi] == "slip") ? 1 : 0;
            if (Ut == "wedge")
            {
                cs.bUTransform[k] = transformRotate;
                cs.bUR[k] = faceT;
            }
            else if (Ut == "symmetryPlane" || Ut == "symmetry" || Ut == "slip")
            {
                cs.bUTransform[k] = transformMirror;
                // symmetryPlane: the patch's one normal; symmetry and slip: patch().nf() = Sf/magSf per face
                cs.bUR[k] = mirrorTensor(Ut == "symmetryPlane" ? nPlane : p.nf[i]);
            }
        }
        k0 += p.size;
    }
    return cs;
}

} // namespace rhoCentral
} // namespace brae
