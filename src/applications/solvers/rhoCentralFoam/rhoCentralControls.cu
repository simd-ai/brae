// rhoCentralControls.cu -- the case: controls, mesh, and every start-up refusal. See the header.
#include "rhoCentralControls.cuh"

#include "foam_constants.cuh"
#include "compressibleRAS_cpp.cuh"
#include "dict_audit.cuh"   // ofSchemeStatement: one fvSchemes lookup for the solver and its RAS model
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "geometric_field.cuh"
#include "scheme_parse.cuh"
#include "start_time.cuh"
#include "time_instances.cuh"   // polyMesh's points instance (milestone K)
#include "wedge_patch.cuh"
#include "device_mesh.cuh"   // the limiter selectors (kVanLeerTwoByk and its kin)

#include <algorithm>
#include <cstdio>
#include <optional>
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

// The scheme statement OpenFOAM resolves for `key` in an fvSchemes block (compressibleRAS::ofSchemeStatement): the entry
// whose keyword IS key, else the last quoted keyword -- a regular expression to OpenFOAM -- that matches it, else
// `default`. Statement by statement: a keyword like reconstruct(rho) carries parentheses, and the comparison is on the
// whole keyword, so reconstruct(rho) never answers for reconstruct(rhoU). This used to unquote a quoted keyword and
// compare it as text, so a pattern answered nothing and a quoted name answered as if it were the name.
std::string schemeStatement(const std::string& caseDir, const std::string& block, const std::string& key)
{
    return compressibleRAS::ofSchemeStatement(caseDir, block, key);
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
    // `linear` (kLinearReconTwoByk): OpenFOAM's plain linear interpolation of the field in both directions, on a
    // scalar and on rhoU alike -- no limiter, no gradient. It was refused, which refused every case written with the
    // density-based solvers' usual `interpolationSchemes { default linear; }`.
    if (name == "linear")
    {
        if (!arg.empty()) refuse("interpolationSchemes " + key + " is `" + stmt + "`: `linear` takes no coefficient.");
        return kLinearReconTwoByk;
    }
    if (!isKnown)
        refuse("interpolationSchemes resolves " + key + " to `" + stmt + "`; this stage ports vanLeer, vanAlbada, "
               "Minmod, MUSCL, SuperBee and limitedLinear (and their V forms on U), upwind and linear "
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
        else if (type != "zeroGradient" && type != "fixedValue" && type != "slip" && type != "inletOutlet"
                 && !(field == "U" && (type == "noSlip" || type == "maxwellSlipU" || type == "movingWallVelocity"
                                       || type == "pressureInletOutletVelocity"))
                 && !(field == "T" && type == "smoluchowskiJumpT")
                 && !(field == "p" && (type == "totalPressure" || type == "waveTransmissive"))
                 && !(field == "T" && type == "totalTemperature"))
            refuse(field + " patch '" + p.name + "' is `" + type + "`. This stage runs zeroGradient, fixedValue, "
                   "slip, inletOutlet, symmetryPlane, symmetry, wedge and empty on U, p and T, noSlip, maxwellSlipU, "
                   "movingWallVelocity and pressureInletOutletVelocity on U, totalPressure and waveTransmissive on p and totalTemperature and smoluchowskiJumpT on T; "
                   "partialSlip, outletInlet and the rest are not ported (applications/solvers/compressible/"
                   "rhoCentralFoam/BCs, src/finiteVolume/fields/fvPatchFields). Refusing rather than run the patch as "
                   "something else.");
        // the nozzle boundaries (milestone H): the forms ported, everything else of theirs refused by name
        if (type == "totalPressure" || type == "totalTemperature" || type == "waveTransmissive")
        {
            const std::string w = field + " patch '" + p.name + "' is `" + type + "`";
            // totalPressure: the psi form (milestone H) or, with psi none, the rho form on rhoCentralFoam's rho
            // (milestone K, movingCone: p0 - 0.5*rho_b*neg(phi)*|U_b|^2); the incompressible form (rho none) refused
            const bool rhoForm = type == "totalPressure" && d->psiName == "none" && d->flowRateRhoName == "rho";
            if (d->psiName != "thermo:psi" && !rhoForm)
                refuse(w + " with psi `" + d->psiName + "`" + (type == "totalPressure" ? " and rho `" +
                       d->flowRateRhoName + "`" : std::string()) + ". This stage ports " + (type == "totalPressure"
                       ? "the isentropic form on rhoCentralFoam's thermo:psi and the rho form on its rho "
                         "(totalPressureFvPatchScalarField.C); the incompressible form is not ported"
                       : "the form on rhoCentralFoam's thermo:psi (its only form)") + ".");
            if (d->phiName != "phi")
                refuse(w + " on `phi " + d->phiName + "`; this stage switches on rhoCentralFoam's mass flux `phi`.");
            if (type != "waveTransmissive" && !(d->hasInletValue && !d->totalRefKey.empty()))
                refuse(w + " without its `" + std::string(type == "totalPressure" ? "p0" : "T0") + "` (OpenFOAM "
                       "requires it).");
            if (type == "waveTransmissive")
            {
                if (!d->advFieldName.empty() && d->advFieldName != "p")
                    refuse(w + " on field `" + d->advFieldName + "`; this stage ports waveTransmissive on p.");
                if (d->flowRateRhoName != "rho")
                    refuse(w + " with rho `" + d->flowRateRhoName + "`; this stage divides phi by rhoCentralFoam's rho.");
                if (d->advLInf > 0 && !d->hasAdvFieldInf)
                    refuse(w + " with lInf and no fieldInf (advectiveFvPatchField.C requires fieldInf with lInf).");
            }
        }
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
    const FoamDict& fvSolution = *cs.fvSolution;
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
               "(rhoCentralFoam.C:220-221); this stage computes Gauss linear only.");
    if (schemeStatement(caseDir, "gradSchemes", "grad(U)") != "Gauss linear")
        refuse("gradSchemes resolves grad(U) to `" + schemeStatement(caseDir, "gradSchemes", "grad(U)") + "` (the literal "
               "entry, else the last matching pattern, else the default). tauMC = muEff*dev2(T(fvc::grad(U))) "
               "(rhoCentralFoam.C:220-221); this stage computes Gauss linear only.");
    // the energy laplacian's NAME is its diffusivity's: turbulence->alphaEff() is thermo.alphahe() under laminar
    // (heThermo.C:812, renamed `alphahe`) and thermo.alphaEff(alphat) under RAS (EddyDiffusivity.H:136-139,
    // heThermo.C:875, renamed `alphaEff`). Under RAS this checked the laminar name, so an entry for
    // laplacian(alphaEff,e) was never seen.
    const char* energyKey = cs.ras ? "laplacian(alphaEff,e)" : "laplacian(alphahe,e)";
    const char* energyWhat = cs.ras ? "fvm::laplacian(alphaEff, e), :278 (the field is thermo.alphaEff(alphat), heThermo.C:875)"
                                    : "fvm::laplacian(alphaEff, e), :278 (the field is thermo.alphahe(), heThermo.C:812)";
    const struct { const char* block; const char* key; const char* want; const char* what; } need[] = {
        {"divSchemes", "div(tauMC)", "Gauss linear", "fvc::div(tauMC), rhoCentralFoam.C:241"},
        {"laplacianSchemes", "laplacian(muEff,U)", "Gauss linear corrected", "fvm::laplacian(muEff, U), :240"},
        {"laplacianSchemes", energyKey, "Gauss linear corrected", energyWhat},
        {"snGradSchemes", "snGrad(U)", "corrected", "fvc::snGrad(U) in sigmaDotU, :251"},
        {"interpolationSchemes", "interpolate(muEff)", "linear", "fvc::interpolate(muEff) in sigmaDotU, :251"},
        // fvc::dotInterpolate(mesh.Sf(), tauMC) in sigmaDotU (:251), the scheme named dotInterpolate(S,tauMC)
        // (surfaceInterpolate.C:338); the device forms it linear (rhoCentralViscous.cu) and it was never checked
        {"interpolationSchemes", "dotInterpolate(S,tauMC)", "linear", "fvc::dotInterpolate(mesh.Sf(), tauMC) in sigmaDotU, :251"},
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

    // ---- mesh motion (milestone K) ---------------------------------------------------------------------------------
    // createDynamicFvMesh.H: dynamicFvMesh::New -- staticFvMesh or no dictionary is a static mesh; dynamicMotionSolverFvMesh
    // with velocityComponentLaplacian (movingCone) runs through brae's shared DynamicMotionSolverFvMesh; the rest of the
    // shared class's motions (solidBody, displacementLaplacian, rigidBodyMotion) and every other dynamicFvMesh refused
    try
    {
        std::unique_ptr<DynamicMotionSolverFvMesh> mo =
            DynamicMotionSolverFvMesh::New(caseDir, caseDir + "/" + cs.startName, /*velocitySolvers=*/true);
        if (mo && mo->motionType() != "velocityComponentLaplacian")
            refuse("constant/dynamicMeshDict asks for the `" + mo->motionType() + "` motion. rhoCentralFoam's port moves "
                   "its mesh with velocityComponentLaplacian (movingCone) only; the shared class's other motions are "
                   "not gated with this solver.");
        cs.motion = std::move(mo);
    }
    catch (const std::exception& e)
    {
        refuse(e.what());
    }

    // ---- fvSchemes ------------------------------------------------------------------------------------------------
    const FoamDict fvSchemes = readDict(caseDir + "/system/fvSchemes");
    const std::string flux = fvSchemes.wordOr("fluxScheme", "Kurganov");   // readFluxScheme.H
    if (flux != "Kurganov" && flux != "Tadmor")
        refuse("fvSchemes fluxScheme `" + flux + "` is not a valid choice; options are Tadmor, Kurganov "
               "(readFluxScheme.H, where OpenFOAM stops with the same FatalError).");
    cs.tadmor = (flux == "Tadmor");

    // ddt: Euler, or localEuler (milestone I). LTS is fv::localEulerDdt::enabled(mesh) -- ddtSchemes `default`
    // localEuler (localEulerDdt.C:39-44) -- and every ddt the solver builds must then be localEuler too: a term on
    // another scheme beside an LTS loop is a mix this port refuses rather than guesses at.
    {
        const std::string def = schemeStatement(caseDir, "ddtSchemes", "default");
        for (const char* term : {"ddt(rho)", "ddt(rhoU)", "ddt(rhoE)", "ddt(rho,U)", "ddt(rho,e)"})
        {
            const std::string ddt = schemeStatement(caseDir, "ddtSchemes", term);
            if (ddt != "Euler" && ddt != "localEuler")
                refuse("ddtSchemes resolves " + std::string(term) + " to `" + ddt + "`. Euler and localEuler (LTS, "
                       "milestone I) are ported; the other ddt schemes are not.");
            if ((def == "localEuler") != (ddt == "localEuler"))
                refuse("ddtSchemes resolves " + std::string(term) + " to `" + ddt + "` with `default " + def + "`: "
                       "LTS is decided by the default (localEulerDdt.C), and a mix of localEuler and Euler terms is "
                       "not ported.");
        }
        cs.lts = (def == "localEuler");
        if (cs.lts)
        {
            const FoamDict cd = readDict(caseDir + "/system/controlDict");
            cs.rDeltaTSmoothingCoeff = cd.scalarOr("rDeltaTSmoothingCoeff", 0.02);   // setRDeltaT.H:4-11
        }
    }

    const std::string rRho = schemeStatement(caseDir, "interpolationSchemes", "reconstruct(rho)");
    const std::string rU   = schemeStatement(caseDir, "interpolationSchemes", "reconstruct(U)");
    const std::string rT   = schemeStatement(caseDir, "interpolationSchemes", "reconstruct(T)");
    cs.limRho = reconstructLimiter(rRho, "reconstruct(rho)", false);
    cs.limT = reconstructLimiter(rT, "reconstruct(T)", false);   // it reconstructs rPsi, e and c
    cs.limU = reconstructLimiter(rU, "reconstruct(U)", true);    // it reconstructs rhoU
    cs.reconRho = rRho;
    cs.reconT = rT;
    cs.reconU = rU;
    // the limiters' gradients: LimitedScheme.C:54 builds them with fvc::grad of the reconstructed field itself -- ONLY a
    // limited scheme does: upwind and linear (upwind.H, linear.H) take no gradient, and a case that names them while its
    // gradSchemes default is something brae does not compute was refused for a gradient OpenFOAM never forms
    const auto limited = [](scalar lim) { return lim != kUpwindTwoByk && lim != kLinearReconTwoByk; };
    std::vector<const char*> limitedFields;
    if (limited(cs.limRho)) limitedFields.push_back("rho");
    if (limited(cs.limU)) limitedFields.push_back("rhoU");
    if (limited(cs.limT)) limitedFields.insert(limitedFields.end(), {"rPsi", "e", "c"});
    for (const char* fld : limitedFields)
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
                   "(LimitedScheme.C:54); this stage computes Gauss linear only.");
        }
        // ...and as OpenFOAM resolves the name, a quoted pattern included, which the text search above does not see
        const std::string ofGrad = schemeStatement(caseDir, "gradSchemes", std::string("grad(") + fld + ")");
        if (ofGrad != "Gauss linear")
            refuse(std::string("gradSchemes resolves grad(") + fld + ") to `" + ofGrad + "` (the literal entry, else the "
                   "last matching pattern, else the default). The limiter reads that gradient (LimitedScheme.C:54); this "
                   "stage computes Gauss linear only.");
    }

    // ---- fvSolution: the three diagonal solves --------------------------------------------------------------------
    // fvm::ddt(rho) + fvc::div(phi) and its two siblings (rhoCentralFoam.C:224-231) are DIAGONAL matrices, and
    // lduMatrix::solver::New returns diagonalSolver for any diagonal matrix whatever the entry names
    // (lduMatrixSolver.C:53-67): the entry must exist (solution::solverDict, a subDict) and carry a `solver` word
    // (get<word>, lduMatrixSolver.C:150), nothing else. Any other name was refused here; it runs now, as the diagonal
    // solve brae always ran (x = source/diag).
    cs.fvSolution = std::make_shared<FoamDict>(readDict(caseDir + "/system/fvSolution"));
    {
        const FoamDict* solvers = cs.fvSolution->subDict("solvers");
        for (const char* fld : {"rho", "rhoU", "rhoE"})
        {
            const FoamDict* sd = solvers ? solvers->subDict(fld) : nullptr;
            if (!sd)
                refuse(std::string("fvSolution solvers has no entry for ") + fld + "; OpenFOAM looks one up for its "
                       "solve (solution::solverDict) and stops without it.");
            if (sd->wordOr("solver", "").empty())
                refuse(std::string("fvSolution solvers' entry for ") + fld + " names no `solver`; OpenFOAM reads the "
                       "word (lduMatrixSolver.C:150) and stops without it, before it finds the matrix diagonal.");
        }
    }

    // ---- thermophysicalProperties ---------------------------------------------------------------------------------
    {
        cs.thermophysicalProperties = std::make_shared<FoamDict>(readDict(caseDir + "/constant/thermophysicalProperties"));
        const FoamDict& tp = *cs.thermophysicalProperties;
        const FoamDict* tt = tp.subDict("thermoType");
        if (!tt) refuse("constant/thermophysicalProperties has no thermoType.");
        const struct { const char* key; const char* want; const char* why; } need[] = {
            {"type", "hePsiThermo", "rhoCentralFoam builds a psiThermo (createFields.H); hePsiThermo is the one ported"},
            {"mixture", "pureMixture", "multi-species mixtures are not ported"},
            {"transport", "const|sutherland", "const and sutherland transport are ported (milestones G and H)"},
            {"thermo", "hConst|janaf", "hConst and janaf are ported (janaf: milestone I); eConst is not"},
            {"equationOfState", "perfectGas", "the psi = 1/(R T) equation of state is the one ported"},
            {"specie", "specie", "the specie model"},
            {"energy", "sensibleInternalEnergy", "rhoCentralFoam transports e (createFields.H: e = thermo.he())"},
        };
        for (const auto& n : need)
        {
            const std::string v = tt->wordOr(n.key, "");
            if (std::string(n.key) == "transport" && (v == "const" || v == "sutherland")) continue;
            if (std::string(n.key) == "thermo" && (v == "hConst" || v == "janaf")) continue;
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
        cs.gas.Tstd  = foamTstd();
        if (tt->wordOr("thermo", "") == "janaf" && tt->wordOr("transport", "") != "sutherland")
            refuse("thermoType " + tt->wordOr("transport", "") + " transport with janaf: OpenFOAM v2412 has no such psiThermo "
                   "(\"Unknown psiThermo type\"; with sensibleInternalEnergy on a perfectGas it builds const<hConst>, "
                   "sutherland<hConst> and sutherland<janaf>).");
        if (tt->wordOr("thermo", "") == "janaf")
        {
            // janafThermo.C:63-79: Tlow, Thigh, Tcommon, highCpCoeffs and lowCpCoeffs (7 each), each coefficient
            // multiplied by R; checkInputData (:33-55): Tlow < Thigh, Tlow < Tcommon <= Thigh
            for (const char* k : {"Tlow", "Thigh", "Tcommon", "highCpCoeffs", "lowCpCoeffs"})
                if (!th->found(k)) refuse(std::string("thermophysicalProperties mixture.thermodynamics (janaf) has no `") + k + "`.");
            cs.gas.janaf   = 1;
            cs.gas.Tlow    = th->scalarOr("Tlow", 0);
            cs.gas.Thigh   = th->scalarOr("Thigh", 0);
            cs.gas.Tcommon = th->scalarOr("Tcommon", 0);
            const std::vector<scalar> hi = th->scalarListOr("highCpCoeffs", {});
            const std::vector<scalar> lo = th->scalarListOr("lowCpCoeffs", {});
            if (hi.size() != 7 || lo.size() != 7)
                refuse("janaf highCpCoeffs and lowCpCoeffs need 7 coefficients each (janafThermo.H nCoeffs_).");
            for (int i = 0; i < 7; ++i)
            {
                cs.gas.highCp[i] = hi[static_cast<std::size_t>(i)]*cs.gas.R;
                cs.gas.lowCp[i]  = lo[static_cast<std::size_t>(i)]*cs.gas.R;
            }
            if (cs.gas.Tlow >= cs.gas.Thigh) refuse("janaf Tlow >= Thigh (janafThermo.C:35-40).");
            if (cs.gas.Tcommon <= cs.gas.Tlow) refuse("janaf Tcommon <= Tlow (janafThermo.C:42-47).");
            if (cs.gas.Tcommon > cs.gas.Thigh) refuse("janaf Tcommon > Thigh (janafThermo.C:49-54).");
        }
        else
        {
        cs.gas.Cp    = th->scalarOr("Cp", 0);
        cs.gas.Tref  = th->scalarOr("Tref", foamTstd());   // hConstThermo.C:41
        cs.gas.Hsref = th->scalarOr("Href", 0);            // hConstThermo.C:41 reads Hsref_ from `Href`
        // Hf: hConstThermo's constructor reads it (hConstThermo.C:39), and only the absolute energy and enthalpy (Ha, Ea)
        // add it; rhoCentralFoam's sensibleInternalEnergy never does. Taken as read, as OpenFOAM takes it.
        th->find("Hf");
        if (!(cs.gas.Cp > cs.gas.R)) refuse("mixture Cp must exceed R (Cv = Cp - R must be positive).");
        }
        if (tt->wordOr("transport", "") == "sutherland")
        {
            // sutherlandTransport.C: As and Ts (Pr is not read); mu(T) > 0 everywhere, so createFieldRefs.H's
            // inviscid is false: the viscous correctors run (milestone H)
            if (!tr->found("As") || !tr->found("Ts"))
                refuse("thermophysicalProperties mixture.transport (sutherland) needs As and Ts (sutherlandTransport.C).");
            cs.gas.sutherland = 1;
            cs.gas.As = tr->scalarOr("As", 0);
            cs.gas.Ts = tr->scalarOr("Ts", 0);
            if (!(cs.gas.As > 0) || cs.gas.Ts < 0)
                refuse("sutherland As must be positive and Ts not negative.");
            cs.viscous = true;
        }
        else
        {
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
    }
    // ---- turbulence: laminar, or RAS (milestone J; the model and its fields: compressibleRAS::readCase) -----------
    {
        std::string tpath = caseDir + "/constant/turbulenceProperties";
        if (!std::filesystem::exists(tpath)) tpath = caseDir + "/constant/momentumTransport";
        if (!std::filesystem::exists(tpath)) refuse("no constant/turbulenceProperties (or momentumTransport).");
        cs.turbulenceProperties = std::make_shared<FoamDict>(readDict(tpath));
        const FoamDict& t = *cs.turbulenceProperties;
        const std::string sim = t.wordOr("simulationType", "");
        if (sim != "laminar" && sim != "RAS")
            refuse("simulationType is `" + sim + "`; laminar and RAS (kOmegaSST, milestone J) are ported; LES is not.");
        cs.ras = (sim == "RAS");
        // createFieldRefs.H decides inviscid on the LAMINAR mu alone: with every mu 0 rhoCentralFoam runs no viscous
        // corrector and never reads muEff or alphaEff, whatever the turbulence model computes
        if (cs.ras && !cs.viscous)
            refuse("simulationType RAS with a laminar mu of 0: rhoCentralFoam then runs inviscid (createFieldRefs.H) and "
                   "the model's muEff and alphaEff reach no equation; not ported as a turbulent case.");
        if (const FoamDict* lam = t.subDict("laminar"))
        {
            const std::string model = lam->wordOr("model", lam->wordOr("laminarModel", "Stokes"));
            if (model != "Stokes")
                refuse("laminar model `" + model + "` is not ported; Stokes (the default) is.");
        }
    }
    // the viscous correctors' controls, once the turbulence is known: the energy laplacian's name is the model's
    if (cs.viscous) readViscousControls(caseDir, cs);

    // ---- mesh -----------------------------------------------------------------------------------------------------
    // polyMesh reads each file from its own instance, searched back from the start time (polyMesh.C): a moving
    // mesh's restart takes its points from the start directory, its faces from constant
    {
        const cpu::timePaths::MeshInstances mi =
            cpu::timePaths::meshInstancesForStartDir(caseDir, caseDir + "/" + cs.startName);
        cs.m.read(mi.pointsDir(caseDir), mi.facesDir(caseDir), mi.boundaryDir(caseDir));
        if (!cs.motion && mi.points != "constant")
            refuse("the start time's points are in " + mi.points + "/polyMesh: a moved mesh, with no motion in "
                   "constant/dynamicMeshDict to continue it.");
    }
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
    const FieldData<scalar> pfd = readField<scalar>(dir + "/p");
    const FieldData<scalar> Tfd = readField<scalar>(dir + "/T");
    const FieldData<vector> Ufd = readField<vector>(dir + "/U");
    // the rarefied-gas walls (milestone I; rhoCentralFoam/BCs, built into librhoCentralFoam): their own keywords, from
    // the field files' dictionaries
    const bool anySlipWall = std::find(Utypes.begin(), Utypes.end(), "maxwellSlipU") != Utypes.end()
                          || std::find(Ttypes.begin(), Ttypes.end(), "smoluchowskiJumpT") != Ttypes.end();
    std::optional<FoamDict> Udict, Tdict;
    if (anySlipWall)
    {
        Udict.emplace(readDict(dir + "/U"));
        Tdict.emplace(readDict(dir + "/T"));
        if (!cs.viscous)
            refuse("maxwellSlipU and smoluchowskiJumpT read mu, tauMC and the viscous solves (rhoCentralFoam/BCs); this "
                   "port runs them on a viscous case only (mu > 0), where milestone I validates them.");
        // smoluchowskiJumpT reads Pr from mixture.transport, "consistent with rhoCentralFoam"
        // (smoluchowskiJumpTFvPatchScalarField.C:171-176): required there even under sutherland, which ignores it
        if (std::find(Ttypes.begin(), Ttypes.end(), "smoluchowskiJumpT") != Ttypes.end())
        {
            // the case's own object (Case::thermophysicalProperties), so the unread-entry audit sees this question
            const FoamDict& tp = *cs.thermophysicalProperties;
            const FoamDict* mix = tp.subDict("mixture");
            const FoamDict* tr = mix ? mix->subDict("transport") : nullptr;
            if (!tr || !tr->found("Pr"))
                refuse("smoluchowskiJumpT reads Pr from thermophysicalProperties mixture.transport "
                       "(smoluchowskiJumpTFvPatchScalarField.C:171-176); it is missing.");
            cs.sjPr = tr->scalarOr("Pr", 0);
        }
    }
    // a patch's `uniform` entry (OpenFOAM's Field(name, dict, size)); nonuniform ones are not ported for these walls
    auto uniformScalar = [&](const FoamDict* d, const std::string& key, const std::string& what) -> scalar {
        const std::vector<std::string>* t = d ? d->find(key) : nullptr;
        if (!t || t->size() < 2 || (*t)[0] != "uniform")
            refuse(what + ": `" + key + "` must be `uniform <value>` here (nonuniform walls are not ported).");
        return std::stod((*t)[1]);
    };
    auto uniformVector = [&](const FoamDict* d, const std::string& key, const std::string& what) -> vector {
        const std::vector<std::string>* t = d ? d->find(key) : nullptr;
        std::vector<scalar> v;
        if (t && !t->empty() && (*t)[0] == "uniform")
            for (std::size_t i = 1; i < t->size(); ++i)
            {
                std::string w = (*t)[i];
                w.erase(std::remove(w.begin(), w.end(), '('), w.end());
                w.erase(std::remove(w.begin(), w.end(), ')'), w.end());
                if (!w.empty()) v.push_back(std::stod(w));
            }
        if (v.size() != 3) refuse(what + ": `" + key + "` must be `uniform (x y z)` here (nonuniform walls are not ported).");
        return vector{v[0], v[1], v[2]};
    };
    auto patchDict = [&](const std::optional<FoamDict>& f, const std::string& name) -> const FoamDict* {
        const FoamDict* bf = f ? f->subDict("boundaryField") : nullptr;
        return bf ? bf->subDict(name) : nullptr;
    };

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
        // the nozzle boundaries' faces and their per-face references (milestone H)
        {
            auto ref = [](const PatchFieldData<scalar>* d, label i) {
                return d->inletUniform ? d->inletUniformValue : d->inletValues[static_cast<std::size_t>(i)];
            };
            if (ptypes[pi] == "totalPressure")
            {
                const PatchFieldData<scalar>* d = findPatchEntry(pfd, p);
                for (label i = 0; i < p.size; ++i)
                {
                    cs.tpFaces.push_back(k0 + i);
                    cs.tpP0.push_back(ref(d, i));
                    cs.tpGamma.push_back(d->gammaTP);
                    cs.tpRho.push_back(d->psiName == "none" ? 1 : 0);   // the rho form (milestone K)
                }
            }
            if (ptypes[pi] == "waveTransmissive")
            {
                const PatchFieldData<scalar>* d = findPatchEntry(pfd, p);
                for (label i = 0; i < p.size; ++i)
                {
                    cs.wtFaces.push_back(k0 + i);
                    cs.wtFieldInf.push_back(d->advFieldInf);
                    cs.wtLInf.push_back(d->advLInf);
                    cs.wtGamma.push_back(d->gammaTP);
                }
            }
            // maxwellSlipU (maxwellSlipUFvPatchVectorField.C:77-117): accommodationCoeff (0 < |a| <= 2, OpenFOAM's
            // check), Uwall, thermalCreep and curvature (default on), refValue and valueFraction when written with a
            // value (else refValue = value, valueFraction 1); the names it reads must be rhoCentralFoam's own
            if (Ut == "maxwellSlipU")
            {
                const std::string w = "U patch '" + p.name + "' is `maxwellSlipU`";
                const FoamDict* d = patchDict(Udict, p.name);
                const PatchFieldData<vector>* fd = findPatchEntry(Ufd, p);
                if (!d || !fd) refuse(w + ": its dictionary could not be read.");
                for (const auto& nm : {std::pair<const char*, const char*>{"T", "T"}, {"rho", "rho"},
                                       {"psi", "thermo:psi"}, {"mu", "thermo:mu"}, {"tauMC", "tauMC"}})
                    if (d->wordOr(nm.first, nm.second) != nm.second)
                        refuse(w + " with " + nm.first + " `" + d->wordOr(nm.first, "") + "`; this port reads "
                               "rhoCentralFoam's own `" + nm.second + "`.");
                if (!d->found("accommodationCoeff")) refuse(w + " without accommodationCoeff (OpenFOAM requires it).");
                const scalar a = d->scalarOr("accommodationCoeff", 0);
                if (std::fabs(a) < kSMALL || std::fabs(a) > 2.0)
                    refuse(w + ": unphysical accommodationCoeff (maxwellSlipUFvPatchVectorField.C:94-104).");
                const vector Uw = uniformVector(d, "Uwall", w);
                const int creep = d->switchOr("thermalCreep", true) ? 1 : 0;
                const int curv = d->switchOr("curvature", true) ? 1 : 0;
                const bool both = fd->hasRefValue && fd->hasValueFraction && fd->hasValue;
                for (label i = 0; i < p.size; ++i)
                {
                    const std::size_t ii = static_cast<std::size_t>(i);
                    cs.msFaces.push_back(k0 + i);
                    cs.msAcc.push_back(a);
                    cs.msUwall.push_back(Uw);
                    cs.msCreep.push_back(creep);
                    cs.msCurv.push_back(curv);
                    const vector val = fd->valueUniform ? fd->uniformValue : fd->values[ii];
                    cs.msRef0.push_back(both ? (fd->refValueUniform ? fd->refValueUniformValue : fd->refValues[ii]) : val);
                    cs.msVf0.push_back(both ? (fd->vfUniform ? fd->vfUniformValue : fd->vfValues[ii]) : scalar(1));
                }
            }
            // smoluchowskiJumpT (smoluchowskiJumpTFvPatchScalarField.C:76-115): accommodationCoeff, Twall, gamma
            // (default 1.4); constructed with refValue = value, refGrad 0, valueFraction 0
            if (Ttypes[pi] == "smoluchowskiJumpT")
            {
                const std::string w = "T patch '" + p.name + "' is `smoluchowskiJumpT`";
                const FoamDict* d = patchDict(Tdict, p.name);
                if (!d) refuse(w + ": its dictionary could not be read.");
                for (const auto& nm : {std::pair<const char*, const char*>{"U", "U"}, {"rho", "rho"},
                                       {"psi", "thermo:psi"}, {"mu", "thermo:mu"}})
                    if (d->wordOr(nm.first, nm.second) != nm.second)
                        refuse(w + " with " + nm.first + " `" + d->wordOr(nm.first, "") + "`; this port reads "
                               "rhoCentralFoam's own `" + nm.second + "`.");
                if (!d->found("accommodationCoeff")) refuse(w + " without accommodationCoeff (OpenFOAM requires it).");
                const scalar a = d->scalarOr("accommodationCoeff", 0);
                if (std::fabs(a) < kSMALL || std::fabs(a) > 2.0)
                    refuse(w + ": unphysical accommodationCoeff (smoluchowskiJumpTFvPatchScalarField.C:94-104).");
                const scalar Tw = uniformScalar(d, "Twall", w);
                const scalar g = d->scalarOr("gamma", 1.4);
                for (label i = 0; i < p.size; ++i)
                {
                    cs.sjFaces.push_back(k0 + i);
                    cs.sjAcc.push_back(a);
                    cs.sjTwall.push_back(Tw);
                    cs.sjGamma.push_back(g);
                }
            }
            if (Ttypes[pi] == "totalTemperature")
            {
                const PatchFieldData<scalar>* d = findPatchEntry(Tfd, p);
                for (label i = 0; i < p.size; ++i)
                {
                    cs.ttFaces.push_back(k0 + i);
                    cs.ttT0.push_back(ref(d, i));
                    cs.ttGamma.push_back(d->gammaTP);
                }
            }
        }
        // milestone K: movingWallVelocity's faces (a fixedValue the motion sets) and pressureInletOutletVelocity
        // (directionMixed on rhoCentralFoam's phi; a tangentialVelocity refused by the shared factory)
        if (Ut == "movingWallVelocity")
            for (label i = 0; i < p.size; ++i) cs.mwFaces.push_back(k0 + i);
        if (Ut == "pressureInletOutletVelocity")
        {
            const PatchFieldData<vector>* fd = findPatchEntry(Ufd, p);
            if (fd && fd->phiName != "phi")
                refuse("U patch '" + p.name + "' is `pressureInletOutletVelocity` on `phi " + fd->phiName + "`; this "
                       "stage switches it on rhoCentralFoam's mass flux `phi`.");
            cs.piovU = true;
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
    // maxwellSlipU's thermal creep takes fvc::grad(T) (maxwellSlipUFvPatchVectorField.C:170-180), the scheme named
    // grad(T); the device forms it Gauss linear (rhoCentralSlipBCs.cu, updateMaxwell) and it was never checked
    if (std::find(cs.msCreep.begin(), cs.msCreep.end(), 1) != cs.msCreep.end()
        && schemeStatement(caseDir, "gradSchemes", "grad(T)") != "Gauss linear")
        refuse("gradSchemes resolves grad(T) to `" + schemeStatement(caseDir, "gradSchemes", "grad(T)") + "`; maxwellSlipU's "
               "thermal creep takes fvc::grad(T) (maxwellSlipUFvPatchVectorField.C:170-180) and this port computes it "
               "Gauss linear only.");
    // waveTransmissive's advective update has its own localEuler branch (advectiveFvPatchField.C: alpha from the
    // patch's rDeltaT), which milestone H did not port
    if (cs.lts && !cs.wtFaces.empty())
        refuse("waveTransmissive with ddtSchemes localEuler: the advective boundary's local-time-step branch "
               "(advectiveFvPatchField.C, rDeltaT on the patch) is not ported.");
    // mesh motion (milestone K) with what its port has not gated, by name
    if (cs.motion)
    {
        if (cs.lts)
            refuse("a dynamicMeshDict with ddtSchemes localEuler: rhoCentralFoam then never calls mesh.update() "
                   "(rhoCentralFoam.C:89-97, `if (!LTS)`), the mesh stands still; not ported.");
        for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
            if (cs.patches[pi].type == "symmetryPlane" || cs.patches[pi].type == "symmetry"
                || Utypes[pi] == "slip")
                refuse("patch '" + cs.patches[pi].name + "' (" + cs.patches[pi].type + ", U " + Utypes[pi] + ") on a moving "
                       "mesh: the mirror its transform takes is built once from the mesh as read; not ported with "
                       "mesh motion.");
        if (cs.slipWalls())
            refuse("maxwellSlipU or smoluchowskiJumpT on a moving mesh is not ported.");
        if (cs.ras)
            refuse("a RAS model on a moving mesh is not ported (the wall distance and the closure's geometry are "
                   "built once).");
        if (!cs.wtFaces.empty())
            refuse("waveTransmissive on a moving mesh is not ported.");
    }
    return cs;
}

void auditRhoCentralDictionaries(const Case& cs)
{
    std::fflush(stdout);
    if (cs.fvSolution) auditDict(*cs.fvSolution, "system/fvSolution");
    if (cs.turbulenceProperties) auditDict(*cs.turbulenceProperties, "constant/turbulenceProperties");
    if (cs.thermophysicalProperties) auditDict(*cs.thermophysicalProperties, "constant/thermophysicalProperties");
    // fvSchemes, at block granularity as dict_audit.cuh's auditFvSchemes does it -- whose list of blocks lacks
    // rhoCentralFoam's own top-level `fluxScheme` (readFluxScheme.H), so the list is this solver's: the blocks its
    // readers resolve, wallDist under RAS only (the model's wallDist::New)
    if (const char* e = std::getenv("BRAE_DICT_AUDIT"))
        if (std::atoi(e) == 0) return;
    FoamDict fvSchemes;
    try { fvSchemes = readDict(cs.caseDir + "/system/fvSchemes"); }
    catch (const std::exception&) { return; }
    std::vector<std::string> read = {"ddtSchemes", "gradSchemes", "divSchemes", "laplacianSchemes",
                                     "interpolationSchemes", "snGradSchemes", "fluxScheme"};
    if (cs.ras) read.push_back("wallDist");
    std::vector<std::string> unread;
    for (const auto& sub : fvSchemes.subs)
        if (std::find(read.begin(), read.end(), sub.first) == read.end()) unread.push_back(sub.first + "/ (whole block)");
    for (const auto& leaf : fvSchemes.leaves)
        if (std::find(read.begin(), read.end(), leaf.first) == read.end()) unread.push_back(leaf.first + " (entry)");
    if (unread.empty()) return;
    std::fprintf(stderr, "brae NOTICE [unread] system/fvSchemes -- %zu entr%s brae never looked at:\n", unread.size(),
                 unread.size() == 1 ? "y" : "ies");
    for (const std::string& k : unread) std::fprintf(stderr, "brae NOTICE [unread]   %s\n", k.c_str());
    std::fprintf(stderr, "brae NOTICE [unread]   (fvSchemes is audited by block: its entries are text-parsed and not "
                         "tracked one by one)\n");
}

} // namespace rhoCentral
} // namespace brae
