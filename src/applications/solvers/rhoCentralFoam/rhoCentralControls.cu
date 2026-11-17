// rhoCentralControls.cu -- the case: controls, mesh, and every start-up refusal. See the header.
#include "rhoCentralControls.cuh"

#include "foam_constants.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "geometric_field.cuh"
#include "scheme_parse.cuh"
#include "start_time.cuh"
#include "wedge_patch.cuh"

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
        else if (type != "zeroGradient" && type != "fixedValue" && type != "slip")
            refuse(field + " patch '" + p.name + "' is `" + type + "`. This stage runs zeroGradient, fixedValue, "
                   "slip, symmetryPlane, symmetry, wedge and empty on U, p and T; inletOutlet, totalPressure, "
                   "totalTemperature, waveTransmissive, partialSlip, maxwellSlipU, smoluchowskiJumpT and the rest are "
                   "not ported (applications/solvers/compressible/rhoCentralFoam/BCs, "
                   "src/finiteVolume/fields/fvPatchFields). Refusing rather than run the patch as something else.");
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

} // namespace


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
    // purgeWrite deletes all but the last N time directories (Time::writeObject); the writer here keeps them all
    if (controlDict.intOr("purgeWrite", 0) > 0)
        refuse("controlDict sets `purgeWrite " + std::to_string(controlDict.intOr("purgeWrite", 0)) + "`. Deleting "
               "the older time directories is not ported yet: set `purgeWrite 0`.");
    cs.timeControls = TimeControls::read(controlDict);   // createTimeControls.H: adjustTimeStep, maxCo, maxDeltaT
    cs.writePrecision = controlDict.intOr("writePrecision", 6);
    const StartTime st = resolveStartTime(caseDir, controlDict);
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
    if (rRho != "vanLeer")
        refuse("interpolationSchemes resolves reconstruct(rho) to `" + rRho + "`; this stage ports vanLeer only "
               "(limitedSchemes/vanLeer, NVDTVD). Refusing rather than reconstruct with another limiter.");
    if (rT != "vanLeer")
        refuse("interpolationSchemes resolves reconstruct(T) to `" + rT + "` (it reconstructs rPsi, e and c); "
               "this stage ports vanLeer only.");
    if (rU != "vanLeerV")
        refuse("interpolationSchemes resolves reconstruct(U) to `" + rU + "` (it reconstructs rhoU); this stage "
               "ports vanLeerV only (NVDVTVDV).");
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
            {"transport", "const", "sutherland's mu(T) only matters with the viscous correctors, which are not ported"},
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
        const scalar mu = tr->scalarOr("mu", -1);
        if (mu != 0)
            refuse("transport mu is " + std::to_string(mu) + ". createFieldRefs.H sets inviscid only when every "
                   "mu is 0; with mu > 0 rhoCentralFoam runs the implicit viscous correctors and sigmaDotU "
                   "(rhoCentralFoam.C:235-244, 247-256, 273-282), which this first stage does not port.");
    }

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
    checkPatchFieldTypes(readField<scalar>(dir + "/p"), cs.patches, "p");
    const std::vector<std::string> Ttypes = checkPatchFieldTypes(readField<scalar>(dir + "/T"), cs.patches, "T");

    // ---- the transform patches, face by face (rhoCentralPatchUpdate.cuh) -----------------------------------------------
    cs.bRhoConstraint.assign(cs.nB, 0);
    cs.bTseedOwn.assign(cs.nB, 0);
    cs.bUTransform.assign(cs.nB, transformNone);
    cs.bUR.assign(cs.nB, tensor{1, 0, 0, 0, 1, 0, 0, 0, 1});
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
        for (label i = 0; i < p.size; ++i)
        {
            const label k = k0 + i;
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
