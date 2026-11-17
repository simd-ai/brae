// rhoCentralFoam_cpp.cu -- the case, createFields and the host reference step. See the header for the scope.
#include "rhoCentralFoam_cpp.cuh"

#include "brae_notice.cuh"
#include "foam_constants.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "geometric_field.cuh"
#include "scheme_parse.cuh"
#include "start_time.cuh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace rhoCentral {

namespace {

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

// The boundary-condition kind of every patch for one field, refusing what this stage does not run.
template <typename T>
std::vector<int> fieldKinds(const FieldData<T>& fd, const std::vector<FvPatch>& patches, const std::string& field)
{
    std::vector<int> k(patches.size(), bcZeroGradient);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        const PatchFieldData<T>* d = findPatchEntry(fd, p);
        const std::string type = d ? d->type : (p.type == "empty" ? std::string("empty") : std::string());
        if (type.empty())
            refuse("no boundaryField entry for patch '" + p.name + "' in " + field + ".");
        if (type == "empty")
        {
            if (p.type != "empty")
                refuse(field + " patch '" + p.name + "' is `empty` on a mesh patch of type " + p.type + ".");
            k[pi] = bcEmpty;
        }
        else if (type == "zeroGradient") k[pi] = bcZeroGradient;
        else if (type == "fixedValue")   k[pi] = bcFixedValue;
        else
            refuse(field + " patch '" + p.name + "' is `" + type + "`. This first stage runs zeroGradient, "
                   "fixedValue and empty on U, p and T; slip, symmetryPlane, wedge, inletOutlet, totalPressure, "
                   "totalTemperature, waveTransmissive, maxwellSlipU, smoluchowskiJumpT and the rest are not ported "
                   "(applications/solvers/compressible/rhoCentralFoam/BCs, src/finiteVolume/fields/fvPatchFields). "
                   "Refusing rather than run the patch as something else.");
    }
    return k;
}

} // namespace


Case readRhoCentralCase(const std::string& caseDir)
{
    Case cs;
    cs.caseDir = caseDir;

    // ---- controlDict: fixed deltaT only ------------------------------------------------------------------------
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const std::string app = controlDict.wordOr("application", "rhoCentralFoam");
    if (app != "rhoCentralFoam")
        refuse("controlDict names application `" + app + "`; this executable runs rhoCentralFoam cases.");
    if (controlDict.switchOr("adjustTimeStep", false))
        refuse("controlDict sets `adjustTimeStep yes`. This first stage runs a FIXED deltaT: the central "
               "Courant number is computed and logged (centralCourantNo.H), but setDeltaT.H (maxCo, maxDeltaT, the "
               "1.2 growth cap and writeControl adjustable's trimming to write times) is not ported. Set "
               "`adjustTimeStep no` and choose deltaT, or wait for the step that ports setDeltaT.");
    cs.deltaT = controlDict.scalarOr("deltaT", 0);
    if (!(cs.deltaT > 0)) refuse("controlDict deltaT must be positive.");
    cs.writePrecision = controlDict.intOr("writePrecision", 6);
    const StartTime st = resolveStartTime(caseDir, controlDict);
    cs.startName = st.name;
    cs.startTime = static_cast<scalar>(st.value);
    const scalar endTime = controlDict.scalarOr("endTime", 0);
    cs.nSteps = openFoamNSteps(static_cast<double>(cs.startTime), static_cast<double>(endTime),
                               static_cast<double>(cs.deltaT));
    if (cs.nSteps < 1)
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
        if (p.type != "patch" && p.type != "wall" && p.type != "empty")
            refuse("mesh patch '" + p.name + "' is of type `" + p.type + "`. This stage runs patch, wall and empty "
                   "patches; symmetryPlane, symmetry, wedge, cyclic and processor patches are not ported.");
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

    // ---- boundary condition kinds -----------------------------------------------------------------------------------
    const std::string dir = caseDir + "/" + cs.startName;
    cs.bcU = fieldKinds(readField<vector>(dir + "/U"), cs.patches, "U");
    cs.bcP = fieldKinds(readField<scalar>(dir + "/p"), cs.patches, "p");
    cs.bcT = fieldKinds(readField<scalar>(dir + "/T"), cs.patches, "T");
    return cs;
}


State createFields(const Case& cs)
{
    const std::string dir = cs.caseDir + "/" + cs.startName;
    const GasCoeffs& g = cs.gas;
    State s;

    // the thermo reads p and T (basicThermo.C), each patch evaluated on construction
    const GeometricField<scalar> pF = buildField(readField<scalar>(dir + "/p"), cs.patches, cs.nC);
    const GeometricField<scalar> TF = buildField(readField<scalar>(dir + "/T"), cs.patches, cs.nC);
    const GeometricField<vector> UF = buildField(readField<vector>(dir + "/U"), cs.patches, cs.nC);
    s.p = pF.internal;
    s.T = TF.internal;
    s.U = UF.internal;
    s.pB.resize(cs.nB);
    s.TB.resize(cs.nB);
    s.UB.resize(cs.nB);
    for (label k = 0; k < cs.nB; ++k)
    {
        const int pi = cs.bPatch[k];
        const label i = k - (cs.patches[pi].start - cs.nIF);
        const label c = cs.bCell[k];
        // a zeroGradient patch's value is its face cell's; brae's patch fields hold that after construction too
        s.pB[k] = (cs.bcP[pi] == bcFixedValue) ? pF.boundary[pi]->value()[i] : s.p[c];
        s.TB[k] = (cs.bcT[pi] == bcFixedValue) ? TF.boundary[pi]->value()[i] : s.T[c];
        s.UB[k] = (cs.bcU[pi] == bcFixedValue) ? UF.boundary[pi]->value()[i] : s.U[c];
    }

    // heThermo::init (heThermo.C:58-90): he from (p, T) on cells and patches. heBoundaryCorrection sets a
    // gradientEnergy patch's gradient to snGrad(he), which every later evaluate recomputes before use.
    s.e.resize(cs.nC);
    s.eB.resize(cs.nB);
    for (label c = 0; c < cs.nC; ++c) s.e[c] = gasEs(s.p[c], s.T[c], g);
    for (label k = 0; k < cs.nB; ++k) s.eB[k] = gasEs(s.pB[k], s.TB[k], g);

    // hePsiThermo's constructor: calculate(p, T, he, psi, mu, alpha) -- T is re-derived from he even here
    s.psi.resize(cs.nC);
    s.psiB.resize(cs.nB);
    for (label c = 0; c < cs.nC; ++c)
    {
        const GasTResult r = gasTEs(s.e[c], s.p[c], s.T[c], g);
        if (r.status) refuse("createFields: the e -> T inversion failed in cell " + std::to_string(c) + ".");
        s.T[c]   = r.T;
        s.psi[c] = gasPsi(s.p[c], s.T[c], g);
    }
    for (label k = 0; k < cs.nB; ++k)
    {
        const int pi = cs.bPatch[k];
        if (cs.bcT[pi] == bcFixedValue)
        {
            s.eB[k] = gasEs(s.pB[k], s.TB[k], g);
        }
        else
        {
            const GasTResult r = gasTEs(s.eB[k], s.pB[k], s.TB[k], g);
            if (r.status) refuse("createFields: the e -> T inversion failed on boundary face " + std::to_string(k) + ".");
            s.TB[k] = r.T;
        }
        s.psiB[k] = gasPsi(s.pB[k], s.TB[k], g);
    }

    // createFields.H: rho = thermo.rho() = p*psi; rhoU = rho*U; rhoE = rho*(e + 0.5*magSqr(U))
    s.rho.resize(cs.nC);
    s.rhoU.resize(cs.nC);
    s.rhoE.resize(cs.nC);
    for (label c = 0; c < cs.nC; ++c)
    {
        s.rho[c]  = s.p[c]*s.psi[c];
        s.rhoU[c] = s.rho[c]*s.U[c];
        s.rhoE[c] = s.rho[c]*(s.e[c] + 0.5*magSqr(s.U[c]));
    }
    s.rhoB.resize(cs.nB);
    s.rhoUB.resize(cs.nB);
    s.rhoEB.resize(cs.nB);
    for (label k = 0; k < cs.nB; ++k)
    {
        s.rhoB[k]  = s.pB[k]*s.psiB[k];
        s.rhoUB[k] = s.rhoB[k]*s.UB[k];
        s.rhoEB[k] = s.rhoB[k]*(s.eB[k] + 0.5*magSqr(s.UB[k]));
    }
    return s;
}


namespace {

// fvc::grad(vf) with gaussGrad and linear interpolation (gaussGrad.C:76-104): internal faces in order,
// owner += Sf*ssf and neighbour -= Sf*ssf, then every non-empty boundary face, then /= V.
void gaussGrad(const Case& cs, const std::vector<scalar>& vf, const std::vector<scalar>& vb,
               std::vector<vector>& grad)
{
    const std::vector<label>& own = cs.m.owner();
    const std::vector<label>& nei = cs.m.neighbour();
    const std::vector<scalar>& w = cs.g.weights();
    const std::vector<vector>& Sf = cs.g.Sf();
    const std::vector<scalar>& V = cs.g.V();
    grad.assign(cs.nC, vector{0, 0, 0});
    for (label f = 0; f < cs.nIF; ++f)
    {
        const scalar ssf = w[f]*(vf[own[f]] - vf[nei[f]]) + vf[nei[f]];
        const vector Sfssf = Sf[f]*ssf;
        grad[own[f]] += Sfssf;
        grad[nei[f]] -= Sfssf;
    }
    for (label k = 0; k < cs.nB; ++k)
    {
        if (cs.bEmpty[k]) continue;
        grad[cs.bCell[k]] += Sf[cs.nIF + k]*vb[k];
    }
    for (label c = 0; c < cs.nC; ++c) grad[c] = grad[c]/V[c];
}

inline scalar sgn(scalar s) { return (s >= 0) ? 1.0 : -1.0; }   // Foam::sign
inline scalar pos0(scalar s) { return (s >= 0) ? 1.0 : 0.0; }   // Foam::pos0
inline scalar fmaxOF(scalar a, scalar b) { return (a > b) ? a : b; }   // Foam::max
inline scalar fminOF(scalar a, scalar b) { return (a < b) ? a : b; }   // Foam::min

// vanLeerLimiter<NVDTVD>::limiter (vanLeer.H, NVDTVD.H:96-121)
scalar vanLeerScalar(scalar faceFlux, scalar phiP, scalar phiN, const vector& gP, const vector& gN, const vector& d)
{
    const scalar gradf = phiN - phiP;
    const scalar gradcf = (faceFlux > 0) ? dot(d, gP) : dot(d, gN);
    scalar r;
    if (std::fabs(gradcf) >= 1000*std::fabs(gradf)) r = 2*1000*sgn(gradcf)*sgn(gradf) - 1;
    else                                           r = 2*(gradcf/gradf) - 1;
    return (r + std::fabs(r))/(1 + std::fabs(r));
}

// d & gradc, a vector & tensor, with the tensor held as the three component gradients g[j] = grad(q_j):
// (d & T)_j = d.x*T_xj + d.y*T_yj + d.z*T_zj (VectorSpace inner product, TensorI.H)
inline vector dDotGrad(const vector& d, const vector g[3])
{
    return vector{d.x*g[0].x + d.y*g[0].y + d.z*g[0].z,
                  d.x*g[1].x + d.y*g[1].y + d.z*g[1].z,
                  d.x*g[2].x + d.y*g[2].y + d.z*g[2].z};
}

// vanLeerLimiter<NVDVTVDV>::limiter (NVDVTVDV.H:96-132)
scalar vanLeerVector(scalar faceFlux, const vector& phiP, const vector& phiN, const vector gP[3], const vector gN[3],
                     const vector& d)
{
    const vector gradfV = phiN - phiP;
    const scalar gradf = dot(gradfV, gradfV);
    const scalar gradcf = (faceFlux > 0) ? dot(gradfV, dDotGrad(d, gP)) : dot(gradfV, dDotGrad(d, gN));
    scalar r;
    if (std::fabs(gradcf) >= 1000*std::fabs(gradf)) r = 2*1000*sgn(gradcf)*sgn(gradf) - 1;
    else                                           r = 2*(gradcf/gradf) - 1;
    return (r + std::fabs(r))/(1 + std::fabs(r));
}

// limitedSurfaceInterpolationScheme::weights: limiter*CDweight + (1 - limiter)*pos0(faceFlux)
inline scalar limitedWeight(scalar limiter, scalar cdw, scalar dir)
{
    return limiter*cdw + (1.0 - limiter)*pos0(dir);
}

// the face core of rhoCentralFoam.C:114-210 for one face, both directions given
struct FaceIn
{
    scalar rho[2], rPsi[2], e[2], c[2];
    vector rhoU[2];
};
struct FaceOut
{
    scalar phi, phiEp, amaxSf;
    vector phiUp;
};

FaceOut faceFlux(const FaceIn& in, const vector& Sf, scalar magSf, bool tadmor)
{
    // [0] = _pos, [1] = _neg
    const vector U_pos = in.rhoU[0]/in.rho[0];
    const vector U_neg = in.rhoU[1]/in.rho[1];
    const scalar p_pos = in.rho[0]*in.rPsi[0];
    const scalar p_neg = in.rho[1]*in.rPsi[1];
    scalar phiv_pos = dot(U_pos, Sf);
    scalar phiv_neg = dot(U_neg, Sf);
    const scalar cSf_pos = in.c[0]*magSf;
    const scalar cSf_neg = in.c[1]*magSf;
    const scalar ap = fmaxOF(fmaxOF(phiv_pos + cSf_pos, phiv_neg + cSf_neg), 0.0);
    const scalar am = fminOF(fminOF(phiv_pos - cSf_pos, phiv_neg - cSf_neg), 0.0);
    scalar a_pos = ap/(ap - am);
    scalar amaxSf = fmaxOF(std::fabs(am), std::fabs(ap));
    scalar aSf = am*a_pos;
    if (tadmor)
    {
        aSf = -0.5*amaxSf;
        a_pos = 0.5;
    }
    const scalar a_neg = 1.0 - a_pos;
    phiv_pos *= a_pos;
    phiv_neg *= a_neg;
    const scalar aphiv_pos = phiv_pos - aSf;
    const scalar aphiv_neg = phiv_neg + aSf;
    amaxSf = fmaxOF(std::fabs(aphiv_pos), std::fabs(aphiv_neg));

    FaceOut o;
    o.amaxSf = amaxSf;
    o.phi = aphiv_pos*in.rho[0] + aphiv_neg*in.rho[1];
    const vector phiU = aphiv_pos*in.rhoU[0] + aphiv_neg*in.rhoU[1];
    o.phiUp = phiU + (a_pos*p_pos + a_neg*p_neg)*Sf;
    o.phiEp = aphiv_pos*(in.rho[0]*(in.e[0] + 0.5*magSqr(U_pos)) + p_pos)
            + aphiv_neg*(in.rho[1]*(in.e[1] + 0.5*magSqr(U_neg)) + p_neg)
            + aSf*p_pos - aSf*p_neg;
    return o;
}

} // namespace


StepReport stepCpp(const Case& cs, State& s)
{
    const GasCoeffs& g = cs.gas;
    const label nC = cs.nC, nIF = cs.nIF, nB = cs.nB;
    const std::vector<label>& own = cs.m.owner();
    const std::vector<label>& nei = cs.m.neighbour();
    const std::vector<scalar>& w = cs.g.weights();
    const std::vector<vector>& Sf = cs.g.Sf();
    const std::vector<scalar>& magSf = cs.g.magSf();
    const std::vector<scalar>& V = cs.g.V();
    const std::vector<vector>& C = cs.g.C();
    StepReport rep;

    // ---- rPsi = 1/psi and c = sqrt(Cp/Cv*rPsi), cells and patches (rhoCentralFoam.C:107, 135) -----------------
    const scalar CpByCv = gasCp(g)/gasCv(g);
    std::vector<scalar> rPsi(nC), c(nC), rPsiB(nB), cB(nB);
    for (label i = 0; i < nC; ++i)
    {
        rPsi[i] = 1.0/s.psi[i];
        c[i] = std::sqrt(CpByCv*rPsi[i]);
    }
    for (label k = 0; k < nB; ++k)
    {
        rPsiB[k] = 1.0/s.psiB[k];
        cB[k] = std::sqrt(CpByCv*rPsiB[k]);
    }

    // ---- the limiters' gradients: grad(rho), grad(rhoU), grad(rPsi), grad(e), grad(c) ---------------------------
    std::vector<scalar> rhoUc[3], rhoUbc[3];
    for (int j = 0; j < 3; ++j)
    {
        rhoUc[j].resize(nC);
        rhoUbc[j].resize(nB);
    }
    for (label i = 0; i < nC; ++i)
    {
        rhoUc[0][i] = s.rhoU[i].x;
        rhoUc[1][i] = s.rhoU[i].y;
        rhoUc[2][i] = s.rhoU[i].z;
    }
    for (label k = 0; k < nB; ++k)
    {
        rhoUbc[0][k] = s.rhoUB[k].x;
        rhoUbc[1][k] = s.rhoUB[k].y;
        rhoUbc[2][k] = s.rhoUB[k].z;
    }
    std::vector<vector> gRho, gRhoU[3], gRPsi, gE, gC;
    gaussGrad(cs, s.rho, s.rhoB, gRho);
    for (int j = 0; j < 3; ++j) gaussGrad(cs, rhoUc[j], rhoUbc[j], gRhoU[j]);
    gaussGrad(cs, rPsi, rPsiB, gRPsi);
    gaussGrad(cs, s.e, s.eB, gE);
    gaussGrad(cs, c, cB, gC);

    // ---- directed interpolation and the central-upwind flux, internal faces --------------------------------------
    std::vector<scalar> phi(nIF), phiEp(nIF), amaxSf(nIF);
    std::vector<vector> phiUp(nIF);
    for (label f = 0; f < nIF; ++f)
    {
        const label P = own[f], N = nei[f];
        const vector d = C[N] - C[P];
        const scalar cdw = w[f];
        FaceIn in;
        for (int k = 0; k < 2; ++k)
        {
            const scalar dir = (k == 0) ? 1.0 : -1.0;   // the `pos` and `neg` surface fields, createFields.H
            const scalar wRho = limitedWeight(vanLeerScalar(dir, s.rho[P], s.rho[N], gRho[P], gRho[N], d), cdw, dir);
            in.rho[k] = wRho*(s.rho[P] - s.rho[N]) + s.rho[N];
            const vector gUP[3] = {gRhoU[0][P], gRhoU[1][P], gRhoU[2][P]};
            const vector gUN[3] = {gRhoU[0][N], gRhoU[1][N], gRhoU[2][N]};
            const scalar wU = limitedWeight(vanLeerVector(dir, s.rhoU[P], s.rhoU[N], gUP, gUN, d), cdw, dir);
            in.rhoU[k] = vector{wU*(s.rhoU[P].x - s.rhoU[N].x) + s.rhoU[N].x,
                                wU*(s.rhoU[P].y - s.rhoU[N].y) + s.rhoU[N].y,
                                wU*(s.rhoU[P].z - s.rhoU[N].z) + s.rhoU[N].z};
            const scalar wP = limitedWeight(vanLeerScalar(dir, rPsi[P], rPsi[N], gRPsi[P], gRPsi[N], d), cdw, dir);
            in.rPsi[k] = wP*(rPsi[P] - rPsi[N]) + rPsi[N];
            const scalar wE = limitedWeight(vanLeerScalar(dir, s.e[P], s.e[N], gE[P], gE[N], d), cdw, dir);
            in.e[k] = wE*(s.e[P] - s.e[N]) + s.e[N];
            const scalar wC = limitedWeight(vanLeerScalar(dir, c[P], c[N], gC[P], gC[N], d), cdw, dir);
            in.c[k] = wC*(c[P] - c[N]) + c[N];
        }
        const FaceOut o = faceFlux(in, Sf[f], magSf[f], cs.tadmor);
        phi[f] = o.phi;
        phiUp[f] = o.phiUp;
        phiEp[f] = o.phiEp;
        amaxSf[f] = o.amaxSf;
    }
    // ...boundary faces: an uncoupled patch's interpolate is the patch value in both directions
    std::vector<scalar> phiB(nB, 0), phiEpB(nB, 0), amaxSfB(nB, 0);
    std::vector<vector> phiUpB(nB, vector{0, 0, 0});
    for (label k = 0; k < nB; ++k)
    {
        if (cs.bEmpty[k]) continue;
        FaceIn in;
        for (int j = 0; j < 2; ++j)
        {
            in.rho[j] = s.rhoB[k];
            in.rhoU[j] = s.rhoUB[k];
            in.rPsi[j] = rPsiB[k];
            in.e[j] = s.eB[k];
            in.c[j] = cB[k];
        }
        const FaceOut o = faceFlux(in, Sf[nIF + k], magSf[nIF + k], cs.tadmor);
        phiB[k] = o.phi;
        phiUpB[k] = o.phiUp;
        phiEpB[k] = o.phiEp;
        amaxSfB[k] = o.amaxSf;
        rep.bndMass += o.phi;
        rep.bndMom += o.phiUp;
        rep.bndEnergy += o.phiEp;
    }

    // ---- centralCourantNo.H -----------------------------------------------------------------------------------------
    if (nIF)
    {
        std::vector<scalar> sumA(nC, 0);
        for (label f = 0; f < nIF; ++f)
        {
            sumA[own[f]] += amaxSf[f];
            sumA[nei[f]] += amaxSf[f];
        }
        for (label k = 0; k < nB; ++k)
            if (!cs.bEmpty[k]) sumA[cs.bCell[k]] += amaxSfB[k];
        scalar mx = -1e300, sA = 0, sV = 0;
        for (label i = 0; i < nC; ++i)
        {
            mx = std::max(mx, sumA[i]/V[i]);
            sA += sumA[i];
            sV += V[i];
        }
        rep.CoNum = 0.5*mx*cs.deltaT;
        rep.meanCoNum = 0.5*(sA/sV)*cs.deltaT;
    }

    // ---- fvc::div = surfaceIntegrate (fvcSurfaceIntegrate.C:45-78) ---------------------------------------------------
    std::vector<scalar> divPhi(nC, 0), divPhiEp(nC, 0);
    std::vector<vector> divPhiUp(nC, vector{0, 0, 0});
    for (label f = 0; f < nIF; ++f)
    {
        divPhi[own[f]] += phi[f];
        divPhi[nei[f]] -= phi[f];
        divPhiUp[own[f]] += phiUp[f];
        divPhiUp[nei[f]] -= phiUp[f];
        divPhiEp[own[f]] += phiEp[f];
        divPhiEp[nei[f]] -= phiEp[f];
    }
    for (label k = 0; k < nB; ++k)
    {
        if (cs.bEmpty[k]) continue;
        divPhi[cs.bCell[k]] += phiB[k];
        divPhiUp[cs.bCell[k]] += phiUpB[k];
        divPhiEp[cs.bCell[k]] += phiEpB[k];
    }
    for (label i = 0; i < nC; ++i)
    {
        divPhi[i] /= V[i];
        divPhiUp[i] = divPhiUp[i]/V[i];
        divPhiEp[i] /= V[i];
    }

    // ---- solve(fvm::ddt(q) + fvc::div(flux)), the diagonal solver: x = source/diag with the Euler matrix
    //      diag = rDeltaT*V, source = rDeltaT*q.oldTime()*V - V*div (EulerDdtScheme.C:381-392, fvMatrix.C:2283)
    const scalar rDeltaT = 1.0/cs.deltaT;
    std::vector<scalar> Told = s.T;   // Tw.evaluate() inside the energy BC's updateCoeffs reads T before correct()
    for (label i = 0; i < nC; ++i)
    {
        const scalar diag = rDeltaT*V[i];
        s.rho[i] = (rDeltaT*s.rho[i]*V[i] - V[i]*divPhi[i])/diag;          // rhoCentralFoam.C:224
        // :227, segregated per component; a component knocked out by an empty patch is not solved
        // (fvMatrixSolve.C:157-164) and keeps its value
        if (cs.sd.valid(0)) s.rhoU[i].x = (rDeltaT*s.rhoU[i].x*V[i] - V[i]*divPhiUp[i].x)/diag;
        if (cs.sd.valid(1)) s.rhoU[i].y = (rDeltaT*s.rhoU[i].y*V[i] - V[i]*divPhiUp[i].y)/diag;
        if (cs.sd.valid(2)) s.rhoU[i].z = (rDeltaT*s.rhoU[i].z*V[i] - V[i]*divPhiUp[i].z)/diag;
        s.U[i] = s.rhoU[i]/s.rho[i];                                          // :229-231
    }
    // U.correctBoundaryConditions(); rhoU.boundaryFieldRef() == rho.boundaryField()*U.boundaryField() (:232-233),
    // with rho's patches still last step's (its `calculated` patches are reset at :288 only)
    for (label k = 0; k < nB; ++k)
    {
        const int pi = cs.bPatch[k];
        if (cs.bcU[pi] != bcFixedValue) s.UB[k] = s.U[cs.bCell[k]];
        s.rhoUB[k] = s.rhoB[k]*s.UB[k];
    }
    // inviscid: no viscous corrector (:235-245), and sigmaDotU is identically zero (muEff = 0), so
    // fvc::div(sigmaDotU) adds V*0 to the energy source: not computed
    for (label i = 0; i < nC; ++i)
    {
        const scalar diag = rDeltaT*V[i];
        s.rhoE[i] = (rDeltaT*s.rhoE[i]*V[i] - V[i]*divPhiEp[i])/diag;       // :258-263
        s.e[i] = s.rhoE[i]/s.rho[i] - 0.5*magSqr(s.U[i]);                   // :264
    }
    // e.correctBoundaryConditions() (:265): gradientEnergy (zeroGradient T) first evaluates T's patch from the cell
    // (Tw.evaluate()), and its gradient Cpv*snGrad(T) + deltaCoeffs*(he(pw,Tw) - he(pw,Tw)) is exactly 0, so e_b = e_c;
    // fixedEnergy (fixedValue T) sets e_b = he(p_b, T_b)
    for (label k = 0; k < nB; ++k)
    {
        const int pi = cs.bPatch[k];
        const label ci = cs.bCell[k];
        if (cs.bcT[pi] == bcFixedValue)
        {
            s.eB[k] = gasEs(s.pB[k], s.TB[k], g);
        }
        else
        {
            s.TB[k] = Told[ci];
            s.eB[k] = s.e[ci];
        }
    }
    // thermo.correct() (:266): hePsiThermo::calculate, cells then patches, p still last step's
    for (label i = 0; i < nC; ++i)
    {
        const GasTResult r = gasTEs(s.e[i], s.p[i], s.T[i], g);
        rep.status |= r.status;
        s.T[i] = r.T;
        s.psi[i] = gasPsi(s.p[i], s.T[i], g);
    }
    for (label k = 0; k < nB; ++k)
    {
        const int pi = cs.bPatch[k];
        if (cs.bcT[pi] == bcFixedValue)
        {
            s.eB[k] = gasEs(s.pB[k], s.TB[k], g);
        }
        else
        {
            const GasTResult r = gasTEs(s.eB[k], s.pB[k], s.TB[k], g);
            rep.status |= r.status;
            s.TB[k] = r.T;
        }
        s.psiB[k] = gasPsi(s.pB[k], s.TB[k], g);
        // rhoE.boundaryFieldRef() == rho.boundaryField()*(e.boundaryField() + 0.5*magSqr(U.boundaryField())) (:267-271)
        s.rhoEB[k] = s.rhoB[k]*(s.eB[k] + 0.5*magSqr(s.UB[k]));
    }
    // p.ref() = rho()/psi(); p.correctBoundaryConditions(); rho.boundaryFieldRef() == psi.boundaryField()*p.boundaryField()
    for (label i = 0; i < nC; ++i) s.p[i] = s.rho[i]/s.psi[i];
    for (label k = 0; k < nB; ++k)
    {
        const int pi = cs.bPatch[k];
        if (cs.bcP[pi] != bcFixedValue) s.pB[k] = s.p[cs.bCell[k]];
        s.rhoB[k] = s.psiB[k]*s.pB[k];
    }
    return rep;
}


Integrals integrals(const Case& cs, const State& s)
{
    const std::vector<scalar>& V = cs.g.V();
    Integrals r;
    r.minRho = r.minP = r.minT = r.minE = 1e300;
    r.maxRho = r.maxP = r.maxT = r.maxMagU = -1e300;
    for (label i = 0; i < cs.nC; ++i)
    {
        r.mass += static_cast<long double>(V[i])*s.rho[i];
        r.energy += static_cast<long double>(V[i])*s.rhoE[i];
        r.mom[0] += static_cast<long double>(V[i])*s.rhoU[i].x;
        r.mom[1] += static_cast<long double>(V[i])*s.rhoU[i].y;
        r.mom[2] += static_cast<long double>(V[i])*s.rhoU[i].z;
        r.minRho = std::min(r.minRho, s.rho[i]);
        r.maxRho = std::max(r.maxRho, s.rho[i]);
        r.minP = std::min(r.minP, s.p[i]);
        r.maxP = std::max(r.maxP, s.p[i]);
        r.minT = std::min(r.minT, s.T[i]);
        r.maxT = std::max(r.maxT, s.T[i]);
        r.minE = std::min(r.minE, s.e[i]);
        r.maxMagU = std::max(r.maxMagU, mag(s.U[i]));
    }
    return r;
}


void writeTime(const Case& cs, const State& s, const std::string& timeName)
{
    const std::string src = cs.caseDir + "/" + cs.startName + "/";
    const std::string out = cs.caseDir + "/" + timeName;
    std::filesystem::create_directories(out);
    const int prec = cs.writePrecision;
    // every patch value, in patch order (writeVolField's flat layout; no coupled patch is accepted here)
    writeVolField(src + "U", out + "/U", s.U, cs.patches, prec, s.UB);
    writeVolField(src + "p", out + "/p", s.p, cs.patches, prec, s.pB);
    writeVolField(src + "T", out + "/T", s.T, cs.patches, prec, s.TB);
    // rho is NO_READ, AUTO_WRITE with calculated patches (createFields.H:24-35)
    static const DerivedFieldSpec rhoSpec{"rho", "dimensions      [1 -3 0 0 0 0 0];"};
    writeVolField(src + "T", out + "/rho", s.rho, cs.patches, prec, s.rhoB, &rhoSpec);
}

} // namespace rhoCentral
} // namespace brae
