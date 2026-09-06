// cf gpuPimpleFoam -- transient incompressible PIMPLE solver, single-GPU device-resident. Reads a standard OpenFOAM
// case (controlDict / fvSolution / fvSchemes / transportProperties / turbulenceProperties + start fields) and marches
// the transient loop on the GPU via DeviceSimpleSolver::pimpleStep -- the SAME three composable phases as steady brae
// (momentum predictor / pressure-velocity / turbulence), with the implicit fvm::ddt(U) folded into the predictor.
// Writes standard OpenFOAM time directories.
//
// Scope: Euler/backward/CrankNicolson ddt; laminar, RAS (kEpsilon/realizableKE/kOmegaSST/kOmegaSSTLM/SpalartAllmaras),
// or DES/LES (SA-DDES/IDDES, kOmegaSST-DDES/IDDES, Smagorinsky). Reuses the steady driver's field I/O
// (foam_field_reader/writer) + dict parsing (foam_dict) + turbulence model setup. BOTH the momentum ddt AND the
// turbulence transport ddt (fvm::ddt(k/eps/omega/nuTilda)) are fully implicit + OF-exact -- the turbulence is transient
// URANS/DES, not quasi-steady (see device_simple_foam.cu:294). fvSchemes div/laplacian schemes are parsed; phi output +
// restart, CrankNicolson ddt0 restart, coded (fixedValue/mixed) BCs and forceCoeffs (Cd/Cl/Cm, sampled on the write
// cadence to postProcessing/forceCoeffs/) are wired. NOT yet: MRF, fvOptions,
// adjustTimeStep + maxCo (adaptive dt), a general functionObject framework (forceCoeffs above is hard-wired, NOT
// a framework -- probes/fieldAverage/sampling/surfaces are still absent and are silently ignored), distributed
// (multi-GPU) transient. The first three are REFUSED at start-up rather than silently skipped (see main).
// Kept a SEPARATE executable (brae_pimpleFoam) so it cannot regress the validated steady brae; `brae` hands
// over to it whenever a case's controlDict says `application pimpleFoam` (solvers/common/solver_dispatch.cuh).
#include "primitive_mesh.cuh"
#include "../common/start_time.cuh"   // OF Time::setControls() startFrom, shared
#include "../common/read_surface_field.cuh"   // readSurfaceField / readPhiIfPresent (OF READ_IF_PRESENT)
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "mrf_read.cuh"             // readMRFProperties: only to REFUSE an MRF case (not applied transient yet)
#include "linear_solver_setup.cuh"   // readLinearSolverControls: shared with gpuSimpleFoam/gpuRhoSimpleFoam
#include "dict_audit.cuh"
#include <regex>
#include "scheme_parse.cuh"         // parseFvSchemesControls: shared fvSchemes div/laplacian scheme parse
#include "turbulence_setup.cuh"    // readTurbulenceModel + readTurbulenceFields (shared with brae)
#include "komega_sst_coeffs.cuh"    // readKOmegaSSTCoeffs
#include "fvc.cuh"
#include "forces.cuh"               // wallForces + forceCoeffs (shared with gpuSimpleFoam)
#include "device_simple_foam.cuh"
#include "coded_bc_setup.cuh"       // CodedBCSpec + parseCodedBCs + setupCodedBCs (shared with gpuSimpleFoam)
#include "acmi_area_scaling.cuh"
#include "solid_body_motion.cuh"
#include "velocity_component_laplacian.cuh"   // the SOLVED (Laplace) mesh motion   // OF dynamicMeshDict + solidBody transform
#include "swept_volume.cuh"        // OF meshPhi / makeRelative / movingWallVelocity
#include "time_controls.cuh"   // OF readTimeControls/CourantNo/setInitialDeltaT/setDeltaT
#include "brae_time.cuh"
#include "fan_pressure.cuh"   // fanPressure: the fan curve + direction, read where caseDir is known
#include "scalar_transport_fo.cuh"   // OF functionObjects::scalarTransport, on the device flux   // OF Time/functionObjectList lifecycle, owned centrally (not per solver)
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace brae;

namespace {

std::string timeName(scalar t)
{
    if (t == std::floor(t) && std::fabs((double)t) < 1e15)
        return std::to_string((long long)std::llround((double)t));
    char b[64];
    std::snprintf(b, sizeof b, "%g", (double)t);
    return std::string(b);
}

// fvSchemes holds multi-token scheme values + $-vars (not a plain key/value dict), so text-parse ddtSchemes.default.
DdtScheme parseDdtScheme(const std::string& fvSchemesPath, scalar& ocCoeff)
{
    ocCoeff = 1.0;                                    // CrankNicolson off-centring; default = pure CN (1)
    std::ifstream f(fvSchemesPath);
    if (!f) throw std::runtime_error("cannot read " + fvSchemesPath);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::size_t blk = text.find("ddtSchemes");
    if (blk == std::string::npos)
        throw std::runtime_error("fvSchemes has no ddtSchemes block (transient pimpleFoam needs Euler|backward).");
    std::size_t d = text.find("default", blk);
    if (d == std::string::npos) throw std::runtime_error("fvSchemes ddtSchemes has no 'default' entry.");
    d += 7;
    while (d < text.size() && std::isspace((unsigned char)text[d])) ++d;
    std::string w;
    while (d < text.size() && !std::isspace((unsigned char)text[d]) && text[d] != ';') w += text[d++];
    if (w == "steadyState")
        throw std::runtime_error("fvSchemes ddtSchemes.default = steadyState -> use the steady solver 'brae'.");
    if (w == "backward")                return DdtScheme::backward;
    if (w == "Euler" || w == "bounded") return DdtScheme::Euler;
    if (w == "CrankNicolson")
    {
        // read the off-centring coefficient (OF "CrankNicolson <ocCoeff>", oc in [0,1]); absent -> pure CN (1).
        while (d < text.size() && std::isspace((unsigned char)text[d])) ++d;
        std::string oc;
        while (d < text.size() && !std::isspace((unsigned char)text[d]) && text[d] != ';') oc += text[d++];
        if (!oc.empty()) { try { ocCoeff = std::stod(oc); } catch (...) {} }
        return DdtScheme::CrankNicolson;
    }
    throw std::runtime_error("fvSchemes ddtSchemes.default '" + w + "' unsupported (Euler|backward|CrankNicolson|steadyState).");
}



// CodedBCSpec + parseCodedBCs + setupCodedBCs now live in coded_bc_setup.cuh (shared with gpuSimpleFoam).

void printUsage()
{
    std::printf(
"brae_pimpleFoam, the GPU-native transient incompressible solver (PIMPLE). Normally reached through `brae`,\n"
"which hands over any case whose controlDict says `application pimpleFoam`.\n\n"
"Usage:\n"
"  brae_pimpleFoam [-case <dir>]  march an OpenFOAM case in time (default: the current directory)\n"
"  brae_pimpleFoam <dir>          same, case directory as a positional argument\n"
"  brae_pimpleFoam --help         show this message\n\n"
"Needs ddtSchemes.default = Euler | backward | CrankNicolson and a PIMPLE dict in fvSolution. Writes standard\n"
"OpenFOAM time directories (U, p, phi, turbulence) that restart seamlessly with startFrom latestTime.\n\n"
"Docs: https://github.com/simd-ai/brae/blob/main/docs/solvers/pimplefoam.md\n");
}

}  // namespace

int main(int argc, char** argv)
try
{
    // Same argument surface as the steady driver, so `brae -case <dir>` forwards here verbatim on dispatch.
    std::string caseDir = ".";
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h")        { printUsage(); return 0; }
        else if (a == "-case" && i + 1 < argc) caseDir = argv[++i];
        else if (a[0] != '-')                  caseDir = a;
    }

    // ---- case dictionaries ----
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");

    // As in the other drivers. turbulenceProperties is deliberately NOT registered: this driver reads it
    // inside an `if (exists(...))` block further down, so it would be destroyed before this scope and the
    // audit would report on a dangling dict.
    DictAuditScope audit;
    audit.add(controlDict, "system/controlDict");
    audit.add(fvSolution,  "system/fvSolution");
    audit.add(transport,   "constant/transportProperties");
    audit.addFvSchemes(caseDir);   // reported at scope exit, once every consumer has run
    scalar ddtOcCoeff = 1.0;
    const DdtScheme ddtScheme  = parseDdtScheme(caseDir + "/system/fvSchemes", ddtOcCoeff);

    scalar       startTime     = controlDict.scalarOr("startTime", 0.0);   // resolved below for startFrom latestTime (restart)
    scalar deltaT              = controlDict.scalarOr("deltaT", 1e-3);   // varies when adjustTimeStep
    const scalar endTime       = controlDict.scalarOr("endTime", 1.0);
    const std::string writeControl = controlDict.wordOr("writeControl", "timeStep");
    const scalar writeInterval = controlDict.scalarOr("writeInterval", 1e30);
    const int    purgeWrite    = std::max(0, controlDict.intOr("purgeWrite", 0));
    const int    precision     = controlDict.intOr("writePrecision", 16);
    if (deltaT <= 0)          throw std::runtime_error("controlDict deltaT must be > 0.");
    if (endTime <= startTime) throw std::runtime_error("controlDict endTime must be > startTime.");

    // Physics the STEADY driver applies and this one does not yet: refuse the case rather than march it with the
    // source terms quietly missing (an MRF rotation or an fvOptions source dropped is wrong physics that still
    // produces a plausible-looking field). Same rule the steady driver applies to unsupported fvOptions sources.
    if (readMRFProperties(caseDir + "/constant").active)
        throw std::runtime_error(
            "constant/MRFProperties has an active zone, and brae's transient solver does not apply MRF yet "
            "(the steady solver does). Marching this case would silently drop the rotation.");
    // fvOptions PRE-FLIGHT, before the mesh: refuse what cannot be applied, and any time window brae
    // cannot honour. The sources themselves are read after the geometry (they need cell volumes) and
    // handed to the solver, whose momentum predictor has applied them all along.
    {
        const FvOptionsPreflight pf = preflightFvOptions(caseDir);
        if (!pf.unsupported.empty())
        {
            std::string msg = "fvOptions contains source(s) brae cannot apply (they would be SILENTLY dropped -> wrong physics):";
            for (const auto& u : pf.unsupported) msg += "\n  - " + u;
            throw std::runtime_error(msg);
        }
        for (const auto& w : pf.windows)
            if (w.start > startTime || w.start + w.duration < endTime)
                throw std::runtime_error(
                    "fvOptions source '" + w.name + "' is active only for t in ["
                    + std::to_string(w.start) + ", " + std::to_string(w.start + w.duration)
                    + "], which does not cover this run [" + std::to_string(startTime) + ", "
                    + std::to_string(endTime) + "]. brae applies its fvOptions for the whole run, so it "
                    "would impose this source outside its window. Shorten the run to the window, or "
                    "remove timeStart/duration if the source is meant to be always on.");
    }

    // adjustTimeStep: Courant-limited dt, via the SHARED cfdTools module (OF readTimeControls /
    // CourantNo / setInitialDeltaT / setDeltaT). Previously refused outright, which cost 11 of the 35
    // OpenFOAM pimpleFoam tutorials -- it is the default for transient cases.
    const TimeControls timeControls = TimeControls::read(controlDict);
    // constant/dynamicMeshDict. Absent -> inactive, so every existing case takes the identical path.
    const MeshMotion meshMotion = readMeshMotion(caseDir);
    // ...and the OTHER motion solver brae has: velocityComponentLaplacian, which SOLVES for the motion
    // instead of prescribing it (movingCone's piston). Both readers return inactive on a dict that names
    // the other one, so exactly one of them can be active.
    VelocityComponentLaplacianMotion vclMotion;
    if (std::filesystem::exists(caseDir + "/constant/dynamicMeshDict"))
        vclMotion = readVelocityComponentLaplacian(readDict(caseDir + "/constant/dynamicMeshDict"));


    // ---- PIMPLE + solver controls (fvSolution) ----
    const FoamDict* pimple  = fvSolution.subDict("PIMPLE");
    const int nOuter   = pimple ? std::max(1, pimple->intOr("nOuterCorrectors", 1)) : 1;
    const int nCorr    = pimple ? std::max(1, pimple->intOr("nCorrectors", 1)) : 1;
    // OF pimpleControl.C:55, default true. pEqn.H guards the whole fvc::ddtCorr term with it.
    const std::string ddtCorrW = pimple ? pimple->wordOr("ddtCorr", "yes") : "yes";
    const bool ddtCorrOn = !(ddtCorrW == "no" || ddtCorrW == "false" || ddtCorrW == "off" || ddtCorrW == "0");
    const int nNonOrth = pimple ? pimple->intOr("nNonOrthogonalCorrectors", 0) : 0;
    const scalar nu = transport.scalarOr("nu", 1e-5);

    DeviceSimpleControls ctl;
    ctl.nu = nu;
    // relaxationFactors come from readRelaxationFactors BELOW (with the turbulence model known). Read here
    // it took epsilon/omega's factor from "k" -- `omega 0.4` was ignored -- skipped the legacy flat form,
    // and had no alpha<=0 guard, where OF's fvMatrix::relax skips relaxation for alpha <= 0.
    // tolerances / relTol / smoothSolver selection come from readLinearSolverControls BELOW, once the
    // turbulence model is known (it picks k+omega vs k+epsilon vs nuTilda). Reading them here read only
    // "k" -- a case with a tighter omega tolerance got the loose one -- and never read relTol or
    // smoothSolver at all, so every transient solve was absolute-tolerance BiCGStab whatever fvSolution said.
    ctl.nNonOrth = nNonOrth;
    ctl.caseDir = caseDir;                                          // AMG-hierarchy cache lives in <caseDir>/constant/polyMesh
    ctl.writeCache = std::getenv("BRAE_MESH_CACHE") != nullptr;     // BRAE_MESH_CACHE=1 -> build the AMG hierarchy ONCE, cache it, reload next run
    // (BRAE_AMG_SA=1 enables smoothed aggregation -- the mesh-independent fast path -- read inside buildAMG; when set at
    //  cache-build time the SA hierarchy is what gets cached, so later runs reload the SA hierarchy directly.)
    // fvSchemes div/laplacian/grad scheme flags (2nd-order div(phi,U) linearUpwind, non-orth correction, limitedLinear
    // turbulence scalars, ...). Same parse as steady brae -> the transient solve honours the case's schemes, not upwind.
    parseFvSchemesControls(caseDir, ctl);

    // ---- turbulence model (constant/turbulenceProperties). Absent -> laminar. ----
    std::string secondName = "epsilon";
    if (std::filesystem::exists(caseDir + "/constant/turbulenceProperties"))
    {
        const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType != "RAS" && simType != "laminar" && simType != "LES")
            throw std::runtime_error("pimpleFoam: unsupported simulationType '" + simType + "' (RAS, LES or laminar).");
        ctl.turbulent = (simType == "RAS" || simType == "LES");   // LES here == SA-DDES (a URANS-based hybrid); readTurbulenceModel sets ctl.des
        readTurbulenceModel(turbProps, ctl);
        secondName = ctl.sst ? "omega" : "epsilon";
    }

    // fvSolution -> ctl, through the SHARED reader. Must come AFTER the turbulence model is read: it
    // branches on ctl.turbulent/sa/sst to pick which scalar solver dicts to look at. "PIMPLE" is the
    // algorithm dict -- OF's solutionControl reads consistent/nNonOrthogonalCorrectors from the dict
    // named by algorithmName_, "PIMPLE" here (pimpleControl.H:135), not "SIMPLE".
    readLinearSolverControls(fvSolution, secondName, ctl, "PIMPLE");
    readRelaxationFactors(fvSolution, ctl);

    // ---- mesh + start fields ----
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    // cyclicACMI needs the face areas split by the overlap mask BEFORE cell volumes are computed --
    // the interface's faces are in the mesh twice. buildGeometryPatchesAndAMI is exactly g.build +
    // buildPatches on a mesh without one.
    FvGeometry g;
    std::vector<FvPatch> fvpBuilt;
    { std::vector<AMIInterface> amisInit; buildGeometryPatchesAndAMI(m, g, fvpBuilt, amisInit, startTime); }
    const std::vector<FvPatch> fvp = std::move(fvpBuilt);
    const label nC = m.nCells();
    // startFrom, via the shared resolver -- the same one the other drivers use, and the behaviour OF
    // puts in Time::setControls() (Time.C:149-188).
    //
    // THIS DRIVER'S OWN COPY WAS WRONG. It tested only `latestTime`, so `startFrom firstTime` fell
    // through to startTime and was silently ignored -- OF resolves both. Found by consolidating three
    // hand-written copies of one behaviour, which is the second such defect in this sweep after phi.
    std::string startStr = resolveStartTime(caseDir,
                                            controlDict.wordOr("startFrom", "startTime"),
                                            timeName(startTime));
    startTime = static_cast<scalar>(std::strtod(startStr.c_str(), nullptr));
    const std::string fieldDir = caseDir + "/" + startStr;
    GeometricField<vector> U = buildField<vector>(readField<vector>(fieldDir + "/U"), fvp, nC); U.evaluateBoundary();
    const FieldData<scalar> pFd = readField<scalar>(fieldDir + "/p");
    GeometricField<scalar> p = buildField<scalar>(pFd, fvp, nC); p.evaluateBoundary();
    // phi: on restart READ the previously-written conservative face flux (surfaceScalarField) so the resumed run
    // continues the EXACT flux state (the >=17-digit write makes the phi write->read round-trip bit-identical) -- no
    // first-step continuity transient. The overall restart is seamless to ~1e-10 (nut re-validated at startup, like OF),
    // not literally bit-for-bit. Fresh start / no phi -> recompute from U (fvc::flux).
    SurfaceScalarField phi;
    const std::string phiPath = fieldDir + "/phi";
    if (std::filesystem::exists(phiPath))
    {
        phi = readSurfaceField(phiPath, fvp, m.nInternalFaces());
        std::printf("gpuPimpleFoam: read phi from %s (restart flux)\n", phiPath.c_str());
    }
    else
        phi = fvc::flux(U, m, g, fvp);

    TurbulenceFields tf = readTurbulenceFields(fieldDir, fvp, nC, ctl, secondName, U);

    // Whether the coupled-patch flux above survives into the solver: the ctor rebuilds cyc_/ami_ phi from
    // U (right for a fresh start), so a restart has to put the stored values back over the top of it.
    const bool phiHadCoupledValues = std::filesystem::exists(phiPath);
    // PIMPLE/momentumPredictor: OF assembles UEqn either way (rAU/HbyA come from it) and only skips the
    // SOLVE. Default true, as pimpleControl's.
    {
        const std::string mp = pimple ? pimple->wordOr("momentumPredictor", "yes") : "yes";
        ctl.momentumPredictor = !(mp == "no" || mp == "false" || mp == "off" || mp == "0");
        if (!ctl.momentumPredictor)
            std::printf("  PIMPLE momentumPredictor off: UEqn is assembled (rAU/HbyA) but not solved\n");
    }
    // PIMPLE loop contract (OF pimpleControl::read). Defaults are OpenFOAM's, not brae's convenience:
    // turbOnFinalIterOnly TRUE, solveFlow TRUE, SIMPLErho FALSE.
    {
        auto yes = [](const std::string& v) { return !(v == "no" || v == "false" || v == "off" || v == "0"); };
        ctl.turbOnFinalIterOnly = pimple ? yes(pimple->wordOr("turbOnFinalIterOnly", "yes")) : true;
        ctl.solveFlow           = pimple ? yes(pimple->wordOr("solveFlow", "yes")) : true;
        ctl.simpleRho           = pimple ? yes(pimple->wordOr("SIMPLErho", "no")) : false;
        ctl.moveMeshOuterCorrectors = pimple ? yes(pimple->wordOr("moveMeshOuterCorrectors", "no")) : false;
        if (ctl.moveMeshOuterCorrectors)
            std::printf("  PIMPLE moveMeshOuterCorrectors on: the mesh moves before EVERY outer corrector\n");
        // dynamicFvMesh::controlledUpdate -- OF's timeControl with the "update" prefix. Read here, where
        // ctl exists, rather than beside the motion-solver parse further up.
        if (std::filesystem::exists(caseDir + "/constant/dynamicMeshDict"))
        {
            const FoamDict dmd = readDict(caseDir + "/constant/dynamicMeshDict");
            ctl.meshUpdateControl  = dmd.wordOr("updateControl", "always");
            ctl.meshUpdateInterval = (int)dmd.scalarOr("updateInterval", 1.0);
            if (ctl.meshUpdateControl != "always")
                std::printf("  dynamicMeshDict updateControl %s interval %d\n",
                            ctl.meshUpdateControl.c_str(), ctl.meshUpdateInterval);
        }
        if (!ctl.turbOnFinalIterOnly)
            std::printf("  PIMPLE turbOnFinalIterOnly off: turbulence corrected on EVERY outer corrector\n");
        if (!ctl.solveFlow) std::printf("  PIMPLE solveFlow off: the momentum/pressure solve is skipped\n");
        // residualControl: outer-loop convergence. Parsed as OF does -- each entry is a field name whose
        // sub-dict carries `tolerance` (absolute) and/or `relTol`.
        if (const FoamDict* rc = pimple ? pimple->subDict("residualControl") : nullptr)
        {
            for (const auto& e : rc->subs)
            {
                DeviceSimpleControls::OuterResidualControl o;
                o.field  = e.first;
                o.absTol = e.second.scalarOr("tolerance", 0.0);
                o.relTol = e.second.scalarOr("relTol", 0.0);
                ctl.outerResidualControl.push_back(o);
            }
            if (!ctl.outerResidualControl.empty())
            {
                std::printf("  PIMPLE residualControl:");
                for (const auto& o : ctl.outerResidualControl)
                    std::printf(" %s(tol=%.3g relTol=%.3g)", o.field.c_str(), (double)o.absTol, (double)o.relTol);
                std::printf("\n");
            }
        }
    }
    // PIMPLE/correctPhi: OF createDyMControls.H defaults it to mesh.dynamic(), i.e. ON whenever a motion
    // solver is active. brae mirrors that -- a case that moves its mesh and says nothing gets the
    // projection, as it does in OpenFOAM, and a case that says `correctPhi no` opts out.
    {
        const bool dyn = meshMotion.active || vclMotion.active;
        const std::string cp = pimple ? pimple->wordOr("correctPhi", dyn ? "yes" : "no") : (dyn ? "yes" : "no");
        ctl.correctPhi = dyn && !(cp == "no" || cp == "false" || cp == "off" || cp == "0");
        if (ctl.correctPhi)
            std::printf("  PIMPLE correctPhi on: phi is rebuilt from Uf and projected divergence-free after each mesh move\n");
    }
    // PREFLIGHT the algorithm block before anything is built. Every control brae intends to honour has
    // been read by this point, so whatever is left in PIMPLE is a control it would ignore -- and an
    // ignored algorithm control is a different algorithm, not a smaller one. See
    // preflightAlgorithmControls: this is the check that would have caught turbOnFinalIterOnly.
    preflightAlgorithmControls(fvSolution, "PIMPLE");
    DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl,
                              (ctl.turbulent && !ctl.les) ? &tf.k : nullptr, (ctl.turbulent && !ctl.sa && !ctl.les) ? &tf.eps : nullptr,
                              ctl.turbulent ? &tf.nut : nullptr, ctl.lm ? &tf.ReThetat : nullptr, ctl.lm ? &tf.gammaInt : nullptr);
    solver.setDdtCorr(ddtCorrOn);

    // Maxwell viscoelastic stress: sigma is a TRANSPORTED symmTensor field with its own initial
    // condition, read here and handed to the solver as its six components -- which is how it is solved
    // (OF's fvSymmTensorMatrix is segregated too). Written back at every output time like any other
    // transported field; a restart that resumed a Newtonian sigma = 0 would be a different problem.
    std::vector<GeometricField<scalar>> sigmaComp;
    if (ctl.maxwell)
    {
        const std::string sigPath = fieldDir + "/sigma";
        if (!std::filesystem::exists(sigPath) && !std::filesystem::exists(sigPath + ".gz"))
            throw std::runtime_error(
                "brae: laminar model Maxwell needs a `sigma` field in " + fieldDir +
                ". OpenFOAM reads it MUST_READ for the same reason: the viscoelastic stress is a state "
                "variable, not something the solver can start from nothing.");
        const FieldData<symmTensor> sfd = readField<symmTensor>(sigPath);
        for (int k = 0; k < 6; ++k)
            sigmaComp.push_back(buildField<scalar>(symmTensorComponent(sfd, k), fvp, nC));
        for (GeometricField<scalar>& f : sigmaComp) f.evaluateBoundary();
        solver.setMaxwellSigma(sigmaComp, fvp, g);
    }

    // fvOptions. The momentum sources themselves live in solveMomentumPredictor, which pimpleStep already
    // calls, so this driver only ever had to READ them and hand them over -- it refused instead, which
    // cost pimpleFoam/laminar/planarPoiseuille and pimpleFoam/LES/periodicPlaneChannel.
    //
    // What a transient driver must add over the steady one is the TIME WINDOW. OF's cellSetOption is
    // active only for timeStart <= t <= timeStart + duration; steady has no time and ignores it, but
    // marching a windowed source outside its window is exactly the silent wrong physics the refusal
    // existed to prevent. brae bakes every source into one set of buffers, so it cannot switch an
    // individual one on and off mid-run: a window that does not cover the whole run is refused, and one
    // that does is applied unconditionally, which is the same thing OF computes.
    FvOptionsData fvo;
    {
        std::map<std::string, std::vector<label>> fvoZones;
        {
            std::ifstream a(caseDir + "/system/fvOptions"), b(caseDir + "/constant/fvOptions");
            if (a.good() || b.good()) fvoZones = readCellZones(caseDir + "/constant/polyMesh");
        }
        fvo = readFvOptions(caseDir, fvoZones, g.V(), nC, g.C());
        if (!fvo.unsupported.empty())
        {
            std::string msg = "fvOptions contains source(s) brae cannot apply (they would be SILENTLY dropped -> wrong physics):";
            for (const auto& u : fvo.unsupported) msg += "\n  - " + u;
            throw std::runtime_error(msg);
        }
        for (const auto& w : fvo.windows)
            if (w.start > startTime || w.start + w.duration < endTime)
                throw std::runtime_error(
                    "fvOptions source '" + w.name + "' is active only for t in ["
                    + std::to_string(w.start) + ", " + std::to_string(w.start + w.duration)
                    + "], which does not cover this run [" + std::to_string(startTime) + ", "
                    + std::to_string(endTime) + "]. brae applies its fvOptions for the whole run, so it "
                    "would impose this source outside its window. Shorten the run to the window, or "
                    "remove timeStart/duration if the source is meant to be always on.");
    }

    if (!fvo.empty())
    {
        solver.setFvOptions(fvo);
        if (fvo.rotor.active)
            solver.setRotorDisk(buildDeviceRotorDisk(fvo.rotor, g.C(), g.Sf(), m.owner(), m.neighbour(), m.nInternalFaces()));
        std::printf("  fvOptions: %d source(s)%s%s%s%s%s%s%s\n", fvo.count, fvo.hasMomentum ? " momentum" : "",
                    fvo.porActive ? " DarcyForchheimer-porosity" : "", fvo.mvfActive ? " meanVelocityForce" : "",
                    fvo.limUActive ? " limitVelocity" : "", fvo.adActive ? " actuationDiskSource" : "",
                    fvo.rotor.active ? " rotorDiskSource" : "", fvo.vdcActive ? " velocityDampingConstraint" : "");
    }
    else std::printf("  No finite volume options present\n");

    // OF createUfIfPresent.H -- the face velocity is constructed at case setup when the mesh is dynamic,
    // before any motion. fvc::ddtCorr's Uf form needs it; without motion there is no Uf and OF uses the
    // phi.oldTime() form instead.
    if (meshMotion.active || vclMotion.active) solver.enableUf();
    if (phiHadCoupledValues)
    {
        std::vector<std::pair<label, std::vector<scalar>>> ifPhi;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (isCoupledInterfaceType(fvp[pi].type) && !phi.boundary[pi].empty())
                ifPhi.push_back({(label)pi, phi.boundary[pi]});
        solver.setInterfacePatchFlux(ifPhi);
    }
    // uniformTotalPressure p0(t). OF samples p0_->value(t) at construction with the CURRENT
    // time and again in every updateCoeffs (uniformTotalPressureFvPatchScalarField.C:73,149),
    // so the tables are handed over before the first step and re-evaluated per step inside the
    // solver. Without this the parsed table sat unused and p0 stayed frozen at its seed --
    // measured on pimpleFoam/RAS/TJunction as inlet p FALLING 9.32 -> 8.62 where the table asks
    // for 13.09 -> 15.11, i.e. a case that runs and silently ignores the prescribed ramp.
    solver.setTimeVaryingP0(DeviceSimpleSolver::collectTimeVaryingP0(pFd, fvp));
    // fanPressure: the curve lives in the case (often an external file), so it is read here rather than in
    // the field parser, which has no case directory. Empty for every case that has no fan patch.
    {
        std::vector<DeviceSimpleSolver::FanPressure> fans = collectFanPressure(caseDir, fieldDir, fvp);
        if (!fans.empty())
            std::printf("  fanPressure: %zu patch(es), fan curve with %zu points\n",
                        fans.size(), fans.front().curve.size());
        solver.setFanPressure(std::move(fans));
    }
    {
        std::vector<DeviceSimpleSolver::FixedMean> fm =
            collectFixedMean(fieldDir, fvp, {"U", "k", "epsilon", "omega", "nuTilda", "nut"});
        if (!fm.empty()) std::printf("  fixedMean: %zu pressure patch(es)\n", fm.size());
        solver.setFixedMean(std::move(fm));
    }
    solver.setTime(startTime);   // seed p0 at the START time, as OF's constructor does
    // OF re-evaluates the turbulent-inlet BCs every updateCoeffs; give the solver the per-face masks so it
    // refreshes them each iteration instead of freezing the set-up value.
    solver.setTurbulentInlets(tf.turbInletMasks.tiMask, tf.turbInletMasks.tiIntensity,
                              tf.turbInletMasks.mlMask, tf.turbInletMasks.mlLength);
    solver.setDdtScheme(ddtScheme, ddtOcCoeff);
    // CrankNicolson RESTART: if the previous run wrote the ddt0 fields (present in the restart time dir), reload them and
    // prime the time state so the FIRST resumed step is full CN using the exact stored old ddt -> seamless (like OF's
    // DDt0Field). Fresh start / no ddt0 file -> ddt0 stays 0 and the first step Euler-bootstraps, exactly as before.
    if (ddtScheme == DdtScheme::CrankNicolson && std::filesystem::exists(fieldDir + "/ddt0(U)")) {
        solver.setUddt0(buildField<vector>(readField<vector>(fieldDir + "/ddt0(U)"), fvp, nC).internal);
        if (ctl.turbulent && !ctl.les) {
            const std::string kn = ctl.sa ? "nuTilda" : "k";
            if (std::filesystem::exists(fieldDir + "/ddt0(" + kn + ")"))
                solver.setKddt0(buildField<scalar>(readField<scalar>(fieldDir + "/ddt0(" + kn + ")"), fvp, nC).internal);
            if (!ctl.sa && std::filesystem::exists(fieldDir + "/ddt0(" + secondName + ")"))
                solver.setE2ddt0(buildField<scalar>(readField<scalar>(fieldDir + "/ddt0(" + secondName + ")"), fvp, nC).internal);
            if (ctl.lm) {
                if (std::filesystem::exists(fieldDir + "/ddt0(ReThetat)")) solver.setReThetatddt0(buildField<scalar>(readField<scalar>(fieldDir + "/ddt0(ReThetat)"), fvp, nC).internal);
                if (std::filesystem::exists(fieldDir + "/ddt0(gammaInt)")) solver.setGammaIntddt0(buildField<scalar>(readField<scalar>(fieldDir + "/ddt0(gammaInt)"), fvp, nC).internal);
            }
        }
        solver.setCnRestart(deltaT);
        std::printf("gpuPimpleFoam: read CrankNicolson ddt0 from %s (seamless CN restart)\n", fieldDir.c_str());
    }
    // codedFixedValue / codedMixed on U / p / turbulence scalars -> NVRTC device kernel per coded patch (compiled once;
    // applied each step in the momentum predictor). Runs fully on the GPU -- no host-side per-step BC evaluation.
    setupCodedBCs(solver, fieldDir, fvp, ctl, secondName, "gpuPimpleFoam");

    std::printf("gpuPimpleFoam: deltaT=%g endTime=%g ddt=%s nOuterCorrectors=%d nCorrectors=%d nNonOrth=%d nu=%g "
                "turbulence=%s nCells=%d\n\n",
                (double)deltaT, (double)endTime, ddtScheme == DdtScheme::backward ? "backward" : ddtScheme == DdtScheme::CrankNicolson ? "CrankNicolson" : "Euler",
                nOuter, nCorr, nNonOrth, (double)nu,
                !ctl.turbulent ? "laminar" : ctl.les ? "Smagorinsky (LES)" : ctl.iddes ? (ctl.sst ? "kOmegaSSTIDDES" : "SpalartAllmarasIDDES") : ctl.sa ? "SpalartAllmaras" : ctl.sst ? (ctl.lm ? "kOmegaSSTLM" : "kOmegaSST") : "kEpsilon",
                nC);

    // ---- write cadence (Foam::Time: writeControl timeStep|runTime + writeInterval + purgeWrite FIFO) ----

    // Write cadence from Time's WriteControl, as OF does (TimeIO.C:277). This driver's own
    // isWriteTime lambda was measured identical to the shared one over 8 controlDict cases x 40
    // steps -- including runTime float accumulation and a non-zero-startTime restart -- before
    // being removed. Unlike its startFrom copy, which had silently lost `firstTime`.
    // Everything a functionObject can need already exists at this point in this driver, so the registry
    // is populated immediately. The lookups still happen on first execute() -- that indirection is what
    // lets the steady drivers build Time at start-up, and costs nothing here.
    ObjectRegistry timeRegistry;
    timeRegistry.store("mesh", &m);
    timeRegistry.store("geometry", &g);
    timeRegistry.store("patches", &fvp);
    timeRegistry.store("solver", &solver);

    const std::vector<vector> points0 = m.points();   // OF points0: transformed absolutely each step
    std::vector<label> movingPts;                     // empty => whole mesh (OF moveAllCells_)
    if (meshMotion.active)
    {
        const auto zones = readCellZones(caseDir + "/constant/polyMesh");
        const auto it = zones.find(meshMotion.cellZone);
        if (it == zones.end())
            throw std::runtime_error("brae: dynamicMeshDict cellZone '" + meshMotion.cellZone +
                "' is not in constant/polyMesh/cellZones; brae will not guess which cells move.");
        movingPts = movingPointIDs(m, it->second);
        std::printf("  mesh motion: %zu of %d points move (cellZone '%s', %zu cells)\n",
                    movingPts.size(), (int)m.nPoints(), meshMotion.cellZone.c_str(), it->second.size());
    }
    // velocityComponentLaplacian state: the cell field being solved (its previous solution is the next
    // solve's initial guess, as OF's cellMotionU_ is) and the cell->point weights, which are pure
    // geometry and therefore rebuilt after every move.
    GeometricField<scalar> cellMotionU;
    VolPointInterpolation vpi;
    std::vector<std::pair<label, scalar>> vclConstraints;
    if (vclMotion.active)
    {
        const std::string mpPath = fieldDir + "/" + vclMotion.fieldName;
        if (!std::filesystem::exists(mpPath) && !std::filesystem::exists(mpPath + ".gz"))
            throw std::runtime_error(
                "brae: velocityComponentLaplacian needs the point-motion field '" + vclMotion.fieldName +
                "' in " + fieldDir + " -- it carries the prescribed wall motion, which is the whole "
                "boundary condition of the motion equation.");
        const FieldData<scalar> pmFd = readField<scalar>(mpPath);
        cellMotionU = buildCellMotionField(pmFd, fvp, nC);
        vclConstraints = pointMotionConstraints(m, fvp, pmFd);
        vpi.build(m, g, fvp);
        std::printf("  mesh motion: velocityComponentLaplacian, component %c, diffusivity (%g %g %g)\n",
                    "xyz"[vclMotion.cmpt], vclMotion.diffusivity.x, vclMotion.diffusivity.y,
                    vclMotion.diffusivity.z);
    }
    std::vector<char> mwvPatch(fvp.size(), 0);
    if (meshMotion.active || vclMotion.active)
    {
        // OF's boundaryField keys are keyTypes: a literal name, or a REGEX. The field reader already
        // resolves them that way, so U's patch fields are built correctly -- but this detection compared
        // the dictionary key to the patch name as a plain string, so a case that groups its patches
        //     "(RUBLADE.*|RUHUB)" { type movingWallVelocity; ... }
        // had the right boundary condition on the field and no moving-wall velocity assigned to it.
        // On RAS/axialTurbine that left the rotating runner's blade and hub walls carrying exactly zero
        // flux where OpenFOAM carries +-5e-05 per face, and it is invisible from the outside: the patch
        // type in the written output is right, because the FIELD was right all along.
        //
        // Same rule as FoamDict's lookup: a literal match wins, otherwise the LAST matching regex.
        const FieldData<vector> uFdM = readField<vector>(fieldDir + "/U");
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::string* hit = nullptr;
            for (const auto& b : uFdM.boundary)
                if (b.name == fvp[pi].name) { hit = &b.type; break; }      // literal wins
            if (!hit)
                for (const auto& b : uFdM.boundary)
                {
                    try
                    {
                        if (std::regex_match(fvp[pi].name, compileFoamRegex(b.name))) hit = &b.type;
                    }
                    catch (const std::regex_error&) {}                     // a name that is not a regex
                }
            if (hit && *hit == "movingWallVelocity") mwvPatch[pi] = 1;
        }
        std::printf("  mesh motion: solidBody on cellZone '%s'\n", meshMotion.cellZone.c_str());
    }

    std::vector<ScalarTransportFO*> scalarTransports;
    std::vector<std::pair<std::string, FunctionObjectList::Factory>> foTypes;
    foTypes.emplace_back(
        "scalarTransport",
        [&](const std::string& foName, const FoamDict& fd) -> std::unique_ptr<FunctionObject>
        {
            // Only OF's constant-D branch is implemented; nut-based and alphaD/alphaDt are refused by
            // name rather than quietly replaced by a constant (scalarTransport.C D()).
            if (!fd.found("D"))
            {
                noticeIgnored("functions/" + foName,
                              "scalarTransport without a constant `D` (nut-based or alphaD/alphaDt "
                              "diffusivity) is not implemented, so this tracer is NOT solved.");
                return nullptr;
            }
            const std::string fld = fd.wordOr("field", foName);
            // OF: schemesField defaults to the field name, and the scheme is looked up as
            // div(phi,<schemesField>). Absent under `default none` is fatal in OF, so it is here too --
            // reported and declined rather than run with a substituted discretisation.
            const std::string schemesField = fd.wordOr("schemesField", fld);
            FieldDivScheme scheme;
            try { scheme = parseFieldDivScheme(caseDir, schemesField); }
            catch (const std::exception& e)
            {
                noticeIgnored("functions/" + foName, std::string(e.what()) + " -- this tracer is NOT solved.");
                return nullptr;
            }
            auto fo = std::make_unique<ScalarTransportFO>(
                foName, fld, fieldDir + "/" + fld, timeRegistry,
                fd.scalarOr("D", 0.0), fd.scalarOr("relaxCoeff", 1.0), fd.scalarOr("tol", 1e-6),
                scheme);
            scalarTransports.push_back(fo.get());
            return fo;
        });
    Time time(controlDict, nullptr, 0, foTypes, {}, {"forceCoeffs"});

    auto writeTimeDir = [&](const std::string& tname) {
        const std::string outDir = caseDir + "/" + tname;
        std::filesystem::create_directories(outDir);
        // scalarTransport tracers: OF's transported field is AUTO_WRITE (scalarTransport.C
        // transportedField()), so it appears in every written time directory. AFTER the directory
        // exists, and pulled from the device only here -- on the write cadence, which is the reason OF
        // splits write() from execute().
        for (ScalarTransportFO* st : scalarTransports)
            if (st->ready())
                writeVolField(fieldDir + "/" + st->fieldName(), outDir + "/" + st->fieldName(),
                              st->hostField(), fvp, 12);
        std::error_code ec;
        if (std::filesystem::exists(fieldDir + "/include"))
            std::filesystem::copy(fieldDir + "/include", outDir + "/include",
                std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec);
        writeVolField(fieldDir + "/U", outDir + "/U", solver.U(), fvp, precision, solver.UBoundary());
        writeVolField(fieldDir + "/p", outDir + "/p", solver.p(), fvp, precision, solver.pBoundary());
        // phi is the restart-critical conservative flux -> write it LOSSLESS (>=17 = double max_digits10) so its
        // write->read round-trip is bit-identical (16 sig figs loses the last bit) and a restart resumes the EXACT flux
        // with no continuity transient. The viz fields (U/p/turbulence) keep the user's writePrecision.
        {
            // ...and that includes the COUPLED patches: a cyclic/AMI face's flux is on the interface, not
            // in phiBoundary, so leaving it out made the restart rebuild it from U instead of resuming it.
            std::map<std::string, std::vector<scalar>> ifPhi;
            for (auto& b : solver.interfacePatchFlux())
                if (b.first >= 0 && b.first < (label)fvp.size()) ifPhi[fvp[b.first].name] = std::move(b.second);
            writeSurfaceField(outDir + "/phi", solver.phiInternal(), solver.phiBoundary(), fvp,
                              std::max(precision, 17), "[0 3 -1 0 0 0 0]", ifPhi);
        }
        if (ctl.maxwell)
        {
            const std::vector<std::vector<scalar>> sc = solver.maxwellSigma();
            std::vector<symmTensor> sig(static_cast<std::size_t>(nC));
            for (label c = 0; c < nC; ++c)
                sig[c] = symmTensor{sc[0][c], sc[1][c], sc[2][c], sc[3][c], sc[4][c], sc[5][c]};
            writeVolField(fieldDir + "/sigma", outDir + "/sigma", sig, fvp, precision);
        }
        if (ctl.les) {   // pure LES Smagorinsky: only the algebraic sub-grid nut (no k/epsilon/omega/nuTilda field)
            writeVolField(fieldDir + "/nut",     outDir + "/nut",     solver.nut(), fvp, precision, solver.nutBoundary());
        } else if (ctl.sa) {
            writeVolField(fieldDir + "/nuTilda", outDir + "/nuTilda", solver.k(),   fvp, precision, solver.nuTildaBoundary());
            writeVolField(fieldDir + "/nut",     outDir + "/nut",     solver.nut(), fvp, precision, solver.nutBoundary());
        } else if (ctl.turbulent) {
            writeVolField(fieldDir + "/k",          outDir + "/k",          solver.k(),   fvp, precision, solver.kBoundary());
            writeVolField(fieldDir + "/" + secondName, outDir + "/" + secondName, solver.eps(), fvp, precision, solver.epsBoundary());
            writeVolField(fieldDir + "/nut",        outDir + "/nut",        solver.nut(), fvp, precision, solver.nutBoundary());
            if (ctl.lm) {
                writeVolField(fieldDir + "/ReThetat", outDir + "/ReThetat", solver.ReThetat(), fvp, precision);
                writeVolField(fieldDir + "/gammaInt", outDir + "/gammaInt", solver.gammaInt(), fvp, precision);
            }
        }
        if (solver.isCrankNicolson()) {
            // CrankNicolson stored old ddt (ddt0) per transported variable -> LOSSLESS (>=17) so a restart resumes the
            // EXACT recurrence (seamless CN restart, like OF's registered DDt0Field). No 0/ template -> writeVolFieldRaw.
            const int p17 = std::max(precision, 17);
            writeVolFieldRaw(outDir + "/ddt0(U)", "ddt0(U)", "[0 1 -2 0 0 0 0]", solver.Uddt0(), fvp, p17);
            if (ctl.turbulent && !ctl.les) {
                const std::string kn = ctl.sa ? "nuTilda" : "k";
                writeVolFieldRaw(outDir + "/ddt0(" + kn + ")", "ddt0(" + kn + ")", "[0 0 0 0 0 0 0]", solver.kddt0(), fvp, p17);
                if (!ctl.sa)
                    writeVolFieldRaw(outDir + "/ddt0(" + secondName + ")", "ddt0(" + secondName + ")", "[0 0 0 0 0 0 0]", solver.e2ddt0(), fvp, p17);
                if (ctl.lm) {
                    writeVolFieldRaw(outDir + "/ddt0(ReThetat)", "ddt0(ReThetat)", "[0 0 0 0 0 0 0]", solver.reThetatddt0(), fvp, p17);
                    writeVolFieldRaw(outDir + "/ddt0(gammaInt)", "ddt0(gammaInt)", "[0 0 0 0 0 0 0]", solver.gammaIntddt0(), fvp, p17);
                }
            }
        }
        std::printf("written %s\n", outDir.c_str());
        // purgeWrite, from Time's WriteControl -- measured identical to this driver's copy and
        // purging at the same point (immediately after the write).
        time.writeControl().recordWritten(caseDir, tname);
    };

    // ---- forceCoeffs (Cd/Cl/Cm) sampling -------------------------------------------------------------------
    // The steady solver evaluates forces ONCE at convergence (gpuSimpleFoam.cu). A DES has no converged state:
    // the wake sheds, so Cd(t) oscillates and only its TIME AVERAGE is meaningful. So sample it through the run
    // and stream to postProcessing/forceCoeffs/<startTime>/coefficient.dat, matching OF's layout closely enough
    // that the same plotting scripts work.
    //
    // COST: wallForces runs fvc::gaussGrad over the whole mesh ON THE HOST and calls nearWallDist internally,
    // both mesh-wide and single-threaded. On 6M cells that is seconds per call, so sampling EVERY timestep would
    // dwarf the solve (16 000 calls). It is therefore sampled on the WRITE cadence by default -- 160 samples over
    // this run, ~7 per shedding cycle, enough for a stable mean if not a spectrum. BRAE_FORCE_INTERVAL overrides
    // it (in timesteps) when a finer Cd(t) trace is worth the cost.
    std::vector<std::string> forceWalls;
    for (const auto& q : fvp)
        if (q.type == "wall") forceWalls.push_back(q.name);
    const FoamDict* fcFuncs = controlDict.subDict("functions");
    // Account for EVERY controlDict.functions entry through the shared list. forceCoeffs is declared
    // APPLIED here (not approximated as in gpuSimpleFoam): this driver samples on the write cadence and
    // writes postProcessing/forceCoeffs/<time>/coefficient.dat, which is what OF's forceCoeffs does.
    // Anything else is reported as ignored instead of being silently passed over.
    // Time owns the functionObject lifecycle (OF Foam::Time). forceCoeffs is declared APPLIED here, not
    // approximated: this driver samples on the write cadence and writes
    // postProcessing/forceCoeffs/<time>/coefficient.dat, which is what OF's forceCoeffs does.

    const FoamDict* fcDict = nullptr;
    if (fcFuncs)
        for (const auto& s : fcFuncs->subs)
            if (s.second.wordOr("type", "") == "forceCoeffs") { fcDict = &s.second; break; }
    const long forceInterval = std::getenv("BRAE_FORCE_INTERVAL")
                             ? std::atol(std::getenv("BRAE_FORCE_INTERVAL")) : 0;   // 0 -> follow the write cadence
    std::ofstream fcOut;
    if (fcDict && ctl.turbulent && !forceWalls.empty()) {
        const std::string fdir = caseDir + "/postProcessing/forceCoeffs/" + timeName(startTime);
        std::error_code fec; std::filesystem::create_directories(fdir, fec);
        fcOut.open(fdir + "/coefficient.dat");
        fcOut << "# Time\tCd\tCl\tCm\tFx\tFy\tFz\n";
        fcOut.setf(std::ios::scientific); fcOut.precision(8);
        std::printf("  forceCoeffs: sampling Cd/Cl/Cm -> postProcessing/forceCoeffs/%s/coefficient.dat%s\n",
                    timeName(startTime).c_str(),
                    forceInterval > 0 ? (" every " + std::to_string(forceInterval) + " steps").c_str() : " on the write cadence");
    } else if (fcDict) {
        // Do not let a forceCoeffs block look like it is running when it is not.
        std::printf("  forceCoeffs: present in controlDict but NOT sampled (%s)\n",
                    !ctl.turbulent ? "laminar case" : "no wall patches");
    }

    // Evaluate forces from the CURRENT device state and append one row. Mirrors the steady solver's block
    // (gpuSimpleFoam.cu): pull U/p back, re-evaluate wall BCs, then wallForces on the requested patches.
    auto sampleForces = [&](scalar tval) {
        if (!fcOut.is_open()) return;
        const std::vector<vector> Ug = solver.U();
        const std::vector<scalar> pg = solver.p();
        for (label c = 0; c < nC; ++c) U.internal[c] = Ug[c];
        p.internal = pg;
        U.evaluateBoundary();
        p.evaluateBoundary();
        const scalar wCmu   = ctl.sst ? ctl.ksstCoeffs.betaStar : ctl.keCoeffs.Cmu;
        const scalar wKappa = ctl.sst ? ctl.ksstCoeffs.kappa    : ctl.keCoeffs.kappa;
        const scalar wE     = ctl.sst ? ctl.ksstCoeffs.E        : ctl.keCoeffs.E;
        const bool velNutWall = ctl.sa || ctl.nutWall != NutWall::Nutk;
        const std::vector<scalar> saNutWall = velNutWall ? solver.nutWall() : std::vector<scalar>();
        const std::vector<scalar>* nwb = velNutWall ? &saNutWall : nullptr;
        auto toV = [](const std::vector<scalar>& a, vector d){ return a.size() >= 3 ? vector{a[0],a[1],a[2]} : d; };
        const std::vector<std::string> fcP = fcDict->wordListOr("patches", forceWalls);
        const scalar rhoInf  = fcDict->scalarOr("rhoInf", 1.0), magUInf = fcDict->scalarOr("magUInf", 1.0);
        const scalar Aref    = fcDict->scalarOr("Aref", 1.0),   lRef    = fcDict->scalarOr("lRef", 1.0);
        const vector liftDir = toV(fcDict->scalarListOr("liftDir", {}), vector{0,1,0});
        const vector dragDir = toV(fcDict->scalarListOr("dragDir", {}), vector{1,0,0});
        const vector pitchAx = toV(fcDict->scalarListOr("pitchAxis", {}), vector{0,0,1});
        const vector CofR    = toV(fcDict->scalarListOr("CofR", {}), vector{0,0,0});
        const ForceResult F  = wallForces(U, p, solver.k(), ctl.nu, m, g, fvp, fcP, rhoInf, 0.0, CofR,
                                          wCmu, wKappa, wE, nwb);
        const ForceCoeffs cc = forceCoeffs(F, dragDir, liftDir, pitchAx, rhoInf, magUInf, Aref, lRef);
        const vector T = F.total();
        fcOut << tval << '\t' << cc.Cd << '\t' << cc.Cl << '\t' << cc.Cm << '\t'
              << T.x << '\t' << T.y << '\t' << T.z << '\n';
        fcOut.flush();                       // a long DES should not lose its Cd trace if the run is killed
        std::printf("  forceCoeffs: Cd=%.6e  Cl=%.6e  Cm=%.6e\n", cc.Cd, cc.Cl, cc.Cm);
    };

    // ---- transient time loop ----
    const scalar tEnd = endTime + 0.5 * deltaT;
    long timeIndex = 0;
    std::string lastWritten;
    // Time drives the functionObject lifecycle; the transient loop keeps its own time-valued
    // advancement, which is a genuinely different shape from the steady solvers' iteration index and is
    // not worth restructuring for this. loop() fires start()/execute() at OF's points.
    // A Courant-adapted run has no fixed step count, so cap the loop generously and let `t <= tEnd`
    // end it; with a fixed dt this is the exact count as before.
    {
        // OF Time::run tests `value() < endTime - 0.5*deltaT` and operator++ ACCUMULATES the value
        // (Time.C:785, :1067). The rounded quotient disagrees at ratio n + 0.5: at startTime 0 /
        // endTime 1 / deltaT 0.4 real OpenFOAM runs 2 steps and this spelling gave 3.
        const long nSteps = openFoamNSteps(static_cast<double>(startTime),
                                           static_cast<double>(tEnd),
                                           static_cast<double>(deltaT));
        time.setSteps(static_cast<int>(timeControls.adjustTimeStep ? std::max(nSteps, 1L)*1000 : nSteps));
    }
    // OF setInitialDeltaT: start AT the requested Courant number rather than ramping to it.
    // Device-side, and it INCLUDES the coupled-interface flux the host path silently omitted. Returns
    // two scalars per call instead of copying the whole internal and boundary flux arrays back every
    // step of an adaptive run. See DeviceSimpleSolver::courantNumbers.
    auto courant = [&]() { return solver.courantNumbers(deltaT); };
    if (timeControls.adjustTimeStep)
    {
        const CourantNumbers c0 = courant();
        deltaT = setInitialDeltaT(deltaT, c0.CoNum, timeControls);
        std::printf("Courant Number mean: %g max: %g\ndeltaT = %g\n", c0.meanCoNum, c0.CoNum, deltaT);
    }
    // Does any cyclicACMI carry a time `scale`? Then the geometry is rebuilt every step even on a
    // static mesh; without one, a static mesh keeps the single setup-time build it always had.
    const bool acmiTimeScale = hasACMITimeScale(m);
    for (scalar t = startTime + deltaT; t <= tEnd && time.loop(); t += deltaT) {
        ++timeIndex;
        solver.setTime(t);   // feed the current time to any codedFixedValue snippet's `t`
        // OF pimpleFoam.C:139-160 -- the mesh update is INSIDE the PIMPLE outer loop, guarded by
        // `pimple.firstIter() || moveMeshOuterCorrectors`. So it is a callback handed to pimpleStep
        // rather than a block that runs before it: with moveMeshOuterCorrectors set, every outer
        // corrector re-moves the mesh and re-assembles on the new geometry. points0 (the ORIGINAL
        // positions) are transformed by an absolute function of t, never the current points -- OF's
        // points0MotionSolver does the same, and transforming incrementally would compound round-off.
        //
        // dynamicFvMesh::controlledUpdate gates the whole thing on updateControl/updateInterval; with
        // the default `always` this fires every step, which is what every brae case has used so far.
        const auto meshUpdate = [&](int /*outerIter*/)
        {
        const bool doUpdate =
            (ctl.meshUpdateControl == "always")
         || (ctl.meshUpdateControl == "timeStep"
             && ctl.meshUpdateInterval > 0 && (timeIndex % ctl.meshUpdateInterval) == 0);
        if (!doUpdate) return;
        if (meshMotion.active)
        {
            const std::vector<vector> oldPoints = m.points();
            // Only the zone's points move (OF zoneMotion + solidBodyMotionSolver::curPoints). Moving
            // the whole mesh instead would rotate the stationary half of the domain with the rotor.
            const std::vector<vector> newPoints =
                curPoints(points0, oldPoints, movingPts, meshMotion.motion, t);
            m.movePoints(newPoints);
            rebuildGeometryWithACMI(m, g, fvp, t);        // host geometry (+ACMI split at time t), then device
            const std::vector<scalar> mp =
                solver.moveMesh(m, g, fvp, oldPoints, newPoints, deltaT);

            // movingWallVelocity: Uwall from the motion, written into the patch refValue. OF assigns it
            // in updateCoeffs() at the same point -- after the move, before assembly.
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (!mwvPatch[pi]) continue;
                const std::vector<vector> Uw = movingWallVelocity(
                    m, fvp[pi].start, fvp[pi].size, oldPoints, newPoints, mp, g.Sf(), g.magSf(), deltaT);
                solver.setPatchVelocity(static_cast<label>(pi), Uw);
            }

            // ...and the wall-function geometry, which is a function of the mesh just as much as the AMI
            // weights are. It is rebuilt LAST so it picks up the movingWallVelocity assignment above.
            solver.refreshWallData(m, g, fvp);
            if (ctl.correctPhi) solver.correctPhi(fvp, ctl.nNonOrth);   // OF pimpleFoam.C, after the move
        }
        else if (vclMotion.active)
        {
            // OF pimpleFoam calls mesh.update() at the same place: the motion is SOLVED on the mesh as
            // it stands, then the points move, then everything geometric is rebuilt.
            const std::vector<vector> oldPoints = m.points();
            const std::vector<vector> newPoints =
                velocityComponentLaplacianPoints(vclMotion, m, g, fvp, cellMotionU, vpi, vclConstraints, deltaT);
            m.movePoints(newPoints);
            rebuildGeometryWithACMI(m, g, fvp, t);
            const std::vector<scalar> mp = solver.moveMesh(m, g, fvp, oldPoints, newPoints, deltaT);
            // movingWallVelocity: the wall's own velocity comes from how far ITS faces moved, and on
            // this solver that is the whole driving force -- movingCone has no inlet, the piston IS the
            // flow. Same call the solidBody branch makes, for the same reason.
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (!mwvPatch[pi]) continue;
                const std::vector<vector> Uw = movingWallVelocity(
                    m, fvp[pi].start, fvp[pi].size, oldPoints, newPoints, mp, g.Sf(), g.magSf(), deltaT);
                solver.setPatchVelocity(static_cast<label>(pi), Uw);
            }
            vpi.build(m, g, fvp);          // the weights are geometry: the mesh just changed
            solver.refreshWallData(m, g, fvp);
            // OF pimpleFoam.C: correctPhi runs AFTER the mesh move and after U's boundary has been
            // refreshed -- correctUphiBCs reads the moving-wall velocity this step just assigned.
            if (ctl.correctPhi) solver.correctPhi(fvp, ctl.nNonOrth);
        }
        // A cyclicACMI `scale` makes the interface open area a function of TIME, so it changes even
        // when nothing moves (TJunctionSwitching's mesh is static and its branch closes between
        // t = 0.2 and t = 0.3). Same rebuild as the moving path, with the points held fixed -- so the
        // mesh flux is identically zero and only the interface areas and AMI weights change.
        else if (acmiTimeScale)
        {
            const std::vector<vector> pts = m.points();
            rebuildGeometryWithACMI(m, g, fvp, t);
            solver.moveMesh(m, g, fvp, pts, pts, deltaT);
            solver.refreshWallData(m, g, fvp);
        }
        };   // meshUpdate
        const bool anyMotion = meshMotion.active || vclMotion.active || acmiTimeScale;
        const DeviceSimpleResidual r =
            solver.pimpleStep(deltaT, nOuter, nCorr, anyMotion ? meshUpdate : std::function<void(int)>());
        // OF order: the Courant number is evaluated on the flux the step just produced, and deltaT for
        // the NEXT step follows from it (CourantNo.H then setDeltaT.H, both at the top of the loop).
        if (timeControls.adjustTimeStep)
        {
            const CourantNumbers c = courant();
            const scalar next = setDeltaT(deltaT, c.CoNum, timeControls);
            std::printf("Courant Number mean: %g max: %g\n", c.meanCoNum, c.CoNum);
            deltaT = next;
        }
        const std::string tn = timeName(t);
        std::printf("Time = %s\n  Ux %.3e  Uy %.3e  Uz %.3e  p %.3e  contLocal %.3e  contGlobal %.3e\n",
                    tn.c_str(), r.Ux, r.Uy, r.Uz, r.p, r.contLocal, r.contGlobal);
        if (!std::getenv("BRAE_ALLOW_NONFINITE")
            && !(std::isfinite(r.p) && std::isfinite(r.Ux) && std::isfinite(r.Uy) && std::isfinite(r.Uz)))
            throw std::runtime_error("solution diverged: non-finite residual at Time = " + tn + ". Reduce deltaT (CFL).");
        const bool isWrite = time.writeControl().isWriteTime(timeIndex, t);
        if (isWrite) { writeTimeDir(tn); lastWritten = tn; time.write(); }
        // Sample Cd on the write cadence, or on BRAE_FORCE_INTERVAL when set (see the cost note above).
        if (forceInterval > 0 ? (timeIndex % forceInterval == 0) : isWrite) sampleForces(t);
    }
    time.end();   // OF Time.C:790-802: a final execute() so the last step is seen, then end()
    {
        const std::string tn = timeName(startTime + deltaT * (scalar)timeIndex);
        if (tn != lastWritten) { writeTimeDir(tn); sampleForces(startTime + deltaT * (scalar)timeIndex); }
    }
    return 0;
}
catch (const std::exception& e)
{
    std::fprintf(stderr, "gpuPimpleFoam error: %s\n", e.what());
    return 1;
}
