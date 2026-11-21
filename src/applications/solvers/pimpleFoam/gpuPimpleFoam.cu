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
// restart, CrankNicolson ddt0 restart, coded (fixedValue/mixed) BCs, and the functionObjects forces and forceCoeffs
// (forces_fo.cuh, every step their controls select) are wired. NOT yet: MRF, fvOptions, adjustTimeStep + maxCo (adaptive
// dt), probes/fieldAverage/sampling/surfaces (each named as NOT run), distributed (multi-GPU) transient. The first three
// are REFUSED at start-up rather than silently skipped (see main).
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
#include "pimple_controls.cuh"       // parsePimpleDdtScheme, readPimpleAlgorithm: shared with `brae check`
#include "check_setup.cuh"           // `brae check --setup` stops before the first step
#include "turbulence_setup.cuh"    // readTurbulenceModel + readTurbulenceFields (shared with brae)
#include "solution_health.cuh"     // healthResidual: the turbulence solves' residuals
#include "turb_blowup.cuh"         // TurbBlowup: sum|k| + sum|second field| tripwire
#include "device_kepsilon.cuh"     // turbulenceReport / clearTurbulenceReport
#include "cell_max_delta.cuh"      // IDDES hmax, brae's and OpenFOAM's maxDeltaxyz
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
#include "forces_fo.cuh"   // OF functionObjects::forces and forceCoeffs, every step, on the device
#include "scalar_transport_fo.cuh"   // OF functionObjects::scalarTransport, on the device flux   // OF Time/functionObjectList lifecycle, owned centrally (not per solver)
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace brae;

namespace {

// Time::timeName at the run's timeFormat and timePrecision (start_time.cuh), shared with the writer. This driver's own
// copy formatted %g whatever the case said.
std::string timeName(scalar t)
{
    return openFoamTimeName(static_cast<double>(t));
}

// parsePimpleDdtScheme: pimple_controls.cuh (shared with `brae check`)



// CodedBCSpec + parseCodedBCs + setupCodedBCs now live in coded_bc_setup.cuh (shared with gpuSimpleFoam).

// constant/transportProperties as singlePhaseTransportModel reads it: `transportModel` with no default
// (viscosityModelNew.C:43), and for Newtonian a `nu` with no default (Newtonian.C:56). This driver read `nu` alone, with a
// default of 1e-5, so a non-Newtonian case ran Newtonian -- TJunctionArrheniusBirdCarreauTransport, which has no `nu`, at
// 1e-5 -- and a case without `nu` ran at 1e-5 where OpenFOAM stops.
void refuseTransportBraeDoesNotRun(const FoamDict& transport)
{
    if (!transport.found("transportModel"))
    {
        throw std::runtime_error(
            "brae: constant/transportProperties has no `transportModel`. OpenFOAM reads it with no default "
            "(viscosityModelNew.C:43) and stops. Add `transportModel Newtonian;` and `nu <value>;`.");
    }
    const std::string model = transport.wordOr("transportModel", "");
    if (model != "Newtonian")
    {
        throw std::runtime_error(
            "brae: constant/transportProperties `transportModel " + model + "` is not ported: brae's pimpleFoam "
            "carries one constant laminar viscosity, so it would run this non-Newtonian fluid as a Newtonian one "
            "(and at a made-up nu when the model has no `nu` entry). Only `transportModel Newtonian` runs.");
    }
    if (!transport.found("nu"))
    {
        throw std::runtime_error(
            "brae: constant/transportProperties says `transportModel Newtonian` and has no `nu`. OpenFOAM reads it "
            "with no default (Newtonian.C:56) and stops; brae used to run at 1e-5. Add `nu <value>;`.");
    }
}

// controlDict deltaT and endTime: OpenFOAM reads deltaT with no default (TimeIO.C:272-275); endTime has none under
// `stopAt endTime` (TimeIO.C:342-347), and with no stopAt entry it is 0, so the run takes no step (TimeIO.C:353-356).
// This driver took 1e-3 and 1.0 for them.
void requireTimeEntries(const FoamDict& controlDict)
{
    if (!controlDict.found("deltaT"))
    {
        throw std::runtime_error(
            "brae: system/controlDict has no `deltaT`. OpenFOAM reads it with no default (TimeIO.C:272-275) and stops; "
            "brae used to march at 1e-3. Add `deltaT <value>;`.");
    }
    if (!controlDict.found("endTime"))
    {
        const std::string why = controlDict.found("stopAt")
            ? "OpenFOAM reads it with no default under `stopAt endTime` (TimeIO.C:342-347) and stops"
            : "with no `stopAt` either, OpenFOAM takes endTime 0 and the run takes no step (TimeIO.C:353-356)";
        throw std::runtime_error(
            "brae: system/controlDict has no `endTime`: " + why + "; brae used to run to 1.0. Add `endTime <value>;`.");
    }
}

// constant/turbulenceProperties, which OpenFOAM's TurbulenceModel::New reads MUST_READ with `simulationType` mandatory
// (TurbulenceModel.C:93-106). This driver ran laminar when the file was absent, including a case written for OpenFOAM.org,
// whose model lives in constant/momentumTransport.
FoamDict readTurbulenceProperties(const std::string& caseDir)
{
    const std::string path = caseDir + "/constant/turbulenceProperties";
    if (!std::filesystem::exists(path))
    {
        const bool org = std::filesystem::exists(caseDir + "/constant/momentumTransport");
        throw std::runtime_error(
            "brae: there is no constant/turbulenceProperties. OpenFOAM v2412 reads it MUST_READ (TurbulenceModel.C:93-104) "
            "and stops; brae used to run such a case laminar." + std::string(org
                ? " This case has constant/momentumTransport, OpenFOAM.org's name for it: brae reads the OpenFOAM v2412 "
                  "(ESI) file only. Write constant/turbulenceProperties, for example `simulationType laminar;`."
                : " Add one, for example `simulationType laminar;`."));
    }
    FoamDict d = readDict(path);
    if (!d.found("simulationType"))
    {
        throw std::runtime_error(
            "brae: constant/turbulenceProperties has no `simulationType`. OpenFOAM reads it with no default "
            "(TurbulenceModel.C:106) and stops; brae used to run laminar. Add `simulationType laminar|RAS|LES;`.");
    }
    return d;
}

// The number tokens of a dictionary entry that is a plain constant, parentheses stripped: `6.28`, `(0 0 1)`, and Function1's
// `constant 6.28`. Empty for anything else (a table, a sine, a sub-dictionary), which brae's motion reader does not evaluate.
std::vector<scalar> plainConstant(const std::vector<std::string>& tokens)
{
    std::vector<std::string> w;
    for (const std::string& t : tokens)
    {
        std::string s;
        for (const char ch : t)
        {
            if (ch == '(' || ch == ')')
            {
                s += ' ';
            }
            else
            {
                s += ch;
            }
        }
        std::istringstream in(s);
        for (std::string x; in >> x; )
        {
            w.push_back(x);
        }
    }
    if (!w.empty() && (w[0] == "constant" || w[0] == "uniform"))
    {
        w.erase(w.begin());
    }
    std::vector<scalar> v;
    for (const std::string& x : w)
    {
        char* end = nullptr;
        const double d = std::strtod(x.c_str(), &end);
        if (end == x.c_str() || *end != '\0')
        {
            return {};
        }
        v.push_back(static_cast<scalar>(d));
    }
    return v;
}

// The solid-body motion's entries, where OpenFOAM reads them. solidBodyMotionSolver hands solidBodyMotionFunction::New its
// coeffDict -- solidBodyCoeffs, or dynamicMeshDict itself -- which reads `solidBodyMotionFunction` there and keeps
// SBMFCoeffs_ = that dictionary's optionalSubDict(<function>Coeffs) (solidBodyMotionFunction.C:48-65). rotatingMotion reads
// origin, axis and omega from it with no default (rotatingMotion.C:58-60); oscillatingLinearMotion omega and amplitude with no
// default and phaseShift, verticalShift if present (oscillatingLinearMotion.C:58-80). brae's reader (solid_body_motion.cuh)
// defaults origin (0 0 0), axis (0 0 1) and omega 0, looks further up the dictionary than OpenFOAM does, and evaluates a
// constant only: a missing omega gave a rotor that never turns, a missing axis a rotation about z.
// Called before readMeshMotion, whose constant read stops on a `table` with a bare "stod"; returns at once for anything but a
// solidBody motion solver.
void refuseSolidBodyEntries(const std::string& caseDir)
{
    if (!std::filesystem::exists(caseDir + "/constant/dynamicMeshDict"))
    {
        return;
    }
    const FoamDict d = readDict(caseDir + "/constant/dynamicMeshDict");
    const std::string solver = d.found("motionSolver") ? d.wordOr("motionSolver", "") : d.wordOr("solver", "");
    if (d.wordOr("dynamicFvMesh", "") != "dynamicMotionSolverFvMesh" || solver != "solidBody")
    {
        return;
    }
    const FoamDict* co = d.subDict("solidBodyCoeffs");
    const FoamDict& c = co ? *co : d;
    const std::string where = co ? "dynamicMeshDict/solidBodyCoeffs" : "dynamicMeshDict";
    if (!c.found("solidBodyMotionFunction"))
    {
        throw std::runtime_error(
            "brae: constant/" + where + " has no `solidBodyMotionFunction`. OpenFOAM reads it there with no default "
            "(solidBodyMotionFunctionNew.C:65-77) and stops; brae would have found one elsewhere in the file.");
    }
    const std::string fn = c.wordOr("solidBodyMotionFunction", "");
    const FoamDict* fc = c.subDict(fn + "Coeffs");
    const FoamDict& k = fc ? *fc : c;
    const std::string kWhere = fc ? where + "/" + fn + "Coeffs" : where;
    struct Entry
    {
        const char* name;
        std::size_t size;
        bool required;
    };
    std::vector<Entry> entries;
    if (fn == "rotatingMotion")
    {
        entries = {{"origin", 3, true}, {"axis", 3, true}, {"omega", 1, true}};
    }
    else if (fn == "oscillatingLinearMotion")
    {
        entries = {{"amplitude", 3, true}, {"omega", 1, true}, {"phaseShift", 1, false}, {"verticalShift", 3, false}};
    }
    for (const Entry& e : entries)
    {
        const std::vector<std::string>* v = k.find(e.name);
        if (!v)
        {
            const bool elsewhere = c.found(e.name) || d.found(e.name) || k.subDict(e.name) != nullptr;
            if (k.subDict(e.name) != nullptr)
            {
                throw std::runtime_error(
                    "brae: constant/" + kWhere + " gives `" + e.name + "` as a Function1 sub-dictionary. brae's solid-body "
                    "motion evaluates a constant only, and would read this one as absent (0, or the axis z). Give a "
                    "constant value, or use a case brae runs.");
            }
            if (e.required)
            {
                throw std::runtime_error(
                    "brae: " + fn + " in constant/" + kWhere + " has no `" + e.name + "`" + std::string(elsewhere
                        ? " (it is elsewhere in dynamicMeshDict, where OpenFOAM does not look)" : "")
                    + ". OpenFOAM reads it there with no default and stops; brae would have moved the mesh with a "
                    "default. Add `" + e.name + "` to " + kWhere + ".");
            }
            if (elsewhere)
            {
                throw std::runtime_error(
                    "brae: " + fn + " `" + e.name + "` is in constant/dynamicMeshDict but not in " + kWhere + ", where "
                    "OpenFOAM reads it (OpenFOAM then takes no " + e.name + "; brae would apply this one). Move it into "
                    + kWhere + " or remove it.");
            }
            continue;
        }
        if (plainConstant(*v).size() != e.size)
        {
            std::string val;
            for (const std::string& t : *v)
            {
                val += (val.empty() ? "" : " ") + t;
            }
            throw std::runtime_error(
                "brae: " + fn + " `" + e.name + " " + val + "` in constant/" + kWhere + " is not a constant brae can "
                "evaluate. OpenFOAM reads it as a Function1 of time; brae's solid-body motion takes a constant only. "
                "Give a constant value.");
        }
    }
}

// dynamicFvMesh::controlledUpdate (dynamicFvMesh.C:167-182) moves the mesh when timeControl::execute says so
// (timeControl.C:178-262): `always`; `timeStep` every updateInterval-th time index, every step when the interval is 1 or less;
// `none` never; the others on write times, run time, wall-clock or CPU time, or at the end. The default is `timeStep` with
// interval 0, every step. brae moves on `always`, on `timeStep` with an interval above 0, and never otherwise, so any
// other control, and an interval below 1, would move the mesh on other steps than OpenFOAM's.
void refuseMeshUpdateControl(const std::string& caseDir)
{
    const FoamDict d = readDict(caseDir + "/constant/dynamicMeshDict");
    const std::string control = d.wordOr("updateControl", "timeStep");
    if (control == "always" || control == "none")
    {
        return;
    }
    if (control != "timeStep")
    {
        static const std::vector<std::string> named = {"writeTime", "outputTime", "runTime", "adjustable",
                                                        "adjustableRunTime", "clockTime", "cpuTime", "onEnd"};
        const bool known = std::find(named.begin(), named.end(), control) != named.end();
        throw std::runtime_error(
            "brae: constant/dynamicMeshDict `updateControl " + control + "`" + (known
                ? " is not ported: brae moves the mesh every step (`always`, `timeStep`) or never (`none`), so it would "
                  "move it on other steps than OpenFOAM's timeControl (timeControl.C:178-262)."
                : " is not a timeControl OpenFOAM knows (timeControl.C:38-51); it stops on it.")
            + " Use `timeStep` (with updateInterval) or `always`.");
    }
    if (!d.found("updateInterval"))
    {
        return;
    }
    const std::vector<scalar> v = plainConstant(*d.find("updateInterval"));
    if (v.size() != 1 || v[0] != std::floor(v[0]))
    {
        throw std::runtime_error(
            "brae: constant/dynamicMeshDict `updateInterval` is not an integer. OpenFOAM reads it as a label and stops; "
            "brae would have truncated it.");
    }
    if (v[0] < 1)
    {
        throw std::runtime_error(
            "brae: constant/dynamicMeshDict `updateInterval " + std::to_string(static_cast<long>(v[0])) + "` under "
            "`updateControl timeStep`. OpenFOAM moves the mesh every step for an interval of 1 or less "
            "(timeControl.C:186-191); brae would never move it. Set `updateInterval 1;` or remove it.");
    }
}

// fvSchemes wallDist `updateInterval`: OpenFOAM's wallDist recomputes y after a mesh move when the time index is a multiple
// of it, default 1 (wallDist.C:127, :193-210); 0 or less never after construction.
long wallDistUpdateInterval(const std::string& caseDir)
{
    const std::string blk = fvSchemesBlock(readFvSchemesText(caseDir), "wallDist");
    const std::size_t q = blk.find("updateInterval");
    if (q == std::string::npos)
    {
        return 1;
    }
    const std::size_t e = blk.find(';', q);
    std::istringstream in(blk.substr(q + 14, e == std::string::npos ? std::string::npos : e - q - 14));
    long n = 1;
    in >> n;
    return n;
}

// The earliest time directory of the case holding `U`: what a run that starts from scratch starts from.
double earliestTime(const std::string& caseDir)
{
    double t0 = std::numeric_limits<double>::max();
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(caseDir, ec))
    {
        if (!e.is_directory())
        {
            continue;
        }
        const std::string nm = e.path().filename().string();
        char* end = nullptr;
        const double t = std::strtod(nm.c_str(), &end);
        if (end == nm.c_str() || *end != '\0')
        {
            continue;
        }
        if (std::filesystem::exists(e.path() / "U") || std::filesystem::exists(e.path() / "U.gz"))
        {
            t0 = std::min(t0, t);
        }
    }
    return t0;
}

// The LES and DES forms brae runs differently from OpenFOAM, from the dictionary alone. Each is a different filter width or a
// different model under the case's name, so a refusal, not a notice. The ones that need the mesh are in refuseLesOnMesh.
void refuseLesForms(
    const FoamDict& turbProps,
    const DeviceSimpleControls& ctl)
{
    if (turbProps.wordOr("simulationType", "laminar") != "LES")
    {
        return;
    }
    const FoamDict* les = turbProps.subDict("LES");
    if (!les)
    {
        return;   // readTurbulenceModel refuses it
    }
    const std::string model = ctl.modelName;
    // SpalartAllmarasDES.C and kOmegaSSTDES.C are the plain DES length scales, without the DDES shielding fd; brae's
    // reader takes them into the DDES path (turbulence_setup.cuh: `saDes` and the kOmegaSST branch make no difference).
    if (model == "SpalartAllmarasDES" || model == "kOmegaSSTDES")
    {
        throw std::runtime_error(
            "brae: constant/turbulenceProperties `LESModel " + model + "` is not ported: brae has the delayed variant "
            "only and would run " + (model == "SpalartAllmarasDES" ? "SpalartAllmarasDDES" : "kOmegaSSTDDES")
            + " (with the shielding function, which moves the RANS/LES switch) under this name. Use the DDES model if "
            "that is what is meant.");
    }
    // OpenFOAM's LESdelta::New reads `delta` with no default; brae's reader takes cubeRootVol when it is absent.
    if (!les->found("delta"))
    {
        throw std::runtime_error(
            "brae: constant/turbulenceProperties LES has no `delta`. OpenFOAM reads it with no default (LESdelta.C) and "
            "stops; brae would have taken cubeRootVol. Name the filter width.");
    }
    const std::string delta = les->wordOr("delta", "");
    // Smagorinsky and WALE: lesDelta_ is built only on the DES path (device_simple_foam.cu:786-810), so the algebraic
    // models fall back to cbrt(V) in their kernels (device_smagorinsky.cu:27) while the log says maxDeltaxyz.
    if (ctl.les && delta == "maxDeltaxyz")
    {
        throw std::runtime_error(
            "brae: constant/turbulenceProperties `LESModel " + model + "` with `delta maxDeltaxyz`: brae's algebraic "
            "LES kernels take cbrt(V) (cubeRootVol) whatever the case names, so it would run another filter width "
            "under this one. Use `delta cubeRootVol` on a 3-D mesh, or a DES model.");
    }
    // cubeRootVolDelta.C:96-104 and maxDeltaxyz.C:117-125 read deltaCoeff from optionalSubDict(<type>Coeffs) -- the
    // sub-dictionary when there is one, else LES{} itself -- with defaults 1 and 2. brae's cbrt(V) has no coefficient, and
    // its maxDeltaxyz reader looks in maxDeltaxyzCoeffs alone.
    if (delta == "cubeRootVol" && !ctl.iddes)
    {
        const scalar dc = les->optionalSubDict("cubeRootVolCoeffs")->scalarOr("deltaCoeff", 1.0);
        if (dc != 1.0)
        {
            throw std::runtime_error(
                "brae: constant/turbulenceProperties LES cubeRootVol `deltaCoeff " + std::to_string(dc) + "`: brae's "
                "cubeRootVol is cbrt(V) with a coefficient of 1 (OpenFOAM's default, cubeRootVolDelta.C:96-104) and "
                "does not read it, so the filter width would be off by that factor. Use deltaCoeff 1.");
        }
    }
    if (delta == "maxDeltaxyz" && !ctl.iddes && !les->subDict("maxDeltaxyzCoeffs") && les->found("deltaCoeff"))
    {
        throw std::runtime_error(
            "brae: constant/turbulenceProperties LES `deltaCoeff` with `delta maxDeltaxyz` and no maxDeltaxyzCoeffs: "
            "OpenFOAM reads that deltaCoeff from LES{} itself (optionalSubDict, maxDeltaxyz.C:117-125); brae reads it "
            "from maxDeltaxyzCoeffs only and would take 2. Move it into maxDeltaxyzCoeffs.");
    }
    if (ctl.iddes)
    {
        // SpalartAllmarasIDDES.C:41-50, kOmegaSSTIDDES.C:41-50: the model refuses any delta but IDDESDelta
        if (delta != "IDDESDelta")
        {
            throw std::runtime_error(
                "brae: constant/turbulenceProperties `LESModel " + model + "` with `delta " + delta + "`. OpenFOAM's "
                "IDDES models take IDDESDelta only and stop on any other (SpalartAllmarasIDDES.C:41-50); brae would "
                "have run its own IDDES length scale under this name. Set `delta IDDESDelta;`.");
        }
        // IDDESDelta.C:154-195: Cw and hmax from optionalSubDict(IDDESDeltaCoeffs), Cw 0.15 by default and hmax a
        // maxDeltaxyz built from the same dictionary (deltaCoeff 2 by default) unless `hmax` names another LESdelta.
        const FoamDict* idc = les->optionalSubDict("IDDESDeltaCoeffs");
        const scalar ofCw = idc->scalarOr("Cw", 0.15);
        const scalar braeCw = ctl.sa ? ctl.saCoeffs.Cw : ctl.ksstCoeffs.Cw;
        if (ofCw != braeCw)
        {
            throw std::runtime_error(
                "brae: IDDES Cw: OpenFOAM's IDDESDelta takes Cw " + std::to_string(ofCw) + " from IDDESDeltaCoeffs "
                "(IDDESDelta.C:154-163); brae takes " + std::to_string(braeCw) + " from " + model + "Coeffs. Put the "
                "same Cw in both, or leave both at 0.15.");
        }
        const std::string hmax = idc->wordOr("hmax", "maxDeltaxyz");
        if (hmax != "maxDeltaxyz" && hmax != "maxDeltaxyzCubeRoot")
        {
            throw std::runtime_error(
                "brae: IDDESDeltaCoeffs `hmax " + hmax + "` is not ported: brae's IDDES hmax is the cell's largest "
                "extent (maxDeltaxyz's measure on a box-shaped cell). Use maxDeltaxyz.");
        }
        // which deltaCoeff each part of the hmax delta reads: maxDeltaxyz's from optionalSubDict(maxDeltaxyzCoeffs) of its
        // dictionary, which for maxDeltaxyzCubeRoot is maxDeltaxyzCubeRootCoeffs (maxDeltaxyzCubeRootLESDelta.C:86-99)
        const FoamDict* hd = (hmax == "maxDeltaxyzCubeRoot") ? idc->subDict("maxDeltaxyzCubeRootCoeffs") : idc;
        if (!hd)
        {
            throw std::runtime_error(
                "brae: IDDESDeltaCoeffs `hmax maxDeltaxyzCubeRoot` without maxDeltaxyzCubeRootCoeffs: OpenFOAM reads "
                "that sub-dictionary with no default (maxDeltaxyzCubeRootLESDelta.C:86-99) and stops.");
        }
        const scalar mc = hd->optionalSubDict("maxDeltaxyzCoeffs")->scalarOr("deltaCoeff", 2.0);
        const scalar cc = (hmax == "maxDeltaxyzCubeRoot") ? hd->optionalSubDict("cubeRootVolCoeffs")->scalarOr("deltaCoeff", 1.0)
                                                          : 1.0;
        if (mc != 2.0 || cc != 1.0)
        {
            throw std::runtime_error(
                "brae: the IDDES hmax delta's deltaCoeff is not the default (maxDeltaxyz 2, cubeRootVol 1): brae's hmax "
                "is the cell's largest extent and takes no coefficient. Remove the deltaCoeff entries.");
        }
        // exactDistance: the IDDES wall-normal spacing needs the wall origin only meshWave carries, so the constructor
        // takes meshWave for y whatever fvSchemes says (device_simple_foam.cu:814-818)
        if (ctl.exactWallDist)
        {
            throw std::runtime_error(
                "brae: fvSchemes `wallDist { method exactDistance; }` with " + model + ": brae's IDDES computes the "
                "wall distance with meshWave whatever the method (it needs the wall origin meshWave carries), so y and "
                "the IDDES length scale would differ from OpenFOAM's. Use `method meshWave;`.");
        }
    }
}

// The LES forms that need the mesh: OpenFOAM's 2-D cubeRootVol, and brae's IDDES hmax against OpenFOAM's.
//   BRAE_CONTROL_LES_MESH_UNCHECKED=1: a gate's CONTROL, deliberately wrong -- neither is checked, as before.
void refuseLesOnMesh(
    const FoamDict& turbProps,
    const DeviceSimpleControls& ctl,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    if (turbProps.wordOr("simulationType", "laminar") != "LES" || !ctl.turbulent)
    {
        return;
    }
    if (std::getenv("BRAE_CONTROL_LES_MESH_UNCHECKED") != nullptr)
    {
        std::printf("  *** CONTROL MODE: the LES filter width is not checked against the mesh. This run is deliberately "
                    "wrong. ***\n");
        return;
    }
    const FoamDict* les = turbProps.subDict("LES");
    const std::string delta = les ? les->wordOr("delta", "cubeRootVol") : std::string("cubeRootVol");
    // cubeRootVolDelta.C:53-75: on a mesh with nGeometricD 2 (an `empty` pair; polyMesh::calcDirections counts the empty
    // patches) delta is deltaCoeff*sqrt(V/thickness); brae's kernels take cbrt(V) (device_smagorinsky.cu:27, the DDES
    // length scales likewise)
    if (delta == "cubeRootVol" && !ctl.iddes)
    {
        for (const FvPatch& p : fvp)
        {
            if (p.type == "empty" && p.size > 0)
            {
                throw std::runtime_error(
                    "brae: LES `delta cubeRootVol` on a mesh with an `empty` patch (" + p.name + "). OpenFOAM's cubeRootVol "
                    "is sqrt(V/thickness) on a 2-D mesh (cubeRootVolDelta.C:53-75); brae's is cbrt(V), so the filter "
                    "width would differ. Run LES on a 3-D mesh.");
            }
        }
    }
    if (!ctl.iddes)
    {
        return;
    }
    // brae's IDDES hmax is the cell's vertex bounding box (cell_max_delta.cuh cellMaxDeltaXYZ); OpenFOAM's is maxDeltaxyz,
    // deltaCoeff * the largest |n_f . (C_f - C)| (maxDeltaxyz.C:46-80), maxed with cbrt(V) under maxDeltaxyzCubeRoot. The
    // two agree on a cell whose faces are axis-aligned; on any other they differ, and so does alpha = 0.25 - y/hmax.
    const FoamDict* idc = les->optionalSubDict("IDDESDeltaCoeffs");
    const bool cubeRoot = idc->wordOr("hmax", "maxDeltaxyz") == "maxDeltaxyzCubeRoot";
    const std::vector<scalar> brae = cellMaxDeltaXYZ(m);
    const std::vector<scalar> of = cellMaxDeltaFaceNormal(m, g, 2.0);
    const std::vector<scalar>& V = g.V();
    scalar worst = 0;
    label at = -1;
    for (label c = 0; c < m.nCells(); ++c)
    {
        const scalar ofh = cubeRoot ? std::max(of[c], std::cbrt(V[c])) : of[c];
        const scalar r = std::fabs(brae[c] - ofh) / std::max(ofh, scalar(1e-300));
        if (r > worst)
        {
            worst = r;
            at = c;
        }
    }
    // round-off of two sums of the same coordinates
    if (worst > 1e-10)
    {
        throw std::runtime_error(
            "brae: " + ctl.modelName + " on this mesh: brae's IDDES hmax (the cell's bounding box) differs from "
            "OpenFOAM's " + std::string(cubeRoot ? "maxDeltaxyzCubeRoot" : "maxDeltaxyz") + " by up to "
            + std::to_string(worst * 100) + " % (cell " + std::to_string(at) + "). They agree only where the cells' "
            "faces are aligned with the axes, so the IDDES length scale would differ from OpenFOAM's here.");
    }
}

// The cyclic (plain, non-AMI) interface data -- weights, deltaCoeffs, the face-to-cell vectors, the separation or rotation
// -- is built once in the solver's constructor and never rebuilt on a move (device_simple_foam.cu:3887-3953 rebuilds the AMI
// alone). It stays right only when every cell beside a cyclic face keeps its shape and the two halves move together by a pure
// translation, or do not move. OpenFOAM rebuilds it on every move (cyclicPolyPatch::movePoints).
void refuseMovingCyclics(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& fvp,
    const std::vector<label>& movingPts,
    bool solidBody,
    bool translation)
{
    std::vector<label> cyc;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type == "cyclic")
        {
            cyc.push_back(static_cast<label>(pi));
        }
    }
    if (cyc.empty())
    {
        return;
    }
    if (!solidBody)
    {
        throw std::runtime_error(
            "brae: patch `" + fvp[cyc[0]].name + "` is a plain cyclic on a mesh velocityComponentLaplacian deforms. brae "
            "builds a cyclic's interpolation weights and face vectors once and does not rebuild them after a move, as "
            "OpenFOAM does, so they would be stale. Not ported for a deforming mesh.");
    }
    // the state of each cell beside a cyclic face: 0 static, 1 every point moves, 2 some do (it deforms)
    std::vector<char> moves(static_cast<std::size_t>(m.nPoints()), movingPts.empty() ? 1 : 0);
    for (const label p : movingPts)
    {
        moves[static_cast<std::size_t>(p)] = 1;
    }
    std::vector<std::vector<label>> cellFaces(static_cast<std::size_t>(m.nCells()));
    for (label f = 0; f < m.nFaces(); ++f)
    {
        cellFaces[static_cast<std::size_t>(m.owner()[f])].push_back(f);
        if (f < m.nInternalFaces())
        {
            cellFaces[static_cast<std::size_t>(m.neighbour()[f])].push_back(f);
        }
    }
    auto cellState = [&](label c)
    {
        int nMove = 0;
        int nStay = 0;
        for (const label f : cellFaces[static_cast<std::size_t>(c)])
        {
            for (label k = 0; k < m.faceSize(f); ++k)
            {
                if (moves[static_cast<std::size_t>(m.faceVert(f, k))])
                {
                    ++nMove;
                }
                else
                {
                    ++nStay;
                }
            }
        }
        return nMove == 0 ? 0 : (nStay == 0 ? 1 : 2);
    };
    // every cell beside a cyclic face of one patch, in one state
    auto patchState = [&](label pi)
    {
        int s0 = -1;
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const int s = cellState(m.owner()[fvp[pi].start + i]);
            if (s == 2 || (s == 1 && !translation))
            {
                throw std::runtime_error(
                    "brae: plain cyclic patch `" + fvp[pi].name + "` on a moving mesh: a cell beside it "
                    + (s == 2 ? std::string("deforms") : std::string("rotates")) + " with the motion. brae builds a "
                    "cyclic's weights and face vectors once and does not rebuild them after a move, as OpenFOAM does, "
                    "so they would be stale. Only cyclics whose cells stay still, or translate rigidly together, run.");
            }
            s0 = (s0 < 0 || s0 == s) ? s : 3;
        }
        return s0;
    };
    // ...and the two halves of a pair together: one still and one translating would change the separation between them
    for (const label pi : cyc)
    {
        std::string nbrName;
        for (const PatchInfo& pinfo : m.patches())
        {
            if (pinfo.name == fvp[pi].name)
            {
                nbrName = pinfo.neighbourPatch;
            }
        }
        label nbr = -1;
        for (std::size_t q = 0; q < fvp.size(); ++q)
        {
            if (fvp[q].name == nbrName)
            {
                nbr = static_cast<label>(q);
            }
        }
        const int s0 = patchState(pi);
        const int s1 = nbr >= 0 ? patchState(nbr) : s0;
        if (s0 == 3 || s0 != s1)
        {
            throw std::runtime_error(
                "brae: plain cyclic pair `" + fvp[pi].name + "` / `" + nbrName + "` on a moving mesh: part of it moves "
                "and part does not, so the separation between the halves changes. brae builds it once and does not "
                "rebuild it after a move, as OpenFOAM does.");
        }
    }
}

// A function object brae runs that would be wrong under this run's settings. The FO step context (brae_time.cuh stepContext)
// takes time = startTime + iteration*deltaT with controlDict's deltaT, which an adjustTimeStep run does not keep; and the
// forces' face areas and centres are copied once (forces_fo.cuh:655-690) while a moving mesh moves them.
void refuseFunctionObject(
    const std::string& type,
    const std::string& name,
    bool adjustTimeStep,
    bool moving)
{
    if (moving && (type == "forces" || type == "forceCoeffs"))
    {
        throw std::runtime_error(
            "brae: functions/" + name + " (" + type + ") on a moving mesh: brae's forces take the patch face areas and "
            "centres once at the start and do not follow the mesh, so the forces and moments would be computed on the "
            "start geometry. Remove the function object, or run the static case.");
    }
    if (adjustTimeStep)
    {
        throw std::runtime_error(
            "brae: functions/" + name + " (" + type + ") with `adjustTimeStep yes`: brae's function objects take the time "
            "as startTime + steps*deltaT with controlDict's deltaT, which an adaptive run does not keep, so their time "
            "column, their execute/write controls and timeStart/timeEnd would be evaluated on another clock than the "
            "run's. Remove the function object or run at a fixed deltaT.");
    }
}

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
    setCaseEnvironment(caseDir);   // FOAM_CASE, before any dictionary is read (argList.C:891-893)

    // ---- case dictionaries ----
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
    refuseTransportBraeDoesNotRun(transport);
    // Read here, with the others, so that it outlives the audit below: it was read inside an `if (exists(...))` further
    // down, and so was never audited -- every entry of it this driver did not read went unreported.
    const FoamDict turbProps   = readTurbulenceProperties(caseDir);

    // As in the other drivers.
    DictAuditScope audit;
    audit.add(controlDict, "system/controlDict");
    audit.add(fvSolution,  "system/fvSolution");
    audit.add(transport,   "constant/transportProperties");
    audit.add(turbProps,   "constant/turbulenceProperties");
    audit.addFvSchemes(caseDir);   // reported at scope exit, once every consumer has run
    scalar ddtOcCoeff = 1.0;
    const DdtScheme ddtScheme  = parsePimpleDdtScheme(caseDir + "/system/fvSchemes", ddtOcCoeff);

    requireTimeEntries(controlDict);
    scalar       startTime     = controlDict.scalarOr("startTime", 0.0);   // resolved below for startFrom latestTime (restart)
    scalar deltaT              = controlDict.scalarOr("deltaT", 0.0);   // varies when adjustTimeStep
    const scalar endTime       = controlDict.scalarOr("endTime", 0.0);
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
    // adjustTimeStep runs on brae's own clock, not OpenFOAM's (issue #53, a later work package): said at start-up, not
    // refused (the coordinator's call, pending the user's). The step SEQUENCE differs from the first step, so the transient
    // fields at a given time differ from OpenFOAM's too -- every step solves OpenFOAM's equations, at another size.
    if (timeControls.adjustTimeStep)
    {
        const std::string wc = controlDict.wordOr("writeControl", "timeStep");
        const bool adjustableWrites = (wc == "adjustable" || wc == "adjustableRunTime");
        noticeApproximated(
            "controlDict adjustTimeStep",
            "under `adjustTimeStep yes` the time-step sequence differs from OpenFOAM's, and so do the transient fields at "
            "any given time, not only the write times (issue #53). OpenFOAM applies setDeltaT before every step, the first "
            "included (pimpleFoam.C:132-136)" + std::string(adjustableWrites
                ? ", and under `writeControl " + wc + "` shortens every step so that the steps land on each write time "
                  "(Time::adjustDeltaT, Time.C:102-145); brae does neither"
                : "; brae does not")
            + ". Each brae step solves OpenFOAM's equations, but at brae's own step sizes, so at a given time its fields "
              "differ from OpenFOAM's by the time-integration error of the different steps. The output differs too: "
            + std::string(adjustableWrites ? "its writes land on the first step past each write time instead of on it, " : "")
            + "the run can stop short of endTime, and the last directory is named after startTime + deltaT*steps with "
            "the last deltaT, not the time reached. A fixed deltaT runs on OpenFOAM's clock.");
    }
    // constant/dynamicMeshDict. Absent -> inactive, so every existing case takes the identical path.
    refuseSolidBodyEntries(caseDir);
    const MeshMotion meshMotion = readMeshMotion(caseDir);
    // ...and the OTHER motion solver brae has: velocityComponentLaplacian, which SOLVES for the motion
    // instead of prescribing it (movingCone's piston). Both readers return inactive on a dict that names
    // the other one, so exactly one of them can be active.
    VelocityComponentLaplacianMotion vclMotion;
    if (std::filesystem::exists(caseDir + "/constant/dynamicMeshDict"))
        vclMotion = readVelocityComponentLaplacian(readDict(caseDir + "/constant/dynamicMeshDict"));
    const bool moving = meshMotion.active || vclMotion.active;
    if (moving)
        refuseMeshUpdateControl(caseDir);

    // ---- PIMPLE + solver controls (fvSolution) ----
    // the PIMPLE block, every control the run takes from it (pRefCell/pRefValue apart, read below where a reference is needed),
    // in one reader `brae check` calls too (pimple_controls.cuh); its values go to ctl where they always have
    const FoamDict* pimple  = fvSolution.subDict("PIMPLE");
    const PimpleAlgorithm pimpleAlg = readPimpleAlgorithm(fvSolution, meshMotion.active || vclMotion.active);
    const int nOuter   = pimpleAlg.nOuter;
    const int nCorr    = pimpleAlg.nCorr;
    const bool ddtCorrOn = pimpleAlg.ddtCorr;
    const int nNonOrth = pimpleAlg.nNonOrth;
    // moveMeshOuterCorrectors: OpenFOAM keeps the old points and the old volumes once per time index (polyMesh::movePoints,
    // curMotionTimeIndex_), so every outer corrector moves from the same start. brae's update takes the CURRENT points as
    // the old ones on every call (the meshUpdate lambda below), so from the second corrector on the mesh flux and the
    // moving-wall velocity are zero, and a velocityComponentLaplacian mesh moves again on each corrector.
    if (moving && pimpleAlg.moveMeshOuterCorrectors && nOuter > 1)
        throw std::runtime_error(
            "brae: system/fvSolution PIMPLE `moveMeshOuterCorrectors yes` with nOuterCorrectors " + std::to_string(nOuter)
            + " on a moving mesh is not ported: brae's mesh update does not keep the old points once per time step as "
            "OpenFOAM does, so from the second outer corrector on the mesh flux and the moving-wall velocity would be "
            "zero. Set `moveMeshOuterCorrectors no;` or nOuterCorrectors 1.");
    const scalar nu = transport.scalarOr("nu", 0.0);   // refuseTransportBraeDoesNotRun: present

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

    // ---- turbulence model (constant/turbulenceProperties, read and refused when absent at the top) ----
    std::string secondName = "epsilon";
    {
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType != "RAS" && simType != "laminar" && simType != "LES")
            throw std::runtime_error("pimpleFoam: unsupported simulationType '" + simType + "' (RAS, LES or laminar).");
        ctl.turbulent = (simType == "RAS" || simType == "LES");   // LES here == SA-DDES (a URANS-based hybrid); readTurbulenceModel sets ctl.des
        // Maxwell is applied here (gpuPimpleFoam.cu:377, :657); the generalized-Newtonian viscosity
        // has no consumer on the transient path.
        readTurbulenceModel(turbProps, ctl, {"pimpleFoam", false, true});
        secondName = ctl.sst ? "omega" : "epsilon";
    }
    refuseLesForms(turbProps, ctl);
    // The wall distance on a moving mesh. OpenFOAM's wallDist is a MeshObject that fvMesh::movePoints updates
    // (fvMesh.C:997, wallDist.C:193-210): y is recomputed after every move unless fvSchemes wallDist sets updateInterval 0.
    // brae computes y (and the IDDES hmax and wall-normal spacing, and the maxDeltaxyz LES delta) once, in the solver's
    // constructor (device_simple_foam.cu:786-834), and never again. The models that read y: kOmegaSST (F1, F2, F23),
    // kOmegaSSTLM, Spalart-Allmaras and every DES and IDDES model on them.
    if (moving && ctl.turbulent && (ctl.sst || ctl.sa) && wallDistUpdateInterval(caseDir) > 0)
        throw std::runtime_error(
            "brae: " + ctl.modelName + " on a moving mesh: the model reads the wall distance, which OpenFOAM recomputes "
            "after every mesh move (wallDist.C:193-210) and brae computes once at the start, so F1/F2 (or dTilda) would "
            "be built from the start geometry. Not ported; kEpsilon, RNGkEpsilon, realizableKE and the algebraic LES "
            "models do not read it.");
    // Named ddt entries, for the fields this run integrates (pimple_controls.cuh).
    {
        std::vector<std::string> ddtNames = {"ddt(U)"};
        if (ctl.turbulent && !ctl.les)
        {
            if (ctl.sa)
                ddtNames.push_back("ddt(nuTilda)");
            else
            {
                ddtNames.push_back("ddt(k)");
                ddtNames.push_back("ddt(" + secondName + ")");
            }
            if (ctl.lm)
            {
                ddtNames.push_back("ddt(ReThetat)");
                ddtNames.push_back("ddt(gammaInt)");
            }
        }
        if (ctl.maxwell)
            ddtNames.push_back("ddt(sigma)");
        refuseNamedDdtSchemes(caseDir, ddtNames, ddtScheme, ddtOcCoeff);
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
    const StartTime start = resolveStartTime(caseDir, controlDict);
    std::string startStr = start.name;
    startTime = static_cast<scalar>(start.value);
    const std::string fieldDir = caseDir + "/" + startStr;
    refuseLesOnMesh(turbProps, ctl, m, g, fvp);
    // A MOVING MESH RESTARTED. OpenFOAM reads the mesh at the start time's instance -- <start>/polyMesh/points once the mesh
    // has moved (polyMesh.C, fvMesh::readUpdate) -- with meshPhi and Uf beside it, and continues the motion from there. brae
    // reads constant/polyMesh whatever the start (above), takes points0 from it, and writes no points, meshPhi or Uf: the
    // first restarted step would sweep the zone from its t = 0 position to its position at t (meshPhi and the moving-wall
    // velocity carrying the whole rotation), and a velocityComponentLaplacian mesh would restart undeformed. A start that is
    // not the case's first time, or a start directory carrying polyMesh/points, is a restart.
    if (moving)
    {
        const bool movedPoints = std::filesystem::exists(fieldDir + "/polyMesh/points")
                              || std::filesystem::exists(fieldDir + "/polyMesh/points.gz");
        const double first = earliestTime(caseDir);
        if (movedPoints || start.value > first)
            throw std::runtime_error(
                "brae: this moving-mesh case starts from time " + startStr + (movedPoints
                    ? ", whose directory holds polyMesh/points (a mesh that has already moved)"
                    : ", after its first time directory " + openFoamTimeName(first))
                + ". brae reads the mesh from constant/polyMesh and moves it from there, and writes no polyMesh/points, "
                "meshPhi or Uf, so a restarted run would start from the unmoved mesh where OpenFOAM continues from the "
                "moved one. A moving-mesh restart is not ported: run the case from its first time.");
    }
    GeometricField<vector> U = buildField<vector>(readField<vector>(fieldDir + "/U"), fvp, nC); U.evaluateBoundary();
    const FieldData<scalar> pFd = readField<scalar>(fieldDir + "/p");
    GeometricField<scalar> p = buildField<scalar>(pFd, fvp, nC); p.evaluateBoundary();
    // THE PRESSURE REFERENCE, which this driver never set. pimpleFoam/createFields.H:34-36 runs
    // setRefCell(p, pimple.dict(), pRefCell, pRefValue) and pEqn.H:13-16,46 then drives adjustPhi and
    // pEqn.setReference off p.needReference() -- exactly as simpleFoam does. DeviceSimpleSolver, which
    // this driver constructs with `ctl`, reads ctl_.needRef at all four of those points
    // (device_simple_foam.cu:2347, 2704, 3164-3170, 3214); with the field left at its default a case
    // whose pressure fixes no value -- every closed or wholly periodic domain, nine of OpenFOAM's own
    // pimpleFoam tutorials among them (LES/decayIsoTurb, LES/periodicHill, laminar/mixerVesselAMI2D,
    // laminar/sloshing2D ...) -- would have run its singular all-Neumann system with neither the flux
    // adjustment nor the reference, and said nothing. The PIMPLE preflight whitelists pRefCell and
    // pRefValue on the stated ground that they are "read only when the pressure needs a reference"
    // (dict_audit.cuh:105-107), which was the condition this omission removed. Found by
    // tools/default_audit.py (the field is set at simpleFoam's site and at the rho drivers', not here).
    // Transcribed from gpuSimpleFoam.cu's block so the two drivers decide it the same way.
    ctl.needRef = true;
    for (const auto& bf : p.boundary)
        if (bf->fixesValue() || bf->bcCategory() == 4)   // fixedValue, or freestreamPressure at outflow
        {
            ctl.needRef = false;
            break;
        }
    // ...and the CONTROL this driver's behaviour before the fix, for the gate that holds it
    // (tests/pimple_pressure_reference_vs_openfoam.sh). An instrument, not a mode.
    if (ctl.needRef && std::getenv("BRAE_NO_PREF"))
    {
        ctl.needRef = false;
        std::printf("  *** BRAE_NO_PREF: the pressure reference is SUPPRESSED (a gate's control). ***\n");
    }
    if (ctl.needRef)
    {
        ctl.pRefCell  = pimple ? (label)pimple->intOr("pRefCell", 0) : 0;
        ctl.pRefValue = pimple ? pimple->scalarOr("pRefValue", 0.0) : 0.0;
        std::printf("  pressure needs reference (no fixedValue-p): pRefCell=%d pRefValue=%.4g\n",
                    (int)ctl.pRefCell, ctl.pRefValue);
    }
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
        ctl.momentumPredictor = pimpleAlg.momentumPredictor;
        if (!ctl.momentumPredictor)
            std::printf("  PIMPLE momentumPredictor off: UEqn is assembled (rAU/HbyA) but not solved\n");
    }
    // PIMPLE loop contract (OF pimpleControl::read). Defaults are OpenFOAM's, not brae's convenience:
    // turbOnFinalIterOnly TRUE, solveFlow TRUE, SIMPLErho FALSE.
    {
        ctl.turbOnFinalIterOnly = pimpleAlg.turbOnFinalIterOnly;
        ctl.solveFlow           = pimpleAlg.solveFlow;
        ctl.simpleRho           = pimpleAlg.simpleRho;
        ctl.moveMeshOuterCorrectors = pimpleAlg.moveMeshOuterCorrectors;
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
        {
            for (const PimpleAlgorithm::ResidualControl& e : pimpleAlg.residualControl)
            {
                DeviceSimpleControls::OuterResidualControl o;
                o.field  = e.field;
                o.absTol = e.absTol;
                o.relTol = e.relTol;
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
        ctl.correctPhi = pimpleAlg.correctPhi;
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
        refuseMovingCyclics(m, fvp, movingPts, true,
                            meshMotion.motion.kind == SolidBodyMotion::Kind::OscillatingLinear);
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
        refuseMovingCyclics(m, fvp, {}, false, false);
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
            try { scheme = parseFieldDivScheme(caseDir, schemesField); requireUncappedLaplacian(scheme, schemesField); }
            catch (const std::exception& e)
            {
                noticeIgnored("functions/" + foName, std::string(e.what()) + " -- this tracer is NOT solved.");
                return nullptr;
            }
            refuseFunctionObject("scalarTransport", foName, timeControls.adjustTimeStep, moving);
            // OF scalarTransport.C: fvm::ddt(s) is looked up as ddt(<field>); brae integrates the tracer with the
            // solver's own ddtSchemes.default
            refuseNamedDdtSchemes(caseDir, {"ddt(" + fld + ")"}, ddtScheme, ddtOcCoeff);
            auto fo = std::make_unique<ScalarTransportFO>(
                foName, fld, fieldDir + "/" + fld, timeRegistry,
                fd.scalarOr("D", 0.0), fd.scalarOr("relaxCoeff", 1.0), fd.scalarOr("tol", 1e-6),
                scheme);
            scalarTransports.push_back(fo.get());
            return fo;
        });
    // OF functionObjects::forces and ::forceCoeffs (forces_fo.cuh): run every step their controls select, written under
    // postProcessing/ as OpenFOAM writes them
    const ForcesContext forcesCtx = forcesContext(caseDir, timeRegistry, controlDict);
    foTypes.emplace_back(
        "forces",
        [&](const std::string& foName, const FoamDict& fd) -> std::unique_ptr<FunctionObject>
        {
            std::unique_ptr<FunctionObject> fo = ForcesFO::make(foName, fd, false, forcesCtx);
            if (fo)
                refuseFunctionObject("forces", foName, timeControls.adjustTimeStep, moving);
            return fo;
        });
    foTypes.emplace_back(
        "forceCoeffs",
        [&](const std::string& foName, const FoamDict& fd) -> std::unique_ptr<FunctionObject>
        {
            std::unique_ptr<FunctionObject> fo = ForcesFO::make(foName, fd, true, forcesCtx);
            if (fo)
                refuseFunctionObject("forceCoeffs", foName, timeControls.adjustTimeStep, moving);
            return fo;
        });
    Time time(controlDict, nullptr, 0, foTypes);
    time.setRunStart(startStr, startTime);
    // The write grid continues from the run's start, as OpenFOAM's does: Time::operator++ measures a runTime or
    // adjustableRunTime write index as (value() - startTime_ + 0.5 deltaT)/writeInterval, with startTime_ the RESOLVED start
    // (Time.C:157-181 sets it from startFrom; Time.C:1116-1128 uses it) and writeTimeIndex_ 0 at construction. The
    // WriteControl measured from controlDict's startTime instead, so after a latestTime restart at 0.37 under
    // `writeControl runTime; writeInterval 0.1` the first step was a write (index 3 > 0) and the later ones sat on 0.4, 0.5
    // where OpenFOAM writes 0.47, 0.57. A start that is controlDict's startTime -- every run from scratch -- is unchanged.
    //   BRAE_CONTROL_WRITE_GRID_ABSOLUTE=1: a gate's CONTROL, deliberately wrong -- the grid from controlDict's startTime.
    if (std::getenv("BRAE_CONTROL_WRITE_GRID_ABSOLUTE") == nullptr)
        time.writeControl().setStartTime(startTime);
    else
        std::printf("  *** CONTROL MODE: the write grid is measured from controlDict's startTime. This run is "
                    "deliberately wrong. ***\n");

    auto writeTimeDirBody = [&](const std::string& tname) {
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

    // BRAE_PHASE_TIME: each time directory's write, the benchmark protocol's E (run_timing.cuh)

    const bool timingMaster = true;

    auto writeTimeDir = [&](const std::string& tname)

    {

        if (!RunTiming::on())

        {

            writeTimeDirBody(tname);

            return;

        }

        RunTiming::drain();

        const double w0 = RunTiming::now();

        writeTimeDirBody(tname);

        RunTiming::drain();

        if (timingMaster) std::printf("brae-timing write %s %.6f\n", tname.c_str(), RunTiming::now() - w0);

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
    // THE TURBULENCE FIELDS' HEALTH. The loop below stopped on a non-finite p or U residual only. A k-epsilon or k-omega pair
    // can run away without reaching U: on pitzDaily with linearUpwind on both scalars omega reached 1e42 with k at its floor,
    // nut collapsed, U stayed bounded and every residual finite (turb_blowup.cuh), and a NaN that Foam::bound's max() turns
    // back into the floor never reaches a residual at all. Measured here (tests/pimple_turbulence_health.sh): pitzDaily at a
    // Courant number in the hundreds, whose epsilon solve goes NaN at step 10 while p's and U's residuals are still finite --
    // brae wrote that step with U and p NaN in every cell. OpenFOAM's log has the same epsilon NaN at the same step; with
    // FPE trapping (trapFpe 1) it aborts there on x86, and on aarch64, whose cores do not trap, it writes NaN to the end.
    // So:
    //   every step, before its write: each turbulence solve's residuals finite (solution_health.cuh healthResidual, on the
    //     host values the solves already hold -- free);
    //   every kHealthStride-th step and before every write: sum|k| + sum|second field| (nuTilda for SA) finite, and not
    //     1e12 times its first value for TurbBlowup::kStreak checks in a row (turb_blowup.cuh). Two device sums with a
    //     blocking read each: MEASURED +0.4 to +0.7 ms a step on pitzDaily (12,225 cells, about 12 ms a step, kEpsilon and
    //     kOmegaSST, 400 steps x 3 runs) when taken every step, so they are taken every tenth and at the writes, where
    //     a field must not leave the device non-finite.
    // BRAE_ALLOW_NONFINITE=1 opts out, as for p and U.
    constexpr long kHealthStride = 10;
    TurbBlowup turbBlowup;
    const bool turbHealth = ctl.turbulent && !ctl.les && std::getenv("BRAE_ALLOW_NONFINITE") == nullptr;
    // TEST CONTROL: <field>:<kind>:<time step>, DeviceSimpleSolver::corruptForTest -- one cell of one field, after the step,
    // so the gate can show the check reads what it claims to (tests/pimple_turbulence_health.sh)
    std::string corruptField;
    std::string corruptKind;
    long corruptStep = -1;
    if (const char* e = std::getenv("BRAE_TEST_CORRUPT"))
    {
        const std::string tc(e);
        const std::size_t a = tc.find(':');
        const std::size_t b = tc.rfind(':');
        if (a == std::string::npos || b == a)
            throw std::runtime_error("BRAE_TEST_CORRUPT wants <field>:<kind>:<time step>");
        corruptField = tc.substr(0, a);
        corruptKind = tc.substr(a + 1, b - a - 1);
        corruptStep = std::atol(tc.c_str() + b + 1);
    }
    long healthChecks = 0;
    auto turbSumHealth = [&](long step, const std::string& tn)
    {
        ++healthChecks;
        if (!turbBlowup.update(solver.turbSumMag(), static_cast<int>(healthChecks)))
            return;
        const bool finite = std::isfinite(turbBlowup.last);
        throw std::runtime_error(
            "solution diverged: sum|" + std::string(ctl.sa ? "nuTilda" : "k") + "|" + (ctl.sa ? "" : " + sum|" + secondName + "|")
            + (finite
                ? " has stood above " + std::to_string(static_cast<double>(TurbBlowup::kRatio)) + " times its first value ("
                  + std::to_string(static_cast<double>(turbBlowup.base)) + ", now " + std::to_string(static_cast<double>(turbBlowup.last))
                  + ") for " + std::to_string(turbBlowup.streak) + " checks in a row (every " + std::to_string(kHealthStride)
                  + " steps and at every write) without coming back"
                : " is not finite (" + std::to_string(static_cast<double>(turbBlowup.last)) + ")")
            + " at time step " + std::to_string(step) + ", Time = " + tn + ". This step is not written; earlier writes are "
            "kept. Set BRAE_ALLOW_NONFINITE=1 to continue anyway.");
    };
    // Does any cyclicACMI carry a time `scale`? Then the geometry is rebuilt every step even on a
    // static mesh; without one, a static mesh keeps the single setup-time build it always had.
    const bool acmiTimeScale = hasACMITimeScale(m);
    // `brae check --setup`: everything above is the run's own setup; stop before its first step, with the Courant numbers the
    // first step starts from at the case's deltaT
    if (cli::checkSetupDone("pimpleFoam"))
    {
        const CourantNumbers c = courant();
        std::printf("brae check --setup: Courant number at the start, deltaT %g: mean %g max %g\n", deltaT, c.meanCoNum, c.CoNum);
        return 0;
    }
    // BRAE_PHASE_TIME: start-up to here (the benchmark protocol's B) and the loop's wall time (run_timing.cuh)
    double loopStart = 0;
    if (RunTiming::on())
    {
        RunTiming::drain();
        loopStart = RunTiming::now();
        if (timingMaster) std::printf("brae-timing startup %.6f\n", RunTiming::sinceProcessStart());
    }
    time.setReportTiming(timingMaster);
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
        clearTurbulenceReport();   // this step's turbulence solves only (read by the health check below)
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
        if (timeIndex == corruptStep)
            solver.corruptForTest(corruptField, corruptKind);
        if (turbHealth)
        {
            std::string why;
            for (const ScalarSolveEntry& e : turbulenceReport())
            {
                if (why.empty())
                    why = healthResidual(e.field, e.perf.initialResidual, static_cast<int>(timeIndex));
                if (why.empty())
                    why = healthResidual(e.field + " (final)", e.perf.finalResidual, static_cast<int>(timeIndex));
            }
            if (!why.empty())
                throw std::runtime_error("solution diverged: " + why + " (time step " + std::to_string(timeIndex)
                                         + ", Time = " + tn + "). This step is not written. Set BRAE_ALLOW_NONFINITE=1 "
                                         "to continue anyway.");
        }
        const bool isWrite = time.writeControl().isWriteTime(timeIndex, t);
        if (turbHealth && (isWrite || timeIndex % kHealthStride == 0))
            turbSumHealth(timeIndex, tn);
        if (isWrite) { writeTimeDir(tn); lastWritten = tn; time.write(); }
    }
    // brae writes the final state below when the cadence did not; OpenFOAM does not, so it is not noted as a write time
    const std::string finalName = timeName(startTime + deltaT * (scalar)timeIndex);
    time.end();   // OF Time.C:790-802: a final execute() so the last step is seen, then end()
    if (RunTiming::on())
    {
        RunTiming::drain();
        if (timingMaster) std::printf("brae-timing loop %.6f %ld\n", RunTiming::now() - loopStart, static_cast<long>(timeIndex));
    }
    if (finalName != lastWritten)
    {
        if (turbHealth)
            turbSumHealth(timeIndex, finalName);
        writeTimeDir(finalName);
    }
    return 0;
}
catch (const std::exception& e)
{
    std::fprintf(stderr, "gpuPimpleFoam error: %s\n", e.what());
    return 1;
}
