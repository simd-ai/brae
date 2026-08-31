// cf gpuSimpleFoam, steady incompressible SIMPLE solver, single-GPU device-resident. Reads the case dicts
// (controlDict / fvSolution / transportProperties / turbulenceProperties) and start fields, runs the whole
// SIMPLE(+kEpsilon) loop on the GPU via DeviceSimpleSolver (U/p/phi/k/eps/nut never leave the device between
// iterations), and writes the converged fields when fvSolution residualControl is met or controlDict endTime
// is reached. The momentum is the faithful incompressible stress div(phi,U)-laplacian(nuEff,U)-div(nuEff*
// dev2(T(grad U))); the pressure uses the device AMG-PCG. Mirrors the host brae_simpleFoam control flow.
//
//   brae -case <caseDir>
#include "primitive_mesh.cuh"
#include "acmi_area_scaling.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "forces.cuh"
#include "mrf_read.cuh"
#include "fv_options.cuh"
#include "turbulent_inlet.cuh"
#include "turb_blowup.cuh"
#include "foam_dict.cuh"
#include "dict_audit.cuh"
#include "scheme_parse.cuh"
#include "linear_solver_setup.cuh"   // readLinearSolverControls (shared with gpuRhoSimpleFoam)   // parseFvSchemesControls: shared fvSchemes div/laplacian scheme parse (steady + transient)
#include "solver_dispatch.cuh"   // dispatchSolver + execSibling: route to the solver / component that owns the work
#include "simpleFoamV2.cuh"      // the rebuilt path + its envelope guard (BRAE_SIMPLEFOAM_V2)
#include "benchmark.cuh"         // brae benchmark [sample]: the standard workload, pulled from the template repo
#include "turbulence_setup.cuh"   // readTurbulenceModel + readTurbulenceFields (shared with pimpleFoam)
#include "sweep_cases.cuh"   // brae -cases c1 c2 ...: multi-GPU mesh/parameter study (orchestrator mode)
#include "../common/read_surface_field.cuh"   // OF READ_IF_PRESENT for phi on restart
#include "fvc.cuh"
#include "device_simple_foam.cuh"
#include "coded_bc_setup.cuh"         // CodedBCSpec + parseCodedBCs + setupCodedBCs (shared with gpuPimpleFoam)
#include "frozen_bc_guard.cuh"
#include "brae_time.cuh"
#include "scalar_transport_fo.cuh"   // OF functionObjects::scalarTransport, on the device flux   // OF Time/functionObjectList lifecycle, owned centrally (not per solver)

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <deque>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace brae;

static void printUsage()
{
    std::printf(
"brae, a GPU-native, OpenFOAM-compatible CFD solver. The whole solve runs on one GPU; reads a standard\n"
"OpenFOAM case and writes standard time dirs.\n\n"
"Solvers (picked from the case's controlDict `application`, so `brae` is the only command you type):\n"
"  simpleFoam       steady incompressible, RAS/laminar\n"
"  pimpleFoam       transient incompressible, URANS/DES/LES/laminar\n"
"Any other application stops at start-up rather than run the case with the wrong solver.\n\n"
"Subcommands (the only two reserved words; anything else leading is a case directory):\n"
"  brae benchmark [sample]        run the standard workload, writes brae-benchmark.json\n"
"  brae benchmark --list          the samples published at github.com/simd-ai/brae-bench\n"
"  brae node register|status|unregister\n"
"                                 join this machine to the Brae network (needs brae-agent)\n"
"A case directory called 'node' or 'benchmark' is addressed as `brae -case node`.\n\n"
"Usage:\n"
"  brae [-case <dir>]             solve an OpenFOAM case (default: the current directory)\n"
"  mpirun -np N brae -case <dir> -parallel\n"
"                                 solve across N GPUs, one rank per GPU (laminar only; needs P2P/NVLink).\n"
"                                 Decomposes in core and reconstructs on write -- no decomposePar step.\n"
"  brae -partition [-case <dir>]  build + cache the mesh and AMG hierarchy, then exit (no solve)\n"
"  brae -cases <d1> <d2> ...      run several cases at once, one per GPU (mesh/parameter study)\n"
"  brae --help                    show this message\n\n"
"A case is a standard OpenFOAM directory (0/ constant/ system/, ASCII or binary mesh). No decomposePar\n"
"needed; brae auto-partitions for the GPU. With -cases, extra cases queue as GPUs free up.\n\n"
"Environment:\n"
"  BRAE_GPUS=N        override the detected GPU count (for -cases)\n"
"  BRAE_JOBS=N        how many cases to run at once with -cases (default: number of GPUs)\n"
"  BRAE_PCG_DEVICE=0  disable the device-resident PCG (on by default)\n"
"  BRAE_AMG_FP32=0    use the FP64 AMG preconditioner instead of FP32 (on by default)\n\n"
"Docs and benchmarks: https://github.com/simd-ai/brae\n");
}

int main(int argc, char** argv)
{
    try
    {
        // Subcommands. ONLY these three leading words are reserved -- everything else is a case directory, so
        // `brae myCase` is untouched. A case actually named `node`, `benchmark` or `job` is reached with -case.
        if (argc > 1 && argv[1][0] != '-')
        {
            const std::string word = argv[1];
            if (word == "benchmark")
            {
                std::vector<std::string> rest(argv + 2, argv + argc);
                std::error_code ec;
                const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
                return bench::runBenchmark(rest, ec ? std::string("brae") : self.string());
            }
            if (word == "node")
            {
                // The node service is a separate, CUDA-free binary; `brae` is only the front door to it.
                std::vector<std::string> args(argv + 1, argv + argc);
                execSibling("brae-agent", args, "node subcommand -> brae-agent", "`brae node` (the node service)");
            }
            if (word == "job")
            {
                // `brae job run`, not `brae run`. A bare `run` was tried and tests/brae_subcommands.sh
                // refused it, correctly: a case directory called `run` is commonplace, and the rule here is
                // that only reserved words are subcommands and everything else is a case. Stealing `run`
                // would have made `brae run` mean different things in different directories.
                std::vector<std::string> args(argv + 1, argv + argc);
                execSibling("brae-agent", args, "job subcommand -> brae-agent", "`brae job` (submitting work)");
            }
        }
        for (int i = 1; i < argc; ++i)
        {
            const std::string h = argv[i];
            if (h == "--help" || h == "-h")
            {
                printUsage();
                return 0;
            }
        }
        // -cases c1 c2 ... : run several cases at once, one per GPU (mesh/parameter study). The parent
        // orchestrates (forks one child `brae -case cX` per GPU, no CUDA here); a plain -case is unaffected.
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "-cases")
            {
                std::vector<std::string> sweep;
                for (int j = i + 1; j < argc && argv[j][0] != '-'; ++j)
                    sweep.push_back(argv[j]);
                if (!sweep.empty()) return braesweep::runSweepCases(sweep);
            }
        // -parallel (the OpenFOAM convention): the distributed DEVICE path, one rank per GPU.
        //   mpirun -np N brae -case <dir> -parallel
        // Gated on the flag rather than on Pstream::nProcs() so a plain `brae -case ...` never initialises
        // MPI -- which also keeps the -cases fork orchestrator above clear of it.
        // -parallel (multi-GPU) is OUT OF SCOPE; the distributed solver lives in legacy/ and is not built.
        // REFUSE rather than fall through to the single-GPU path: under `mpirun -np N` that would run N
        // redundant identical solves, each writing over the others' time directories.
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "-parallel")
                throw std::runtime_error(
                    "brae: -parallel (multi-GPU) is not supported in this build. The distributed solver was "
                    "moved to legacy/ and is out of scope; see legacy/README.md. Run brae single-GPU without "
                    "-parallel (and without mpirun).");
        std::string caseDir = ".";
        bool partition = false;                              // -partition: build mesh + AMG caches, then exit (no solve)
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "-case" && i + 1 < argc)
                caseDir = argv[++i];
            else if (a == "-partition")
                partition = true;
            else if (a[0] != '-')
                caseDir = a;
        }
        // The case names its solver (controlDict `application`), so `brae` is the only command a user types: this
        // executable keeps the steady cases and hands the others to the brae solver that owns them, before any dict
        // read or CUDA init. -partition is solver-agnostic prep (mesh + AMG cache) and always runs here.
        // Registry + rules: solvers/common/solver_dispatch.cuh.
        if (!partition) dispatchSolver(caseDir, argc, argv);

        // The REBUILT simpleFoam (UEqn.cu + pEqn.cu + simpleFoam.cu), selected with BRAE_SIMPLEFOAM_V2=1.
        // It covers a strict subset of what the code below runs, so it is opt-in -- but once selected it
        // either runs the case or REFUSES with the reason. It must never quietly fall through to the old
        // solver: a user who asked for the new path and silently got the old one cannot tell from the
        // output which algorithm produced their answer, and that is the failure mode this rebuild exists
        // to remove. Hence no try/catch here.
        if (!partition && gpu::simpleFoamV2Selected())
        {
            gpu::runSimpleFoamV2(caseDir);
            return 0;
        }

        // -partition is cf's analogue of OF decomposePar: do the one-time prep (parse mesh + build AMG hierarchy) and
        // persist it to constant/polyMesh/.brae_mesh|amgcache, so the actual run reloads it warm. Forces the cache write.
        if (partition) setenv("BRAE_MESH_CACHE", "1", 1);

        // controls from the case dictionaries
        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
        const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
        const FoamDict turbProps   = readDict(caseDir + "/constant/turbulenceProperties");

        // Report every dictionary entry read off disk and then ignored, on EVERY exit including a
        // refusal. Declared AFTER the dicts so it is destroyed FIRST -- it holds pointers to them.
        // Until now this ran only in gpuRhoSimpleFoam, so the incompressible solvers had no unread-entry
        // reporting at all: an input this driver parsed and never applied was invisible.
        DictAuditScope audit;
        audit.add(controlDict, "system/controlDict");
        audit.add(fvSolution,  "system/fvSolution");
        audit.add(transport,   "constant/transportProperties");
        audit.add(turbProps,   "constant/turbulenceProperties");
        audit.addFvSchemes(caseDir);   // reported at scope exit, once every consumer has run

        // Account for controlDict.functions at STARTUP, not after the run: a case that refuses on an
        // unsupported model, or that never reaches the post-run forces block, must still be told which
        // of its functionObjects brae will not honour. forceCoeffs is declared APPROXIMATED because this
        // driver prints it once at the end, whereas OF's runs every time step and writes a history
        // (forceCoeffs.H:547,550) -- gpuPimpleFoam declares the same type APPLIED, because it does
        // sample on the write cadence.


        // startFrom is resolved by Time, as OF does in Time::setControls() (Time.C:149-188). This
        // driver carried its own copy of that scan; measured identical to the shared one on an
        // adversarial directory (numeric dir without U, non-integer time, 0.orig) before removing it.
        Time time(caseDir, controlDict);   // startFrom resolves here; functionObjects are read below
        std::string startStr = time.startName();
        // OF `restore0Dir` convention (NOT a solver fallback): tutorials needing a mesh-prep step (snappyHexMesh /
        // setFields / changeDictionary, e.g. motorBike) ship the flow fields in <startTime>.orig, because
        // snappyHexMesh writes mesh-level fields (cellLevel/pointLevel/...) INTO <startTime>. OpenFOAM's Allrun copies
        // 0.orig -> 0 before solving; cf does exactly the same SETUP step here (only when <startTime> has no U), so the
        // solver then reads <startTime> identically to OF, behaviour unchanged vs OF, no read-from-.orig special case.
        {
            namespace fs = std::filesystem;
            std::error_code ec;
            const std::string d0 = caseDir + "/" + startStr, dOrig = d0 + ".orig";
            const bool haveU = fs::exists(d0 + "/U") || fs::exists(d0 + "/U.gz");
            if (!haveU && (fs::exists(dOrig + "/U") || fs::exists(dOrig + "/U.gz")))
            {
                std::fprintf(stderr, "brae: %s/U not found -> restoring %s/* into %s (OpenFOAM restore0Dir convention)\n",
                             d0.c_str(), dOrig.c_str(), d0.c_str());
                fs::create_directories(d0, ec);
                for (const auto& e : fs::directory_iterator(dOrig, ec))
                    fs::copy(e.path(), d0 + "/" + e.path().filename().string(),
                             fs::copy_options::overwrite_existing | fs::copy_options::recursive, ec);
            }
        }
        const std::string fieldDir = caseDir + "/" + startStr;   // solver reads <startTime> exactly as in OpenFOAM

        // Time owns the functionObject lifecycle, as OF's Foam::Time does: the loop below never mentions
        // functionObjects, which is what stops a solver from being able to forget them. Placed here --
        // after the start directory resolves, but still BEFORE mesh, geometry and solver -- so the report
        // reaches a case that refuses later, while the objects resolve those dependencies from the
        // registry on first execute(). forceCoeffs is APPROXIMATED in this driver (a single post-run
        // print); gpuPimpleFoam declares the same type APPLIED, because it samples on the write cadence.
        ObjectRegistry timeRegistry;
        std::vector<ScalarTransportFO*> scalarTransports;
        std::vector<std::pair<std::string, FunctionObjectList::Factory>> foTypes;
        foTypes.emplace_back(
            "scalarTransport",
            [&](const std::string& foName, const FoamDict& fd) -> std::unique_ptr<FunctionObject>
            {
                // Only OF's constant-D branch is implemented; nut-based and alphaD/alphaDt are refused
                // by name rather than quietly replaced by a constant (scalarTransport.C D()).
                if (!fd.found("D"))
                {
                    noticeIgnored("functions/" + foName,
                                  "scalarTransport without a constant `D` (nut-based or alphaD/alphaDt "
                                  "diffusivity) is not implemented, so this tracer is NOT solved.");
                    return nullptr;
                }
                const std::string fld = fd.wordOr("field", foName);
                // OF: schemesField defaults to the field name, and the scheme is looked up as
                // div(phi,<schemesField>). Absent under `default none` is fatal in OF, so it is here
                // too -- reported and declined rather than run with a substituted discretisation.
                const std::string schemesField = fd.wordOr("schemesField", fld);
                FieldDivScheme scheme;
                try { scheme = parseFieldDivScheme(caseDir, schemesField); }
                catch (const std::exception& e)
                {
                    noticeIgnored("functions/" + foName, std::string(e.what()) +
                                  " -- this tracer is NOT solved.");
                    return nullptr;
                }
                auto fo = std::make_unique<ScalarTransportFO>(
                    foName, fld, fieldDir + "/" + fld, timeRegistry,
                    fd.scalarOr("D", 0.0), fd.scalarOr("relaxCoeff", 1.0), fd.scalarOr("tol", 1e-6),
                    scheme);
                scalarTransports.push_back(fo.get());
                return fo;
            });
        time.readFunctionObjects(controlDict, foTypes, {"forceCoeffs"});
        const int   endTime   = controlDict.intOr("endTime", 1000);
        // Steady simpleFoam's endTime is an INTEGER iteration count. A fractional endTime (e.g. 0.3, copy-pasted from
        // a transient case) truncates to 0 -> the loop runs 0 iterations and would write the initial field as "the
        // solution". Refuse endTime < 1.
        if (endTime < 1)
            throw std::runtime_error("controlDict endTime = " + std::to_string(endTime) + " (< 1). Steady simpleFoam"
                " runs an integer iteration count; a fractional endTime truncates to 0 and would write the initial"
                " field as the solution. Set endTime to the number of SIMPLE iterations.");
        const int   precision = controlDict.intOr("writePrecision", 16);

        DeviceSimpleControls ctl;
        ctl.caseDir = caseDir;
        ctl.writeCache = std::getenv("BRAE_MESH_CACHE") != nullptr;   // -partition (above) or the env
        ctl.nu = transport.scalarOr("nu", 1e-5);
        // brae is Newtonian-only; a non-Newtonian transportModel has no top-level nu, so it would silently run with
        // the default constant nu (the wrong viscosity for a shear-thinning/thickening fluid). Fail loud instead.
        {
            const std::string tModel = transport.wordOr("transportModel", "Newtonian");
            if (tModel != "Newtonian")
                throw std::runtime_error("constant/transportProperties transportModel '" + tModel + "' is not"
                    " supported -- brae is Newtonian-only and would silently use a constant nu. Only 'Newtonian' is supported.");
        }
        // read schemes: div(phi,U) "bounded" -> -Sp(div(phi),.); "linearUpwind" -> deferred gradient correction;
        // laplacian/snGrad "corrected" -> non-orthogonal correction (nonOrthDeltaCoeffs implicit + corrVec.grad explicit).
        parseFvSchemesControls(caseDir, ctl);
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        ctl.turbulent = (simType == "RAS");
        if (simType != "RAS" && simType != "laminar")
            throw std::runtime_error("brae: unsupported simulationType '" + simType + "' (RAS or laminar)");
        readTurbulenceModel(turbProps, ctl);

        // Scalar linearUpwind is gated OFF here as a COLD-START STABILITY guard, not for accuracy. The
        // original comment claimed it "degrades turbulence accuracy vs OF"; that was measured with the old
        // line-based fvSchemes parser (the one that leaked schemes between statements) and is wrong. What
        // the re-measurement actually shows:
        //   - discretisation is CORRECT: one iteration from OF's own converged pitzDaily state, tight
        //     solvers, linearUpwind on k -> k agrees with OF to 1.6e-06 (omega 3.3e-05, p exact).
        //   - the compressible duct converges to OF at k 2.0e-06 / nut 8.1e-07 with it honoured, and is
        //     1.6e-02 off on nut with it downgraded -- so rhoSimpleFoam now HONOURS it.
        //   - but from a COLD start pitzDaily SST diverges where OF converges. Bisected cleanly, with the
        //     `div(phi,epsilon)` line pinned so it cannot leak into luEps (an earlier bisect was
        //     contaminated by exactly that and wrongly blamed k alone):
        //         luK=1 luEps=0  converges     luK=0 luEps=1  converges
        //         luK=0 luEps=0  converges     luK=1 luEps=1  NaN
        //     So it is a COUPLED k-omega instability -- neither correction destabilises anything by
        //     itself. Ruled out by measurement: SIMPLEC, gradient limiting, regex relaxation, the Pk
        //     limiter, k/omega bounding, constant preservation, the matrix assembly (diagonal and source
        //     match OF's fvScalarMatrix to 1.8e-06), and the near-wall matrix manipulation.
        //
        // SPALART-ALLMARAS IS EXEMPT, and the reason is structural rather than empirical: SA transports ONE
        // scalar. deviceSpalartAllmarasCorrect takes luK only and never reads luEps, so the luK=1,luEps=1
        // state the divergence requires is unreachable. Downgrading SA was protecting it from a
        // two-equation failure mode it cannot have, and the cost was large: airFoil2D nuTilda 1.67 upwind
        // vs 3.6e-03 honoured, a factor of 460. gpuPimpleFoam never had this guard, so the same SA-IDDES
        // case already ran linearUpwind there -- the steady path was giving a different answer to the
        // transient one for identical input.
        //
        // The two-equation guard stays until the coupled divergence is understood.
        if (!std::getenv("BRAE_SCALAR_LINEARUPWIND") && !ctl.sa)
        {
            // Never silently honour-then-ignore: if fvSchemes asked for it, say that upwind is running.
            if (ctl.luK || ctl.luEps)
                std::fprintf(stderr, "brae WARNING: div(phi,k|epsilon|omega) requested 'linearUpwind' but brae is "
                             "running UPWIND on the TWO-equation models (cold-start guard against a coupled "
                             "k-omega divergence; the discretisation itself matches OF to 1.6e-06, and SA is "
                             "unaffected and honours it). Set BRAE_SCALAR_LINEARUPWIND=1 to honour it anyway.\n");
            ctl.luK = false;
            ctl.luEps = false;
        }

        readRelaxationFactors(fvSolution, ctl);   // shared: solvers/common/linear_solver_setup.cuh

        // fvSolution -> ctl, through the SHARED reader in solvers/common/linear_solver_setup.cuh.
        // This used to be an inline copy here; the compressible driver was ported from it and silently
        // dropped fifteen of these controls. One reader means a new driver gets the whole set or none.
        const std::string second = ctl.sst ? "omega" : "epsilon";
        readLinearSolverControls(fvSolution, second, ctl);

        const FoamDict* simple = fvSolution.subDict("SIMPLE");
        const FoamDict* resCtl = simple ? simple->subDict("residualControl") : nullptr;
        const bool hasRC = (resCtl != nullptr);
        const scalar rcP = resCtl ? resCtl->scalarOr("p", -1) : -1, rcU = resCtl ? resCtl->scalarOr("U", -1) : -1;
        // consistent / nNonOrthogonalCorrectors / bodyForce are read by readLinearSolverControls above.

        std::printf("brae (device-resident) | case=%s | %s%s | nu=%.3g\n", caseDir.c_str(),
                    simType.c_str(), ctl.turbulent ? (ctl.sst ? " (kOmegaSST)" : " (kEpsilon)") : "", ctl.nu);
        std::printf("  relax U=%.2g p=%.2g | tol p=%.1g U=%.1g | endTime=%d | residualControl=%s\n",
                    ctl.relaxU, ctl.relaxP, ctl.tolP, ctl.tolU, endTime, hasRC ? "on" : "off");
        std::printf("  schemes: bounded(U=%d,k=%d,eps=%d) linearUpwind(U)=%d nonOrth(corrected)=%d nonOrthLimit=%.3g consistent(SIMPLEC)=%d limitedLinear(k=%d,eps=%d) linearUpwind(k=%d,eps=%d)\n",
                    ctl.bounded, ctl.boundedK, ctl.boundedEps, ctl.linearUpwind, ctl.nonOrth, ctl.nonOrthLimit, ctl.consistent, ctl.limitedK, ctl.limitedEps, ctl.luK, ctl.luEps);
        std::printf("  grad(U) cellLimited k=%.3g (0=unlimited)\n", ctl.gradULimitK);
        std::printf("  linear solver (GaussSeidel from fvSolution; else BiCGStab): U=%d k|nuTilda=%d eps|omega=%d\n", ctl.gsU, ctl.gsK, ctl.gsEps);

        // mesh + start fields (single GPU: read directly, no decomposition)
        const bool timeStartup = std::getenv("BRAE_TIME_STARTUP") != nullptr;
        auto _tsClk = std::chrono::high_resolution_clock::now();
        auto _tsLap = [&](const char* what)
        {
            if (timeStartup)
            {
                auto n = std::chrono::high_resolution_clock::now();
                std::fprintf(stderr, "[startup] %-22s %6.2f s\n", what, std::chrono::duration<double>(n-_tsClk).count());
                _tsClk = n;
            }
        };
        PrimitiveMesh m;
        m.read(caseDir + "/constant/polyMesh");
        _tsLap("mesh read");
        FvGeometry g;
        // cyclicACMI splits the coincident interface faces' areas before cell volumes are computed;
        // on a mesh without one this is exactly g.build + buildPatches.
        std::vector<FvPatch> fvpBuilt;
        { std::vector<AMIInterface> amisInit; buildGeometryPatchesAndAMI(m, g, fvpBuilt, amisInit); }
        _tsLap("geometry build");
        const std::vector<FvPatch> fvp = std::move(fvpBuilt);
        timeRegistry.store("mesh", &m);
        timeRegistry.store("geometry", &g);
        timeRegistry.store("patches", &fvp);
        const label nC = m.nCells();

        // This driver maintains the coded pair per step (setupCodedBCs below) but NOT fixedMean or
        // fanPressure -- collectFixedMean/collectFanPressure are wired in gpuPimpleFoam only -- so
        // those two would freeze at the file `value`. Refuse them here, where the type still exists.
        const FieldData<vector> UFd = readField<vector>(fieldDir + "/U");
        refuseFrozenPerStepBC(UFd, "U", "gpuSimpleFoam", true);
        GeometricField<vector> U = buildField<vector>(UFd, fvp, nC);
        U.evaluateBoundary();
        const FieldData<scalar> pFd = readField<scalar>(fieldDir + "/p");
        refuseFrozenPerStepBC(pFd, "p", "gpuSimpleFoam", true);
        GeometricField<scalar> p = buildField<scalar>(pFd, fvp, nC);
        p.evaluateBoundary();
        // pressure needs a reference iff NO p patch fixes the value (singular all-Neumann system, e.g. closed
        // lid-driven cavity / fixedFluxPressure-walled domains). Then adjustPhi + pEqn.setReference (OF needReference).
        ctl.needRef = true;
        // a fixedValue-p OR a freestreamPressure (outletInlet, cat 4: fixedValue at outflow) references the pressure.
        for (const auto& bf : p.boundary)
            if (bf->fixesValue() || bf->bcCategory() == 4)
            {
                ctl.needRef = false;
                break;
            }
        if (ctl.needRef)
        {
            ctl.pRefCell  = simple ? (label)simple->intOr("pRefCell", 0) : 0;
            ctl.pRefValue = simple ? simple->scalarOr("pRefValue", 0.0) : 0.0;
            std::printf("  pressure needs reference (no fixedValue-p): pRefCell=%d pRefValue=%.4g\n",
                        (int)ctl.pRefCell, ctl.pRefValue);
        }
        // OF createPhi.H:45 builds phi with IOobject::READ_IF_PRESENT, so a restart RESUMES the stored
        // flux instead of rebuilding it. A converged SIMPLE phi is NOT fvc::flux(U): it carries the
        // Rhie-Chow pressure correction, so rebuilding discards that and restarts from a different
        // state than OF would. Measured on validation/channel, restart-at-30 vs continuous-to-60:
        // U 7.85e-06 / p 2.08e-05 relative, against a restart-to-restart reproducibility floor of
        // 9.5e-12 -- six orders above noise, so not round-off.
        //
        // Identical to the defect already fixed for the compressible driver; this driver simply never
        // received it, which is what carrying three copies of the same behaviour costs.
        bool phiWasRead = false;
        SurfaceScalarField phi = readPhiIfPresent(fieldDir, fvp, m.nInternalFaces(),
                                                  fvc::flux(U, m, g, fvp), &phiWasRead);

        const std::string secondName = ctl.sst ? "omega" : "epsilon";   // the 2nd turbulence scalar
        TurbulenceFields tf = readTurbulenceFields(fieldDir, fvp, nC, ctl, secondName, U,
                                                   "gpuSimpleFoam", true);

        // MRF rotating zone (constant/MRFProperties + polyMesh/cellZones), if present
        const MRFConfig mrfCfg = readMRFProperties(caseDir + "/constant");
        MRFZone mrfZone;
        if (mrfCfg.active)
        {
            const auto zones = readCellZones(caseDir + "/constant/polyMesh");
            const auto it = zones.find(mrfCfg.cellZone);
            std::vector<label> zoneCells = (it != zones.end()) ? it->second : std::vector<label>{};
            if (zoneCells.empty())
            {
                // A named zone that isn't in cellZones is almost always a typo (or a binary/unparsed zone): the
                // old silent whole-mesh fallback then turned the ENTIRE domain into a rotating frame. Refuse that,
                // and report the zones that ARE present -- EXCEPT for the explicit 'all' convention (deliberate
                // whole-domain rotation, e.g. rotatingCylinders), which we honour but announce loudly.
                if (mrfCfg.cellZone == "all")
                {
                    std::printf("  MRF: cellZone 'all' -> rotating the WHOLE mesh (%d cells)\n", nC);
                    zoneCells.resize(nC);
                    for (label c = 0; c < nC; ++c) zoneCells[c] = c;
                }
                else
                {
                    std::string avail;
                    for (const auto& z : zones) avail += (avail.empty() ? "" : ", ") + z.first;
                    throw std::runtime_error(
                        std::string("MRF cellZone '") + mrfCfg.cellZone
                        + "' not found or empty in constant/polyMesh/cellZones (available: "
                        + (avail.empty() ? std::string("<none>") : avail)
                        + "). Refusing to silently rotate the ENTIRE domain -- fix the 'cellZone' name in"
                          " constant/MRFProperties (use 'all' for deliberate whole-mesh rotation).");
                }
            }
            mrfZone = buildMRFZone(m, zoneCells, mrfCfg.axis, mrfCfg.omega, mrfCfg.origin);
            mrfCorrectBoundaryVelocity(U, mrfZone, g, fvp, mrfCfg.nonRotatingPatches);   // in-zone walls -> Omega x r
            phi = fvc::flux(U, m, g, fvp);                                               // re-flux (absolute; setMRF makes it relative)
            std::printf("  MRF: cellZone=%s omega=%.4g axis=(%.2g %.2g %.2g) origin=(%.2g %.2g %.2g) nonRotating=%zu zoneCells=%zu\n",
                        mrfCfg.cellZone.c_str(), mrfCfg.omega, mrfCfg.axis.x, mrfCfg.axis.y, mrfCfg.axis.z,
                        mrfCfg.origin.x, mrfCfg.origin.y, mrfCfg.origin.z, mrfCfg.nonRotatingPatches.size(), zoneCells.size());
        }

        // device-resident SIMPLE loop
        _tsLap("fields + patches");
        DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl,
                                  ctl.turbulent ? &tf.k : nullptr, (ctl.turbulent && !ctl.sa) ? &tf.eps : nullptr, ctl.turbulent ? &tf.nut : nullptr,
                                  ctl.lm ? &tf.ReThetat : nullptr, ctl.lm ? &tf.gammaInt : nullptr,
                                  phiWasRead);
        // uniformTotalPressure p0(t). OF samples p0_->value(t) at construction with the CURRENT
        // time and again in every updateCoeffs (uniformTotalPressureFvPatchScalarField.C:73,149),
        // so the tables are handed over before the first step and re-evaluated per step inside the
        // solver. Without this the parsed table sat unused and p0 stayed frozen at its seed --
        // measured on pimpleFoam/RAS/TJunction as inlet p FALLING 9.32 -> 8.62 where the table asks
        // for 13.09 -> 15.11, i.e. a case that runs and silently ignores the prescribed ramp.
        solver.setTimeVaryingP0(DeviceSimpleSolver::collectTimeVaryingP0(pFd, fvp));
        solver.setTime(static_cast<scalar>(std::strtod(startStr.c_str(), nullptr)));   // seed p0 at the START time, as OF's constructor does
        timeRegistry.store("solver", &solver);   // from here the functionObjects can resolve
        _tsLap("solver ctor (incl AMG)");
        if (partition)   // caches written by the mesh read + the AMG build above; done, like decomposePar finishing.
        {
            std::printf("brae -partition: mesh + AMG hierarchy cached to %s/constant/polyMesh/ (.brae_meshcache + .brae_amgcache).\n"
                        "  Run the solve normally; it will reload them warm.\n", caseDir.c_str());
            return 0;
        }
        if (mrfCfg.active) solver.setMRF(mrfZone, m, g, mrfCfg.nonRotatingPatches);

        // codedFixedValue / codedMixed on U / p / turbulence scalars -> NVRTC device kernel per coded patch (compiled
        // once; applied each SIMPLE iteration in the momentum predictor). Steady: the coded `t` stays 0 (position-based
        // profiles); a codedFixedValue set here overrides its seed `value`. Fully device-resident.
        setupCodedBCs(solver, fieldDir, fvp, ctl, secondName, "gpuSimpleFoam");

        // fvOptions (system/fvOptions or constant/fvOptions), if present (else an empty no-op list)
        // Read cellZones only when an fvOptions file exists (avoids touching cellZones on cases that have none,
        // and a binary cellZones, which readCellZones does not parse).
        std::map<std::string, std::vector<label>> fvoZones;
        {
            std::ifstream a(caseDir + "/system/fvOptions"), b(caseDir + "/constant/fvOptions");
            if (a.good() || b.good()) fvoZones = readCellZones(caseDir + "/constant/polyMesh");
        }
        const FvOptionsData fvo = readFvOptions(caseDir, fvoZones, g.V(), nC, g.C());
        if (!fvo.unsupported.empty())   // fail loud rather than run a valid-looking case with a silently-dropped source
        {
            std::string msg = "fvOptions contains source(s) brae cannot apply (they would be SILENTLY dropped -> wrong physics):";
            for (const auto& u : fvo.unsupported) msg += "\n  - " + u;
            msg += "\nRemove/disable them, or use a supported form. Supported: vectorSemiImplicitSource,"
                   " explicitPorositySource[DarcyForchheimer], meanVelocityForce, limitVelocity,"
                   " actuationDiskSource[Froude], rotorDisk, velocityDampingConstraint; selectionMode all|cellZone.";
            throw std::runtime_error(msg);
        }
        // OF re-evaluates the turbulent-inlet BCs every updateCoeffs; give the solver the per-face masks
        // so it refreshes them each iteration instead of freezing the set-up value.
        //
        // THIS IS NOT AN fvOptions CONCERN, and sitting inside the `if (!fvo.empty())` below meant a case
        // with no fvOptions never got it -- the intensity-based k inlet and the mixing-length
        // epsilon/omega inlet stayed at whatever the file's `value` entry said for the whole run. The
        // OpenFOAM tutorials write `value $internalField` there, so on pipeCyclic that was k = 1 against
        // the 0.0038-0.0067 the 5% intensity implies: a 200x inlet that fed the entrance region and
        // decayed only by the pipe exit (wall-cell k 31x OpenFOAM's at x<1.3, 1.02x at x>8.7).
        solver.setTurbulentInlets(tf.turbInletMasks.tiMask, tf.turbInletMasks.tiIntensity,
                                  tf.turbInletMasks.mlMask, tf.turbInletMasks.mlLength);
        if (!fvo.empty())
        {
            solver.setFvOptions(fvo);
            if (fvo.rotor.active)   // build the BEM rotor geometry from the mesh (cell centres + face areas) and hand it over
                solver.setRotorDisk(buildDeviceRotorDisk(fvo.rotor, g.C(), g.Sf(), m.owner(), m.neighbour(), m.nInternalFaces()));
            std::printf("  fvOptions: %d source(s)%s%s%s%s%s%s%s%s\n", fvo.count, fvo.hasMomentum ? " momentum" : "",
                        fvo.porActive ? " DarcyForchheimer-porosity" : "", fvo.mvfActive ? " meanVelocityForce" : "",
                        fvo.limUActive ? " limitVelocity" : "", fvo.adActive ? " actuationDiskSource" : "",
                        fvo.rotor.active ? " rotorDiskSource" : "", fvo.vdcActive ? " velocityDampingConstraint" : "",
                        (!fvo.scaSu.empty() || !fvo.scaSp.empty()) ? " scalar(WARN: turbulence-field sources read but not yet applied per-model)" : "");
        }
        else
        {
            std::printf("  No finite volume options present\n");   // OF createFvOptions.H message
        }
        // OF simpleControl::criteriaSatisfied: an unlisted field is not a criterion, and a run only
        // converges if at least one criterion was ACTUALLY checked (see solvers/common/residual_control.cuh).
        int rcChecked = 0;
        TurbBlowup turbBlowup;   // sum|turb| tripwire; see solvers/common/turb_blowup.cuh
        auto ok = [&](scalar res, scalar ctlv) { if (ctlv < 0) return true; ++rcChecked; return res < ctlv; };
        // OF controlDict write cadence: writeControl / writeInterval / purgeWrite (ported from Foam::Time)
        const std::string writeControl = controlDict.wordOr("writeControl", "timeStep");
        const scalar writeInterval = controlDict.scalarOr("writeInterval", 1e30);   // OF default GREAT -> only the final state
        const int    purgeWrite    = std::max(0, controlDict.intOr("purgeWrite", 0));
        const scalar deltaT        = controlDict.scalarOr("deltaT", 1.0);
        // The RESOLVED start, not controlDict's startTime: `startFrom latestTime` can make them differ,
        // and every time value below is measured from the start. Taking the dict's value made a run
        // restarted from 10 name its output 1, 2, 3... -- overwriting the case's own early history.
        const scalar startTimeVal  = static_cast<scalar>(std::strtod(startStr.c_str(), nullptr));
        // Write cadence comes from Time, which owns the WriteControl -- as OF does in
        // Time::operator++ / TimeIO.C:277. This driver carried its own isWriteTime lambda; it was
        // measured identical to the shared one over 8 controlDict cases x 40 steps (including
        // runTime float accumulation and a non-zero-startTime restart) before being removed.
        const std::string wsrc = fieldDir + "/";
        auto writeTimeDir = [&](const std::string& tname)   // reconstruct + write one time directory
        {
            const std::string outDir = caseDir + "/" + tname;
            std::filesystem::create_directories(outDir);
            // scalarTransport tracers. OF builds its transported field with AUTO_WRITE
            // (scalarTransport.C transportedField()), so it appears in every written time directory
            // alongside the solved fields. Pulled from the device only HERE -- on the write cadence,
            // not every iteration -- which is why OF splits write() from execute().
            for (ScalarTransportFO* st : scalarTransports)
                if (st->ready())
                    writeVolField(fieldDir + "/" + st->fieldName(), outDir + "/" + st->fieldName(),
                                  st->hostField(), fvp, 12);
            // The written fields keep the source field's `#include "include/..."` directives (resolved relative to the
            // time dir). Copy the startTime include/ dir alongside so OpenFOAM readers (paraFoam, postProcess, foamToVTK)
            // resolve them, otherwise the field points at a nonexistent <time>/include/ and fails to load.
            {
                std::error_code ec2;
                if (std::filesystem::exists(fieldDir + "/include"))
                    std::filesystem::copy(fieldDir + "/include", outDir + "/include",
                        std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec2);
            }
            // phi, so a brae->brae restart RESUMES the corrected flux. OF's createPhi.H pairs
            // READ_IF_PRESENT with AUTO_WRITE; reading it without writing it leaves every restart
            // falling back to fvc::flux(U), which is the very field the read exists to avoid.
            // INCOMPRESSIBLE dimensions: volumetric flux [0 3 -1 0 0 0 0], not the compressible
            // driver's mass flux [1 0 -1 0 0 0 0].
            writeSurfaceField(outDir + "/phi", solver.phiInternal(), solver.phiBoundary(), fvp,
                              17, "[0 3 -1 0 0 0 0]");
            writeVolField(wsrc + "U", outDir + "/U", solver.U(), fvp, precision, solver.UBoundary());
            writeVolField(wsrc + "p", outDir + "/p", solver.p(), fvp, precision, solver.pBoundary());
            if (ctl.sa)   // one-equation: the k slot holds nuTilda
            {
                writeVolField(wsrc + "nuTilda", outDir + "/nuTilda", solver.k(),   fvp, precision, solver.nuTildaBoundary());
                writeVolField(wsrc + "nut",     outDir + "/nut",     solver.nut(), fvp, precision, solver.nutBoundary());
            }
            else if (ctl.turbulent)
            {
                writeVolField(wsrc + "k", outDir + "/k", solver.k(), fvp, precision, solver.kBoundary());
                writeVolField(wsrc + secondName, outDir + "/" + secondName, solver.eps(), fvp, precision, solver.epsBoundary());
                writeVolField(wsrc + "nut", outDir + "/nut", solver.nut(), fvp, precision, solver.nutBoundary());
                if (ctl.lm)   // kOmegaSSTLM transition fields
                {
                    writeVolField(wsrc + "ReThetat", outDir + "/ReThetat", solver.ReThetat(), fvp, precision);
                    writeVolField(wsrc + "gammaInt", outDir + "/gammaInt", solver.gammaInt(), fvp, precision);
                }
            }
            std::printf("written %s/{U,p%s}\n", outDir.c_str(),
                        ctl.turbulent ? (ctl.sa ? ",nuTilda,nut" : ctl.sst ? ",k,omega,nut" : ",k,epsilon,nut") : "");
            // purgeWrite, from Time's WriteControl. This driver's copy was measured identical over 6
            // FIFO cases (disabled / keep-1 / keep-2 / never-exceeds / repeated tname / non-integer
            // names), and all three copies purged at the SAME point -- immediately after the write --
            // so consolidating changes no filesystem side effect.
            time.writeControl().recordWritten(caseDir, tname);
        };

        // endTime is ABSOLUTE, not a run length. OF's Time::run() tests `value() < endTime - 0.5*deltaT`,
        // so a case restarted at 10 with endTime 20 runs TEN more steps and finishes at 20. Looping
        // `iter <= endTime` from 1 ran TWENTY and finished at 30 -- silently changing the iteration
        // count, the write times, and any comparison of a restarted run against a continuous one. Only
        // correct when startTime is 0, which is why every fresh-start case hid it.
        const long nSteps = std::lround((static_cast<double>(endTime) - static_cast<double>(startTimeVal))
                                        / static_cast<double>(deltaT));
        if (nSteps < 1)
            throw std::runtime_error(
                "controlDict endTime (" + std::to_string(endTime) + ") is not beyond the start time ("
                + startStr + "): there is nothing to run. endTime is an ABSOLUTE time, not a number of "
                "iterations -- on a restart set it past the time you are restarting from.");
        int iter = 0;
        bool converged = false;
        const auto _runStart = std::chrono::high_resolution_clock::now();          // for OpenFOAM-style ExecutionTime
        double _cumCont = 0.0;                                                     // cumulative continuity error (OF continuityErrs.H)
        time.setSteps(static_cast<int>(nSteps));
        while (!converged && time.loop())
        {
            iter = time.timeIndex();
            const DeviceSimpleResidual r = solver.step();
            {
                // OpenFOAM-style per-iteration report: Time, per-field solver residuals, continuity, turbulence, ExecutionTime.
                const double _et = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _runStart).count();
                const double cl = (double)deltaT * r.contLocal, cg = (double)deltaT * r.contGlobal;
                _cumCont += cg;
                std::printf("Time = %d\n\n"
                            "smoothSolver:  Solving for Ux, Initial residual = %g, Final residual = %g, No Iterations %d\n"
                            "smoothSolver:  Solving for Uy, Initial residual = %g, Final residual = %g, No Iterations %d\n"
                            "smoothSolver:  Solving for Uz, Initial residual = %g, Final residual = %g, No Iterations %d\n"
                            "GAMG:  Solving for p, Initial residual = %g, Final residual = %g, No Iterations %d\n"
                            "time step continuity errors : sum local = %g, global = %g, cumulative = %g\n",
                            iter, r.Ux, r.UxFinal, r.UxIters, r.Uy, r.UyFinal, r.UyIters, r.Uz, r.UzFinal, r.UzIters,
                            r.p, r.pFinal, r.pIters, cl, cg, _cumCont);
                for (const auto& e : turbulenceReport())   // Solving for omega/k/epsilon/... in solve order, like OF
                    std::printf("smoothSolver:  Solving for %s, Initial residual = %g, Final residual = %g, No Iterations %d\n",
                                e.field.c_str(), e.perf.initialResidual, e.perf.finalResidual, e.perf.nIterations);
                std::printf("ExecutionTime = %.2f s  ClockTime = %.0f s\n\n", _et, _et);
            }
            // NaN/divergence guard: a non-finite momentum/pressure residual means the solve blew up (FP32 overflow,
            // singular pressure, turbulence blow-up, or an under-stabilised case). Without this the loop runs to
            // endTime (ok(NaN,tol) is always false -> never "converges") and WRITES the NaN field as the solution.
            // Abort loudly and write nothing. Opt out (e.g. to inspect the field) with BRAE_ALLOW_NONFINITE=1.
            if (!std::getenv("BRAE_ALLOW_NONFINITE")
                && !(std::isfinite(r.p) && std::isfinite(r.Ux) && std::isfinite(r.Uy) && std::isfinite(r.Uz)))
                throw std::runtime_error(
                    "solution diverged: non-finite residual at iteration " + std::to_string(iter)
                    + " (p=" + std::to_string(r.p) + " Ux=" + std::to_string(r.Ux)
                    + " Uy=" + std::to_string(r.Uy) + " Uz=" + std::to_string(r.Uz) + "). Likely causes:"
                    + " too-loose relaxation, a high-non-orthogonality mesh, a singular pressure system, or"
                    + " turbulence blow-up. No field written. Set BRAE_ALLOW_NONFINITE=1 to continue anyway.");
            // Turbulence blow-up that stays FINITE. The check above only catches NaN/Inf, and a diverging
            // k-omega pair need not get there: measured on pitzDaily with linearUpwind on BOTH scalars,
            // omega reached 1e42 with k pinned at its 1e-15 floor, U stayed bounded (nut collapses, so the
            // flow just goes near-laminar), every residual stayed finite, and the run marched to endTime and
            // WROTE the fields reporting success. That is worse than a crash -- the output looks plausible.
            //
            // Growth is measured against the first iteration rather than an absolute value, so the bar is
            // independent of mesh size and of the case's units -- but crossing the bar is NOT on its own
            // divergence. A violent start-up transient crosses it and recovers, and real OpenFOAM goes
            // through the same excursion on the case that exposed this. The tripwire therefore requires the
            // excursion to PERSIST; see solvers/common/turb_blowup.cuh for the measurements behind that.
            if (!std::getenv("BRAE_ALLOW_NONFINITE") && ctl.turbulent)
            {
                if (turbBlowup.update(solver.turbSumMag(), (int)iter))
                    throw std::runtime_error(turbBlowup.message((int)iter));
            }
            // OF residualControl: also gate on every turbulence field (k/epsilon/omega/nuTilda) that lists a target.
            // Previously ONLY p and Ux were checked, so a turbulent case could report "converged" with k/epsilon
            // still far from tol -- the substantive bug this fixes. Unlisted fields have target -1 -> ok() ignores
            // them (OF). U stays gated on Ux alone: brae tracks no valid/solved directions, so the out-of-plane
            // component of a 2D/empty or wedge case has a DEGENERATE residual (stuck ~0.1, never reaching tol) that
            // would wrongly block convergence on every 2D case -- gating all U components needs that infra first.
            rcChecked = 0;
            converged = hasRC && ok(r.p, rcP) && ok(r.Ux, rcU);
            if (converged)
                for (const auto& e : turbulenceReport())
                    if (!ok(e.perf.initialResidual, resCtl->scalarOr(e.field, -1))) { converged = false; break; }
            // OF's `checked` safety: `residualControl { }`, or a dict naming only fields brae does not
            // check, must NOT report convergence. Without this brae stopped after ONE iteration and wrote
            // a plausible-looking field set (simpleControl.C:51-57).
            converged = converged && rcChecked > 0;
            const scalar tval = startTimeVal + (scalar)iter * deltaT;               // OF time value at this step
            if (!converged && iter != nSteps && time.writeControl().isWriteTime(iter, tval))
            {
                writeTimeDir(WriteControl::timeName(tval));
                time.write();   // OF splits write() from execute(); this driver owns its own cadence
            }
        }
        time.end();   // OF Time.C:790-802: a final execute() so the last step is seen, then end()
        // timeIndex() is the iteration that ran, so no -1: the for-loop this replaced incremented past it.
        const int nIter = converged ? time.timeIndex() : static_cast<int>(nSteps);
        std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                              : "SIMPLE reached endTime (%d iterations)\n", nIter);

        // BRAE_DUMP_PHI: write the conservative face flux, the one quantity that carries between SIMPLE
        // iterations and is not in any written cell field. Diagnostic only.
        if (std::getenv("BRAE_DUMP_PHI"))
        {
            const std::vector<scalar> phiI = solver.phiInternal();
            FILE* fp = std::fopen(std::getenv("BRAE_DUMP_PHI"), "w");
            if (fp)
            {
                for (std::size_t i = 0; i < phiI.size(); ++i) std::fprintf(fp, "%.17g\n", phiI[i]);
                std::fclose(fp);
            }
        }
        // BRAE_DUMP_CONTINUITY: localise the per-cell continuity imbalance R[c]=sum_f phi_f and bucket sum|R| by
        // region (wall-adjacent / farfield-adjacent / interior) to find WHERE continuity fails to close.
        if (std::getenv("BRAE_DUMP_CONTINUITY"))
        {
            const std::vector<scalar> phiI = solver.phiInternal(), phiB = solver.phiBoundary();
            std::vector<scalar> R(nC, 0.0);
            for (label f = 0; f < m.nInternalFaces(); ++f)
            {
                R[m.owner()[f]] += phiI[f];
                R[m.neighbour()[f]] -= phiI[f];
            }
            {
                label bi = 0;
                for (const auto& pp : fvp)
                    for (label i = 0; i < pp.size; ++i)
                        R[pp.faceCells[i]] += phiB[bi++];
            }
            std::vector<int> tag(nC, 0);   // 0=interior 1=wall-adjacent 2=farfield(patch)-adjacent
            for (const auto& pp : fvp)
            {
                const int t = (pp.type == "wall") ? 1 : (pp.type == "patch" ? 2 : 0);
                if (t)
                    for (label i = 0; i < pp.size; ++i)
                        tag[pp.faceCells[i]] = std::max(tag[pp.faceCells[i]], t);
            }
            double sAll = 0, sW = 0, sF = 0, sI = 0, mx = 0;
            label mxc = 0;
            for (label c = 0; c < nC; ++c)
            {
                const double a = std::fabs(R[c]);
                sAll += a;
                if (tag[c] == 1) sW += a;
                else if (tag[c] == 2) sF += a;
                else sI += a;
                if (a > mx) { mx = a; mxc = c; }
            }
            const double inv = sAll > 0 ? 100.0 / sAll : 0.0;
            std::printf("CONTINUITY: sum|div phi|=%.4e  max=%.4e @cell %d  | region share: wall=%.1f%% farfield=%.1f%% interior=%.1f%%\n",
                        sAll, mx, (int)mxc, sW * inv, sF * inv, sI * inv);
            std::vector<label> idx(nC);
            for (label c = 0; c < nC; ++c)
                idx[c] = c;
            const label topN = std::min<label>(12, nC);
            std::partial_sort(idx.begin(), idx.begin() + topN, idx.end(), [&](label a, label b){ return std::fabs(R[a]) > std::fabs(R[b]); });
            std::printf("  top cells (|imbalance|, tag, centroid):\n");
            for (label q = 0; q < topN; ++q)
            {
                const label c = idx[q];
                const vector cc = g.C()[c];
                std::printf("    cell %6d  |R|=%.3e  tag=%d  C=(%.4f %.4f %.4f)\n", (int)c, std::fabs(R[c]), tag[c], cc.x, cc.y, cc.z);
            }
        }

        if (std::getenv("BRAE_DUMP_Y") && ctl.turbulent)   // cell wall distance stats vs OF wallDist (SA destruction ~ 1/y^2)
        {
            const std::vector<scalar> yv = solver.cellY();
            if (!yv.empty())
            {
                double mn = 1e300, mx = 0, sm = 0;
                for (scalar v : yv)
                {
                    mn = std::min(mn, (double)v);
                    mx = std::max(mx, (double)v);
                    sm += v;
                }
                std::printf("cellY: min=%.4e max=%.4e mean=%.4e (n=%zu)\n", mn, mx, sm / yv.size(), yv.size());
            }
        }

        // always write the final (converged / endTime) state; matches OF's writeAndEnd, and feeds purgeWrite
        writeTimeDir(WriteControl::timeName(startTimeVal + (scalar)nIter * deltaT));
        const std::vector<vector> Ug = solver.U();   // converged fields, reused by the force calculation below
        const std::vector<scalar> pg = solver.p();

        // forces on wall patches (OF functionObjects::forces; kinematic, rhoInf=1). Validated vs OF (ctest forces).
        if (ctl.turbulent)
        {
            std::vector<std::string> walls;
            for (const auto& q : fvp)
                if (q.type == "wall") walls.push_back(q.name);
            if (!walls.empty())
            {
                for (label c = 0; c < nC; ++c)
                    U.internal[c] = Ug[c];   // converged fields -> evaluate wall BCs
                p.internal = pg;
                U.evaluateBoundary();
                p.evaluateBoundary();
                const scalar wCmu   = ctl.sst ? ctl.ksstCoeffs.betaStar : ctl.keCoeffs.Cmu;
                const scalar wKappa = ctl.sst ? ctl.ksstCoeffs.kappa    : ctl.keCoeffs.kappa;
                const scalar wE     = ctl.sst ? ctl.ksstCoeffs.E        : ctl.keCoeffs.E;
                // Velocity-based wall nut (SA-Spalding, or nutUSpalding/nutUBlended on kEps/kOmegaSST): use the device
                // wall nut for the viscous force, not the k-based one, matching the BC. nutk cases keep nwb=nullptr.
                const bool velNutWall = ctl.sa || ctl.nutWall != NutWall::Nutk;
                const std::vector<scalar> saNutWall = velNutWall ? solver.nutWall() : std::vector<scalar>();
                const std::vector<scalar>* nwb = velNutWall ? &saNutWall : nullptr;
                const ForceResult F = wallForces(U, p, solver.k(), ctl.nu, m, g, fvp, walls, 1.0, 0.0, vector{0,0,0}, wCmu, wKappa, wE, nwb);
                std::printf("forces (walls, rhoInf=1):  pressure=(%.5e %.5e %.5e)  viscous=(%.5e %.5e %.5e)  total=(%.5e %.5e %.5e)\n",
                            F.pressure.x, F.pressure.y, F.pressure.z, F.viscous.x, F.viscous.y, F.viscous.z, F.total().x, F.total().y, F.total().z);

                // controlDict.functions. Every entry is now ACCOUNTED FOR through the shared
                // FunctionObjectList: forceCoeffs is computed just below (reported as approximated,
                // because brae prints it once here while OF's forceCoeffs runs every time step and
                // writes a history -- forceCoeffs.H:547,550), and anything else is reported as ignored
                // rather than silently skipped. Before this, a case could carry any number of
                // functionObjects and the only one even looked at was forceCoeffs.
                const FoamDict* funcs = controlDict.subDict("functions");
                const FoamDict* fcd = nullptr;
                if (funcs)
                    for (const auto& s : funcs->subs)
                        if (s.second.wordOr("type", "") == "forceCoeffs")
                        {
                            fcd = &s.second;
                            break;
                        }
                if (fcd)
                {
                    auto toV = [](const std::vector<scalar>& a, vector d){ return a.size() >= 3 ? vector{a[0],a[1],a[2]} : d; };
                    const std::vector<std::string> fcP = fcd->wordListOr("patches", walls);
                    const scalar rhoInf = fcd->scalarOr("rhoInf", 1.0), magUInf = fcd->scalarOr("magUInf", 1.0);
                    const scalar Aref = fcd->scalarOr("Aref", 1.0), lRef = fcd->scalarOr("lRef", 1.0);
                    const vector liftDir = toV(fcd->scalarListOr("liftDir", {}), vector{0,1,0});
                    const vector dragDir = toV(fcd->scalarListOr("dragDir", {}), vector{1,0,0});
                    const vector pitchAxis = toV(fcd->scalarListOr("pitchAxis", {}), vector{0,0,1});
                    const vector CofR = toV(fcd->scalarListOr("CofR", {}), vector{0,0,0});
                    const ForceResult Fc = wallForces(U, p, solver.k(), ctl.nu, m, g, fvp, fcP, rhoInf, 0.0, CofR, wCmu, wKappa, wE, nwb);
                    const ForceCoeffs cc = forceCoeffs(Fc, dragDir, liftDir, pitchAxis, rhoInf, magUInf, Aref, lRef);
                    std::printf("forceCoeffs (rhoInf=%.3g magUInf=%.3g Aref=%.3g lRef=%.3g):  Cd=%.6e  Cl=%.6e  Cm=%.6e\n",
                                rhoInf, magUInf, Aref, lRef, cc.Cd, cc.Cl, cc.Cm);
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "brae ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
