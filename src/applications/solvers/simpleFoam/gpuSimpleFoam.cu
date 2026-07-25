// cf gpuSimpleFoam, steady incompressible SIMPLE solver, single-GPU device-resident. Reads the case dicts
// (controlDict / fvSolution / transportProperties / turbulenceProperties) and start fields, runs the whole
// SIMPLE(+kEpsilon) loop on the GPU via DeviceSimpleSolver (U/p/phi/k/eps/nut never leave the device between
// iterations), and writes the converged fields when fvSolution residualControl is met or controlDict endTime
// is reached. The momentum is the faithful incompressible stress div(phi,U)-laplacian(nuEff,U)-div(nuEff*
// dev2(T(grad U))); the pressure uses the device AMG-PCG. Mirrors the host brae_simpleFoam control flow.
//
//   brae -case <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "forces.cuh"
#include "mrf_read.cuh"
#include "fv_options.cuh"
#include "turbulent_inlet.cuh"
#include "foam_dict.cuh"
#include "sweep_cases.cuh"   // brae -cases c1 c2 ...: multi-GPU mesh/parameter study (orchestrator mode)
#include "fvc.cuh"
#include "device_simple_foam.cuh"
#include "parallel_device_foam.cuh"   // brae ... -parallel: the distributed DEVICE path (one rank per GPU)

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
"brae, a GPU-native, OpenFOAM-compatible CFD solver (steady incompressible, simpleFoam).\n"
"The whole SIMPLE loop runs on one GPU; reads a standard OpenFOAM case and writes standard time dirs.\n\n"
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
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "-parallel") return runParallelDeviceFoam(argc, argv);
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
        // -partition is cf's analogue of OF decomposePar: do the one-time prep (parse mesh + build AMG hierarchy) and
        // persist it to constant/polyMesh/.brae_mesh|amgcache, so the actual run reloads it warm. Forces the cache write.
        if (partition) setenv("BRAE_MESH_CACHE", "1", 1);

        // controls from the case dictionaries
        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
        const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
        const FoamDict turbProps   = readDict(caseDir + "/constant/turbulenceProperties");

        std::string startStr = controlDict.wordOr("startTime", "0");
        // OF startFrom: 'latestTime' restarts from the newest numeric time directory, 'firstTime' from the earliest
        // (default 'startTime' uses startTime). brae previously IGNORED startFrom, so a 'latestTime' restart silently
        // re-ran from startTime (scratch). Resolve it by scanning the case's numeric time dirs that hold a U field.
        {
            const std::string startFrom = controlDict.wordOr("startFrom", "startTime");
            if (startFrom == "latestTime" || startFrom == "firstTime")
            {
                namespace fs = std::filesystem;
                std::error_code ec;
                double best = 0.0; std::string bestName;
                for (const auto& e : fs::directory_iterator(caseDir, ec))
                {
                    if (!e.is_directory()) continue;
                    const std::string nm = e.path().filename().string();
                    char* end = nullptr; const double t = std::strtod(nm.c_str(), &end);
                    if (end == nm.c_str() || *end != '\0') continue;   // not a pure number -> skip constant/system/*.orig
                    if (!(fs::exists(e.path() / "U") || fs::exists(e.path() / "U.gz"))) continue;   // must be a field dir
                    if (bestName.empty() || (startFrom == "latestTime" ? t > best : t < best)) { best = t; bestName = nm; }
                }
                if (!bestName.empty() && bestName != startStr)
                {
                    std::fprintf(stderr, "brae: startFrom %s -> starting from time '%s'\n", startFrom.c_str(), bestName.c_str());
                    startStr = bestName;
                }
            }
        }
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
        {
            const std::string schemesText = readFileExpanded(caseDir + "/system/fvSchemes");   // $var expanded ($turbulence)
            std::istringstream fsch(schemesText);
            std::string ln;
            bool foundDivU = false;   // require an EXPLICIT div(phi,U); brae does not resolve the divSchemes 'default'
            bool warnedLeastSq = false, warnedCellMD = false;   // #14: warn-once on grad schemes brae approximates
            auto hasWord = [](const std::string& s, const std::string& w)   // whole-word match (so "uncorrected" != "corrected")
            {
                for (std::size_t p = s.find(w); p != std::string::npos; p = s.find(w, p + 1))
                {
                    const bool lb = (p == 0 || !std::isalpha((unsigned char)s[p-1]));
                    const bool rb = (p + w.size() >= s.size() || !std::isalpha((unsigned char)s[p+w.size()]));
                    if (lb && rb) return true;
                }
                return false;
            };
            // "limitedLinear <k_>" on a div line -> twoByk = 2/max(k_,SMALL); returns 0 if the scheme is absent.
            auto limitedTwoByk = [](const std::string& s) -> scalar
            {
                const std::size_t p = s.find("limitedLinear");
                if (p == std::string::npos) return 0.0;
                scalar kc = 1.0;
                std::sscanf(s.c_str() + p + 13, "%lf", &kc);   // coefficient after "limitedLinear"
                return 2.0 / std::max(kc, (scalar)1e-30);
            };
            // the interpolation-scheme word after "Gauss" on a div(phi,*) line (OF: "[bounded] Gauss <scheme> [args]").
            auto divSchemeWord = [](const std::string& s) -> std::string
            {
                const std::size_t g = s.find("Gauss");
                if (g == std::string::npos) return std::string();
                const char* p = s.c_str() + g + 5;
                while (*p && std::isspace((unsigned char)*p)) ++p;
                std::string w;
                while (*p && !std::isspace((unsigned char)*p) && *p != ';') { w += *p; ++p; }
                return w;
            };
            // OF-faithful fail-fast (mirrors surfaceInterpolationScheme::New "Unknown discretisation scheme ... Valid
            // schemes are : (...)" + exit(FatalIOError)): throw on a scheme cf models nothing close to; warn loudly on
            // one cf only APPROXIMATES (so the route is detected, never silently covered). `ok` = exact; `approx` = warned.
            auto checkDiv = [&](
                const std::string& s,
                const char* field,
                std::initializer_list<const char*> ok,
                std::initializer_list<const char*> approx)
            {
                const std::string w = divSchemeWord(s);
                for (const char* o : ok)
                    if (w == o) return;
                for (const char* o : approx)
                    if (w == o)
                    {
                        std::fprintf(stderr, "brae WARNING: div(phi,%s) 'Gauss %s' has no exact cf kernel -- run as a near-equivalent "
                                     "(NOT OF-bit-identical). Set the scheme to an exact one to avoid this.\n", field, w.c_str());
                        return;
                    }
                std::string valid;
                for (const char* o : ok)     (valid += " ") += o;
                for (const char* o : approx) (valid += " ") += o;
                throw std::runtime_error(std::string("brae: unknown/unsupported div(phi,") + field +
                    ") scheme 'Gauss " + w + "'; cf supports : (" + valid + " )");
            };
            while (std::getline(fsch, ln))
            {
                if (ln.find("div(phi,U)") != std::string::npos)
                {
                    foundDivU = true;
                    checkDiv(ln, "U", {"upwind", "linearUpwind", "linearUpwindV", "LUST"}, {"limitedLinear", "limitedLinearV"});
                    if (ln.find("bounded") != std::string::npos)      ctl.bounded = true;
                    if (ln.find("linearUpwind") != std::string::npos) ctl.linearUpwind = true;   // linearUpwindV contains this -> upwind matrix + gradients
                    if (ln.find("linearUpwindV") != std::string::npos) ctl.linearUpwindV = true; // + OF vector direction limiter
                    if (ln.find("LUST") != std::string::npos)         ctl.lust = true;   // 0.75 linear + 0.25 linearUpwind
                }
                // grad(U) cellLimited Gauss linear <k> (OF cellLimitedGrad<minmod>): k is the first number after
                // "cellLimited" (the basicScheme between has no digits). 0 = unlimited. cellMDLimited not yet handled.
                if (ln.find("grad(U)") != std::string::npos && hasWord(ln, "cellLimited"))
                {
                    ctl.gradULimitK = 1.0;
                    const char* s = ln.c_str() + ln.find("cellLimited") + 11;
                    while (*s && !(std::isdigit((unsigned char)*s) || *s == '.')) ++s;
                    scalar kc;
                    if (std::sscanf(s, "%lf", &kc) == 1) ctl.gradULimitK = kc;
                }
                // #14: brae computes gradients via Gauss linear only. These are valid ALTERNATIVE discretisations
                // (not wrong answers), so warn-once rather than fail -- the user should know brae is approximating.
                if (!warnedLeastSq && ln.find("leastSquares") != std::string::npos)
                { warnedLeastSq = true; std::fprintf(stderr, "brae WARNING: gradScheme 'leastSquares' is approximated as Gauss linear (differs on skewed meshes)\n"); }
                if (!warnedCellMD && ln.find("cellMDLimited") != std::string::npos)
                { warnedCellMD = true; std::fprintf(stderr, "brae WARNING: grad limiter 'cellMDLimited' is not applied (runs unlimited)\n"); }
                if (std::getenv("BRAE_SCHEME_DEBUG") && ln.find("div(phi,") != std::string::npos) std::fprintf(stderr, "[scheme] %s\n", ln.c_str());
                if (ln.find("div(phi,k)") != std::string::npos)
                {
                    checkDiv(ln, "k", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedK = true; ctl.twoBykK = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luK = true;
                }
                if (ln.find("div(phi,epsilon)") != std::string::npos || ln.find("div(phi,omega)") != std::string::npos)
                {
                    checkDiv(ln, "epsilon|omega", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedEps = true; ctl.twoBykEps = t; }   // 2nd turb scalar (eps|omega)
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luEps = true;
                }
                if (ln.find("div(phi,nuTilda)") != std::string::npos)   // SA: nuTilda uses the k-slot scheme flags
                {
                    checkDiv(ln, "nuTilda", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedK = true; ctl.twoBykK = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luK = true;
                }
                if (ln.find("Gauss") != std::string::npos || ln.find("snGrad") != std::string::npos ||
                    ln.find("laplacian") != std::string::npos || ln.find("default") != std::string::npos)
                {
                    if (hasWord(ln, "corrected")) ctl.nonOrth = true;     // unlimited non-orth correction (psi = 1)
                    // OF fv::limitedSnGrad "limited [<correctedScheme>] <psi>" (psi in [0,1]): non-orth correction
                    // capped per-face. hasWord avoids matching "unlimited" and "limitedLinear" (a div scheme); the coeff
                    // is the next numeric token after "limited" (skip an optional scheme word like "corrected").
                    if (hasWord(ln, "limited"))
                    {
                        ctl.nonOrth = true;
                        scalar psi = 1.0;
                        const char* s = ln.c_str() + ln.find("limited") + 7;
                        while (*s && !(std::isdigit((unsigned char)*s) || *s == '.')) ++s;   // skip to the coefficient
                        if (std::sscanf(s, "%lf", &psi) == 1) ctl.nonOrthLimit = psi;
                    }
                }
            }
            if (!schemesText.empty() && !foundDivU)
                throw std::runtime_error("fvSchemes: no explicit div(phi,U) scheme. brae does not resolve the"
                    " divSchemes 'default' for momentum convection (it would silently run first-order upwind)."
                    " Add e.g. 'div(phi,U)  bounded Gauss linearUpwind grad(U);'.");
        }
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        ctl.turbulent = (simType == "RAS");
        if (simType != "RAS" && simType != "laminar")
            throw std::runtime_error("brae: unsupported simulationType '" + simType + "' (RAS or laminar)");
        if (ctl.turbulent)
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            ctl.lm  = (model == "kOmegaSSTLM");                 // Langtry-Menter transition = kOmegaSST + gamma-ReThetat
            ctl.sst = (model == "kOmegaSST") || ctl.lm;
            ctl.sa  = (model == "SpalartAllmaras");
            const bool rke = (model == "realizableKE");
            if (model != "kEpsilon" && !rke && !ctl.sst && !ctl.sa)
                throw std::runtime_error("brae: unsupported RASModel '" + model + "' (kEpsilon, realizableKE, kOmegaSST, kOmegaSSTLM or SpalartAllmaras)");
            if (ctl.sa)
            {
                // Spalart-Allmaras: OF defaults (coeffs read from RAS.SpalartAllmarasCoeffs would override; not needed here).
                const SpalartAllmarasCoeffs& c = ctl.saCoeffs;
                std::printf("  SpalartAllmaras (OF defaults): sigmaNut=%.4g kappa=%.4g Cb1=%.4g Cb2=%.4g Cw1=%.4g Cw2=%.3g Cw3=%.3g Cv1=%.3g Cs=%.3g\n",
                            c.sigmaNut, c.kappa, c.Cb1, c.Cb2, c.Cw1(), c.Cw2, c.Cw3, c.Cv1, c.Cs);
            }
            else if (ctl.sst)
            {
                // kOmegaSST coefficients: OF turbulenceProperties RAS.kOmegaSSTCoeffs (absent keys keep OF defaults).
                readKOmegaSSTCoeffs(ras, ctl.ksstCoeffs);
                const KOmegaSSTCoeffs& c = ctl.ksstCoeffs;
                std::printf("  kOmegaSSTCoeffs: a1=%.4g betaStar=%.4g alphaK1=%.4g alphaOmega1=%.4g gamma1=%.4g beta1=%.4g\n",
                            c.a1, c.betaStar, c.alphaK1, c.alphaOmega1, c.gamma1, c.beta1);
            }
            else
            {
                // k-eps coefficients: RAS.kEpsilonCoeffs (kappa/E wall-function coeffs accepted if present).
                KEpsilonCoeffs& c = ctl.keCoeffs;
                if (rke)   // realizableKE: variable Cmu + strain production; OF defaults A0=4, C2=1.9, sigmak=1, sigmaEps=1.2
                {
                    c.realizable = true;
                    c.A0 = 4.0;
                    c.C2 = 1.9;
                    c.sigmaK = 1.0;
                    c.sigmaEps = 1.2;
                    if (const FoamDict* rkc = ras ? ras->subDict("realizableKECoeffs") : nullptr)
                    {
                        c.A0 = rkc->scalarOr("A0", c.A0);
                        c.C2 = rkc->scalarOr("C2", c.C2);
                        c.sigmaK = rkc->scalarOr("sigmak", c.sigmaK);
                        c.sigmaEps = rkc->scalarOr("sigmaEps", c.sigmaEps);
                        c.kappa = rkc->scalarOr("kappa", c.kappa);
                        c.E = rkc->scalarOr("E", c.E);
                    }
                    std::printf("  realizableKECoeffs: A0=%.4g C2=%.4g sigmak=%.4g sigmaEps=%.4g kappa=%.4g E=%.4g (variable Cmu)\n",
                                c.A0, c.C2, c.sigmaK, c.sigmaEps, c.kappa, c.E);
                }
                else
                {
                    const FoamDict* kec = ras ? ras->subDict("kEpsilonCoeffs") : nullptr;
                    if (kec)
                    {
                        c.Cmu = kec->scalarOr("Cmu", c.Cmu);
                        c.C1 = kec->scalarOr("C1", c.C1);
                        c.C2 = kec->scalarOr("C2", c.C2);
                        c.C3 = kec->scalarOr("C3", c.C3);
                        c.sigmaK = kec->scalarOr("sigmak", c.sigmaK);
                        c.sigmaEps = kec->scalarOr("sigmaEps", c.sigmaEps);
                        c.kappa = kec->scalarOr("kappa", c.kappa);
                        c.E = kec->scalarOr("E", c.E);
                    }
                    std::printf("  kEpsilonCoeffs%s: Cmu=%.4g C1=%.4g C2=%.4g C3=%.4g sigmak=%.4g sigmaEps=%.4g kappa=%.4g E=%.4g\n",
                                kec ? " (from dict)" : " (OF defaults)", c.Cmu, c.C1, c.C2, c.C3, c.sigmaK, c.sigmaEps, c.kappa, c.E);
                }
            }
        }

        // Scalar linearUpwind (deferred correction) is implemented in the scalar scaffold but currently DEGRADES
        // turbulence accuracy (explicit lagged correction + loose under-relaxed turb solve + steep near-wall
        // gradients): airFoil2D SA overshoots nuTilda (Cd worse), pitzDaily SST k/p worse than the upwind baseline.
        // Default OFF -> scalars use upwind (the validated behavior). BRAE_SCALAR_LINEARUPWIND=1 honours the scheme (A/B).
        if (!std::getenv("BRAE_SCALAR_LINEARUPWIND"))
        {
            // Detect-the-route: if fvSchemes ASKED for linearUpwind on a turbulence scalar, say loudly that cf is
            // running upwind instead (do not silently honour-then-ignore). The gating is deliberate (accuracy), but
            // the user must see that the requested scheme is not the one being applied.
            if (ctl.luK || ctl.luEps)
                std::fprintf(stderr, "brae WARNING: div(phi,<turbulence scalar>) requested 'linearUpwind' but cf is running "
                             "UPWIND (default: the scalar linearUpwind deferred correction degrades turbulence accuracy vs "
                             "OF). Set BRAE_SCALAR_LINEARUPWIND=1 to honour the requested scheme.\n");
            ctl.luK = false;
            ctl.luEps = false;
        }

        const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
        const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
        const FoamDict* fld = rf ? rf->subDict("fields") : nullptr;
        // OF accepts BOTH the modern nested {equations{} fields{}} and the legacy FLAT {p ..; U ..;} form. Fall back
        // to the flat keys (read from rf itself) when a sub-dict is absent, so a legacy case isn't silently left
        // un-relaxed (all factors 1.0 -> the steady SIMPLE loop typically diverges).
        const FoamDict* eqSrc  = eqs ? eqs : rf;
        const FoamDict* fldSrc = fld ? fld : rf;
        ctl.relaxU   = eqSrc  ? eqSrc->scalarOr("U", 1.0) : 1.0;
        ctl.relaxK   = eqSrc  ? eqSrc->scalarOr(ctl.sa ? "nuTilda" : "k", 1.0) : 1.0;   // SA: relaxK carries the nuTilda relax
        ctl.relaxEps = eqSrc  ? eqSrc->scalarOr(ctl.sst ? "omega" : "epsilon", 1.0) : 1.0;
        ctl.relaxP   = fldSrc ? fldSrc->scalarOr("p", 1.0) : 1.0;
        // A relaxation factor <= 0 divides by zero in the diagonal-relaxation kernel (Inf diag -> NaN). OF's
        // fvMatrix::relax skips relaxation for alpha <= 0; match that (treat as 1.0 = no under-relaxation) + warn.
        auto fixRelax = [](scalar& a, const char* nm) {
            if (a <= 0.0) { std::fprintf(stderr, "brae WARNING: relaxationFactors %s = %g <= 0; using 1.0 (no under-relaxation)\n", nm, a); a = 1.0; }
        };
        fixRelax(ctl.relaxU, "U");
        fixRelax(ctl.relaxK, ctl.sa ? "nuTilda" : "k");
        fixRelax(ctl.relaxEps, ctl.sst ? "omega" : "epsilon");
        fixRelax(ctl.relaxP, "p");

        const FoamDict* solvers = fvSolution.subDict("solvers");
        auto solverTol = [&](const std::string& f, scalar def)
        {
            const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
            return s ? s->scalarOr("tolerance", def) : def;
        };
        auto solverRelTol = [&](const std::string& f)   // SIMPLE only needs a loose per-step solve
        {
            const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
            return s ? s->scalarOr("relTol", 0.0) : 0.0;
        };
        ctl.tolP = solverTol("p", 1e-6);
        ctl.tolU = solverTol("U", 1e-8);
        const std::string second = ctl.sst ? "omega" : "epsilon";
        ctl.tolKE = ctl.sa ? solverTol("nuTilda", 1e-8) : std::fmin(solverTol("k", 1e-8), solverTol(second, 1e-8));
        ctl.relTolP = solverRelTol("p");
        ctl.relTolU = solverRelTol("U");
        ctl.relTolKE = ctl.sa ? solverRelTol("nuTilda")                     // SA: single nuTilda solve
                              : std::fmin(solverRelTol("k"), solverRelTol(second));   // OF solves k/(eps|omega) loosely per SIMPLE step
        // Per-field scalar linear solver from fvSolution, EXACTLY as OF selects it: solver smoothSolver +
        // smoother symGaussSeidel -> cf's deviceSymGaussSeidel. Any other (PBiCG[Stab]/GAMG/...) keeps BiCGStab.
        // OF smoothSolver with a GaussSeidel-family smoother (GaussSeidel = forward sweep, symGaussSeidel = fwd+bwd);
        // cf maps BOTH to its symmetric multicolor deviceSymGaussSeidel (a superset of forward GaussSeidel). Other
        // solvers (PBiCG[Stab]/GAMG/...) keep BiCGStab. motorBike uses "GaussSeidel" for U/k/omega; pitzDaily "symGaussSeidel".
        auto useSymGS = [&](const std::string& f)
        {
            const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
            if (!s || s->wordOr("solver", "") != "smoothSolver") return false;
            const std::string sm = s->wordOr("smoother", "");
            return sm == "symGaussSeidel" || sm == "GaussSeidel";
        };
        ctl.gsK   = useSymGS(ctl.sa ? "nuTilda" : "k");
        ctl.gsEps = ctl.sa ? false : useSymGS(second);
        // Momentum solver READ FROM fvSolution exactly like OF (fvVectorMatrix::solveSegregated solves each U component
        // with the U solver) and like cf already does for the scalars above. OF uses smoothSolver+symGaussSeidel for U
        // on most tutorials; cf's default Jacobi-preconditioned BiCGStab STALLS on the ill-conditioned high-aspect-ratio
        // viscous momentum matrix (backwardFacingStep2D AR~7600: the iter-1 predictor came out ~half of OF's, seeding the
        // divergence, divDevReff and rAU are OF-exact, the gap was purely the linear solve), whereas GS is robust like
        // OF's smoothSolver. BRAE_GS_U=0 forces BiCGStab (perf escape: GS is slower over the 3 components).
        const char* gsuEnv = std::getenv("BRAE_GS_U");
        ctl.gsU   = gsuEnv ? (std::atoi(gsuEnv) != 0 && useSymGS("U")) : useSymGS("U");
        // BiCGStab batched convergence DEFAULT OFF: measured net-NEGATIVE once relTol makes the solves few-iter
        // (overshoot to the K-boundary costs more than the saved syncs). Kept behind the env for tight-tol cases
        // (many inner iters), where it would help like the pressure pcgCheckEvery does.
        if (const char* be = std::getenv("BRAE_BICG_CHECK_EVERY"))
        {
            const int k = std::atoi(be);
            if (k >= 1) ctl.bicgCheckEvery = k;
        }
        ctl.pcgCheckEvery = 4;   // application default: batched PCG residual read (1.30x; OF-validated identical to K=1).
        if (const char* ce = std::getenv("BRAE_PCG_CHECK_EVERY"))
        {
            const int k = std::atoi(ce);
            if (k >= 1) ctl.pcgCheckEvery = k;   // override (1 = exact)
        }
        if (const char* cs = std::getenv("BRAE_CORR_SCALING")) ctl.corrScaling = (std::atoi(cs) != 0);   // AMG corr-scaling + flexible CG
        if (const char* ug = std::getenv("BRAE_USE_GRAPH")) ctl.useGraph = (std::atoi(ug) != 0);          // V-cycle graph replay toggle (debug)

        const FoamDict* simple = fvSolution.subDict("SIMPLE");
        const FoamDict* resCtl = simple ? simple->subDict("residualControl") : nullptr;
        const bool hasRC = (resCtl != nullptr);
        const scalar rcP = resCtl ? resCtl->scalarOr("p", -1) : -1, rcU = resCtl ? resCtl->scalarOr("U", -1) : -1;
        {
            const std::string cons = simple ? simple->wordOr("consistent", "no") : "no";
            ctl.consistent = (cons == "yes" || cons == "true" || cons == "on" || cons == "1");   // SIMPLEC
        }
        ctl.nNonOrth = simple ? simple->intOr("nNonOrthogonalCorrectors", 0) : 0;   // extra pressure passes on non-orthogonal meshes
        {
            const std::vector<scalar> bf = simple ? simple->scalarListOr("bodyForce", {}) : std::vector<scalar>{};
            if (bf.size() >= 3) ctl.bodyForce = vector{bf[0], bf[1], bf[2]};   // constant momentum source (drives cyclic/periodic channels)
        }

        std::printf("brae (device-resident) | case=%s | %s%s | nu=%.3g\n", caseDir.c_str(),
                    simType.c_str(), ctl.turbulent ? (ctl.sst ? " (kOmegaSST)" : " (kEpsilon)") : "", ctl.nu);
        std::printf("  relax U=%.2g p=%.2g | tol p=%.1g U=%.1g | endTime=%d | residualControl=%s\n",
                    ctl.relaxU, ctl.relaxP, ctl.tolP, ctl.tolU, endTime, hasRC ? "on" : "off");
        std::printf("  schemes: bounded=%d linearUpwind(U)=%d nonOrth(corrected)=%d nonOrthLimit=%.3g consistent(SIMPLEC)=%d limitedLinear(k=%d,eps=%d) linearUpwind(k=%d,eps=%d)\n",
                    ctl.bounded, ctl.linearUpwind, ctl.nonOrth, ctl.nonOrthLimit, ctl.consistent, ctl.limitedK, ctl.limitedEps, ctl.luK, ctl.luEps);
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
        g.build(m);
        _tsLap("geometry build");
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        const label nC = m.nCells();

        GeometricField<vector> U = buildField<vector>(readField<vector>(fieldDir + "/U"), fvp, nC);
        U.evaluateBoundary();
        GeometricField<scalar> p = buildField<scalar>(readField<scalar>(fieldDir + "/p"), fvp, nC);
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
        SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

        const std::string secondName = ctl.sst ? "omega" : "epsilon";   // the 2nd turbulence scalar
        // Wall-function fidelity guard -- fail loud on the two silently-wrong cases the BC parser can't catch:
        //  (1) nutUSpaldingWallFunction on a non-SA model: brae picks the wall model from the RAS model (Spalding for
        //      SA, nutk otherwise), NOT the BC, so it would silently apply nutkWallFunction (wrong wall shear/Cf/Cd).
        //  (2) a nut/epsilon/omega wall-function BC on a patch NOT typed 'wall': brae gates the near-wall model on the
        //      geometric patch type, so the wall function would be SILENTLY inert. Conservative on the patch match --
        //      only errors when the BC patch resolves to a concrete non-'wall' patch (group/regex names are skipped).
        auto patchGeoType = [&](const std::string& nm) -> std::string {
            for (const auto& q : fvp) if (q.name == nm) return q.type;
            return "";
        };
        auto guardWallFn = [&](const FieldData<scalar>& fd, const std::string& field) {
            auto isWF = [](const std::string& t) {
                return t == "nutkWallFunction" || t == "nutUSpaldingWallFunction" || t == "nutLowReWallFunction"
                    || t == "nutUBlendedWallFunction" || t == "epsilonWallFunction" || t == "omegaWallFunction";
            };
            for (const auto& pb : fd.boundary)
            {
                if (!isWF(pb.type)) continue;
                const std::string gt = patchGeoType(pb.name);
                if (!gt.empty() && gt != "wall")
                    throw std::runtime_error(field + " boundary '" + pb.name + "' uses " + pb.type + ", but the patch"
                        " is type '" + gt + "' (not 'wall'). brae applies the near-wall model only on 'wall' patches, so"
                        " it would be SILENTLY inert (no wall shear / near-wall constraint). Retype the patch as 'wall'"
                        " in constant/polyMesh/boundary.");
                if (pb.type == "nutUSpaldingWallFunction" && !ctl.sa)
                    throw std::runtime_error(field + " boundary '" + pb.name + "' uses nutUSpaldingWallFunction, but"
                        " this is a " + std::string(ctl.sst ? "kOmegaSST" : "kEpsilon") + " case -- brae applies"
                        " nutkWallFunction there (Spalding is SA-only), silently changing the wall shear. Use"
                        " nutkWallFunction, or run the SpalartAllmaras model.");
            }
        };
        GeometricField<scalar> k, eps, nut, ReThetat, gammaInt;   // ReThetat/gammaInt: kOmegaSSTLM transition
        if (ctl.sa)   // Spalart-Allmaras (one-equation): nuTilda -> the "k" slot, nut. BCs (freestream + fixedValue-0 wall) need no inlet calc.
        {
            k   = buildField<scalar>(readField<scalar>(fieldDir + "/nuTilda"), fvp, nC);
            k.evaluateBoundary();
            const FieldData<scalar> nutFD = readField<scalar>(fieldDir + "/nut");
            guardWallFn(nutFD, "nut");
            nut = buildField<scalar>(nutFD, fvp, nC);
            nut.evaluateBoundary();
        }
        else if (ctl.turbulent)
        {
            const FieldData<scalar> kFD = readField<scalar>(fieldDir + "/k");
            const FieldData<scalar> sFD = readField<scalar>(fieldDir + "/" + secondName);
            k   = buildField<scalar>(kFD, fvp, nC);
            k.evaluateBoundary();
            eps = buildField<scalar>(sFD, fvp, nC);
            eps.evaluateBoundary();
            const FieldData<scalar> nutFD = readField<scalar>(fieldDir + "/nut");
            guardWallFn(nutFD, "nut");
            guardWallFn(sFD, secondName);
            nut = buildField<scalar>(nutFD, fvp, nC);
            nut.evaluateBoundary();
            if (ctl.lm)   // kOmegaSSTLM transition fields
            {
                ReThetat = buildField<scalar>(readField<scalar>(fieldDir + "/ReThetat"), fvp, nC);
                ReThetat.evaluateBoundary();
                gammaInt = buildField<scalar>(readField<scalar>(fieldDir + "/gammaInt"), fvp, nC);
                gammaInt.evaluateBoundary();
                std::printf("  kOmegaSSTLM (Langtry-Menter transition): + ReThetat + gammaInt transport\n");
            }
            // turbulent-inlet BCs: inflow value computed from U (k) then from k (epsilon/omega).
            const scalar wCmu = ctl.sst ? ctl.ksstCoeffs.betaStar : ctl.keCoeffs.Cmu;
            applyTurbulentInletK(k, kFD, U, fvp);
            applyTurbulentInletSecond(eps, sFD, k, wCmu, fvp);
        }

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
                                  ctl.turbulent ? &k : nullptr, (ctl.turbulent && !ctl.sa) ? &eps : nullptr, ctl.turbulent ? &nut : nullptr,
                                  ctl.lm ? &ReThetat : nullptr, ctl.lm ? &gammaInt : nullptr);
        _tsLap("solver ctor (incl AMG)");
        if (partition)   // caches written by the mesh read + the AMG build above; done, like decomposePar finishing.
        {
            std::printf("brae -partition: mesh + AMG hierarchy cached to %s/constant/polyMesh/ (.brae_meshcache + .brae_amgcache).\n"
                        "  Run the solve normally; it will reload them warm.\n", caseDir.c_str());
            return 0;
        }
        if (mrfCfg.active) solver.setMRF(mrfZone, m, g, mrfCfg.nonRotatingPatches);

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
        auto ok = [](scalar res, scalar ctlv) { return ctlv < 0 || res < ctlv; };
        // OF controlDict write cadence: writeControl / writeInterval / purgeWrite (ported from Foam::Time)
        const std::string writeControl = controlDict.wordOr("writeControl", "timeStep");
        const scalar writeInterval = controlDict.scalarOr("writeInterval", 1e30);   // OF default GREAT -> only the final state
        const int    purgeWrite    = std::max(0, controlDict.intOr("purgeWrite", 0));
        const scalar deltaT        = controlDict.scalarOr("deltaT", 1.0);
        const scalar startTimeVal  = controlDict.scalarOr("startTime", 0.0);
        long writeTimeIndex = 0;
        auto timeName = [](scalar t) -> std::string   // integer name for whole times (deltaT=1), else %g
        {
            if (t == std::floor(t) && std::fabs((double)t) < 1e15) return std::to_string((long long)std::llround((double)t));
            char b[64];
            std::snprintf(b, sizeof b, "%g", (double)t);
            return std::string(b);
        };
        auto isWriteTime = [&](int it, scalar tval) -> bool   // Foam::Time::operator++ write switch
        {
            if (writeControl == "timeStep")
                return writeInterval >= 1 && (it % (long)writeInterval) == 0;       // Time.C: !(timeIndex % writeInterval)
            if (writeControl == "runTime" || writeControl == "adjustable" || writeControl == "adjustableRunTime")
            {
                const long wi = (long)(((tval - startTimeVal) + 0.5 * deltaT) / writeInterval);
                if (wi > writeTimeIndex)
                {
                    writeTimeIndex = wi;
                    return true;
                }
                return false;
            }
            return false;   // none / clockTime / cpuTime: only the final state (written after the loop)
        };
        std::deque<std::string> writtenTimes;                                       // purgeWrite FIFO (keep the last N)
        const std::string wsrc = fieldDir + "/";
        auto writeTimeDir = [&](const std::string& tname)   // reconstruct + write one time directory
        {
            const std::string outDir = caseDir + "/" + tname;
            std::filesystem::create_directories(outDir);
            // The written fields keep the source field's `#include "include/..."` directives (resolved relative to the
            // time dir). Copy the startTime include/ dir alongside so OpenFOAM readers (paraFoam, postProcess, foamToVTK)
            // resolve them, otherwise the field points at a nonexistent <time>/include/ and fails to load.
            {
                std::error_code ec2;
                if (std::filesystem::exists(fieldDir + "/include"))
                    std::filesystem::copy(fieldDir + "/include", outDir + "/include",
                        std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec2);
            }
            writeVolField(wsrc + "U", outDir + "/U", solver.U(), fvp, precision);
            writeVolField(wsrc + "p", outDir + "/p", solver.p(), fvp, precision);
            if (ctl.sa)   // one-equation: the k slot holds nuTilda
            {
                writeVolField(wsrc + "nuTilda", outDir + "/nuTilda", solver.k(),   fvp, precision);
                writeVolField(wsrc + "nut",     outDir + "/nut",     solver.nut(), fvp, precision);
            }
            else if (ctl.turbulent)
            {
                writeVolField(wsrc + "k", outDir + "/k", solver.k(), fvp, precision);
                writeVolField(wsrc + secondName, outDir + "/" + secondName, solver.eps(), fvp, precision);
                writeVolField(wsrc + "nut", outDir + "/nut", solver.nut(), fvp, precision);
                if (ctl.lm)   // kOmegaSSTLM transition fields
                {
                    writeVolField(wsrc + "ReThetat", outDir + "/ReThetat", solver.ReThetat(), fvp, precision);
                    writeVolField(wsrc + "gammaInt", outDir + "/gammaInt", solver.gammaInt(), fvp, precision);
                }
            }
            std::printf("written %s/{U,p%s}\n", outDir.c_str(),
                        ctl.turbulent ? (ctl.sa ? ",nuTilda,nut" : ctl.sst ? ",k,omega,nut" : ",k,epsilon,nut") : "");
            if (purgeWrite > 0)   // Foam::Time TimeIO: keep only the last purgeWrite dirs
            {
                if (writtenTimes.empty() || writtenTimes.back() != tname) writtenTimes.push_back(tname);
                while ((int)writtenTimes.size() > purgeWrite)
                {
                    std::error_code pec;
                    std::filesystem::remove_all(caseDir + "/" + writtenTimes.front(), pec);
                    writtenTimes.pop_front();
                }
            }
        };

        int iter = 0;
        bool converged = false;
        const auto _runStart = std::chrono::high_resolution_clock::now();          // for OpenFOAM-style ExecutionTime
        double _cumCont = 0.0;                                                     // cumulative continuity error (OF continuityErrs.H)
        for (iter = 1; iter <= endTime && !converged; ++iter)
        {
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
            // OF residualControl: also gate on every turbulence field (k/epsilon/omega/nuTilda) that lists a target.
            // Previously ONLY p and Ux were checked, so a turbulent case could report "converged" with k/epsilon
            // still far from tol -- the substantive bug this fixes. Unlisted fields have target -1 -> ok() ignores
            // them (OF). U stays gated on Ux alone: brae tracks no valid/solved directions, so the out-of-plane
            // component of a 2D/empty or wedge case has a DEGENERATE residual (stuck ~0.1, never reaching tol) that
            // would wrongly block convergence on every 2D case -- gating all U components needs that infra first.
            converged = hasRC && ok(r.p, rcP) && ok(r.Ux, rcU);
            if (converged)
                for (const auto& e : turbulenceReport())
                    if (!ok(e.perf.initialResidual, resCtl->scalarOr(e.field, -1))) { converged = false; break; }
            const scalar tval = startTimeVal + (scalar)iter * deltaT;               // OF time value at this step
            if (!converged && iter != endTime && isWriteTime(iter, tval)) writeTimeDir(timeName(tval));  // intermediate writes
        }
        const int nIter = converged ? iter - 1 : endTime;
        std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                              : "SIMPLE reached endTime (%d iterations)\n", nIter);

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
        writeTimeDir(timeName(startTimeVal + (scalar)nIter * deltaT));
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
                // SA: use the device nutUSpaldingWallFunction wall nut for the viscous force (not the k-based one).
                const std::vector<scalar> saNutWall = ctl.sa ? solver.nutWall() : std::vector<scalar>();
                const std::vector<scalar>* nwb = ctl.sa ? &saNutWall : nullptr;
                const ForceResult F = wallForces(U, p, solver.k(), ctl.nu, m, g, fvp, walls, 1.0, 0.0, vector{0,0,0}, wCmu, wKappa, wE, nwb);
                std::printf("forces (walls, rhoInf=1):  pressure=(%.5e %.5e %.5e)  viscous=(%.5e %.5e %.5e)  total=(%.5e %.5e %.5e)\n",
                            F.pressure.x, F.pressure.y, F.pressure.z, F.viscous.x, F.viscous.y, F.viscous.z, F.total().x, F.total().y, F.total().z);

                // forceCoeffs (Cd/Cl/Cm) if the case defines a forceCoeffs functionObject (controlDict.functions).
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
