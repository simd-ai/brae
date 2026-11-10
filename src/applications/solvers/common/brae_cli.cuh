#pragma once
// The `brae` command line's own commands -- `brae --version`, `brae solvers`, `brae capabilities`, `brae check`, `brae run` -- and
// the rule that decides when a leading word is one of them and when it is a case directory.
//
// Every list printed here is a list the compiled code checks a case against: braeSolvers (the dispatch table),
// patchFieldTypeNames (the boundary-condition factory's), divSchemeRules (parseFvSchemesControls'), rasModelNames / lesModelNames
// / lesDeltaNames (readTurbulenceModel's), pimpleDdtSchemeNames (brae_pimpleFoam's). Nothing is typed out a second time here, so
// what `brae capabilities` says and what a run accepts cannot drift apart. What no table carries (the linear-solver policy, the
// known approximations) is stated as a policy with the place in the code that applies it.
//
// `brae check -case D` reads the case's own dictionaries through the same functions a run calls, collects every refusal instead
// of stopping at the first, and reports the notices a run would print. `--setup` goes further: it runs the case's solver through
// its whole setup (mesh, fields, every boundary condition, the device arrays) and stops before the first step.
//
// The multi-GPU layer is a separate program (brae-mgpu, the private build). `brae` links none of it: it hands `-parallel` runs to
// it and asks it for its own capabilities, when it is installed next to brae.

#include "solver_dispatch.cuh"
#include "scheme_parse.cuh"
#include "turbulence_setup.cuh"
#include "linear_solver_setup.cuh"
#include "dict_audit.cuh"
#include "start_time.cuh"
#include "solid_body_motion.cuh"   // readMeshMotion: what pimpleFoam refuses of a dynamicMeshDict
#include "mrf_read.cuh"            // readMRFProperties: the zones simpleFoam rotates
#include "patch_entry_lookup.cuh"  // findPatchEntry: the entry a patch takes, the solver's rule
#include "pimple_controls.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "check_setup.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef BRAE_VERSION_STRING
#define BRAE_VERSION_STRING "unknown"
#endif
#ifndef BRAE_GIT_COMMIT
#define BRAE_GIT_COMMIT "unknown"
#endif
#ifndef BRAE_COMMIT_ORIGIN
#define BRAE_COMMIT_ORIGIN "none"
#endif
#ifndef BRAE_CUDA_ARCHS_STRING
#define BRAE_CUDA_ARCHS_STRING "unknown"
#endif

namespace brae {
namespace cli {

// ---- the command words -------------------------------------------------------------------------------------------------------

// The words `brae` takes as commands besides the reserved benchmark / node / job. A case directory of the same name stays a case
// (isCommand below), so none of these takes a case path away from anyone.
inline const std::vector<std::string>& commandWords()
{
    static const std::vector<std::string> w = {"run", "check", "solvers", "capabilities"};
    return w;
}

inline bool isCaseDirectory(const std::string& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(std::filesystem::path(path) / "system" / "controlDict", ec);
}

// Is argv[1] a command? A command word followed by an option (`brae run -case D`, `brae capabilities --json`) always is. A bare
// command word is a command unless a case directory of that name is here: then it is that case, as it always was (brae <dir>),
// and brae says so. Decided by the arguments and that one directory test alone, so the same command line in the same directory
// always means the same thing, and an existing case path is never read as a command.
inline bool isCommand(int argc, char** argv)
{
    if (argc < 2) return false;
    const std::string w = argv[1];
    if (!listHas(commandWords(), w)) return false;
    if (argc > 2 && argv[2][0] == '-') return true;
    if (isCaseDirectory(w))
    {
        std::fprintf(stderr, "brae: './%s' is a case directory, so `brae %s` runs that case, as it always has. The command is "
                             "`brae %s -case <dir>`%s.\n", w.c_str(), w.c_str(), w.c_str(),
                     (w == "solvers" || w == "capabilities") ? " (or with --json)" : "");
        return false;
    }
    return true;
}

// The case of a command line: the last `-case <dir>`, else the last bare word after the command, else the current directory.
inline std::string caseOf(int argc, char** argv, int first)
{
    std::string c = ".";
    for (int i = first; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "-case" && i + 1 < argc)
            c = argv[++i];
        else if (!a.empty() && a[0] != '-')
            c = a;
    }
    return c;
}

inline bool hasFlag(int argc, char** argv, int first, const char* flag)
{
    for (int i = first; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// ---- the multi-GPU sibling ---------------------------------------------------------------------------------------------------

// brae-mgpu next to this binary, "" when this install has none
inline std::string mgpuSibling()
{
    std::error_code ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) return "";
    const std::filesystem::path s = self.parent_path() / "brae-mgpu";
    return std::filesystem::exists(s, ec) ? s.string() : std::string();
}

// A command's standard output, "" on failure
inline std::string captureCommand(const std::string& cmd)
{
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return out;
    char buf[4096];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
    if (pclose(p) != 0) return "";
    return out;
}

// ---- --version ---------------------------------------------------------------------------------------------------------------

inline int version()
{
    int rt = 0;
    int drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    // where the commit came from (CMakeLists.txt): git at configure, the packer's BRAE_SOURCE_COMMIT in a source tarball, or
    // neither
    const std::string origin = BRAE_COMMIT_ORIGIN;
    if (origin == "stamp")
        std::printf("brae %s (commit %s, stamped in the source tree's BRAE_SOURCE_COMMIT)\n", BRAE_VERSION_STRING, BRAE_GIT_COMMIT);
    else if (origin == "none")
        std::printf("brae %s (commit unknown: no git and no BRAE_SOURCE_COMMIT at configure)\n", BRAE_VERSION_STRING);
    else
        std::printf("brae %s (commit %s at configure)\n", BRAE_VERSION_STRING, BRAE_GIT_COMMIT);
    std::printf("  CUDA runtime %d.%d, built for CUDA %d.%d, device code for architectures %s\n", rt / 1000, (rt % 1000) / 10,
                CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10, BRAE_CUDA_ARCHS_STRING);
    if (drv > 0)
        std::printf("  CUDA driver supports up to %d.%d\n", drv / 1000, (drv % 1000) / 10);
    else
        std::printf("  CUDA driver: none found (brae needs one to run a case; these commands do not)\n");
    std::printf("  arithmetic: FP64 throughout\n");
    const std::string sib = mgpuSibling();
    if (sib.empty())
        std::printf("  multi-GPU layer: not installed (brae-mgpu next to brae)\n");
    else
    {
        std::string v = captureCommand("'" + sib + "' --version 2>/dev/null");
        while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
        std::printf("  multi-GPU layer: %s\n", v.empty() ? (sib + " (did not report a version)").c_str() : v.c_str());
    }
    return 0;
}

// ---- the capability registry -------------------------------------------------------------------------------------------------

// One thing brae can be asked to do, as a case would ask for it. status: validated (a named gate holds it against OpenFOAM),
// implemented (in the code, no gate of its own named here), experimental (an opt-in path, off unless an environment variable asks
// for it), approximated (runs a near-equivalent and says so), refused (named so the refusal can say why). mode: "1 GPU", "multi-GPU"
// or both.
struct Capability
{
    std::string category;
    std::string name;
    std::string status;
    std::string solvers;
    std::string mode;
    std::string note;
    std::string gate;
};

// The gates that hold a capability against OpenFOAM, by "category/name". Each is a registered test (tests/brae_cli.sh holds every
// name here to `ctest -N`); a capability without one is reported as implemented, never as validated.
inline const std::map<std::string, std::string>& validationGates()
{
    static const std::map<std::string, std::string> g = {
        {"solver/simpleFoam", "simple_ahmed_limited_vs_openfoam simple_sst_start_vs_openfoam"},
        {"solver/pimpleFoam", "simple_ahmed_limited_vs_openfoam pimple_run"},
        {"solver/rhoSimpleFoam", "rho_simple_step_cuda rho_angledduct_structural"},
        {"solver/interFoam", "interfoam_dambreak_vs_openfoam interfoam_ras_dambreak_vs_openfoam"},
        {"turbulence/kOmegaSST", "simple_ahmed_limited_vs_openfoam"},
        {"turbulence/kEpsilon", "simple_ahmed_limited_vs_openfoam"},
        {"turbulence/SpalartAllmaras", "simple_ahmed_limited_vs_openfoam"},
        {"turbulence/kOmegaSSTDDES", "gpu_ksstddes"},
        {"turbulence/kOmegaSSTIDDES", "gpu_ksstiddes"},
        {"turbulence/SpalartAllmarasDDES", "gpu_saddes"},
        {"turbulence/SpalartAllmarasIDDES", "gpu_saiddes"},
        {"snGrad/limited [corrected] <psi>", "simple_ahmed_limited_vs_openfoam"},
        {"snGrad/corrected", "simple_ahmed_limited_vs_openfoam"},
        {"snGrad/uncorrected", "simple_scalar_uncorrected_vs_openfoam"},
        {"ddt/backward", "ddt_corr_backward"},
        {"ddt/Euler", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/nutUSpaldingWallFunction", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/nutkWallFunction", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/kqRWallFunction", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/omegaWallFunction", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/epsilonWallFunction", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/inletOutlet", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/noSlip", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/fixedValue", "simple_ahmed_limited_vs_openfoam"},
        {"boundary condition/zeroGradient", "simple_ahmed_limited_vs_openfoam"},
        {"div scheme/U linearUpwind", "simple_ahmed_limited_vs_openfoam"},
        {"div scheme/k limitedLinear", "simple_ahmed_limited_vs_openfoam"},
        {"div scheme/epsilon|omega limitedLinear", "simple_ahmed_limited_vs_openfoam"},
        {"div scheme/nuTilda limitedLinear", "simple_ahmed_limited_vs_openfoam"},
    };
    return g;
}

inline Capability cap(const std::string& category, const std::string& name, const std::string& status, const std::string& solvers,
                      const std::string& mode, const std::string& note)
{
    Capability c{category, name, status, solvers, mode, note, ""};
    const auto it = validationGates().find(category + "/" + name);
    if (it != validationGates().end())
    {
        c.gate = it->second;
        if (c.status == "implemented") c.status = "validated";
    }
    return c;
}

// The multi-GPU layer's own entries, as `brae-mgpu capabilities --lines` prints them (category|name|status|solvers|note, one a
// line), each marked multi-GPU; none when brae-mgpu is not installed or does not answer.
inline std::vector<Capability> mgpuCapabilities()
{
    std::vector<Capability> out;
    const std::string sib = mgpuSibling();
    if (sib.empty()) return out;
    std::istringstream in(captureCommand("'" + sib + "' capabilities --lines 2>/dev/null"));
    std::string line;
    while (std::getline(in, line))
    {
        std::vector<std::string> f;
        std::string cell;
        std::istringstream ls(line);
        while (std::getline(ls, cell, '|')) f.push_back(cell);
        if (f.size() < 5) continue;
        Capability c{f[0], f[1], f[2], f[3], "multi-GPU", f[4], f.size() > 5 ? f[5] : ""};
        out.push_back(c);
    }
    return out;
}

inline std::vector<Capability> capabilities()
{
    std::vector<Capability> v;
    const std::string legacy3 = "simpleFoam pimpleFoam rhoSimpleFoam";
    // solvers: the dispatch table
    for (const BraeSolver& s : braeSolvers())
        v.push_back(cap("solver", s.application, "implemented", s.application, "1 GPU",
                        std::string(s.what) + "; executable " + s.exe + (s.transient ? ", transient" : ", steady")));
    // turbulence: readTurbulenceModel's lists
    for (const std::string& m : rasModelNames())
        v.push_back(cap("turbulence", m, "implemented", m == "kOmegaSSTLM" ? "simpleFoam pimpleFoam" : legacy3, "1 GPU",
                        "RAS (rhoSimpleFoam's mirror: kEpsilon and kOmegaSST only; interFoam: kEpsilon and kOmegaSST)"));
    for (const std::string& m : lesModelNames())
    {
        const bool plainDes = (m == "kOmegaSSTDES" || m == "SpalartAllmarasDES");
        v.push_back(cap("turbulence", m, plainDes ? "approximated" : "implemented", "pimpleFoam", "1 GPU",
                        plainDes ? "LES; runs with DDES shielding, which OpenFOAM's plain DES has not" : "LES"));
    }
    for (const std::string& d : lesDeltaNames())
        v.push_back(cap("LES delta", d, "implemented", "pimpleFoam", "1 GPU", "IDDES makes its own length scale"));
    v.push_back(cap("turbulence", "laminar (Stokes)", "implemented", "simpleFoam pimpleFoam rhoSimpleFoam interFoam", "1 GPU",
                    "Maxwell and generalizedNewtonian powerLaw per solver (readLaminarModel's envelope)"));
    // boundary conditions: the factory's list
    for (const std::string& t : patchFieldTypeNames())
    {
        const bool refused = listHas(patchFieldTypesAlwaysRefused(), t);
        std::string note = "one factory for every solver (fv_patch_field.cuh)";
        if (t == "fixedMean" || t == "fanPressure") note = "refused by simpleFoam (built once, never updated per step)";
        if (refused) note = "named so its refusal can say why; not implemented";
        v.push_back(cap("boundary condition", t, refused ? "refused" : "implemented", refused ? "" : "all", "1 GPU", note));
    }
    // div schemes: parseFvSchemesControls' rules
    for (const DivSchemeRule& r : divSchemeRules())
    {
        for (const std::string& s : r.exact)
            v.push_back(cap("div scheme", std::string(r.label) + " " + s, "implemented", legacy3, "1 GPU",
                            "div(phi," + std::string(r.label) + ") Gauss " + s + ", bounded or not"));
        for (const std::string& s : r.approximated)
            v.push_back(cap("div scheme", std::string(r.label) + " " + s, "approximated", legacy3, "1 GPU",
                            "run as a near-equivalent, with a brae WARNING (not OpenFOAM to the bit)"));
    }
    // snGrad / laplacian: readSnGrad's four
    for (const char* s : {"corrected", "uncorrected", "orthogonal", "limited [corrected] <psi>"})
        v.push_back(cap("snGrad", s, "implemented", legacy3, "1 GPU",
                        "laplacianSchemes and snGradSchemes; one snGrad for every laplacian; any other word is refused"));
    // ddt
    v.push_back(cap("ddt", "steadyState", "implemented", "simpleFoam rhoSimpleFoam", "1 GPU", "steady solvers"));
    for (const std::string& d : pimpleDdtSchemeNames())
        v.push_back(cap("ddt", d, "implemented", "pimpleFoam", "1 GPU", "parsePimpleDdtScheme (pimple_controls.cuh)"));
    v.push_back(cap("ddt", "Euler / backward / CrankNicolson / localEuler", "implemented", "interFoam", "1 GPU",
                    "interFoam's own reader (inter_case_cpp.cu)"));
    // linear solvers: a policy, not a table
    v.push_back(cap("linear solver", "p, p_rgh, pcorr: any entry", "approximated", "all", "1 GPU",
                    "brae runs its AMG-preconditioned PCG to the entry's tolerance and relTol, whatever solver it names, and says "
                    "so (NOTICE [approximated]); the policy in CLAUDE.md"));
    v.push_back(cap("linear solver", "U, k, epsilon, omega, nuTilda: smoothSolver GaussSeidel|symGaussSeidel", "implemented",
                    legacy3, "1 GPU", "OpenFOAM's sweep, its stopping rule (tolerance, relTol, maxIter, minIter, nSweeps)"));
    v.push_back(cap("linear solver", "U, k, epsilon, omega, nuTilda: any other entry", "approximated", legacy3, "1 GPU",
                    "brae's PBiCGStab (DILU honoured) to the entry's criteria, with a NOTICE"));
    // gradients
    v.push_back(cap("grad scheme", "Gauss linear, cellLimited Gauss linear <k>", "implemented", legacy3, "1 GPU", ""));
    v.push_back(cap("grad scheme", "leastSquares", "approximated", legacy3, "1 GPU", "run as Gauss linear, with a brae WARNING"));
    v.push_back(cap("grad scheme", "cellMDLimited", "approximated", legacy3, "1 GPU", "the limiter is not applied, with a brae WARNING"));
    // mesh and parallel
    v.push_back(cap("mesh", "polyMesh ascii / binary / gzip", "implemented", "all", "1 GPU", "cellZones ascii and binary"));
    v.push_back(cap("mesh", "cyclic, cyclicAMI, cyclicACMI", "implemented", "all", "1 GPU", "simpleFoamV2 refuses coupled patches"));
    v.push_back(cap("parallel", "-parallel (processor patches, decomposition)", mgpuSibling().empty() ? "refused" : "implemented",
                    "", "multi-GPU", mgpuSibling().empty() ? "this install has no multi-GPU layer (brae-mgpu)"
                                                         : "handed to brae-mgpu; its entries follow"));
    // opt-in solver paths: off unless asked for, never chosen by a case's own settings
    v.push_back(cap("solver path", "simpleFoam V2 driver", "experimental", "simpleFoam", "1 GPU",
                    "BRAE_SIMPLEFOAM_V2=1; refuses coupled patches; no [unread] report"));
    v.push_back(cap("solver path", "rhoSimpleFoam OF-mirror", "experimental", "rhoSimpleFoam", "1 GPU",
                    "BRAE_RHOSIMPLEFOAM_MIRROR=1 (host) or cuda; the CPU-first re-port"));
    // time and restart
    v.push_back(cap("time", "startFrom startTime | latestTime | firstTime", "implemented", "all", "1 GPU",
                    "a restart reads phi when written (READ_IF_PRESENT); <startTime>.orig restored when <startTime> has no U"));
    // known numerical limitations that are not one table's entry
    v.push_back(cap("limitation", "div(phi,k|epsilon|omega) linearUpwind on simpleFoam", "approximated", "simpleFoam", "1 GPU",
                    "runs upwind (cold-start guard, brae WARNING); BRAE_SCALAR_LINEARUPWIND=1 honours it"));
    v.push_back(cap("limitation", "unread-entry audit", "implemented", "simpleFoam pimpleFoam rhoSimpleFoam", "1 GPU",
                    "interFoam and simpleFoamV2 print no [unread] report"));
    for (const Capability& c : mgpuCapabilities()) v.push_back(c);
    return v;
}

inline std::string jsonEsc(const std::string& s)
{
    std::string o;
    for (const char c : s)
    {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

inline int capabilitiesCommand(int argc, char** argv)
{
    const std::vector<Capability> v = capabilities();
    if (hasFlag(argc, argv, 2, "--json"))
    {
        std::printf("{\n  \"brae\": \"%s\",\n  \"commit\": \"%s\",\n  \"capabilities\": [\n", BRAE_VERSION_STRING, BRAE_GIT_COMMIT);
        for (std::size_t i = 0; i < v.size(); ++i)
            std::printf("    {\"category\": \"%s\", \"name\": \"%s\", \"status\": \"%s\", \"solvers\": \"%s\", \"mode\": \"%s\", "
                        "\"note\": \"%s\", \"gate\": \"%s\"}%s\n",
                        jsonEsc(v[i].category).c_str(), jsonEsc(v[i].name).c_str(), jsonEsc(v[i].status).c_str(),
                        jsonEsc(v[i].solvers).c_str(), jsonEsc(v[i].mode).c_str(), jsonEsc(v[i].note).c_str(),
                        jsonEsc(v[i].gate).c_str(), i + 1 < v.size() ? "," : "");
        std::printf("  ]\n}\n");
        return 0;
    }
    std::printf("brae %s capabilities: what this build accepts in a case, from the tables the code checks cases against.\n"
                "status: validated (a named gate holds it against OpenFOAM) | implemented | experimental (opt-in, off by default) | "
                "approximated (a near-equivalent, announced) | refused (named, not run)\n", BRAE_VERSION_STRING);
    std::string cat;
    for (const Capability& c : v)
    {
        if (c.category != cat)
        {
            cat = c.category;
            std::printf("\n%s\n", cat.c_str());
        }
        std::printf("  %-12s %-44s %-9s %s%s%s%s\n", c.status.c_str(), c.name.c_str(), c.mode.c_str(),
                    c.solvers.empty() ? "" : ("[" + c.solvers + "] ").c_str(), c.note.c_str(),
                    c.gate.empty() ? "" : "  gate: ", c.gate.c_str());
    }
    return 0;
}

inline int solversCommand()
{
    std::printf("brae %s solvers (picked from the case's controlDict `application`):\n\n", BRAE_VERSION_STRING);
    std::printf("  %-16s %-20s %-10s %-14s %s\n", "application", "executable", "1 GPU", "multi-GPU", "what");
    const std::vector<Capability> m = mgpuCapabilities();
    for (const BraeSolver& s : braeSolvers())
    {
        std::string multi = "no";
        for (const Capability& c : m)
            if (c.category == "solver" && c.name == s.application) multi = c.status;
        if (m.empty()) multi = "not installed";
        const auto it = validationGates().find(std::string("solver/") + s.application);
        std::printf("  %-16s %-20s %-10s %-14s %s\n", s.application, s.exe,
                    it != validationGates().end() ? "validated" : "yes", multi.c_str(), s.what);
    }
    std::printf("\nAny other application is refused at start-up rather than run with the wrong solver. Multi-GPU runs "
                "(-parallel) are the brae-mgpu layer's%s.\n", mgpuSibling().empty() ? ", which this install does not have" : "");
    return 0;
}

// ---- check -------------------------------------------------------------------------------------------------------------------

// Everything written to stderr between start() and stop(): the NOTICE and WARNING lines the parsers print
class StderrCapture
{
public:
    void start()
    {
        std::fflush(stderr);
        saved_ = dup(2);
        tmp_ = std::tmpfile();
        if (tmp_) dup2(fileno(tmp_), 2);
    }
    std::string stop()
    {
        std::fflush(stderr);
        std::string out;
        if (saved_ >= 0)
        {
            dup2(saved_, 2);
            close(saved_);
            saved_ = -1;
        }
        if (tmp_)
        {
            std::rewind(tmp_);
            char buf[4096];
            std::size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, tmp_)) > 0) out.append(buf, n);
            std::fclose(tmp_);
            tmp_ = nullptr;
        }
        return out;
    }

private:
    int saved_ = -1;
    FILE* tmp_ = nullptr;
};

struct CheckReport
{
    struct Line
    {
        std::string topic;
        std::string verdict;   // ok, notice, refused, error
        std::string text;
    };
    std::vector<Line> lines;
    void add(const std::string& topic, const std::string& verdict, const std::string& text) { lines.push_back({topic, verdict, text}); }
    bool any(const std::string& verdict) const
    {
        for (const Line& l : lines)
            if (l.verdict == verdict) return true;
        return false;
    }
};

// Runs one step of the check: a throw is the step's refusal, each NOTICE / WARNING line it printed is a notice, and what it
// returns is its line when it passes
template <typename F>
inline void checkStep(CheckReport& r, const std::string& topic, F&& f)
{
    StderrCapture cap;
    cap.start();
    std::string err;
    std::string okText;
    try
    {
        okText = f();
    }
    catch (const std::exception& e)
    {
        err = e.what();
    }
    const std::string printed = cap.stop();
    std::istringstream in(printed);
    std::string line;
    while (std::getline(in, line))
        if (line.find("NOTICE") != std::string::npos || line.find("WARNING") != std::string::npos)
            r.add(topic, "notice", line);
    if (!err.empty())
        r.add(topic, "refused", err);
    else
        r.add(topic, "ok", okText);
}

// The class word of an OpenFOAM field file's header, "" if it has none
inline std::string fieldClass(const std::string& path)
{
    std::ifstream f(path);
    std::string head(4096, '\0');
    f.read(&head[0], static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<std::size_t>(f.gcount()));
    const std::size_t c = head.find("class");
    if (c == std::string::npos) return "";
    std::istringstream ls(head.substr(c + 5));
    std::string w;
    ls >> w;
    while (!w.empty() && w.back() == ';') w.pop_back();
    return w;
}

inline int checkCommand(int argc, char** argv)
{
    const std::string caseDir = caseOf(argc, argv, 2);
    const bool setup = hasFlag(argc, argv, 2, "--setup");
    std::printf("brae check %s\n", caseDir.c_str());
    if (!isCaseDirectory(caseDir))
    {
        std::printf("  [error]   %s has no system/controlDict: not an OpenFOAM case\nverdict: CASE ERROR\n", caseDir.c_str());
        return 3;
    }
    setCaseEnvironment(caseDir);   // FOAM_CASE, as the solver will have it (argList.C:891-893)
    CheckReport r;
    FoamDict controlDict;
    const BraeSolver* solver = nullptr;
    checkStep(r, "solver", [&]() {
        controlDict = readDict(caseDir + "/system/controlDict");
        const SolverDecision d = decideSolver(caseDir, controlDict);
        solver = d.row;
        return std::string(solver->application) + " (" + solver->exe + ", 1 GPU): " + solver->what;
    });
    const std::string app = solver ? solver->application : "";
    const bool legacyFamily = (app == "simpleFoam" || app == "pimpleFoam" || app == "rhoSimpleFoam");

    // the mesh: its patches, as the boundary file names them -- name, type and inGroups, what the solver's field reader
    // resolves each boundaryField entry against (findPatchEntry)
    struct CheckPatch
    {
        std::string              name;
        std::string              type;
        std::vector<std::string> inGroups;
    };
    std::vector<CheckPatch> meshPatches;
    checkStep(r, "mesh", [&]() {
        std::ifstream f(caseDir + "/constant/polyMesh/boundary");
        if (!f) throw std::runtime_error("no constant/polyMesh/boundary");
        // without its comments, which OpenFOAM's lexer drops: a comment holding a "(" before the list was taken for it
        const std::string text = stripFoamComments(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
        // the patch list: `name { ... }` entries at depth 1 of the list's parentheses, after the FoamFile header
        std::size_t i = text.find("FoamFile");
        i = (i == std::string::npos) ? 0 : text.find('}', i) + 1;
        i = text.find('(', i);
        if (i == std::string::npos) throw std::runtime_error("constant/polyMesh/boundary has no patch list");
        std::map<std::string, int> kinds;
        int n = 0;
        std::size_t k = i + 1;
        std::string word;
        bool afterSpace = true;
        while (k < text.size())
        {
            const char c = text[k];
            if (c == '{')
            {
                // the patch body, to its matching brace
                int depth = 1;
                const std::size_t open = k;
                while (++k < text.size() && depth > 0)
                {
                    if (text[k] == '{') ++depth;
                    if (text[k] == '}') --depth;
                }
                const std::string patchBody = text.substr(open, k - open);
                std::string type;
                const std::size_t t = patchBody.find("type");
                if (t != std::string::npos)
                {
                    std::istringstream ls(patchBody.substr(t + 4));
                    ls >> type;
                    while (!type.empty() && type.back() == ';') type.pop_back();
                }
                CheckPatch cp;
                cp.name = word;
                cp.type = type;
                // inGroups N(a b) or (a b)
                const std::size_t g = patchBody.find("inGroups");
                if (g != std::string::npos)
                {
                    const std::size_t lp = patchBody.find('(', g);
                    const std::size_t rp = patchBody.find(')', lp == std::string::npos ? g : lp);
                    if (lp != std::string::npos && rp != std::string::npos)
                    {
                        std::istringstream gs(patchBody.substr(lp + 1, rp - lp - 1));
                        std::string grp;
                        while (gs >> grp) cp.inGroups.push_back(grp);
                    }
                }
                meshPatches.push_back(cp);
                ++kinds[type];
                ++n;
                word.clear();
                afterSpace = true;
                continue;
            }
            if (c == ')') break;
            // a patch's name is the last token before its `{`
            if (std::isspace(static_cast<unsigned char>(c)))
            {
                afterSpace = true;
            }
            else
            {
                if (afterSpace)
                    word.clear();
                afterSpace = false;
                word += c;
            }
            ++k;
        }
        if (n == 0) throw std::runtime_error("constant/polyMesh/boundary lists no patch");
        if (kinds.count("processor"))
            throw std::runtime_error("processor patches: a decomposed mesh needs the multi-GPU layer (brae-mgpu)");
        std::string list;
        for (const auto& e : kinds) list += (list.empty() ? "" : ", ") + std::to_string(e.second) + " " + e.first;
        return std::to_string(n) + " patches (" + list + ")";
    });

    // the turbulence model, through readTurbulenceModel, as the solver's main calls it
    std::string second = "epsilon";
    if (legacyFamily)
    {
        checkStep(r, "turbulence", [&]() {
            const FoamDict tp = readDict(caseDir + "/constant/turbulenceProperties");
            const std::string simType = tp.wordOr("simulationType", "laminar");
            const bool allowed = (app == "pimpleFoam") ? (simType == "RAS" || simType == "LES" || simType == "laminar")
                                                       : (simType == "RAS" || simType == "laminar");
            if (!allowed) throw std::runtime_error("simulationType '" + simType + "' is not one " + app + " runs");
            DeviceSimpleControls checkCtl;
            checkCtl.caseDir = caseDir;
            checkCtl.turbulent = (simType == "RAS" || simType == "LES");
            readTurbulenceModel(tp, checkCtl, {app.c_str(), app == "rhoSimpleFoam", true});
            if (checkCtl.sst) second = "omega";
            return simType + (checkCtl.modelName.empty() ? std::string() : " " + checkCtl.modelName);
        });
    }

    // fvSchemes and fvSolution, through parseFvSchemesControls, pimpleFoam's ddt reader and readLinearSolverControls
    if (legacyFamily)
    {
        checkStep(r, "fvSchemes", [&]() {
            DeviceSimpleControls checkCtl;
            checkCtl.caseDir = caseDir;
            parseFvSchemesControls(caseDir, checkCtl);
            if (app == "pimpleFoam")
            {
                scalar oc = 1.0;
                parsePimpleDdtScheme(caseDir + "/system/fvSchemes", oc);
            }
            return std::string("every scheme is one brae implements");
        });
        checkStep(r, "fvSolution", [&]() {
            const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
            DeviceSimpleControls checkCtl;
            checkCtl.caseDir = caseDir;
            const std::string algo = (app == "pimpleFoam") ? "PIMPLE" : "SIMPLE";
            readLinearSolverControls(fvSolution, second, checkCtl, algo);
            if (app == "pimpleFoam")
            {
                // pimpleFoam's own reads of the block come first, or its audit names the controls the run takes as unread. A
                // dynamicMeshDict only moves correctPhi's default, which is read either way.
                readPimpleAlgorithm(fvSolution, std::filesystem::exists(caseDir + "/constant/dynamicMeshDict"));
                preflightAlgorithmControls(fvSolution, "PIMPLE");
            }
            return std::string("every solver entry and the ") + algo + " controls are read";
        });
    }
    else if (solver)
    {
        r.add("fvSchemes", "notice", app + " reads its fvSchemes and fvSolution with its own reader: `brae check --setup` runs it");
    }

    // MRF, through the reader simpleFoam calls (mrf_read.cuh): one active zone runs; a second is refused, by the solver and so
    // here. pimpleFoam applies none, and refuses an active zone at its setup.
    if ((app == "simpleFoam" || app == "pimpleFoam") && std::filesystem::exists(caseDir + "/constant/MRFProperties"))
    {
        checkStep(r, "MRF", [&]() {
            const MRFConfig mrf = readMRFProperties(caseDir + "/constant");
            if (mrf.active && app == "pimpleFoam")
                throw std::runtime_error("constant/MRFProperties has an active zone, and pimpleFoam does not apply MRF");
            return mrf.active ? std::string("one rotating zone, cellZone ") + mrf.cellZone : std::string("no active zone");
        });
    }

    // the mesh's motion, through the reader pimpleFoam calls (solid_body_motion.cuh). check passed occDrivAerRotMesh's
    // dynamicMultiMotionSolverFvMesh, which the solver then refused at its setup: the two disagreed (2026-11-10).
    if (app == "pimpleFoam" && std::filesystem::exists(caseDir + "/constant/dynamicMeshDict"))
    {
        checkStep(r, "motion", [&]() {
            const MeshMotion mm = readMeshMotion(caseDir);
            return mm.active ? std::string("solid-body motion of cellZone ") + mm.cellZone
                             : std::string("constant/dynamicMeshDict moves nothing this reader names; --setup reads the "
                                           "solver's other motion readers");
        });
    }

    // every field's boundary conditions at the start time, through the solvers' own field reader and the factory's list
    checkStep(r, "boundary", [&]() {
        const std::string start = resolveStartTime(caseDir, controlDict).name;
        const std::string dir = caseDir + "/" + start;
        if (!std::filesystem::is_directory(dir)) throw std::runtime_error("no start directory " + dir);
        int fields = 0;
        int entries = 0;
        int unusedEntries = 0;
        std::vector<std::string> refused;
        for (const auto& e : std::filesystem::directory_iterator(dir))
        {
            if (!e.is_regular_file()) continue;
            const std::string path = e.path().string();
            const std::string cls = fieldClass(path);
            std::vector<std::pair<std::string, std::string>> types;
            const std::string field = e.path().filename().string();
            // ONLY the entries a mesh patch resolves to, by the solver's own rule (findPatchEntry: name, group, regex): an
            // entry no patch takes is never built, by OpenFOAM or by brae. occDrivAerStaticMesh's fields #includeEtc
            // caseDicts/setConstraintTypes, whose cyclicSlip, processor, overset ... entries check refused on a mesh that has
            // none of those patches. A patch no entry covers stops both codes.
            auto resolve = [&](const auto& boundary) {
                if (meshPatches.empty())
                {
                    for (const auto& b : boundary) types.emplace_back(b.name, b.type);
                    return;
                }
                std::set<const void*> taken;
                for (const CheckPatch& mp : meshPatches)
                {
                    const auto* b = findPatchEntry(boundary, mp);
                    if (!b)
                    {
                        refused.push_back(field + " " + mp.name + ": no boundaryField entry covers this patch");
                        continue;
                    }
                    if (taken.insert(static_cast<const void*>(b)).second) types.emplace_back(b->name, b->type);
                }
                unusedEntries += static_cast<int>(boundary.size()) - static_cast<int>(taken.size());
            };
            if (cls == "volVectorField")
                resolve(readField<vector>(path).boundary);
            else if (cls == "volScalarField")
                resolve(readField<scalar>(path).boundary);
            else
                continue;
            ++fields;
            for (const auto& t : types)
            {
                ++entries;
                const bool known = isPatchFieldTypeName(t.second);
                const bool always = listHas(patchFieldTypesAlwaysRefused(), t.second);
                const bool frozen = (app == "simpleFoam") && (t.second == "fixedMean" || t.second == "fanPressure");
                if (!known || always || frozen)
                    refused.push_back(field + " " + t.first + ": '" + t.second + "'" +
                                      (!known ? " is not a boundary condition brae implements"
                                              : always ? " is named but not implemented"
                                                       : " is built once and never updated by simpleFoam"));
            }
        }
        if (!refused.empty())
        {
            std::string all;
            for (const std::string& s : refused) all += "\n              " + s;
            throw std::runtime_error(std::to_string(refused.size()) + " of " + std::to_string(entries) +
                                     " patch entries refused:" + all);
        }
        return std::to_string(fields) + " fields at time " + start + ", " + std::to_string(entries) +
               " patch entries, every type one the factory implements" +
               (unusedEntries ? " (" + std::to_string(unusedEntries) + " entries no patch takes, not built)" : std::string());
    });

    // the GPU this machine has (the check itself needs none)
    {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            r.add("GPU", "notice", "no CUDA GPU visible here: brae check needs none, brae run needs one");
        else
        {
            cudaDeviceProp p{};
            cudaGetDeviceProperties(&p, 0);
            r.add("GPU", "ok", std::to_string(n) + " x " + p.name + ", sm_" + std::to_string(p.major) + std::to_string(p.minor) + ", " +
                                   std::to_string(p.totalGlobalMem >> 30) + " GiB");
        }
    }

    for (const CheckReport::Line& l : r.lines)
        std::printf("  [%-7s] %-11s %s\n", l.verdict.c_str(), l.topic.c_str(), l.text.c_str());
    int code = 0;
    if (r.any("refused"))
    {
        std::printf("verdict: REFUSED -- brae would stop on this case; nothing was run\n");
        code = 2;
    }
    else if (r.any("notice"))
    {
        std::printf("verdict: SUPPORTED WITH NOTICES -- brae runs the case and says what it runs differently\n");
        code = 1;
    }
    else
        std::printf("verdict: SUPPORTED\n");
    if (setup && code < 2 && solver)
    {
        // Only a solver path that stops before its first step under BRAE_CHECK_SETUP (check_setup.cuh) is handed the case: any
        // other would ignore the variable and RUN it. simpleFoamV2 and rhoSimpleFoam's mirror are such paths, and interFoam.
        static const std::vector<std::string> stops = {"simpleFoam", "pimpleFoam", "rhoSimpleFoam"};
        const char* v2 = std::getenv("BRAE_SIMPLEFOAM_V2");
        const char* mirror = std::getenv("BRAE_RHOSIMPLEFOAM_MIRROR");
        if (!listHas(stops, app) || (app == "simpleFoam" && v2 && *v2 && std::string(v2) != "0")
            || (app == "rhoSimpleFoam" && mirror && *mirror))
        {
            std::printf("--setup: not run: %s on this path does not stop before its first step yet, so setting it up would "
                        "run it\n", app.c_str());
            return code;
        }
        // the solver's own setup, to the first step: every conditional refusal and the device memory
        std::fflush(stdout);
        setenv("BRAE_CHECK_SETUP", "1", 1);
        std::vector<std::string> args = {"-case", caseDir};
        std::istringstream la(solver->launchArgs);
        std::string w;
        while (la >> w) args.push_back(w);
        if (selfExecutableName() == solver->exe) return -1;   // this binary sets the case up: main carries on
        execSibling(solver->exe, args, "check --setup -> " + std::string(solver->exe), std::string("cases using ") + app);
    }
    return code;
}

}  // namespace cli
}  // namespace brae
