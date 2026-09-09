#pragma once
#include "io_precision.cuh"   // setIOPrecision: OF ties every Info line to writePrecision
#include "brae_notice.cuh"
// Which brae solver owns a case -- the one place that maps an OpenFOAM case to a brae executable.
//
// `brae` is the single command a user types; the case itself names the solver it wants, in controlDict's
// `application` entry, exactly as OpenFOAM does. This header holds that registry: one row per solver brae
// implements. Adding a solver (rhoSimpleFoam, potentialFoam, ...) is one row plus its executable -- no new
// branching in the drivers.
//
// Selection, in order:
//   1. controlDict `application` names a registered solver -> that solver runs it (in-process if it IS this
//      executable, otherwise exec the sibling binary with the same argv).
//   2. controlDict `application` names something brae does not implement -> stop and say so, listing what brae
//      does have. brae never guesses a solver: a wrong one is a silently wrong answer.
//   3. controlDict has no `application` (hand-written cases) -> fall back to the case's fvSchemes ddtSchemes,
//      which still says steady (steadyState) or transient, and pick the registered solver of that kind.
// A case whose `application` disagrees with its ddtSchemes (application simpleFoam + Euler, say) is a case
// error, not something to paper over, so that stops too.
#include "foam_dict.cuh"   // readDict, readFileExpanded
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>   // getenv/setenv: the one-hand-over-per-run guard
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>   // execv/execvp: hand the case to the solver that owns it

namespace brae {

struct BraeSolver
{
    const char* application;   // OF controlDict `application` this row claims
    const char* exe;           // brae executable that runs it
    bool        transient;     // marches in time (ddtSchemes.default != steadyState)
    const char* what;          // one-line description, shown when a case asks for a solver brae lacks
};

// The registry. Every row has the same shape, including the steady one: `brae` is the launcher AND the steady
// solver's executable, which is a fact about today's build, not a rule the dispatcher should know. It finds out
// by comparing the chosen row's executable with the binary that is actually running (selfExecutableName below)
// -- so the steady case costs no extra process, and the day the steady solver moves into its own binary, only
// this table changes.
inline const std::vector<BraeSolver>& braeSolvers()
{
    static const std::vector<BraeSolver> reg = {
        {"simpleFoam", "brae",            false, "steady incompressible, RAS/laminar"},
        {"pimpleFoam", "brae_pimpleFoam", true,  "transient incompressible, URANS/DES/LES/laminar"},
        {"rhoSimpleFoam", "brae_rhoSimpleFoam", false, "steady compressible (subsonic, laminar, perfectGas+hConst)"},
    };
    return reg;
}

// Filename of the running binary, "" if it cannot be determined (then nothing matches and the chosen solver is
// exec'd, which is the safe direction -- the loop guard in execSolver catches a misconfiguration).
inline std::string selfExecutableName()
{
    std::error_code ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : self.filename().string();
}

// ddtSchemes.default as written in system/fvSchemes, "" if the case has no readable ddtSchemes block.
// fvSchemes is not a plain key/value dict (multi-token values, $-vars), so scan the text, bounded by the
// block's own braces -- an empty ddtSchemes must not pick up the next block's `default`.
inline std::string readDdtSchemeWord(const std::string& fvSchemesPath)
{
    std::string text;
    try { text = readFileExpanded(fvSchemesPath); } catch (...) { return ""; }
    const std::size_t blk = text.find("ddtSchemes");
    if (blk == std::string::npos) return "";
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return "";
    const std::size_t close = text.find('}', open);
    std::size_t d = text.find("default", open);
    if (d == std::string::npos || (close != std::string::npos && d > close)) return "";
    d += 7;
    while (d < text.size() && std::isspace(static_cast<unsigned char>(text[d]))) ++d;
    std::string w;
    while (d < text.size() && !std::isspace(static_cast<unsigned char>(text[d])) && text[d] != ';') w += text[d++];
    return w;
}

// The solvers brae has, formatted for an error message.
inline std::string braeSolverList()
{
    std::string s;
    for (const BraeSolver& e : braeSolvers())
    {
        s += "\n    ";
        s += e.application;
        s.append(std::max<std::size_t>(1, 16 - std::strlen(e.application)), ' ');
        s += e.what;
    }
    return s;
}

// Replace this process with a sibling brae executable, forwarding `args` as its argv[1..]. Prefers the binary
// next to this one, so a build tree runs its own components and an install runs the installed set; falls back to
// PATH. Never returns on success. This is the ONE exec path in brae -- solver hand-over and the `brae node`
// subcommand both come through here.
[[noreturn]] inline void execSibling(const std::string& exe, const std::vector<std::string>& args,
                                     const std::string& message, const std::string& target)
{
    // One hand-over per run. A second one means a renamed binary or a table pointing at the wrong executable, so
    // stop with something readable instead of forking forever.
    if (const char* from = std::getenv("BRAE_DISPATCHED_FROM"))
        throw std::runtime_error(
            std::string("dispatch loop: already handed over to '") + from + "' and now asked for '" + exe +
            "'. Check that " + exe + " is the binary that was meant.");
    setenv("BRAE_DISPATCHED_FROM", exe.c_str(), 1);

    std::string path = exe;
    std::error_code ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec)
    {
        const std::filesystem::path sibling = self.parent_path() / exe;
        if (std::filesystem::exists(sibling, ec)) path = sibling.string();
    }
    std::fprintf(stderr, "brae: %s\n", message.c_str());
    std::vector<char*> av;
    av.push_back(const_cast<char*>(path.c_str()));
    for (const std::string& a : args) av.push_back(const_cast<char*>(a.c_str()));
    av.push_back(nullptr);
    execv(path.c_str(), av.data());   // the sibling/installed path
    execvp(exe.c_str(), av.data());   // else whatever is on PATH
    throw std::runtime_error(
        "cannot start '" + exe + "', which brae needs for " + target +
        ". Build it with: cmake --build build --target " + exe);
}

// Hand the case to the solver that owns it, forwarding argv unchanged (every driver takes `-case <dir>` and a
// positional case dir).
[[noreturn]] inline void execSolver(const BraeSolver& s, int argc, char** argv, const std::string& why)
{
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    execSibling(s.exe, args,
                why + " -> " + s.application + " (" + s.exe + ")",
                std::string("cases using ") + s.application);
}

// Route the case to its solver. Returns only when the running binary IS the chosen solver; otherwise it has
// already exec'd the right one, or thrown because brae does not have it.
inline void dispatchSolver(const std::string& caseDir, int argc, char** argv)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    // OF: Time::readDict -> IOstream::defaultPrecision(writePrecision) -> Sout, which is what Info
    // writes through (TimeIO.C:375-383). Set here because every driver reaches the case through this
    // function, so the one entry cannot be honoured by one arm and dropped by another.
    setIOPrecision(controlDict.intOr("writePrecision", 6));
    const std::string application = controlDict.wordOr("application", "");
    const std::string ddt = readDdtSchemeWord(caseDir + "/system/fvSchemes");
    const bool transientCase = !ddt.empty() && ddt != "steadyState";

    const BraeSolver* chosen = nullptr;
    std::string why;
    if (!application.empty())
    {
        for (const BraeSolver& e : braeSolvers())
            if (application == e.application) chosen = &e;
        if (!chosen)
            throw std::runtime_error(
                "controlDict application '" + application + "' is not a solver brae implements yet."
                "\n  brae runs:" + braeSolverList() +
                "\n  brae stops rather than run a case with the wrong solver.");
        // The case names a solver but its ddtSchemes says the other thing -- one of the two is wrong, and
        // guessing which would silently produce the wrong answer.
        // A STEADY solver with a transient ddtSchemes entry is NOT an error in OpenFOAM. simpleFoam's
        // UEqn has no ddt term at all --
        //     fvm::div(phi, U) + MRF.DDt(U) + turbulence->divDevSigma(U) == fvOptions(U)
        // -- so the entry is never looked up and is simply inert. OF's OWN squareBend tutorial ships
        // `application simpleFoam` with `ddtSchemes { default Euler; }` and runs fine; brae refused it.
        // Report that the entry is ignored (the steady path forces steadyState regardless) rather than
        // reject a valid, shipped case.
        if (!ddt.empty() && chosen->transient != transientCase && !chosen->transient)
        {
            noticeIgnored("fvSchemes/ddtSchemes.default",
                          "'" + ddt + "' on the steady solver '" + application + "'. OpenFOAM's steady "
                          "solvers construct no ddt term, so this entry is inert there and is ignored "
                          "here too -- the run is steady.");
        }
        // The reverse IS an error: a transient solver with steadyState would silently drop the time
        // derivative, which is a different equation rather than an unused entry.
        else if (!ddt.empty() && chosen->transient != transientCase)
            throw std::runtime_error(
                "controlDict application '" + application + "' is transient, but fvSchemes "
                "ddtSchemes.default is '" + ddt + "'. A transient solver needs "
                "Euler|backward|CrankNicolson; steadyState would drop the time derivative entirely.");
        why = "controlDict application " + application;
    }
    else
    {
        // No `application` entry: the ddt scheme still says steady or transient, so pick that kind of solver.
        for (const BraeSolver& e : braeSolvers())
            if (e.transient == transientCase) { chosen = &e; break; }
        if (!chosen)
            throw std::runtime_error(
                "system/controlDict has no `application` entry and fvSchemes ddtSchemes.default '" + ddt +
                "' matches no brae solver."
                "\n  brae runs:" + braeSolverList());
        why = "no controlDict application, ddtSchemes.default = " + (ddt.empty() ? std::string("(none)") : ddt);
    }

    if (selfExecutableName() == chosen->exe) return;   // already the right binary: solve in this process
    execSolver(*chosen, argc, argv, why);              // does not return
}

}  // namespace brae
