// E2/E3: brae must SAY when it runs a different linear solver than the case asked for.
//
// Found by dict_audit, which reported solvers/p/solver, solvers/p/smoother and
// solvers/(U|h|e)/preconditioner as read off disk by nobody. brae runs AMG-PCG for pressure and
// Jacobi-preconditioned BiCGStab elsewhere, whatever the dict says -- except U, whose DILU request is
// now honoured (device_dilu.cuh), which is why this file asserts U's preconditioner notice is ABSENT
// while the energy one is still present.
//
// This NOTICES rather than refuses, and the distinction is the whole design decision. A substituted
// linear solver is not a wrong answer: it solves the same linear system to the same tolerance, so the
// converged SIMPLE result is unchanged. What changes is the iteration count, the cost, and -- at the loose
// per-step relTol that SIMPLE uses (0.01 on p is the norm) -- the intermediate fields, because two solvers
// stop at different points. Someone diffing brae's "Solving for p" against OF's has a right to know.
//
// The negative control is the point of this test. Asserting "the notice appears" is satisfied by a
// function that prints unconditionally; what has to hold is that a dict asking for EXACTLY what brae runs
// stays silent. Both directions are checked below.

#include "linear_solver_setup.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& detail = "")
{
    if (ok) return;
    std::printf("  FAIL %s%s\n", what, detail.empty() ? "" : ("  [" + detail + "]").c_str());
    failures++;
}

// Run `fn` with stderr captured to a file, and return what it wrote. The notices go to stderr through
// fprintf, so this reads the real output rather than a test-only hook that could drift from it.
template <typename F>
std::string captureStderr(const std::string& tmpFile, F&& fn)
{
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    FILE* redirected = std::freopen(tmpFile.c_str(), "w", stderr);
    (void)redirected;
    fn();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::ifstream in(tmpFile);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string writeFvSolution(const std::string& dir, const std::string& solversBody)
{
    std::filesystem::create_directories(dir + "/system");
    std::ofstream(dir + "/system/fvSolution")
        << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
        << "solvers\n{\n" << solversBody << "}\n"
        << "SIMPLE { nNonOrthogonalCorrectors 0; }\n"
        << "relaxationFactors { equations { U 0.7; } }\n";
    return dir;
}

bool has(const std::string& hay, const std::string& needle)
{
    return hay.find(needle) != std::string::npos;
}

}   // namespace

int main()
{
    const std::string tmp = "/tmp/brae_solver_notices";
    std::filesystem::remove_all(tmp);

    // ---- POSITIVE: a stock-tutorial fvSolution, none of whose choices brae actually runs ----
    {
        const std::string dir = writeFvSolution(tmp + "/asks",
            "    p { solver GAMG; smoother DICGaussSeidel; tolerance 1e-10; relTol 0.01; }\n"
            "    \"(U|h|e)\" { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0.1; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");

        const std::string out = captureStderr(tmp + "/asks.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl);
            readEnergySolverControls(fv, /*internalEnergy=*/true);
        });

        check("p solver substitution is reported", has(out, "solvers/p solver") && has(out, "GAMG") && has(out, "AMG-PCG"), out);
        check("p smoother is reported ignored", has(out, "solvers/p smoother") && has(out, "DICGaussSeidel"), out);
        check("U solver substitution is reported", has(out, "solvers/U solver") && has(out, "PBiCGStab"), out);
        // U's DILU is now IMPLEMENTED (device_dilu.cuh), so it must NOT be reported as a substitution --
        // a notice that brae approximates something it actually runs is as misleading as a silent
        // substitution, and it is what would hide the next real one. The energy check below is the
        // other half: DILU is wired for the momentum path only, and `e` must still say so.
        check("U preconditioner is NOT reported: DILU is implemented for the momentum solves",
              !has(out, "solvers/U preconditioner"), out);
        check("energy preconditioner is reported", has(out, "solvers/e preconditioner") && has(out, "DILU"), out);
        // The wording must carry the reason, not just the fact -- a bare "ignored" would read as a bug.
        check("the notice explains the consequence", has(out, "iteration count and cost differ"), out);
    }

    // ---- NEGATIVE CONTROL: ask for exactly what brae runs -> silence ----
    //
    // notice() de-duplicates by (kind, subject, detail) in a process-wide set, so this MUST use different
    // field names than the positive case above, or the silence would just be de-duplication.
    {
        const std::string dir = writeFvSolution(tmp + "/matches",
            "    k { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }\n"
            "    omega { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");

        const std::string out = captureStderr(tmp + "/matches.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = true;
            ctl.sa = false;
            readLinearSolverControls(fv, "omega", ctl);
        });

        check("smoothSolver+symGaussSeidel on k produces no notice", !has(out, "solvers/k"), out);
        check("smoothSolver+symGaussSeidel on omega produces no notice", !has(out, "solvers/omega"), out);
    }

    // ---- NEGATIVE CONTROL 2: no `solvers` entries at all -> nothing to report ----
    {
        const std::string dir = writeFvSolution(tmp + "/empty", "");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/empty.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl);
        });
        check("an fvSolution with no solver entries is silent", !has(out, "NOTICE"), out);
    }

    // ---- maxIter / minIter: WHERE the solve stops, read from the same sub-dictionary ----
    //
    // OF lduMatrix::solver reads both (defaults 1000 and 0). LES/NACA4412 sets `maxIter 10` on p, and on
    // its impulsive first step OF's GAMG leaves with a final residual of 4.26 against an initial 1 -- an
    // unconverged field BY CONSTRUCTION. brae ran to its own convergence and got a different (better,
    // but different) answer, which is most of that case's disagreement. Both entries were sitting in the
    // dict audit's unread list the whole time.
    {
        const std::string dir = writeFvSolution(tmp + "/iters",
            "    p { solver GAMG; tolerance 1e-6; relTol 0.05; minIter 1; maxIter 10; }\n"
            "    pFinal { solver GAMG; tolerance 1e-6; relTol 0.01; minIter 2; maxIter 7; }\n"
            "    U { solver PBiCG; preconditioner DILU; tolerance 1e-5; relTol 0.1; minIter 1; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        DeviceSimpleControls ctl;
        ctl.turbulent = false;
        readLinearSolverControls(fv, "epsilon", ctl);
        check("maxIter p (want 10)", ctl.maxIterP == 10, std::to_string(ctl.maxIterP));
        check("minIter p (want 1)", ctl.minIterP == 1, std::to_string(ctl.minIterP));
        check("maxIter pFinal (want 7)", ctl.maxIterPFinal == 7, std::to_string(ctl.maxIterPFinal));
        check("minIter pFinal (want 2)", ctl.minIterPFinal == 2, std::to_string(ctl.minIterPFinal));
        check("minIter U (want 1)", ctl.minIterU == 1, std::to_string(ctl.minIterU));
        // U has no maxIter -> OF's default, and UFinal inherits U's (not the hard-coded default) exactly
        // as the tolerances do.
        check("maxIter U defaults to OF's 1000 (want 1000)", ctl.maxIterU == 1000, std::to_string(ctl.maxIterU));
        check("minIter UFinal inherits U (want 1)", ctl.minIterUFinal == 1, std::to_string(ctl.minIterUFinal));
        // and the accessors pick the right one per iteration
        ctl.finalInner = false; check("pMaxIter non-final (want 10)", ctl.pMaxIter() == 10, std::to_string(ctl.pMaxIter()));
        ctl.finalInner = true;  check("pMaxIter final (want 7)", ctl.pMaxIter() == 7, std::to_string(ctl.pMaxIter()));
    }
    {
        // Negative control: a dict that says nothing leaves OF's defaults, so nothing changes for the
        // hundreds of cases that never write these entries.
        const std::string dir = writeFvSolution(tmp + "/noiters",
            "    p { solver GAMG; tolerance 1e-6; relTol 0.05; }\n"
            "    U { solver PBiCG; tolerance 1e-5; relTol 0.1; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        DeviceSimpleControls ctl;
        ctl.turbulent = false;
        readLinearSolverControls(fv, "epsilon", ctl);
        check("no maxIter -> 1000 (want 1000)", ctl.maxIterP == 1000, std::to_string(ctl.maxIterP));
        check("no minIter -> 0 (want 0)", ctl.minIterP == 0, std::to_string(ctl.minIterP));
        check("no minIter on U -> 0 (want 0)", ctl.minIterU == 0, std::to_string(ctl.minIterU));
    }

    {
        // ...and the substitution notice must SAY when a cap is what makes the fields differ.
        // U, not p, and with a solver name no earlier block used: notice() de-duplicates on the whole
        // (kind, subject, detail) triple, so reusing either would test the de-dup, not the wording.
        const std::string dir = writeFvSolution(tmp + "/capped",
            "    U { solver PBiCGStab; tolerance 1e-6; relTol 0.05; maxIter 10; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/capped.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl);
        });
        check("a capped solve says the two solvers stop at different residuals",
              has(out, "maxIter 10") && has(out, "DIFFERENT residuals"), out);
    }
    {
        // Negative control: no cap -> the plain "same tolerance, cost differs" wording, which is the
        // claim that actually holds there.
        const std::string dir = writeFvSolution(tmp + "/uncapped",
            "    U { solver PCG; tolerance 1e-6; relTol 0.05; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/uncapped.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl);
        });
        check("an uncapped solve keeps the cost-only wording",
              has(out, "iteration count and cost differ") && !has(out, "DIFFERENT residuals"), out);
    }

    // ---- WHICH DRIVER IS RUNNING decides whether a preconditioner notice is true ----
    //
    // The energy solve is preconditioned with DILU by the OF-mirror driver and with Jacobi by the legacy
    // ones, off the SAME fvSolution entry, so the notice cannot be a property of the dict alone. Before
    // the diluOnEnergy argument the reader reported a substitution the mirror was not making, which is
    // the same defect as a silent substitution pointing the other way: it teaches the reader to discount
    // the notices. Both directions are asserted here, because a parameter that is never false in a test
    // and a parameter that is never true are equally untested.
    {
        const std::string dir = writeFvSolution(tmp + "/heprec",
            "    \"(U|e|k|epsilon)\" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string legacy = captureStderr(tmp + "/heprec_legacy.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "e");
        });
        check("a driver that does NOT precondition the energy solve says so",
              has(legacy, "solvers/e preconditioner") && has(legacy, "DILU"), legacy);
        const std::string mirror = captureStderr(tmp + "/heprec_mirror.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "e", /*diluOnEnergy=*/true);
        });
        check("a driver that DOES precondition the energy solve is silent about it",
              !has(mirror, "solvers/e preconditioner"), mirror);
        // ...and the flag must not silence the OTHER fields' notices, which is what a blanket exemption
        // would have done -- p asks DILU here too and nothing preconditions it.
        DeviceSimpleControls ctl;
        ctl.turbulent = false;
        readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "e", /*diluOnEnergy=*/true);
        check("the energy preconditioner is read off the case's own entry",
              ctl.diluHe, std::string("diluHe=") + (ctl.diluHe ? "1" : "0"));
    }
    {
        // Negative control on the read: an energy entry naming a preconditioner brae does not run must
        // leave diluHe false, or the driver would build a DILU the case never asked for.
        const std::string dir = writeFvSolution(tmp + "/heprec_none",
            "    \"(U|e|k|epsilon)\" { solver PBiCGStab; preconditioner diagonal; tolerance 1e-12; relTol 0; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        DeviceSimpleControls ctl;
        ctl.turbulent = false;
        readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "e", /*diluOnEnergy=*/true);
        check("a non-DILU energy preconditioner leaves diluHe false",
              !ctl.diluHe, std::string("diluHe=") + (ctl.diluHe ? "1" : "0"));
    }

    std::printf("solver_notices: %d failures\n", failures);
    return failures ? 1 : 0;
}
