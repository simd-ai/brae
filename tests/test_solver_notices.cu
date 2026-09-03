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

        // brae's AMG-PCG is OF's PCG with a GAMG preconditioner, so the notice names both -- `GAMG` as
        // what the case asked for and `PCG preconditioned with GAMG` as what runs. OF's GAMG is
        // multigrid AS the solver, which brae does not run, so this substitution is real.
        check("p solver substitution is reported",
              has(out, "solvers/p solver") && has(out, "GAMG") && has(out, "PCG preconditioned with GAMG"), out);
        check("p smoother is reported ignored", has(out, "solvers/p smoother") && has(out, "DICGaussSeidel"), out);
        // NOTHING about U. This block used to assert the OPPOSITE -- that asking `PBiCGStab` on U was
        // reported as a substitution -- which was the defect, not the contract: PBiCGStab IS a
        // preconditioned BiCGStab with a run-time selectable preconditioner (PBiCGStab.H), so brae
        // running BiCGStab with the DILU this entry names is running exactly what the case asked for.
        // The old assertion made every stock tutorial announce a substitution on U that was not
        // happening, and a notice that brae approximates something it actually runs hides the next real
        // one just as well as silence does (queue item 28).
        check("U solver is NOT reported: PBiCGStab with the case's DILU is what brae runs",
              !has(out, "solvers/U solver"), out);
        check("U preconditioner is NOT reported: DILU is implemented for the momentum solves",
              !has(out, "solvers/U preconditioner"), out);
        // The energy half: this driver does NOT wire DILU there, so the preconditioner IS substituted and
        // must say so -- while the SOLVER is the one the case named and must not.
        check("energy preconditioner is reported", has(out, "solvers/e preconditioner") && has(out, "DILU"), out);
        check("energy solver is NOT reported", !has(out, "solvers/e solver"), out);
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
        // PBiCG, not PBiCGStab: the stabilized one is what brae runs, so naming it here would produce no
        // notice at all and this block would assert nothing (queue item 28). PBiCG is a different Krylov
        // method (PBiCG.H: bi-conjugate gradient, not the stabilized variant), so it is a real one.
        const std::string dir = writeFvSolution(tmp + "/capped",
            "    U { solver PBiCG; tolerance 1e-6; relTol 0.05; maxIter 10; }\n");
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
            SolverRunsAs mirrorRuns;
            mirrorRuns.diluOnEnergy = true;
            readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "e", mirrorRuns);
        });
        check("a driver that DOES precondition the energy solve is silent about it",
              !has(mirror, "solvers/e preconditioner"), mirror);
        // ...and the flag must not silence the OTHER fields' notices, which is what a blanket exemption
        // would have done -- p asks DILU here too and nothing preconditions it.
        DeviceSimpleControls ctl;
        ctl.turbulent = false;
        SolverRunsAs mirrorRuns;
        mirrorRuns.diluOnEnergy = true;
        readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "e", mirrorRuns);
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
        SolverRunsAs mirrorRuns;
        mirrorRuns.diluOnEnergy = true;
        readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "e", mirrorRuns);
        check("a non-DILU energy preconditioner leaves diluHe false",
              !ctl.diluHe, std::string("diluHe=") + (ctl.diluHe ? "1" : "0"));
    }

    // ---- ITEM 28: A CASE THAT NAMES WHAT BRAE RUNS MUST BE MET WITH SILENCE ----
    //
    // The headline of the fix. Every field here names the solver AND the preconditioner brae actually
    // runs, in OpenFOAM's own vocabulary, so there is nothing to report. Before this the same dictionary
    // produced a solver notice on all four fields, because the comparison held the dict's `PBiCGStab`
    // against a display string of "Jacobi-BiCGStab" and the dict's `PCG` against "AMG-PCG".
    //
    // The field names and values differ from every block above: notice() de-duplicates on the whole
    // (kind, subject, detail) triple, and a silence assertion is the one kind that a de-dup could pass
    // for the wrong reason.
    {
        const std::string dir = writeFvSolution(tmp + "/asksexactly",
            "    p { solver PCG; preconditioner GAMG; tolerance 1e-7; relTol 0.02; }\n"
            "    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.02; }\n"
            "    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.02; }\n"
            "    omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.02; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/asksexactly.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = true;
            ctl.sa = false;
            readLinearSolverControls(fv, "omega", ctl);
        });
        check("a case naming exactly what brae runs is met with silence", !has(out, "NOTICE"), out);
    }
    {
        // The control for that silence: the SAME dictionary with exactly one field changed to a solver
        // brae does not run. If the block above passed because the reader had gone quiet altogether,
        // this one fails.
        //
        // On `k`, and not on U as first written: notice() de-duplicates on the whole (kind, subject,
        // detail) triple, and an earlier block in this file already emits `solvers/U solver: case asks
        // 'PBiCG', brae runs PBiCGStab preconditioned with DILU`. The duplicate was suppressed and this
        // control read an empty capture -- a false FAIL, which is the safe direction, but the same trap
        // pointed the other way is what makes a silence assertion pass for the wrong reason.
        const std::string dir = writeFvSolution(tmp + "/asksexactly2",
            "    p { solver PCG; preconditioner GAMG; tolerance 1e-7; relTol 0.02; }\n"
            "    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.02; }\n"
            "    k { solver PBiCG; preconditioner DILU; tolerance 1e-7; relTol 0.02; }\n"
            "    omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.02; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/asksexactly2.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = true;
            ctl.sa = false;
            readLinearSolverControls(fv, "omega", ctl);
        });
        check("...and the same block with one solver changed is NOT silent",
              has(out, "solvers/k solver") && has(out, "PBiCG"), out);
        check("...and only that field is reported",
              !has(out, "solvers/U solver") && !has(out, "solvers/omega solver") && !has(out, "solvers/p solver"), out);
    }

    // ---- `none` IS NOT `diagonal`, and was exempted as though it were ----
    //
    // OF's noPreconditioner is wA = rA and its diagonalPreconditioner is wA = rA/diag (their .C files).
    // The old exemption list skipped both words, so a case asking for NO preconditioner was answered
    // with Jacobi and told nothing. Comparing against what the driver runs removes the list and the hole
    // with it.
    {
        const std::string dir = writeFvSolution(tmp + "/nonecon",
            "    U { solver PBiCGStab; preconditioner none; tolerance 1e-9; relTol 0.03; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/nonecon.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl);
        });
        check("asking for no preconditioner at all is reported",
              has(out, "solvers/U preconditioner") && has(out, "'none'") && has(out, "diagonal"), out);
    }
    {
        // Control: `diagonal` IS what brae runs there, so it stays silent.
        const std::string dir = writeFvSolution(tmp + "/diagcon",
            "    U { solver PBiCGStab; preconditioner diagonal; tolerance 1e-9; relTol 0.03; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/diagcon.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl);
        });
        check("asking for the diagonal preconditioner brae runs is silent",
              !has(out, "solvers/U preconditioner"), out);
    }

    // ---- THE SUBSTITUTION CAN POINT THE OTHER WAY ----
    //
    // The OF-mirror's HOST arm solves everything through pbicgstab.cuh, which is DILU whatever the dict
    // says. A case asking `diagonal` there is being substituted just as surely as one asking DILU on a
    // driver that runs Jacobi, and nothing reported it because the old rule only ever looked for the
    // word DILU in the DICT.
    {
        const std::string dir = writeFvSolution(tmp + "/alwaysdilu",
            "    U { solver PBiCGStab; preconditioner diagonal; tolerance 1e-11; relTol 0.04; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/alwaysdilu.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            SolverRunsAs hostArm;
            hostArm.alwaysDilu = true;
            hostArm.pSolver = "PBiCGStab";
            hostArm.pPrecon = "DILU";
            readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "", hostArm);
        });
        check("a driver that always runs DILU reports the case that asked for diagonal",
              has(out, "solvers/U preconditioner") && has(out, "brae preconditions with DILU"), out);
    }

    std::printf("solver_notices: %d failures\n", failures);
    return failures ? 1 : 0;
}
