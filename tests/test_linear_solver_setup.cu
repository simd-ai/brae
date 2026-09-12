// readLinearSolverControls must read the ALGORITHM dict by name, and must read the whole control set.
//
// Two separate claims, and the second is the one that needs a negative control.
//
// 1. Completeness. The compressible driver was ported by copying only the parts of gpuSimpleFoam it
//    needed, and left fifteen controls behind; each fell back to a struct default (relTol -> 0, gs* ->
//    false, consistent -> unread). A converged field on a case that does not use them looks fine, which
//    is why it survived. So assert every control lands, from a dict where each has a DISTINCT value.
//
// 2. Dict-name parameterisation. OF reads consistent/nNonOrthogonalCorrectors from
//    solutionDict().subOrEmptyDict(algorithmName_) -- solutionControl.C:46,51,302 -- where the name is a
//    ctor argument: "SIMPLE" in simpleControl.H:100, "PIMPLE" in pimpleControl.H:135. brae hardcoded
//    "SIMPLE", so a transient case's nNonOrthogonalCorrectors read as 0 and its `consistent` as false.
//    The NEGATIVE CONTROL is the point here: reading a PIMPLE-only fvSolution while asking for "SIMPLE"
//    must come back with the DEFAULTS. Without that check, deleting the algorithmDict parameter and
//    hardcoding "PIMPLE" would pass the positive test just as well -- and would then break every steady
//    case in exactly the way this fix was meant to repair.

#include "linear_solver_setup.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& got, const std::string& want)
{
    if (ok) return;
    std::printf("  FAIL %s: got %s, want %s\n", what, got.c_str(), want.c_str());
    failures++;
}

void eqScalar(const char* what, scalar got, scalar want)
{
    check(what, got == want, std::to_string((double)got), std::to_string((double)want));
}

void eqBool(const char* what, bool got, bool want)
{
    check(what, got == want, got ? "true" : "false", want ? "true" : "false");
}

void eqInt(const char* what, int got, int want)
{
    check(what, got == want, std::to_string(got), std::to_string(want));
}

// An fvSolution whose algorithm controls live in a dict named `algoName`. Every value is distinct so a
// control read from the wrong key cannot coincidentally match.
std::string writeCase(const std::string& dir, const std::string& algoName)
{
    std::filesystem::create_directories(dir + "/system");
    std::ofstream f(dir + "/system/fvSolution");
    f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
      << "solvers\n{\n"
      << "    p       { solver GAMG;         tolerance 1e-07; relTol 0.05; }\n"
      << "    U       { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-09; relTol 0.02; }\n"
      << "    k       { solver smoothSolver; smoother GaussSeidel;    tolerance 1e-11; relTol 0.03; }\n"
      << "    omega   { solver PBiCGStab;    tolerance 1e-10; relTol 0.04; }\n"
      << "}\n"
      << algoName << "\n{\n"
      << "    consistent yes;\n"
      << "    nNonOrthogonalCorrectors 3;\n"
      << "}\n";
    return dir;
}

}   // namespace

int main()
{
    const std::string base = "/tmp/brae_linear_solver_setup";
    std::filesystem::remove_all(base);

    // ---- 1. completeness, on a steady (SIMPLE) case with kOmegaSST ----
    {
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        ctl.sst = true;
        const FoamDict fvSolution = readDict(writeCase(base + "/simple", "SIMPLE") + "/system/fvSolution");
        readLinearSolverControls(fvSolution, "omega", ctl, "SIMPLE");

        eqScalar("tolP", ctl.tolP, 1e-07);
        eqScalar("tolU", ctl.tolU, 1e-09);
        eqScalar("relTolP", ctl.relTolP, 0.05);
        eqScalar("relTolU", ctl.relTolU, 0.02);
        // k/omega share one solve tolerance in brae: OF solves them separately, so take the TIGHTER.
        eqScalar("tolKE = min(tol k, tol omega)", ctl.tolKE, 1e-11);
        eqScalar("relTolKE = min(relTol k, relTol omega)", ctl.relTolKE, 0.03);
        // smoothSolver + a GaussSeidel-family smoother -> symGS; anything else stays BiCGStab.
        eqBool("gsU (smoothSolver/symGaussSeidel)", ctl.gsU, true);
        eqBool("gsK (smoothSolver/GaussSeidel)", ctl.gsK, true);
        eqBool("gsEps (PBiCGStab -> NOT symGS)", ctl.gsEps, false);
        eqBool("consistent", ctl.consistent, true);
        eqInt("nNonOrth", ctl.nNonOrth, 3);
        if (failures == 0) std::printf("  OK   SIMPLE case: all 11 controls read from fvSolution\n");
    }

    // ---- 2. the same file, but the algorithm dict is named PIMPLE ----
    const int beforePimple = failures;
    {
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        ctl.sst = true;
        const FoamDict fvSolution = readDict(writeCase(base + "/pimple", "PIMPLE") + "/system/fvSolution");
        readLinearSolverControls(fvSolution, "omega", ctl, "PIMPLE");

        eqBool("PIMPLE consistent", ctl.consistent, true);
        eqInt("PIMPLE nNonOrth", ctl.nNonOrth, 3);
        eqScalar("PIMPLE relTolU", ctl.relTolU, 0.02);   // solvers dict is algorithm-independent
        if (failures == beforePimple) std::printf("  OK   PIMPLE case: algorithm controls read from the PIMPLE dict\n");
    }

    // ---- 3. NEGATIVE CONTROL: wrong dict name must yield DEFAULTS, not the PIMPLE values ----
    // This is what fails if algorithmDict is dropped and either name is hardcoded again.
    {
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        ctl.sst = true;
        const FoamDict fvSolution = readDict(base + "/pimple/system/fvSolution");
        readLinearSolverControls(fvSolution, "omega", ctl, "SIMPLE");   // deliberately the wrong dict

        const bool defaulted = (ctl.consistent == false && ctl.nNonOrth == 0);
        if (defaulted)
        {
            std::printf("  OK   wrong algorithm dict -> defaults (so the name is load-bearing)\n");
        }
        else
        {
            std::printf("  FAIL asking for 'SIMPLE' found the PIMPLE dict's values "
                        "(consistent=%d nNonOrth=%d) -- the dict name is being ignored, so this test\n"
                        "       cannot detect a driver reading the wrong algorithm dict\n",
                        (int)ctl.consistent, ctl.nNonOrth);
            failures++;
        }
        // ...but the solvers dict is NOT algorithm-scoped, so it must still be read.
        eqScalar("solvers still read with the wrong algorithm dict", ctl.relTolU, 0.02);
    }

    // ---- 4. laminar: the turbulence solver dicts must be skipped, not read ----
    {
        DeviceSimpleControls ctl;   // turbulent = false
        const DeviceSimpleControls dflt;
        const FoamDict fvSolution = readDict(base + "/simple/system/fvSolution");
        readLinearSolverControls(fvSolution, "omega", ctl, "SIMPLE");
        eqScalar("laminar tolKE untouched", ctl.tolKE, dflt.tolKE);
        eqScalar("laminar relTolKE untouched", ctl.relTolKE, dflt.relTolKE);
        eqScalar("laminar tolP still read", ctl.tolP, 1e-07);
    }

    // ---- 4b. the ENERGY slot and the turbulence CAPS, from their own entries ----
    // The rho mirror drivers took the energy tolerance from tolKE -- the turbulence slot -- and every
    // equation's maxIter from p's; maxIterKE/minIterKE were never read at all. Distinct values per
    // entry are the test, and the two negatives (no heName -> untouched; laminar -> caps untouched)
    // are what stop a reader that fills every slot from one entry passing it.
    {
        const std::string d = base + "/energy";
        std::filesystem::create_directories(d + "/system");
        {
            std::ofstream f(d + "/system/fvSolution");
            f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
              << "solvers\n{\n"
              << "    p         { solver GAMG;      tolerance 1e-07; relTol 0.05; maxIter 13; minIter 1; }\n"
              << "    U         { solver PBiCGStab; tolerance 1e-09; relTol 0.02; maxIter 17; minIter 2; }\n"
              << "    \"(h|e)\"   { solver PBiCGStab; tolerance 1e-03; relTol 0.21; maxIter 7;  minIter 3; }\n"
              << "    k         { solver PBiCGStab; tolerance 1e-11; relTol 0.03; maxIter 9;  minIter 1; }\n"
              << "    epsilon   { solver PBiCGStab; tolerance 1e-10; relTol 0.04; maxIter 9;  minIter 1; }\n"
              << "}\n"
              << "SIMPLE { }\n";
        }
        const FoamDict fv = readDict(d + "/system/fvSolution");
        const DeviceSimpleControls dflt;

        DeviceSimpleControls ctl;   ctl.turbulent = true;
        readLinearSolverControls(fv, "epsilon", ctl, "SIMPLE", "e");
        eqScalar("tolHe from the (h|e) entry",     ctl.tolHe,     1e-03);
        eqScalar("relTolHe from the (h|e) entry",  ctl.relTolHe,  0.21);
        eqInt   ("maxIterHe from the (h|e) entry", ctl.maxIterHe, 7);
        eqInt   ("minIterHe from the (h|e) entry", ctl.minIterHe, 3);
        eqInt   ("maxIterP is p's own",            ctl.maxIterP,  13);
        eqInt   ("maxIterU is U's own",            ctl.maxIterU,  17);
        eqInt   ("minIterU is U's own",            ctl.minIterU,  2);
        eqInt   ("maxIterKE from the k/epsilon entries", ctl.maxIterKE, 9);
        eqInt   ("minIterKE from the k/epsilon entries", ctl.minIterKE, 1);
        eqScalar("tolKE = min(tol k, tol epsilon)", ctl.tolKE, 1e-11);

        DeviceSimpleControls noHe;  noHe.turbulent = true;
        readLinearSolverControls(fv, "epsilon", noHe, "SIMPLE");   // no energy field named
        eqScalar("no heName -> tolHe untouched",   noHe.tolHe,     dflt.tolHe);
        eqInt   ("no heName -> maxIterHe untouched", noHe.maxIterHe, dflt.maxIterHe);

        DeviceSimpleControls lam;   // turbulent = false
        readLinearSolverControls(fv, "epsilon", lam, "SIMPLE", "e");
        eqInt   ("laminar -> maxIterKE untouched", lam.maxIterKE, dflt.maxIterKE);
        eqScalar("laminar -> tolHe still read",    lam.tolHe,     1e-03);
        if (failures == 0) std::printf("  OK   energy slot and turbulence caps read from their own entries\n");
    }

    // ---- 5. relaxationFactors: modern nested form, per-FIELD keys ----
    // The transient driver took epsilon/omega's factor from "k", so distinct values here are the test.
    {
        const std::string d = base + "/relax";
        std::filesystem::create_directories(d + "/system");
        {
            std::ofstream f(d + "/system/fvSolution");
            f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
              << "solvers { }\n"
              << "relaxationFactors\n{\n"
              << "    fields    { p 0.31; }\n"
              << "    equations { U 0.72; k 0.61; omega 0.43; epsilon 0.55; nuTilda 0.66; }\n"
              << "}\n";
        }
        const FoamDict fv = readDict(d + "/system/fvSolution");

        DeviceSimpleControls sst;   sst.turbulent = true; sst.sst = true;
        readRelaxationFactors(fv, sst);
        eqScalar("relaxP", sst.relaxP, 0.31);
        eqScalar("relaxU", sst.relaxU, 0.72);
        eqScalar("relaxK", sst.relaxK, 0.61);
        eqScalar("SST relaxEps from 'omega' (NOT k)", sst.relaxEps, 0.43);

        DeviceSimpleControls ke;   ke.turbulent = true;   // kEpsilon: sst = false
        readRelaxationFactors(fv, ke);
        eqScalar("kEpsilon relaxEps from 'epsilon'", ke.relaxEps, 0.55);

        DeviceSimpleControls sa;   sa.turbulent = true; sa.sa = true;
        readRelaxationFactors(fv, sa);
        eqScalar("SA relaxK from 'nuTilda'", sa.relaxK, 0.66);
    }

    // ---- 6. relaxationFactors: LEGACY flat form ----
    {
        const std::string d = base + "/relaxflat";
        std::filesystem::create_directories(d + "/system");
        {
            std::ofstream f(d + "/system/fvSolution");
            f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
              << "solvers { }\n"
              << "relaxationFactors { p 0.3; U 0.7; k 0.6; epsilon 0.5; }\n";   // no equations{}/fields{}
        }
        DeviceSimpleControls ctl;   ctl.turbulent = true;
        readRelaxationFactors(readDict(d + "/system/fvSolution"), ctl);
        eqScalar("legacy flat relaxP", ctl.relaxP, 0.3);
        eqScalar("legacy flat relaxU", ctl.relaxU, 0.7);
        eqScalar("legacy flat relaxEps", ctl.relaxEps, 0.5);
    }

    // ---- 7. alpha <= 0 -> 1.0, matching OF's fvMatrix::relax ----
    // Without this a factor of 0 makes the diagonal-relaxation kernel divide by zero: Inf diag, NaN field.
    {
        const std::string d = base + "/relaxzero";
        std::filesystem::create_directories(d + "/system");
        {
            std::ofstream f(d + "/system/fvSolution");
            f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
              << "solvers { }\n"
              << "relaxationFactors { equations { U 0; k -0.5; } fields { p 0; } }\n";
        }
        DeviceSimpleControls ctl;   ctl.turbulent = true;
        readRelaxationFactors(readDict(d + "/system/fvSolution"), ctl);   // warns on stderr, by design
        eqScalar("relaxU 0 -> 1.0", ctl.relaxU, 1.0);
        eqScalar("relaxK -0.5 -> 1.0", ctl.relaxK, 1.0);
        eqScalar("relaxP 0 -> 1.0", ctl.relaxP, 1.0);
    }

    // ---- 8. THE `Final` VARIANTS, through OpenFOAM's own $macro idiom ----
    //
    // Two defects met here, and the test needs both halves because either one alone hides the other.
    //
    // (a) brae read only the base entries, so `pFinal` was inert: every PIMPLE corrector was solved to
    //     the loose `p` tolerance. OF gives the LAST pressure corrector its own settings
    //     (pEqn.solve(p.select(pimple.finalInnerIter()))), which is why its log shows one corrector
    //     stopping at 2.8e-04 and the next at 6e-11 on the same step.
    //
    // (b) brae's $macro expansion is textual, and p's captured body ends in its own ';' -- so `$p;`
    //     expanded to `... relTol 0.01 ; ;` and the parser read that second ';' as the KEY of the entry
    //     that followed, swallowing the `tolerance 1e-10` meant to override it. pFinal silently became p.
    //
    // THE FIXTURE IS THE CANONICAL OPENFOAM IDIOM, verbatim, because that is the only form in which (b)
    // bites -- writing pFinal out longhand passes even with the parser bug, which is exactly why this
    // went unnoticed. Leg (ii) is that negative control: the same numbers, written without the macro.
    {
        const std::string d = base + "/pfinal";
        std::filesystem::create_directories(d + "/system");
        {
            std::ofstream f(d + "/system/fvSolution");
            f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
              << "solvers\n{\n"
              << "    p      { solver GAMG; tolerance 1e-5; relTol 0.01; }\n"
              << "    pFinal { $p; tolerance 1e-10; relTol 0; }\n"
              << "    U      { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6; relTol 0.1; }\n"
              << "    UFinal { $U; relTol 0; }\n"
              << "}\n"
              << "PIMPLE { nOuterCorrectors 1; nCorrectors 2; }\n";
        }
        DeviceSimpleControls ctl;
        readLinearSolverControls(readDict(d + "/system/fvSolution"), "epsilon", ctl, "PIMPLE");
        eqScalar("base p tolerance untouched", ctl.tolP, 1e-5);
        eqScalar("base p relTol untouched",    ctl.relTolP, 0.01);
        eqScalar("pFinal tolerance overrides the $p macro", ctl.tolPFinal, 1e-10);
        eqScalar("pFinal relTol overrides the $p macro",    ctl.relTolPFinal, 0.0);
        eqScalar("UFinal inherits tolerance through $U",    ctl.tolUFinal, 1e-6);
        eqScalar("UFinal relTol overrides",                 ctl.relTolUFinal, 0.0);
        // VACUITY GUARD: if Final and base agreed, every assertion above would pass under both defects.
        if (ctl.tolPFinal == ctl.tolP || ctl.relTolUFinal == ctl.relTolU)
        {
            std::printf("  FAIL vacuous: the Final entries do not differ from the base ones, so this leg\n"
                        "       cannot tell a working override from an ignored one\n");
            failures++;
        }
    }
    {
        // (ii) NEGATIVE CONTROL for the macro: the identical settings written longhand must give the
        // identical controls. If these two legs ever disagree, the expansion is the thing that broke.
        const std::string d = base + "/pfinal_nomacro";
        std::filesystem::create_directories(d + "/system");
        {
            std::ofstream f(d + "/system/fvSolution");
            f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
              << "solvers\n{\n"
              << "    p      { solver GAMG; tolerance 1e-5; relTol 0.01; }\n"
              << "    pFinal { solver GAMG; tolerance 1e-10; relTol 0; }\n"
              << "}\n"
              << "PIMPLE { nOuterCorrectors 1; nCorrectors 2; }\n";
        }
        DeviceSimpleControls ctl;
        readLinearSolverControls(readDict(d + "/system/fvSolution"), "epsilon", ctl, "PIMPLE");
        eqScalar("longhand pFinal tolerance", ctl.tolPFinal, 1e-10);
        eqScalar("longhand pFinal relTol",    ctl.relTolPFinal, 0.0);
    }
    {
        // (iii) NO Final entry -> fall back to the base one. OF would FatalError; refusing a case OF runs
        // is worse than solving the last corrector exactly as tightly as the others.
        const std::string d = base + "/nofinal";
        std::filesystem::create_directories(d + "/system");
        {
            std::ofstream f(d + "/system/fvSolution");
            f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
              << "solvers { p { solver GAMG; tolerance 3e-7; relTol 0.02; } }\n"
              << "PIMPLE { nCorrectors 2; }\n";
        }
        DeviceSimpleControls ctl;
        readLinearSolverControls(readDict(d + "/system/fvSolution"), "epsilon", ctl, "PIMPLE");
        eqScalar("absent pFinal falls back to p tolerance", ctl.tolPFinal, 3e-7);
        eqScalar("absent pFinal falls back to p relTol",    ctl.relTolPFinal, 0.02);
    }

    std::printf("linear_solver_setup: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
