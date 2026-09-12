#pragma once
// of_residual_log.cuh -- print one SIMPLE iteration's linear solves in OpenFOAM's own log format.
//
// WHY. brae prints a compact one-line summary every 50 iterations. That is fine for watching a run and
// useless for the thing this project actually does: comparing brae against OF iteration by iteration to
// find where an error is introduced. OF prints, per solve:
//
//     DILUPBiCGStab:  Solving for Ux, Initial residual = 1, Final residual = 0.0833, No Iterations 1
//     GAMG:  Solving for p, Initial residual = 1, Final residual = 0.00798, No Iterations 31
//     time step continuity errors : sum local = 0.0266, global = 0.0047, cumulative = 0.0047
//
// Every number in those lines already exists in brae (DeviceSimpleResidual carries the initial/final
// residual and iteration count for Ux/Uy/Uz and p; turbulenceReport() carries the same for he, k and
// epsilon|omega). Nothing new is computed here -- this only formats what the solve already produced, so
// enabling it cannot change a single result.
//
// The INITIAL residual is the one worth diffing. It is the residual of the previous iterate against the
// freshly assembled matrix, so it reports the state the equation was handed -- which is exactly what
// "where did the error come from" needs. A final residual mostly reports the linear solver's tolerance.
//
// Enabled by BRAE_OF_LOG=1, off by default: the existing compact output is what the gates grep, and a
// solver that changes its stdout because a debugging aid was added is its own kind of defect. Off costs
// one getenv per run.
//
// WHICH COLUMNS ARE ACTUALLY COMPARABLE. Measured on aerofoilNACA0012, a case whose converged fields
// agree with OF to 3.2e-04:
//
//     field   OF initial      brae initial
//     Ux      1               1              <- comparable
//     p       1               1              <- comparable
//     e       0.99999         1.44e-07       <- NOT comparable (7e6x)
//     k       1               9.92e-04       <- NOT comparable (1000x)
//
// The MOMENTUM and PRESSURE residuals normalise identically on both sides, so a difference there is real.
// The SCALAR equations (e, k, epsilon, omega) use a different normFactor, so a large ratio there means
// nothing on its own -- the case above is provably correct and shows 7e6x. Reading those columns as a
// defect is a trap this project has already fallen into once: an apparent "e 117x off, k 6.7x off" on
// angledDuct evaporated when this control was run.
//
// The CONTINUITY line is a physical quantity, not normalised, and is comparable -- subject to both sides
// having solved the pressure equation to a similar tolerance, which different linear solvers do not
// guarantee at a loose relTol.
//
// Iteration counts and FINAL residuals are never comparable: different solvers, different preconditioners.
//
// Solver names are brae's actual ones, not OF's. Printing "GAMG:" when brae ran AMG-PCG would make the
// two logs diff clean while hiding that a different solver produced the number -- the diff is supposed
// to show that.
#include "solver_controls.cuh"
#include "device_kepsilon.cuh"   // turbulenceReport() / ScalarSolveEntry
#include <cstdio>
#include <cstdlib>
#include <string>

namespace brae {

inline bool ofResidualLog()
{
    static const bool on = []() {
        const char* e = std::getenv("BRAE_OF_LOG");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}

// The solver name on the Ux/Uy/Uz lines: what RAN, set by the driver from the branch it wired. The
// header above says these names are brae's actual ones, and until this holder existed the three lines
// said "JacobiBiCGStab" whatever the momentum solve was -- DILU-preconditioned, OpenFOAM's own sweep,
// or the colour-order sweep (the default since 2026-09-08). Defaults to the old string, so a driver
// that never sets it -- the legacy drivers do not -- keeps printing "JacobiBiCGStab" whatever it ran,
// which is the known-wrong line this holder was added to fix on the mirror. OpenFOAM composes its own name as
// preconditioner + solver (PBiCGStab.C:80-83), so the DILU form is spelled `DILUPBiCGStab` to diff
// line for line against an OpenFOAM log; the others are brae's own names for brae's own solvers.
inline const char*& momentumSolverName()
{
    static const char* name = "JacobiBiCGStab";
    return name;
}

// One solve line in OpenFOAM's format (SolverPerformance::print, SolverPerformance.C:95-117:
// `<solver>:  Solving for <field>, Initial residual = a, Final residual = b, No Iterations n`).
// Exposed so a driver that can stand behind only some of the lines -- the rhoSimpleFoam mirror
// prints the momentum ones, whose numbers its step returns -- prints them in the same format.
inline void printOfSolveLine(
    const char* solver,
    const char* field,
    scalar init,
    scalar fin,
    int nIter)
{
    std::printf("%s:  Solving for %s, Initial residual = %g, Final residual = %g, No Iterations %d\n",
                solver, field, (double)init, (double)fin, nIter);
}

// One line per solved field, in OF's order for rhoSimpleFoam: U, then he, then p, then the continuity
// errors, then the turbulence scalars (rhoSimpleFoam.C -> UEqn.H, EEqn.H, pEqn.H, turbulence->correct()).
inline void printOfResidualLog(int iter, const DeviceSimpleResidual& r, scalar cumulativeCont)
{
    if (!ofResidualLog()) return;

    std::printf("\nTime = %d\n\n", iter);

    auto line = [](const char* solver, const char* field, scalar init, scalar fin, int nIter)
    {
        printOfSolveLine(solver, field, init, fin, nIter);
    };

    line(momentumSolverName(), "Ux", r.Ux, r.UxFinal, r.UxIters);
    line(momentumSolverName(), "Uy", r.Uy, r.UyFinal, r.UyIters);
    line(momentumSolverName(), "Uz", r.Uz, r.UzFinal, r.UzIters);

    // The energy, from the residual struct: correctTurbulence() clears the shared report store before the
    // turbulence solves, so the EEqn's entry is not there by the time this runs.
    if (r.heIters > 0 || r.he > 0) line("JacobiBiCGStab", "e", r.he, r.heFinal, r.heIters);

    line("AMG-PCG", "p", r.p, r.pFinal, r.pIters);

    // OF prints this straight after the pressure solve, from pEqn.H's continuityErrs.H.
    std::printf("time step continuity errors : sum local = %g, global = %g, cumulative = %g\n",
                (double)r.contLocal, (double)r.contGlobal, (double)cumulativeCont);

    for (const ScalarSolveEntry& e : turbulenceReport())
        line("JacobiBiCGStab", e.field.c_str(), e.perf.initialResidual, e.perf.finalResidual, e.perf.nIterations);
}

}   // namespace brae
