#pragma once
// The solver's own log against OpenFOAM's, for the interFoam gates -- shared because damBreak and
// capillaryRise both need it and a second copy is a second place for the parse to go wrong.
#include "cf_types.cuh"
#include "inter_peqn_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace brae {
namespace gatecheck {

using brae::cpu::interFoam::LinearSolveRecord;
using brae::cpu::interFoam::PressureSolveRecord;

// OpenFOAM's own account of every solve of one field: "<solver>:  Solving for <field>, Initial
// residual = X, Final residual = Y, No Iterations N", in the order it ran them. The trailing comma is
// part of the match, so "p_rgh" does not also collect a field that merely starts with it.
inline std::vector<LinearSolveRecord> readOfSolves(
    const std::string& logPath,
    const std::string& field)
{
    std::vector<LinearSolveRecord> out;
    std::ifstream in(logPath);
    std::string line;
    const std::string kI = "Initial residual = ", kF = "Final residual = ", kN = "No Iterations ";
    while (std::getline(in, line))
    {
        if (line.find("Solving for " + field + ",") == std::string::npos) continue;
        const std::size_t a = line.find(kI), b = line.find(kF), c = line.find(kN);
        if (a == std::string::npos || b == std::string::npos || c == std::string::npos) continue;
        LinearSolveRecord r;
        r.initialResidual = std::atof(line.c_str() + a + kI.size());
        r.finalResidual = std::atof(line.c_str() + b + kF.size());
        r.nIterations = std::atoi(line.c_str() + c + kN.size());
        out.push_back(r);
    }
    return out;
}

inline std::vector<LinearSolveRecord> readOfPressureSolves(const std::string& logPath)
{
    return readOfSolves(logPath, "p_rgh");
}

// THE SOLVER'S OWN LOG AGAINST OpenFOAM'S, solve by solve. This is the arm that says a solve STOPPED
// where OpenFOAM's did. A field comparison cannot: with p_rgh at relTol 0.05 the iterate handed back is
// wherever the preconditioner left it, and the device sat 9.2e-04 of |U| from OpenFOAM on this case
// with its discretisation exact to 3.5e-08, purely because it ran a different solver to the same
// relTol. An iteration count is the sharpest statement of "the same solver" a log can make -- Jacobi
// in the same CG loop takes 21 iterations where DIC takes 8 (tests/test_device_dic.cu).
// The relative difference of two NORMALISED residuals, with the floor a normalised residual has. It is
// sum|b - A.psi| over a normFactor of the same size as sum|b|, so each term is a cancellation of O(1)
// quantities and the quotient cannot be resolved below about 1e-16 whatever the two codes do. Without
// the floor this gate failed on agreement it should have praised: a non-orthogonal corrector's second
// solve starting from an already-converged p_rgh (initial residual 5e-8, the two codes 6e-18 apart,
// "1.3e-10 relative"), and a second outer corrector's alpha solve ending at 1.935e-14 in BOTH logs to
// four digits ("2.2e-05 relative", 4e-19 absolute). A different SOLVER is still nine orders above it.
inline scalar residualRelDiff(
    scalar mine,
    scalar of)
{
    const scalar d = std::fabs(mine - of);
    if (d <= scalar(1e-16)) return scalar(0);
    return d/std::fmax(of, scalar(1e-300));
}

inline int compareSolves(
    const char* who,
    const std::vector<LinearSolveRecord>& mine,
    const std::vector<LinearSolveRecord>& of,
    label nSteps,
    const char* field = "p_rgh",
    // the two bounds on the initial residual; the defaults are p_rgh's and are argued below
    scalar stepOneBound = scalar(1e-10),
    scalar runBound = scalar(1e-5),
    // a bound on the FINAL residual's relative difference; negative = not asserted. It is only
    // meaningful where the two codes run the same solver in the same operation order -- it is the arm
    // that tells symGaussSeidel from a Krylov solver that also happens to take one iteration.
    scalar finalBound = scalar(-1),
    // out: the worst relative difference of the final residuals, for a caller building a control
    scalar* worstFinalOut = nullptr,
    // false = MEASURE ONLY: print the comparison, assert nothing, return 0. For a control, whose whole
    // purpose is to disagree -- an uncounted "FAIL:" line in a passing log is a trap for the next reader.
    bool assertArms = true,
    // THE SOLVE'S ABSOLUTE TOLERANCE, for the one disagreement in an iteration count that is not a
    // different solver: a residual that lands ON the tolerance. A solve at `tolerance 1e-13` whose
    // sixteenth residual is 1.000e-13 stops or does not on the last bit of a number that is itself a
    // difference of two O(1) quantities, and the other code's sixteenth, summed in another order,
    // falls a bit above or below. MEASURED on waves/stokesI `tight`: OpenFOAM 16 with a final residual
    // of 1.000e-13, the host 17 ending at 5.3e-14 and the device 16 -- the same solver on the same
    // system. So a count that differs by EXACTLY ONE, where the shorter run's final residual is within
    // a thousandth of the tolerance, is counted as equal and printed as an edge stop. Negative = no
    // tolerance known, no exception. This is not a bound on the count: two apart, or one apart with
    // the shorter run's residual anywhere else, is still a different solve.
    scalar tolerance = scalar(-1))
{
    int failures = 0;
    auto check = [&](
        const char* what,
        bool ok)
    {
        if (!assertArms) return;
        std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
        if (!ok)
        {
            ++failures;
        }
    };
    std::size_t nSame = 0;
    std::size_t nEdge = 0;
    scalar wInit = 0, wInitFirst = 0;
    const std::size_t perStep = nSteps > 0 ? of.size()/static_cast<std::size_t>(nSteps) : 0;
    for (std::size_t k = 0; k < mine.size() && k < of.size(); ++k)
    {
        if (mine[k].nIterations == of[k].nIterations)
        {
            ++nSame;
        }
        else if (tolerance > scalar(0) && std::abs(mine[k].nIterations - of[k].nIterations) == 1)
        {
            const LinearSolveRecord& shorter = mine[k].nIterations < of[k].nIterations ? mine[k] : of[k];
            if (shorter.finalResidual <= tolerance
                && shorter.finalResidual >= tolerance*(scalar(1) - scalar(1e-3)))
            {
                ++nSame;
                ++nEdge;
                std::printf("  (solve %zu: %d against %d iterations, the shorter ending at %.4e on a "
                            "tolerance of %.1e -- an edge stop, counted as equal)\n",
                            k + 1, of[k].nIterations, mine[k].nIterations, (double)shorter.finalResidual,
                            (double)tolerance);
            }
        }
        const scalar e = residualRelDiff(mine[k].initialResidual, of[k].initialResidual);
        wInit = std::fmax(wInit, e);
        if (k < perStep)
        {
            wInitFirst = std::fmax(wInitFirst, e);
        }
    }
    std::printf("  %s %s solves against OpenFOAM's log: %zu of %zu iteration counts equal%s;  initial "
                "residual worst %.3e in step one, %.3e over the run\n",
                who, field, nSame, of.size(), nEdge ? " (one of them an edge stop)" : "",
                (double)wInitFirst, (double)wInit);
    std::printf("    iterations (OpenFOAM/%s):", who);
    for (std::size_t k = 0; k < mine.size() && k < of.size(); ++k)
    {
        std::printf(" %d/%d", of[k].nIterations, mine[k].nIterations);
    }
    std::printf("\n    final residual (OpenFOAM/%s):", who);
    for (std::size_t k = 0; k < mine.size() && k < of.size(); ++k)
    {
        std::printf(" %.3e/%.3e", (double)of[k].finalResidual, (double)mine[k].finalResidual);
    }
    scalar wFinal = 0;
    for (std::size_t k = 0; k < mine.size() && k < of.size(); ++k)
    {
        if (of[k].finalResidual > scalar(0))
        {
            wFinal = std::fmax(wFinal, residualRelDiff(mine[k].finalResidual, of[k].finalResidual));
        }
    }
    if (worstFinalOut)
    {
        *worstFinalOut = wFinal;
    }
    std::printf("  [worst relative difference %.3e]", (double)wFinal);
    std::printf("\n    initial residual, relative difference per solve:");
    for (std::size_t k = 0; k < mine.size() && k < of.size(); ++k)
    {
        std::printf(" %.1e", (double)residualRelDiff(mine[k].initialResidual, of[k].initialResidual));
    }
    std::printf("\n");
    check("...it ran as many solves of that field as OpenFOAM logged", mine.size() == of.size() && !of.empty());
    check("...EVERY ONE of them took OpenFOAM's iteration count", nSame == of.size() && !of.empty());
    // Step one starts both codes from the same fields, so its residuals are the solver's alone: 1e-10.
    // Later steps inherit the last corrector's `tolerance 1e-07` ball, inside which two correct CG
    // runs land in different places, so over the run the bound is that tolerance's and not round-off's.
    check("...from OpenFOAM's initial residuals in step one", wInitFirst < stepOneBound);
    check("...and over the run", wInit < runBound);
    if (finalBound > scalar(0))
    {
        check("...and it left OpenFOAM's FINAL residuals, which only the same solver does", wFinal < finalBound);
    }
    return failures;
}

}   // namespace gatecheck
}   // namespace brae
