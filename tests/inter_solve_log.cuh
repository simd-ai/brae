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

using brae::cpu::interFoam::PressureSolveRecord;

// OpenFOAM's own account of every p_rgh solve: "DICPCG:  Solving for p_rgh, Initial residual = X,
// Final residual = Y, No Iterations N", in the order it ran them.
inline std::vector<PressureSolveRecord> readOfPressureSolves(const std::string& logPath)
{
    std::vector<PressureSolveRecord> out;
    std::ifstream in(logPath);
    std::string line;
    const std::string kI = "Initial residual = ", kF = "Final residual = ", kN = "No Iterations ";
    while (std::getline(in, line))
    {
        if (line.find("Solving for p_rgh,") == std::string::npos) continue;
        const std::size_t a = line.find(kI), b = line.find(kF), c = line.find(kN);
        if (a == std::string::npos || b == std::string::npos || c == std::string::npos) continue;
        PressureSolveRecord r;
        r.initialResidual = std::atof(line.c_str() + a + kI.size());
        r.finalResidual = std::atof(line.c_str() + b + kF.size());
        r.nIterations = std::atoi(line.c_str() + c + kN.size());
        out.push_back(r);
    }
    return out;
}

// THE SOLVER'S OWN LOG AGAINST OpenFOAM'S, solve by solve. This is the arm that says a solve STOPPED
// where OpenFOAM's did. A field comparison cannot: with p_rgh at relTol 0.05 the iterate handed back is
// wherever the preconditioner left it, and the device sat 9.2e-04 of |U| from OpenFOAM on this case
// with its discretisation exact to 3.5e-08, purely because it ran a different solver to the same
// relTol. An iteration count is the sharpest statement of "the same solver" a log can make -- Jacobi
// in the same CG loop takes 21 iterations where DIC takes 8 (tests/test_device_dic.cu).
inline int compareSolves(
    const char* who,
    const std::vector<PressureSolveRecord>& mine,
    const std::vector<PressureSolveRecord>& of,
    label nSteps)
{
    int failures = 0;
    auto check = [&](
        const char* what,
        bool ok)
    {
        std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
        if (!ok)
        {
            ++failures;
        }
    };
    std::size_t nSame = 0;
    scalar wInit = 0, wInitFirst = 0;
    const std::size_t perStep = nSteps > 0 ? of.size()/static_cast<std::size_t>(nSteps) : 0;
    for (std::size_t k = 0; k < mine.size() && k < of.size(); ++k)
    {
        if (mine[k].nIterations == of[k].nIterations)
        {
            ++nSame;
        }
        const scalar e = std::fabs(mine[k].initialResidual - of[k].initialResidual)
                       / std::fmax(of[k].initialResidual, scalar(1e-300));
        wInit = std::fmax(wInit, e);
        if (k < perStep)
        {
            wInitFirst = std::fmax(wInitFirst, e);
        }
    }
    std::printf("  %s p_rgh solves against OpenFOAM's log: %zu of %zu iteration counts equal;  initial "
                "residual worst %.3e in step one, %.3e over the run\n",
                who, nSame, of.size(), (double)wInitFirst, (double)wInit);
    std::printf("    iterations (OpenFOAM/%s):", who);
    for (std::size_t k = 0; k < mine.size() && k < of.size(); ++k)
    {
        std::printf(" %d/%d", of[k].nIterations, mine[k].nIterations);
    }
    std::printf("\n    initial residual, relative difference per solve:");
    for (std::size_t k = 0; k < mine.size() && k < of.size(); ++k)
    {
        std::printf(" %.1e", (double)(std::fabs(mine[k].initialResidual - of[k].initialResidual)
                                      / std::fmax(of[k].initialResidual, scalar(1e-300))));
    }
    std::printf("\n");
    check("...it ran as many p_rgh solves as OpenFOAM logged", mine.size() == of.size() && !of.empty());
    check("...EVERY ONE of them took OpenFOAM's iteration count", nSame == of.size() && !of.empty());
    // Step one starts both codes from the same fields, so its residuals are the solver's alone: 1e-10.
    // Later steps inherit the last corrector's `tolerance 1e-07` ball, inside which two correct CG
    // runs land in different places, so over the run the bound is that tolerance's and not round-off's.
    check("...from OpenFOAM's initial residuals, to 1e-10 in step one", wInitFirst < scalar(1e-10));
    check("...and to 1e-5 over the run, which is the final corrector's tolerance feeding forward",
          wInit < scalar(1e-5));
    return failures;
}

}   // namespace gatecheck
}   // namespace brae
