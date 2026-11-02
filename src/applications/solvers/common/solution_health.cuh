#pragma once
// The SIMPLE loop's divergence detector, as decisions that can be tested on their own.
//
// WHY THE RESIDUALS AND THE SUM ARE NOT ENOUGH. A blow-up of k and omega on pitzDailySST with linearUpwind honoured
// (jobs 4835242, 4835375) took omega from 1e5 to 1e14 and to 9e17 in one iteration, left k at its floor in 2,000 of
// 12,225 cells and froze there: every residual fell to 1e-9, which reads as convergence, and sum|k|+sum|omega| stayed
// under turb_blowup.cuh's 1e12 times its first value, so the run wrote the fields and reported success.
//
// THREE TESTS, none of them a magnitude picked by hand.
//   A residual that is not a finite non-negative number: nothing can follow it, so at once.
//   A field holding a non-finite value, or a transported turbulence field below zero after Foam::bound, whose floor
//   is positive: neither can recover, so at once.
//   k and epsilon or omega above what their own equation can sustain. omega has no source that holds it above its
//   wall function's and its fixed faces' values or the balance of production and destruction, gamma S2 = beta omega^2
//   (so sqrt(gamma/beta) sqrt(S2)); epsilon the same with C1 G = C2 epsilon. The caller works that limit out from the
//   fields of the iteration. Being above it is not divergence by itself: the 5.96 M-cell car crosses every such bar
//   on a cold start, in OpenFOAM as well (omega 1.4e30 at iteration 3), and comes back. So the test is that the run
//   has stopped getting back: far above the limit for two windows of iterations, the second's lowest no lower than the first's.
#include "cf_types.cuh"
#include <cmath>
#include <string>

namespace brae {

// One field's reductions over the cells of this rank, on the same iteration.
struct HealthField
{
    const char* name = "";
    scalar      min = 0.0;
    scalar      max = 0.0;
    scalar      mean = 0.0;
    bool        boundedBelow = false;   // transported turbulence: Foam::bound has just floored it above zero
};

inline std::string healthResidual(
    const std::string& field,
    scalar             residual,
    int                iter)
{
    if (std::isfinite(residual) && residual >= scalar(0)) return {};
    return "invalid residual for " + field + " at iteration " + std::to_string(iter) + " (" + std::to_string((double)residual) + ")";
}

inline std::string healthField(
    const HealthField& f,
    int                iter)
{
    // The mean as well: a min or a max kernel can pass over a NaN, and a sum cannot.
    if (!(std::isfinite(f.min) && std::isfinite(f.max) && std::isfinite(f.mean)))
    {
        return std::string("non-finite value in ") + f.name + " at iteration " + std::to_string(iter) + " (min "
             + std::to_string((double)f.min) + " max " + std::to_string((double)f.max) + " mean " + std::to_string((double)f.mean) + ")";
    }
    if (f.boundedBelow && f.min < scalar(0))
    {
        return std::string("negative ") + f.name + " after Foam::bound at iteration " + std::to_string(iter) + " (min " + std::to_string((double)f.min) + ")";
    }
    return {};
}

// Rolling state for one run: the turbulence field against the limit its own equation sets.
struct HealthAdmissible
{
    // Far above the limit rather than above it: a limit from one iteration's cell values is a scale and the scheme can
    // overshoot it. MEASURED, the largest ratio on a run that reaches a steady state: see tests/simple_divergence_detector.sh.
    static constexpr scalar kExcess = 1e3;
    // Two whole windows above the bar, the second's lowest excess no real step under the first's. Compared by window because a run
    // that recovers need not fall every iteration and one that runs away need not rise every iteration: one that gains a decade and
    // gives back half every other iteration falls half the time, and a peak falling back is still above every value before the peak.
    static constexpr int kWindow = 50;
    static constexpr scalar kNewLow = 0.9;

    scalar last = 0.0;
    scalar windowLow = 0.0;
    scalar previousLow = 0.0;
    int    count = 0;
    int    above = 0;

    bool update(
        scalar value,
        scalar limit)
    {
        if (!(limit > scalar(0)))
        {
            return false;
        }
        last = value / limit;
        if (last <= kExcess)
        {
            previousLow = 0.0;
            count = 0;
            above = 0;
            return false;
        }
        ++above;
        windowLow = count == 0 ? last : std::fmin(windowLow, last);
        ++count;
        if (count < kWindow)
        {
            return false;
        }
        const bool stuck = previousLow > scalar(0) && windowLow >= kNewLow * previousLow;
        previousLow = windowLow;
        count = 0;
        return stuck;
    }

    std::string message(
        const std::string& field,
        int                iter) const
    {
        return field + " stands " + std::to_string((double)last) + " times above what its equation can sustain at iteration "
             + std::to_string(iter) + ", " + std::to_string(above) + " iterations above the bar without coming back";
    }
};

}   // namespace brae
