#pragma once
// The turbulence blow-up tripwire, as a decision that can be tested on its own.
//
// WHY IT EXISTS. A diverging k-epsilon / k-omega pair need not reach NaN, so the non-finite residual check
// does not catch it. Measured on pitzDaily with linearUpwind on BOTH turbulence scalars, omega reached
// 1e42 with k pinned at its 1e-15 floor, U stayed bounded (nut collapses, so the flow just goes
// near-laminar), every residual stayed finite, and the run marched to endTime and WROTE the fields
// reporting success. That is worse than a crash, because the output looks plausible.
//
// WHY A RATIO ON ITS OWN IS NOT ENOUGH, and this is what this file fixes. The first version tripped as
// soon as sum|turb| exceeded 1e12 times its iteration-1 value. That also fires on a violent START-UP
// TRANSIENT which recovers, and refuses a case the solver can actually solve. Measured on a 5.96M-cell
// snappyHexMesh car case (kOmegaSST, SIMPLEC, cold uniform start), sum|turb| crosses the bar at iteration
// 2, peaks around iteration 3, and is back to a physical field by iteration 200 -- k max 134, omega max
// 2.1e5, nut max 0.105, nothing non-finite. Real OpenFOAM on the same mesh and the same dictionaries goes
// through the SAME excursion and peaks HIGHER (omega 1.4e30 against brae's 5.9e26 at iteration 3), then
// recovers as well. So the excursion is a property of the case, not a defect in the solver, and aborting
// on it refused a case both codes solve.
//
// WHAT ACTUALLY SEPARATES THE TWO. A diverging run stays up; a transient peaks and comes back DOWN
// THROUGH THE BAR. So the bar is not "went above the threshold once" but "STAYED above the threshold for
// kStreak consecutive iterations", and only falling back below the bar resets the count. The car case
// crosses for a handful of iterations and then drops decades below it, so it is let through, while a
// diverging run never comes back and trips kStreak iterations after it crosses.
//
// The count is deliberately NOT "kept growing". A diverging run need not grow monotonically -- one that
// gains a decade and gives back half every other iteration still runs away, and a growth test lets it
// through forever. Staying above the bar is the property divergence actually has.
#include "cf_types.cuh"
#include <cmath>
#include <string>

namespace brae {

// Rolling state for one run. `update()` is called once per outer iteration with sum|k| + sum|eps|omega|,
// and returns whether the run should be stopped without writing.
struct TurbBlowup
{
    // A tripwire, not a convergence criterion. Growth is measured against iteration 1 rather than an
    // absolute value so the bar is independent of mesh size and of the case's units.
    static constexpr scalar kRatio  = 1e12;
    // Consecutive iterations that must stay above the bar before this calls it divergence. The car case
    // crosses for only a few iterations before dropping decades below, so this has margin; a diverging run
    // never comes back and pays only these iterations before stopping.
    static constexpr int    kStreak = 25;

    scalar base   = 0.0;   // sum|turb| at iteration 1
    scalar last   = 0.0;   // most recent sum|turb|, for the message
    int    streak = 0;     // consecutive iterations above the bar

    // True when the run has to stop. Non-finite is fatal immediately -- there is nothing to recover from.
    bool update(scalar tm, int iter)
    {
        last = tm;
        if (!std::isfinite(tm)) return true;
        if (iter <= 1)
        {
            base   = tm;
            streak = 0;
            return false;
        }
        const bool above = (base > 0.0) && (tm > kRatio * base);
        streak = above ? streak + 1 : 0;
        return streak >= kStreak;
    }

    std::string message(int iter) const
    {
        return "solution diverged: turbulence blow-up at iteration " + std::to_string(iter)
             + " (sum|k|+sum|eps/omega| grew from " + std::to_string((double)base) + " to "
             + std::to_string((double)last)
             + (std::isfinite(last)
                    ? " and stayed above " + std::to_string((double)kRatio) + "x that for "
                      + std::to_string(streak) + " consecutive iterations without coming back down"
                    : "")
             + "). The momentum residuals can stay finite while this happens, so the run would otherwise"
             + " write a plausible-looking but wrong field. No field written."
             + " Set BRAE_ALLOW_NONFINITE=1 to continue anyway.";
    }
};

}   // namespace brae
