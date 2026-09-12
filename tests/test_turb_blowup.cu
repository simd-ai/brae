// The turbulence blow-up tripwire (solvers/common/turb_blowup.cuh).
//
// The tripwire has to separate two things that both cross its ratio bar:
//
//   a run that DIVERGES        -- sum|turb| climbs and never comes back; the fields at endTime are wrong
//                                 but finite and plausible, which is the whole reason the check exists
//   a start-up TRANSIENT       -- sum|turb| spikes and falls back; the run goes on to a physical field
//
// The first version of the check only tested "did it cross the bar", so it stopped a 5.96M-cell car case
// at iteration 3 that both brae and real OpenFOAM solve (OpenFOAM's own peak on that case is HIGHER).
// The sequences below are taken from those two measured shapes, so a regression to a bar-only test fails
// SPIKE_RECOVER, and dropping the check altogether fails the divergence cases.
#include "turb_blowup.cuh"
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using brae::TurbBlowup;

static int failures = 0;

static void check(
    const std::string& name,
    bool got,
    bool want)
{
    const bool ok = (got == want);
    if (!ok) ++failures;
    std::printf("  %-34s tripped=%-5s want=%-5s  %s\n",
                name.c_str(),
                got ? "yes" : "no",
                want ? "yes" : "no",
                ok ? "ok" : "FAIL");
}

// Feed a whole sequence through one detector and report whether it ever tripped.
static bool run(const std::vector<double>& mags)
{
    TurbBlowup d;
    for (std::size_t i = 0; i < mags.size(); ++i)
    {
        if (d.update((brae::scalar)mags[i], (int)i + 1)) return true;
    }
    return false;
}

int main()
{
    std::printf("turbulence blow-up tripwire\n");

    // 1. A healthy solve. sum|turb| grows by ~1e2 from a cold start and settles; nowhere near the bar.
    {
        std::vector<double> m;
        for (int i = 0; i < 200; ++i) m.push_back(1.0e3 * (1.0 + 99.0 * (double)i / 199.0));
        check("healthy solve", run(m), false);
    }

    // 2. THE CASE THAT EXPOSED THE BUG. Crosses the bar at iteration 2, peaks at 3, falls back, and ends
    //    on a physical field. Shape taken from the measured car case: base ~4e9, peak ~6e26, recovered
    //    ~2e5. A bar-only tripwire stops this at iteration 3; it must not.
    {
        std::vector<double> m = { 3.94e9, 4.06e16, 5.92e26, 5.92e25, 8.0e23, 4.0e20, 9.0e16, 3.0e13 };
        for (int i = 0; i < 192; ++i) m.push_back(2.0e5);
        check("spike then recover", run(m), false);
    }

    // 3. Divergence, monotone. The pitzDaily linearUpwind-on-both-scalars shape: climbs decade by decade
    //    to 1e42 and never returns. This is what the check is FOR.
    {
        std::vector<double> m = { 1.0e3 };
        for (int i = 0; i < 60; ++i) m.push_back(m.back() * 10.0);
        check("divergence, monotone", run(m), true);
    }

    // 4. Divergence that SAWTOOTHS. Trends up but dips every other iteration. Comparing against the
    //    previous iteration alone would let this run forever, which is why the streak is measured against
    //    the level it started at.
    {
        std::vector<double> m = { 1.0e3 };
        for (int i = 0; i < 60; ++i)
        {
            const double next = m.back() * ((i % 2 == 0) ? 100.0 : 0.5);
            m.push_back(next);
        }
        check("divergence, sawtooth", run(m), true);
    }

    // 5. Non-finite is fatal at once -- there is nothing to recover from.
    {
        std::vector<double> m = { 1.0e3, 2.0e3, std::numeric_limits<double>::quiet_NaN() };
        check("NaN trips immediately", run(m), true);
        std::vector<double> mi = { 1.0e3, 2.0e3, std::numeric_limits<double>::infinity() };
        check("Inf trips immediately", run(mi), true);
    }

    // 6. A run that sits just BELOW the bar forever is not divergence, however long it runs.
    {
        std::vector<double> m = { 1.0e3 };
        for (int i = 0; i < 500; ++i) m.push_back(1.0e3 * TurbBlowup::kRatio * 0.5);
        check("just below the bar", run(m), false);
    }

    // 7. THE CONTROL, and it is what makes the numbers above mean anything: a detector whose streak
    //    requirement is removed (bar only, the old behaviour) MUST stop the recovering transient. If this
    //    reports no trip, the sequence in test 2 is too tame to discriminate and tests 2 and 3 prove
    //    nothing.
    {
        const std::vector<double> m = { 3.94e9, 4.06e16, 5.92e26, 5.92e25 };
        TurbBlowup d;
        bool barOnly = false;
        for (std::size_t i = 0; i < m.size(); ++i)
        {
            d.update((brae::scalar)m[i], (int)i + 1);
            if (i > 0 && d.base > 0.0 && m[i] > TurbBlowup::kRatio * d.base) barOnly = true;
        }
        check("control: bar-only stops it", barOnly, true);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
