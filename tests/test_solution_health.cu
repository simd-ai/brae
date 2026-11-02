// The divergence detector's decisions (solvers/common/solution_health.cuh), with no solver in the way.
// Every invalid residual and field it claims to read stops the run; the measured trajectories decide the third test:
// the frozen blow-ups of pitzDailySST stop, a car's cold start that falls back does not. Each arm has its opposite.
#include "solution_health.cuh"
#include <cstdio>
#include <limits>
#include <vector>

using brae::HealthAdmissible;
using brae::HealthField;
using brae::healthField;
using brae::healthResidual;

static int failures = 0;

static void check(
    const char* name,
    bool        got,
    bool        want)
{
    const bool ok = got == want;
    if (!ok) ++failures;
    std::printf("  %-52s stopped=%-5s want=%-5s %s\n", name, got ? "yes" : "no", want ? "yes" : "no", ok ? "ok" : "FAIL");
}

static HealthField field(
    double a,
    double b,
    double m,
    bool   bounded)
{
    HealthField f;
    f.name = "x";
    f.min = a;
    f.max = b;
    f.mean = m;
    f.boundedBelow = bounded;
    return f;
}

// The ratios of value to limit, one an iteration
static bool run(const std::vector<double>& r)
{
    HealthAdmissible h;
    for (double x : r)
    {
        if (h.update(x, 1.0)) return true;
    }
    return false;
}

int main()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::printf("residuals\n");
    check("a residual of 0.5", !healthResidual("k", 0.5, 1).empty(), false);
    check("a residual of 0", !healthResidual("k", 0.0, 1).empty(), false);
    check("NaN", !healthResidual("k", nan, 1).empty(), true);
    check("Inf", !healthResidual("k", inf, 1).empty(), true);
    check("negative", !healthResidual("k", -1e-3, 1).empty(), true);
    std::printf("fields\n");
    check("finite, negative where that is allowed", !healthField(field(-5, 5, 0, false), 1).empty(), false);
    check("floored and positive", !healthField(field(1e-15, 5, 1, true), 1).empty(), false);
    check("a NaN only the mean shows", !healthField(field(0, 5, nan, false), 1).empty(), true);
    check("a NaN in the min", !healthField(field(nan, 5, 1, false), 1).empty(), true);
    check("Inf in the max", !healthField(field(0, inf, inf, false), 1).empty(), true);
    check("negative after Foam::bound", !healthField(field(-1e-20, 5, 1, true), 1).empty(), true);
    std::printf("the limit a field's own equation sets\n");
    {
        std::vector<double> m(5000, 0.3);
        check("under the limit for good", run(m), false);
    }
    {
        // Above the limit, but by less than the bar, for good: a scheme's overshoot.
        std::vector<double> m(5000, HealthAdmissible::kExcess * 0.5);
        check("over the limit, under the bar, for good", run(m), false);
    }
    {
        // pitzDailySST, job 4835375 P-hon-dn: from 1 to 1e9 in one iteration, then frozen
        std::vector<double> m(2461, 0.5);
        m.resize(3000, 1e9);
        check("a frozen blow-up", run(m), true);
    }
    {
        // Rounding on the frozen value is not a way back
        std::vector<double> m(100, 0.5);
        for (int i = 0; i < 500; ++i) m.push_back(i % 2 ? 1e9 : 1e9 * (1.0 + 1e-6));
        check("frozen with rounding", run(m), true);
    }
    {
        // Growing without limit
        std::vector<double> m;
        double x = 1e4;
        for (int i = 0; i < 200; ++i)
        {
            m.push_back(x);
            x *= 1.5;
        }
        check("growing", run(m), true);
    }
    {
        // The car's cold start: peaks decades above any bar at iteration 3, in OpenFOAM as well, and falls back over 200.
        std::vector<double> m = {1.0, 1e15, 1e24};
        double x = 1e24;
        for (int i = 0; i < 200; ++i)
        {
            x *= 0.8;
            m.push_back(x);
        }
        m.resize(1000, 0.5);
        check("a start-up that falls back", run(m), false);
    }
    {
        // Falling back in steps: long plateaus, each ending in a new low within a window
        std::vector<double> m;
        double x = 1e12;
        for (int i = 0; i < 12; ++i)
        {
            m.insert(m.end(), HealthAdmissible::kWindow - 2, x);
            x *= 0.1;
        }
        m.resize(2000, 0.5);
        check("falling back in steps", run(m), false);
    }
    {
        // ...and the fail-proof of that arm: the same plateau two windows long stops
        std::vector<double> m(2 * HealthAdmissible::kWindow, 1e12);
        check("one plateau two windows long", run(m), true);
        m.pop_back();
        check("...one iteration short of that", run(m), false);
    }
    {
        // Gains a decade, gives back half, every other iteration: half of its iterations fall
        std::vector<double> m;
        double x = 1e4;
        for (int i = 0; i < 150; ++i)
        {
            x *= (i % 2) ? 0.5 : 10.0;
            m.push_back(x);
        }
        check("rising in a saw-tooth", run(m), true);
    }
    {
        // Back under the bar resets: two excursions, each under the two windows, do not add up
        std::vector<double> m(2 * HealthAdmissible::kWindow - 2, 1e12);
        m.push_back(0.5);
        m.insert(m.end(), 2 * HealthAdmissible::kWindow - 2, 1e12);
        m.resize(500, 0.5);
        check("two short excursions", run(m), false);
    }
    {
        HealthAdmissible h;
        check("no limit set: never", h.update(1e30, 0.0), false);
    }
    std::printf(failures ? "FAILED %d\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
