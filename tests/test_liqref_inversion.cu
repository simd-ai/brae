// brae's H2O he->T inversion against OpenFOAM's OWN answer, PATH AND ALL.
//
// OpenFOAM's species::thermo<>::T stops when the temperature STEP falls below T0*1e-4, a tolerance
// fixed from the initial guess (thermoI.H:43-88, thermo.C:33). It therefore does not iterate to
// convergence and its answer DEPENDS ON T0 -- measured with tools/liqref at p = 1e5, Ttrue = 400:
// six starting guesses give six answers spanning 2.4e-08 K. Any inversion that converges properly
// returns one number for all six and is a different function.
//
// This reads tools/liqref's table (OpenFOAM calling itself) and requires brae to reproduce every
// (form, p, Ttrue, T0) row. It also reports the spread of OF's own answers, so a reader can see that
// the rows differ and the comparison is not trivially satisfiable.
//   Run: test_liqref_inversion <oracle.txt> [bound]
#include "nsrds_functions.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace brae;

namespace {
// THE CONTROL, and it has to be live rather than remembered. This is what brae's inversion used to be:
// Newton driven to a tight ENERGY residual instead of OpenFOAM's temperature-step test. It lands on the
// true root, so it is arguably the better numeric -- and that is exactly why it is wrong here. If this
// reproduced OpenFOAM's table, the table would not be testing the path and the rewrite would be
// unfalsifiable.
scalar convergedInvert(EnergyForm form, scalar target, scalar p, scalar T0)
{
    scalar T = T0;
    for (int it = 0; it < 200; ++it)
    {
        const scalar err = h2oEnergy(form, p, T) - target;
        if (std::fabs(err) <= 1e-12 * std::fabs(target)) break;
        T -= err / h2oCpv(form, p, T);
        if (!(T == T)) break;
    }
    return T;
}
}   // namespace

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s <oracle.txt> [bound]\n", argv[0]); return 2; }
    const double bound = (argc > 2) ? std::atof(argv[2]) : 1e-12;

    std::FILE* fp = std::fopen(argv[1], "r");
    if (!fp) { std::printf("FAIL: cannot open %s\n", argv[1]); return 1; }

    char line[512];
    int rows = 0, bad = 0, ofNan = 0;
    double worst = 0.0, worstOfSpreadRel = 0.0;
    // Track OF's own spread per (form,p,Ttrue) so the report shows the rows are not identical.
    std::string key; double lo = 0, hi = 0; bool have = false;

    while (std::fgets(line, sizeof(line), fp))
    {
        if (std::strncmp(line, "INV ", 4) != 0) continue;
        char form[8]; double p, Ttrue, T0, Tof;
        if (std::sscanf(line, "INV %7s %lf %lf %lf %lf", form, &p, &Ttrue, &T0, &Tof) != 5) continue;
        const EnergyForm ef = (form[0] == 'e') ? EnergyForm::sensibleInternalEnergy
                                               : EnergyForm::sensibleEnthalpy;
        // The target energy is evaluated at the TRUE temperature, exactly as liqref did, so the only
        // thing under test is the inversion -- not the correlations, which test_nsrds already gates.
        const scalar target = h2oEnergy(ef, p, Ttrue);
        const HeToTResult r = h2oEnergyToT(ef, target, p, T0);
        ++rows;
        // OpenFOAM ITSELF diverges on some (p, T0) pairs and prints nan -- at p = 1e6 its Newton walks
        // out of the correlation range from a hot start and never comes back. Mirroring means failing
        // there too, so those rows assert that brae ALSO refuses rather than quietly returning a number.
        if (!(Tof == Tof))
        {
            if (r.converged)
            { std::printf("  MISMATCH OpenFOAM diverged (nan) but brae converged to %.17g: %s", (double)r.T, line); ++bad; }
            ++ofNan;
            continue;
        }
        const double rel = std::fabs(r.T - Tof) / std::fabs(Tof);
        if (!r.converged) { std::printf("  MISMATCH brae did not converge where OpenFOAM did: %s", line); ++bad; continue; }
        if (rel > bound)
        {
            if (bad < 8) std::printf("  MISMATCH form=%s p=%.0f Ttrue=%g T0=%g  brae %.17g  OF %.17g  rel %.3e\n",
                                     form, p, Ttrue, T0, (double)r.T, Tof, rel);
            ++bad;
        }
        worst = std::fmax(worst, rel);

        char k[128]; std::snprintf(k, sizeof(k), "%s|%.0f|%g", form, p, Ttrue);
        if (!have || key != k) { if (have && lo > 0) worstOfSpreadRel = std::fmax(worstOfSpreadRel, (hi - lo)/lo);
                                 key = k; lo = hi = Tof; have = true; }
        else { lo = std::fmin(lo, Tof); hi = std::fmax(hi, Tof); }
    }
    std::fclose(fp);
    if (have && lo > 0) worstOfSpreadRel = std::fmax(worstOfSpreadRel, (hi - lo)/lo);

    if (rows == 0) { std::printf("FAIL: no INV rows in %s\n", argv[1]); return 1; }

    // Re-read for the control: the converged inversion must MISS OpenFOAM's answer on rows whose T0 is
    // far from the root, because OpenFOAM stops before it gets there.
    int ctlRows = 0, ctlMiss = 0; double ctlWorst = 0.0;
    fp = std::fopen(argv[1], "r");
    while (fp && std::fgets(line, sizeof(line), fp))
    {
        if (std::strncmp(line, "INV ", 4) != 0) continue;
        char form[8]; double p, Ttrue, T0, Tof;
        if (std::sscanf(line, "INV %7s %lf %lf %lf %lf", form, &p, &Ttrue, &T0, &Tof) != 5) continue;
        if (!(Tof == Tof)) continue;
        const EnergyForm ef = (form[0] == 'e') ? EnergyForm::sensibleInternalEnergy
                                               : EnergyForm::sensibleEnthalpy;
        const scalar target = h2oEnergy(ef, p, Ttrue);
        const double Tc = (double)convergedInvert(ef, target, p, T0);
        if (!(Tc == Tc)) continue;
        const double rel = std::fabs(Tc - Tof) / std::fabs(Tof);
        ++ctlRows; ctlWorst = std::fmax(ctlWorst, rel);
        if (rel > bound) ++ctlMiss;
    }
    if (fp) std::fclose(fp);
    const bool ctlOk = ctlMiss > 0;
    std::printf("  CONTROL: a residual-converged inversion misses OpenFOAM on %d of %d rows"
                " (worst %.3e)  %s\n", ctlMiss, ctlRows, ctlWorst,
                ctlOk ? "ok" : "FAIL -- the bound cannot see the path, so this gate proves nothing");
    if (!ctlOk) ++bad;
    std::printf("  rows %d (%d where OpenFOAM itself diverged and brae must too)   worst |dT|/T %.3e   (bound %.1e)\n",
                rows, ofNan, worst, bound);
    std::printf("  OpenFOAM's OWN answers for one energy span %.3e relative across starting guesses --\n"
                "  a converged inversion cannot reproduce that, which is what makes these rows a test\n",
                worstOfSpreadRel);
    std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
