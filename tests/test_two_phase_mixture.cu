// The two-phase mixture: what the case says, and the three blends built from it.
//
// TWO THINGS THIS FILE EXISTS FOR, and neither is "does arithmetic work".
//
// 1. PHASE ORDER. `phases (water air)` makes alpha1 WATER. Read them the other way round and every
//    field is inverted while the solver runs perfectly happily -- the densities are 1000 and 1, so the
//    dam breaks upwards. Asserted against damBreak's own file, by name and by value.
//
// 2. RHO IS NOT CLAMPED AND MU IS. OpenFOAM:
//        rho = alpha1*rho1 + alpha2*rho2                        createFields.H:54    RAW alpha
//        mu  = a*rho1*nu1 + (1-a)*rho2*nu2,  a = clamp(alpha1)   ...Mixture.C:137     CLAMPED
//        nu  = mu/(a*rho1 + (1-a)*rho2)                          ...Mixture.C:55      CLAMPED
//    On a converged field MULES holds alpha1 in [0,1] and the two agree, so a port that clamps rho too
//    passes every smooth test and differs only on the overshoot MULES is ALLOWED to leave. The sweep
//    below therefore includes alpha OUTSIDE [0,1] on purpose, and the control at the end fails if rho
//    is clamped -- without it this gate could not tell the two apart.
//
// The oracle is OpenFOAM's own expression, transcribed from the lines cited above and evaluated here,
// exactly as tests/ does for the other closed-form blends. No solve, no tolerance beyond round-off.
#include "two_phase_mixture_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::twoPhase;

namespace {
int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

void checkNum(const char* what, scalar got, scalar want, scalar tol = scalar(1e-12))
{
    const bool ok = std::fabs(got - want) <= tol * std::fmax(scalar(1), std::fabs(want));
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s (got %.17g want %.17g)\n", what, (double)got, (double)want);
    if (!ok) ++failures;
}

// OpenFOAM's own expressions, kept separate from the implementation under test.
scalar ofClamp(scalar a) { return a < 0 ? scalar(0) : (a > 1 ? scalar(1) : a); }
scalar ofRho(scalar a1, scalar a2, const PhaseProperties& p) { return a1*p.rho1 + a2*p.rho2; }
scalar ofMu (scalar a1, const PhaseProperties& p)
{ const scalar a = ofClamp(a1); return a*p.rho1*p.nu1 + (1-a)*p.rho2*p.nu2; }
scalar ofNu (scalar a1, scalar mu, const PhaseProperties& p)
{ const scalar a = ofClamp(a1); return mu/(a*p.rho1 + (1-a)*p.rho2); }

}   // namespace

int main(int argc, char** argv)
{
    std::printf("== two-phase mixture ==\n");

    // ---- 1. the case's own transportProperties ---------------------------------------------------
    const std::string tut = (argc > 1) ? argv[1] : "";
    PhaseProperties p;
    if (!tut.empty() && std::filesystem::exists(tut + "/constant/transportProperties"))
    {
        const MixtureSpec m = readTransportProperties(tut);
        check    ("phases (water air) -> phase1 is water", m.phase1Name == "water");
        check    ("...and phase2 is air",                  m.phase2Name == "air");
        checkNum ("rho1 (water)", m.phases.rho1, scalar(1000));
        checkNum ("nu1  (water)", m.phases.nu1,  scalar(1e-06));
        checkNum ("rho2 (air)",   m.phases.rho2, scalar(1));
        checkNum ("nu2  (air)",   m.phases.nu2,  scalar(1.48e-05));
        checkNum ("sigma",        m.sigma,       scalar(0.07));
        p = m.phases;
    }
    else
    {
        // A SKIP, not a pass. This arm is the only one that reads damBreak's own phase ORDER, and it
        // reported green for its whole life because $FOAM_TUTORIALS is empty outside an OpenFOAM shell
        // and the path silently became "/multiphase/interFoam/...". ctest SKIP_RETURN_CODE is 77.
        std::printf("  SKIP: OpenFOAM's damBreak tutorial not found at \"%s\"\n", tut.c_str());
        return 77;
    }

    // ---- 2. the three blends, including alpha OUTSIDE [0,1] ---------------------------------------
    // MULES is bound-preserving to round-off, not exactly, so the overshoot is the interesting region:
    // it is the only place rho's raw alpha and mu's clamped alpha disagree.
    const std::vector<scalar> alphas =
        {scalar(-0.05), scalar(0), scalar(1e-9), scalar(0.25), scalar(0.5),
         scalar(0.75), scalar(1) - scalar(1e-9), scalar(1), scalar(1.05)};
    {
        std::vector<scalar> a1, a2, rho, mu, nu;
        for (scalar a : alphas) { a1.push_back(a); a2.push_back(scalar(1) - a); }
        mixtureRho(a1, a2, p, rho);
        mixtureMu (a1, p, mu);
        mixtureNu (a1, mu, p, nu);

        scalar wr = 0, wm = 0, wn = 0;
        for (std::size_t i = 0; i < alphas.size(); ++i)
        {
            const scalar orho = ofRho(a1[i], a2[i], p);
            const scalar omu  = ofMu (a1[i], p);
            const scalar onu  = ofNu (a1[i], omu, p);
            wr = std::fmax(wr, std::fabs(rho[i] - orho)/std::fmax(scalar(1), std::fabs(orho)));
            wm = std::fmax(wm, std::fabs(mu[i]  - omu )/std::fmax(scalar(1), std::fabs(omu )));
            wn = std::fmax(wn, std::fabs(nu[i]  - onu )/std::fmax(scalar(1), std::fabs(onu )));
        }
        std::printf("  worst relative vs OpenFOAM's own expression: rho %.3e  mu %.3e  nu %.3e\n",
                    (double)wr, (double)wm, (double)wn);
        check("rho matches OpenFOAM", wr == scalar(0));
        check("mu  matches OpenFOAM", wm == scalar(0));
        check("nu  matches OpenFOAM", wn == scalar(0));
    }

    // ---- 3. THE CONTROL: rho must NOT be clamped -------------------------------------------------
    // Without this the gate cannot distinguish the implementation from one that clamps everything, and
    // that difference is invisible on any in-range field. alpha1 = 1.05 with (1000, 1):
    //     raw     rho = 1.05*1000 + (-0.05)*1 = 1049.95
    //     clamped rho = 1.00*1000 +  0.00 *1  = 1000
    {
        const scalar over = scalar(1.05);
        std::vector<scalar> a1{over}, a2{scalar(1) - over}, rho;
        mixtureRho(a1, a2, p, rho);
        const scalar raw     = ofRho(over, scalar(1) - over, p);
        const scalar clamped = ofRho(ofClamp(over), scalar(1) - ofClamp(over), p);
        std::printf("  alpha1 = 1.05: raw rho %.6g, clamped rho %.6g (they differ by %.6g)\n",
                    (double)raw, (double)clamped, (double)std::fabs(raw - clamped));
        check("the two differ, so this control can discriminate", std::fabs(raw - clamped) > scalar(1));
        checkNum("rho takes the RAW alpha, as OpenFOAM does", rho[0], raw);
        check("...and is NOT the clamped value", std::fabs(rho[0] - clamped) > scalar(1));
    }

    // ---- 4. refusals -----------------------------------------------------------------------------
    {
        const std::string base = "/tmp/brae_two_phase_mixture";
        std::filesystem::remove_all(base);
        auto write = [&](const std::string& dir, const std::string& body)
        {
            std::filesystem::create_directories(base + "/" + dir + "/constant");
            std::ofstream(base + "/" + dir + "/constant/transportProperties")
                << "FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }\n"
                << body;
            return base + "/" + dir;
        };
        auto refuses = [&](const std::string& path)
        {
            try { (void)readTransportProperties(path); } catch (const std::exception&) { return true; }
            return false;
        };
        check("a non-Newtonian transportModel is refused by name",
              refuses(write("bird",
                  "phases (water air);\n"
                  "water { transportModel BirdCarreau; nu 1e-06; rho 1000; }\n"
                  "air   { transportModel Newtonian;  nu 1.48e-05; rho 1; }\n")));
        check("a phase named in `phases` with no sub-dictionary is refused",
              refuses(write("missing",
                  "phases (water air);\n"
                  "water { transportModel Newtonian; nu 1e-06; rho 1000; }\n")));
        check("three phases are refused (this mixture is two)",
              refuses(write("three",
                  "phases (water air oil);\n"
                  "water { transportModel Newtonian; nu 1e-06; rho 1000; }\n"
                  "air   { transportModel Newtonian; nu 1.48e-05; rho 1; }\n"
                  "oil   { transportModel Newtonian; nu 1e-04; rho 900; }\n")));
        // CONTROL: the ordinary file must still be accepted, or the refusals above would be worthless.
        check("a well-formed two-phase file still builds",
              !refuses(write("good",
                  "phases (water air);\n"
                  "water { transportModel Newtonian; nu 1e-06; rho 1000; }\n"
                  "air   { transportModel Newtonian; nu 1.48e-05; rho 1; }\n"
                  "sigma 0.07;\n")));
    }

    std::printf("test_two_phase_mixture: %d failures\n", failures);
    return failures ? 1 : 0;
}
