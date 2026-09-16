// interfaceProperties: the coefficients, the stabiliser, and the face normal.
//
// WHAT THIS CATCHES, none of it arithmetic:
//
// 1. THE SOLVER KEY IS A REGEX. damBreak writes `"alpha.water.*"`, not `alpha.water`, so a reader that
//    looks up a literal finds nothing and -- if it then DEFAULTS cAlpha -- runs the case with an
//    interface compression the user never chose. OpenFOAM has no default for cAlpha
//    (interfaceProperties.C:184, get<scalar>), so the only correct behaviours are "resolve the regex"
//    or "refuse". This gate asserts the first against damBreak's own file.
//
// 2. deltaN IS MESH-DEPENDENT: 1e-8/cbrt(average(V)), interfaceProperties.C:194. It is what keeps nHat
//    finite where gradAlpha vanishes, which is most of the domain. A hard-coded 1e-8 is wrong on any
//    mesh whose cells are not of order unit volume, and the error is in the interface NORMAL, so it
//    shows up as curvature -- not as an obviously broken field.
//
// 3. nAlphaSmoothCurvature DOES default to 0 while cAlpha does not. Treating them alike either refuses
//    a case OpenFOAM runs or runs one it refuses.
//
// The oracle is OpenFOAM's own expressions from the lines cited; no solve.
#include "interface_properties_cpp.cuh"
#include "two_phase_mixture_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interfaceProps;

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
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== interfaceProperties ==\n");
    const std::string tut = (argc > 1) ? argv[1] : "";

    // ---- 1. damBreak's own dictionaries, regex key and all ----------------------------------------
    if (!tut.empty() && std::filesystem::exists(tut + "/system/fvSolution"))
    {
        const FoamDict fvSolution = readDict(tut + "/system/fvSolution");
        const cpu::twoPhase::MixtureSpec m = cpu::twoPhase::readTransportProperties(tut);
        // alpha1's field name is "alpha." + phase1Name -- damBreak's phase1 is water, so alpha.water,
        // and fvSolution keys it as the REGEX "alpha.water.*".
        const std::string alphaName = "alpha." + m.phase1Name;
        const InterfaceCoeffs c = readInterfaceCoeffs(fvSolution, alphaName, m.sigma);
        std::printf("  alpha field: %s   (fvSolution keys it \"%s.*\")\n", alphaName.c_str(), alphaName.c_str());
        checkNum("cAlpha read through the regex key", c.cAlpha, scalar(1));
        checkNum("nAlphaSmoothCurvature absent -> 0", scalar(c.nAlphaSmoothCurvature), scalar(0));
        checkNum("sigma from transportProperties", c.sigma, scalar(0.07));
    }
    else
    {
        std::printf("  (no tutorial given; dictionary arm skipped)\n");
    }

    // ---- 2. deltaN is mesh-dependent --------------------------------------------------------------
    {
        // Unit cells: average V = 1, so deltaN = 1e-8. This is the only mesh where the hard-coded
        // constant happens to be right, which is exactly why it is not a safe default.
        const std::vector<scalar> unitV(8, scalar(1));
        checkNum("deltaN on unit cells = 1e-8", deltaN(unitV), scalar(1e-8));

        // A millimetre-scale mesh: V = 1e-9, cbrt = 1e-3, so deltaN = 1e-5 -- a THOUSAND times the
        // constant. damBreak's cells are of this order.
        const std::vector<scalar> mmV(8, scalar(1e-9));
        checkNum("deltaN on 1 mm cells = 1e-5", deltaN(mmV), scalar(1e-5));
        check("...which is 1000x the hard-coded 1e-8", deltaN(mmV) > scalar(100) * scalar(1e-8));

        // Mixed volumes take the AVERAGE, not the min or the max.
        std::vector<scalar> mixed{scalar(1e-9), scalar(3e-9)};
        checkNum("deltaN uses average(V)", deltaN(mixed), scalar(1e-8) / std::cbrt(scalar(2e-9)));
    }

    // ---- 3. nHatfv = gradAlphaf/(mag + deltaN) ----------------------------------------------------
    {
        const scalar dN = scalar(1e-5);
        std::vector<vector> g{
            vector{3, 4, 0},              // mag 5, well inside the interface
            vector{0, 0, 0},              // AWAY from the interface: the stabiliser is all there is
            vector{1e-12, 0, 0}};         // near-zero gradient, where a bare divide would blow up
        std::vector<vector> nHatfv;
        faceUnitNormal(g, dN, nHatfv);

        checkNum("nHat.x on a mag-5 gradient", nHatfv[0].x, scalar(3) / (scalar(5) + dN));
        checkNum("nHat.y on a mag-5 gradient", nHatfv[0].y, scalar(4) / (scalar(5) + dN));
        check   ("a ZERO gradient gives a zero normal, not a NaN",
                 nHatfv[1].x == scalar(0) && std::isfinite(nHatfv[1].x));
        check   ("a 1e-12 gradient stays finite",  std::isfinite(nHatfv[2].x));
        check   ("...and is damped by deltaN rather than normalised to 1",
                 std::fabs(nHatfv[2].x) < scalar(1e-6));

        // nHatf = nHatfv & Sf
        std::vector<vector> Sf{vector{2, 0, 0}, vector{0, 1, 0}, vector{0, 0, 3}};
        std::vector<scalar> nHatf;
        faceNormalFlux(nHatfv, Sf, nHatf);
        checkNum("nHatf = nHatfv & Sf", nHatf[0], nHatfv[0].x * scalar(2));
    }

    // ---- 4. REFUSALS, and the control ------------------------------------------------------------
    {
        const std::string base = "/tmp/brae_interface_properties";
        std::filesystem::remove_all(base);
        auto write = [&](const std::string& dir, const std::string& solversBody)
        {
            std::filesystem::create_directories(base + "/" + dir + "/system");
            std::ofstream(base + "/" + dir + "/system/fvSolution")
                << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
                << "solvers\n{\n" << solversBody << "}\n";
            return readDict(base + "/" + dir + "/system/fvSolution");
        };
        auto refuses = [&](const FoamDict& d, const std::string& name)
        {
            try { (void)readInterfaceCoeffs(d, name, scalar(0.07)); } catch (const std::exception&) { return true; }
            return false;
        };

        check("a MISSING cAlpha is refused, not defaulted (OpenFOAM FatalErrors)",
              refuses(write("nocalpha", "    \"alpha.water.*\" { nAlphaSubCycles 1; }\n"), "alpha.water"));
        check("no solvers entry for the alpha field at all is refused",
              refuses(write("nokey",    "    p_rgh { solver PCG; tolerance 1e-7; }\n"), "alpha.water"));

        // CONTROL: the ordinary form must be accepted, or the two refusals above prove nothing.
        const FoamDict good = write("good", "    \"alpha.water.*\" { cAlpha 1; nAlphaSubCycles 1; }\n");
        check("a well-formed entry still reads", !refuses(good, "alpha.water"));
        checkNum("...and gives cAlpha 1", readInterfaceCoeffs(good, "alpha.water", scalar(0.07)).cAlpha, scalar(1));

        // ...and a LITERAL key must work too -- not every case writes the regex form.
        const FoamDict lit = write("literal", "    alpha.water { cAlpha 0.5; }\n");
        checkNum("a literal (non-regex) key also resolves",
                 readInterfaceCoeffs(lit, "alpha.water", scalar(0.07)).cAlpha, scalar(0.5));
    }

    std::printf("test_interface_properties: %d failures\n", failures);
    return failures ? 1 : 0;
}
