// interFoam's gravity fields: g, hRef, ghRef, gh/ghf, and p = p_rgh + rho*gh.
//
// THE POINT OF THIS FILE IS THE hRef ARM, and it is a gate about a gate.
//
// ghRef = g & (cmptMag(g)/mag(g))*hRef   (gh.H:2-7) -- g dotted with its own UNIT DIRECTION, times
// hRef, so the result carries g's SIGN. With g = (0,-9.81,0) it is NEGATIVE. Writing the plausible
// mag(g)*hRef flips it, and the error reaches p through `p = p_rgh + rho*gh` as a uniform offset that
// reads as a datum choice rather than a defect.
//
// readhRef.H is READ_IF_PRESENT with a default of 0, and NO interFoam tutorial that ships defines
// constant/hRef -- damBreak included. So on every stock case ghRef is 0 and the sign is unobservable:
// a gate built only on the tutorials passes with the formula written either way. This test therefore
// supplies its own non-zero hRef, which is the only way the arm can discriminate.
#include "inter_create_fields_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

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

std::string writeCase(const std::string& dir, const char* gValue, const char* hRefValue)
{
    std::filesystem::create_directories(dir + "/constant");
    std::ofstream(dir + "/constant/g")
        << "FoamFile { version 2.0; format ascii; class uniformDimensionedVectorField; object g; }\n"
        << "dimensions [0 1 -2 0 0 0 0];\nvalue " << gValue << ";\n";
    if (hRefValue)
        std::ofstream(dir + "/constant/hRef")
            << "FoamFile { version 2.0; format ascii; class uniformDimensionedScalarField; object hRef; }\n"
            << "dimensions [0 1 0 0 0 0 0];\nvalue " << hRefValue << ";\n";
    return dir;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== interFoam createFields: gravity ==\n");
    const std::string base = "/tmp/brae_inter_create_fields";
    std::filesystem::remove_all(base);

    // ---- 1. damBreak's own constant/g -------------------------------------------------------------
    const std::string tut = (argc > 1) ? argv[1] : "";
    if (tut.empty() || !std::filesystem::exists(tut + "/constant/g"))
    {
        // A SKIP, not a silent pass -- see tests/test_two_phase_mixture.cu for why this mattered.
        std::printf("  SKIP: OpenFOAM's damBreak tutorial not found at \"%s\"\n", tut.c_str());
        return 77;
    }
    {
        const vector g = readGravity(tut);
        checkNum("damBreak g.x", g.x, scalar(0));
        checkNum("damBreak g.y", g.y, scalar(-9.81));
        checkNum("damBreak g.z", g.z, scalar(0));
        checkNum("damBreak has no constant/hRef -> 0", readHRef(tut), scalar(0));
        check   ("...so its ghRef is 0 and CANNOT show a sign error", ghRef(g, readHRef(tut)) == scalar(0));
    }

    // ---- 2. THE SIGN, on a case that actually sets hRef -------------------------------------------
    {
        const std::string d = writeCase(base + "/href", "(0 -9.81 0)", "0.5");
        const vector g = readGravity(d);
        const scalar h = readHRef(d);
        checkNum("hRef is read when present", h, scalar(0.5));
        // g & (cmptMag(g)/mag(g)) = (0,-9.81,0) & (0,1,0) = -9.81, times 0.5 = -4.905
        checkNum("ghRef carries g's sign: -9.81*0.5", ghRef(g, h), scalar(-4.905));
        check   ("...and is NOT +mag(g)*hRef", std::fabs(ghRef(g, h) - scalar(4.905)) > scalar(1));

        // and with gravity pointing the other way it flips with it
        const std::string du = writeCase(base + "/up", "(0 9.81 0)", "0.5");
        checkNum("g upward -> ghRef positive", ghRef(readGravity(du), readHRef(du)), scalar(4.905));
    }

    // ---- 3. gh and ghf, same expression over different geometry -----------------------------------
    {
        const vector g{0, scalar(-9.81), 0};
        const scalar gr = scalar(-4.905);                 // hRef 0.5, as above
        const std::vector<vector> C {vector{0, 0, 0}, vector{0, 1, 0}, vector{0, scalar(2.5), 0}};
        const std::vector<vector> Cf{vector{0, scalar(0.5), 0}, vector{1, scalar(1.5), 0}};
        std::vector<scalar> gh, ghf;
        ghField(g, gr, C,  gh);
        ghField(g, gr, Cf, ghf);
        checkNum("gh at y=0",   gh[0],  scalar(0)              - gr);
        checkNum("gh at y=1",   gh[1],  scalar(-9.81)          - gr);
        checkNum("gh at y=2.5", gh[2],  scalar(-9.81) * scalar(2.5) - gr);
        checkNum("ghf at y=0.5", ghf[0], scalar(-9.81) * scalar(0.5) - gr);
        check("gh ignores x and z for a y-only g", ghf[1] == scalar(-9.81) * scalar(1.5) - gr);
    }

    // ---- 4. p = p_rgh + rho*gh --------------------------------------------------------------------
    // The field interFoam WRITES but never solves. rho is the mixture blend, so p carries the density
    // jump across the interface -- 1000 vs 1 on damBreak, which is why the two phases must already be
    // blended before this is formed.
    {
        const std::vector<scalar> p_rgh{scalar(1e5), scalar(1e5)};
        const std::vector<scalar> rho  {scalar(1000), scalar(1)};      // water cell, air cell
        const std::vector<scalar> gh   {scalar(-9.81), scalar(-9.81)};
        std::vector<scalar> p;
        staticPressure(p_rgh, rho, gh, p);
        checkNum("p in the water cell", p[0], scalar(1e5) + scalar(1000) * scalar(-9.81));
        checkNum("p in the air cell",   p[1], scalar(1e5) + scalar(1)    * scalar(-9.81));
        check("the two differ by the density ratio, not by a constant",
              std::fabs((p[0] - scalar(1e5)) / (p[1] - scalar(1e5)) - scalar(1000)) < scalar(1e-9));
    }

    // ---- 5. refusal, and the control --------------------------------------------------------------
    {
        std::filesystem::create_directories(base + "/nog/constant");
        std::ofstream(base + "/nog/constant/g")
            << "FoamFile { version 2.0; format ascii; class uniformDimensionedVectorField; object g; }\n"
            << "dimensions [0 1 -2 0 0 0 0];\n";               // no `value`
        bool threw = false;
        try { (void)readGravity(base + "/nog"); } catch (const std::exception&) { threw = true; }
        check("constant/g with no `value` is refused", threw);
        check("...and a well-formed one still reads (control)",
              readGravity(writeCase(base + "/ok", "(0 -9.81 0)", nullptr)).y == scalar(-9.81));
        // zero gravity: OF takes the `: 0` branch rather than dividing by mag(g)
        check("g = 0 gives ghRef 0 without dividing by zero",
              ghRef(vector{0, 0, 0}, scalar(0.5)) == scalar(0));
    }

    std::printf("test_inter_create_fields: %d failures\n", failures);
    return failures ? 1 : 0;
}
