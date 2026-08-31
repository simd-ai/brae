// generalizedNewtonian powerLaw's dictionary contract, against OpenFOAM's own.
//
// OF powerLaw.C:63-65 constructs n_, nuMin_ and nuMax_ straight from the coefficients dictionary --
// `dimensionedScalar(name, dims, powerLawCoeffs_)` -- which FATALS on a missing entry. There are no
// defaults. brae's guard tested only nuMax, so a case missing `n` silently got n = 1.0, which makes
// nu = nu0 identically: the NEWTONIAN answer on a case that asked for shear thinning. On
// squareBendLiqNoNewtonian's numbers that is a ~1120x viscosity error over essentially the whole field
// (turbulence_setup.cuh:33-40 carries the measurement).
//
// NO FIXTURE IS NEEDED: the guard is dictionary arithmetic, so the dicts are constructed exactly --
// including the three single-key deletions a real tutorial would only reach by a typo.
#include "turbulence_setup.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

static int failures = 0;

static void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

// One parse of one dict body, reporting whether readTurbulenceModel threw and what it said.
struct Result { bool threw = false; std::string msg; DeviceSimpleControls ctl; };

static Result run(const std::string& coeffs)
{
    const std::string dir = "/tmp/brae_powerlaw_guards";
    std::filesystem::create_directories(dir);
    std::ofstream f(dir + "/turbulenceProperties");
    f << "FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\n"
      << "simulationType laminar;\n"
      << "laminar { model generalizedNewtonian; viscosityModel powerLaw; " << coeffs << " }\n";
    f.close();

    Result r;
    r.ctl.turbulent = false;   // the powerLaw branch is the laminar one
    try { readTurbulenceModel(readDict(dir + "/turbulenceProperties"), r.ctl); }
    catch (const std::exception& e) { r.threw = true; r.msg = e.what(); }
    return r;
}

int main()
{
    std::printf("powerLaw dict guards: n, nuMin, nuMax all required (OF powerLaw.C:63-65)\n");

    // ---- THE NEGATIVE CONTROL: squareBendLiqNoNewtonian's own trio must be ACCEPTED. --------------
    // Without this, every refusal below passes on a parser that throws for any reason at all.
    {
        const Result r = run("n 0.4; nuMin 1e-3; nuMax 1;");
        check("the full trio is ACCEPTED (negative control)", !r.threw);
        // Non-vacuity: n must be the CASE's 0.4, not the old silent default of 1.0. A guard that threw
        // correctly but read wrongly would still converge to the Newtonian answer.
        check("...and n is the case's 0.4, not the old default 1.0",
              !r.threw && r.ctl.gnN > 0.39 && r.ctl.gnN < 0.41);
        check("...and nuMin/nuMax are the case's", !r.threw
              && r.ctl.gnNuMin > 0.9e-3 && r.ctl.gnNuMin < 1.1e-3 && r.ctl.gnNuMax > 0.99);
    }

    // ---- THE REFUSALS: each deletion OF would fatal on must refuse, and name the missing key. ------
    {
        const Result r = run("nuMin 1e-3; nuMax 1;");        // no n: the old guard ACCEPTED this
        check("missing n is refused (was silently n = 1.0, Newtonian)", r.threw);
        check("...and the refusal names n", r.msg.find("missing: n") != std::string::npos);
    }
    {
        const Result r = run("n 0.4; nuMax 1;");             // no nuMin: old guard ACCEPTED, clamp at 0
        check("missing nuMin is refused (was silently 0.0)", r.threw);
        check("...and the refusal names nuMin", r.msg.find("nuMin") != std::string::npos);
    }
    {
        const Result r = run("n 0.4; nuMin 1e-3;");          // no nuMax: the one the old guard caught
        check("missing nuMax is still refused", r.threw);
    }

    // ---- powerLawCoeffs{} spelling: OF's optionalSubDict, so the sub-dict form must work too. ------
    {
        const Result r = run("powerLawCoeffs { n 0.4; nuMin 1e-3; nuMax 1; }");
        check("the powerLawCoeffs{} spelling is ACCEPTED", !r.threw);
        const Result r2 = run("powerLawCoeffs { nuMin 1e-3; nuMax 1; }");
        check("...and missing n inside it is refused", r2.threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
