// Four case-dictionary readers refuse, or honour, exactly what OpenFOAM would -- items 16d/e/g/h/i.
//
//   (i) parseFieldDivScheme matches the scheme word EXACTLY: `LUST` used to set no flag and run as
//       upwind; `linearUpwindV` / `limitedLinearV` matched their unlimited cousins by substring. On a
//       vector each is its own flag; on a scalar the V forms and LUST are refused by name; an unported
//       name is refused by name.
//   readTurbulenceModel expects ctl.turbulent preset by its caller (the drivers derive it from
//   simulationType), so the tests set it.
//   (e) readTurbulenceModel refuses an LES `delta` it has not ported instead of warning and running
//       cubeRootVol under it.
//   (d) readTurbulenceModel reads the Spalart-Allmaras coefficients from the coeffDict, as
//       SpalartAllmarasBase.C does, and refuses `ft2 true` (unimplemented term).
//   (g) readThermoCoeffs takes Prt from constant/momentumTransport when that is the name the case
//       carries (createFields accepts both names; Prt used to be read from turbulenceProperties only).
//   (h) readField stores a wall function's per-patch Cmu/kappa/E instead of skipping them.
//
// Every "refuses" line has a "still accepts" twin on the same reader, so a reader that started
// throwing on everything would fail too. Fail-proofs were the pre-fix readers: (i) LUST parsed with
// every flag false, (e) a warning and cubeRootVol, (d) Cb1 stayed at 0.1355 under `Cb1 0.2`,
// (g) Prt stayed 1.0 under momentumTransport, (h) kappa was skipped.
#include "foam_dict.cuh"
#include "scheme_parse.cuh"
#include "turbulence_setup.cuh"
#include "solver_controls.cuh"
#include "thermo_parse.cuh"
#include "foam_field_reader.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace brae;
namespace fs = std::filesystem;

static int g_fails = 0;
static void say(bool ok, const std::string& what)
{
    std::printf("  %-78s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fails;
}
template <class F>
static bool throws(F f, std::string* msg = nullptr)
{
    try { f(); return false; }
    catch (const std::exception& e) { if (msg) *msg = e.what(); return true; }
}
static void write(const std::string& path, const std::string& text)
{
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    f << "FoamFile { version 2.0; format ascii; class dictionary; object x; }\n" << text;
}
static std::string schemesCase(const std::string& dir, const std::string& uScheme, const std::string& tScheme)
{
    write(dir + "/system/fvSchemes",
          "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\n"
          "divSchemes\n{\n    default none;\n    div(phi,U) " + uScheme + ";\n    div(phi,T) " + tScheme + ";\n}\n"
          "laplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\n"
          "snGradSchemes { default corrected; }\n");
    return dir;
}

int main()
{
    const std::string base = fs::temp_directory_path().string() + "/brae_case_dict_refusals_" + std::to_string(::getpid());
    fs::remove_all(base);

    // ---- (i) the field scheme parser -----------------------------------------------------------
    {
        std::printf("(i) parseFieldDivScheme\n");
        const std::string c = schemesCase(base + "/i1", "Gauss LUST grad(U)", "Gauss LUST grad(T)");
        const FieldDivScheme u = parseFieldDivScheme(c, "U", /*vectorField=*/true);
        say(u.lust && !u.linearUpwind && !u.limited, "vector `Gauss LUST grad(U)` sets lust and nothing else");
        say(throws([&] { parseFieldDivScheme(c, "T"); }), "scalar `Gauss LUST` is refused by name");
        const std::string c2 = schemesCase(base + "/i2", "bounded Gauss linearUpwindV grad(U)", "bounded Gauss linearUpwind grad(T)");
        const FieldDivScheme u2 = parseFieldDivScheme(c2, "U", true);
        say(u2.linearUpwindV && !u2.linearUpwind && u2.bounded, "vector `linearUpwindV` sets linearUpwindV, not linearUpwind");
        const FieldDivScheme t2 = parseFieldDivScheme(c2, "T");
        say(t2.linearUpwind && !t2.linearUpwindV, "scalar `linearUpwind` still accepted");
        const std::string c3 = schemesCase(base + "/i3", "Gauss limitedLinearV 0.5", "Gauss limitedLinear 1");
        const FieldDivScheme u3 = parseFieldDivScheme(c3, "U", true);
        say(u3.limitedLinearV && !u3.limited && u3.coeff == scalar(0.5), "vector `limitedLinearV 0.5` sets limitedLinearV with its coefficient");
        say(throws([&] { parseFieldDivScheme(c3, "U", /*vectorField=*/false); }), "`limitedLinearV` on a scalar is refused");
        const FieldDivScheme t3 = parseFieldDivScheme(c3, "T");
        say(t3.limited && t3.coeff == scalar(1.0), "scalar `limitedLinear 1` still accepted");
        const std::string c4 = schemesCase(base + "/i4", "Gauss QUICK", "Gauss upwind");
        std::string msg;
        say(throws([&] { parseFieldDivScheme(c4, "U", true); }, &msg) && msg.find("QUICK") != std::string::npos,
            "an unported scheme name (QUICK) is refused BY NAME");
        say(!throws([&] { parseFieldDivScheme(c4, "T"); }), "`upwind` still accepted");
    }

    // ---- (e) LES delta -----------------------------------------------------------------------
    {
        std::printf("(e) LES delta\n");
        auto turb = [&](const std::string& dir, const std::string& delta)
        {
            write(dir + "/constant/turbulenceProperties",
                  "simulationType LES;\nLES\n{\n    LESModel Smagorinsky;\n    turbulence on;\n    delta " + delta + ";\n"
                  "    SmagorinskyCoeffs { Ck 0.094; Ce 1.048; }\n}\n");
            return readDict(dir + "/constant/turbulenceProperties");
        };
        std::string msg;
        say(throws([&] { DeviceSimpleControls ctl; ctl.turbulent = true; readTurbulenceModel(turb(base + "/e1", "vanDriest"), ctl, {"test", true, true}); }, &msg)
            && msg.find("vanDriest") != std::string::npos, "LES `delta vanDriest` is refused by name");
        say(!throws([&] { DeviceSimpleControls ctl; ctl.turbulent = true; readTurbulenceModel(turb(base + "/e2", "cubeRootVol"), ctl, {"test", true, true}); }),
            "LES `delta cubeRootVol` still accepted");
        say(!throws([&] { DeviceSimpleControls ctl; ctl.turbulent = true; readTurbulenceModel(turb(base + "/e3", "maxDeltaxyz"), ctl, {"test", true, true}); }),
            "LES `delta maxDeltaxyz` still accepted");
    }

    // ---- (d) Spalart-Allmaras coefficients ---------------------------------------------------
    {
        std::printf("(d) SpalartAllmaras coefficients\n");
        auto turb = [&](const std::string& dir, const std::string& coeffs)
        {
            write(dir + "/constant/turbulenceProperties",
                  "simulationType RAS;\nRAS\n{\n    RASModel SpalartAllmaras;\n    turbulence on;\n" + coeffs + "}\n");
            return readDict(dir + "/constant/turbulenceProperties");
        };
        DeviceSimpleControls a; a.turbulent = true; readTurbulenceModel(turb(base + "/d1", ""), a, {"test", true, true});
        say(a.sa && a.saCoeffs.Cb1 == scalar(0.1355) && a.saCoeffs.sigmaNut == scalar(0.66666), "no coeffDict: OpenFOAM's defaults");
        DeviceSimpleControls b; b.turbulent = true; readTurbulenceModel(turb(base + "/d2", "    SpalartAllmarasCoeffs { Cb1 0.2; sigmaNut 0.8; Cw2 0.35; }\n"), b, {"test", true, true});
        say(b.saCoeffs.Cb1 == scalar(0.2) && b.saCoeffs.sigmaNut == scalar(0.8) && b.saCoeffs.Cw2 == scalar(0.35)
            && b.saCoeffs.Cb2 == scalar(0.622), "SpalartAllmarasCoeffs { Cb1 sigmaNut Cw2 } are read; the rest keep their defaults");
        DeviceSimpleControls d; d.turbulent = true; readTurbulenceModel(turb(base + "/d4", "    Cb1 0.19;\n"), d, {"test", true, true});
        say(d.saCoeffs.Cb1 == scalar(0.19), "a coefficient at the RAS level reaches the model (optionalSubDict fallback, as OpenFOAM)");
        std::string msg;
        say(throws([&] { DeviceSimpleControls c; c.turbulent = true; readTurbulenceModel(turb(base + "/d3", "    SpalartAllmarasCoeffs { ft2 true; }\n"), c, {"test", true, true}); }, &msg)
            && msg.find("ft2") != std::string::npos, "`ft2 true` is refused by name (term not implemented)");
    }

    // ---- (g) Prt from momentumTransport ------------------------------------------------------
    {
        std::printf("(g) Prt source\n");
        auto thermoCase = [&](const std::string& dir, const std::string& turbFile)
        {
            write(dir + "/constant/thermophysicalProperties",
                  "thermoType\n{\n    type hePsiThermo;\n    mixture pureMixture;\n    transport const;\n    thermo hConst;\n"
                  "    equationOfState perfectGas;\n    specie specie;\n    energy sensibleEnthalpy;\n}\n"
                  "mixture\n{\n    specie { molWeight 28.9; }\n    thermodynamics { Cp 1005; Hf 0; }\n    transport { mu 1.8e-05; Pr 0.7; }\n}\n");
            write(dir + "/system/fvSolution", "solvers { p { solver GAMG; tolerance 1e-8; relTol 0.01; } }\nSIMPLE { }\n");
            write(dir + "/constant/" + turbFile,
                  "simulationType RAS;\nRAS\n{\n    RASModel kEpsilon;\n    turbulence on;\n    kEpsilonCoeffs { Prt 0.7; }\n}\n");
            return dir;
        };
        const ThermoCoeffs a = readThermoCoeffs(thermoCase(base + "/g1", "momentumTransport"));
        say(a.Prt == scalar(0.7), "Prt 0.7 in constant/momentumTransport reaches the thermo");
        const ThermoCoeffs b = readThermoCoeffs(thermoCase(base + "/g2", "turbulenceProperties"));
        say(b.Prt == scalar(0.7), "Prt 0.7 in constant/turbulenceProperties still reaches the thermo");
    }

    // ---- (h) per-patch wall-function coefficients --------------------------------------------
    {
        std::printf("(h) wall-function coefficients on the patch\n");
        const std::string dir = base + "/h1";
        fs::create_directories(dir + "/0");
        {
            std::ofstream f(dir + "/0/nut");
            f << "FoamFile { version 2.0; format ascii; class volScalarField; object nut; }\n"
                 "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
                 "    wall  { kappa 0.4; type nutkWallFunction; E 9.8; value uniform 0; }\n"
                 "    inlet { type calculated; value uniform 0; }\n}\n";
        }
        const FieldData<scalar> fd = readField<scalar>(dir + "/0/nut");
        bool found = false, right = false;
        for (const auto& b : fd.boundary)
            if (b.name == "wall") { found = true; right = b.hasWfKappa && b.wfKappa == scalar(0.4) && b.hasWfE && b.wfE == scalar(9.8) && !b.hasWfCmu; }
        say(found && right, "`kappa 0.4; E 9.8;` on a nutkWallFunction patch are stored (kappa before `type` too)");
    }

    fs::remove_all(base);
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
