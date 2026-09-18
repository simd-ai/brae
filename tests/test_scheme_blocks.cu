// fvSchemes entries must be resolved WITHIN their own sub-dictionary (group C: C1-C4).
//
// The parser used to sniff keywords across a flat token stream: any statement containing "Gauss" was a
// candidate laplacian/snGrad entry, and the four energy-related div keys all wrote the same flags. Both
// are silent scheme substitutions -- the run converges to a plausible answer computed with a
// discretisation the case never asked for.
//
// The trigger is not exotic. aerofoilNACA0012, a stock OF v2412 tutorial, uses the standard macro idiom:
//
//     gradSchemes { limited  cellLimited Gauss linear 1;  grad(U) $limited;  grad(k) $limited; ... }
//     divSchemes  { div(phi,U)  bounded Gauss linearUpwind limited;  ... }
//
// where "limited" NAMES a gradient scheme. Both statements matched hasWord(ln, "limited") and set
// ctl.nonOrth, so a case whose laplacianSchemes AND snGradSchemes both say `orthogonal` ran WITH
// non-orthogonal correction. Measured on a laplacian-orthogonal case: nonOrth 0 -> 1.
//
// Each case below is paired with its negative control, because "the flag is off" is only evidence if the
// same parser turns it ON for the input that genuinely asks for it.

#include "scheme_parse.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& got, const std::string& want)
{
    if (ok) return;
    std::printf("  FAIL %s: got %s, want %s\n", what, got.c_str(), want.c_str());
    failures++;
}

void checkFlag(const char* what, bool got, bool want)
{
    check(what, got == want, got ? "true" : "false", want ? "true" : "false");
}

void checkNum(const char* what, scalar got, scalar want)
{
    check(what, std::fabs(got - want) < 1e-12, std::to_string((double)got), std::to_string((double)want));
}

// Write an fvSchemes into a throwaway case and parse it.
DeviceSimpleControls parse(const std::string& dir, const std::string& body)
{
    std::filesystem::create_directories(dir + "/system");
    std::ofstream(dir + "/system/fvSchemes") << body;
    DeviceSimpleControls ctl;
    parseFvSchemesControls(dir, ctl);
    return ctl;
}

const char* LAP_ORTHOGONAL =
    "laplacianSchemes { default Gauss linear orthogonal; }\n"
    "snGradSchemes    { default orthogonal; }\n"
    "interpolationSchemes { default linear; }\n";

const char* LAP_CORRECTED =
    "laplacianSchemes { default Gauss linear corrected; }\n"
    "snGradSchemes    { default corrected; }\n"
    "interpolationSchemes { default linear; }\n";

}   // namespace

int main()
{
    const std::string tmp = "/tmp/brae_scheme_blocks";
    std::filesystem::remove_all(tmp);

    // ---- C3: a gradScheme NAMED "limited", referenced from a div line, must not touch the laplacian ----
    {
        const std::string body =
            std::string("ddtSchemes { default steadyState; }\n")
            + "gradSchemes { default Gauss linear;\n"
              "              limited cellLimited Gauss linear 1;\n"
              "              grad(U) cellLimited Gauss linear 1; }\n"
              "divSchemes  { default none;\n"
              "              div(phi,U) bounded Gauss linearUpwind limited; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls c = parse(tmp + "/c3_orthogonal", body);
        checkFlag("C3 nonOrth off when laplacian+snGrad are orthogonal", c.nonOrth, false);
        // Negative control: the SAME div/grad lines with a corrected laplacian must still set it, so the
        // fix cannot be "nonOrth is never set from anywhere".
        const DeviceSimpleControls d = parse(tmp + "/c3_corrected",
            body.substr(0, body.size() - std::string(LAP_ORTHOGONAL).size()) + LAP_CORRECTED);
        checkFlag("C3 nonOrth ON when the laplacian IS corrected", d.nonOrth, true);
        // And the grad limiter must still be read from the gradSchemes line it belongs to.
        checkNum("C3 grad(U) cellLimited still parsed", c.gradULimitK, 1.0);
    }

    // ---- C3b: `limited 0.33` inside laplacianSchemes is the REAL limited-snGrad and must be honoured ----
    {
        const std::string body =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes  { default none; div(phi,U) bounded Gauss upwind; }\n"
              "laplacianSchemes { default Gauss linear limited corrected 0.33; }\n"
              "snGradSchemes    { default limited corrected 0.33; }\n"
              "interpolationSchemes { default linear; }\n";
        const DeviceSimpleControls c = parse(tmp + "/c3b", body);
        checkFlag("C3b limited snGrad sets nonOrth", c.nonOrth, true);
        checkNum("C3b limited coefficient", c.nonOrthLimit, 0.33);
    }

    // ---- C4: div(phi,e) and div(phi,K) are separate entries and must not share a slot ----
    {
        const std::string body =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none;\n"
              "             div(phi,U)   bounded Gauss upwind;\n"
              "             div(phi,e)   bounded Gauss linearUpwind limited;\n"
              "             div(phi,K)   bounded Gauss upwind;\n"
              "             div(phi,Ekp) bounded Gauss upwind; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls c = parse(tmp + "/c4_split", body);
        checkFlag("C4 energy honours linearUpwind", c.luHe, true);
        checkFlag("C4 kinetic term stays upwind", c.luKin, false);
        checkFlag("C4 kinetic entry was seen", c.foundKinScheme, true);
        // Negative control: the reverse assignment must flip both, so the test cannot pass on a parser
        // that simply hardcodes luKin = false.
        const std::string swapped =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none;\n"
              "             div(phi,U)   bounded Gauss upwind;\n"
              "             div(phi,e)   bounded Gauss upwind;\n"
              "             div(phi,K)   bounded Gauss linearUpwind limited;\n"
              "             div(phi,Ekp) bounded Gauss linearUpwind limited; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls d = parse(tmp + "/c4_swapped", swapped);
        checkFlag("C4 swapped: energy upwind", d.luHe, false);
        checkFlag("C4 swapped: kinetic linearUpwind", d.luKin, true);
        // No explicit K/Ekp entry -> inherit the energy scheme, and SAY so via foundKinScheme.
        const std::string none =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none;\n"
              "             div(phi,U) bounded Gauss upwind;\n"
              "             div(phi,e) bounded Gauss linearUpwind limited; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls e = parse(tmp + "/c4_none", none);
        checkFlag("C4 absent kinetic entry inherits energy", e.luKin, true);
        checkFlag("C4 absent kinetic entry is flagged", e.foundKinScheme, false);
    }

    // ---- C2: cellLimited on the TURBULENCE and ENERGY gradients, not just grad(U) ----
    {
        const std::string body =
            std::string("gradSchemes { default Gauss linear;\n")
            + "              grad(U)     cellLimited Gauss linear 1;\n"
              "              grad(k)     cellLimited Gauss linear 0.5;\n"
              "              grad(omega) cellLimited Gauss linear 0.5; }\n"
              "divSchemes { default none; div(phi,U) bounded Gauss upwind; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls c = parse(tmp + "/c2", body);
        checkNum("C2 grad(U) limiter", c.gradULimitK, 1.0);
        checkNum("C2 grad(k)/grad(omega) limiter", c.gradKLimitK, 0.5);
        checkNum("C2 grad(h|e) unlisted stays unlimited", c.gradHeLimitK, 0.0);
        // Negative control: with no cellLimited anywhere, every limiter must be 0.
        const std::string plain =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none; div(phi,U) bounded Gauss upwind; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls d = parse(tmp + "/c2_plain", plain);
        checkNum("C2 negative control grad(U)", d.gradULimitK, 0.0);
        checkNum("C2 negative control grad(k)", d.gradKLimitK, 0.0);
    }

    // ---- C1: interpolationSchemes was never parsed; a non-linear default must be refused ----
    {
        const std::string body =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none; div(phi,U) bounded Gauss upwind; }\n"
              "laplacianSchemes { default Gauss linear orthogonal; }\n"
              "snGradSchemes    { default orthogonal; }\n"
              "interpolationSchemes { default midPoint; }\n";
        bool threw = false;
        try { parse(tmp + "/c1", body); }
        catch (const std::exception&) { threw = true; }
        checkFlag("C1 non-linear interpolation default is refused", threw, true);
        // Negative control: `linear` (OF's default, and what brae actually does) must still be accepted.
        bool threwLinear = false;
        try
        {
            parse(tmp + "/c1_linear",
                  std::string("gradSchemes { default Gauss linear; }\n")
                  + "divSchemes { default none; div(phi,U) bounded Gauss upwind; }\n"
                  + LAP_ORTHOGONAL);
        }
        catch (const std::exception&) { threwLinear = true; }
        checkFlag("C1 linear interpolation is accepted", threwLinear, false);
    }

    // ---- D: div(phi,U) resolved through the divSchemes `default` -------------------------------
    //
    // OF looks the key up and falls back to `default`; `default none` means there is no fallback and OF
    // itself throws. brae refused every case without an explicit div(phi,U), which stopped
    // laminar/cylinder2D -- `default Gauss linear;` and nothing else -- over a scheme brae implements.
    // The pair below is the whole rule: resolve when there is a default, refuse when it is `none`.
    {
        const DeviceSimpleControls ctl = parse(tmp + "/d1",
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default Gauss linear; div((nuEff*dev(T(grad(U))))) Gauss linear; }\n"
            + LAP_CORRECTED);
        checkFlag("D1 `default Gauss linear` resolves div(phi,U) to central differencing", ctl.divULinear, true);
    }
    {
        // ...and the resolved scheme is the DEFAULT's, not a guess: a limitedLinearV default must land on
        // the limitedLinearV path, with its coefficient.
        const DeviceSimpleControls ctl = parse(tmp + "/d2",
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default Gauss limitedLinearV 1; }\n"
            + LAP_CORRECTED);
        checkFlag("D2 a limitedLinearV default resolves to limitedLinearV", ctl.divULimitedV, true);
        checkNum("D2 twoByk from the default line", ctl.divUTwoBykV, 2.0);
        checkFlag("D2 ...and not to plain linear", ctl.divULinear, false);
    }
    {
        // An EXPLICIT div(phi,U) still wins over the default (dictionary lookup order).
        const DeviceSimpleControls ctl = parse(tmp + "/d3",
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default Gauss linear; div(phi,U) bounded Gauss upwind; }\n"
            + LAP_CORRECTED);
        checkFlag("D3 an explicit div(phi,U) overrides the default", ctl.divULinear, false);
        checkFlag("D3 ...and its own `bounded` is read", ctl.bounded, true);
    }
    {
        // Negative control 1: `default none` has nothing to resolve to -- still a refusal, as in OF.
        // The MESSAGE is asserted, not just the throw: feeding the literal word `none` through as if it
        // were a scheme also fails, but with "unknown scheme ''", which sends the reader looking for a
        // typo in a line they never wrote.
        bool threw = false;
        std::string msg;
        try
        {
            parse(tmp + "/d4",
                  std::string("gradSchemes { default Gauss linear; }\n")
                  + "divSchemes { default none; div(phi,k) Gauss upwind; }\n"
                  + LAP_CORRECTED);
        }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        checkFlag("D4 `default none` with no div(phi,U) is refused", threw, true);
        check("D4 the refusal names the `none` default", msg.find("`none`") != std::string::npos, msg, "mentions `none`");
    }
    {
        // Negative control 2: no default at all is the same refusal.
        bool threw = false;
        try
        {
            parse(tmp + "/d5",
                  std::string("gradSchemes { default Gauss linear; }\n")
                  + "divSchemes { div(phi,k) Gauss upwind; }\n"
                  + LAP_CORRECTED);
        }
        catch (const std::exception&) { threw = true; }
        checkFlag("D5 no divSchemes default and no div(phi,U) is refused", threw, true);
    }

    // ---- interFoam's alpha transport -----------------------------------------------------------
    // The div(phi,sigma) defect this file already documents was a limiter that was PLUMBED but never
    // SELECTED: the flag existed, defaulted false, and nothing assigned it, so the case's vanAlbada ran
    // unlimited. These arms exist so div(phi,alpha) cannot repeat it -- the parse is asserted, not the
    // presence of a field.
    {
        const std::string base = "/tmp/brae_scheme_blocks_alpha";
        std::filesystem::remove_all(base);
        const std::string TAIL = std::string(LAP_ORTHOGONAL) + "gradSchemes { default Gauss linear; }\n"
                                 "ddtSchemes { default Euler; }\n";

        // Every interFoam tutorial, verbatim: vanLeer on the transport, linear on the compression flux.
        {
            const DeviceSimpleControls c = parse(base + "/vanleer",
                "divSchemes { default none; div(phi,U) Gauss upwind;\n"
                "             div(phi,alpha) Gauss vanLeer;\n"
                "             div(phirb,alpha) Gauss linear; }\n" + TAIL);
            // BOTH halves, and the found-flag is the load-bearing one: the DEFAULT is already vanLeer,
            // so asserting the value alone passes even if the parser never selected anything.
            checkFlag("div(phi,alpha) Gauss vanLeer was actually SELECTED", c.foundDivAlpha, true);
            checkNum ("div(phi,alpha) Gauss vanLeer -> kVanLeerTwoByk", c.divAlphaTwoByk, scalar(-1.0));
            checkFlag("div(phirb,alpha) Gauss linear was SELECTED", c.foundDivAlphaRb, true);
            checkFlag("div(phirb,alpha) Gauss linear", c.divAlphaRbLinear, true);
        }
        // The other two limiters brae has, so the selector is read rather than defaulted.
        {
            const DeviceSimpleControls c = parse(base + "/valbada",
                "divSchemes { default none; div(phi,U) Gauss upwind;\n"
                "             div(phi,alpha) Gauss vanAlbada; }\n" + TAIL);
            checkFlag("vanAlbada selected", c.foundDivAlpha, true);
            checkNum ("div(phi,alpha) Gauss vanAlbada -> 0", c.divAlphaTwoByk, scalar(0.0));
        }
        {
            const DeviceSimpleControls c = parse(base + "/limlin",
                "divSchemes { default none; div(phi,U) Gauss upwind;\n"
                "             div(phi,alpha) Gauss limitedLinear 1; }\n" + TAIL);
            checkFlag("limitedLinear selected", c.foundDivAlpha, true);
            checkNum ("div(phi,alpha) Gauss limitedLinear 1 -> twoByk 2", c.divAlphaTwoByk, scalar(2.0));
        }
        // REFUSALS. An unimplemented limiter on the VoF transport moves the interface, so it is named.
        {
            bool threw = false;
            try {
                parse(base + "/bad", "divSchemes { default none; div(phi,U) Gauss upwind;\n"
                                     "             div(phi,alpha) Gauss MUSCL; }\n" + TAIL);
            } catch (const std::exception&) { threw = true; }
            checkFlag("div(phi,alpha) Gauss MUSCL is refused by name", threw, true);
        }
        {
            bool threw = false;
            try {
                parse(base + "/badrb", "divSchemes { default none; div(phi,U) Gauss upwind;\n"
                                       "             div(phirb,alpha) Gauss vanLeer; }\n" + TAIL);
            } catch (const std::exception&) { threw = true; }
            checkFlag("div(phirb,alpha) Gauss vanLeer is refused (compression flux is linear)", threw, true);
        }
        // CONTROL: a case with no alpha entry at all must not be refused, and keeps the defaults.
        {
            const DeviceSimpleControls c = parse(base + "/none",
                "divSchemes { default none; div(phi,U) Gauss upwind; }\n" + TAIL);
            checkFlag("no div(phi,alpha) entry -> NOT marked as found", c.foundDivAlpha, false);
            checkFlag("no div(phirb,alpha) entry -> NOT marked as found", c.foundDivAlphaRb, false);
        }
    }

    std::printf("scheme_blocks: %d failures\n", failures);
    return failures ? 1 : 0;
}
