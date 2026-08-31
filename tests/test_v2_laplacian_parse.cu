// simpleFoamV2's laplacianSchemes parse, against limitedSnGrad's own algebra.
//
//     limiter = min( k*|orth| / ((1 - k)*|corr| + SMALL), 1 )
//
// so `limited 1` is exactly `corrected` and `limited 0` is exactly `uncorrected` -- the parse's doc
// comment has said so from the start, while the code mapped `limited 0` onto its OWN sentinel
// (limitCoeff = 0 = "no limiter") and ran the FULL correction under a name that asked for none. The
// parse was file-local, so nothing could call it; it is hoisted precisely for this test.
#include "simpleFoamV2.cuh"
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

static gpu::LaplacianScheme parse(const std::string& entry)
{
    static int n = 0;
    const std::string dir = "/tmp/brae_v2_laplacian_parse/" + std::to_string(n++);
    std::filesystem::create_directories(dir + "/system");
    std::ofstream f(dir + "/system/fvSchemes");
    f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
      << "laplacianSchemes { default " << entry << "; }\n";
    f.close();
    return gpu::laplacianScheme(dir);
}

int main()
{
    std::printf("V2 laplacianSchemes parse vs limitedSnGrad's algebra\n");
    std::filesystem::remove_all("/tmp/brae_v2_laplacian_parse");

    { const auto r = parse("Gauss linear corrected");
      check("`corrected` -> corrected, no limiter", r.corrected && r.limitCoeff == 0.0 && r.unsupported.empty()); }
    { const auto r = parse("Gauss linear uncorrected");
      check("`uncorrected` -> not corrected", !r.corrected && r.unsupported.empty()); }
    { const auto r = parse("Gauss linear limited 0.33");
      check("`limited 0.33` -> corrected with the limiter",
            r.corrected && r.limitCoeff > 0.32 && r.limitCoeff < 0.34 && r.unsupported.empty()); }
    { const auto r = parse("Gauss linear limited corrected 0.33");   // turbineSiting's spelling
      check("`limited corrected 0.33` (coefficient after the word) parses too",
            r.corrected && r.limitCoeff > 0.32 && r.limitCoeff < 0.34); }
    { const auto r = parse("Gauss linear limited 1");
      check("`limited 1` -> plain corrected (the limiter never binds)",
            r.corrected && r.limitCoeff == 0.0); }
    // THE FIX: k = 0 makes the limiter identically zero -- the correction is fully suppressed.
    // The old parse returned corrected=true with limitCoeff=0 (its no-limiter sentinel): the FULL
    // correction where the case asked for NONE.
    { const auto r = parse("Gauss linear limited 0");
      check("`limited 0` -> UNCORRECTED (limiter identically zero)",
            !r.corrected && r.unsupported.empty()); }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
