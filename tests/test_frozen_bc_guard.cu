// refuseFrozenPerStepBC: the guard for boundaries that need a per-step driver update.
//
// The factory accepts fixedMean, fanPressure and the coded pair on the strength of a per-step update
// only some drivers perform. Drivers that do not (simpleFoamV2, the mirror createFields, gpuSimpleFoam
// for the non-coded pair) call this at their read sites. Parse level on purpose: the claim is about
// what the guard refuses and what it lets through, and that is deterministic.
#include "frozen_bc_guard.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

std::string writeU(const std::string& dir, const std::string& body)
{
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/U";
    std::ofstream f(path);
    f << "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
      << "dimensions [0 1 -1 0 0 0 0];\n"
      << "internalField uniform (0 0 0);\n"
      << "boundaryField\n{\n" << body << "\n}\n";
    return path;
}

// One guard call against one U file; reports the throw and its message.
struct R { bool threw = false; std::string msg; };
R run(const std::string& body, bool codedMaintained)
{
    static int n = 0;
    const FieldData<vector> fd = readField<vector>(
        writeU("/tmp/brae_frozen_bc_guard/" + std::to_string(n++), body));
    R r;
    try { refuseFrozenPerStepBC(fd, "U", "testDriver", codedMaintained); }
    catch (const std::exception& e) { r.threw = true; r.msg = e.what(); }
    return r;
}

}   // namespace

int main()
{
    std::printf("frozen_bc_guard: per-step BCs refused where no driver maintains them\n");
    std::filesystem::remove_all("/tmp/brae_frozen_bc_guard");

    {
        const R r = run("    inlet { type fixedMean; meanValue uniform (10 0 0); value uniform (10 0 0); }", false);
        check("fixedMean is refused", r.threw);
        check("...naming the driver, the patch and the type",
              r.msg.find("testDriver") != std::string::npos
           && r.msg.find("inlet") != std::string::npos
           && r.msg.find("fixedMean") != std::string::npos);
    }
    {
        const R r = run("    outlet { type fanPressure; value uniform (0 0 0); }", false);
        check("fanPressure is refused", r.threw);
    }
    {
        const R off = run("    inlet { type codedFixedValue; value uniform (1 0 0); name swirl; }", false);
        const R on  = run("    inlet { type codedFixedValue; value uniform (1 0 0); name swirl; }", true);
        check("codedFixedValue refused where coded is NOT maintained", off.threw);
        check("...and ACCEPTED where it is (gpuSimpleFoam wires NVRTC)", !on.threw);
    }
    {
        const R off = run("    wall { type codedMixed; refValue uniform (0 0 0); value uniform (0 0 0); }", false);
        check("codedMixed refused where coded is NOT maintained", off.threw);
    }
    // NEGATIVE CONTROL: an ordinary boundary set must pass, or this refuses every case in the tree.
    {
        const R r = run("    inlet { type fixedValue; value uniform (44.2 0 0); }\n"
                        "    outlet { type zeroGradient; }\n"
                        "    wall { type noSlip; }", false);
        check("an ordinary fixedValue/zeroGradient/noSlip set passes (negative control)", !r.threw);
    }
    // fixedMean stays refused even where coded is maintained: the coded flag must not widen past coded.
    {
        const R r = run("    inlet { type fixedMean; meanValue uniform (10 0 0); value uniform (10 0 0); }", true);
        check("fixedMean refused even with codedMaintained=true", r.threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
