// The polyMesh boundary parser must skip an unsupported SUB-DICTIONARY, and must refuse a time-varying
// cyclicACMI scale rather than run past it.
//
// The catch-all for unknown keys was `while (peek() != ";") next();`, which assumes every value is a
// simple one. cyclicACMI can carry
//     scale           table;
//     scaleCoeffs     { values 3((0 1)(0.2 1)(0.3 0)); }
// and there the scan stops at the ';' INSIDE the block; the block's own '}' is then taken for the
// patch's, every later patch is off by one, and the parse dies on a patch name far away. That is exactly
// how pimpleFoam/RAS/TJunctionSwitching failed -- "expected '{' got 'bottom_central_blockage'", which
// names a patch two entries later and points nowhere near the cause.
//
// Leg 2 is the more important half. Once the block parses, `scale` must not be silently dropped: it makes
// the interface's open area a prescribed function of time (TJunctionSwitching uses it to close a branch),
// so ignoring it solves a different problem and converges happily to it. It used to be REFUSED for that
// reason; now that brae evaluates it, the leg asserts the parsed table instead -- same requirement, one
// step further on.
#include "primitive_mesh.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>

using namespace brae;

namespace {
int failures = 0;

void writeBoundary(const std::string& dir, const char* extra, const char* type)
{
    std::ofstream o(dir + "/boundary");
    o << "FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }\n"
      << "2\n(\n"
      << "    inlet\n    {\n        type            patch;\n"
      << "        nFaces          5;\n        startFace       0;\n    }\n"
      << "    couple\n    {\n        type            " << type << ";\n"
      << "        inGroups        1(" << type << ");\n"
      << "        nFaces          7;\n        startFace       5;\n"
      << "        neighbourPatch  other;\n"
      << "        nonOverlapPatch couple_blockage;\n"
      << extra
      << "    }\n)\n";
}
}   // namespace

int main()
{
    const char* tmp = std::getenv("TMPDIR");
    const std::string dir = std::string(tmp ? tmp : "/tmp") + "/brae_bnd_subdict";
    std::system(("rm -rf " + dir + " && mkdir -p " + dir).c_str());

    // the block that used to derail the parser, written exactly as OpenFOAM writes it
    const char* scaleBlock =
        "        scale           table;\n"
        "        scaleCoeffs\n        {\n            values          \n3\n(\n(0 1)\n(0.2 1)\n(0.3 0)\n)\n;\n        }\n";

    // ---- 1. a sub-dictionary on a NON-ACMI patch parses, and the patch after it survives ----
    {
        writeBoundary(dir, scaleBlock, "cyclic");
        PrimitiveMesh m;
        bool threw = false;
        try { m.readBoundary(dir); } catch (const std::exception& e)
        { threw = true; std::printf("  FAIL parse threw: %s\n", e.what()); ++failures; }
        if (!threw)
        {
            std::printf("  sub-dictionary skipped: %zu patches read\n", m.patches().size());
            if (m.patches().size() != 2)
            { std::printf("  FAIL expected 2 patches, got %zu -- the block desynchronised the parse\n",
                          m.patches().size()); ++failures; }
            else if (m.patches()[1].name != "couple" || m.patches()[1].size != 7 || m.patches()[1].start != 5)
            {
                std::printf("  FAIL the patch AFTER the sub-dictionary is wrong: name '%s' nFaces %d startFace %d\n",
                            m.patches()[1].name.c_str(), (int)m.patches()[1].size, (int)m.patches()[1].start);
                ++failures;
            }
            else if (m.patches()[1].nonOverlapPatch != "couple_blockage")
            { std::printf("  FAIL nonOverlapPatch lost: '%s'\n", m.patches()[1].nonOverlapPatch.c_str()); ++failures; }
        }
    }

    // ---- 2. the same block on a cyclicACMI must be READ, table and all ----
    {
        writeBoundary(dir, scaleBlock, "cyclicACMI");
        PrimitiveMesh m;
        try { m.readBoundary(dir); }
        catch (const std::exception& e)
        { std::printf("  FAIL cyclicACMI + scale threw: %s\n", e.what()); ++failures; }
        if (m.patches().size() == 2)
        {
            const Function1& f = m.patches()[1].acmiScale;
            if (f.empty())
            {
                std::printf("  FAIL the scale table was dropped. The interface's open area is then a\n"
                            "       constant where the case says it opens and closes, and the run converges\n"
                            "       to the wrong problem without saying so.\n");
                ++failures;
            }
            else
            {
                // (0 1)(0.2 1)(0.3 0): flat, then a ramp to zero, then clamped below
                const bool ok = std::fabs(f.value(0.0)  - 1.0) < 1e-12
                             && std::fabs(f.value(0.2)  - 1.0) < 1e-12
                             && std::fabs(f.value(0.25) - 0.5) < 1e-12
                             && std::fabs(f.value(0.3)  - 0.0) < 1e-12
                             && std::fabs(f.value(9.9)  - 0.0) < 1e-12;
                std::printf("  cyclicACMI + scale: read (1 -> %.3f at t=0.25 -> 0)\n", (double)f.value(0.25));
                if (!ok)
                { std::printf("  FAIL the table does not interpolate as OpenFOAM's does\n"); ++failures; }
            }
        }
    }

    // ---- 2b. THE DICTIONARY FORM: `scale { type coded; ... }` is refused BY NAME, and the same form
    // holding a table is read. OpenFOAM reads `scale` as a PatchFunction1 (cyclicACMIPolyPatch.C:625);
    // RAS/damBreakLeakage writes a coded one, and the parser died there on "expected ';' got 'type'".
    {
        writeBoundary(dir,
            "        scale\n        {\n            type            coded;\n"
            "            name            leak;\n            sub { a 1; }\n        }\n",
            "cyclicACMI");
        PrimitiveMesh m;
        bool named = false;
        try
        {
            m.readBoundary(dir);
        }
        catch (const std::exception& e)
        {
            named = std::string(e.what()).find("`scale coded`") != std::string::npos;
            std::printf("  scale { type coded; }: %s\n", e.what());
        }
        if (!named)
        {
            std::printf("  FAIL a coded scale block is not refused under its own name\n");
            ++failures;
        }
        writeBoundary(dir,
            "        scale\n        {\n            type            table;\n"
            "            values          3((0 1)(0.2 1)(0.3 0));\n        }\n",
            "cyclicACMI");
        PrimitiveMesh m2;
        try
        {
            m2.readBoundary(dir);
            const bool ok = m2.patches().size() == 2
                         && std::fabs(m2.patches()[1].acmiScale.value(0.25) - 0.5) < 1e-12;
            if (!ok)
            {
                std::printf("  FAIL scale { type table; values ...; } does not read as the inline table does\n");
                ++failures;
            }
        }
        catch (const std::exception& e)
        {
            std::printf("  FAIL scale { type table; } threw (%s)\n", e.what());
            ++failures;
        }
    }

    // ---- 3. VACUITY GUARD: without the block, the same cyclicACMI patch must parse fine ----
    {
        writeBoundary(dir, "", "cyclicACMI");
        PrimitiveMesh m;
        try
        {
            m.readBoundary(dir);
            std::printf("  control: the same patch without the scale block parses, %zu patches\n", m.patches().size());
            if (m.patches().size() != 2)
            { std::printf("  FAIL control parse is wrong too, so leg 2 proves nothing\n"); ++failures; }
        }
        catch (const std::exception& e)
        {
            std::printf("  FAIL control threw (%s) -- leg 2's refusal cannot be attributed to `scale`\n", e.what());
            ++failures;
        }
    }

    std::printf("boundary_subdict_skip: %d failures\n", failures);
    return failures ? 1 : 0;
}
