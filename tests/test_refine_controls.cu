// The dynamicRefineFvMeshCoeffs reader, against the three shipped AMR tutorials' own dictionaries.
//
// THE ORACLE IS THE DICTIONARY. Every value asserted below is written in the file, so this gate cannot
// drift from OpenFOAM: if a tutorial changes, the assertion fails and says which entry moved.
//
// ALL THREE ARE NEEDED, because they are written three different ways and no one of them exercises the
// lookup (`optionalSubDict`, dynamicRefineFvMesh.C:181 and :1292):
//   damBreakWithObstacle  the entries FLAT at the top level
//   oscillatingBox        FLAT, beside a `solvers { VF { ... } }` sub-dict for its motion solver --
//                         so a reader that took "the first sub-dictionary" would read the motion solver
//   motorBike             WRAPPED in `dynamicRefineFvMeshCoeffs`, and with NO `unrefineLevel`, which is
//                         the only entry OpenFOAM defaults (to GREAT = 1e+15). Reading 0 there would
//                         unrefine the whole mesh on the first step.
//
// THE REFUSALS are OpenFOAM's own three FatalErrors plus the missing-mandatory-entry case, each
// exercised by deleting or corrupting one entry of a real dictionary.
#include "dynamic_refine_fv_mesh_cpp.cuh"
#include "foam_dict.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::dynamicRefine;

namespace {
int failures = 0;

void check(
    const char* what,
    bool        ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

bool threwWith(
    const std::string& dictText,
    const std::string& needle,
    std::string&       got)
{
    const std::string tmp = "/tmp/brae_refine_controls_probe.dict";
    { std::ofstream f(tmp); f << dictText; }
    try
    {
        readRefineControls(readDict(tmp));
        got = "(it did not throw)";
        return false;
    }
    catch (const std::exception& e)
    {
        got = e.what();
        return got.find(needle) != std::string::npos;
    }
}

std::string slurp(
    const std::string& path)
{
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae's dynamicRefineFvMeshCoeffs reader vs the tutorials' own dictionaries ==\n");
    if (argc < 2)
    {
        std::printf("  SKIP: usage: %s <tutorialsRoot>\n", argv[0]);
        return 77;
    }
    const std::string T = std::string(argv[1]) + "/multiphase/interFoam";

    // ---- 1. FLAT: laminar/damBreakWithObstacle -----------------------------------------------------
    {
        const RefineControls c = readRefineControls(readDict(T + "/laminar/damBreakWithObstacle/constant/dynamicMeshDict"));
        check("damBreakWithObstacle: refineInterval 1", c.refineInterval == 1);
        check("damBreakWithObstacle: field alpha.water", c.field == "alpha.water");
        check("damBreakWithObstacle: lowerRefineLevel 0.001", c.lowerRefineLevel == scalar(0.001));
        check("damBreakWithObstacle: upperRefineLevel 0.999", c.upperRefineLevel == scalar(0.999));
        check("damBreakWithObstacle: unrefineLevel 10, as the file gives it", c.unrefineLevel == scalar(10));
        check("damBreakWithObstacle: nBufferLayers 1", c.nBufferLayers == 1);
        check("damBreakWithObstacle: maxRefinement 2", c.maxRefinement == 2);
        check("damBreakWithObstacle: maxCells 200000", c.maxCells == 200000);
        check("damBreakWithObstacle: dumpLevel true", c.dumpLevel);
        check("damBreakWithObstacle: six correctFluxes pairs", c.correctFluxes.size() == 6);
        bool phiNone = false;
        for (const auto& pr : c.correctFluxes) if (pr.first == "phi" && pr.second == "none") phiNone = true;
        check("damBreakWithObstacle: (phi none) is one of them", phiNone);
    }

    // ---- 2. FLAT BESIDE A SUB-DICT: laminar/oscillatingBox ------------------------------------------
    // Its `solvers { VF { ... } }` is a sub-dictionary that is NOT the coeffs dict. A reader that took
    // any sub-dictionary would read the motion solver and find none of these entries.
    {
        const RefineControls c = readRefineControls(readDict(T + "/laminar/oscillatingBox/constant/dynamicMeshDict"));
        check("oscillatingBox: read past the `solvers` sub-dict to the flat entries", c.refineInterval == 1);
        check("oscillatingBox: field alpha.water", c.field == "alpha.water");
        check("oscillatingBox: maxRefinement 2", c.maxRefinement == 2);
        check("oscillatingBox: maxCells 200000", c.maxCells == 200000);
        check("oscillatingBox: unrefineLevel 10", c.unrefineLevel == scalar(10));
    }

    // ---- 3. WRAPPED, AND NO unrefineLevel: RAS/motorBike -------------------------------------------
    {
        const RefineControls c = readRefineControls(readDict(T + "/RAS/motorBike/constant/dynamicMeshDict"));
        check("motorBike: read from the dynamicRefineFvMeshCoeffs sub-dict", c.refineInterval == 1);
        check("motorBike: maxRefinement 4", c.maxRefinement == 4);
        check("motorBike: maxCells 2000000", c.maxCells == 2000000);
        check("motorBike: seven correctFluxes pairs", c.correctFluxes.size() == 7);
        // THE ONE DEFAULT, and the reason it matters: the file has no unrefineLevel, and 0 would
        // unrefine everything. OpenFOAM's getOrDefault gives GREAT (doubleScalar.H:58).
        check("motorBike: unrefineLevel DEFAULTS to GREAT, the file having none",
              c.unrefineLevel == scalar(1.0e+15));
    }

    // ---- 4. the refusals ---------------------------------------------------------------------------
    {
        const std::string base = slurp(T + "/laminar/damBreakWithObstacle/constant/dynamicMeshDict");
        std::string got;

        auto sub = [&](const std::string& from, const std::string& to)
        {
            std::string s = base;
            const std::size_t k = s.find(from);
            if (k != std::string::npos) s.replace(k, from.size(), to);
            return s;
        };

        check("a negative refineInterval is refused, naming it",
              threwWith(sub("refineInterval  1;", "refineInterval  -1;"), "refineInterval", got));
        check("maxCells 0 is refused, naming it",
              threwWith(sub("maxCells        200000;", "maxCells        0;"), "maximum number of cells", got));
        check("maxRefinement 0 is refused, naming it",
              threwWith(sub("maxRefinement   2;", "maxRefinement   0;"), "maximum refinement level", got));
        check("a missing mandatory entry is refused, naming it",
              threwWith(sub("nBufferLayers   1;", ""), "nBufferLayers", got));
        check("a missing correctFluxes is refused, naming it",
              threwWith(sub("correctFluxes", "correctFluxesXX"), "correctFluxes", got));
        check("a missing dumpLevel is refused, naming it",
              threwWith(sub("dumpLevel       true;", ""), "dumpLevel", got));
        // THE ORDER OpenFOAM STOPS IN, and its words: the constructor's readDict() reads correctFluxes, then
        // dumpLevel (dynamicRefineFvMesh.C:184, :193); refineInterval comes at the first update() (:1295).
        // A bare `dynamicFvMesh dynamicRefineFvMesh;` -- tests/interfoam_refusals.sh's mesh_dynamicRefine
        // arms -- is what real OpenFOAM v2412 stops on with "Entry 'correctFluxes' not found in dictionary".
        // brae read refineInterval first, so the same dictionary named a different entry.
        {
            const std::string bare =
                "FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\n"
                "dynamicFvMesh dynamicRefineFvMesh;\n";
            check("a bare dynamicRefineFvMesh stops on correctFluxes, in OpenFOAM's words",
                  threwWith(bare, "Entry 'correctFluxes' not found in dictionary", got));
            check("...with correctFluxes given it stops on dumpLevel, not refineInterval",
                  threwWith(bare + "correctFluxes ((phi none));\n", "Entry 'dumpLevel' not found", got));
            check("...and with both, on refineInterval",
                  threwWith(bare + "correctFluxes ((phi none));\ndumpLevel true;\n",
                            "Entry 'refineInterval' not found", got));
        }
        // dumpLevel is a bool readEntry: a token that is not a Switch word stops, not reads as false
        check("a dumpLevel that is not a Switch word is refused",
              threwWith(sub("dumpLevel       true;", "dumpLevel       maybe;"), "maybe", got));
        // ...and refineInterval 0 is NOT an error: it means "never refine" (dynamicRefineFvMesh.C:1299)
        {
            const std::string tmp = "/tmp/brae_refine_controls_zero.dict";
            { std::ofstream f(tmp); f << sub("refineInterval  1;", "refineInterval  0;"); }
            bool ok = false;
            try { ok = (readRefineControls(readDict(tmp)).refineInterval == 0); } catch (...) { ok = false; }
            check("refineInterval 0 is accepted -- it means `never refine`, not an error", ok);
        }
    }

    std::printf("test_refine_controls: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
