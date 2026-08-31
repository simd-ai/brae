// uniformFixedValue with a NON-CONSTANT uniformValue must be refused, not silently degraded.
//
// OF's uniformFixedValue takes a Function1: `constant`, `table`, `polynomial`, `expression`, `coded`, ...
// brae evaluates only the constant forms. The reader used to skip the others and rely on "dispatch throws
// when there is no value" -- which does not hold, because a case that OVERRIDES an earlier entry leaves a
// stale `value` behind:
//
//     coldWall { type fixedValue; value uniform 350; type uniformFixedValue;
//                uniformValue { type expression; expression "..."; } }
//
// hasValue is still true, so the expression silently became the constant 350. squareBendLiq's T walls are
// exactly this shape. A wrong wall temperature converges perfectly happily.
//
// Parse level on purpose: the claim is about what the READER records and what construction does with it,
// and that is deterministic -- no solve, no tolerance, no GPU.

#include "foam_field_reader.cuh"
#include "fv_patch_field.cuh"
#include "fv_patch.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

std::string writeField(const std::string& dir, const std::string& patchBody)
{
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/T";
    std::ofstream f(path);
    f << "FoamFile { version 2.0; format ascii; class volScalarField; object T; }\n"
      << "dimensions [0 0 0 1 0 0 0];\n"
      << "internalField uniform 300;\n"
      << "boundaryField\n{\n" << patchBody << "\n}\n";
    return path;
}

}   // namespace

int main()
{
    const std::string base = "/tmp/brae_uniform_function1";
    std::filesystem::remove_all(base);

    // 1. constant form -> parsed, no refusal marker.
    {
        const FieldData<scalar> fd = readField<scalar>(
            writeField(base + "/const", "    wall { type uniformFixedValue; uniformValue constant 300; }"));
        const PatchFieldData<scalar>& b = fd.boundary.at(0);
        if (b.unsupportedFunction1.empty() && b.hasValue && b.uniformValue == 300.0)
            std::printf("  OK   constant uniformValue parses to 300 and is accepted\n");
        else
        { std::printf("  FAIL constant uniformValue mis-parsed (marker='%s' value=%g)\n",
                      b.unsupportedFunction1.c_str(), (double)b.uniformValue); failures++; }
    }

    // 2. expression form WITH a stale value -> must be marked, and must name the Function1.
    {
        const FieldData<scalar> fd = readField<scalar>(writeField(base + "/expr",
            "    wall { type fixedValue; value uniform 350; type uniformFixedValue;\n"
            "           uniformValue { type expression; expression \"300 + 50\"; } }"));
        const PatchFieldData<scalar>& b = fd.boundary.at(0);
        if (b.unsupportedFunction1 == "expression")
        {
            std::printf("  OK   expression uniformValue is marked (named '%s'), not silently dropped\n",
                        b.unsupportedFunction1.c_str());
        }
        else
        {
            std::printf("  FAIL expression uniformValue left marker '%s'; the stale value %g would be used\n",
                        b.unsupportedFunction1.c_str(), (double)b.uniformValue);
            failures++;
        }

        // ...and construction must refuse it. This is the half that actually protects the user: the
        // marker is useless if makePatchField ignores it.
        FvPatch p;
        p.name = "wall";
        p.type = "wall";
        p.size = 1;
        p.faceCells.assign(1, 0);
        p.deltaCoeffs.assign(1, 1.0);
        p.nf.assign(1, vector{1, 0, 0});
        p.magSf.assign(1, 1.0);
        bool refused = false;
        try { (void)makePatchField<scalar>(p, b); }
        catch (const std::exception& e)
        { refused = std::string(e.what()).find("uniformFixedValue") != std::string::npos; }
        if (refused) std::printf("  OK   construction refuses it by name\n");
        else { std::printf("  FAIL construction accepted a non-constant uniformValue\n"); failures++; }
    }

    // 3. NEGATIVE CONTROL: a plain fixedValue must be untouched, or this would refuse ordinary cases.
    {
        const FieldData<scalar> fd = readField<scalar>(
            writeField(base + "/plain", "    wall { type fixedValue; value uniform 321; }"));
        const PatchFieldData<scalar>& b = fd.boundary.at(0);
        if (b.unsupportedFunction1.empty() && b.uniformValue == 321.0)
            std::printf("  OK   plain fixedValue unaffected\n");
        else { std::printf("  FAIL plain fixedValue disturbed\n"); failures++; }
    }

    // 4. pressureInletOutletVelocity with `tangentialVelocity` must be refused, and without it accepted.
    // OF drives the tangential component (refValue = tv - n*(n & tv),
    // pressureInletOutletVelocityFvPatchVectorField.C:135); brae leaves it at zero, so honouring the case
    // would mean solving a swirl-free inlet where the case asked for swirl. The header said "not
    // supported"; nothing enforced it.
    {
        std::filesystem::create_directories(base + "/tv");
        auto writeU = [&](const std::string& dir, const std::string& body)
        {
            std::filesystem::create_directories(dir);
            const std::string path = dir + "/U";
            std::ofstream f(path);
            f << "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
              << "dimensions [0 1 -1 0 0 0 0];\n"
              << "internalField uniform (0 0 0);\n"
              << "boundaryField\n{\n" << body << "\n}\n";
            return path;
        };
        FvPatch p;
        p.name = "outlet";
        p.type = "patch";
        p.size = 1;
        p.faceCells.assign(1, 0);
        p.deltaCoeffs.assign(1, 1.0);
        p.nf.assign(1, vector{1, 0, 0});
        p.magSf.assign(1, 1.0);

        const FieldData<vector> with = readField<vector>(writeU(base + "/tv/with",
            "    outlet { type pressureInletOutletVelocity; tangentialVelocity uniform (0 5 0);\n"
            "             value uniform (0 0 0); }"));
        bool refused = false;
        try { (void)makePatchField<vector>(p, with.boundary.at(0)); }
        catch (const std::exception& e)
        { refused = std::string(e.what()).find("tangentialVelocity") != std::string::npos; }
        if (refused) std::printf("  OK   pressureInletOutletVelocity + tangentialVelocity refused by name\n");
        else { std::printf("  FAIL tangentialVelocity accepted -- the tangential component would be zeroed\n"); failures++; }

        // NEGATIVE CONTROL: the plain form is the one every simpleFoam tutorial uses; it must still build.
        const FieldData<vector> without = readField<vector>(writeU(base + "/tv/without",
            "    outlet { type pressureInletOutletVelocity; value uniform (0 0 0); }"));
        bool built = true;
        try { (void)makePatchField<vector>(p, without.boundary.at(0)); }
        catch (const std::exception&) { built = false; }
        if (built) std::printf("  OK   plain pressureInletOutletVelocity still accepted\n");
        else { std::printf("  FAIL the plain form was refused too -- that breaks every case using it\n"); failures++; }
    }

    // 5. surfaceNormalFixedValue / uniformNormalFixedValue with a Function1 refValue. OF samples a full
    // PatchFunction1 every updateCoeffs; brae's reader used to SKIP the forms it cannot evaluate without
    // marking them, so the patch built with an empty value array and every face got U_b = 0*n -- a silent
    // zero inlet where the case prescribed a ramp (the class comment even said "any Function1 is
    // ignored"). The reader must mark, and construction must refuse by name.
    {
        auto writeU = [&](const std::string& dir, const std::string& body)
        {
            std::filesystem::create_directories(dir);
            const std::string path = dir + "/U";
            std::ofstream f(path);
            f << "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
              << "dimensions [0 1 -1 0 0 0 0];\n"
              << "internalField uniform (0 0 0);\n"
              << "boundaryField\n{\n" << body << "\n}\n";
            return path;
        };
        FvPatch p;
        p.name = "inlet";
        p.type = "patch";
        p.size = 1;
        p.faceCells.assign(1, 0);
        p.deltaCoeffs.assign(1, 1.0);
        p.nf.assign(1, vector{1, 0, 0});
        p.magSf.assign(1, 1.0);

        // A ramped inflow, straight from the OF tutorials' shape. Must refuse and name the table.
        const FieldData<vector> tab = readField<vector>(writeU(base + "/snf/table",
            "    inlet { type surfaceNormalFixedValue; refValue table ((0 0) (1 -10));\n"
            "            value uniform (0 0 0); }"));
        bool refused = false;
        try { (void)makePatchField<vector>(p, tab.boundary.at(0)); }
        catch (const std::exception& e)
        {
            const std::string w = e.what();
            refused = w.find("surfaceNormalFixedValue") != std::string::npos
                   && w.find("table") != std::string::npos;
        }
        if (refused) std::printf("  OK   surfaceNormalFixedValue + table refValue refused by name\n");
        else { std::printf("  FAIL a table refValue built -- every face would get U_b = 0*n\n"); failures++; }

        // NEGATIVE CONTROL: the constant form must still build, AND still compute refValue*n -- an
        // accepted patch with the wrong arithmetic would pass a build-only check.
        const FieldData<vector> plain = readField<vector>(writeU(base + "/snf/plain",
            "    inlet { type surfaceNormalFixedValue; refValue uniform -10;\n"
            "            value uniform (0 0 0); }"));
        bool built = true; vector v0{};
        try
        {
            auto pf = makePatchField<vector>(p, plain.boundary.at(0));
            pf->evaluate({});   // buildField's job in real use; value_ is unset before it
            v0 = pf->value().at(0);
        }
        catch (const std::exception&) { built = false; }
        if (built && v0.x == -10.0 && v0.y == 0.0 && v0.z == 0.0)
            std::printf("  OK   constant refValue still builds U_b = refValue*n = (-10 0 0)\n");
        else { std::printf("  FAIL the constant form broke (built=%d, U_b=(%g %g %g))\n",
                           (int)built, (double)v0.x, (double)v0.y, (double)v0.z); failures++; }

        // The uniformNormalFixedValue spelling routes through the same slot; same rule.
        const FieldData<vector> unf = readField<vector>(writeU(base + "/snf/uniform",
            "    inlet { type uniformNormalFixedValue; uniformValue table ((0 0) (1 -10));\n"
            "            value uniform (0 0 0); }"));
        bool refused2 = false;
        try { (void)makePatchField<vector>(p, unf.boundary.at(0)); }
        catch (const std::exception& e)
        { refused2 = std::string(e.what()).find("uniformValue") != std::string::npos; }
        if (refused2) std::printf("  OK   uniformNormalFixedValue + table uniformValue refused too\n");
        else { std::printf("  FAIL the uniformNormalFixedValue spelling still slips through\n"); failures++; }

        // The MISSING entry arrives at the same zero inlet through a typo instead of a table. OF reads
        // the entry unconditionally (PatchFunction1::New), so absence must refuse as well.
        const FieldData<vector> miss = readField<vector>(writeU(base + "/snf/missing",
            "    inlet { type surfaceNormalFixedValue; value uniform (0 0 0); }"));
        bool refused3 = false;
        try { (void)makePatchField<vector>(p, miss.boundary.at(0)); }
        catch (const std::exception& e)
        { refused3 = std::string(e.what()).find("without the refValue") != std::string::npos; }
        if (refused3) std::printf("  OK   a missing refValue is refused, not run as U_b = 0\n");
        else { std::printf("  FAIL a missing refValue built as a silent zero inlet\n"); failures++; }
    }

    std::printf("uniform_function1: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
