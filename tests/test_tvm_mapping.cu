// timeVaryingMappedFixedValue's mapping, against the properties OpenFOAM's own pipeline guarantees.
//
// OF's DEFAULT is planar Delaunay interpolation (pointToPointPlanarInterpolation); brae ran 3D
// nearest for everything -- a silent substitution that staircases any profile coarser than the mesh.
// The SHARP check needs no oracle run: barycentric interpolation reproduces a LINEAR field exactly,
// and nearest cannot (it returns station values verbatim), so one synthetic profile discriminates the
// two mappings to machine precision. The steady-scope refusals (multi-dir table, setAverage, sample
// readers, non-constant offset, run time below the table) are each exercised by mutation.
#include "foam_field_reader.cuh"
#include "fv_patch_field.cuh"
#include "fv_patch.cuh"
#include <cmath>
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

static const std::string BASE = "/tmp/brae_tvm_mapping";

// One synthetic case: 6 y-stations duplicated at two z (the pitzDailyExptInlet shape), value 2 + 3y.
static void writeCase(const std::string& tag, const std::string& patchEntryExtras,
                      const std::string& timeDir = "0", const std::string& fieldTime = "0")
{
    const std::string c = BASE + "/" + tag;
    std::filesystem::create_directories(c + "/" + fieldTime);
    std::filesystem::create_directories(c + "/constant/boundaryData/inlet/" + timeDir);
    {
        std::ofstream f(c + "/constant/boundaryData/inlet/points");
        f << "12\n(\n";
        for (int zi = 0; zi < 2; ++zi)
            for (int j = 0; j < 6; ++j)
                f << "(0 " << 0.1*j << " " << 0.05*zi << ")\n";
        f << ")\n";
    }
    {
        std::ofstream f(c + "/constant/boundaryData/inlet/" + timeDir + "/T");
        f << "12\n(\n";
        for (int zi = 0; zi < 2; ++zi)
            for (int j = 0; j < 6; ++j)
                f << (2.0 + 3.0*(0.1*j)) << "\n";
        f << ")\n";
    }
    {
        std::ofstream f(c + "/" + fieldTime + "/T");
        f << "FoamFile { version 2.0; format ascii; class volScalarField; object T; }\n"
          << "dimensions [0 0 0 1 0 0 0];\ninternalField uniform 300;\n"
          << "boundaryField {\n  inlet { type timeVaryingMappedFixedValue; "
          << patchEntryExtras << " value uniform 300; }\n}\n";
    }
}

static FvPatch midPatch()   // face centres BETWEEN the y-stations, inside the point cloud's z-band
{
    FvPatch p;
    p.name = "inlet";
    p.type = "patch";
    p.size = 5;
    for (label i = 0; i < 5; ++i)
    {
        p.faceCells.push_back(0);
        p.deltaCoeffs.push_back(1.0);
        p.nf.push_back(vector{1, 0, 0});
        p.magSf.push_back(1.0);
        p.Cf.push_back(vector{0, 0.05 + 0.1*i, 0.025});   // midway between stations, mid-z
    }
    return p;
}

struct R { bool threw = false; std::string msg; std::vector<scalar> v; };
static R run(const std::string& tag, const std::string& fieldTime = "0")
{
    R r;
    try
    {
        const FieldData<scalar> fd = readField<scalar>(BASE + "/" + tag + "/" + fieldTime + "/T");
        FvPatch p = midPatch();
        auto pf = makePatchField<scalar>(p, fd.boundary.at(0));
        pf->evaluate({});
        r.v = pf->value();
    }
    catch (const std::exception& e) { r.threw = true; r.msg = e.what(); }
    return r;
}

int main()
{
    std::filesystem::remove_all(BASE);

    // ---- the DEFAULT mapping reproduces a linear profile exactly (planar barycentric) -------------
    writeCase("planar", "");
    {
        const R r = run("planar");
        check("the default mapping builds", !r.threw);
        double dMax = 0;
        for (int i = 0; i < 5 && !r.threw; ++i)
            dMax = std::fmax(dMax, std::fabs((double)r.v[i] - (2.0 + 3.0*(0.05 + 0.1*i))));
        std::printf("     max |planar - exact linear| = %.3e\n", dMax);
        check("planar (the OF default) reproduces the linear profile EXACTLY", !r.threw && dMax < 1e-12);
    }

    // ---- mapMethod nearest staircases -- proving the dispatch is real, not two names one path ----
    writeCase("nearest", "mapMethod nearest; ");
    {
        const R r = run("nearest");
        check("mapMethod nearest builds", !r.threw);
        double dMax = 0;
        for (int i = 0; i < 5 && !r.threw; ++i)
            dMax = std::fmax(dMax, std::fabs((double)r.v[i] - (2.0 + 3.0*(0.05 + 0.1*i))));
        std::printf("     max |nearest - exact linear| = %.3e (a half-step is 0.15)\n", dMax);
        check("...and STAIRCASES the same profile (the two mappings differ)", !r.threw && dMax > 0.1);
    }

    // ---- constant offset is added after mapping ---------------------------------------------------
    writeCase("offset", "offset constant 10; ");
    {
        const R r = run("offset");
        check("a constant offset is applied after mapping",
              !r.threw && std::fabs((double)r.v[0] - (2.0 + 3.0*0.05 + 10.0)) < 1e-12);
    }

    // ---- the refusals, each by name ---------------------------------------------------------------
    writeCase("twodirs", "");
    std::filesystem::create_directories(BASE + "/twodirs/constant/boundaryData/inlet/1");
    { std::ofstream f(BASE + "/twodirs/constant/boundaryData/inlet/1/T"); f << "12\n(\n"; for (int i=0;i<12;++i) f << "5\n"; f << ")\n"; }
    { const R r = run("twodirs");
      check("a multi-dir time table is refused (steady pseudo-time)", r.threw && r.msg.find("time directories") != std::string::npos); }

    writeCase("setavg", "setAverage true; ");
    { const R r = run("setavg");
      check("setAverage true is refused by name", r.threw && r.msg.find("setAverage") != std::string::npos); }

    writeCase("sfmt", "sampleFormat someReader; ");
    { const R r = run("sfmt");
      check("sampleFormat is refused by name", r.threw && r.msg.find("sampleFormat") != std::string::npos); }

    writeCase("offtab", "offset table ((0 0) (1 5)); ");
    { const R r = run("offtab");
      check("a non-constant offset is refused by name", r.threw && r.msg.find("offset") != std::string::npos); }

    writeCase("badmap", "mapMethod voronoi; ");
    { const R r = run("badmap");
      check("an unknown mapMethod is refused (OF fatals too)", r.threw && r.msg.find("mapMethod") != std::string::npos); }

    // run time BELOW the table start: OF fatals (MappedFile.C:593-604); brae used to run silently
    writeCase("early", "", /*timeDir*/"10", /*fieldTime*/"0");
    { const R r = run("early", "0");
      check("a run time below the sample time is refused (OF fatals)", r.threw && r.msg.find("below") != std::string::npos); }

    // fieldTable is DEAD on this BC in v2412 -- accepted, not refused
    writeCase("ftable", "fieldTable T; ");
    { const R r = run("ftable");
      check("fieldTable (dead in v2412) is accepted", !r.threw); }

    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
