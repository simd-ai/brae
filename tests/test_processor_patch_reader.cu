// Processor patches and per-rank case roots (host only, no GPU work).
//
//   1. a REAL processor `boundary` (OpenFOAM v2606 decomposePar of tools/mgpu_oracle's tank case, 4 domains,
//      processor1 -- two processor patches, one zero-size physical patch) is read with myProcNo, neighbProcNo,
//      the zero-size patch kept and the processor patches last;
//   2. np = 1 (processor0 of a one-domain decomposition) has no processor patch;
//   3. a processorCyclic carries referPatch and is refused BY NAME under -parallel;
//   4. the mesh cache (magic CFM5) round-trips the new fields;
//   5. FvPatch carries them from a tiny 2-domain mesh read whole (points, faces, owner, neighbour, boundary);
//   6. every polyMesh/boundary the repository tracks parses EXACTLY as before (golden produced by the reader of
//      2a7a5268, tests/boundary_parse_dump.cuh);
//   7. the per-rank roots: -rank/-nprocs, the launcher's environment, processor<rank>, processor0 for the time
//      list; and the refusals under -parallel by name.
//
//   test_processor_patch_reader <repo root>
#include "boundary_parse_dump.cuh"
#include "case_roots.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "inter_parallel.cuh"
#include "primitive_mesh.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace brae;

static int fails = 0;
#define CHECK(cond, ...)                                                                      \
    do {                                                                                      \
        if (!(cond)) { ++fails; std::printf("  FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond);  \
                       std::printf(__VA_ARGS__); std::printf("\n"); }                         \
    } while (0)

static std::string readAll(const std::string& p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
static void writeAll(const std::string& p, const std::string& s)
{
    fs::create_directories(fs::path(p).parent_path());
    std::ofstream(p) << s;
}
template <class F>
static std::string thrown(F f)
{
    try { f(); } catch (const std::exception& e) { return e.what(); }
    return "";
}
static bool has(const std::string& s, const std::string& w) { return s.find(w) != std::string::npos; }

static PrimitiveMesh readBoundaryFile(const std::string& dir, const std::string& boundaryFile)
{
    fs::create_directories(dir);
    fs::copy_file(boundaryFile, dir + "/boundary", fs::copy_options::overwrite_existing);
    PrimitiveMesh m;
    m.readBoundary(dir);
    return m;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s <repo root>\n", argv[0]); return 2; }
    const std::string root = argv[1];
    const std::string data = root + "/tests/data";
    const std::string tmp = (fs::temp_directory_path() / ("brae_proc_reader_" + std::to_string(::getpid()))).string();
    fs::remove_all(tmp);
    std::printf("test_processor_patch_reader:\n");

    // ---- 1. a real processor boundary -------------------------------------------------------------------
    {
        const PrimitiveMesh m = readBoundaryFile(tmp + "/np4p1", data + "/processor_boundary/tank_np4_processor1.boundary");
        const auto& P = m.patches();
        CHECK(P.size() == 4u, "got %zu", P.size());
        if (P.size() == 4u)
        {
            CHECK(P[0].name == "tankWall" && P[0].type == "wall" && P[0].size == 2304 && P[0].start == 39456, "tankWall");
            CHECK(P[1].name == "lid" && P[1].size == 0 && P[1].start == 41760, "zero-size physical patch kept");
            CHECK(P[2].name == "procBoundary1to0" && P[2].type == "processor", "name/type %s %s", P[2].name.c_str(), P[2].type.c_str());
            CHECK(P[2].size == 1152 && P[2].start == 41760, "size/start %ld %ld", (long)P[2].size, (long)P[2].start);
            CHECK(P[2].myProcNo == 1 && P[2].neighbProcNo == 0, "ranks %d %d", (int)P[2].myProcNo, (int)P[2].neighbProcNo);
            CHECK(P[2].transform == "unknown" && P[2].matchTolerance == 1e-4, "transform/tol");
            CHECK(P[2].inGroups.size() == 1u && P[2].inGroups[0] == "processor", "inGroups");
            CHECK(P[3].name == "procBoundary1to3" && P[3].size == 576 && P[3].start == 42912, "second processor patch");
            CHECK(P[3].myProcNo == 1 && P[3].neighbProcNo == 3, "ranks %d %d", (int)P[3].myProcNo, (int)P[3].neighbProcNo);
            for (int i = 0; i < 2; ++i) CHECK(P[i].myProcNo == -1 && P[i].neighbProcNo == -1 && P[i].referPatch.empty(), "wall carries rank");

            CaseRoots r;
            r.parallel = true; r.rank = 1; r.nProcs = 4; r.dictDir = "case"; r.procDir = "case/processor1";
            const std::string table = cpu::interFoam::processorPatchTable(r, P);
            CHECK(has(table, "procBoundary1to3") && has(table, "rank 1 <-> rank 3") && has(table, "1728 processor face"),
                  "table:\n%s", table.c_str());
            const std::string why = thrown([&] { cpu::interFoam::refuseProcessorCoupling(r, P); });
            CHECK(why.empty(), "-parallel couples processor patches, it does not refuse them: %s", why.c_str());
            const std::string serial = thrown([&] { cpu::interFoam::refuseProcessorCoupling(CaseRoots("case"), P); });
            CHECK(has(serial, "-parallel") && has(serial, "ordinary wall"), "serial refusal: %s", serial.c_str());
        }
    }

    // ---- 2. np = 1: no processor patch, runs as an ordinary case -----------------------------------------
    {
        const PrimitiveMesh m = readBoundaryFile(tmp + "/np1", data + "/processor_boundary/tank_np1_processor0.boundary");
        CHECK(m.patches().size() == 2u, "np=1 patches %zu", m.patches().size());
        for (const PatchInfo& p : m.patches()) CHECK(p.type == "wall", "type %s", p.type.c_str());
        CaseRoots r;
        r.parallel = true; r.rank = 0; r.nProcs = 1; r.dictDir = "case"; r.procDir = "case/processor0";
        CHECK(thrown([&] { cpu::interFoam::refuseProcessorCoupling(r, m.patches()); }).empty(), "np=1 refused");
        CHECK(thrown([&] { cpu::interFoam::refuseUnderParallel(r, m.patches()); }).empty(), "np=1 refused by name");
    }

    // ---- 3. processorCyclic ------------------------------------------------------------------------------
    {
        const std::string text =
            "FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }\n"
            "2\n(\n"
            "    procBoundary0to1throughcyclicLeft\n    {\n        type            processorCyclic;\n"
            "        inGroups        2(cyclic processorCyclic);\n        nFaces          16;\n        startFace       100;\n"
            "        matchTolerance  0.0001;\n        transform       translational;\n        separationVector (1 0 0);\n"
            "        myProcNo        0;\n        neighbProcNo    1;\n        referPatch      cyclicLeft;\n        tag 1;\n    }\n"
            "    procBoundary0to2\n    {\n        type            processor;\n        inGroups        1(processor);\n"
            "        nFaces          8;\n        startFace       116;\n        matchTolerance  0.0001;\n        transform unknown;\n"
            "        myProcNo        0;\n        neighbProcNo    2;\n    }\n)\n";
        writeAll(tmp + "/pc/boundary", text);
        PrimitiveMesh m;
        m.readBoundary(tmp + "/pc");
        CHECK(m.patches().size() == 2u, "patches");
        if (m.patches().size() == 2u)
        {
            const PatchInfo& p = m.patches()[0];
            CHECK(p.type == "processorCyclic" && p.referPatch == "cyclicLeft" && p.myProcNo == 0 && p.neighbProcNo == 1
                  && p.transform == "translational" && p.size == 16, "processorCyclic fields");
            CHECK(m.patches()[1].referPatch.empty(), "referPatch leaked");
        }
        CaseRoots r;
        r.parallel = true; r.rank = 0; r.nProcs = 3; r.dictDir = tmp + "/pcCase"; r.procDir = r.dictDir + "/processor0";
        const std::string why = thrown([&] { cpu::interFoam::refuseUnderParallel(r, m.patches()); });
        CHECK(has(why, "processorCyclic") && has(why, "cyclicLeft") && has(why, "preservePatches"), "refusal: %s", why.c_str());
    }

    // ---- 4. the mesh cache round-trips the fields, and a CFM4 cache is not read ---------------------------
    {
        const std::string d = tmp + "/cache";
        fs::create_directories(d);
        for (const char* f : {"points", "faces", "owner", "neighbour", "boundary"})
            fs::copy_file(data + "/processor_mesh_np2/processor0/constant/polyMesh/" + f, d + "/" + f);
        ::setenv("BRAE_MESH_CACHE", "1", 1);
        PrimitiveMesh cold;
        cold.read(d);
        ::unsetenv("BRAE_MESH_CACHE");
        CHECK(fs::exists(d + "/.brae_meshcache"), "no cache written");
        writeAll(d + "/boundary", "garbage that cannot parse");     // a warm read must not look at it
        PrimitiveMesh warm;
        warm.read(d);
        CHECK(warm.patches().size() == cold.patches().size() && !warm.patches().empty(), "warm patches %zu", warm.patches().size());
        bool same = warm.patches().size() == cold.patches().size();
        for (std::size_t i = 0; same && i < warm.patches().size(); ++i)
            same = brae_test::dumpLegacyPatch(warm.patches()[i]) == brae_test::dumpLegacyPatch(cold.patches()[i])
                && warm.patches()[i].myProcNo == cold.patches()[i].myProcNo
                && warm.patches()[i].neighbProcNo == cold.patches()[i].neighbProcNo
                && warm.patches()[i].referPatch == cold.patches()[i].referPatch;
        CHECK(same, "cache round trip differs");
        CHECK(warm.patches().back().type == "processor" && warm.patches().back().myProcNo == 0
              && warm.patches().back().neighbProcNo == 1, "processor fields lost in the cache");
        std::string bytes = readAll(d + "/.brae_meshcache");
        unsigned magic = 0;
        std::memcpy(&magic, bytes.data(), sizeof magic);
        CHECK(magic == 0x43464D35u, "magic %08x, expect CFM5", magic);
        // an old-format cache (magic CFM4) is rejected: the ascii parse runs and meets the garbage boundary
        const unsigned old = 0x43464D34u;
        std::memcpy(&bytes[0], &old, sizeof old);
        writeAll(d + "/.brae_meshcache", bytes);
        const std::string why = thrown([&] { PrimitiveMesh m; m.read(d); });
        CHECK(!why.empty(), "a CFM4 cache was accepted");
    }

    // ---- 5. FvPatch carries the fields, from a whole tiny 2-domain mesh ----------------------------------
    for (int rank = 0; rank < 2; ++rank)
    {
        PrimitiveMesh m;
        m.read(data + "/processor_mesh_np2/processor" + std::to_string(rank) + "/constant/polyMesh");
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> patches = buildPatches(m, g, /*mirrorACMI=*/true);
        CHECK(m.nCells() == 8, "cells %ld", (long)m.nCells());
        CHECK(patches.size() == 5u, "patches %zu", patches.size());
        if (patches.size() != 5u) continue;
        const FvPatch& pp = patches.back();
        CHECK(pp.type == "processor" && pp.size == 4 && pp.myProcNo == rank && pp.neighbProcNo == 1 - rank
              && pp.name == "procBoundary" + std::to_string(rank) + "to" + std::to_string(1 - rank), "FvPatch %s", pp.name.c_str());
        CHECK(!pp.coupled, "reading a processor patch must not couple it");
        for (int i = 0; i < 4; ++i) CHECK(patches[i].myProcNo == -1 && patches[i].neighbProcNo == -1, "wall");
        if (rank == 1) CHECK(patches[1].name == "inlet" && patches[1].size == 0, "zero-size inlet kept on rank 1");
        else           CHECK(patches[1].name == "inlet" && patches[1].size == 4, "inlet on rank 0");
        CaseRoots r;
        r.parallel = true; r.rank = rank; r.nProcs = 2; r.dictDir = "c"; r.procDir = "c/processor" + std::to_string(rank);
        const std::string why = thrown([&] { cpu::interFoam::refuseProcessorCoupling(r, m.patches()); });
        CHECK(why.empty(), "-parallel couples processor patches, it does not refuse them: %s", why.c_str());
        const std::string serial = thrown([&] { cpu::interFoam::refuseProcessorCoupling(CaseRoots("c"), m.patches()); });
        CHECK(has(serial, "ordinary wall") && has(serial, "-parallel"), "serial refusal: %s", serial.c_str());
    }

    // ---- 6. every tracked boundary parses as it did before ------------------------------------------------
    {
        const std::string golden = readAll(data + "/boundary_parse_golden.txt");
        CHECK(!golden.empty(), "no golden");
        std::istringstream in(golden);
        std::string line, expect, rel;
        long nFiles = 0, nBad = 0;
        auto flush = [&]() {
            if (rel.empty()) return;
            ++nFiles;
            const std::string got = brae_test::dumpLegacyBoundary(root, rel);
            if (got != expect)
            {
                ++nBad;
                std::printf("  FAIL %s parses differently\n--- expected\n%s--- got\n%s", rel.c_str(), expect.c_str(), got.c_str());
            }
        };
        while (std::getline(in, line))
        {
            if (line.compare(0, 3, "## ") == 0)
            {
                flush();
                rel = line.substr(3);
                const std::size_t t = rel.find(" THROWS");
                if (t != std::string::npos) rel.resize(t);
                expect = line + "\n";
            }
            else
            {
                expect += line + "\n";
            }
        }
        flush();
        CHECK(nFiles >= 50, "only %ld boundary files in the golden", nFiles);
        CHECK(nBad == 0, "%ld of %ld boundaries parse differently", nBad, nFiles);
        std::printf("  %ld tracked boundary files compared with the golden\n", nFiles);
    }

    // ---- 7. roots and refusals ---------------------------------------------------------------------------
    {
        const std::string c = tmp + "/case";
        for (int r = 0; r < 4; ++r) fs::create_directories(c + "/processor" + std::to_string(r) + "/constant/polyMesh");
        fs::create_directories(c + "/processor9x");            // not a processor directory
        for (const char* v : {"OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "PMI_RANK", "PMI_SIZE", "SLURM_PROCID", "SLURM_NTASKS"})
            ::unsetenv(v);

        const CaseRoots serial = makeCaseRoots(c, false);
        CHECK(!serial.parallel && serial.dictDir == c && serial.procDir == c && serial.rank == 0 && serial.nProcs == 1
              && serial.isMaster() && serial.masterProcDir() == c, "serial roots");
        const CaseRoots viaString = c;                          // the converting constructor: a serial caller
        CHECK(viaString.dictDir == c && viaString.procDir == c, "implicit conversion");

        const CaseRoots f = makeCaseRoots(c, true, 2, 4);
        CHECK(f.parallel && f.rank == 2 && f.nProcs == 4 && f.dictDir == c && f.procDir == c + "/processor2"
              && !f.isMaster() && f.masterProcDir() == c + "/processor0", "flag roots");
        CHECK(has(thrown([&] { makeCaseRoots(c, true); }), "-rank N -nprocs M"), "parallel without rank");
        CHECK(has(thrown([&] { makeCaseRoots(c, true, 1, -1); }), "go together"), "-rank without -nprocs");
        CHECK(has(thrown([&] { makeCaseRoots(c, false, 0, 1); }), "-parallel"), "-rank without -parallel");
        CHECK(has(thrown([&] { makeCaseRoots(c, true, 4, 4); }), "not a valid rank"), "rank out of range");
        CHECK(has(thrown([&] { makeCaseRoots(c, true, 0, 2); }), "4 processor directories"), "size != processor dirs");

        ::setenv("OMPI_COMM_WORLD_RANK", "3", 1);
        ::setenv("OMPI_COMM_WORLD_SIZE", "4", 1);
        const CaseRoots e1 = makeCaseRoots(c, true);
        CHECK(e1.rank == 3 && e1.nProcs == 4 && e1.procDir == c + "/processor3", "OMPI env");
        const CaseRoots e2 = makeCaseRoots(c, true, 1, 4);
        CHECK(e2.rank == 1, "flags take precedence over the environment");
        ::unsetenv("OMPI_COMM_WORLD_RANK"); ::unsetenv("OMPI_COMM_WORLD_SIZE");
        ::setenv("PMI_RANK", "0", 1); ::setenv("PMI_SIZE", "4", 1);
        CHECK(makeCaseRoots(c, true).rank == 0, "PMI env");
        ::unsetenv("PMI_RANK"); ::unsetenv("PMI_SIZE");
        ::setenv("SLURM_PROCID", "2", 1); ::setenv("SLURM_NTASKS", "4", 1);
        CHECK(makeCaseRoots(c, true).rank == 2, "SLURM env");
        ::unsetenv("SLURM_PROCID"); ::unsetenv("SLURM_NTASKS");

        // refusals by name, each from its own dictionary of the case (the dictionaries are the CASE's)
        CaseRoots r = makeCaseRoots(c, true, 0, 4);
        const std::vector<PatchInfo> none;
        CHECK(thrown([&] { cpu::interFoam::refuseUnderParallel(r, none); }).empty(), "plain case refused");
        writeAll(c + "/constant/turbulenceProperties", "FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\nsimulationType laminar;\n");
        CHECK(thrown([&] { cpu::interFoam::refuseUnderParallel(r, none); }).empty(), "laminar refused");
        writeAll(c + "/constant/turbulenceProperties", "FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\nsimulationType RAS;\nRAS { model kEpsilon; }\n");
        CHECK(has(thrown([&] { cpu::interFoam::refuseUnderParallel(r, none); }), "simulationType `RAS`"), "turbulence");
        writeAll(c + "/constant/turbulenceProperties", "FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\nsimulationType laminar;\n");
        writeAll(c + "/constant/dynamicMeshDict", "FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\ndynamicFvMesh dynamicRefineFvMesh;\n");
        CHECK(has(thrown([&] { cpu::interFoam::refuseUnderParallel(r, none); }), "dynamicRefineFvMesh"), "dynamic refinement");
        writeAll(c + "/constant/dynamicMeshDict", "FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\ndynamicFvMesh staticFvMesh;\n");
        CHECK(thrown([&] { cpu::interFoam::refuseUnderParallel(r, none); }).empty(), "staticFvMesh refused");
        writeAll(c + "/constant/MRFProperties", "FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\nMRF1 { cellZone rotor; active yes; origin (0 0 0); axis (0 0 1); omega 1; }\n");
        CHECK(has(thrown([&] { cpu::interFoam::refuseUnderParallel(r, none); }), "MRF zone"), "MRF");
        writeAll(c + "/constant/MRFProperties", "FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\nMRF1 { cellZone rotor; active no; omega 1; }\n");
        CHECK(thrown([&] { cpu::interFoam::refuseUnderParallel(r, none); }).empty(), "inactive MRF refused");
        writeAll(c + "/system/fvOptions", "FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\nsink { type semiImplicitSource; active yes; }\n");
        CHECK(has(thrown([&] { cpu::interFoam::refuseUnderParallel(r, none); }), "fvOptions"), "fvOptions");
        fs::remove(c + "/system/fvOptions");
        for (const char* t : {"cyclicAMI", "cyclicACMI"})
        {
            PatchInfo p; p.name = "amiPatch"; p.type = t; p.size = 4;
            CHECK(has(thrown([&] { cpu::interFoam::refuseUnderParallel(r, {p}); }), std::string("is ") + t), "%s", t);
        }
    }

    fs::remove_all(tmp);
    std::printf(fails ? "FAILED (%d)\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
