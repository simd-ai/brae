// proc_topology_tool: rank -> GPU placement advisor for multi-GPU brae (CommSolvers.md step D).
//
//   proc_topology_tool [--topology dgx1v | nvswitch | probe | smi=<file with `nvidia-smi topo -m`>]   (default probe)
//                      [--np N]
//                      (--case <decomposed case dir> | --matrix <file of np*np face counts> | --chain | --cube | --all-to-all)
//                      [--bytes-per-face B] [--program "<cmd>"] [--map-only]
//
// Prints the GPU link table, the cut weights, the optimal placement (exhaustive over <= 8! maps, deterministic), the
// rank pairs that are not on NVLink (and how many hops no placement can avoid), and the CPU/socket binding advice.
// `--map-only` prints just the BRAE_GPU_MAP value (for `export BRAE_GPU_MAP=$(proc_topology_tool --map-only ...)`);
// Pstream::init binds node-local rank i to GPU map[i] before any CUDA context exists.
// Runs on the login/launch shell BEFORE mpirun: `--topology probe` queries every GPU of the node (needs the GPUs).
#include "proc_topology.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>

using namespace brae;

namespace {

std::string slurp(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// processorN/constant/polyMesh/boundary: every patch with `neighbProcNo K;` and `nFaces M;`
std::vector<long long> matrixFromCase(const std::string& dir, int& np)
{
    np = 0;
    while (std::ifstream(dir + "/processor" + std::to_string(np) + "/constant/polyMesh/boundary")) ++np;
    if (np == 0) throw std::runtime_error("no processor0/constant/polyMesh/boundary under " + dir + " (decomposePar first)");
    std::vector<std::vector<std::pair<int, long long>>> nf(np);
    for (int r = 0; r < np; ++r)
    {
        const std::string t = slurp(dir + "/processor" + std::to_string(r) + "/constant/polyMesh/boundary");
        for (std::size_t pos = t.find("neighbProcNo"); pos != std::string::npos; pos = t.find("neighbProcNo", pos + 1))
        {
            const std::size_t open = t.rfind('{', pos), close = t.find('}', pos);
            if (open == std::string::npos || close == std::string::npos) throw std::runtime_error("malformed boundary file of processor" + std::to_string(r));
            const std::string blk = t.substr(open, close - open);
            const std::size_t nfp = blk.find("nFaces");
            if (nfp == std::string::npos) throw std::runtime_error("processor patch without nFaces in processor" + std::to_string(r));
            const int nb = std::atoi(t.c_str() + pos + std::strlen("neighbProcNo"));
            const long long n = std::atoll(blk.c_str() + nfp + std::strlen("nFaces"));
            nf[r].emplace_back(nb, n);
        }
    }
    return cutWeightsFromNeighbours(nf);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string topoArg = "probe", caseDir, matrixFile, program = "./brae_interFoam -parallel", pattern;
        int np = 0;
        bool mapOnly = false;
        PlacementModel model;
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value"); return argv[++i]; };
            if (a == "--topology") topoArg = next();
            else if (a == "--np") np = std::atoi(next().c_str());
            else if (a == "--case") caseDir = next();
            else if (a == "--matrix") matrixFile = next();
            else if (a == "--chain" || a == "--cube" || a == "--all-to-all") pattern = a;
            else if (a == "--bytes-per-face") model.bytesPerFace = std::atof(next().c_str());
            else if (a == "--program") program = next();
            else if (a == "--map-only") mapOnly = true;
            else throw std::invalid_argument("unknown option " + a);
        }
        GpuTopology topo;
        std::string note;
        if (topoArg == "dgx1v") topo = dgx1V100Topology();
        else if (topoArg == "nvswitch") topo = dgxNvSwitchTopology(8);
        else if (topoArg == "probe") topo = probeTopology(true, &note);
        else if (topoArg.rfind("smi=", 0) == 0) topo = parseNvidiaSmiTopo(slurp(topoArg.substr(4)));
        else throw std::invalid_argument("--topology: dgx1v | nvswitch | probe | smi=<file>");

        std::vector<long long> W;
        if (!caseDir.empty()) { int n = 0; W = matrixFromCase(caseDir, n); if (np && np != n) throw std::invalid_argument("--np disagrees with the case"); np = n; }
        else if (!matrixFile.empty())
        {
            if (np < 1) throw std::invalid_argument("--matrix needs --np");
            std::istringstream is(slurp(matrixFile));
            W.resize(static_cast<std::size_t>(np)*np);
            for (auto& w : W) if (!(is >> w)) throw std::invalid_argument("matrix file: need np*np numbers");
        }
        else if (!pattern.empty())
        {
            if (np < 1) np = topo.n;
            W.assign(static_cast<std::size_t>(np)*np, 0);
            for (int i = 0; i < np; ++i)
                for (int j = 0; j < np; ++j)
                {
                    if (i == j) continue;
                    bool nb = pattern == "--all-to-all" || (pattern == "--chain" && (j == i + 1 || i == j + 1)) ||
                              (pattern == "--cube" && np == 8 && __builtin_popcount(i ^ j) == 1);
                    if (nb) W[static_cast<std::size_t>(i)*np + j] = 10000;
                }
        }
        else throw std::invalid_argument("give --case, --matrix, --chain, --cube or --all-to-all");

        const PlacementResult r = placeRanks(W, np, topo, model);
        if (mapOnly) { std::cout << gpuMapString(r) << "\n"; return 0; }
        if (!note.empty()) std::cout << "note: " << note << "\n";
        printTopology(std::cout, topo);
        std::cout << "\n";
        printPlacement(std::cout, r, topo, W);
        std::cout << "\n" << bindingAdvice(r, topo, program).text;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "proc_topology_tool: %s\n", e.what());
        return 1;
    }
}
