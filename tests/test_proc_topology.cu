// proc_topology unit test: rank -> GPU placement on mocked cut-weight matrices and the DGX-1V cube-mesh link table
// (no GPU needed except the last probe check, which skips without one).
#include "proc_comm.cuh"
#include "proc_topology.cuh"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <sstream>

using namespace brae;

namespace {

int g_fail = 0;
#define CHECK(cond, ...)                                                                                            \
    do { if (!(cond)) { ++g_fail; std::printf("FAIL line %d [" #cond "]: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

std::vector<long long> chain(int np, long long w = 1000)
{
    std::vector<long long> W(static_cast<std::size_t>(np)*np, 0);
    for (int i = 0; i + 1 < np; ++i) W[static_cast<std::size_t>(i)*np + i + 1] = W[static_cast<std::size_t>(i + 1)*np + i] = w;
    return W;
}

std::vector<long long> cube(int np = 8, long long w = 1000)   // Q3: ranks differing in one bit
{
    std::vector<long long> W(static_cast<std::size_t>(np)*np, 0);
    for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j)
            if (__builtin_popcount(i ^ j) == 1) W[static_cast<std::size_t>(i)*np + j] = w;
    return W;
}

std::vector<long long> full(int np, bool varied)
{
    std::vector<long long> W(static_cast<std::size_t>(np)*np, 0);
    for (int i = 0; i < np; ++i)
        for (int j = i + 1; j < np; ++j)
            W[static_cast<std::size_t>(i)*np + j] = W[static_cast<std::size_t>(j)*np + i] = varied ? 100 + 37*((i*5 + j*11) % 13) : 500;
    return W;
}

double costOf(const std::vector<long long>& W, int np, const GpuTopology& t, const std::vector<int>& map, const PlacementModel& m)
{
    double c = 0;
    for (int i = 0; i < np; ++i)
        for (int j = i + 1; j < np; ++j)
            c += m.pairUs(t.at(map[i], map[j]), W[static_cast<std::size_t>(i)*np + j] + W[static_cast<std::size_t>(j)*np + i]);
    return c;
}

// independent exhaustive reference over the prefixes of all permutations of the GPUs
double refMin(const std::vector<long long>& W, int np, const GpuTopology& t, const PlacementModel& m, int* minSlow)
{
    std::vector<int> p(t.n);
    std::iota(p.begin(), p.end(), 0);
    double best = 1e300;
    int bs = 1 << 30;
    do
    {
        best = std::min(best, costOf(W, np, t, p, m));
        int s = 0;
        for (int i = 0; i < np; ++i)
            for (int j = i + 1; j < np; ++j)
                if (W[static_cast<std::size_t>(i)*np + j] + W[static_cast<std::size_t>(j)*np + i] > 0 && !isNvLinkClass(t.at(p[i], p[j]).cls)) ++s;
        bs = std::min(bs, s);
    } while (std::next_permutation(p.begin(), p.end()));
    *minSlow = bs;
    return best;
}

bool isPerm(const std::vector<int>& m, int nGpu)
{
    std::vector<int> s = m;
    std::sort(s.begin(), s.end());
    return std::adjacent_find(s.begin(), s.end()) == s.end() && s.front() >= 0 && s.back() < nGpu;
}

const char* kSmiDgx1 =
    "\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tmlx5_0\tCPU Affinity\tNUMA Affinity\n"
    "GPU0\t X \tNV1\tNV1\tNV2\tNV2\tSYS\tSYS\tSYS\tPIX\t0-19,40-59\t0\n"
    "GPU1\tNV1\t X \tNV2\tNV1\tSYS\tNV2\tSYS\tSYS\tPIX\t0-19,40-59\t0\n"
    "GPU2\tNV1\tNV2\t X \tNV2\tSYS\tSYS\tNV1\tSYS\tPHB\t0-19,40-59\t0\n"
    "GPU3\tNV2\tNV1\tNV2\t X \tSYS\tSYS\tSYS\tNV1\tPHB\t0-19,40-59\t0\n"
    "GPU4\tNV2\tSYS\tSYS\tSYS\t X \tNV1\tNV1\tNV2\tSYS\t20-39,60-79\t1\n"
    "GPU5\tSYS\tNV2\tSYS\tSYS\tNV1\t X \tNV2\tNV1\tSYS\t20-39,60-79\t1\n"
    "GPU6\tSYS\tSYS\tNV1\tSYS\tNV1\tNV2\t X \tNV2\tSYS\t20-39,60-79\t1\n"
    "GPU7\tSYS\tSYS\tSYS\tNV1\tNV2\tNV1\tNV2\t X \tSYS\t20-39,60-79\t1\n"
    "mlx5_0\tPIX\tPIX\tPHB\tPHB\tSYS\tSYS\tSYS\tSYS\t X \n"
    "\nLegend:\n  X = Self\n";

} // namespace

int main()
{
    const PlacementModel model;
    const GpuTopology dgx1 = dgx1V100Topology();
    dgx1.validate();

    // --- the table itself: 6 NVLinks per GPU, 4 NVLink peers, 3 SYS peers, all in-quad pairs linked
    for (int a = 0; a < 8; ++a)
    {
        int lanes = 0, nv = 0, sys = 0;
        for (int b = 0; b < 8; ++b)
        {
            if (a == b) continue;
            const LinkInfo& l = dgx1.at(a, b);
            if (l.cls == LinkClass::NVLink) { lanes += l.lanes; ++nv; }
            else if (l.cls == LinkClass::CrossSocket) ++sys;
            if (a/4 == b/4) CHECK(l.cls == LinkClass::NVLink, "GPUs %d,%d of one quad are not NVLinked", a, b);
        }
        CHECK(lanes == 6 && nv == 4 && sys == 3, "GPU %d: %d lanes, %d NVLink peers, %d SYS peers", a, lanes, nv, sys);
    }

    // --- nvidia-smi parser reproduces the mock table (incl. a NIC column and the socket from the NUMA column)
    {
        const GpuTopology p = parseNvidiaSmiTopo(kSmiDgx1);
        CHECK(p.n == 8, "parsed %d GPUs", p.n);
        bool same = p.n == 8;
        for (int a = 0; same && a < 8; ++a)
            for (int b = 0; b < 8; ++b)
                if (p.at(a, b).cls != dgx1.at(a, b).cls || p.at(a, b).lanes != dgx1.at(a, b).lanes) { same = false; break; }
        CHECK(same, "parsed link table differs from dgx1V100Topology()");
        CHECK(p.socket[0] == 0 && p.socket[3] == 0 && p.socket[4] == 1 && p.socket[7] == 1, "parsed sockets");
        CHECK(p.cpuList[5] == "20-39,60-79", "parsed cpu list '%s'", p.cpuList[5].c_str());
        bool threw = false;
        try { parseNvidiaSmiTopo("nothing here\n"); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "garbage text must throw");
    }

    // --- chain of 8: every neighbouring pair lands on an NVLink pair; matches the independent exhaustive search;
    //     the identity placement (0-1-2-3-4: pair 3-4 is SYS) is worse
    {
        const std::vector<long long> W = chain(8);
        const PlacementResult r = placeRanks(W, 8, dgx1, model);
        int ms = 0;
        const double ref = refMin(W, 8, dgx1, model, &ms);
        CHECK(isPerm(r.gpuOfRank, 8), "not a permutation");
        CHECK(r.evaluated == 40320, "evaluated %lld", r.evaluated);
        CHECK(std::abs(r.costUs - ref) < 1e-6*ref, "cost %.6f != exhaustive reference %.6f", r.costUs, ref);
        CHECK(r.slowPairs == 0 && r.minSlowPairs == 0, "chain has %d non-NVLink pairs", r.slowPairs);
        for (int i = 0; i + 1 < 8; ++i)
            CHECK(dgx1.at(r.gpuOfRank[i], r.gpuOfRank[i + 1]).cls == LinkClass::NVLink, "chain pair %d-%d on %s", i, i + 1,
                  linkClassName(dgx1.at(r.gpuOfRank[i], r.gpuOfRank[i + 1]).cls));
        CHECK(r.costUs < r.identityCostUs, "optimal %.1f us is not better than identity %.1f us", r.costUs, r.identityCostUs);
        // all-NV2 chain exists? the optimum must use as many NV2 edges as it can: 7 pairs, check against reference above
        const PlacementResult again = placeRanks(W, 8, dgx1, model);
        CHECK(again.gpuOfRank == r.gpuOfRank, "placement is not deterministic");
        std::printf("chain 8 on DGX-1V: rank->GPU %s, %.1f us (identity %.1f us)\n", gpuMapString(r).c_str(), r.costUs, r.identityCostUs);
    }

    // --- hierarchical (2 2 2): the cube rank graph embeds in the cube-mesh, all 12 pairs on NVLink
    {
        const std::vector<long long> W = cube();
        const PlacementResult r = placeRanks(W, 8, dgx1, model);
        int ms = 0;
        const double ref = refMin(W, 8, dgx1, model, &ms);
        CHECK(r.slowPairs == 0 && ms == 0, "Q3 on the cube-mesh: %d slow pairs (reference %d)", r.slowPairs, ms);
        CHECK(r.pairs.size() == 12, "Q3 has %zu communicating pairs", r.pairs.size());
        CHECK(std::abs(r.costUs - ref) < 1e-6*ref, "cost %.6f != reference %.6f", r.costUs, ref);
        std::printf("cube 2x2x2 on DGX-1V: rank->GPU %s, all %zu pairs on NVLink\n", gpuMapString(r).c_str(), r.pairs.size());
    }

    // --- scotch-like: every rank talks to all 7 others. 28 pairs, 16 NVLink pairs exist: 12 PCIe/SYS hops are unavoidable
    for (int varied = 0; varied < 2; ++varied)
    {
        const std::vector<long long> W = full(8, varied);
        const PlacementResult r = placeRanks(W, 8, dgx1, model);
        int ms = 0;
        const double ref = refMin(W, 8, dgx1, model, &ms);
        CHECK(ms == 12 && r.minSlowPairs == 12, "unavoidable hops: %d (reference %d), expected 12", r.minSlowPairs, ms);
        CHECK(r.slowPairs >= 12 && r.pairs.size() == 28, "slow pairs %d of %zu", r.slowPairs, r.pairs.size());
        CHECK(std::abs(r.costUs - ref) < 1e-6*ref, "cost %.6f != reference %.6f", r.costUs, ref);
        if (!varied) CHECK(r.slowPairs == 12, "uniform weights: placement has %d slow pairs, 12 is the minimum", r.slowPairs);
        std::ostringstream os;
        printPlacement(os, r, dgx1, W);
        const std::string txt = os.str();
        CHECK(txt.find("UNAVOIDABLE") != std::string::npos || varied, "report must call the 12 hops unavoidable:\n%s", txt.c_str());
        CHECK(txt.find("BRAE_GPU_MAP=") != std::string::npos && txt.find("not NVLink") != std::string::npos, "report content");
        if (varied == 0) std::printf("--- report for the 7-neighbour pattern ---\n%s", txt.c_str());
    }

    // --- fewer ranks than GPUs: np=2 takes an NV2 pair, np=4 stays in one quad (never straddles the sockets)
    {
        std::vector<long long> W2(4, 0);
        W2[1] = W2[2] = 5000;
        const PlacementResult r2 = placeRanks(W2, 2, dgx1, model);
        CHECK(dgx1.at(r2.gpuOfRank[0], r2.gpuOfRank[1]).lanes == 2, "np=2 not on an NV2 pair");
        CHECK(r2.gpuOfRank == (std::vector<int>{0, 3}), "np=2 tie-break: lexicographically smallest NV2 pair is 0-3, got %s", gpuMapString(r2).c_str());
        CHECK(r2.evaluated == 56, "np=2 evaluated %lld maps (8*7 = 56)", r2.evaluated);
        const PlacementResult r4 = placeRanks(chain(4), 4, dgx1, model);
        int ms = 0;
        const double ref = refMin(chain(4), 4, dgx1, model, &ms);
        CHECK(std::abs(r4.costUs - ref) < 1e-6*ref, "np=4 cost");
        const int q = r4.gpuOfRank[0]/4;
        for (int g : r4.gpuOfRank) CHECK(g/4 == q, "np=4 chain straddles the sockets: %s", gpuMapString(r4).c_str());
        CHECK(r4.slowPairs == 0, "np=4 chain has slow pairs");
        const BindingAdvice ba = bindingAdvice(r4, dgx1);
        CHECK(ba.socket[0] == ba.socket[3] && ba.text.find("numactl --cpunodebind=") != std::string::npos, "binding advice");
        std::printf("np=2 -> GPUs %s; np=4 chain -> GPUs %s\n", gpuMapString(r2).c_str(), gpuMapString(r4).c_str());
        std::printf("%s", ba.text.c_str());
    }

    // --- NVSwitch: every placement ties, the tie-break returns the identity
    {
        const GpuTopology sw = dgxNvSwitchTopology(8);
        const PlacementResult r = placeRanks(full(8, true), 8, sw, model);
        CHECK(r.gpuOfRank == (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}), "NVSwitch tie-break: %s", gpuMapString(r).c_str());
        CHECK(r.slowPairs == 0, "NVSwitch has no slow pair");
    }

    // --- weights from real plans: a 2x2x2 block decomposition of a box gives the cube rank graph
    {
        const int nx = 16, ny = 16, nz = 16;
        const SyntheticPartition sp = partitionBox(nx, ny, nz, 8, blockOwner(nx, ny, nz, 2, 2, 2));
        const std::vector<long long> W = cutWeightsFromDescs(sp.descs);
        CHECK(W[0*8 + 1] == 8*8 && W[0*8 + 2] == 8*8 && W[0*8 + 4] == 8*8 && W[0*8 + 7] == 0, "cut weights of block 2x2x2: %lld %lld %lld %lld",
              W[1], W[2], W[4], W[7]);
        const PlacementResult r = placeRanks(W, 8, dgx1, model);
        CHECK(r.slowPairs == 0, "2x2x2 block decomposition has %d non-NVLink pairs", r.slowPairs);
    }

    // --- the probe runs on this machine (1..n GPUs) and gives a valid table
    {
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev < 1) { cudaGetLastError(); std::printf("SKIP probe: no CUDA device\n"); }
        else
        {
            std::string note;
            const GpuTopology t = probeTopology(true, &note);
            t.validate();
            CHECK(t.n == nDev, "probe n %d != %d", t.n, nDev);
            std::ostringstream os;
            printTopology(os, t);
            std::printf("probe on this machine (%s)%s%s\n%s", t.source.c_str(), note.empty() ? "" : " note: ", note.c_str(), os.str().c_str());
        }
    }

    std::printf("test_proc_topology: %s (%d failing checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
