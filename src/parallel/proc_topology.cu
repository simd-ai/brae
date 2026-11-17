// brae topology-aware rank -> GPU placement. See proc_topology.cuh for the contract.
#include "proc_topology.cuh"

#include "proc_comm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace brae {

const char* linkClassName(LinkClass c)
{
    switch (c)
    {
        case LinkClass::Self: return "X";
        case LinkClass::NVSwitch: return "NVS";
        case LinkClass::NVLink: return "NV";
        case LinkClass::PciSwitch: return "PIX";
        case LinkClass::PciHost: return "PHB";
        case LinkClass::CrossSocket: return "SYS";
        default: return "?";
    }
}

bool isNvLinkClass(LinkClass c) { return c == LinkClass::NVSwitch || c == LinkClass::NVLink; }

namespace {

std::string linkLabel(const LinkInfo& l)
{
    if (l.cls == LinkClass::NVLink) return "NV" + std::to_string(l.lanes);
    return linkClassName(l.cls);
}

} // namespace

void GpuTopology::set(int a, int b, LinkClass c, int lanes)
{
    link[static_cast<std::size_t>(a)*n + b] = {c, lanes};
    link[static_cast<std::size_t>(b)*n + a] = {c, lanes};
}

void GpuTopology::validate() const
{
    if (n < 1 || link.size() != static_cast<std::size_t>(n)*n || static_cast<int>(socket.size()) != n)
        throw std::invalid_argument("GpuTopology: inconsistent sizes");
    for (int a = 0; a < n; ++a)
    {
        if (at(a, a).cls != LinkClass::Self) throw std::invalid_argument("GpuTopology: diagonal must be Self");
        for (int b = 0; b < n; ++b)
            if (at(a, b).cls != at(b, a).cls || at(a, b).lanes != at(b, a).lanes)
                throw std::invalid_argument("GpuTopology: link table not symmetric at " + std::to_string(a) + "," + std::to_string(b));
    }
}

GpuTopology uniformTopology(int n, LinkClass c, int lanes)
{
    GpuTopology t;
    t.n = n;
    t.link.assign(static_cast<std::size_t>(n)*n, LinkInfo{c, lanes});
    for (int a = 0; a < n; ++a) t.link[static_cast<std::size_t>(a)*n + a] = {LinkClass::Self, 0};
    t.socket.assign(n, 0);
    t.cpuList.assign(n, "");
    t.source = std::string("mock uniform ") + linkClassName(c);
    return t;
}

GpuTopology dgx1V100Topology()
{
    // `nvidia-smi topo -m` of a DGX-1V (hybrid cube-mesh; 6 NVLinks per GPU: two NV2 + two NV1 peers; the three GPUs of
    // the other quad that are not linked are SYS; GPUs 0-3 hang off socket 0, 4-7 off socket 1).
    static const char* rows[8] = {
        "X NV1 NV1 NV2 NV2 SYS SYS SYS",
        "NV1 X NV2 NV1 SYS NV2 SYS SYS",
        "NV1 NV2 X NV2 SYS SYS NV1 SYS",
        "NV2 NV1 NV2 X SYS SYS SYS NV1",
        "NV2 SYS SYS SYS X NV1 NV1 NV2",
        "SYS NV2 SYS SYS NV1 X NV2 NV1",
        "SYS SYS NV1 SYS NV1 NV2 X NV2",
        "SYS SYS SYS NV1 NV2 NV1 NV2 X"};
    GpuTopology t = uniformTopology(8, LinkClass::CrossSocket);
    for (int a = 0; a < 8; ++a)
    {
        std::istringstream is(rows[a]);
        std::string tok;
        for (int b = 0; b < 8; ++b)
        {
            is >> tok;
            if (tok == "X") continue;
            if (tok == "SYS") t.link[static_cast<std::size_t>(a)*8 + b] = {LinkClass::CrossSocket, 0};
            else t.link[static_cast<std::size_t>(a)*8 + b] = {LinkClass::NVLink, tok == "NV2" ? 2 : 1};
        }
        t.socket[a] = a < 4 ? 0 : 1;
        t.cpuList[a] = a < 4 ? "0-19,40-59" : "20-39,60-79";
    }
    t.source = "mock dgx1v";
    t.validate();
    return t;
}

GpuTopology dgxNvSwitchTopology(int n)
{
    GpuTopology t = uniformTopology(n, LinkClass::NVSwitch);
    for (int a = 0; a < n; ++a)
    {
        t.socket[a] = a < n/2 ? 0 : 1;
        t.cpuList[a] = a < n/2 ? "0-55,112-167" : "56-111,168-223";
    }
    t.source = "mock nvswitch";
    return t;
}

// ------------------------------------------------------------------------------------------------------------
// nvidia-smi topo -m
// ------------------------------------------------------------------------------------------------------------
namespace {

std::string trim(const std::string& s)
{
    std::size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

std::vector<std::string> splitTabs(const std::string& line)
{
    std::vector<std::string> v;
    std::string cur;
    for (char ch : line)
    {
        if (ch == '\t') { v.push_back(trim(cur)); cur.clear(); }
        else cur += ch;
    }
    v.push_back(trim(cur));
    return v;
}

// strip ANSI colour escapes (nvidia-smi colours the matrix on a terminal)
std::string stripAnsi(const std::string& s)
{
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        if (s[i] == '\x1b' && i + 1 < s.size() && s[i + 1] == '[')
        {
            while (i < s.size() && s[i] != 'm') ++i;
            continue;
        }
        o += s[i];
    }
    return o;
}

LinkInfo parseCell(const std::string& c)
{
    if (c == "X") return {LinkClass::Self, 0};
    if (c == "SYS") return {LinkClass::CrossSocket, 0};
    if (c == "NODE" || c == "PHB") return {LinkClass::PciHost, 0};
    if (c == "PXB" || c == "PIX") return {LinkClass::PciSwitch, 0};
    if (c.size() > 2 && c[0] == 'N' && c[1] == 'V')
    {
        const int lanes = std::atoi(c.c_str() + 2);
        if (lanes >= 12) return {LinkClass::NVSwitch, lanes};
        if (lanes >= 1) return {LinkClass::NVLink, lanes};
    }
    return {LinkClass::Unknown, 0};
}

} // namespace

GpuTopology parseNvidiaSmiTopo(const std::string& textIn)
{
    const std::string text = stripAnsi(textIn);
    std::istringstream is(text);
    std::string line;
    std::vector<std::string> header;
    while (std::getline(is, line))
    {
        std::vector<std::string> c = splitTabs(line);
        if (c.size() >= 2 && c[0].empty() && c[1] == "GPU0") { header = c; break; }
        if (c.size() >= 1 && c[0] == "GPU0" && c.size() >= 2 && c[1].rfind("GPU", 0) == 0) { header = c; break; }
    }
    if (header.empty()) throw std::invalid_argument("parseNvidiaSmiTopo: no GPU matrix header found");
    int n = 0, cpuCol = -1, numaCol = -1;
    for (std::size_t i = 0; i < header.size(); ++i)
    {
        if (header[i].rfind("GPU", 0) == 0 && header[i].size() > 3 && std::isdigit(static_cast<unsigned char>(header[i][3])) && cpuCol < 0 && numaCol < 0)
            ++n;
        if (header[i] == "CPU Affinity") cpuCol = static_cast<int>(i);
        if (header[i] == "NUMA Affinity") numaCol = static_cast<int>(i);
    }
    if (n < 1) throw std::invalid_argument("parseNvidiaSmiTopo: no GPU columns");
    GpuTopology t = uniformTopology(n, LinkClass::Unknown);
    int rows = 0;
    // the first GPU column of the header is cell index 1 (cell 0 is the empty corner), or 0 if the corner is absent
    const int off = header[0].empty() ? 1 : 0;
    while (std::getline(is, line) && rows < n)
    {
        std::vector<std::string> c = splitTabs(line);
        if (c.empty() || c[0].rfind("GPU", 0) != 0) continue;
        const int a = std::atoi(c[0].c_str() + 3);
        if (a < 0 || a >= n || static_cast<int>(c.size()) < n + 1) throw std::invalid_argument("parseNvidiaSmiTopo: malformed row '" + c[0] + "'");
        for (int b = 0; b < n; ++b)
        {
            const LinkInfo l = parseCell(c[1 + b]);
            if (l.cls == LinkClass::Unknown) throw std::invalid_argument("parseNvidiaSmiTopo: unknown link '" + c[1 + b] + "'");
            t.link[static_cast<std::size_t>(a)*n + b] = l;
        }
        const int cc = cpuCol - off + 1, nc = numaCol - off + 1;
        if (cpuCol >= 0 && cc < static_cast<int>(c.size())) t.cpuList[a] = c[cc];
        if (numaCol >= 0 && nc < static_cast<int>(c.size()) && !c[nc].empty() && std::isdigit(static_cast<unsigned char>(c[nc][0])))
            t.socket[a] = std::atoi(c[nc].c_str());
        ++rows;
    }
    if (rows != n) throw std::invalid_argument("parseNvidiaSmiTopo: expected " + std::to_string(n) + " GPU rows, got " + std::to_string(rows));
    // no NUMA column: sockets = classes of equal CPU affinity list
    if (numaCol < 0)
    {
        std::map<std::string, int> id;
        for (int a = 0; a < n; ++a) t.socket[a] = id.emplace(t.cpuList[a], static_cast<int>(id.size())).first->second;
    }
    t.source = "nvidia-smi";
    t.validate();
    return t;
}

GpuTopology probeTopology(bool useNvidiaSmi, std::string* note)
{
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev < 1) { cudaGetLastError(); throw std::runtime_error("probeTopology: no CUDA device"); }
    GpuTopology t = uniformTopology(nDev, LinkClass::CrossSocket);
    t.source = "cuda-p2p";
    for (int a = 0; a < nDev; ++a)
        for (int b = a + 1; b < nDev; ++b)
        {
            int can = 0, atomic = 0, rank = 0;
            cudaDeviceCanAccessPeer(&can, a, b);
            cudaDeviceGetP2PAttribute(&atomic, cudaDevP2PAttrNativeAtomicSupported, a, b);
            cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, a, b);
            cudaGetLastError();
            // native atomics over the link = NVLink; peer access without = PCIe P2P; none = host-staged
            if (can && atomic) t.set(a, b, LinkClass::NVLink, 1);
            else if (can) t.set(a, b, LinkClass::PciSwitch);
            else t.set(a, b, LinkClass::CrossSocket);
            (void)rank;
        }
    if (!useNvidiaSmi) return t;
    FILE* p = popen("nvidia-smi topo -m 2>/dev/null", "r");
    if (!p) { if (note) *note = "nvidia-smi not runnable: classes from cudaDeviceGetP2PAttribute only (NVLink lanes unknown, counted as 1)"; return t; }
    std::string out;
    char buf[4096];
    while (std::fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
    try
    {
        GpuTopology s = parseNvidiaSmiTopo(out);
        if (s.n != nDev) throw std::invalid_argument("nvidia-smi lists " + std::to_string(s.n) + " GPUs, CUDA sees " + std::to_string(nDev));
        // CUDA's device order is FASTEST_FIRST unless CUDA_DEVICE_ORDER=PCI_BUS_ID: flag a disagreement on NVLink-ness
        int disagree = 0;
        for (int a = 0; a < nDev; ++a)
            for (int b = a + 1; b < nDev; ++b)
                if (isNvLinkClass(s.at(a, b).cls) != isNvLinkClass(t.at(a, b).cls)) ++disagree;
        s.source = "nvidia-smi + cuda-p2p";
        if (note && disagree)
            *note = std::to_string(disagree) + " GPU pairs differ between nvidia-smi and CUDA about NVLink: set CUDA_DEVICE_ORDER=PCI_BUS_ID";
        return s;
    }
    catch (const std::exception& e)
    {
        if (note) *note = std::string("nvidia-smi topo -m not usable (") + e.what() + "): classes from cudaDeviceGetP2PAttribute only";
        return t;
    }
}

// ------------------------------------------------------------------------------------------------------------
// cost model and search
// ------------------------------------------------------------------------------------------------------------
double PlacementModel::pairUs(const LinkInfo& l, long long faces) const
{
    if (faces <= 0) return 0.0;
    LinkModel m;
    switch (l.cls)
    {
        case LinkClass::NVSwitch: m = nvswitch; break;
        case LinkClass::NVLink: m = {nvlinkGbpsPerLane*std::max(1, l.lanes), nvlinkLatencyUs}; break;
        case LinkClass::PciSwitch: m = pciSwitch; break;
        case LinkClass::PciHost: m = pciHost; break;
        case LinkClass::CrossSocket: m = crossSocket; break;
        default: m = unknown; break;
    }
    return m.latencyUs + static_cast<double>(faces)*bytesPerFace/(m.gbps*1.0e3);   // GB/s = 1e3 bytes/us
}

namespace {

struct Search
{
    int np, nGpu;
    const std::vector<long long>& Ws;                 // symmetrised W'
    const GpuTopology& topo;
    const PlacementModel& model;
    std::vector<int> cur, bestMap;
    std::vector<char> used;
    double bestCost = 1e300;
    int bestSlow = 1 << 30;                            // minimum slow pairs over all maps
    long long visited = 0;

    // us of rank pair (i,j) on gpus (a,b)
    double us(int i, int j, int a, int b) const { return model.pairUs(topo.at(a, b), Ws[static_cast<std::size_t>(i)*np + j]); }
    bool slow(int i, int j, int a, int b) const
    {
        return Ws[static_cast<std::size_t>(i)*np + j] > 0 && !isNvLinkClass(topo.at(a, b).cls);
    }

    void dfs(int r, double cost, int slowCount)
    {
        if (r == np)
        {
            ++visited;
            if (cost < bestCost*(1.0 - 1e-9) - 1e-12) { bestCost = cost; bestMap = cur; }   // strictly better only: lexicographic first wins ties
            bestSlow = std::min(bestSlow, slowCount);
            return;
        }
        for (int g = 0; g < nGpu; ++g)
        {
            if (used[g]) continue;
            double c = cost;
            int s = slowCount;
            for (int i = 0; i < r; ++i)
            {
                c += us(i, r, cur[i], g);
                s += slow(i, r, cur[i], g) ? 1 : 0;
            }
            used[g] = 1;
            cur[r] = g;
            dfs(r + 1, c, s);
            used[g] = 0;
        }
    }
};

} // namespace

PlacementResult placeRanks(const std::vector<long long>& W, int np, const GpuTopology& topo, const PlacementModel& model)
{
    topo.validate();
    if (np < 1 || np > topo.n) throw std::invalid_argument("placeRanks: need 1 <= np <= number of GPUs (np " + std::to_string(np) + ", GPUs " + std::to_string(topo.n) + ")");
    if (topo.n > 8) throw std::invalid_argument("placeRanks: exhaustive search is limited to 8 GPUs");
    if (W.size() != static_cast<std::size_t>(np)*np) throw std::invalid_argument("placeRanks: W must be np*np");
    std::vector<long long> Ws(static_cast<std::size_t>(np)*np, 0);
    for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j)
        {
            if (i == j) continue;
            if (W[static_cast<std::size_t>(i)*np + j] < 0) throw std::invalid_argument("placeRanks: negative cut weight");
            Ws[static_cast<std::size_t>(i)*np + j] = W[static_cast<std::size_t>(i)*np + j] + W[static_cast<std::size_t>(j)*np + i];
        }
    Search s{np, topo.n, Ws, topo, model, std::vector<int>(np), {}, std::vector<char>(topo.n, 0)};
    s.dfs(0, 0.0, 0);

    PlacementResult r;
    r.np = np;
    r.gpuOfRank = s.bestMap;
    r.costUs = s.bestCost;
    r.minSlowPairs = s.bestSlow;
    r.evaluated = s.visited;
    for (int i = 0; i < np; ++i)
        for (int j = i + 1; j < np; ++j)
        {
            r.identityCostUs += model.pairUs(topo.at(i, j), Ws[static_cast<std::size_t>(i)*np + j]);
            const long long f = Ws[static_cast<std::size_t>(i)*np + j];
            if (f <= 0) continue;
            PlacementPair p;
            p.rankA = i; p.rankB = j; p.faces = f;
            p.gpuA = r.gpuOfRank[i]; p.gpuB = r.gpuOfRank[j];
            p.link = topo.at(p.gpuA, p.gpuB);
            p.us = model.pairUs(p.link, f);
            if (!isNvLinkClass(p.link.cls)) ++r.slowPairs;
            r.pairs.push_back(p);
        }
    std::stable_sort(r.pairs.begin(), r.pairs.end(), [](const PlacementPair& a, const PlacementPair& b) { return a.faces > b.faces; });
    return r;
}

// ------------------------------------------------------------------------------------------------------------
// cut weights
// ------------------------------------------------------------------------------------------------------------
std::vector<long long> cutWeightsFromNeighbours(const std::vector<std::vector<std::pair<int, long long>>>& nbrFaces)
{
    const int np = static_cast<int>(nbrFaces.size());
    std::vector<long long> W(static_cast<std::size_t>(np)*np, 0);
    for (int r = 0; r < np; ++r)
        for (const auto& pr : nbrFaces[r])
        {
            if (pr.first < 0 || pr.first >= np || pr.first == r) throw std::invalid_argument("cutWeightsFromNeighbours: bad neighbour rank");
            W[static_cast<std::size_t>(r)*np + pr.first] += pr.second;
        }
    return W;
}

std::vector<long long> cutWeightsFromDescs(const std::vector<HaloPlanDesc>& descs)
{
    std::vector<std::vector<std::pair<int, long long>>> nf(descs.size());
    for (std::size_t r = 0; r < descs.size(); ++r)
        for (const HaloNeighbour& n : descs[r].nbrs) nf[r].emplace_back(n.rank, n.faceCount);
    return cutWeightsFromNeighbours(nf);
}

std::vector<long long> gatherCutWeights(ProcComm& comm, const HaloPlanDesc& mine)
{
    const int np = comm.size();
    std::vector<double> rows(static_cast<std::size_t>(np)*np, 0.0);   // exact: face counts << 2^53
    for (const HaloNeighbour& n : mine.nbrs) rows[static_cast<std::size_t>(comm.rank())*np + n.rank] += static_cast<double>(n.faceCount);
    comm.allReduceHost(rows.data(), np*np, ReduceOp::Sum);
    std::vector<long long> W(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) W[i] = static_cast<long long>(rows[i]);
    return W;
}

// ------------------------------------------------------------------------------------------------------------
// reports
// ------------------------------------------------------------------------------------------------------------
void printTopology(std::ostream& os, const GpuTopology& t)
{
    os << "GPU link table (" << t.source << "), sockets/CPU affinity:\n      ";
    for (int b = 0; b < t.n; ++b) os << std::setw(5) << ("GPU" + std::to_string(b));
    os << "   socket  cpus\n";
    for (int a = 0; a < t.n; ++a)
    {
        os << std::setw(5) << ("GPU" + std::to_string(a)) << ' ';
        for (int b = 0; b < t.n; ++b) os << std::setw(5) << linkLabel(t.at(a, b));
        os << "   " << std::setw(6) << t.socket[a] << "  " << t.cpuList[a] << "\n";
    }
}

std::string gpuMapString(const PlacementResult& r)
{
    std::string s;
    for (std::size_t i = 0; i < r.gpuOfRank.size(); ++i) s += (i ? "," : "") + std::to_string(r.gpuOfRank[i]);
    return s;
}

void printPlacement(std::ostream& os, const PlacementResult& r, const GpuTopology& t, const std::vector<long long>& W)
{
    const int np = r.np;
    os << "cut weights W (processor faces shared by rank i and rank j):\n      ";
    for (int j = 0; j < np; ++j) os << std::setw(8) << ("r" + std::to_string(j));
    os << "\n";
    for (int i = 0; i < np; ++i)
    {
        os << std::setw(5) << ("r" + std::to_string(i)) << ' ';
        for (int j = 0; j < np; ++j) os << std::setw(8) << (i == j ? 0 : W[static_cast<std::size_t>(i)*np + j]);
        os << "\n";
    }
    os << "placement (" << r.evaluated << " maps searched, exhaustive, deterministic lexicographic tie-break):\n";
    os << "  rank -> GPU:";
    for (int i = 0; i < np; ++i) os << ' ' << i << "->" << r.gpuOfRank[i];
    os << "\n  BRAE_GPU_MAP=" << gpuMapString(r) << "\n";
    os << std::fixed << std::setprecision(1);
    os << "  modelled exchange cost " << r.costUs << " us (identity rank r -> GPU r: " << r.identityCostUs << " us)\n";
    os << "  communicating rank pairs (heaviest first):\n";
    os << "    ranks    faces   GPUs   link   us\n";
    for (const PlacementPair& p : r.pairs)
        os << "    " << std::setw(2) << p.rankA << "-" << std::left << std::setw(2) << p.rankB << std::right << std::setw(8) << p.faces << "   "
           << p.gpuA << "-" << p.gpuB << "   " << std::setw(4) << linkLabel(p.link) << "  " << p.us << (isNvLinkClass(p.link.cls) ? "" : "   <-- not NVLink") << "\n";
    if (r.slowPairs == 0)
        os << "  every communicating pair is on NVLink\n";
    else
    {
        os << "  " << r.slowPairs << " communicating pair(s) are NOT on NVLink (PCIe / host-staged); no placement can do better than "
           << r.minSlowPairs << " (" << (r.slowPairs == r.minSlowPairs ? "so these hops are UNAVOIDABLE for this decomposition on this GPU graph"
                                                                         : "this placement trades pair COUNT for modelled time: the heavy pairs are on NVLink")
           << ")\n";
        if (t.n == 8 && np == 8)
            os << "  hint: `hierarchical` decomposition with n (2 2 2) gives a cube rank graph that embeds in the DGX-1 NVLink cube-mesh\n";
    }
}

BindingAdvice bindingAdvice(const PlacementResult& r, const GpuTopology& t, const std::string& program)
{
    BindingAdvice a;
    std::ostringstream os;
    os << "CPU/socket binding (each rank next to its GPU's socket, so its launch thread and pinned memory are local):\n";
    std::set<int> sockets;
    for (int i = 0; i < r.np; ++i)
    {
        const int g = r.gpuOfRank[i];
        a.gpu.push_back(g);
        a.socket.push_back(t.socket[g]);
        a.cpuList.push_back(t.cpuList[g]);
        sockets.insert(t.socket[g]);
        os << "  rank " << i << ": GPU " << g << ", socket " << t.socket[g] << (t.cpuList[g].empty() ? "" : ", cpus " + t.cpuList[g]) << "\n";
    }
    os << "  [one process per GPU; do NOT use mpirun's default bind-to-core with this map: use --bind-to none and the wrapper]\n";
    os << "  mpirun --bind-to none -np " << r.np << " -x BRAE_GPU_MAP=" << gpuMapString(r) << " bash -c '\n";
    os << "    case $OMPI_COMM_WORLD_RANK in\n";
    for (int i = 0; i < r.np; ++i)
        os << "      " << i << ") exec numactl --cpunodebind=" << a.socket[i] << " --membind=" << a.socket[i] << " \"$@\";;\n";
    os << "    esac' _ " << program << "\n";
    os << "  srun equivalent: srun -N1 -n " << r.np << " --gpu-bind=none --cpu-bind=none --export=ALL,BRAE_GPU_MAP=" << gpuMapString(r)
       << " <same wrapper with $SLURM_PROCID>\n";
    if (sockets.size() > 1 && r.np <= static_cast<int>(t.n)/static_cast<int>(sockets.size()))
        os << "  note: np <= GPUs of one socket, but this placement straddles " << sockets.size()
           << " sockets (the cut weights decided it); fewer ranks per quad is only better when the weights say so\n";
    a.text = os.str();
    return a;
}

} // namespace brae
