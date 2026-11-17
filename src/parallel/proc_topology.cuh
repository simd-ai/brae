#pragma once
// brae topology-aware rank -> GPU placement (CommSolvers.md step D, section 1.4).
//
// Input: the per-rank cut-weight matrix W (W[i*np+j] = processor faces rank i shares with rank j, i.e. the halo
// traffic of the decomposition; every rank can build it by allgathering the face counts of its processor patches,
// see cutWeightsFromDescs / gatherCutWeights) and the physical GPU link table (GpuTopology: mocked, parsed from
// `nvidia-smi topo -m`, or probed with cudaDeviceGetP2PAttribute).
// Output: gpuOfRank[r], the GPU rank r must use, minimising
//     sum over rank pairs i<j with W'[i][j] > 0 of  latency(link) + W'[i][j] * bytesPerFace / bandwidth(link),
//     W' = W + W^T (both directions of the exchange),
// found by exhaustive search over the injective maps rank -> GPU (np <= nGpu <= 8: at most 8! = 40320 maps) with a
// DETERMINISTIC tie-break: among maps of equal cost (relative 1e-9) the lexicographically smallest gpuOfRank wins,
// so every rank that runs it on the same input gets the same answer.
//
// How it is used: the launcher/bootstrap runs placeRanks() BEFORE any CUDA context exists (tools/proc_topology
// prints `BRAE_GPU_MAP=...`; Pstream::init reads BRAE_GPU_MAP and binds rank (node-local) i to GPU map[i] with
// cudaSetDevice before it creates anything). Processor directories are indexed by MPI rank, so no relabelling of the
// decomposition is needed. The numbers of the cost model (bandwidths/latencies per link class) are [INFERENCE]
// from NVLink/PCIe nominal figures; override them in PlacementModel when measured data exists.
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace brae {

struct HaloPlanDesc;
class ProcComm;

// Link between two GPUs, in the vocabulary of `nvidia-smi topo -m`.
enum class LinkClass
{
    Self,         // X
    NVSwitch,     // NVLink through NVSwitch (DGX-H100/A100): every pair equal
    NVLink,       // NV# direct NVLink lanes (DGX-1: NV1 / NV2)
    PciSwitch,    // PIX / PXB (or CUDA P2P over PCIe): peer access, never leaves the PCIe switch/root
    PciHost,      // PHB / NODE: through the host bridge of one socket
    CrossSocket,  // SYS: no peer access, host-staged through the SMP interconnect (NCCL SHM)
    Unknown
};
const char* linkClassName(LinkClass c);     // "X" "NVS" "NV" "PIX" "PHB" "SYS" "?"
bool isNvLinkClass(LinkClass c);            // NVSwitch or NVLink

struct LinkInfo
{
    LinkClass cls = LinkClass::Unknown;
    int lanes = 0;                          // NVLink lanes for LinkClass::NVLink (NV1 -> 1, NV2 -> 2)
};

struct GpuTopology
{
    int n = 0;                              // GPUs
    std::vector<LinkInfo> link;             // [a*n + b], symmetric, link[a*n+a] = Self
    std::vector<int> socket;                // [gpu] CPU socket / NUMA node the GPU hangs off (0 when unknown)
    std::vector<std::string> cpuList;       // [gpu] CPU affinity list as printed by nvidia-smi ("0-19,40-59"), may be empty
    std::string source;                     // "mock dgx1v" | "nvidia-smi" | "cuda-p2p" | ...

    const LinkInfo& at(int a, int b) const { return link[static_cast<std::size_t>(a)*n + b]; }
    void set(int a, int b, LinkClass c, int lanes = 0);   // symmetric
    void validate() const;                                // throws std::invalid_argument
};

// Mocked / built-in tables.
GpuTopology dgx1V100Topology();             // 8 V100, hybrid cube-mesh: 6 NVLinks/GPU (NV1/NV2), SYS across the quads, sockets {0-3},{4-7}
GpuTopology dgxNvSwitchTopology(int n = 8); // DGX-H100/A100: NVSwitch between all pairs, 2 sockets
GpuTopology uniformTopology(int n, LinkClass c, int lanes = 0);
// Parse the text of `nvidia-smi topo -m` (tab separated cells; GPU columns first). Throws std::invalid_argument when
// no GPU matrix is found. NIC columns are ignored; NVx with x >= 12 is read as NVSwitch.
GpuTopology parseNvidiaSmiTopo(const std::string& text);
// Probe the visible GPUs: cudaDeviceCanAccessPeer + cudaDeviceGetP2PAttribute (NativeAtomicSupported = NVLink,
// PerformanceRank), then, when `nvidia-smi` runs and reports the same GPU count, refine the classes and lanes and
// read the CPU affinity from `nvidia-smi topo -m`. Needs no CUDA context of its own beyond the attribute queries
// (they may create one on each device: call it from the launcher, not from a rank).
GpuTopology probeTopology(bool useNvidiaSmi = true, std::string* note = nullptr);

// Cost model: time of one halo exchange of a rank pair over a link (us).
struct LinkModel { double gbps; double latencyUs; };
struct PlacementModel
{
    double bytesPerFace = 64.0;             // payload per processor face and exchange (8 doubles: a few batched fields)
    LinkModel nvswitch{450.0, 5.0};         // NVLink4 x18 lanes
    double nvlinkGbpsPerLane = 25.0;        // NVLink2 lane, one direction
    double nvlinkLatencyUs = 5.0;
    LinkModel pciSwitch{10.0, 10.0};
    LinkModel pciHost{8.0, 15.0};
    LinkModel crossSocket{6.0, 35.0};       // host-staged: 1/6 GB/s + 30 us over the PCIe path
    LinkModel unknown{6.0, 35.0};

    // microseconds of one exchange of `faces` faces over link l (0 for faces == 0)
    double pairUs(const LinkInfo& l, long long faces) const;
};

struct PlacementPair
{
    int rankA = 0, rankB = 0;               // rankA < rankB
    long long faces = 0;                    // W'[a][b]
    int gpuA = 0, gpuB = 0;
    LinkInfo link;
    double us = 0;
};

struct PlacementResult
{
    int np = 0;
    std::vector<int> gpuOfRank;             // the answer
    double costUs = 0;                      // objective of gpuOfRank
    double identityCostUs = 0;              // objective of rank r -> GPU r (what Pstream does without BRAE_GPU_MAP)
    int slowPairs = 0;                      // communicating rank pairs of gpuOfRank NOT on NVLink/NVSwitch
    int minSlowPairs = 0;                   // the minimum of that number over ALL placements: the unavoidable PCIe/SYS hops
    long long evaluated = 0;                // maps visited
    std::vector<PlacementPair> pairs;       // communicating pairs of gpuOfRank, heaviest first (ties: lower ranks first)
};

// Exhaustive deterministic search. W is np*np (row-major), non-negative; the diagonal is ignored. np <= topo.n <= 8.
PlacementResult placeRanks(const std::vector<long long>& W, int np, const GpuTopology& topo, const PlacementModel& m = {});

// W from the plans of ALL ranks (tests, loopback, a single process that read every processorN).
std::vector<long long> cutWeightsFromDescs(const std::vector<HaloPlanDesc>& descs);
// W from the face counts of ranks' processor patches: nbrFaces[r] = {(neighbour rank, faces)...}.
std::vector<long long> cutWeightsFromNeighbours(const std::vector<std::vector<std::pair<int, long long>>>& nbrFaces);
// COLLECTIVE: every rank passes its own plan; returns the full W on every rank (host allReduce of the rows).
std::vector<long long> gatherCutWeights(ProcComm& comm, const HaloPlanDesc& mine);

// ---- reports -----------------------------------------------------------------------------------------------------
void printTopology(std::ostream& os, const GpuTopology& t);                                    // the link table
void printPlacement(std::ostream& os, const PlacementResult& r, const GpuTopology& t, const std::vector<long long>& W);
std::string gpuMapString(const PlacementResult& r);                                            // "3,0,1,2" for BRAE_GPU_MAP

// CPU / socket binding advice: per rank the GPU, socket, CPU list and a numactl prefix, plus an OpenMPI-style wrapper.
struct BindingAdvice
{
    std::vector<int> gpu, socket;
    std::vector<std::string> cpuList;
    std::string text;                       // human readable block incl. the launch lines
};
BindingAdvice bindingAdvice(const PlacementResult& r, const GpuTopology& t, const std::string& program = "./brae_interFoam -parallel");

} // namespace brae
