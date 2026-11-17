// brae::ProcComm: backends (loopback, MPI-host, NCCL), halo plans, synthetic partitioner.
// The contract is documented in proc_comm.cuh.
#include "proc_comm.cuh"
#include "proc_hooks.cuh"   // procGraphDestroyAll: captured solver graphs die before the communicators

#include <mpi.h>
#include <unistd.h>
#ifdef BRAE_WITH_NCCL
#include <nccl.h>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace brae {

namespace {

// ------------------------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------------------------
std::atomic<std::size_t> g_allocs{0};
ProcComm* g_current = nullptr;
thread_local ProcComm* t_current = nullptr;
// ------------------------------------------------------------------------------------------------------------
// Failure propagation shared by the MPI-host and NCCL backends.
//
// The blocking primitive of a point-to-point exchange is a MPI_Waitall: a rank that fails AFTER the opening
// status phase would leave its peers in that wait for ever. Two mechanisms close the window, and the no-failure
// hot path pays only an atomic load:
//  * every local, fallible step of an exchange runs BEFORE any message is posted and its outcome is carried by a
//    status allgather, so a local failure aborts all ranks BEFORE anyone waits on a message;
//  * the nonblocking wait itself is bounded (watchdog) and a separate duplicated "side" communicator carries a
//    continuous nonblocking allgather of the abort record, so a peer that is stuck in a device synchronisation
//    still learns the abort and (NCCL) aborts its communicators, which releases its NCCL kernels.
// The side channel is driven by one background thread per process; it needs MPI_THREAD_MULTIPLE (Pstream::init
// requests it). Without that level the thread only polls/aborts NCCL and the main thread drives the side channel.
struct AbortState
{
    std::mutex m;
    char msg[248] = {0};        // published abort message (the first one wins)
    int origin = -1;            // rank that raised it
    std::atomic<bool> raised{false};     // a LOCAL rank called collectiveAbort
    std::atomic<bool> pending{false};    // an abort is known (local or remote); ops must throw
    std::atomic<bool> ncclAbort{false};  // ask the NCCL watchdog to abort both communicators
    std::atomic<bool> stop{false};       // backend is being destroyed

    void publish(int rank, const std::string& text)
    {
        {
            std::lock_guard<std::mutex> lk(m);
            if (origin < 0)
            {
                origin = rank;
                std::strncpy(msg, text.c_str(), sizeof(msg) - 1);
            }
        }
        pending.store(true, std::memory_order_release);
    }
    // Lowest raising rank wins, exactly like the MPI status allgather. Called by the side-channel thread.
    void publishRemote(int rank, const char* text)
    {
        {
            std::lock_guard<std::mutex> lk(m);
            if (origin < 0 || rank < origin)
            {
                origin = rank;
                std::strncpy(msg, text, sizeof(msg) - 1);
            }
        }
        pending.store(true, std::memory_order_release);
    }
    std::pair<int, std::string> get() const
    {
        std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(m));
        return {origin, msg};
    }
};

int watchdogMs(const char* env, int fallback)
{
    const char* e = std::getenv(env);
    if (!e) return fallback;
    const int v = std::atoi(e);
    return v < 0 ? fallback : v;
}

void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ProcComm cuda: ") + what + ": " + cudaGetErrorString(e));
}

template<class T> T* dalloc(std::size_t n)
{
    if (n == 0) return nullptr;
    T* p = nullptr;
    ck(cudaMalloc(&p, n*sizeof(T)), "cudaMalloc");
    ++g_allocs;
    return p;
}

template<class T> T* dupload(const std::vector<T>& v)
{
    T* p = dalloc<T>(v.size());
    if (p)
    {
        // pageable H2D may return before the DMA is done: complete it before another stream can read the memory
        ck(cudaMemcpyAsync(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice, cudaStreamPerThread), "upload");
        ck(cudaStreamSynchronize(cudaStreamPerThread), "upload sync");
    }
    return p;
}

constexpr int kMaxComps = 16;      // components per pack/unpack launch (kernel parameter space)
constexpr int kMaxTotalComps = 64; // components per exchange
constexpr int kMaxRanks = 64;      // loopback ranks (kernel parameter space of the fold)

// Pack/unpack descriptors. Staging layout of a cell exchange, neighbour-major: the data of neighbour p with
// `count` cells occupies [C*off, C*(off+count)), component c of element j at C*off + c*count + j.
struct SendEntry { int off, count; };
struct RecvEntry { const double* src; int srcOff, count, dstSlot; };
struct FaceEntry { const double* src; int srcOff, count, start; };

struct CompList
{
    double* p[kMaxComps];
    long long cs[kMaxComps];
    int n;      // components in this launch
    int base;   // global index of component 0 of this launch
};

struct SlotList { const double* p[kMaxRanks]; int n; };

__global__ void packCellsK(const int* sendCells, const SendEntry* tab, CompList cl, int C, double* out)
{
    const SendEntry e = tab[blockIdx.y];
    const long long total = static_cast<long long>(e.count)*cl.n;
    for (long long t = static_cast<long long>(blockIdx.x)*blockDim.x + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x)*blockDim.x)
    {
        const int c = static_cast<int>(t / e.count);
        const int j = static_cast<int>(t - static_cast<long long>(c)*e.count);
        out[static_cast<long long>(C)*e.off + static_cast<long long>(cl.base + c)*e.count + j] =
            cl.p[c][static_cast<long long>(sendCells[e.off + j])*cl.cs[c]];
    }
}

__global__ void unpackCellsK(const RecvEntry* tab, CompList cl, int C, int nC)
{
    const RecvEntry e = tab[blockIdx.y];
    const long long total = static_cast<long long>(e.count)*cl.n;
    for (long long t = static_cast<long long>(blockIdx.x)*blockDim.x + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x)*blockDim.x)
    {
        const int c = static_cast<int>(t / e.count);
        const int j = static_cast<int>(t - static_cast<long long>(c)*e.count);
        cl.p[c][static_cast<long long>(nC + e.dstSlot + j)*cl.cs[c]] =
            e.src[static_cast<long long>(C)*e.srcOff + static_cast<long long>(cl.base + c)*e.count + j];
    }
}

__global__ void combineFacesK(const FaceEntry* tab, double* arr, int op)
{
    const FaceEntry e = tab[blockIdx.y];
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < e.count; i += gridDim.x*blockDim.x)
    {
        const double s = e.src[e.srcOff + i];
        double& a = arr[e.start + i];
        switch (op)
        {
            case 0: a = s; break;
            case 1: a = s < a ? s : a; break;
            case 2: a = s > a ? s : a; break;
            default: a = a + s; break;
        }
    }
}

// op: 0 sum, 1 min, 2 max, 3 bcast of slot `root`. Rank order 0..n-1: the same bits on every rank.
__global__ void foldSlotsK(double* out, SlotList sl, int n, int op, int root)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    {
        if (op == 3) { out[i] = sl.p[root][i]; continue; }
        double acc = sl.p[0][i];
        for (int r = 1; r < sl.n; ++r)
        {
            const double v = sl.p[r][i];
            if (op == 0) acc = acc + v;
            else if (op == 1) acc = v < acc ? v : acc;
            else acc = v > acc ? v : acc;
        }
        out[i] = acc;
    }
}

int opCode(ReduceOp op) { return op == ReduceOp::Sum ? 0 : op == ReduceOp::Min ? 1 : 2; }

void foldHost(double* out, const std::vector<const double*>& s, int n, int op, int root)
{
    for (int i = 0; i < n; ++i)
    {
        if (op == 3) { out[i] = s[root][i]; continue; }
        double acc = s[0][i];
        for (std::size_t r = 1; r < s.size(); ++r)
        {
            const double v = s[r][i];
            if (op == 0) acc = acc + v;
            else if (op == 1) acc = v < acc ? v : acc;
            else acc = v > acc ? v : acc;
        }
        out[i] = acc;
    }
}

int faceOpCode(FaceOp op)
{
    switch (op) { case FaceOp::Replace: return 0; case FaceOp::Min: return 1; case FaceOp::Max: return 2; default: return 3; }
}

struct FlatComps
{
    double* p[kMaxTotalComps];
    long long cs[kMaxTotalComps];
    int n = 0;
};

void flatten(const HaloField* f, int nFields, FlatComps& out)
{
    out.n = 0;
    for (int i = 0; i < nFields; ++i)
    {
        if (f[i].nComp < 1) throw std::invalid_argument("ProcComm: HaloField with nComp < 1");
        if (f[i].nComp > 1 && f[i].compStride == 0) throw std::invalid_argument("ProcComm: HaloField with nComp > 1 needs compStride");
        for (int k = 0; k < f[i].nComp; ++k)
        {
            if (out.n >= kMaxTotalComps) throw std::length_error("ProcComm: more than 64 components in one exchange");
            out.p[out.n] = f[i].data + static_cast<long long>(k)*f[i].compStride;
            out.cs[out.n] = f[i].cellStride;
            ++out.n;
        }
    }
}

void fillList(CompList& cl, const FlatComps& fc, int c0)
{
    cl.n = std::min(kMaxComps, fc.n - c0);
    cl.base = c0;
    for (int k = 0; k < cl.n; ++k) { cl.p[k] = fc.p[c0 + k]; cl.cs[k] = fc.cs[c0 + k]; }
}

int gridFor(long long total)
{
    return static_cast<int>(std::min<long long>((total + 255)/256, 1 << 20));
}

// Backend-independent device-side staging of a plan: index tables and the staging buffers.
struct StagePlan : PlanState
{
    int maxComps = 0;
    std::size_t sendCap = 0, recvCap = 0;      // doubles
    double* dSend = nullptr;                   // pack target (peers read it on loopback)
    double* dRecv = nullptr;                   // MPI: unpack source
    SendEntry* dSendTab = nullptr;
    RecvEntry* dRecvTab = nullptr;
    FaceEntry* dFaceTab = nullptr;
    int nNbr = 0, maxSend = 0, maxRecv = 0, maxFace = 0;
    const HaloPlanDesc* desc = nullptr;

    ~StagePlan() override
    {
        cudaFree(dSend); cudaFree(dRecv); cudaFree(dSendTab); cudaFree(dRecvTab); cudaFree(dFaceTab);
    }

    void init(const HaloPlanDesc& d, int maxC, bool wantRecv)
    {
        desc = &d;
        maxComps = maxC;
        nNbr = static_cast<int>(d.nbrs.size());
        sendCap = std::max<std::size_t>(static_cast<std::size_t>(maxC)*d.sendCells.size(), d.nProcFaces);
        recvCap = std::max<std::size_t>(static_cast<std::size_t>(maxC)*d.nHalo, d.nProcFaces);
        dSend = dalloc<double>(sendCap);
        if (wantRecv) dRecv = dalloc<double>(recvCap);
        std::vector<SendEntry> st(nNbr);
        for (int k = 0; k < nNbr; ++k)
        {
            st[k] = {d.nbrs[k].sendOffset, d.nbrs[k].sendCount};
            maxSend = std::max(maxSend, d.nbrs[k].sendCount);
            maxRecv = std::max(maxRecv, d.nbrs[k].recvCount);
            maxFace = std::max(maxFace, d.nbrs[k].faceCount);
        }
        dSendTab = dupload(st);
    }

    // rtab/ftab: per neighbour, filled by the backend (the source buffers differ).
    void setTables(const std::vector<RecvEntry>& rt, const std::vector<FaceEntry>& ft)
    {
        cudaFree(dRecvTab); cudaFree(dFaceTab);
        dRecvTab = dupload(rt);
        dFaceTab = dupload(ft);
    }

    void pack(const HaloPlan& plan, const FlatComps& fc, cudaStream_t s) const
    {
        if (nNbr == 0 || maxSend == 0) return;
        for (int c0 = 0; c0 < fc.n; c0 += kMaxComps)
        {
            CompList cl; fillList(cl, fc, c0);
            packCellsK<<<dim3(gridFor(static_cast<long long>(maxSend)*cl.n), nNbr), 256, 0, s>>>(
                plan.dSendCells(), dSendTab, cl, fc.n, dSend);
        }
        ck(cudaGetLastError(), "packCells launch");
    }

    void unpack(const HaloPlan& plan, const FlatComps& fc, cudaStream_t s) const
    {
        if (nNbr == 0 || maxRecv == 0) return;
        for (int c0 = 0; c0 < fc.n; c0 += kMaxComps)
        {
            CompList cl; fillList(cl, fc, c0);
            unpackCellsK<<<dim3(gridFor(static_cast<long long>(maxRecv)*cl.n), nNbr), 256, 0, s>>>(
                dRecvTab, cl, fc.n, plan.nCells());
        }
        ck(cudaGetLastError(), "unpackCells launch");
    }

    void combineFaces(double* arr, FaceOp op, cudaStream_t s) const
    {
        if (nNbr == 0 || maxFace == 0) return;
        combineFacesK<<<dim3(gridFor(maxFace), nNbr), 256, 0, s>>>(dFaceTab, arr, faceOpCode(op));
        ck(cudaGetLastError(), "combineFaces launch");
    }
};

void checkCapacity(const StagePlan& st, int C)
{
    if (C > st.maxComps)
        throw std::length_error("ProcComm: exchange of " + std::to_string(C) + " components exceeds the plan's maxComps "
                                + std::to_string(st.maxComps));
}

} // namespace

namespace detail {
struct PlanAccess
{
    static PlanState* state(const HaloPlan& p) { return p.state_.get(); }
    static void set(HaloPlan& p, std::unique_ptr<PlanState> s) { p.state_ = std::move(s); }
    static const ProcComm* owner(const HaloPlan& p) { return p.owner_; }
    static const HaloPlanDesc& desc(const HaloPlan& p) { return p.desc_; }
    static void upload(HaloPlan& p)
    {
        const HaloPlanDesc& d = p.desc_;
        p.dSendCells_ = dupload(d.sendCells);
        p.dFaceCells_ = dupload(d.faceCells);
        std::vector<int> fh(d.faceToSlot.size());
        for (std::size_t i = 0; i < fh.size(); ++i) fh[i] = d.nCells + d.faceToSlot[i];
        p.dFaceHalo_ = dupload(fh);
    }
};
} // namespace detail

// ------------------------------------------------------------------------------------------------------------
// HaloPlanDesc, builder
// ------------------------------------------------------------------------------------------------------------
void HaloPlanDesc::validate() const
{
    auto bad = [](const std::string& m) { throw std::invalid_argument("HaloPlanDesc: " + m); };
    if (nCells < 0 || nHalo < 0 || nProcFaces < 0) bad("negative size");
    int so = 0, rs = 0, fs = 0;
    for (std::size_t k = 0; k < nbrs.size(); ++k)
    {
        const HaloNeighbour& n = nbrs[k];
        if (n.rank < 0) bad("negative neighbour rank");
        for (std::size_t j = 0; j < k; ++j) if (nbrs[j].rank == n.rank) bad("neighbour rank listed twice");
        if (n.sendCount < 0 || n.recvCount < 0 || n.faceCount < 0) bad("negative count");
        if (n.sendOffset != so || n.recvStart != rs || n.faceStart != fs) bad("ranges are not contiguous in neighbour order");
        so += n.sendCount; rs += n.recvCount; fs += n.faceCount;
    }
    if (static_cast<std::size_t>(so) != sendCells.size()) bad("sendCells size != sum of sendCount");
    if (rs != nHalo) bad("sum of recvCount != nHalo");
    if (fs != nProcFaces) bad("sum of faceCount != nProcFaces");
    if (faceCells.size() != static_cast<std::size_t>(nProcFaces) || faceToSlot.size() != static_cast<std::size_t>(nProcFaces))
        bad("faceCells/faceToSlot size != nProcFaces");
    for (int c : sendCells) if (c < 0 || c >= nCells) bad("sendCells entry out of range");
    for (int c : faceCells) if (c < 0 || c >= nCells) bad("faceCells entry out of range");
    for (const HaloNeighbour& n : nbrs)
        for (int i = n.faceStart; i < n.faceStart + n.faceCount; ++i)
            if (faceToSlot[i] < n.recvStart || faceToSlot[i] >= n.recvStart + n.recvCount)
                bad("faceToSlot outside its neighbour's slot range");
}

HaloPlanDesc buildHaloPlanFromProcessorPatches(int nCells, const std::vector<ProcessorPatchDesc>& patches)
{
    HaloPlanDesc d;
    d.nCells = nCells;
    for (const ProcessorPatchDesc& p : patches)
    {
        if (p.faceCells.size() != p.nbrFaceCells.size())
            throw std::invalid_argument("buildHaloPlanFromProcessorPatches: faceCells and nbrFaceCells differ in length "
                                        "(fetch the neighbour's face cells first)");
        HaloNeighbour n;
        n.rank = p.nbrRank;
        n.sendOffset = static_cast<int>(d.sendCells.size());
        n.recvStart = d.nHalo;
        n.faceStart = d.nProcFaces;
        n.faceCount = static_cast<int>(p.faceCells.size());
        std::unordered_map<int, int> sent, slotOf;
        for (std::size_t i = 0; i < p.faceCells.size(); ++i)
        {
            const int mine = p.faceCells[i];
            if (sent.emplace(mine, static_cast<int>(sent.size())).second) d.sendCells.push_back(mine);
            const auto it = slotOf.emplace(p.nbrFaceCells[i], static_cast<int>(slotOf.size())).first;
            d.faceCells.push_back(mine);
            d.faceToSlot.push_back(n.recvStart + it->second);
        }
        n.sendCount = static_cast<int>(sent.size());
        n.recvCount = static_cast<int>(slotOf.size());
        d.nHalo += n.recvCount;
        d.nProcFaces += n.faceCount;
        d.nbrs.push_back(n);
    }
    d.validate();
    return d;
}

// ------------------------------------------------------------------------------------------------------------
// HaloPlan
// ------------------------------------------------------------------------------------------------------------
HaloPlan::HaloPlan(const ProcComm* owner, HaloPlanDesc d) : owner_(owner), desc_(std::move(d)) {}

HaloPlan::~HaloPlan()
{
    state_.reset();
    cudaFree(dSendCells_);
    cudaFree(dFaceCells_);
    cudaFree(dFaceHalo_);
}

// ------------------------------------------------------------------------------------------------------------
// ProcComm base
// ------------------------------------------------------------------------------------------------------------
ProcComm::ProcComm(Backend b, int rank, int size) : rank_(rank), size_(size), backend_(b) {}
ProcComm::~ProcComm()
{
    procGraphDestroyAll(this);          // graphs captured on this comm's graph channel die BEFORE its communicators
    if (g_current == this) g_current = nullptr;
    if (t_current == this) t_current = nullptr;
}

ProcComm* ProcComm::current() { return t_current ? t_current : g_current; }
void ProcComm::setCurrent(ProcComm* c) { g_current = c; }
void ProcComm::setThreadCurrent(ProcComm* c) { t_current = c; }
std::size_t ProcComm::allocationCount() { return g_allocs.load(); }

void ProcComm::bindThread()
{
    if (device_ >= 0) ck(cudaSetDevice(device_), "cudaSetDevice");
    setThreadCurrent(this);
}

std::unique_ptr<HaloPlan> ProcComm::newPlan(HaloPlanDesc desc)
{
    desc.validate();
    std::unique_ptr<HaloPlan> p(new HaloPlan(this, std::move(desc)));
    detail::PlanAccess::upload(*p);
    return p;
}

PlanState* ProcComm::planState(const HaloPlan& p) { return detail::PlanAccess::state(p); }
void ProcComm::setPlanState(HaloPlan& p, std::unique_ptr<PlanState> s) { detail::PlanAccess::set(p, std::move(s)); }

void ProcComm::checkPlan(const HaloPlan& p) const
{
    if (detail::PlanAccess::owner(p) != this)
        throw std::logic_error("ProcComm: HaloPlan used with a communicator other than the one that made it");
}

std::unique_ptr<HaloPlan> ProcComm::makePlan(HaloPlanDesc desc, int maxComps)
{
    if (maxComps < 1 || maxComps > kMaxTotalComps) throw std::invalid_argument("ProcComm::makePlan: maxComps must be 1..64");
    if (size_ == 1)
    {
        if (!desc.nbrs.empty() || desc.nHalo != 0)
            throw std::invalid_argument("ProcComm::makePlan: np == 1 cannot have neighbours");
        desc.validate();
        return std::unique_ptr<HaloPlan>(new HaloPlan(this, std::move(desc)));   // no device memory
    }
    return doMakePlan(std::move(desc), maxComps);
}

std::unique_ptr<HaloPlan> ProcComm::buildPlanFromPatches(int nCells, std::vector<ProcessorPatchDesc> patches, int maxComps)
{
    // Participation and sizes are agreed BEFORE any point-to-point exchange, empty patches included: a zero-face
    // patch on one side with a nonzero one on the other is a decomposePar/input error, and it must abort every
    // rank with the same message instead of letting a peer wait for a message that will never be sent (finding 9).
    if (size_ > 1)
    {
        std::vector<int> part(static_cast<std::size_t>(size_)*size_, 0);
        for (const ProcessorPatchDesc& p : patches)
        {
            if (p.nbrRank < 0 || p.nbrRank >= size_)
                collectiveAbort("ProcComm::buildPlanFromPatches: neighbour rank out of range");
            part[static_cast<std::size_t>(rank_)*size_ + p.nbrRank] = static_cast<int>(p.faceCells.size()) + 1;
            // +1 distinguishes "no patch" from "a patch with zero faces"
        }
        // allgather through the backend's own (bounded, abort-aware) path: loopback answers locally, MPI-host
        // and NCCL run the bounded nonblocking allgather on the status communicator.
        std::vector<int> matrix;
        allgatherInts(part, matrix);
        std::string bad;
        for (int a = 0; a < size_ && bad.empty(); ++a)
            for (int b = 0; b < size_; ++b)
            {
                const int ab = matrix[static_cast<std::size_t>(a)*part.size() + static_cast<std::size_t>(a)*size_ + b];
                const int ba = matrix[static_cast<std::size_t>(b)*part.size() + static_cast<std::size_t>(b)*size_ + a];
                if ((ab == 0) != (ba == 0))
                {
                    bad = "one-sided processor patch between ranks " + std::to_string(a) + " and " + std::to_string(b)
                          + ": one side announces " + std::to_string(ab) + ", the other " + std::to_string(ba);
                    break;
                }
                if (ab > 0 && ba > 0 && ab != ba)
                {
                    bad = "processor patch size mismatch between ranks " + std::to_string(a) + " and " + std::to_string(b)
                          + ": " + std::to_string(ab - 1) + " faces on one side, " + std::to_string(ba - 1) + " on the other";
                    break;
                }
            }
        if (!bad.empty()) collectiveAbort("ProcComm::buildPlanFromPatches: " + bad);
    }
    std::vector<int> nbr;
    std::vector<std::vector<int>> send;
    for (const ProcessorPatchDesc& p : patches) { nbr.push_back(p.nbrRank); send.push_back(p.faceCells); }
    std::vector<std::vector<int>> recv = exchangeIntLists(nbr, send);
    for (std::size_t k = 0; k < patches.size(); ++k) patches[k].nbrFaceCells = std::move(recv[k]);
    return makePlan(buildHaloPlanFromProcessorPatches(nCells, patches), maxComps);
}

std::vector<std::vector<int>> ProcComm::exchangeIntLists(const std::vector<int>& nbrRanks,
                                                         const std::vector<std::vector<int>>& send)
{
    if (nbrRanks.size() != send.size()) throw std::invalid_argument("ProcComm::exchangeIntLists: size mismatch");
    if (size_ == 1)
    {
        if (!nbrRanks.empty()) throw std::invalid_argument("ProcComm::exchangeIntLists: np == 1 has no neighbours");
        return {};
    }
    return doExchangeInts(nbrRanks, send);
}

std::vector<std::vector<double>> ProcComm::exchangeFaceData(const std::vector<int>& nbrRanks,
                                                            const std::vector<std::vector<double>>& send)
{
    static_assert(sizeof(double) == 2*sizeof(int), "exchangeFaceData packs a double as two ints");
    if (nbrRanks.size() != send.size()) throw std::invalid_argument("ProcComm::exchangeFaceData: size mismatch");
    std::vector<std::vector<int>> packed(send.size());
    for (std::size_t k = 0; k < send.size(); ++k)
    {
        packed[k].resize(2*send[k].size());
        if (!send[k].empty()) std::memcpy(packed[k].data(), send[k].data(), send[k].size()*sizeof(double));
    }
    const std::vector<std::vector<int>> got = exchangeIntLists(nbrRanks, packed);   // np == 1: no neighbours, empty
    std::vector<std::vector<double>> recv(got.size());
    for (std::size_t k = 0; k < got.size(); ++k)
    {
        recv[k].resize(got[k].size()/2);
        if (!recv[k].empty()) std::memcpy(recv[k].data(), got[k].data(), recv[k].size()*sizeof(double));
    }
    return recv;
}

void ProcComm::collectiveAbort(const std::string& msg)
{
    if (size_ == 1) throw std::runtime_error(msg);
    doAbort(msg);
    throw std::logic_error("ProcComm::doAbort returned");
}

void ProcComm::exchangeCells(const HaloPlan& plan, std::initializer_list<double*> scalars, cudaStream_t stream, CommChannel ch)
{
    if (size_ == 1) return;
    HaloField f[kMaxTotalComps];
    int n = 0;
    for (double* p : scalars)
    {
        if (n >= kMaxTotalComps) throw std::length_error("ProcComm: more than 64 fields in one exchange");
        f[n++] = HaloField{p, 1, 0, 1};
    }
    exchangeCells(plan, f, n, stream, ch);
}

void ProcComm::exchangeVector(const HaloPlan& plan, double* x, double* y, double* z, cudaStream_t stream, CommChannel ch)
{
    if (size_ == 1) return;
    const HaloField f[3] = {{x, 1, 0, 1}, {y, 1, 0, 1}, {z, 1, 0, 1}};
    exchangeCells(plan, f, 3, stream, ch);
}

void ProcComm::exchangeCellsPacked(const HaloPlan& plan, double* data, int nComp, long long compStride, cudaStream_t stream, CommChannel ch)
{
    if (size_ == 1) return;
    const HaloField f{data, nComp, compStride, 1};
    exchangeCells(plan, &f, 1, stream, ch);
}

void ProcComm::exchangeCellsAoS(const HaloPlan& plan, double* data, int nComp, cudaStream_t stream, CommChannel ch)
{
    if (size_ == 1) return;
    const HaloField f{data, nComp, 1, nComp};
    exchangeCells(plan, &f, 1, stream, ch);
}

// ------------------------------------------------------------------------------------------------------------
// LOOPBACK backend
// ------------------------------------------------------------------------------------------------------------
namespace {

struct LbPlan : StagePlan {};

struct LbGroup
{
    int np;
    std::size_t slotCap;

    // abort-aware host rendezvous of the np rank threads
    std::mutex m;
    std::condition_variable cv;
    int arrived = 0;
    unsigned long gen = 0;
    bool aborted = false;
    std::string abortMsg;
    int abortRank = -1;

    std::vector<cudaEvent_t> ready, done;      // per rank, recorded on that rank's stream
    std::vector<double*> dSlot;                // per rank device slot of the device reductions
    std::vector<std::vector<double>> hSlot;    // per rank host slot of the host reductions
    std::vector<std::vector<std::vector<int>>> mail;   // [from][to]
    std::map<int, std::vector<LbPlan*>> reg;   // plan id -> per-rank plan, for makePlan()

    LbGroup(int n, std::size_t cap)
    : np(n), slotCap(cap), ready(n, nullptr), done(n, nullptr), dSlot(n, nullptr), hSlot(n),
      mail(n, std::vector<std::vector<int>>(n))
    {
        for (int r = 0; r < np; ++r)
        {
            ck(cudaEventCreateWithFlags(&ready[r], cudaEventDisableTiming), "cudaEventCreate");
            ck(cudaEventCreateWithFlags(&done[r], cudaEventDisableTiming), "cudaEventCreate");
            dSlot[r] = dalloc<double>(slotCap);
        }
    }
    ~LbGroup()
    {
        for (int r = 0; r < np; ++r)
        {
            cudaEventDestroy(ready[r]);
            cudaEventDestroy(done[r]);
            cudaFree(dSlot[r]);
        }
    }

    void sync()
    {
        std::unique_lock<std::mutex> lk(m);
        if (aborted) throw ProcCommAbort(abortMsg, abortRank);
        const unsigned long g = gen;
        if (++arrived == np) { arrived = 0; ++gen; cv.notify_all(); return; }
        cv.wait(lk, [&] { return gen != g || aborted; });
        if (gen == g) throw ProcCommAbort(abortMsg, abortRank);   // aborted before the barrier was released
    }

    // First abort wins; sticky.
    void abort(int rank, const std::string& msg)
    {
        std::lock_guard<std::mutex> lk(m);
        if (!aborted) { aborted = true; abortMsg = msg; abortRank = rank; }
        cv.notify_all();
    }
    bool isAborted()
    {
        std::lock_guard<std::mutex> lk(m);
        return aborted;
    }
    std::string message() { std::lock_guard<std::mutex> lk(m); return abortMsg; }
    int originRank() { std::lock_guard<std::mutex> lk(m); return abortRank; }
};

class LoopbackComm final : public ProcComm
{
public:
    LoopbackComm(std::shared_ptr<LbGroup> g, int rank, int np) : ProcComm(Backend::Loopback, rank, np), g_(std::move(g))
    {
        if (np > 1) ck(cudaGetDevice(&device_), "cudaGetDevice");
    }
    bool graphCapturable() const override { return true; }
    bool abortPending() const override { return g_->isAborted(); }

protected:
    // Collective: every rank registers its plan, then reads the peers' staging pointers/offsets.
    std::unique_ptr<HaloPlan> doMakePlan(HaloPlanDesc desc, int maxComps) override
    {
        std::unique_ptr<HaloPlan> plan = newPlan(std::move(desc));
        auto st = std::make_unique<LbPlan>();
        LbPlan* sp = st.get();
        sp->init(detail::PlanAccess::desc(*plan), maxComps, false);
        setPlanState(*plan, std::move(st));
        const int id = planSeq_++;
        {
            std::lock_guard<std::mutex> lk(g_->m);
            auto& v = g_->reg[id];
            v.resize(g_->np, nullptr);
            v[rank_] = sp;
        }
        g_->sync();
        const HaloPlanDesc& d = *sp->desc;
        std::vector<RecvEntry> rt(d.nbrs.size());
        std::vector<FaceEntry> ft(d.nbrs.size());
        std::string err;
        for (std::size_t k = 0; k < d.nbrs.size() && err.empty(); ++k)
        {
            const HaloNeighbour& n = d.nbrs[k];
            if (n.rank >= g_->np) { err = "neighbour rank out of range"; break; }
            const LbPlan* peer = g_->reg[id][n.rank];
            const HaloNeighbour* pn = nullptr;
            for (const HaloNeighbour& q : peer->desc->nbrs) if (q.rank == rank_) pn = &q;
            if (!pn) { err = "rank " + std::to_string(n.rank) + " does not list rank " + std::to_string(rank_) + " as a neighbour"; break; }
            if (pn->sendCount != n.recvCount || pn->recvCount != n.sendCount || pn->faceCount != n.faceCount)
            {
                err = "plan mismatch between ranks " + std::to_string(rank_) + " and " + std::to_string(n.rank);
                break;
            }
            rt[k] = {peer->dSend, pn->sendOffset, n.recvCount, n.recvStart};
            ft[k] = {peer->dSend, pn->faceStart, n.faceCount, n.faceStart};
        }
        if (!err.empty()) g_->abort(rank_, "ProcComm::makePlan: " + err);   // the others throw at the sync below
        g_->sync();
        if (!err.empty()) throw ProcCommAbort(g_->message(), g_->originRank());
        sp->setTables(rt, ft);
        g_->sync();   // no rank exchanges before every rank's tables point at live peers
        return plan;
    }

    // pack -> ready -> [rendezvous] -> peers' staging read by my unpack -> done -> [rendezvous] -> wait done
    void doExchangeCells(const HaloPlan& plan, const HaloField* fields, int nFields, cudaStream_t s, CommChannel) override
    {
        auto* st = static_cast<LbPlan*>(planState(plan));
        FlatComps fc;
        flatten(fields, nFields, fc);
        if (fc.n == 0) return;
        checkCapacity(*st, fc.n);
        st->pack(plan, fc, s);
        rendezvous(s, neighbourRanks(*st), [&] { st->unpack(plan, fc, s); });
    }

    void doExchangeFaces(const HaloPlan& plan, double* arr, FaceOp op, cudaStream_t s, CommChannel) override
    {
        auto* st = static_cast<LbPlan*>(planState(plan));
        const int nf = plan.nProcFaces();
        if (nf > 0) ck(cudaMemcpyAsync(st->dSend, arr, static_cast<std::size_t>(nf)*sizeof(double), cudaMemcpyDeviceToDevice, s), "face pack");
        rendezvous(s, neighbourRanks(*st), [&] { st->combineFaces(arr, op, s); });
    }

    void doAllReduceDevice(double* d, int n, ReduceOp op, cudaStream_t s, CommChannel) override
    {
        deviceFold(d, n, opCode(op), 0, s);
    }
    void doBcastDevice(double* d, int n, int root, cudaStream_t s, CommChannel) override
    {
        if (root < 0 || root >= size_) throw std::invalid_argument("ProcComm::bcast: root out of range");
        deviceFold(d, n, 3, root, s);
    }

    void doAllReduceHost(double* h, int n, ReduceOp op) override { hostFold(h, n, opCode(op), 0); }
    void doBcastHost(double* h, int n, int root) override
    {
        if (root < 0 || root >= size_) throw std::invalid_argument("ProcComm::bcast: root out of range");
        hostFold(h, n, 3, root);
    }

    void doBarrier() override { g_->sync(); }

    std::vector<std::vector<int>> doExchangeInts(const std::vector<int>& nbr, const std::vector<std::vector<int>>& send) override
    {
        for (std::size_t k = 0; k < nbr.size(); ++k)
        {
            if (nbr[k] < 0 || nbr[k] >= size_) throw std::invalid_argument("ProcComm::exchangeIntLists: rank out of range");
            g_->mail[rank_][nbr[k]] = send[k];
        }
        g_->sync();
        std::vector<std::vector<int>> recv(nbr.size());
        for (std::size_t k = 0; k < nbr.size(); ++k) recv[k] = g_->mail[nbr[k]][rank_];
        g_->sync();
        for (std::size_t k = 0; k < nbr.size(); ++k) g_->mail[rank_][nbr[k]].clear();
        return recv;
    }

    void allgatherInts(const std::vector<int>& send, std::vector<int>& recv) override
    {
        // Every rank publishes its row in the mailbox, rendezvous, then reads all rows in rank order. recv is
        // [rank][row-entry], exactly like the MPI backends' allgather.
        g_->mail[rank_][rank_] = send;
        g_->sync();
        recv.clear();
        for (int r = 0; r < size_; ++r)
        {
            const auto& row = g_->mail[r][r];
            recv.insert(recv.end(), row.begin(), row.end());
        }
        g_->sync();
        g_->mail[rank_][rank_].clear();
    }

    [[noreturn]] void doAbort(const std::string& msg) override
    {
        g_->abort(rank_, msg);
        throw ProcCommAbort(g_->message(), g_->originRank());
    }

private:
    static std::vector<int> neighbourRanks(const StagePlan& st)
    {
        std::vector<int> r;
        for (const HaloNeighbour& n : st.desc->nbrs) r.push_back(n.rank);
        return r;
    }

    template<class Body> void rendezvous(cudaStream_t s, const std::vector<int>& peers, Body body)
    {
        ck(cudaEventRecord(g_->ready[rank_], s), "record ready");
        g_->sync();
        for (int p : peers) ck(cudaStreamWaitEvent(s, g_->ready[p], 0), "wait ready");
        body();
        ck(cudaEventRecord(g_->done[rank_], s), "record done");
        g_->sync();
        for (int p : peers) ck(cudaStreamWaitEvent(s, g_->done[p], 0), "wait done");
    }

    void deviceFold(double* d, int n, int op, int root, cudaStream_t s)
    {
        if (n < 0 || static_cast<std::size_t>(n) > g_->slotCap)
            throw std::length_error("ProcComm: device reduction of " + std::to_string(n) + " doubles exceeds the loopback reduceCapacity "
                                    + std::to_string(g_->slotCap));
        if (n == 0) return;
        SlotList sl;
        sl.n = g_->np;
        for (int r = 0; r < g_->np; ++r) sl.p[r] = g_->dSlot[r];
        std::vector<int> all(g_->np);
        for (int r = 0; r < g_->np; ++r) all[r] = r;
        ck(cudaMemcpyAsync(g_->dSlot[rank_], d, static_cast<std::size_t>(n)*sizeof(double), cudaMemcpyDeviceToDevice, s), "reduce stage");
        rendezvous(s, all, [&] {
            foldSlotsK<<<gridFor(n), 256, 0, s>>>(d, sl, n, op, root);
            ck(cudaGetLastError(), "fold launch");
        });
    }

    void hostFold(double* h, int n, int op, int root)
    {
        g_->hSlot[rank_].assign(h, h + n);
        g_->sync();
        std::string err;
        std::vector<const double*> s(g_->np);
        for (int r = 0; r < g_->np; ++r)
        {
            if (static_cast<int>(g_->hSlot[r].size()) != n) err = "host reduction size differs between ranks";
            s[r] = g_->hSlot[r].data();
        }
        if (!err.empty()) g_->abort(rank_, "ProcComm: " + err);
        else foldHost(h, s, n, op, root);
        g_->sync();   // slots may be rewritten only after everyone has read them (throws for everyone after an abort)
    }

    std::shared_ptr<LbGroup> g_;
    int planSeq_ = 0;
};

} // namespace

std::vector<std::unique_ptr<ProcComm>> ProcComm::createLoopback(int np, std::size_t reduceCapacity)
{
    if (np < 1 || np > kMaxRanks) throw std::invalid_argument("ProcComm::createLoopback: np must be 1..64");
    std::shared_ptr<LbGroup> g;
    if (np > 1) g = std::make_shared<LbGroup>(np, reduceCapacity);   // np == 1 allocates nothing
    std::vector<std::unique_ptr<ProcComm>> v;
    for (int r = 0; r < np; ++r) v.push_back(std::unique_ptr<ProcComm>(new LoopbackComm(g, r, np)));
    return v;
}

// ------------------------------------------------------------------------------------------------------------
// MPI-HOST backend
// ------------------------------------------------------------------------------------------------------------
namespace {

constexpr int kTagCells = 7001, kTagFaces = 7002, kTagInts = 7003;

struct AbortRec { int flag; int pad; char msg[248]; };

struct MpiPlan : StagePlan
{
    double* hSend = nullptr;     // pinned
    double* hRecv = nullptr;     // pinned
    cudaEvent_t lastUse = nullptr;
    ~MpiPlan() override
    {
        if (lastUse) cudaEventDestroy(lastUse);
        cudaFreeHost(hSend);
        cudaFreeHost(hRecv);
    }
};

double* hostAlloc(std::size_t n)
{
    if (n == 0) return nullptr;
    double* p = nullptr;
    ck(cudaMallocHost(&p, n*sizeof(double)), "cudaMallocHost");
    ++g_allocs;
    return p;
}

void mpiCheck(int rc, const char* what)
{
    if (rc != MPI_SUCCESS) throw std::runtime_error(std::string("ProcComm MPI: ") + what + " failed");
}

// The NCCL backend derives from this class: its host-resident collectives (barrier, host reductions/bcast, int lists,
// the abort status phase) are the MPI ones, its device ops are NCCL.
class MpiHostComm : public ProcComm
{
public:
    MpiHostComm() : MpiHostComm(Backend::MpiHost) {}
    bool graphCapturable() const override { return false; }

protected:
    explicit MpiHostComm(Backend b) : ProcComm(b, Pstream::myProcNo(), Pstream::nProcs())
    {
        device_ = Pstream::deviceId();
        // The abort record travels on a duplicated communicator with a continuous nonblocking allgather, so a
        // rank can learn a peer's abort WITHOUT joining the peer's current collective on MPI_COMM_WORLD. This is
        // what makes "one rank aborts, the others are mid-exchange" recoverable: no rank ever waits for an
        // allgather that the aborting rank will not join.
        startSideChannel();
    }
    // Called on EVERY rank, from inside the status phase that is about to throw a ProcCommAbort (all ranks reach it
    // at the same collective). The NCCL backend uses it to abort and re-create its communicators.
    virtual void abortSeen() {}

    std::unique_ptr<HaloPlan> doMakePlan(HaloPlanDesc desc, int maxComps) override
    {
        status(nullptr);
        std::unique_ptr<HaloPlan> plan = newPlan(std::move(desc));
        auto st = std::make_unique<MpiPlan>();
        MpiPlan* sp = st.get();
        const HaloPlanDesc& d = detail::PlanAccess::desc(*plan);
        sp->init(d, maxComps, true);
        sp->hSend = hostAlloc(sp->sendCap);
        sp->hRecv = hostAlloc(sp->recvCap);
        ck(cudaEventCreateWithFlags(&sp->lastUse, cudaEventDisableTiming), "cudaEventCreate");
        std::vector<RecvEntry> rt(d.nbrs.size());
        std::vector<FaceEntry> ft(d.nbrs.size());
        for (std::size_t k = 0; k < d.nbrs.size(); ++k)
        {
            const HaloNeighbour& n = d.nbrs[k];
            rt[k] = {sp->dRecv, n.recvStart, n.recvCount, n.recvStart};
            ft[k] = {sp->dRecv, n.faceStart, n.faceCount, n.faceStart};
        }
        sp->setTables(rt, ft);
        setPlanState(*plan, std::move(st));
        verifyPlanSizes(d);
        return plan;
    }

    // The neighbours must agree on the message sizes (collective; every rank throws the same ProcCommAbort).
    void verifyPlanSizes(const HaloPlanDesc& d)
    {
        std::vector<int> nbr;
        std::vector<std::vector<int>> mine;
        for (const HaloNeighbour& n : d.nbrs) { nbr.push_back(n.rank); mine.push_back({n.sendCount, n.recvCount, n.faceCount}); }
        const auto theirs = exchangeInts(nbr, mine);
        double bad = 0;
        for (std::size_t k = 0; k < nbr.size(); ++k)
            if (theirs[k].size() != 3 || theirs[k][0] != d.nbrs[k].recvCount || theirs[k][1] != d.nbrs[k].sendCount
                || theirs[k][2] != d.nbrs[k].faceCount)
                bad = 1;
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "Allreduce");
        if (bad != 0)
            throw ProcCommAbort("ProcComm::makePlan: the ranks' halo plans do not match (send/recv/face counts)", 0);
    }

    void doExchangeCells(const HaloPlan& plan, const HaloField* fields, int nFields, cudaStream_t s, CommChannel) override
    {
        noCapture(s);
        FlatComps fc;
        flatten(fields, nFields, fc);
        auto* st = static_cast<MpiPlan*>(planState(plan));
        checkCapacity(*st, fc.n);            // BEFORE the status phase: capacity errors abort everyone together
        status(nullptr);
        if (fc.n == 0) return;
        const HaloPlanDesc& d = *st->desc;
        const int C = fc.n;
        const std::size_t nSend = static_cast<std::size_t>(C)*d.sendCells.size();
        const std::size_t nRecv = static_cast<std::size_t>(C)*d.nHalo;
        // Local, fallible work: nothing has been posted yet, so a failure here is still a clean collective abort.
        localPhase([&] {
            ck(cudaEventSynchronize(st->lastUse), "wait last use");
            st->pack(plan, fc, s);
            if (nSend) ck(cudaMemcpyAsync(st->hSend, st->dSend, nSend*sizeof(double), cudaMemcpyDeviceToHost, s), "D2H");
            ck(cudaStreamSynchronize(s), "stream sync");
        });
        // Negative-control hook (tests/test_proc_comm_failure.cu): fail AFTER the opening status phase and the
        // local staging, before the nonblocking wait -- the window in which a rank-local error used to hang peers.
        if (backend_ == Backend::MpiHost)
            if (const char* inj = std::getenv("BRAE_COMM_FAIL_AFTER_STATUS"); inj && inj[0] == '1')
                throw std::runtime_error("injected post-status failure");
        std::vector<MPI_Request> req;
        req.reserve(2*d.nbrs.size());
        localPhase([&] {
            for (const HaloNeighbour& n : d.nbrs)
            {
                MPI_Request r;
                if (n.recvCount)
                {
                    mpiCheck(MPI_Irecv(st->hRecv + static_cast<std::size_t>(C)*n.recvStart, C*n.recvCount, MPI_DOUBLE, n.rank, kTagCells, MPI_COMM_WORLD, &r), "Irecv");
                    req.push_back(r);
                }
                if (n.sendCount)
                {
                    mpiCheck(MPI_Isend(st->hSend + static_cast<std::size_t>(C)*n.sendOffset, C*n.sendCount, MPI_DOUBLE, n.rank, kTagCells, MPI_COMM_WORLD, &r), "Isend");
                    req.push_back(r);
                }
            }
        }, true);
        waitGuarded(req, "exchangeCells");
        if (nRecv) ck(cudaMemcpyAsync(st->dRecv, st->hRecv, nRecv*sizeof(double), cudaMemcpyHostToDevice, s), "H2D");
        st->unpack(plan, fc, s);
        ck(cudaEventRecord(st->lastUse, s), "record lastUse");
    }

    void doExchangeFaces(const HaloPlan& plan, double* arr, FaceOp op, cudaStream_t s, CommChannel) override
    {
        noCapture(s);
        status(nullptr);
        auto* st = static_cast<MpiPlan*>(planState(plan));
        const HaloPlanDesc& d = *st->desc;
        const std::size_t nf = static_cast<std::size_t>(d.nProcFaces);
        localPhase([&] {
            ck(cudaEventSynchronize(st->lastUse), "wait last use");
            if (nf) ck(cudaMemcpyAsync(st->hSend, arr, nf*sizeof(double), cudaMemcpyDeviceToHost, s), "D2H");
            ck(cudaStreamSynchronize(s), "stream sync");
        });
        std::vector<MPI_Request> req;
        req.reserve(2*d.nbrs.size());
        localPhase([&] {
            for (const HaloNeighbour& n : d.nbrs)
            {
                if (!n.faceCount) continue;
                MPI_Request r;
                mpiCheck(MPI_Irecv(st->hRecv + n.faceStart, n.faceCount, MPI_DOUBLE, n.rank, kTagFaces, MPI_COMM_WORLD, &r), "Irecv");
                req.push_back(r);
                mpiCheck(MPI_Isend(st->hSend + n.faceStart, n.faceCount, MPI_DOUBLE, n.rank, kTagFaces, MPI_COMM_WORLD, &r), "Isend");
                req.push_back(r);
            }
        }, true);
        waitGuarded(req, "exchangeFaces");
        if (nf) ck(cudaMemcpyAsync(st->dRecv, st->hRecv, nf*sizeof(double), cudaMemcpyHostToDevice, s), "H2D");
        st->combineFaces(arr, op, s);
        ck(cudaEventRecord(st->lastUse, s), "record lastUse");
    }

    void doAllReduceDevice(double* d, int n, ReduceOp op, cudaStream_t s, CommChannel) override { deviceColl(d, n, opCode(op), 0, s); }
    void doBcastDevice(double* d, int n, int root, cudaStream_t s, CommChannel) override
    {
        if (root < 0 || root >= size_) throw std::invalid_argument("ProcComm::bcast: root out of range");
        deviceColl(d, n, 3, root, s);
    }
    void doAllReduceHost(double* h, int n, ReduceOp op) override { hostColl(h, n, opCode(op), 0); }
    void doBcastHost(double* h, int n, int root) override
    {
        if (root < 0 || root >= size_) throw std::invalid_argument("ProcComm::bcast: root out of range");
        hostColl(h, n, 3, root);
    }

    void doBarrier() override
    {
        status(nullptr);
        mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    }

    std::vector<std::vector<int>> doExchangeInts(const std::vector<int>& nbr, const std::vector<std::vector<int>>& send) override
    {
        status(nullptr);
        return exchangeInts(nbr, send);
    }

    [[noreturn]] void doAbort(const std::string& msg) override
    {
        // Publish first, then join the status phase (peers that are already inside a host collective need this
        // rank's contribution), but with the BOUNDED nonblocking wait of status(): a peer that is stuck in a
        // point-to-point exchange is released by the side channel, and this rank never blocks for ever either.
        publishAbort(msg);
        status(&msg);
        throw ProcCommAbort(msg, rank_);
    }

    // A rank-local, fallible step of an exchange. Its outcome is agreed with the status allgather BEFORE any
    // rank can block on a message, so one rank's CUDA error or capacity problem aborts EVERY rank with the same
    // ProcCommAbort instead of hanging the peers in MPI_Waitall. With `posted` some requests may already be
    // nonblocking: they are then detached (never waited on) and the backend refuses further work, because the
    // process is expected to abort.
    template<class F> void localPhase(F&& f, bool posted = false)
    {
        std::string err;
        try { f(); }
        catch (const std::exception& e) { err = e.what(); }
        catch (...) { err = "unknown local failure"; }
        if (posted && !err.empty()) cancelLive();   // a partially posted exchange must not leave live requests
        status(err.empty() ? nullptr : &err);
    }

    // Bounded nonblocking wait. MPI_Testall is polled on the caller's thread (MPI_THREAD_FUNNELED is enough), so
    // a peer that died AFTER the status phase is detected by the watchdog instead of hanging this rank for ever.
    // No extra collective on the hot path.
    // Every request is registered until it completes, and the registry is CANCELLED+WAITED on the abort path:
    // a rank that aborts while its peers never posted the matching messages still has live nonblocking
    // operations. Freeing or reusing their buffers then would be a use-after-free, and a later exchange (same
    // tags) would consume the stale message and desynchronise the job -- the post-abort hang this guard removes.
    void waitGuarded(std::vector<MPI_Request>& req, const char* what)
    {
        if (req.empty()) return;
        live_.push_back({});
        Live& slot = live_.back();
        slot.reqs.swap(req);
        const auto t0 = std::chrono::steady_clock::now();
        for (;;)
        {
            int done = 0;
            mpiCheck(MPI_Testall(static_cast<int>(slot.reqs.size()), slot.reqs.data(), &done, MPI_STATUSES_IGNORE), "Testall");
            if (done) { live_.pop_back(); return; }
            // A peer's abort releases this wait too (no watchdog delay): cancel the outstanding messages.
            {
                std::string msg;
                int origin = -1;
                if (sideAborted(msg, origin))
                {
                    live_.pop_back();
                    cancelLive();
                    abortSeen();
                    throw ProcCommAbort(msg, origin);
                }
            }
            if (watchdogMs_ > 0
                && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() > watchdogMs_)
            {
                cancelLive();                   // nothing is freed or reused while a request is outstanding
                collectiveAbort(std::string("ProcComm: ") + what + " timed out after " + std::to_string(watchdogMs_)
                                + " ms waiting for a neighbour (a rank failed after the status phase)");
            }
            usleep(200);
        }
    }

    // A nonblocking COLLECTIVE cannot be cancelled. If a peer aborts, its send/receive
    // buffers must remain alive while MPI progresses/finalizes the request. The caller moves
    // them into these members before throwing; stopSideChannel leaves a communicator with
    // outstanding work for MPI_Finalize rather than explicitly destroying it a second time.
    static void dropCollective(MPI_Request& r) { r = MPI_REQUEST_NULL; }
    AbortRec abortedSend_{};
    std::vector<AbortRec> abortedRecv_;
    bool abortedCollective_ = false;

    struct Live { std::vector<MPI_Request> reqs; };
    std::vector<Live> live_;                    // requests of the exchange(s) currently in flight on this rank

    void cancelLive()
    {
        for (Live& l : live_)
        {
            for (auto& r : l.reqs)
                if (r != MPI_REQUEST_NULL) MPI_Cancel(&r);      // point-to-point requests: cancellable
            for (auto& r : l.reqs)
                if (r != MPI_REQUEST_NULL) MPI_Wait(&r, MPI_STATUS_IGNORE);
        }
        live_.clear();
    }

    ~MpiHostComm() override;                    // defined below: cancels live requests, stops the side channel

    void allgatherInts(const std::vector<int>& send, std::vector<int>& recv) override
    {
        recv.resize(static_cast<std::size_t>(size_)*send.size());
        if (statusComm_ == MPI_COMM_NULL)
            throw std::runtime_error("ProcComm::allgatherInts: the status communicator was not initialized");
        MPI_Request req = MPI_REQUEST_NULL;
        mpiCheck(MPI_Iallgather(send.data(), static_cast<int>(send.size()), MPI_INT, recv.data(),
                                static_cast<int>(send.size()), MPI_INT, statusComm_, &req), "Iallgather");
        const auto t0 = std::chrono::steady_clock::now();
        for (;;)
        {
            int done = 0;
            mpiCheck(MPI_Test(&req, &done, MPI_STATUS_IGNORE), "Test");
            if (done) return;
            std::string msg;
            int origin = -1;
            if (sideAborted(msg, origin)) { dropCollective(req); abortSeen(); throw ProcCommAbort(msg, origin); }
            if (watchdogMs_ > 0
                && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() > watchdogMs_)
            {
                dropCollective(req);
                collectiveAbort("ProcComm: patch participation allgather timed out (a rank left the sequence)");
            }
            usleep(200);
        }
    }

    // A collective that must NEVER hang the abort recovery: the status allgather already uses a bounded
    // nonblocking wait, and the recovery's allreduce/barrier need the same property because a peer may have left
    // the sequence after its own ProcCommAbort. The result is the MAX of the local flags when the collective
    // completes; if the peers are gone the wait ends after the watchdog and `completed` is false.
    int agreeMax(int local, bool& completed)
    {
        int out = local;
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Iallreduce(&local, &out, 1, MPI_INT, MPI_MAX, statusComm_, &req) != MPI_SUCCESS)
        {
            completed = false;
            return out;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (;;)
        {
            int done = 0;
            if (MPI_Test(&req, &done, MPI_STATUS_IGNORE) != MPI_SUCCESS) { completed = false; return out; }
            if (done) { completed = true; return out; }
            if (watchdogMs_ > 0
                && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() > watchdogMs_)
            {
                dropCollective(req);
                completed = false;
                return out;
            }
            usleep(200);
        }
    }

    // Publishes the abort to the side channel so peers blocked in NCCL work learn it without reaching a host
    // collective. Overridden by the NCCL backend, which also asks its watchdog to abort the communicators.
    virtual void publishAbort(const std::string& msg)
    {
        if (!abortState_) return;
        abortState_->raised.store(true, std::memory_order_release);
        abortState_->publish(rank_, msg);
    }

    std::shared_ptr<AbortState> abortState_ = std::make_shared<AbortState>();
    int watchdogMs_ = watchdogMs("BRAE_COMM_WATCHDOG_MS", 30000);

protected:
    // The status phase opens every collective: all ranks allgather their abort flag (+ message). A raised
    // abort is thrown, with the message of the lowest raising rank, by every rank including the raiser.
    void status(const std::string* raise)
    {
        pollSide();
        if (!raise)
        {
            std::string msg;
            int origin = -1;
            if (sideAborted(msg, origin)) { abortSeen(); throw ProcCommAbort(msg, origin); }
        }
        AbortRec mine;
        std::memset(&mine, 0, sizeof(mine));
        if (raise)
        {
            mine.flag = 1;
            std::strncpy(mine.msg, raise->c_str(), sizeof(mine.msg) - 1);
        }
        std::vector<AbortRec> all(size_);
        // NONBLOCKING, on the status communicator: the allgather is progressed in a bounded poll that also
        // watches the side channel, so a peer that aborted and left (doAbort throws at once) cannot strand this
        // rank in a blocking MPI_Allgather. MPI_COMM_WORLD is dup'd so a stuck status never collides with the
        // point-to-point tags of an exchange.
        MPI_Request req = MPI_REQUEST_NULL;
        mpiCheck(MPI_Iallgather(&mine, sizeof(AbortRec), MPI_BYTE, all.data(), sizeof(AbortRec), MPI_BYTE, statusComm_, &req), "Iallgather");
        const auto t0 = std::chrono::steady_clock::now();
        for (;;)
        {
            int done = 0;
            mpiCheck(MPI_Test(&req, &done, MPI_STATUS_IGNORE), "Test");
            if (done) break;
            {
                std::string msg;
                int origin = -1;
                // While RAISING, only a DIFFERENT rank's abort interrupts: my own record is already published and
                // the peers still need this allgather. With no raise, any side-channel abort releases the wait.
                if (sideAborted(msg, origin) && (!raise || origin != rank_))
                {
                    abortedCollective_ = true;
                    abortedRecv_ = std::move(all);
                    abortedSend_ = mine;
                    dropCollective(req);
                    abortSeen();
                    throw ProcCommAbort(msg, origin);
                }
            }
            if (watchdogMs_ > 0
                && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() > watchdogMs_)
            {
                abortedCollective_ = true;
                abortedRecv_ = std::move(all);
                abortedSend_ = mine;
                dropCollective(req);
                if (raise) { abortSeen(); throw ProcCommAbort(*raise, rank_); }
                const std::string msg = "ProcComm: status phase timed out (a rank left the collective sequence)";
                publishAbort(msg);
                abortSeen();
                throw ProcCommAbort(msg, rank_);
            }
            usleep(100);
        }
        for (int r = 0; r < size_; ++r)
            if (all[r].flag)
            {
                abortSeen();
                throw ProcCommAbort(std::string(all[r].msg), r);
            }
    }

    static void noCapture(cudaStream_t s)
    {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        ck(cudaStreamIsCapturing(s, &cs), "cudaStreamIsCapturing");
        if (cs != cudaStreamCaptureStatusNone)
            throw std::logic_error("ProcComm MPI-host backend is host-synchronous and cannot be captured in a CUDA graph");
    }

    std::vector<std::vector<int>> exchangeInts(const std::vector<int>& nbr, const std::vector<std::vector<int>>& send)
    {
        std::vector<MPI_Request> req(nbr.size());
        // Receives are posted FIRST with the length this rank announced, and empty lists are real messages, so
        // participation is symmetric and no MPI_Probe can wait for a message a one-sided peer never sends.
        std::vector<MPI_Request> rreq(nbr.size());
        std::vector<std::vector<int>> recv(nbr.size());
        localPhase([&] {
            for (std::size_t k = 0; k < nbr.size(); ++k)
            {
                if (nbr[k] < 0 || nbr[k] >= size_) throw std::invalid_argument("ProcComm::exchangeIntLists: rank out of range");
                recv[k].resize(k < send.size() ? send[k].size() : 0);
            }
            for (std::size_t k = 0; k < nbr.size(); ++k)
                mpiCheck(MPI_Irecv(recv[k].data(), static_cast<int>(recv[k].size()), MPI_INT, nbr[k], kTagInts, MPI_COMM_WORLD, &rreq[k]), "Irecv");
            for (std::size_t k = 0; k < nbr.size(); ++k)
                mpiCheck(MPI_Isend(send[k].data(), static_cast<int>(send[k].size()), MPI_INT, nbr[k], kTagInts, MPI_COMM_WORLD, &req[k]), "Isend");
        }, true);
        waitGuarded(rreq, "exchangeIntLists receive");
        waitGuarded(req, "exchangeIntLists send");
        return recv;
    }

    // allgather of the n partials, then the fixed-rank-order fold on every rank
    void gatherFold(double* h, int n, int op, int root)
    {
        std::vector<double> all(static_cast<std::size_t>(size_)*n);
        mpiCheck(MPI_Allgather(h, n, MPI_DOUBLE, all.data(), n, MPI_DOUBLE, MPI_COMM_WORLD), "Allgather");
        std::vector<const double*> s(size_);
        for (int r = 0; r < size_; ++r) s[r] = all.data() + static_cast<std::size_t>(r)*n;
        foldHost(h, s, n, op, root);
    }

    // ---- abort side channel ---------------------------------------------------------------------------------
    // A duplicated communicator carries a CONTINUOUS nonblocking allgather of the abort record. The status
    // allgather on MPI_COMM_WORLD only reaches ranks that enter a host collective; this channel reaches a rank
    // whose main thread is blocked in a device synchronisation (the NCCL watchdog then aborts its communicators,
    // releasing the NCCL kernels so the main thread can reach the throw).
    // Started lazily by the NCCL backend (the only one with device-blocking collectives). One background thread
    // per process, the sole user of sideComm_, and only with MPI_THREAD_MULTIPLE. Without that level the main
    // thread drives the rounds from pollSide() (device ops then propagate at their next host call).
    void startSideChannel()
    {
        if (sideOn_ || size_ == 1) return;
        sideOn_ = true;
        side_ = std::make_shared<SideChannel>();
        side_->recv.resize(size_);
        mpiCheck(MPI_Comm_dup(MPI_COMM_WORLD, &sideComm_), "Comm_dup");
        mpiCheck(MPI_Comm_dup(MPI_COMM_WORLD, &statusComm_), "Comm_dup");
        // The abort channel must never let OpenMPI kill the process: a peer that aborts mid-round leaves a
        // truncated/incomplete nonblocking allgather behind, which is EXPECTED during failure propagation.
        // Handle those errors as "channel is dead, stop polling" instead of MPI_ERRORS_ARE_FATAL.
        MPI_Comm_set_errhandler(sideComm_, MPI_ERRORS_RETURN);
        MPI_Comm_set_errhandler(statusComm_, MPI_ERRORS_RETURN);
        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        sideThreadOk_ = provided >= MPI_THREAD_MULTIPLE;
        if (sideThreadOk_)
        {
            auto sc = side_;
            auto st = abortState_;
            sideThread_ = std::thread([this, sc, st] { sideLoop(sc, st); });
        }
    }

    // COLLECTIVE shutdown. Every rank calls it (the destructor, or explicitly before Pstream::finalize): finish
    // the outstanding abort-record round, then run one final allgather so no rank enters MPI_Finalize with an
    // NBC subrequest its peers still owe. An abort already delivered stays delivered; shutdown never revives a
    // communicator, and a failed peer cannot hold the others past the watchdog.
    void quiesceSideChannel()
    {
        if (!sideOn_ || sideQuiesced_) return;
        sideQuiesced_ = true;
        sideClosing_.store(true, std::memory_order_release);
        for (;;)
        {
            bool active = false;
            if (sideThreadOk_)
            {
                std::lock_guard<std::mutex> lk(side_->m);
                active = side_->active;
            }
            else
            {
                sideRound(*side_);
                active = side_->active;
            }
            if (!active) break;
            usleep(200);
            if (watchdogMs_ > 0 && ++sideShutdownSpins_ > watchdogMs_)
                throw std::runtime_error("ProcComm: side channel did not finish its last abort round");
        }
        // One agreed, bounded round on the STATUS communicator (the side thread owns only sideComm_). This is
        // the collective quiesce: after it every rank has left the channel, so Pstream::finalize cannot strand a
        // nonblocking allgather and OpenMPI reports no NBC-subrequest error.
        AbortRec mine;
        std::memset(&mine, 0, sizeof(mine));
        std::vector<AbortRec> all(size_);
        MPI_Request req = MPI_REQUEST_NULL;
        mpiCheck(MPI_Iallgather(&mine, sizeof(AbortRec), MPI_BYTE, all.data(), sizeof(AbortRec), MPI_BYTE,
                                statusComm_, &req), "shutdown Iallgather");
        const auto t0 = std::chrono::steady_clock::now();
        for (;;)
        {
            int done = 0;
            mpiCheck(MPI_Test(&req, &done, MPI_STATUS_IGNORE), "shutdown Test");
            if (done) break;
            if (watchdogMs_ > 0
                && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
                   > watchdogMs_)
            {
                abortedCollective_ = true;
                abortedRecv_ = std::move(all);
                abortedSend_ = mine;
                dropCollective(req);
                throw std::runtime_error("ProcComm: collective shutdown timed out (a rank skipped communicator destruction)");
            }
            usleep(200);
        }
        abortState_->stop.store(true, std::memory_order_release);
        joinSideThread();
    }

    void joinSideThread()
    {
        if (sideThread_.joinable())
        {
            sideThread_.join();
            side_->done.store(true, std::memory_order_release);
        }
    }

    void stopSideChannel()
    {
        if (!sideOn_) return;
        abortState_->stop.store(true, std::memory_order_release);
        joinSideThread();
        // OpenMPI's duplicated-communicator free is collective: do not let one rank
        // release it while another is still leaving its final status round. If a peer
        // cannot rendezvous after an abort, leave both communicators to MPI_Finalize.
        const bool joined = MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS;
        if (joined && !abortedCollective_)
        {
            MPI_Comm_free(&sideComm_);
            MPI_Comm_free(&statusComm_);
        }
        sideOn_ = false;
    }

    struct SideChannel
    {
        std::mutex m;                    // side thread's MPI round vs. collective shutdown
        std::vector<AbortRec> recv;
        MPI_Request req = MPI_REQUEST_NULL;
        bool active = false;
        std::atomic<bool> done{false};   // the thread has left its loop: the comms may be freed
    };
    std::shared_ptr<SideChannel> side_;

    // Main-thread driver when the MPI thread level does not allow the background thread.
    void pollSide()
    {
        if (!sideOn_ || sideThreadOk_) return;
        sideRound(*side_);
        if (abortState_->pending.load(std::memory_order_acquire)) onSideAbort();
    }

    // The side channel's abort, if any: every rank throws the same ProcCommAbort as the raiser.
    bool sideAborted(std::string& msg, int& origin)
    {
        if (!abortState_->pending.load(std::memory_order_acquire)) return false;
        auto got = abortState_->get();
        if (got.first < 0) return false;
        msg = got.second;
        origin = got.first;
        return true;
    }

    MPI_Comm sideComm_ = MPI_COMM_NULL;
    MPI_Comm statusComm_ = MPI_COMM_NULL;
    bool sideOn_ = false, sideThreadOk_ = false, sideQuiesced_ = false;
    int sideShutdownSpins_ = 0;
    std::atomic<bool> sideClosing_{false};
    std::thread sideThread_;

private:
    void sideRound(SideChannel& sc)
    {
        if (!sideOn_ || sideClosing_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lk(sc.m);
        if (!sc.active)
        {
            AbortRec mine;
            std::memset(&mine, 0, sizeof(mine));
            if (abortState_->raised.load(std::memory_order_acquire))
            {
                auto got = abortState_->get();
                mine.flag = 1;
                std::strncpy(mine.msg, got.second.c_str(), sizeof(mine.msg) - 1);
            }
            const int rc = MPI_Iallgather(&mine, sizeof(AbortRec), MPI_BYTE, sc.recv.data(), sizeof(AbortRec), MPI_BYTE, sideComm_, &sc.req);
            if (rc != MPI_SUCCESS)
            {
                // A peer already left (abort/finalize): the channel cannot complete another round. Stop it
                // rather than retrying forever -- the abort record has already been published to `abortState_`.
                sideClosing_.store(true, std::memory_order_release);
                return;
            }
            sc.active = true;
            return;
        }
        int done = 0;
        if (MPI_Test(&sc.req, &done, MPI_STATUS_IGNORE) != MPI_SUCCESS)
        {
            // Truncated/incomplete round after a peer's abort: not a process-fatal error. The request is dropped
            // (a collective cannot be cancelled) and the channel stops; the published record still propagates.
            sideComm_ = MPI_COMM_NULL;   // MPI owns the failed request until finalize; sc owns its buffer
            sc.req = MPI_REQUEST_NULL;
            sc.active = false;
            sideClosing_.store(true, std::memory_order_release);
            return;
        }
        // MPI_Test only releases the request when done is true. An incomplete poll must
        // leave both the request and its receive buffer owned by this round: starting the
        // next allgather here leaks live requests and lets them overwrite one another (and
        // eventually MPI's own teardown allocations).
        if (!done) return;
        sc.active = false;
        for (int r = 0; r < size_; ++r)
            if (sc.recv[r].flag) abortState_->publishRemote(r, sc.recv[r].msg);
    }

    void sideLoop(const std::shared_ptr<SideChannel>& sc, const std::shared_ptr<AbortState>& st)
    {
        while (!st->stop.load(std::memory_order_acquire))
        {
            sideRound(*sc);
            // A remote abort releases this rank's NCCL kernels even while the main thread is synchronising.
            if (st->pending.load(std::memory_order_acquire)) onSideAbort();
            usleep(1000);
        }
        // quiesceSideChannel completed the last round collectively: no abandoned NBC request is left here, and
        // the communicator is freed only after this thread has been joined.
        sc->done.store(true, std::memory_order_release);
    }

protected:
    // NCCL: abort both communicators from the watchdog thread. MPI-host: nothing to release.
    virtual void onSideAbort() {}

    void hostColl(double* h, int n, int op, int root)
    {
        status(nullptr);
        if (n == 0) return;
        gatherFold(h, n, op, root);
    }

    void deviceColl(double* d, int n, int op, int root, cudaStream_t s)
    {
        noCapture(s);
        status(nullptr);
        if (n == 0) return;
        std::vector<double> h(n);
        ck(cudaMemcpyAsync(h.data(), d, static_cast<std::size_t>(n)*sizeof(double), cudaMemcpyDeviceToHost, s), "D2H");
        ck(cudaStreamSynchronize(s), "stream sync");
        gatherFold(h.data(), n, op, root);
        ck(cudaMemcpyAsync(d, h.data(), static_cast<std::size_t>(n)*sizeof(double), cudaMemcpyHostToDevice, s), "H2D");
    }
};

} // namespace

namespace {
MpiHostComm::~MpiHostComm()
{
    cancelLive();         // no pinned/device buffer may be released with an MPI request still pointing at it
    try { quiesceSideChannel(); }
    catch (const std::exception&) { stopSideChannel(); }   // destruction still joins the thread before freeing
    stopSideChannel();
}
} // namespace

std::unique_ptr<ProcComm> ProcComm::createMpiHost()
{
    if (!Pstream::initialized()) throw std::logic_error("ProcComm::createMpiHost: Pstream::init() has not been called");
    return std::unique_ptr<ProcComm>(new MpiHostComm());
}

// ------------------------------------------------------------------------------------------------------------
// NCCL backend (CommSolvers.md step C), compiled with -DBRAE_WITH_NCCL (CMake option BRAE_WITH_NCCL).
//   one process per GPU under mpirun; bootstrap: rank 0 ncclGetUniqueId, MPI_Bcast, ncclCommInitRankConfig.
//   TWO communicators over the same ranks: `ce` (eager: warm-up, host-driven paths, function objects) and `cg`
//   (graph-only, ncclConfig graphUsageMode = 0: every op that is captured in a CUDA graph / conditional WHILE
//   body). An op uses cg when the caller asks for CommChannel::Graph or its stream is capturing; else ce.
//   exchangeCells  pack kernel -> ONE ncclGroupStart/ncclRecv.../ncclSend.../ncclGroupEnd for all neighbours (one
//                  message per neighbour carrying all components) -> unpack kernel; all on the caller's stream.
//   exchangeFaces  the same group, sending straight from the caller's face array, then the combine kernel.
//   allReduce      ncclAllGather of the n partials into a persistent buffer, then the rank-order fold kernel of the
//                  other backends (Sum, Min, Max alike: identical bits on every rank, independent of NCCL's
//                  algorithm/ring/tree; min/max would be exact with ncclAllReduce too, one code path is simpler).
//   bcast          ncclBroadcast in place (exact).
//   host ops       (barrier, allReduceHost, bcastHost, exchangeIntLists, makePlan's size check, abort) are the MPI
//                  ones of MpiHostComm: they carry the abort record. The device ops only ENQUEUE and cannot throw a
//                  remote abort; a rank learns it at its next HOST collective. Peers that already enqueued NCCL work
//                  against the failed rank would hang on the stream, so every rank that sees an abort calls
//                  ncclCommAbort on both communicators (kernels exit) and re-creates them (collective: all ranks
//                  pass the same status phase). Graphs captured before an abort are invalid afterwards.
// ------------------------------------------------------------------------------------------------------------
#ifdef BRAE_WITH_NCCL
namespace {

void ncclCk(ncclResult_t r, const char* what)
{
    if (r != ncclSuccess)
        throw std::runtime_error(std::string("ProcComm NCCL: ") + what + ": " + ncclGetErrorString(r));
}

unsigned long long hostHash()
{
    char name[256] = {0};
    gethostname(name, sizeof(name) - 1);
    unsigned long long h = 1469598103934665603ull;
    for (const char* c = name; *c; ++c) { h ^= static_cast<unsigned char>(*c); h *= 1099511628211ull; }
    return h;
}

struct NcclPlan : StagePlan
{
    // Launches every kernel the plan's ops use once, with no work, on `s`. CUDA loads kernels lazily at their first
    // launch, and a lazy load can block behind a running (spinning) NCCL kernel: the first use must therefore not
    // happen after an abort left kernels waiting for a failed peer.
    void warmKernels(cudaStream_t s) const
    {
        if (nNbr == 0) return;
        CompList cl;
        cl.n = 0; cl.base = 0;
        packCellsK<<<dim3(1, nNbr), 32, 0, s>>>(nullptr, dSendTab, cl, 1, dSend);
        unpackCellsK<<<dim3(1, nNbr), 32, 0, s>>>(dRecvTab, cl, 1, 0);
        combineFacesK<<<dim3(1, nNbr), 32, 0, s>>>(dFaceTab, dRecv, 2);
        ck(cudaGetLastError(), "warm-up kernel launch");
    }
};

class NcclComm final : public MpiHostComm
{
public:
    NcclComm() : MpiHostComm(Backend::Nccl)
    {
        if (size_ == 1) return;   // identity: no communicator, no allocation
        // one rank per GPU (NCCL refuses two ranks on one device); every rank sees the same table, so all throw
        // alike. A rank WITHOUT a CUDA device reports dev = -1 and the error is raised AFTER this allgather, so a
        // local device problem can never leave the peers inside a different collective sequence.
        struct Id { unsigned long long host; int dev; int pad; };
        Id mine{hostHash(), device_, 0};
        std::vector<Id> all(size_);
        mpiCheck(MPI_Allgather(&mine, sizeof(Id), MPI_BYTE, all.data(), sizeof(Id), MPI_BYTE, MPI_COMM_WORLD), "Allgather");
        for (int a = 0; a < size_; ++a)
            if (all[a].dev < 0)
                throw std::runtime_error("ProcComm NCCL: rank " + std::to_string(a) + " has no CUDA device");
        // bring-up hook for a node with fewer GPUs than ranks: NCCL >= 2.3x runs several ranks on one GPU when
        // NCCL_MULTI_RANK_GPU_ENABLE=1 (slow, only for testing the data path; never for measurements)
        const char* share = std::getenv("BRAE_NCCL_ALLOW_SHARED_GPU");
        if (share && share[0] == '1')
        {
            setenv("NCCL_MULTI_RANK_GPU_ENABLE", "1", 0);
            // all ranks' communicator buffers live on the one GPU: a few channels keep 8 ranks inside its memory
            setenv("NCCL_MAX_NCHANNELS", "4", 0);
            setenv("NCCL_MAX_P2P_NCHANNELS", "2", 0);
        }
        for (int a = 0; a < size_ && !(share && share[0] == '1'); ++a)
            for (int b = a + 1; b < size_; ++b)
                if (all[a].host == all[b].host && all[a].dev == all[b].dev)
                    throw std::runtime_error("ProcComm NCCL: ranks " + std::to_string(a) + " and " + std::to_string(b)
                                             + " are bound to the same GPU (device " + std::to_string(all[a].dev)
                                             + "); NCCL needs one rank per GPU: use np <= GPUs per node, or BRAE_COMM=mpi");
        ck(cudaSetDevice(device_), "cudaSetDevice");
        // graph communicator without mixing support (the per-communicator form is ncclConfig.graphUsageMode; this
        // env is the fallback for an NCCL without it and affects only comms that are never launched eagerly)
        setenv("NCCL_GRAPH_MIXING_SUPPORT", "0", 0);
        const char* e = std::getenv("BRAE_COMM_REDUCE_CAPACITY");
        if (e && std::atoi(e) > 0) cap_ = static_cast<std::size_t>(std::atoi(e));
        ck(cudaStreamCreateWithFlags(&wst_, cudaStreamNonBlocking), "warm-up stream");
        for (int c = 0; c < 2; ++c) gath_[c] = dalloc<double>(cap_*static_cast<std::size_t>(size_));
        scratch_ = dalloc<double>(8);
        initComms();
    }

    ~NcclComm() override
    {
        if (size_ == 1) return;
        // Every rank must arrive here together: finish the side channel's outstanding NBC round BEFORE freeing
        // either NCCL or MPI communicators. The base destructor repeats this idempotently after NCCL teardown.
        try { quiesceSideChannel(); }
        catch (const std::exception&) { stopSideChannel(); }
        stopSideChannel();    // joins the watchdog before the communicators it polls are destroyed
        std::lock_guard<std::mutex> lk(ncclM_);   // no watchdog abort may interleave with NCCL teardown
        cudaStreamSynchronize(wst_);
        // The two communicators are PROCESS-LIFETIME. They are deliberately NOT destroyed here: once conditional-WHILE
        // solver graphs have run on the graph communicator, ncclCommFinalize/Destroy/Abort all funnel into NCCL's
        // commDestroySync, which waits for a GLOBAL quiescence that never arrives (measured at np=2 on one shared GPU:
        // both ranks stuck in commDestroySync with every graph already destroyed and the device synchronised; the
        // results were correct and complete). Destroying them would hang the job at exit. The OS reclaims the device
        // and host resources when the process ends. Abort/re-create uses initComms() directly and is unaffected.
        // gath_/scratch_/wst_ are likewise left to the process teardown: freeing them while NCCL still references them
        // is the same hazard.
        comm_[0] = nullptr;
        comm_[1] = nullptr;
    }


    bool graphCapturable() const override { return true; }

protected:
    std::unique_ptr<HaloPlan> doMakePlan(HaloPlanDesc desc, int maxComps) override
    {
        status(nullptr);
        if (!broken_.empty()) throw std::runtime_error("ProcComm NCCL: communicators unusable after an abort: " + broken_);
        std::unique_ptr<HaloPlan> plan = newPlan(std::move(desc));
        auto st = std::make_unique<NcclPlan>();
        NcclPlan* sp = st.get();
        const HaloPlanDesc& d = detail::PlanAccess::desc(*plan);
        sp->init(d, maxComps, true);
        std::vector<RecvEntry> rt(d.nbrs.size());
        std::vector<FaceEntry> ft(d.nbrs.size());
        for (std::size_t k = 0; k < d.nbrs.size(); ++k)
        {
            const HaloNeighbour& n = d.nbrs[k];
            rt[k] = {sp->dRecv, n.recvStart, n.recvCount, n.recvStart};
            ft[k] = {sp->dRecv, n.faceStart, n.faceCount, n.faceStart};
        }
        sp->setTables(rt, ft);
        setPlanState(*plan, std::move(st));
        verifyPlanSizes(d);
        // Warm-up: the lazy peer-to-peer connections of BOTH communicators are set up here with the largest
        // messages this plan will send, so that no first-use latency/allocation happens in a timed region or a
        // capture. A rank whose peers already aborted must NOT enter it: NCCL work against a rank that left
        // would hang the stream, and the abort is already on its way through the side channel.
        {
            std::string msg;
            int origin = -1;
            if (sideAborted(msg, origin)) { onSideAbort(); throw ProcCommAbort(msg, origin); }
        }
        for (int c = 0; c < 2; ++c)
        {
            sendRecvGroup(c, wst_, d, sp->dSend, sp->dRecv, maxComps, true);
            sendRecvGroup(c, wst_, d, sp->dSend, sp->dRecv, 1, false);
            ck(cudaStreamSynchronize(wst_), "warm-up sync");
        }
        sp->warmKernels(wst_);
        ck(cudaStreamSynchronize(wst_), "warm-up sync");
        return plan;
    }

    void doExchangeCells(const HaloPlan& plan, const HaloField* fields, int nFields, cudaStream_t s, CommChannel ch) override
    {
        s = realStream(s);
        FlatComps fc;
        flatten(fields, nFields, fc);
        auto* st = static_cast<NcclPlan*>(planState(plan));
        deviceLocal([&] { if (fc.n) checkCapacity(*st, fc.n); }, s);
        if (fc.n == 0) return;
        const int c = channel(s, ch);
        st->pack(plan, fc, s);
        sendRecvGroup(c, s, *st->desc, st->dSend, st->dRecv, fc.n, true);
        st->unpack(plan, fc, s);
    }

    void doExchangeFaces(const HaloPlan& plan, double* arr, FaceOp op, cudaStream_t s, CommChannel ch) override
    {
        s = realStream(s);
        auto* st = static_cast<NcclPlan*>(planState(plan));
        deviceLocal([] {}, s);
        const HaloPlanDesc& d = *st->desc;
        if (d.nProcFaces == 0) return;
        const int c = channel(s, ch);
        grouped([&] {
            for (const HaloNeighbour& n : d.nbrs)
            {
                if (!n.faceCount) continue;
                ncclCk(ncclRecv(st->dRecv + n.faceStart, n.faceCount, ncclDouble, n.rank, comm_[c], s), "ncclRecv");
                ncclCk(ncclSend(arr + n.faceStart, n.faceCount, ncclDouble, n.rank, comm_[c], s), "ncclSend");
            }
        });
        st->combineFaces(arr, op, s);
    }

    void doAllReduceDevice(double* d, int n, ReduceOp op, cudaStream_t s, CommChannel ch) override
    {
        deviceLocal([&] {
            if (n > 0 && static_cast<std::size_t>(n) > cap_)
                throw std::length_error("ProcComm NCCL: device reduction of " + std::to_string(n) + " doubles exceeds the capacity "
                                        + std::to_string(cap_) + " (env BRAE_COMM_REDUCE_CAPACITY)");
        }, s);
        if (n <= 0) return;
        s = realStream(s);
        const int c = channel(s, ch);
        ncclCk(ncclAllGather(d, gath_[c], static_cast<std::size_t>(n), ncclDouble, comm_[c], s), "ncclAllGather");
        SlotList sl;
        sl.n = size_;
        for (int r = 0; r < size_; ++r) sl.p[r] = gath_[c] + static_cast<std::size_t>(r)*n;
        foldSlotsK<<<gridFor(n), 256, 0, s>>>(d, sl, n, opCode(op), 0);
        ck(cudaGetLastError(), "fold launch");
    }

    void doBcastDevice(double* d, int n, int root, cudaStream_t s, CommChannel ch) override
    {
        s = realStream(s);
        deviceLocal([&] { if (root < 0 || root >= size_) throw std::invalid_argument("ProcComm::bcast: root out of range"); }, s);
        if (n <= 0) return;
        const int c = channel(s, ch);
        ncclCk(ncclBroadcast(d, d, static_cast<std::size_t>(n), ncclDouble, root, comm_[c], s), "ncclBroadcast");
    }

    void abortSeen() override
    {
        // Once per communicator generation. The watchdog can publish and abort before the status phase; repeated
        // throws on later ranks must not revoke a partially recovered generation a second time.
        if (recoveryDone_) return;
        recoveryDone_ = true;
        std::lock_guard<std::mutex> lk(ncclM_);   // the watchdog's ncclCommAbort runs outside this whole recovery
        // Every rank passes through here at the same status phase. Peers may have NCCL kernels enqueued against
        // the failed rank that never complete: stop them, then rebuild both communicators collectively.
        // A LOCAL failure of revoke/quiesce/destroy must NEVER skip the collective barrier: the local outcome is
        // captured, every rank reaches the barrier and an allreduce of the status flags, and only the AGREED
        // result decides whether the communicators are re-created. One rank's revoke error therefore leaves every
        // rank in the same (broken) state with the same message instead of hanging the others in MPI_Barrier.
        // NCCL_REVOKE: phase 1 stops every in-flight kernel of this rank WITHOUT freeing what a peer's kernel may
        // still touch (ncclCommAbort frees at once: a peer that has not aborted yet could then read freed
        // IPC/SHM buffers), an MPI barrier makes all ranks quiescent, only then are the communicators destroyed.
        // Older NCCL without ncclCommRevoke falls back to ncclCommAbort on each rank (that race is possible there).
        std::string err;
        auto step = [&](const char* what, auto&& f) {
            if (!err.empty()) return;      // keep the first local failure; still reach the collectives below
            try { f(); }
            catch (const std::exception& e) { err = std::string(what) + ": " + e.what(); }
            catch (...) { err = std::string(what) + ": unknown failure"; }
        };
        const int forceFail = revokeFailRank_();   // test hook: force a local revoke/quiesce failure
        step("ncclCommRevoke", [&] {
#ifdef NCCL_REVOKE_DEFAULT
            for (int c = 0; c < 2; ++c)
                if (comm_[c])
                {
                    if (rank_ == forceFail) throw std::runtime_error("injected revoke failure");
                    ncclCk(ncclCommRevoke(comm_[c], NCCL_REVOKE_DEFAULT), "ncclCommRevoke");
                }
#else
            for (int c = 0; c < 2; ++c)
                if (comm_[c])
                {
                    if (rank_ == forceFail) throw std::runtime_error("injected revoke failure");
                    ncclCommAbort(comm_[c]);
                    comm_[c] = nullptr;
                }
#endif
        });
        step("quiesce", [&] {
#ifdef NCCL_REVOKE_DEFAULT
            for (int c = 0; c < 2; ++c)
            {
                if (!comm_[c]) continue;
                ncclResult_t st = ncclInProgress;
                for (int spin = 0; st == ncclInProgress; ++spin)
                {
                    ncclCk(ncclCommGetAsyncError(comm_[c], &st), "ncclCommGetAsyncError");
                    if (spin > 600000) throw std::runtime_error("ProcComm NCCL: communicator did not quiesce after ncclCommRevoke");
                    if (st == ncclInProgress) usleep(100);
                }
            }
#endif
        });
        // Collective agreement, reached by EVERY rank whatever happened above, and BOUNDED: a peer that already
        // left the sequence after its own ProcCommAbort cannot strand this rank in a blocking allreduce/barrier.
        // A failed agreement marks the communicators unusable on every rank alike instead of hanging.
        int localBad = err.empty() ? 0 : 1, anyBad = 0;
        bool agreed = false;
        anyBad = agreeMax(localBad, agreed);
        if (!agreed && err.empty()) err = "ProcComm NCCL: the abort recovery could not agree on the ranks' status";
        int dummy = 0;
        bool barrierDone = false;
        agreeMax(dummy, barrierDone);   // the quiescence barrier, with the same bound
        if (agreed && barrierDone && anyBad == 0)
        {
            step("ncclCommDestroy", [&] {
                for (int c = 0; c < 2; ++c) if (comm_[c]) { ncclCommDestroy(comm_[c]); comm_[c] = nullptr; }
            });
            step("re-init", [&] {
                ck(cudaStreamSynchronize(wst_), "warm-up stream sync");
                initComms();
            });
            localBad = err.empty() ? 0 : 1;
            anyBad = 0;
            bool agreed2 = false;
            anyBad = agreeMax(localBad, agreed2);
            agreed = agreed && agreed2;
        }
        for (int c = 0; c < 2; ++c)
            if (comm_[c] && (anyBad != 0 || !agreed)) { ncclCommAbort(comm_[c]); comm_[c] = nullptr; }
        if (anyBad != 0 || !err.empty() || !agreed)
        {
            if (err.empty()) err = "ProcComm NCCL: a peer failed to re-create its communicators";
            broken_ = err;   // the abort itself is still delivered; device ops refuse until the process restarts
        }
        else { broken_.clear(); abortedByWatchdog_.store(false); }
    }

    // BRAE_NCCL_FAIL_REVOKE=<rank>: the negative control forces that rank's revoke step to throw.
    static int revokeFailRank_()
    {
        const char* e = std::getenv("BRAE_NCCL_FAIL_REVOKE");
        return e ? std::atoi(e) : -1;
    }

    // Called by the abort side channel (background thread, or the main thread at an op entry). Releases the NCCL
    // kernels that are waiting on the failed rank, so a cudaStreamSynchronize in the solver returns and the rank
    // reaches the host collective that throws the shared ProcCommAbort. Never throws.
    void onSideAbort() override
    {
        if (abortedByWatchdog_.exchange(true)) return;
        // NCCL is not re-entrant across threads: the watchdog must not run ncclCommAbort while the main thread
        // is inside ncclCommRevoke / recovery / destruction (it deadlocks in commFree). Take the same lock the
        // recovery and the destructor hold, so the abort happens between two main-thread NCCL phases.
        std::lock_guard<std::mutex> lk(ncclM_);
        for (int c = 0; c < 2; ++c)
            if (comm_[c]) { ncclCommAbort(comm_[c]); comm_[c] = nullptr; }   // releases blocked NCCL kernels
    }
    std::mutex ncclM_;                          // serializes NCCL calls: watchdog abort vs. recovery/destruction

private:
    ncclComm_t comm_[2] = {nullptr, nullptr};   // 0 eager, 1 graph-only
    double* gath_[2] = {nullptr, nullptr};      // [cap_*size_] allgather targets, one per channel
    double* scratch_ = nullptr;
    std::size_t cap_ = 4096;                    // doubles per rank of one device reduction
    std::string broken_;                        // set when re-creating the communicators after an abort failed
    std::atomic<bool> abortedByWatchdog_{false};  // onSideAbort runs once per communicator generation
    bool recoveryDone_ = false;                 // abortSeen runs once per communicator generation
    int opStatus_ = -1;                             // BRAE_NCCL_OP_STATUS=0: pure enqueue-only device ops
    cudaStream_t wst_ = nullptr;

    // `0` means the per-thread default stream in this TU (compiled --default-stream per-thread), but NCCL (a separate
    // library) would read it as the legacy stream: pass the explicit handle.
    static cudaStream_t realStream(cudaStream_t s) { return s ? s : cudaStreamPerThread; }

    static bool capturing(cudaStream_t s)
    {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        ck(cudaStreamIsCapturing(s, &cs), "cudaStreamIsCapturing");
        return cs != cudaStreamCaptureStatusNone;
    }
    int channel(cudaStream_t s, CommChannel ch)
    {
        if (!broken_.empty()) throw std::runtime_error("ProcComm NCCL: communicators unusable after an abort: " + broken_);
        checkSideAbort();       // a remote abort reached through the side channel: one atomic load when healthy
        if (abortedByWatchdog_.load()) throw ProcCommAbort("ProcComm NCCL: communicators were aborted after a peer failure", rank_);
        return (ch == CommChannel::Graph || capturing(s)) ? 1 : 0;
    }

    // Device ops cannot join a host status phase while they are being CAPTURED (they only enqueue), so the abort
    // of a peer is delivered by the side channel. Every rank throws the SAME ProcCommAbort. `err` is a rank-local
    // validation failure (capacity, injected error): outside a capture it is agreed with the status allgather
    // BEFORE any NCCL work is enqueued, so no peer ever waits on the failed rank's contribution. Inside a capture
    // there is no host collective to agree with: the local error is published to the side channel and thrown, and
    // the peers' watchdogs release their NCCL kernels.
    void checkSideAbort(const std::string& err = std::string())
    {
        pollSide();
        if (opStatus_ < 0) opStatus_ = [] {
            const char* e = std::getenv("BRAE_NCCL_OP_STATUS");
            return e && e[0] == '0' ? 0 : 1;
        }();
        if (opStatus_ && !err.empty())
        {
            publishAbort(err);
            status(&err);            // throws ProcCommAbort on every rank, including this one
        }
        else if (!err.empty())
        {
            publishAbort(err);       // capture / opted out: the peers' watchdogs release their NCCL kernels
            throw ProcCommAbort(err, rank_);
        }
        std::string msg;
        int origin = -1;
        if (sideAborted(msg, origin)) { onSideAbort(); throw ProcCommAbort(msg, origin); }
    }

    // Local, fallible validation shared by the device ops; runs before any NCCL call.
    template<class F> void deviceLocal(F&& f, cudaStream_t s = nullptr)
    {
        std::string err;
        try { f(); }
        catch (const std::exception& e) { err = e.what(); }
        catch (...) { err = "unknown local failure"; }
        if (const char* inj = std::getenv("BRAE_COMM_FAIL_AFTER_STATUS"); inj && inj[0] == '1' && err.empty())
            err = "injected post-status failure";
        // A host allgather is illegal inside a CUDA-graph capture (and would be recorded nowhere): only the side
        // channel is used there. Eager ops agree on local failures BEFORE enqueueing any NCCL work.
        if (s && capturing(s))
        {
            pollSide();
            if (!err.empty()) { publishAbort(err); throw ProcCommAbort(err, rank_); }
            std::string msg;
            int origin = -1;
            if (sideAborted(msg, origin)) { onSideAbort(); throw ProcCommAbort(msg, origin); }
            return;
        }
        checkSideAbort(err);
    }

    template<class F> static void grouped(F&& f)
    {
        ncclCk(ncclGroupStart(), "ncclGroupStart");
        try { f(); }
        catch (...) { ncclGroupEnd(); throw; }
        ncclCk(ncclGroupEnd(), "ncclGroupEnd");
    }

    // One group with every neighbour's receive and send (C components per cell); zero-length neighbours are skipped.
    // NCCL's group semantics make the order inside the group irrelevant (no deadlock for any neighbour graph).
    void sendRecvGroup(int c, cudaStream_t s, const HaloPlanDesc& d, double* dSend, double* dRecv, int C, bool cells)
    {
        if (cells)
        {
            grouped([&] {
                for (const HaloNeighbour& n : d.nbrs)
                {
                    if (n.recvCount)
                        ncclCk(ncclRecv(dRecv + static_cast<std::size_t>(C)*n.recvStart, static_cast<std::size_t>(C)*n.recvCount,
                                        ncclDouble, n.rank, comm_[c], s), "ncclRecv");
                    if (n.sendCount)
                        ncclCk(ncclSend(dSend + static_cast<std::size_t>(C)*n.sendOffset, static_cast<std::size_t>(C)*n.sendCount,
                                        ncclDouble, n.rank, comm_[c], s), "ncclSend");
                }
            });
        }
        else
        {
            grouped([&] {
                for (const HaloNeighbour& n : d.nbrs)
                {
                    if (!n.faceCount) continue;
                    ncclCk(ncclRecv(dRecv + n.faceStart, n.faceCount, ncclDouble, n.rank, comm_[c], s), "ncclRecv");
                    ncclCk(ncclSend(dSend + n.faceStart, n.faceCount, ncclDouble, n.rank, comm_[c], s), "ncclSend");
                }
            });
        }
    }

    // Both communicators, then one allgather on each (their kernels and rings exist before the first real use).
    void initComms()
    {
        for (int c = 0; c < 2; ++c)
        {
            struct Boot { int ok; int pad; ncclUniqueId id; } boot;
            std::memset(&boot, 0, sizeof(boot));
            std::string why;
            if (rank_ == 0)
            {
                const ncclResult_t r = ncclGetUniqueId(&boot.id);
                boot.ok = r == ncclSuccess;
                if (!boot.ok) why = ncclGetErrorString(r);
            }
            mpiCheck(MPI_Bcast(&boot, sizeof(boot), MPI_BYTE, 0, MPI_COMM_WORLD), "Bcast");
            if (!boot.ok) throw std::runtime_error("ProcComm NCCL: ncclGetUniqueId failed on rank 0");
            ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
#ifdef BRAE_NCCL_HAVE_GRAPH_USAGE_MODE
            if (c == 1) cfg.graphUsageMode = 0;
#endif
            ncclCk(ncclCommInitRankConfig(&comm_[c], size_, boot.id, rank_, &cfg), "ncclCommInitRankConfig");
        }
        for (int c = 0; c < 2; ++c)
        {
            ncclCk(ncclAllGather(scratch_, gath_[c], 1, ncclDouble, comm_[c], wst_), "warm-up ncclAllGather");
            SlotList sl;
            sl.n = size_;
            for (int r = 0; r < size_; ++r) sl.p[r] = gath_[c] + r;
            for (int op = 0; op < 3; ++op) foldSlotsK<<<1, 32, 0, wst_>>>(scratch_ + 1, sl, 1, op, 0);   // module load of the fold kernel
            ck(cudaGetLastError(), "warm-up fold launch");
            ck(cudaStreamSynchronize(wst_), "warm-up sync");
        }
    }
};

} // namespace
#endif // BRAE_WITH_NCCL

bool ProcComm::ncclBuilt()
{
#ifdef BRAE_WITH_NCCL
    return true;
#else
    return false;
#endif
}

std::unique_ptr<ProcComm> ProcComm::createNccl()
{
#ifdef BRAE_WITH_NCCL
    if (!Pstream::initialized()) throw std::logic_error("ProcComm::createNccl: Pstream::init() has not been called");
    return std::unique_ptr<ProcComm>(new NcclComm());
#else
    throw std::runtime_error("ProcComm NCCL backend: this build has no NCCL (configure with -DBRAE_WITH_NCCL=ON and NCCL_ROOT); use loopback or mpi");
#endif
}

const char* ProcComm::backendName(Backend b)
{
    return b == Backend::Loopback ? "loopback" : b == Backend::MpiHost ? "mpi" : "nccl";
}

namespace {
// Ranks of this job that share this host (MPI_COMM_TYPE_SHARED).
int nodeLocalRanks()
{
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, Pstream::myProcNo(), MPI_INFO_NULL, &node);
    int n = 1;
    MPI_Comm_size(node, &n);
    MPI_Comm_free(&node);
    return n;
}
} // namespace

std::vector<std::unique_ptr<ProcComm>> ProcComm::create(const std::string& backendIn, int np)
{
    std::string b = backendIn;
    if (b.empty()) { const char* e = std::getenv("BRAE_COMM"); if (e) b = e; }
    if (b.empty()) { const char* e = std::getenv("BRAE_COMM_BACKEND"); if (e) b = e; }
    if (b.empty() || b == "auto")
    {
        // NCCL when this build has it and EVERY rank of the job can own a GPU, else MPI-host. The choice is a
        // collective agreement, not a per-rank guess: a rank-specific CUDA visibility/device-query failure used
        // to send that rank to the other backend (or skip the node-local collective) and hang the job at start-up
        // (review finding 8). Every rank publishes {nccl built, device count, node-local rank count}; the MIN of
        // the eligibility flags and the agreed device/rank record decide, and a rank with no usable device is
        // reported by rank number on every rank.
        b = "mpi";
        if (Pstream::initialized() && Pstream::nProcs() > 1)
        {
            struct Elig { int built; int nDev; int localRanks; int localRank; };
            int nDev = 0;
            if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
            int localRanks = 1, localRank = 0;
            {
                MPI_Comm node;
                if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, Pstream::myProcNo(), MPI_INFO_NULL, &node) == MPI_SUCCESS)
                {
                    MPI_Comm_size(node, &localRanks);
                    MPI_Comm_rank(node, &localRank);
                    MPI_Comm_free(&node);
                }
            }
            Elig mine{ncclBuilt() ? 1 : 0, nDev, localRanks, localRank};
            std::vector<Elig> all(Pstream::nProcs());
            MPI_Allgather(&mine, sizeof(Elig), MPI_BYTE, all.data(), sizeof(Elig), MPI_BYTE, MPI_COMM_WORLD);
            int ok = 1;
            for (const Elig& e : all)
                if (!e.built || e.nDev <= 0 || e.localRanks > e.nDev) ok = 0;
            if (ok) b = "nccl";
        }
        else if (ncclBuilt() && Pstream::initialized() && Pstream::nProcs() > 1)
        {
            b = "nccl";
        }
    }
    std::vector<std::unique_ptr<ProcComm>> v;
    if (b == "loopback")
    {
        if (np <= 0) { const char* e = std::getenv("BRAE_COMM_NP"); np = e ? std::atoi(e) : 1; }
        return createLoopback(np);
    }
    if (b == "mpi") { v.push_back(createMpiHost()); return v; }
    if (b == "nccl") { v.push_back(createNccl()); return v; }
    throw std::invalid_argument("ProcComm::create: unknown backend '" + b + "' (loopback | mpi | nccl | auto)");
}

// ------------------------------------------------------------------------------------------------------------
// synthetic partitioner
// ------------------------------------------------------------------------------------------------------------
SyntheticPartition partitionBox(int nx, int ny, int nz, int np, const std::vector<int>& owner)
{
    const long long N = static_cast<long long>(nx)*ny*nz;
    if (nx < 1 || ny < 1 || nz < 1 || static_cast<long long>(owner.size()) != N)
        throw std::invalid_argument("partitionBox: owner size != nx*ny*nz");
    SyntheticPartition sp;
    sp.np = np; sp.nx = nx; sp.ny = ny; sp.nz = nz; sp.owner = owner;
    sp.cellGlobal.assign(np, {});
    std::vector<int> local(N);
    for (int g = 0; g < N; ++g)
    {
        if (owner[g] < 0 || owner[g] >= np) throw std::invalid_argument("partitionBox: owner out of range");
        local[g] = static_cast<int>(sp.cellGlobal[owner[g]].size());
        sp.cellGlobal[owner[g]].push_back(g);
    }
    for (int r = 0; r < np; ++r)
        if (sp.cellGlobal[r].empty()) throw std::invalid_argument("partitionBox: rank " + std::to_string(r) + " owns no cell");

    struct Face { long long key; int mine, nbr; };   // global ids
    std::map<std::pair<int, int>, std::vector<Face>> pairs;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const int g = i + nx*(j + ny*k);
                const int nb[3] = {i + 1 < nx ? g + 1 : -1, j + 1 < ny ? g + nx : -1, k + 1 < nz ? g + nx*ny : -1};
                for (int dir = 0; dir < 3; ++dir)
                {
                    const int g2 = nb[dir];
                    if (g2 < 0 || owner[g] == owner[g2]) continue;
                    const long long key = static_cast<long long>(g)*3 + dir;   // g < g2: the lower id keys the face
                    pairs[{owner[g], owner[g2]}].push_back({key, g, g2});
                    pairs[{owner[g2], owner[g]}].push_back({key, g2, g});
                }
            }
    sp.patches.assign(np, {});
    sp.faceKey.assign(np, {});
    sp.haloGlobal.assign(np, {});
    sp.descs.assign(np, {});
    std::vector<std::vector<int>> nbrGlobal(np);   // per rank, per processor face: global id of the neighbour cell
    for (auto& kv : pairs)
    {
        const int r = kv.first.first, p = kv.first.second;
        std::vector<Face>& fs = kv.second;
        std::sort(fs.begin(), fs.end(), [](const Face& a, const Face& b) { return a.key < b.key; });
        ProcessorPatchDesc pd;
        pd.nbrRank = p;
        for (const Face& f : fs)
        {
            pd.faceCells.push_back(local[f.mine]);
            pd.nbrFaceCells.push_back(local[f.nbr]);
            sp.faceKey[r].push_back(f.key);
            nbrGlobal[r].push_back(f.nbr);
        }
        sp.patches[r].push_back(std::move(pd));
    }
    for (int r = 0; r < np; ++r)
    {
        sp.descs[r] = buildHaloPlanFromProcessorPatches(static_cast<int>(sp.cellGlobal[r].size()), sp.patches[r]);
        sp.haloGlobal[r].assign(sp.descs[r].nHalo, -1);
        for (int i = 0; i < sp.descs[r].nProcFaces; ++i) sp.haloGlobal[r][sp.descs[r].faceToSlot[i]] = nbrGlobal[r][i];
    }
    return sp;
}

std::vector<int> blockOwner(int nx, int ny, int nz, int px, int py, int pz)
{
    auto axis = [](int n, int P) {
        std::vector<int> b(n);
        for (int i = 0; i < n; ++i)
            for (int q = 0; q < P; ++q)
                if (i < static_cast<int>((static_cast<long long>(q + 1)*n)/P)) { b[i] = q; break; }
        return b;
    };
    const std::vector<int> bx = axis(nx, px), by = axis(ny, py), bz = axis(nz, pz);
    std::vector<int> o(static_cast<std::size_t>(nx)*ny*nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) o[i + nx*(j + static_cast<std::size_t>(ny)*k)] = bx[i] + px*(by[j] + py*bz[k]);
    return o;
}

std::vector<int> scatteredOwner(int nx, int ny, int nz, int np, int chunk)
{
    const int cx = (nx + chunk - 1)/chunk, cy = (ny + chunk - 1)/chunk, cz = (nz + chunk - 1)/chunk;
    std::vector<int> chunkRank(static_cast<std::size_t>(cx)*cy*cz);
    for (int k = 0; k < cz; ++k)
        for (int j = 0; j < cy; ++j)
            for (int i = 0; i < cx; ++i)
            {
                const unsigned h = static_cast<unsigned>(i)*73856093u ^ static_cast<unsigned>(j)*19349663u ^ static_cast<unsigned>(k)*83492791u;
                chunkRank[i + cx*(j + static_cast<std::size_t>(cy)*k)] = static_cast<int>((h >> 3) % static_cast<unsigned>(np));
            }
    std::vector<int> cnt(np, 0);
    for (int r : chunkRank) ++cnt[r];
    for (int r = 0; r < np && r < static_cast<int>(chunkRank.size()); ++r)
        if (cnt[r] == 0) chunkRank[r] = r;
    std::vector<int> o(static_cast<std::size_t>(nx)*ny*nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                o[i + nx*(j + static_cast<std::size_t>(ny)*k)] =
                    chunkRank[(i/chunk) + cx*((j/chunk) + static_cast<std::size_t>(cy)*(k/chunk))];
    return o;
}

} // namespace brae
