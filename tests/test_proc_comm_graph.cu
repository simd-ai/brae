// ProcComm CUDA-graph gate: on the LOOPBACK backend one step of an SPMD "solver" (batched halo exchange of
// cell fields + a processor-face exchange (min) + a device-resident allReduce sum and max + a device-resident
// bcast, with kernels between them) is captured into ONE joint CUDA graph for all virtual ranks and replayed
// 1000 times; the state of every rank is compared with an integer reference after every replay. All values are
// integers below 2^53, so the comparison is bitwise independent of fma/ordering.
//
//   test_proc_comm_graph                loopback np = 2,3,4,8
//   mpirun -np 2 test_proc_comm_graph mpi    the MPI-host backend must REFUSE to be captured (throws logic_error)
//   mpirun -np N test_proc_comm_graph nccl   NCCL backend (BRAE_WITH_NCCL), one process per GPU: EVERY rank captures its
//                                            OWN graph (no joint graph): (a) the step as a plain graph, 1000 replays;
//                                            (b) the step inside the body of a conditional WHILE node (device-side loop
//                                            count decided by the allreduced sum, varies between replays), 1000 replays.
//                                            Every replay is checked bit for bit against the integer reference.
#include "proc_comm.cuh"
#include "cuda_graph_compat.cuh"

#include <mpi.h>

#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <thread>

using namespace brae;

namespace {

void cu(cudaError_t e, const char* w)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string("cuda ") + w + ": " + cudaGetErrorString(e));
}

#define REQUIRE(cond, ...)                                                                                          \
    do { if (!(cond)) {                                                                                             \
        char b_[512]; std::snprintf(b_, sizeof b_, __VA_ARGS__);                                                    \
        throw std::runtime_error(std::string(b_) + "  [" #cond "] at line " + std::to_string(__LINE__)); } } while (0)

constexpr double kMod = 1000003.0;

__global__ void wK(const double* u, double* w, int nC)
{
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < nC; c += gridDim.x*blockDim.x) w[c] = 2.0*u[c] + 1.0;
}
__global__ void faceK(const double* u, const int* faceCells, double* fa, int nf)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nf; i += gridDim.x*blockDim.x) fa[i] = u[faceCells[i]];
}
__global__ void sumK(const double* u, const double* w, const double* fa, int nC, int nH, int nf, double* partial, double* stamp)
{
    double s = 0, m = 0;
    for (int k = 0; k < nH; ++k) { s += u[nC + k]; const double v = w[nC + k]; m = v > m ? v : m; }
    for (int i = 0; i < nf; ++i) s += fa[i];
    partial[0] = s;
    partial[1] = m;
    stamp[0] = u[0];
}
__global__ void updateK(double* u, const int* glob, int nC, const double* partial, const double* stamp)
{
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < nC; c += gridDim.x*blockDim.x)
        u[c] = fmod(3.0*u[c] + partial[0] + partial[1] + static_cast<double>(glob[c] % 7) + stamp[0], kMod);
}

#if CUDART_VERSION >= 12040
// WHILE bookkeeping: the loop count is decided on the device from the allreduced sum (identical on every rank)
constexpr double kWhileMod = 3.0;
constexpr int kWhileMax = 24;
__global__ void whileInitK(cudaGraphConditionalHandle h, int* iter) { *iter = 0; cudaGraphSetConditional(h, 1u); }
__global__ void whileCondK(cudaGraphConditionalHandle h, const double* partial, int* iter)
{
    const int it = ++*iter;
    cudaGraphSetConditional(h, (fmod(partial[0], kWhileMod) != 0.0 && it < kWhileMax) ? 1u : 0u);
}
#endif

template<class T> struct DBuf
{
    T* p = nullptr;
    std::size_t n;
    explicit DBuf(std::size_t n_) : n(n_) { cu(cudaMalloc(&p, std::max<std::size_t>(n, 1)*sizeof(T)), "malloc"); cu(cudaMemsetAsync(p, 0, std::max<std::size_t>(n, 1)*sizeof(T), cudaStreamPerThread), "memset"); cu(cudaStreamSynchronize(cudaStreamPerThread), "memset sync"); }
    template<class V> explicit DBuf(const std::vector<V>& v) : DBuf(v.size())
    {
        if (!v.empty())
        {
            cu(cudaMemcpyAsync(p, v.data(), v.size()*sizeof(V), cudaMemcpyHostToDevice, cudaStreamPerThread), "upload");
            cu(cudaStreamSynchronize(cudaStreamPerThread), "upload sync");
        }
    }
    DBuf(const DBuf&) = delete;
    ~DBuf() { cudaFree(p); }
};

struct RankState
{
    std::unique_ptr<HaloPlan> plan;
    std::unique_ptr<DBuf<double>> u, w, fa, partial, stamp;
    std::unique_ptr<DBuf<int>> glob;
    cudaStream_t stream = nullptr;
};

int blocks(int n) { return std::max(1, std::min((n + 255)/256, 4096)); }

// One step, enqueued on the rank's stream. Identical code eager and captured.
void rankStep(ProcComm& c, RankState& st, cudaStream_t s)
{
    const HaloPlan& pl = *st.plan;
    const int nC = pl.nCells(), nH = pl.nHalo(), nf = pl.nProcFaces();
    wK<<<blocks(nC), 256, 0, s>>>(st.u->p, st.w->p, nC);
    faceK<<<blocks(nf), 256, 0, s>>>(st.u->p, pl.dFaceCells(), st.fa->p, nf);
    c.exchangeCells(pl, {st.u->p, st.w->p}, s, CommChannel::Graph);
    c.exchangeFaces(pl, st.fa->p, FaceOp::Min, s, CommChannel::Graph);
    sumK<<<1, 1, 0, s>>>(st.u->p, st.w->p, st.fa->p, nC, nH, nf, st.partial->p, st.stamp->p);
    c.allReduce(st.partial->p, 1, ReduceOp::Sum, s, CommChannel::Graph);
    c.allReduce(st.partial->p + 1, 1, ReduceOp::Max, s, CommChannel::Graph);
    c.bcastDevice(st.stamp->p, 1, c.size() - 1, s, CommChannel::Graph);
    updateK<<<blocks(nC), 256, 0, s>>>(st.u->p, st.glob->p, nC, st.partial->p, st.stamp->p);
    cu(cudaGetLastError(), "step kernels");
}

// reference: the same step on global integer arrays
long long refStep(const SyntheticPartition& sp, std::vector<long long>& U)
{
    long long total = 0, M = 0;
    for (int r = 0; r < sp.np; ++r)
    {
        const HaloPlanDesc& d = sp.descs[r];
        long long s = 0, m = 0;
        for (int k = 0; k < d.nHalo; ++k)
        {
            const long long v = U[sp.haloGlobal[r][k]];
            s += v;
            m = std::max(m, 2*v + 1);
        }
        for (int i = 0; i < d.nProcFaces; ++i)
            s += std::min(U[sp.cellGlobal[r][d.faceCells[i]]], U[sp.haloGlobal[r][d.faceToSlot[i]]]);
        total += s;
        M = std::max(M, m);
    }
    const long long stamp = U[sp.cellGlobal[sp.np - 1][0]];
    for (std::size_t g = 0; g < U.size(); ++g)
        U[g] = static_cast<long long>(std::fmod(static_cast<double>(3*U[g] + total + M + static_cast<long long>(g % 7) + stamp), kMod));
    return total;
}

void checkState(const SyntheticPartition& sp, const std::vector<std::unique_ptr<RankState>>& rs, const std::vector<long long>& U, const char* when)
{
    for (int r = 0; r < sp.np; ++r)
    {
        const int nC = rs[r]->plan->nCells();
        std::vector<double> h(nC);
        cu(cudaMemcpy(h.data(), rs[r]->u->p, nC*sizeof(double), cudaMemcpyDeviceToHost), "download");
        for (int c = 0; c < nC; ++c)
            REQUIRE(h[c] == static_cast<double>(U[sp.cellGlobal[r][c]]), "%s: rank %d cell %d: %.17g != %lld", when, r, c, h[c], U[sp.cellGlobal[r][c]]);
    }
}

struct Watchdog
{
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::thread t;
    Watchdog(const char* what, int seconds)
    {
        t = std::thread([this, what, seconds] {
            std::unique_lock<std::mutex> lk(m);
            if (!cv.wait_for(lk, std::chrono::seconds(seconds), [&] { return done; }))
            {
                std::printf("FAIL: watchdog: '%s' hung for %d s\n", what, seconds);
                std::fflush(stdout);
                std::_Exit(3);
            }
        });
    }
    ~Watchdog() { { std::lock_guard<std::mutex> lk(m); done = true; } cv.notify_all(); t.join(); }
};

int runGraphCase(const std::string& name, const SyntheticPartition& sp, int replays)
{
    const int np = sp.np;
    Watchdog wd(name.c_str(), 240);
    std::vector<std::unique_ptr<ProcComm>> comms = ProcComm::createLoopback(np);
    std::vector<std::unique_ptr<RankState>> rs(np);
    std::vector<std::string> err(np);
    cudaEvent_t fork = nullptr;
    std::vector<cudaEvent_t> join(np, nullptr);
    cu(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "event");
    for (auto& e : join) cu(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<long long> U(sp.owner.size());
    for (std::size_t g = 0; g < U.size(); ++g) U[g] = static_cast<long long>((g*7919 + 13) % 1000003);

    std::vector<std::thread> th;
    for (int r = 0; r < np; ++r)
        th.emplace_back([&, r] {
            ProcComm& c = *comms[r];
            try
            {
                c.bindThread();
                rs[r].reset(new RankState);
                RankState& st = *rs[r];
                cu(cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking), "stream");
                st.plan = c.makePlan(sp.descs[r], 4);
                const int nExt = st.plan->nExt(), nC = st.plan->nCells();
                st.u.reset(new DBuf<double>(nExt));
                st.w.reset(new DBuf<double>(nExt));
                st.fa.reset(new DBuf<double>(st.plan->nProcFaces()));
                st.partial.reset(new DBuf<double>(2));
                st.stamp.reset(new DBuf<double>(1));
                st.glob.reset(new DBuf<int>(sp.cellGlobal[r]));
                std::vector<double> u0(nC);
                for (int i = 0; i < nC; ++i) u0[i] = static_cast<double>((static_cast<std::size_t>(sp.cellGlobal[r][i])*7919 + 13) % 1000003);
                cu(cudaMemcpy(st.u->p, u0.data(), nC*sizeof(double), cudaMemcpyHostToDevice), "upload u");
                cu(cudaStreamSynchronize(cudaStreamPerThread), "sync");
                c.barrier();
                // eager warm-up step: wakes every code path (and every lazily created resource) before capture
                rankStep(c, st, st.stream);
                cu(cudaStreamSynchronize(st.stream), "sync");
                c.barrier();
                // ---- joint capture: rank 0 forks everybody into ONE capture ----
                if (r == 0)
                {
                    cu(cudaStreamBeginCapture(st.stream, cudaStreamCaptureModeGlobal), "begin capture");
                    cu(cudaEventRecord(fork, st.stream), "fork");
                }
                c.barrier();
                if (r != 0) cu(cudaStreamWaitEvent(st.stream, fork, 0), "join capture");
                rankStep(c, st, st.stream);
                if (r != 0) cu(cudaEventRecord(join[r], st.stream), "record join");
                c.barrier();
                if (r == 0)
                {
                    for (int q = 1; q < np; ++q) cu(cudaStreamWaitEvent(st.stream, join[q], 0), "wait join");
                    cu(cudaStreamEndCapture(st.stream, &graph), "end capture");
                    cu(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
                }
                c.barrier();
            }
            catch (const std::exception& e)
            {
                err[r] = e.what();
                try { c.collectiveAbort(e.what()); } catch (...) {}
            }
        });
    for (auto& t : th) t.join();
    int fails = 0;
    for (int r = 0; r < np; ++r)
        if (!err[r].empty()) { ++fails; std::printf("FAIL %s rank %d: %s\n", name.c_str(), r, err[r].c_str()); }
    if (fails) return fails;
    try
    {
        refStep(sp, U);   // the eager step
        checkState(sp, rs, U, "eager step");
        size_t nNodes = 0;
        cu(cudaGraphGetNodes(graph, nullptr, &nNodes), "nodes");
        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < replays; ++rep)
        {
            cu(cudaGraphLaunch(exec, rs[0]->stream), "graph launch");
            cu(cudaStreamSynchronize(rs[0]->stream), "graph sync");
            refStep(sp, U);
            const std::string when = "replay " + std::to_string(rep);
            checkState(sp, rs, U, when.c_str());
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("PASS %s: %d replays bitwise correct, graph of %zu nodes (%.1f ms incl. verification)\n", name.c_str(), replays, nNodes, ms);
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL %s: %s\n", name.c_str(), e.what());
        fails = 1;
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    for (auto& r : rs) { cudaStreamSynchronize(r->stream); cudaStreamDestroy(r->stream); }
    cudaEventDestroy(fork);
    for (auto e : join) if (e) cudaEventDestroy(e);
    return fails;
}

// The MPI-host backend is host-synchronous: a call inside a capture must throw, not corrupt the capture.
int mpiRefuses(int argc, char** argv)
{
    Pstream::init(argc, argv);
    int fails = 0;
    {
        std::unique_ptr<ProcComm> c = ProcComm::createMpiHost();
        if (c->size() > 1)
        {
            cudaStream_t s;
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            double* d = nullptr;
            cu(cudaMalloc(&d, 8*sizeof(double)), "malloc");
            REQUIRE(!c->graphCapturable(), "MPI-host backend claims to be graph capturable");
            cu(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "begin capture");
            bool threw = false;
            try { c->allReduce(d, 1, ReduceOp::Sum, s); }
            catch (const std::logic_error&) { threw = true; }
            cudaGraph_t g = nullptr;
            cudaStreamEndCapture(s, &g);
            if (g) cudaGraphDestroy(g);
            cudaGetLastError();
            if (!threw) { std::printf("FAIL rank %d: MPI-host allReduce inside a capture did not throw\n", c->rank()); ++fails; }
            cudaFree(d);
            cudaStreamDestroy(s);
        }
    }
    const int total = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master()) std::printf("%s test_proc_comm_graph mpi: backend refuses capture\n", total == 0 ? "PASS" : "FAIL");
    Pstream::finalize();
    return total == 0 ? 0 : 1;
}


// ---------------------------------------------------------------------------------------------------------------
// NCCL: one process per GPU, every rank captures its own graph
// ---------------------------------------------------------------------------------------------------------------
int checkOwn(ProcComm& c, const SyntheticPartition& sp, RankState& st, const std::vector<long long>& U, const char* when, int& bad)
{
    const int r = c.rank(), nC = st.plan->nCells();
    std::vector<double> h(nC);
    cu(cudaMemcpy(h.data(), st.u->p, nC*sizeof(double), cudaMemcpyDeviceToHost), "download");
    int nb = 0;
    for (int i = 0; i < nC; ++i)
        if (h[i] != static_cast<double>(U[sp.cellGlobal[r][i]])) { if (nb++ == 0 && bad == 0) std::printf("FAIL rank %d %s: cell %d: %.17g != %lld\n", r, when, i, h[i], U[sp.cellGlobal[r][i]]); }
    bad += nb ? 1 : 0;
    return nb;
}

// whileMode: the body of a conditional WHILE node; else a plain graph. Returns the number of failing ranks' checks.
int runNcclGraph(ProcComm& c, const std::string& name, const SyntheticPartition& sp, int replays, bool whileMode)
{
    const int r = c.rank();
    Watchdog wd(name.c_str(), 300);
    RankState st;
    cu(cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking), "stream");
    st.plan = c.makePlan(sp.descs[r], 4);
    const int nExt = st.plan->nExt(), nC = st.plan->nCells();
    st.u.reset(new DBuf<double>(nExt));
    st.w.reset(new DBuf<double>(nExt));
    st.fa.reset(new DBuf<double>(st.plan->nProcFaces()));
    st.partial.reset(new DBuf<double>(2));
    st.stamp.reset(new DBuf<double>(1));
    st.glob.reset(new DBuf<int>(sp.cellGlobal[r]));
    DBuf<int> iter(1);
    std::vector<long long> U(sp.owner.size());
    for (std::size_t g = 0; g < U.size(); ++g) U[g] = static_cast<long long>((g*7919 + 13) % 1000003);
    std::vector<double> u0(nC);
    for (int i = 0; i < nC; ++i) u0[i] = static_cast<double>((static_cast<std::size_t>(sp.cellGlobal[r][i])*7919 + 13) % 1000003);
    cu(cudaMemcpy(st.u->p, u0.data(), nC*sizeof(double), cudaMemcpyHostToDevice), "upload u");
    cu(cudaStreamSynchronize(cudaStreamPerThread), "sync");
    c.barrier();
    // eager warm-up step (also on the graph communicator): wakes every code path before capture
    rankStep(c, st, st.stream);
    cu(cudaStreamSynchronize(st.stream), "sync");
    c.barrier();
    refStep(sp, U);
    int bad = 0;
    checkOwn(c, sp, st, U, "eager step", bad);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    size_t nNodes = 0;
    if (!whileMode)
    {
        cu(cudaStreamBeginCapture(st.stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        rankStep(c, st, st.stream);
        cu(cudaStreamEndCapture(st.stream, &graph), "end capture");
        cu(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        cu(cudaGraphGetNodes(graph, nullptr, &nNodes), "nodes");
    }
    else
    {
#if CUDART_VERSION >= 12040
        cudaGraphConditionalHandle h;
        cu(cudaGraphCreate(&graph, 0), "graph create");
        cu(cudaGraphConditionalHandleCreate(&h, graph, 0, cudaGraphCondAssignDefault), "handle");
        cu(cudaStreamBeginCaptureToGraph(st.stream, graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "begin prologue");
        whileInitK<<<1, 1, 0, st.stream>>>(h, iter.p);
        const cudaGraphNode_t* deps = nullptr;
        std::size_t nDeps = 0;
        cudaStreamCaptureStatus cs;
        cu(braeStreamGetCaptureInfo(st.stream, &cs, &deps, &nDeps), "capture info");
        std::vector<cudaGraphNode_t> dep(deps, deps + nDeps);
        cudaGraph_t tmp = nullptr;
        cu(cudaStreamEndCapture(st.stream, &tmp), "end prologue");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = h;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t node;
        cu(braeGraphAddNode(&node, graph, dep.data(), dep.size(), &cp), "add WHILE node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cu(cudaStreamBeginCaptureToGraph(st.stream, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "begin body");
        rankStep(c, st, st.stream);
        whileCondK<<<1, 1, 0, st.stream>>>(h, st.partial->p, iter.p);
        cu(cudaStreamEndCapture(st.stream, &tmp), "end body");
        cu(cudaGraphInstantiate(&exec, graph, 0), "instantiate WHILE graph");
        cu(cudaGraphGetNodes(graph, nullptr, &nNodes), "nodes");
#else
        throw std::runtime_error("WHILE graphs need CUDA >= 12.4");
#endif
    }
    c.barrier();
    std::map<int, int> counts;
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < replays; ++rep)
    {
        cu(cudaGraphLaunch(exec, st.stream), "graph launch");
        cu(cudaStreamSynchronize(st.stream), "graph sync");
        int its = 1;
        if (!whileMode) refStep(sp, U);
        else
        {
#if CUDART_VERSION >= 12040
            its = 0;
            for (;;)
            {
                const long long total = refStep(sp, U);
                ++its;
                if (!(std::fmod(static_cast<double>(total), kWhileMod) != 0.0 && its < kWhileMax)) break;
            }
            int got = -1;
            cu(cudaMemcpy(&got, iter.p, sizeof(int), cudaMemcpyDeviceToHost), "iter");
            if (got != its) { if (bad == 0) std::printf("FAIL rank %d replay %d: WHILE ran %d iterations, reference %d\n", r, rep, got, its); ++bad; }
#endif
        }
        ++counts[its];
        const std::string when = "replay " + std::to_string(rep);
        checkOwn(c, sp, st, U, when.c_str(), bad);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double nbad = c.allReduce(static_cast<double>(bad), ReduceOp::Sum);
    std::string hist;
    if (whileMode) { hist = ", WHILE iteration counts"; for (auto& kv : counts) hist += " " + std::to_string(kv.first) + "x" + std::to_string(kv.second); }
    if (whileMode && c.size() > 1 && counts.size() < 3) {   // np=1 has no halo: the reduced sum is 0 and the loop runs once
        if (r == 0) std::printf("FAIL %s: the loop count did not vary (%zu distinct values)\n", name.c_str(), counts.size()); ++bad; }
    const int failing = static_cast<int>(c.allReduce(static_cast<double>(bad), ReduceOp::Max) != 0.0);
    if (r == 0)
        std::printf("%s %s: %d replays, %s, graph of %zu nodes, %.0f ms incl. verification%s\n", failing ? "FAIL" : "PASS", name.c_str(), replays,
                    nbad == 0 ? "bitwise correct on every rank" : "MISMATCHES", nNodes, ms, hist.c_str());
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cu(cudaStreamSynchronize(st.stream), "sync");
    cudaStreamDestroy(st.stream);
    c.barrier();   // nobody destroys a plan while a peer may still use its staging
    return failing;
}

int ncclSuite(int argc, char** argv, const std::string& backendArg)
{
    Pstream::init(argc, argv);
    int fails = 0;
    {
        std::unique_ptr<ProcComm> c = std::move(ProcComm::create(backendArg == "auto" ? "" : backendArg)[0]);
        const int np = c->size();
        if (c->isMaster()) std::printf("backend: %s (np=%d, rank 0 on GPU %d, graphCapturable=%d)\n", ProcComm::backendName(c->backend()), np, c->device(), c->graphCapturable());
        if (!c->graphCapturable()) { std::printf("FAIL: backend is not graph capturable\n"); return 1; }
        static const int fac[9][3] = {{0,0,0},{1,1,1},{2,1,1},{1,3,1},{2,2,1},{0,0,0},{0,0,0},{0,0,0},{2,2,2}};
        std::vector<std::pair<std::string, SyntheticPartition>> parts;
        const int nx = 20, ny = 15, nz = 8;
        if (np <= 8 && fac[np][0] > 0)
            parts.emplace_back("block", partitionBox(nx, ny, nz, np, blockOwner(nx, ny, nz, fac[np][0], fac[np][1], fac[np][2])));
        else
            parts.emplace_back("block np x 1 x 1", partitionBox(nx*2, ny, nz, np, blockOwner(nx*2, ny, nz, np, 1, 1)));
        if (np > 1) parts.emplace_back("scattered", partitionBox(18, 15, 12, np, scatteredOwner(18, 15, 12, np, 3)));
        for (auto& pc : parts)
        {
            fails += runNcclGraph(*c, "nccl graph np=" + std::to_string(np) + " " + pc.first, pc.second, 1000, false);
#if CUDART_VERSION >= 12040
            fails += runNcclGraph(*c, "nccl WHILE-body graph np=" + std::to_string(np) + " " + pc.first, pc.second, 1000, true);
#else
            if (c->isMaster()) std::printf("SKIP WHILE-body graph: built with CUDA < 12.4\n");
#endif
        }
        if (c->isMaster()) std::printf("test_proc_comm_graph nccl: %s\n", fails ? "FAIL" : "PASS");
    }
    Pstream::finalize();
    return fails ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc > 1 && std::string(argv[1]) == "mpi") return mpiRefuses(argc, argv);
        if (argc > 1 && (std::string(argv[1]) == "nccl" || std::string(argv[1]) == "auto")) return ncclSuite(argc, argv, argv[1]);
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev <= 0) { std::printf("SKIP: no CUDA device\n"); return 77; }
        std::printf("backend: loopback (joint graph over np = 2,3,4,8 virtual ranks on one GPU)\n");
        int fails = 0;
        for (int np : {2, 3, 4, 8})
        {
            static const int fac[9][3] = {{0,0,0},{1,1,1},{2,1,1},{1,3,1},{2,2,1},{0,0,0},{0,0,0},{0,0,0},{2,2,2}};
            const int nx = 20, ny = 15, nz = 8;
            fails += runGraphCase("graph np=" + std::to_string(np) + " block",
                                  partitionBox(nx, ny, nz, np, blockOwner(nx, ny, nz, fac[np][0], fac[np][1], fac[np][2])), 1000);
            fails += runGraphCase("graph np=" + std::to_string(np) + " scattered",
                                  partitionBox(18, 15, 12, np, scatteredOwner(18, 15, 12, np, 3)), 1000);
        }
        std::printf("test_proc_comm_graph: %s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
