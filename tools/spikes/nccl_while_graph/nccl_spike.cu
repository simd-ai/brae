// NCCL inside a CUDA graph and inside the body of a conditional WHILE node: the transport spike for multi-GPU brae.
//
// Two builds of this one file:
//   nccl_spike      one process drives N GPUs (ncclCommInitAll)         ./nccl_spike [--ngpu N]
//   nccl_spike_mpi  one MPI rank per GPU of one node (ncclCommInitRank) mpirun -np N ./nccl_spike_mpi
//
// Tests (each prints PASS / FAIL / SKIP, exit status != 0 if any FAIL):
//   1 eager   halo (grouped ncclSend/ncclRecv to every peer) + 3 deterministic reductions (ncclAllGather of one double
//             per rank, summed in rank order by a kernel) on a stream, validated bit for bit against the host
//   2 graph   the same iteration captured into a CUDA graph, replayed 100 times, validated
//   3 while   the iteration captured into the BODY of a conditional WHILE node exactly like
//             src/matrices/lduMatrix/solvers/PCG/device_amg_pcg.cu builds its PCG loop; the loop count depends on the
//             allgathered sum, 1000 replays with a varying start value, iteration count / sum / state validated bit for bit
//   4 timing  one PCG-like iteration (halo + local kernel + 3 reductions): eager vs graph vs graph-in-WHILE, us/iteration,
//             message sizes (per peer) 1 KB .. 8 MB
//   5 pairs   every GPU pair: latency and bidirectional bandwidth, classified from `nvidia-smi topo -m`
//
// Graphs use the communicator `cg`, eager work the communicator `ce` (run with and without
// NCCL_GRAPH_MIXING_SUPPORT=0, see README.md).
#include <cuda_runtime.h>
#include <nccl.h>
#ifdef SPIKE_MPI
#include <mpi.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <unistd.h>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------------------------- error handling
static int g_myRank = 0;
static std::atomic<long> g_deadline{0};
static const char* g_stage = "init";
static long nowS() { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void arm(const char* stage, int seconds = 60) { g_stage = stage; g_deadline = nowS() + seconds; }

[[noreturn]] static void die(const char* what, const char* file, int line, const char* msg)
{
    fprintf(stderr, "[FAIL] rank %d: %s at %s:%d: %s (stage: %s)\n", g_myRank, what, file, line, msg, g_stage);
    fflush(stderr);
#ifdef SPIKE_MPI
    MPI_Abort(MPI_COMM_WORLD, 2);
#endif
    _exit(2);
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) die(#x, __FILE__, __LINE__, cudaGetErrorString(e_)); } while (0)
#define NK(x) do { ncclResult_t e_ = (x); if (e_ != ncclSuccess) die(#x, __FILE__, __LINE__, ncclGetErrorString(e_)); } while (0)

// ---------------------------------------------------------------------------------------------- device side
struct St { double v, s; int iter, err, pad0, pad1; };

__host__ __device__ static inline uint32_t hsh(uint32_t a, uint32_t b, uint32_t i, uint32_t it)
{
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ i * 0xC2B2AE35u ^ (it + 1u) * 0x27D4EB2Fu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
__host__ __device__ static inline int peerOf(int slot, int rank, int R) { return R == 1 ? 0 : (slot < rank ? slot : slot + 1); }

__global__ void iterIncK(St* st) { st->iter++; }
__global__ void resetK(St* st, double v0) { st->v = v0; st->s = 0; st->iter = 0; }
__global__ void resetIterK(St* st) { st->iter = 0; }
__global__ void fillK(uint32_t* snd, int wpp, int nPeer, int rank, int R, const St* st)
{
    const int it = st->iter;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)wpp * nPeer; i += (size_t)gridDim.x * blockDim.x)
    {
        const int slot = (int)(i / wpp);
        snd[i] = hsh(rank, peerOf(slot, rank, R), (uint32_t)(i % wpp), it);
    }
}
__global__ void checkK(const uint32_t* rcv, int wpp, int nPeer, int rank, int R, St* st)
{
    const int it = st->iter;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)wpp * nPeer; i += (size_t)gridDim.x * blockDim.x)
    {
        const int slot = (int)(i / wpp);
        if (rcv[i] != hsh(peerOf(slot, rank, R), rank, (uint32_t)(i % wpp), it)) atomicAdd(&st->err, 1);
    }
}
__global__ void localK(const uint32_t* rcv, uint32_t* work, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) work[i] = rcv[i] * 3u + 1u;
}
// the reduction operand: non-trivial doubles so that the summation order matters; explicit rounding intrinsics so
// the host reference below (compiled with -ffp-contract=off) is bit identical
__device__ static inline double xvalD(int rank, int k, int iter)
{
    return __dadd_rn(__dadd_rn(__dmul_rn((double)(rank + 1), 0.1), __dmul_rn((double)k, 0.37)), __dmul_rn((double)(iter & 7), 0.013));
}
static inline double xvalH(int rank, int k, int iter) { return (double)(rank + 1) * 0.1 + (double)k * 0.37 + (double)(iter & 7) * 0.013; }
__global__ void xK(const St* st, double* x, int k, int rank) { *x = xvalD(rank, k, st->iter); }
__global__ void sumK(const double* g, int R, double* out)
{
    double a = g[0];
    for (int i = 1; i < R; i++) a = __dadd_rn(a, g[i]);
    *out = a;
}
// test 3: the body's value update, then the loop decision from the allgathered sum
__global__ void vUpdK(St* st, double* x) { st->v = __dmul_rn(st->v, 0.5); *x = st->v; }
__global__ void sumCondK(const double* g, int R, St* st, cudaGraphConditionalHandle h)
{
    double a = g[0];
    for (int i = 1; i < R; i++) a = __dadd_rn(a, g[i]);
    st->s = a;
    cudaGraphSetConditional(h, (a > 1.0 && st->iter < 64) ? 1u : 0u);
}
__global__ void setOneK(cudaGraphConditionalHandle h) { cudaGraphSetConditional(h, 1u); }
__global__ void condIterK(cudaGraphConditionalHandle h, const St* st, int K) { cudaGraphSetConditional(h, st->iter < K ? 1u : 0u); }

// ---------------------------------------------------------------------------------------------- ranks
struct Rk
{
    int rank, dev;
    cudaStream_t s;
    ncclComm_t ce, cg;
    cudaGraph_t g;
    cudaGraphExec_t e;
    cudaGraphConditionalHandle h;
    cudaGraph_t bg;
    uint32_t *snd, *rcv, *work;
    double *x, *gath, *out;
    St* st;
    double* pin;     // pinned host results
};
static std::vector<Rk> L;
static int R = 1;                       // total ranks
static int g_graphMode = 0;             // graphUsageMode of the graph communicator: 0 = graph-only (no mixing), 1 = mixing, -1 = leave to NCCL/env
static ncclConfig_t cfgFor(bool graphComm)
{
    ncclConfig_t c = NCCL_CONFIG_INITIALIZER;
#ifdef HAVE_GRAPH_USAGE_MODE
    if (graphComm && g_graphMode >= 0) c.graphUsageMode = g_graphMode;
#else
    (void)graphComm;
#endif
    return c;
}
static cudaStreamCaptureMode g_capMode = cudaStreamCaptureModeThreadLocal;

template <class F> static void each(F f) { for (auto& r : L) { CK(cudaSetDevice(r.dev)); f(r); } }
static bool mpiMode() {
#ifdef SPIKE_MPI
    return true;
#else
    return false;
#endif
}
static void syncAll()
{
    each([](Rk& r) { CK(cudaStreamSynchronize(r.s)); });
#ifdef SPIKE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}
static bool allOk(bool ok)
{
#ifdef SPIKE_MPI
    int a = ok, b = 1;
    MPI_Allreduce(&a, &b, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return b != 0;
#else
    return ok;
#endif
}
static double maxD(double v)
{
#ifdef SPIKE_MPI
    double m;
    MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return m;
#else
    return v;
#endif
}
template <class F> static double timeIt(F f)
{
    syncAll();
    auto t0 = std::chrono::steady_clock::now();
    f();
    syncAll();
    return maxD(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}
#define OUT(...) do { if (g_myRank == 0) { printf(__VA_ARGS__); fflush(stdout); } } while (0)
static int g_fails = 0;
static void verdict(const char* name, bool ok, const std::string& detail = "")
{
    ok = allOk(ok);
    if (!ok) g_fails++;
    OUT("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", name, detail.empty() ? "" : " : ", detail.c_str());
}

// ---------------------------------------------------------------------------------------------- the operations
static size_t g_S = 0;                  // bytes per peer
static int nPeer() { return R == 1 ? 1 : R - 1; }

static void allocBufs(size_t S)
{
    g_S = S;
    each([&](Rk& r) {
        cudaFree(r.snd); cudaFree(r.rcv); cudaFree(r.work);
        const size_t n = (size_t)nPeer() * S;
        CK(cudaMalloc(&r.snd, n)); CK(cudaMalloc(&r.rcv, n)); CK(cudaMalloc(&r.work, n));
        CK(cudaMemsetAsync(r.rcv, 0, n, r.s));
        CK(cudaStreamSynchronize(r.s));
    });
}
static void halo(ncclComm_t Rk::*cm)
{
    NK(ncclGroupStart());
    for (auto& r : L)
    {
        CK(cudaSetDevice(r.dev));
        for (int slot = 0; slot < nPeer(); slot++)
        {
            const int p = peerOf(slot, r.rank, R);
            NK(ncclSend((char*)r.snd + slot * g_S, g_S, ncclChar, p, r.*cm, r.s));
            NK(ncclRecv((char*)r.rcv + slot * g_S, g_S, ncclChar, p, r.*cm, r.s));
        }
    }
    NK(ncclGroupEnd());
}
static void gather(ncclComm_t Rk::*cm)
{
    NK(ncclGroupStart());
    for (auto& r : L)
    {
        CK(cudaSetDevice(r.dev));
        NK(ncclAllGather(r.x, r.gath, 1, ncclDouble, r.*cm, r.s));
    }
    NK(ncclGroupEnd());
}
static void haloPhase(ncclComm_t Rk::*cm)
{
    const int wpp = (int)(g_S / 4);
    each([&](Rk& r) { fillK<<<256, 256, 0, r.s>>>(r.snd, wpp, nPeer(), r.rank, R, r.st); });
    halo(cm);
    each([&](Rk& r) { checkK<<<256, 256, 0, r.s>>>(r.rcv, wpp, nPeer(), r.rank, R, r.st); });
}
// one PCG-like iteration: halo, a local kernel, 3 deterministic reductions
static void step(ncclComm_t Rk::*cm)
{
    each([&](Rk& r) { iterIncK<<<1, 1, 0, r.s>>>(r.st); });
    haloPhase(cm);
    each([&](Rk& r) { localK<<<256, 256, 0, r.s>>>(r.rcv, r.work, (size_t)nPeer() * g_S / 4); });
    for (int k = 0; k < 3; k++)
    {
        each([&](Rk& r) { xK<<<1, 1, 0, r.s>>>(r.st, r.x, k, r.rank); });
        gather(cm);
        each([&](Rk& r) { sumK<<<1, 1, 0, r.s>>>(r.gath, R, r.out + k); });
    }
}
static double refSum(int k, int iter)
{
    double a = xvalH(0, k, iter);
    for (int i = 1; i < R; i++) a += xvalH(i, k, iter);
    return a;
}
static int errCount()
{
    int bad = 0;
    each([&](Rk& r) { CK(cudaStreamSynchronize(r.s)); St h; CK(cudaMemcpy(&h, r.st, sizeof h, cudaMemcpyDeviceToHost)); bad += h.err; });
    return bad;
}
static void zeroState() { each([](Rk& r) { CK(cudaMemsetAsync(r.st, 0, sizeof(St), r.s)); CK(cudaStreamSynchronize(r.s)); }); }

// ---------------------------------------------------------------------------------------------- graph capture
static void getDeps(cudaStream_t s, std::vector<cudaGraphNode_t>& v)
{
    cudaStreamCaptureStatus st;
    const cudaGraphNode_t* deps = nullptr;
    size_t n = 0;
#if CUDART_VERSION >= 13000
    CK(cudaStreamGetCaptureInfo(s, &st, nullptr, nullptr, &deps, nullptr, &n));
#else
    CK(cudaStreamGetCaptureInfo(s, &st, nullptr, nullptr, &deps, &n));
#endif
    v.assign(deps, deps + n);
}
static cudaGraphNode_t addCondNode(cudaGraph_t g, cudaGraphConditionalHandle h, const std::vector<cudaGraphNode_t>& d, cudaGraph_t* body)
{
    cudaGraphNodeParams cp = {};
    cp.type = cudaGraphNodeTypeConditional;
    cp.conditional.handle = h;
    cp.conditional.type = cudaGraphCondTypeWhile;
    cp.conditional.size = 1;
    cudaGraphNode_t n;
#if CUDART_VERSION >= 13000
    CK(cudaGraphAddNode(&n, g, d.data(), nullptr, d.size(), &cp));
#else
    CK(cudaGraphAddNode(&n, g, d.data(), d.size(), &cp));
#endif
    *body = cp.conditional.phGraph_out[0];
    return n;
}
static void freeGraphs()
{
    each([](Rk& r) {
        if (r.e) cudaGraphExecDestroy(r.e);
        if (r.g) cudaGraphDestroy(r.g);
        r.e = nullptr; r.g = nullptr;
    });
}
static void describeGraph(const char* what, cudaGraph_t g)
{
    if (const char* d = getenv("SPIKE_DOT")) cudaGraphDebugDotPrint(g, d, cudaGraphDebugDotFlagsVerbose);
    size_t n = 0;
    cudaGraphGetNodes(g, nullptr, &n);
    std::vector<cudaGraphNode_t> nodes(n);
    cudaGraphGetNodes(g, nodes.data(), &n);
    std::map<int, int> cnt;
    for (auto nd : nodes)
    {
        cudaGraphNodeType t;
        cudaGraphNodeGetType(nd, &t);
        cnt[(int)t]++;
        if (t == cudaGraphNodeTypeKernel)
        {
            cudaKernelNodeParams kp;
            cudaGraphKernelNodeGetParams(nd, &kp);
            if (true)
            {
                cudaKernelNodeAttrValue v;
                std::string at;
                if (cudaGraphKernelNodeGetAttribute(nd, cudaKernelNodeAttributeClusterDimension, &v) == cudaSuccess) at += " cluster=" + std::to_string(v.clusterDim.x);
                if (cudaGraphKernelNodeGetAttribute(nd, cudaKernelNodeAttributeCooperative, &v) == cudaSuccess) at += " cooperative=" + std::to_string(v.cooperative);
                if (cudaGraphKernelNodeGetAttribute(nd, cudaKernelNodeAttributePriority, &v) == cudaSuccess) at += " priority=" + std::to_string(v.priority);
                if (cudaGraphKernelNodeGetAttribute(nd, cudaKernelNodeAttributeMemSyncDomain, &v) == cudaSuccess) at += " memsyncdomain=" + std::to_string((int)v.memSyncDomain);
                fprintf(stderr, "rank %d:   kernel node: grid %u block %u smem %u params %p%s\n", g_myRank, kp.gridDim.x, kp.blockDim.x, kp.sharedMemBytes, kp.func, at.c_str());
            }
        }
        if (t == cudaGraphNodeTypeMemcpy)
        {
            cudaMemcpy3DParms mp;
            cudaGraphMemcpyNodeGetParams(nd, &mp);
            fprintf(stderr, "rank %d:   memcpy node: kind %d, extent %zu x %zu x %zu, src %p (ptr %p pitch %zu) dst %p (ptr %p pitch %zu)\n", g_myRank, (int)mp.kind,
                    mp.extent.width, mp.extent.height, mp.extent.depth, (void*)mp.srcArray, mp.srcPtr.ptr, mp.srcPtr.pitch, (void*)mp.dstArray, mp.dstPtr.ptr, mp.dstPtr.pitch);
        }
    }
    fprintf(stderr, "rank %d: %s has %zu nodes; type histogram (cudaGraphNodeType: count):", g_myRank, what, n);
    for (auto& kv : cnt) fprintf(stderr, " %d:%d", kv.first, kv.second);
    fprintf(stderr, "   [0 kernel, 1 memcpy, 2 memset, 3 host, 4 child graph, 5 empty, 6 wait event, 7 event record, 8/9 semaphores, 10 mem alloc, 11 mem free, 12 batch mem op, 13 conditional]\n");
}
// plain capture of fn into a graph per rank
template <class F> static void captureGraph(F fn)
{
    freeGraphs();
    each([](Rk& r) { CK(cudaStreamBeginCapture(r.s, g_capMode)); });
    fn();
    each([](Rk& r) { CK(cudaStreamEndCapture(r.s, &r.g)); CK(cudaGraphInstantiate(&r.e, r.g, 0)); });
}
// WHILE capture, the structure of device_amg_pcg.cu: a prologue sets the handle, the conditional node follows it, the
// body is captured into the node's body graph; body() must end with a kernel that sets the handle
template <class F> static bool captureWhile(F body, bool dieOnFail = true)
{
    freeGraphs();
    each([](Rk& r) {
        CK(cudaGraphCreate(&r.g, 0));
        CK(cudaGraphConditionalHandleCreate(&r.h, r.g, 0, cudaGraphCondAssignDefault));
        CK(cudaStreamBeginCaptureToGraph(r.s, r.g, nullptr, nullptr, 0, g_capMode));
        setOneK<<<1, 1, 0, r.s>>>(r.h);
        std::vector<cudaGraphNode_t> dep;
        getDeps(r.s, dep);
        cudaGraph_t tmp;
        CK(cudaStreamEndCapture(r.s, &tmp));
        addCondNode(r.g, r.h, dep, &r.bg);
    });
    // only now open the body captures (no other API calls happen on any rank while a capture is open)
    each([](Rk& r) { CK(cudaStreamBeginCaptureToGraph(r.s, r.bg, nullptr, nullptr, 0, g_capMode)); });
    body();
    bool ok = true;
    each([&](Rk& r) {
        cudaGraph_t tmp;
        CK(cudaStreamEndCapture(r.s, &tmp));
        cudaError_t ie = cudaGraphInstantiate(&r.e, r.g, 0);
        if (ie != cudaSuccess)
        {
            describeGraph("WHILE body", r.bg);
            if (dieOnFail) die("cudaGraphInstantiate of the WHILE graph", __FILE__, __LINE__, cudaGetErrorString(ie));
            cudaGetLastError();
            r.e = nullptr;
            ok = false;
        }
    });
    return ok;
}
static void launch() { each([](Rk& r) { CK(cudaGraphLaunch(r.e, r.s)); }); }

// ---------------------------------------------------------------------------------------------- tests
static void test1_eager()
{
    arm("test1");
    allocBufs(4096);
    zeroState();
    const int N = 100;
    each([&](Rk& r) { CK(cudaMallocHost(&r.pin, N * 3 * sizeof(double))); });
    for (int j = 0; j < N; j++)
    {
        step(&Rk::ce);
        each([&](Rk& r) { CK(cudaMemcpyAsync(r.pin + j * 3, r.out, 3 * sizeof(double), cudaMemcpyDeviceToHost, r.s)); });
    }
    syncAll();
    bool ok = errCount() == 0;
    int badv = 0;
    each([&](Rk& r) { for (int j = 0; j < N; j++) for (int k = 0; k < 3; k++) if (r.pin[j * 3 + k] != refSum(k, j + 1)) badv++; cudaFreeHost(r.pin); });
    verdict("test1 eager halo + allgather-sum, 100 iterations", ok && badv == 0, "halo errors " + std::to_string(errCount()) + ", reduction mismatches " + std::to_string(badv));
}
static void test2_graph()
{
    arm("test2");
    zeroState();
    captureGraph([] { step(&Rk::cg); });
    const int N = 100;
    each([&](Rk& r) { CK(cudaMallocHost(&r.pin, N * 3 * sizeof(double))); });
    for (int j = 0; j < N; j++)
    {
        launch();
        each([&](Rk& r) { CK(cudaMemcpyAsync(r.pin + j * 3, r.out, 3 * sizeof(double), cudaMemcpyDeviceToHost, r.s)); });
    }
    syncAll();
    int badv = 0;
    each([&](Rk& r) { for (int j = 0; j < N; j++) for (int k = 0; k < 3; k++) if (r.pin[j * 3 + k] != refSum(k, j + 1)) badv++; cudaFreeHost(r.pin); });
    verdict("test2 graph capture + 100 replays of the same iteration", errCount() == 0 && badv == 0, "halo errors " + std::to_string(errCount()) + ", reduction mismatches " + std::to_string(badv));
    freeGraphs();
}
// which NCCL operation makes a WHILE body uninstantiable? informational probe of single-ingredient bodies
static void test3_probe()
{
    arm("test3 probe");
    allocBufs(4096);
    zeroState();
    struct { const char* name; int mask; } cases[] = {{"kernels only", 0}, {"ncclAllGather only", 1}, {"ncclSend/ncclRecv only", 2}, {"both", 3}, {"cudaMemcpyAsync D2D", 4}, {"kernel w/ cluster dim 1", 8}, {"event record+wait fork/join", 16}, {"kernel 61760 B dyn smem", 32}, {"kernel cluster 1 + 61760 B smem", 64}};
    for (auto& c : cases)
    {
        const int m = c.mask;
        const bool ok = captureWhile([&] {
            each([](Rk& r) { iterIncK<<<1, 1, 0, r.s>>>(r.st); });
            if (m & 2) halo(&Rk::cg);
            if (m & 8) each([](Rk& r) {
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3(1); cfg.blockDim = dim3(1); cfg.stream = r.s;
                cudaLaunchAttribute at[1];
                at[0].id = cudaLaunchAttributeClusterDimension;
                at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
                cfg.attrs = at; cfg.numAttrs = 1;
                CK(cudaLaunchKernelEx(&cfg, iterIncK, r.st));
            });
            if (m & 96) each([&](Rk& r) {
                static bool once = false;
                if (!once) { CK(cudaFuncSetAttribute(iterIncK, cudaFuncAttributeMaxDynamicSharedMemorySize, 61760)); once = true; }
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3(1); cfg.blockDim = dim3(640); cfg.stream = r.s; cfg.dynamicSmemBytes = 61760;
                cudaLaunchAttribute at[1];
                at[0].id = cudaLaunchAttributeClusterDimension;
                at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
                cfg.attrs = at; cfg.numAttrs = (m & 64) ? 1 : 0;
                CK(cudaLaunchKernelEx(&cfg, iterIncK, r.st));
            });
            if (m & 16) each([](Rk& r) {
                cudaStream_t s2; cudaEvent_t e1, e2;
                CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
                CK(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));
                CK(cudaEventRecord(e1, r.s)); CK(cudaStreamWaitEvent(s2, e1, 0));
                iterIncK<<<1, 1, 0, s2>>>(r.st);
                CK(cudaEventRecord(e2, s2)); CK(cudaStreamWaitEvent(r.s, e2, 0));
            });
            if (m & 4) each([](Rk& r) { CK(cudaMemcpyAsync(r.rcv, r.snd, 4096, cudaMemcpyDeviceToDevice, r.s)); });
            each([](Rk& r) { vUpdK<<<1, 1, 0, r.s>>>(r.st, r.x); });
            if (m & 1) gather(&Rk::cg);
            each([](Rk& r) { sumCondK<<<1, 1, 0, r.s>>>(r.gath, R, r.st, r.h); });
        }, false);
        OUT("  probe: WHILE body with %-24s instantiate %s\n", c.name, ok ? "ok" : "REJECTED");
        freeGraphs();
    }
}
static void test3_while()
{
    test3_probe();
    arm("test3");
    allocBufs(4096);
    zeroState();
    captureWhile([] {
        each([](Rk& r) { iterIncK<<<1, 1, 0, r.s>>>(r.st); });
        haloPhase(&Rk::cg);
        each([](Rk& r) { vUpdK<<<1, 1, 0, r.s>>>(r.st, r.x); });
        gather(&Rk::cg);
        each([](Rk& r) { sumCondK<<<1, 1, 0, r.s>>>(r.gath, R, r.st, r.h); });
    });
    const int N = 1000;
    each([&](Rk& r) { CK(cudaMallocHost(&r.pin, N * sizeof(St))); });
    for (int j = 0; j < N; j++)
    {
        each([&](Rk& r) {
            resetK<<<1, 1, 0, r.s>>>(r.st, (double)(r.rank + 1) * (1 + j % 13) + 0.5);
            CK(cudaGraphLaunch(r.e, r.s));
            CK(cudaMemcpyAsync((St*)r.pin + j, r.st, sizeof(St), cudaMemcpyDeviceToHost, r.s));
        });
    }
    syncAll();
    int bad = 0;
    std::set<int> counts;
    each([&](Rk& r) {
        for (int j = 0; j < N; j++)
        {
            std::vector<double> v(R);
            for (int q = 0; q < R; q++) v[q] = (double)(q + 1) * (1 + j % 13) + 0.5;
            int it = 0;
            double s;
            do
            {
                it++;
                for (int q = 0; q < R; q++) v[q] *= 0.5;
                s = v[0];
                for (int q = 1; q < R; q++) s += v[q];
            } while (s > 1.0 && it < 64);
            const St& h = ((St*)r.pin)[j];
            if (h.iter != it || h.s != s || h.v != v[r.rank]) { if (bad < 3) fprintf(stderr, "rank %d replay %d: iter %d (ref %d) s %.17g (ref %.17g)\n", r.rank, j, h.iter, it, h.s, s); bad++; }
            if (r.rank == L[0].rank) counts.insert(it);
        }
        cudaFreeHost(r.pin);
    });
    verdict("test3 NCCL in conditional WHILE body, 1000 replays, loop count from allgathered sum", bad == 0 && errCount() == 0,
            "mismatches " + std::to_string(bad) + ", halo errors " + std::to_string(errCount()) + ", distinct loop counts " + std::to_string(counts.size()));
    freeGraphs();
}

static void test4_timing()
{
    const int K = 20;
    OUT("\ntest4: us per PCG-like iteration (halo to %d peer(s) of the given size + local kernel + 3 reductions)\n", nPeer());
    OUT("%10s %12s %12s %14s %10s\n", "bytes/peer", "eager", "graph", "graph-in-WHILE", "check");
    bool ok = true;
    for (size_t S : {1u << 10, 4u << 10, 16u << 10, 64u << 10, 256u << 10, 1u << 20, 4u << 20, 8u << 20})
    {
        arm("test4");
        allocBufs(S);
        zeroState();
        const int reps = S >= (1u << 20) ? 3 : 10;
        auto eager = [&] { for (int i = 0; i < K; i++) step(&Rk::ce); };
        eager();
        const double te = timeIt([&] { for (int q = 0; q < reps; q++) { each([](Rk& r) { resetIterK<<<1, 1, 0, r.s>>>(r.st); }); eager(); } });
        const int err1 = errCount();
        captureGraph([] { step(&Rk::cg); });
        launch();
        const double tg = timeIt([&] { for (int q = 0; q < reps; q++) { each([](Rk& r) { resetIterK<<<1, 1, 0, r.s>>>(r.st); }); for (int i = 0; i < K; i++) launch(); } });
        const int err2 = errCount();
        captureWhile([&] {
            step(&Rk::cg);
            each([&](Rk& r) { condIterK<<<1, 1, 0, r.s>>>(r.h, r.st, K); });
        });
        launch();
        const double tw = timeIt([&] { for (int q = 0; q < reps; q++) { each([](Rk& r) { resetIterK<<<1, 1, 0, r.s>>>(r.st); }); launch(); } });
        const int err = errCount();
        if (err) OUT("       (halo errors after: warmup+eager %d, +graph %d, +while %d)\n", err1, err2, err);
        ok = ok && err == 0;
        freeGraphs();
        const double d = (double)reps * K / 1e6;
        OUT("%10zu %12.1f %12.1f %14.1f %10s\n", S, te / d, tg / d, tw / d, err ? "ERRORS" : "ok");
    }
    verdict("test4 timing runs, halo data verified in every variant", ok);
}

static std::vector<std::vector<std::string>> g_topo;
static void loadTopo()
{
    FILE* f = popen("nvidia-smi topo -m 2>/dev/null", "r");
    if (!f) return;
    char line[4096];
    while (fgets(line, sizeof line, f))
    {
        if (strncmp(line, "GPU", 3) != 0 || !isdigit((unsigned char)line[3])) continue;
        std::istringstream is(line);
        std::vector<std::string> t;
        std::string w;
        while (is >> w) t.push_back(w);
        g_topo.push_back(t);
    }
    pclose(f);
}
static std::string pairClass(int a, int b)
{
    std::string t = "?";
    if (a < (int)g_topo.size() && (size_t)b + 1 < g_topo[a].size()) t = g_topo[a][b + 1];
    std::string c;
    if (t.rfind("NV", 0) == 0) c = "NVLink(" + t + ")";
    else if (t == "SYS") c = "cross-socket(SYS)";
    else c = (a / 4 == b / 4 ? "same-quad(" : "cross-quad(") + t + ")";
    if (t.rfind("NV", 0) != 0 && t != "SYS" && a / 4 != b / 4) c = "cross-quad(" + t + ")";
    return c;
}
static void test5_pairs()
{
    if (R < 2) { OUT("[SKIP] test5 pair sweep needs >= 2 GPUs\n"); return; }
    loadTopo();
    const size_t BW = 32u << 20;
    each([&](Rk& r) {
        cudaFree(r.snd); cudaFree(r.rcv);
        CK(cudaMalloc(&r.snd, BW)); CK(cudaMalloc(&r.rcv, BW));
        CK(cudaMemsetAsync(r.snd, r.rank + 1, BW, r.s));
        CK(cudaStreamSynchronize(r.s));
    });
    OUT("\ntest5: pair sweep (bidirectional exchange; latency = 8 B, bandwidth = 32 MB per direction)\n");
    OUT("%5s %-24s %6s %12s %14s %6s\n", "pair", "class", "p2p", "latency us", "GB/s per dir", "data");
    std::map<std::string, std::vector<std::pair<double, double>>> cls;
    bool ok = true;
    for (int a = 0; a < R; a++)
        for (int b = a + 1; b < R; b++)
        {
            arm("test5");
            auto exch = [&](size_t bytes) {
                NK(ncclGroupStart());
                for (auto& r : L)
                    if (r.rank == a || r.rank == b)
                    {
                        CK(cudaSetDevice(r.dev));
                        const int p = r.rank == a ? b : a;
                        NK(ncclSend(r.snd, bytes, ncclChar, p, r.ce, r.s));
                        NK(ncclRecv(r.rcv, bytes, ncclChar, p, r.ce, r.s));
                    }
                NK(ncclGroupEnd());
            };
            exch(BW);
            const double tl = timeIt([&] { for (int i = 0; i < 200; i++) exch(8); });
            const double tb = timeIt([&] { for (int i = 0; i < 5; i++) exch(BW); });
            bool dataOk = true;
            each([&](Rk& r) {
                if (r.rank != a && r.rank != b) return;
                unsigned char c0, c1;
                CK(cudaStreamSynchronize(r.s));
                CK(cudaMemcpy(&c0, r.rcv, 1, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&c1, (char*)r.rcv + BW - 1, 1, cudaMemcpyDeviceToHost));
                const unsigned char want = (unsigned char)((r.rank == a ? b : a) + 1);
                if (c0 != want || c1 != want) dataOk = false;
            });
            dataOk = allOk(dataOk);
            ok = ok && dataOk;
            std::string p2p = "?";
            if (!mpiMode()) { int can = 0; CK(cudaDeviceCanAccessPeer(&can, L[a].dev, L[b].dev)); p2p = can ? "yes" : "no"; }
            const double lat = tl / 200 * 1e6, bw = (double)BW * 5 / tb / 1e9;
            const std::string c = pairClass(a, b);
            cls[c].push_back({lat, bw});
            OUT("%2d-%-2d %-24s %6s %12.1f %14.2f %6s\n", a, b, c.c_str(), p2p.c_str(), lat, bw, dataOk ? "ok" : "BAD");
        }
    OUT("\nper class (mean over pairs):\n");
    for (auto& kv : cls)
    {
        double l = 0, b = 0, bmin = 1e30;
        for (auto& p : kv.second) { l += p.first; b += p.second; bmin = std::min(bmin, p.second); }
        OUT("  %-24s %2zu pairs   latency %8.1f us   %8.2f GB/s per direction (min %.2f)\n", kv.first.c_str(), kv.second.size(), l / kv.second.size(), b / kv.second.size(), bmin);
    }
    verdict("test5 every pair exchanged and verified", ok);
}

// ---------------------------------------------------------------------------------------------- main
int main(int argc, char** argv)
{
    int ngpu = 0;
    std::set<int> tests = {1, 2, 3, 4, 5};
#ifdef SPIKE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &R);
#endif
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--ngpu") && i + 1 < argc) ngpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tests") && i + 1 < argc)
        {
            tests.clear();
            for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) tests.insert(atoi(t));
        }
        else if (!strcmp(argv[i], "--graph-mode") && i + 1 < argc) g_graphMode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--relaxed")) g_capMode = cudaStreamCaptureModeRelaxed;
        else if (!strcmp(argv[i], "--global")) g_capMode = cudaStreamCaptureModeGlobal;
        else { fprintf(stderr, "usage: %s [--ngpu N] [--tests 1,2,3,4,5] [--graph-mode -1|0|1] [--relaxed|--global]\n", argv[0]); return 1; }
    }
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (nowS() > g_deadline) { fprintf(stderr, "[FAIL] rank %d: HANG (no progress for 60 s) in stage %s\n", g_myRank, g_stage); fflush(stderr); _exit(3); }
        }
    }).detach();
    arm("setup");

    int nDev = 0;
    CK(cudaGetDeviceCount(&nDev));
    int ver = 0;
    NK(ncclGetVersion(&ver));
    int rtv = 0;
    CK(cudaRuntimeGetVersion(&rtv));
    ncclComm_t ce[16], cg[16];
    std::vector<int> devs;
    if (!mpiMode())
    {
        R = ngpu > 0 ? ngpu : nDev;
        if (R > nDev || R > 16) die("ngpu", __FILE__, __LINE__, "more GPUs requested than visible");
        for (int i = 0; i < R; i++) devs.push_back(i);
        // like ncclCommInitAll, but with a config per communicator (the graph communicator is created graph-only)
        for (int c = 0; c < 2; c++)
        {
            ncclUniqueId id;
            NK(ncclGetUniqueId(&id));
            ncclConfig_t cfg = cfgFor(c == 1);
            NK(ncclGroupStart());
            for (int i = 0; i < R; i++) { CK(cudaSetDevice(i)); NK(ncclCommInitRankConfig(c ? &cg[i] : &ce[i], R, id, i, &cfg)); }
            NK(ncclGroupEnd());
        }
        for (int i = 0; i < R; i++) { Rk r = {}; r.rank = i; r.dev = i; r.ce = ce[i]; r.cg = cg[i]; L.push_back(r); }
    }
    else
    {
#ifdef SPIKE_MPI
        MPI_Comm loc;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &loc);
        int lr, ls;
        MPI_Comm_rank(loc, &lr);
        MPI_Comm_size(loc, &ls);
        if (ls != R) die("topology", __FILE__, __LINE__, "all ranks must be on ONE node");
        if (lr >= nDev) die("device", __FILE__, __LINE__, "more ranks than visible GPUs on the node");
        CK(cudaSetDevice(lr));
        Rk r = {};
        r.rank = g_myRank; r.dev = lr;
        for (int c = 0; c < 2; c++)
        {
            ncclUniqueId id;
            if (g_myRank == 0) NK(ncclGetUniqueId(&id));
            MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
            ncclConfig_t cfg = cfgFor(c == 1);
            NK(ncclCommInitRankConfig(c ? &r.cg : &r.ce, R, id, g_myRank, &cfg));
        }
        L.push_back(r);
#endif
    }
    each([](Rk& r) {
        CK(cudaStreamCreateWithFlags(&r.s, cudaStreamNonBlocking));
        CK(cudaMalloc(&r.x, sizeof(double)));
        CK(cudaMalloc(&r.gath, 16 * sizeof(double)));
        CK(cudaMalloc(&r.out, 4 * sizeof(double)));
        CK(cudaMalloc(&r.st, sizeof(St)));
    });
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, L[0].dev));
    OUT("nccl_spike (%s): %d rank(s), GPU0 %s sm_%d%d, NCCL %d.%d.%d, CUDA runtime %d, NCCL_GRAPH_MIXING_SUPPORT=%s, graph comm graphUsageMode=%s, capture mode %s\n",
        mpiMode() ? "MPI" : "single process", R, pr.name, pr.major, pr.minor, ver / 10000, (ver / 100) % 100, ver % 100, rtv,
        getenv("NCCL_GRAPH_MIXING_SUPPORT") ? getenv("NCCL_GRAPH_MIXING_SUPPORT") : "(unset=1)",
#ifdef HAVE_GRAPH_USAGE_MODE
        g_graphMode < 0 ? "default" : g_graphMode == 0 ? "0 (graph-only)" : "1 (mixing)",
#else
        "n/a (this nccl.h has no graphUsageMode: use NCCL_GRAPH_MIXING_SUPPORT=0)",
#endif
        g_capMode == cudaStreamCaptureModeThreadLocal ? "thread-local" : g_capMode == cudaStreamCaptureModeRelaxed ? "relaxed" : "global");
    if (R == 1) OUT("NOTE: 1 rank: halo is a self send/recv, test5 is skipped, cross-device behaviour is NOT exercised\n");

    if (tests.count(1)) test1_eager();
    if (tests.count(2)) test2_graph();
    if (tests.count(3)) test3_while();
    if (tests.count(4)) test4_timing();
    if (tests.count(5)) test5_pairs();

    arm("teardown");
    freeGraphs();
    OUT("\n%s (%d failing test(s))\n", g_fails ? "OVERALL FAIL" : "OVERALL PASS", g_fails);
    fflush(stdout);
#ifdef SPIKE_MPI
    MPI_Finalize();
#endif
    _exit(g_fails ? 1 : 0);   // skip ncclCommDestroy: a failed run may leave peers blocked and we want the exit status
}
