// ProcComm tests (InterFoamStep.md WP2 gate G2): halo round trips, face exchange, deterministic reductions,
// np=1 identity, collective abort, stream ordering, on the synthetic box partitions of proc_comm.cuh.
//
//   test_proc_comm            LOOPBACK backend: np = 1,2,3,4,8 virtual ranks (threads) on one GPU
//   mpirun -np N test_proc_comm mpi     MPI-host backend, one process per rank (np > GPUs is fine)
//   mpirun -np N test_proc_comm nccl    NCCL backend (build with BRAE_WITH_NCCL), one process per GPU: np <= GPUs
//   mpirun -np N test_proc_comm auto    ProcComm::create() decides (env BRAE_COMM, else NCCL if built and np <= GPUs, else mpi)
//   An optional second argument keeps only the tests whose name contains it (e.g. `nccl collectiveAbort`).
//   The backend that ran is printed ("backend: ...") and prefixes every PASS line.
//
// Every blocking region runs under a watchdog: a hang prints FAIL and exits instead of stalling ctest.
#include "proc_comm.cuh"
#include "device_buffer.cuh"

#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <mpi.h>
#include <thread>

using namespace brae;

namespace {

#define REQUIRE(cond, ...)                                                                                          \
    do { if (!(cond)) {                                                                                             \
        char b_[512]; std::snprintf(b_, sizeof b_, __VA_ARGS__);                                                    \
        throw std::runtime_error(std::string(b_) + "  [" #cond "] at line " + std::to_string(__LINE__)); } } while (0)

void cu(cudaError_t e, const char* w)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string("cuda ") + w + ": " + cudaGetErrorString(e));
}

template<class T> struct DBuf
{
    T* p = nullptr;
    std::size_t n = 0;
    explicit DBuf(std::size_t n_) : n(n_) { cu(cudaMalloc(&p, std::max<std::size_t>(n, 1)*sizeof(T)), "malloc"); }
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
    std::vector<T> host() const
    {
        cu(cudaDeviceSynchronize(), "device sync");   // the kernels that wrote it ran on the caller's streams
        std::vector<T> h(n);
        if (n) cu(cudaMemcpy(h.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost), "download");
        return h;
    }
};

__host__ __device__ inline double hv(unsigned long long a, unsigned long long b)
{
    unsigned long long z = a*0x9E3779B97F4A7C15ull + b*0xBF58476D1CE4E5B9ull + 0x94D049BB133111EBull;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull; z ^= z >> 27; z *= 0x94D049BB133111EBull; z ^= z >> 31;
    return static_cast<double>(z >> 11)*(1.0/9007199254740992.0) + 0.25;
}

bool sameBits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

__global__ void delayK(long long cycles)
{
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

// f(cell c, component comp) = hv(global*16 + comp, it), stored at f[c*cs]
__global__ void fillOwnedK(double* f, long long cs, int nC, const int* glob, int comp, int it)
{
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < nC; c += gridDim.x*blockDim.x)
        f[c*cs] = hv(static_cast<unsigned long long>(glob[c])*16 + comp, it);
}

// counts owned cells whose value changed and halo slots that differ from the owner's value
__global__ void verifyK(const double* f, long long cs, int nC, int nHalo, const int* glob, const int* haloGlob,
                        int comp, int it, unsigned* bad)
{
    const int n = nC + nHalo;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < n; c += gridDim.x*blockDim.x)
    {
        const int g = c < nC ? glob[c] : haloGlob[c - nC];
        const double want = hv(static_cast<unsigned long long>(g)*16 + comp, it);
        if (__double_as_longlong(f[c*cs]) != __double_as_longlong(want)) atomicAdd(bad, 1u);
    }
}

__global__ void copyHaloK(const double* f, double* out, int nC, int nHalo)
{
    for (int s = blockIdx.x*blockDim.x + threadIdx.x; s < nHalo; s += gridDim.x*blockDim.x) out[s] = f[nC + s];
}

__global__ void fillFacesK(double* a, int n, const long long* key, int rank, int it)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
        a[i] = hv(static_cast<unsigned long long>(key[i]), static_cast<unsigned long long>(rank) + 1000ull*it);
}

int blocks(int n) { return std::max(1, std::min((n + 255)/256, 4096)); }

using Body = std::function<void(ProcComm&, int rank, int np, cudaStream_t)>;

// ---------------------------------------------------------------------------------------------------------
// the rank-side test bodies
// ---------------------------------------------------------------------------------------------------------

// Fills all components of a field with fillOwnedK, NaN-fills the halo tails, exchanges with `doExchange`, verifies.
struct FieldSet
{
    std::vector<std::unique_ptr<DBuf<double>>> bufs;
    std::vector<HaloField> fields;
    std::vector<std::pair<double*, long long>> comps;   // (component base pointer, cellStride)
};

void checkCells(ProcComm& c, const SyntheticPartition& sp, const HaloPlan& plan, cudaStream_t s,
                const char* name, const std::function<FieldSet(int nExt)>& make,
                const std::function<void(FieldSet&)>& exchange, int iterations)
{
    const int r = c.rank();
    const int nC = plan.nCells(), nH = plan.nHalo(), nExt = plan.nExt();
    DBuf<int> dGlob(sp.cellGlobal[r]), dHalo(sp.haloGlobal[r]);
    DBuf<unsigned> bad(1);
    FieldSet fs = make(nExt);
    cu(cudaMemsetAsync(bad.p, 0, sizeof(unsigned), s), "memset");
    for (int it = 0; it < iterations; ++it)
    {
        int k = 0;
        for (auto& cp : fs.comps)
        {
            // halo tail := NaN (0xFF bytes) before every exchange; owned cells := f(global, comp, it)
            if (nH > 0)
                cu(cudaMemset2DAsync(cp.first + static_cast<long long>(nC)*cp.second, sizeof(double)*cp.second, 0xFF,
                                     sizeof(double), nH, s), "halo sentinel");
            fillOwnedK<<<blocks(nC), 256, 0, s>>>(cp.first, cp.second, nC, dGlob.p, k, it);
            ++k;
        }
        exchange(fs);
        k = 0;
        for (auto& cp : fs.comps)
        {
            verifyK<<<blocks(nExt), 256, 0, s>>>(cp.first, cp.second, nC, nH, dGlob.p, dHalo.p, k, it, bad.p);
            ++k;
        }
    }
    cu(cudaStreamSynchronize(s), "sync");
    cu(cudaGetLastError(), "kernels");
    const unsigned nBad = bad.host()[0];
    REQUIRE(nBad == 0, "rank %d: %s: %u halo/owned values differ after exchange", r, name, nBad);
}

FieldSet makeScalars(int n, int nExt)
{
    FieldSet fs;
    for (int i = 0; i < n; ++i)
    {
        fs.bufs.emplace_back(new DBuf<double>(nExt));
        fs.fields.push_back({fs.bufs.back()->p, 1, 0, 1});
        fs.comps.push_back({fs.bufs.back()->p, 1});
    }
    return fs;
}

void testCells(ProcComm& c, const SyntheticPartition& sp, const HaloPlan& plan, cudaStream_t s)
{
    // scalar
    checkCells(c, sp, plan, s, "scalar", [](int nExt) { return makeScalars(1, nExt); },
               [&](FieldSet& f) { c.exchangeCells(plan, f.fields[0].data, s); }, 3);
    // vector as 3 component arrays
    checkCells(c, sp, plan, s, "vector3", [](int nExt) { return makeScalars(3, nExt); },
               [&](FieldSet& f) { c.exchangeVector(plan, f.fields[0].data, f.fields[1].data, f.fields[2].data, s); }, 3);
    // batch of scalars
    checkCells(c, sp, plan, s, "batch5", [](int nExt) { return makeScalars(5, nExt); },
               [&](FieldSet& f) { c.exchangeCells(plan, {f.fields[0].data, f.fields[1].data, f.fields[2].data, f.fields[3].data, f.fields[4].data}, s); }, 3);
    // 20 components: more than one pack/unpack launch
    checkCells(c, sp, plan, s, "batch20", [](int nExt) { return makeScalars(20, nExt); },
               [&](FieldSet& f) { c.exchangeCells(plan, f.fields.data(), static_cast<int>(f.fields.size()), s); }, 2);
    // packed SoA, stride larger than nC+nHalo
    checkCells(c, sp, plan, s, "packedSoA9", [](int nExt) {
                   FieldSet fs;
                   const long long stride = nExt + 3;
                   fs.bufs.emplace_back(new DBuf<double>(9*stride));
                   for (int k = 0; k < 9; ++k) fs.comps.push_back({fs.bufs[0]->p + k*stride, 1});
                   fs.fields.push_back({fs.bufs[0]->p, 9, stride, 1});
                   return fs;
               },
               [&](FieldSet& f) { c.exchangeCellsPacked(plan, f.fields[0].data, 9, f.fields[0].compStride, s); }, 2);
    // packed AoS
    checkCells(c, sp, plan, s, "packedAoS4", [](int nExt) {
                   FieldSet fs;
                   fs.bufs.emplace_back(new DBuf<double>(static_cast<std::size_t>(4)*nExt));
                   for (int k = 0; k < 4; ++k) fs.comps.push_back({fs.bufs[0]->p + k, 4});
                   fs.fields.push_back({fs.bufs[0]->p, 4, 1, 4});
                   return fs;
               },
               [&](FieldSet& f) { c.exchangeCellsAoS(plan, f.fields[0].data, 4, s); }, 2);
    // back-to-back pipeline with changing values: reuse of the staging buffers must not corrupt anything
    checkCells(c, sp, plan, s, "pipeline200", [](int nExt) { return makeScalars(2, nExt); },
               [&](FieldSet& f) { c.exchangeCells(plan, {f.fields[0].data, f.fields[1].data}, s); }, 200);
}

void testFaces(ProcComm& c, const SyntheticPartition& sp, const HaloPlan& plan, cudaStream_t s)
{
    const int r = c.rank();
    const int nf = plan.nProcFaces();
    std::vector<int> nbrOf(nf);
    for (const HaloNeighbour& n : plan.desc().nbrs)
        for (int i = n.faceStart; i < n.faceStart + n.faceCount; ++i) nbrOf[i] = n.rank;
    DBuf<long long> dKey(sp.faceKey[r]);
    DBuf<double> arr(nf);
    const FaceOp ops[4] = {FaceOp::Replace, FaceOp::Min, FaceOp::Max, FaceOp::Sum};
    const char* names[4] = {"replace", "min", "max", "sum"};
    for (int it = 0; it < 4; ++it)
        for (int o = 0; o < 4; ++o)
        {
            fillFacesK<<<blocks(nf), 256, 0, s>>>(arr.p, nf, dKey.p, r, it);
            c.exchangeFaces(plan, arr.p, ops[o], s);
            cu(cudaStreamSynchronize(s), "sync");
            const std::vector<double> h = arr.host();
            for (int i = 0; i < nf; ++i)
            {
                const double mine = hv(static_cast<unsigned long long>(sp.faceKey[r][i]), static_cast<unsigned long long>(r) + 1000ull*it);
                const double nb = hv(static_cast<unsigned long long>(sp.faceKey[r][i]), static_cast<unsigned long long>(nbrOf[i]) + 1000ull*it);
                const double want = o == 0 ? nb : o == 1 ? (nb < mine ? nb : mine) : o == 2 ? (nb > mine ? nb : mine) : mine + nb;
                REQUIRE(sameBits(h[i], want), "rank %d face %d (%s, it %d): %.17g != %.17g", r, i, names[o], it, h[i], want);
            }
        }
}

double rv(int rank, int i, int rep)
{
    const double m = hv(static_cast<unsigned long long>(rank)*977 + i*13 + 1, static_cast<unsigned long long>(rep)*7 + 3);
    const double e = hv(static_cast<unsigned long long>(rank)*31 + i*101 + 5, static_cast<unsigned long long>(rep)*11 + 1);
    return (m - 0.75)*std::ldexp(1.0, static_cast<int>(e*24) - 12);
}

double refFold(int np, int i, int rep, ReduceOp op, int root, bool bcast)
{
    if (bcast) return rv(root, i, rep);
    double acc = rv(0, i, rep);
    for (int q = 1; q < np; ++q)
    {
        const double v = rv(q, i, rep);
        acc = op == ReduceOp::Sum ? acc + v : op == ReduceOp::Min ? (v < acc ? v : acc) : (v > acc ? v : acc);
    }
    return acc;
}

void testReductions(ProcComm& c, int np, cudaStream_t s)
{
    const int r = c.rank();
    const ReduceOp ops[3] = {ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max};
    const char* names[3] = {"sum", "min", "max"};
    const int sizes[3] = {1, 7, 64};
    for (int n : sizes)
    {
        DBuf<double> d(n);
        for (int o = 0; o < 3; ++o)
            for (int rep = 0; rep < 100; ++rep)
            {
                std::vector<double> mine(n), host(n);
                for (int i = 0; i < n; ++i) mine[i] = host[i] = rv(r, i, rep);
                cu(cudaMemcpyAsync(d.p, mine.data(), n*sizeof(double), cudaMemcpyHostToDevice, s), "H2D");
                cu(cudaStreamSynchronize(s), "sync");
                c.allReduce(d.p, n, ops[o], s);
                c.allReduceHost(host.data(), n, ops[o]);
                const std::vector<double> got = d.host();
                for (int i = 0; i < n; ++i)
                {
                    const double want = refFold(np, i, rep, ops[o], 0, false);
                    REQUIRE(sameBits(got[i], want), "rank %d device allReduce %s n=%d rep=%d i=%d: %.17g != %.17g", r, names[o], n, rep, i, got[i], want);
                    REQUIRE(sameBits(host[i], want), "rank %d host allReduce %s n=%d rep=%d i=%d: %.17g != %.17g", r, names[o], n, rep, i, host[i], want);
                }
            }
        // scalar convenience + bcast from every root
        for (int root = 0; root < np; ++root)
        {
            std::vector<double> mine(n), host(n);
            for (int i = 0; i < n; ++i) mine[i] = host[i] = rv(r, i, 5);
            cu(cudaMemcpyAsync(d.p, mine.data(), n*sizeof(double), cudaMemcpyHostToDevice, s), "H2D");
            c.bcastDevice(d.p, n, root, s);
            c.bcastHost(host.data(), n, root);
            const std::vector<double> got = d.host();
            for (int i = 0; i < n; ++i)
            {
                REQUIRE(sameBits(got[i], rv(root, i, 5)), "rank %d bcastDevice root %d", r, root);
                REQUIRE(sameBits(host[i], rv(root, i, 5)), "rank %d bcastHost root %d", r, root);
            }
            REQUIRE(sameBits(c.bcastScalar(rv(r, 0, 9), root), rv(root, 0, 9)), "rank %d bcastScalar root %d", r, root);
        }
    }
    REQUIRE(sameBits(c.allReduce(rv(r, 0, 1), ReduceOp::Sum), refFold(np, 0, 1, ReduceOp::Sum, 0, false)), "rank %d scalar allReduce", r);
    // the reference really depends on the rank order (guards against a test that cannot see an order bug)
    if (r == 0 && np >= 3)
    {
        bool orderMatters = false;
        for (int rep = 0; rep < 100 && !orderMatters; ++rep)
        {
            double fwd = rv(0, 0, rep), rev = rv(np - 1, 0, rep);
            for (int q = 1; q < np; ++q) fwd += rv(q, 0, rep);
            for (int q = np - 2; q >= 0; --q) rev += rv(q, 0, rep);
            orderMatters = !sameBits(fwd, rev);
        }
        REQUIRE(orderMatters, "test values are not rounding sensitive to the summation order");
    }
    c.barrier();
}

// An exchange enqueued between two kernels on the stream is ordered with them, across ranks: the slow rank's
// peers must see its NEW values.
void testStreamOrder(ProcComm& c, const SyntheticPartition& sp, const HaloPlan& plan, cudaStream_t s)
{
    const int r = c.rank(), nC = plan.nCells(), nH = plan.nHalo(), nExt = plan.nExt();
    DBuf<int> dGlob(sp.cellGlobal[r]);
    DBuf<double> f(nExt), out(nH);
    cu(cudaMemsetAsync(f.p, 0, nExt*sizeof(double), s), "zero");
    cu(cudaMemsetAsync(out.p, 0, std::max(nH, 1)*sizeof(double), s), "zero");
    cu(cudaStreamSynchronize(s), "sync");
    for (int rep = 0; rep < 3; ++rep)
    {
        delayK<<<1, 1, 0, s>>>(r == 0 ? 200000000LL : 1000LL);       // rank 0 is late: ~0.1 s
        fillOwnedK<<<blocks(nC), 256, 0, s>>>(f.p, 1, nC, dGlob.p, 0, 100 + rep);   // kernel A
        c.exchangeCells(plan, f.p, s);                                // the exchange under test
        copyHaloK<<<blocks(nH), 256, 0, s>>>(f.p, out.p, nC, nH);     // kernel B
        cu(cudaStreamSynchronize(s), "sync");
        const std::vector<double> h = out.host();
        for (int k = 0; k < nH; ++k)
            REQUIRE(sameBits(h[k], hv(static_cast<unsigned long long>(sp.haloGlobal[r][k])*16, 100 + rep)),
                    "rank %d rep %d halo slot %d not ordered with the kernels around the exchange", r, rep, k);
        // refresh the halo with stale values for the next round
        cu(cudaMemsetAsync(f.p, 0, nExt*sizeof(double), s), "zero");
    }
    c.barrier();
}

void runCase(ProcComm& c, int np, const SyntheticPartition& sp, cudaStream_t s)
{
    const int r = c.rank();
    {
        std::unique_ptr<HaloPlan> plan = c.makePlan(sp.descs[r], 24);
        REQUIRE(plan->nCells() == static_cast<int>(sp.cellGlobal[r].size()), "plan nCells");
        testCells(c, sp, *plan, s);
        testFaces(c, sp, *plan, s);
        testStreamOrder(c, sp, *plan, s);
        cu(cudaStreamSynchronize(s), "sync");
        c.barrier();   // every rank is done with every plan's staging before any plan is destroyed
    }
    testReductions(c, np, s);
}

// plan built the way the later steps do it: from processor patches, neighbour face cells learnt by the comm
void testBuildFromPatches(ProcComm& c, const SyntheticPartition& sp, cudaStream_t s)
{
    const int r = c.rank();
    std::vector<ProcessorPatchDesc> patches = sp.patches[r];
    for (auto& p : patches) p.nbrFaceCells.clear();
    std::unique_ptr<HaloPlan> plan = c.buildPlanFromPatches(static_cast<int>(sp.cellGlobal[r].size()), patches, 8);
    const HaloPlanDesc& want = sp.descs[r];
    REQUIRE(plan->desc().nHalo == want.nHalo && plan->desc().nProcFaces == want.nProcFaces, "rank %d plan sizes", r);
    REQUIRE(plan->desc().sendCells == want.sendCells && plan->desc().faceToSlot == want.faceToSlot, "rank %d plan index lists", r);
    testStreamOrder(c, sp, *plan, s);
    cu(cudaStreamSynchronize(s), "sync");
    c.barrier();
}

// np = 1: identity, no allocation, no stream interaction
void testIdentity(ProcComm& c, cudaStream_t s)
{
    REQUIRE(c.size() == 1 && c.rank() == 0 && c.isMaster(), "np=1 rank/size");
    const std::size_t a0 = ProcComm::allocationCount();
    const std::size_t p0 = detail::devicePool().mallocs();
    SyntheticPartition sp = partitionBox(6, 5, 4, 1, std::vector<int>(120, 0));
    std::unique_ptr<HaloPlan> plan = c.makePlan(sp.descs[0]);
    REQUIRE(plan->nHalo() == 0 && plan->nProcFaces() == 0 && plan->nNeighbours() == 0, "np=1 plan");
    double* d = nullptr;
    cu(cudaMalloc(&d, 64*sizeof(double)), "malloc");
    cu(cudaMemsetAsync(d, 0, 64*sizeof(double), s), "memset");
    cu(cudaStreamSynchronize(s), "sync");
    const std::size_t a1 = ProcComm::allocationCount();
    delayK<<<1, 1, 0, s>>>(300000000LL);   // keeps the stream busy ~0.15 s: any sync in a call would show
    const auto t0 = std::chrono::steady_clock::now();
    double v = 3.5;
    c.exchangeCells(*plan, d, s);
    c.exchangeCells(*plan, {d, d + 1}, s);
    c.exchangeVector(*plan, d, d, d, s);
    c.exchangeFaces(*plan, d, FaceOp::Min, s);
    c.allReduce(d, 8, ReduceOp::Sum, s);
    c.bcastDevice(d, 8, 0, s);
    c.allReduceHost(&v, 1, ReduceOp::Max);
    REQUIRE(c.allReduce(v, ReduceOp::Sum) == 3.5 && c.bcastScalar(v, 0) == 3.5, "np=1 reductions are the identity");
    c.barrier();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE(cudaStreamQuery(s) == cudaErrorNotReady, "np=1 call synchronised the stream (took %.0f us)", us);
    cu(cudaStreamSynchronize(s), "sync");
    REQUIRE(ProcComm::allocationCount() == a1 && a1 == a0, "np=1 allocated through ProcComm (%zu -> %zu -> %zu)", a0, a1, ProcComm::allocationCount());
    REQUIRE(detail::devicePool().mallocs() == p0, "np=1 allocated through the device pool");
    // the abort of one rank is a plain exception
    bool threw = false;
    try { c.collectiveAbort("solo"); } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "solo"; }
    REQUIRE(threw, "np=1 collectiveAbort");
    cudaFree(d);
}

// A rank-local failure: every rank throws the SAME ProcCommAbort, with no hang. `late`: the failing rank is slow, the others
// are already waiting in the collective.
void testAbort(ProcComm& c, int np, const SyntheticPartition& sp, cudaStream_t s, bool late, bool inExchange)
{
    const int r = c.rank();
    const int bad = np - 1;
    std::unique_ptr<HaloPlan> plan = c.makePlan(sp.descs[r], 4);
    DBuf<double> d(plan->nExt() + 8);
    cu(cudaMemsetAsync(d.p, 0, (plan->nExt() + 8)*sizeof(double), s), "zero");
    cu(cudaStreamSynchronize(s), "sync");
    std::string got;
    int origin = -1;
    try
    {
        c.guarded([&] {
            if (r == bad)
            {
                if (late) std::this_thread::sleep_for(std::chrono::milliseconds(300));
                throw std::runtime_error("injected failure on the last rank");
            }
            if (inExchange) c.exchangeCells(*plan, d.p, s);
            else c.allReduce(d.p, 4, ReduceOp::Sum, s);
            c.barrier();
        });
    }
    catch (const ProcCommAbort& e) { got = e.what(); origin = e.originRank(); }
    REQUIRE(got == "injected failure on the last rank", "rank %d got abort message '%s'", r, got.c_str());
    REQUIRE(origin == bad, "rank %d abort origin %d != %d", r, origin, bad);
    cu(cudaStreamSynchronize(s), "sync");
}

// ---------------------------------------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------------------------------------
struct Watchdog
{
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::thread t;
    explicit Watchdog(const char* what, int seconds)
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
    ~Watchdog()
    {
        { std::lock_guard<std::mutex> lk(m); done = true; }
        cv.notify_all();
        t.join();
    }
};

int runLoopback(const char* name, int np, const Body& body)
{
    Watchdog wd(name, 180);
    std::vector<std::unique_ptr<ProcComm>> comms = ProcComm::createLoopback(np);
    std::vector<std::string> err(np);
    std::vector<std::thread> th;
    for (int r = 0; r < np; ++r)
        th.emplace_back([&, r] {
            try
            {
                comms[r]->bindThread();
                cudaStream_t s;
                cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
                try { body(*comms[r], r, np, s); }
                catch (const ProcCommAbort&) { throw; }
                catch (const std::exception& e) { if (np > 1) { try { comms[r]->collectiveAbort(e.what()); } catch (...) {} } throw; }
                cudaStreamDestroy(s);
            }
            catch (const std::exception& e) { err[r] = e.what(); }
        });
    for (auto& t : th) t.join();
    int fails = 0;
    for (int r = 0; r < np; ++r)
        if (!err[r].empty()) { ++fails; std::printf("FAIL %s np=%d rank %d: %s\n", name, np, r, err[r].c_str()); }
    return fails;
}

std::vector<std::pair<std::string, SyntheticPartition>> partitionsFor(int np)
{
    std::vector<std::pair<std::string, SyntheticPartition>> v;
    static const int fac[9][3] = {{0,0,0},{1,1,1},{2,1,1},{1,3,1},{2,2,1},{0,0,0},{0,0,0},{0,0,0},{2,2,2}};
    if (np == 1 || np == 2 || np == 3 || np == 4 || np == 8)
    {
        const int nx = 20, ny = 15, nz = 12;
        v.emplace_back("block " + std::to_string(fac[np][0]) + "x" + std::to_string(fac[np][1]) + "x" + std::to_string(fac[np][2]),
                       partitionBox(nx, ny, nz, np, blockOwner(nx, ny, nz, fac[np][0], fac[np][1], fac[np][2])));
        if (np > 1)
            v.emplace_back("scattered chunk 3", partitionBox(18, 15, 12, np, scatteredOwner(18, 15, 12, np, 3)));
        if (np == 3) v.emplace_back("block 3x1x1 uneven", partitionBox(11, 7, 5, 3, blockOwner(11, 7, 5, 3, 1, 1)));
    }
    return v;
}

int loopbackSuite()
{
    int fails = 0;
    for (int np : {1, 2, 3, 4, 8})
    {
        for (auto& pc : partitionsFor(np))
        {
            const std::string name = "np=" + std::to_string(np) + " " + pc.first;
            const SyntheticPartition& sp = pc.second;
            const int f = runLoopback(name.c_str(), np, [&](ProcComm& c, int, int, cudaStream_t s) { runCase(c, np, sp, s); });
            std::printf("%s %s\n", f ? "FAIL" : "PASS", name.c_str());
            fails += f;
            if (np > 1)
            {
                const int g = runLoopback(name.c_str(), np, [&](ProcComm& c, int, int, cudaStream_t s) { testBuildFromPatches(c, sp, s); });
                std::printf("%s %s plan from processor patches\n", g ? "FAIL" : "PASS", name.c_str());
                fails += g;
            }
        }
        if (np == 1)
        {
            const int f = runLoopback("np=1 identity", 1, [&](ProcComm& c, int, int, cudaStream_t s) { testIdentity(c, s); });
            std::printf("%s np=1 identity (no allocation, no sync)\n", f ? "FAIL" : "PASS");
            fails += f;
        }
        else
        {
            const SyntheticPartition sp = partitionBox(12, 9, 6, np, np == 8 ? blockOwner(12, 9, 6, 2, 2, 2) : np == 4 ? blockOwner(12, 9, 6, 2, 2, 1) : np == 3 ? blockOwner(12, 9, 6, 3, 1, 1) : blockOwner(12, 9, 6, 2, 1, 1));
            for (int late = 0; late < 2; ++late)
                for (int inEx = 0; inEx < 2; ++inEx)
                {
                    const int f = runLoopback("abort", np, [&](ProcComm& c, int, int, cudaStream_t s) { testAbort(c, np, sp, s, late, inEx); });
                    std::printf("%s np=%d collectiveAbort (%s failing rank, in %s)\n", f ? "FAIL" : "PASS", np, late ? "late" : "early",
                                inEx ? "exchangeCells" : "allReduce");
                    fails += f;
                }
        }
    }
    return fails;
}

int mpiSuite(int argc, char** argv, const std::string& backendArg)
{
    Pstream::init(argc, argv);
    const int np = Pstream::nProcs();
    int fails = 0;
    {
        std::unique_ptr<ProcComm> c = std::move(ProcComm::create(backendArg == "auto" ? "" : backendArg)[0]);
        const std::string bn = ProcComm::backendName(c->backend());
        if (c->isMaster())
            std::printf("backend: %s (np=%d, rank 0 on GPU %d%s)\n", bn.c_str(), np, c->device(),
                        np == 1 ? "; np=1 is the identity: no communicator is created" : "");
        cudaStream_t s;
        cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        // optional argv[2]: run only the tests whose name contains it (every rank applies the same filter)
        const std::string only = argc > 2 ? argv[2] : "";
        auto run = [&](const char* name, const std::function<void()>& f) {
            if (!only.empty() && std::string(name).find(only) == std::string::npos) return;
            Watchdog wd(name, 180);
            int bad = 0;
            // An abort test RAISES a ProcCommAbort on purpose: it is not a failure, but it MUST still be agreed
            // on collectively. Treating it as a rank-local failure made one rank call MPI_Abort while its peers
            // went on to the next test's allgather and hung there (a harness bug, not a backend one).
            const bool expectAbort = std::string(name).find("collectiveAbort") != std::string::npos;
            try { c->guarded(f); }
            catch (const ProcCommAbort& e)
            {
                if (!expectAbort) { bad = 1; std::printf("FAIL %s rank %d: %s\n", name, c->rank(), e.what()); }
            }
            catch (const std::exception& e) { bad = 1; std::printf("FAIL %s rank %d: %s\n", name, c->rank(), e.what()); }
            // collective verdict (an exception leaves a rank out of step: abort the whole run on a failure)
            int total = Pstream::allReduce(static_cast<label>(bad), ReduceOp::Sum);
            if (total != 0 && bad) std::fflush(stdout);
            if (total != 0)
            {
                if (c->isMaster()) std::printf("FAIL %s (np=%d)\n", name, np);
                std::fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (c->isMaster()) std::printf("PASS %s (np=%d)\n", name, np);
        };
        if (np == 1)
        {
            run((bn + " np=1 identity").c_str(), [&] { testIdentity(*c, s); });
        }
        else
        {
            std::vector<std::pair<std::string, SyntheticPartition>> parts = partitionsFor(np);
            if (parts.empty())
            {
                const int nx = 16, ny = 12, nz = 10;
                parts.emplace_back("block np x 1 x 1", partitionBox(nx, ny, nz, np, blockOwner(nx, ny, nz, np, 1, 1)));
            }
            for (auto& pc : parts)
            {
                const std::string name = bn + " " + pc.first;
                run(name.c_str(), [&] { runCase(*c, np, pc.second, s); });
                run((name + " plan from processor patches").c_str(), [&] { testBuildFromPatches(*c, pc.second, s); });
            }
            const SyntheticPartition& ap = parts.front().second;
            for (int late = 0; late < 2; ++late)
                for (int inEx = 0; inEx < 2; ++inEx)
                {
                    const std::string an = bn + " collectiveAbort (" + (late ? "late" : "early") + " failing rank, in "
                                           + (inEx ? "exchangeCells" : "allReduce") + ")";
                    run(an.c_str(), [&] { testAbort(*c, np, ap, s, late, inEx); });
                }
        }
    }
    Pstream::finalize();
    return fails;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc > 1 && (std::string(argv[1]) == "mpi" || std::string(argv[1]) == "nccl" || std::string(argv[1]) == "auto"))
            return mpiSuite(argc, argv, argv[1]) == 0 ? 0 : 1;
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev <= 0) { std::printf("SKIP: no CUDA device\n"); return 77; }
        std::printf("backend: loopback (np = 1,2,3,4,8 virtual ranks on one GPU)\n");
        const int fails = loopbackSuite();
        std::printf("test_proc_comm loopback: %s (%d failing rank runs)\n", fails ? "FAIL" : "PASS", fails);
        return fails ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
