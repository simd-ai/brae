// ProcComm negative controls for the deadlock windows (review findings 1, 2, 3, 7, 8).
// Each mode injects a failure at the point the old implementation could not survive and asserts that EVERY rank
// returns the same ProcCommAbort (origin + message) within a watchdog, instead of hanging.
//
//   mpirun -np N test_proc_comm_failure mpi  <mode>     (N >= 2; N = 2,3,4,8 in ctest)
//   mpirun -np N test_proc_comm_failure nccl <mode>     (NCCL build; shared GPU: BRAE_NCCL_ALLOW_SHARED_GPU=1)
//
// modes:
//   post-status   rank N-1 fails AFTER the opening status phase of exchangeCells, before the nonblocking wait
//                 (env BRAE_COMM_FAIL_AFTER_STATUS=1 is honoured by the backend; without it the test throws the
//                 same error from the caller, which is the pre-fix behaviour this control must catch).
//   capacity      rank N-1 exchanges more components than its plan allows (a local capacity error): every rank
//                 must abort together, none may wait for a message.
//   nccl-queued   rank N-1 fails while the peers have queued NCCL device collectives and are stream-synchronising.
//   revoke        BRAE_NCCL_FAIL_REVOKE=<rank> forces one rank's revoke step of the abort recovery to throw:
//                 no rank may skip the recovery barrier, and all ranks must agree on the outcome.
//   asym          one-sided zero/nonzero processor patch: participation must be rejected collectively, not hang.
//
// Exit 0 = every rank saw the expected abort; 1 = failure; 3 = watchdog (the pre-fix hang); 77 = skipped.
#include "proc_comm.cuh"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mpi.h>
#include <thread>

using namespace brae;

namespace {

#define REQUIRE(cond, ...)                                                                                          \
    do { if (!(cond)) {                                                                                             \
        char b_[512]; std::snprintf(b_, sizeof b_, __VA_ARGS__);                                                    \
        throw std::runtime_error(std::string(b_) + "  [" #cond "] at line " + std::to_string(__LINE__)); } } while (0)

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
                std::printf("FAIL: watchdog: '%s' hung for %d s (a rank is blocked in a collective)\n", what, seconds);
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

int watchdogSeconds()
{
    const char* e = std::getenv("BRAE_FAILURE_TEST_WATCHDOG_S");
    return e ? std::atoi(e) : 120;
}

// Two-rank (or N-rank chain) plan: each rank has 4 cells, one neighbour with 2 cells sent / 2 halo slots.
HaloPlanDesc pairDesc(int rank, int np)
{
    HaloPlanDesc d;
    d.nCells = 4;
    const int nbr = (rank + 1) % np;
    const int prev = (rank + np - 1) % np;
    if (np == 2)
    {
        d.nbrs.resize(1);
        d.nbrs[0].rank = nbr;
        d.nbrs[0].sendOffset = 0;
        d.nbrs[0].sendCount = 4;
        d.nbrs[0].recvStart = 0;
        d.nbrs[0].recvCount = 4;
        d.nbrs[0].faceStart = 0;
        d.nbrs[0].faceCount = 4;
        d.sendCells = {0, 1, 2, 3};
        d.faceCells = {0, 1, 2, 3};
        d.faceToSlot = {0, 1, 2, 3};
        d.nHalo = 4;
        d.nProcFaces = 4;
        return d;
    }
    d.nbrs.resize(2);
    d.nbrs[0].rank = nbr;
    d.nbrs[0].sendOffset = 0;
    d.nbrs[0].sendCount = 2;
    d.nbrs[0].recvStart = 0;
    d.nbrs[0].recvCount = 2;
    d.nbrs[0].faceStart = 0;
    d.nbrs[0].faceCount = 2;
    d.nbrs[1].rank = prev;
    d.nbrs[1].sendOffset = 2;
    d.nbrs[1].sendCount = 2;
    d.nbrs[1].recvStart = 2;
    d.nbrs[1].recvCount = 2;
    d.nbrs[1].faceStart = 2;
    d.nbrs[1].faceCount = 2;
    d.sendCells = {0, 1, 2, 3};
    d.faceCells = {0, 1, 2, 3};
    d.faceToSlot = {0, 1, 2, 3};
    d.nHalo = 4;
    d.nProcFaces = 4;
    return d;
}

// The post-status window: the backend raises a local CUDA failure after the opening status phase.
void postStatus(ProcComm& c, int np, cudaStream_t s)
{
    const int r = c.rank();
    const int bad = np - 1;
    std::unique_ptr<HaloPlan> plan = c.makePlan(pairDesc(r, np), 4);
    double* d = nullptr;
    REQUIRE(cudaMalloc(&d, 64*sizeof(double)) == cudaSuccess, "malloc");
    cudaMemsetAsync(d, 0, 64*sizeof(double), s);
    cudaStreamSynchronize(s);
    // Two ways to enter the post-status window:
    //  - the fixed backend honours BRAE_COMM_FAIL_AFTER_STATUS (a deterministic injection point);
    //  - without that hook (main) the failing rank hands the exchange an INVALID device pointer: the pack kernel
    //    faults, cudaStreamSynchronize returns an error AFTER the opening status phase, and the rank aborts while
    //    its peers are already in the nonblocking wait. Both are the same rank-local failure window.
    const bool hook = std::getenv("BRAE_FAILURE_NO_HOOK") == nullptr;
    if (r == bad && hook) setenv("BRAE_COMM_FAIL_AFTER_STATUS", "1", 1);
    double* badPtr = reinterpret_cast<double*>(static_cast<uintptr_t>(0xdead0000));
    std::string got;
    int origin = -1;
    try
    {
        c.guarded([&] {
            c.exchangeCells(*plan, (r == bad && !hook) ? badPtr : d, s);
            c.barrier();
        });
    }
    catch (const ProcCommAbort& e) { got = e.what(); origin = e.originRank(); }
    if (r == bad && hook) unsetenv("BRAE_COMM_FAIL_AFTER_STATUS");
    REQUIRE(!got.empty(), "rank %d: no abort was raised (the failing rank's error did not propagate)", r);
    REQUIRE(origin == bad, "rank %d: abort origin %d != %d (message '%s')", r, origin, bad, got.c_str());
    // Every rank must report the SAME message: agree on it through a fresh communicator-independent channel.
    int len = static_cast<int>(got.size());
    std::vector<int> lens(np);
    MPI_Allgather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int q = 0; q < np; ++q) REQUIRE(lens[q] == len, "rank %d: abort message length %d != rank %d's %d", r, len, q, lens[q]);
    cudaFree(d);
}

// A local capacity error must abort every rank before anyone waits on a message.
void capacity(ProcComm& c, int np, cudaStream_t s)
{
    const int r = c.rank();
    std::unique_ptr<HaloPlan> plan = c.makePlan(pairDesc(r, np), 1);   // maxComps = 1
    double* d = nullptr;
    REQUIRE(cudaMalloc(&d, 64*sizeof(double)) == cudaSuccess, "malloc");
    std::string got;
    int origin = -1;
    try
    {
        c.guarded([&] {
            c.exchangeCellsPacked(*plan, d, 3, 16, s);   // 3 components > maxComps on every rank
            c.barrier();
        });
    }
    catch (const ProcCommAbort& e) { got = e.what(); origin = e.originRank(); }
    REQUIRE(!got.empty(), "rank %d: a capacity error did not abort the exchange", r);
    REQUIRE(got.find("maxComps") != std::string::npos, "rank %d: unexpected message '%s'", r, got.c_str());
    REQUIRE(origin >= 0, "rank %d: no abort origin", r);
    cudaFree(d);
}

// Peers queue NCCL device collectives and synchronise while one rank fails.
void ncclQueued(ProcComm& c, int np, cudaStream_t s)
{
    const int r = c.rank();
    const int bad = np - 1;
    std::unique_ptr<HaloPlan> plan = c.makePlan(pairDesc(r, np), 4);
    double* d = nullptr;
    REQUIRE(cudaMalloc(&d, 64*sizeof(double)) == cudaSuccess, "malloc");
    cudaMemsetAsync(d, 0, 64*sizeof(double), s);
    cudaStreamSynchronize(s);
    c.barrier();
    std::string got;
    int origin = -1;
    try
    {
        c.guarded([&] {
            if (r != bad)
            {
                // Real NCCL work against the rank that is about to fail, then a blocking device wait: the old
                // code left these ranks in cudaStreamSynchronize for ever.
                for (int k = 0; k < 8; ++k)
                {
                    c.exchangeCells(*plan, d, s);
                    c.allReduce(d, 4, ReduceOp::Sum, s);
                }
                cudaStreamSynchronize(s);
            }
            else
            {
                // Let the peers enqueue their collectives first.
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                throw std::runtime_error("injected NCCL-side failure on the last rank");
            }
            c.barrier();
        });
    }
    catch (const ProcCommAbort& e) { got = e.what(); origin = e.originRank(); }
    REQUIRE(got == "injected NCCL-side failure on the last rank", "rank %d: got '%s'", r, got.c_str());
    REQUIRE(origin == bad, "rank %d: origin %d != %d", r, origin, bad);
    cudaFree(d);
}

// One rank's revoke/quiesce step fails during the recovery: nobody may skip the barrier.
void revoke(ProcComm& c, int np, cudaStream_t s)
{
    const int r = c.rank();
    const int bad = np - 1;
    setenv("BRAE_NCCL_FAIL_REVOKE", std::to_string(bad).c_str(), 1);
    std::string got;
    int origin = -1;
    try
    {
        c.guarded([&] {
            if (r == bad) throw std::runtime_error("injected failure forcing revoke recovery");
            c.barrier();
        });
    }
    catch (const ProcCommAbort& e) { got = e.what(); origin = e.originRank(); }
    unsetenv("BRAE_NCCL_FAIL_REVOKE");
    REQUIRE(got == "injected failure forcing revoke recovery", "rank %d: got '%s'", r, got.c_str());
    REQUIRE(origin == bad, "rank %d: origin %d != %d", r, origin, bad);
    // The recovery is collective: every rank must reach this barrier (a rank that skipped the recovery barrier
    // would leave the others here for ever, which the watchdog catches).
    MPI_Barrier(MPI_COMM_WORLD);
}

// One-sided zero/nonzero processor patch: rejected collectively, no probe waiting for a missing message.
void asym(ProcComm& c, int np, cudaStream_t)
{
    const int r = c.rank();
    // Rank 0 drops its patch to rank 1 (as if a zero-face patch had been filtered out); rank 1 keeps a 2-face one.
    std::vector<ProcessorPatchDesc> patches;
    if (np == 2)
    {
        if (r == 1)
        {
            ProcessorPatchDesc p;
            p.nbrRank = 0;
            p.faceCells = {0, 1};
            patches.push_back(p);
        }
    }
    else
    {
        ProcessorPatchDesc p;
        p.nbrRank = (r + 1) % np;
        p.faceCells = {0, 1};
        if (r != 0) patches.push_back(p);   // rank 0 is the one-sided deserter
    }
    std::string got;
    try
    {
        c.guarded([&] { auto plan = c.buildPlanFromPatches(4, patches); });
    }
    catch (const ProcCommAbort& e) { got = e.what(); }
    catch (const std::exception& e) { got = e.what(); }
    REQUIRE(!got.empty(), "rank %d: an asymmetric processor patch was accepted", r);
    int len = static_cast<int>(got.size());
    std::vector<int> lens(np);
    MPI_Allgather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int q = 0; q < np; ++q) REQUIRE(lens[q] > 0, "rank %d: rank %d saw no error", r, q);
}

} // namespace

int main(int argc, char** argv)
{
    const std::string backend = argc > 1 ? argv[1] : "mpi";
    const std::string mode = argc > 2 ? argv[2] : "post-status";
    // Small watchdog for the exchange: the control must fail fast instead of waiting 30 s per rank.
    setenv("BRAE_COMM_WATCHDOG_MS", "4000", 0);
    Pstream::init(argc, argv);
    const int np = Pstream::nProcs();
    int rc = 0;
    if (np < 2)
    {
        std::printf("SKIP: %s needs np >= 2\n", mode.c_str());
        Pstream::finalize();
        return 77;
    }
    {
        Watchdog wd(mode.c_str(), watchdogSeconds());
        try
        {
            std::unique_ptr<ProcComm> c = std::move(ProcComm::create(backend)[0]);
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            if (mode == "post-status") postStatus(*c, np, s);
            else if (mode == "capacity") capacity(*c, np, s);
            else if (mode == "nccl-queued") ncclQueued(*c, np, s);
            else if (mode == "revoke") revoke(*c, np, s);
            else if (mode == "asym") asym(*c, np, s);
            else throw std::runtime_error("unknown mode '" + mode + "'");
            cudaStreamDestroy(s);
        }
        catch (const std::exception& e)
        {
            std::printf("FAIL %s %s rank %d: %s\n", backend.c_str(), mode.c_str(), Pstream::myProcNo(), e.what());
            std::fflush(stdout);
            rc = 1;
        }
        if (rc == 0 && Pstream::myProcNo() == 0)
            std::printf("PASS %s %s (np=%d): every rank saw the same abort\n", backend.c_str(), mode.c_str(), np);
        std::fflush(stdout);
    }
    Pstream::finalize();
    return rc;
}
