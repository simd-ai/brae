#pragma once
// The solver-side view of ProcComm: which channel the hooks use. The matrix product's halo exchange
// (device_spmv.cu) and the reductions' allReduce (reductions.cu) enqueue on the CHANNEL set here, so a solver loop that
// is captured into a CUDA graph (a conditional WHILE body) runs them on the graph communicator (CommChannel::Graph;
// NCCL's graph-only communicator, graphUsageMode = 0) and everything else stays on the eager one. Loopback and MPI-host
// ignore the channel. Thread-local, default Eager.
#include "proc_comm.cuh"
#include <cstdlib>

namespace brae {

inline CommChannel& procChannelRef()
{
    thread_local CommChannel ch = CommChannel::Eager;
    return ch;
}
inline CommChannel procChannel() { return procChannelRef(); }

// RAII: ops issued by the hooks inside the scope use `ch`.
class ProcChannelScope
{
public:
    explicit ProcChannelScope(CommChannel ch) : saved_(procChannelRef()) { procChannelRef() = ch; }
    ~ProcChannelScope() { procChannelRef() = saved_; }
    ProcChannelScope(const ProcChannelScope&) = delete;
    ProcChannelScope& operator=(const ProcChannelScope&) = delete;
private:
    CommChannel saved_;
};

// Do the distributed solver loops (AMG-PCG, symGaussSeidel) run as CUDA-graph WHILE loops with the collectives inside
// the body? Only on the NCCL backend (each rank captures its own graph; MPI-host throws inside a capture and a
// loopback body would need one joint graph); BRAE_PROC_GRAPH=0 selects the host-driven loops there too. Read at every
// solve (the tests switch it), not cached.
inline bool procGraphLoops(const ProcComm& c)
{
    const char* e = std::getenv("BRAE_PROC_GRAPH");
    if (e && e[0] == '0') return false;
    return c.backend() == ProcComm::Backend::Nccl && c.graphCapturable();
}

// REGISTRY of the CUDA graphs captured on a communicator's graph channel. A communicator's destructor destroys these
// BEFORE it destroys its NCCL communicators: a graph that outlives them makes ncclCommDestroy wait forever for the
// graph's work to be globally quiescent (measured: np=2, both ranks in commDestroySync after the solver WHILE graphs).
// Each captured cache registers a small closure (destroy its graphs) keyed by the communicator it captured on.
void procGraphRegister(const ProcComm* c, void (*destroy)(void*), void* arg);
void procGraphDestroyAll(const ProcComm* c);   // ProcComm's destructor; idempotent, safe with no graphs
void procGraphUnregister(const ProcComm* c, void* arg);   // a cache that dies before its communicator

} // namespace brae
