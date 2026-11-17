// brae::Pstream MPI backend. Re-expressed from OpenFOAM src/Pstream/mpi semantics
// (non-blocking exchange + Allreduce); not copied verbatim.
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "cf_pstream.cuh"

#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>

namespace brae {
namespace {
    bool                     g_init    = false;
    int                      g_rank    = 0;
    int                      g_nprocs  = 1;
    int                      g_device  = -1;   // CUDA device this rank owns (round-robin over visible GPUs)
    std::vector<MPI_Request> g_requests;   // pending isend/irecv

    // Bind this rank to one GPU (nodeLocalRank % visibleGPUs), so each host's ranks map onto that host's own
    // GPUs -- correct for both single-node (localRank == g_rank) and multi-node runs. Defensive: a host-only
    // build or a CPU-only CI node has no device, so a missing/zero device count is not an error -- the binding
    // is simply skipped and any CUDA error state is cleared so later code is unaffected. On GB10 (one GPU)
    // every rank binds to device 0.
    //
    // BRAE_GPU_MAP="g0,g1,...": the topology-aware placement (tools/proc_comm_nccl/proc_topology_tool, proc_topology.cuh):
    // node-local rank i uses GPU gi. It is applied here, before any CUDA context exists. An invalid map (not
    // `localSize` distinct in-range GPU ids) aborts the job instead of silently running on the wrong GPUs.
    void bindRankToDevice(int nodeLocalRank, int localSize)
    {
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev <= 0)
        {
            cudaGetLastError();   // swallow the no-device error, leave the CUDA state clean
            g_device = -1;
            return;
        }
        g_device = nodeLocalRank % nDev;
        if (const char* m = std::getenv("BRAE_GPU_MAP"))
        {
            std::vector<int> map;
            const char* p = m;
            bool ok = *p != '\0';
            while (ok && *p)
            {
                char* end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p || v < 0 || v >= nDev) { ok = false; break; }
                for (int g : map) if (g == v) ok = false;
                map.push_back(static_cast<int>(v));
                p = end;
                if (*p == ',') ++p;
                else if (*p != '\0') ok = false;
            }
            if (!ok || static_cast<int>(map.size()) != localSize)
            {
                if (nodeLocalRank == 0)
                    std::fprintf(stderr, "brae FATAL: BRAE_GPU_MAP='%s' must list %d distinct GPU ids in 0..%d (one per rank on the node)\n",
                                 m, localSize, nDev - 1);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            g_device = map[nodeLocalRank];
        }
        cudaSetDevice(g_device);
    }

    inline MPI_Op mpiOp(ReduceOp op)
    {
        switch (op)
        {
            case ReduceOp::Sum: return MPI_SUM;
            case ReduceOp::Min: return MPI_MIN;
            case ReduceOp::Max: return MPI_MAX;
        }
        return MPI_SUM;
    }
} // namespace

void Pstream::init(int& argc, char**& argv)
{
    int already = 0;
    MPI_Initialized(&already);
    if (!already)
    {
        // MULTIPLE: the ProcComm abort side channel is a background thread that runs a nonblocking allgather on
        // its own duplicated communicator while the main thread is blocked in a device synchronisation. OpenMPI
        // provides it; with a lesser level ProcComm falls back to driving the channel from the main thread.
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    // Node-local rank/size (ranks sharing this host) -> per-node GPU binding.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localSize);
    MPI_Comm_free(&node);

    bindRankToDevice(localRank, localSize);
    g_init = true;
}

void Pstream::finalize()
{
    if (!g_requests.empty())
        waitAll();
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
    g_init = false;
}

bool Pstream::initialized() { return g_init; }
int  Pstream::myProcNo()    { return g_rank; }
int  Pstream::nProcs()      { return g_nprocs; }
bool Pstream::parRun()      { return g_nprocs > 1; }
bool Pstream::master()      { return g_rank == 0; }
int  Pstream::deviceId()    { return g_device; }

void Pstream::isend(const scalar* data, int count, int toProc, int tag)
{
    MPI_Request r;
    MPI_Isend(data, count, MPI_DOUBLE, toProc, tag, MPI_COMM_WORLD, &r);
    g_requests.push_back(r);
}

void Pstream::irecv(scalar* buf, int count, int fromProc, int tag)
{
    MPI_Request r;
    MPI_Irecv(buf, count, MPI_DOUBLE, fromProc, tag, MPI_COMM_WORLD, &r);
    g_requests.push_back(r);
}

void Pstream::waitAll()
{
    if (g_requests.empty()) return;
    MPI_Waitall(static_cast<int>(g_requests.size()),
                g_requests.data(), MPI_STATUSES_IGNORE);
    g_requests.clear();
}


scalar Pstream::allReduce(scalar v, ReduceOp op)
{
    scalar out = v;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, mpiOp(op), MPI_COMM_WORLD);
    return out;
}

label Pstream::allReduce(label v, ReduceOp op)
{
    label out = v;
    MPI_Allreduce(&v, &out, 1, MPI_INT32_T, mpiOp(op), MPI_COMM_WORLD);
    return out;
}

void Pstream::allReduce(scalar* data, int count, ReduceOp op)
{
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, mpiOp(op), MPI_COMM_WORLD);
}

void Pstream::broadcast(label* data, int count, int root)
{
    MPI_Bcast(data, count, MPI_INT32_T, root, MPI_COMM_WORLD);
}

void Pstream::barrier() { MPI_Barrier(MPI_COMM_WORLD); }

} // namespace brae
