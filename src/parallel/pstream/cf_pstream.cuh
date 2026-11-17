#pragma once
// brae::Pstream, the COMMUNICATION SEAM.
//
// CPU baseline backend = MPI (Open MPI). Mirrors OpenFOAM src/Pstream + UPstream:
// the same SPMD model, non-blocking point-to-point for halo exchange
// (Isend/Irecv/wait) plus global reductions (Allreduce).
//
// The GPU backend (intra-cluster DSM / inter-cluster global halo) will implement
// this SAME surface later; every call site, ordering and barrier stays put. Only the
// transport changes. No <mpi.h> leaks into this header on purpose, backends are swappable.
#include "cf_types.cuh"

namespace brae {

enum class ReduceOp { Sum, Min, Max };

// All-static facade over the active parallel transport.
class Pstream
{
public:
    static void init(int& argc, char**& argv);
    static void finalize();
    static bool initialized();

    static int  myProcNo();   // this partition/rank id
    static int  nProcs();     // number of partitions/ranks
    static bool parRun();     // nProcs() > 1
    static bool master();     // myProcNo() == 0
    static int  deviceId();   // CUDA device this rank is bound to in init(), or -1 if none

    // Halo transport: non-blocking scalar buffer exchange (the lduInterface uses these).
    static void isend(const scalar* data, int count, int toProc, int tag);
    static void irecv(scalar* buf,        int count, int fromProc, int tag);
    static void waitAll();    // Pstream::wait equivalent, completes all pending isend/irecv

    // Global reductions (Allreduce).
    static scalar allReduce(scalar v, ReduceOp op);
    static label  allReduce(label  v, ReduceOp op);
    static void   allReduce(scalar* data, int count, ReduceOp op);   // in-place array reduce (field gather)

    // Broadcast a label buffer from root to all ranks (e.g. the decomposition map).
    static void broadcast(label* data, int count, int root = 0);

    static void barrier();
};

} // namespace brae
