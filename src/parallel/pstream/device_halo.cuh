#pragma once
// brae DeviceHalo: device-resident processor halo exchange over NVSHMEM.
//
// Manages ALL of a rank's processor interfaces together (one symmetric send/recv buffer pair, one barrier) --
// the device counterpart of the host ProcessorInterface loop (init-all / wait / update-all). The transport is
// the NVSHMEM on-stream host API (nvshmemx_putmem_on_stream + nvshmemx_barrier_all_on_stream), so the pack and
// scatter are plain CUDA kernels and NOTHING needs relocatable device code. The fully device-initiated,
// pack-fused, signalled variant is a Phase-5 performance change behind this same interface.
//
// Symmetric-heap layout: the send/recv buffers are sized to the MAX total interface faces across ALL PEs, so
// the collective nvshmem_malloc is symmetric and recvBuf.data() is the same symmetric address on every PE.
// Interface i occupies [recvOffset_[i], +size_[i]). A one-time MPI offset exchange tells each neighbour where
// in our recv buffer it should write (remoteOffset_[i] is where WE write in the neighbour's buffer).
#include "cf_types.cuh"
#include "cf_pstream.cuh"
#include "sym_buffer.cuh"
#include "device_reduce.cuh"   // DeviceReducer owned here so its symmetric heap is freed with the halo (pre-finalize)
#include "device_buffer.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace brae {

class DeviceHalo
{
public:
    // Collective across all ranks. nbrParts[i] = neighbour partition of interface i; faceCells[i] = the local
    // owner cell of each of that interface's faces (one interface per neighbour partition).
    DeviceHalo(
        int myPart,
        const std::vector<int>& nbrParts,
        const std::vector<std::vector<label>>& faceCells);

    int   nInterfaces() const { return static_cast<int>(nbr_.size()); }
    int   neighbour(int i) const { return nbr_[i]; }
    label size(int i)      const { return size_[i]; }
    // Interface i's local owner cells (one per face), for callers that scatter per-interface-face data (e.g. the
    // Galerkin coarse-operator assembly: aggregate of each cut face's owner). Same order as size(i)/recvData(i).
    const label* faceCellsDev(int i) const { return faceCellsD_[i].data(); }
    // Device pointer to interface i's local owner cells (one per face). Lets callers fold the interface into their
    // own kernels (e.g. the Galerkin coarse-operator scatter) without a round-trip.
    const label* faceCells(int i) const { return faceCellsD_[i].data(); }

    // Post the exchange: pack psi at the interface faceCells and put to the neighbours on `stream` (no wait).
    // Do the local product between postExchange and waitExchange to overlap it with the transfer.
    //
    // The default stream is cudaStreamPerThread, NOT 0: brae's kernels compile with --default-stream
    // per-thread, so their implicit stream is the per-thread default; NVSHMEM (a precompiled library) would
    // read a bare 0 as the LEGACY default stream -- a different stream -- and the put could then race the pack.
    // Passing cudaStreamPerThread (0x2) makes both agree, so pack/put/local-product/barrier/scatter are ordered.
    void postExchange(
        const scalar* psi_d,
        cudaStream_t stream = cudaStreamPerThread);
    // Complete the exchange (barrier). After the stream completes, recv region i holds the neighbour values.
    void waitExchange(cudaStream_t stream = cudaStreamPerThread);
    // postExchange + waitExchange as one call (standalone use, e.g. the halo unit test).
    //
    // HAZARD -- the recv buffer is REUSED by every exchange. After waitExchange the ranks run on independently,
    // so a neighbour can post its NEXT exchange (writing our recv buffer) while we are still READING the
    // previous one. Back-to-back exchanges therefore need an intervening global sync: either a collective the
    // caller already performs (e.g. the Krylov allReduce between two products) or an explicit waitExchange()
    // AFTER the read. Exchanging several fields (e.g. the 3 components of a vector) without that sync silently
    // corrupts the earlier field.
    void exchange(
        const scalar* psi_d,
        cudaStream_t stream = cudaStreamPerThread);

    // Interface i's off-diagonal coupling: result[faceCells[i][f]] -= coeff[f] * recvNeighbour[f].
    void updateInterfaceMatrix(
        int i,
        scalar* result_d,
        const scalar* coeff_d,
        cudaStream_t stream = cudaStreamPerThread);

    // After exchange(psi_d), write the interpolated processor-face boundary values into `bval_d`, so the
    // unchanged device explicit operators (grad/div/interpolate) treat a processor face as a coupled boundary:
    //   bval[procStart[i] + f] = weights[i][f]*psi[faceCells[i][f]] + (1 - weights[i][f])*psiNeighbour[i][f]
    // weights[i] and procStart[i] are this partition's processor patches, same order as `halo`'s interfaces.
    void scatterBoundaryValues(
        const scalar* psi_d,
        const std::vector<DeviceBuffer<scalar>>& weights,
        const std::vector<label>& procStart,
        scalar* bval_d,
        cudaStream_t stream = cudaStreamPerThread);

    // Device pointer to interface i's received neighbour values (valid after exchange/waitExchange). Lets
    // callers fold the halo into their own kernels (e.g. the H() interface term) without a round-trip.
    const scalar* recvData(int i) const { return recvBuf_.data() + recvOffset_[i]; }

    // Neighbour values received for interface i (D2H; syncs `stream` first). For validation / tests.
    std::vector<scalar> neighbourField(
        int i,
        cudaStream_t stream = cudaStreamPerThread) const;

    // On-stream NVSHMEM global reducer for the distributed Krylov solvers, owned here so its symmetric heap is
    // allocated (collectively) with the halo and freed BEFORE Pstream::finalize -- a process-static reducer's
    // nvshmem_free runs after finalize and crashes ("API called before init completed").
    DeviceReducer& reducer() { return red_; }

private:
    int                              myPart_ = 0;
    std::vector<int>                 nbr_;
    std::vector<label>               size_, recvOffset_, remoteOffset_;
    std::vector<DeviceBuffer<label>> faceCellsD_;
    SymBuffer<scalar>                sendBuf_, recvBuf_;
    DeviceReducer                    red_{8};   // sized for the largest fused reduction group (BiCGStab: 2 dots)
    label                            total_ = 0;
};

} // namespace brae
