#pragma once
// brae::ProcComm, the OPERATOR-SIDE communication shim of multi-GPU interFoam (InterFoamStep.md WP2,
// CommSolvers.md step C interface). Every multi-GPU operator talks to this class and to nothing else; the
// transport behind it is a run-time choice:
//
//   Loopback  N virtual ranks inside ONE process on ONE GPU. A rank is a host thread with its own stream;
//             an exchange is a device-to-device copy between the ranks' staging buffers, ordered across ranks
//             with CUDA events plus a host rendezvous. Deterministic, and CAPTURABLE into a CUDA graph (below).
//   MpiHost   one process per rank under mpirun, cudaMemcpy staging through pinned host memory + MPI
//             Isend/Irecv, reductions by MPI_Allgather + a fixed-rank-order host fold. For CI and np > GPUs.
//             Host-synchronous per call, so NOT graph-capturable.
//   Nccl      one process per GPU under mpirun (CMake option BRAE_WITH_NCCL). A HaloPlan is the per-peer send/recv
//             list of one ncclGroupStart/ncclSend/ncclRecv/ncclGroupEnd on the op's stream, reductions are
//             "allgather the n partials, fold locally in rank order" (identical bits on every rank, independent of
//             NCCL's algorithm), and CommChannel selects the one of the two communicators (graph-only with
//             graphUsageMode=0 / eager) an op uses. Every op is CUDA-graph capturable (each rank captures its OWN
//             graph, no joint graph); host-resident ops and aborts go over MPI (see proc_comm.cu).
//
// ---- semantics every backend keeps ---------------------------------------------------------------------------
//  * np == 1 (size() == 1) is the IDENTITY with zero overhead: every op returns on its first line, before any
//    allocation, stream/event call or sync. Single-GPU code passes ProcComm::current() == nullptr and never
//    reaches this header at all (`if (auto* pc = ProcComm::current()) ...`).
//  * Every op is a COLLECTIVE: all ranks call the same ops in the same order with the same sizes (SPMD). Device
//    ops only ENQUEUE on `stream`: no host sync, no allocation (except where stated), so they are ordered with
//    the caller's kernels on that stream. The host-resident variants block.
//  * Reductions are deterministic: the n partials are combined in rank order 0..np-1, so every rank gets the SAME
//    BITS, independent of timing and of the number of repeats. Sum is ((v0+v1)+v2)+..., Min/Max are exact.
//  * (NCCL/MPI backends) device ops only enqueue, so a remote abort surfaces at the next HOST collective (barrier,
//    host reduction/bcast, makePlan, exchangeIntLists); loopback and MPI-host also check inside every op.
//  * A rank-local exception must not deadlock the others: route it through collectiveAbort(msg) (or guarded()).
//    Every rank then throws the SAME ProcCommAbort at its next collective. An exception that leaves a rank
//    WITHOUT going through collectiveAbort hangs the other ranks (this is the SPMD contract of MPI as well).
//
// ---- halo storage convention (InterFoamStep.md section 7, hybrid interface) ---------------------------------
//  Cell arrays have length nC + nHalo. Halo slot s (0 <= s < nHalo) lives at index nC + s. The halo slots of one
//  neighbour rank are one CONTIGUOUS range. A HaloPlan lists, per neighbour rank, the owned cells to SEND (in the
//  order the neighbour's slots expect them) and the slot range to RECEIVE into. Processor faces stay the LAST
//  boundary faces of the mesh; a plan also carries them (faceCells, faceToSlot) so an operator can read the
//  neighbour cell of processor face i at index nC + faceToSlot[i] (= dFaceHaloCell()[i]).
//
// ---- CUDA graphs on the loopback backend ---------------------------------------------------------------------
//  Per-rank independent captures cannot synchronise through events (an event must be recorded in the SAME capture
//  that waits on it). So the loopback backend is captured as ONE joint graph for all ranks: rank 0 begins a
//  capture on its stream and forks the other ranks' streams into it with an event (cudaStreamWaitEvent), every
//  rank thread then runs its normal SPMD code on its own stream (the host rendezvous happens only while
//  capturing), the other streams join back into stream 0 with events, rank 0 ends the capture. One launch of that
//  graph replays all ranks. Warm up (one eager call of each op) before capturing: plan and slot buffers are
//  allocated at makePlan()/create time and NEVER inside an op, so no op allocates in a capture. The graph bakes
//  the device pointers passed to the ops: keep them stable. MPI-host ops refuse (throw) inside a capture.
#include "cf_pstream.cuh"      // ReduceOp { Sum, Min, Max }

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

class ProcComm;

// The error every rank throws at its next collective after any rank called collectiveAbort().
class ProcCommAbort : public std::runtime_error
{
public:
    ProcCommAbort(const std::string& msg, int originRank)
    : std::runtime_error(msg), origin_(originRank) {}
    int originRank() const { return origin_; }   // the (lowest) rank that raised the abort
private:
    int origin_;
};

// How exchangeFaces() combines my value with the neighbour's value of the SAME face (face i of my patch pairs with
// face i of the neighbour's patch, OpenFOAM's matching processor-patch order). Both sides compute the same result
// for the symmetric ops: Min/Max/Sum are commutative bit for bit.
enum class FaceOp { Replace, Min, Max, Sum };

// Which communicator an op is issued on. Eager = setup, host-driven paths, function objects. Graph = ops that are
// captured in a CUDA graph / conditional-WHILE body. NCCL needs two communicators (graph-only with
// graphUsageMode = 0, NCCL_GRAPH_MIXING_SUPPORT = 0); Loopback and MpiHost ignore it.
enum class CommChannel { Eager, Graph };

// ------------------------------------------------------------------------------------------------------------
// Halo plan: pure host description (what the next steps build from decomposePar meshes) ...
// ------------------------------------------------------------------------------------------------------------
struct HaloNeighbour
{
    int rank = -1;
    int sendOffset = 0, sendCount = 0;   // range of HaloPlanDesc::sendCells sent to this neighbour
    int recvStart = 0,  recvCount = 0;   // halo slot range [recvStart, recvStart+recvCount) filled by this neighbour
    int faceStart = 0,  faceCount = 0;   // processor-face range [faceStart, faceStart+faceCount) shared with it
};

struct HaloPlanDesc
{
    int nCells = 0;                      // owned cells (nC)
    int nHalo = 0;                       // halo slots; cell arrays have length nC + nHalo
    int nProcFaces = 0;                  // processor faces, grouped by neighbour in `nbrs` order
    std::vector<HaloNeighbour> nbrs;     // ascending patch order; one entry per neighbour rank
    std::vector<int> sendCells;          // owned cells to send, grouped by neighbour; the neighbour's slot k of this
                                         // group is sendCells[sendOffset + k]
    std::vector<int> faceCells;          // [nProcFaces] owned cell of processor face i
    std::vector<int> faceToSlot;         // [nProcFaces] halo slot (0..nHalo-1) of the neighbour cell of face i

    int nExt() const { return nCells + nHalo; }
    // Internal consistency (ranges contiguous and in bounds, ids in range). Throws std::invalid_argument.
    void validate() const;
};

// One processor patch as decomposePar writes it: the neighbour rank, my cell of every face (patch face order) and
// the NEIGHBOUR's cell of every face (its local cell ids; face i of my patch pairs with face i of its patch).
// nbrFaceCells is what a rank cannot know locally: fetch it with ProcComm::buildPlanFromPatches, or fill it
// yourself when all ranks live in one process.
struct ProcessorPatchDesc
{
    int nbrRank = -1;
    std::vector<int> faceCells;          // my owner cell per face
    std::vector<int> nbrFaceCells;       // the neighbour's owner cell per face (same length)
};

// Build the plan of one rank from its processor patches, in patch order (one patch per neighbour rank). The
// halo slots of a patch are the DISTINCT neighbour cells in order of first appearance over the patch faces; the
// send list is the distinct own cells in order of first appearance. Both sides apply the same rule to the same
// face order, so slot k of the receiver is element k of the sender's list, with no further negotiation.
HaloPlanDesc buildHaloPlanFromProcessorPatches(int nCells, const std::vector<ProcessorPatchDesc>& patches);

// ... and its device realisation, created by ProcComm::makePlan (a collective: same order on every rank).
// Owns the persistent staging memory of its backend. Destroy plans before the ProcComms that made them, and (loopback)
// destroy a plan only after every rank has finished its last exchange on it.
struct PlanState { virtual ~PlanState() = default; };   // backend-private staging (defined in proc_comm.cu)
namespace detail { struct PlanAccess; }

class HaloPlan
{
public:
    ~HaloPlan();
    HaloPlan(const HaloPlan&) = delete;
    HaloPlan& operator=(const HaloPlan&) = delete;

    const HaloPlanDesc& desc() const { return desc_; }
    int nCells() const { return desc_.nCells; }
    int nHalo() const { return desc_.nHalo; }
    int nExt() const { return desc_.nCells + desc_.nHalo; }
    int nProcFaces() const { return desc_.nProcFaces; }
    int nNeighbours() const { return static_cast<int>(desc_.nbrs.size()); }

    // Device copies for the operators' kernels (null when the array is empty, e.g. np == 1).
    const int* dSendCells() const { return dSendCells_; }
    const int* dFaceCells() const { return dFaceCells_; }       // owned cell of processor face i
    const int* dFaceHaloCell() const { return dFaceHalo_; }     // nC + slot: where the neighbour cell of face i lives

private:
    friend class ProcComm;
    friend struct detail::PlanAccess;
    HaloPlan(const ProcComm* owner, HaloPlanDesc d);
    const ProcComm* owner_;
    HaloPlanDesc desc_;
    int* dSendCells_ = nullptr;
    int* dFaceCells_ = nullptr;
    int* dFaceHalo_ = nullptr;
    std::unique_ptr<PlanState> state_;
};

// A field handed to exchangeCells: nComp components, component k of cell c at
//   data[k*compStride + c*cellStride]            (length of each component >= nC + nHalo)
// Scalar: {p}. SoA packed vector/tensor: {p, nComp, stride}. AoS (cell-major): {p, nComp, 1, nComp}.
struct HaloField
{
    double* data = nullptr;
    int nComp = 1;
    long long compStride = 0;
    long long cellStride = 1;
};

// ------------------------------------------------------------------------------------------------------------
class ProcComm
{
public:
    enum class Backend { Loopback, MpiHost, Nccl };

    virtual ~ProcComm();
    ProcComm(const ProcComm&) = delete;
    ProcComm& operator=(const ProcComm&) = delete;

    // ---- process-wide handle: the hooks find the communicator here (null = single GPU, no communication) ------
    // current() returns the thread-local handle if one is set, else the process-wide one. Loopback rank threads
    // call bindThread() (thread-local); an MPI process calls setCurrent() once.
    static ProcComm* current();
    static void setCurrent(ProcComm* c);          // process-wide; pass nullptr to go back to single GPU
    static void setThreadCurrent(ProcComm* c);    // this thread only (overrides the process-wide one)

    // ---- creation ------------------------------------------------------------------------------------------
    // Loopback: np rank handles of ONE group; hand handle r to rank thread r. reduceCapacity = largest n
    // (doubles) of a device-resident allReduce/bcastDevice. np must be 1..64.
    static std::vector<std::unique_ptr<ProcComm>> createLoopback(int np, std::size_t reduceCapacity = 4096);
    // MPI-host: this process's rank handle; needs Pstream::init() (uses MPI_COMM_WORLD, Pstream rank/size).
    static std::unique_ptr<ProcComm> createMpiHost();
    // NCCL: one process per GPU under mpirun (needs Pstream::init(); GPU = Pstream::deviceId(), np <= GPUs per node).
    // Throws std::runtime_error when this build has no NCCL (ncclBuilt() == false) or two ranks share a GPU.
    static std::unique_ptr<ProcComm> createNccl();
    static bool ncclBuilt();                       // compiled with BRAE_WITH_NCCL
    static const char* backendName(Backend b);     // "loopback" | "mpi" | "nccl"
    // Run-time selection: backend name "loopback" | "mpi" | "nccl" | "auto". Default: env BRAE_COMM, then env
    // BRAE_COMM_BACKEND, then auto = NCCL when built with it, np > 1 and every rank of a node can own a GPU, else mpi.
    // Loopback returns np handles (np: argument, else env BRAE_COMM_NP, else 1); the others return one.
    static std::vector<std::unique_ptr<ProcComm>> create(const std::string& backend = "", int np = 0);

    // Binds the calling thread to this rank: cudaSetDevice(device()) and setThreadCurrent(this).
    void bindThread();

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool isMaster() const { return rank_ == 0; }
    Backend backend() const { return backend_; }
    int device() const { return device_; }
    // True when ops may be enqueued inside a CUDA-graph capture (loopback: via the joint capture above).
    virtual bool graphCapturable() const = 0;

    // ---- plans ----------------------------------------------------------------------------------------------
    // COLLECTIVE. Allocates the persistent staging for up to maxComps scalar components per exchangeCells.
    // np == 1: returns a plan with no neighbours and allocates nothing (desc must have no neighbours).
    std::unique_ptr<HaloPlan> makePlan(HaloPlanDesc desc, int maxComps = 32);
    // Fetches the neighbours' face cells (host collective), then buildHaloPlanFromProcessorPatches + makePlan.
    // `patches[k].nbrFaceCells` is filled in.
    std::unique_ptr<HaloPlan> buildPlanFromPatches(int nCells, std::vector<ProcessorPatchDesc> patches, int maxComps = 32);

    // ---- halo exchange of cell fields (refresh the halo tails; owned values are untouched) ----------------------
    void exchangeCells(const HaloPlan& plan, const HaloField* fields, int nFields, cudaStream_t stream,
                       CommChannel ch = CommChannel::Eager)
    {
        if (size_ == 1) return;
        checkPlan(plan);
        doExchangeCells(plan, fields, nFields, stream, ch);
    }
    void exchangeCells(const HaloPlan& plan, double* scalar, cudaStream_t stream, CommChannel ch = CommChannel::Eager)
    {
        const HaloField f{scalar, 1, 0, 1};
        exchangeCells(plan, &f, 1, stream, ch);
    }
    // A batch of scalar fields in ONE exchange.
    void exchangeCells(const HaloPlan& plan, std::initializer_list<double*> scalars, cudaStream_t stream,
                       CommChannel ch = CommChannel::Eager);
    // A vector stored as 3 component arrays.
    void exchangeVector(const HaloPlan& plan, double* x, double* y, double* z, cudaStream_t stream,
                        CommChannel ch = CommChannel::Eager);
    // nComp packed components, component k at data + k*compStride (SoA).
    void exchangeCellsPacked(const HaloPlan& plan, double* data, int nComp, long long compStride, cudaStream_t stream,
                             CommChannel ch = CommChannel::Eager);
    // nComp packed components, cell-major (AoS): component k of cell c at data[c*nComp + k].
    void exchangeCellsAoS(const HaloPlan& plan, double* data, int nComp, cudaStream_t stream,
                          CommChannel ch = CommChannel::Eager);

    // ---- per-processor-face exchange (MULES lambda) ---------------------------------------------------------
    // arr[0..nProcFaces) is a device array over the plan's processor faces (the processor part of a boundary-face
    // array). arr[i] = op(arr[i], neighbour's arr[i]).
    void exchangeFaces(const HaloPlan& plan, double* arr, FaceOp op, cudaStream_t stream,
                       CommChannel ch = CommChannel::Eager)
    {
        if (size_ == 1) return;
        checkPlan(plan);
        requireDeviceMemory(arr, "exchangeFaces");
        doExchangeFaces(plan, arr, op, stream, ch);
    }

    // THE DEVICE-RESIDENT ENTRY POINTS TAKE GPU MEMORY. Ordinary (pageable) host memory handed to one of them works
    // where the driver lets the GPU read pageable memory (HMM: the laptop's) and fails everywhere else -- MEASURED on
    // the V100 nodes (driver 535): the MPI-host copy "D2H: invalid argument", NCCL an illegal memory access and a hang,
    // from one host std::vector given to allReduce. Refused here by name on every machine; allReduceHost takes host data.
    static void requireDeviceMemory(const void* p, const char* what)
    {
        cudaPointerAttributes a{};
        const cudaError_t e = cudaPointerGetAttributes(&a, p);
        if (e != cudaSuccess) (void)cudaGetLastError();
        if (e != cudaSuccess || a.type == cudaMemoryTypeUnregistered)
            throw std::invalid_argument(std::string("ProcComm::") + what + ": the data is not GPU memory "
                                        "(ordinary host memory; the host-resident overload takes that)");
    }

    // ---- collectives ----------------------------------------------------------------------------------------
    // Device-resident, in place: dData[0..n) is replaced by the rank-ordered reduction of all ranks' dData. Only
    // enqueues on `stream`; the result never visits the host. n <= the loopback reduceCapacity.
    void allReduce(double* dData, int n, ReduceOp op, cudaStream_t stream, CommChannel ch = CommChannel::Eager)
    {
        if (size_ == 1) return;
        if (n > 0) requireDeviceMemory(dData, "allReduce (device-resident)");
        doAllReduceDevice(dData, n, op, stream, ch);
    }
    // Host-resident, in place, blocking.
    void allReduceHost(double* hData, int n, ReduceOp op)
    {
        if (size_ == 1) return;
        doAllReduceHost(hData, n, op);
    }
    double allReduce(double v, ReduceOp op) { allReduceHost(&v, 1, op); return v; }

    // Broadcast from `root`: device-resident (enqueued) and host (blocking).
    void bcastDevice(double* dData, int n, int root, cudaStream_t stream, CommChannel ch = CommChannel::Eager)
    {
        if (size_ == 1) return;
        if (n > 0) requireDeviceMemory(dData, "bcastDevice");
        doBcastDevice(dData, n, root, stream, ch);
    }
    void bcastHost(double* hData, int n, int root)
    {
        if (size_ == 1) return;
        doBcastHost(hData, n, root);
    }
    double bcastScalar(double v, int root) { bcastHost(&v, 1, root); return v; }

    // Host rendezvous of all ranks (also an abort checkpoint). No device effect.
    void barrier() { if (size_ == 1) return; doBarrier(); }

    // Allgather of one integer vector per rank (host, blocking, bounded): agrees on processor-patch
    // participation before any point-to-point exchange. Loopback/np==1: recv = send.
    virtual void allgatherInts(const std::vector<int>& send, std::vector<int>& recv) { recv = send; }

    // Neighbour exchange of integer lists (host, blocking): recv[k] = list sent by nbrRanks[k]. Used to learn the
    // neighbour's face order (ProcessorPatchDesc::nbrFaceCells).
    std::vector<std::vector<int>> exchangeIntLists(const std::vector<int>& nbrRanks,
                                                   const std::vector<std::vector<int>>& send);
    // Neighbour exchange of double lists (host, blocking), BIT-EXACT: recv[k] = list sent by nbrRanks[k]. Setup-time
    // data of a processor patch (the neighbour's Cf, Sf and cell centre per face). Carried by exchangeIntLists as
    // int pairs, so every backend has it and none had to change.
    std::vector<std::vector<double>> exchangeFaceData(const std::vector<int>& nbrRanks,
                                                      const std::vector<std::vector<double>>& send);

    // ---- collective error handling --------------------------------------------------------------------------
    // Records the abort (the first one wins; the lowest raising rank on the MPI backend), then throws
    // ProcCommAbort(msg) on THIS rank. Every other rank throws the same ProcCommAbort at its next collective
    // (barrier, exchange, reduction, bcast, makePlan). np == 1: throws std::runtime_error(msg) directly.
    [[noreturn]] void collectiveAbort(const std::string& msg);
    // True once any rank aborted (loopback; sticky). The MPI backend learns only inside collectives.
    virtual bool abortPending() const { return false; }
    // Runs f(); a std::exception that is not already a ProcCommAbort becomes a collectiveAbort(e.what()).
    template<class F> void guarded(F&& f)
    {
        try { f(); }
        catch (const ProcCommAbort&) { throw; }
        catch (const std::exception& e) { collectiveAbort(e.what()); }
    }

    // Counts the device/pinned allocations this layer made (plans, slots, staging). For the "np == 1 allocates
    // nothing" gate.
    static std::size_t allocationCount();

protected:
    ProcComm(Backend b, int rank, int size);

    virtual std::unique_ptr<HaloPlan> doMakePlan(HaloPlanDesc desc, int maxComps) = 0;
    virtual void doExchangeCells(const HaloPlan&, const HaloField*, int, cudaStream_t, CommChannel) = 0;
    virtual void doExchangeFaces(const HaloPlan&, double*, FaceOp, cudaStream_t, CommChannel) = 0;
    virtual void doAllReduceDevice(double*, int, ReduceOp, cudaStream_t, CommChannel) = 0;
    virtual void doAllReduceHost(double*, int, ReduceOp) = 0;
    virtual void doBcastDevice(double*, int, int, cudaStream_t, CommChannel) = 0;
    virtual void doBcastHost(double*, int, int) = 0;
    virtual void doBarrier() = 0;
    virtual std::vector<std::vector<int>> doExchangeInts(const std::vector<int>&, const std::vector<std::vector<int>>&) = 0;
    [[noreturn]] virtual void doAbort(const std::string& msg) = 0;

    // Shared plan plumbing for the backends.
    std::unique_ptr<HaloPlan> newPlan(HaloPlanDesc desc);           // validates + uploads the index arrays
    static PlanState* planState(const HaloPlan& p);
    static void setPlanState(HaloPlan& p, std::unique_ptr<PlanState> s);
    void checkPlan(const HaloPlan& p) const;

    int rank_, size_, device_ = -1;
    Backend backend_;
};

// ------------------------------------------------------------------------------------------------------------
// Synthetic partitioner: split a structured nx*ny*nz box of cells among ranks and build every rank's patches and
// plan, with no OpenFOAM involved (tests, loopback bring-up). Cell (i,j,k) has global id i + nx*(j + ny*k); a
// rank's local cells are its global ids in ascending order; face (g, g+step) is shared by 6-connectivity; the
// faces of a rank pair are ordered by (lower global id, direction), identical on both sides.
// ------------------------------------------------------------------------------------------------------------
struct SyntheticPartition
{
    int np = 0, nx = 0, ny = 0, nz = 0;
    std::vector<int> owner;                                    // [nx*ny*nz] rank of each global cell
    std::vector<std::vector<int>> cellGlobal;                  // [rank][local cell] -> global id
    std::vector<std::vector<int>> haloGlobal;                  // [rank][halo slot] -> global id of that neighbour cell
    std::vector<std::vector<long long>> faceKey;               // [rank][processor face] -> id equal on both sides
    std::vector<std::vector<ProcessorPatchDesc>> patches;      // [rank][patch]
    std::vector<HaloPlanDesc> descs;                           // [rank]
};

// owner[] -> partition. Every rank in 0..np-1 must own at least one cell.
SyntheticPartition partitionBox(int nx, int ny, int nz, int np, const std::vector<int>& owner);
// Owner map of a px*py*pz block decomposition (near-equal blocks along each axis); np = px*py*pz,
// rank = bi + px*(bj + py*bk).
std::vector<int> blockOwner(int nx, int ny, int nz, int px, int py, int pz);
// Owner map of cubes of side `chunk` dealt to ranks by a hash of the cube coordinates: ragged interfaces in which
// one cell has several faces to the same neighbour rank (exercises the distinct-slot rule). Every rank gets cells
// when (nx/chunk)*(ny/chunk)*(nz/chunk) >= 2*np.
std::vector<int> scatteredOwner(int nx, int ny, int nz, int np, int chunk);

} // namespace brae
