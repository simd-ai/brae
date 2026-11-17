#pragma once
// The processor interface as the neighbour-list operators use it (InterFoamStep.md WP4).
//
// A processor face lives twice on a rank's device mesh (device_processor.cuh): as boundary face
// `dm.nBndFacesPhys + j` (bcType 8: the sum-class operators -- gradient, divergence, reconstruct, Courant -- take
// its value from the boundary-value slot, in OpenFOAM's face order) and as face j of `dp.cyc` (a DeviceCyclic
// whose neighbour index is the halo slot `nC + slot`: every kernel that reads the OTHER side's cell value or
// gradient). So an operator across a processor face does three things, all of them here:
//
//   1. refresh the halo tail of every cell array the neighbour kernels read       procExchange
//   2. let the interface kernel (the cyclic one, unchanged) form the face value   deviceCyclic* on ctx.cyc()
//   3. put the result in the boundary-face array's processor segment              procScatter
//      (and gather the other way, for kernels that read the pair's own `phi`)     procGather
// and the sum-class operators then see the face through the boundary array like any boundary face. The cyclic
// kernels that add the face to its owner cell (deviceCyclicAddGrad / AddDivFlux / AddToOwner) are NOT called for a
// processor face: the boundary route already sums it, and calling both would count it twice.
//
// Nothing here exists without processor faces. The context is the driver's: it publishes it with
// DeviceProcCtx::setCurrent after the interface is built, and an operator that must act across a processor face asks
// `DeviceProcCtx::active(dm)`; with no processor face (every single-GPU run, np = 1) that is false and the operator
// is its old self, instruction for instruction.
#include "device_buffer.cuh"
#include "device_cyclic.cuh"
#include "device_mesh.cuh"
#include "device_processor.cuh"
#include "proc_comm.cuh"

#include <cstdlib>
#include <initializer_list>
#include <vector>
#include <stdexcept>
#include <string>

namespace brae {

struct DeviceProcCommCounts
{
    long long cellExchanges = 0;
    long long faceExchanges = 0;
};
inline DeviceProcCommCounts& deviceProcCommCountsRef()
{
    static DeviceProcCommCounts c;
    return c;
}

class DeviceProcCtx
{
public:
    DeviceProcCtx(DeviceProcInterface& dp, ProcComm& comm)
        : dp_(&dp), comm_(&comm)
    {
        if (dp.plan == nullptr)
        {
            throw std::runtime_error("brae: a processor context needs the halo plan of its interface.");
        }
    }

    DeviceProcInterface& interface() const { return *dp_; }
    DeviceCyclic& cyc() const { return dp_->cyc; }
    ProcComm& comm() const { return *comm_; }
    int nFaces() const { return dp_->nProcFaces; }
    int bndStart() const { return dp_->bndStart; }
    int nCellsExt() const { return dp_->nCellsExt; }
    const HaloPlan& plan() const { return *dp_->plan; }
    // The distributed LDU view's plan/global-cell metadata and this interface's coefficient storage.
    // The coefficient pointer is replaced with the matrix's own saved array by MomentumMatrix::view or
    // the pressure assembly before a solve.
    DeviceProcView interfaceView() const;
    void setGlobalCells(long long n) { globalCells_ = n; }

    // the context of this thread's rank (null: no processor interface, the single-GPU run)
    static DeviceProcCtx* current() { return slot(); }
    static void setCurrent(DeviceProcCtx* c) { slot() = c; }
    // true when `dm` has processor faces AND a context publishes their interface
    static bool active(const DeviceMesh& dm)
    {
        return dm.nHalo > 0 && slot() != nullptr;
    }

private:
    static DeviceProcCtx*& slot()
    {
        static thread_local DeviceProcCtx* c = nullptr;
        return c;
    }
    DeviceProcInterface* dp_;
    ProcComm* comm_;
    long long globalCells_ = 0;
};


namespace procdetail {
inline void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string("brae processor interface: ") + what + ": " + cudaGetErrorString(e));
}
inline void needHalo(const DeviceProcCtx& c, const DeviceBuffer<scalar>& b, const char* what)
{
    if (b.capacity() < static_cast<std::size_t>(c.nCellsExt()))
    {
        throw std::runtime_error(
            std::string("brae processor interface: the cell array `") + what + "` has room for " + std::to_string(b.capacity())
            + " entries and a processor operator reads " + std::to_string(c.nCellsExt())
            + " (nCells + nHalo): it was allocated without its halo tail (cellArrayLength, device_mesh.cuh).");
    }
}
}   // namespace procdetail

inline DeviceProcView DeviceProcCtx::interfaceView() const
{
    DeviceProcView v;
    if (nFaces() == 0) return v;
    v.comm = comm_;
    v.plan = dp_->plan;
    v.nFaces = dp_->nProcFaces;
    v.faceCell = dp_->cyc.ownCell.data();
    v.nbrIdx = dp_->cyc.nbrCell.data();
    v.coeff = dp_->cyc.ifCoeff.data();
    v.rank = dp_->cyc.ifRank.data();
    if (dp_->cyc.nIfCells > 0 && !std::getenv("BRAE_CONTROL_PROC_PER_PLACE_LAUNCH"))
    {
        v.nCellsWith = dp_->cyc.nIfCells;
        v.cellList = dp_->cyc.ifCellList.data();
        v.cellStart = dp_->cyc.ifCellStart.data();
        v.perm = dp_->cyc.ifPerm.data();
    }
    v.nRanks = dp_->cyc.nIfRanks;
    v.nGlobalCells = globalCells_;
    return v;
}

// 1. REFRESH THE HALO TAILS of these cell arrays from the neighbour ranks (one batched exchange; the components of
// a vector or a gradient go together). Every array must have been allocated with its halo capacity.
inline void procExchange(
    const DeviceProcCtx& c,
    std::initializer_list<DeviceBuffer<scalar>*> fields,
    const char* what = "field")
{
    //   BRAE_CONTROL_PROC_NO_EXCHANGE=1: a gate's CONTROL, deliberately wrong -- the halo tails are never refreshed
    //   (they keep their zeros), which the stage gate against OpenFOAM must catch on every processor face
    static const bool noExchange = std::getenv("BRAE_CONTROL_PROC_NO_EXCHANGE") != nullptr;
    if (noExchange) return;
    std::vector<HaloField> hf;
    hf.reserve(fields.size());
    for (DeviceBuffer<scalar>* f : fields)
    {
        procdetail::needHalo(c, *f, what);
        hf.push_back(HaloField{f->data(), 1, 0, 1});
    }
    c.comm().exchangeCells(c.plan(), hf.data(), static_cast<int>(hf.size()), cudaStreamPerThread);
    ++deviceProcCommCountsRef().cellExchanges;
}

// 2b. syncTools::syncFaceList(minEqOp) across processor patches. The input is the interface array (one
// entry per processor face); both ranks keep the smaller limiter. A local periodic pair's twin sync is
// a separate, local operation in the MULES kernel.
inline void procSyncFacesMin(const DeviceProcCtx& c, DeviceBuffer<scalar>& ifArray)
{
    if (c.nFaces() == 0) return;
    if (ifArray.size() < static_cast<std::size_t>(c.nFaces()))
        throw std::runtime_error("brae processor interface: a face limiter has no processor segment.");
    //   BRAE_CONTROL_PROC_NO_LAMBDA_SYNC=1: a gate's CONTROL, deliberately wrong -- each rank keeps its
    //   own one-sided limiter, so the two sides of a cut face can transport different alpha fluxes.
    static const bool noSync = std::getenv("BRAE_CONTROL_PROC_NO_LAMBDA_SYNC") != nullptr;
    if (noSync) return;
    c.comm().exchangeFaces(c.plan(), ifArray.data(), FaceOp::Min, cudaStreamPerThread);
    ++deviceProcCommCountsRef().faceExchanges;
}

// 3. the interface array (one entry per processor face, the order of the processor patches) into / out of the
// processor segment of a boundary-face array of nBndFaces entries
// ...the same into a raw pointer to the first BOUNDARY face of an array that holds the internal faces first
// (`stf.data() + nIf`: the full-face layout of the surface-tension flux and snGrad(rho))
inline void procScatter(const DeviceProcCtx& c, const DeviceBuffer<scalar>& ifArray, scalar* bndBase)
{
    if (c.nFaces() == 0) return;
    procdetail::check(cudaMemcpyAsync(bndBase + c.bndStart(), ifArray.data(),
                                      static_cast<std::size_t>(c.nFaces())*sizeof(scalar),
                                      cudaMemcpyDeviceToDevice, cudaStreamPerThread), "scatter to the boundary faces");
}
inline void procScatter(const DeviceProcCtx& c, const DeviceBuffer<scalar>& ifArray, DeviceBuffer<scalar>& bnd)
{
    if (c.nFaces() == 0) return;
    if (ifArray.size() < static_cast<std::size_t>(c.nFaces())
     || bnd.size() < static_cast<std::size_t>(c.bndStart()) + static_cast<std::size_t>(c.nFaces()))
    {
        throw std::runtime_error("brae processor interface: a boundary-face array of " + std::to_string(bnd.size())
                                 + " entries has no processor segment of " + std::to_string(c.nFaces()) + " faces at "
                                 + std::to_string(c.bndStart()) + ".");
    }
    procdetail::check(cudaMemcpyAsync(bnd.data() + c.bndStart(), ifArray.data(),
                                      static_cast<std::size_t>(c.nFaces())*sizeof(scalar),
                                      cudaMemcpyDeviceToDevice, cudaStreamPerThread), "scatter to the boundary faces");
}
inline void procGather(const DeviceProcCtx& c, const DeviceBuffer<scalar>& bnd, DeviceBuffer<scalar>& ifArray)
{
    if (c.nFaces() == 0) return;
    ifArray.resize(static_cast<std::size_t>(c.nFaces()));
    if (bnd.size() < static_cast<std::size_t>(c.bndStart()) + static_cast<std::size_t>(c.nFaces()))
    {
        throw std::runtime_error("brae processor interface: a boundary-face array of " + std::to_string(bnd.size())
                                 + " entries has no processor segment of " + std::to_string(c.nFaces()) + " faces at "
                                 + std::to_string(c.bndStart()) + ".");
    }
    procdetail::check(cudaMemcpyAsync(ifArray.data(), bnd.data() + c.bndStart(),
                                      static_cast<std::size_t>(c.nFaces())*sizeof(scalar),
                                      cudaMemcpyDeviceToDevice, cudaStreamPerThread), "gather from the boundary faces");
}

// The face value of a cell field on the processor faces, w*own + (1-w)*neighbour, written into the processor
// segment of `bnd` -- the boundary-value slot of the sum-class operators (fvc::interpolate's coupled branch, which
// is what a processor patch field holds after coupledFvPatchField::evaluate). `cell`'s halo must be current.
inline void procFaceValue(const DeviceProcCtx& c, const DeviceBuffer<scalar>& cell, DeviceBuffer<scalar>& bnd)
{
    if (c.nFaces() == 0) return;
    DeviceBuffer<scalar> face;
    deviceCyclicFaceValue(c.cyc(), cell, face);
    procScatter(c, face, bnd);
}

}   // namespace brae
