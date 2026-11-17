#pragma once
// brae::DeviceProcInterface, the multi-GPU interface of one rank's device mesh (InterFoamStep.md WP1/WP4).
//
// A processor patch lives TWICE on the device, on purpose:
//   * as the LAST boundary faces of DeviceMesh (bcType 8, DeviceBoundary), so that the sum-class
//     operators (div, grad, reconstruct, Courant) add them in OpenFOAM's face order through the boundary-value
//     slot: processor face j is boundary face dm.nBndFacesPhys + j;
//   * as `cyc`, a DeviceCyclic (the translational interface kernels, the per-owner-cell CSR with its ifRank sum
//     order, dOwn/dNbr/delta/corrVec) whose neighbour index addresses HALO SLOTS: nbrCell[j] = nC + slot, so every
//     kernel that reads cyclicNbrValue(nbr, x, j) finds the neighbour rank's cell once the halo tail of x was
//     refreshed with ProcComm::exchangeCells(plan, x). twin = -1 everywhere: MULES' lambda sync across a rank
//     boundary is ProcComm::exchangeFaces over the processor part of the boundary-face array.
// Cell arrays that processor operators read have length dm.nCellsExt = nC + nHalo (the halo-extended storage is the
// operator step's work; this struct and the mesh only fix the numbering).
//
// Nothing here exists without processor faces: buildDeviceProcInterface returns an empty interface (n == 0,
// nHalo == 0) and allocates nothing then.
#include "device_cyclic.cuh"
#include "device_mesh.cuh"
#include "processor_coupling.cuh"

#include <stdexcept>
#include <string>

namespace brae {

struct DeviceProcInterface
{
    DeviceCyclic cyc;                  // the interface faces of every processor patch, neighbour = halo slot
    int nProcFaces = 0;                // = cyc.n = dm.nBndFaces - dm.nBndFacesPhys
    int nHalo = 0;                     // halo slots after the nC owned cells
    int nCellsExt = 0;                 // nC + nHalo
    int bndStart = 0;                  // dm.nBndFacesPhys: processor face j is boundary face bndStart + j
    const HaloPlan* plan = nullptr;    // the plan of this rank (owned by the caller's ProcessorCoupling)
    bool empty() const { return nProcFaces == 0; }
};

// Build the interface of this rank from the coupled processor patches and their plan, and check that what the
// device mesh and the plan say agree: the boundary suffix of `dm` is exactly the plan's processor faces (same
// owner cells, same count) and the neighbour index of every face is nC + the plan's slot of it.
inline DeviceProcInterface buildDeviceProcInterface(
    const DeviceMesh& dm,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const ProcessorCoupling& pc)
{
    DeviceProcInterface out;
    out.plan = pc.plan.get();
    const std::vector<CyclicInterface> ifs = buildProcessorInterfaces(g, fvp);
    std::size_t n = 0;
    for (const CyclicInterface& c : ifs) n += c.faceCells.size();
    if (static_cast<label>(n) != pc.nProcFaces || dm.nHalo != pc.nHalo
     || dm.nBndFaces - dm.nBndFacesPhys != static_cast<int>(n))
    {
        throw std::runtime_error(
            "brae: the processor interface disagrees with the plan and the device mesh: " + std::to_string(n)
            + " interface faces, plan " + std::to_string(pc.nProcFaces) + ", mesh boundary suffix "
            + std::to_string(dm.nBndFaces - dm.nBndFacesPhys) + "; halo " + std::to_string(dm.nHalo) + " vs plan "
            + std::to_string(pc.nHalo) + ".");
    }
    if (n == 0)
    {
        return out;
    }
    const HaloPlanDesc& pd = pc.plan->desc();
    std::size_t j = 0;
    for (const CyclicInterface& c : ifs)
    {
        for (std::size_t i = 0; i < c.faceCells.size(); ++i, ++j)
        {
            if (c.faceCells[i] != pd.faceCells[j] || c.nbrFaceCells[i] != dm.nCells + pd.faceToSlot[j])
            {
                throw std::runtime_error(
                    "brae: processor face " + std::to_string(j) + " of patch " + std::to_string(c.patch)
                    + " is not where the halo plan has it (cell " + std::to_string(c.faceCells[i]) + ", halo "
                    + std::to_string(c.nbrFaceCells[i]) + ").");
            }
        }
    }
    out.cyc = buildDeviceCyclic(ifs, g, fvp);
    out.nProcFaces = static_cast<int>(n);
    out.nHalo = dm.nHalo;
    out.nCellsExt = dm.nCellsExt;
    out.bndStart = dm.nBndFacesPhys;
    return out;
}

} // namespace brae
