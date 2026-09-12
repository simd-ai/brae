#pragma once
// cf GPU offload (G1): device lduMatrix + SpMV (Amul). Atomic-free, deterministic per-CELL gather,
// exactly the OpenFOAM lduMatrix::Amul traversal:
//   Apsi[c] = diag[c]*psi[c]
//           + sum over faces OWNED by c    : upper[f]*psi[neighbour[f]]   (faces ordered by owner -> ownerStart)
//           + sum over faces NEIGHBOURing c : lower[f]*psi[owner[f]]      (faces sorted by neighbour -> losort)
// One thread per cell, no write races (the scatter is turned into a gather), reproducible order.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include <vector>

namespace brae {

// A non-owning view of an lduMatrix on device (raw pointers): lets the G4-assembled coefficients reuse a
// DeviceMesh's addressing without copying, and lets a self-contained DeviceLduMatrix expose the same shape.
struct DeviceLduView
{
    int nCells = 0, nInternalFaces = 0;
    const scalar* diag = nullptr;
    const scalar* upper = nullptr;
    const scalar* lower = nullptr;
    const label*  owner = nullptr;
    const label* nei = nullptr;
    const label*  ownerStart = nullptr;
    const label* losort = nullptr;
    const label* losortStart = nullptr;
    // optional cyclic (periodic) interface, OpenFOAM cyclicFvPatchField::updateInterfaceMatrix. Applied in
    // deviceAmul as Apsi[cycOwn[j]] += cycCoeff[j]*psi[cycNbr[j]] (the diagonal part is folded into `diag`).
    int nCyc = 0;
    const label* cycOwn = nullptr;
    const label* cycNbr = nullptr;
    const scalar* cycCoeff = nullptr;
    // optional cyclicAMI interface, OF cyclicAMIFvPatchField::updateInterfaceMatrix. Applied in deviceAmul as
    // Apsi[amiOwn[i]] += amiIfc[i] * sum_{k in [amiOff[i],amiOff[i+1])} amiW[k]*psi[amiNbr[k]] (weighted stencil).
    int nAmi = 0;
    const label* amiOwn = nullptr;
    const label* amiOff = nullptr;
    const label* amiNbr = nullptr;
    const scalar* amiW = nullptr;
    const scalar* amiIfc = nullptr;
};

struct DeviceLduMatrix
{
    int nCells = 0, nFaces = 0;
    DeviceBuffer<scalar> diag, upper, lower;
    DeviceBuffer<label>  owner, nei;
    DeviceBuffer<label>  ownerStart, losort, losortStart;

    DeviceLduView view() const
    {
        return {nCells, nFaces, diag.data(), upper.data(), lower.data(), owner.data(), nei.data(),
                ownerStart.data(), losort.data(), losortStart.data()};
    }
};

// Build the device matrix from host lduMatrix arrays. owner/neighbour are upper-triangular ordered
// (faces sorted by owner) as in cf's mesh, so ownerStart is a plain prefix sum; losort is a counting
// sort of the faces by neighbour cell.
inline DeviceLduMatrix buildDeviceLdu(
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower,
    const std::vector<label>& owner,
    const std::vector<label>& nei,
    int nCells)
{
    const int nF = static_cast<int>(owner.size());
    std::vector<label> ownerStart(nCells + 1, 0), losortStart(nCells + 1, 0), losort(nF);
    for (int f = 0; f < nF; ++f)
    {
        ownerStart[owner[f] + 1]++;
        losortStart[nei[f] + 1]++;
    }
    for (int c = 0; c < nCells; ++c)
    {
        ownerStart[c + 1] += ownerStart[c];
        losortStart[c + 1] += losortStart[c];
    }
    std::vector<label> pos(losortStart.begin(), losortStart.end());
    for (int f = 0; f < nF; ++f)
        losort[pos[nei[f]]++] = f;

    DeviceLduMatrix A;
    A.nCells = nCells;
    A.nFaces = nF;
    A.diag.copyFrom(diag);
    A.upper.copyFrom(upper);
    A.lower.copyFrom(lower);
    A.owner.copyFrom(owner);
    A.nei.copyFrom(nei);
    A.ownerStart.copyFrom(ownerStart);
    A.losort.copyFrom(losort);
    A.losortStart.copyFrom(losortStart);
    return A;
}

// A view over a DeviceMesh's addressing + externally-assembled (G4) coefficients, no copy.
inline DeviceLduView deviceLduView(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& diag,
    const DeviceBuffer<scalar>& upper,
    const DeviceBuffer<scalar>& lower)
{
    return {dm.nCells, dm.nInternalFaces, diag.data(), upper.data(), lower.data(), dm.owner.data(), dm.nei.data(),
            dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), 0, nullptr, nullptr, nullptr};
}
// Same view, augmented with a cyclic interface (cycOwn/cycNbr/cycCoeff over nCyc periodic faces). deviceAmul
// applies the interface off-diagonal; the matching diagonal contribution must already be folded into `diag`.
inline DeviceLduView deviceLduViewCyclic(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& diag,
    const DeviceBuffer<scalar>& upper,
    const DeviceBuffer<scalar>& lower,
    int nCyc,
    const label* cycOwn,
    const label* cycNbr,
    const scalar* cycCoeff)
{
    return {dm.nCells, dm.nInternalFaces, diag.data(), upper.data(), lower.data(), dm.owner.data(), dm.nei.data(),
            dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), nCyc, cycOwn, cycNbr, cycCoeff};
}
// BOTH interfaces at once. A mesh may carry cyclic AND cyclicAMI patches -- pimpleFoam/RAS/
// oscillatingInletPeriodicAMI2D has a y-periodic `cyclic` pair on the sliding channel and a
// cyclicPeriodicAMI joining it to the duct -- and the view has always had room for both. Picking one
// with a ternary (`hasCyclic_ ? cyclicView : amiView`) silently DROPS the other's off-diagonal while
// leaving its diagonal folded into `diag` and its flux in the RHS, which is not a lossy approximation
// but an inconsistent linear system: the pressure solve on that case ran to its 50-iteration cap every
// time with the final residual ABOVE the initial one (1.00 -> 5.14), and continuity never closed --
// contGlobal pinned at -0.33 for the whole run, 0.1 of mass in and 0.013 out.
inline DeviceLduView deviceLduViewCyclicAmi(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& diag,
    const DeviceBuffer<scalar>& upper,
    const DeviceBuffer<scalar>& lower,
    int nCyc,
    const label* cycOwn,
    const label* cycNbr,
    const scalar* cycCoeff,
    int nAmi,
    const label* amiOwn,
    const label* amiOff,
    const label* amiNbr,
    const scalar* amiW,
    const scalar* amiIfc)
{
    DeviceLduView v = deviceLduViewCyclic(dm, diag, upper, lower, nCyc, cycOwn, cycNbr, cycCoeff);
    v.nAmi = nAmi;
    v.amiOwn = amiOwn;
    v.amiOff = amiOff;
    v.amiNbr = amiNbr;
    v.amiW = amiW;
    v.amiIfc = amiIfc;
    return v;
}
// Same view, augmented with a cyclicAMI interface (the weighted CSR stencil over nAmi source faces). deviceAmul
// applies the weighted off-diagonal; the matching diagonal contribution must already be folded into `diag`.
inline DeviceLduView deviceLduViewAmi(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& diag,
    const DeviceBuffer<scalar>& upper,
    const DeviceBuffer<scalar>& lower,
    int nAmi,
    const label* amiOwn,
    const label* amiOff,
    const label* amiNbr,
    const scalar* amiW,
    const scalar* amiIfc)
{
    DeviceLduView v = deviceLduView(dm, diag, upper, lower);
    v.nAmi = nAmi;
    v.amiOwn = amiOwn;
    v.amiOff = amiOff;
    v.amiNbr = amiNbr;
    v.amiW = amiW;
    v.amiIfc = amiIfc;
    return v;
}

void deviceAmul(const DeviceLduView& A, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi);

class DeviceHalo;      // forward (parallel/pstream/device_halo.cuh)
struct DistributedAMI; // forward (cuda/distributed_ami.cuh) -- optional cyclicAMI coupling in the matvec

// Distributed matrix-vector product: the local cell-gather Amul plus the processor-interface coupling over the
// NVSHMEM halo (post exchange -> local product overlaps it -> wait -> interface scatter). ifaceCoeffs[i] holds
// interface i's boundary coefficients (-upper on the owner side / -lower on the neighbour side), in the same
// order as `halo`'s interfaces. Mirrors host parallelAmul.
//   `ami` (optional): a cyclicAMI interface. When non-null, after the halo interface scatter the AMI target cells are
//   gathered on-GPU (NVSHMEM) and deviceAmiAmul adds their weighted contribution -> Apsi carries the AMI coupling
//   too, so the whole implicit solve sees it. Null (default) -> plain distributed matvec, unchanged.
void deviceParallelAmul(
    const DeviceLduView& A,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& psi,
    DeviceBuffer<scalar>& Apsi,
    const DistributedAMI* ami = nullptr);

} // namespace brae
