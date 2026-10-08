#pragma once
// cf GPU offload (G1): device lduMatrix + SpMV (Amul). Atomic-free, deterministic per-CELL gather, in the
// order OpenFOAM's lduMatrix::Amul face loop adds into each cell (lduMatrixATmul.C:121-135):
//   Apsi[c] = diag[c]*psi[c], then every face of c in INCREASING FACE INDEX, adding
//             upper[f]*psi[neighbour[f]] where c owns f and lower[f]*psi[owner[f]] where c neighbours it
// (the owned faces come from ownerStart, the neighbouring ones from losort; the kernel merges the two). It
// used to add every owned face and then every neighbouring one -- the same terms in another order, which
// is another last bit, and on a solve whose residual is all cancellation another stopping iteration.
// One thread per cell, no write races (the scatter is turned into a gather), reproducible order.
#include <algorithm>
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
    // A JUMP ACROSS THE PAIR (fixedJump, porousBafflePressure), ALREADY SIGNED: the owner side holds the
    // file's value and the other side its negative, as fixedJumpFvPatchField::jump() gives them. The
    // neighbour value a jump cyclic hands the matrix is psi[nbr] - jump (jumpCyclicFvPatchField.C:94-125)
    // and it is applied ONLY when the operand is the SOLUTION FIELD -- "only apply jump to original
    // field", which in a Krylov solve is the one Amul that builds the initial residual and the
    // normalisation, never the search directions. deviceAmul's `onField` says which call this is.
    // Null = no jump on this pair.
    const scalar* cycJump = nullptr;
    // optional cyclicAMI interface, OF cyclicAMIFvPatchField::updateInterfaceMatrix. Applied in deviceAmul as
    // Apsi[amiOwn[i]] += amiIfc[i] * sum_{k in [amiOff[i],amiOff[i+1])} amiW[k]*psi[amiNbr[k]] (weighted stencil).
    int nAmi = 0;
    const label* amiOwn = nullptr;
    const label* amiOff = nullptr;
    const label* amiNbr = nullptr;
    const scalar* amiW = nullptr;
    const scalar* amiIfc = nullptr;
    // The identity of owner/nei's CONTENT, from nextDeviceAddressingId (device_mesh.cuh): what the
    // topology caches (Gauss-Seidel levels, colouring, captured graphs) compare, since the pointer alone
    // is recycled by the pool. Trailing so the brace initialisers above stay valid; 0 = not stamped,
    // which every cache treats as "match on the pointer and the sizes alone", as they always did.
    unsigned long long addressingId = 0;
    // THE ORDER A CELL'S PAIR FACES ARE ADDED IN by a product (deviceAmul, deviceResidual). The interface
    // kernels add a face's term to its own cell with atomicAdd; a cell that owns several faces of the pair
    // was then summed in the order the threads arrived (DeviceCyclic::ifRank has the measurement: RAS/
    // mixerVesselAMI, 21,392 such cells). pairRank[i] is face i's place among its own cell's faces, in face
    // order, and the kernels are launched once a place -- nPairRanks launches, the faces of one place each.
    // Null and 1: one launch, every face, as before (what a view built without them gets).
    const label* pairRank = nullptr;
    int nPairRanks = 1;
};

// each entry's place among the entries of the same own cell, in list order, and the most one cell has
inline std::vector<label> pairOwnerRanks(
    const std::vector<label>& own,
    int& mostOfACell)
{
    std::vector<label> rank(own.size(), 0);
    label top = -1;
    for (const label c : own)
    {
        top = std::max(top, c);
    }
    std::vector<label> seen(static_cast<std::size_t>(top + 1), 0);
    mostOfACell = 1;
    for (std::size_t i = 0; i < own.size(); ++i)
    {
        rank[i] = seen[static_cast<std::size_t>(own[i])]++;
        mostOfACell = std::max(mostOfACell, static_cast<int>(rank[i]) + 1);
    }
    return rank;
}

struct DeviceLduMatrix
{
    int nCells = 0, nFaces = 0;
    DeviceBuffer<scalar> diag, upper, lower;
    DeviceBuffer<label>  owner, nei;
    DeviceBuffer<label>  ownerStart, losort, losortStart;
    unsigned long long addressingId = 0;                // stamped by buildDeviceLdu

    DeviceLduView view() const
    {
        DeviceLduView v{nCells, nFaces, diag.data(), upper.data(), lower.data(), owner.data(), nei.data(),
                        ownerStart.data(), losort.data(), losortStart.data()};
        v.addressingId = addressingId;
        return v;
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
    A.addressingId = nextDeviceAddressingId();
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
    DeviceLduView v{dm.nCells, dm.nInternalFaces, diag.data(), upper.data(), lower.data(), dm.owner.data(),
                    dm.nei.data(), dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                    0, nullptr, nullptr, nullptr};
    v.addressingId = dm.addressingId;
    return v;
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
    const scalar* cycCoeff,
    // the pair's already-signed jump, or null -- see DeviceLduView::cycJump
    const scalar* cycJump = nullptr)
{
    DeviceLduView v{dm.nCells, dm.nInternalFaces, diag.data(), upper.data(), lower.data(), dm.owner.data(),
                    dm.nei.data(), dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                    nCyc, cycOwn, cycNbr, cycCoeff, cycJump};
    v.addressingId = dm.addressingId;
    return v;
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

// `onField` = the operand IS the solution field, the one case in which a jump cyclic subtracts its
// jump from the neighbour cell. False everywhere else, which is every search direction in a Krylov
// solve; see DeviceLduView::cycJump.
void deviceAmul(const DeviceLduView& A, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi,
                bool onField = false);

// lduMatrix::residual (lduMatrixATmul.C:268-340): rA = source - diag*psi with every face term then subtracted
// from it in face order -- what smoothSolver's LOOP evaluates (smoothSolver.C:190-197), where its initial
// residual is source - A.psi. The two round differently, and on a solve that stalls at round-off the
// difference decides whether it stops. MEASURED on RAS/DTCHullMoving, step two's k solve, tolerance 1e-13:
// OpenFOAM stalls at 1.08e-13 and runs its 1000 sweeps, source - A.psi read 4.5e-14 and stopped after 2.
void deviceResidual(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& source,
    DeviceBuffer<scalar>& rA,
    bool onField = false);
// ...and the control that puts source - A.psi back (BRAE_CONTROL_DEVICE_GS_RESIDUAL_AMUL)
bool deviceResidualAsAmul();

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
