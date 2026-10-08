// cf GPU offload (G1): the device lduMatrix SpMV kernel (Amul). One thread per cell gathers the diagonal and
// then its faces in increasing face order -- OpenFOAM's face loop's order per cell -- race-free, deterministic.
#include "device_ldu.cuh"
#include "device_halo.cuh"
#include "distributed_ami.cuh"   // DistributedAMI + distributedAmiAmul: optional cyclicAMI coupling in the matvec
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace brae {

namespace {
constexpr int TPB = 256;


__global__
void amulKernel(
    int nC,
    const scalar* __restrict__ diag,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    const label* __restrict__ nei,
    const label* __restrict__ owner,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ psi,
    scalar* __restrict__ Apsi)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // lduMatrixATmul.C:121-135, the branch a solve takes (nothing calls lowerCSR(), so hasLowerCSR() is false):
    // Apsi = diag*psi, then ONE loop over the faces in index order, each adding its lower term to the
    // neighbour and its upper term to the owner. A cell therefore receives its contributions in increasing
    // FACE order, owned and neighbouring faces interleaved -- not every owned face and then every
    // neighbouring one, which is the same sum in another order and another last bit. Both lists are
    // ascending in face index (ownerStart: faces are upper-triangular ordered; losort: stable by
    // neighbour), so the walk below is a merge of the two by face index.
    scalar s = diag[c] * psi[c];
    int f = ownerStart[c];
    const int u1 = ownerStart[c + 1];
    int k = losortStart[c];
    const int l1 = losortStart[c + 1];
    while (f < u1 || k < l1)
    {
        const int fl = (k < l1) ? losort[k] : 0x7fffffff;
        if (f < u1 && f < fl)
        {
            s += upper[f] * psi[nei[f]];      // c owns face f
            ++f;
        }
        else
        {
            s += lower[fl] * psi[owner[fl]];  // c is face fl's neighbour
            ++k;
        }
    }
    Apsi[c] = s;
}


// cyclic (periodic) interface off-diagonal, OpenFOAM cyclicFvPatchField::updateInterfaceMatrix. One thread per
// cyclic face: Apsi[own] += coeff*psi[nbr]. atomicAdd because a cell may own faces on several interfaces.
__global__
void cyclicAmulKernel(
    int nCyc,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ coeff,
    const scalar* __restrict__ jump,        // already signed; null = no jump on this pair
    const scalar* __restrict__ psi,
    scalar* __restrict__ Apsi,
    const label* __restrict__ pairRank,
    int pairPass)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nCyc) return;
    // one of its own cell's faces a launch, in face order (DeviceLduView::pairRank)
    if (pairRank && pairRank[j] != pairPass) return;

    // jumpCyclicFvPatchField::updateInterfaceMatrix: the neighbour value is psi[nbr] - jump
    const scalar pnf = jump ? (psi[nbr[j]] - jump[j]) : psi[nbr[j]];
    atomicAdd(&Apsi[own[j]], coeff[j] * pnf);
}


// cyclicAMI weighted-stencil off-diagonal, one thread per source face: Apsi[own] += ifc * sum_k w*psi[nbr].
__global__
void amiAmulKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ psi,
    scalar* __restrict__ Apsi,
    const label* __restrict__ pairRank,
    int pairPass)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // one of its own cell's faces a launch, in face order (DeviceLduView::pairRank)
    if (pairRank && pairRank[i] != pairPass) return;

    scalar s = 0;
    for (label k = off[i]; k < off[i+1]; ++k)
        s += w[k] * psi[nbr[k]];
    atomicAdd(&Apsi[own[i]], ifc[i] * s);
}
// lduMatrix::residual (lduMatrixATmul.C:268-340) for one cell: rA = source - diag*psi, then the face loop
// SUBTRACTS each term from it, in face order. Not source - (A.psi): the same number to another last bit,
// and at a converged iterate the last bits are all the residual is.
__global__
void residualKernel(
    int nC,
    const scalar* __restrict__ diag,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    const label* __restrict__ nei,
    const label* __restrict__ owner,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ source,
    scalar* __restrict__ rA)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    // every product rounded before it is subtracted (__dmul_rn, __dsub_rn): OpenFOAM's x86-64 build does
    // not fuse them
    scalar s = __dsub_rn(source[c], __dmul_rn(diag[c], psi[c]));
    int f = ownerStart[c];
    const int u1 = ownerStart[c + 1];
    int k = losortStart[c];
    const int l1 = losortStart[c + 1];
    while (f < u1 || k < l1)
    {
        const int fl = (k < l1) ? losort[k] : 0x7fffffff;
        if (f < u1 && f < fl)
        {
            s = __dsub_rn(s, __dmul_rn(upper[f], psi[nei[f]]));
            ++f;
        }
        else
        {
            s = __dsub_rn(s, __dmul_rn(lower[fl], psi[owner[fl]]));
            ++k;
        }
    }
    rA[c] = s;
}

// ...and the interfaces' share of it, the mirror of cyclicAmulKernel and amiAmulKernel
__global__
void cyclicResidualKernel(
    int nCyc,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ coeff,
    const scalar* __restrict__ jump,
    const scalar* __restrict__ psi,
    scalar* __restrict__ rA,
    const label* __restrict__ pairRank,
    int pairPass)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nCyc) return;
    // one of its own cell's faces a launch, in face order (DeviceLduView::pairRank)
    if (pairRank && pairRank[j] != pairPass) return;

    const scalar pnf = jump ? (psi[nbr[j]] - jump[j]) : psi[nbr[j]];
    atomicAdd(&rA[own[j]], -__dmul_rn(coeff[j], pnf));
}

__global__
void amiResidualKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ psi,
    scalar* __restrict__ rA,
    const label* __restrict__ pairRank,
    int pairPass)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // one of its own cell's faces a launch, in face order (DeviceLduView::pairRank)
    if (pairRank && pairRank[i] != pairPass) return;

    scalar s = 0;
    for (label k = off[i]; k < off[i+1]; ++k)
    {
        s += w[k] * psi[nbr[k]];
    }
    atomicAdd(&rA[own[i]], -__dmul_rn(ifc[i], s));
}
} // namespace


// BRAE_CONTROL_DEVICE_GS_RESIDUAL_AMUL=1 puts source - A.psi back in the smoothing loops -- the gate's control
bool deviceResidualAsAmul()
{
    static const bool on = []
    {
        const bool v = std::getenv("BRAE_CONTROL_DEVICE_GS_RESIDUAL_AMUL") != nullptr;
        if (v)
        {
            std::printf("  *** CONTROL MODE: the device smoothSolver's loop residual is source - A.psi, not "
                        "lduMatrix::residual. This run is deliberately wrong. ***\n");
        }
        return v;
    }();
    return on;
}

void deviceResidual(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& source,
    DeviceBuffer<scalar>& rA,
    bool onField)
{
    rA.resize(A.nCells);
    const int blocks = (A.nCells + TPB - 1) / TPB;
    residualKernel<<<blocks, TPB>>>(A.nCells, A.diag, A.upper, A.lower, A.nei, A.owner,
                                    A.ownerStart, A.losort, A.losortStart, psi.data(), source.data(), rA.data());
    cudaCheck(cudaGetLastError(), "residual");
    if (A.nCyc > 0)
    {
        for (int pairPass = 0; pairPass < A.nPairRanks; ++pairPass)
        {
            cyclicResidualKernel<<<(A.nCyc + TPB - 1) / TPB, TPB>>>(A.nCyc, A.cycOwn, A.cycNbr, A.cycCoeff,
                                                                    onField ? A.cycJump : nullptr,
                                                                    psi.data(), rA.data(),
                A.nPairRanks > 1 ? A.pairRank : nullptr, pairPass);
        }
        cudaCheck(cudaGetLastError(), "cyclicResidual");
    }
    if (A.nAmi > 0)
    {
        for (int pairPass = 0; pairPass < A.nPairRanks; ++pairPass)
        {
            amiResidualKernel<<<(A.nAmi + TPB - 1) / TPB, TPB>>>(A.nAmi, A.amiOwn, A.amiOff, A.amiNbr, A.amiW,
                                                                 A.amiIfc, psi.data(), rA.data(),
                A.nPairRanks > 1 ? A.pairRank : nullptr, pairPass);
        }
        cudaCheck(cudaGetLastError(), "amiResidual");
    }
}


// FP-11 TRIED THE CONTIGUOUS-ROW LAYOUT HERE AND IT DOES NOT TRANSFER. This product is the largest
// single item at scale -- on squareBend at 896,000 cells it is 22.5 of the turbulence phase's 44.9 GPU
// ms per iteration and 30.6 of the pressure phase's 69.0, 53 ms of a 172 ms iteration -- and its
// neighbour loop reads losort[k], then lower[f] and psi[owner[f]], three indirections into arrays
// ordered by FACE. The same shape in the FP32 AMG operator gave 37% to a row layout (device_amg.cuh).
//
// It was built here too, bit-identical (the row is the cell's owner faces in ownerStart order then its
// neighbour faces in losort order, which is the order below), wired into the BiCGStab solves and their
// series, and measured on the 896k case: the products it took over went 450 us to 397 (12%, not 37),
// the turbulence phase 49.0 to 47.4 ms per iteration, the whole iteration's GPU time 153.0 to 151.0,
// and the wall not at all. Reverted.
//
// WHY IT DOES NOT TRANSFER: a row layout stores each off-diagonal TWICE, once for the owner and once
// for the neighbour, where the LDU form stores it once. In FP32 that traded 10.5 MB of values for 21
// and the coalescing won; in FP64 it trades 21 MB for 42 on top of a refill pass, and the extra bytes
// eat the gain. The cost here is bandwidth, and a layout that doubles the bytes cannot fix bandwidth.

void deviceAmul(const DeviceLduView& A, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi,
                bool onField)
{
    Apsi.resize(A.nCells);
    const int blocks = (A.nCells + TPB - 1) / TPB;
    amulKernel<<<blocks, TPB>>>(A.nCells, A.diag, A.upper, A.lower, A.nei, A.owner,
                                A.ownerStart, A.losort, A.losortStart, psi.data(), Apsi.data());
    cudaCheck(cudaGetLastError(), "amul");
    if (A.nCyc > 0)
    {
        for (int pairPass = 0; pairPass < A.nPairRanks; ++pairPass)
        {
            cyclicAmulKernel<<<(A.nCyc + TPB - 1) / TPB, TPB>>>(A.nCyc, A.cycOwn, A.cycNbr, A.cycCoeff,
                                                                onField ? A.cycJump : nullptr,
                                                                psi.data(), Apsi.data(),
                A.nPairRanks > 1 ? A.pairRank : nullptr, pairPass);
        }
        cudaCheck(cudaGetLastError(), "cyclicAmul");
    }
    if (A.nAmi > 0)
    {
        for (int pairPass = 0; pairPass < A.nPairRanks; ++pairPass)
        {
            amiAmulKernel<<<(A.nAmi + TPB - 1) / TPB, TPB>>>(A.nAmi, A.amiOwn, A.amiOff, A.amiNbr, A.amiW,
                A.amiIfc, psi.data(), Apsi.data(),
                A.nPairRanks > 1 ? A.pairRank : nullptr, pairPass);
        }
        cudaCheck(cudaGetLastError(), "amiAmul");
    }
}

// Distributed product: overlap the halo transfer with the local product, then apply the interface coupling.
// Same ordering as host parallelAmul (post -> local -> wait -> update), all on the per-thread default stream.
void deviceParallelAmul(
    const DeviceLduView& A,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& psi,
    DeviceBuffer<scalar>& Apsi,
    const DistributedAMI* ami)
{
    halo.postExchange(psi.data());          // pack + put (async)
    deviceAmul(A, psi, Apsi);               // local diag + upper/lower gather, overlaps the transfer
    halo.waitExchange();                    // barrier: neighbour values now in the recv buffers
    for (int i = 0; i < halo.nInterfaces(); ++i)
        halo.updateInterfaceMatrix(i, Apsi.data(), ifaceCoeffs[i].data());   // Apsi[fc] -= coeff * psiNbr
    if (ami) distributedAmiAmul(*ami, psi, Apsi);   // + cyclicAMI: gather remote target cells (NVSHMEM) then amul
}

} // namespace brae
