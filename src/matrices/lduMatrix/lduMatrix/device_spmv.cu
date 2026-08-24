// cf GPU offload (G1): the device lduMatrix SpMV kernel (Amul). One thread per cell gathers the diagonal
// plus the owner-ordered upper faces and the neighbour-sorted lower faces, race-free, deterministic.
#include "device_ldu.cuh"
#include "device_halo.cuh"
#include "distributed_ami.cuh"   // DistributedAMI + distributedAmiAmul: optional cyclicAMI coupling in the matvec
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;


__device__
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

    scalar s = diag[c] * psi[c];
    const int u0 = ownerStart[c], u1 = ownerStart[c + 1];
    for (int f = u0; f < u1; ++f)
        s += upper[f] * psi[nei[f]];          // faces owned by c
    const int l0 = losortStart[c], l1 = losortStart[c + 1];
    for (int k = l0; k < l1; ++k)
    {
        const int f = losort[k];
        s += lower[f] * psi[owner[f]];  // faces neighbouring c
    }
    Apsi[c] = s;
}


// cyclic (periodic) interface off-diagonal, OpenFOAM cyclicFvPatchField::updateInterfaceMatrix. One thread per
// cyclic face: Apsi[own] += coeff*psi[nbr]. atomicAdd because a cell may own faces on several interfaces.
__device__
void cyclicAmulKernel(
    int nCyc,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ coeff,
    const scalar* __restrict__ psi,
    scalar* __restrict__ Apsi)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nCyc) return;

    atomicAdd(&Apsi[own[j]], coeff[j] * psi[nbr[j]]);
}


// cyclicAMI weighted-stencil off-diagonal, one thread per source face: Apsi[own] += ifc * sum_k w*psi[nbr].
__device__
void amiAmulKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ off,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ psi,
    scalar* __restrict__ Apsi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    scalar s = 0;
    for (label k = off[i]; k < off[i+1]; ++k)
        s += w[k] * psi[nbr[k]];
    atomicAdd(&Apsi[own[i]], ifc[i] * s);
}
} // namespace


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

void deviceAmul(const DeviceLduView& A, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi)
{
    Apsi.resize(A.nCells);
    const int blocks = (A.nCells + TPB - 1) / TPB;
    const scalar* psid = psi.data();
    scalar* Apsid = Apsi.data();
    const int nC = A.nCells, nCyc = A.nCyc, nAmi = A.nAmi;
    const scalar *diag = A.diag, *upper = A.upper, *lower = A.lower;
    const label *nei = A.nei, *owner = A.owner, *ownerStart = A.ownerStart, *losort = A.losort, *losortStart = A.losortStart;
    pcudaParallelFor(blocks, TPB, [=] __device__ () {
        amulKernel(nC, diag, upper, lower, nei, owner, ownerStart, losort, losortStart, psid, Apsid);
    });
    cudaCheck(cudaGetLastError(), "amul");
    if (A.nCyc > 0)
    {
        const label* cycOwn = A.cycOwn; const label* cycNbr = A.cycNbr; const scalar* cycCoeff = A.cycCoeff;
        pcudaParallelFor((nCyc + TPB - 1) / TPB, TPB, [=] __device__ () {
            cyclicAmulKernel(nCyc, cycOwn, cycNbr, cycCoeff, psid, Apsid);
        });
        cudaCheck(cudaGetLastError(), "cyclicAmul");
    }
    if (A.nAmi > 0)
    {
        const label* amiOwn = A.amiOwn; const label* amiOff = A.amiOff; const label* amiNbr = A.amiNbr;
        const scalar* amiW = A.amiW; const scalar* amiIfc = A.amiIfc;
        pcudaParallelFor((nAmi + TPB - 1) / TPB, TPB, [=] __device__ () {
            amiAmulKernel(nAmi, amiOwn, amiOff, amiNbr, amiW, amiIfc, psid, Apsid);
        });
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
