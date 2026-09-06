// cf GPU offload (G1): the device lduMatrix SpMV kernel (Amul). One thread per cell gathers the diagonal
// plus the owner-ordered upper faces and the neighbour-sorted lower faces, race-free, deterministic.
#include "device_ldu.cuh"
#include "device_halo.cuh"
#include "distributed_ami.cuh"   // DistributedAMI + distributedAmiAmul: optional cyclicAMI coupling in the matvec
#include <cuda_runtime.h>

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
__global__
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
__global__
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


void deviceAmul(const DeviceLduView& A, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi)
{
    Apsi.resize(A.nCells);
    const int blocks = (A.nCells + TPB - 1) / TPB;
    amulKernel<<<blocks, TPB>>>(A.nCells, A.diag, A.upper, A.lower, A.nei, A.owner,
                                A.ownerStart, A.losort, A.losortStart, psi.data(), Apsi.data());
    cudaCheck(cudaGetLastError(), "amul");
    if (A.nCyc > 0)
    {
        cyclicAmulKernel<<<(A.nCyc + TPB - 1) / TPB, TPB>>>(A.nCyc, A.cycOwn, A.cycNbr, A.cycCoeff, psi.data(), Apsi.data());
        cudaCheck(cudaGetLastError(), "cyclicAmul");
    }
    if (A.nAmi > 0)
    {
        amiAmulKernel<<<(A.nAmi + TPB - 1) / TPB, TPB>>>(A.nAmi, A.amiOwn, A.amiOff, A.amiNbr, A.amiW, A.amiIfc, psi.data(), Apsi.data());
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
