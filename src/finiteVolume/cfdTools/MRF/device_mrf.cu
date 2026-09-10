// cf device MRF, Coriolis source + relative-flux kernels. See device_mrf.cuh. Mirrors the host MRFZone
// (mrf_zone.cuh), validated end-to-end by ctest gpu_mrf (closed rotating box -> solid-body rotation).
#include "device_mrf.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


// src[c] -= V[c]*(Omega x U)_kk on zone cells.  (Omega x U)_x = oy*Uz-oz*Uy, etc.
__device__
void mrfCoriolisKernel(
    int nC,
    const label* __restrict__ zc,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar ox,
    scalar oy,
    scalar oz,
    int kk,
    scalar* __restrict__ src)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC || !zc[c]) return;

    const scalar comp = (kk == 0) ? (oy*Uz[c] - oz*Uy[c])
                      : (kk == 1) ? (oz*Ux[c] - ox*Uz[c])
                                  : (ox*Uy[c] - oy*Ux[c]);
    src[c] -= V[c] * comp;
}


__device__
void mrfFrameFluxKernel(
    int n,
    const scalar* __restrict__ ff,
    scalar sign,
    scalar* __restrict__ phi)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) phi[f] -= sign * ff[f];
}
} // namespace


void deviceMrfCoriolis(
    const DeviceMRF& mrf,
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    int kk,
    DeviceBuffer<scalar>& src)
{
    const int nC = static_cast<int>(V.size());
    const label* zc = mrf.zoneCell.data();
    const scalar* Vd = V.data();
    const scalar* Uxd = Ux.data();
    const scalar* Uyd = Uy.data();
    const scalar* Uzd = Uz.data();
    const scalar ox = mrf.Omega.x, oy = mrf.Omega.y, oz = mrf.Omega.z;
    scalar* srcd = src.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
        mrfCoriolisKernel(nC, zc, Vd, Uxd, Uyd, Uzd, ox, oy, oz, kk, srcd);
    });
    cudaCheck(cudaGetLastError(), "mrfCoriolis");
}


void deviceMrfApplyFrameFlux(
    const DeviceMRF& mrf,
    scalar sign,
    DeviceBuffer<scalar>& phiInt,
    DeviceBuffer<scalar>& phiBnd)
{
    const int nIf = static_cast<int>(mrf.frameFluxInt.size()), nB = static_cast<int>(mrf.frameFluxBnd.size());
    if (nIf)
    {
        const scalar* ffi = mrf.frameFluxInt.data();
        scalar* phiIntd = phiInt.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () { mrfFrameFluxKernel(nIf, ffi, sign, phiIntd); });
    }
    if (nB)
    {
        const scalar* ffb = mrf.frameFluxBnd.data();
        scalar* phiBndd = phiBnd.data();
        pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () { mrfFrameFluxKernel(nB, ffb, sign, phiBndd); });
    }
    cudaCheck(cudaGetLastError(), "mrfFrameFlux");
}

} // namespace brae
