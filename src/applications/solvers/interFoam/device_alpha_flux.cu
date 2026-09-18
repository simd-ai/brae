// interFoam's alpha fluxes on the device -- see the header for why the nested flux is composed on the
// host rather than fused.
#include "device_alpha_flux.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

inline void ckA(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae deviceAlphaFlux: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// gaussConvectionScheme.C:64-73 -- flux(faceFlux, vf) == faceFlux*interpolate(vf).
__global__ void faceFluxKernel(
    const label*  __restrict__ own, const label* __restrict__ nei,
    const scalar* __restrict__ psi, const scalar* __restrict__ w,
    const scalar* __restrict__ field,
    int n, scalar* __restrict__ out)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const scalar wf = w[f];
    out[f] = psi[f] * (wf*field[own[f]] + (scalar(1) - wf)*field[nei[f]]);
}

__global__ void negateKernel(const scalar* __restrict__ in, int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}

__global__ void phicKernel(
    const scalar* __restrict__ phi, const scalar* __restrict__ magSf,
    int n, scalar cAlpha, scalar* __restrict__ phic)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) phic[f] = cAlpha * fabs(phi[f] / magSf[f]);
}

__global__ void fillKernel(scalar* __restrict__ x, int n, scalar v)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ void massFluxKernel(
    const scalar* __restrict__ alphaPhi, const scalar* __restrict__ phi,
    int n, scalar dRho, scalar rho2, scalar* __restrict__ rhoPhi)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) rhoPhi[f] = alphaPhi[f]*dRho + phi[f]*rho2;
}

__global__ void mulKernel(const scalar* __restrict__ a, const scalar* __restrict__ b,
                          int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i]*b[i];
}

__global__ void subKernel(const scalar* __restrict__ a, const scalar* __restrict__ b,
                          int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

}   // namespace


void deviceMultiplyFaces(int n, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b,
                         DeviceBuffer<scalar>& out)
{
    if (n <= 0) return;
    out.resize(static_cast<std::size_t>(n));
    mulKernel<<<nBlocks(n), TPB>>>(a.data(), b.data(), n, out.data());
    ckA(cudaGetLastError(), "multiply faces");
}


void deviceSubtractFaces(int n, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b,
                         DeviceBuffer<scalar>& out)
{
    if (n <= 0) return;
    out.resize(static_cast<std::size_t>(n));
    subKernel<<<nBlocks(n), TPB>>>(a.data(), b.data(), n, out.data());
    ckA(cudaGetLastError(), "subtract faces");
}


void deviceAlphaFaceFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& psiInt,
    const DeviceBuffer<scalar>& w,
    const DeviceBuffer<scalar>& field,
    DeviceBuffer<scalar>&       out)
{
    if (nInternalFaces <= 0) return;
    out.resize(static_cast<std::size_t>(nInternalFaces));
    faceFluxKernel<<<nBlocks(nInternalFaces), TPB>>>(
        dm.owner.data(), dm.nei.data(), psiInt.data(), w.data(), field.data(),
        nInternalFaces, out.data());
    ckA(cudaGetLastError(), "face flux");
}


void deviceNegateFaces(int n, const DeviceBuffer<scalar>& in, DeviceBuffer<scalar>& out)
{
    if (n <= 0) return;
    out.resize(static_cast<std::size_t>(n));
    negateKernel<<<nBlocks(n), TPB>>>(in.data(), n, out.data());
    ckA(cudaGetLastError(), "negate");
}


void deviceCompressionFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    int                         nBoundaryFaces,
    const DeviceBuffer<scalar>& phiInt,
    scalar                      cAlpha,
    DeviceBuffer<scalar>&       phicInt,
    DeviceBuffer<scalar>&       phicBnd)
{
    phicInt.resize(static_cast<std::size_t>(nInternalFaces));
    phicBnd.resize(static_cast<std::size_t>(nBoundaryFaces));
    if (nInternalFaces > 0)
    {
        // magSf is the mesh's FULL face array -- internal faces first -- so the internal faces index
        // straight through it.
        phicKernel<<<nBlocks(nInternalFaces), TPB>>>(
            phiInt.data(), dm.magSf.data(), nInternalFaces, cAlpha, phicInt.data());
        ckA(cudaGetLastError(), "phic");
    }
    if (nBoundaryFaces > 0)
    {
        // "Do not compress interface at non-coupled boundary faces" -- alphaEqn.H:79-89. brae has no
        // coupled patch in a VoF case yet, so every boundary face is zeroed; when one is added this
        // becomes a per-face branch on the same flag MULES already takes.
        fillKernel<<<nBlocks(nBoundaryFaces), TPB>>>(phicBnd.data(), nBoundaryFaces, scalar(0));
        ckA(cudaGetLastError(), "phic boundary");
    }
}


void deviceMassFlux(
    int                         n,
    const DeviceBuffer<scalar>& alphaPhi,
    const DeviceBuffer<scalar>& phiForRho2,
    scalar                      rho1,
    scalar                      rho2,
    DeviceBuffer<scalar>&       rhoPhi)
{
    if (n <= 0) return;
    rhoPhi.resize(static_cast<std::size_t>(n));
    massFluxKernel<<<nBlocks(n), TPB>>>(
        alphaPhi.data(), phiForRho2.data(), n, rho1 - rho2, rho2, rhoPhi.data());
    ckA(cudaGetLastError(), "mass flux");
}

} // namespace brae
