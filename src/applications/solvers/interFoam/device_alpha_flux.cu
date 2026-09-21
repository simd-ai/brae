// interFoam's alpha fluxes on the device -- see the header for why the nested flux is composed on the
// host rather than fused.
#include "device_alpha_flux.cuh"
#include "device_limiter.cuh"
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

// fvc::flux(psi, vf, scheme) on a COUPLED face. The scheme's own weight applies there exactly as on an
// internal face -- LimitedScheme.C's calcLimiter takes its bLim[patchi].coupled() branch -- with the
// neighbour CELL across the pair and the patch's own delta and central weight, which is what the host
// arm does (alpha_eqn_cpp.cu:263-303). An uncoupled patch has no second cell and takes its patch value
// instead, which is why the boundary kernels above carry no weights.
__global__ void cyclicFaceFluxKernel(
    const label*  __restrict__ own, const label* __restrict__ nbr,
    const scalar* __restrict__ phi, const scalar* __restrict__ w,
    const scalar* __restrict__ field,
    const scalar* __restrict__ gx, const scalar* __restrict__ gy, const scalar* __restrict__ gz,
    const scalar* __restrict__ dx, const scalar* __restrict__ dy, const scalar* __restrict__ dz,
    int scheme,                     // 0 linear, 1 upwind, 2 vanLeer, 3 interfaceCompression
    int n, scalar* __restrict__ out)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int P = own[j], N = nbr[j];
    const scalar pb = phi[j];
    const scalar vfN = field[N];
    const scalar up = (pb >= scalar(0)) ? scalar(1) : scalar(0);
    scalar wf = w[j];
    if (scheme == 1)
    {
        wf = up;
    }
    else if (scheme == 2)
    {
        // NVDTVD::r with the pair's delta, then vanLeer -- the shared limiter, not a second copy
        const int U = (pb > scalar(0)) ? P : N;
        const scalar gradcf = dx[j]*gx[U] + dy[j]*gy[U] + dz[j]*gz[U];
        const scalar gradf = vfN - field[P];
        scalar r;
        if (fabs(gradcf) >= 1000.0 * fabs(gradf))
            r = 2.0 * 1000.0 * ((gradcf >= 0.0) ? 1.0 : -1.0) * ((gradf >= 0.0) ? 1.0 : -1.0) - 1.0;
        else
            r = 2.0 * (gradcf / gradf) - 1.0;
        const scalar lim = limiterOfR(r, kVanLeerTwoByk);
        wf = lim*w[j] + (scalar(1) - lim)*up;
    }
    else if (scheme == 3)
    {
        // interfaceCompression's quartic limiter, on the pair's two CELLS -- the same expression the
        // internal faces take (device_fvm.cu, interfaceCompressionWeightsKernel). It reads no
        // gradient, so the pair costs nothing the internal faces do not.
        const scalar phiP = field[P];
        const scalar aP = scalar(1) - scalar(4)*phiP*(scalar(1) - phiP);
        const scalar aN = scalar(1) - scalar(4)*vfN*(scalar(1) - vfN);
        scalar lim = scalar(1) - fmax(aP*aP, aN*aN);
        lim = fmin(fmax(lim, scalar(0)), scalar(1));
        wf = lim*w[j] + (scalar(1) - lim)*up;
    }
    out[j] = pb * (wf*field[P] + (scalar(1) - wf)*vfN);
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


__global__ void cyclicPhicKernel(
    const scalar* __restrict__ phi, const scalar* __restrict__ magSf,
    int n, scalar cAlpha, scalar* __restrict__ phic)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) phic[j] = cAlpha * fabs(phi[j] / magSf[j]);
}

void deviceAlphaCyclicCompressionFlux(
    const DeviceCyclic&         cyc,
    scalar                      cAlpha,
    DeviceBuffer<scalar>&       phicIf)
{
    if (cyc.n == 0) { phicIf.resize(0); return; }
    phicIf.resize(static_cast<std::size_t>(cyc.n));
    cyclicPhicKernel<<<nBlocks(cyc.n), TPB>>>(cyc.phi.data(), cyc.magSf.data(), cyc.n, cAlpha,
                                              phicIf.data());
    cudaCheck(cudaGetLastError(), "phic, interface");
}


void deviceAlphaCyclicFluxWith(
    const DeviceCyclic&         cyc,
    const DeviceBuffer<scalar>& phi,
    int                         scheme,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>&       out)
{
    if (cyc.n == 0) { out.resize(0); return; }
    out.resize(static_cast<std::size_t>(cyc.n));
    cyclicFaceFluxKernel<<<nBlocks(cyc.n), TPB>>>(
        cyc.ownCell.data(), cyc.nbrCell.data(), phi.data(), cyc.weights.data(), field.data(),
        gx.data(), gy.data(), gz.data(), cyc.dX.data(), cyc.dY.data(), cyc.dZ.data(),
        scheme, cyc.n, out.data());
    cudaCheck(cudaGetLastError(), "alpha flux, interface");
}


void deviceAlphaCyclicFlux(
    const DeviceCyclic&         cyc,
    int                         scheme,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>&       out)
{
    deviceAlphaCyclicFluxWith(cyc, cyc.phi, scheme, field, gx, gy, gz, out);
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
