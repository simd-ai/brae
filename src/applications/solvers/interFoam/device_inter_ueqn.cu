// interFoam's momentum predictor on the device -- see device_inter_ueqn.cuh for the two rho fields and
// for why the body force is a face flux.
#include "device_inter_ueqn.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

void ckU(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae interFoam device UEqn: ") + what + ": "
                                 + cudaGetErrorString(e));
}

__global__ void ddtRhoUKernel(
    const scalar* __restrict__ rho, const scalar* __restrict__ rhoOld,
    const scalar* __restrict__ uox, const scalar* __restrict__ uoy, const scalar* __restrict__ uoz,
    const scalar* __restrict__ V, int nC, scalar rDeltaT,
    scalar* __restrict__ diag,
    scalar* __restrict__ sx, scalar* __restrict__ sy, scalar* __restrict__ sz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    diag[c] += rDeltaT * rho[c] * V[c];
    // rhoOld, NOT rho. At a VoF interface these differ by the density ratio.
    const scalar w = rDeltaT * rhoOld[c] * V[c];
    sx[c] += w * uox[c];
    sy[c] += w * uoy[c];
    sz[c] += w * uoz[c];
}

__global__ void productKernel(const scalar* __restrict__ a, const scalar* __restrict__ b,
                              int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i]*b[i];
}

// interpolate(rho*nuEff) -- the PRODUCT, interpolated once.
__global__ void interpProductKernel(
    const label* __restrict__ own, const label* __restrict__ nei, const scalar* __restrict__ w,
    const scalar* __restrict__ mu, int nIf, scalar* __restrict__ out)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf) out[f] = w[f]*mu[own[f]] + (scalar(1) - w[f])*mu[nei[f]];
}

__global__ void sourceFluxKernel(
    const scalar* __restrict__ stf, const scalar* __restrict__ ghf,
    const scalar* __restrict__ snGradRho, const scalar* __restrict__ snGradPrgh,
    const scalar* __restrict__ magSf, int n, scalar* __restrict__ out)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) out[f] = (stf[f] - ghf[f]*snGradRho[f] - snGradPrgh[f]) * magSf[f];
}

}   // namespace


void deviceInterEulerDdtRhoU(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& rhoOld,
    const DeviceBuffer<scalar>& UOldX,
    const DeviceBuffer<scalar>& UOldY,
    const DeviceBuffer<scalar>& UOldZ,
    scalar                      deltaT,
    DeviceBuffer<scalar>&       diag,
    DeviceBuffer<scalar>&       srcX,
    DeviceBuffer<scalar>&       srcY,
    DeviceBuffer<scalar>&       srcZ)
{
    if (deltaT <= scalar(0))
        throw std::runtime_error(
            "brae interFoam device UEqn: deltaT must be positive; interFoam has no steady path.");
    const int nC = dm.nCells;
    if (static_cast<int>(diag.size()) != nC || static_cast<int>(srcX.size()) != nC)
        throw std::runtime_error(
            "brae interFoam device UEqn: the ddt adds INTO an existing diagonal and source, so both "
            "must already be sized to the cell count -- the div and stress terms are assembled first.");
    ddtRhoUKernel<<<nBlocks(nC), TPB>>>(
        rho.data(), rhoOld.data(), UOldX.data(), UOldY.data(), UOldZ.data(),
        dm.V.data(), nC, scalar(1)/deltaT, diag.data(), srcX.data(), srcY.data(), srcZ.data());
    ckU(cudaGetLastError(), "ddt(rho, U)");
}


void deviceMomentumSourceFlux(
    int                         n,
    const DeviceBuffer<scalar>& surfaceTensionForce,
    const DeviceBuffer<scalar>& ghf,
    const DeviceBuffer<scalar>& snGradRho,
    const DeviceBuffer<scalar>& snGradPrgh,
    const DeviceBuffer<scalar>& magSf,
    DeviceBuffer<scalar>&       out)
{
    if (n <= 0) return;
    if (static_cast<int>(magSf.size()) < n)
        throw std::runtime_error(
            "brae interFoam device UEqn: magSf is shorter than the face fields it scales. It is the "
            "mesh's FULL face array, internal faces first, not an internal-only one.");
    out.resize(static_cast<std::size_t>(n));
    sourceFluxKernel<<<nBlocks(n), TPB>>>(
        surfaceTensionForce.data(), ghf.data(), snGradRho.data(), snGradPrgh.data(),
        magSf.data(), n, out.data());
    ckU(cudaGetLastError(), "momentum source flux");
}


void deviceInterMuEff(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& nuEff,
    const DeviceBuffer<scalar>& rhoBnd,
    const DeviceBuffer<scalar>& nuEffBnd,
    DeviceBuffer<scalar>&       muCell,
    DeviceBuffer<scalar>&       muFace,
    DeviceBuffer<scalar>&       muBnd)
{
    const int nC = dm.nCells, nIf = dm.nInternalFaces, nBf = dm.nBndFaces;
    if (static_cast<int>(rho.size()) != nC || static_cast<int>(nuEff.size()) != nC)
        throw std::runtime_error("brae interFoam device UEqn: rho and nuEff must be cell fields.");
    if (static_cast<int>(rhoBnd.size()) != nBf || static_cast<int>(nuEffBnd.size()) != nBf)
        throw std::runtime_error(
            "brae interFoam device UEqn: rho and nuEff must also come with their PATCH values -- the "
            "stress at a wall is what the boundary condition says they are there, not what the face "
            "cell holds.");

    muCell.resize(static_cast<std::size_t>(nC));
    productKernel<<<nBlocks(nC), TPB>>>(rho.data(), nuEff.data(), nC, muCell.data());
    ckU(cudaGetLastError(), "mu = rho*nuEff");

    muFace.resize(static_cast<std::size_t>(nIf));
    if (nIf > 0)
    {
        interpProductKernel<<<nBlocks(nIf), TPB>>>(dm.owner.data(), dm.nei.data(), dm.w.data(),
                                                   muCell.data(), nIf, muFace.data());
        ckU(cudaGetLastError(), "interpolate(rho*nuEff)");
    }

    muBnd.resize(static_cast<std::size_t>(nBf));
    if (nBf > 0)
    {
        productKernel<<<nBlocks(nBf), TPB>>>(rhoBnd.data(), nuEffBnd.data(), nBf, muBnd.data());
        ckU(cudaGetLastError(), "mu at the patch");
    }
}

} // namespace brae
