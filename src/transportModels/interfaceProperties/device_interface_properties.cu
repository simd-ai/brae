// Interface normal and curvature on the device -- see the header for what is here and what is not.
#include "device_interface_properties.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

inline void ckIP(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae deviceInterfaceProperties: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// interpolate -> normalise -> dot, in one pass. The face gradient is a register triple and never
// reaches memory.
__global__ void nHatfInternalKernel(
    const label*  __restrict__ own,
    const label*  __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    int nIf, scalar deltaN,
    scalar* __restrict__ nHatf)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;

    const label o = own[f], n = nei[f];
    const scalar wf = w[f], wn = scalar(1) - wf;
    const scalar fx = wf*gx[o] + wn*gx[n];
    const scalar fy = wf*gy[o] + wn*gy[n];
    const scalar fz = wf*gz[o] + wn*gz[n];

    // nHatfv = gradAlphaf/(mag + deltaN). deltaN is what keeps this finite where gradAlpha vanishes,
    // which is most of the domain -- so the divide is unguarded ON PURPOSE and the stabiliser is the
    // guard. A branch on mag == 0 here would be a second, different guard.
    const scalar m = sqrt(fx*fx + fy*fy + fz*fz);
    const scalar s = scalar(1) / (m + deltaN);

    nHatf[f] = (fx*s)*Sfx[f] + (fy*s)*Sfy[f] + (fz*s)*Sfz[f];
}

// ...and the faces of a PERIODIC PAIR, which are in neither list. OpenFOAM's line is the same one:
// gradAlphaf = fvc::interpolate(grad(alpha)), and on a coupled patch that interpolation is the two
// CELLS' gradients weighted (surfaceInterpolationScheme.C:284-293); then nHatfv = gradAlphaf/(mag +
// deltaN) and nHatf = nHatfv & Sf (interfaceProperties.C:134-150). correctContactAngle does not reach a
// cyclic: it walks the alphaContactAngle patches, and a periodic pair is not one.
__global__ void nHatfCyclicKernel(
    const label*  __restrict__ own,
    const label*  __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    int n, scalar deltaN,
    scalar* __restrict__ nHatf)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    const scalar wf = w[j], wn = scalar(1) - wf;
    const scalar fx = wf*gx[o] + wn*gx[nb];
    const scalar fy = wf*gy[o] + wn*gy[nb];
    const scalar fz = wf*gz[o] + wn*gz[nb];

    const scalar m = sqrt(fx*fx + fy*fy + fz*fz);
    const scalar s = scalar(1) / (m + deltaN);

    nHatf[j] = (fx*s)*Sfx[j] + (fy*s)*Sfy[j] + (fz*s)*Sfz[j];
}

// The boundary faces, from normals the caller has already corrected. Sf is indexed by the GLOBAL face
// number -- boundary faces follow the internal ones in the mesh's own ordering -- which is why this
// takes nInternalFaces as well.
__global__ void nHatfBoundaryKernel(
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ nbx,
    const scalar* __restrict__ nby,
    const scalar* __restrict__ nbz,
    int nBf, int nIf,
    scalar* __restrict__ nHatfBnd)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    const int f = nIf + b;
    nHatfBnd[b] = nbx[b]*Sfx[f] + nby[b]*Sfy[f] + nbz[b]*Sfz[f];
}

__global__ void negateKernel(scalar* __restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = -x[i];
}

}   // namespace


void deviceInterfaceNormalFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar                      deltaN,
    DeviceBuffer<scalar>&       nHatfInt)
{
    if (nInternalFaces <= 0) return;
    nHatfInt.resize(static_cast<std::size_t>(nInternalFaces));
    nHatfInternalKernel<<<nBlocks(nInternalFaces), TPB>>>(
        dm.owner.data(), dm.nei.data(), dm.w.data(),
        dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
        gx.data(), gy.data(), gz.data(),
        nInternalFaces, deltaN, nHatfInt.data());
    ckIP(cudaGetLastError(), "nHatf internal");
}


void deviceInterfaceNormalFluxBoundary(
    const DeviceMesh&           dm,
    int                         nBoundaryFaces,
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& nbx,
    const DeviceBuffer<scalar>& nby,
    const DeviceBuffer<scalar>& nbz,
    DeviceBuffer<scalar>&       nHatfBnd)
{
    if (nBoundaryFaces <= 0) return;
    nHatfBnd.resize(static_cast<std::size_t>(nBoundaryFaces));
    nHatfBoundaryKernel<<<nBlocks(nBoundaryFaces), TPB>>>(
        dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
        nbx.data(), nby.data(), nbz.data(),
        nBoundaryFaces, nInternalFaces, nHatfBnd.data());
    ckIP(cudaGetLastError(), "nHatf boundary");
}


void deviceInterfaceCurvature(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& nHatfInt,
    const DeviceBuffer<scalar>& nHatfBnd,
    DeviceBuffer<scalar>&       K)
{
    deviceDiv(dm, nHatfInt, nHatfBnd, K);
    const int n = static_cast<int>(K.size());
    if (n <= 0) return;
    negateKernel<<<nBlocks(n), TPB>>>(K.data(), n);
    ckIP(cudaGetLastError(), "curvature negate");
}


void deviceInterfaceNormalFluxCyclic(
    DeviceCyclic&               cyc,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar                      deltaN,
    DeviceBuffer<scalar>&       nHatfIf)
{
    if (cyc.n == 0) { nHatfIf.resize(0); return; }
    nHatfIf.resize(static_cast<std::size_t>(cyc.n));
    nHatfCyclicKernel<<<nBlocks(cyc.n), TPB>>>(
        cyc.ownCell.data(), cyc.nbrCell.data(), cyc.weights.data(),
        cyc.Sfx.data(), cyc.Sfy.data(), cyc.Sfz.data(),
        gx.data(), gy.data(), gz.data(), cyc.n, deltaN, nHatfIf.data());
    ckIP(cudaGetLastError(), "nHatf, interface");
}


void deviceInterfaceCorrect(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& alpha1,
    const DeviceBuffer<scalar>& alpha1Bnd,
    const DeviceBuffer<scalar>& nHatfBnd,
    scalar                      deltaN,
    DeviceBuffer<scalar>&       nHatfInt,
    DeviceBuffer<scalar>&       K,
    DeviceCyclic*               cyc,
    DeviceBuffer<scalar>*       nHatfIf)
{
    DeviceBuffer<scalar> gx, gy, gz;
    deviceGaussGrad(dm, alpha1, alpha1Bnd, gx, gy, gz);
    // THE PAIR's own contribution to grad(alpha) first: nHatf is built from that gradient, and a
    // gradient without the pair is the gradient of a mesh with a wall there.
    if (cyc && cyc->n > 0)
    {
        deviceCyclicAddGrad(*cyc, alpha1, dm.V, gx, gy, gz);
    }
    deviceInterfaceNormalFlux(dm, dm.nInternalFaces, gx, gy, gz, deltaN, nHatfInt);
    if (cyc && cyc->n > 0)
    {
        if (!nHatfIf)
        {
            throw std::runtime_error(
                "brae interfaceProperties (device): the mesh has a periodic pair and no array was given "
                "for nHatf there. The curvature is -div(nHatf), which sums a coupled face like any "
                "other, and the compressive flux reads the same normal.");
        }
        deviceInterfaceNormalFluxCyclic(*cyc, gx, gy, gz, deltaN, *nHatfIf);
    }
    deviceInterfaceCurvature(dm, nHatfInt, nHatfBnd, K);
    // ...and the pair's own face in -div(nHatf): K is a divergence, and fvc::div sums a coupled patch's
    // face into its cell like any patch's (fvc.cu:548-550). The negate above has already run, so this
    // subtracts what that sum would have added.
    if (cyc && cyc->n > 0)
    {
        DeviceBuffer<scalar> negIf(static_cast<std::size_t>(cyc->n));
        ckIP(cudaMemcpy(negIf.data(), nHatfIf->data(), sizeof(scalar)*cyc->n,
                        cudaMemcpyDeviceToDevice), "nHatf copy, interface");
        negateKernel<<<nBlocks(cyc->n), TPB>>>(negIf.data(), cyc->n);
        ckIP(cudaGetLastError(), "nHatf negate, interface");
        deviceCyclicAddDivFlux(*cyc, negIf, dm.V, K);
    }
}

} // namespace brae
