#include "device_MRF.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }
}

namespace {

__global__
void mrfCoriolisKernel(
    int n,
    const label*  __restrict__ zoneCell,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar ox,
    scalar oy,
    scalar oz,
    int cmpt,
    scalar* __restrict__ src)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n || !zoneCell[c]) return;

    // (Omega x U), component cmpt
    scalar r;
    if      (cmpt == 0) r = oy * Uz[c] - oz * Uy[c];
    else if (cmpt == 1) r = oz * Ux[c] - ox * Uz[c];
    else                r = ox * Uy[c] - oy * Ux[c];
    src[c] -= V[c] * r;
}

__global__
void mrfSubtractKernel(int n, const scalar* __restrict__ ff, scalar* __restrict__ phi)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    phi[f] -= ff[f];
}

__global__
void mrfBoundaryKernel(
    int n,
    const scalar* __restrict__ ff,
    const label*  __restrict__ zeroMask,
    scalar* __restrict__ phi)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    // Included faces move WITH the frame, so their relative flux is zero outright -- not the
    // subtraction the internal and excluded faces take.
    if (zeroMask[f]) phi[f] = scalar(0);
    else             phi[f] -= ff[f];
}

} // namespace

DeviceMRFZone buildDeviceMRFZone(
    const cpu::MRF::Zone&       z,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    DeviceMRFZone d;
    d.Omega  = z.Omega;
    d.active = z.active;

    const label nC = m.nCells(), nIf = m.nInternalFaces();
    std::vector<label> zc(nC, 0);
    for (label c : z.cells)
    {
        zc[c] = 1;
    }
    d.zoneCell.copyFrom(zc);

    std::vector<scalar> ffi(nIf, 0.0);
    for (label f : z.internalFaces)
    {
        ffi[f] = dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]);
    }
    d.frameFluxInt.copyFrom(ffi);

    // The boundary arrays are flat over ALL patch faces, in patch order, matching the driver's phiBnd.
    std::vector<scalar> ffb;
    std::vector<label>  zb;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        std::vector<char> included(patches[pi].size, 0), excluded(patches[pi].size, 0);
        if (pi < z.includedFaces.size())
        {
            for (label i : z.includedFaces[pi]) included[i] = 1;
        }
        if (pi < z.excludedFaces.size())
        {
            for (label i : z.excludedFaces[pi]) excluded[i] = 1;
        }
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label f = patches[pi].start + i;
            ffb.push_back(excluded[i] ? dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]) : 0.0);
            zb.push_back(included[i] ? 1 : 0);
        }
    }
    d.frameFluxBnd.copyFrom(ffb);
    d.zeroBnd.copyFrom(zb);
    return d;
}

void deviceMrfCoriolisZone(
    const std::vector<DeviceMRFZone>& zones,
    const DeviceBuffer<scalar>&       V,
    const DeviceBuffer<scalar>&       Ux,
    const DeviceBuffer<scalar>&       Uy,
    const DeviceBuffer<scalar>&       Uz,
    int                               cmpt,
    DeviceBuffer<scalar>&             src)
{
    for (const DeviceMRFZone& z : zones)
    {
        if (!z.active) continue;
        const int n = static_cast<int>(z.zoneCell.size());
        if (!n) continue;
        mrfCoriolisKernel<<<nBlocks(n), TPB>>>(n, z.zoneCell.data(), V.data(),
                                               Ux.data(), Uy.data(), Uz.data(),
                                               z.Omega.x, z.Omega.y, z.Omega.z, cmpt, src.data());
    }
    cudaCheck(cudaGetLastError(), "mrfCoriolisZone");
}

void deviceMrfMakeRelative(
    const std::vector<DeviceMRFZone>& zones,
    DeviceBuffer<scalar>&             phiInt,
    DeviceBuffer<scalar>&             phiBnd)
{
    for (const DeviceMRFZone& z : zones)
    {
        if (!z.active) continue;
        const int nIf = static_cast<int>(z.frameFluxInt.size());
        const int nB  = static_cast<int>(z.frameFluxBnd.size());
        if (nIf)
        {
            mrfSubtractKernel<<<nBlocks(nIf), TPB>>>(nIf, z.frameFluxInt.data(), phiInt.data());
        }
        if (nB && nB == static_cast<int>(phiBnd.size()))
        {
            mrfBoundaryKernel<<<nBlocks(nB), TPB>>>(nB, z.frameFluxBnd.data(), z.zeroBnd.data(),
                                                    phiBnd.data());
        }
    }
    cudaCheck(cudaGetLastError(), "mrfMakeRelative");
}

} // namespace brae
