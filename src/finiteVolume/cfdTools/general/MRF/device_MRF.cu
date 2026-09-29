#include <stdexcept>
#include <string>
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

    // (Omega x U), component cmpt, then src -= V*r -- with the host's contraction, read off the object code
    // g++ makes of addCoriolis (MRF_cpp.cu:271 over cross(), cf_types.cuh; objdump -dl): x and y round their
    // FIRST product and fuse the second, z rounds its SECOND and fuses the first, and src - V*r is one
    // fmsub. Written as fma() so nvcc's own contraction choice cannot move it.
    scalar r;
    if      (cmpt == 0) r = fma(-oz, Uy[c], oy * Uz[c]);
    else if (cmpt == 1) r = fma(-ox, Uz[c], oz * Ux[c]);
    else                r = fma(ox, Uy[c], -(oy * Ux[c]));
    src[c] = fma(-V[c], r, src[c]);
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

__global__
void mrfZeroKernel(
    int n,
    const label* __restrict__ mask,
    scalar* __restrict__ phi)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n && mask[f]) phi[f] = scalar(0);
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
    std::vector<label>  fli(nIf, 0);
    for (label f : z.internalFaces)
    {
        ffi[f] = dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]);
        fli[f] = 1;
    }
    d.frameFluxInt.copyFrom(ffi);

    // The boundary arrays are flat over ALL patch faces, in patch order, matching the driver's phiBnd.
    std::vector<scalar> ffb;
    std::vector<label>  zb;
    std::vector<label>  flb;
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
            flb.push_back((included[i] || excluded[i]) ? 1 : 0);
        }
    }
    d.frameFluxBnd.copyFrom(ffb);
    d.zeroBnd.copyFrom(zb);
    d.filterInt.copyFrom(fli);
    d.filterBnd.copyFrom(flb);
    return d;
}

namespace {

// A ZONE WHOSE ARRAYS ARE NOT THE CURRENT MESH'S IS A MISSED REBUILD, and it must say so rather than
// half-apply itself. deviceMrfMakeRelative used to carry `if (nB && nB == phiBnd.size())` -- a SILENT SKIP
// of the boundary half, with the internal half launched unguarded beside it -- so a zone left over from
// before a topology change would subtract the frame flux on the wrong internal faces and do nothing at all
// on the boundary, with no message. That is the same shape as the porosity cell list uploaded once before
// the time loop, which cost a measured alpha 1.4218e-02 before it was found by comparing against a control.
// FAIL-PROOF, 2026-09-27: with buildMrf() taken out of interFoam's device change branch, the mrf profile's
// device arm now stops at the FIRST change with "deviceMrfCoriolisZone: this MRF zone's per-cell mask is
// 3072 where the field is 3660" instead of running to completion on the wrong faces.
void requireCurrent(const char* who, const char* what, std::size_t have, std::size_t want)
{
    if (have == want) return;
    throw std::runtime_error(
        std::string("brae ") + who + ": this MRF zone's " + what + " is " + std::to_string(have)
        + " where the field is " + std::to_string(want) + ". The zone was built for a different mesh, so a "
        "topology change did not rebuild it (buildDeviceMRFZone from the host zone MRF::update rebuilt). "
        "Refused rather than apply the frame to the wrong faces.");
}

} // namespace

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
        requireCurrent("deviceMrfCoriolisZone", "per-cell mask", z.zoneCell.size(), V.size());
        mrfCoriolisKernel<<<nBlocks(n), TPB>>>(n, z.zoneCell.data(), V.data(),
                                               Ux.data(), Uy.data(), Uz.data(),
                                               z.Omega.x, z.Omega.y, z.Omega.z, cmpt, src.data());
    }
    cudaCheck(cudaGetLastError(), "mrfCoriolisZone");
}

void deviceMrfZeroFilter(
    const std::vector<DeviceMRFZone>& zones,
    DeviceBuffer<scalar>&             phiInt,
    DeviceBuffer<scalar>&             phiBnd)
{
    for (const DeviceMRFZone& z : zones)
    {
        if (!z.active) continue;
        const int nIf = static_cast<int>(z.filterInt.size());
        const int nBf = static_cast<int>(z.filterBnd.size());
        requireCurrent("deviceMrfZeroFilter", "internal filter", z.filterInt.size(), phiInt.size());
        requireCurrent("deviceMrfZeroFilter", "boundary filter", z.filterBnd.size(), phiBnd.size());
        if (nIf) mrfZeroKernel<<<nBlocks(nIf), TPB>>>(nIf, z.filterInt.data(), phiInt.data());
        if (nBf) mrfZeroKernel<<<nBlocks(nBf), TPB>>>(nBf, z.filterBnd.data(), phiBnd.data());
    }
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
        requireCurrent("deviceMrfMakeRelative", "internal frame flux", z.frameFluxInt.size(), phiInt.size());
        requireCurrent("deviceMrfMakeRelative", "boundary frame flux", z.frameFluxBnd.size(), phiBnd.size());
        if (nIf)
        {
            mrfSubtractKernel<<<nBlocks(nIf), TPB>>>(nIf, z.frameFluxInt.data(), phiInt.data());
        }
        if (nB)
        {
            mrfBoundaryKernel<<<nBlocks(nB), TPB>>>(nB, z.frameFluxBnd.data(), z.zeroBnd.data(),
                                                    phiBnd.data());
        }
    }
    cudaCheck(cudaGetLastError(), "mrfMakeRelative");
}

} // namespace brae
