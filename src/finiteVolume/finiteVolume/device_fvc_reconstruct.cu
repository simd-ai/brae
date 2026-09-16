// fvc::reconstruct on the device -- see device_fvc_reconstruct.cuh for the sign and the normalisation.
#include "device_fvc_reconstruct.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

void ckR(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae deviceReconstruct: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// One face into this cell's tensor and vector. Written as a device function so the three gather loops
// below cannot drift apart -- the owner loop, the neighbour loop and the boundary loop all accumulate
// IDENTICALLY, which is exactly what makes surfaceSum not a divergence.
__device__ __forceinline__ void accumulate(
    scalar Sfx, scalar Sfy, scalar Sfz, scalar ssf,
    scalar& Txx, scalar& Txy, scalar& Txz,
    scalar& Tyx, scalar& Tyy, scalar& Tyz,
    scalar& Tzx, scalar& Tzy, scalar& Tzz,
    scalar& vx,  scalar& vy,  scalar& vz)
{
    const scalar mag = sqrt(Sfx*Sfx + Sfy*Sfy + Sfz*Sfz);
    const scalar hx = Sfx/mag, hy = Sfy/mag, hz = Sfz/mag;      // SfHat
    Txx += hx*Sfx; Txy += hx*Sfy; Txz += hx*Sfz;
    Tyx += hy*Sfx; Tyy += hy*Sfy; Tyz += hy*Sfz;
    Tzx += hz*Sfx; Tzy += hz*Sfy; Tzz += hz*Sfz;
    vx  += hx*ssf; vy  += hy*ssf; vz  += hz*ssf;
}

__global__ void reconstructKernel(
    int nC, int nIf,
    const label*  __restrict__ nei,
    const label*  __restrict__ ownerStart,
    const label*  __restrict__ losort,   const label* __restrict__ losortStart,
    const label*  __restrict__ bndCellStart, const label* __restrict__ bndPerm,
    const label*  __restrict__ bndIsEmpty,   const label* __restrict__ bndGFace,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ ssfInt, const scalar* __restrict__ ssfBnd,
    scalar* __restrict__ outX, scalar* __restrict__ outY, scalar* __restrict__ outZ)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar Txx = 0, Txy = 0, Txz = 0, Tyx = 0, Tyy = 0, Tyz = 0, Tzx = 0, Tzy = 0, Tzz = 0;
    scalar vx = 0, vy = 0, vz = 0;

    for (int k = ownerStart[c]; k < ownerStart[c + 1]; ++k)
        accumulate(Sfx[k], Sfy[k], Sfz[k], ssfInt[k],
                   Txx,Txy,Txz, Tyx,Tyy,Tyz, Tzx,Tzy,Tzz, vx,vy,vz);
    // THE SAME SIGN on the neighbour side -- fvcSurfaceIntegrate.C:165-166. deviceDiv subtracts here.
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        accumulate(Sfx[f], Sfy[f], Sfz[f], ssfInt[f],
                   Txx,Txy,Txz, Tyx,Tyy,Tyz, Tzx,Tzy,Tzz, vx,vy,vz);
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int b = bndPerm[k];
        if (bndIsEmpty[b]) continue;          // emptyFvPatch::size() == 0 in OpenFOAM
        const int gf = bndGFace[b];
        accumulate(Sfx[gf], Sfy[gf], Sfz[gf], ssfBnd[b],
                   Txx,Txy,Txz, Tyx,Tyy,Tyz, Tzx,Tzy,Tzz, vx,vy,vz);
    }

    // 3x3 inverse by cofactors. The tensor is sum(SfHat (x) Sf) over a closed cell, symmetric positive
    // definite, so no pivoting is needed.
    const scalar c00 = Tyy*Tzz - Tyz*Tzy;
    const scalar c01 = Tyz*Tzx - Tyx*Tzz;
    const scalar c02 = Tyx*Tzy - Tyy*Tzx;
    const scalar s   = scalar(1) / (Txx*c00 + Txy*c01 + Txz*c02);

    const scalar i00 = c00*s,                     i01 = (Txz*Tzy - Txy*Tzz)*s, i02 = (Txy*Tyz - Txz*Tyy)*s;
    const scalar i10 = c01*s,                     i11 = (Txx*Tzz - Txz*Tzx)*s, i12 = (Txz*Tyx - Txx*Tyz)*s;
    const scalar i20 = c02*s,                     i21 = (Txy*Tzx - Txx*Tzy)*s, i22 = (Txx*Tyy - Txy*Tyx)*s;

    outX[c] = i00*vx + i01*vy + i02*vz;
    outY[c] = i10*vx + i11*vy + i12*vz;
    outZ[c] = i20*vx + i21*vy + i22*vz;
    (void)nei;
    (void)nIf;
}

}   // namespace


void deviceReconstruct(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& ssfInt,
    const DeviceBuffer<scalar>& ssfBnd,
    DeviceBuffer<scalar>&       outX,
    DeviceBuffer<scalar>&       outY,
    DeviceBuffer<scalar>&       outZ)
{
    const int nC = dm.nCells;
    outX.resize(static_cast<std::size_t>(nC));
    outY.resize(static_cast<std::size_t>(nC));
    outZ.resize(static_cast<std::size_t>(nC));
    if (nC == 0) return;
    reconstructKernel<<<nBlocks(nC), TPB>>>(
        nC, dm.nInternalFaces, dm.nei.data(), dm.ownerStart.data(),
        dm.losort.data(), dm.losortStart.data(), dm.bndCellStart.data(), dm.bndPerm.data(),
        dm.bndIsEmpty.data(), dm.bndGFace.data(),
        dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
        ssfInt.data(), ssfBnd.data(), outX.data(), outY.data(), outZ.data());
    ckR(cudaGetLastError(), "reconstruct");
}

} // namespace brae
