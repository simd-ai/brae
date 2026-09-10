// cf GPU offload (G3): fvc explicit operators on device. interpolate is per-internal-face; div and
// gaussGrad are per-cell gathers (internal owner/neighbour faces via ownerStart/losort, boundary faces via
// bndCellStart), race-free, deterministic, matching the CPU fvc to machine precision.
#include "device_mesh.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__device__
void interpKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ vol,
    scalar* __restrict__ sf)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf) sf[f] = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
}


__device__
void divKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ phiInt,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ bval,
    const scalar* __restrict__ V,
    scalar* __restrict__ d)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
        s += phiInt[f];              // +owner internal
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
        s -= phiInt[losort[k]];    // -neighbour internal
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
        s += bval[bndPerm[k]];   // +boundary
    d[c] = s / V[c];
}


__device__
void gradKernel(
    int nC,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ vol,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ bval,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar sx = 0.0, sy = 0.0, sz = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)   // +owner internal
    {
        const scalar pf = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
        sx += Sfx[f] * pf;
        sy += Sfy[f] * pf;
        sz += Sfz[f] * pf;
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)   // -neighbour internal
    {
        const int f = losort[k];
        const scalar pf = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
        sx -= Sfx[f] * pf;
        sy -= Sfy[f] * pf;
        sz -= Sfz[f] * pf;
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)   // +boundary
    {
        const int kk = bndPerm[k];
        const int f = bndGFace[kk];
        const scalar pv = bval[kk];
        sx += Sfx[f] * pv;
        sy += Sfy[f] * pv;
        sz += Sfz[f] * pv;
    }
    gx[c] = sx / V[c];
    gy[c] = sy / V[c];
    gz[c] = sz / V[c];
}
} // namespace


void deviceInterpolate(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol, DeviceBuffer<scalar>& sfInt)
{
    sfInt.resize(dm.nInternalFaces);
    const int nIf = dm.nInternalFaces;
    const label* own = dm.owner.data(); const label* nei = dm.nei.data(); const scalar* w = dm.w.data();
    const scalar* vold = vol.data(); scalar* sfd = sfInt.data();
    pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
        interpKernel(nIf, own, nei, w, vold, sfd); });
    cudaCheck(cudaGetLastError(), "interp");
}


void deviceDiv(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& bval, DeviceBuffer<scalar>& d)
{
    d.resize(dm.nCells);
    const int nC = dm.nCells;
    const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
    const label* losortStart = dm.losortStart.data(); const scalar* phiIntD = phiInt.data();
    const label* bndCellStart = dm.bndCellStart.data(); const label* bndPerm = dm.bndPerm.data();
    const scalar* bvalD = bval.data(); const scalar* Vd = dm.V.data(); scalar* dd = d.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
        divKernel(nC, ownerStart, losort, losortStart, phiIntD, bndCellStart, bndPerm, bvalD, Vd, dd); });
    cudaCheck(cudaGetLastError(), "div");
}


void deviceGaussGrad(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& vol,
    const DeviceBuffer<scalar>& bval,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz)
{
    gx.resize(dm.nCells);
    gy.resize(dm.nCells);
    gz.resize(dm.nCells);
    {
        const int nC = dm.nCells;
        const label* own = dm.owner.data(); const label* nei = dm.nei.data(); const scalar* w = dm.w.data();
        const scalar* Sfx = dm.Sfx.data(); const scalar* Sfy = dm.Sfy.data(); const scalar* Sfz = dm.Sfz.data();
        const scalar* vold = vol.data();
        const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
        const label* losortStart = dm.losortStart.data();
        const label* bndCellStart = dm.bndCellStart.data(); const label* bndPerm = dm.bndPerm.data();
        const label* bndGFace = dm.bndGFace.data(); const scalar* bvalD = bval.data();
        const scalar* Vd = dm.V.data(); scalar* gxd = gx.data(); scalar* gyd = gy.data(); scalar* gzd = gz.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            gradKernel(nC, own, nei, w, Sfx, Sfy, Sfz, vold, ownerStart, losort, losortStart,
                       bndCellStart, bndPerm, bndGFace, bvalD, Vd, gxd, gyd, gzd); });
    }
    cudaCheck(cudaGetLastError(), "gaussGrad");
}


namespace {
// OF cellLimitedGrad<minmod>::limitFaceCmpt: r = maxDelta/extrapolate (or minDelta/extrapolate); minmod limiter
// min(r,1). |extrapolate|~0 -> face imposes no constraint (returns 1). SMALL = Foam::SMALL (1e-15, double).
__device__ __forceinline__
scalar limFace(scalar maxD, scalar minD, scalar ex)
{
    if (ex >  1e-15) return fmin(maxD / ex, 1.0);
    if (ex < -1e-15) return fmin(minD / ex, 1.0);
    return 1.0;
}


__device__
void cellLimitGradKernel(
    int nC,
    scalar k,
    const scalar* __restrict__ U,
    const scalar* __restrict__ Ubnd,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ owner,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    const scalar* __restrict__ dBndX,
    const scalar* __restrict__ dBndY,
    const scalar* __restrict__ dBndZ,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar uc = U[c];
    scalar maxD = 0.0, minD = 0.0;   // incl. the cell itself (U[c]-U[c]=0)
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
    {
        const scalar d = U[nei[f]] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    {
        const scalar d = U[owner[losort[j]]] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const scalar d = Ubnd[bndPerm[j]] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    if (k < 1.0)   // OF k<1 widening
    {
        const scalar wdn = (1.0/k - 1.0)*(maxD - minD);
        maxD += wdn;
        minD -= wdn;
    }
    const scalar gcx = gx[c], gcy = gy[c], gcz = gz[c];
    scalar lim = 1.0;
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        lim = fmin(lim, limFace(maxD, minD, dOwnX[f]*gcx + dOwnY[f]*gcy + dOwnZ[f]*gcz));
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    {
        const int f = losort[j];
        lim = fmin(lim, limFace(maxD, minD, dNeiX[f]*gcx + dNeiY[f]*gcy + dNeiZ[f]*gcz));
    }
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const int bk = bndPerm[j];
        lim = fmin(lim, limFace(maxD, minD, dBndX[bk]*gcx + dBndY[bk]*gcy + dBndZ[bk]*gcz));
    }
    gx[c] = gcx*lim;
    gy[c] = gcy*lim;
    gz[c] = gcz*lim;
}

// ---- the coupled-patch path (see CellLimitInterface) -------------------------------------------------
// max/min and the limiter are all order-independent reductions, so scattering them from the faces with
// atomics gives the same answer every run whatever order the faces land in.
__device__ __forceinline__ void atomicFmax(scalar* a, scalar v)
{
    unsigned long long* p = reinterpret_cast<unsigned long long*>(a);
    unsigned long long old = *p, assumed;
    while (v > __longlong_as_double(old))
    {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(v));
        if (old == assumed) break;
    }
}
__device__ __forceinline__ void atomicFmin(scalar* a, scalar v)
{
    unsigned long long* p = reinterpret_cast<unsigned long long*>(a);
    unsigned long long old = *p, assumed;
    while (v < __longlong_as_double(old))
    {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(v));
        if (old == assumed) break;
    }
}

// Phase 1 on cells: maxD/minD from the internal + non-coupled boundary faces only. Split out of
// cellLimitGradKernel so the interface faces can be folded in between the two halves.
__device__
void cellLimitMinMaxKernel(
    int nC,
    const scalar* __restrict__ U,
    const scalar* __restrict__ Ubnd,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ owner,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    scalar* __restrict__ maxD,
    scalar* __restrict__ minD)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar uc = U[c];
    scalar mx = 0.0, mn = 0.0;   // incl. the cell itself (U[c]-U[c]=0)
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
    { const scalar d = U[nei[f]] - uc; mx = fmax(mx,d); mn = fmin(mn,d); }
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    { const scalar d = U[owner[losort[j]]] - uc; mx = fmax(mx,d); mn = fmin(mn,d); }
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    { const scalar d = Ubnd[bndPerm[j]] - uc; mx = fmax(mx,d); mn = fmin(mn,d); }
    maxD[c] = mx;
    minD[c] = mn;
}

// Phase 2 on interface faces: the coupled neighbour joins the cell's range.
__device__
void ifMinMaxKernel(int n, const label* __restrict__ own, const scalar* __restrict__ nbrVal,
                    const scalar* __restrict__ U, scalar* __restrict__ maxD, scalar* __restrict__ minD)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = own[i];
    const scalar d = nbrVal[i] - U[c];
    atomicFmax(&maxD[c], d);
    atomicFmin(&minD[c], d);
}

// Phase 3: OF's k<1 widening, applied once the range is complete (interfaces included).
__device__
void widenKernel(int nC, scalar k, scalar* __restrict__ maxD, scalar* __restrict__ minD)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar wdn = (1.0/k - 1.0)*(maxD[c] - minD[c]);
    maxD[c] += wdn;
    minD[c] -= wdn;
}

// Phase 4 on cells: the limiter from the internal + non-coupled boundary face extrapolations.
__device__
void cellLimitFactorKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
    const scalar* __restrict__ dBndX, const scalar* __restrict__ dBndY, const scalar* __restrict__ dBndZ,
    const scalar* __restrict__ maxD, const scalar* __restrict__ minD,
    const scalar* __restrict__ gx, const scalar* __restrict__ gy, const scalar* __restrict__ gz,
    scalar* __restrict__ lim)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar gcx = gx[c], gcy = gy[c], gcz = gz[c], mx = maxD[c], mn = minD[c];
    scalar l = 1.0;
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        l = fmin(l, limFace(mx, mn, dOwnX[f]*gcx + dOwnY[f]*gcy + dOwnZ[f]*gcz));
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    { const int f = losort[j]; l = fmin(l, limFace(mx, mn, dNeiX[f]*gcx + dNeiY[f]*gcy + dNeiZ[f]*gcz)); }
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    { const int bk = bndPerm[j]; l = fmin(l, limFace(mx, mn, dBndX[bk]*gcx + dBndY[bk]*gcy + dBndZ[bk]*gcz)); }
    lim[c] = l;
}

// Phase 5 on interface faces: OF's second loop is over EVERY boundary face, not just the uncoupled ones.
__device__
void ifLimitKernel(int n, const label* __restrict__ own,
                   const scalar* __restrict__ dx, const scalar* __restrict__ dy, const scalar* __restrict__ dz,
                   const scalar* __restrict__ maxD, const scalar* __restrict__ minD,
                   const scalar* __restrict__ gx, const scalar* __restrict__ gy, const scalar* __restrict__ gz,
                   scalar* __restrict__ lim)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = own[i];
    atomicFmin(&lim[c], limFace(maxD[c], minD[c], dx[i]*gx[c] + dy[i]*gy[c] + dz[i]*gz[c]));
}

__device__
void applyLimitKernel(int nC, const scalar* __restrict__ lim,
                      scalar* __restrict__ gx, scalar* __restrict__ gy, scalar* __restrict__ gz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar l = lim[c];
    gx[c] *= l; gy[c] *= l; gz[c] *= l;
}
} // namespace


// cellLimited Gauss linear <k> (OF cellLimitedGrad<vector,minmod>), applied to ONE component's gradient: scale grad
// so the reconstructed face value C[c] + (Cf-C).grad stays within the cell's face-neighbour min/max. Per-component
// (caller loops the 3 U components). k=1 = full limiting (motorBike); k<1 widens the bounds (less limiting).
void deviceCellLimitGrad(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& U,
    const DeviceBuffer<scalar>& Ubnd,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz,
    scalar k,
    const CellLimitInterface* ifs,
    int nIfs)
{
    int nIfFaces = 0;
    for (int i = 0; i < nIfs; ++i) if (ifs[i].n > 0 && ifs[i].nbrVal) nIfFaces += ifs[i].n;
    if (nIfFaces > 0)
    {
        // The five-phase form: the interface faces have to enter the range BEFORE the widening and the
        // limiter, and the extrapolation limit AFTER the range is complete. Same arithmetic as the
        // single-kernel path below, just split where a scatter can be inserted.
        const int nC = dm.nCells;
        DeviceBuffer<scalar> maxD(nC), minD(nC), lim(nC);
        {
            const scalar* Ud = U.data(); const scalar* Ubndd = Ubnd.data();
            const label* ownerStart = dm.ownerStart.data(); const label* nei = dm.nei.data();
            const label* losort = dm.losort.data(); const label* losortStart = dm.losortStart.data();
            const label* owner = dm.owner.data();
            const label* bndCellStart = dm.bndCellStart.data(); const label* bndPerm = dm.bndPerm.data();
            scalar* maxDd = maxD.data(); scalar* minDd = minD.data();
            pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
                cellLimitMinMaxKernel(nC, Ud, Ubndd, ownerStart, nei, losort, losortStart, owner,
                                       bndCellStart, bndPerm, maxDd, minDd); });
        }
        for (int i = 0; i < nIfs; ++i)
            if (ifs[i].n > 0 && ifs[i].nbrVal)
            {
                const int n = ifs[i].n; const label* ownCell = ifs[i].ownCell; const scalar* nbrVal = ifs[i].nbrVal;
                const scalar* Ud = U.data(); scalar* maxDd = maxD.data(); scalar* minDd = minD.data();
                pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
                    ifMinMaxKernel(n, ownCell, nbrVal, Ud, maxDd, minDd); });
            }
        if (k < 1.0)
        {
            scalar* maxDd = maxD.data(); scalar* minDd = minD.data();
            pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { widenKernel(nC, k, maxDd, minDd); });
        }
        {
            const label* ownerStart = dm.ownerStart.data(); const label* losort = dm.losort.data();
            const label* losortStart = dm.losortStart.data();
            const label* bndCellStart = dm.bndCellStart.data(); const label* bndPerm = dm.bndPerm.data();
            const scalar* dOwnX = dm.dOwnX.data(); const scalar* dOwnY = dm.dOwnY.data(); const scalar* dOwnZ = dm.dOwnZ.data();
            const scalar* dNeiX = dm.dNeiX.data(); const scalar* dNeiY = dm.dNeiY.data(); const scalar* dNeiZ = dm.dNeiZ.data();
            const scalar* dBndX = dm.dBndX.data(); const scalar* dBndY = dm.dBndY.data(); const scalar* dBndZ = dm.dBndZ.data();
            const scalar* maxDd = maxD.data(); const scalar* minDd = minD.data();
            const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
            scalar* limd = lim.data();
            pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
                cellLimitFactorKernel(nC, ownerStart, losort, losortStart, bndCellStart, bndPerm,
                                       dOwnX, dOwnY, dOwnZ, dNeiX, dNeiY, dNeiZ, dBndX, dBndY, dBndZ,
                                       maxDd, minDd, gxd, gyd, gzd, limd); });
        }
        for (int i = 0; i < nIfs; ++i)
            if (ifs[i].n > 0 && ifs[i].nbrVal)
            {
                const int n = ifs[i].n; const label* ownCell = ifs[i].ownCell;
                const scalar* dOwnX = ifs[i].dOwnX; const scalar* dOwnY = ifs[i].dOwnY; const scalar* dOwnZ = ifs[i].dOwnZ;
                const scalar* maxDd = maxD.data(); const scalar* minDd = minD.data();
                const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
                scalar* limd = lim.data();
                pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
                    ifLimitKernel(n, ownCell, dOwnX, dOwnY, dOwnZ, maxDd, minDd, gxd, gyd, gzd, limd); });
            }
        {
            const scalar* limd = lim.data(); scalar* gxd = gx.data(); scalar* gyd = gy.data(); scalar* gzd = gz.data();
            pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { applyLimitKernel(nC, limd, gxd, gyd, gzd); });
        }
        cudaCheck(cudaGetLastError(), "cellLimitGradInterface");
        return;
    }
    {
        const int nC = dm.nCells;
        const scalar* Ud = U.data(); const scalar* Ubndd = Ubnd.data();
        const label* ownerStart = dm.ownerStart.data(); const label* nei = dm.nei.data();
        const label* losort = dm.losort.data(); const label* losortStart = dm.losortStart.data();
        const label* owner = dm.owner.data();
        const label* bndCellStart = dm.bndCellStart.data(); const label* bndPerm = dm.bndPerm.data();
        const scalar* dOwnX = dm.dOwnX.data(); const scalar* dOwnY = dm.dOwnY.data(); const scalar* dOwnZ = dm.dOwnZ.data();
        const scalar* dNeiX = dm.dNeiX.data(); const scalar* dNeiY = dm.dNeiY.data(); const scalar* dNeiZ = dm.dNeiZ.data();
        const scalar* dBndX = dm.dBndX.data(); const scalar* dBndY = dm.dBndY.data(); const scalar* dBndZ = dm.dBndZ.data();
        scalar* gxd = gx.data(); scalar* gyd = gy.data(); scalar* gzd = gz.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            cellLimitGradKernel(nC, k, Ud, Ubndd, ownerStart, nei, losort, losortStart, owner,
                                 bndCellStart, bndPerm, dOwnX, dOwnY, dOwnZ, dNeiX, dNeiY, dNeiZ,
                                 dBndX, dBndY, dBndZ, gxd, gyd, gzd); });
    }
    cudaCheck(cudaGetLastError(), "cellLimitGrad");
}

} // namespace brae
