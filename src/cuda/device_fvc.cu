// cf GPU offload (G3): fvc explicit operators on device. interpolate is per-internal-face; div and
// gaussGrad are per-cell gathers (internal owner/neighbour faces via ownerStart/losort, boundary faces via
// bndCellStart), race-free, deterministic, matching the CPU fvc to machine precision.
#include "device_mesh.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__global__
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


__global__
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


__global__
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
    interpKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(), dm.w.data(), vol.data(), sfInt.data());
    cudaCheck(cudaGetLastError(), "interp");
}


void deviceDiv(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& bval, DeviceBuffer<scalar>& d)
{
    d.resize(dm.nCells);
    divKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                                           phiInt.data(), dm.bndCellStart.data(), dm.bndPerm.data(), bval.data(), dm.V.data(), d.data());
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
    gradKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.owner.data(), dm.nei.data(), dm.w.data(),
                                            dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), vol.data(),
                                            dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                                            dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndGFace.data(), bval.data(),
                                            dm.V.data(), gx.data(), gy.data(), gz.data());
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


__global__
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
    const label* __restrict__ bndIsEmpty,
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
    // EMPTY PATCHES CONTRIBUTE NOTHING, because in OpenFOAM they cannot: emptyFvPatchField is a
    // ZERO-SIZED patch field (emptyFvPatchField.C:41), so there are no faces to evaluate. brae keeps
    // those faces in its addressing and has to skip them explicitly, and the host reference does
    // (cellLimitedGrad_cpp.cu:76 and :116). In the FACE loop it is not harmless: on a 2D mesh Cf - C for
    // an empty face points out of the plane, so the extrapolate is the out-of-plane gradient -- round-off
    // rather than physics -- and r = maxDelta/extrapolate is then a ratio of a real number to noise,
    // which can clamp the limiter far below what any real face asks for. Measured on the host when it
    // was fixed there: grad(U) 1.39e-02 -> 1.28e-14.
    //
    // LATENT ON EVERY REGISTERED FIXTURE, and saying so is the point. rho_ueqn_cuda_cell_limited is the
    // only arm that drives the limiter on a 2D mesh, and it passes at 2e-12 WITH the skip and WITHOUT
    // it. Both loops are inert on pitzDailyTurb for a reason worth writing down: an empty face's stored
    // value equals its cell's, so Ubnd - uc is 0 and cannot move a range that already includes the cell
    // itself at 0; and the mesh is axis-aligned, so dBnd is out-of-plane while the gradient is in it and
    // dBnd . g underflows to a limiter of 1. Neither holds in general -- the host's own measurement
    // above is what a case that breaks the second one costs -- so this brings the device into line with
    // the host and with OpenFOAM's zero-sized empty patch rather than fixing an observed number.
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const int bk = bndPerm[j];
        if (bndIsEmpty[bk]) continue;
        const scalar d = Ubnd[bk] - uc;
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
        if (bndIsEmpty[bk]) continue;   // see the note above -- here it is NOT round-off-harmless
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
__global__
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
    const label* __restrict__ bndIsEmpty,
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
    // Empty patches contribute nothing -- emptyFvPatchField is zero-sized in OpenFOAM, so these faces
    // do not exist there. Same skip as the single-kernel path, and the host reference at
    // cellLimitedGrad_cpp.cu:76.
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    { const int bk = bndPerm[j]; if (bndIsEmpty[bk]) continue;
      const scalar d = Ubnd[bk] - uc; mx = fmax(mx,d); mn = fmin(mn,d); }
    maxD[c] = mx;
    minD[c] = mn;
}

// Phase 2 on interface faces: the coupled neighbour joins the cell's range.
__global__
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
__global__
void widenKernel(int nC, scalar k, scalar* __restrict__ maxD, scalar* __restrict__ minD)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar wdn = (1.0/k - 1.0)*(maxD[c] - minD[c]);
    maxD[c] += wdn;
    minD[c] -= wdn;
}

// Phase 4 on cells: the limiter from the internal + non-coupled boundary face extrapolations.
__global__
void cellLimitFactorKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
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
    // ...and here the skip is load-bearing rather than tidy: Cf - C on an empty face points out of the
    // 2D plane, so this extrapolate is round-off and the ratio against maxDelta can clamp the limiter far
    // below what any real face asks for (cellLimitedGrad_cpp.cu:111-116).
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    { const int bk = bndPerm[j]; if (bndIsEmpty[bk]) continue;
      l = fmin(l, limFace(mx, mn, dBndX[bk]*gcx + dBndY[bk]*gcy + dBndZ[bk]*gcz)); }
    lim[c] = l;
}

// Phase 5 on interface faces: OF's second loop is over EVERY boundary face, not just the uncoupled ones.
__global__
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

__global__
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
        cellLimitMinMaxKernel<<<nBlocks(nC), TPB>>>(nC, U.data(), Ubnd.data(),
            dm.ownerStart.data(), dm.nei.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
            dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(), maxD.data(), minD.data());
        for (int i = 0; i < nIfs; ++i)
            if (ifs[i].n > 0 && ifs[i].nbrVal)
                ifMinMaxKernel<<<nBlocks(ifs[i].n), TPB>>>(ifs[i].n, ifs[i].ownCell, ifs[i].nbrVal,
                                                           U.data(), maxD.data(), minD.data());
        if (k < 1.0) widenKernel<<<nBlocks(nC), TPB>>>(nC, k, maxD.data(), minD.data());
        cellLimitFactorKernel<<<nBlocks(nC), TPB>>>(nC,
            dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
            dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(),
            dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
            dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), maxD.data(), minD.data(),
            gx.data(), gy.data(), gz.data(), lim.data());
        for (int i = 0; i < nIfs; ++i)
            if (ifs[i].n > 0 && ifs[i].nbrVal)
                ifLimitKernel<<<nBlocks(ifs[i].n), TPB>>>(ifs[i].n, ifs[i].ownCell,
                    ifs[i].dOwnX, ifs[i].dOwnY, ifs[i].dOwnZ, maxD.data(), minD.data(),
                    gx.data(), gy.data(), gz.data(), lim.data());
        applyLimitKernel<<<nBlocks(nC), TPB>>>(nC, lim.data(), gx.data(), gy.data(), gz.data());
        cudaCheck(cudaGetLastError(), "cellLimitGradInterface");
        return;
    }
    cellLimitGradKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, k, U.data(), Ubnd.data(),
        dm.ownerStart.data(), dm.nei.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
        dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), gx.data(), gy.data(), gz.data());
    cudaCheck(cudaGetLastError(), "cellLimitGrad");
}

} // namespace brae
