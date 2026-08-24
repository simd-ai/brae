// cf GPU offload, cyclic (periodic) interface kernels. See device_cyclic.cuh. Both sides of each periodic
// pair are stored with their OWN outward Sf and own flux, so the coupling/continuity are symmetric and
// conservative automatically (the N-side flux = -the O-side flux; the N-side weight = 1 - the O-side weight).
// Sign convention matches device_fvm.cu: off-diagonal ifCoeff is added in Amul as Apsi[own]+=ifCoeff*psi[nbr];
// diag[own] -= ifCoeff (Laplacian) / += max(phi,0) (upwind convection).
#include "device_cyclic.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {
namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__device__
void laplKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ gamma,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    scalar* __restrict__ ifCoeff,
    scalar* __restrict__ diag,
    int addToDiag)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const scalar gf = w[j] * gamma[own[j]] + (1.0 - w[j]) * gamma[nbr[j]];   // gamma interpolated to the face
    const scalar c = gf * dc[j] * magSf[j];
    ifCoeff[j] = c;
    if (addToDiag) atomicAdd(&diag[own[j]], -c);
}


__device__
void convKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ phi,
    scalar* __restrict__ ifCoeff,
    scalar* __restrict__ diag)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const scalar p = phi[j];
    ifCoeff[j] += (p < 0.0) ? p : 0.0;                  // off-diag += min(phi,0)
    atomicAdd(&diag[own[j]], (p > 0.0) ? p : 0.0);      // diag    += max(phi,0)
}


__device__
void momKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ nu,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ phi,
    scalar* __restrict__ ifCoeff,
    scalar* __restrict__ diag)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const scalar nf = w[j] * nu[own[j]] + (1.0 - w[j]) * nu[nbr[j]];
    const scalar lap = nf * dc[j] * magSf[j];           // diffusion magnitude (>0)
    const scalar p = phi[j];
    ifCoeff[j] = -lap + ((p < 0.0) ? p : 0.0);          // off-diag: -laplacian + min(phi,0)
    atomicAdd(&diag[own[j]], lap + ((p > 0.0) ? p : 0.0));   // diag: +laplacian + max(phi,0)
}


__device__
void addHKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifCoeff,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ V,
    scalar* __restrict__ H)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    atomicAdd(&H[own[j]], -ifCoeff[j] * psi[nbr[j]] / V[own[j]]);
}


__device__
void offSumKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ ifCoeff,
    scalar* __restrict__ sumOff)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    atomicAdd(&sumOff[own[j]], fabs(ifCoeff[j]));
}


__device__
void fluxKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Hx,
    const scalar* __restrict__ Hy,
    const scalar* __restrict__ Hz,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    scalar* __restrict__ phi)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    const scalar wj = w[j], wn = 1.0 - wj;
    const scalar fx = wj * Hx[o] + wn * Hx[nb], fy = wj * Hy[o] + wn * Hy[nb], fz = wj * Hz[o] + wn * Hz[nb];
    phi[j] = fx * Sfx[j] + fy * Sfy[j] + fz * Sfz[j];
}


__device__
void divAddKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ V,
    scalar* __restrict__ div)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    atomicAdd(&div[own[j]], phi[j] / V[own[j]]);   // deviceDiv returns the volume-normalized divergence (Sum phi / V)
}


__device__
void fluxCorrKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifCoeff,
    const scalar* __restrict__ p,
    scalar* __restrict__ phi)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    phi[j] -= ifCoeff[j] * (p[nbr[j]] - p[own[j]]);     // snGrad(p) flux: -coeff*(p_nbr - p_own)
}


__device__
void cycTensorDivKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ sigmaC,
    int nC,
    const scalar* __restrict__ fT,
    int rotational,
    scalar* __restrict__ dX,
    scalar* __restrict__ dY,
    scalar* __restrict__ dZ)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int o = own[j], nb = nbr[j];
    const scalar wf = w[j], sx = Sfx[j], sy = Sfy[j], sz = Sfz[j];
    // neighbour stress tensor; for ROTATIONAL cyclic a rank-2 tensor transforms as sigma' = R*sigma*R^T (NOT the
    // vector rotation) before interpolation to the face, else the cyclic divDevReff injects a spurious stress.
    scalar sn[9];
    for (int q = 0; q < 9; ++q)
        sn[q] = sigmaC[q*nC + nb];
    if (rotational)
    {
        scalar R[9];
        for (int q = 0; q < 9; ++q)
            R[q] = fT[q*n + j];
        scalar M[9];   // M = R*sigma
        for (int i = 0; i < 3; ++i)
            for (int l = 0; l < 3; ++l)
            {
                scalar s = 0;
                for (int k = 0; k < 3; ++k)
                    s += R[3*i+k]*sn[3*k+l];
                M[3*i+l] = s;
            }
        scalar sr[9];   // sigma' = M*R^T : sr[i][jj] = sum_l M[i][l]*R[jj][l]
        for (int i = 0; i < 3; ++i)
            for (int jj = 0; jj < 3; ++jj)
            {
                scalar s = 0;
                for (int l = 0; l < 3; ++l)
                    s += M[3*i+l]*R[3*jj+l];
                sr[3*i+jj] = s;
            }
        for (int q = 0; q < 9; ++q)
            sn[q] = sr[q];
    }
    scalar d[3];
    for (int jc = 0; jc < 3; ++jc)   // sigma_face = w*sigma[own] + (1-w)*sigma'[nbr]
    {
        const scalar s0 = wf*sigmaC[(0*3+jc)*nC+o] + (1.0-wf)*sn[0*3+jc];
        const scalar s1 = wf*sigmaC[(1*3+jc)*nC+o] + (1.0-wf)*sn[1*3+jc];
        const scalar s2 = wf*sigmaC[(2*3+jc)*nC+o] + (1.0-wf)*sn[2*3+jc];
        d[jc] = sx*s0 + sy*s1 + sz*s2;
    }
    // raw sum (= V*fvc::div), like tensorDivKernel
    atomicAdd(&dX[o], d[0]);
    atomicAdd(&dY[o], d[1]);
    atomicAdd(&dZ[o], d[2]);
}


__device__
void gradAddKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j];
    const scalar fv = (w[j] * psi[o] + (1.0 - w[j]) * psi[nbr[j]]) / V[o];
    atomicAdd(&gx[o], Sfx[j] * fv);
    atomicAdd(&gy[o], Sfy[j] * fv);
    atomicAdd(&gz[o], Sfz[j] * fv);
}


// rotational helpers. forwardT packed (3*i+j)*n + face ; (forwardT.v)[i] = sum_k fT[(3i+k)n+j]*v_k
__device__ __forceinline__
void rotNbr(const scalar* fT, int n, int j, scalar vx, scalar vy, scalar vz,
            scalar& rx, scalar& ry, scalar& rz)
{
    rx = fT[(0*3+0)*n+j]*vx + fT[(0*3+1)*n+j]*vy + fT[(0*3+2)*n+j]*vz;
    ry = fT[(1*3+0)*n+j]*vx + fT[(1*3+1)*n+j]*vy + fT[(1*3+2)*n+j]*vz;
    rz = fT[(2*3+0)*n+j]*vx + fT[(2*3+1)*n+j]*vy + fT[(2*3+2)*n+j]*vz;
}


__device__
void scaleImplicitKernel(
    int n,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ fT,
    scalar* __restrict__ ic0,
    scalar* __restrict__ ic1,
    scalar* __restrict__ ic2)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    ic0[j] = ifc[j] * fT[(0*3+0)*n+j];   // * forwardT[0][0]
    ic1[j] = ifc[j] * fT[(1*3+1)*n+j];   // * forwardT[1][1]
    ic2[j] = ifc[j] * fT[(2*3+2)*n+j];   // * forwardT[2][2]
}


__device__
void fluxRotKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Hx,
    const scalar* __restrict__ Hy,
    const scalar* __restrict__ Hz,
    const scalar* __restrict__ fT,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    scalar* __restrict__ phi)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    const scalar wj = w[j], wn = 1.0 - wj;
    scalar rx, ry, rz;
    rotNbr(fT, n, j, Hx[nb], Hy[nb], Hz[nb], rx, ry, rz);
    const scalar fx = wj*Hx[o] + wn*rx, fy = wj*Hy[o] + wn*ry, fz = wj*Hz[o] + wn*rz;
    phi[j] = fx*Sfx[j] + fy*Sfy[j] + fz*Sfz[j];
}


__device__
void addHRotKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ fT,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ V,
    scalar* __restrict__ Hx,
    scalar* __restrict__ Hy,
    scalar* __restrict__ Hz)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    scalar rx, ry, rz;
    rotNbr(fT, n, j, Ux[nb], Uy[nb], Uz[nb], rx, ry, rz);
    const scalar c = ifc[j] / V[o];
    atomicAdd(&Hx[o], -c*rx);
    atomicAdd(&Hy[o], -c*ry);
    atomicAdd(&Hz[o], -c*rz);
}


__device__
void gradRotKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ fT,
    int comp,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    scalar rx, ry, rz;
    rotNbr(fT, n, j, Ux[nb], Uy[nb], Uz[nb], rx, ry, rz);
    const scalar uOwn = (comp==0)?Ux[o]:(comp==1)?Uy[o]:Uz[o];
    const scalar uNbrR = (comp==0)?rx:(comp==1)?ry:rz;            // (forwardT.U[nbr])[comp]
    const scalar fv = (w[j]*uOwn + (1.0-w[j])*uNbrR) / V[o];
    atomicAdd(&gx[o], Sfx[j]*fv);
    atomicAdd(&gy[o], Sfy[j]*fv);
    atomicAdd(&gz[o], Sfz[j]*fv);
}


__device__
void deferredRotKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ fT,
    int comp,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ src)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    scalar rx, ry, rz;
    rotNbr(fT, n, j, Ux[nb], Uy[nb], Uz[nb], rx, ry, rz);
    const scalar uNbrComp = (comp==0)?Ux[nb]:(comp==1)?Uy[nb]:Uz[nb];
    const scalar diag     = fT[(3*comp+comp)*n+j];               // forwardT[comp][comp]
    const scalar rComp    = (comp==0)?rx:(comp==1)?ry:rz;        // (forwardT.U[nbr])[comp]
    atomicAdd(&src[o], -ifc[j] * (rComp - diag*uNbrComp));       // -ifc*(full - diag) mixing
}


__device__
void addHDiagKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifcC,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ V,
    scalar* __restrict__ H)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    atomicAdd(&H[own[j]], -ifcC[j] * psi[nbr[j]] / V[own[j]]);   // diag cyclic off-diag (ifCoeffC[comp])
}
} // namespace


void deviceCyclicAssembleLaplacian(
    DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& gammaCell,
    DeviceBuffer<scalar>& diag,
    bool addToDiag)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* gamma = gammaCell.data(); const scalar* dc = cyc.deltaCoeffs.data();
        const scalar* w = cyc.weights.data(); const scalar* magSf = cyc.magSf.data();
        scalar* ifc = cyc.ifCoeff.data(); scalar* diagd = diag.data(); const int addD = addToDiag ? 1 : 0;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            laplKernel(n, own, nbr, gamma, dc, w, magSf, ifc, diagd, addD); });
    }
    cudaCheck(cudaGetLastError(), "cyclicLapl");
}


void deviceCyclicAddConvection(DeviceCyclic& cyc, DeviceBuffer<scalar>& diag)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const scalar* phi = cyc.phi.data();
        scalar* ifc = cyc.ifCoeff.data(); scalar* diagd = diag.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { convKernel(n, own, phi, ifc, diagd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicConv");
}


void deviceCyclicAssembleMomentum(DeviceCyclic& cyc, const DeviceBuffer<scalar>& nuEffCell, DeviceBuffer<scalar>& diag)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* nu = nuEffCell.data(); const scalar* dc = cyc.deltaCoeffs.data();
        const scalar* w = cyc.weights.data(); const scalar* magSf = cyc.magSf.data(); const scalar* phi = cyc.phi.data();
        scalar* ifc = cyc.ifCoeff.data(); scalar* diagd = diag.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            momKernel(n, own, nbr, nu, dc, w, magSf, phi, ifc, diagd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicMom");
}


void deviceCyclicAddH(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& H)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* ifc = cyc.ifCoeff.data(); const scalar* psid = psi.data(); const scalar* Vd = V.data();
        scalar* Hd = H.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { addHKernel(n, own, nbr, ifc, psid, Vd, Hd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicAddH");
}


void deviceCyclicOffDiagSum(const DeviceCyclic& cyc, DeviceBuffer<scalar>& sumOff)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const scalar* ifc = cyc.ifCoeff.data();
        scalar* sumOffd = sumOff.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { offSumKernel(n, own, ifc, sumOffd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicOffSum");
}


void deviceCyclicFlux(
    DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Hx,
    const DeviceBuffer<scalar>& Hy,
    const DeviceBuffer<scalar>& Hz)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* w = cyc.weights.data();
        const scalar* Hxd = Hx.data(); const scalar* Hyd = Hy.data(); const scalar* Hzd = Hz.data();
        const scalar* sfx = cyc.Sfx.data(); const scalar* sfy = cyc.Sfy.data(); const scalar* sfz = cyc.Sfz.data();
        scalar* phid = cyc.phi.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            fluxKernel(n, own, nbr, w, Hxd, Hyd, Hzd, sfx, sfy, sfz, phid); });
    }
    cudaCheck(cudaGetLastError(), "cyclicFlux");
}


void deviceCyclicAddDiv(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const scalar* phi = cyc.phi.data();
        const scalar* Vd = V.data(); scalar* divd = div.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { divAddKernel(n, own, phi, Vd, divd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicDiv");
}


namespace {
__device__
void cycZeroWallKernel(int n, const label* __restrict__ own, const label* __restrict__ isW, scalar* __restrict__ ifc)
{
    const int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j<n && isW[own[j]]) ifc[j]=0.0;
}
} // namespace


// epsilon setValues: a wall cell's eps is fixed (eps0); zero the cyclic interface off-diagonal for wall-cell owners.
void deviceCyclicZeroWallIfCoeff(DeviceCyclic& cyc, const DeviceBuffer<label>& isWallCell)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* isW = isWallCell.data();
        scalar* ifc = cyc.ifCoeff.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { cycZeroWallKernel(n, own, isW, ifc); });
    }
    cudaCheck(cudaGetLastError(), "cyclicZeroWall");
}


void deviceCyclicCorrectFlux(DeviceCyclic& cyc, const DeviceBuffer<scalar>& p)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* ifc = cyc.ifCoeff.data(); const scalar* pd = p.data(); scalar* phid = cyc.phi.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { fluxCorrKernel(n, own, nbr, ifc, pd, phid); });
    }
    cudaCheck(cudaGetLastError(), "cyclicFluxCorr");
}


namespace {
// face value of a CELL field on a cyclic face: w*psi[own] + (1-w)*psi[nbr] -- fvc::interpolate on a
// coupled patch, the 1:1 counterpart of deviceAmiFaceValue.
__device__
void cyclicFaceValueKernel(int n, const label* __restrict__ own, const label* __restrict__ nbr,
                           const scalar* __restrict__ w, const scalar* __restrict__ cell,
                           scalar* __restrict__ out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = w[i]*cell[own[i]] + (scalar(1) - w[i])*cell[nbr[i]];
}
}   // namespace

namespace {
// The RAW periodic-neighbour value per cyclic face -- OF's patchNeighbourField(), not a face
// interpolation. cellLimitedGrad needs it to fold the coupled neighbour into a cell's min/max range.
// Rotational: the neighbour vector is rotated by forwardT first, so component `comp` mixes all three.
__device__
void cyclicNbrValueKernel(int n, const label* __restrict__ nbr, const scalar* __restrict__ cell,
                          const scalar* __restrict__ c0, const scalar* __restrict__ c1,
                          const scalar* __restrict__ c2, const scalar* __restrict__ fT,
                          int rotational, int comp, scalar* __restrict__ out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int nb = nbr[i];
    if (!rotational) { out[i] = cell[nb]; return; }
    out[i] = fT[(3*comp+0)*n + i]*c0[nb] + fT[(3*comp+1)*n + i]*c1[nb] + fT[(3*comp+2)*n + i]*c2[nb];
}
}   // namespace

void deviceCyclicNbrValue(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& cell,
                          const DeviceBuffer<scalar>& c0, const DeviceBuffer<scalar>& c1,
                          const DeviceBuffer<scalar>& c2, int comp, DeviceBuffer<scalar>& out)
{
    out.resize(cyc.n);
    if (cyc.n == 0) return;
    const bool rot = cyc.rotational && c0.size() && c1.size() && c2.size();
    {
        const int n = cyc.n; const label* nbr = cyc.nbrCell.data(); const scalar* celld = cell.data();
        const scalar* c0d = rot ? c0.data() : nullptr; const scalar* c1d = rot ? c1.data() : nullptr;
        const scalar* c2d = rot ? c2.data() : nullptr;
        const scalar* fT = cyc.rotational ? cyc.fT.data() : nullptr; const int rotD = cyc.rotational ? 1 : 0;
        const int compD = comp; scalar* outd = out.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            cyclicNbrValueKernel(n, nbr, celld, c0d, c1d, c2d, fT, rotD, compD, outd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicNbrValue");
}


void deviceCyclicFaceValue(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& cell, DeviceBuffer<scalar>& out)
{
    out.resize(cyc.n);
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* w = cyc.weights.data(); const scalar* celld = cell.data(); scalar* outd = out.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            cyclicFaceValueKernel(n, own, nbr, w, celld, outd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicFaceValue");
}


void deviceCyclicAddGrad(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* w = cyc.weights.data(); const scalar* psid = psi.data();
        const scalar* sfx = cyc.Sfx.data(); const scalar* sfy = cyc.Sfy.data(); const scalar* sfz = cyc.Sfz.data();
        const scalar* Vd = V.data(); scalar* gxd = gx.data(); scalar* gyd = gy.data(); scalar* gzd = gz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            gradAddKernel(n, own, nbr, w, psid, sfx, sfy, sfz, Vd, gxd, gyd, gzd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicGrad");
}


void deviceCyclicScaleImplicit(DeviceCyclic& cyc)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const scalar* ifc = cyc.ifCoeff.data(); const scalar* fT = cyc.fT.data();
        scalar* ic0 = cyc.ifCoeffC[0].data(); scalar* ic1 = cyc.ifCoeffC[1].data(); scalar* ic2 = cyc.ifCoeffC[2].data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            scaleImplicitKernel(n, ifc, fT, ic0, ic1, ic2); });
    }
    cudaCheck(cudaGetLastError(), "cyclicScaleImplicit");
}


void deviceCyclicFluxRot(
    DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Hx,
    const DeviceBuffer<scalar>& Hy,
    const DeviceBuffer<scalar>& Hz)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* w = cyc.weights.data();
        const scalar* Hxd = Hx.data(); const scalar* Hyd = Hy.data(); const scalar* Hzd = Hz.data();
        const scalar* fT = cyc.fT.data();
        const scalar* sfx = cyc.Sfx.data(); const scalar* sfy = cyc.Sfy.data(); const scalar* sfz = cyc.Sfz.data();
        scalar* phid = cyc.phi.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            fluxRotKernel(n, own, nbr, w, Hxd, Hyd, Hzd, fT, sfx, sfy, sfz, phid); });
    }
    cudaCheck(cudaGetLastError(), "cyclicFluxRot");
}


void deviceCyclicAddHRot(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& Hx,
    DeviceBuffer<scalar>& Hy,
    DeviceBuffer<scalar>& Hz)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* ifc = cyc.ifCoeff.data(); const scalar* fT = cyc.fT.data();
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        const scalar* Vd = V.data(); scalar* Hxd = Hx.data(); scalar* Hyd = Hy.data(); scalar* Hzd = Hz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            addHRotKernel(n, own, nbr, ifc, fT, Uxd, Uyd, Uzd, Vd, Hxd, Hyd, Hzd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicAddHRot");
}


void deviceCyclicAddGradRot(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    int comp,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* w = cyc.weights.data(); const scalar* fT = cyc.fT.data(); const int compD = comp;
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        const scalar* sfx = cyc.Sfx.data(); const scalar* sfy = cyc.Sfy.data(); const scalar* sfz = cyc.Sfz.data();
        const scalar* Vd = V.data(); scalar* gxd = gx.data(); scalar* gyd = gy.data(); scalar* gzd = gz.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            gradRotKernel(n, own, nbr, w, fT, compD, Uxd, Uyd, Uzd, sfx, sfy, sfz, Vd, gxd, gyd, gzd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicGradRot");
}


void deviceCyclicAddDeferredRot(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    int comp,
    DeviceBuffer<scalar>& src)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* ifc = cyc.ifCoeff.data(); const scalar* fT = cyc.fT.data(); const int compD = comp;
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        scalar* srcd = src.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            deferredRotKernel(n, own, nbr, ifc, fT, compD, Uxd, Uyd, Uzd, srcd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicDeferredRot");
}


void deviceCyclicAddHDiag(
    const DeviceCyclic& cyc,
    int comp,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& H)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* ifcC = cyc.ifCoeffC[comp].data();
        const scalar* psid = psi.data(); const scalar* Vd = V.data(); scalar* Hd = H.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            addHDiagKernel(n, own, nbr, ifcC, psid, Vd, Hd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicAddHDiag");
}


__device__
void cycLinUpwindKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ gx0,
    const scalar* __restrict__ gy0,
    const scalar* __restrict__ gz0,
    const scalar* __restrict__ gx1,
    const scalar* __restrict__ gy1,
    const scalar* __restrict__ gz1,
    const scalar* __restrict__ gx2,
    const scalar* __restrict__ gy2,
    const scalar* __restrict__ gz2,
    const scalar* __restrict__ dox,
    const scalar* __restrict__ doy,
    const scalar* __restrict__ doz,
    const scalar* __restrict__ dnx,
    const scalar* __restrict__ dny,
    const scalar* __restrict__ dnz,
    const scalar* __restrict__ fT,
    int rotational,
    int comp,
    scalar* __restrict__ corr)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int o = own[j], nb = nbr[j];
    const scalar pf = phi[j];
    scalar c;
    if (pf >= 0)   // own is upwind: grad(U_comp)[own] . dOwn  (no rotation)
    {
        const scalar* gx = (comp==0)?gx0:(comp==1)?gx1:gx2;
        const scalar* gy = (comp==0)?gy0:(comp==1)?gy1:gy2;
        const scalar* gz = (comp==0)?gz0:(comp==1)?gz1:gz2;
        c = pf * (gx[o]*dox[j] + gy[o]*doy[j] + gz[o]*doz[j]);
    }
    else   // nbr is upwind: forwardT . (gradU[nbr] . dNbr), take comp
    {
        const scalar rl0 = gx0[nb]*dnx[j] + gy0[nb]*dny[j] + gz0[nb]*dnz[j];   // grad(U_l)[nbr] . dNbr (nbr frame)
        const scalar rl1 = gx1[nb]*dnx[j] + gy1[nb]*dny[j] + gz1[nb]*dnz[j];
        const scalar rl2 = gx2[nb]*dnx[j] + gy2[nb]*dny[j] + gz2[nb]*dnz[j];
        c = rotational ? pf * (fT[(3*comp+0)*n+j]*rl0 + fT[(3*comp+1)*n+j]*rl1 + fT[(3*comp+2)*n+j]*rl2)
                       : pf * ((comp==0)?rl0:(comp==1)?rl1:rl2);
    }
    atomicAdd(&corr[o], c);                             // own is the "+" side (mirrors the internal owner pass)
}


void deviceCyclicAddLinUpwindCorr(
    const DeviceCyclic& cyc,
    int comp,
    const DeviceBuffer<scalar>* gUx,
    const DeviceBuffer<scalar>* gUy,
    const DeviceBuffer<scalar>* gUz,
    DeviceBuffer<scalar>& corr)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* phi = cyc.phi.data();
        const scalar* gx0 = gUx[0].data(); const scalar* gy0 = gUy[0].data(); const scalar* gz0 = gUz[0].data();
        const scalar* gx1 = gUx[1].data(); const scalar* gy1 = gUy[1].data(); const scalar* gz1 = gUz[1].data();
        const scalar* gx2 = gUx[2].data(); const scalar* gy2 = gUy[2].data(); const scalar* gz2 = gUz[2].data();
        const scalar* dox = cyc.dOwnX.data(); const scalar* doy = cyc.dOwnY.data(); const scalar* doz = cyc.dOwnZ.data();
        const scalar* dnx = cyc.dNbrX.data(); const scalar* dny = cyc.dNbrY.data(); const scalar* dnz = cyc.dNbrZ.data();
        const scalar* fT = cyc.rotational ? cyc.fT.data() : nullptr; const int rot = cyc.rotational ? 1 : 0;
        const int compD = comp; scalar* corrd = corr.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            cycLinUpwindKernel(n, own, nbr, phi, gx0, gy0, gz0, gx1, gy1, gz1, gx2, gy2, gz2,
                                dox, doy, doz, dnx, dny, dnz, fT, rot, compD, corrd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicLinUpwind");
}

// SCALAR overload -- see the AMI one. comp = 0, rotation off, one gradient in all three slots.
void deviceCyclicAddLinUpwindCorr(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& corr)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* phi = cyc.phi.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        const scalar* dox = cyc.dOwnX.data(); const scalar* doy = cyc.dOwnY.data(); const scalar* doz = cyc.dOwnZ.data();
        const scalar* dnx = cyc.dNbrX.data(); const scalar* dny = cyc.dNbrY.data(); const scalar* dnz = cyc.dNbrZ.data();
        scalar* corrd = corr.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            cycLinUpwindKernel(n, own, nbr, phi, gxd, gyd, gzd, gxd, gyd, gzd, gxd, gyd, gzd,
                                dox, doy, doz, dnx, dny, dnz, nullptr, 0, 0, corrd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicLinUpwindScalar");
}


__device__
void cycLapCorrKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ nu,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ gx0,
    const scalar* __restrict__ gy0,
    const scalar* __restrict__ gz0,
    const scalar* __restrict__ gx1,
    const scalar* __restrict__ gy1,
    const scalar* __restrict__ gz1,
    const scalar* __restrict__ gx2,
    const scalar* __restrict__ gy2,
    const scalar* __restrict__ gz2,
    const scalar* __restrict__ fT,
    int rotational,
    int comp,
    scalar* __restrict__ src)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int o = own[j], nb = nbr[j];
    const scalar wf = w[j], wn = 1.0 - wf;
    const scalar* gxc = (comp==0)?gx0:(comp==1)?gx1:gx2;
    const scalar* gyc = (comp==0)?gy0:(comp==1)?gy1:gy2;
    const scalar* gzc = (comp==0)?gz0:(comp==1)?gz1:gz2;
    scalar gnx, gny, gnz;                              // grad(U_comp)[nbr] in the OWN frame (rotated)
    if (rotational)                                    // grad((R.U)_comp) = (R . gradU[nbr] . R^T)[comp][:]
    {
        const scalar G[9] = { gx0[nb],gy0[nb],gz0[nb], gx1[nb],gy1[nb],gz1[nb], gx2[nb],gy2[nb],gz2[nb] };
        scalar R[9];
        for (int q = 0; q < 9; ++q)
            R[q] = fT[q*n + j];
        scalar gn[3];
        for (int jj = 0; jj < 3; ++jj)
        {
            scalar s = 0;
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k)
                    s += R[3*comp+i]*G[3*i+k]*R[3*jj+k];
            gn[jj] = s;
        }
        gnx = gn[0]; gny = gn[1]; gnz = gn[2];
    }
    else { gnx = gxc[nb]; gny = gyc[nb]; gnz = gzc[nb]; }
    const scalar gfx = wf*gxc[o] + wn*gnx, gfy = wf*gyc[o] + wn*gny, gfz = wf*gzc[o] + wn*gnz;
    const scalar gammaf = wf*nu[o] + wn*nu[nb];
    const scalar ffc = gammaf * magSf[j] * (cvx[j]*gfx + cvy[j]*gfy + cvz[j]*gfz);
    atomicAdd(&src[o], -ffc);                          // owner contribution (gather: src[c] -= ffc for owned faces)
}


void deviceCyclicAddLapCorr(
    const DeviceCyclic& cyc,
    int comp,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>* gUx,
    const DeviceBuffer<scalar>* gUy,
    const DeviceBuffer<scalar>* gUz,
    DeviceBuffer<scalar>& corr)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* nu = gammaCell.data(); const scalar* w = cyc.weights.data(); const scalar* magSf = cyc.magSf.data();
        const scalar* cvx = cyc.corrVecX.data(); const scalar* cvy = cyc.corrVecY.data(); const scalar* cvz = cyc.corrVecZ.data();
        const scalar* gx0 = gUx[0].data(); const scalar* gy0 = gUy[0].data(); const scalar* gz0 = gUz[0].data();
        const scalar* gx1 = gUx[1].data(); const scalar* gy1 = gUy[1].data(); const scalar* gz1 = gUz[1].data();
        const scalar* gx2 = gUx[2].data(); const scalar* gy2 = gUy[2].data(); const scalar* gz2 = gUz[2].data();
        const scalar* fT = cyc.rotational ? cyc.fT.data() : nullptr; const int rot = cyc.rotational ? 1 : 0;
        const int compD = comp; scalar* srcd = corr.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            cycLapCorrKernel(n, own, nbr, nu, w, magSf, cvx, cvy, cvz,
                              gx0, gy0, gz0, gx1, gy1, gz1, gx2, gy2, gz2, fT, rot, compD, srcd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicLapCorr");
}

void deviceCyclicAddLapCorr(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& corr)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* nu = gammaCell.data(); const scalar* w = cyc.weights.data(); const scalar* magSf = cyc.magSf.data();
        const scalar* cvx = cyc.corrVecX.data(); const scalar* cvy = cyc.corrVecY.data(); const scalar* cvz = cyc.corrVecZ.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        scalar* srcd = corr.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            cycLapCorrKernel(n, own, nbr, nu, w, magSf, cvx, cvy, cvz,
                              gxd, gyd, gzd, gxd, gyd, gzd, gxd, gyd, gzd, nullptr, 0, 0, srcd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicLapCorrScalar");
}


__device__
void cycLapCorrScalarKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ gamma,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    scalar* __restrict__ bp,
    scalar* __restrict__ ffcOut)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int o = own[j], nb = nbr[j];
    const scalar wf = w[j], wn = 1.0 - wf;
    const scalar gfx = wf*gx[o]+wn*gx[nb], gfy = wf*gy[o]+wn*gy[nb], gfz = wf*gz[o]+wn*gz[nb];   // grad(p)_face (scalar)
    const scalar gammaf = wf*gamma[o] + wn*gamma[nb];                                            // rAtU_face
    const scalar ffc = gammaf * magSf[j] * (cvx[j]*gfx + cvy[j]*gfy + cvz[j]*gfz);
    ffcOut[j] = ffc;                                    // for the post-solve flux correction (cyc.phi -= ffc)
    atomicAdd(&bp[o], -ffc);                            // -V*div(ffc): owner contribution to the pressure source
}


void deviceCyclicLapCorrP(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& bp,
    DeviceBuffer<scalar>& ffcOut)
{
    if (cyc.n == 0) return;
    ffcOut.resize(cyc.n);
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* gamma = gammaCell.data(); const scalar* w = cyc.weights.data(); const scalar* magSf = cyc.magSf.data();
        const scalar* cvx = cyc.corrVecX.data(); const scalar* cvy = cyc.corrVecY.data(); const scalar* cvz = cyc.corrVecZ.data();
        const scalar* gxd = gx.data(); const scalar* gyd = gy.data(); const scalar* gzd = gz.data();
        scalar* bpd = bp.data(); scalar* ffcOutd = ffcOut.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            cycLapCorrScalarKernel(n, own, nbr, gamma, w, magSf, cvx, cvy, cvz, gxd, gyd, gzd, bpd, ffcOutd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicLapCorrP");
}


void deviceCyclicAddTensorDiv(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& sigmaC,
    int nC,
    DeviceBuffer<scalar>& srcX,
    DeviceBuffer<scalar>& srcY,
    DeviceBuffer<scalar>& srcZ)
{
    if (cyc.n == 0) return;
    {
        const int n = cyc.n; const label* own = cyc.ownCell.data(); const label* nbr = cyc.nbrCell.data();
        const scalar* w = cyc.weights.data();
        const scalar* sfx = cyc.Sfx.data(); const scalar* sfy = cyc.Sfy.data(); const scalar* sfz = cyc.Sfz.data();
        const scalar* sigmaCd = sigmaC.data(); const int nCd = nC;
        const scalar* fT = cyc.rotational ? cyc.fT.data() : nullptr; const int rot = cyc.rotational ? 1 : 0;
        scalar* dXd = srcX.data(); scalar* dYd = srcY.data(); scalar* dZd = srcZ.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            cycTensorDivKernel(n, own, nbr, w, sfx, sfy, sfz, sigmaCd, nCd, fT, rot, dXd, dYd, dZd); });
    }
    cudaCheck(cudaGetLastError(), "cyclicTensorDiv");
}

} // namespace brae
