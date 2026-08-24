// cf GPU offload (G7): SIMPLE coupling-glue kernels. matrixH and matrixFlux reuse the ownerStart/losort
// gather; rAU / corrector are elementwise; the HbyA flux is a per-face vector interpolation dotted with Sf.
#include "device_simple.cuh"
#include "pcuda_compat.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

__device__
void matrixHKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ source,
    const scalar* __restrict__ V,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ bdDiag,
    const scalar* __restrict__ bdSrc,
    scalar* __restrict__ H)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar h = source[c];
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
        h -= upper[f] * psi[nei[f]];   // -upper*psi[nei] (owned faces)
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        h -= lower[f] * psi[own[f]];  // -lower*psi[own]
    }
    scalar bd = 0, bs = 0;
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int kk = bndPerm[k];
        bd += bdDiag[kk];
        bs += bdSrc[kk];
    }
    h += bd * psi[c] + bs;
    H[c] = h / V[c];
}
__device__
void reciprocalVKernel(
    int nC,
    const scalar* __restrict__ diagC,
    const scalar* __restrict__ V,
    scalar* __restrict__ rAU)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) rAU[c] = V[c] / diagC[c];
}
__device__
void vectorFluxKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ phi)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar wf = w[f];
    const int o = own[f], n = nei[f];
    const scalar ux = wf * Ux[o] + (1.0 - wf) * Ux[n];
    const scalar uy = wf * Uy[o] + (1.0 - wf) * Uy[n];
    const scalar uz = wf * Uz[o] + (1.0 - wf) * Uz[n];
    phi[f] = ux * Sfx[f] + uy * Sfy[f] + uz * Sfz[f];
}
__device__
void matrixFluxKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    const scalar* __restrict__ p,
    scalar* __restrict__ flux)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf) flux[f] = upper[f] * p[nei[f]] - lower[f] * p[own[f]];
}
__device__
void correctorKernel(
    int nC,
    const scalar* __restrict__ HbyA,
    const scalar* __restrict__ rAU,
    const scalar* __restrict__ gradP,
    scalar* __restrict__ U)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) U[c] = HbyA[c] - rAU[c] * gradP[c];
}
__device__
void bndFluxKernel(
    int nB,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ uxb,
    const scalar* __restrict__ uyb,
    const scalar* __restrict__ uzb,
    scalar* __restrict__ phiB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    const int f = bndGFace[i];
    phiB[i] = uxb[i] * Sfx[f] + uyb[i] * Sfy[f] + uzb[i] * Sfz[f];
}
__device__
void relaxKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ iCbnd,
    const scalar* __restrict__ rawDiag,
    scalar alpha,
    scalar* __restrict__ relaxedDiag,
    scalar* __restrict__ delta,
    const scalar* __restrict__ cycSumOff,
    const scalar* __restrict__ iCmaxMag,
    const scalar* __restrict__ iCmin)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar sumOff = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
        sumOff += fabs(upper[f]);             // |upper| over owned faces
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
        sumOff += fabs(lower[losort[k]]);   // |lower| over neighbour faces
    if (cycSumOff) sumOff += cycSumOff[c];                                                        // |cyclic off-diagonal|
    scalar magSum = 0.0, minSum = 0.0;
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int p = bndPerm[k];
        magSum += iCmaxMag ? iCmaxMag[p] : fabs(iCbnd[p]);   // OF relax: cmptMax(cmptMag(iC)) over comps (= |iC0| when equal)
        minSum += iCmin ? iCmin[p] : iCbnd[p];               // OF relax REMOVES cmptMin(iC) -- signed, across comps
    }
    const scalar d0 = rawDiag[c];
    const scalar dr = fmax(fabs(d0 + magSum), sumOff) / alpha - minSum;
    relaxedDiag[c] = dr;
    delta[c] = dr - d0;
}
} // namespace

void deviceMatrixH(
    const DeviceLduView& A,
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& psiK,
    const DeviceBuffer<scalar>& sourceK,
    const DeviceBuffer<scalar>& bdDiagK,
    const DeviceBuffer<scalar>& bdSrcK,
    DeviceBuffer<scalar>& Hk)
{
    const int nC = dm.nCells;
    Hk.resize(nC);
    {
        const label *ownerStart=A.ownerStart, *losort=A.losort, *losortStart=A.losortStart, *own=A.owner, *nei=A.nei;
        const scalar *upper=A.upper, *lower=A.lower;
        const scalar *psid=psiK.data(), *sourced=sourceK.data(), *Vd=dm.V.data();
        const label *bndCellStart=dm.bndCellStart.data(), *bndPerm=dm.bndPerm.data();
        const scalar *bdDiagd=bdDiagK.data(), *bdSrcd=bdSrcK.data(); scalar* Hd = Hk.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            matrixHKernel(nC, ownerStart, losort, losortStart, upper, lower, own, nei, psid, sourced, Vd,
                         bndCellStart, bndPerm, bdDiagd, bdSrcd, Hd);
        });
    }
    cudaCheck(cudaGetLastError(), "matrixH");
}
// linearUpwind / linearUpwindV / LUST deferred convection corrections moved to device_deferred_correction.cu

namespace {
__device__
void setRefKernel(
    label ref,
    scalar refValue,
    scalar* __restrict__ diag,
    scalar* __restrict__ b)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        b[ref] += diag[ref] * refValue;
        diag[ref] += diag[ref];
    }
}
// adjustPhi: per-face classify (massIn / fixedMassOut / adjustableMassOut) via atomics into sums[0..2].
__device__
void adjustReduceKernel(
    int n,
    const label* __restrict__ adj,
    const scalar* __restrict__ phiB,
    scalar* __restrict__ sums)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar f = phiB[i];
    if (f < 0.0) atomicAdd(&sums[0], -f);                 // massIn
    else if (adj[i]) atomicAdd(&sums[2], f);              // adjustableMassOut
    else atomicAdd(&sums[1], f);                          // fixedMassOut
}
__device__
void adjustScaleKernel(
    int n,
    const label* __restrict__ adj,
    scalar massCorr,
    scalar* __restrict__ phiB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (adj[i] && phiB[i] > 0.0) phiB[i] *= massCorr;     // scale only the adjustable OUTFLOW
}
} // namespace
scalar deviceAdjustPhi(const DeviceBuffer<label>& adjustable, DeviceBuffer<scalar>& phiB)
{
    const int n = static_cast<int>(phiB.size());
    if (n == 0) return 1.0;
    DeviceBuffer<scalar> sums(3);
    cudaCheck(cudaMemset(sums.data(), 0, 3 * sizeof(scalar)), "adjustPhi memset");
    const label* adjd = adjustable.data(); scalar* phiBd = phiB.data(); scalar* sumsd = sums.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { adjustReduceKernel(n, adjd, phiBd, sumsd); });
    scalar h[3];
    cudaCheck(cudaMemcpy(h, sums.data(), 3 * sizeof(scalar), cudaMemcpyDeviceToHost), "adjustPhi D2H");
    const scalar massIn = h[0], fixedOut = h[1], adjOut = h[2];
    scalar massCorr = 1.0;
    if (std::fabs(adjOut) > 1e-300) massCorr = (massIn - fixedOut) / adjOut;       // OF: (massIn-fixedMassOut)/adjustableMassOut
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { adjustScaleKernel(n, adjd, massCorr, phiBd); });
    cudaCheck(cudaGetLastError(), "adjustPhi");
    return massCorr;
}
void deviceSetReference(DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& b, label refCell, scalar refValue)
{
    if (refCell < 0) return;
    scalar* diagd = diag.data(); scalar* bd = b.data();
    pcudaParallelFor(1, 1, [=] __device__ () { setRefKernel(refCell, refValue, diagd, bd); });
    cudaCheck(cudaGetLastError(), "setReference");
}

void deviceReciprocalV(const DeviceMesh& dm, const DeviceBuffer<scalar>& diagC, DeviceBuffer<scalar>& rAU)
{
    const int nC = dm.nCells;
    rAU.resize(nC);
    const scalar* diagCd = diagC.data(); const scalar* Vd = dm.V.data(); scalar* rAUd = rAU.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { reciprocalVKernel(nC, diagCd, Vd, rAUd); });
    cudaCheck(cudaGetLastError(), "reciprocalV");
}
// SIMPLEC rAtU = 1/(1/rAU - H1) = V/rowSum, where rowSum = A*1 = diagA + sum(off-diag) (= 1/rAU - H1).
// OF (pEqn.H:12): rAtU = 1.0/(1.0/rAU - UEqn.H1()) -- NO floor/guard. Faithful: divide by rowSum directly.
__device__
void simplecRAtUKernel(
    int n,
    const scalar* __restrict__ V,
    const scalar* __restrict__ rowSum,
    const scalar* __restrict__ diagA,
    scalar* __restrict__ rAtU)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    (void)diagA;
    if (c < n) rAtU[c] = V[c] / rowSum[c];
}
void deviceSimplecRAtU(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& rowSum,
    const DeviceBuffer<scalar>& diagA,
    DeviceBuffer<scalar>& rAtU)
{
    const int nC = dm.nCells;
    rAtU.resize(nC);
    const scalar* Vd = dm.V.data(); const scalar* rowSumd = rowSum.data(); const scalar* diagAd = diagA.data();
    scalar* rAtUd = rAtU.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { simplecRAtUKernel(nC, Vd, rowSumd, diagAd, rAtUd); });
    cudaCheck(cudaGetLastError(), "simplecRAtU");
}
void deviceVectorFlux(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& phiInt)
{
    const int nIf = dm.nInternalFaces;
    phiInt.resize(nIf);
    {
        const label *own=dm.owner.data(), *nei=dm.nei.data(); const scalar *wd=dm.w.data();
        const scalar *Sfxd=dm.Sfx.data(), *Sfyd=dm.Sfy.data(), *Sfzd=dm.Sfz.data();
        const scalar *Uxd=Ux.data(), *Uyd=Uy.data(), *Uzd=Uz.data(); scalar* phiIntd = phiInt.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            vectorFluxKernel(nIf, own, nei, wd, Sfxd, Sfyd, Sfzd, Uxd, Uyd, Uzd, phiIntd);
        });
    }
    cudaCheck(cudaGetLastError(), "vectorFlux");
}
void deviceMatrixFluxInternal(const DeviceLduView& A, const DeviceBuffer<scalar>& p, DeviceBuffer<scalar>& fluxInt)
{
    fluxInt.resize(A.nInternalFaces);
    {
        const int nIf = A.nInternalFaces;
        const label *own=A.owner, *nei=A.nei; const scalar *upper=A.upper, *lower=A.lower;
        const scalar* pd = p.data(); scalar* fluxIntd = fluxInt.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () { matrixFluxKernel(nIf, own, nei, upper, lower, pd, fluxIntd); });
    }
    cudaCheck(cudaGetLastError(), "matrixFlux");
}
void deviceCorrector(
    const DeviceBuffer<scalar>& HbyA,
    const DeviceBuffer<scalar>& rAU,
    const DeviceBuffer<scalar>& gradP,
    DeviceBuffer<scalar>& U)
{
    const int nC = static_cast<int>(HbyA.size());
    U.resize(nC);
    const scalar *HbyAd = HbyA.data(), *rAUd = rAU.data(), *gradPd = gradP.data(); scalar* Ud = U.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () { correctorKernel(nC, HbyAd, rAUd, gradPd, Ud); });
    cudaCheck(cudaGetLastError(), "corrector");
}
void deviceBoundaryFlux(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& uxb,
    const DeviceBuffer<scalar>& uyb,
    const DeviceBuffer<scalar>& uzb,
    DeviceBuffer<scalar>& phiB)
{
    phiB.resize(dm.nBndFaces);
    {
        const int nB = dm.nBndFaces;
        const label* bndGFaced = dm.bndGFace.data();
        const scalar *Sfxd=dm.Sfx.data(), *Sfyd=dm.Sfy.data(), *Sfzd=dm.Sfz.data();
        const scalar *uxbd=uxb.data(), *uybd=uyb.data(), *uzbd=uzb.data(); scalar* phiBd = phiB.data();
        pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () {
            bndFluxKernel(nB, bndGFaced, Sfxd, Sfyd, Sfzd, uxbd, uybd, uzbd, phiBd);
        });
    }
    cudaCheck(cudaGetLastError(), "bndFlux");
}
void deviceRelaxDiag(
    const DeviceLduView& A,
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& iCbnd,
    scalar alpha,
    DeviceBuffer<scalar>& relaxedDiag,
    DeviceBuffer<scalar>& delta,
    const scalar* cycSumOff,
    const scalar* iCmaxMag,
    const scalar* iCmin)
{
    relaxedDiag.resize(A.nCells);
    delta.resize(A.nCells);
    {
        const int nC = A.nCells;
        const label *ownerStart=A.ownerStart, *losort=A.losort, *losortStart=A.losortStart;
        const scalar *upper=A.upper, *lower=A.lower;
        const label *bndCellStart=dm.bndCellStart.data(), *bndPerm=dm.bndPerm.data();
        const scalar *iCbndd = iCbnd.data(), *rawDiag = A.diag;
        scalar *relaxedDiagd = relaxedDiag.data(), *deltad = delta.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            relaxKernel(nC, ownerStart, losort, losortStart, upper, lower, bndCellStart, bndPerm, iCbndd, rawDiag,
                       alpha, relaxedDiagd, deltad, cycSumOff, iCmaxMag, iCmin);
        });
    }
    cudaCheck(cudaGetLastError(), "relaxDiag");
}
namespace {
__device__
void cmptMaxMag3Kernel(
    int n,
    const scalar* __restrict__ a,
    const scalar* __restrict__ b,
    const scalar* __restrict__ c,
    scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmax(fabs(a[i]), fmax(fabs(b[i]), fabs(c[i])));
}
} // namespace
void deviceCmptMaxMag3(
    const DeviceBuffer<scalar>& a,
    const DeviceBuffer<scalar>& b,
    const DeviceBuffer<scalar>& c,
    DeviceBuffer<scalar>& out)
{
    const int n = static_cast<int>(a.size());
    out.resize(n);
    const scalar *ad = a.data(), *bd = b.data(), *cd = c.data(); scalar* outd = out.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { cmptMaxMag3Kernel(n, ad, bd, cd, outd); });
    cudaCheck(cudaGetLastError(), "cmptMaxMag3");
}


__device__
void cmptMin3Kernel(
    int n,
    const scalar* __restrict__ a,
    const scalar* __restrict__ b,
    const scalar* __restrict__ c,
    scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmin(a[i], fmin(b[i], c[i]));
}


void deviceCmptMin3(
    const DeviceBuffer<scalar>& a,
    const DeviceBuffer<scalar>& b,
    const DeviceBuffer<scalar>& c,
    DeviceBuffer<scalar>& out)
{
    const int n = static_cast<int>(a.size());
    out.resize(n);
    const scalar *ad = a.data(), *bd = b.data(), *cd = c.data(); scalar* outd = out.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { cmptMin3Kernel(n, ad, bd, cd, outd); });
    cudaCheck(cudaGetLastError(), "cmptMin3");
}

} // namespace brae
