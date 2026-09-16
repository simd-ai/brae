// MULES on the device -- see device_mules.cuh for the gather-not-atomics argument and for what cannot
// be bit-identical to the host.
#include "device_mules.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// OpenFOAM's ROOTVSMALL. Named because the two divisors below run in EVERY cell -- sumPhip and
// mSumPhim are zero wherever the correction does not reach -- so this constant is what lambda is in
// the untouched majority of the domain, not a guard against a rare zero.
__device__ constexpr scalar kRootVSmall = scalar(1.0e-150);

inline void ckM(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae deviceMules: ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ scalar clamp01d(scalar x)
{
    return x < scalar(0) ? scalar(0) : (x > scalar(1) ? scalar(1) : x);
}

__device__ __forceinline__ scalar at(const scalar* f, int i, scalar k) { return f ? f[i] : k; }

// phiBD = upwind(phi).flux(psi) on the internal faces.
__global__ void donorInternalKernel(
    const label* __restrict__ own, const label* __restrict__ nei,
    const scalar* __restrict__ phi, const scalar* __restrict__ psi,
    int nIf, scalar* __restrict__ phiBD)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar p = phi[f];
    phiBD[f] = p * ((p >= scalar(0)) ? psi[own[f]] : psi[nei[f]]);
}

// ...and the boundary, which is phiPsi verbatim -- the overwrite that makes phiCorr zero there.
__global__ void donorBoundaryKernel(const scalar* __restrict__ phiPsiBnd, int nBf,
                                    scalar* __restrict__ phiBDBnd)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    phiBDBnd[b] = phiPsiBnd[b];
}

// THE SETUP, one gather per cell: the neighbourhood extrema, sumPhiBD, sumPhip and mSumPhim, then the
// extremaCoeff/smoothLimiter relaxation and the transform into flux-space budgets.
__global__ void setupKernel(
    int nC,
    const label*  __restrict__ own,      const label*  __restrict__ nei,
    const label*  __restrict__ ownerStart,
    const label*  __restrict__ losort,   const label*  __restrict__ losortStart,
    const label*  __restrict__ bndCellStart, const label* __restrict__ bndPerm,
    const int*    __restrict__ bndFlag,  const int*    __restrict__ bndFixes,
    const scalar* __restrict__ psi,      const scalar* __restrict__ psiOld,
    const scalar* __restrict__ psiBndValue,
    const scalar* __restrict__ phiBD,    const scalar* __restrict__ phiBDBnd,
    const scalar* __restrict__ phiCorr,  const scalar* __restrict__ phiCorrBnd,
    const scalar* __restrict__ V,
    const scalar* __restrict__ rho, const scalar* __restrict__ rhoOld,
    const scalar* __restrict__ Sp,  const scalar* __restrict__ Su,
    const scalar* __restrict__ psiMaxF, const scalar* __restrict__ psiMinF,
    scalar rDeltaT, scalar extremaCoeff, scalar boundaryDeltaExtremaCoeff, scalar smoothLimiter,
    scalar* __restrict__ psiMaxn, scalar* __restrict__ psiMinn,
    scalar* __restrict__ sumPhip, scalar* __restrict__ mSumPhim)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar pMax = at(psiMaxF, c, scalar(1));
    const scalar pMin = at(psiMinF, c, scalar(0));

    // THE SWAPPED INITIALISATION: psiMaxn starts at the LOWER bound so the neighbour scan builds it up
    // from the far end (MULESTemplates.C:280-281).
    scalar mx = pMin, mn = pMax;
    scalar sBD = 0, sP = 0, mSP = 0;

    // faces where this cell is the OWNER
    for (int k = ownerStart[c]; k < ownerStart[c + 1]; ++k)
    {
        const scalar pn = psi[nei[k]];
        mx = fmax(mx, pn);
        mn = fmin(mn, pn);
        sBD += phiBD[k];
        const scalar pc = phiCorr[k];
        if (pc > scalar(0)) sP  += pc;
        else                mSP -= pc;
    }
    // ...and where it is the NEIGHBOUR. The sign of sumPhiBD flips, and so does which of the two
    // correction sums the face lands in -- that crossing is the whole content of these four lines.
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        const scalar po = psi[own[f]];
        mx = fmax(mx, po);
        mn = fmin(mn, po);
        sBD -= phiBD[f];
        const scalar pc = phiCorr[f];
        if (pc > scalar(0)) mSP += pc;
        else                sP  -= pc;
    }
    // boundary faces
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int b = bndPerm[k];
        if (bndFlag[b] == 1) continue;                 // empty: OpenFOAM never enters this body
        if (bndFixes[b])
        {
            mx = fmax(mx, psiBndValue[b]);
            mn = fmin(mn, psiBndValue[b]);
        }
        else if (boundaryDeltaExtremaCoeff > scalar(0))
        {
            const scalar extrema = boundaryDeltaExtremaCoeff * (pMax - pMin);
            mx += extrema;
            mn -= extrema;
        }
        sBD += phiBDBnd[b];
        const scalar pc = phiCorrBnd[b];
        if (pc > scalar(0)) sP  += pc;
        else                mSP -= pc;
    }

    mx = fmin(mx + extremaCoeff*(pMax - pMin), pMax);
    mn = fmax(mn - extremaCoeff*(pMax - pMin), pMin);
    if (smoothLimiter > scalar(1e-15))
    {
        mx = fmin(smoothLimiter*psi[c] + (scalar(1) - smoothLimiter)*mx, pMax);
        mn = fmax(smoothLimiter*psi[c] + (scalar(1) - smoothLimiter)*mn, pMin);
    }

    // ...into flux-space budgets (MULESTemplates.C:418-436, the fixed-mesh branch).
    const scalar a = at(rho, c, scalar(1))*rDeltaT - at(Sp, c, scalar(0));
    const scalar b = at(rhoOld, c, scalar(1))*rDeltaT*psiOld[c];
    const scalar SuC = at(Su, c, scalar(0));
    psiMaxn[c]  = V[c]*(a*mx - SuC - b) + sBD;
    psiMinn[c]  = V[c]*(SuC - a*mn + b) - sBD;
    sumPhip[c]  = sP;
    mSumPhim[c] = mSP;
}

// ONE ITERATION, part 1: gather lambda*phiCorr per cell and form the two per-cell limiters.
__global__ void iterCellKernel(
    int nC,
    const label*  __restrict__ own,      const label*  __restrict__ nei,
    const label*  __restrict__ ownerStart,
    const label*  __restrict__ losort,   const label*  __restrict__ losortStart,
    const label*  __restrict__ bndCellStart, const label* __restrict__ bndPerm,
    const int*    __restrict__ bndFlag,
    const scalar* __restrict__ lambda,   const scalar* __restrict__ lambdaBnd,
    const scalar* __restrict__ phiCorr,  const scalar* __restrict__ phiCorrBnd,
    const scalar* __restrict__ psiMaxn,  const scalar* __restrict__ psiMinn,
    const scalar* __restrict__ sumPhip,  const scalar* __restrict__ mSumPhim,
    scalar* __restrict__ lambdam, scalar* __restrict__ lambdap)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar sl = 0, msl = 0;
    for (int k = ownerStart[c]; k < ownerStart[c + 1]; ++k)
    {
        const scalar lpc = lambda[k] * phiCorr[k];
        if (lpc > scalar(0)) sl  += lpc;
        else                 msl -= lpc;
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        const scalar lpc = lambda[f] * phiCorr[f];
        if (lpc > scalar(0)) msl += lpc;
        else                 sl  -= lpc;
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int b = bndPerm[k];
        if (bndFlag[b] == 1) continue;                 // empty
        const scalar lpc = lambdaBnd[b] * phiCorrBnd[b];
        if (lpc > scalar(0)) sl  += lpc;
        else                 msl -= lpc;
    }

    // OpenFOAM reuses the accumulators as the limiters and then aliases them the other way round --
    // lambdam IS sumlPhip and lambdap IS mSumlPhim (MULESTemplates.C:501-502). The crossing is
    // deliberate: a face carrying a POSITIVE correction out of the owner is constrained by the owner's
    // lower-bound limiter and the neighbour's upper-bound one.
    lambdam[c] = clamp01d((sl  + psiMaxn[c]) / (mSumPhim[c] + kRootVSmall));
    lambdap[c] = clamp01d((msl + psiMinn[c]) / (sumPhip[c]  + kRootVSmall));
}

// ...part 2: tighten each face against the two cells it joins.
__global__ void iterFaceKernel(
    int nIf,
    const label*  __restrict__ own, const label* __restrict__ nei,
    const scalar* __restrict__ phiCorr,
    const scalar* __restrict__ lambdam, const scalar* __restrict__ lambdap,
    scalar* __restrict__ lambda)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const int o = own[f], n = nei[f];
    lambda[f] = (phiCorr[f] > scalar(0))
        ? fmin(lambda[f], fmin(lambdap[o], lambdam[n]))
        : fmin(lambda[f], fmin(lambdam[o], lambdap[n]));
}

// A WEDGE patch is zeroed outright, before any coupled logic (MULESTemplates.C:530-533). Uncoupled
// patches are left alone: phiCorr is zero on them by construction, so lambda there multiplies nothing.
__global__ void iterBoundaryKernel(int nBf, const int* __restrict__ bndFlag,
                                   scalar* __restrict__ lambdaBnd)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    if (bndFlag[b] == 2) lambdaBnd[b] = scalar(0);
}

__global__ void fillKernel(scalar* __restrict__ x, int n, scalar v)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ void blendKernel(
    const scalar* __restrict__ bd, const scalar* __restrict__ lam,
    const scalar* __restrict__ corr, int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bd[i] + lam[i]*corr[i];
}

__global__ void explicitSolveKernel(
    const scalar* __restrict__ psiOld, const scalar* __restrict__ divPhiPsi,
    const scalar* __restrict__ rho, const scalar* __restrict__ rhoOld,
    const scalar* __restrict__ Sp, const scalar* __restrict__ Su,
    int nC, scalar rDeltaT, scalar* __restrict__ psi)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    // rho.oldTime() in the numerator and rho in the denominator -- not the same field on a VoF
    // interface, where the two differ by the density ratio in every cell the interface crossed.
    const scalar num = at(rhoOld, c, scalar(1))*psiOld[c]*rDeltaT + at(Su, c, scalar(0)) - divPhiPsi[c];
    const scalar den = at(rho, c, scalar(1))*rDeltaT - at(Sp, c, scalar(0));
    psi[c] = num / den;
}

}   // namespace


void deviceMulesBlend(
    int                         nInternalFaces,
    int                         nBoundaryFaces,
    const DeviceBuffer<scalar>& phiBDInt,  const DeviceBuffer<scalar>& phiBDBnd,
    const DeviceBuffer<scalar>& lambdaInt, const DeviceBuffer<scalar>& lambdaBnd,
    const DeviceBuffer<scalar>& phiCorrInt,const DeviceBuffer<scalar>& phiCorrBnd,
    DeviceBuffer<scalar>&       phiPsiInt, DeviceBuffer<scalar>&       phiPsiBnd)
{
    if (nInternalFaces > 0)
    {
        phiPsiInt.resize(static_cast<std::size_t>(nInternalFaces));
        blendKernel<<<nBlocks(nInternalFaces), TPB>>>(
            phiBDInt.data(), lambdaInt.data(), phiCorrInt.data(), nInternalFaces, phiPsiInt.data());
        ckM(cudaGetLastError(), "blend internal");
    }
    if (nBoundaryFaces > 0)
    {
        phiPsiBnd.resize(static_cast<std::size_t>(nBoundaryFaces));
        blendKernel<<<nBlocks(nBoundaryFaces), TPB>>>(
            phiBDBnd.data(), lambdaBnd.data(), phiCorrBnd.data(), nBoundaryFaces, phiPsiBnd.data());
        ckM(cudaGetLastError(), "blend boundary");
    }
}


void deviceMulesExplicitSolve(
    const DeviceMesh&           dm,
    scalar                      rDeltaT,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& phiPsiInt,
    const DeviceBuffer<scalar>& phiPsiBnd,
    const DeviceMulesFields&    f,
    DeviceBuffer<scalar>&       psi)
{
    DeviceBuffer<scalar> divPhiPsi(dm.nCells);
    deviceDiv(dm, phiPsiInt, phiPsiBnd, divPhiPsi);
    psi.resize(static_cast<std::size_t>(dm.nCells));
    explicitSolveKernel<<<nBlocks(dm.nCells), TPB>>>(
        psiOld.data(), divPhiPsi.data(), f.rho, f.rhoOld, f.Sp, f.Su,
        dm.nCells, rDeltaT, psi.data());
    ckM(cudaGetLastError(), "explicit solve");
}


void deviceMulesDonorFlux(
    const DeviceMesh&           dm,
    int                         nInternalFaces,
    int                         nBoundaryFaces,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& phiPsiBnd,
    DeviceBuffer<scalar>&       phiBDInt,
    DeviceBuffer<scalar>&       phiBDBnd)
{
    phiBDInt.resize(static_cast<std::size_t>(nInternalFaces));
    phiBDBnd.resize(static_cast<std::size_t>(nBoundaryFaces));
    if (nInternalFaces > 0)
    {
        donorInternalKernel<<<nBlocks(nInternalFaces), TPB>>>(
            dm.owner.data(), dm.nei.data(), phiInt.data(), psi.data(),
            nInternalFaces, phiBDInt.data());
        ckM(cudaGetLastError(), "donor internal");
    }
    if (nBoundaryFaces > 0)
    {
        donorBoundaryKernel<<<nBlocks(nBoundaryFaces), TPB>>>(
            phiPsiBnd.data(), nBoundaryFaces, phiBDBnd.data());
        ckM(cudaGetLastError(), "donor boundary");
    }
}


void deviceMulesLimiter(
    const DeviceMesh&            dm,
    int                          nInternalFaces,
    int                          nBoundaryFaces,
    scalar                       rDeltaT,
    const DeviceBuffer<scalar>&  psi,
    const DeviceBuffer<scalar>&  psiOld,
    const DeviceBuffer<scalar>&  psiBndValue,
    const DeviceBuffer<int>&     bndFixesValue,
    const DeviceBuffer<int>&     bndFlag,
    const DeviceBuffer<scalar>&  phiBDInt,
    const DeviceBuffer<scalar>&  phiBDBnd,
    const DeviceBuffer<scalar>&  phiCorrInt,
    const DeviceBuffer<scalar>&  phiCorrBnd,
    const DeviceMulesFields&     f,
    const DeviceMulesControls&   c,
    DeviceBuffer<scalar>&        lambdaInt,
    DeviceBuffer<scalar>&        lambdaBnd)
{
    if (c.nLimiterIter < 1)
        throw std::runtime_error("brae deviceMules: nLimiterIter must be at least 1.");

    const int nC = dm.nCells;
    const scalar boundaryDeltaExtremaCoeff =
        (c.boundaryExtremaCoeff - c.extremaCoeff) > scalar(0)
            ? (c.boundaryExtremaCoeff - c.extremaCoeff) : scalar(0);

    // lambda starts at 1 on every face and only ever decreases.
    lambdaInt.resize(static_cast<std::size_t>(nInternalFaces));
    lambdaBnd.resize(static_cast<std::size_t>(nBoundaryFaces));
    if (nInternalFaces > 0)
    { fillKernel<<<nBlocks(nInternalFaces), TPB>>>(lambdaInt.data(), nInternalFaces, scalar(1)); }
    if (nBoundaryFaces > 0)
    { fillKernel<<<nBlocks(nBoundaryFaces), TPB>>>(lambdaBnd.data(), nBoundaryFaces, scalar(1)); }
    ckM(cudaGetLastError(), "lambda init");

    DeviceBuffer<scalar> psiMaxn(nC), psiMinn(nC), sumPhip(nC), mSumPhim(nC), lambdam(nC), lambdap(nC);

    setupKernel<<<nBlocks(nC), TPB>>>(
        nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(),
        dm.losort.data(), dm.losortStart.data(), dm.bndCellStart.data(), dm.bndPerm.data(),
        bndFlag.data(), bndFixesValue.data(),
        psi.data(), psiOld.data(), psiBndValue.data(),
        phiBDInt.data(), phiBDBnd.data(), phiCorrInt.data(), phiCorrBnd.data(),
        dm.V.data(), f.rho, f.rhoOld, f.Sp, f.Su, f.psiMax, f.psiMin,
        rDeltaT, c.extremaCoeff, boundaryDeltaExtremaCoeff, c.smoothLimiter,
        psiMaxn.data(), psiMinn.data(), sumPhip.data(), mSumPhim.data());
    ckM(cudaGetLastError(), "setup");

    for (label it = 0; it < c.nLimiterIter; ++it)
    {
        iterCellKernel<<<nBlocks(nC), TPB>>>(
            nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(),
            dm.losort.data(), dm.losortStart.data(), dm.bndCellStart.data(), dm.bndPerm.data(),
            bndFlag.data(), lambdaInt.data(), lambdaBnd.data(),
            phiCorrInt.data(), phiCorrBnd.data(),
            psiMaxn.data(), psiMinn.data(), sumPhip.data(), mSumPhim.data(),
            lambdam.data(), lambdap.data());
        ckM(cudaGetLastError(), "iter cell");

        if (nInternalFaces > 0)
        {
            iterFaceKernel<<<nBlocks(nInternalFaces), TPB>>>(
                nInternalFaces, dm.owner.data(), dm.nei.data(), phiCorrInt.data(),
                lambdam.data(), lambdap.data(), lambdaInt.data());
            ckM(cudaGetLastError(), "iter face");
        }
        if (nBoundaryFaces > 0)
        {
            iterBoundaryKernel<<<nBlocks(nBoundaryFaces), TPB>>>(
                nBoundaryFaces, bndFlag.data(), lambdaBnd.data());
            ckM(cudaGetLastError(), "iter boundary");
        }
    }
}

} // namespace brae
