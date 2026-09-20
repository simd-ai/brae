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

// OpenFOAM's SMALL*SMALL, the threshold CMULES tests the TOTAL boundary flux against. It is not zero,
// and that matters: a face whose flux is numerically nothing must count as an inlet and be left alone.
__device__ constexpr scalar kSmallSquared = scalar(1.0e-15) * scalar(1.0e-15);

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

// ...and a COUPLED face, which is NOT overwritten: it keeps upwind's own flux, taken from the cell the
// flux leaves (MULESTemplates.C:605 `if (!phiBDPf.coupled())`, and mules_cpp.cu:88-94). The neighbour
// is a real cell on the other side of the pair, so this is the internal-face form with the interface's
// own addressing -- not a patch value, of which a cyclic patch has none that MULES would read.
__global__ void donorCyclicKernel(
    const label* __restrict__ own, const label* __restrict__ nbr,
    const scalar* __restrict__ phi, const scalar* __restrict__ psi,
    int n, scalar* __restrict__ phiBD)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const scalar p = phi[j];
    phiBD[j] = p * ((p >= scalar(0)) ? psi[own[j]] : psi[nbr[j]]);
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
    // the COUPLED faces, which are in none of the three lists above: the device mesh keeps a cyclic
    // patch out of its boundary gather entirely (device_mesh.cuh:41-44)
    const label* __restrict__ ifCellStart, const label* __restrict__ ifPerm,
    const label* __restrict__ ifNbrCell,
    const scalar* __restrict__ ifPhiBD, const scalar* __restrict__ ifPhiCorr,
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
    // COUPLED faces, transcribed from the host's patch loop (mules_cpp.cu:181-204): the neighbour
    // CELL's value enters the extrema -- not a stored patch value, because psi really is that on the
    // other side -- and phiBD/phiCorr enter the budgets with the owner-face sign, as any patch face does.
    if (ifCellStart)
    {
        for (int k = ifCellStart[c]; k < ifCellStart[c + 1]; ++k)
        {
            const int j = ifPerm[k];
            const scalar pn = psi[ifNbrCell[j]];
            mx = fmax(mx, pn);
            mn = fmin(mn, pn);
            sBD += ifPhiBD[j];
            const scalar pc = ifPhiCorr[j];
            if (pc > scalar(0)) sP  += pc;
            else                mSP -= pc;
        }
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
    const label*  __restrict__ ifCellStart, const label* __restrict__ ifPerm,
    const scalar* __restrict__ ifLambda, const scalar* __restrict__ ifPhiCorr,
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

    if (ifCellStart)
    {
        for (int k = ifCellStart[c]; k < ifCellStart[c + 1]; ++k)
        {
            const int j = ifPerm[k];
            const scalar lpc = ifLambda[j] * ifPhiCorr[j];
            if (lpc > scalar(0)) sl  += lpc;
            else                 msl -= lpc;
        }
    }

    // OpenFOAM reuses the accumulators as the limiters and then aliases them the other way round --
    // lambdam IS sumlPhip and lambdap IS mSumlPhim (MULESTemplates.C:501-502). The crossing is
    // deliberate: a face carrying a POSITIVE correction out of the owner is constrained by the owner's
    // lower-bound limiter and the neighbour's upper-bound one.
    lambdam[c] = clamp01d((sl  + psiMaxn[c]) / (mSumPhim[c] + kRootVSmall));
    lambdap[c] = clamp01d((msl + psiMinn[c]) / (sumPhip[c]  + kRootVSmall));
}

// ...part 2a: a COUPLED face takes THIS SIDE's per-cell limiter (mules_cpp.cu:311-318, and
// MULESTemplates.C:543). The other side's arrives through the sync below, which is what makes the pair
// behave as the one internal face it is.
__global__ void iterIfFaceKernel(
    int n,
    const label*  __restrict__ own,
    const scalar* __restrict__ phiCorr,
    const scalar* __restrict__ lambdam, const scalar* __restrict__ lambdap,
    scalar* __restrict__ lambda)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int o = own[j];
    lambda[j] = (phiCorr[j] > scalar(0)) ? fmin(lambda[j], lambdap[o]) : fmin(lambda[j], lambdam[o]);
}

// ...part 2b: syncTools::syncFaceList with minEqOp -- NOT a no-op in serial, it leaves both sides of a
// cyclic face with the SMALLER limiter (mules_cpp.cu:321-345). It does NOT sync an AMI pair, whose twin
// is -1 here. Reads a snapshot so the two writes of a pair cannot race.
__global__ void syncIfLambdaKernel(
    int n,
    const label*  __restrict__ twin,
    const scalar* __restrict__ lambdaIn,
    scalar* __restrict__ lambda)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const label t = twin[j];
    if (t < 0) return;
    lambda[j] = fmin(lambdaIn[j], lambdaIn[t]);
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


// -- CMULES ------------------------------------------------------------------------------------------
// THE SETUP, without a donor flux. Everything that differs from setupKernel above is marked; what is
// NOT marked is identical on purpose, because the extrema scan, the extremaCoeff relaxation and the
// smoothLimiter blend are shared between the two limiters in OpenFOAM too.
__global__ void setupCorrKernel(
    int nC,
    const label*  __restrict__ own,      const label*  __restrict__ nei,
    const label*  __restrict__ ownerStart,
    const label*  __restrict__ losort,   const label*  __restrict__ losortStart,
    const label*  __restrict__ bndCellStart, const label* __restrict__ bndPerm,
    const int*    __restrict__ bndFlag,  const int*    __restrict__ bndFixes,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ psiBndValue,
    const scalar* __restrict__ phiCorr,  const scalar* __restrict__ phiCorrBnd,
    // the pair, in none of the three lists above (device_mesh.cuh:41-44). CMULES has no donor flux
    // (note B), so a coupled face brings its neighbour's psi to the extrema and its phiCorr to the
    // budgets, and nothing else -- mules_cpp.cu:534-556, CMULESTemplates.C:327.
    const label*  __restrict__ ifCellStart, const label* __restrict__ ifPerm,
    const label*  __restrict__ ifNbrCell,   const scalar* __restrict__ ifPhiCorr,
    const scalar* __restrict__ V,
    const scalar* __restrict__ rho,
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

    scalar mx = pMin, mn = pMax;                       // the swapped initialisation, as above
    scalar sP = 0, mSP = 0;                            // B: no sumPhiBD -- there is no donor flux

    for (int k = ownerStart[c]; k < ownerStart[c + 1]; ++k)
    {
        const scalar pn = psi[nei[k]];
        mx = fmax(mx, pn);
        mn = fmin(mn, pn);
        const scalar pc = phiCorr[k];
        if (pc > scalar(0)) sP  += pc;
        else                mSP -= pc;
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        const scalar po = psi[own[f]];
        mx = fmax(mx, po);
        mn = fmin(mn, po);
        const scalar pc = phiCorr[f];
        if (pc > scalar(0)) mSP += pc;
        else                sP  -= pc;
    }
    if (ifCellStart)
    {
        for (int k = ifCellStart[c]; k < ifCellStart[c + 1]; ++k)
        {
            const int j = ifPerm[k];
            const scalar pn = psi[ifNbrCell[j]];
            mx = fmax(mx, pn);
            mn = fmin(mn, pn);
            const scalar pc = ifPhiCorr[j];
            if (pc > scalar(0)) sP  += pc;
            else                mSP -= pc;
        }
    }

    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int b = bndPerm[k];
        if (bndFlag[b] == 1) continue;                 // empty
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

    // A and B together (CMULESTemplates.C:400-412): rho and psi are the CURRENT ones, and nothing is
    // added for a donor step because there was not one.
    const scalar rhoC = at(rho, c, scalar(1));
    const scalar a   = rhoC*rDeltaT - at(Sp, c, scalar(0));
    const scalar b   = rhoC*psi[c]*rDeltaT;
    const scalar SuC = at(Su, c, scalar(0));
    psiMaxn[c]  = V[c]*(a*mx - SuC - b);
    psiMinn[c]  = V[c]*(SuC - a*mn + b);
    sumPhip[c]  = sP;
    mSumPhim[c] = mSP;
}

// C: the boundary tightening the explicit path does not have. A wedge is zeroed outright, and every
// other face is limited ONLY where the TOTAL flux phi + phiCorr leaves the domain -- OpenFOAM's own
// comment is "Limit outlet faces only" (CMULESTemplates.C:537-561). The threshold is SMALL*SMALL, not
// zero, so a face whose flux is numerically nothing counts as an inlet and is left alone. Limiting an
// inlet would throttle a prescribed inflow.
// A COUPLED face is limited whichever way its flux goes -- the inlet test an uncoupled face gets
// (total <= SMALL*SMALL) does not apply to it (CMULESTemplates.C:516, mules_cpp.cu:647-657).
__global__ void iterIfFaceCorrKernel(
    int n,
    const label*  __restrict__ own,
    const scalar* __restrict__ phiCorr,
    const scalar* __restrict__ lambdam, const scalar* __restrict__ lambdap,
    scalar* __restrict__ lambda)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int o = own[j];
    lambda[j] = (phiCorr[j] > scalar(0)) ? fmin(lambda[j], lambdap[o]) : fmin(lambda[j], lambdam[o]);
}

__global__ void iterBoundaryCorrKernel(
    int nBf,
    const label*  __restrict__ bndCell,
    const int*    __restrict__ bndFlag,
    const scalar* __restrict__ phiBnd, const scalar* __restrict__ phiCorrBnd,
    const scalar* __restrict__ lambdam, const scalar* __restrict__ lambdap,
    scalar* __restrict__ lambdaBnd)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    if (bndFlag[b] == 2) { lambdaBnd[b] = scalar(0); return; }       // wedge
    const scalar pc = phiCorrBnd[b];
    if (phiBnd[b] + pc <= kSmallSquared) return;                     // inlet, or no flux at all
    const int c = bndCell[b];
    lambdaBnd[b] = (pc > scalar(0)) ? fmin(lambdaBnd[b], lambdap[c])
                                    : fmin(lambdaBnd[b], lambdam[c]);
}

__global__ void scaleKernel(const scalar* __restrict__ lam, int n, scalar* __restrict__ x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= lam[i];
}

// A: rho*psi with BOTH current, where explicitSolveKernel takes rhoOld*psiOld.
__global__ void correctKernel(
    const scalar* __restrict__ divPhiCorr,
    const scalar* __restrict__ rho, const scalar* __restrict__ Sp, const scalar* __restrict__ Su,
    int nC, scalar rDeltaT, scalar* __restrict__ psi)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar rhoC = at(rho, c, scalar(1));
    const scalar num  = rhoC*psi[c]*rDeltaT + at(Su, c, scalar(0)) - divPhiCorr[c];
    const scalar den  = rhoC*rDeltaT - at(Sp, c, scalar(0));
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
    DeviceBuffer<scalar>&       psi,
    const DeviceCyclic*         cyc,
    const DeviceBuffer<scalar>* phiPsiIf)
{
    DeviceBuffer<scalar> divPhiPsi(dm.nCells);
    deviceDiv(dm, phiPsiInt, phiPsiBnd, divPhiPsi);
    // ...and the PAIR's flux, which fvc::div sums into its face cell like any patch's (fvc.cu:548-550)
    // and which the device mesh's boundary gather does not contain. Without it a periodic face carries
    // no alpha at all: the divergence is that of a mesh with a wall there.
    if (cyc && cyc->n > 0)
    {
        if (!phiPsiIf)
        {
            throw std::runtime_error(
                "brae deviceMules: the mesh has a periodic pair and its limited flux was not handed "
                "over. fvc::div sums a coupled patch's flux into its face cell; dropping it is a wall.");
        }
        DeviceCyclic tmp;   // deviceCyclicAddDiv reads phi from the interface itself
        deviceCyclicAddDivFlux(*cyc, *phiPsiIf, dm.V, divPhiPsi);
    }
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


void deviceMulesDonorFluxCyclic(
    const DeviceCyclic&         cyc,
    const DeviceBuffer<scalar>& psi,
    DeviceBuffer<scalar>&       phiBDIf)
{
    if (cyc.n == 0) { phiBDIf.resize(0); return; }
    phiBDIf.resize(static_cast<std::size_t>(cyc.n));
    donorCyclicKernel<<<nBlocks(cyc.n), TPB>>>(
        cyc.ownCell.data(), cyc.nbrCell.data(), cyc.phi.data(), psi.data(), cyc.n, phiBDIf.data());
    ckM(cudaGetLastError(), "donor flux, interface");
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
    DeviceBuffer<scalar>&        lambdaBnd,
    const DeviceCyclic*          cyc,
    const DeviceBuffer<scalar>*  phiBDIf,
    const DeviceBuffer<scalar>*  phiCorrIf,
    DeviceBuffer<scalar>*        lambdaIf)
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

    // the COUPLED faces, if the caller has any. All four arrive together or none does: a limiter that
    // saw the interface's extrema but not its fluxes would be a third thing, neither the host's nor a
    // refusal.
    const int nIf2 = (cyc && phiBDIf && phiCorrIf && lambdaIf) ? cyc->n : 0;
    if ((cyc && cyc->n > 0) && nIf2 == 0)
    {
        throw std::runtime_error(
            "brae deviceMules: a cyclic interface was given without its phiBD, phiCorr and lambda. "
            "MULES limits a coupled face like an internal one (MULESTemplates.C:543 and the sync that "
            "follows); limiting it with only some of them is not that.");
    }
    if (nIf2 > 0)
    {
        lambdaIf->resize(static_cast<std::size_t>(nIf2));
        fillKernel<<<nBlocks(nIf2), TPB>>>(lambdaIf->data(), nIf2, scalar(1));
        ckM(cudaGetLastError(), "lambda init, interface");
    }

    DeviceBuffer<scalar> psiMaxn(nC), psiMinn(nC), sumPhip(nC), mSumPhim(nC), lambdam(nC), lambdap(nC);

    setupKernel<<<nBlocks(nC), TPB>>>(
        nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(),
        dm.losort.data(), dm.losortStart.data(), dm.bndCellStart.data(), dm.bndPerm.data(),
        bndFlag.data(), bndFixesValue.data(),
        psi.data(), psiOld.data(), psiBndValue.data(),
        phiBDInt.data(), phiBDBnd.data(), phiCorrInt.data(), phiCorrBnd.data(),
        dm.V.data(), f.rho, f.rhoOld, f.Sp, f.Su, f.psiMax, f.psiMin,
        nIf2 ? cyc->ifCellStart.data() : nullptr, nIf2 ? cyc->ifPerm.data() : nullptr,
        nIf2 ? cyc->nbrCell.data() : nullptr,
        nIf2 ? phiBDIf->data() : nullptr, nIf2 ? phiCorrIf->data() : nullptr,
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
            nIf2 ? cyc->ifCellStart.data() : nullptr, nIf2 ? cyc->ifPerm.data() : nullptr,
            nIf2 ? lambdaIf->data() : nullptr, nIf2 ? phiCorrIf->data() : nullptr,
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
        if (nIf2 > 0)
        {
            iterIfFaceKernel<<<nBlocks(nIf2), TPB>>>(
                nIf2, cyc->ownCell.data(), phiCorrIf->data(),
                lambdam.data(), lambdap.data(), lambdaIf->data());
            ckM(cudaGetLastError(), "iter interface face");
            // ...and the sync, off a snapshot, in the host's own order: the face limiter first, the
            // pair's minimum after it (mules_cpp.cu:311-345)
            DeviceBuffer<scalar> snapshot(static_cast<std::size_t>(nIf2));
            ckM(cudaMemcpy(snapshot.data(), lambdaIf->data(), sizeof(scalar)*nIf2,
                           cudaMemcpyDeviceToDevice), "lambda snapshot");
            syncIfLambdaKernel<<<nBlocks(nIf2), TPB>>>(
                nIf2, cyc->twin.data(), snapshot.data(), lambdaIf->data());
            ckM(cudaGetLastError(), "sync interface lambda");
        }
    }
}


// -- CMULES ------------------------------------------------------------------------------------------

void deviceMulesLimiterCorr(
    const DeviceMesh&            dm,
    int                          nInternalFaces,
    int                          nBoundaryFaces,
    scalar                       rDeltaT,
    const DeviceBuffer<scalar>&  psi,
    const DeviceBuffer<scalar>&  psiBndValue,
    const DeviceBuffer<int>&     bndFixesValue,
    const DeviceBuffer<int>&     bndFlag,
    const DeviceBuffer<scalar>&  phiBnd,
    const DeviceBuffer<scalar>&  phiCorrInt,
    const DeviceBuffer<scalar>&  phiCorrBnd,
    const DeviceMulesFields&     f,
    const DeviceMulesControls&   c,
    DeviceBuffer<scalar>&        lambdaInt,
    DeviceBuffer<scalar>&        lambdaBnd,
    const DeviceCyclic*          cyc,
    const DeviceBuffer<scalar>*  phiCorrIf,
    DeviceBuffer<scalar>*        lambdaIf)
{
    const int nC = dm.nCells;

    // the pair: all three together or none, as the explicit limiter has it
    const int nIfC = (cyc && phiCorrIf && lambdaIf) ? cyc->n : 0;
    if (cyc && cyc->n > 0 && nIfC == 0)
    {
        throw std::runtime_error(
            "brae deviceMules (CMULES): a cyclic interface was given without its phiCorr and lambda. "
            "A coupled face is limited whichever way its flux goes (CMULESTemplates.C:516); limiting "
            "it with only some of them is not that.");
    }
    if (nIfC > 0)
    {
        lambdaIf->resize(static_cast<std::size_t>(nIfC));
        fillKernel<<<nBlocks(nIfC), TPB>>>(lambdaIf->data(), nIfC, scalar(1));
        ckM(cudaGetLastError(), "CMULES lambda init, interface");
    }

    lambdaInt.resize(static_cast<std::size_t>(nInternalFaces));
    lambdaBnd.resize(static_cast<std::size_t>(nBoundaryFaces));
    if (nInternalFaces > 0)
    {
        fillKernel<<<nBlocks(nInternalFaces), TPB>>>(lambdaInt.data(), nInternalFaces, scalar(1));
        ckM(cudaGetLastError(), "lambda init");
    }
    if (nBoundaryFaces > 0)
    {
        fillKernel<<<nBlocks(nBoundaryFaces), TPB>>>(lambdaBnd.data(), nBoundaryFaces, scalar(1));
        ckM(cudaGetLastError(), "lambda boundary init");
    }

    // boundaryExtremaCoeff DEFAULTS TO extremaCoeff, not to 0 (MULESTemplates.C:220), so what the scan
    // adds at a boundary is the DIFFERENCE and is normally nothing.
    const scalar boundaryDelta = std::max(c.boundaryExtremaCoeff - c.extremaCoeff, scalar(0));

    DeviceBuffer<scalar> psiMaxn(nC), psiMinn(nC), sumPhip(nC), mSumPhim(nC);
    setupCorrKernel<<<nBlocks(nC), TPB>>>(
        nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(),
        dm.losort.data(), dm.losortStart.data(), dm.bndCellStart.data(), dm.bndPerm.data(),
        bndFlag.data(), bndFixesValue.data(),
        psi.data(), psiBndValue.data(), phiCorrInt.data(), phiCorrBnd.data(),
        nIfC ? cyc->ifCellStart.data() : nullptr, nIfC ? cyc->ifPerm.data() : nullptr,
        nIfC ? cyc->nbrCell.data() : nullptr,     nIfC ? phiCorrIf->data() : nullptr,
        dm.V.data(), f.rho, f.Sp, f.Su, f.psiMax, f.psiMin,
        rDeltaT, c.extremaCoeff, boundaryDelta, c.smoothLimiter,
        psiMaxn.data(), psiMinn.data(), sumPhip.data(), mSumPhim.data());
    ckM(cudaGetLastError(), "CMULES setup");

    DeviceBuffer<scalar> lambdam(nC), lambdap(nC);
    for (label it = 0; it < c.nLimiterIter; ++it)
    {
        // part 1 is shared with the explicit limiter: it reads only lambda and phiCorr, neither of
        // which A, B or C touches.
        iterCellKernel<<<nBlocks(nC), TPB>>>(
            nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(),
            dm.losort.data(), dm.losortStart.data(), dm.bndCellStart.data(), dm.bndPerm.data(),
            bndFlag.data(), lambdaInt.data(), lambdaBnd.data(),
            phiCorrInt.data(), phiCorrBnd.data(),
            psiMaxn.data(), psiMinn.data(), sumPhip.data(), mSumPhim.data(),
            nIfC ? cyc->ifCellStart.data() : nullptr, nIfC ? cyc->ifPerm.data() : nullptr,
            nIfC ? lambdaIf->data() : nullptr, nIfC ? phiCorrIf->data() : nullptr,
            lambdam.data(), lambdap.data());
        ckM(cudaGetLastError(), "CMULES iteration, cells");

        if (nInternalFaces > 0)
        {
            iterFaceKernel<<<nBlocks(nInternalFaces), TPB>>>(
                nInternalFaces, dm.owner.data(), dm.nei.data(), phiCorrInt.data(),
                lambdam.data(), lambdap.data(), lambdaInt.data());
            ckM(cudaGetLastError(), "CMULES iteration, faces");
        }
        if (nBoundaryFaces > 0)
        {
            if (nIfC > 0)
            {
                iterIfFaceCorrKernel<<<nBlocks(nIfC), TPB>>>(
                    nIfC, cyc->ownCell.data(), phiCorrIf->data(),
                    lambdam.data(), lambdap.data(), lambdaIf->data());
                ckM(cudaGetLastError(), "CMULES iteration, interface face");
                DeviceBuffer<scalar> snap(static_cast<std::size_t>(nIfC));
                ckM(cudaMemcpy(snap.data(), lambdaIf->data(), sizeof(scalar)*nIfC,
                               cudaMemcpyDeviceToDevice), "CMULES lambda snapshot");
                syncIfLambdaKernel<<<nBlocks(nIfC), TPB>>>(
                    nIfC, cyc->twin.data(), snap.data(), lambdaIf->data());
                ckM(cudaGetLastError(), "CMULES sync, interface");
            }
            iterBoundaryCorrKernel<<<nBlocks(nBoundaryFaces), TPB>>>(
                nBoundaryFaces, dm.bndCell.data(), bndFlag.data(),
                phiBnd.data(), phiCorrBnd.data(), lambdam.data(), lambdap.data(), lambdaBnd.data());
            ckM(cudaGetLastError(), "CMULES iteration, boundary");
        }
    }
}


void deviceMulesLimitCorr(
    const DeviceMesh&            dm,
    int                          nInternalFaces,
    int                          nBoundaryFaces,
    scalar                       rDeltaT,
    const DeviceBuffer<scalar>&  psi,
    const DeviceBuffer<scalar>&  psiBndValue,
    const DeviceBuffer<int>&     bndFixesValue,
    const DeviceBuffer<int>&     bndFlag,
    const DeviceBuffer<scalar>&  phiBnd,
    DeviceBuffer<scalar>&        phiCorrInt,
    DeviceBuffer<scalar>&        phiCorrBnd,
    const DeviceMulesFields&     f,
    const DeviceMulesControls&   c,
    DeviceBuffer<scalar>*        lambdaIntOut,
    DeviceBuffer<scalar>*        lambdaBndOut,
    const DeviceCyclic*          cyc,
    DeviceBuffer<scalar>*        phiCorrIf)
{
    DeviceBuffer<scalar> li, lb;
    DeviceBuffer<scalar>& lambdaInt = lambdaIntOut ? *lambdaIntOut : li;
    DeviceBuffer<scalar>& lambdaBnd = lambdaBndOut ? *lambdaBndOut : lb;

    DeviceBuffer<scalar> lambdaIf;
    deviceMulesLimiterCorr(dm, nInternalFaces, nBoundaryFaces, rDeltaT, psi, psiBndValue,
                           bndFixesValue, bndFlag, phiBnd, phiCorrInt, phiCorrBnd, f, c,
                           lambdaInt, lambdaBnd,
                           (cyc && phiCorrIf) ? cyc : nullptr,
                           (cyc && phiCorrIf) ? phiCorrIf : nullptr,
                           (cyc && phiCorrIf) ? &lambdaIf : nullptr);

    // phiCorr *= lambda, in place. No blended flux: see B.
    if (nInternalFaces > 0)
    {
        scaleKernel<<<nBlocks(nInternalFaces), TPB>>>(lambdaInt.data(), nInternalFaces, phiCorrInt.data());
        ckM(cudaGetLastError(), "phiCorr *= lambda");
    }
    if (nBoundaryFaces > 0)
    {
        scaleKernel<<<nBlocks(nBoundaryFaces), TPB>>>(lambdaBnd.data(), nBoundaryFaces, phiCorrBnd.data());
        ckM(cudaGetLastError(), "phiCorr *= lambda, boundary");
    }
    if (cyc && phiCorrIf && cyc->n > 0)
    {
        scaleKernel<<<nBlocks(cyc->n), TPB>>>(lambdaIf.data(), cyc->n, phiCorrIf->data());
        ckM(cudaGetLastError(), "phiCorr *= lambda, interface");
    }
}


void deviceMulesCorrect(
    const DeviceMesh&           dm,
    scalar                      rDeltaT,
    const DeviceBuffer<scalar>& phiCorrInt,
    const DeviceBuffer<scalar>& phiCorrBnd,
    const DeviceMulesFields&    f,
    DeviceBuffer<scalar>&       psi,
    const DeviceCyclic*         cyc,
    const DeviceBuffer<scalar>* phiCorrIf)
{
    DeviceBuffer<scalar> divPhiCorr(dm.nCells);
    deviceDiv(dm, phiCorrInt, phiCorrBnd, divPhiCorr);
    // ...and the pair's correction, which surfaceIntegrate sums into its face cell like any patch's
    // (fvc.cu:548-550). Leaving it out is the same wall the explicit solve's divergence would build.
    if (cyc && cyc->n > 0)
    {
        if (!phiCorrIf)
        {
            throw std::runtime_error(
                "brae deviceMules (CMULES correct): the mesh has a periodic pair and its limited "
                "correction was not handed over. It is summed into the cells like any patch's.");
        }
        deviceCyclicAddDivFlux(*cyc, *phiCorrIf, dm.V, divPhiCorr);
    }
    correctKernel<<<nBlocks(dm.nCells), TPB>>>(
        divPhiCorr.data(), f.rho, f.Sp, f.Su, dm.nCells, rDeltaT, psi.data());
    ckM(cudaGetLastError(), "CMULES correct");
}

} // namespace brae
