// cf GPU offload (G7): SIMPLE coupling-glue kernels. matrixH and matrixFlux reuse the ownerStart/losort
// gather; rAU / corrector are elementwise; the HbyA flux is a per-face vector interpolation dotted with Sf.
#include <stdexcept>
#include "device_boundary.cuh"
#include <string>
#include "device_simple.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

__global__
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
__global__
void reciprocalVKernel(
    int nC,
    const scalar* __restrict__ diagC,
    const scalar* __restrict__ V,
    scalar* __restrict__ rAU)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) rAU[c] = V[c] / diagC[c];
}
__global__
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
__global__
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
__global__
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
__global__
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
__global__
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
    matrixHKernel<<<nBlocks(nC), TPB>>>(nC, A.ownerStart, A.losort, A.losortStart, A.upper, A.lower, A.owner, A.nei,
                                        psiK.data(), sourceK.data(), dm.V.data(), dm.bndCellStart.data(), dm.bndPerm.data(),
                                        bdDiagK.data(), bdSrcK.data(), Hk.data());
    cudaCheck(cudaGetLastError(), "matrixH");
}
// linearUpwind / linearUpwindV / LUST deferred convection corrections moved to device_deferred_correction.cu

namespace {
__global__
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
__global__
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
    // NO |phi| into sums[3] here: OF's totalFlux = VSMALL + sum(mag(phi)) sums the INTERNAL faces only
    // -- Foam::sum() of a GeometricField is gSum(f1.primitiveField()) (GeometricFieldFunctions.C:470-
    // 497). This kernel used to add the boundary too, inflating the normaliser behind the massCorr
    // threshold, the fatal and closedVolume; test_adjust_phi_guards arm 7 straddles exactly that gap
    // (a 2e-8 relative continuity error that OF refuses and the inflated form waved through).
}


// sum|phi| over the INTERNAL faces -- the whole of OF's normaliser (see the note above).
__global__
void absSumKernel(int n, const scalar* __restrict__ x, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(out, fabs(x[i]));
}
__global__
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
scalar deviceAdjustPhi(const DeviceBuffer<label>& adjustable,
                       DeviceBuffer<scalar>& phiB,
                       const DeviceBuffer<scalar>* phiInt,
                       bool* closedVolume)
{
    const int n = static_cast<int>(phiB.size());
    if (n == 0) { if (closedVolume) *closedVolume = true; return 1.0; }
    DeviceBuffer<scalar> sums(4);
    cudaCheck(cudaMemset(sums.data(), 0, 4 * sizeof(scalar)), "adjustPhi memset");
    adjustReduceKernel<<<nBlocks(n), TPB>>>(n, adjustable.data(), phiB.data(), sums.data());
    if (phiInt && phiInt->size())
    {
        const int ni = static_cast<int>(phiInt->size());
        absSumKernel<<<nBlocks(ni), TPB>>>(ni, phiInt->data(), sums.data() + 3);
    }
    scalar h[4];
    cudaCheck(cudaMemcpy(h, sums.data(), 4 * sizeof(scalar), cudaMemcpyDeviceToHost), "adjustPhi D2H");
    const scalar massIn = h[0], fixedOut = h[1], adjOut = h[2];

    // OF adjustPhi.C:90-119, both clauses. This used to be `if (fabs(adjOut) > 1e-300)` and nothing else,
    // which differs from OpenFOAM twice over:
    //
    //   1. OF's test is RELATIVE -- magAdjustableMassOut/totalFlux > SMALL (1e-15) -- against
    //      totalFlux = VSMALL + sum(mag(phi)), the INTERNAL faces (Foam::sum() of a GeometricField is
    //      gSum of the primitive field). An absolute 1e-300 admits an
    //      adjustable outflow that is negligible beside the flux in the domain, and then divides by it:
    //      exactly the uninitialised-outflow case OF's message tells you to fix with potentialFoam.
    //   2. OF RAISES A FATAL ERROR on the other branch when the residual continuity error is more than
    //      1e-8 of the total flux. brae carried on with massCorr = 1.0 and handed the pressure equation
    //      an inconsistent right-hand side, which converges to something plausible.
    //
    // Both host twins already do this -- rhoPEqn_cpp.cu:99-122 and simpleFoam/pEqn_cpp.cu:92-125, the
    // latter with OpenFOAM's own wording -- so the device kernel was the only one of the three that did
    // not, while being the one all six device call sites reach.
    const scalar kVSmall  = scalar(1e-300);
    const scalar kSmall   = scalar(1e-15);
    const scalar totalFlux = kVSmall + h[3];
    scalar massCorr = 1.0;
    const scalar magAdj = std::fabs(adjOut);
    if (magAdj > kVSmall && magAdj / totalFlux > kSmall)
    {
        massCorr = (massIn - fixedOut) / adjOut;
    }
    else if (std::fabs(fixedOut - massIn) / totalFlux > scalar(1e-8))
    {
        throw std::runtime_error(
            "brae deviceAdjustPhi: continuity error cannot be removed by adjusting the outflow "
            "(adjustPhi.C:108-119). Check the velocity boundary conditions, or run potentialFoam to "
            "initialise the outflow. Specified mass inflow " + std::to_string((double)massIn)
            + ", specified mass outflow " + std::to_string((double)fixedOut)
            + ", adjustable mass outflow " + std::to_string((double)adjOut)
            + ", total flux " + std::to_string((double)totalFlux) + ".");
    }
    adjustScaleKernel<<<nBlocks(n), TPB>>>(n, adjustable.data(), massCorr, phiB.data());
    cudaCheck(cudaGetLastError(), "adjustPhi");
    if (closedVolume)
        *closedVolume = std::fabs(massIn)   / totalFlux < kSmall
                     && std::fabs(fixedOut) / totalFlux < kSmall
                     && std::fabs(adjOut)   / totalFlux < kSmall;
    return massCorr;
}
void deviceSetReference(DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& b, label refCell, scalar refValue)
{
    if (refCell < 0) return;
    setRefKernel<<<1, 1>>>(refCell, refValue, diag.data(), b.data());
    cudaCheck(cudaGetLastError(), "setReference");
}

void deviceReciprocalV(const DeviceMesh& dm, const DeviceBuffer<scalar>& diagC, DeviceBuffer<scalar>& rAU)
{
    const int nC = dm.nCells;
    rAU.resize(nC);
    reciprocalVKernel<<<nBlocks(nC), TPB>>>(nC, diagC.data(), dm.V.data(), rAU.data());
    cudaCheck(cudaGetLastError(), "reciprocalV");
}
// SIMPLEC rAtU = 1/(1/rAU - H1) = V/rowSum, where rowSum = A*1 = diagA + sum(off-diag) (= 1/rAU - H1).
// OF (pEqn.H:12): rAtU = 1.0/(1.0/rAU - UEqn.H1()) -- NO floor/guard. Faithful: divide by rowSum directly.
__global__
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
    simplecRAtUKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), rowSum.data(), diagA.data(), rAtU.data());
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
    vectorFluxKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), dm.Sfx.data(), dm.Sfy.data(),
                                            dm.Sfz.data(), Ux.data(), Uy.data(), Uz.data(), phiInt.data());
    cudaCheck(cudaGetLastError(), "vectorFlux");
}
void deviceMatrixFluxInternal(const DeviceLduView& A, const DeviceBuffer<scalar>& p, DeviceBuffer<scalar>& fluxInt)
{
    fluxInt.resize(A.nInternalFaces);
    matrixFluxKernel<<<nBlocks(A.nInternalFaces), TPB>>>(A.nInternalFaces, A.owner, A.nei, A.upper, A.lower, p.data(), fluxInt.data());
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
    correctorKernel<<<nBlocks(nC), TPB>>>(nC, HbyA.data(), rAU.data(), gradP.data(), U.data());
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
    bndFluxKernel<<<nBlocks(dm.nBndFaces), TPB>>>(dm.nBndFaces, dm.bndGFace.data(), dm.Sfx.data(), dm.Sfy.data(),
                                                  dm.Sfz.data(), uxb.data(), uyb.data(), uzb.data(), phiB.data());
    cudaCheck(cudaGetLastError(), "bndFlux");
}

namespace
{
__global__ void constrainPressureKernel(
    int                        n,
    const label*  __restrict__ mask,
    const scalar* __restrict__ phiHbyABnd,
    const scalar* __restrict__ sfUBnd,
    const scalar* __restrict__ rhoBnd,     // null = 1 (incompressible)
    const scalar* __restrict__ denBnd,
    const scalar* __restrict__ magSf,
    scalar*       __restrict__ refGrad)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !mask[i]) return;
    const scalar rho = rhoBnd ? rhoBnd[i] : scalar(1);
    refGrad[i] = (phiHbyABnd[i] - rho * sfUBnd[i]) / (magSf[i] * denBnd[i]);
}

__global__ void ownerGatherKernel(
    int                        n,
    const label*  __restrict__ bndCell,
    const scalar* __restrict__ cellField,
    scalar*       __restrict__ bnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bnd[i] = cellField[bndCell[i]];
}
} // namespace

void deviceConstrainPressure(
    const DeviceBoundary&       dbP,
    const DeviceBuffer<scalar>& phiHbyABnd,
    const DeviceBuffer<scalar>& sfUBnd,
    const scalar*               rhoBnd,
    const DeviceBuffer<scalar>& denBnd)
{
    const int n = static_cast<int>(dbP.snGradMask.size());
    if (n == 0) return;   // empty boundary; a mask of zeros makes the kernel itself a no-op
    constrainPressureKernel<<<(n + 255) / 256, 256>>>(
        n, dbP.snGradMask.data(), phiHbyABnd.data(), sfUBnd.data(), rhoBnd, denBnd.data(),
        dbP.magSf.data(),
        const_cast<DeviceBuffer<scalar>&>(dbP.refGrad).data());
    cudaCheck(cudaGetLastError(), "deviceConstrainPressure");
}

void deviceOwnerGather(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& cellField,
    DeviceBuffer<scalar>&       bnd)
{
    bnd.resize(dm.nBndFaces);
    if (dm.nBndFaces == 0) return;
    ownerGatherKernel<<<(dm.nBndFaces + 255) / 256, 256>>>(
        dm.nBndFaces, dm.bndCell.data(), cellField.data(), bnd.data());
    cudaCheck(cudaGetLastError(), "deviceOwnerGather");
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
    relaxKernel<<<nBlocks(A.nCells), TPB>>>(A.nCells, A.ownerStart, A.losort, A.losortStart, A.upper, A.lower,
                                            dm.bndCellStart.data(), dm.bndPerm.data(), iCbnd.data(), A.diag, alpha,
                                            relaxedDiag.data(), delta.data(), cycSumOff, iCmaxMag, iCmin);
    cudaCheck(cudaGetLastError(), "relaxDiag");
}
namespace {
__global__
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
    cmptMaxMag3Kernel<<<nBlocks(n), TPB>>>(n, a.data(), b.data(), c.data(), out.data());
    cudaCheck(cudaGetLastError(), "cmptMaxMag3");
}


__global__
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
    cmptMin3Kernel<<<nBlocks(n), TPB>>>(n, a.data(), b.data(), c.data(), out.data());
    cudaCheck(cudaGetLastError(), "cmptMin3");
}

} // namespace brae
