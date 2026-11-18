// Implicit transient ddt kernels -- see device_ddt.cuh for the OF-2412 correspondence + source refs.
#include "device_ddt.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// diag[i] += a*V[i]   with a = coefft*rDeltaT*rho.
__global__ void ddtDiagKernel(const scalar* __restrict__ V, int n, scalar a, scalar* __restrict__ diag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    diag[i] += a * V[i];
}

// backward/Euler : source[i] += V[i]*(b0*old[i] - b00*old2[i])   (b0=rDeltaT*rho*coefft0, b00=rDeltaT*rho*coefft00)
// CrankNicolson  : source[i] += V[i]*(b0*old[i] + dc*ddt0[i])    (dc=rho*ocCoeff; the ddt0 term REPLACES the old2 term)
// old2/ddt0 may be null (Euler / bootstrap): the corresponding term is then structurally absent.
__global__ void ddtSourceKernel(
    const scalar* __restrict__ V, int n, scalar b0, scalar b00, scalar dc,
    const scalar* __restrict__ old, const scalar* __restrict__ old2, const scalar* __restrict__ ddt0,
    scalar* __restrict__ source)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scalar s = b0 * old[i];
    if (ddt0)      s += dc * ddt0[i];       // CrankNicolson stored-old-ddt term
    else if (old2) s -= b00 * old2[i];      // backward second-old-level term
    source[i] += V[i] * s;
}

// CrankNicolson ddt0 recurrence: ddt0[i] = a*(old[i] - old2[i]) - oc*ddt0[i]   with a = coefft*rDeltaT0. In place.
__global__ void ddt0UpdateKernel(
    int n, scalar a, scalar oc, const scalar* __restrict__ old, const scalar* __restrict__ old2, scalar* __restrict__ ddt0)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ddt0[i] = a * (old[i] - old2[i]) - oc * ddt0[i];
}
}  // namespace

void deviceFvmDdtDiag(
    const DeviceBuffer<scalar>& V, const DdtCoeffs& c, scalar rho, DeviceBuffer<scalar>& diag)
{
    if (!c.active) return;                                   // steadyState / bootstrap -> no-op
    const int n = static_cast<int>(V.size());
    ddtDiagKernel<<<nBlocks(n), TPB>>>(V.data(), n, c.coefft * c.rDeltaT * rho, diag.data());
    cudaCheck(cudaGetLastError(), "fvmDdtDiag");
}

void deviceFvmDdtSource(
    const DeviceBuffer<scalar>& V, const DdtCoeffs& c, scalar rho,
    const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2, DeviceBuffer<scalar>& source,
    const DeviceBuffer<scalar>* ddt0)
{
    if (!c.active) return;
    const int n = static_cast<int>(V.size());
    const scalar b0  = c.rDeltaT * rho * c.coefft0;
    const scalar b00 = c.rDeltaT * rho * c.coefft00;
    const scalar dc  = rho * c.ocCoeff;                                    // CrankNicolson ddt0 weight
    const scalar* old2p = psiOld2.size() ? psiOld2.data() : nullptr;
    const scalar* ddt0p = (c.cn && ddt0 && ddt0->size()) ? ddt0->data() : nullptr;   // CN uses ddt0, NOT old2
    ddtSourceKernel<<<nBlocks(n), TPB>>>(V.data(), n, b0, b00, dc, psiOld.data(), old2p, ddt0p, source.data());
    cudaCheck(cudaGetLastError(), "fvmDdtSource");
}

void deviceFvmDdtUpdateDdt0(
    const DdtCoeffs& c, const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2, DeviceBuffer<scalar>& ddt0)
{
    if (!c.cn || !psiOld2.size() || !ddt0.size()) return;                  // steady/Euler/backward/first-step -> ddt0 untouched (stays 0)
    const int n = static_cast<int>(psiOld.size());
    ddt0UpdateKernel<<<nBlocks(n), TPB>>>(n, c.coefft0dd * c.rDeltaT0, c.ocCoeff, psiOld.data(), psiOld2.data(), ddt0.data());
    cudaCheck(cudaGetLastError(), "fvmDdtUpdateDdt0");
}

void deviceFvmDdt(
    const DeviceBuffer<scalar>& V, const DdtCoeffs& c, scalar rho,
    const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source, const DeviceBuffer<scalar>* ddt0)
{
    deviceFvmDdtDiag(V, c, rho, diag);
    deviceFvmDdtSource(V, c, rho, psiOld, psiOld2, source, ddt0);
}

namespace {
// One face. Written once and used for both the internal and the boundary sweep so the two cannot drift:
// the only difference between them is where phi_old and rAU_face come from.
// `phiCorrEff` is the quantity actually corrected; it equals phiCorr for Euler and carries backward's
// two-level combination otherwise. It is a SEPARATE argument because OF computes the coupling coefficient
// from the single old level -- fvcDdtPhiCoeff(U.oldTime(), Sf & Uf.oldTime()) -- and then applies that
// coefficient to the scheme's own multi-level correction. Folding the combination into the coefficient
// too would change the limiter, not just the term it limits.
__device__ __forceinline__ scalar ddtCorrFace(scalar phiOld, scalar fluxUold, scalar phiCorrEff,
                                              scalar rAUf, scalar rDeltaT)
{
    const scalar phiCorr = phiOld - fluxUold;
    // ddtScheme<Type>::fvcDdtPhiCoeff, the ddtPhiCoeff_ < 0 branch (the default -- v2412 never reads
    // ddtPhiCoeff from fvSchemes, so that branch is the only one a case can reach):
    //     coeff = 1 - min(mag(phiCorr)/(mag(phi) + SMALL), 1)
    // SMALL is Foam::SMALL = 1e-15 in double precision.
    const scalar coeff = scalar(1) - fmin(fabs(phiCorr)/(fabs(phiOld) + scalar(1e-15)), scalar(1));
    return rAUf*coeff*rDeltaT*phiCorrEff;
}

__global__ void ddtCorrIntK(int nIf, const scalar* phiOld, const scalar* fluxUold, const scalar* rAUf,
                            const scalar* phiCorrEff, scalar rDeltaT, scalar* out)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar eff = phiCorrEff ? phiCorrEff[f] : (phiOld[f] - fluxUold[f]);
    out[f] += ddtCorrFace(phiOld[f], fluxUold[f], eff, rAUf[f], rDeltaT);
}

__global__ void ddtCorrBndK(int nBf, const scalar* phiOld, const scalar* fluxUold, const scalar* rAUb,
                            const scalar* mask, const scalar* phiCorrEff, scalar rDeltaT, scalar* out)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    if (mask[b] == scalar(0)) return;   // fixesValue / cyclicAMI: OF sets the coupling coefficient to 0
    const scalar eff = phiCorrEff ? phiCorrEff[b] : (phiOld[b] - fluxUold[b]);
    out[b] += ddtCorrFace(phiOld[b], fluxUold[b], eff, rAUb[b], rDeltaT);
}
}   // namespace

void deviceDdtCorrFlux(
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& phiOldInt,
    const DeviceBuffer<scalar>& phiOldBnd,
    const DeviceBuffer<scalar>& fluxUoldInt,
    const DeviceBuffer<scalar>& fluxUoldBnd,
    const DeviceBuffer<scalar>& rAUf,
    const DeviceBuffer<scalar>& rAUbnd,
    const DeviceBuffer<scalar>& coeffMask,
    scalar                      rDeltaT,
    DeviceBuffer<scalar>&       outInt,
    DeviceBuffer<scalar>&       outBnd,
    const DeviceBuffer<scalar>* phiCorrEffInt,
    const DeviceBuffer<scalar>* phiCorrEffBnd)
{
    const int nIf = nInternalFaces;
    if (nIf > 0 && (int)phiOldInt.size() >= nIf && (int)fluxUoldInt.size() >= nIf
        && (int)rAUf.size() >= nIf && (int)outInt.size() >= nIf)
    {
        ddtCorrIntK<<<nBlocks(nIf), TPB>>>(nIf, phiOldInt.data(), fluxUoldInt.data(), rAUf.data(),
                                           (phiCorrEffInt && (int)phiCorrEffInt->size() >= nIf)
                                               ? phiCorrEffInt->data() : nullptr,
                                           rDeltaT, outInt.data());
        cudaCheck(cudaGetLastError(), "ddtCorrInt");
    }
    const int nBf = (int)outBnd.size();
    if (nBf > 0 && (int)phiOldBnd.size() >= nBf && (int)fluxUoldBnd.size() >= nBf
        && (int)rAUbnd.size() >= nBf && (int)coeffMask.size() >= nBf)
    {
        ddtCorrBndK<<<nBlocks(nBf), TPB>>>(nBf, phiOldBnd.data(), fluxUoldBnd.data(), rAUbnd.data(),
                                           coeffMask.data(),
                                           (phiCorrEffBnd && (int)phiCorrEffBnd->size() >= nBf)
                                               ? phiCorrEffBnd->data() : nullptr,
                                           rDeltaT, outBnd.data());
        cudaCheck(cudaGetLastError(), "ddtCorrBnd");
    }
}

namespace {
__global__ void correctUfK(int n, const label* __restrict__ idx,
                           const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                           const scalar* __restrict__ Sfz, const scalar* __restrict__ magSf,
                           const scalar* __restrict__ phi,
                           scalar* __restrict__ ux, scalar* __restrict__ uy, scalar* __restrict__ uz)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int f = idx ? (int)idx[i] : i;
    const scalar ms = magSf[f];
    if (!(ms > scalar(0))) return;              // a zero-area face (an uncovered ACMI face) has no normal
    const scalar nx = Sfx[f]/ms, ny = Sfy[f]/ms, nz = Sfz[f]/ms;
    const scalar nu = nx*ux[i] + ny*uy[i] + nz*uz[i];
    const scalar c  = phi[i]/ms - nu;
    ux[i] += nx*c;  uy[i] += ny*c;  uz[i] += nz*c;
}

__global__ void dotSfK(int n, const label* __restrict__ idx,
                       const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                       const scalar* __restrict__ Sfz, const scalar* __restrict__ ux,
                       const scalar* __restrict__ uy, const scalar* __restrict__ uz,
                       scalar* __restrict__ out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int f = idx ? (int)idx[i] : i;
    out[i] = Sfx[f]*ux[i] + Sfy[f]*uy[i] + Sfz[f]*uz[i];
}
}   // namespace

void deviceCorrectUf(
    int n, const label* faceIdx,
    const DeviceBuffer<scalar>& Sfx, const DeviceBuffer<scalar>& Sfy, const DeviceBuffer<scalar>& Sfz,
    const DeviceBuffer<scalar>& magSf, const DeviceBuffer<scalar>& phi,
    DeviceBuffer<scalar>& ufx, DeviceBuffer<scalar>& ufy, DeviceBuffer<scalar>& ufz)
{
    if (n <= 0 || (int)phi.size() < n || (int)ufx.size() < n) return;
    correctUfK<<<nBlocks(n), TPB>>>(n, faceIdx, Sfx.data(), Sfy.data(), Sfz.data(), magSf.data(),
                                    phi.data(), ufx.data(), ufy.data(), ufz.data());
    cudaCheck(cudaGetLastError(), "correctUf");
}

namespace {
// OF fvc::surfaceSum(mag(phi)) per cell: |phi| lands on the owner AND the neighbour of every internal
// face, and on the owner of every boundary face.
__global__
void surfSumMagIntK(int nIf, const label* __restrict__ own, const label* __restrict__ nei,
                    const scalar* __restrict__ phi, scalar* __restrict__ out)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar a = fabs(phi[f]);
    atomicAdd(&out[own[f]], a);
    atomicAdd(&out[nei[f]], a);
}
__global__
void surfSumMagOwnK(int n, const label* __restrict__ own, const scalar* __restrict__ phi,
                    scalar* __restrict__ out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&out[own[i]], fabs(phi[i]));
}
} // namespace

// surfaceSum(mag(phi)) on the device, INCLUDING the coupled interfaces.
//
// The host version this replaces read the internal and DeviceBoundary flux arrays only. cyclic and AMI
// flux live in their own buffers and were simply absent, so every cell on a coupled interface had its
// Courant number computed from a partial flux sum -- understated exactly where a rotating interface
// makes it largest. On an adaptive-deltaT case that is not a diagnostic error: the understated maxCo
// feeds setDeltaT and the next step is taken too large.
void deviceSurfaceSumMagPhi(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<label>*  cycOwn, const DeviceBuffer<scalar>* cycPhi,
    const DeviceBuffer<label>*  amiOwn, const DeviceBuffer<scalar>* amiPhi,
    DeviceBuffer<scalar>&       out)
{
    out.copyFrom(std::vector<scalar>(static_cast<std::size_t>(dm.nCells), scalar(0)));
    if (dm.nInternalFaces > 0)
    {
        surfSumMagIntK<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(),
                                                            dm.nei.data(), phiInt.data(), out.data());
        cudaCheck(cudaGetLastError(), "surfSumMagInt");
    }
    if (dm.nBndFaces > 0 && (int)phiBnd.size() >= dm.nBndFaces)
    {
        surfSumMagOwnK<<<nBlocks(dm.nBndFaces), TPB>>>(dm.nBndFaces, dm.bndCell.data(), phiBnd.data(),
                                                       out.data());
        cudaCheck(cudaGetLastError(), "surfSumMagBnd");
    }
    auto addIface = [&](const DeviceBuffer<label>* own, const DeviceBuffer<scalar>* phi)
    {
        if (!own || !phi) return;
        const int n = static_cast<int>(phi->size());
        if (n == 0 || (int)own->size() < n) return;
        surfSumMagOwnK<<<nBlocks(n), TPB>>>(n, own->data(), phi->data(), out.data());
        cudaCheck(cudaGetLastError(), "surfSumMagIface");
    };
    addIface(cycOwn, cycPhi);
    addIface(amiOwn, amiPhi);
}


void deviceDotSf(
    int n, const label* faceIdx,
    const DeviceBuffer<scalar>& Sfx, const DeviceBuffer<scalar>& Sfy, const DeviceBuffer<scalar>& Sfz,
    const DeviceBuffer<scalar>& ufx, const DeviceBuffer<scalar>& ufy, const DeviceBuffer<scalar>& ufz,
    DeviceBuffer<scalar>& out)
{
    out.resize(n);
    if (n <= 0 || (int)ufx.size() < n) return;
    dotSfK<<<nBlocks(n), TPB>>>(n, faceIdx, Sfx.data(), Sfy.data(), Sfz.data(),
                                ufx.data(), ufy.data(), ufz.data(), out.data());
    cudaCheck(cudaGetLastError(), "dotSf");
}

namespace {
// rDT: localEuler's per-cell rDeltaT (null: Euler's one rDeltaT). localEulerDdtScheme.C:283-310 and :139-160 are
// EulerDdtScheme's expressions with the field in the scalar's place, term for term
__global__ void eulerFvmDdtRhoKernel(int n, scalar rDeltaT, const scalar* __restrict__ rho,
                                     const scalar* __restrict__ rho0, const scalar* __restrict__ vf0,
                                     const scalar* __restrict__ V, scalar* __restrict__ diag,
                                     scalar* __restrict__ source, const scalar* __restrict__ rDT)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    const scalar r = rDT ? rDT[c] : rDeltaT;
    diag[c] = __dmul_rn(__dmul_rn(r, rho[c]), V[c]);                          // rDeltaT*rho*V
    source[c] = __dmul_rn(__dmul_rn(__dmul_rn(r, rho0[c]), vf0[c]), V[c]);    // rDeltaT*rho0*vf0*V
}
__global__ void eulerFvcDdtRhoSubKernel(int n, scalar rDeltaT, const scalar* __restrict__ rho,
                                        const scalar* __restrict__ vf, const scalar* __restrict__ rho0,
                                        const scalar* __restrict__ vf0, const scalar* __restrict__ V,
                                        scalar* __restrict__ source, const scalar* __restrict__ rDT)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    const scalar r = rDT ? rDT[c] : rDeltaT;
    const scalar d = __dmul_rn(r, __dmul_rn(rho[c], vf[c]) - __dmul_rn(rho0[c], vf0[c]));   // fvc::ddt
    source[c] = source[c] + __dmul_rn(V[c], d);                                     // tC.source() += V*su
}
} // namespace

void deviceEulerFvmDdtRho(scalar rDeltaT, const DeviceBuffer<scalar>& rho, const DeviceBuffer<scalar>& rho0,
                          const DeviceBuffer<scalar>& vf0, const DeviceBuffer<scalar>& V,
                          DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source)
{
    const int n = static_cast<int>(V.size());
    diag.resize(n);
    source.resize(n);
    eulerFvmDdtRhoKernel<<<nBlocks(n), TPB>>>(n, rDeltaT, rho.data(), rho0.data(), vf0.data(), V.data(),
                                              diag.data(), source.data(), nullptr);
    cudaCheck(cudaGetLastError(), "fvm::ddt(rho, vf)");
}

void deviceEulerFvcDdtRhoSub(scalar rDeltaT, const DeviceBuffer<scalar>& rho, const DeviceBuffer<scalar>& vf,
                             const DeviceBuffer<scalar>& rho0, const DeviceBuffer<scalar>& vf0,
                             const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& source)
{
    const int n = static_cast<int>(V.size());
    eulerFvcDdtRhoSubKernel<<<nBlocks(n), TPB>>>(n, rDeltaT, rho.data(), vf.data(), rho0.data(), vf0.data(),
                                                 V.data(), source.data(), nullptr);
    cudaCheck(cudaGetLastError(), "- fvc::ddt(rho, vf)");
}

void deviceLocalEulerFvmDdtRho(const DeviceBuffer<scalar>& rDeltaT, const DeviceBuffer<scalar>& rho,
                               const DeviceBuffer<scalar>& rho0, const DeviceBuffer<scalar>& vf0,
                               const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source)
{
    const int n = static_cast<int>(V.size());
    diag.resize(n);
    source.resize(n);
    eulerFvmDdtRhoKernel<<<nBlocks(n), TPB>>>(n, 0, rho.data(), rho0.data(), vf0.data(), V.data(), diag.data(),
                                              source.data(), rDeltaT.data());
    cudaCheck(cudaGetLastError(), "fvm::ddt(rho, vf), localEuler");
}

void deviceLocalEulerFvcDdtRhoSub(const DeviceBuffer<scalar>& rDeltaT, const DeviceBuffer<scalar>& rho,
                                  const DeviceBuffer<scalar>& vf, const DeviceBuffer<scalar>& rho0,
                                  const DeviceBuffer<scalar>& vf0, const DeviceBuffer<scalar>& V,
                                  DeviceBuffer<scalar>& source)
{
    const int n = static_cast<int>(V.size());
    eulerFvcDdtRhoSubKernel<<<nBlocks(n), TPB>>>(n, 0, rho.data(), vf.data(), rho0.data(), vf0.data(), V.data(),
                                                 source.data(), rDeltaT.data());
    cudaCheck(cudaGetLastError(), "- fvc::ddt(rho, vf), localEuler");
}

}  // namespace brae
