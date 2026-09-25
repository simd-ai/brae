// Implicit transient ddt kernels -- see device_ddt.cuh for the OF-2412 correspondence + source refs.
#include "device_ddt.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// diag[i] += a*V[i]   with a = coefft*rDeltaT*rho.
__device__ void ddtDiagKernel(const scalar* __restrict__ V, int n, scalar a, scalar* __restrict__ diag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    diag[i] += a * V[i];
}

// backward/Euler : source[i] += V[i]*(b0*old[i] - b00*old2[i])   (b0=rDeltaT*rho*coefft0, b00=rDeltaT*rho*coefft00)
// CrankNicolson  : source[i] += V[i]*(b0*old[i] + dc*ddt0[i])    (dc=rho*ocCoeff; the ddt0 term REPLACES the old2 term)
// old2/ddt0 may be null (Euler / bootstrap): the corresponding term is then structurally absent.
__device__ void ddtSourceKernel(
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
__device__ void ddt0UpdateKernel(
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
    const scalar* Vd = V.data(); const scalar a = c.coefft * c.rDeltaT * rho; scalar* diagd = diag.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { ddtDiagKernel(Vd, n, a, diagd); });
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
    const scalar* Vd = V.data(); const scalar* psiOldd = psiOld.data(); scalar* sourced = source.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
        ddtSourceKernel(Vd, n, b0, b00, dc, psiOldd, old2p, ddt0p, sourced);
    });
    cudaCheck(cudaGetLastError(), "fvmDdtSource");
}

void deviceFvmDdtUpdateDdt0(
    const DdtCoeffs& c, const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2, DeviceBuffer<scalar>& ddt0)
{
    if (!c.cn || !psiOld2.size() || !ddt0.size()) return;                  // steady/Euler/backward/first-step -> ddt0 untouched (stays 0)
    const int n = static_cast<int>(psiOld.size());
    const scalar a = c.coefft0dd * c.rDeltaT0, oc = c.ocCoeff;
    const scalar* psiOldd = psiOld.data(); const scalar* psiOld2d = psiOld2.data(); scalar* ddt0d = ddt0.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { ddt0UpdateKernel(n, a, oc, psiOldd, psiOld2d, ddt0d); });
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

__device__ void ddtCorrIntK(int nIf, const scalar* phiOld, const scalar* fluxUold, const scalar* rAUf,
                            const scalar* phiCorrEff, scalar rDeltaT, scalar* out)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar eff = phiCorrEff ? phiCorrEff[f] : (phiOld[f] - fluxUold[f]);
    out[f] += ddtCorrFace(phiOld[f], fluxUold[f], eff, rAUf[f], rDeltaT);
}

__device__ void ddtCorrBndK(int nBf, const scalar* phiOld, const scalar* fluxUold, const scalar* rAUb,
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
        const scalar* phiOldd = phiOldInt.data(); const scalar* fluxUoldd = fluxUoldInt.data(); const scalar* rAUfd = rAUf.data();
        const scalar* eff = (phiCorrEffInt && (int)phiCorrEffInt->size() >= nIf) ? phiCorrEffInt->data() : nullptr;
        scalar* outIntd = outInt.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () {
            ddtCorrIntK(nIf, phiOldd, fluxUoldd, rAUfd, eff, rDeltaT, outIntd);
        });
        cudaCheck(cudaGetLastError(), "ddtCorrInt");
    }
    const int nBf = (int)outBnd.size();
    if (nBf > 0 && (int)phiOldBnd.size() >= nBf && (int)fluxUoldBnd.size() >= nBf
        && (int)rAUbnd.size() >= nBf && (int)coeffMask.size() >= nBf)
    {
        const scalar* phiOldd = phiOldBnd.data(); const scalar* fluxUoldd = fluxUoldBnd.data(); const scalar* rAUbndd = rAUbnd.data();
        const scalar* maskd = coeffMask.data();
        const scalar* eff = (phiCorrEffBnd && (int)phiCorrEffBnd->size() >= nBf) ? phiCorrEffBnd->data() : nullptr;
        scalar* outBndd = outBnd.data();
        pcudaParallelFor(nBlocks(nBf), TPB, [=] __device__ () {
            ddtCorrBndK(nBf, phiOldd, fluxUoldd, rAUbndd, maskd, eff, rDeltaT, outBndd);
        });
        cudaCheck(cudaGetLastError(), "ddtCorrBnd");
    }
}

namespace {
__device__ void correctUfK(int n, const label* __restrict__ idx,
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

__device__ void dotSfK(int n, const label* __restrict__ idx,
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
    const scalar *Sfxd=Sfx.data(),*Sfyd=Sfy.data(),*Sfzd=Sfz.data(),*magSfd=magSf.data(),*phid=phi.data();
    scalar *ufxd=ufx.data(),*ufyd=ufy.data(),*ufzd=ufz.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
        correctUfK(n, faceIdx, Sfxd, Sfyd, Sfzd, magSfd, phid, ufxd, ufyd, ufzd);
    });
    cudaCheck(cudaGetLastError(), "correctUf");
}

namespace {
// OF fvc::surfaceSum(mag(phi)) per cell: |phi| lands on the owner AND the neighbour of every internal
// face, and on the owner of every boundary face.
__device__
void surfSumMagIntK(int nIf, const label* __restrict__ own, const label* __restrict__ nei,
                    const scalar* __restrict__ phi, scalar* __restrict__ out)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar a = fabs(phi[f]);
    atomicAdd(&out[own[f]], a);
    atomicAdd(&out[nei[f]], a);
}
__device__
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
    scalar* outd = out.data();
    if (dm.nInternalFaces > 0)
    {
        const int nIf = dm.nInternalFaces;
        const label *ownd = dm.owner.data(), *neid = dm.nei.data(); const scalar* phiIntd = phiInt.data();
        pcudaParallelFor(nBlocks(nIf), TPB, [=] __device__ () { surfSumMagIntK(nIf, ownd, neid, phiIntd, outd); });
        cudaCheck(cudaGetLastError(), "surfSumMagInt");
    }
    if (dm.nBndFaces > 0 && (int)phiBnd.size() >= dm.nBndFaces)
    {
        const int nBf = dm.nBndFaces;
        const label* bndCelld = dm.bndCell.data(); const scalar* phiBndd = phiBnd.data();
        pcudaParallelFor(nBlocks(nBf), TPB, [=] __device__ () { surfSumMagOwnK(nBf, bndCelld, phiBndd, outd); });
        cudaCheck(cudaGetLastError(), "surfSumMagBnd");
    }
    auto addIface = [&](const DeviceBuffer<label>* own, const DeviceBuffer<scalar>* phi)
    {
        if (!own || !phi) return;
        const int n = static_cast<int>(phi->size());
        if (n == 0 || (int)own->size() < n) return;
        const label* ownd = own->data(); const scalar* phid = phi->data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { surfSumMagOwnK(n, ownd, phid, outd); });
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
    const scalar *Sfxd=Sfx.data(),*Sfyd=Sfy.data(),*Sfzd=Sfz.data(),*ufxd=ufx.data(),*ufyd=ufy.data(),*ufzd=ufz.data();
    scalar* outd = out.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { dotSfK(n, faceIdx, Sfxd, Sfyd, Sfzd, ufxd, ufyd, ufzd, outd); });
    cudaCheck(cudaGetLastError(), "dotSf");
}

}  // namespace brae
