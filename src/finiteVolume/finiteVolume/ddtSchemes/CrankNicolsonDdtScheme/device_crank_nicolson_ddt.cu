// OpenFOAM's CrankNicolson ddt scheme on the device -- see the header.
#include "device_crank_nicolson_ddt.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace brae {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1)/TPB; }

__device__ inline scalar offCentre(
    scalar oc,
    scalar x)
{
    return (oc < scalar(1)) ? oc*x : x;
}

// ddt0 = rDtCoef0*(rhoOld*old - rhoOO*oo) - offCentre(ddt0), one component (rho null = 1)
__global__ void cnDdt0UpdateKernel(
    int n,
    scalar rDtCoef0,
    scalar oc,
    const scalar* rhoOld,
    const scalar* rhoOO,
    const scalar* old,
    const scalar* oo,
    scalar* ddt0)
{
    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    if (c >= n) return;
    const scalar ro = rhoOld ? rhoOld[c] : scalar(1);
    const scalar roo = rhoOO ? rhoOO[c] : scalar(1);
    ddt0[c] = rDtCoef0*(ro*old[c] - roo*oo[c]) - offCentre(oc, ddt0[c]);
}

// diag += (rDtCoef*rho)*V, once
__global__ void cnDiagKernel(
    int n,
    scalar rDtCoef,
    const scalar* rho,
    const scalar* V,
    scalar* diag)
{
    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    if (c >= n) return;
    const scalar r = rho ? rho[c] : scalar(1);
    diag[c] += (rDtCoef*r)*V[c];
}

// src += ((rDtCoef*rhoOld)*old + offCentre(ddt0))*V, per component
__global__ void cnSourceKernel(
    int n,
    scalar rDtCoef,
    scalar oc,
    const scalar* rhoOld,
    const scalar* old,
    const scalar* ddt0,
    const scalar* V,
    scalar* src)
{
    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    if (c >= n) return;
    const scalar ro = rhoOld ? rhoOld[c] : scalar(1);
    src[c] += ((rDtCoef*ro)*old[c] + offCentre(oc, ddt0[c]))*V[c];
}

// W = rDtCoef*old + offCentre(ddt0), per component, cells or boundary faces alike
__global__ void cnWKernel(
    int n,
    scalar rDtCoef,
    scalar oc,
    const scalar* old,
    const scalar* ddt0,
    scalar* W)
{
    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    if (c >= n) return;
    W[c] = rDtCoef*old[c] + offCentre(oc, ddt0[c]);
}

__device__ inline scalar ddtCoeffOf(
    scalar phi,
    scalar phiCorr,
    scalar given)
{
    // doubleScalar.H's SMALL, as the host reference has it
    const scalar kSmall = scalar(1e-15);
    return (given < scalar(0)) ? scalar(1) - fmin(fabs(phiCorr)/(fabs(phi) + kSmall), scalar(1)) : given;
}

// out = coeff*((rDtCoef*phiOld + offCentre(dphidt0)) - (Sf & interpolate(W))) on an internal face, with
// dotInterpolate's Sf & (lambda*(P - N) + N)
__global__ void cnDdtCorrInternalKernel(
    int nIf,
    const label* own,
    const label* nei,
    const scalar* lambda,
    const scalar* Sfx,
    const scalar* Sfy,
    const scalar* Sfz,
    const scalar* phiOld,
    const scalar* dphidt0,
    const scalar* uox,
    const scalar* uoy,
    const scalar* uoz,
    const scalar* Wx,
    const scalar* Wy,
    const scalar* Wz,
    scalar rDtCoef,
    scalar oc,
    scalar given,
    scalar* out)
{
    const int f = blockDim.x*blockIdx.x + threadIdx.x;
    if (f >= nIf) return;
    const int P = own[f];
    const int N = nei[f];
    const scalar l = lambda[f];
    const scalar ux = l*(uox[P] - uox[N]) + uox[N];
    const scalar uy = l*(uoy[P] - uoy[N]) + uoy[N];
    const scalar uz = l*(uoz[P] - uoz[N]) + uoz[N];
    const scalar phiCorr = phiOld[f] - (Sfx[f]*ux + Sfy[f]*uy + Sfz[f]*uz);
    const scalar wx = l*(Wx[P] - Wx[N]) + Wx[N];
    const scalar wy = l*(Wy[P] - Wy[N]) + Wy[N];
    const scalar wz = l*(Wz[P] - Wz[N]) + Wz[N];
    const scalar corr = (rDtCoef*phiOld[f] + offCentre(oc, dphidt0[f])) - (Sfx[f]*wx + Sfy[f]*wy + Sfz[f]*wz);
    out[f] = ddtCoeffOf(phiOld[f], phiCorr, given)*corr;
}

__global__ void cnDdtCorrBoundaryKernel(
    int nBf,
    const label* bndGFace,
    const int* fixesValue,
    const scalar* Sfx,
    const scalar* Sfy,
    const scalar* Sfz,
    const scalar* phiOldBnd,
    const scalar* dphidt0Bnd,
    const scalar* uobx,
    const scalar* uoby,
    const scalar* uobz,
    const scalar* Wbx,
    const scalar* Wby,
    const scalar* Wbz,
    scalar rDtCoef,
    scalar oc,
    scalar given,
    scalar* out)
{
    const int b = blockDim.x*blockIdx.x + threadIdx.x;
    if (b >= nBf) return;
    if (fixesValue[b])
    {
        out[b] = scalar(0);
        return;
    }
    const int gf = bndGFace[b];
    const scalar phiCorr = phiOldBnd[b] - (Sfx[gf]*uobx[b] + Sfy[gf]*uoby[b] + Sfz[gf]*uobz[b]);
    const scalar corr = (rDtCoef*phiOldBnd[b] + offCentre(oc, dphidt0Bnd[b]))
                      - (Sfx[gf]*Wbx[b] + Sfy[gf]*Wby[b] + Sfz[gf]*Wbz[b]);
    out[b] = ddtCoeffOf(phiOldBnd[b], phiCorr, given)*corr;
}

void zeroBuffer(
    DeviceBuffer<scalar>& b,
    std::size_t n)
{
    b.resize(n);
    if (n)
    {
        cudaCheck(cudaMemsetAsync(b.data(), 0, n*sizeof(scalar), cudaStreamPerThread), "cn ddt0 zero");
    }
}

}   // namespace


void DeviceCnDdt0::lookupOrCreate(
    const cpu::fv::CrankNicolsonClock& clock,
    int nComponents,
    std::size_t nInternal,
    std::size_t nBoundary)
{
    if (exists)
    {
        if (nComp != nComponents || internal[0].size() != nInternal || boundary[0].size() != nBoundary)
            throw std::runtime_error("brae CrankNicolson (device): the ddt0 field `" + name + "` does not fit the mesh.");
        return;
    }
    nComp = nComponents;
    for (int k = 0; k < nComp; ++k)
    {
        zeroBuffer(internal[k], nInternal);
        zeroBuffer(boundary[k], nBoundary);
    }
    startTimeIndex = clock.timeIndex;
    timeIndex = clock.timeIndex;
    exists = true;
}


void deviceCnFvmDdt(
    const cpu::fv::CrankNicolsonClock& clock,
    DeviceCnDdt0& ddt0,
    const DeviceBuffer<scalar>* rho,
    const DeviceBuffer<scalar>* rhoOld,
    const DeviceBuffer<scalar>* rhoOO,
    int nComp,
    const DeviceBuffer<scalar>* const* vfOld,
    const DeviceBuffer<scalar>* const* vfOO,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>* const* src)
{
    const std::size_t nC = V.size();
    const int n = static_cast<int>(nC);
    if (clock.deltaT <= scalar(0) || clock.deltaT0 <= scalar(0))
        throw std::runtime_error("brae CrankNicolson (device) fvm::ddt: deltaT and deltaT0 must both be positive.");
    const bool withRho = (rho != nullptr);
    if (withRho != (rhoOld != nullptr) || withRho != (rhoOO != nullptr))
        throw std::runtime_error(
            "brae CrankNicolson (device) fvm::ddt(" + ddt0.name + "): rho, rho.oldTime() and "
            "rho.oldTime().oldTime() come together or not at all.");
    if (withRho && (rho->size() != nC || rhoOld->size() != nC || rhoOO->size() != nC))
        throw std::runtime_error("brae CrankNicolson (device) fvm::ddt(" + ddt0.name + "): the densities must be one value per cell.");
    if (diag.size() != nC)
        throw std::runtime_error("brae CrankNicolson (device) fvm::ddt(" + ddt0.name + "): the diagonal is not one value per cell.");
    for (int k = 0; k < nComp; ++k)
    {
        if (!vfOld[k] || !vfOO[k] || !src[k] || vfOld[k]->size() != nC || vfOO[k]->size() != nC || src[k]->size() != nC)
            throw std::runtime_error(
                "brae CrankNicolson (device) fvm::ddt(" + ddt0.name + "): component " + std::to_string(k)
                + " of vf.oldTime(), vf.oldTime().oldTime() or the source is not one value per cell.");
    }
    ddt0.lookupOrCreate(clock, nComp, nC, 0);
    if (n == 0) return;

    const scalar rDtCoef = ddt0.rDtCoef(clock);
    cnDiagKernel<<<nBlocks(n), TPB>>>(n, rDtCoef, withRho ? rho->data() : nullptr, V.data(), diag.data());
    cudaCheck(cudaGetLastError(), "cn diag");
    if (ddt0.evaluate(clock))
    {
        const scalar rDtCoef0 = ddt0.rDtCoef0(clock);
        for (int k = 0; k < nComp; ++k)
        {
            cnDdt0UpdateKernel<<<nBlocks(n), TPB>>>(n, rDtCoef0, clock.ocCoeff,
                                                    withRho ? rhoOld->data() : nullptr,
                                                    withRho ? rhoOO->data() : nullptr,
                                                    vfOld[k]->data(), vfOO[k]->data(), ddt0.internal[k].data());
            cudaCheck(cudaGetLastError(), "cn ddt0");
        }
    }
    for (int k = 0; k < nComp; ++k)
    {
        cnSourceKernel<<<nBlocks(n), TPB>>>(n, rDtCoef, clock.ocCoeff, withRho ? rhoOld->data() : nullptr,
                                            vfOld[k]->data(), ddt0.internal[k].data(), V.data(), src[k]->data());
        cudaCheck(cudaGetLastError(), "cn source");
    }
}


void deviceCnDdtCorr(
    const DeviceMesh& dm,
    const cpu::fv::CrankNicolsonClock& clock,
    DeviceCnDdt0& ddt0,
    DeviceCnDdt0& dphidt0,
    const DeviceBuffer<scalar>* const* UOld,
    const DeviceBuffer<scalar>* const* UOO,
    const DeviceBuffer<scalar>* const* UOldBnd,
    const DeviceBuffer<scalar>* const* UOOBnd,
    const DeviceBuffer<scalar>& phiOldInt,
    const DeviceBuffer<scalar>& phiOldBnd,
    const DeviceBuffer<scalar>& phiOOInt,
    const DeviceBuffer<scalar>& phiOOBnd,
    const DeviceBuffer<int>& bndUFixesValue,
    scalar ddtPhiCoeff,
    DeviceBuffer<scalar>& outInt,
    DeviceBuffer<scalar>& outBnd)
{
    const int nC = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nBf = dm.nBndFaces;
    if (clock.deltaT <= scalar(0) || clock.deltaT0 <= scalar(0))
        throw std::runtime_error("brae CrankNicolson (device) ddtCorr: deltaT and deltaT0 must both be positive.");
    for (int k = 0; k < 3; ++k)
    {
        if (!UOld[k] || !UOO[k] || !UOldBnd[k] || !UOOBnd[k]
         || UOld[k]->size() != static_cast<std::size_t>(nC) || UOO[k]->size() != static_cast<std::size_t>(nC)
         || UOldBnd[k]->size() != static_cast<std::size_t>(nBf) || UOOBnd[k]->size() != static_cast<std::size_t>(nBf))
            throw std::runtime_error(
                "brae CrankNicolson (device) ddtCorr: U.oldTime() and U.oldTime().oldTime() must be given on "
                "every cell and every boundary face, three components each.");
    }
    if (phiOldInt.size() != static_cast<std::size_t>(nIf) || phiOOInt.size() != static_cast<std::size_t>(nIf)
     || phiOldBnd.size() != static_cast<std::size_t>(nBf) || phiOOBnd.size() != static_cast<std::size_t>(nBf)
     || bndUFixesValue.size() != static_cast<std::size_t>(nBf))
        throw std::runtime_error(
            "brae CrankNicolson (device) ddtCorr: phi.oldTime(), phi.oldTime().oldTime() and the fixes-value "
            "mask must be the mesh's.");
    ddt0.lookupOrCreate(clock, 3, static_cast<std::size_t>(nC), static_cast<std::size_t>(nBf));
    dphidt0.lookupOrCreate(clock, 1, static_cast<std::size_t>(nIf), static_cast<std::size_t>(nBf));

    const scalar rDtCoef = ddt0.rDtCoef(clock);
    if (ddt0.evaluate(clock))
    {
        const scalar rDtCoef0 = ddt0.rDtCoef0(clock);
        for (int k = 0; k < 3; ++k)
        {
            if (nC) cnDdt0UpdateKernel<<<nBlocks(nC), TPB>>>(nC, rDtCoef0, clock.ocCoeff, nullptr, nullptr,
                                                             UOld[k]->data(), UOO[k]->data(), ddt0.internal[k].data());
            if (nBf) cnDdt0UpdateKernel<<<nBlocks(nBf), TPB>>>(nBf, rDtCoef0, clock.ocCoeff, nullptr, nullptr,
                                                               UOldBnd[k]->data(), UOOBnd[k]->data(), ddt0.boundary[k].data());
            cudaCheck(cudaGetLastError(), "cn ddtCorr ddt0");
        }
    }
    if (dphidt0.evaluate(clock))
    {
        const scalar rDtCoef0 = dphidt0.rDtCoef0(clock);
        if (nIf) cnDdt0UpdateKernel<<<nBlocks(nIf), TPB>>>(nIf, rDtCoef0, clock.ocCoeff, nullptr, nullptr,
                                                           phiOldInt.data(), phiOOInt.data(), dphidt0.internal[0].data());
        if (nBf) cnDdt0UpdateKernel<<<nBlocks(nBf), TPB>>>(nBf, rDtCoef0, clock.ocCoeff, nullptr, nullptr,
                                                           phiOldBnd.data(), phiOOBnd.data(), dphidt0.boundary[0].data());
        cudaCheck(cudaGetLastError(), "cn ddtCorr dphidt0");
    }

    // W = rDtCoef*U.oldTime() + offCentre(ddt0), cells and patch values
    DeviceBuffer<scalar> W[3], Wb[3];
    for (int k = 0; k < 3; ++k)
    {
        W[k].resize(static_cast<std::size_t>(nC));
        Wb[k].resize(static_cast<std::size_t>(nBf));
        if (nC) cnWKernel<<<nBlocks(nC), TPB>>>(nC, rDtCoef, clock.ocCoeff, UOld[k]->data(), ddt0.internal[k].data(), W[k].data());
        if (nBf) cnWKernel<<<nBlocks(nBf), TPB>>>(nBf, rDtCoef, clock.ocCoeff, UOldBnd[k]->data(), ddt0.boundary[k].data(), Wb[k].data());
        cudaCheck(cudaGetLastError(), "cn ddtCorr W");
    }
    outInt.resize(static_cast<std::size_t>(nIf));
    outBnd.resize(static_cast<std::size_t>(nBf));
    if (nIf)
    {
        cnDdtCorrInternalKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(),
                                                       dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                                       phiOldInt.data(), dphidt0.internal[0].data(),
                                                       UOld[0]->data(), UOld[1]->data(), UOld[2]->data(),
                                                       W[0].data(), W[1].data(), W[2].data(),
                                                       rDtCoef, clock.ocCoeff, ddtPhiCoeff, outInt.data());
        cudaCheck(cudaGetLastError(), "cn ddtCorr internal");
    }
    if (nBf)
    {
        cnDdtCorrBoundaryKernel<<<nBlocks(nBf), TPB>>>(nBf, dm.bndGFace.data(), bndUFixesValue.data(),
                                                       dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                                       phiOldBnd.data(), dphidt0.boundary[0].data(),
                                                       UOldBnd[0]->data(), UOldBnd[1]->data(), UOldBnd[2]->data(),
                                                       Wb[0].data(), Wb[1].data(), Wb[2].data(),
                                                       rDtCoef, clock.ocCoeff, ddtPhiCoeff, outBnd.data());
        cudaCheck(cudaGetLastError(), "cn ddtCorr boundary");
    }
}

}   // namespace brae
