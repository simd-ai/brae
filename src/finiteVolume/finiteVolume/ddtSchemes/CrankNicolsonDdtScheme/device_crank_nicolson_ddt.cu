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

// THE MOVING BRANCH's ddt0 (CrankNicolsonDdtScheme.C:1029-1047), transcribed from the host reference
// term for term and in its order -- (V0*ro)*old, then rDtCoef0*(a - b), then the reciprocal of V0 as
// a multiply, because that is how `times(scalar(1)/V0, num)` rounds:
//   ddt0 = (rDtCoef0*(V0*rhoOld*old - V00*rhoOO*oo) - V00*offCentre(ddt0))/V0
__global__ void cnDdt0UpdateMovingKernel(
    int n,
    scalar rDtCoef0,
    scalar oc,
    const scalar* rhoOld,
    const scalar* rhoOO,
    const scalar* old,
    const scalar* oo,
    const scalar* V0,
    const scalar* V00,
    scalar* ddt0)
{
    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    if (c >= n) return;
    const scalar ro = rhoOld ? rhoOld[c] : scalar(1);
    const scalar roo = rhoOO ? rhoOO[c] : scalar(1);
    const scalar a = (V0[c]*ro)*old[c];
    const scalar b = (V00[c]*roo)*oo[c];
    const scalar num = rDtCoef0*(a - b) - V00[c]*offCentre(oc, ddt0[c]);
    ddt0[c] = (scalar(1)/V0[c])*num;
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

// THE PERIODIC PAIR's half of the same expression. Its faces are in neither the internal nor the
// boundary array -- they carry their own owner/neighbour cells, weights and Sf (DeviceCyclic) -- and
// the arithmetic is the internal kernel's, face for face. The Euler twin is
// deviceInterDdtCorrCyclic; without this one the pair silently took THAT under CrankNicolson's name.
__global__ void cnDdtCorrCyclicKernel(
    int nIf,
    const label* own,
    const label* nbr,
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
    const int N = nbr[f];
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

// fvcDdtUfCorr's internal face, transcribed from the host reference (crank_nicolson_ddt_scheme_cpp.cu,
// fvcDdtUfCorr): the flux side is (Sf & Uf.oldTime()), the two vectors are subtracted BEFORE the dot,
// and the interpolation is lambda*(P - N) + N.
__global__ void cnDdtUfCorrInternalKernel(
    int nIf,
    const label* own,
    const label* nei,
    const scalar* lambda,
    const scalar* Sfx,
    const scalar* Sfy,
    const scalar* Sfz,
    const scalar* ufox,
    const scalar* ufoy,
    const scalar* ufoz,
    const scalar* dUfdt0x,
    const scalar* dUfdt0y,
    const scalar* dUfdt0z,
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
    const scalar sx = Sfx[f], sy = Sfy[f], sz = Sfz[f];
    // the coefficient's two arguments: (Sf & Uf.oldTime()) and the flux of interpolate(U.oldTime())
    const scalar phiOld = sx*ufox[f] + sy*ufoy[f] + sz*ufoz[f];
    const scalar ux = l*(uox[P] - uox[N]) + uox[N];
    const scalar uy = l*(uoy[P] - uoy[N]) + uoy[N];
    const scalar uz = l*(uoz[P] - uoz[N]) + uoz[N];
    const scalar phiCorr = phiOld - (sx*ux + sy*uy + sz*uz);
    // lhs = rDtCoef*Uf.oldTime() + offCentre(dUfdt0), rhs = interpolate(W), both vectors
    const scalar lx = rDtCoef*ufox[f] + offCentre(oc, dUfdt0x[f]);
    const scalar ly = rDtCoef*ufoy[f] + offCentre(oc, dUfdt0y[f]);
    const scalar lz = rDtCoef*ufoz[f] + offCentre(oc, dUfdt0z[f]);
    const scalar wx = l*(Wx[P] - Wx[N]) + Wx[N];
    const scalar wy = l*(Wy[P] - Wy[N]) + Wy[N];
    const scalar wz = l*(Wz[P] - Wz[N]) + Wz[N];
    out[f] = ddtCoeffOf(phiOld, phiCorr, given)*(sx*(lx - wx) + sy*(ly - wy) + sz*(lz - wz));
}

// ...and its boundary face. Zero where U fixes a value, as the static twin is; a coupled patch is not
// in this array (the device keeps those faces apart) and the caller refuses a pair.
__global__ void cnDdtUfCorrBoundaryKernel(
    int nBf,
    const label* bndGFace,
    const int* fixesValue,
    const scalar* Sfx,
    const scalar* Sfy,
    const scalar* Sfz,
    const scalar* ufobx,
    const scalar* ufoby,
    const scalar* ufobz,
    const scalar* dUfdt0bx,
    const scalar* dUfdt0by,
    const scalar* dUfdt0bz,
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
    const scalar sx = Sfx[gf], sy = Sfy[gf], sz = Sfz[gf];
    const scalar phiOld = sx*ufobx[b] + sy*ufoby[b] + sz*ufobz[b];
    const scalar phiCorr = phiOld - (sx*uobx[b] + sy*uoby[b] + sz*uobz[b]);
    const scalar lx = rDtCoef*ufobx[b] + offCentre(oc, dUfdt0bx[b]);
    const scalar ly = rDtCoef*ufoby[b] + offCentre(oc, dUfdt0by[b]);
    const scalar lz = rDtCoef*ufobz[b] + offCentre(oc, dUfdt0bz[b]);
    out[b] = ddtCoeffOf(phiOld, phiCorr, given)
           *(sx*(lx - Wbx[b]) + sy*(ly - Wby[b]) + sz*(lz - Wbz[b]));
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
    DeviceBuffer<scalar>* const* src,
    const DeviceBuffer<scalar>* V0,
    const DeviceBuffer<scalar>* V00)
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
    // V0 and V00 come TOGETHER: one of them alone is half a moving mesh, and the static form would
    // then run under the scheme's name on a mesh whose volumes changed
    const bool moving = (V0 != nullptr && V00 != nullptr);
    if ((V0 != nullptr) != (V00 != nullptr))
        throw std::runtime_error(
            "brae CrankNicolson (device) fvm::ddt(" + ddt0.name + "): the moving branch weights the two "
            "old levels by mesh().V0() and mesh().V00(); the caller gave one of them.");
    if (moving && (V0->size() != nC || V00->size() != nC))
        throw std::runtime_error(
            "brae CrankNicolson (device) fvm::ddt(" + ddt0.name + "): V0 and V00 must be one value per cell.");
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
            if (moving)
            {
                cnDdt0UpdateMovingKernel<<<nBlocks(n), TPB>>>(n, rDtCoef0, clock.ocCoeff,
                                                              withRho ? rhoOld->data() : nullptr,
                                                              withRho ? rhoOO->data() : nullptr,
                                                              vfOld[k]->data(), vfOO[k]->data(),
                                                              V0->data(), V00->data(),
                                                              ddt0.internal[k].data());
            }
            else
            {
                cnDdt0UpdateKernel<<<nBlocks(n), TPB>>>(n, rDtCoef0, clock.ocCoeff,
                                                        withRho ? rhoOld->data() : nullptr,
                                                        withRho ? rhoOO->data() : nullptr,
                                                        vfOld[k]->data(), vfOO[k]->data(), ddt0.internal[k].data());
            }
            cudaCheck(cudaGetLastError(), "cn ddt0");
        }
    }
    // ...and the source on the OLD volumes when the mesh moved, which is where the old-time field
    // lives (:1060-1065 against :1077-1082). One kernel, handed a different volume array.
    const scalar* Vsrc = moving ? V0->data() : V.data();
    for (int k = 0; k < nComp; ++k)
    {
        cnSourceKernel<<<nBlocks(n), TPB>>>(n, rDtCoef, clock.ocCoeff, withRho ? rhoOld->data() : nullptr,
                                            vfOld[k]->data(), ddt0.internal[k].data(), Vsrc, src[k]->data());
        cudaCheck(cudaGetLastError(), "cn source");
    }
}


void deviceCnDdtUfCorr(
    const DeviceMesh& dm,
    const cpu::fv::CrankNicolsonClock& clock,
    DeviceCnDdt0& ddt0,
    DeviceCnDdt0& dUfdt0,
    const DeviceBuffer<scalar>* const* UOld,
    const DeviceBuffer<scalar>* const* UOO,
    const DeviceBuffer<scalar>* const* UOldBnd,
    const DeviceBuffer<scalar>* const* UOOBnd,
    const DeviceBuffer<scalar>* const* UfOld,
    const DeviceBuffer<scalar>* const* UfOldBnd,
    const DeviceBuffer<scalar>* const* UfOO,
    const DeviceBuffer<scalar>* const* UfOOBnd,
    const DeviceBuffer<int>& bndUFixesValue,
    scalar ddtPhiCoeff,
    DeviceBuffer<scalar>& outInt,
    DeviceBuffer<scalar>& outBnd,
    const DeviceCyclic* cyc)
{
    const int nC = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nBf = dm.nBndFaces;
    if (clock.deltaT <= scalar(0) || clock.deltaT0 <= scalar(0))
        throw std::runtime_error("brae CrankNicolson (device) ddtCorr(U, Uf): deltaT and deltaT0 must both be positive.");
    if (cyc && cyc->n > 0)
        throw std::runtime_error(
            "brae CrankNicolson (device) ddtCorr(U, Uf): the mesh has a periodic pair and a moving "
            "mesh. Those faces are in neither the internal nor the boundary array, and no moving case "
            "with a pair is gated on this arm.");
    for (int k = 0; k < 3; ++k)
    {
        if (!UOld[k] || !UOO[k] || !UOldBnd[k] || !UOOBnd[k]
         || UOld[k]->size() != static_cast<std::size_t>(nC) || UOO[k]->size() != static_cast<std::size_t>(nC)
         || UOldBnd[k]->size() != static_cast<std::size_t>(nBf) || UOOBnd[k]->size() != static_cast<std::size_t>(nBf))
            throw std::runtime_error(
                "brae CrankNicolson (device) ddtCorr(U, Uf): U.oldTime() and U.oldTime().oldTime() must "
                "be given on every cell and every boundary face, three components each.");
        if (!UfOld[k] || !UfOO[k] || !UfOldBnd[k] || !UfOOBnd[k]
         || UfOld[k]->size() != static_cast<std::size_t>(nIf) || UfOO[k]->size() != static_cast<std::size_t>(nIf)
         || UfOldBnd[k]->size() != static_cast<std::size_t>(nBf) || UfOOBnd[k]->size() != static_cast<std::size_t>(nBf))
            throw std::runtime_error(
                "brae CrankNicolson (device) ddtCorr(U, Uf): Uf.oldTime() and Uf.oldTime().oldTime() must "
                "be given on every internal face and every boundary face, three components each.");
    }
    if (bndUFixesValue.size() != static_cast<std::size_t>(nBf))
        throw std::runtime_error("brae CrankNicolson (device) ddtCorr(U, Uf): the fixes-value mask is not the mesh's.");
    ddt0.lookupOrCreate(clock, 3, static_cast<std::size_t>(nC), static_cast<std::size_t>(nBf));
    dUfdt0.lookupOrCreate(clock, 3, static_cast<std::size_t>(nIf), static_cast<std::size_t>(nBf));

    // rDtCoef is ddt0's, as the static twin has it; dUfdt0's is never asked for
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
            cudaCheck(cudaGetLastError(), "cn ddtCorr(U, Uf) ddt0");
        }
    }
    if (dUfdt0.evaluate(clock))
    {
        const scalar rDtCoef0 = dUfdt0.rDtCoef0(clock);
        for (int k = 0; k < 3; ++k)
        {
            if (nIf) cnDdt0UpdateKernel<<<nBlocks(nIf), TPB>>>(nIf, rDtCoef0, clock.ocCoeff, nullptr, nullptr,
                                                               UfOld[k]->data(), UfOO[k]->data(), dUfdt0.internal[k].data());
            if (nBf) cnDdt0UpdateKernel<<<nBlocks(nBf), TPB>>>(nBf, rDtCoef0, clock.ocCoeff, nullptr, nullptr,
                                                               UfOldBnd[k]->data(), UfOOBnd[k]->data(), dUfdt0.boundary[k].data());
            cudaCheck(cudaGetLastError(), "cn ddtCorr(U, Uf) dUfdt0");
        }
    }

    // W = rDtCoef*U.oldTime() + offCentre(ddt0), the vol field the interpolation is handed
    DeviceBuffer<scalar> W[3], Wb[3];
    for (int k = 0; k < 3; ++k)
    {
        W[k].resize(static_cast<std::size_t>(nC));
        Wb[k].resize(static_cast<std::size_t>(nBf));
        if (nC) cnWKernel<<<nBlocks(nC), TPB>>>(nC, rDtCoef, clock.ocCoeff, UOld[k]->data(), ddt0.internal[k].data(), W[k].data());
        if (nBf) cnWKernel<<<nBlocks(nBf), TPB>>>(nBf, rDtCoef, clock.ocCoeff, UOldBnd[k]->data(), ddt0.boundary[k].data(), Wb[k].data());
        cudaCheck(cudaGetLastError(), "cn ddtCorr(U, Uf) W");
    }
    outInt.resize(static_cast<std::size_t>(nIf));
    outBnd.resize(static_cast<std::size_t>(nBf));
    if (nIf)
    {
        cnDdtUfCorrInternalKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(),
                                                         dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                                         UfOld[0]->data(), UfOld[1]->data(), UfOld[2]->data(),
                                                         dUfdt0.internal[0].data(), dUfdt0.internal[1].data(),
                                                         dUfdt0.internal[2].data(),
                                                         UOld[0]->data(), UOld[1]->data(), UOld[2]->data(),
                                                         W[0].data(), W[1].data(), W[2].data(),
                                                         rDtCoef, clock.ocCoeff, ddtPhiCoeff, outInt.data());
        cudaCheck(cudaGetLastError(), "cn ddtCorr(U, Uf) internal");
    }
    if (nBf)
    {
        cnDdtUfCorrBoundaryKernel<<<nBlocks(nBf), TPB>>>(nBf, dm.bndGFace.data(), bndUFixesValue.data(),
                                                         dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                                         UfOldBnd[0]->data(), UfOldBnd[1]->data(), UfOldBnd[2]->data(),
                                                         dUfdt0.boundary[0].data(), dUfdt0.boundary[1].data(),
                                                         dUfdt0.boundary[2].data(),
                                                         UOldBnd[0]->data(), UOldBnd[1]->data(), UOldBnd[2]->data(),
                                                         Wb[0].data(), Wb[1].data(), Wb[2].data(),
                                                         rDtCoef, clock.ocCoeff, ddtPhiCoeff, outBnd.data());
        cudaCheck(cudaGetLastError(), "cn ddtCorr(U, Uf) boundary");
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
    DeviceBuffer<scalar>& outBnd,
    const DeviceCyclic* cyc,
    DeviceCnDdt0* dphidt0If,
    const DeviceBuffer<scalar>* phiOldIf,
    const DeviceBuffer<scalar>* phiOOIf,
    DeviceBuffer<scalar>* outIf)
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
    // ...and THE PAIR, with W and the coefficients built above: the same face expression on a third
    // array. dphidt0's level for those faces is its own object, which sees the same clock and so
    // carries the same startTimeIndex and the same two coefficients -- splitting the storage of one
    // OpenFOAM surfaceScalarField's levels, not splitting the scheme.
    if (cyc && cyc->n > 0)
    {
        if (!dphidt0If || !phiOldIf || !phiOOIf || !outIf
         || static_cast<int>(phiOldIf->size()) != cyc->n
         || static_cast<int>(phiOOIf->size()) != cyc->n)
            throw std::runtime_error(
                "brae CrankNicolson (device) ddtCorr: the mesh has a periodic pair and the caller gave "
                "no dphidt0 level, no phi.oldTime()/oldTime().oldTime() on its faces, or nowhere to "
                "write. Those faces are in neither the internal nor the boundary array.");
        dphidt0If->lookupOrCreate(clock, 1, static_cast<std::size_t>(cyc->n), 0);
        if (dphidt0If->evaluate(clock))
        {
            const scalar rDtCoef0 = dphidt0If->rDtCoef0(clock);
            cnDdt0UpdateKernel<<<nBlocks(cyc->n), TPB>>>(cyc->n, rDtCoef0, clock.ocCoeff, nullptr, nullptr,
                                                         phiOldIf->data(), phiOOIf->data(),
                                                         dphidt0If->internal[0].data());
            cudaCheck(cudaGetLastError(), "cn ddtCorr dphidt0, interface");
        }
        outIf->resize(static_cast<std::size_t>(cyc->n));
        cnDdtCorrCyclicKernel<<<nBlocks(cyc->n), TPB>>>(cyc->n, cyc->ownCell.data(), cyc->nbrCell.data(),
                                                        cyc->weights.data(),
                                                        cyc->Sfx.data(), cyc->Sfy.data(), cyc->Sfz.data(),
                                                        phiOldIf->data(), dphidt0If->internal[0].data(),
                                                        UOld[0]->data(), UOld[1]->data(), UOld[2]->data(),
                                                        W[0].data(), W[1].data(), W[2].data(),
                                                        rDtCoef, clock.ocCoeff, ddtPhiCoeff, outIf->data());
        cudaCheck(cudaGetLastError(), "cn ddtCorr, interface");
    }
}

}   // namespace brae
