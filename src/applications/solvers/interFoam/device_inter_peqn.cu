// interFoam's pressure corrector on the device -- see device_inter_peqn.cuh for the four.
#include "device_inter_peqn.cuh"
#include "device_fvc_reconstruct.cuh"
#include "device_mesh.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// OpenFOAM's SMALL, the guard in fvcDdtPhiCoeff's denominator.
__device__ constexpr scalar kSmall = scalar(1.0e-37);

void ckP(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae interFoam device pEqn: ") + what + ": "
                                 + cudaGetErrorString(e));
}

__global__ void buoyancyKernel(
    const scalar* __restrict__ stf, const scalar* __restrict__ ghf,
    const scalar* __restrict__ snGradRho, const scalar* __restrict__ rAUf,
    const scalar* __restrict__ magSf, int n, scalar* __restrict__ out)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    // NO snGrad(p_rgh) -- it is the laplacian being solved, and adding it here counts it twice.
    if (f < n) out[f] = (stf[f] - ghf[f]*snGradRho[f]) * rAUf[f] * magSf[f];
}

__global__ void rhoRAUfKernel(
    const label* __restrict__ own, const label* __restrict__ nei, const scalar* __restrict__ w,
    const scalar* __restrict__ rho, const scalar* __restrict__ rAU, int nIf,
    scalar* __restrict__ out)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    // THE PRODUCT PER CELL, interpolated once. interpolate(rho)*interpolate(rAU) is a different field.
    const scalar po = rho[own[f]] * rAU[own[f]];
    const scalar pn = rho[nei[f]] * rAU[nei[f]];
    out[f] = w[f]*po + (scalar(1) - w[f])*pn;
}

__device__ __forceinline__ scalar ddtCoeff(scalar phiCorr, scalar phiOld, scalar given)
{
    // A NEGATIVE ddtPhiCoeff selects the limiter, which is OpenFOAM's default (ddtScheme.H:135). It
    // switches the correction OFF where it is large compared with the flux -- the opposite of what a
    // constant 1 would do.
    if (given >= scalar(0)) return given;
    const scalar r = fabs(phiCorr) / (fabs(phiOld) + kSmall);
    return scalar(1) - (r < scalar(1) ? r : scalar(1));
}

__global__ void ddtCorrInternalKernel(
    const label* __restrict__ own, const label* __restrict__ nei, const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ phiOld,
    const scalar* __restrict__ uox, const scalar* __restrict__ uoy, const scalar* __restrict__ uoz,
    int nIf, scalar given, scalar rDeltaT, scalar* __restrict__ out)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const int o = own[f], n = nei[f];
    const scalar wf = w[f], wm = scalar(1) - w[f];
    const scalar ux = wf*uox[o] + wm*uox[n];
    const scalar uy = wf*uoy[o] + wm*uoy[n];
    const scalar uz = wf*uoz[o] + wm*uoz[n];
    const scalar phiCorr = phiOld[f] - (ux*Sfx[f] + uy*Sfy[f] + uz*Sfz[f]);
    out[f] = ddtCoeff(phiCorr, phiOld[f], given) * rDeltaT * phiCorr;
}

__global__ void ddtCorrBoundaryKernel(
    const label* __restrict__ bndCell, const label* __restrict__ bndGFace,
    const int* __restrict__ fixesValue,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ phiOldBnd,
    const scalar* __restrict__ uox, const scalar* __restrict__ uoy, const scalar* __restrict__ uoz,
    int nBf, scalar given, scalar rDeltaT, scalar* __restrict__ out)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    // ZERO wherever U fixes a value: the flux there is what the boundary condition says, and there is
    // no inconsistency to correct (ddtScheme.C:~275).
    if (fixesValue[b]) { out[b] = scalar(0); return; }
    const int c = bndCell[b], gf = bndGFace[b];
    // the patch's own face value of U.oldTime() is the face CELL's -- fvcDdtPhiCorr takes
    // patchInternalField there, not the stored patch value
    const scalar interpFlux = uox[c]*Sfx[gf] + uoy[c]*Sfy[gf] + uoz[c]*Sfz[gf];
    const scalar pOld    = phiOldBnd[b];
    const scalar phiCorr = pOld - interpFlux;
    out[b] = ddtCoeff(phiCorr, pOld, given) * rDeltaT * phiCorr;
}

// pe.source += fvc::div(phiHbyA)*V -- a PLUS, fvMatrix::operator== (fvMatrix.C:1855-1862).
__global__ void addDivSourceKernel(const scalar* __restrict__ div, const scalar* __restrict__ V,
                                   int nC, scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) source[c] += div[c]*V[c];
}

// fvMatrix::setReference: source += diag*refValue, then diag += diag. It DOUBLES the entry; replacing
// the row instead gives a different matrix that still solves.
__global__ void setReferenceKernel(int cell, scalar refValue,
                                   scalar* __restrict__ diag, scalar* __restrict__ source)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        source[cell] += diag[cell]*refValue;
        diag[cell]   += diag[cell];
    }
}

__global__ void divideKernel(const scalar* __restrict__ a, const scalar* __restrict__ b,
                             int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

__global__ void addScaledKernel(
    const scalar* __restrict__ hx, const scalar* __restrict__ hy, const scalar* __restrict__ hz,
    const scalar* __restrict__ rAU,
    const scalar* __restrict__ rx, const scalar* __restrict__ ry, const scalar* __restrict__ rz,
    int nC, scalar* __restrict__ ux, scalar* __restrict__ uy, scalar* __restrict__ uz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    ux[c] = hx[c] + rAU[c]*rx[c];
    uy[c] = hy[c] + rAU[c]*ry[c];
    uz[c] = hz[c] + rAU[c]*rz[c];
}

__global__ void staticPressureKernel(
    const scalar* __restrict__ p_rgh, const scalar* __restrict__ rho, const scalar* __restrict__ gh,
    int nC, scalar* __restrict__ p)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) p[c] = p_rgh[c] + rho[c]*gh[c];
}

}   // namespace


void deviceBuoyancyFlux(
    int                         n,
    const DeviceBuffer<scalar>& surfaceTensionForce,
    const DeviceBuffer<scalar>& ghf,
    const DeviceBuffer<scalar>& snGradRho,
    const DeviceBuffer<scalar>& rAUf,
    const DeviceBuffer<scalar>& magSf,
    DeviceBuffer<scalar>&       phig)
{
    if (n <= 0) return;
    if (static_cast<int>(magSf.size()) < n || static_cast<int>(rAUf.size()) < n)
        throw std::runtime_error(
            "brae interFoam device pEqn: magSf and rAUf are the mesh's FULL face arrays here, "
            "internal faces first, and must be at least as long as the fields they scale.");
    phig.resize(static_cast<std::size_t>(n));
    buoyancyKernel<<<nBlocks(n), TPB>>>(surfaceTensionForce.data(), ghf.data(), snGradRho.data(),
                                        rAUf.data(), magSf.data(), n, phig.data());
    ckP(cudaGetLastError(), "buoyancy flux");
}


void deviceRhoRAUf(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& rAU,
    DeviceBuffer<scalar>&       out)
{
    const int nIf = dm.nInternalFaces;
    out.resize(static_cast<std::size_t>(nIf));
    if (nIf == 0) return;
    rhoRAUfKernel<<<nBlocks(nIf), TPB>>>(dm.owner.data(), dm.nei.data(), dm.w.data(),
                                         rho.data(), rAU.data(), nIf, out.data());
    ckP(cudaGetLastError(), "interpolate(rho*rAU)");
}


void deviceDdtCorr(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& phiOldInt,
    const DeviceBuffer<scalar>& phiOldBnd,
    const DeviceBuffer<scalar>& UOldX,
    const DeviceBuffer<scalar>& UOldY,
    const DeviceBuffer<scalar>& UOldZ,
    const DeviceBuffer<int>&    bndUFixesValue,
    scalar                      ddtPhiCoeff,
    scalar                      deltaT,
    DeviceBuffer<scalar>&       outInt,
    DeviceBuffer<scalar>&       outBnd)
{
    if (deltaT <= scalar(0))
        throw std::runtime_error("brae interFoam device ddtCorr: deltaT must be positive.");
    const int nIf = dm.nInternalFaces, nBf = dm.nBndFaces;
    const scalar rDeltaT = scalar(1) / deltaT;

    outInt.resize(static_cast<std::size_t>(nIf));
    if (nIf > 0)
    {
        ddtCorrInternalKernel<<<nBlocks(nIf), TPB>>>(
            dm.owner.data(), dm.nei.data(), dm.w.data(),
            dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), phiOldInt.data(),
            UOldX.data(), UOldY.data(), UOldZ.data(), nIf, ddtPhiCoeff, rDeltaT, outInt.data());
        ckP(cudaGetLastError(), "ddtCorr, internal");
    }
    outBnd.resize(static_cast<std::size_t>(nBf));
    if (nBf > 0)
    {
        ddtCorrBoundaryKernel<<<nBlocks(nBf), TPB>>>(
            dm.bndCell.data(), dm.bndGFace.data(), bndUFixesValue.data(),
            dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), phiOldBnd.data(),
            UOldX.data(), UOldY.data(), UOldZ.data(), nBf, ddtPhiCoeff, rDeltaT, outBnd.data());
        ckP(cudaGetLastError(), "ddtCorr, boundary");
    }
}


void deviceCorrectVelocity(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& HbyAX,
    const DeviceBuffer<scalar>& HbyAY,
    const DeviceBuffer<scalar>& HbyAZ,
    const DeviceBuffer<scalar>& rAU,
    const DeviceBuffer<scalar>& faceFluxInt,
    const DeviceBuffer<scalar>& rAUfInt,
    const DeviceBuffer<scalar>& faceFluxBnd,
    const DeviceBuffer<scalar>& rAUfBnd,
    DeviceBuffer<scalar>&       UX,
    DeviceBuffer<scalar>&       UY,
    DeviceBuffer<scalar>&       UZ)
{
    const int nC = dm.nCells, nIf = dm.nInternalFaces, nBf = dm.nBndFaces;

    // (phig - flux)/rAUf, face by face, BEFORE the reconstruction. The division here and the
    // multiplication by rAU below do not cancel on a non-uniform rAU.
    DeviceBuffer<scalar> ssfInt(static_cast<std::size_t>(nIf)), ssfBnd(static_cast<std::size_t>(nBf));
    if (nIf > 0)
    {
        divideKernel<<<nBlocks(nIf), TPB>>>(faceFluxInt.data(), rAUfInt.data(), nIf, ssfInt.data());
        ckP(cudaGetLastError(), "faceFlux/rAUf");
    }
    if (nBf > 0)
    {
        divideKernel<<<nBlocks(nBf), TPB>>>(faceFluxBnd.data(), rAUfBnd.data(), nBf, ssfBnd.data());
        ckP(cudaGetLastError(), "faceFlux/rAUf, boundary");
    }

    DeviceBuffer<scalar> rx, ry, rz;
    deviceReconstruct(dm, ssfInt, ssfBnd, rx, ry, rz);

    UX.resize(static_cast<std::size_t>(nC));
    UY.resize(static_cast<std::size_t>(nC));
    UZ.resize(static_cast<std::size_t>(nC));
    addScaledKernel<<<nBlocks(nC), TPB>>>(HbyAX.data(), HbyAY.data(), HbyAZ.data(), rAU.data(),
                                          rx.data(), ry.data(), rz.data(), nC,
                                          UX.data(), UY.data(), UZ.data());
    ckP(cudaGetLastError(), "U = HbyA + rAU*reconstruct(...)");
}


void deviceStaticPressure(
    int                         nC,
    const DeviceBuffer<scalar>& p_rgh,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& gh,
    DeviceBuffer<scalar>&       p)
{
    if (nC <= 0) return;
    p.resize(static_cast<std::size_t>(nC));
    staticPressureKernel<<<nBlocks(nC), TPB>>>(p_rgh.data(), rho.data(), gh.data(), nC, p.data());
    ckP(cudaGetLastError(), "p = p_rgh + rho*gh");
}


void deviceInterAssemblePEqn(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rAUfInt,
    const DeviceBuffer<scalar>& phiHbyAInt,
    const DeviceBuffer<scalar>& phiHbyABnd,
    bool                        needReference,
    int                         pRefCell,
    scalar                      pRefValue,
    DevicePressureMatrix&       P)
{
    const int nC = dm.nCells;

    // fvm::laplacian(rAUf, p_rgh), UNCORRECTED -- interFoam's pEqn passes corrected=false, and the
    // shipped tutorials' `laplacianSchemes default Gauss linear corrected` applies to the momentum
    // equation, not to this one. A corrected laplacian here would need its deferred source AND the
    // matching faceFluxCorrection, or phi comes out non-conservative while the equation still solves.
    deviceLaplacianCoeffs(dm, rAUfInt, P.diag, P.upper, P.lower, /*nonOrth=*/false);

    // == fvc::div(phiHbyA): source += div*V, a PLUS.
    DeviceBuffer<scalar> div(static_cast<std::size_t>(nC));
    deviceDiv(dm, phiHbyAInt, phiHbyABnd, div);
    P.source.resize(static_cast<std::size_t>(nC));
    cudaMemset(P.source.data(), 0, sizeof(scalar)*nC);
    addDivSourceKernel<<<nBlocks(nC), TPB>>>(div.data(), dm.V.data(), nC, P.source.data());
    ckP(cudaGetLastError(), "source += div(phiHbyA)*V");

    if (needReference)
    {
        if (pRefCell < 0 || pRefCell >= nC)
            throw std::runtime_error(
                "brae interFoam device pEqn: pRefCell is outside the mesh. A case whose p_rgh has no "
                "value-fixing patch is singular without it, so this cannot be defaulted away.");
        setReferenceKernel<<<1, 1>>>(pRefCell, pRefValue, P.diag.data(), P.source.data());
        ckP(cudaGetLastError(), "setReference");
    }
}

} // namespace brae
