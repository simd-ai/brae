// interFoam's pressure corrector on the device -- see device_inter_peqn.cuh for the four.
#include "device_inter_peqn.cuh"
#include "device_fvc_reconstruct.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_simple.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// OpenFOAM's SMALL IN A DOUBLE BUILD (doubleScalar.H:62), the guard in fvcDdtPhiCoeff's denominator.
// This was 1e-37, the FLOAT build's VSMALL, as the host's ddtCorr was; the two differ only where
// |phi| is at or below 1e-15.
__device__ constexpr scalar kSmall = scalar(1.0e-15);

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

// ...and on a PERIODIC PAIR, which is the internal kernel with the pair's own arrays: OpenFOAM's
// fvc::ddtCorr is a whole surfaceScalarField and a coupled patch is live in it, because ddtScheme.C's
// exclusions are `U fixes a value` and cyclicAMI, and a plain cyclic is neither
// (inter_peqn_cpp.cu:252-283). fvc::dotInterpolate(Sf, U.oldTime()) there is the two CELLS' old
// velocities interpolated, never a stored patch value -- which is why this is the internal form.
__global__ void ddtCorrCyclicKernel(
    const label* __restrict__ own, const label* __restrict__ nbr, const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ phiOld,
    const scalar* __restrict__ uox, const scalar* __restrict__ uoy, const scalar* __restrict__ uoz,
    int n, scalar given, scalar rDeltaT, scalar* __restrict__ out)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int o = own[j], nb = nbr[j];
    const scalar wf = w[j], wm = scalar(1) - w[j];
    const scalar ux = wf*uox[o] + wm*uox[nb];
    const scalar uy = wf*uoy[o] + wm*uoy[nb];
    const scalar uz = wf*uoz[o] + wm*uoz[nb];
    const scalar phiCorr = phiOld[j] - (ux*Sfx[j] + uy*Sfy[j] + uz*Sfz[j]);
    out[j] = ddtCoeff(phiCorr, phiOld[j], given) * rDeltaT * phiCorr;
}

__global__ void ddtCorrBoundaryKernel(
    const label* __restrict__ bndCell, const label* __restrict__ bndGFace,
    const int* __restrict__ fixesValue,
    const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
    const scalar* __restrict__ phiOldBnd,
    const scalar* __restrict__ uox, const scalar* __restrict__ uoy, const scalar* __restrict__ uoz,
    const scalar* __restrict__ uobx, const scalar* __restrict__ uoby, const scalar* __restrict__ uobz,
    int nBf, scalar given, scalar rDeltaT, scalar* __restrict__ out)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    // ZERO wherever U fixes a value: the flux there is what the boundary condition says, and there is
    // no inconsistency to correct (ddtScheme.C:~275).
    if (fixesValue[b]) { out[b] = scalar(0); return; }
    const int c = bndCell[b], gf = bndGFace[b];
    // fvc::dotInterpolate(Sf, U.oldTime()) on an UNCOUPLED patch is `pSf & vf.boundaryField()[pi]` --
    // the STORED patch value (surfaceInterpolationScheme.C:296-298), not patchInternalField. This read
    // the face cell, which is the same number only where the patch value follows the cell. On a SLIP
    // wall it does not: MEASURED on RAS/angledDuct, whose `porosityWall` is one, the boundary half of
    // ddtCorr put the `inactive` arm 1.4e-11 from OpenFOAM where it is 7.6e-15 with the patch value.
    // The host arm reads it the same way (inter_peqn_cpp.cu:263-270).
    const scalar uX = uobx ? uobx[b] : uox[c];
    const scalar uY = uoby ? uoby[b] : uoy[c];
    const scalar uZ = uobz ? uobz[b] : uoz[c];
    const scalar interpFlux = uX*Sfx[gf] + uY*Sfy[gf] + uZ*Sfz[gf];
    const scalar pOld    = phiOldBnd[b];
    const scalar phiCorr = pOld - interpFlux;
    out[b] = ddtCoeff(phiCorr, pOld, given) * rDeltaT * phiCorr;
}

// pe.source += fvc::div(phiHbyA)*V -- a PLUS, fvMatrix::operator== (fvMatrix.C:1855-1862).
__global__ void interpolateFullKernel(
    const label* __restrict__ own, const label* __restrict__ nei, const scalar* __restrict__ w,
    const label* __restrict__ bndCell, const label* __restrict__ bndGFace,
    const scalar* __restrict__ vf, int nIf, int nBf, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nIf)
    {
        out[i] = w[i]*vf[own[i]] + (scalar(1) - w[i])*vf[nei[i]];
    }
    else if (i < nIf + nBf)
    {
        const int b = i - nIf;
        // at an uncoupled patch there is no second cell to weight against, so the face value IS the
        // cell's. bndGFace puts it where the mesh's face order expects it.
        out[bndGFace[b]] = vf[bndCell[b]];
    }
}

__global__ void gatherBoundaryKernel(const label* __restrict__ bndCell, const scalar* __restrict__ vf,
                                     int nBf, scalar* __restrict__ out)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nBf) out[b] = vf[bndCell[b]];
}

__global__ void addPhiHbyATermsKernel(
    const scalar* __restrict__ rhoRAUf, const scalar* __restrict__ ddtCorr,
    const scalar* __restrict__ phig, int n, int haveDdt, scalar* __restrict__ phiHbyA)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    if (haveDdt) phiHbyA[f] += rhoRAUf[f]*ddtCorr[f];
    phiHbyA[f] += phig[f];
}

// The same two adds, SEPARATED, because MRF.makeRelative(phiHbyA) runs between them: pEqn.H:14-36 builds
// phiHbyA from flux(HbyA) and the ddtCorr term, makes it relative, and only then adds phig. The
// arithmetic is unchanged -- (phiHbyA + rhoRAUf*ddtCorr) + phig in that order either way.
__global__ void addDdtCorrTermKernel(
    const scalar* __restrict__ rhoRAUf, const scalar* __restrict__ ddtCorr,
    int n, scalar* __restrict__ phiHbyA)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) phiHbyA[f] += rhoRAUf[f]*ddtCorr[f];
}

__global__ void addPhigInternalKernel(const scalar* __restrict__ phig, int n,
                                      scalar* __restrict__ phiHbyA)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) phiHbyA[f] += phig[f];
}

__global__ void addPhigBoundaryKernel(const scalar* __restrict__ phig, int n,
                                      scalar* __restrict__ phiHbyA)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < n) phiHbyA[b] += phig[b];
}

// ddtCorr's BOUNDARY half, the host's own loop (inter_peqn_cpp.cu:531-573) for an uncoupled patch:
//     phiHbyA_b += rho_b * rAU[faceCell] * corr_b
// with interpolate(rho*rAU) on the patch being rho's own boundary value and rAU's extrapolated one, which
// is 1/A of the face cell (fvMatrix::A() is extrapolatedCalculated). corr_b is already ZERO wherever U
// fixes a value, because fvcDdtPhiCoeff zeroed the coupling coefficient there (deviceDdtCorr takes that
// mask), so no patch test is needed here -- the arithmetic is the test.
__global__ void addDdtCorrBoundaryKernel(
    const scalar* __restrict__ rhoBnd,
    const scalar* __restrict__ rAU,
    const label*  __restrict__ faceCell,
    const scalar* __restrict__ corrBnd,
    const int*    __restrict__ uFixesValue,
    int n,
    scalar* __restrict__ phiHbyA)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    // The host's own test, kept EXPLICIT (inter_peqn_cpp.cu:533): a patch whose U fixes a value is
    // skipped outright. Relying on the correction being zero there instead was not the same thing --
    // MEASURED on RAS/angledDuct, where it moved the `inactive` arm from 7.6e-15 to 1.4e-11.
    if (uFixesValue[b]) return;
    phiHbyA[b] += rhoBnd[b] * rAU[faceCell[b]] * corrBnd[b];
}

// fvMatrix::flux() at a boundary face: internalCoeffs*p[faceCell] - boundaryCoeffs.
__global__ void pFluxBoundaryKernel(
    const label* __restrict__ bndCell, const scalar* __restrict__ iC, const scalar* __restrict__ bC,
    const scalar* __restrict__ p, int nBf, scalar* __restrict__ flux)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nBf) flux[b] = iC[b]*p[bndCell[b]] - bC[b];
}

// source += s, where s is deviceFaceDivSource's output -- the host's laplacianNonOrthSource negated, so
// adding it is the host's `source -= corr`
__global__ void addKernel(const scalar* __restrict__ s, int nC, scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) source[c] += s[c];
}

__global__ void addDivSourceKernel(const scalar* __restrict__ div, const scalar* __restrict__ V,
                                   int nC, scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) source[c] += div[c]*V[c];
}

// fvMatrix::setReference: source += diag*refValue, then diag += diag. It DOUBLES the entry; replacing
// the row instead gives a different matrix that still solves.
__global__ void setReferenceKernel(int cell, const scalar* __restrict__ pRgh,
                                   scalar* __restrict__ diag, scalar* __restrict__ source)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // getRefCellValue(p_rgh, pRefCell), pEqn.H:47: the cell is pinned at ITS CURRENT p_rgh, not at
        // pRefValue -- pRefValue is p's level and is applied after the solve (deviceInterPressureReference).
        // The host arm reads it the same way (inter_peqn_cpp.cu:729-737).
        source[cell] += diag[cell]*pRgh[cell];
        diag[cell]   += diag[cell];
    }
}

// applyPressureReference (inter_peqn_cpp.cu:179-196), pEqn.H:74-83: p += pRefValue - p[pRefCell], and
// p_rgh is REBUILT from the shifted p rather than keeping what the solve gave it. Two passes, because
// every cell reads p[pRefCell] and the reference cell's own thread writes it.
__global__ void refShiftKernel(int cell, scalar pRefValue,
                              const scalar* __restrict__ p, scalar* __restrict__ shift)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) shift[0] = pRefValue - p[cell];
}

__global__ void applyRefShiftKernel(int nC, const scalar* __restrict__ shift,
                                    const scalar* __restrict__ rho, const scalar* __restrict__ gh,
                                    scalar* __restrict__ p, scalar* __restrict__ p_rgh)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    p[c] += shift[0];
    p_rgh[c] = p[c] - rho[c]*gh[c];
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


void deviceInterDdtCorrCyclic(
    const DeviceCyclic&         cyc,
    const DeviceBuffer<scalar>& phiOldIf,
    const DeviceBuffer<scalar>& UOldX,
    const DeviceBuffer<scalar>& UOldY,
    const DeviceBuffer<scalar>& UOldZ,
    scalar                      ddtPhiCoeff,
    scalar                      deltaT,
    DeviceBuffer<scalar>&       outIf)
{
    if (deltaT <= scalar(0))
        throw std::runtime_error("brae interFoam device ddtCorr: deltaT must be positive.");
    outIf.resize(static_cast<std::size_t>(cyc.n));
    if (cyc.n == 0) return;
    if (static_cast<int>(phiOldIf.size()) != cyc.n)
        throw std::runtime_error(
            "brae interFoam device ddtCorr: the pair's OLD flux is the wrong length. ddtCorr compares "
            "phi.oldTime() with the flux of U.oldTime(), so the caller must snapshot the pair's flux "
            "at the top of the step, before the pressure corrector rewrites it.");
    ddtCorrCyclicKernel<<<nBlocks(cyc.n), TPB>>>(
        cyc.ownCell.data(), cyc.nbrCell.data(), cyc.weights.data(),
        cyc.Sfx.data(), cyc.Sfy.data(), cyc.Sfz.data(), phiOldIf.data(),
        UOldX.data(), UOldY.data(), UOldZ.data(), cyc.n, ddtPhiCoeff,
        scalar(1)/deltaT, outIf.data());
    ckP(cudaGetLastError(), "ddtCorr, interface");
}


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
    DeviceBuffer<scalar>&       outBnd,
    const DeviceBuffer<scalar>* UOldBndX,
    const DeviceBuffer<scalar>* UOldBndY,
    const DeviceBuffer<scalar>* UOldBndZ,
    const DeviceBuffer<scalar>* phiUfOldInt)
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
            dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
            // ON A MOVING MESH (Sf & Uf.oldTime()) takes phi.oldTime()'s place, in phiCorr AND in the
            // limiter's denominator -- the host reference's own ternary (inter_peqn_cpp.cu:237), and
            // OpenFOAM's fvcDdtUfCorr. The kernel is unchanged; only which array it is handed is.
            (phiUfOldInt && static_cast<int>(phiUfOldInt->size()) == nIf) ? phiUfOldInt->data()
                                                                         : phiOldInt.data(),
            UOldX.data(), UOldY.data(), UOldZ.data(), nIf, ddtPhiCoeff, rDeltaT, outInt.data());
        ckP(cudaGetLastError(), "ddtCorr, internal");
    }
    outBnd.resize(static_cast<std::size_t>(nBf));
    if (nBf > 0)
    {
        ddtCorrBoundaryKernel<<<nBlocks(nBf), TPB>>>(
            dm.bndCell.data(), dm.bndGFace.data(), bndUFixesValue.data(),
            dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), phiOldBnd.data(),
            UOldX.data(), UOldY.data(), UOldZ.data(),
            UOldBndX ? UOldBndX->data() : nullptr,
            UOldBndY ? UOldBndY->data() : nullptr,
            UOldBndZ ? UOldBndZ->data() : nullptr, nBf, ddtPhiCoeff, rDeltaT, outBnd.data());
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
    DeviceBuffer<scalar>&       UZ,
    const DeviceCyclic*         cyc,
    const DeviceBuffer<scalar>* faceFluxIf,
    const DeviceBuffer<scalar>* rAUfIf)
{
    const int nC = dm.nCells, nIf = dm.nInternalFaces, nBf = dm.nBndFaces;
    const bool havePair = cyc && cyc->n > 0;
    if (havePair && (!faceFluxIf || !rAUfIf))
    {
        throw std::runtime_error(
            "brae interFoam device pEqn: the mesh has a periodic pair and its (phig - flux) or its rAUf "
            "was not handed to the velocity correction. fvc::reconstruct sums a coupled face like any "
            "other patch face, so leaving it out gives a different U in every cell the pair touches.");
    }

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

    // ...and the pair's, the same divide on its own faces
    DeviceBuffer<scalar> ssfIf;
    if (havePair)
    {
        ssfIf.resize(static_cast<std::size_t>(cyc->n));
        divideKernel<<<nBlocks(cyc->n), TPB>>>(faceFluxIf->data(), rAUfIf->data(), cyc->n, ssfIf.data());
        ckP(cudaGetLastError(), "faceFlux/rAUf, interface");
    }

    DeviceBuffer<scalar> rx, ry, rz;
    deviceReconstruct(dm, ssfInt, ssfBnd, rx, ry, rz, havePair ? cyc : nullptr,
                      havePair ? &ssfIf : nullptr);

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
    const DeviceBuffer<scalar>* pRghForRef,
    DevicePressureMatrix&       P,
    bool                        corrected,
    const DeviceBuffer<scalar>* nonOrthSource,
    DeviceCyclic*               cyc,
    const DeviceBuffer<scalar>* rAUCell,
    const DeviceBuffer<scalar>* phiHbyAIf,
    DeviceBuffer<scalar>*       divTapOut)
{
    const int nC = dm.nCells;

    // fvm::laplacian(rAUf, p_rgh) under the case's laplacianSchemes, as the host's pressureCorrector
    // (inter_peqn_cpp.cu) assembles it: nonOrthDeltaCoeffs on the internal faces when `corrected`.
    // (This said pEqn passes corrected=false; the host takes the case's scheme, and so does this.)
    deviceLaplacianCoeffs(dm, rAUfInt, P.diag, P.upper, P.lower, corrected);

    // the source, in the host's order: the laplacian's own (zero), then `source -= corr` -- the
    // explicit non-orthogonal correction -- then `source += div*V`
    P.source.resize(static_cast<std::size_t>(nC));
    cudaMemset(P.source.data(), 0, sizeof(scalar)*nC);
    if (nonOrthSource)
    {
        if (!corrected)
            throw std::runtime_error(
                "brae interFoam device pEqn: a non-orthogonal correction was handed to an orthogonal assembly.");
        addKernel<<<nBlocks(nC), TPB>>>(nonOrthSource->data(), nC, P.source.data());
        ckP(cudaGetLastError(), "source -= laplacianNonOrthSource");
    }
    // == fvc::div(phiHbyA): source += div*V, a PLUS.
    DeviceBuffer<scalar> div(static_cast<std::size_t>(nC));
    deviceDiv(dm, phiHbyAInt, phiHbyABnd, div);
    if (divTapOut)
    {
        divTapOut->resize(static_cast<std::size_t>(nC));
        ckP(cudaMemcpy(divTapOut->data(), div.data(), sizeof(scalar)*nC, cudaMemcpyDeviceToDevice),
            "div tap");
    }
    // ...and the PAIR's phiHbyA, which fvc::div sums into its face cell like any patch's
    // (fvc.cu:548-550). The matrix's own coupling was already here; leaving the SOURCE without the
    // pair's flux is a different equation, not a smaller one -- MEASURED on validation/interFoamCyclic,
    // the first pressure solve then put the Courant number at 2.9 and alpha at 1.53 by step two.
    if (cyc && cyc->n > 0)
    {
        if (!phiHbyAIf)
        {
            throw std::runtime_error(
                "brae interFoam device pEqn: the mesh has a periodic pair and its phiHbyA was not "
                "handed to the assembly. fvc::div(phiHbyA) sums a coupled face like any other.");
        }
        deviceCyclicAddDivFlux(*cyc, *phiHbyAIf, dm.V, div);
    }
    addDivSourceKernel<<<nBlocks(nC), TPB>>>(div.data(), dm.V.data(), nC, P.source.data());
    ckP(cudaGetLastError(), "source += div(phiHbyA)*V");

    // THE COUPLED FACES' laplacian coefficients. rAU is a CELL field here and the interface kernel
    // interpolates it with the pair's own weight, which is what interpolate(rAU) gives on a coupled
    // patch (fvc.cu:472, coupledLinear) -- the same number the host's gammaf.boundary carries. Gated
    // face by face in tests/test_device_cyclic_laplacian_vs_host.cu.
    if (cyc && cyc->n > 0)
    {
        if (!rAUCell)
        {
            throw std::runtime_error(
                "brae interFoam device pEqn: the mesh has a periodic pair and the laplacian's gamma was "
                "not handed over as a cell field. interpolate(rAU) on a coupled patch is the two cells' "
                "rAU interpolated; there is nothing on the patch to read instead.");
        }
        deviceCyclicAssembleLaplacian(*cyc, *rAUCell, P.diag, /*addToDiag=*/true, corrected);
    }

    if (needReference)
    {
        if (pRefCell < 0 || pRefCell >= nC)
            throw std::runtime_error(
                "brae interFoam device pEqn: pRefCell is outside the mesh. A case whose p_rgh has no "
                "value-fixing patch is singular without it, so this cannot be defaulted away.");
        if (!pRghForRef)
            throw std::runtime_error(
                "brae interFoam device pEqn: setReference pins the reference cell at p_rgh's CURRENT value "
                "there (pEqn.H:47), and the field was not handed over.");
        setReferenceKernel<<<1, 1>>>(pRefCell, pRghForRef->data(), P.diag.data(), P.source.data());
        ckP(cudaGetLastError(), "setReference");
    }
}


void deviceInterPressureReference(
    int                         nC,
    int                         pRefCell,
    scalar                      pRefValue,
    const DeviceBuffer<scalar>& rho,
    const DeviceBuffer<scalar>& gh,
    DeviceBuffer<scalar>&       p,
    DeviceBuffer<scalar>&       p_rgh)
{
    if (nC <= 0) return;
    if (pRefCell < 0 || pRefCell >= nC)
        throw std::runtime_error(
            "brae interFoam device pEqn: pRefCell is outside the mesh, and pEqn.H:74-83 reads p there.");
    DeviceBuffer<scalar> shift(1);
    refShiftKernel<<<1, 1>>>(pRefCell, pRefValue, p.data(), shift.data());
    ckP(cudaGetLastError(), "the reference shift");
    applyRefShiftKernel<<<nBlocks(nC), TPB>>>(nC, shift.data(), rho.data(), gh.data(),
                                              p.data(), p_rgh.data());
    ckP(cudaGetLastError(), "p += shift; p_rgh = p - rho*gh");
}


void deviceInterAddPhiHbyATerms(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& rhoRAUfInt,
    const DeviceBuffer<scalar>& ddtCorrInt,
    const DeviceBuffer<scalar>& phigInt,
    const DeviceBuffer<scalar>& phigBnd,
    bool                        haveDdtCorr,
    DeviceBuffer<scalar>&       phiHbyAInt,
    DeviceBuffer<scalar>&       phiHbyABnd,
    const std::vector<DeviceMRFZone>* mrf,
    const DeviceBuffer<scalar>* ddtCorrBnd,
    const DeviceBuffer<scalar>* rhoBnd,
    const DeviceBuffer<scalar>* rAU,
    const DeviceBuffer<int>*    uFixesValue)
{
    const int nIf = dm.nInternalFaces, nBf = dm.nBndFaces;
    if (static_cast<int>(phiHbyAInt.size()) != nIf || static_cast<int>(phiHbyABnd.size()) != nBf)
        throw std::runtime_error(
            "brae interFoam device pEqn: phiHbyA must already hold fvc::flux(HbyA) on BOTH sides. The "
            "boundary half is not decoration -- fvc::div(phiHbyA) sums it, so it is how a wall's "
            "buoyancy and surface tension reach the pressure equation's source.");
    // MRF.makeRelative(phiHbyA) sits BETWEEN the two adds (pEqn.H:19, before phig at :36), so with a zone
    // the ddtCorr term goes in on its own first. Without one the fused kernel stands, unchanged.
    if (mrf && !mrf->empty())
    {
        if (nIf > 0 && haveDdtCorr)
        {
            addDdtCorrTermKernel<<<nBlocks(nIf), TPB>>>(
                rhoRAUfInt.data(), ddtCorrInt.data(), nIf, phiHbyAInt.data());
            ckP(cudaGetLastError(), "phiHbyA += rhoRAUf*ddtCorr");
        }
        deviceMrfMakeRelative(*mrf, phiHbyAInt, phiHbyABnd);
        if (nIf > 0)
        {
            addPhigInternalKernel<<<nBlocks(nIf), TPB>>>(phigInt.data(), nIf, phiHbyAInt.data());
            ckP(cudaGetLastError(), "phiHbyA += phig");
        }
    }
    else if (nIf > 0)
    {
        addPhiHbyATermsKernel<<<nBlocks(nIf), TPB>>>(
            rhoRAUfInt.data(), haveDdtCorr ? ddtCorrInt.data() : nullptr, phigInt.data(),
            nIf, haveDdtCorr ? 1 : 0, phiHbyAInt.data());
        ckP(cudaGetLastError(), "phiHbyA += rhoRAUf*ddtCorr + phig");
    }
    if (nBf > 0)
    {
        // the ddtCorr term goes in BEFORE phig on the boundary too, in the host's order
        if (ddtCorrBnd && rhoBnd && rAU && uFixesValue)
        {
            addDdtCorrBoundaryKernel<<<nBlocks(nBf), TPB>>>(
                rhoBnd->data(), rAU->data(), dm.bndCell.data(), ddtCorrBnd->data(),
                uFixesValue->data(), nBf, phiHbyABnd.data());
            ckP(cudaGetLastError(), "phiHbyA += rho_b*rAU*ddtCorr, boundary");
        }
        addPhigBoundaryKernel<<<nBlocks(nBf), TPB>>>(phigBnd.data(), nBf, phiHbyABnd.data());
        ckP(cudaGetLastError(), "phiHbyA += phig, boundary");
    }
}


void deviceInterPEqnFlux(
    const DeviceMesh&           dm,
    const DevicePressureMatrix& P,
    const DeviceBuffer<scalar>& iC,
    const DeviceBuffer<scalar>& bC,
    const DeviceBuffer<scalar>& pSolved,
    DeviceBuffer<scalar>&       fluxInt,
    DeviceBuffer<scalar>&       fluxBnd,
    const DeviceBuffer<scalar>* faceFluxCorrection)
{
    const int nBf = dm.nBndFaces;
    DeviceLduView A{};
    A.nCells = dm.nCells;
    A.nInternalFaces = dm.nInternalFaces;
    A.diag = P.diag.data();
    A.upper = P.upper.data();
    A.lower = P.lower.data();
    A.owner = dm.owner.data();
    A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data();
    A.losortStart = dm.losortStart.data();
    A.addressingId = dm.addressingId;
    deviceMatrixFluxInternal(A, pSolved, fluxInt);
    // fvMatrix::flux(): `fieldFlux += *faceFluxCorrectionPtr_` (fvMatrix.C:1688, matrixFlux on the host)
    if (faceFluxCorrection && dm.nInternalFaces > 0)
    {
        addKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(faceFluxCorrection->data(), dm.nInternalFaces,
                                                       fluxInt.data());
        ckP(cudaGetLastError(), "pEqn.flux() += faceFluxCorrection");
    }

    fluxBnd.resize(static_cast<std::size_t>(nBf));
    if (nBf > 0)
    {
        pFluxBoundaryKernel<<<nBlocks(nBf), TPB>>>(dm.bndCell.data(), iC.data(), bC.data(),
                                                   pSolved.data(), nBf, fluxBnd.data());
        ckP(cudaGetLastError(), "pEqn.flux(), boundary");
    }
}


void deviceInterpolateFull(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& vf,
    DeviceBuffer<scalar>&       out)
{
    const int nIf = dm.nInternalFaces, nBf = dm.nBndFaces;
    out.resize(static_cast<std::size_t>(nIf + nBf));
    if (nIf + nBf == 0) return;
    interpolateFullKernel<<<nBlocks(nIf + nBf), TPB>>>(
        dm.owner.data(), dm.nei.data(), dm.w.data(), dm.bndCell.data(), dm.bndGFace.data(),
        vf.data(), nIf, nBf, out.data());
    ckP(cudaGetLastError(), "fvc::interpolate, full face array");
}


void deviceGatherBoundary(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& vf,
    DeviceBuffer<scalar>&       out)
{
    const int nBf = dm.nBndFaces;
    out.resize(static_cast<std::size_t>(nBf));
    if (nBf == 0) return;
    gatherBoundaryKernel<<<nBlocks(nBf), TPB>>>(dm.bndCell.data(), vf.data(), nBf, out.data());
    ckP(cudaGetLastError(), "gather to the boundary");
}

} // namespace brae
