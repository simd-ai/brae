// CUDA implementation -- see rhoPcEqn.cuh for the provenance, the six differences from pEqn.H and the
// branch ordering. Every term is transcribed from rhoPcEqn_cpp.cu in the order that file produces it.
#include "rhoPcEqn.cuh"
#include "device_blas.cuh"
#include "device_simple.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

__global__
void constrainHbyAKernel(
    int                        nB,
    const label*  __restrict__ faceCell,
    const label*  __restrict__ takeU,
    const scalar* __restrict__ Ub,
    const scalar* __restrict__ HbyACell,
    scalar*       __restrict__ HbyAb)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    HbyAb[i] = takeU[i] ? Ub[i] : HbyACell[faceCell[i]];
}

__global__
void foldBoundaryDiagKernel(
    int                        nC,
    const label*  __restrict__ bndCellStart,
    const label*  __restrict__ bndPerm,
    const scalar* __restrict__ icAv,
    scalar*       __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = 0.0;
    for (label k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k) s += icAv[bndPerm[k]];
    D[c] += s;
}

// gammaBnd[i] = rhoBnd[i] * cellField[faceCell[i]] -- rho's PATCH value times a calculated volume field's
// OWNER-CELL value. rhoPcEqn_cpp.cu builds rhorAtUb exactly this way, and it is why the laplacian's
// boundary coefficients need the FACE variant rather than the cell one.
__global__
void mixedBndKernel(
    int                        nB,
    const label*  __restrict__ faceCell,
    const scalar* __restrict__ rhoBnd,
    const scalar* __restrict__ cellField,
    scalar*       __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    out[i] = rhoBnd[i] * cellField[faceCell[i]];
}

__global__
void phidKernel(
    int                        n,
    const scalar* __restrict__ psif,
    const scalar* __restrict__ rhof,
    const scalar* __restrict__ phiHbyA,
    scalar*       __restrict__ phid)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    phid[f] = (psif[f] / rhof[f]) * phiHbyA[f];
}

// TRANSONIC, in ONE statement as pcEqn.H writes it:
//     phiHbyA = phiHbyA0 + simplecCorr - interp(psi*p)*phiHbyA0/interp(rho)
// The subtraction is taken against the UNCORRECTED phiHbyA0, not against the corrected value, so there is
// no intermediate state where only one of the two has been applied. Splitting this into two passes would
// subtract from a flux that already carried the correction.
__global__
void transonicPhiHbyAKernel(
    int                        n,
    const scalar* __restrict__ phi0,
    const scalar* __restrict__ corr,
    const scalar* __restrict__ psipf,
    const scalar* __restrict__ rhof,
    scalar*       __restrict__ out)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    out[f] = phi0[f] + corr[f] - psipf[f] * phi0[f] / rhof[f];
}

void zeroed(DeviceBuffer<scalar>& b, int n)
{
    b.resize(n);
    if (n) cudaCheck(cudaMemsetAsync(b.data(), 0, sizeof(scalar) * n, cudaStreamPerThread),
                     "rhoPcEqn zero");
}

__global__
void negateKernel(int n, scalar* __restrict__ a)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = -a[i];
}

void negate(DeviceBuffer<scalar>& a)
{
    if (a.size()) negateKernel<<<nBlocks((int)a.size()), TPB>>>((int)a.size(), a.data());
}

void refuseUnsupported(const RhoPressureInput& in)
{
    if (in.hasMRF)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn(cuda): the case declares MRF, and pcEqn.H calls MRF.makeRelative on "
            "phiHbyA before adjustPhi. Refusing rather than balancing a flux that still carries the "
            "frame motion.");
    }
    if (in.hasFvOptions)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn(cuda): the case declares an fvOption this path does not implement"
            + (in.fvOptionUnsupported.empty() ? std::string()
                                              : std::string(" (") + in.fvOptionUnsupported + ")")
            + ". pcEqn.H applies fvOptions(psi, p, rho.name()).");
    }
    if (in.hasFixedFluxPressure)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn(cuda): the case uses fixedFluxPressure, whose boundary gradient is set "
            "by constrainPressure -- not ported. Refusing rather than solving with a zeroGradient.");
    }
    if (in.hasCoupledPatches)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn(cuda): the mesh has cyclic/AMI/processor patches, which buildDeviceMesh "
            "keeps out of the LDU -- they would contribute nothing, silently.");
    }
    if (!in.rhoCell || !in.rhoBndFace || !in.psiCell || !in.psiBndFace)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn(cuda): rho and psi are required on cells AND boundary faces.");
    }
}

} // namespace


void consistentPressurePredictor(
    ConsistentPressureStages&   st,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary&       dbP,
    const MomentumMatrix&       UEqn,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& p,
    const RhoPressureInput&     in)
{
    refuseUnsupported(in);
    if (!in.takeUAtBoundary)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn(cuda): constrainHbyA needs the per-face `assignable` mask -- and "
            "assignable() is not fixesValue(), so it cannot be recovered from bcType here.");
    }
    st.transonic = in.transonic;
    const int nC = dm.nCells;

    DeviceBuffer<scalar> icAv;
    deviceCopy(icAv, UEqn.iC[0]);
    deviceAxpy(1.0, UEqn.iC[1], icAv);
    deviceAxpy(1.0, UEqn.iC[2], icAv);
    deviceScale(icAv, 1.0 / 3.0);

    // ---- rAU = 1/UEqn.A(), then rAtU = 1/(1/rAU - UEqn.H1()) --------------------------------
    // D is kept: H1's row sum needs the SAME folded diagonal, and recomputing it would be a second
    // chance to fold differently.
    DeviceBuffer<scalar> D;
    {
        deviceCopy(D, UEqn.relaxed ? UEqn.relaxedDiag : UEqn.diag);
        foldBoundaryDiagKernel<<<nBlocks(nC), TPB>>>(
            nC, dm.bndCellStart.data(), dm.bndPerm.data(), icAv.data(), D.data());
        cudaCheck(cudaGetLastError(), "rhoPcEqn foldBoundaryDiag");
        deviceReciprocalV(dm, D, st.rAU);
    }
    {
        // H1 is the row sum of the off-diagonals, obtained as A*1 -- the matrix applied to a vector of
        // ones -- which is how the incompressible twin gets it and is gated there.
        DeviceBuffer<scalar> ones, rowSum;
        ones.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nC), scalar(1)));
        deviceAmul(deviceLduView(dm, D, UEqn.upper, UEqn.lower), ones, rowSum);
        deviceSimplecRAtU(dm, rowSum, D, st.rAtU);
    }

    // rhorAtU: a VOLUME field, interpolated here rather than handed to fvm::laplacian as one, plus the
    // mixed boundary form -- see the header.
    deviceHadamard(st.rhorAtU, *in.rhoCell, st.rAtU);
    deviceInterpolate(dm, st.rhorAtU, st.rhorAtUf);
    st.rhorAtUfBnd.resize(dm.nBndFaces);
    if (dm.nBndFaces > 0)
    {
        mixedBndKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
            dm.nBndFaces, dm.bndCell.data(), in.rhoBndFace->data(), st.rAtU.data(),
            st.rhorAtUfBnd.data());
        cudaCheck(cudaGetLastError(), "rhoPcEqn rhorAtUBnd");
    }

    // ---- HbyA0 = rAU*UEqn.H(), then constrainHbyA -------------------------------------------
    const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
    const DeviceLduView A = UEqn.view(dm);
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> bdDiagK;
        deviceCopy(bdDiagK, icAv);
        deviceAxpy(-1.0, UEqn.iC[k], bdDiagK);
        DeviceBuffer<scalar> Hk;
        deviceMatrixH(A, dm, *U[k], UEqn.source[k], bdDiagK, UEqn.bC[k], Hk);
        deviceHadamard(st.HbyA0[k], st.rAU, Hk);
    }
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> Ub;
        deviceBCValue(dbU.comp[k], *U[k], Ub);
        st.HbyAb[k].resize(dm.nBndFaces);
        if (dm.nBndFaces > 0)
        {
            constrainHbyAKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
                dm.nBndFaces, dm.bndCell.data(), in.takeUAtBoundary->data(),
                Ub.data(), st.HbyA0[k].data(), st.HbyAb[k].data());
        }
    }
    cudaCheck(cudaGetLastError(), "rhoPcEqn constrainHbyA");

    // ---- phiHbyA0 = interpolate(rho)*flux(HbyA0) --------------------------------------------
    DeviceBuffer<scalar> rhofInt;
    deviceInterpolate(dm, *in.rhoCell, rhofInt);
    {
        DeviceBuffer<scalar> fluxInt, fluxBnd;
        deviceVectorFlux(dm, st.HbyA0[0], st.HbyA0[1], st.HbyA0[2], fluxInt);
        deviceBoundaryFlux(dm, st.HbyAb[0], st.HbyAb[1], st.HbyAb[2], fluxBnd);
        deviceHadamard(st.phiHbyA0Int, rhofInt, fluxInt);
        deviceHadamard(st.phiHbyA0Bnd, *in.rhoBndFace, fluxBnd);
    }
    deviceCopy(st.phiHbyAInt, st.phiHbyA0Int);
    deviceCopy(st.phiHbyABnd, st.phiHbyA0Bnd);

    // ---- the SIMPLEC flux correction, interpolate(rho*(rAtU - rAU))*snGrad(p)*magSf ---------
    // Built as the FLUX of a laplacian with that gamma, because gamma_f*snGrad(p)*magSf IS that flux.
    // One code path rather than a second snGrad, and it is the shape the incompressible twin uses.
    DeviceBuffer<scalar> corrInt, corrBnd, pb, gx, gy, gz;
    {
        DeviceBuffer<scalar> drAtU, rhodrAtU, gammaf, gammaBnd;
        deviceCopy(drAtU, st.rAtU);
        deviceAxpy(-1.0, st.rAU, drAtU);                 // rAtU - rAU
        deviceHadamard(rhodrAtU, *in.rhoCell, drAtU);    // rho*(rAtU - rAU)
        deviceInterpolate(dm, rhodrAtU, gammaf);

        deviceBCValue(dbP, p, pb);
        deviceGaussGrad(dm, p, pb, gx, gy, gz);

        DeviceBuffer<scalar> ld, lu, ll;
        deviceLaplacianCoeffs(dm, gammaf, ld, lu, ll, in.correctedLaplacian);
        deviceMatrixFluxInternal(deviceLduView(dm, ld, lu, ll), p, corrInt);
        if (in.correctedLaplacian)
        {
            DeviceBuffer<scalar> ffc;
            deviceLaplacianCorrFlux(dm, gammaf, gx, gy, gz, ffc);
            deviceAxpy(1.0, ffc, corrInt);
        }
        gammaBnd.resize(dm.nBndFaces);
        if (dm.nBndFaces > 0)
        {
            mixedBndKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
                dm.nBndFaces, dm.bndCell.data(), in.rhoBndFace->data(), drAtU.data(),
                gammaBnd.data());
        }
        DeviceBuffer<scalar> dIC, dBC;
        deviceBCLaplacianCoeffsFace(dbP, gammaBnd, dIC, dBC);
        deviceMatrixFluxBoundary(dbP, dIC, dBC, p, corrBnd);
        cudaCheck(cudaGetLastError(), "rhoPcEqn simplec correction");
    }

    st.closedVolume = false;
    st.massCorr = 1.0;

    if (in.transonic)
    {
        // phid from the UNCORRECTED phiHbyA, FIRST.
        DeviceBuffer<scalar> psifInt;
        deviceInterpolate(dm, *in.psiCell, psifInt);
        st.phidInt.resize(st.phiHbyA0Int.size());
        if (st.phiHbyA0Int.size())
        {
            phidKernel<<<nBlocks((int)st.phiHbyA0Int.size()), TPB>>>(
                (int)st.phiHbyA0Int.size(), psifInt.data(), rhofInt.data(),
                st.phiHbyA0Int.data(), st.phidInt.data());
        }
        st.phidBnd.resize(dm.nBndFaces);
        if (dm.nBndFaces > 0)
        {
            phidKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
                dm.nBndFaces, in.psiBndFace->data(), in.rhoBndFace->data(),
                st.phiHbyA0Bnd.data(), st.phidBnd.data());
        }

        // ...then the correction and the psi*p subtraction TOGETHER, against phiHbyA0.
        DeviceBuffer<scalar> psip, psipfInt, psipBnd, pBndVal;
        deviceHadamard(psip, *in.psiCell, p);
        deviceInterpolate(dm, psip, psipfInt);
        deviceBCValue(dbP, p, pBndVal);
        deviceHadamard(psipBnd, *in.psiBndFace, pBndVal);

        if (st.phiHbyA0Int.size())
        {
            transonicPhiHbyAKernel<<<nBlocks((int)st.phiHbyA0Int.size()), TPB>>>(
                (int)st.phiHbyA0Int.size(), st.phiHbyA0Int.data(), corrInt.data(),
                psipfInt.data(), rhofInt.data(), st.phiHbyAInt.data());
        }
        if (dm.nBndFaces > 0)
        {
            transonicPhiHbyAKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
                dm.nBndFaces, st.phiHbyA0Bnd.data(), corrBnd.data(),
                psipBnd.data(), in.rhoBndFace->data(), st.phiHbyABnd.data());
        }
        cudaCheck(cudaGetLastError(), "rhoPcEqn transonic phiHbyA");
    }
    else
    {
        // adjustPhi FIRST, correction AFTER -- see the header. Reversing them makes adjustPhi balance a
        // different flux and scale the adjustable outflow by a different factor.
        if (in.pRefCell >= 0)
        {
            if (!in.adjustable)
            {
                throw std::runtime_error(
                    "rhoSimpleFoam pcEqn(cuda): adjustPhi needs the per-face `adjustable` mask, and this "
                    "case needs adjustPhi -- p has no fixed-value patch.");
            }
            st.massCorr = deviceAdjustPhi(*in.adjustable, st.phiHbyABnd, &st.phiHbyAInt);
            st.closedVolume = true;
        }
        deviceAxpy(1.0, corrInt, st.phiHbyAInt);
        deviceAxpy(1.0, corrBnd, st.phiHbyABnd);
    }

    // ---- HbyA -= (rAU - rAtU)*fvc::grad(p) --------------------------------------------------
    // The other half of SIMPLEC. Omitting it while keeping rAtU made the converged velocity worse than
    // doing neither, in the incompressible lineage -- the two are one change.
    {
        DeviceBuffer<scalar> dAU;
        deviceCopy(dAU, st.rAU);
        deviceAxpy(-1.0, st.rAtU, dAU);                  // rAU - rAtU
        const DeviceBuffer<scalar>* gp[3] = {&gx, &gy, &gz};
        for (int k = 0; k < 3; ++k)
        {
            deviceCopy(st.HbyA[k], st.HbyA0[k]);
            DeviceBuffer<scalar> t;
            deviceHadamard(t, dAU, *gp[k]);
            deviceAxpy(-1.0, t, st.HbyA[k]);
        }
    }
}


void assemblePcEqn(
    PressureMatrix&                 P,
    const ConsistentPressureStages& st,
    const DeviceMesh&               dm,
    const DeviceBoundary&           dbP,
    const DeviceBuffer<scalar>&     p,
    const RhoPressureInput&         in)
{
    refuseUnsupported(in);
    const int nC = dm.nCells;

    // ---- -fvm::laplacian(rhorAtU, p) --------------------------------------------------------
    deviceLaplacianCoeffs(dm, st.rhorAtUf, P.diag, P.upper, P.lower, in.correctedLaplacian);
    deviceBCLaplacianCoeffsFace(dbP, st.rhorAtUfBnd, P.iC, P.bC);
    zeroed(P.source, nC);

    if (in.correctedLaplacian)
    {
        DeviceBuffer<scalar> pb, gx, gy, gz, lc;
        deviceBCValue(dbP, p, pb);
        deviceGaussGrad(dm, p, pb, gx, gy, gz);
        if (in.snGradLimitCoeff > 0.0)
        {
            DeviceBuffer<scalar> ffcL;
            deviceLaplacianCorrFluxLimited(dm, st.rhorAtUf, p, gx, gy, gz, in.snGradLimitCoeff, ffcL);
            deviceFaceDivSource(dm, ffcL, lc);
            deviceLaplacianCorrFluxLimited(dm, st.rhorAtUf, p, gx, gy, gz, in.snGradLimitCoeff,
                                           P.faceFluxCorr);
        }
        else
        {
            deviceLaplacianCorr(dm, st.rhorAtUf, gx, gy, gz, lc);
            deviceLaplacianCorrFlux(dm, st.rhorAtUf, gx, gy, gz, P.faceFluxCorr);
        }
        deviceAxpy(1.0, lc, P.source);
    }
    else
    {
        P.faceFluxCorr.resize(0);
    }

    // pcEqn.H's implicit operator is MINUS the laplacian, as pEqn.H's is. faceFluxCorr is negated with
    // the rest: fvMatrix::flux() adds it back, and a correction of the wrong sign leaves div(phi) != 0.
    negate(P.diag);
    negate(P.upper);
    negate(P.lower);
    negate(P.source);
    negate(P.iC);
    negate(P.bC);
    negate(P.faceFluxCorr);

    // ---- + fvm::div(phid, p) -- TRANSONIC ONLY, and what makes this branch ASYMMETRIC -------
    if (st.transonic)
    {
        DeviceBuffer<scalar> dD, dU, dL, dIC, dBC;
        deviceDivUpwindCoeffs(dm, st.phidInt, dD, dU, dL);
        deviceBCDivCoeffs(dbP, st.phidBnd, dIC, dBC);
        deviceAxpy(1.0, dD, P.diag);
        deviceAxpy(1.0, dU, P.upper);
        deviceAxpy(1.0, dL, P.lower);
        deviceAxpy(1.0, dIC, P.iC);
        deviceAxpy(1.0, dBC, P.bC);
    }

    // ---- + fvc::div(phiHbyA) ----------------------------------------------------------------
    {
        DeviceBuffer<scalar> divPhi, t;
        deviceDiv(dm, st.phiHbyAInt, st.phiHbyABnd, divPhi);
        deviceHadamard(t, divPhi, dm.V);
        deviceAxpy(-1.0, t, P.source);
    }

    // ---- pcEqn.relax() -- TRANSONIC ONLY ----------------------------------------------------
    if (st.transonic && in.relaxPSpecified && in.relaxP > 0.0)
    {
        DeviceBuffer<scalar> relaxedDiag, delta, t;
        deviceRelaxDiag(P.view(dm), dm, P.iC, in.relaxP, relaxedDiag, delta);
        deviceCopy(P.diag, relaxedDiag);
        deviceHadamard(t, delta, p);
        deviceAxpy(1.0, t, P.source);
    }

    // ---- setReference: source += diag*value; diag += diag (it DOUBLES the diagonal) ---------
    if (in.pRefCell >= 0)
    {
        deviceSetReference(P.diag, P.source, in.pRefCell, in.pRefValue);
    }
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
