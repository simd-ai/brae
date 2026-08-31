// CUDA implementation -- see rhoPEqn.cuh for the provenance, the two branches, and the contract with the
// _cpp reference. Every term here is transcribed from rhoPEqn_cpp.cu in the order that file produces it.
#include "rhoPEqn.cuh"
#include "device_blas.cuh"
#include "device_simple.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

// Launch config, file-local exactly as simpleFoam/pEqn.cu:14-15 keeps it -- one translation unit, one
// definition, no header pollution.
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// fvm::laplacian on the HOST does `M.source.assign(nC, 0)`; deviceLaplacianCoeffs fills only
// diag/upper/lower and leaves the source untouched. The incompressible module never noticed because it
// ASSIGNS its source (deviceHadamard writes it) where this one ACCUMULATES into it -- the non-orthogonal
// correction lands there before div(phiHbyA) does. Without this the first deviceAxpy reads an
// unallocated buffer, which compute-sanitizer reports as an invalid __global__ read at address 0x100.
void zeroed(DeviceBuffer<scalar>& b, int n)
{
    b.resize(n);
    if (n) cudaCheck(cudaMemsetAsync(b.data(), 0, sizeof(scalar) * n, cudaStreamPerThread),
                     "rhoPEqn zero");
}

// constrainHbyA: HbyA takes U's boundary value on a patch whose U is NOT assignable. Mirrors the kernel
// of the same name in simpleFoam/pEqn.cu -- duplicated rather than shared because that one is file-local
// there; the alternative is exporting it, which is the better fix the moment a third lineage needs it.
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

// A() = D/V with D = diag + cmptAv(internalCoeffs): fold each cell's boundary contribution onto its
// diagonal. Same kernel as the incompressible module's, for the same reason as above.
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

// rhorAU's BOUNDARY is rho's PATCH value times rAU's, and rAU is a calculated field whose patch value is
// the OWNER CELL's. That mix -- patch rho, owner-cell rAU -- is what rhoPEqn_cpp.cu:166-169 builds, and it
// is why the laplacian below needs the FACE boundary-coefficient variant rather than the cell one the
// incompressible module uses.
__global__
void rhorAUBndKernel(
    int                        nB,
    const label*  __restrict__ faceCell,
    const scalar* __restrict__ rhoBnd,
    const scalar* __restrict__ rAU,
    scalar*       __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    out[i] = rhoBnd[i] * rAU[faceCell[i]];
}

// phid = (interpolate(psi)/interpolate(rho))*phiHbyA, TRANSONIC only, built from phiHbyA BEFORE the psi*p
// subtraction -- the order in pEqn.H is not incidental and reversing it changes the matrix.
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

// phiHbyA -= interpolate(psi*p)*phiHbyA/interpolate(rho). psi*p is formed PER CELL first and then
// interpolated: interpolate(psi*p) is not interpolate(psi)*interpolate(p) on a non-uniform field.
__global__
void psipSubtractKernel(
    int                        n,
    const scalar* __restrict__ psipf,
    const scalar* __restrict__ rhof,
    scalar*       __restrict__ phiHbyA)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    phiHbyA[f] -= psipf[f] * phiHbyA[f] / rhof[f];
}

// The whole matrix is negated after assembly: pEqn.H is `fvc::div(phiHbyA) - fvm::laplacian(rhorAUf, p)`,
// so the implicit operator is MINUS the laplacian. The incompressible lineage writes the same equation
// with the opposite overall sign (`fvm::laplacian(rAtU, p) == fvc::div(phiHbyA)`), which solves to the
// same p but assembles a matrix of the other sign -- so this cannot be inherited from that module.
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
            "rhoSimpleFoam pEqn(cuda): the case declares MRF, and pEqn.H calls MRF.makeRelative on "
            "phiHbyA BEFORE adjustPhi (pEqn.H:9). Refusing rather than balancing a flux that still "
            "carries the frame motion.");
    }
    if (in.hasFvOptions)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pEqn(cuda): the case declares an fvOption this path does not implement"
            + (in.fvOptionUnsupported.empty() ? std::string()
                                              : std::string(" (") + in.fvOptionUnsupported + ")")
            + ". pEqn.H applies fvOptions(psi, p, rho.name()) to the pressure equation.");
    }
    if (in.hasFixedFluxPressure)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pEqn(cuda): the case uses fixedFluxPressure, whose boundary gradient is set by "
            "constrainPressure(p, rho, U, phiHbyA, rhorAUf) -- not ported. Refusing rather than solving "
            "with a zeroGradient in its place.");
    }
    if (in.hasCoupledPatches)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pEqn(cuda): the mesh has cyclic/AMI/processor patches. buildDeviceMesh keeps "
            "those faces out of the LDU by design, so they would contribute nothing to the laplacian, to "
            "div(phiHbyA) or to the dominance sum -- silently.");
    }
    if (!in.rhoCell || !in.rhoBndFace || !in.psiCell || !in.psiBndFace)
    {
        throw std::runtime_error(
            "rhoSimpleFoam pEqn(cuda): rho and psi are required on cells AND boundary faces. psi is not "
            "transonic-only -- the closed-volume correction after the solve is psi-weighted.");
    }
}

} // namespace


void pressurePredictor(
    RhoPressureStages&          st,
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
            "rhoSimpleFoam pEqn(cuda): constrainHbyA needs the per-face `assignable` mask. assignable() "
            "is not fixesValue() -- slip and inletOutlet are non-assignable without fixing a value -- so "
            "it cannot be recovered from bcType here.");
    }
    st.transonic = in.transonic;
    const int nC = dm.nCells;

    // cmptAv(internalCoeffs) -- shared by A() and by H()'s boundary-diagonal term.
    DeviceBuffer<scalar> icAv;
    deviceCopy(icAv, UEqn.iC[0]);
    deviceAxpy(1.0, UEqn.iC[1], icAv);
    deviceAxpy(1.0, UEqn.iC[2], icAv);
    deviceScale(icAv, 1.0 / 3.0);

    // ---- rAU = 1/UEqn.A() --------------------------------------------------------------------
    // `diag` is the RELAXED diagonal when the matrix has been relaxed: OpenFOAM's relax() writes diag_ in
    // place and A() is taken afterwards, so the boundary average is added ON TOP of the relaxed value.
    {
        DeviceBuffer<scalar> D;
        deviceCopy(D, UEqn.relaxed ? UEqn.relaxedDiag : UEqn.diag);
        foldBoundaryDiagKernel<<<nBlocks(nC), TPB>>>(
            nC, dm.bndCellStart.data(), dm.bndPerm.data(), icAv.data(), D.data());
        cudaCheck(cudaGetLastError(), "rhoPEqn foldBoundaryDiag");
        deviceReciprocalV(dm, D, st.rAU);
    }

    // ---- rhorAUf = fvc::interpolate(rho*rAU) -------------------------------------------------
    // THE PRODUCT is formed per cell and then interpolated. interpolate(rho)*interpolate(rAU) is a
    // different field on a non-uniform rho, and this is the diffusivity of the whole pressure equation.
    {
        DeviceBuffer<scalar> rhorAU;
        deviceHadamard(rhorAU, *in.rhoCell, st.rAU);
        deviceInterpolate(dm, rhorAU, st.rhorAUf);

        st.rhorAUfBnd.resize(dm.nBndFaces);
        if (dm.nBndFaces > 0)
        {
            rhorAUBndKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
                dm.nBndFaces, dm.bndCell.data(), in.rhoBndFace->data(), st.rAU.data(),
                st.rhorAUfBnd.data());
            cudaCheck(cudaGetLastError(), "rhoPEqn rhorAUBnd");
        }
    }

    // ---- HbyA = rAU*UEqn.H(), then constrainHbyA(HbyA, U, p) ---------------------------------
    const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
    const DeviceLduView A = UEqn.view(dm);
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> bdDiagK;
        deviceCopy(bdDiagK, icAv);
        deviceAxpy(-1.0, UEqn.iC[k], bdDiagK);

        DeviceBuffer<scalar> Hk;
        deviceMatrixH(A, dm, *U[k], UEqn.source[k], bdDiagK, UEqn.bC[k], Hk);
        deviceHadamard(st.HbyA[k], st.rAU, Hk);
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
                Ub.data(), st.HbyA[k].data(), st.HbyAb[k].data());
        }
    }
    cudaCheck(cudaGetLastError(), "rhoPEqn constrainHbyA");

    // ---- phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA) -------------------------------------
    // THE FACTORS, interpolated separately -- unlike compressibleCreatePhi.H one file earlier, which
    // interpolates the product rho*U. Both forms are OpenFOAM's and each belongs to exactly one place.
    DeviceBuffer<scalar> rhofInt;
    deviceInterpolate(dm, *in.rhoCell, rhofInt);
    {
        DeviceBuffer<scalar> fluxInt, fluxBnd;
        deviceVectorFlux(dm, st.HbyA[0], st.HbyA[1], st.HbyA[2], fluxInt);
        deviceBoundaryFlux(dm, st.HbyAb[0], st.HbyAb[1], st.HbyAb[2], fluxBnd);
        deviceHadamard(st.phiHbyA0Int, rhofInt, fluxInt);
        deviceHadamard(st.phiHbyA0Bnd, *in.rhoBndFace, fluxBnd);
    }
    deviceCopy(st.phiHbyAInt, st.phiHbyA0Int);
    deviceCopy(st.phiHbyABnd, st.phiHbyA0Bnd);

    st.closedVolume = false;
    st.massCorr = 1.0;

    if (in.transonic)
    {
        // phid FIRST, from phiHbyA BEFORE the subtraction below.
        DeviceBuffer<scalar> psifInt;
        deviceInterpolate(dm, *in.psiCell, psifInt);
        st.phidInt.resize(st.phiHbyAInt.size());
        if (st.phiHbyAInt.size())
        {
            phidKernel<<<nBlocks((int)st.phiHbyAInt.size()), TPB>>>(
                (int)st.phiHbyAInt.size(), psifInt.data(), rhofInt.data(),
                st.phiHbyAInt.data(), st.phidInt.data());
        }
        st.phidBnd.resize(dm.nBndFaces);
        if (dm.nBndFaces > 0)
        {
            phidKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
                dm.nBndFaces, in.psiBndFace->data(), in.rhoBndFace->data(),
                st.phiHbyABnd.data(), st.phidBnd.data());
        }
        cudaCheck(cudaGetLastError(), "rhoPEqn phid");

        // phiHbyA -= fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho).
        DeviceBuffer<scalar> psip, psipfInt, pBnd, psipBnd;
        deviceHadamard(psip, *in.psiCell, p);
        deviceInterpolate(dm, psip, psipfInt);
        deviceBCValue(dbP, p, pBnd);
        deviceHadamard(psipBnd, *in.psiBndFace, pBnd);

        if (st.phiHbyAInt.size())
        {
            psipSubtractKernel<<<nBlocks((int)st.phiHbyAInt.size()), TPB>>>(
                (int)st.phiHbyAInt.size(), psipfInt.data(), rhofInt.data(), st.phiHbyAInt.data());
        }
        if (dm.nBndFaces > 0)
        {
            psipSubtractKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
                dm.nBndFaces, psipBnd.data(), in.rhoBndFace->data(), st.phiHbyABnd.data());
        }
        cudaCheck(cudaGetLastError(), "rhoPEqn psip subtraction");
        // closedVolume stays false: pEqn.H never runs adjustPhi on this branch.
    }
    else
    {
        // adjustPhi(phiHbyA, U, p). It runs exactly when p needs a reference -- adjustPhi.C returns early
        // if any p patch fixesValue -- which is the same condition as pRefCell >= 0. Without it a closed
        // domain hands the singular operator a right-hand side with no solution at all.
        if (in.pRefCell >= 0)
        {
            if (!in.adjustable)
            {
                throw std::runtime_error(
                    "rhoSimpleFoam pEqn(cuda): adjustPhi needs the per-face `adjustable` mask, and this "
                    "case needs adjustPhi -- p has no fixed-value patch.");
            }
            st.massCorr = deviceAdjustPhi(*in.adjustable, st.phiHbyABnd, &st.phiHbyAInt);
            st.closedVolume = true;
        }
    }
}


void assemblePEqn(
    PressureMatrix&             P,
    const RhoPressureStages&    st,
    const DeviceMesh&           dm,
    const DeviceBoundary&       dbP,
    const DeviceBuffer<scalar>& p,
    const RhoPressureInput&     in)
{
    refuseUnsupported(in);
    const int nC = dm.nCells;

    // ---- -fvm::laplacian(rhorAUf, p), the term BOTH branches share --------------------------
    // `corrected` has two halves and both are required: the implicit coefficient takes
    // nonOrthDeltaCoeffs AND an explicit source -V*div(gamma*magSf*(corrVec & interp(grad p))) is added.
    // Implementing only the implicit half is the defect this port already paid for on the energy and
    // pressure equations on the host side -- it moves the source while leaving the diagonal exact, so
    // every gate that compared D() passed. See PORT.md.
    deviceLaplacianCoeffs(dm, st.rhorAUf, P.diag, P.upper, P.lower, in.correctedLaplacian);
    zeroed(P.source, nC);
    // The FACE variant, not the cell one the incompressible module uses: rhorAUf's boundary is rho's
    // PATCH value times rAU's owner-cell value, which is not any single cell's field.
    deviceBCLaplacianCoeffsFace(dbP, st.rhorAUfBnd, P.iC, P.bC);

    if (in.correctedLaplacian)
    {
        DeviceBuffer<scalar> pb, gx, gy, gz, lc;
        deviceBCValue(dbP, p, pb);
        deviceGaussGrad(dm, p, pb, gx, gy, gz);
        // The explicit half: source gets -V*div(gamma*magSf*(corrVec & interp(grad p))), which
        // deviceLaplacianCorr already returns pre-negated (device_mesh.cuh:288), hence the +1 here.
        if (in.snGradLimitCoeff > 0.0)
        {
            DeviceBuffer<scalar> ffcL;
            deviceLaplacianCorrFluxLimited(dm, st.rhorAUf, p, gx, gy, gz, in.snGradLimitCoeff, ffcL);
            deviceFaceDivSource(dm, ffcL, lc);
        }
        else
        {
            deviceLaplacianCorr(dm, st.rhorAUf, gx, gy, gz, lc);
        }
        deviceAxpy(1.0, lc, P.source);
        // ...and the matching FACE FLUX. fvMatrix::flux() adds it back, and `phi = phiHbyA - pEqn.flux()`
        // is what makes phi discretely conservative: a flux missing the correction its source carries
        // leaves div(phi) != 0 on a non-orthogonal mesh, silently, because the equation still solves.
        if (in.snGradLimitCoeff > 0.0)
        {
            deviceLaplacianCorrFluxLimited(dm, st.rhorAUf, p, gx, gy, gz, in.snGradLimitCoeff,
                                           P.faceFluxCorr);
        }
        else
        {
            deviceLaplacianCorrFlux(dm, st.rhorAUf, gx, gy, gz, P.faceFluxCorr);
        }
    }
    else
    {
        P.faceFluxCorr.resize(0);
    }

    // The sign flip: pEqn.H's implicit operator is MINUS the laplacian. Everything the laplacian produced
    // is negated together, including the stored face-flux correction -- fvMatrix::flux() adds that back,
    // and a correction of the wrong sign leaves div(phi) != 0 on a non-orthogonal mesh, silently.
    negate(P.diag);
    negate(P.upper);
    negate(P.lower);
    negate(P.source);
    negate(P.iC);
    negate(P.bC);
    negate(P.faceFluxCorr);

    // ---- + fvm::div(phid, p) -- TRANSONIC ONLY ----------------------------------------------
    // This is what makes the pressure equation convective and it is the entire structural difference
    // between the branches' matrices. It is also what makes this branch ASYMMETRIC: upwind sets
    // lower = -w*phi and upper = lower + phi, so CG is not a legal solver here. See rhoPEqn.cuh.
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
    // An explicit field on the LEFT of the equation, so source -= V*div. fvc::div returns the per-volume
    // divergence, so the V multiplies back out.
    {
        DeviceBuffer<scalar> divPhi, t;
        deviceDiv(dm, st.phiHbyAInt, st.phiHbyABnd, divPhi);
        deviceHadamard(t, divPhi, dm.V);
        deviceAxpy(-1.0, t, P.source);
    }

    // ---- pEqn.relax() -- TRANSONIC ONLY -----------------------------------------------------
    // pEqn.H relaxes on that branch and not on the subsonic one. Relaxing both, or neither, changes the
    // iteration path without changing the converged answer -- exactly the difference a converged-field
    // comparison cannot see. The guard is "the case NAMES a factor", not "the factor is below 1":
    // relax(1.0) still runs the dominance clamp and adds (D - D0)*psi to the source.
    if (st.transonic && in.relaxPSpecified && in.relaxP > 0.0)
    {
        DeviceBuffer<scalar> relaxedDiag, delta;
        deviceRelaxDiag(P.view(dm), dm, P.iC, in.relaxP, relaxedDiag, delta);
        deviceCopy(P.diag, relaxedDiag);
        DeviceBuffer<scalar> t;
        deviceHadamard(t, delta, p);
        deviceAxpy(1.0, t, P.source);
    }

    // ---- pEqn.setReference(refCell, refValue) -----------------------------------------------
    // fvMatrix.C:1011-1023, verbatim: source[cell] += diag[cell]*value; diag[cell] += diag[cell]. It
    // DOUBLES the diagonal rather than setting it. A "pin the cell to a value" reading gives a different
    // matrix that still solves.
    if (in.pRefCell >= 0)
    {
        deviceSetReference(P.diag, P.source, in.pRefCell, in.pRefValue);
    }
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
