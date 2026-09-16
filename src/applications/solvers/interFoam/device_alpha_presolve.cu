// The implicit upwind pre-solve -- see device_alpha_presolve.cuh for the provenance and for why the
// convection here is upwind and not the case's scheme.
#include "device_alpha_presolve.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_blas.cuh"
#include "device_simple.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

void ckP(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae interFoam alpha pre-solve: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// fvm::ddt(alpha1), Euler, rho == 1 (EulerDdtScheme.C, the scalar overload):
//     diag   += V/dt
//     source += V*alpha.oldTime()/dt
// The source is BUILT here rather than added to an existing one because Su is zeroField for interFoam
// -- there is nothing else in it.
__global__ void eulerDdtKernel(const scalar* __restrict__ V, const scalar* __restrict__ psiOld,
                               int nC, scalar rDeltaT,
                               scalar* __restrict__ diag, scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    diag[c]  += rDeltaT * V[c];
    source[c] = rDeltaT * V[c] * psiOld[c];
}

// alpha1Eqn.flux() at a BOUNDARY face: internalCoeffs*psi[faceCell] - boundaryCoeffs (fvMatrix.C:1688).
// It reads the face cell's INTERNAL value, not the patch value -- the matrix's boundary coefficients
// already carry everything the patch condition contributes.
__global__ void boundaryFluxKernel(const label* __restrict__ bndCell,
                                   const scalar* __restrict__ iC, const scalar* __restrict__ bC,
                                   const scalar* __restrict__ psi, int nBf,
                                   scalar* __restrict__ flux)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nBf) flux[b] = iC[b]*psi[bndCell[b]] - bC[b];
}

}   // namespace


scalar deviceAlphaPreSolve(
    const DeviceMesh&             dm,
    DeviceBuffer<scalar>&         alpha1,
    const DeviceBuffer<scalar>&   alpha1Old,
    const DeviceBuffer<scalar>&   phiCNInt,
    const DeviceBuffer<scalar>&   iC,
    const DeviceBuffer<scalar>&   bC,
    scalar                        deltaT,
    const DeviceAlphaSolverControls& sc,
    DeviceBuffer<scalar>&         alphaPhi10Int,
    DeviceBuffer<scalar>&         alphaPhi10Bnd)
{
    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nBf = dm.nBndFaces;
    if (static_cast<int>(iC.size()) != nBf || static_cast<int>(bC.size()) != nBf)
        throw std::runtime_error(
            "brae interFoam alpha pre-solve: internalCoeffs and boundaryCoeffs must have one entry per "
            "boundary face, flattened in the mesh's boundary-face order. They come from the host's "
            "per-patch BC dispatch -- see device_alpha_presolve.cuh.");

    // fvm::div(phiCN, alpha1) with UPWIND weights -- OpenFOAM names the scheme in the code rather than
    // reading it from fvSchemes, and so does this.
    DeviceBuffer<scalar> rawDiag, upper, lower;
    deviceDivUpwindCoeffs(dm, phiCNInt, rawDiag, upper, lower);

    DeviceBuffer<scalar> source(static_cast<std::size_t>(nC));
    eulerDdtKernel<<<nBlocks(nC), TPB>>>(dm.V.data(), alpha1Old.data(), nC, scalar(1)/deltaT,
                                         rawDiag.data(), source.data());
    ckP(cudaGetLastError(), "Euler ddt");

    // fvMatrix::solve's completion: the boundary internalCoeffs go onto the diagonal and the
    // boundaryCoeffs into the source, which is what makes the system square.
    DeviceBuffer<scalar> diagC, b;
    deviceFold(dm, rawDiag, source, iC, bC, diagC, b);

    const DeviceLduView A = deviceLduView(dm, diagC, upper, lower);

    // OpenFOAM's lduMatrix::solver::normFactor, not sum|b| -- it scales every residual the solver
    // reports and tests, so the absolute `tolerance` means something different under the other one.
    DeviceBuffer<scalar> dNf;
    deviceNormFactorInto(A, alpha1, b, deviceOnes(nC), dNf);

    // BiCGStab because the matrix is ASYMMETRIC: upwind convection gives upper != lower.
    const DeviceSolverPerf perf =
        deviceJacobiBiCGStab(A, b, alpha1, dNf.data(), sc.tol, sc.relTol, sc.maxIter);

    // alphaPhi10 = alpha1Eqn.flux(), the CONSERVATIVE flux of the solved matrix -- not the upwind flux
    // of the solved field. The two differ by the solver's residual.
    deviceMatrixFluxInternal(A, alpha1, alphaPhi10Int);
    alphaPhi10Bnd.resize(static_cast<std::size_t>(nBf));
    if (nBf > 0)
    {
        boundaryFluxKernel<<<nBlocks(nBf), TPB>>>(dm.bndCell.data(), iC.data(), bC.data(),
                                                  alpha1.data(), nBf, alphaPhi10Bnd.data());
        ckP(cudaGetLastError(), "matrix flux, boundary");
    }
    (void)nIf;
    return perf.finalResidual;
}

} // namespace brae
