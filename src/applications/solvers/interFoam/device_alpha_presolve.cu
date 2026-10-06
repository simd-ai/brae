// The implicit upwind pre-solve -- see device_alpha_presolve.cuh for the provenance and for why the
// convection here is upwind and not the case's scheme.
#include "device_alpha_presolve.cuh"
#include "device_mules.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"   // deviceSymGaussSeidel
#include "device_blas.cuh"
#include "device_simple.cuh"
#include "inter_phase_time.cuh"
#include <optional>
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
__global__ void eulerDdtKernel(const scalar* __restrict__ V,      // mesh.Vsc()
                               const scalar* __restrict__ V0,     // mesh.Vsc0(), null if not moving
                               const scalar* __restrict__ psiOld,
                               int nC, scalar rDeltaT,
                               scalar* __restrict__ diag, scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    if (V0)
    {
        // the moving branch, in the host's own order (alpha_eqn_cpp.cu:497-501)
        diag[c]  += rDeltaT * V[c];
        source[c] = rDeltaT * psiOld[c] * V0[c];
        return;
    }
    diag[c]  += rDeltaT * V[c];
    source[c] = rDeltaT * V[c] * psiOld[c];
}

// fvm::ddt(alpha1) under localEuler (localEulerDdtScheme.C:245-246), static mesh:
//     diag   += rDeltaT*V
//     source  = rDeltaT*alpha.oldTime()*V      -- the host's order, alpha_eqn_cpp.cu
__global__ void localEulerDdtKernel(const scalar* __restrict__ V,
                                    const scalar* __restrict__ psiOld,
                                    const scalar* __restrict__ rDeltaT,
                                    int nC,
                                    scalar* __restrict__ diag, scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar rDT = rDeltaT[c];
    diag[c]  += rDT * V[c];
    source[c] = rDT * psiOld[c] * V[c];
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
    DeviceBuffer<scalar>&         alphaPhi10Bnd,
    DeviceSolverPerf*             perfOut,
    DeviceCyclic*                 cyc,
    DeviceBuffer<scalar>*         alphaPhi10If,
    const DeviceBuffer<scalar>*   Vsc,
    const DeviceBuffer<scalar>*   Vsc0,
    const DeviceBuffer<scalar>* rDeltaT,
    const DeviceBuffer<scalar>* phiCNIf)
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
    std::optional<interPhase::Nested> part;
    part.emplace("alpha pre-solve: the matrix");
    deviceDivUpwindCoeffs(dm, phiCNInt, rawDiag, upper, lower);
    // ...and the PAIR, whose faces are in neither the internal list nor the boundary one. Upwind gives
    // a coupled face internalCoeffs = phi*w and boundaryCoeffs = -(phi*(1 - w)) with w = pos0(phi), as
    // on an internal face (fvm.cuh:536-549); deviceCyclicAddConvection is that, gated face by face in
    // tests/test_device_cyclic_laplacian_vs_host.cu. The convecting flux there is phiCN's, handed in.
    if (cyc && cyc->n > 0)
    {
        if (!phiCNIf || static_cast<label>(phiCNIf->size()) != cyc->n)
        {
            throw std::runtime_error(
                "brae interFoam alpha pre-solve: the mesh has a coupled pair and the caller handed no phiCN "
                "for its faces. fvm::div(phiCN, alpha1) convects with phiCN there as on every face "
                "(alphaEqn.H:110-115); the pair's own flux is not it under CrankNicolson.");
        }
        std::vector<scalar> zeros(static_cast<std::size_t>(cyc->n), scalar(0));
        cyc->ifCoeff.copyFrom(zeros);      // ADDS to the interface coefficient, so it starts clean
        deviceCyclicAddConvection(*cyc, *phiCNIf, rawDiag);
    }

    DeviceBuffer<scalar> source(static_cast<std::size_t>(nC));
    if (rDeltaT)
    {
        if (Vsc || static_cast<int>(rDeltaT->size()) != nC)
            throw std::runtime_error(
                "brae interFoam alpha pre-solve: a local time step on a moving mesh is not ported, and the "
                "rDeltaT field must have one value per cell.");
        localEulerDdtKernel<<<nBlocks(nC), TPB>>>(dm.V.data(), alpha1Old.data(), rDeltaT->data(), nC,
                                                  rawDiag.data(), source.data());
        ckP(cudaGetLastError(), "localEuler ddt");
    }
    else
    {
        eulerDdtKernel<<<nBlocks(nC), TPB>>>(Vsc ? Vsc->data() : dm.V.data(),
                                             Vsc0 ? Vsc0->data() : nullptr,
                                             alpha1Old.data(), nC, scalar(1)/deltaT,
                                             rawDiag.data(), source.data());
        ckP(cudaGetLastError(), "Euler ddt");
    }

    // fvMatrix::solve's completion: the boundary internalCoeffs go onto the diagonal and the
    // boundaryCoeffs into the source, which is what makes the system square.
    DeviceBuffer<scalar> diagC, b;
    deviceFold(dm, rawDiag, source, iC, bC, diagC, b);

    const DeviceLduView A = (cyc && cyc->n > 0)
        ? deviceLduViewPair(dm, diagC, upper, lower, *cyc)
        : deviceLduView(dm, diagC, upper, lower);

    // OpenFOAM's lduMatrix::solver::normFactor, not sum|b| -- it scales every residual the solver
    // reports and tests, so the absolute `tolerance` means something different under the other one.
    DeviceBuffer<scalar> dNf;
    deviceNormFactorInto(A, alpha1, b, deviceOnes(nC), dNf);

    // THE CASE'S OWN SMOOTHER WHERE IT NAMES ONE, and the choice decides the answer. An implicit upwind
    // matrix is nearly triangular in flow order, so a Gauss-Seidel sweep is nearly an exact solve:
    // OpenFOAM's log on damBreak reads "No Iterations 2, Final residual 9.3e-14" at `tolerance 1e-8`.
    // Jacobi-BiCGStab has no such property and stops AT 1e-8. Measured against real OpenFOAM, five
    // steps of damBreak with this solve at the case's 1e-8: alpha 3.3e-06 and U 1.2e-03 out. The driver
    // used to hide that by hardcoding 1e-12 (alpha 1.6e-10) -- a tolerance nobody chose, standing in
    // for a solver the case did not ask for.
    part.emplace("alpha pre-solve: the linear solve");
    DeviceSolverPerf perf;
    if (sc.smoothSolver)
    {
        deviceSymGaussSeidel(A, b, alpha1, dNf.data(), sc.tol, sc.relTol, sc.maxIter,
                             &perf, sc.minIter, sc.nSweeps, sc.symmetric);
    }
    else
    {
        // BiCGStab because the matrix is ASYMMETRIC: upwind convection gives upper != lower.
        perf = deviceJacobiBiCGStab(A, b, alpha1, dNf.data(), sc.tol, sc.relTol, sc.maxIter,
                                    /*checkEvery=*/1, sc.minIter);
    }

    part.emplace("alpha pre-solve: the flux of the solved matrix");
    // alphaPhi10 = alpha1Eqn.flux(), the CONSERVATIVE flux of the solved matrix. For a pure upwind
    // matrix that IS the upwind flux of the solved field, bit for bit; it is taken from the matrix
    // because that is what alphaEqn.H does and what stays right if the scheme ever changes.
    deviceMatrixFluxInternal(A, alpha1, alphaPhi10Int);
    // ...and the pair's own face flux. fvMatrix::flux() on a coupled patch is
    // internalCoeffs*pif - boundaryCoeffs*pnf (fvMatrix.C:1483-1512); with upwind's w = pos0(phi) that
    // is phi*alpha[own] on an outflow face and phi*alpha[nbr] on an inflow one -- the upwind flux of
    // the SOLVED alpha, which is what deviceMulesDonorFluxCyclic computes.
    if (cyc && cyc->n > 0 && alphaPhi10If)
    {
        deviceMulesDonorFluxCyclic(*cyc, *phiCNIf, alpha1, *alphaPhi10If);
    }
    alphaPhi10Bnd.resize(static_cast<std::size_t>(nBf));
    if (nBf > 0)
    {
        boundaryFluxKernel<<<nBlocks(nBf), TPB>>>(dm.bndCell.data(), iC.data(), bC.data(),
                                                  alpha1.data(), nBf, alphaPhi10Bnd.data());
        ckP(cudaGetLastError(), "matrix flux, boundary");
    }
    (void)nIf;
    if (perfOut)
    {
        *perfOut = perf;
    }
    return perf.finalResidual;
}

} // namespace brae
