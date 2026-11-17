// The implicit upwind pre-solve -- see device_alpha_presolve.cuh for the provenance and for why the
// convection here is upwind and not the case's scheme.
#include "device_alpha_presolve.cuh"
#include "device_mules.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"   // deviceSymGaussSeidel
#include "device_blas.cuh"
#include "device_processor_ops.cuh"
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

__global__ void foldPhysicalKernel(
    int nC, int nPhys, const label* __restrict__ start, const label* __restrict__ perm,
    const scalar* __restrict__ rawDiag, const scalar* __restrict__ source,
    const scalar* __restrict__ iC, const scalar* __restrict__ bC,
    scalar* __restrict__ diag, scalar* __restrict__ b)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar dd = rawDiag[c], bb = source[c];
    for (int k = start[c]; k < start[c+1]; ++k)
    {
        const int j = perm[k];
        if (j < nPhys) { dd += iC[j]; bb += bC[j]; }
    }
    diag[c] = dd; b[c] = bb;
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
    const bool procMesh = DeviceProcCtx::active(dm);
    if ((!procMesh && (static_cast<int>(iC.size()) != nBf || static_cast<int>(bC.size()) != nBf))
     || static_cast<int>(iC.size()) > nBf || static_cast<int>(bC.size()) != static_cast<int>(iC.size()))
        throw std::runtime_error(
            "brae interFoam alpha pre-solve: internalCoeffs and boundaryCoeffs must have one entry per "
            "boundary face, flattened in the mesh's boundary-face order. They come from the host's "
            "per-patch BC dispatch -- see device_alpha_presolve.cuh.");
    DeviceBuffer<scalar> paddedIC, paddedBC;
    const DeviceBuffer<scalar>* icPtr = &iC;
    const DeviceBuffer<scalar>* bcPtr = &bC;
    if (procMesh && static_cast<int>(iC.size()) < nBf)
    {
        paddedIC.resize(static_cast<std::size_t>(nBf));
        paddedBC.resize(static_cast<std::size_t>(nBf));
        const std::size_t bytes = iC.size()*sizeof(scalar);
        if (bytes)
        {
            ckP(cudaMemcpy(paddedIC.data(), iC.data(), bytes, cudaMemcpyDeviceToDevice), "physical iC");
            ckP(cudaMemcpy(paddedBC.data(), bC.data(), bytes, cudaMemcpyDeviceToDevice), "physical bC");
        }
        ckP(cudaMemset(paddedIC.data() + iC.size(), 0, sizeof(scalar)*(nBf-iC.size())), "processor iC zero");
        ckP(cudaMemset(paddedBC.data() + bC.size(), 0, sizeof(scalar)*(nBf-bC.size())), "processor bC zero");
        icPtr = &paddedIC; bcPtr = &paddedBC;
    }

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

    // Processor patches are not ordinary valueInternal/valueBoundary patches: their upwind coupling
    // is exactly the cyclic convection kernel, with the halo as its neighbour. The placeholder patch
    // coefficients must not enter the folded diagonal/source on these faces.
    DeviceProcCtx* const proc = (DeviceProcCtx::active(dm) && !(cyc && cyc->n > 0))
        ? DeviceProcCtx::current() : nullptr;
    if (proc)
    {
        if (!phiCNIf) throw std::runtime_error("brae interFoam alpha pre-solve: processor faces need phiCN.");
        procExchange(*proc, {&alpha1}, "alpha1 pre-solve initial guess");
        proc->cyc().ifCoeff.resize(static_cast<std::size_t>(proc->nFaces()));
        ckP(cudaMemset(proc->cyc().ifCoeff.data(), 0, sizeof(scalar)*proc->nFaces()), "processor upwind coefficients");
        deviceCyclicAddConvection(proc->cyc(), *phiCNIf, rawDiag);
        if (sc.matrixProcCoeffOut) deviceCopy(*sc.matrixProcCoeffOut, proc->cyc().ifCoeff);
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

    // Fold physical patches only. The processor suffix is a coupled interface and is folded by the
    // convection kernel above; its placeholder patch has no matrix coefficients.
    DeviceBuffer<scalar> diagC(static_cast<std::size_t>(nC)), b(static_cast<std::size_t>(nC));
    if (proc)
    {
        foldPhysicalKernel<<<nBlocks(nC), TPB>>>(nC, proc->bndStart(), dm.bndCellStart.data(),
            dm.bndPerm.data(), rawDiag.data(), source.data(), icPtr->data(), bcPtr->data(), diagC.data(), b.data());
        ckP(cudaGetLastError(), "fold physical alpha boundary");
    }
    else
    {
        deviceFold(dm, rawDiag, source, *icPtr, *bcPtr, diagC, b);
    }
    if (sc.matrixDiagOut) deviceCopy(*sc.matrixDiagOut, diagC);
    if (sc.matrixSourceOut) deviceCopy(*sc.matrixSourceOut, b);
    if (sc.matrixUpperOut) deviceCopy(*sc.matrixUpperOut, upper);
    if (sc.matrixLowerOut) deviceCopy(*sc.matrixLowerOut, lower);

    if (sc.matrixICOut) deviceCopy(*sc.matrixICOut, *icPtr);
    if (sc.matrixBCOut) deviceCopy(*sc.matrixBCOut, *bcPtr);
    DeviceLduView A = (cyc && cyc->n > 0)
        ? deviceLduViewPair(dm, diagC, upper, lower, *cyc)
        : deviceLduView(dm, diagC, upper, lower);
    if (proc)
    {
        A.proc.comm = &proc->comm();
        A.proc.plan = &proc->plan();
        A.proc.nFaces = proc->nFaces();
        A.proc.faceCell = proc->plan().dFaceCells();
        A.proc.nbrIdx = proc->plan().dFaceHaloCell();
        A.proc.coeff = proc->cyc().ifCoeff.data();
        A.proc.rank = proc->cyc().ownerRank();
        A.proc.nRanks = proc->cyc().ownerPasses();
        A.proc.nGlobalCells = static_cast<long long>(proc->comm().allReduce(static_cast<double>(nC), ReduceOp::Sum));
    }

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
    DeviceBuffer<scalar> procSolvedFlux;
    if (proc)
    {
        deviceMulesDonorFluxCyclic(proc->cyc(), *phiCNIf, alpha1, procSolvedFlux);
    }
    alphaPhi10Bnd.resize(static_cast<std::size_t>(nBf));
    if (nBf > 0)
    {
        boundaryFluxKernel<<<nBlocks(nBf), TPB>>>(dm.bndCell.data(), icPtr->data(), bcPtr->data(),
                                                  alpha1.data(), nBf, alphaPhi10Bnd.data());
        ckP(cudaGetLastError(), "matrix flux, boundary");
    }
    if (proc) procScatter(*proc, procSolvedFlux, alphaPhi10Bnd);
    (void)nIf;
    if (perfOut)
    {
        *perfOut = perf;
    }
    return perf.finalResidual;
}

} // namespace brae
