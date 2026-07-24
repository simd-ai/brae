// cf GPU offload (G2): device-resident Jacobi-PCG. Mirrors brae::pcg's CG recurrence exactly, but the
// preconditioner is Jacobi (wA = rA/diag) and every vector op is a device kernel.
#include "device_pcg.cuh"
#include "device_amg.cuh"   // AMGData + amgVCycleApply (the distributed AMG-PCG preconditioner)
#include "device_blas.cuh"
#include "device_halo.cuh"
#include "device_reduce.cuh"   // DeviceReducer: on-stream NVSHMEM global reduction (replaces host MPI_Allreduce)
#include "cf_pstream.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <vector>


namespace brae {

namespace {

// Aggregation coarse-space kernels (balanced two-level, BRAE_COARSE_SPACE != 0). Z is a partition-of-unity: each fine cell
// belongs to exactly one (global) aggregate. aggSumK = Z^T (sum fine values into their aggregate, atomic scatter);
// aggGatherK = Z (broadcast an aggregate's coarse value back to its fine cells). CS_TPB/csBlocks are local here.
constexpr int CS_TPB = 256;
inline int csBlocks(int n) { return (n + CS_TPB - 1) / CS_TPB; }
__global__ void aggSumK(int n, const label* __restrict__ agg, const scalar* __restrict__ v, scalar* __restrict__ csum)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(&csum[agg[i]], v[i]); }
__global__ void aggGatherK(int n, const label* __restrict__ agg, const scalar* __restrict__ mu, scalar* __restrict__ out)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = mu[agg[i]]; }

// GALERKIN coarse-operator assembly: E = Z^T A Z scattered straight from the LDU matrix (no matvecs). E[a][b] =
// sum over matrix entries (i,j) with i in aggregate a, j in aggregate b, of A[i][j]. Diagonal + internal faces are
// local; interface faces use the neighbour's (exchanged) aggregate id. E is row-major totalCoarse x totalCoarse.
__global__ void galDiagK(int n, const label* __restrict__ agg, const scalar* __restrict__ diag, int tc, scalar* __restrict__ E)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { const label a = agg[i]; atomicAdd(&E[static_cast<std::size_t>(a) * tc + a], diag[i]); } }
__global__ void galFaceK(int nF, const label* __restrict__ own, const label* __restrict__ nei, const label* __restrict__ agg,
                         const scalar* __restrict__ up, const scalar* __restrict__ lo, int tc, scalar* __restrict__ E)
{ const int f = blockIdx.x * blockDim.x + threadIdx.x; if (f < nF) { const label ao = agg[own[f]], an = agg[nei[f]];
    atomicAdd(&E[static_cast<std::size_t>(ao) * tc + an], up[f]); atomicAdd(&E[static_cast<std::size_t>(an) * tc + ao], lo[f]); } }
__global__ void galIfaceK(int n, const label* __restrict__ faceCells, const label* __restrict__ agg,
                          const scalar* __restrict__ nbrAggS, const scalar* __restrict__ ifCoeff, int tc, scalar* __restrict__ E)
{ const int f = blockIdx.x * blockDim.x + threadIdx.x; if (f < n) { const label ao = agg[faceCells[f]]; const int an = static_cast<int>(nbrAggS[f] + 0.5);
    atomicAdd(&E[static_cast<std::size_t>(ao) * tc + an], -ifCoeff[f]); } }
__global__ void labelToScalarK(int n, const label* __restrict__ src, scalar* __restrict__ dst)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = static_cast<scalar>(src[i]); }

// BiCGStab device-scalar recurrence kernels (1 thread): keep omega/beta off the host (like the AMG-PCG scalars).
// Breakdown is OF PBiCGStab's SolverPerformance::checkSingularity: |rho(rA0rA)| or |omega| < VSMALL. We detect it
// ON-DEVICE into a flag `bd` (read only on the K-cadence convergence check, not every iter -> removes the 2 per-iter
// breakdown D2H reads). The guarded recurrence stays NaN-safe between checks; for a NON-breakdown solve nothing trips
// (rho/omega stay >> VSMALL until convergence breaks first) so the iteration is bit-identical to the host-guarded form.
constexpr scalar BICG_VSMALL = 1e-300;   // = OF solveScalar VSMALL (SolverPerformance.C checkSingularity: mag(x) < vsmall_)
__global__
void omegaK(
    const scalar* __restrict__ ts,
    const scalar* __restrict__ tt,
    scalar* __restrict__ om,
    scalar* __restrict__ negOm,
    scalar* __restrict__ bd)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar t=*tt;
        const scalar o = (t > BICG_VSMALL) ? (*ts)/t : 0.0;
        *om=o;
        *negOm=-o;
        if (fabs(o) < BICG_VSMALL) *bd = 1.0;            // OF: checkSingularity(mag(omega)) (guards next-iter beta)
    }
}
__global__
void bicgBetaK(
    const scalar* __restrict__ rr,
    const scalar* __restrict__ rrOld,
    const scalar* __restrict__ al,
    const scalar* __restrict__ om,
    scalar* __restrict__ beta,
    scalar* __restrict__ bd)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar ro=*rrOld, o=*om;
        const bool ok = fabs(ro) >= BICG_VSMALL && fabs(o) >= BICG_VSMALL;
        *beta = ok ? (*rr / ro) * (*al / o) : 0.0;       // beta = (rA0rA/rA0rAold)*(alpha/omega), guarded
        if (!ok) *bd = 1.0;
    }
}
__global__
void bicgRhoSingK(
    const scalar* __restrict__ rr,
    scalar* __restrict__ bd)
{
    if (threadIdx.x==0 && blockIdx.x==0 && fabs(*rr) < BICG_VSMALL) *bd = 1.0;   // OF: checkSingularity(mag(rA0rA))
}
__global__
void bicgAlphaK(
    const scalar* __restrict__ rr,
    const scalar* __restrict__ r0Ay,
    scalar* __restrict__ al,
    scalar* __restrict__ negAl,
    scalar* __restrict__ bd)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar d=*r0Ay;
        const bool ok = fabs(d) >= BICG_VSMALL;
        const scalar q = ok ? (*rr)/d : 0.0;
        *al=q;
        *negAl=-q;                                // OF: alpha = rA0rA/rA0AyA (guarded vs 0/0 breakdown)
        if (!ok) *bd = 1.0;
    }
}
// Persistent BiCGStab device scalars (one-time alloc; single solve at a time, sequential across the 3 U components).
struct BiCGScalars
{
    DeviceBuffer<scalar> rr, rrOld, alpha, negAlpha, omega, negOmega, beta, r0Ay, tt, ts, sNorm, rNorm, bd;
    BiCGScalars()
    {
        for (auto* p : {&rr,&rrOld,&alpha,&negAlpha,&omega,&negOmega,&beta,&r0Ay,&tt,&ts,&sNorm,&rNorm,&bd})
            p->resize(1);
    }
};
inline BiCGScalars& bicgScalars() { static BiCGScalars s; return s; }
} // namespace

DeviceSolverPerf deviceJacobiPCG(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const int nC = A.nCells;
    DeviceBuffer<scalar> wA(nC), rA(nC), pA(nC), Ax(nC);

    deviceAmul(A, psi, Ax);                                  // rA = b - A*psi
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);

    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    scalar wArA = 1e300, wArAold;
    int nIter = 0;
    if (!converged(perf.finalResidual))
    {
        do
        {
            wArAold = wArA;
            deviceJacobi(wA, rA, A.diag);                   // wA = M^-1 rA  (Jacobi)
            wArA = deviceDot(wA, rA);
            if (nIter == 0) deviceCopy(pA, wA);             // pA = wA
            else                                            // pA = wA + beta*pA
            {
                const scalar beta = wArA / wArAold;
                deviceScale(pA, beta);
                deviceAxpy(1.0, wA, pA);
            }
            deviceAmul(A, pA, wA);                          // wA = A*pA
            const scalar wApA  = deviceDot(wA, pA);
            const scalar alpha = wArA / wApA;
            deviceAxpy(alpha, pA, psi);                     // psi += alpha*pA
            deviceAxpy(-alpha, wA, rA);                     // rA  -= alpha*wA
            perf.finalResidual = deviceSumMag(rA) / normFactor;
            ++nIter;
        } while (nIter < maxIter && !converged(perf.finalResidual));
    }
    perf.nIterations = nIter;
    return perf;
}

scalar deviceNormFactor(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& b,
    const DeviceBuffer<scalar>& ones)
{
    const int nC = A.nCells;
    DeviceBuffer<scalar> Apsi(nC), sumA(nC), tmp(nC), t(nC);
    // Device-resident: the 3 reductions (avgPsi, n1, n2) stay on the device; only the final normFactor is read to
    // the host (it scales the residual for the convergence check). Same kernels + same IEEE ops (the divide by nC,
    // the avg-multiply, and the (n1+n2)+1e-20 add are reproduced exactly) -> bit-identical, 3 D2H syncs -> 1.
    DeviceBuffer<scalar> dAvg(1), dN1(1), dN2(1), dNorm(1);
    deviceAmul(A, psi, Apsi);                                // A*psi
    deviceAmul(A, ones, sumA);                               // sumA = rowSum(A) = A*1
    deviceDotInto(psi, ones, dAvg.data());                  // psi.ones
    deviceScalarDivConst(dAvg.data(), (scalar)nC, dAvg.data());   // avgPsi = gAverage(psi) = (psi.ones)/nC
    deviceCopy(tmp, sumA);
    deviceScaleDev(dAvg.data(), tmp);      // tmp = sumA*avg(psi)
    deviceCopy(t, Apsi);
    deviceAxpy(-1.0, tmp, t);
    deviceSumMagInto(t, dN1.data());   // n1 = |A*psi - tmp|
    deviceCopy(t, b);
    deviceAxpy(-1.0, tmp, t);
    deviceSumMagInto(t, dN2.data());   // n2 = |b - tmp|
    deviceScalarAdd2(dN1.data(), dN2.data(), 1e-20, dNorm.data());                    // n1 + n2 + 1e-20
    return deviceReadScalar(dNorm.data());                                            // the only host sync
}

DeviceSolverPerf deviceJacobiBiCGStab(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int checkEvery)
{
    const int nC = A.nCells;
    const int K = (checkEvery > 1) ? checkEvery : 1;             // convergence-read cadence (1 = exact per-iter)
    DeviceBuffer<scalar> rA(nC), rA0(nC), pA(nC), yA(nC), AyA(nC), sA(nC), zA(nC), tA(nC), Ax(nC);

    deviceAmul(A, psi, Ax);                                  // rA = b - A*psi
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);
    deviceCopy(rA0, rA);

    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    // DEVICE-RESIDENT BiCGStab: alpha/omega/beta and all dots live on the device (fed to the *Dev kernels by pointer).
    // Breakdown (OF checkSingularity: |rA0rA| or |omega| < VSMALL) is detected ON-DEVICE into `bd` and read only on the
    // K-cadence check (with |s|/|r|), so the host reads NOTHING per off-check iter (was 2 per-iter breakdown D2H reads).
    // For any non-breakdown solve nothing trips (rho/omega stay >> VSMALL until convergence breaks first) -> same kernels
    // + IEEE ops + branches -> bit-identical. On a true breakdown the guarded recurrence damps (no NaN) and the batched
    // bd read stops it within K iters (vs OF's immediate break -- equivalent result; breakdown is pathological & rare).
    BiCGScalars& s = bicgScalars();
    cudaCheck(cudaMemsetAsync(s.bd.data(), 0, sizeof(scalar), cudaStreamPerThread), "bicg bd zero");
    int nIter = 0;
    if (!converged(perf.finalResidual))
    {
        do
        {
            if (nIter > 0) deviceScalarCopy(s.rr.data(), s.rrOld.data());   // rA0rAold = rA0rA
            deviceDotInto(rA0, rA, s.rr.data());                  // rA0rA = rA0 . rA
            bicgRhoSingK<<<1,1>>>(s.rr.data(), s.bd.data());      // OF checkSingularity(mag(rA0rA)) -> flag (no host read)
            if (nIter == 0) deviceCopy(pA, rA);
            else
            {
                bicgBetaK<<<1,1>>>(s.rr.data(), s.rrOld.data(), s.alpha.data(), s.omega.data(), s.beta.data(), s.bd.data());
                deviceFusedBicgP(rA, pA, AyA, s.beta.data(), s.negOmega.data());            // pA = rA + beta*(pA - omega*AyA)  [fused 3->1]
            }
            deviceJacobi(yA, pA, A.diag);
            deviceAmul(A, yA, AyA);          // yA = M^-1 pA; AyA = A yA
            deviceDotInto(rA0, AyA, s.r0Ay.data());
            bicgAlphaK<<<1,1>>>(s.rr.data(), s.r0Ay.data(), s.alpha.data(), s.negAlpha.data(), s.bd.data());   // alpha = rA0rA/(rA0.AyA), guarded
            deviceFusedSxpy(sA, rA, s.negAlpha.data(), AyA);                            // sA = rA - alpha*AyA  [fused 2->1]
            const bool check = ((nIter + 1) % K == 0) || (nIter + 1 >= maxIter);        // read |s|/|r|/bd only on check iters
            if (check)                                          // mid-iter early-exit only when we read |s|
            {
                deviceSumMagInto(sA, s.sNorm.data());
                perf.finalResidual = deviceReadScalar(s.sNorm.data()) / normFactor;
                if (converged(perf.finalResidual))
                {
                    deviceAxpyDev(s.alpha.data(), yA, psi);
                    ++nIter;
                    break;
                }
            }
            deviceJacobi(zA, sA, A.diag);
            deviceAmul(A, zA, tA);           // zA = M^-1 sA; tA = A zA
            deviceDotInto(tA, tA, s.tt.data());
            deviceDotInto(tA, sA, s.ts.data());
            omegaK<<<1,1>>>(s.ts.data(), s.tt.data(), s.omega.data(), s.negOmega.data(), s.bd.data());     // omega = tt>tiny ? ts/tt : 0 (+singularity flag)
            deviceFusedAxpy2(psi, s.alpha.data(), yA, s.omega.data(), zA);               // psi += alpha*yA + omega*zA  [fused 2->1]
            deviceFusedSxpy(rA, sA, s.negOmega.data(), tA);                             // rA = sA - omega*tA  [fused 2->1]
            if (check)
            {
                deviceSumMagInto(rA, s.rNorm.data());
                perf.finalResidual = deviceReadScalar(s.rNorm.data()) / normFactor;
                if (deviceReadScalar(s.bd.data()) != 0.0)   // OF break on singularity, batched to K
                {
                    ++nIter;
                    break;
                }
            }
            ++nIter;
        } while (nIter < maxIter && !converged(perf.finalResidual));
    }
    perf.nIterations = nIter;
    return perf;
}

// ---- distributed (multi-GPU) Jacobi-PCG ------------------------------------------------------------------
// The device counterpart of host parallelPCG: same recurrence as deviceJacobiPCG, but A*x uses the
// interface-coupled deviceParallelAmul and every reduction is global (Pstream::allReduce, tier-1).

scalar deviceParallelNormFactor(
    const DeviceLduView& A,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& b,
    const DeviceBuffer<scalar>& ones,
    label globalNCells,
    const DistributedAMI* ami)
{
    const int nC = A.nCells;
    DeviceBuffer<scalar> Apsi(nC), sumA(nC), tmp(nC), t(nC);
    deviceParallelAmul(A, halo, ifaceCoeffs, psi, Apsi, ami);     // A*psi (with interface + AMI coupling)
    deviceParallelAmul(A, halo, ifaceCoeffs, ones, sumA, ami);    // rowSum(A) = A*1 (includes interfaces + AMI)
    const scalar avgPsi = Pstream::allReduce(deviceDot(psi, ones), ReduceOp::Sum)
                        / static_cast<scalar>(globalNCells);
    deviceCopy(tmp, sumA);
    deviceScale(tmp, avgPsi);                                     // tmp = sumA*avg(psi)
    deviceCopy(t, Apsi);
    deviceAxpy(-1.0, tmp, t);
    const scalar n1 = Pstream::allReduce(deviceSumMag(t), ReduceOp::Sum);   // sum|A*psi - tmp|
    deviceCopy(t, b);
    deviceAxpy(-1.0, tmp, t);
    const scalar n2 = Pstream::allReduce(deviceSumMag(t), ReduceOp::Sum);   // sum|b - tmp|
    return n1 + n2 + 1e-20;
}

DeviceSolverPerf deviceParallelJacobiPCG(
    const DeviceLduView& A,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    const DistributedAMI* ami)
{
    const int nC = A.nCells;
    DeviceBuffer<scalar> wA(nC), rA(nC), pA(nC), Ax(nC);

    deviceParallelAmul(A, halo, ifaceCoeffs, psi, Ax, ami);      // rA = b - A*psi (+ AMI coupling if ami)
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);

    DeviceSolverPerf perf;

    // FUSED REDUCTIONS. Every reduction here is a HOST Pstream::allReduce, and deviceDot/deviceSumMag copy
    // their scalar to the host first -- so each one is a full GPU sync + blocking collective. At ~100 Krylov
    // iterations per SIMPLE step that latency, NOT the NVLink halo, is what caps the multi-GPU speedup (it is
    // a fixed cost that does not shrink as ranks are added, and at np=1 a 1-rank allReduce is nearly free, so
    // it shows up only when you parallelise). The textbook loop does THREE per iteration:
    //     wArA = allReduce(dot(wA,rA))     -> beta
    //     wApA = allReduce(dot(wA,pA))     -> alpha
    //     resid = allReduce(sumMag(rA))    -> convergence test only
    // but the residual test and the NEXT iteration's wArA both reduce over the freshly updated rA, so they
    // fuse into ONE collective of two values: 3 -> 2 per iteration, ~33% fewer round-trips. The arithmetic is
    // UNCHANGED -- same wArA/wApA/alpha/beta/residual sequence, same iteration count -- it is purely the
    // packing of the collectives. (`wA` can no longer be reused for A*pA, hence the separate ApA buffer.)
    DeviceBuffer<scalar> ApA;
    scalar red[2];
    auto fusedReduce = [&]()   // wA = M^-1 rA ; red = [ dot(wA,rA), sumMag(rA) ] in ONE collective
    {
        deviceJacobi(wA, rA, A.diag);
        red[0] = deviceDot(wA, rA);
        red[1] = deviceSumMag(rA);
        Pstream::allReduce(red, 2, ReduceOp::Sum);
    };

    fusedReduce();
    perf.initialResidual = red[1] / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    scalar wArA = red[0], wArAold = 1e300;
    int nIter = 0;
    if (!converged(perf.finalResidual))
    {
        do
        {
            if (nIter == 0) deviceCopy(pA, wA);
            else
            {
                const scalar beta = wArA / wArAold;
                deviceScale(pA, beta);
                deviceAxpy(1.0, wA, pA);
            }
            deviceParallelAmul(A, halo, ifaceCoeffs, pA, ApA, ami);   // ApA = A*pA (+ AMI coupling if ami)
            const scalar wApA  = Pstream::allReduce(deviceDot(ApA, pA), ReduceOp::Sum);
            const scalar alpha = wArA / wApA;
            deviceAxpy(alpha, pA, psi);
            deviceAxpy(-alpha, ApA, rA);
            wArAold = wArA;
            fusedReduce();                                       // next wArA AND the residual, one collective
            wArA = red[0];
            perf.finalResidual = red[1] / normFactor;
            ++nIter;
        } while (nIter < maxIter && !converged(perf.finalResidual));
    }
    perf.nIterations = nIter;
    return perf;
}

// AMG-preconditioned distributed CG: deviceParallelJacobiPCG with the point-Jacobi preconditioner (wA = rA/diag)
// replaced by a per-rank LOCAL AMG V-cycle (wA = M^-1 rA via amgVCycleApply). Everything else is IDENTICAL -- the
// halo-coupled matvec (deviceParallelAmul), the two fused GLOBAL reductions per iteration, the CG recurrence and
// the convergence test are byte-for-byte the same, so at np=1 with the same AMG this is the single-GPU AMG-PCG
// path and at np>1 it is that with the interface added to the matvec. The V-cycle is block-Jacobi (it drops the
// interface), which the outer CG corrects for; the gain is a vastly stronger preconditioner than point-Jacobi,
// so the pressure converges on graded meshes instead of hitting maxIter and diverging.
DeviceSolverPerf deviceParallelAMGPCG(
    const DeviceLduView& A,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    AMGData& amg,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const int nC = A.nCells;
    DeviceBuffer<scalar> wA(nC), rA(nC), pA(nC), Ax(nC), ApA;

    amgPrepareFP32(amg, A);   // cast the local hierarchy to FP32 once (matrices are current post-amgGalerkin) -> FP32 V-cycles
    // BRAE_PARALLEL_GRAPH: 0 off, 1 = V-cycle-only graph replay (below), 2 = WHOLE-loop conditional graph (the 2x lever).
    static const int g_parGraphLevel = std::getenv("BRAE_PARALLEL_GRAPH") ? std::atoi(std::getenv("BRAE_PARALLEL_GRAPH")) : 0;
    if (g_parGraphLevel >= 2)
    {
        const DeviceSolverPerf gp = deviceParallelAMGPCGGraph(A, amg, halo, ifaceCoeffs, b, psi, normFactor, tol, relTol, maxIter);
        if (gp.nIterations >= 0) return gp;   // whole-loop graph ran; else (CUDA<13) fall through to the direct path
    }
    const bool g_parGraph = (g_parGraphLevel == 1);

    deviceParallelAmul(A, halo, ifaceCoeffs, psi, Ax);          // rA = b - A*psi  (interface-coupled)
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);

    // ---- TWO-LEVEL preconditioner (BRAE_COARSE_SPACE != 0): aggregation coarse space + balanced Schwarz --------
    // The local AMG V-cycle alone is a ONE-LEVEL, zero-overlap block-Jacobi preconditioner: it never communicates,
    // so the condition number grows with the subdomain count and Krylov counts climb with nProcs. The optional
    // two-level path below adds an AMG-aggregation coarse space Z (~BRAE_COARSE_DIM aggregates/rank) and applies it
    // in BALANCED (hybrid) form, which removes the slow cross-subdomain mode the block-Jacobi solve leaves behind.
    // DEFAULT 0 = one-level (local AMG only) -- the shipping single-node behaviour; nonzero = balanced two-level.
    static const int coarseMode = std::getenv("BRAE_COARSE_SPACE") ? std::atoi(std::getenv("BRAE_COARSE_SPACE")) : 0;
    const int nSub = Pstream::nProcs();
    const int myP  = Pstream::myProcNo();

    // ===== AGGREGATION coarse space + BALANCED two-level =====================================================
    // Z is ~kc aggregate-indicators PER subdomain, taken from the AMG agglomeration map composed down to the first
    // level with <= BRAE_COARSE_DIM cells (coarse dim = sum_r kc_r). E = Z^T A Z is assembled by GALERKIN SCATTER
    // straight from the LDU matrix (the halo supplies the cross-rank coupling), and the nullspace of a closed
    // all-Neumann domain is handled by a mean-free saddle solve (the aggregate indicators sum to the global 1).
    if (coarseMode != 0 && nSub > 1 && amg.nLevels() > 0)
    {
        const int targetKc = std::getenv("BRAE_COARSE_DIM") ? std::atoi(std::getenv("BRAE_COARSE_DIM")) : 32;
        // compose fine -> level-(L+1) aggregation until the coarse count drops to <= targetKc
        std::vector<label> f2a(nC);
        for (int i = 0; i < nC; ++i) f2a[i] = i;
        int kcLocal = nC;
        for (int L = 0; L < amg.nLevels(); ++L)
        {
            const std::vector<label> m = amg.level[L].map.host();     // grid L -> grid L+1
            for (int i = 0; i < nC; ++i) f2a[i] = m[f2a[i]];
            kcLocal = amg.level[L].nCoarse;
            if (kcLocal <= targetKc) break;
        }
        if (kcLocal < 1) kcLocal = 1;
        // global coarse indexing: each rank fills its own slot, allReduce gives every rank the offsets
        std::vector<scalar> kcS(static_cast<std::size_t>(nSub), 0.0); kcS[myP] = kcLocal;
        Pstream::allReduce(kcS.data(), nSub, ReduceOp::Sum);
        std::vector<int> off(static_cast<std::size_t>(nSub) + 1, 0);
        for (int r = 0; r < nSub; ++r) off[r + 1] = off[r] + static_cast<int>(kcS[r] + 0.5);
        const int totalCoarse = off[nSub], myOff = off[myP];
        DeviceBuffer<label> aggDev;
        { std::vector<label> ag(nC); for (int i = 0; i < nC; ++i) ag[i] = f2a[i] + myOff; aggDev.copyFrom(ag); }
        static bool announced = false;
        if (Pstream::master() && !announced)
        { std::fprintf(stderr, "brae coarse space (balanced two-level): ~%d aggregates/rank, totalCoarse=%d\n", kcLocal, totalCoarse); announced = true; }

        DeviceBuffer<scalar> muDev(totalCoarse), tmpZ(nC);
        auto Zt = [&](const DeviceBuffer<scalar>& v)                  // Z^T v -> replicated coarse vector (len totalCoarse)
        {
            DeviceBuffer<scalar> cdev(totalCoarse);
            cudaMemset(cdev.data(), 0, static_cast<std::size_t>(totalCoarse) * sizeof(scalar));
            aggSumK<<<csBlocks(nC), CS_TPB>>>(nC, aggDev.data(), v.data(), cdev.data());
            std::vector<scalar> c = cdev.host();
            Pstream::allReduce(c.data(), totalCoarse, ReduceOp::Sum);
            return c;
        };
        auto Zmu = [&](const std::vector<scalar>& mu, DeviceBuffer<scalar>& out)   // out = Z mu (broadcast to fine)
        {
            muDev.copyFrom(mu);
            aggGatherK<<<csBlocks(nC), CS_TPB>>>(nC, aggDev.data(), muDev.data(), out.data());
        };
        // E = Z^T A Z (totalCoarse x totalCoarse, replicated) by GALERKIN SCATTER straight from the LDU matrix --
        // replaces the O(totalCoarse) collective matvecs (the dominant cost) with 3 device kernels + ONE halo
        // exchange of the aggregate ids. Diagonal + internal faces are local; interface faces scatter the -ifCoeff
        // coupling into E[agg(owner)][nbrAgg], where nbrAgg is the neighbour cell's (exchanged) global aggregate id.
        std::vector<scalar> E(static_cast<std::size_t>(totalCoarse) * totalCoarse, 0.0);
        {
            DeviceBuffer<scalar> Edev(static_cast<int>(E.size()));
            cudaMemset(Edev.data(), 0, E.size() * sizeof(scalar));
            galDiagK<<<csBlocks(nC), CS_TPB>>>(nC, aggDev.data(), A.diag, totalCoarse, Edev.data());
            if (A.nInternalFaces > 0)
                galFaceK<<<csBlocks(A.nInternalFaces), CS_TPB>>>(
                    A.nInternalFaces, A.owner, A.nei, aggDev.data(), A.upper, A.lower, totalCoarse, Edev.data());
            DeviceBuffer<scalar> aggS(nC);                              // aggregate ids as scalars, to exchange over the halo
            labelToScalarK<<<csBlocks(nC), CS_TPB>>>(nC, aggDev.data(), aggS.data());
            halo.exchange(aggS.data());                                // recvData(i) = neighbour's aggregate ids at the cut
            for (int i = 0; i < halo.nInterfaces(); ++i)
            {
                const int nIf2 = static_cast<int>(halo.size(i));
                if (nIf2 <= 0) continue;
                galIfaceK<<<csBlocks(nIf2), CS_TPB>>>(
                    nIf2, halo.faceCellsDev(i), aggDev.data(), halo.recvData(i), ifaceCoeffs[i].data(), totalCoarse, Edev.data());
            }
            E = Edev.host();
            Pstream::allReduce(E.data(), static_cast<int>(E.size()), ReduceOp::Sum);
        }
        // Is the coarse operator SINGULAR? A closed/all-Neumann domain has A 1 = 0, hence E 1 = 0 (row sums ~0),
        // and the coarse solve MUST be mean-free (saddle constraint 1^T y = 0). For a NON-singular (open, with a
        // fixedValue-p patch) system that constraint is WRONG: it removes a real coarse dof and injects a spurious
        // correction that diverges (measured: mode 4 -> pressure residual 23, mode 3 -> NaN, on open pitzDaily).
        // Detect from E's row sums vs its diagonal scale; apply the saddle constraint ONLY when singular.
        scalar csMaxRow = 0.0, csMaxDiag = 0.0;
        for (int i = 0; i < totalCoarse; ++i)
        {
            scalar rs = 0.0;
            for (int j = 0; j < totalCoarse; ++j) rs += E[static_cast<std::size_t>(i) * totalCoarse + j];
            csMaxRow  = std::max(csMaxRow,  std::fabs(rs));
            csMaxDiag = std::max(csMaxDiag, std::fabs(E[static_cast<std::size_t>(i) * totalCoarse + i]));
        }
        const bool singularE = (csMaxDiag > 0.0) && (csMaxRow < 1e-8 * csMaxDiag);
        // Factor the coarse (saddle) operator ONCE per pressure solve; Esolve then only substitutes. E is fixed
        // within a solve, so re-factoring it on every PCG iteration was O(nIter * n^3) waste -> now O(n^3 + nIter*n^2).
        const int nE = totalCoarse, mM = singularE ? nE + 1 : nE;
        std::vector<scalar> LU(static_cast<std::size_t>(mM) * mM, 0.0);
        std::vector<int>    piv(static_cast<std::size_t>(mM));
        for (int i = 0; i < nE; ++i)
            for (int j = 0; j < nE; ++j) LU[static_cast<std::size_t>(i) * mM + j] = E[static_cast<std::size_t>(i) * nE + j];
        if (singularE)
            for (int i = 0; i < nE; ++i) { LU[static_cast<std::size_t>(i) * mM + nE] = 1.0; LU[static_cast<std::size_t>(nE) * mM + i] = 1.0; }
        for (int k = 0; k < mM; ++k)   // LU factorisation with partial pivoting, in place; L stored below the diagonal
        {
            int p = k; scalar mx = std::fabs(LU[static_cast<std::size_t>(k) * mM + k]);
            for (int i = k + 1; i < mM; ++i) { const scalar v = std::fabs(LU[static_cast<std::size_t>(i) * mM + k]); if (v > mx) { mx = v; p = i; } }
            piv[k] = p;
            if (p != k) for (int j = 0; j < mM; ++j) std::swap(LU[static_cast<std::size_t>(k) * mM + j], LU[static_cast<std::size_t>(p) * mM + j]);
            const scalar d = LU[static_cast<std::size_t>(k) * mM + k];
            if (std::fabs(d) < 1e-300) continue;
            for (int i = k + 1; i < mM; ++i)
            {
                const scalar f = LU[static_cast<std::size_t>(i) * mM + k] / d;
                LU[static_cast<std::size_t>(i) * mM + k] = f;
                for (int j = k + 1; j < mM; ++j) LU[static_cast<std::size_t>(i) * mM + j] -= f * LU[static_cast<std::size_t>(k) * mM + j];
            }
        }
        auto Esolve = [&](const std::vector<scalar>& c)   // solve E y = c via the stored LU (forward + back substitution)
        {
            std::vector<scalar> y(static_cast<std::size_t>(mM), 0.0);
            for (int i = 0; i < nE; ++i) y[i] = c[i];
            for (int k = 0; k < mM; ++k) if (piv[k] != k) std::swap(y[k], y[piv[k]]);       // apply the row swaps
            for (int i = 0; i < mM; ++i) { scalar s = y[i]; for (int j = 0; j < i; ++j) s -= LU[static_cast<std::size_t>(i) * mM + j] * y[j]; y[i] = s; }   // forward (unit L)
            for (int i = mM - 1; i >= 0; --i)
            {
                scalar s = y[i];
                for (int j = i + 1; j < mM; ++j) s -= LU[static_cast<std::size_t>(i) * mM + j] * y[j];
                const scalar d = LU[static_cast<std::size_t>(i) * mM + i];
                y[i] = (std::fabs(d) < 1e-300) ? 0.0 : s / d;
            }
            return std::vector<scalar>(y.begin(), y.begin() + nE);
        };

        // ===== BALANCED (hybrid) TWO-LEVEL apply + PCG ==========================================================
        // M_bal^-1 r = Q r + (I - Q A) M_L^-1 (I - A Q) r, with Q = Z E^-1 Z^T the aggregation coarse correction and
        // M_L^-1 the local AMG V-cycle (block-Jacobi Schwarz). The balanced form maps every coarse direction to
        // eigenvalue EXACTLY 1 -- unlike the additive M_L^-1 + Q, which shifts theta -> theta+1 and over-corrects --
        // so the coarse space cleanly removes the slow cross-subdomain mode instead of fighting the local solve.
        // Measured at scale: ~2x fewer pressure Krylov iterations and a net wall-clock win despite the extra apply.
        // The preconditioner is SPD (symmetric V-cycle in a symmetric Q sandwich), so ordinary PCG stays valid.
        DeviceSolverPerf perf5;
        DeviceBuffer<scalar> zz(nC), pp(nC), Ap(nC), bAz(nC), bt(nC), by(nC), bAy(nC), bzn(nC);
        auto applyQ = [&](const DeviceBuffer<scalar>& r, DeviceBuffer<scalar>& out)   // out = Q r = Z E^-1 Z^T r
        {
            const std::vector<scalar> mu = Esolve(Zt(r)); Zmu(mu, out);
        };
        auto applyM = [&](const DeviceBuffer<scalar>& r, DeviceBuffer<scalar>& z)
        {
            applyQ(r, tmpZ);                                            // tmpZ = Q r
            deviceParallelAmul(A, halo, ifaceCoeffs, tmpZ, bAz);        // A Q r
            deviceCopy(bt, r); deviceAxpy(-1.0, bAz, bt);              // t = (I - A Q) r
            amgVCycleApply(amg, A, bt, by, g_parGraph);                // y = M_L^-1 t
            deviceParallelAmul(A, halo, ifaceCoeffs, by, bAy);         // A y
            applyQ(bAy, bzn);                                          // Q A y
            deviceCopy(z, tmpZ); deviceAxpy(1.0, by, z); deviceAxpy(-1.0, bzn, z);   // z = Q r + y - Q A y
        };
        applyM(rA, zz);                                        // rA already = b - A*psi
        deviceCopy(pp, zz);
        scalar rd[2]; rd[0] = deviceDot(rA, zz); rd[1] = deviceSumMag(rA);
        Pstream::allReduce(rd, 2, ReduceOp::Sum);
        perf5.initialResidual = rd[1] / normFactor;
        perf5.finalResidual   = perf5.initialResidual;
        auto conv5 = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf5.initialResidual); };
        scalar rzOld = rd[0]; int nit = 0;
        if (!conv5(perf5.finalResidual))
        {
            do {
                deviceParallelAmul(A, halo, ifaceCoeffs, pp, Ap);
                const scalar pAp = Pstream::allReduce(deviceDot(pp, Ap), ReduceOp::Sum);
                const scalar alpha = rzOld / pAp;
                deviceAxpy(alpha, pp, psi);
                deviceAxpy(-alpha, Ap, rA);
                applyM(rA, zz);
                scalar rd2[2]; rd2[0] = deviceDot(rA, zz); rd2[1] = deviceSumMag(rA);
                Pstream::allReduce(rd2, 2, ReduceOp::Sum);
                perf5.finalResidual = rd2[1] / normFactor;
                ++nit;
                if (conv5(perf5.finalResidual)) break;
                const scalar beta = rd2[0] / rzOld;
                deviceScale(pp, beta); deviceAxpy(1.0, zz, pp);
                rzOld = rd2[0];
            } while (nit < maxIter);
        }
        perf5.nIterations = nit;
        return perf5;
    }

    DeviceSolverPerf perf;
    scalar red[2];
    auto fusedReduce = [&]()   // wA = M^-1 rA (local AMG V-cycle) ; red = [ dot(wA,rA), sumMag(rA) ] in ONE collective
    {
        amgVCycleApply(amg, A, rA, wA, g_parGraph);
        red[0] = deviceDot(wA, rA);
        red[1] = deviceSumMag(rA);
        Pstream::allReduce(red, 2, ReduceOp::Sum);
    };

    fusedReduce();
    perf.initialResidual = red[1] / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    scalar wArA = red[0], wArAold = 1e300;
    int nIter = 0;
    if (!converged(perf.finalResidual))
    {
        do
        {
            if (nIter == 0) deviceCopy(pA, wA);
            else
            {
                const scalar beta = wArA / wArAold;
                deviceScale(pA, beta);
                deviceAxpy(1.0, wA, pA);
            }
            deviceParallelAmul(A, halo, ifaceCoeffs, pA, ApA);  // ApA = A*pA
            const scalar wApA  = Pstream::allReduce(deviceDot(ApA, pA), ReduceOp::Sum);
            const scalar alpha = wArA / wApA;
            deviceAxpy(alpha, pA, psi);
            deviceAxpy(-alpha, ApA, rA);
            wArAold = wArA;
            fusedReduce();                                      // next wArA AND the residual, one collective
            wArA = red[0];
            perf.finalResidual = red[1] / normFactor;
            ++nIter;
        } while (nIter < maxIter && !converged(perf.finalResidual));
    }
    perf.nIterations = nIter;
    return perf;
}

// Distributed Jacobi-BiCGStab (non-symmetric momentum matrix). Mirrors deviceJacobiBiCGStab's recurrence with
// the interface-coupled product and global reductions. Breakdown guards follow OF checkSingularity (VSMALL).
DeviceSolverPerf deviceParallelJacobiBiCGStab(
    const DeviceLduView& A,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter,
    int checkEvery,
    BiCGGraphCache* gcache,
    const DistributedAMI* ami)
{
    const int nC = A.nCells;
    const int K  = (checkEvery > 1) ? checkEvery : 1;            // convergence-read cadence (1 = exact per-iter)
    const cudaStream_t strm = cudaStreamPerThread;               // the halo/reduce stream: everything is stream-ordered
    // BRAE_PARALLEL_GRAPH>=3: whole-loop conditional-graph momentum solve (the momentum twin of the pressure
    // deviceParallelAMGPCGGraph). Falls through to the direct device-resident path below on CUDA<13 (sentinel -1).
    // With a cyclicAMI (`ami`) the matvec does an NVSHMEM gather + barrier that is not captured into the while-graph,
    // so the graph path is skipped (as the single-GPU path skips the V-cycle graph for AMI) -> direct path below.
    static const int g_parGraphLevel = std::getenv("BRAE_PARALLEL_GRAPH") ? std::atoi(std::getenv("BRAE_PARALLEL_GRAPH")) : 0;
    if (g_parGraphLevel >= 3 && gcache && !ami)
    {
        const DeviceSolverPerf gp =
            deviceParallelJacobiBiCGStabGraph(A, halo, ifaceCoeffs, b, psi, *gcache, normFactor, tol, relTol, maxIter);
        if (gp.nIterations >= 0) return gp;
    }
    DeviceBuffer<scalar> rA(nC), rA0(nC), pA(nC), yA(nC), AyA(nC), sA(nC), zA(nC), tA(nC), Ax(nC);

    deviceParallelAmul(A, halo, ifaceCoeffs, psi, Ax, ami);      // rA = b - A*psi (+ AMI coupling if ami)
    deviceCopy(rA, b);
    deviceAxpy(-1.0, Ax, rA);
    deviceCopy(rA0, rA);

    // DEVICE-RESIDENT distributed BiCGStab: the exact recurrence of the single-GPU deviceJacobiBiCGStab (alpha/
    // omega/beta and every dot live on the device, fed to the fused *Dev kernels by pointer), with two changes for
    // multi-GPU: A*x is deviceParallelAmul (halo-coupled), and each LOCAL dot is globalised by an ON-STREAM NVSHMEM
    // sum-reduce (DeviceReducer) instead of a blocking host MPI_Allreduce -- so the host is untouched per iteration
    // except the K-cadence convergence read. This removes the ~6 D2H + 6 MPI_Allreduce per iteration that were the
    // multi-GPU momentum-solve bottleneck. Breakdown (OF checkSingularity) is detected ON-DEVICE into `bd` from the
    // GLOBAL rr/omega (identical on every PE), read only on check iters. gdot/gsum wrap local-dot -> reduce -> copy.
    DeviceReducer& R = halo.reducer();
    auto gdot = [&](const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, DeviceBuffer<scalar>& out)
    {   // out (device scalar) = global sum of x.y across all PEs
        deviceDotInto(x, y, R.src());
        R.sumReduce(1, strm);
        deviceScalarCopy(R.dst(), out.data());
    };
    auto gsumHost = [&](const DeviceBuffer<scalar>& x)   // global |x|_1, returned to the host (the only per-check D2H)
    {
        deviceSumMagInto(x, R.src());
        R.sumReduce(1, strm);
        return deviceReadScalar(R.dst());
    };

    DeviceSolverPerf perf;
    perf.initialResidual = gsumHost(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    BiCGScalars& s = bicgScalars();
    cudaCheck(cudaMemsetAsync(s.bd.data(), 0, sizeof(scalar), strm), "pbicg bd zero");
    int nIter = 0;
    if (!converged(perf.finalResidual))
    {
        do
        {
            if (nIter > 0) deviceScalarCopy(s.rr.data(), s.rrOld.data());   // rrOld = rr
            gdot(rA0, rA, s.rr);                                            // rr = rA0.rA (global)
            bicgRhoSingK<<<1,1,0,strm>>>(s.rr.data(), s.bd.data());         // OF checkSingularity(rA0rA) -> flag
            if (nIter == 0) deviceCopy(pA, rA);
            else
            {
                bicgBetaK<<<1,1,0,strm>>>(s.rr.data(), s.rrOld.data(), s.alpha.data(), s.omega.data(), s.beta.data(), s.bd.data());
                deviceFusedBicgP(rA, pA, AyA, s.beta.data(), s.negOmega.data());   // pA = rA + beta*(pA - omega*AyA)
            }
            deviceJacobi(yA, pA, A.diag);                                   // yA = M^-1 pA (local block-Jacobi)
            deviceParallelAmul(A, halo, ifaceCoeffs, yA, AyA, ami);
            gdot(rA0, AyA, s.r0Ay);
            bicgAlphaK<<<1,1,0,strm>>>(s.rr.data(), s.r0Ay.data(), s.alpha.data(), s.negAlpha.data(), s.bd.data());
            deviceFusedSxpy(sA, rA, s.negAlpha.data(), AyA);               // sA = rA - alpha*AyA
            const bool check = ((nIter + 1) % K == 0) || (nIter + 1 >= maxIter);
            if (check)                                                     // mid-iter early exit only when we read |s|
            {
                perf.finalResidual = gsumHost(sA) / normFactor;
                if (converged(perf.finalResidual))
                {
                    deviceAxpyDev(s.alpha.data(), yA, psi);
                    ++nIter;
                    break;
                }
            }
            deviceJacobi(zA, sA, A.diag);                                  // zA = M^-1 sA
            deviceParallelAmul(A, halo, ifaceCoeffs, zA, tA, ami);
            deviceDotInto(tA, tA, R.src() + 0);                            // tt AND ts in ONE fused collective
            deviceDotInto(tA, sA, R.src() + 1);
            R.sumReduce(2, strm);
            deviceScalarCopy(R.dst() + 0, s.tt.data());
            deviceScalarCopy(R.dst() + 1, s.ts.data());
            omegaK<<<1,1,0,strm>>>(s.ts.data(), s.tt.data(), s.omega.data(), s.negOmega.data(), s.bd.data());
            deviceFusedAxpy2(psi, s.alpha.data(), yA, s.omega.data(), zA); // psi += alpha*yA + omega*zA
            deviceFusedSxpy(rA, sA, s.negOmega.data(), tA);               // rA = sA - omega*tA
            if (check)
            {
                perf.finalResidual = gsumHost(rA) / normFactor;
                if (deviceReadScalar(s.bd.data()) != 0.0) { ++nIter; break; }   // OF break on singularity (batched to K)
            }
            ++nIter;
        } while (nIter < maxIter && !converged(perf.finalResidual));
    }
    perf.nIterations = nIter;
    return perf;
}

// ---- whole-loop conditional-graph momentum BiCGStab (BRAE_PARALLEL_GRAPH=3) --------------------------------
BiCGGraphCache::~BiCGGraphCache()
{
    if (exec)  cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
}

#if CUDART_VERSION >= 13000
namespace {
__global__
void bicgScaleInvK(scalar* v, const scalar* nf)   // v = v / normFactor (device-resident residual normalize)
{
    if (threadIdx.x==0 && blockIdx.x==0) *v = *v / *nf;
}
__global__
void bicgSetCondK(                                 // drive the WHILE conditional from the on-device stop predicate
    cudaGraphConditionalHandle h,
    const scalar* res,
    scalar tol,
    const scalar* init,
    scalar relTol,
    int* iter,
    int maxIter,
    const scalar* bd)
{
    if (threadIdx.x || blockIdx.x) return;
    const int it = ++(*iter);
    const scalar fr = *res;                        // already normalized by normFactor
    const bool conv = (fr < tol) || (relTol > 0.0 && fr < relTol * (*init));
    cudaGraphSetConditional(h, (conv || (*bd != 0.0) || it >= maxIter) ? 0u : 1u);   // stop on converge/breakdown/maxIter
}
} // namespace

// The momentum twin of deviceParallelAMGPCGGraph: the entire steady-state BiCGStab iteration (interface-coupled
// matvec + on-stream NVSHMEM reductions + the guarded recurrence) captured once into a cudaGraphCondTypeWhile graph
// and replayed on-device, removing the ~15 host kernel launches/Krylov-iter the direct device-resident path still
// pays. The momentum matrix is rebuilt with fresh buffers each SIMPLE step, so the graph body references PERSISTENT
// copies (c.diagP/upperP/lowerP/ifaceP) that we memcpy-refresh here -> capture ONCE, replay every step. The direct
// path's mid-iter |s| early exit is dropped (the body must be uniform: at most < 1 extra half-iteration, same psi);
// convergence + OF breakdown are tested at the END of the body from the full-iteration residual r.
DeviceSolverPerf deviceParallelJacobiBiCGStabGraph(
    const DeviceLduView& A,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    BiCGGraphCache& c,
    scalar normFactor,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const cudaStream_t strm = cudaStreamPerThread;
    const int nC = A.nCells;
    const int nF = A.nInternalFaces;

    c.rA.resize(nC); c.rA0.resize(nC); c.pA.resize(nC); c.yA.resize(nC); c.AyA.resize(nC);
    c.sA.resize(nC); c.zA.resize(nC); c.tA.resize(nC); c.Ax.resize(nC);
    c.sNormF.resize(1); c.sInit.resize(1); c.sRes.resize(1); c.sIter.resize(1);
    cudaMemcpyAsync(c.sNormF.data(), &normFactor, sizeof(scalar), cudaMemcpyHostToDevice, strm);

    // refresh the persistent matrix copies from THIS step's momentum matrix (stable addresses -> capture once)
    c.diagP.resize(nC); c.upperP.resize(nF); c.lowerP.resize(nF);
    cudaMemcpyAsync(c.diagP.data(),  A.diag,  nC*sizeof(scalar), cudaMemcpyDeviceToDevice, strm);
    cudaMemcpyAsync(c.upperP.data(), A.upper, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, strm);
    cudaMemcpyAsync(c.lowerP.data(), A.lower, nF*sizeof(scalar), cudaMemcpyDeviceToDevice, strm);
    DeviceLduView Ap = A;
    Ap.diag = c.diagP.data(); Ap.upper = c.upperP.data(); Ap.lower = c.lowerP.data();
    c.ifaceP.resize(ifaceCoeffs.size());
    for (std::size_t i = 0; i < ifaceCoeffs.size(); ++i)
    {
        c.ifaceP[i].resize(ifaceCoeffs[i].size());
        deviceCopy(c.ifaceP[i], ifaceCoeffs[i]);
    }

    DeviceReducer& R = halo.reducer();
    auto gdot = [&](const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, scalar* dOut)
    { deviceDotInto(x, y, R.src()); R.sumReduce(1, strm); deviceScalarCopy(R.dst(), dOut); };
    auto gsum = [&](const DeviceBuffer<scalar>& x, scalar* dOut)
    { deviceSumMagInto(x, R.src()); R.sumReduce(1, strm); deviceScalarCopy(R.dst(), dOut); };

    BiCGScalars& s = bicgScalars();
    cudaMemsetAsync(s.bd.data(), 0, sizeof(scalar), strm);

    // r = b - A psi ; r0 = r ; initial residual (one host read for the converge-in-0 early-out)
    deviceParallelAmul(Ap, halo, c.ifaceP, psi, c.Ax);
    deviceCopy(c.rA, b);
    deviceAxpy(-1.0, c.Ax, c.rA);
    deviceCopy(c.rA0, c.rA);
    DeviceSolverPerf perf;
    gsum(c.rA, c.sInit.data());
    bicgScaleInvK<<<1,1,0,strm>>>(c.sInit.data(), c.sNormF.data());
    scalar initRes;
    cudaCheck(cudaMemcpyAsync(&initRes, c.sInit.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "mbicg init D2H");
    cudaStreamSynchronize(strm);
    perf.initialResidual = initRes; perf.finalResidual = initRes;
    auto convergedHost = [&](scalar fr){ return (fr < tol) || (relTol > 0.0 && fr < relTol*initRes); };
    if (convergedHost(initRes)) { perf.nIterations = 0; return perf; }

    // iteration 0 (explicit: p = r, no beta) -- one full BiCGStab step
    gdot(c.rA0, c.rA, s.rr.data());
    bicgRhoSingK<<<1,1,0,strm>>>(s.rr.data(), s.bd.data());
    deviceCopy(c.pA, c.rA);
    deviceJacobi(c.yA, c.pA, Ap.diag);
    deviceParallelAmul(Ap, halo, c.ifaceP, c.yA, c.AyA);
    gdot(c.rA0, c.AyA, s.r0Ay.data());
    bicgAlphaK<<<1,1,0,strm>>>(s.rr.data(), s.r0Ay.data(), s.alpha.data(), s.negAlpha.data(), s.bd.data());
    deviceFusedSxpy(c.sA, c.rA, s.negAlpha.data(), c.AyA);
    deviceJacobi(c.zA, c.sA, Ap.diag);
    deviceParallelAmul(Ap, halo, c.ifaceP, c.zA, c.tA);
    deviceDotInto(c.tA, c.tA, R.src() + 0);
    deviceDotInto(c.tA, c.sA, R.src() + 1);
    R.sumReduce(2, strm);
    deviceScalarCopy(R.dst() + 0, s.tt.data());
    deviceScalarCopy(R.dst() + 1, s.ts.data());
    omegaK<<<1,1,0,strm>>>(s.ts.data(), s.tt.data(), s.omega.data(), s.negOmega.data(), s.bd.data());
    deviceFusedAxpy2(psi, s.alpha.data(), c.yA, s.omega.data(), c.zA);
    deviceFusedSxpy(c.rA, c.sA, s.negOmega.data(), c.tA);
    gsum(c.rA, c.sRes.data());
    bicgScaleInvK<<<1,1,0,strm>>>(c.sRes.data(), c.sNormF.data());
    scalar res1;
    cudaCheck(cudaMemcpyAsync(&res1, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "mbicg it0 D2H");
    cudaStreamSynchronize(strm);
    if (convergedHost(res1) || maxIter <= 1) { perf.finalResidual = res1; perf.nIterations = 1; return perf; }
    cudaMemsetAsync(c.sIter.data(), 0, sizeof(int), strm);

    // WHILE body = steady-state iteration 1+ (captured once, replayed on-device)
    if (!c.exec || c.key != psi.data())
    {
        if (c.exec)  { cudaGraphExecDestroy(c.exec);  c.exec  = nullptr; }
        if (c.graph) { cudaGraphDestroy(c.graph);     c.graph = nullptr; }
        cudaCheck(cudaGraphCreate(&c.graph, 0), "mbicg graph create");
        cudaCheck(cudaGraphConditionalHandleCreate(&c.handle, c.graph, 1, cudaGraphCondAssignDefault), "mbicg cond handle");
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = c.handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t cnode;
        cudaCheck(cudaGraphAddNode(&cnode, c.graph, nullptr, nullptr, 0, &cp), "mbicg cond node");
        cudaGraph_t body = cp.conditional.phGraph_out[0];
        cudaCheck(cudaStreamBeginCaptureToGraph(strm, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal), "mbicg capture begin");
        deviceScalarCopy(s.rr.data(), s.rrOld.data());              // rrOld = rr
        gdot(c.rA0, c.rA, s.rr.data());
        bicgRhoSingK<<<1,1,0,strm>>>(s.rr.data(), s.bd.data());
        bicgBetaK<<<1,1,0,strm>>>(s.rr.data(), s.rrOld.data(), s.alpha.data(), s.omega.data(), s.beta.data(), s.bd.data());
        deviceFusedBicgP(c.rA, c.pA, c.AyA, s.beta.data(), s.negOmega.data());   // p = r + beta*(p - omega*AyA)
        deviceJacobi(c.yA, c.pA, Ap.diag);
        deviceParallelAmul(Ap, halo, c.ifaceP, c.yA, c.AyA);
        gdot(c.rA0, c.AyA, s.r0Ay.data());
        bicgAlphaK<<<1,1,0,strm>>>(s.rr.data(), s.r0Ay.data(), s.alpha.data(), s.negAlpha.data(), s.bd.data());
        deviceFusedSxpy(c.sA, c.rA, s.negAlpha.data(), c.AyA);     // s = r - alpha*AyA
        deviceJacobi(c.zA, c.sA, Ap.diag);
        deviceParallelAmul(Ap, halo, c.ifaceP, c.zA, c.tA);
        deviceDotInto(c.tA, c.tA, R.src() + 0);
        deviceDotInto(c.tA, c.sA, R.src() + 1);
        R.sumReduce(2, strm);
        deviceScalarCopy(R.dst() + 0, s.tt.data());
        deviceScalarCopy(R.dst() + 1, s.ts.data());
        omegaK<<<1,1,0,strm>>>(s.ts.data(), s.tt.data(), s.omega.data(), s.negOmega.data(), s.bd.data());
        deviceFusedAxpy2(psi, s.alpha.data(), c.yA, s.omega.data(), c.zA);       // psi += alpha*y + omega*z
        deviceFusedSxpy(c.rA, c.sA, s.negOmega.data(), c.tA);     // r = s - omega*tA
        gsum(c.rA, c.sRes.data());
        bicgScaleInvK<<<1,1,0,strm>>>(c.sRes.data(), c.sNormF.data());
        bicgSetCondK<<<1,1,0,strm>>>(c.handle, c.sRes.data(), tol, c.sInit.data(), relTol, c.sIter.data(), maxIter-1, s.bd.data());
        cudaGraph_t tmp;
        cudaCheck(cudaStreamEndCapture(strm, &tmp), "mbicg capture end");
        cudaCheck(cudaGraphInstantiate(&c.exec, c.graph, 0), "mbicg graph instantiate");
        c.key = psi.data();
    }
    cudaCheck(cudaGraphLaunch(c.exec, strm), "mbicg graph launch");
    scalar finalRes; int whileIters;
    cudaCheck(cudaMemcpyAsync(&finalRes, c.sRes.data(), sizeof(scalar), cudaMemcpyDeviceToHost, strm), "mbicg final D2H");
    cudaCheck(cudaMemcpyAsync(&whileIters, c.sIter.data(), sizeof(int), cudaMemcpyDeviceToHost, strm), "mbicg iters D2H");
    cudaStreamSynchronize(strm);
    perf.finalResidual = finalRes;
    perf.nIterations = 1 + whileIters;
    return perf;
}
#endif // CUDART_VERSION >= 13000

#if CUDART_VERSION < 13000
// CUDA < 13: WHILE conditional-graph nodes are unavailable -> sentinel (nIterations < 0); caller uses the direct path.
DeviceSolverPerf deviceParallelJacobiBiCGStabGraph(
    const DeviceLduView& A, DeviceHalo&,
    const std::vector<DeviceBuffer<scalar>>&,
    const DeviceBuffer<scalar>&, DeviceBuffer<scalar>&,
    BiCGGraphCache&, scalar, scalar, scalar, int)
{
    (void)A;
    DeviceSolverPerf p; p.nIterations = -1; return p;
}
#endif

} // namespace brae
