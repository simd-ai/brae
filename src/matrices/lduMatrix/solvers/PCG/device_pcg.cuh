#pragma once
// cf GPU offload (G2): device-resident Jacobi-preconditioned CG. Composes the G1 SpMV (deviceAmul) with
// the G0 BLAS-1 kernels; the whole Krylov loop runs on the GPU, only the scalar reductions (wArA, wApA,
// residual) return to the host each iteration. Same CG recurrence as cf's CPU pcg, with a Jacobi (diagonal)
// preconditioner in place of DIC (DIC's sweeps are sequential, a later phase). normFactor is the OF
// normalisation, passed from the host. Solves to the same converged solution as the CPU solver.
#include "cf_types.cuh"
#include "device_ldu.cuh"
#include "device_dilu.cuh"   // optional DILU preconditioner (nullptr -> Jacobi, the historical behaviour)
#include "device_buffer.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace brae {

struct DeviceSolverPerf { scalar initialResidual = 0, finalResidual = 0; int nIterations = 0; };

// Cached conditional-graph of the distributed momentum BiCGStab WHILE-body (BRAE_PARALLEL_GRAPH>=3). One cache
// per U-component, owned by the solver (Uk_[k]'s address is the stable key). The momentum matrix is REBUILT with
// fresh buffers every SIMPLE step (unlike the stable pressure matrix), so the graph body references PERSISTENT
// copies here (diagP/upperP/lowerP/ifaceP) that the caller memcpy-refreshes each step -> the graph is captured
// ONCE and replayed, no per-step re-instantiation. Holds the graph-referenced Krylov vectors + control scalars.
struct BiCGGraphCache {
    cudaGraphExec_t exec = nullptr; cudaGraph_t graph = nullptr;
    cudaGraphConditionalHandle handle{}; const void* key = nullptr;
    DeviceBuffer<scalar> rA, rA0, pA, yA, AyA, sA, zA, tA, Ax;      // persistent Krylov vectors
    DeviceBuffer<scalar> diagP, upperP, lowerP;                     // persistent matrix copies (refreshed per step)
    std::vector<DeviceBuffer<scalar>> ifaceP;                       // persistent interface-coeff copies
    DeviceBuffer<scalar> sNormF, sInit, sRes; DeviceBuffer<int> sIter;   // control scalars
    ~BiCGGraphCache();
};

// OpenFOAM's lduMatrix::solver::normFactor: sum(|A*psi - sumA*avg(psi)| + |b - sumA*avg(psi)|) + small,
// where sumA = rowSum(A) = A*ones. Pass this as the solver's normFactor so the reported initial residual
// (and the tolerance test) match OF's residualControl convention. ones is a cached unit vector (length nCells).
scalar deviceNormFactor(const DeviceLduView& A, const DeviceBuffer<scalar>& psi,
                        const DeviceBuffer<scalar>& b, const DeviceBuffer<scalar>& ones);
// THE NORMFACTOR STAYS ON THE DEVICE (item 66). Every solver divides by it on the device already (the graph
// loops upload it and scale there), so reading it to the host was a queue drain per solve for a number the
// host never used -- 15 of the ~36 drains per T3A iteration. This writes it into `dNf` (one scalar) and the
// pointer-taking solver entries below consume it in place. Same kernels, same operations: the value and every
// quotient are bit-identical to the host-read path, which BRAE_NORMFACTOR_HOST=1 restores for the gate.
void deviceNormFactorInto(const DeviceLduView& A, const DeviceBuffer<scalar>& psi,
                          const DeviceBuffer<scalar>& b, const DeviceBuffer<scalar>& ones,
                          DeviceBuffer<scalar>& dNf);
bool normFactorOnHost();                    // BRAE_NORMFACTOR_HOST=1: the pointer entries read it and take the scalar path
void announceNormFactorMode();              // once per process, from every pointer-taking solver entry


DeviceSolverPerf deviceJacobiPCG(const DeviceLduView& A, const DeviceBuffer<scalar>& b,
                                 DeviceBuffer<scalar>& psi, scalar normFactor,
                                 scalar tol, scalar relTol, int maxIter, int minIter = 0);

// Jacobi-preconditioned BiCGStab for the NON-symmetric momentum matrix (upwind convection -> upper!=lower).
// Same recurrence as brae::pbicgstab; device-resident.
//
// `precon` selects the preconditioner: nullptr keeps Jacobi (w = r/diag), a DeviceDilu runs OpenFOAM's
// DILU. That is not a tuning knob -- both reach the requested relTol, but they leave the residual error
// in different modes, and on a boundary-layer mesh Jacobi leaves it in the wall-normal direction where
// the PIMPLE outer loop amplifies it (see device_dilu.cuh).
// checkEvery: read the |s|/|r| convergence norms (the 2 of 4 D2H reads/iter that aren't breakdown guards) only every
// K iters -> batched convergence, like deviceAMGPCG's checkEvery. Breakdown guards (rA0rA, omega) stay per-iter for
// safety. K=1 = exact (bit-identical); K>1 overshoots convergence by < K iters. Default 1.
DeviceSolverPerf deviceJacobiBiCGStab(const DeviceLduView& A, const DeviceBuffer<scalar>& b,
                                      DeviceBuffer<scalar>& psi, scalar normFactor,
                                      scalar tol, scalar relTol, int maxIter, int checkEvery = 1, int minIter = 0,
                                      const DeviceDilu* precon = nullptr);
// the same solve with the normFactor on the device (item 66)
DeviceSolverPerf deviceJacobiBiCGStab(const DeviceLduView& A, const DeviceBuffer<scalar>& b,
                                      DeviceBuffer<scalar>& psi, const scalar* dNormFactor,
                                      scalar tol, scalar relTol, int maxIter, int checkEvery = 1, int minIter = 0,
                                      const DeviceDilu* precon = nullptr);


class DeviceHalo;   // forward (parallel/pstream/device_halo.cuh)

// Distributed (multi-GPU) counterparts of the two functions above: A*x via deviceParallelAmul (halo exchange +
// interface coupling), and every dot / sumMag becomes a GLOBAL reduction (tier-1: Pstream::allReduce on the
// host-side scalar). The Jacobi preconditioner stays LOCAL per rank (interfaces dropped), exactly as host
// parallelPCG. ifaceCoeffs[i] holds interface i's boundary coefficients, same order as `halo`'s interfaces.
scalar deviceParallelNormFactor(
    const DeviceLduView& A,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<scalar>>& ifaceCoeffs,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& b,
    const DeviceBuffer<scalar>& ones,
    label globalNCells,
    const DistributedAMI* ami = nullptr);

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
    const DistributedAMI* ami = nullptr);

// Distributed Jacobi-BiCGStab for the NON-symmetric momentum matrix (upwind convection -> upper != lower):
// the device counterpart of host parallelPBiCGStab, and the distributed twin of deviceJacobiBiCGStab. Same
// recurrence, with A*x via deviceParallelAmul and every dot / sumMag a GLOBAL reduction (tier-1).
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
    int checkEvery = 1,          // convergence-read cadence: read |r| to the host every K iters (1 = exact per-iter)
    BiCGGraphCache* gcache = nullptr,    // if BRAE_PARALLEL_GRAPH>=3 and non-null, use the whole-loop graph path
    const DistributedAMI* ami = nullptr);   // optional cyclicAMI: every matvec carries the AMI coupling (graph off)

// Whole-loop conditional-graph momentum BiCGStab (BRAE_PARALLEL_GRAPH=3): the entire steady-state BiCGStab WHILE
// body -- interface-coupled matvec + on-stream NVSHMEM reductions + the recurrence -- captured once into a
// cudaGraphCondTypeWhile graph and replayed on-device, removing the ~15 host kernel launches/Krylov-iter that the
// direct device-resident path still pays. Mirrors deviceParallelAMGPCGGraph. Requires CUDA >= 13 (WHILE nodes) and
// 1 PE/GPU (the on-stream reduce must be capturable); returns nIterations = -1 otherwise so the caller falls back.
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
    int maxIter);

struct AMGData;   // fwd (device_amg.cuh); passed by ref so this header need not include the AMG hierarchy

// AMG-preconditioned distributed CG: identical recurrence to deviceParallelJacobiPCG (halo-coupled matvec +
// GLOBAL fused reductions), but the preconditioner z=M^-1 r is a per-rank LOCAL AMG V-cycle (amgVCycleApply)
// instead of point-Jacobi. This is the block-Jacobi/additive-Schwarz AMG preconditioner: it converges the
// pressure on stiff/graded meshes where point-Jacobi caps its iteration count and the SIMPLE loop diverges.
// `amg` must be built on A's LOCAL internal addressing (buildAMG) and current for this step (amgGalerkin).
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
    int maxIter);

} // namespace brae
