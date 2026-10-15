// Coarse-grid solvers for the AMG V-cycle: the single-launch coarsest-grid solves (cluster-fused Jacobi,
// single-block Jacobi, single-block Jacobi-PCG) + deviceCoarseFitsCluster. Split out of device_amg.cu; the
// kernels reuse the shared inline device helpers (safeDiag/blockDot/warpReduceSum) via device_amg_detail.cuh,
// so each keeps its own inlined copy (no cross-TU inlining loss). deviceCoarseJacobiLoop lives here too (it
// launches the V-cycle's smoothT<> from amg_kernels.cuh + deviceAmul).
#include "device_amg.cuh"          // public deviceCoarse* decls + DeviceLduView/DeviceBuffer
#include "device_amg_detail.cuh"   // safeDiag/blockDot/warpReduceSum + TPB/OMEGA/CCL/... constants
#include "amg_kernels.cuh"       // smoothT<> for deviceCoarseJacobiLoop
#include "device_amg_coarse.cuh"   // deviceCoarsePCG/JacobiSingleBlock internal decls (match defs here)
#include "device_ldu.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mutex>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

namespace cg = cooperative_groups;

namespace brae {

namespace {
// All nSweeps of coarse weighted-Jacobi in ONE cluster kernel. The coarse vector is held in distributed shared
// memory: block 'rank' owns cells [rank*cpb, rank*cpb+myCount); a cell's SpMV reads neighbour values from the
// owning sibling block via cluster.map_shared_rank. Ping-pong (rd/wr) gives proper Jacobi (all-old-x per sweep);
// cluster.sync() between sweeps makes every block's update visible. Same arithmetic/order as deviceAmul+smoothK.
__global__
void coarseJacobiFusedKernel(
    int nC,
    int cpb,
    int nSweeps,
    scalar omega,
    const scalar* __restrict__ rc,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    scalar* __restrict__ xc)
{
#if __CUDA_ARCH__ >= 900   // clusters+DSM exist only on sm_90+. Pre-Hopper compiles this as a no-op; it is never
                           // launched there because deviceCoarseFitsCluster() returns false without cluster support,
                           // so the coarsest-solve dispatch falls through to the global-memory Jacobi fallback.
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank();
    extern __shared__ scalar sh[];
    scalar* buf0 = sh;
    scalar* buf1 = sh + cpb;
    const int base = rank * cpb;
    const int myCount = max(0, min(cpb, nC - base));
    for (int i = threadIdx.x; i < cpb; i += blockDim.x)
    {
        buf0[i] = (i < myCount) ? xc[base + i] : 0.0;
        buf1[i] = 0.0;
    }
    __syncthreads();
    cluster.sync();                                          // initial guess visible cluster-wide
    scalar* rd = buf0;
    scalar* wr = buf1;
    for (int s = 0; s < nSweeps; ++s)
    {
        for (int i = threadIdx.x; i < myCount; i += blockDim.x)
        {
            const int c = base + i;
            scalar Ax = diag[c] * rd[i];
            for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)   // faces owned by c
            {
                const int g = nei[f];
                const scalar* rem = cluster.map_shared_rank(rd, g / cpb);
                Ax += upper[f] * rem[g % cpb];
            }
            for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)   // faces neighbouring c
            {
                const int f = losort[k], g = owner[f];
                const scalar* rem = cluster.map_shared_rank(rd, g / cpb);
                Ax += lower[f] * rem[g % cpb];
            }
            wr[i] = rd[i] + omega * (rc[c] - Ax) / safeDiag(diag[c]);
        }
        cluster.sync();                                       // all writes (wr) + reads (rd) done -> swap
        scalar* t = rd;
        rd = wr;
        wr = t;
    }
    for (int i = threadIdx.x; i < myCount; i += blockDim.x)
        xc[base + i] = rd[i];
#endif
}

// SINGLE-BLOCK coarsest Jacobi: the whole (tiny) coarse vector lives in shared memory, nSweeps of ping-pong Jacobi
// with cheap __syncthreads() between them (NOT cluster.sync, the latter's per-sweep cost dominates at the high
// sweep counts a well-solved coarsest needs). Same arithmetic as deviceAmul+smoothK. Used for nC <= SB_MAX.
__global__
void coarseJacobiSingleBlockKernel(
    int nC,
    int nSweeps,
    scalar omega,
    const scalar* __restrict__ rc,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    scalar* __restrict__ xc)
{
    extern __shared__ scalar sh[];
    scalar* rd = sh;
    scalar* wr = sh + nC;
    for (int i = threadIdx.x; i < nC; i += blockDim.x)
        rd[i] = xc[i];
    __syncthreads();
    for (int s = 0; s < nSweeps; ++s)
    {
        for (int c = threadIdx.x; c < nC; c += blockDim.x)
        {
            scalar Ax = diag[c] * rd[c];
            for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
                Ax += upper[f] * rd[nei[f]];
            for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
            {
                const int f = losort[k];
                Ax += lower[f] * rd[owner[f]];
            }
            wr[c] = rd[c] + omega * (rc[c] - Ax) / safeDiag(diag[c]);
        }
        __syncthreads();
        scalar* t = rd;
        rd = wr;
        wr = t;
    }
    for (int i = threadIdx.x; i < nC; i += blockDim.x)
        xc[i] = rd[i];
}

// Single-block coarsest solve by Jacobi-preconditioned CG (z = r/diag). CG converges as O(sqrt(kappa)) vs Jacobi's
// O(kappa), so a handful of iterations solve the tiny coarsest as well as hundreds of Jacobi sweeps. The whole CG
// (vectors + dot reductions) lives in shared memory in one block -> no host sync, capturable in the V-cycle graph.
// alpha/beta guarded so an early-converged solve can't NaN.
//
// It stops on a RELATIVE RESIDUAL, not a count, for the reason its asymmetric twin does and which was
// measured there (see COARSE_REL_TOL in device_amg_detail.cuh): a fixed count leaves an approximation
// whose error depends on the right-hand side, so M^-1 is no longer a fixed linear operator and the outer
// Krylov method loses the premise it is built on. That the twin was converged and this one still counted
// was the remainder of item 80. `nIters` is the CAP.
__global__
void coarsePCGKernel(
    int nC,
    int nIters,
    const scalar* __restrict__ rc,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    scalar* __restrict__ xc)
{
    extern __shared__ scalar sh[];
    scalar* x = sh;
    scalar* r = sh + nC;
    scalar* p = sh + 2*nC;
    scalar* Ap = sh + 3*nC;
    scalar* z = sh + 4*nC;
    scalar* red = sh + 5*nC;                                     // TPB scratch for the block reduction
    const int tid = threadIdx.x;
    for (int i = tid; i < nC; i += blockDim.x)   // x0=0, M^-1 r
    {
        x[i]=0.0;
        r[i]=rc[i];
        z[i]=r[i]/safeDiag(diag[i]);
        p[i]=z[i];
    }
    __syncthreads();
    scalar rz = blockDot(r, z, nC, red);                         // r . z
    // The same measure the asymmetric twin stops on -- r.r, not CG's own r.z -- so the two coarsest
    // solves return an inverse of the same accuracy and the V-cycle behaves the same on either operator.
    const scalar stop = COARSE_REL_TOL*COARSE_REL_TOL*blockDot(r, r, nC, red);
    for (int it = 0; it < nIters; ++it)
    {
        if (it > 0 && blockDot(r, r, nC, red) <= stop)
        {
            break;
        }
        for (int c = tid; c < nC; c += blockDim.x)   // Ap = A p
        {
            scalar a = diag[c]*p[c];
            for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
                a += upper[f]*p[nei[f]];
            for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
            {
                const int f = losort[k];
                a += lower[f]*p[owner[f]];
            }
            Ap[c] = a;
        }
        __syncthreads();
        const scalar pAp = blockDot(p, Ap, nC, red);
        const scalar alpha = (pAp > 1e-300 || pAp < -1e-300) ? rz/pAp : 0.0;
        for (int i = tid; i < nC; i += blockDim.x)
        {
            x[i]+=alpha*p[i];
            r[i]-=alpha*Ap[i];
            z[i]=r[i]/safeDiag(diag[i]);
        }
        __syncthreads();
        const scalar rznew = blockDot(r, z, nC, red);
        const scalar beta = (rz > 1e-300 || rz < -1e-300) ? rznew/rz : 0.0;
        rz = rznew;
        for (int i = tid; i < nC; i += blockDim.x)
            p[i] = z[i] + beta*p[i];
        __syncthreads();
    }
    for (int i = tid; i < nC; i += blockDim.x)
        xc[i] = x[i];
}

// Single-block coarsest solve by Jacobi-preconditioned BiCGStab: the ASYMMETRIC twin of coarsePCGKernel.
// CG is not a solve on an operator with upper != lower -- its alpha = (r.z)/(p.Ap) is only a step length
// when p.Ap is the A-norm of p, i.e. when A is symmetric -- and its guards (coarsePCGKernel's alpha/beta
// tests) make it return finite garbage rather than fail. OpenFOAM's own coarsest level for an asymmetric
// matrix is a PBiCGStab (GAMGSolver.C:299-328), so this is that recurrence (PBiCGStab.C:160-249), in one
// block, with the same fixed iteration count and the same divide-by-zero guards as coarsePCGKernel.
//
// FIVE shared vectors, exactly coarsePCGKernel's footprint (5*nC + 32 doubles), so SB_CG_MAX needs no
// change and no >48KB shared-memory opt-in (which is a non-stream runtime call and this kernel is
// reachable from a stream-captured V-cycle). Two of BiCGStab's nine vectors are elided rather than
// stored: sA overwrites rA (OF never reads the old rA once sA exists -- PBiCGStab.C:209 is the last use,
// and pA's next update at :192 reads the NEW rA), and the preconditioned yA / zA are re-derived where
// they are used, which for the Jacobi preconditioner is one division (y = p/diag, z = s/diag).
__global__
void coarseBiCGStabKernel(
    int nC,
    int nIters,
    const scalar* __restrict__ rc,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    scalar* __restrict__ xc)
{
    extern __shared__ scalar sh[];
    scalar* r = sh;                                              // rA, and sA after the half step
    scalar* r0 = sh + nC;
    scalar* p = sh + 2*nC;
    scalar* Ay = sh + 3*nC;
    scalar* t = sh + 4*nC;
    scalar* red = sh + 5*nC;                                     // TPB scratch for the block reduction
    const int tid = threadIdx.x;
    for (int i = tid; i < nC; i += blockDim.x)   // x0 = 0 -> rA = b, and OF's iteration-0 pA = rA
    {
        xc[i] = 0.0;
        r[i] = rc[i];
        r0[i] = rc[i];
        p[i] = rc[i];
    }
    __syncthreads();
    // The coarsest solve must be CONVERGED, not merely cheap: a fixed count leaves an approximation
    // whose error depends on the right-hand side, so the V-cycle stops being a fixed linear operator
    // and the OUTER Krylov method breaks on it. Measured on validation/sbMatched (112k, transonic p,
    // tolerance 1e-12) with this kernel at a fixed 16 iterations: the outer BiCGStab took 187, then
    // 1000 (its cap, unconverged), then 89 iterations on three consecutive outer iterations. At 64 it
    // takes 50, 50, 43, and at 256 the same 50, 50, 43 -- i.e. 64 is where the coarsest converges and
    // the outer method becomes well behaved. So this iterates to a RELATIVE RESIDUAL, as OpenFOAM's
    // own coarsest solver does (GAMGSolver.C:299-328 constructs a PBiCGStab with the GAMG dict's
    // tolerance), and `nIters` is now the CAP rather than the count.
    const scalar tol0 = blockDot(r, r, nC, red);
    const scalar stop = COARSE_REL_TOL*COARSE_REL_TOL*tol0;
    scalar alpha = 0.0, omega = 0.0, rr = 0.0;
    for (int it = 0; it < nIters; ++it)
    {
        if (it > 0)
        {
            // r is the current residual (the recurrence keeps it): stop once it is small enough that
            // the correction this returns is the coarse inverse to COARSE_REL_TOL.
            const scalar rn = blockDot(r, r, nC, red);
            if (rn <= stop)
            {
                break;
            }
        }
        const scalar rrOld = rr;
        rr = blockDot(r0, r, nC, red);                           // rA0rA (PBiCGStab.C:166)
        if (it > 0)
        {
            // beta = (rA0rA/rA0rAold)*(alpha/omega) (PBiCGStab.C:190). OF instead BREAKS the loop when
            // mag(rA0rA) or mag(omega) is singular; a fixed-count kernel cannot break, so a singular
            // denominator restarts the direction (beta = 0, pA = rA) rather than producing a NaN.
            const bool ok = (rrOld > 1e-300 || rrOld < -1e-300) && (omega > 1e-300 || omega < -1e-300);
            const scalar beta = ok ? (rr/rrOld)*(alpha/omega) : 0.0;
            for (int c = tid; c < nC; c += blockDim.x)
                p[c] = r[c] + beta*(p[c] - omega*Ay[c]);
            __syncthreads();
        }
        for (int c = tid; c < nC; c += blockDim.x)   // AyA = A yA, yA = M^-1 pA (PBiCGStab.C:198-201)
        {
            scalar a = diag[c]*(p[c]/safeDiag(diag[c]));
            for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
            {
                const int g = nei[f];
                a += upper[f]*(p[g]/safeDiag(diag[g]));
            }
            for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
            {
                const int f = losort[k], g = owner[f];
                a += lower[f]*(p[g]/safeDiag(diag[g]));
            }
            Ay[c] = a;
        }
        __syncthreads();
        const scalar r0Ay = blockDot(r0, Ay, nC, red);           // rA0AyA (PBiCGStab.C:203)
        alpha = (r0Ay > 1e-300 || r0Ay < -1e-300) ? rr/r0Ay : 0.0;
        for (int c = tid; c < nC; c += blockDim.x)
            r[c] -= alpha*Ay[c];                                 // sA = rA - alpha*AyA (PBiCGStab.C:209), in rA's slot
        __syncthreads();
        for (int c = tid; c < nC; c += blockDim.x)   // tA = A zA, zA = M^-1 sA (PBiCGStab.C:232-235)
        {
            scalar a = diag[c]*(r[c]/safeDiag(diag[c]));
            for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
            {
                const int g = nei[f];
                a += upper[f]*(r[g]/safeDiag(diag[g]));
            }
            for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
            {
                const int f = losort[k], g = owner[f];
                a += lower[f]*(r[g]/safeDiag(diag[g]));
            }
            t[c] = a;
        }
        __syncthreads();
        const scalar tt = blockDot(t, t, nC, red);               // tAtA (PBiCGStab.C:237)
        const scalar ts = blockDot(t, r, nC, red);               // gSumProd(tA, sA) (PBiCGStab.C:241)
        omega = (tt > 1e-300) ? ts/tt : 0.0;
        for (int c = tid; c < nC; c += blockDim.x)
        {
            const scalar d = safeDiag(diag[c]);
            xc[c] += alpha*(p[c]/d) + omega*(r[c]/d);            // psi += alpha*yA + omega*zA (PBiCGStab.C:246)
            r[c] -= omega*t[c];                                  // rA = sA - omega*tA (PBiCGStab.C:247)
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------------
// DIRECT COARSEST SOLVE: dense LU with partial pivoting (device_amg_detail.cuh, DENSE_COARSE_MAX).
// OpenFOAM's own `directSolveCoarsest` alternative at this level (GAMGSolver.C:266-278 -> LUscalarMatrix).
// Factorisation runs once per Galerkin update (amgGalerkin), the substitutions once per V-cycle.
// ---------------------------------------------------------------------------------------------------

// A_c -> P A_c = L U, in ONE block, lu[] row-major (n <= DENSE_COARSE_MAX, so n*n doubles is at most
// 72 KB and the working copy fits in shared). Right-looking, one k at a time: the pivot search and
// the row swap are serial in k, the multipliers and the trailing rank-1 update are spread over the block.
__global__
void coarseLUFactorKernel(
    int n,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    scalar* __restrict__ lu,
    int* __restrict__ piv,
    bool inShared,
    const scalar* __restrict__ extra)
{
    const int tid = threadIdx.x;
    __shared__ scalar shPiv;      // the chosen pivot's value, and the scale below
    __shared__ int shRow;         // the chosen pivot's row
    __shared__ scalar shScale;    // max|diag| of this level: the floor DENSE_LU_EPS is relative to
    // The factorisation is n sequential k-steps of very little arithmetic each, so it is latency
    // bound, not bandwidth bound: run it in SHARED memory whenever n*n doubles fit (the caller sets
    // inShared and the launch's dynamic size together) and write the factors out once at the end.
    // Measured on squareBend at 305,760 cells, whose coarsest level is 64 cells: 336 us per outer
    // iteration through global memory, 31 us through shared.
    extern __shared__ scalar sh[];
    scalar* __restrict__ A = inShared ? sh : lu;
    for (int i = tid; i < n*n; i += blockDim.x) A[i] = 0.0;
    __syncthreads();
    // Dense fill from the LDU addressing, one row per thread. lduMatrix::Amul is
    // Apsi[owner] += upper[f]*psi[nei], Apsi[nei] += lower[f]*psi[owner], so row c holds upper[f] in
    // column nei[f] for the faces c owns and lower[f] in column owner[f] for the faces c neighbours --
    // exactly the two gathers coarseBiCGStabKernel's SpMV runs above.
    for (int c = tid; c < n; c += blockDim.x)
    {
        A[c*n + c] = diag[c];
        for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f) A[c*n + nei[f]] = upper[f];
        for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
        {
            const int f = losort[k];
            A[c*n + owner[f]] = lower[f];
        }
    }
    __syncthreads();
    // ...and the entries the LDU form does not hold: a coupled pair's, on this grid (AMGPair::dense)
    if (extra)
    {
        for (int i = tid; i < n*n; i += blockDim.x) A[i] += extra[i];
        __syncthreads();
    }
    if (tid == 0)
    {
        scalar m = 0.0;
        for (int c = 0; c < n; ++c) m = fmax(m, fabs(diag[c]));
        shScale = (m > 0.0) ? m : 1.0;
    }
    __syncthreads();
    for (int k = 0; k < n; ++k)
    {
        if (tid < 32)   // partial pivot: the largest |a_ik| at or below the diagonal, one warp
        {
            // One warp rather than one thread: a serial scan of column k costs a shared-memory
            // latency per row, and there are n of these steps, which is where the whole factorisation
            // was spending its time (measured: 246 us per outer iteration at n = 64, 63 us with this).
            scalar bv = -1.0;
            int bi = k;
            for (int i = k + tid; i < n; i += 32)
            {
                const scalar v = fabs(A[i*n + k]);
                if (v > bv) { bv = v; bi = i; }
            }
            for (int off = 16; off > 0; off >>= 1)
            {
                const scalar ov = __shfl_down_sync(0xffffffffu, bv, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (ov > bv) { bv = ov; bi = oi; }
            }
            if (tid == 0)
            {
                shRow = bi;
                piv[k] = bi;
            }
        }
        __syncthreads();
        if (shRow != k)
        {
            for (int j = tid; j < n; j += blockDim.x)
            {
                const scalar t = A[k*n + j];
                A[k*n + j] = A[shRow*n + j];
                A[shRow*n + j] = t;
            }
            __syncthreads();
        }
        if (tid == 0)
        {
            // A singular level (an all-Neumann pressure whose reference cell never reached this grid)
            // leaves a zero pivot here. Dividing by it would put Inf into the correction and take the
            // outer solve to maxIter on garbage, so the pivot is floored at DENSE_LU_EPS*max|diag|
            // instead: the factorisation stays a FIXED linear operator (which is all the outer Krylov
            // method needs of a preconditioner), it just stops being the exact inverse in that
            // direction. There is no fallback to the iterative twin because there is nothing to fall
            // back to -- the same singular direction stalls it as well.
            scalar d = A[k*n + k];
            const scalar floor_ = DENSE_LU_EPS*shScale;
            if (!(fabs(d) > floor_)) d = (d < 0.0) ? -floor_ : floor_;
            A[k*n + k] = d;
            shPiv = d;
        }
        __syncthreads();
        const scalar rp = 1.0/shPiv;
        for (int i = k+1+tid; i < n; i += blockDim.x) A[i*n + k] *= rp;   // multipliers, kept in L's slot
        __syncthreads();
        const int m = n - k - 1;
        for (int idx = tid; idx < m*m; idx += blockDim.x)                  // trailing rank-1 update
        {
            const int i = k + 1 + idx/m;
            const int j = k + 1 + idx%m;
            A[i*n + j] -= A[i*n + k]*A[k*n + j];
        }
        __syncthreads();
    }
    if (inShared) for (int i = tid; i < n*n; i += blockDim.x) lu[i] = A[i];
}

// x = U^-1 L^-1 P b, in ONE block. Column-oriented substitutions (the row-oriented form is serial in i);
// 2n barriers at n <= 256 is a few microseconds against the 706 us the iterative coarsest solve measured
// on squareBend at 305,760 cells.
__global__
void coarseLUSolveKernel(
    int n,
    const scalar* __restrict__ lu,
    const int* __restrict__ piv,
    const scalar* __restrict__ rc,
    scalar* __restrict__ xc,
    bool inShared)
{
    // sh holds y (n), and the factors too when they fit: both substitutions are n sequential steps
    // whose scalar reads sit on the critical path, so reading them from global memory costs a device
    // latency per step (measured: 50 us per call at n = 64, 9 us with the factors in shared).
    extern __shared__ scalar sh[];
    scalar* y = sh;
    const scalar* __restrict__ A = inShared ? sh + n : lu;
    const int tid = threadIdx.x;
    if (inShared)
    {
        for (int i = tid; i < n*n; i += blockDim.x) sh[n + i] = lu[i];
        __syncthreads();
    }
    for (int i = tid; i < n; i += blockDim.x) y[i] = rc[i];
    __syncthreads();
    if (tid == 0)                                   // apply the row interchanges in factorisation order
    {
        for (int k = 0; k < n; ++k)
        {
            const int r = piv[k];
            if (r != k) { const scalar t = y[k]; y[k] = y[r]; y[r] = t; }
        }
    }
    __syncthreads();
    for (int k = 0; k < n; ++k)                     // forward: L has a unit diagonal
    {
        const scalar yk = y[k];
        for (int i = k+1+tid; i < n; i += blockDim.x) y[i] -= A[i*n + k]*yk;
        __syncthreads();
    }
    for (int k = n-1; k >= 0; --k)                  // backward
    {
        if (tid == 0) y[k] /= A[k*n + k];
        __syncthreads();
        const scalar yk = y[k];
        for (int i = tid; i < k; i += blockDim.x) y[i] -= A[i*n + k]*yk;
        __syncthreads();
    }
    for (int i = tid; i < n; i += blockDim.x) xc[i] = y[i];
}

// ---------------------------------------------------------------------------------------------------
// THE COARSEST SOLVE'S ALTERNATIVES (BRAE_AMG_COARSE_SOLVE; branch perf/amg-coarse-solve, an experiment).
// Each is prepared from the factorisation amgGalerkin makes, outside every capture, and applied inside the
// V-cycle: `tri` the same LU applied by one warp, `inverse` A^-1 formed once and applied as a product,
// `cholesky` L L^T where the level is symmetric and its pivots positive, the LU otherwise.
// ---------------------------------------------------------------------------------------------------

// rows a lane holds: the warp kernels below take n <= 32*TRI_ROWS
constexpr int TRI_ROWS = 4;
// Cholesky is taken only on a level symmetric to this, relative to max|diag| (a Galerkin product sums its
// upper and lower entries in different orders, so a symmetric level is symmetric to a rounding, not exactly)
constexpr scalar CHOL_SYM_TOL = 1e-12;
// and only with every pivot above this, relative to max|diag|: a singular level (an all-Neumann pressure)
// leaves a pivot of a rounding, and keeps the LU's floored pivot (DENSE_LU_EPS) by falling back to it
constexpr scalar CHOL_PIVOT_MIN = 1e-10;

// lu (row-major, unit L and U in place) -> fc column-major, fc[k*n + i] = lu[i*n + k], so that a warp reading
// column k of a factor reads consecutive words; and the row interchanges composed into one gather,
// y[i] = b[perm[i]]: the swaps of coarseLUSolveKernel applied to the positions, in factorisation order
__global__
void coarseLUColumnMajorKernel(
    int n,
    const scalar* __restrict__ lu,
    const int* __restrict__ piv,
    scalar* __restrict__ fc,
    int* __restrict__ perm)
{
    for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < n*n; idx += gridDim.x*blockDim.x)
    {
        const int i = idx / n;
        const int k = idx % n;
        fc[k*n + i] = lu[idx];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < n; ++i) perm[i] = i;
        for (int k = 0; k < n; ++k)
        {
            const int r = piv[k];
            if (r != k)
            {
                const int t = perm[k];
                perm[k] = perm[r];
                perm[r] = t;
            }
        }
    }
}

// row i lives in lane i % 32, slot i / 32 (a constant index into the slots, so they stay in registers)
__device__ __forceinline__
scalar warpRow(
    const scalar (&yr)[TRI_ROWS],
    int k)
{
    const int s = k >> 5;
    const scalar v = (s == 0) ? yr[0] : (s == 1) ? yr[1] : (s == 2) ? yr[2] : yr[3];
    return __shfl_sync(0xffffffffu, v, k & 31);
}

__device__ __forceinline__
void warpDivideRow(
    scalar (&yr)[TRI_ROWS],
    int k,
    int lane,
    scalar d)
{
    if ((k & 31) != lane) return;
    const int s = k >> 5;
    if (s == 0) yr[0] /= d;
    else if (s == 1) yr[1] /= d;
    else if (s == 2) yr[2] /= d;
    else yr[3] /= d;
}

// coarseLUSolveKernel's two substitutions, row for row and step for step (y_i -= F_ik y_k in k order, the
// backward step's division at its own row first), by one warp: the same arithmetic in the same order is the
// same bits. F is column-major. unitLower: L's diagonal is 1 and not stored (the LU); otherwise the forward
// step divides by F_kk first (Cholesky's L).
__device__ __forceinline__
void warpSubstitute(
    int n,
    const scalar* __restrict__ F,
    bool unitLower,
    scalar (&yr)[TRI_ROWS],
    int lane)
{
    for (int k = 0; k < n; ++k)
    {
        if (!unitLower) warpDivideRow(yr, k, lane, F[k*n + k]);
        const scalar yk = warpRow(yr, k);
        #pragma unroll
        for (int s = 0; s < TRI_ROWS; ++s)
        {
            const int i = lane + 32*s;
            if (i > k && i < n) yr[s] -= F[k*n + i]*yk;
        }
    }
    for (int k = n - 1; k >= 0; --k)
    {
        warpDivideRow(yr, k, lane, F[k*n + k]);
        const scalar yk = warpRow(yr, k);
        #pragma unroll
        for (int s = 0; s < TRI_ROWS; ++s)
        {
            const int i = lane + 32*s;
            if (i < k) yr[s] -= F[k*n + i]*yk;
        }
    }
}

// x = A^-1 r in one block: the block brings the factors into shared memory in one coalesced pass when they
// fit, and ONE warp then runs both substitutions with the vector in its registers -- no block barrier in the
// n steps, where coarseLUSolveKernel takes about 3n. chol (a device flag, so the choice made at the
// factorisation holds inside a captured graph): 1 = F2 holds Cholesky's L and L^T, taken with no
// interchange; otherwise the LU (F, perm).
__global__
void coarseWarpSolveKernel(
    int n,
    const scalar* __restrict__ F,
    const int* __restrict__ perm,
    const scalar* __restrict__ F2,
    const int* __restrict__ chol,
    const scalar* __restrict__ rc,
    scalar* __restrict__ xc,
    bool inShared)
{
    extern __shared__ scalar sh[];
    const bool useChol = F2 != nullptr && chol != nullptr && *chol == 1;
    const scalar* __restrict__ G = useChol ? F2 : F;
    if (inShared)
    {
        for (int i = threadIdx.x; i < n*n; i += blockDim.x) sh[i] = G[i];
        __syncthreads();
        G = sh;
    }
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    scalar yr[TRI_ROWS];
    #pragma unroll
    for (int s = 0; s < TRI_ROWS; ++s)
    {
        const int i = lane + 32*s;
        yr[s] = (i < n) ? rc[useChol ? i : perm[i]] : 0.0;
    }
    warpSubstitute(n, G, !useChol, yr, lane);
    #pragma unroll
    for (int s = 0; s < TRI_ROWS; ++s)
    {
        const int i = lane + 32*s;
        if (i < n) xc[i] = yr[s];
    }
}

// A^-1, a column a warp: column j is the LU's solve of e_j by the same substitutions, written row-major
// (inv[i*n + j]) for the product below. Reads the column-major factors from global memory: once a factorisation.
__global__
void coarseInverseKernel(
    int n,
    const scalar* __restrict__ F,
    const int* __restrict__ perm,
    scalar* __restrict__ inv)
{
    const int j = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (j >= n) return;
    scalar yr[TRI_ROWS];
    #pragma unroll
    for (int s = 0; s < TRI_ROWS; ++s)
    {
        const int i = lane + 32*s;
        yr[s] = (i < n && perm[i] == j) ? 1.0 : 0.0;
    }
    warpSubstitute(n, F, true, yr, lane);
    #pragma unroll
    for (int s = 0; s < TRI_ROWS; ++s)
    {
        const int i = lane + 32*s;
        if (i < n) inv[i*n + j] = yr[s];
    }
}

// x = A^-1 r as a product: a warp a row, each lane every 32nd column, then a fixed shuffle tree
__global__
void coarseInverseApplyKernel(
    int n,
    const scalar* __restrict__ inv,
    const scalar* __restrict__ rc,
    scalar* __restrict__ xc)
{
    const int i = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= n) return;
    scalar s = 0.0;
    for (int j = lane; j < n; j += 32) s += inv[i*n + j]*rc[j];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) xc[i] = s;
}

// A_c = L L^T where the level allows it, in ONE block (the fill is coarseLUFactorKernel's, the coupled pair's
// dense entries with it). work is the n*n row-major working matrix (shared memory when inShared). Writes lc
// column-major with both factors, lc[k*n + i] = L[i][k] for i >= k and L[k][i] for i < k, so that the warp
// solve reads L's columns forward and L^T's backward; chol = 1 then, 0 (and lc untouched) where the level is
// not symmetric to CHOL_SYM_TOL or a pivot falls below CHOL_PIVOT_MIN.
__global__
void coarseCholeskyKernel(
    int n,
    const scalar* __restrict__ diag,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    const scalar* __restrict__ extra,
    scalar* __restrict__ work,
    scalar* __restrict__ lc,
    int* __restrict__ chol,
    bool inShared)
{
    const int tid = threadIdx.x;
    __shared__ scalar shScale;
    __shared__ scalar shAsym[TPB];
    __shared__ scalar shL;
    __shared__ int shFail;
    extern __shared__ scalar sh[];
    scalar* __restrict__ A = inShared ? sh : work;
    for (int i = tid; i < n*n; i += blockDim.x) A[i] = 0.0;
    __syncthreads();
    for (int c = tid; c < n; c += blockDim.x)
    {
        A[c*n + c] = diag[c];
        for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f) A[c*n + nei[f]] = upper[f];
        for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
        {
            const int f = losort[k];
            A[c*n + owner[f]] = lower[f];
        }
    }
    __syncthreads();
    if (extra)
    {
        for (int i = tid; i < n*n; i += blockDim.x) A[i] += extra[i];
        __syncthreads();
    }
    scalar asym = 0.0;
    for (int idx = tid; idx < n*n; idx += blockDim.x)
    {
        const int i = idx / n;
        const int j = idx % n;
        if (j > i) asym = fmax(asym, fabs(A[i*n + j] - A[j*n + i]));
    }
    shAsym[tid] = asym;
    if (tid == 0)
    {
        scalar m = 0.0;
        for (int c = 0; c < n; ++c) m = fmax(m, fabs(diag[c]));
        shScale = (m > 0.0) ? m : 1.0;
        shFail = 0;
    }
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2)
    {
        if (tid < width) shAsym[tid] = fmax(shAsym[tid], shAsym[tid + width]);
        __syncthreads();
    }
    if (!(shAsym[0] <= CHOL_SYM_TOL*shScale))
    {
        if (tid == 0) *chol = 0;
        return;
    }
    for (int k = 0; k < n; ++k)
    {
        if (tid == 0)
        {
            const scalar d = A[k*n + k];
            if (!(d > CHOL_PIVOT_MIN*shScale)) shFail = 1;
            else
            {
                shL = sqrt(d);
                A[k*n + k] = shL;
            }
        }
        __syncthreads();
        if (shFail)
        {
            if (tid == 0) *chol = 0;
            return;
        }
        const scalar l = shL;
        for (int i = k+1+tid; i < n; i += blockDim.x) A[i*n + k] /= l;
        __syncthreads();
        const int m = n - k - 1;
        for (int idx = tid; idx < m*m; idx += blockDim.x)            // the lower triangle of the trailing block
        {
            const int i = k + 1 + idx/m;
            const int j = k + 1 + idx%m;
            if (j <= i) A[i*n + j] -= A[i*n + k]*A[j*n + k];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < n*n; idx += blockDim.x)
    {
        const int k = idx / n;
        const int i = idx % n;
        lc[idx] = (i >= k) ? A[i*n + k] : A[k*n + i];
    }
    if (tid == 0) *chol = 1;
}

} // anon

bool deviceCoarseFitsCluster(int nCoarse)
{
    static const bool clusterOK = []()
    {
        int v = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, dev);
        return v != 0;
    }();
    if (!clusterOK) return false;                                            // pre-Hopper: no cluster launch -> global-mem Jacobi fallback
    const int cpb = (nCoarse + CCL - 1) / CCL;
    return 2 * static_cast<std::size_t>(cpb) * sizeof(scalar) <= 99 * 1024;   // GB10 opt-in DSM/block
}

void deviceCoarseJacobiFused(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nSweeps)
{
    const int nC = cv.nCells, cpb = (nC + CCL - 1) / CCL;
    const std::size_t shBytes = 2 * static_cast<std::size_t>(cpb) * sizeof(scalar);
    // Opt into the &gt;48KB dynamic shared memory ONCE. This is a non-stream host runtime call, and
    // deviceCoarseJacobiFused is reachable from vcycleAt, which is stream-captured -- issuing it
    // during capture is at best ignored and at worst refuses the capture. The attribute is a
    // per-function property, so setting it a single time at first use is sufficient.
    static std::once_flag coarseShmemOptin;
    std::call_once(coarseShmemOptin, []()
    {
        cudaCheck(cudaFuncSetAttribute(coarseJacobiFusedKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024), "coarse shmem optin");
    });
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CCL);
    cfg.blockDim = dim3(TPB);
    cfg.dynamicSmemBytes = shBytes;
    cfg.stream = cudaStreamPerThread;
    cudaLaunchAttribute attr[1] = {};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CCL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaCheck(cudaLaunchKernelEx(&cfg, coarseJacobiFusedKernel, nC, cpb, nSweeps, OMEGA,
        rc.data(), cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower, xc.data()),
        "coarseJacobiFused");
    cudaCheck(cudaGetLastError(), "coarseJacobiFused launch");
}

// Single-block coarsest solve (nC <= SB_MAX): nSweeps of in-shared-memory ping-pong Jacobi, cheap __syncthreads.
void deviceCoarseJacobiSingleBlock(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nSweeps)
{
    const int nC = cv.nCells;
    const std::size_t shBytes = 2 * static_cast<std::size_t>(nC) * sizeof(scalar);
    coarseJacobiSingleBlockKernel<<<1, TPB, shBytes, cudaStreamPerThread>>>(nC, nSweeps, OMEGA,
        rc.data(), cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower, xc.data());
    cudaCheck(cudaGetLastError(), "coarseJacobiSingleBlock launch");
}

// Single-block coarsest solve (nC <= SB_CG_MAX) by Jacobi-preconditioned CG, nIters iterations, one launch.
// Right-size the block to the tiny coarsest n (one warp per 32 cells, capped at TPB) so idle warps don't pay the
// block-reduce path. blockDim stays a multiple of 32 (full-warp shuffle mask) and >= 32 (one warp minimum).
void deviceCoarsePCG(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nIters)
{
    const int nC = cv.nCells;
    const int bs = nC >= TPB ? TPB : ((nC + 31) / 32) * 32;     // warp-rounded, in [32, TPB]
    const std::size_t shBytes = (5 * static_cast<std::size_t>(nC) + 32) * sizeof(scalar);   // red[] needs <= bs/32 slots
    coarsePCGKernel<<<1, bs, shBytes, cudaStreamPerThread>>>(nC, nIters,
        rc.data(), cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower, xc.data());
    cudaCheck(cudaGetLastError(), "coarsePCG launch");
}

// Single-block ASYMMETRIC coarsest solve (nC <= SB_CG_MAX) by Jacobi-preconditioned BiCGStab, nIters
// iterations, one launch. Same block sizing and the same 5*nC+32 shared footprint as deviceCoarsePCG.
void deviceCoarseBiCGStab(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nIters)
{
    const int nC = cv.nCells;
    const int bs = nC >= TPB ? TPB : ((nC + 31) / 32) * 32;     // warp-rounded, in [32, TPB]
    const std::size_t shBytes = (5 * static_cast<std::size_t>(nC) + 32) * sizeof(scalar);   // red[] needs <= bs/32 slots
    coarseBiCGStabKernel<<<1, bs, shBytes, cudaStreamPerThread>>>(nC, nIters,
        rc.data(), cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower, xc.data());
    cudaCheck(cudaGetLastError(), "coarseBiCGStab launch");
}


// Factorise the coarsest matrix. lu is sized n*n and piv n by the caller's DeviceBuffer; one block, and
// blockDim is warp-rounded to the level's size exactly as the iterative twins are.
void deviceCoarseLUFactor(
    const DeviceLduView& cv,
    DeviceBuffer<scalar>& lu,
    DeviceBuffer<int>& piv,
    const scalar* extra)
{
    const int nC = cv.nCells;
    // The trailing update is (n-k)^2 elements wide at the top of the loop, so take the whole block
    // rather than warp-rounding to nC as the iterative twins do -- their shared vectors are nC long,
    // this one's work is nC^2.
    const int bs = TPB;
    // Dynamic shared for the working matrix when n*n doubles clear the 48KB a block gets without the
    // opt-in (a non-stream runtime call; amgGalerkin is outside graph capture but the opt-in would
    // still be a per-launch surprise). At DENSE_COARSE_MAX = 96 that is 72 KB, so a level near the cap
    // falls back to global memory rather than asking for the opt-in.
    const std::size_t shBytes = static_cast<std::size_t>(nC)*nC*sizeof(scalar);
    const bool inShared = shBytes <= 48u*1024u;
    coarseLUFactorKernel<<<1, bs, inShared ? shBytes : 0, cudaStreamPerThread>>>(nC,
        cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower,
        lu.data(), piv.data(), inShared, extra);
    cudaCheck(cudaGetLastError(), "coarseLUFactor launch");
}

// Apply that factorisation: xc = A_c^-1 rc, exactly (to round-off), in one launch and with no host sync.
void deviceCoarseLUSolve(
    int nC,
    const DeviceBuffer<scalar>& lu,
    const DeviceBuffer<int>& piv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc)
{
    const int bs = TPB;
    // y always; the factors as well when n*(n+1) doubles clear the 48KB a block gets without the opt-in.
    const std::size_t both = static_cast<std::size_t>(nC)*(nC + 1)*sizeof(scalar);
    const bool inShared = both <= 48u*1024u;
    coarseLUSolveKernel<<<1, bs, inShared ? both : nC*sizeof(scalar), cudaStreamPerThread>>>(
        nC, lu.data(), piv.data(), rc.data(), xc.data(), inShared);
    cudaCheck(cudaGetLastError(), "coarseLUSolve launch");
}


CoarseSolveKind coarseSolveKind()
{
    static const CoarseSolveKind kind = []()
    {
        const char* e = std::getenv("BRAE_AMG_COARSE_SOLVE");
        const std::string v = e ? e : "";
        CoarseSolveKind k = CoarseSolveKind::LU;
        if (v.empty() || v == "lu") k = CoarseSolveKind::LU;
        else if (v == "tri") k = CoarseSolveKind::Tri;
        else if (v == "inverse") k = CoarseSolveKind::Inverse;
        else if (v == "cholesky") k = CoarseSolveKind::Cholesky;
        else throw std::runtime_error("BRAE_AMG_COARSE_SOLVE='" + v + "' is not one of lu, tri, inverse, cholesky");
        if (k != CoarseSolveKind::LU) std::printf("NOTICE: the AMG's coarsest direct solve is '%s' (BRAE_AMG_COARSE_SOLVE)\n", v.c_str());
        return k;
    }();
    return kind;
}

namespace
{
// the largest dynamic shared memory a block may ask for here, granted to the warp solve once (outside every
// capture: the first prepare)
int warpSolveSharedMax()
{
    static int bytes = 0;
    static std::once_flag once;
    std::call_once(once, []()
    {
        int dev = 0;
        cudaCheck(cudaGetDevice(&dev), "warp solve device");
        int optin = 0;
        cudaCheck(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), "warp solve shared");
        cudaCheck(cudaFuncSetAttribute(coarseWarpSolveKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin),
                  "warp solve shared optin");
        bytes = optin;
    });
    return bytes;
}
} // anon

void deviceCoarseDirectPrepare(
    CoarseSolveKind kind,
    const DeviceLduView& cv,
    const scalar* extra,
    const DeviceBuffer<scalar>& lu,
    const DeviceBuffer<int>& piv,
    CoarseDirect& d)
{
    const int n = cv.nCells;
    // the warp kernels hold n <= 32*TRI_ROWS rows; above it every kind is the LU
    d.kind = (n <= 32*TRI_ROWS) ? kind : CoarseSolveKind::LU;
    d.n = n;
    if (d.kind == CoarseSolveKind::LU) return;
    const std::size_t nn = static_cast<std::size_t>(n)*n;
    d.fc.resize(nn);
    d.perm.resize(n);
    warpSolveSharedMax();
    coarseLUColumnMajorKernel<<<nBlocks(static_cast<int>(nn)), TPB, 0, cudaStreamPerThread>>>(n, lu.data(), piv.data(), d.fc.data(), d.perm.data());
    cudaCheck(cudaGetLastError(), "coarse LU column-major");
    if (d.kind == CoarseSolveKind::Inverse)
    {
        d.f2.resize(nn);
        coarseInverseKernel<<<nBlocks(32*n), TPB, 0, cudaStreamPerThread>>>(n, d.fc.data(), d.perm.data(), d.f2.data());
        cudaCheck(cudaGetLastError(), "coarse inverse");
    }
    else if (d.kind == CoarseSolveKind::Cholesky)
    {
        d.f2.resize(nn);
        d.chol.resize(1);
        const std::size_t shBytes = nn*sizeof(scalar);
        // the static shared (the asymmetry reduction, TPB doubles) shares the 48 KB with the working matrix
        const bool inShared = shBytes + TPB*sizeof(scalar) + 64 <= 48u*1024u;
        if (!inShared) d.work.resize(nn);
        coarseCholeskyKernel<<<1, TPB, inShared ? shBytes : 0, cudaStreamPerThread>>>(n,
            cv.diag, cv.ownerStart, cv.nei, cv.upper, cv.losortStart, cv.losort, cv.owner, cv.lower, extra,
            inShared ? nullptr : d.work.data(), d.f2.data(), d.chol.data(), inShared);
        cudaCheck(cudaGetLastError(), "coarse cholesky");
    }
}

void deviceCoarseDirectSolve(
    int nC,
    const DeviceBuffer<scalar>& lu,
    const DeviceBuffer<int>& piv,
    const CoarseDirect& d,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc)
{
    if (d.kind == CoarseSolveKind::LU || d.n != nC)
    {
        deviceCoarseLUSolve(nC, lu, piv, rc, xc);
        return;
    }
    if (d.kind == CoarseSolveKind::Inverse)
    {
        coarseInverseApplyKernel<<<nBlocks(32*nC), TPB, 0, cudaStreamPerThread>>>(nC, d.f2.data(), rc.data(), xc.data());
        cudaCheck(cudaGetLastError(), "coarse inverse apply");
        return;
    }
    const std::size_t shBytes = static_cast<std::size_t>(nC)*nC*sizeof(scalar);
    const bool inShared = static_cast<int>(shBytes) <= warpSolveSharedMax();
    const bool chol = d.kind == CoarseSolveKind::Cholesky;
    coarseWarpSolveKernel<<<1, TPB, inShared ? shBytes : 0, cudaStreamPerThread>>>(nC, d.fc.data(), d.perm.data(),
        chol ? d.f2.data() : nullptr, chol ? d.chol.data() : nullptr, rc.data(), xc.data(), inShared);
    cudaCheck(cudaGetLastError(), "coarse warp solve");
}

// Iterated coarsest solve: nSweeps of global weighted-Jacobi (deviceAmul + smoothT). Reuses the V-cycle's
// smoothT<> kernel (amg_kernels.cuh) and deviceAmul; kept out of the single-launch solvers above for that reason.
void deviceCoarseJacobiLoop(
    const DeviceLduView& cv,
    const DeviceBuffer<scalar>& rc,
    DeviceBuffer<scalar>& xc,
    int nSweeps)
{
    const int nC = cv.nCells;
    DeviceBuffer<scalar> Axc(nC);
    for (int s = 0; s < nSweeps; ++s)
    {
        deviceAmul(cv, xc, Axc);
        smoothT<scalar><<<nBlocks(nC),TPB>>>(nC, rc.data(), Axc.data(), cv.diag, xc.data());
    }
}

} // namespace brae
