// AMG V-cycle smoothers: the per-level smoothing kernels + apply functions the V-cycle calls each level.
//   * weighted-Jacobi is inline in the V-cycle (smoothT, amg_kernels.cuh); this file adds the upgrades --
//     Chebyshev polynomial (chebyshevSmooth), OpenFOAM v2606 twoStageGaussSeidel (twoStageGSSmooth), and one
//     multicolor Gauss-Seidel sweep (gsSweep) -- plus the power-iteration spectrum estimate the Chebyshev
//     interval needs (estimateLambdaMax/ensureSpectrum). The V-cycle (device_amg.cu) calls these via decls
//     in device_amg_internal.cuh; the standalone symGaussSeidel solver (device_amg.cu) reuses gsSweep too.
// Verbatim split of device_amg.cu -- no logic change.
#include "device_amg.cuh"          // AMGData / GridColoring
#include "device_amg_detail.cuh"   // safeDiag / OMEGA / CHEB_* / useChebyshev / nBlocks / TPB
#include "device_amg_internal.cuh" // matching decls for the smoother interface
#include "amg_kernels.cuh"         // smoothT / zeroT / gsColorT (templated V-cycle kernels)
#include "device_ldu.cuh"          // DeviceLduView / deviceAmul
#include "device_blas.cuh"         // deviceDot / deviceScale / deviceCopy
#include <cuda_runtime.h>
#include <cmath>

namespace brae {

namespace {
// z = src/diag: applies the diagonal (Jacobi) preconditioner, used by the power iteration on D^-1 A.
__global__
void invDiagMulK(
    int n,
    const scalar* __restrict__ src,
    const scalar* __restrict__ diag,
    scalar* __restrict__ z)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) z[i] = src[i]/diag[i];
}
// Chebyshev polynomial smoother step: d = c1*d + c2*(b - Ax)/diag ; x += d.
__global__
void chebStepK(
    int n,
    const scalar* __restrict__ b,
    const scalar* __restrict__ Ax,
    const scalar* __restrict__ diag,
    scalar c1,
    scalar c2,
    scalar* __restrict__ d,
    scalar* __restrict__ x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = c1*d[i] + c2*(b[i]-Ax[i])/diag[i];
    x[i] += d[i];
}
// twoStageGaussSeidel kernels (BRAE_AMG_TSGS). upperMul = strictly-upper-triangle SpMV r = U z; the tsgs* kernels
// below are the fused Jacobi/correction updates of the polynomial expansion.
__global__
void upperMulK(
    int n,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ z,
    scalar* __restrict__ r)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        s += upper[f] * z[nei[f]];
    r[c] = s;
}
// lowerMul = strictly-lower-triangle SpMV r = L z, the transpose of upperMul for the symmetric matrix. Pre-smooth
// with U and post-smooth with L so the V-cycle preconditioner stays symmetric, as the AMG-PCG outer solver requires.
__global__
void lowerMulK(
    int n,
    const label* __restrict__ losortStart,
    const label* __restrict__ losort,
    const label* __restrict__ owner,
    const scalar* __restrict__ lower,
    const scalar* __restrict__ z,
    scalar* __restrict__ r)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n) return;
    scalar s = 0.0;
    for (int k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const int f = losort[k];
        s += lower[f] * z[owner[f]];
    }
    r[c] = s;
}
__global__
void tsgsZUpdateK(
    int n,
    const scalar* __restrict__ b,
    const scalar* __restrict__ Ax,
    const scalar* __restrict__ diag,
    scalar* __restrict__ z,
    scalar* __restrict__ x)
{   // z = (b-Ax)/D; x += z
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const scalar zi = (b[i]-Ax[i])/diag[i];
        z[i]=zi;
        x[i]+=zi;
    }
}
__global__
void tsgsCorrSaveK(
    int n,
    const scalar* __restrict__ r,
    const scalar* __restrict__ diag,
    scalar mult,
    scalar* __restrict__ z,
    scalar* __restrict__ x)
{   // z = r/D; x += mult*z
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const scalar zi = r[i]/diag[i];
        z[i]=zi;
        x[i]+=mult*zi;
    }
}
__global__
void tsgsCorrLastK(
    int n,
    const scalar* __restrict__ r,
    const scalar* __restrict__ diag,
    scalar mult,
    scalar* __restrict__ x)
{                            // last term: x += mult*r/D (no z save)
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] += mult*r[i]/diag[i];
}

// High-frequency seed for the power iteration. The CONSTANT vector is the Laplacian's
// near-null-space (the SMALLEST eigenvalue of D^-1 A), so starting there makes power iteration
// reach lambda_max only through round-off: in exact arithmetic a constant start is orthogonal to
// every non-constant mode and never leaves the null space. In few iterations it UNDER-estimates
// lambda_max badly, and an under-estimated Chebyshev interval [., 1.1*lambdaMax] leaves the true
// top modes ABOVE the interval, where the Chebyshev polynomial amplifies rather than damps them
// -> the smoother diverges (observed as a stall at ~1.6e-3 on a 1M-cell Laplacian). A period-7
// sawtooth injects broad spectral content, exactly as the SA path's hostSpectralRadius does.
__global__
void fillSeedK(
    int n,
    scalar* __restrict__ x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0 + 0.1*static_cast<scalar>(i % 7);
}
} // anon

// amg.spectrumReady. x/ax/z are grid-g sized scratch (the V-cycle's vX/vAx/vR, free before the first cycle).
static scalar estimateLambdaMax(
    const DeviceLduView& A,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& ax,
    DeviceBuffer<scalar>& z)
{
    const int n = A.nCells;
    fillSeedK<<<nBlocks(n),TPB>>>(n, x.data());
    scalar lam = 1.0;
    // 25 iterations, not 12: power iteration converges to lambda_max FROM BELOW, and every ms of
    // under-estimate risks divergence in the Chebyshev smoother, while an over-estimate only mildly
    // slows convergence -> spend the extra iterations, they run once per matrix.
    for (int it = 0; it < 25; ++it)
    {
        const scalar nrm = std::sqrt(deviceDot(x, x));
        if (!(nrm > 0.0)) break;
        deviceScale(x, 1.0/nrm);                                  // x <- x/||x||
        deviceAmul(A, x, ax);                                     // ax = A x
        invDiagMulK<<<nBlocks(n),TPB>>>(n, ax.data(), A.diag, z.data());  // z = D^-1 A x
        lam = std::sqrt(deviceDot(z, z));                         // ||D^-1 A x|| (||x||=1) -> Rayleigh-quotient bound
        deviceCopy(x, z);
    }
    return lam;
}

// One-time Chebyshev spectrum estimate per matrix (host scalars -> must precede any graph capture). lambdaMax[g] for
// the SMOOTHED grids 0..nLevels-1; the coarsest grid stays a many-sweep Jacobi solve. No-op unless Chebyshev is
// selected (no power-iteration cost when lambdaMax is unused) or already computed for this matrix.
void ensureSpectrum(
    AMGData& amg,
    const DeviceLduView& A)
{
    if (!useChebyshev() || amg.spectrumReady) return;
    amg.lambdaMax[0] = estimateLambdaMax(A, amg.vX[0], amg.vAx[0], amg.vR[0]);
    for (int g = 1; g < amg.nLevels(); ++g)
        amg.lambdaMax[g] = estimateLambdaMax(amg.level[g-1].coarseView(), amg.vX[g], amg.vAx[g], amg.vR[g]);
    amg.spectrumReady = true;
}

// Preconditioned Chebyshev (polynomial) smoother of degree `deg` on the interval [upper/CHEB_EIGRATIO, upper] of
// D^-1 A (upper = CHEB_UPPER_SAFETY*lambdaMax). Damps the high-frequency error far better per flop than weighted-Jacobi, this is
// the standard GPU-AMG smoother upgrade. x is the in/out guess (zero on pre-smooth, prolonged correction on post-);
// d is grid-sized scratch. All coefficients are host CONSTANTS (no device reductions) -> graph-capturable.
void chebyshevSmooth(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& d,
    DeviceBuffer<scalar>& ax,
    scalar lambdaMax,
    int deg)
{
    const int n = A.nCells;
    // Bias the interval top ABOVE the power-iteration estimate: the estimate converges to
    // lambda_max from below, so the true top eigenvalues can sit just past it, exactly where the
    // Chebyshev polynomial amplifies. Over-covering the interval only mildly slows smoothing;
    // under-covering it diverges. 1.2 (not 1.1) buys margin for the residual under-estimate.
    const scalar upper = CHEB_UPPER_SAFETY*lambdaMax, lower = upper/CHEB_EIGRATIO;
    const scalar theta = 0.5*(upper+lower), delta = 0.5*(upper-lower), sigma = theta/delta;
    zeroT<scalar><<<nBlocks(n),TPB>>>(n, d.data());                       // avoid 0*Inf=NaN on the c1=0 first step
    deviceAmul(A, x, ax);                                        // step 0: d = (1/theta)(b-Ax)/D ; x += d
    chebStepK<<<nBlocks(n),TPB>>>(n, b.data(), ax.data(), A.diag, 0.0, 1.0/theta, d.data(), x.data());
    scalar rho = 1.0/sigma;
    for (int k = 1; k < deg; ++k)
    {
        const scalar rhoNew = 1.0/(2.0*sigma - rho);
        const scalar c1 = rho*rhoNew, c2 = 2.0*rhoNew/delta;
        deviceAmul(A, x, ax);
        chebStepK<<<nBlocks(n),TPB>>>(n, b.data(), ax.data(), A.diag, c1, c2, d.data(), x.data());
        rho = rhoNew;
    }
}

// Two-stage Gauss-Seidel smoother (BRAE_AMG_TSGS), OpenFOAM v2606 twoStageGaussSeidel (arXiv:2111.09512, "Scaled
// Smoothers for Navier-Stokes Pressure Projection"). A fully parallel polynomial smoother: per sweep,
//   r = A x ; z = (b - r)/D ; x += z ;                              (weighted-Jacobi stage, w baked into the /D update)
//   for k in 0..order-1: r = U z ; x += (-1)^(k+1) z_k ; z = r/D    (Neumann-series correction in the strictly-upper
//                                                                    triangle, alternating signs)
// No serial dependency and no coloring -> graph-capturable at a fixed order (all coefficients are host constants,
// mult = +-1). order 0 reduces to the weighted-Jacobi smoothT exactly. z/r are level-sized scratch (the V-cycle's
// vD/vAx). Symmetry: the AMG-PCG outer solver needs an SPD preconditioner, so the correction triangle switches
// between pre (forward=U) and post (backward=L=U^T) smooth to keep the V-cycle symmetric; a U-only pre+post is
// non-symmetric here and wrecks PCG convergence.
void twoStageGSSmooth(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    DeviceBuffer<scalar>& z,
    DeviceBuffer<scalar>& r,
    int nSweeps,
    int order,
    bool forward)
{
    const int n = A.nCells;
    for (int s = 0; s < nSweeps; ++s)
    {
        deviceAmul(A, x, r);                                                                 // r = A x
        if (order == 0)   // == weighted Jacobi
        {
            smoothT<scalar><<<nBlocks(n),TPB>>>(n, b.data(), r.data(), A.diag, x.data());
            continue;
        }
        tsgsZUpdateK<<<nBlocks(n),TPB>>>(n, b.data(), r.data(), A.diag, z.data(), x.data());  // z=(b-Ax)/D; x+=z
        scalar mult = -1.0;
        for (int k = 0; k < order; ++k)
        {
            if (forward) upperMulK<<<nBlocks(n),TPB>>>(n, A.ownerStart, A.nei, A.upper, z.data(), r.data());              // r = U z (pre)
            else         lowerMulK<<<nBlocks(n),TPB>>>(n, A.losortStart, A.losort, A.owner, A.lower, z.data(), r.data()); // r = L z (post -> symmetric)
            if (k < order - 1) tsgsCorrSaveK<<<nBlocks(n),TPB>>>(n, r.data(), A.diag, mult, z.data(), x.data());
            else               tsgsCorrLastK<<<nBlocks(n),TPB>>>(n, r.data(), A.diag, mult, x.data());
            mult = -mult;
        }
    }
}

// One multicolor Gauss-Seidel sweep over grid g. forward = colors 0..nColors-1, backward = reverse. The V-cycle
// pre-smooths forward and post-smooths backward so the whole preconditioner is SYMMETRIC (symmetric GS), required
// by the plain-CG callers; the SIMPLE loop's flexible CG would tolerate either. One kernel launch per color, all
// bounds host-constant -> the sweep is still capturable into the V-cycle graph (same as the Jacobi/Chebyshev paths).
void gsSweep(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& x,
    const GridColoring& gc,
    bool forward)
{
    for (int ci = 0; ci < gc.nColors; ++ci)
    {
        const int col = forward ? ci : (gc.nColors-1-ci);
        const int lo = gc.startH[col], hi = gc.startH[col+1];
        const int nc = hi - lo;
        if (nc <= 0) continue;
        gsColorT<scalar><<<nBlocks(nc),TPB>>>(lo, hi, gc.cells.data(), b.data(), A.diag,
            A.ownerStart, A.nei, A.upper, A.losortStart, A.losort, A.owner, A.lower, x.data());
    }
}

} // namespace brae
