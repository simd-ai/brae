// cf GPU offload -- BLAS-1 ELEMENTWISE ops: axpy/scale/copy/jacobi/hadamard, the device-resident-scalar variants
// (coefficient by pointer, no host sync), the single-thread scalar-recurrence ops, and the FUSED multi-term Krylov
// updates (one pass, bit-identical FP sequence). Split from device_blas.cu (reductions in reductions.cu).
#include "device_blas.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__device__
void axpyKernel(scalar a, const scalar* __restrict__ x, scalar* __restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}


__device__
void scaleKernel(scalar a, scalar* __restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}


__device__
void jacobiKernel(const scalar* __restrict__ r, const scalar* __restrict__ diag, scalar* __restrict__ z, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = r[i] / diag[i];
}


__device__
void hadamardKernel(const scalar* __restrict__ a, const scalar* __restrict__ b, scalar* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}


// device-resident-scalar variants: the coefficient lives in device memory (read by pointer), never on the host.
__device__
void axpyDevKernel(const scalar* __restrict__ a, const scalar* __restrict__ x, scalar* __restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += (*a) * x[i];
}


__device__
void scaleDevKernel(const scalar* __restrict__ a, scalar* __restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= (*a);
}


// single-thread scalar recurrence ops (cheap launch, NO host sync, same IEEE double op as the host did).
__device__
void scalarDivK(const scalar* num, const scalar* den, scalar* out)
{
    // Guard the pressure-PCG divide (den = pAp / wArAold): a near-zero/breakdown denominator yields 0 (no update this
    // iteration) instead of Inf/NaN, mirroring the momentum BiCGStab's OF-checkSingularity guard. Healthy solve: bit-identical.
    if (threadIdx.x == 0 && blockIdx.x == 0) { const scalar d = *den; *out = (fabs(d) > scalar(1e-300)) ? (*num) / d : scalar(0); }
}


__device__
void scalarDivNegK(const scalar* num, const scalar* den, scalar* out, scalar* outNeg)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) { const scalar d = *den; const scalar q = (fabs(d) > scalar(1e-300)) ? (*num) / d : scalar(0); *out = q; *outNeg = -q; }
}


__device__
void scalarCopyK(const scalar* src, scalar* dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}


__device__
void scalarDivConstK(const scalar* num, scalar denom, scalar* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = (*num) / denom;
}


__device__
void scalarAdd2K(const scalar* a, const scalar* b, scalar c, scalar* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = (*a) + (*b) + c;   // matches host (a+b)+c associativity
}


// FUSED multi-term BLAS-1 (one pass: fewer launches + less memory traffic). Each reproduces the EXACT FP-op
// sequence of the separate kernels it replaces (fma() where axpy/axpyDev contract under --fmad=true; __dmul_rn for a
// scale so a new mul+add does NOT contract) -> BIT-IDENTICAL, not just machine-precision. Device-scalar coeffs by ptr.
// pA = rA + beta*(pA - omega*AyA)   [replaces axpyDev(negOmega,AyA,pA) + scaleDev(beta,pA) + axpy(1,rA,pA)]
__device__
void fusedBicgPK(
    const scalar* __restrict__ rA,
    scalar* __restrict__ pA,
    const scalar* __restrict__ AyA,
    const scalar* __restrict__ beta,
    const scalar* __restrict__ negOmega,
    int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar t = fma(*negOmega, AyA[i], pA[i]);          // pA - omega*AyA  (axpyDev fma)
    pA[i] = __dmul_rn(t, *beta) + rA[i];                     // *beta (scale, rounded) then + rA (axpy)
}


// out = src + a*x   [replaces copy(src->out) + axpyDev(a,x,out)]
__device__
void fusedSxpyK(
    scalar* __restrict__ out,
    const scalar* __restrict__ src,
    const scalar* __restrict__ a,
    const scalar* __restrict__ x,
    int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = fma(*a, x[i], src[i]);   // src + a*x (fma)
}


// y += a*x1 + b*x2   [replaces axpyDev(a,x1,y) + axpyDev(b,x2,y)]
__device__
void fusedAxpy2K(
    scalar* __restrict__ y,
    const scalar* __restrict__ a,
    const scalar* __restrict__ x1,
    const scalar* __restrict__ b,
    const scalar* __restrict__ x2,
    int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar t = fma(*a, x1[i], y[i]);                   // y + a*x1 (axpyDev fma)
    y[i] = fma(*b, x2[i], t);                                // + b*x2 (axpyDev fma)
}


// p = b*p + w   [replaces scaleDev(b,p) + axpy(1,w,p)]
__device__
void fusedScaleAxpyK(scalar* __restrict__ p, const scalar* __restrict__ b, const scalar* __restrict__ w, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) p[i] = __dmul_rn(*b, p[i]) + w[i];   // b*p (scale) + w (axpy)
}
} // namespace


void deviceAxpy(scalar a, const DeviceBuffer<scalar>& x, DeviceBuffer<scalar>& y)
{
    const int n = static_cast<int>(x.size());
    const scalar* xd = x.data();
    scalar* yd = y.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { axpyKernel(a, xd, yd, n); });
    cudaCheck(cudaGetLastError(), "axpy");
}


void deviceScale(DeviceBuffer<scalar>& x, scalar a)
{
    const int n = static_cast<int>(x.size());
    scalar* xd = x.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { scaleKernel(a, xd, n); });
    cudaCheck(cudaGetLastError(), "scale");
}


void deviceCopy(DeviceBuffer<scalar>& dst, const DeviceBuffer<scalar>& src)
{
    dst.resize(src.size());
    // D2D ASYNC on the per-thread stream: the copy stays ordered with the kernels that consume dst, and the host
    // never reads it here, so there is no reason to block the host (a plain cudaMemcpy would drain the GPU pipeline).
    // Host reads go through copyTo()/.host(), which sync explicitly.
    cudaCheck(cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(scalar), cudaMemcpyDeviceToDevice,
                              cudaStreamPerThread), "copy");
}


void deviceJacobi(DeviceBuffer<scalar>& z, const DeviceBuffer<scalar>& r, const scalar* diag)
{
    const int n = static_cast<int>(r.size());
    z.resize(n);
    const scalar* rd = r.data();
    scalar* zd = z.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { jacobiKernel(rd, diag, zd, n); });
    cudaCheck(cudaGetLastError(), "jacobi");
}


// out = a ./ b. A zero denominator yields zero rather than a NaN: the only caller is the transonic
// pressure equation, where a face with rho_f = 0 has no flux to carry anyway, and a NaN there would
// propagate silently into the whole matrix.
__device__
void divideKernel(const scalar* __restrict__ a, const scalar* __restrict__ b, scalar* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (b[i] != scalar(0)) ? a[i] / b[i] : scalar(0);
}


void deviceDivide(DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b)
{
    const int n = static_cast<int>(a.size());
    out.resize(n);
    const scalar* ad = a.data();
    const scalar* bd = b.data();
    scalar* od = out.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { divideKernel(ad, bd, od, n); });
    cudaCheck(cudaGetLastError(), "divide");
}


void deviceHadamard(DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b)
{
    const int n = static_cast<int>(a.size());
    out.resize(n);
    const scalar* ad = a.data();
    const scalar* bd = b.data();
    scalar* od = out.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { hadamardKernel(ad, bd, od, n); });
    cudaCheck(cudaGetLastError(), "hadamard");
}


// fused multi-term updates (replace 2-3 BLAS-1 launches each; device-scalar coeffs by ptr)
void deviceFusedBicgP(
    const DeviceBuffer<scalar>& rA,
    DeviceBuffer<scalar>& pA,
    const DeviceBuffer<scalar>& AyA,
    const scalar* beta,
    const scalar* negOmega)
{
    const int n = static_cast<int>(pA.size());
    const scalar* rAd = rA.data();
    scalar* pAd = pA.data();
    const scalar* AyAd = AyA.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { fusedBicgPK(rAd, pAd, AyAd, beta, negOmega, n); });
    cudaCheck(cudaGetLastError(), "fusedBicgP");
}


void deviceFusedSxpy(DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& src, const scalar* a, const DeviceBuffer<scalar>& x)
{
    const int n = static_cast<int>(src.size());
    out.resize(n);
    scalar* outd = out.data();
    const scalar* srcd = src.data();
    const scalar* xd = x.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { fusedSxpyK(outd, srcd, a, xd, n); });
    cudaCheck(cudaGetLastError(), "fusedSxpy");
}


void deviceFusedAxpy2(DeviceBuffer<scalar>& y, const scalar* a, const DeviceBuffer<scalar>& x1, const scalar* b, const DeviceBuffer<scalar>& x2)
{
    const int n = static_cast<int>(y.size());
    scalar* yd = y.data();
    const scalar* x1d = x1.data();
    const scalar* x2d = x2.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { fusedAxpy2K(yd, a, x1d, b, x2d, n); });
    cudaCheck(cudaGetLastError(), "fusedAxpy2");
}


void deviceFusedScaleAxpy(DeviceBuffer<scalar>& p, const scalar* b, const DeviceBuffer<scalar>& w)
{
    const int n = static_cast<int>(p.size());
    scalar* pd = p.data();
    const scalar* wd = w.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { fusedScaleAxpyK(pd, b, wd, n); });
    cudaCheck(cudaGetLastError(), "fusedScaleAxpy");
}


void deviceAxpyDev(const scalar* dA, const DeviceBuffer<scalar>& x, DeviceBuffer<scalar>& y)
{
    const int n = static_cast<int>(x.size());
    const scalar* xd = x.data();
    scalar* yd = y.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { axpyDevKernel(dA, xd, yd, n); });
    cudaCheck(cudaGetLastError(), "axpyDev");
}


void deviceScaleDev(const scalar* dA, DeviceBuffer<scalar>& x)
{
    const int n = static_cast<int>(x.size());
    scalar* xd = x.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { scaleDevKernel(dA, xd, n); });
    cudaCheck(cudaGetLastError(), "scaleDev");
}


void deviceScalarDiv(const scalar* num, const scalar* den, scalar* out)
{
    pcudaParallelFor(1, 1, [=] __device__ () { scalarDivK(num, den, out); });
    cudaCheck(cudaGetLastError(), "scalarDiv");
}


void deviceScalarDivNeg(const scalar* num, const scalar* den, scalar* out, scalar* outNeg)
{
    pcudaParallelFor(1, 1, [=] __device__ () { scalarDivNegK(num, den, out, outNeg); });
    cudaCheck(cudaGetLastError(), "scalarDivNeg");
}


void deviceScalarCopy(const scalar* src, scalar* dst)
{
    pcudaParallelFor(1, 1, [=] __device__ () { scalarCopyK(src, dst); });
    cudaCheck(cudaGetLastError(), "scalarCopy");
}


void deviceScalarDivConst(const scalar* num, scalar denom, scalar* out)
{
    pcudaParallelFor(1, 1, [=] __device__ () { scalarDivConstK(num, denom, out); });
    cudaCheck(cudaGetLastError(), "scalarDivConst");
}


void deviceScalarAdd2(const scalar* a, const scalar* b, scalar c, scalar* out)
{
    pcudaParallelFor(1, 1, [=] __device__ () { scalarAdd2K(a, b, c, out); });
    cudaCheck(cudaGetLastError(), "scalarAdd2");
}

} // namespace brae
