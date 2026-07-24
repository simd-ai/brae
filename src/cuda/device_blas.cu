// cf GPU offload (G0): BLAS-1 device kernels. Block-reduction dot (shared memory + atomicAdd to a device
// accumulator); the reduction order differs from the CPU sequential sum, so the result matches to
// machine precision (FP non-associativity), not bit-for-bit, the validation criterion for reductions.
#include "device_blas.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// Persistent reduction scratch (one-time alloc, process-lifetime): a device accumulator + a PINNED host mirror.
// Removes the per-reduction cudaMalloc/cudaFree + H2D-zero-init that dominated the device SIMPLE-loop wall; the
// zero-init is now a cheap async memset on the per-thread stream and the result D2H uses pinned memory. Single
// solve at a time (cf is one host thread per solve), so a function-local static accumulator is safe.
scalar* g_redDev = nullptr;       // device accumulator (1 scalar)
scalar* g_redPinned = nullptr;    // pinned host mirror (1 scalar)
scalar* g_readPinned = nullptr;   // pinned host mirror for deviceReadScalar (separate so it never clobbers g_redPinned)
inline void ensureRedScratch()
{
    if (!g_redDev)    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g_redDev), sizeof(scalar)), "red dev alloc");
    if (!g_redPinned) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&g_redPinned), sizeof(scalar)), "red pinned alloc");
}


__global__
void axpyKernel(scalar a, const scalar* __restrict__ x, scalar* __restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}


__global__
void scaleKernel(scalar a, scalar* __restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}


__global__
void dotKernel(const scalar* __restrict__ x, const scalar* __restrict__ y, scalar* result, int n)
{
    __shared__ scalar sdata[TPB];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? x[i] * y[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}


__global__
void sumMagKernel(const scalar* __restrict__ x, scalar* result, int n)
{
    __shared__ scalar sdata[TPB];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? fabs(x[i]) : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}


__global__
void jacobiKernel(const scalar* __restrict__ r, const scalar* __restrict__ diag, scalar* __restrict__ z, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = r[i] / diag[i];
}


__global__
void hadamardKernel(const scalar* __restrict__ a, const scalar* __restrict__ b, scalar* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}


// device-resident-scalar variants: the coefficient lives in device memory (read by pointer), never on the host.
__global__
void axpyDevKernel(const scalar* __restrict__ a, const scalar* __restrict__ x, scalar* __restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += (*a) * x[i];
}


__global__
void scaleDevKernel(const scalar* __restrict__ a, scalar* __restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= (*a);
}


// single-thread scalar recurrence ops (cheap launch, NO host sync, same IEEE double op as the host did).
__global__
void scalarDivK(const scalar* num, const scalar* den, scalar* out)
{
    // Guard the pressure-PCG divide (den = pAp / wArAold): a near-zero/breakdown denominator yields 0 (no update this
    // iteration) instead of Inf/NaN, mirroring the momentum BiCGStab's OF-checkSingularity guard. Healthy solve: bit-identical.
    if (threadIdx.x == 0 && blockIdx.x == 0) { const scalar d = *den; *out = (fabs(d) > scalar(1e-300)) ? (*num) / d : scalar(0); }
}


__global__
void scalarDivNegK(const scalar* num, const scalar* den, scalar* out, scalar* outNeg)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) { const scalar d = *den; const scalar q = (fabs(d) > scalar(1e-300)) ? (*num) / d : scalar(0); *out = q; *outNeg = -q; }
}


__global__
void scalarCopyK(const scalar* src, scalar* dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}


__global__
void scalarDivConstK(const scalar* num, scalar denom, scalar* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = (*num) / denom;
}


__global__
void scalarAdd2K(const scalar* a, const scalar* b, scalar c, scalar* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = (*a) + (*b) + c;   // matches host (a+b)+c associativity
}


// FUSED multi-term BLAS-1 (one pass: fewer launches + less memory traffic). Each reproduces the EXACT FP-op
// sequence of the separate kernels it replaces (fma() where axpy/axpyDev contract under --fmad=true; __dmul_rn for a
// scale so a new mul+add does NOT contract) -> BIT-IDENTICAL, not just machine-precision. Device-scalar coeffs by ptr.
// pA = rA + beta*(pA - omega*AyA)   [replaces axpyDev(negOmega,AyA,pA) + scaleDev(beta,pA) + axpy(1,rA,pA)]
__global__
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
__global__
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
__global__
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
__global__
void fusedScaleAxpyK(scalar* __restrict__ p, const scalar* __restrict__ b, const scalar* __restrict__ w, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) p[i] = __dmul_rn(*b, p[i]) + w[i];   // b*p (scale) + w (axpy)
}

} // namespace


void deviceAxpy(scalar a, const DeviceBuffer<scalar>& x, DeviceBuffer<scalar>& y)
{
    const int n = static_cast<int>(x.size());
    axpyKernel<<<nBlocks(n), TPB>>>(a, x.data(), y.data(), n);
    cudaCheck(cudaGetLastError(), "axpy");
}


void deviceScale(DeviceBuffer<scalar>& x, scalar a)
{
    const int n = static_cast<int>(x.size());
    scaleKernel<<<nBlocks(n), TPB>>>(a, x.data(), n);
    cudaCheck(cudaGetLastError(), "scale");
}


scalar deviceDot(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y)
{
    const int n = static_cast<int>(x.size());
    ensureRedScratch();
    cudaCheck(cudaMemsetAsync(g_redDev, 0, sizeof(scalar), cudaStreamPerThread), "dot zero");
    dotKernel<<<nBlocks(n), TPB>>>(x.data(), y.data(), g_redDev, n);
    cudaCheck(cudaGetLastError(), "dot");
    cudaCheck(cudaMemcpy(g_redPinned, g_redDev, sizeof(scalar), cudaMemcpyDeviceToHost), "dot result");
    return *g_redPinned;
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
    jacobiKernel<<<nBlocks(n), TPB>>>(r.data(), diag, z.data(), n);
    cudaCheck(cudaGetLastError(), "jacobi");
}


scalar deviceSumMag(const DeviceBuffer<scalar>& x)
{
    const int n = static_cast<int>(x.size());
    ensureRedScratch();
    cudaCheck(cudaMemsetAsync(g_redDev, 0, sizeof(scalar), cudaStreamPerThread), "summag zero");
    sumMagKernel<<<nBlocks(n), TPB>>>(x.data(), g_redDev, n);
    cudaCheck(cudaGetLastError(), "summag");
    cudaCheck(cudaMemcpy(g_redPinned, g_redDev, sizeof(scalar), cudaMemcpyDeviceToHost), "summag result");
    return *g_redPinned;
}


void deviceHadamard(DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b)
{
    const int n = static_cast<int>(a.size());
    out.resize(n);
    hadamardKernel<<<nBlocks(n), TPB>>>(a.data(), b.data(), out.data(), n);
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
    fusedBicgPK<<<nBlocks(n), TPB>>>(rA.data(), pA.data(), AyA.data(), beta, negOmega, n);
    cudaCheck(cudaGetLastError(), "fusedBicgP");
}


void deviceFusedSxpy(DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& src, const scalar* a, const DeviceBuffer<scalar>& x)
{
    const int n = static_cast<int>(src.size());
    out.resize(n);
    fusedSxpyK<<<nBlocks(n), TPB>>>(out.data(), src.data(), a, x.data(), n);
    cudaCheck(cudaGetLastError(), "fusedSxpy");
}


void deviceFusedAxpy2(DeviceBuffer<scalar>& y, const scalar* a, const DeviceBuffer<scalar>& x1, const scalar* b, const DeviceBuffer<scalar>& x2)
{
    const int n = static_cast<int>(y.size());
    fusedAxpy2K<<<nBlocks(n), TPB>>>(y.data(), a, x1.data(), b, x2.data(), n);
    cudaCheck(cudaGetLastError(), "fusedAxpy2");
}


void deviceFusedScaleAxpy(DeviceBuffer<scalar>& p, const scalar* b, const DeviceBuffer<scalar>& w)
{
    const int n = static_cast<int>(p.size());
    fusedScaleAxpyK<<<nBlocks(n), TPB>>>(p.data(), b, w.data(), n);
    cudaCheck(cudaGetLastError(), "fusedScaleAxpy");
}


// device-resident scalar plumbing (no host sync)
void deviceDotInto(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, scalar* dResult)
{
    const int n = static_cast<int>(x.size());
    cudaCheck(cudaMemsetAsync(dResult, 0, sizeof(scalar), cudaStreamPerThread), "dotInto zero");
    dotKernel<<<nBlocks(n), TPB>>>(x.data(), y.data(), dResult, n);
    cudaCheck(cudaGetLastError(), "dotInto");
}


void deviceSumMagInto(const DeviceBuffer<scalar>& x, scalar* dResult)
{
    const int n = static_cast<int>(x.size());
    cudaCheck(cudaMemsetAsync(dResult, 0, sizeof(scalar), cudaStreamPerThread), "summagInto zero");
    sumMagKernel<<<nBlocks(n), TPB>>>(x.data(), dResult, n);
    cudaCheck(cudaGetLastError(), "summagInto");
}


void deviceAxpyDev(const scalar* dA, const DeviceBuffer<scalar>& x, DeviceBuffer<scalar>& y)
{
    const int n = static_cast<int>(x.size());
    axpyDevKernel<<<nBlocks(n), TPB>>>(dA, x.data(), y.data(), n);
    cudaCheck(cudaGetLastError(), "axpyDev");
}


void deviceScaleDev(const scalar* dA, DeviceBuffer<scalar>& x)
{
    const int n = static_cast<int>(x.size());
    scaleDevKernel<<<nBlocks(n), TPB>>>(dA, x.data(), n);
    cudaCheck(cudaGetLastError(), "scaleDev");
}


void deviceScalarDiv(const scalar* num, const scalar* den, scalar* out)
{
    scalarDivK<<<1, 1>>>(num, den, out);
    cudaCheck(cudaGetLastError(), "scalarDiv");
}


void deviceScalarDivNeg(const scalar* num, const scalar* den, scalar* out, scalar* outNeg)
{
    scalarDivNegK<<<1, 1>>>(num, den, out, outNeg);
    cudaCheck(cudaGetLastError(), "scalarDivNeg");
}


void deviceScalarCopy(const scalar* src, scalar* dst)
{
    scalarCopyK<<<1, 1>>>(src, dst);
    cudaCheck(cudaGetLastError(), "scalarCopy");
}


void deviceScalarDivConst(const scalar* num, scalar denom, scalar* out)
{
    scalarDivConstK<<<1, 1>>>(num, denom, out);
    cudaCheck(cudaGetLastError(), "scalarDivConst");
}


void deviceScalarAdd2(const scalar* a, const scalar* b, scalar c, scalar* out)
{
    scalarAdd2K<<<1, 1>>>(a, b, c, out);
    cudaCheck(cudaGetLastError(), "scalarAdd2");
}


scalar deviceReadScalar(const scalar* dSrc)
{
    ensureRedScratch();
    if (!g_readPinned) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&g_readPinned), sizeof(scalar)), "read pinned alloc");
    cudaCheck(cudaMemcpy(g_readPinned, dSrc, sizeof(scalar), cudaMemcpyDeviceToHost), "readScalar");
    return *g_readPinned;
}

} // namespace brae
