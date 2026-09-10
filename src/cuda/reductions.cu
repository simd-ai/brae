// cf GPU offload -- BLAS-1 REDUCTIONS: block-reduction dot / sum(|x|) (shared memory + atomicAdd to a device
// accumulator), the device-resident "Into" variants, and the device->host scalar read-back. The reduction order
// differs from the CPU sequential sum, so results match to machine precision (FP non-associativity), not
// bit-for-bit -- the validation criterion for reductions. Split from device_blas.cu (elementwise ops in blas1.cu).
//
// Kernels' static __shared__ arrays work the same as __device__ functions here as they did as __global__.
#include "device_blas.cuh"
#include "pcuda_compat.cuh"
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


__device__
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


__device__
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
} // namespace


scalar deviceDot(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y)
{
    const int n = static_cast<int>(x.size());
    ensureRedScratch();
    cudaCheck(cudaMemsetAsync(g_redDev, 0, sizeof(scalar), cudaStreamPerThread), "dot zero");
    const scalar* xd = x.data();
    const scalar* yd = y.data();
    scalar* result = g_redDev;
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { dotKernel(xd, yd, result, n); });
    cudaCheck(cudaGetLastError(), "dot");
    cudaCheck(cudaMemcpy(g_redPinned, g_redDev, sizeof(scalar), cudaMemcpyDeviceToHost), "dot result");
    return *g_redPinned;
}


// max over cells of x/y, skipping y <= 0. This is the Courant reduction's only genuinely new shape:
// sum and dot already exist, but the Courant NUMBER is a maximum, and a maximum cannot be assembled
// from them. Kept as ratio-of-two-arrays rather than max(x) so the division happens in the same pass
// and no per-cell ratio array is ever materialised.
__device__
void maxRatioKernel(const scalar* __restrict__ x, const scalar* __restrict__ y, scalar* result, int n)
{
    __shared__ scalar sdata[TPB];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n && y[i] > scalar(0)) ? (x[i]/y[i]) : scalar(0);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMax((unsigned long long*)result, __double_as_longlong(sdata[0]));
}

scalar deviceMaxRatio(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y)
{
    const int n = static_cast<int>(x.size());
    if (n == 0 || (int)y.size() < n) return 0;
    ensureRedScratch();
    cudaCheck(cudaMemsetAsync(g_redDev, 0, sizeof(scalar), cudaStreamPerThread), "maxratio zero");
    const scalar* xd = x.data();
    const scalar* yd = y.data();
    scalar* result = g_redDev;
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { maxRatioKernel(xd, yd, result, n); });
    cudaCheck(cudaGetLastError(), "maxratio");
    cudaCheck(cudaMemcpy(g_redPinned, g_redDev, sizeof(scalar), cudaMemcpyDeviceToHost), "maxratio result");
    return *g_redPinned;
}


scalar deviceSumMag(const DeviceBuffer<scalar>& x)
{
    const int n = static_cast<int>(x.size());
    ensureRedScratch();
    cudaCheck(cudaMemsetAsync(g_redDev, 0, sizeof(scalar), cudaStreamPerThread), "summag zero");
    const scalar* xd = x.data();
    scalar* result = g_redDev;
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { sumMagKernel(xd, result, n); });
    cudaCheck(cudaGetLastError(), "summag");
    cudaCheck(cudaMemcpy(g_redPinned, g_redDev, sizeof(scalar), cudaMemcpyDeviceToHost), "summag result");
    return *g_redPinned;
}


// device-resident scalar plumbing (no host sync): the reduction writes into a caller-owned device scalar.
void deviceDotInto(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, scalar* dResult)
{
    const int n = static_cast<int>(x.size());
    cudaCheck(cudaMemsetAsync(dResult, 0, sizeof(scalar), cudaStreamPerThread), "dotInto zero");
    const scalar* xd = x.data();
    const scalar* yd = y.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { dotKernel(xd, yd, dResult, n); });
    cudaCheck(cudaGetLastError(), "dotInto");
}


void deviceSumMagInto(const DeviceBuffer<scalar>& x, scalar* dResult)
{
    const int n = static_cast<int>(x.size());
    cudaCheck(cudaMemsetAsync(dResult, 0, sizeof(scalar), cudaStreamPerThread), "summagInto zero");
    const scalar* xd = x.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { sumMagKernel(xd, dResult, n); });
    cudaCheck(cudaGetLastError(), "summagInto");
}


scalar deviceReadScalar(const scalar* dSrc)
{
    ensureRedScratch();
    if (!g_readPinned) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&g_readPinned), sizeof(scalar)), "read pinned alloc");
    cudaCheck(cudaMemcpy(g_readPinned, dSrc, sizeof(scalar), cudaMemcpyDeviceToHost), "readScalar");
    return *g_readPinned;
}

} // namespace brae
