// cf GPU offload -- BLAS-1 REDUCTIONS: dot / sum(|x|) / max-ratio, the device-resident "Into" variants, and the
// device->host scalar read-back. Split from device_blas.cu (elementwise ops in blas1.cu).
//
// THE SUMS ARE TWO-STAGE AND DETERMINISTIC, not atomicAdd. They used to accumulate each block's partial into one
// device scalar with atomicAdd, which makes the summation order depend on the order blocks happen to finish -- so
// the SAME binary on the SAME input returned a different last bit every run.
//
// That is not a cosmetic reproducibility complaint. Every Krylov solver takes its stopping decision from one of
// these reductions, so a last-bit difference decides which iteration crosses the tolerance. Measured on
// compressible/rhoSimpleFoam/squareBend (transonic -> the asymmetric pressure matrix on Jacobi-BiCGStab, relTol
// 0.1): the pressure solve stopped at 484 iterations with a final residual of 5.70e-02 on one run and 480 with
// 9.75e-02 on the next. At that residual the iterate is nowhere near converged, so the two runs' velocity fields
// differed by 2.6% on all 112000 cells -- from the same binary and the same case.
//
// Stage 1 writes block partials to a fixed slot (partials[blockIdx.x]); stage 2 reduces that array in ONE block,
// in index order. Both stages have a fixed traversal, so the result is bit-identical run to run. max is exempt:
// it is associative and exact in floating point, so atomicMax was already deterministic.
//
// This does NOT make brae bit-reproducible on its own. There are ~68 other atomicAdd sites, most of them
// scatter-accumulates into per-cell arrays (out[own[f]] += ...), which are order-dependent for the same reason.
// What this fixes is the reduction that the convergence test reads.
#include <map>
#include "device_blas.cuh"
#include <cuda_runtime.h>
#include <cstddef>

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
scalar* g_partials = nullptr;     // stage-1 block partials; grown on demand, never shrunk
int     g_partialsCap = 0;
inline scalar* ensurePartials(int nb)
{
    if (nb > g_partialsCap)
    {
        if (g_partials) cudaCheck(cudaFree(g_partials), "partials free");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g_partials), (std::size_t)nb*sizeof(scalar)), "partials alloc");
        g_partialsCap = nb;
    }
    return g_partials;
}
inline void ensureRedScratch()
{
    if (!g_redDev)    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g_redDev), sizeof(scalar)), "red dev alloc");
    if (!g_redPinned) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&g_redPinned), sizeof(scalar)), "red pinned alloc");
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
    if (tid == 0) result[blockIdx.x] = sdata[0];   // fixed slot, not atomicAdd -- see the note at the top
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
    if (tid == 0) result[blockIdx.x] = sdata[0];   // fixed slot, not atomicAdd
}


// Stage 2: sum `partials[0..nb)` in ONE block, in index order. A single block with a fixed grid-stride walk and a
// fixed shared-memory tree visits the values in the same order on every launch, which is what makes the whole
// reduction bit-reproducible.
__global__
void finalSumKernel(const scalar* __restrict__ partials, scalar* out, int nb)
{
    __shared__ scalar sdata[TPB];
    const int tid = threadIdx.x;
    scalar acc = 0;
    for (int i = tid; i < nb; i += TPB) acc += partials[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = TPB / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = sdata[0];
}

// One deterministic sum reduction: stage 1 into partials, stage 2 into `dResult`. No memset needed -- stage 2
// assigns rather than accumulates.
template <class K>
inline void reduceInto(K launchStage1, int n, scalar* dResult)
{
    if (n <= 0) { cudaCheck(cudaMemsetAsync(dResult, 0, sizeof(scalar), cudaStreamPerThread), "reduce zero"); return; }
    const int nb = nBlocks(n);
    scalar* part = ensurePartials(nb);
    launchStage1(nb, part);
    finalSumKernel<<<1, TPB>>>(part, dResult, nb);
}
} // namespace


scalar deviceDot(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y)
{
    const int n = static_cast<int>(x.size());
    ensureRedScratch();
    reduceInto([&](int nb, scalar* part){ dotKernel<<<nb, TPB>>>(x.data(), y.data(), part, n); }, n, g_redDev);
    cudaCheck(cudaGetLastError(), "dot");
    cudaCheck(cudaMemcpy(g_redPinned, g_redDev, sizeof(scalar), cudaMemcpyDeviceToHost), "dot result");
    return *g_redPinned;
}


// max over cells of x/y, skipping y <= 0. This is the Courant reduction's only genuinely new shape:
// sum and dot already exist, but the Courant NUMBER is a maximum, and a maximum cannot be assembled
// from them. Kept as ratio-of-two-arrays rather than max(x) so the division happens in the same pass
// and no per-cell ratio array is ever materialised.
__global__
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
    maxRatioKernel<<<nBlocks(n), TPB>>>(x.data(), y.data(), g_redDev, n);
    cudaCheck(cudaGetLastError(), "maxratio");
    cudaCheck(cudaMemcpy(g_redPinned, g_redDev, sizeof(scalar), cudaMemcpyDeviceToHost), "maxratio result");
    return *g_redPinned;
}


scalar deviceSumMag(const DeviceBuffer<scalar>& x)
{
    const int n = static_cast<int>(x.size());
    ensureRedScratch();
    reduceInto([&](int nb, scalar* part){ sumMagKernel<<<nb, TPB>>>(x.data(), part, n); }, n, g_redDev);
    cudaCheck(cudaGetLastError(), "summag");
    cudaCheck(cudaMemcpy(g_redPinned, g_redDev, sizeof(scalar), cudaMemcpyDeviceToHost), "summag result");
    return *g_redPinned;
}


// device-resident scalar plumbing (no host sync): the reduction writes into a caller-owned device scalar.
void deviceDotInto(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, scalar* dResult)
{
    const int n = static_cast<int>(x.size());
    reduceInto([&](int nb, scalar* part){ dotKernel<<<nb, TPB>>>(x.data(), y.data(), part, n); }, n, dResult);
    cudaCheck(cudaGetLastError(), "dotInto");
}


const DeviceBuffer<scalar>& deviceOnes(int n)
{
    // Leaked on purpose, like the other device caches: no static destructor may run after the CUDA
    // context is torn down.
    static auto& cache = *new std::map<int, DeviceBuffer<scalar>>();
    auto it = cache.find(n);
    if (it == cache.end())
    {
        DeviceBuffer<scalar> ones;
        ones.copyFrom(std::vector<scalar>(static_cast<std::size_t>(n), scalar(1)));
        it = cache.emplace(n, std::move(ones)).first;
    }
    return it->second;
}

void deviceSumMagInto(const DeviceBuffer<scalar>& x, scalar* dResult)
{
    const int n = static_cast<int>(x.size());
    reduceInto([&](int nb, scalar* part){ sumMagKernel<<<nb, TPB>>>(x.data(), part, n); }, n, dResult);
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
