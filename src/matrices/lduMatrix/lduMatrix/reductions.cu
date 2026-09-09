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
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <string>

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
scalar* g_partials = nullptr;     // stage-1 block partials; grown on demand, never shrunk
int     g_partialsCap = 0;
int     g_partialsEpoch = 0;      // bumped on every regrow: a captured graph holding the old pointer must rebuild
inline scalar* ensurePartials(int nb)
{
    if (nb > g_partialsCap)
    {
        if (g_partials) cudaCheck(cudaFree(g_partials), "partials free");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g_partials), (std::size_t)nb*sizeof(scalar)), "partials alloc");
        g_partialsCap = nb;
        ++g_partialsEpoch;
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


// MIN, MAX AND SUM IN ONE PASS, the three numbers OpenFOAM's bound() needs (bound.C:38-46: it computes
// min unconditionally as its guard and prints all three when the guard fires). One kernel rather than
// three because bound() runs on every turbulence solve and the field is read once either way; the
// partials are interleaved [min|max|sum] so the second stage sees three contiguous blocks.
__global__
void minMaxSumKernel(const scalar* __restrict__ x, scalar* result, int n, int nb)
{
    __shared__ scalar smin[TPB];
    __shared__ scalar smax[TPB];
    __shared__ scalar ssum[TPB];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in = (i < n);
    smin[tid] = in ? x[i] :  1e300;
    smax[tid] = in ? x[i] : -1e300;
    ssum[tid] = in ? x[i] :  0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smin[tid] = fmin(smin[tid], smin[tid + s]);
            smax[tid] = fmax(smax[tid], smax[tid + s]);
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        result[blockIdx.x]          = smin[0];      // fixed slots, not atomics -- see the note at the top
        result[nb + blockIdx.x]     = smax[0];
        result[2 * nb + blockIdx.x] = ssum[0];
    }
}


// ...and the sum becomes the mean, on the device, so the host reads three finished numbers.
__global__
void scaleMeanKernel(scalar* out, int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) out[2] /= static_cast<scalar>(n);
}


// Stage 2 for that one: the same single-block, fixed-order walk, three times over its own third.
__global__
void finalMinMaxSumKernel(const scalar* __restrict__ partials, scalar* out, int nb)
{
    __shared__ scalar sd[TPB];
    const int tid = threadIdx.x;
    const int which = blockIdx.x;                   // 0 = min, 1 = max, 2 = sum
    const scalar ident = (which == 0) ? 1e300 : (which == 1) ? -1e300 : 0.0;
    scalar acc = ident;
    for (int i = tid; i < nb; i += blockDim.x)
    {
        const scalar v = partials[which * nb + i];
        acc = (which == 0) ? fmin(acc, v) : (which == 1) ? fmax(acc, v) : acc + v;
    }
    sd[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            const scalar v = sd[tid + s];
            sd[tid] = (which == 0) ? fmin(sd[tid], v) : (which == 1) ? fmax(sd[tid], v) : sd[tid] + v;
        }
        __syncthreads();
    }
    if (tid == 0) out[which] = sd[0];
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

// min, max and MEAN of x, into three consecutive device scalars. The mean is the arithmetic one over the
// internal field, which is what OpenFOAM's bound() prints (gAverage(vsf.primitiveField()), not a
// volume-weighted average).
void deviceMinMaxMeanInto(const DeviceBuffer<scalar>& x, scalar* dOut3)
{
    const int n = static_cast<int>(x.size());
    if (n <= 0) { cudaCheck(cudaMemsetAsync(dOut3, 0, 3*sizeof(scalar), cudaStreamPerThread), "minmax zero"); return; }
    const int nb = nBlocks(n);
    scalar* part = ensurePartials(3 * nb);
    minMaxSumKernel<<<nb, TPB>>>(x.data(), part, n, nb);
    finalMinMaxSumKernel<<<3, TPB>>>(part, dOut3, nb);
    scaleMeanKernel<<<1, 1>>>(dOut3, n);
    cudaCheck(cudaGetLastError(), "minMaxMeanInto");
}


void deviceSumMagInto(const DeviceBuffer<scalar>& x, scalar* dResult)
{
    const int n = static_cast<int>(x.size());
    reduceInto([&](int nb, scalar* part){ sumMagKernel<<<nb, TPB>>>(x.data(), part, n); }, n, dResult);
    cudaCheck(cudaGetLastError(), "summagInto");
}


int deviceReductionScratchEpoch()
{
    return g_partialsEpoch;
}


namespace {

// THE READ-BACK MAILBOX. deviceReadScalar was a blocking cudaMemcpy, and the driver returned from it only
// after it had walked the whole queue. nsys on the 306k compressible case (steady-state iterations, the
// cold one excluded) put 306 of the 425 device-to-host copies in one outer iteration on single scalars
// like this one -- convergence checks -- and every GPU-idle gap over 5 us ended at one of them: 17 gaps
// totalling 7.19 ms in the pressure phase, 22 totalling 6.26 ms in turbulence, out of a 50.1 ms iteration
// that was 38% idle. The momentum phase was the outlier at 10 gaps totalling 0.06 ms BECAUSE its per-pass
// read had already been replaced by the mailbox in device_colour_gauss_seidel.cu (18.2 -> 14.1 ms per
// outer iteration). This is that mailbox generalised, so every solver's read gets it and not just the
// momentum one.
//
// A one-thread kernel copies the value into mapped pinned host memory, fences it system-wide, then
// publishes an incrementing sequence number; the host spins on the number. The kernel is enqueued on
// cudaStreamPerThread at exactly the point in the stream the cudaMemcpy occupied, so it snapshots the
// same value the copy would have copied and NO CALLER CHANGES:
//   * whatever produced the value was enqueued earlier on this stream, so the publish runs after it;
//   * whatever overwrites the value later is enqueued after the publish, so it cannot reach the value
//     the publish already read;
//   * the host does not return until the publish has run, so a read is never in flight across a call --
//     one host thread has at most one outstanding publish, and the mailbox is per host thread anyway.
// Every kernel in brae is launched on the per-thread stream (there is no cudaStreamCreate in src/), and
// the whole tree compiles with --default-stream per-thread, so "ordered on this stream" is the same
// ordering the blocking copy had. Measured on this box: a producer on an unrelated stream is NOT waited
// on by the blocking copy either, so the mailbox is stale exactly where the copy was stale
// (tests/test_scalar_mailbox holds both modes to the same answer there).
//
// Measured here, 179 timed reads of a 305,760-cell reduction on GB10: with nothing else queued the
// mailbox takes 21.3 us mean / 20.6 us best against the blocking copy's 269.4 us mean / 20.8 us best --
// the same GPU work, and the whole difference is the driver's return. Behind 40 queued 305k kernels:
// 205.4 us mean against 522.1 us.
//
// It carries a GROUP of values, not one: the graph solvers report a residual AND an iteration count, and
// the fused Gauss-Seidel one of each per component, and each group was a run of async D2H copies with a
// single cudaStreamSynchronize behind it. One publish reads the whole group in one thread at the point
// the first copy occupied. That is the same snapshot the copies took: no stream work ran between them
// either, so no source could change between the first value and the last in either scheme.
struct ValueMailbox
{
    unsigned long long value[DEVICE_READ_MAX_VALUES];
    unsigned long long seq;
};

static_assert(sizeof(scalar) == sizeof(unsigned long long), "the mailbox publishes a scalar as raw bits");
static_assert(sizeof(int) == 4, "the mailbox publishes an int as its low 32 bits");

// The sources travel in the kernel's parameter block rather than a device array: a device array would be
// one more allocation to keep alive for the length of a solve, and eight pointers plus eight flags is 96
// bytes against the 4 KB a launch may carry.
struct PublishSources
{
    const void* p[DEVICE_READ_MAX_VALUES];
    int isInt[DEVICE_READ_MAX_VALUES];
};

__global__
void publishValuesKernel(
    PublishSources src,
    int n,
    ValueMailbox* box,
    unsigned long long seq)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    // Copied as raw bits. A double load/store preserves them too, but the bit copy leaves no room for a
    // signalling NaN to be canonicalised on its way through a register: the contract is that this returns
    // exactly what the memcpy would have returned, for any producer. An int rides in the low 32 bits.
    for (int i = 0; i < n; ++i)
    {
        box->value[i] = src.isInt[i]
            ? (unsigned long long)(*reinterpret_cast<const unsigned int*>(src.p[i]))
            : *reinterpret_cast<const unsigned long long*>(src.p[i]);
    }
    __threadfence_system();
    *reinterpret_cast<volatile unsigned long long*>(&box->seq) = seq;
}

// Allocated once and never freed: no static destructor may run after the CUDA context is torn down (the
// same rule the other device caches here follow). thread_local rather than one global because the stream
// it publishes on is cudaStreamPerThread -- one mailbox per stream is what keeps two host threads' reads
// from overwriting each other's values and sequence number.
ValueMailbox* valueMailbox(ValueMailbox** devPtr)
{
    thread_local ValueMailbox* box = nullptr;
    thread_local ValueMailbox* boxDev = nullptr;
    if (!boxDev)
    {
        ValueMailbox* h = nullptr;
        ValueMailbox* d = nullptr;
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&h), sizeof(ValueMailbox), cudaHostAllocMapped),
                  "read mailbox alloc");
        std::memset(h, 0, sizeof(ValueMailbox));
        cudaCheck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), h, 0),
                  "read mailbox device pointer");
        box = h;                                  // both set only once BOTH calls succeeded, so a partial
        boxDev = d;                               // failure retries rather than handing back a null device pointer
    }
    *devPtr = boxDev;
    return box;
}

// The spin, bounded. Past two seconds it falls back to the stream sync and, if the number still has not
// arrived, throws: a wedge is reported, never waited on forever. Same bound as the momentum mailbox.
// `what` names the caller, so the message says which read gave up.
void waitForMailboxSequence(
    const ValueMailbox* box,
    unsigned long long seq,
    const char* what)
{
    const volatile unsigned long long* p = reinterpret_cast<const volatile unsigned long long*>(&box->seq);
    const auto t0 = std::chrono::steady_clock::now();
    while (*p != seq)
    {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2))
        {
            cudaCheck(cudaStreamSynchronize(cudaStreamPerThread), "read mailbox fallback sync");
            if (*p != seq)
            {
                throw std::runtime_error(std::string("brae ") + what + ": the read-back mailbox never "
                                         "carried this read's sequence number, even after a stream sync");
            }
            break;
        }
    }
    // The values were written before the number on the device side; the acquire fence keeps this side's
    // reads of value[] after its read of seq (the host is an aarch64 here, which reorders loads).
    std::atomic_thread_fence(std::memory_order_acquire);
}

// BRAE_READ_SCALAR_SYNC=1 restores the blocking copies: an escape hatch for measuring the two against
// each other and for a machine whose host spin behaves badly. Both paths return the same bits.
bool readScalarSync()
{
    static const bool on = std::getenv("BRAE_READ_SCALAR_SYNC") != nullptr;
    return on;
}

// The mailbox lives in MAPPED pinned memory, so a device that cannot map host memory has nothing to spin
// on. That machine takes the blocking copy and is told why, rather than failing to allocate.
bool readScalarUsesMailbox()
{
    static const bool on = []()
    {
        if (readScalarSync()) return false;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        int canMap = 0;
        if (cudaDeviceGetAttribute(&canMap, cudaDevAttrCanMapHostMemory, dev) != cudaSuccess) return false;
        return canMap != 0;
    }();
    return on;
}

void announceReadScalarMode()
{
    static bool announced = false;
    if (announced) return;
    announced = true;
    std::printf(readScalarUsesMailbox()
        ? "  scalar read-back: mailbox, host spins on a published sequence number; BRAE_READ_SCALAR_SYNC=1 restores the blocking copy\n"
        : "  scalar read-back: blocking cudaMemcpy (BRAE_READ_SCALAR_SYNC, or this device cannot map host memory)\n");
}

// The one read-back, for one value or several. Every caller's group used to be a run of async D2H copies
// with one cudaStreamSynchronize behind it; the mailbox replaces the whole group with one publish and one
// wait, at the point in the stream the first copy occupied.
void readValues(
    const DeviceReadValue* values,
    int n,
    const char* what)
{
    announceReadScalarMode();
    if (!values || n <= 0 || n > DEVICE_READ_MAX_VALUES)
    {
        throw std::runtime_error(std::string("brae ") + what + ": asked to read " + std::to_string(n)
                                 + " device values in one wait; the mailbox carries 1 to "
                                 + std::to_string(DEVICE_READ_MAX_VALUES));
    }
    for (int i = 0; i < n; ++i)
    {
        if (!values[i].dSrc || !values[i].hDst)
        {
            // The blocking copy returned an invalid-argument error here; the publish kernel would instead
            // fault and poison the context, so name it before launching anything.
            throw std::runtime_error(std::string("brae ") + what + ": asked to read a null device value");
        }
    }
    if (!readScalarUsesMailbox())
    {
        // Exactly the copies and the single sync the converted sites used to write out themselves: same
        // stream, same order, so this mode is the mailbox's oracle rather than a different read.
        for (int i = 0; i < n; ++i)
        {
            cudaCheck(cudaMemcpyAsync(values[i].hDst,
                                      values[i].dSrc,
                                      values[i].isInt ? sizeof(int) : sizeof(scalar),
                                      cudaMemcpyDeviceToHost,
                                      cudaStreamPerThread),
                      "readValues D2H");
        }
        cudaCheck(cudaStreamSynchronize(cudaStreamPerThread), "readValues sync");
        return;
    }
    // A capture cannot be spun on: the publish kernel would be recorded into the graph instead of run, so
    // the host would wait out the two seconds for a number nobody is going to write. The blocking copy
    // this replaced failed outright under capture, so naming the situation here loses nothing.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaCheck(cudaStreamIsCapturing(cudaStreamPerThread, &cap), "readValues capture query");
    if (cap != cudaStreamCaptureStatusNone)
    {
        throw std::runtime_error(std::string("brae ") + what + ": called while cudaStreamPerThread is "
                                 "capturing a graph. A value cannot be read to the host from inside a "
                                 "capture -- keep it device-resident, or read it outside the capture.");
    }
    PublishSources src;
    for (int i = 0; i < DEVICE_READ_MAX_VALUES; ++i)
    {
        src.p[i] = (i < n) ? values[i].dSrc : nullptr;
        src.isInt[i] = (i < n && values[i].isInt) ? 1 : 0;
    }
    ValueMailbox* boxDev = nullptr;
    ValueMailbox* box = valueMailbox(&boxDev);
    // 64-bit and per-thread, so it cannot wrap and cannot be shared: one read per nanosecond would take
    // 584 years to come back to a number the host is still waiting for. Read as one aligned 8-byte word,
    // so it cannot be observed torn either.
    thread_local unsigned long long seqCounter = 0;
    const unsigned long long seq = ++seqCounter;
    publishValuesKernel<<<1, 1, 0, cudaStreamPerThread>>>(src, n, boxDev, seq);
    cudaCheck(cudaGetLastError(), "readValues publish");
    waitForMailboxSequence(box, seq, what);
    for (int i = 0; i < n; ++i)
    {
        const unsigned long long bits = reinterpret_cast<const volatile unsigned long long*>(box->value)[i];
        if (values[i].isInt)
        {
            const unsigned int lo = static_cast<unsigned int>(bits);
            std::memcpy(values[i].hDst, &lo, sizeof(int));
        }
        else
        {
            std::memcpy(values[i].hDst, &bits, sizeof(scalar));
        }
    }
}

} // namespace


void deviceReadValues(
    const DeviceReadValue* values,
    int n)
{
    readValues(values, n, "deviceReadValues");
}


scalar deviceReadScalar(const scalar* dSrc)
{
    scalar v = 0;
    DeviceReadValue one;
    one.dSrc = dSrc;
    one.hDst = &v;
    one.isInt = false;
    readValues(&one, 1, "deviceReadScalar");
    return v;
}


void deviceReadScalarWaitProbe()
{
    ValueMailbox* boxDev = nullptr;
    ValueMailbox* box = valueMailbox(&boxDev);
    (void)boxDev;
    // A number no publish kernel will ever carry, and none is launched: the wait has to give up on the
    // clock, not on the queue. The read counter is untouched, so reads after this one still work.
    waitForMailboxSequence(box, ~0ull, "deviceReadScalar");
}

} // namespace brae
