#pragma once
// cf GPU offload foundation (G0): DeviceBuffer<T>, RAII device memory with explicit H2D/D2H. Explicit
// device buffers (not managed memory) keep the residency discipline visible: every host<->device copy is
// a named call, so the G7 goal (device-resident SIMPLE loop, zero copies per iteration) is measurable.
// The CPU baseline stays the oracle; each GPU kernel is validated against it.
#include "cf_types.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>          // BRAE_POOL_STATS: allocation-size histogram
#include <algorithm>    // sort, for the histogram report
#include <unordered_map>
#include <cstdlib>
#include <cstdio>

namespace brae {

inline void cudaCheck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string("brae cuda: ") + what + ": " + cudaGetErrorString(e));
}

// Entry-count census for the caches keyed on a field pointer, under BRAE_CACHE_STATS=1. Such a cache
// stays bounded only while the pointer is stable: the legacy driver allocates its psi fresh every outer
// iteration, so a cache keyed on it gains an entry per solve (items 75, 76). Prints on each new high mark.
inline void cacheStat(const char* name, std::size_t n)
{
    static const bool on = std::getenv("BRAE_CACHE_STATS") != nullptr;
    if (!on) return;
    static std::unordered_map<std::string, std::size_t> high;
    std::size_t& h = high[name];
    if (n > h) { h = n; std::fprintf(stderr, "[cache] %s entries=%zu\n", name, n); }
}

namespace detail {
// Caching device allocator: blocks freed by a DeviceBuffer are RETAINED in a size-keyed free list and handed back
// to the next same-size request, instead of round-tripping through cudaMalloc/cudaFree on every temporary. The
// device SIMPLE loop churns ~290 malloc+free/iter on per-iteration temporaries (nsys: cudaMalloc+cudaFree = 36.5%
// of host API time after the sync work); once every size is warm (after iter 1) steady-state malloc/free -> ~0.
// Returned memory is NOT zeroed, same contract as cudaMalloc, and cf always writes a buffer before reading it.
// Single-threaded use (cf runs one solve on one host thread; the GPU path is not entered concurrently). Disable
// with BRAE_NO_DEVICE_POOL=1 to A/B against the raw-cudaMalloc behaviour. The pool is intentionally leaked (never
// destructed) so there is no static-destruction-order hazard and no cudaFree after the CUDA context is torn down.
class DevicePool
{
public:
    DevicePool()
    :
        enabled_(std::getenv("BRAE_NO_DEVICE_POOL") == nullptr),
        exact_(std::getenv("BRAE_CONTROL_POOL_EXACT") != nullptr),
        short_(std::getenv("BRAE_CONTROL_POOL_SHORT_CLASS") != nullptr)
    {
        const char* e = std::getenv("BRAE_POOL_KEEP_STEPS");
        const long asked = e ? std::atol(e) : 0;
        if (e && asked < 1)
        {
            throw std::runtime_error(std::string("brae: BRAE_POOL_KEEP_STEPS=") + e + " is not a count of 1 or more.");
        }
        if (asked > 0) keep_ = static_cast<unsigned long>(asked);
    }
    // A REQUEST IS ROUNDED UP TO A SIZE CLASS, and the free list is keyed on the class. Keyed on the exact byte
    // count, a block was reusable only by a request of exactly that size -- right on a mesh that keeps its
    // sizes, where every size is warm after the first step. On a mesh whose sizes CHANGE nothing was ever reused
    // and nothing was ever freed. MEASURED 2026-10-05 after the first step, over the benchmark interval:
    //   RAS/motorBike (refines every step)   839 cudaMalloc a step, 10.5 ms of a 134 ms step; idle list
    //                                        28 MB -> 3,607 MB in 30 steps
    //   damBreakWithObstacle (refines)       680 a step, 15.3 ms of 169; idle 78 MB -> 5,221 MB in 25 steps
    //   RAS/mixerVesselAMI (the AMI's lists) 38 a step; idle 395 MB -> 2,392 MB in 29 steps
    //   RAS/DTCHullMoving (fixed topology)   0.7 a step; idle 683 MB -> 709 MB
    // -- a refining or rotating case run to its end would exhaust the device. The classes are eight to an octave
    // (a request above 1 KiB is rounded up to a multiple of an eighth of the power of two below it; one below
    // to a power of two, 64 bytes at least), so a block is at most 12.5% larger than asked and a size that
    // drifts by a few percent stays in its class.
    //   BRAE_CONTROL_POOL_EXACT=1         the exact-size keys and no trimming, as before
    //   BRAE_CONTROL_POOL_SHORT_CLASS=1   a gate's CONTROL, deliberately wrong: a request is rounded DOWN, so a
    //                                     block is shorter than its buffer (the run is stopped at the first one)
    std::size_t classOf(std::size_t bytes) const
    {
        if (exact_) return bytes;
        std::size_t base = 64;
        while ((base << 1) < bytes)
        {
            base <<= 1;
        }
        if (bytes <= base) return base;
        if (base < 1024) return base << 1;
        const std::size_t step = base >> 3;
        const std::size_t over = bytes - base;
        if (short_) return base + (over/step)*step;
        return base + ((over + step - 1)/step)*step;
    }
    void* take(std::size_t bytes)
    {
        if (bytes == 0) return nullptr;
        const std::size_t cls = classOf(bytes);
        if (cls < bytes)
        {
            throw std::runtime_error(
                "brae device pool: a request of " + std::to_string(bytes) + " bytes was given the class of "
                + std::to_string(cls) + ", which is shorter than the buffer it is for.");
        }
        if (enabled_)
        {
            auto it = free_.find(cls);
            if (it != free_.end() && !it->second.empty())
            {
                void* p = it->second.back().p;
                it->second.pop_back();
                heldBytes_ -= cls;
                return p;
            }
        }
        void* p = nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        cudaCheck(cudaMalloc(&p, cls), "pool cudaMalloc");
        mallocSeconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        mallocBytes_ += cls;
        ++mallocs_;
        hist_[cls].first += cls;
        ++hist_[cls].second;
        return p;
    }
    void give(void* p, std::size_t bytes)
    {
        if (!p) return;
        if (enabled_ && bytes)
        {
            const std::size_t cls = classOf(bytes);
            free_[cls].push_back(Idle{p, step_});
            heldBytes_ += cls;
        }
        else
        {
            cudaFree(p);
        }
    }
    // A STEP OF THE CALLER'S LOOP HAS ENDED: a block that has sat idle for `keep` steps is handed back to the
    // driver. A step's own temporaries are taken and given at every step, so what stays idle that long is a size
    // the run has moved away from -- the mesh before a change, the start-up's work arrays. A class's idle blocks
    // are taken from the back and given to the back, so the oldest are at the front. Not called by a loop that
    // wants the list kept for good (every block is then retained, as before); never under the exact-size keys.
    void endOfStep()
    {
        ++step_;
        if (!enabled_ || exact_ || step_ <= keep_) return;
        const auto t0 = std::chrono::steady_clock::now();
        for (auto& kv : free_)
        {
            std::vector<Idle>& idle = kv.second;
            std::size_t n = 0;
            while (n < idle.size() && idle[n].step + keep_ < step_)
            {
                cudaFree(idle[n].p);
                ++n;
            }
            if (n == 0) continue;
            idle.erase(idle.begin(), idle.begin() + static_cast<std::ptrdiff_t>(n));
            heldBytes_ -= n*kv.first;
            freedBytes_ += n*kv.first;
            frees_ += n;
        }
        freeSeconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    // what the pool has asked the driver for so far: the calls, the seconds in them, and the bytes idle in the
    // free list -- for a caller that reports them per step (the interFoam device loop, BRAE_INTER_PHASE_TIME)
    std::size_t mallocs() const
    {
        return mallocs_;
    }
    double mallocSeconds() const
    {
        return mallocSeconds_;
    }
    std::size_t heldBytes() const
    {
        return heldBytes_;
    }
    std::size_t mallocBytes() const
    {
        return mallocBytes_;
    }
    std::size_t frees() const
    {
        return frees_;
    }
    double freeSeconds() const
    {
        return freeSeconds_;
    }
    std::size_t freedBytes() const
    {
        return freedBytes_;
    }
    bool exactKeys() const
    {
        return exact_;
    }
    // BRAE_POOL_STATS=1: what the pool asked the driver for, and how much of that is sitting in the
    // free list rather than in use.
    void report(const char* when) const
    {
        if (!std::getenv("BRAE_POOL_STATS")) return;
        std::size_t sizes = free_.size(), blocks = 0;
        for (const auto& kv : free_) blocks += kv.second.size();
        std::fprintf(stderr,
            "[pool] %-10s cudaMalloc %8.3f GB in %zu calls | idle in free list %8.3f GB "
            "(%zu distinct sizes, %zu blocks)\n",
            when, mallocBytes_/1073741824.0, mallocs_, heldBytes_/1073741824.0, sizes, blocks);
        // The biggest consumers, by total bytes. Sizes are the attribution: a buffer is sized by what it
        // spans, so nCells*8 is a cell field, nFaces*8 a face field, and the tail of odd sizes is the
        // AMG hierarchy's coarse levels. Cheaper and more honest than guessing at call sites.
        std::vector<std::pair<std::size_t, std::pair<std::size_t, std::size_t>>> v(hist_.begin(), hist_.end());
        std::sort(v.begin(), v.end(),
                  [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        const std::size_t n = v.size() < 12 ? v.size() : 12;
        for (std::size_t i = 0; i < n; ++i)
        {
            std::fprintf(stderr, "[pool]   %10.2f MB total  %5zu x %10.2f MB\n",
                         v[i].second.first/1048576.0, v[i].second.second, v[i].first/1048576.0);
        }
    }
private:
    struct Idle
    {
        void*         p = nullptr;
        unsigned long step = 0;
    };
    std::unordered_map<std::size_t, std::vector<Idle>> free_;
    std::size_t mallocBytes_ = 0, heldBytes_ = 0, mallocs_ = 0;
    std::size_t frees_ = 0;
    std::size_t freedBytes_ = 0;
    double mallocSeconds_ = 0;
    double freeSeconds_ = 0;
    unsigned long step_ = 0;
    // steps a block may sit idle before it goes back to the driver (BRAE_POOL_KEEP_STEPS)
    unsigned long keep_ = 4;
    std::map<std::size_t, std::pair<std::size_t, std::size_t>> hist_;   // size -> {total bytes, count}
    bool enabled_;
    bool exact_;
    bool short_;
};
inline DevicePool& devicePool() { static DevicePool* p = new DevicePool(); return *p; }   // intentionally leaked
} // namespace detail

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { resize(n); }
    explicit DeviceBuffer(const std::vector<T>& h) { resize(h.size()); copyFrom(h); }
    ~DeviceBuffer() { if (d_) detail::devicePool().give(d_, n_ * sizeof(T)); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : d_(o.d_), n_(o.n_) { o.d_ = nullptr; o.n_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept
    {
        if (this != &o)
        {
            if (d_) detail::devicePool().give(d_, n_ * sizeof(T));
            d_ = o.d_;
            n_ = o.n_;
            o.d_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    void resize(std::size_t n)
    {
        if (n == n_) return;
        if (d_) detail::devicePool().give(d_, n_ * sizeof(T));            // return OLD block (OLD n_) to the pool
        n_ = n;
        d_ = nullptr;
        if (n_) d_ = static_cast<T*>(detail::devicePool().take(n_ * sizeof(T)));
    }
    void copyFrom(const std::vector<T>& h)                                // H2D
    {
        if (h.size() != n_) resize(h.size());
        cudaCheck(cudaMemcpy(d_, h.data(), n_ * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    }
    void copyTo(std::vector<T>& h) const                                 // D2H
    {
        h.resize(n_);
        cudaCheck(cudaMemcpy(h.data(), d_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
    }
    std::vector<T> host() const { std::vector<T> h; copyTo(h); return h; }

    T*       data()       { return d_; }
    const T* data() const { return d_; }
    std::size_t size() const { return n_; }

private:
    T*          d_ = nullptr;
    std::size_t n_ = 0;
};

} // namespace brae
