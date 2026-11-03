#pragma once
// BRAE_MEMORY_PROBE=<n,m,...>: what a run holds, read inside the process at fixed iterations of ONE run.
//
// WHY NOT THE PEAK AT EXIT. host_smoother_memory compared the kernel's peak resident set at exit (/usr/bin/time's
// %M, ru_maxrss) between a 100- and a 400-iteration run. On the JUPITER login node (GH200, 72 CPUs, 64 KiB pages,
// 2026-11-03) that figure was 189 to 243 MB between identical runs of one binary, every value a multiple of 9 MiB,
// while VmHWM read from /proc during the loop stayed flat to 0.1 MB from iteration ~64 to 361. 9 MiB is 144 pages of
// 64 KiB, 2 x 72 CPUs: the batch of a per-CPU counter, so the exit figure moves by that counter's slack and not by
// what the run held. Read here, the iterations are fixed and the run is the same one.
//
// One line a probe, on stdout:
//   brae memory: at=<start-up|iteration|loop-end|end> iteration=<n> pool_reserved=<B> pool_allocated=<B>
//       pool_peak_reserved=<B> device_used=<B> device_total=<B> rss_kb=<kB> hwm_kb=<kB> rss_exact_kb=<kB>
//       host_used_kb=<kB>
//   pool_reserved       the device pool's footprint: what it has cudaMalloc'ed less what it handed back
//   pool_allocated      that less its idle blocks: what the DeviceBuffers alive now span
//   pool_peak_reserved  the most pool_reserved has been so far
//   device_used         cudaMemGetInfo's total less free: the whole device, the context and other processes included
//   rss_kb, hwm_kb      the process: /proc/self/status VmRSS and VmHWM
//   rss_exact_kb        the process: /proc/self/smaps_rollup Rss, a walk of the page tables, not a counter
//   host_used_kb        the machine: /proc/meminfo MemTotal less MemAvailable. Page-locked memory comes out of the
//                       machine (item 68's leak was pinned), but other users' processes are in this figure too.
// start-up is after the set-up and before the first iteration, loop-end after the last iteration, end after the
// final write. The pool figures are -1 under BRAE_NO_DEVICE_POOL=1, whose frees the pool does not see.
//
// BRAE_CONTROL_MEMORY_LEAK=<host|device>:<MiB>, host_smoother_memory's CONTROL and deliberately wrong: every
// iteration keeps that many MiB for good -- host memory written so that it is resident, or a device-pool block that
// is never handed back.
#include "device_buffer.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

class MemoryProbe
{
public:
    MemoryProbe()
    {
        const char* e = std::getenv("BRAE_MEMORY_PROBE");
        on_ = e != nullptr;
        if (on_)
        {
            const std::string list(e);
            std::size_t i = 0;
            while (i < list.size())
            {
                std::size_t j = list.find(',', i);
                if (j == std::string::npos)
                {
                    j = list.size();
                }
                const std::string item = list.substr(i, j - i);
                char* stop = nullptr;
                const long n = std::strtol(item.c_str(), &stop, 10);
                if (item.empty() || *stop != '\0' || n < 1)
                {
                    throw std::runtime_error("brae: BRAE_MEMORY_PROBE=" + list
                                             + " is not a comma list of iterations of 1 or more.");
                }
                iters_.insert(n);
                i = j + 1;
            }
        }
        if (const char* c = std::getenv("BRAE_CONTROL_MEMORY_LEAK"))
        {
            const std::string spec(c);
            const std::size_t colon = spec.find(':');
            const std::string kind = spec.substr(0, colon);
            const long mib = colon == std::string::npos ? 0 : std::atol(spec.c_str() + colon + 1);
            if ((kind != "host" && kind != "device") || mib < 1)
            {
                throw std::runtime_error("brae: BRAE_CONTROL_MEMORY_LEAK=" + spec
                                         + " is not <host|device>:<MiB of 1 or more>.");
            }
            leakDevice_ = kind == "device";
            leakBytes_ = static_cast<std::size_t>(mib) << 20;
            std::printf("brae CONTROL: BRAE_CONTROL_MEMORY_LEAK=%s, %ld MiB of %s memory kept at every iteration\n",
                        spec.c_str(),
                        mib,
                        kind.c_str());
        }
    }
    // after the set-up, before the first iteration
    void startUp()
    {
        print("start-up", 0);
    }
    // the end of iteration `iter`: the control's leak, then the line when the iteration is listed
    void iteration(
        long iter)
    {
        leak();
        if (iters_.count(iter))
        {
            print("iteration", iter);
        }
    }
    void loopEnd(
        long iter)
    {
        print("loop-end", iter);
    }
    void end(
        long iter)
    {
        print("end", iter);
    }
private:
    // the number on the line of `file` that starts with `key`, in the kB those files use; -1 when it is absent
    static long readKb(
        const char* file,
        const char* key)
    {
        std::ifstream in(file);
        std::string line;
        const std::size_t n = std::strlen(key);
        while (std::getline(in, line))
        {
            if (line.compare(0, n, key) == 0 && line.size() > n && line[n] == ':')
            {
                return std::atol(line.c_str() + n + 1);
            }
        }
        return -1;
    }
    void print(
        const char* at,
        long iter) const
    {
        if (!on_)
        {
            return;
        }
        const detail::DevicePool& pool = detail::devicePool();
        long long reserved = -1;
        long long allocated = -1;
        long long peak = -1;
        if (pool.enabled())
        {
            reserved = static_cast<long long>(pool.mallocBytes() - pool.freedBytes());
            allocated = reserved - static_cast<long long>(pool.heldBytes());
            peak = static_cast<long long>(pool.peakFootprint());
        }
        std::size_t freeB = 0;
        std::size_t totalB = 0;
        cudaCheck(cudaMemGetInfo(&freeB, &totalB), "BRAE_MEMORY_PROBE cudaMemGetInfo");
        const long memTotal = readKb("/proc/meminfo", "MemTotal");
        const long memAvail = readKb("/proc/meminfo", "MemAvailable");
        std::printf("brae memory: at=%s iteration=%ld pool_reserved=%lld pool_allocated=%lld pool_peak_reserved=%lld "
                    "device_used=%zu device_total=%zu rss_kb=%ld hwm_kb=%ld rss_exact_kb=%ld host_used_kb=%ld\n",
                    at,
                    iter,
                    reserved,
                    allocated,
                    peak,
                    totalB - freeB,
                    totalB,
                    readKb("/proc/self/status", "VmRSS"),
                    readKb("/proc/self/status", "VmHWM"),
                    readKb("/proc/self/smaps_rollup", "Rss"),
                    (memTotal < 0 || memAvail < 0) ? -1L : memTotal - memAvail);
        // the gate reads these lines while the run is still going, to time its shutdown window from `end`
        std::fflush(stdout);
    }
    void leak()
    {
        if (leakBytes_ == 0)
        {
            return;
        }
        if (leakDevice_)
        {
            detail::devicePool().take(leakBytes_);
            return;
        }
        char* p = static_cast<char*>(std::malloc(leakBytes_));
        if (!p)
        {
            throw std::runtime_error("brae: BRAE_CONTROL_MEMORY_LEAK could not allocate its host block.");
        }
        std::memset(p, 1, leakBytes_);
        kept_.push_back(p);
    }
    bool on_ = false;
    std::set<long> iters_;
    bool leakDevice_ = false;
    std::size_t leakBytes_ = 0;
    std::vector<char*> kept_;
};

} // namespace brae
