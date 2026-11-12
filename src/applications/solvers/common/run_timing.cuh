#pragma once
// The solver-only benchmark's measurements (brief #17 task 3, the protocol's B, C, D and E), opt-in with BRAE_PHASE_TIME as the
// per-iteration phase times are. One line an event, for scripts/container/brae_benchmark.sh to read:
//
//   brae-timing startup <s>                        B: process start to the first iteration (mesh and fields read, GPU set up)
//   brae-timing fo <name> execute <n> <s> write <m> <s>   D: each functionObject's own wall time over the run, every call
//   brae-timing write <time> <s>                   E: each time directory the run writes
//   brae-timing loop <s> <iterations>              the iterations' wall time, functionObjects and writes inside it included;
//                                                  C, the solver's own, is the loop less D and the loop's writes
//
// WALL TIME, NOT KERNEL TIME. Each measured span starts and ends with the device drained (cudaDeviceSynchronize), so that a span
// holds its own device work and not the solver work queued before it: a functionObject's first host read would otherwise wait
// for the step's last kernels and be charged for them. Kernel time is a profiler's (Nsight Systems) and is reported apart, never
// mixed with these. The drains cost a little and exist only when BRAE_PHASE_TIME is set; without it nothing here runs.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include <unistd.h>

namespace brae
{

struct RunTiming
{
    static bool on()
    {
        static const bool b = std::getenv("BRAE_PHASE_TIME") != nullptr;
        return b;
    }

    static void drain()
    {
        cudaDeviceSynchronize();
    }

    static double now()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // seconds since this process started: /proc/uptime less the process's start (field 22 of /proc/self/stat, in clock ticks
    // since boot). -1 where /proc does not say
    static double sinceProcessStart()
    {
        std::ifstream up("/proc/uptime");
        double uptime = -1;
        if (!(up >> uptime)) return -1;
        std::ifstream st("/proc/self/stat");
        std::string line;
        std::getline(st, line);
        const std::size_t close = line.rfind(')');
        if (close == std::string::npos) return -1;
        std::istringstream rest(line.substr(close + 2));
        std::string field;
        // fields from the third (state) on; starttime is the 22nd overall, the 20th here
        for (int i = 3; i <= 22; ++i)
        {
            if (!(rest >> field)) return -1;
        }
        const double ticks = static_cast<double>(sysconf(_SC_CLK_TCK));
        return uptime - std::strtod(field.c_str(), nullptr) / ticks;
    }
};

} // namespace brae
