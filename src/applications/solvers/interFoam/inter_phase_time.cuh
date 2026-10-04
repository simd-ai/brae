// BRAE_INTER_PHASE_TIME=1: where a step of the interFoam DEVICE loop goes. Off unless the variable is set --
// attributing device work needs a synchronise at every boundary, which is itself a cost, so the instrument is
// never part of a measured run. `mark(name)` charges the wall time since the previous mark to `name`, so the
// phases partition the loop; `Nested` charges a scope INSIDE a phase (a host hook, a linear solve) to a
// second table that is reported beside the phases and is not added to them.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

namespace brae {
namespace interPhase {

inline bool on()
{
    static const bool v = std::getenv("BRAE_INTER_PHASE_TIME") != nullptr;
    return v;
}

struct State
{
    std::vector<std::pair<std::string, double>> phases;
    std::vector<std::pair<std::string, double>> nested;
    std::vector<std::pair<std::string, long>> nestedCalls;
    std::chrono::steady_clock::time_point last;
    bool started = false;
};

inline State& state()
{
    static State s;
    return s;
}

inline double& slot(
    std::vector<std::pair<std::string, double>>& table,
    const char* name)
{
    for (auto& e : table)
    {
        if (e.first == name) return e.second;
    }
    table.emplace_back(name, 0.0);
    return table.back().second;
}

inline void start()
{
    if (!on()) return;
    cudaDeviceSynchronize();
    state().last = std::chrono::steady_clock::now();
    state().started = true;
}

inline void mark(const char* name)
{
    if (!on() || !state().started) return;
    cudaDeviceSynchronize();
    const auto now = std::chrono::steady_clock::now();
    slot(state().phases, name) += std::chrono::duration<double>(now - state().last).count();
    state().last = now;
}

struct Nested
{
    const char* name;
    std::chrono::steady_clock::time_point t0;

    explicit Nested(const char* n)
    :
        name(n)
    {
        if (!on()) return;
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
    }

    ~Nested()
    {
        if (!on()) return;
        cudaDeviceSynchronize();
        slot(state().nested, name) += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (auto& e : state().nestedCalls)
        {
            if (e.first == name)
            {
                ++e.second;
                return;
            }
        }
        state().nestedCalls.emplace_back(name, 1);
    }
};

// `seconds` and `calls` charged to `name` in the nested table, by something that timed itself
inline void charge(
    const char* name,
    double seconds,
    long calls)
{
    if (!on()) return;
    slot(state().nested, name) += seconds;
    for (auto& e : state().nestedCalls)
    {
        if (e.first == name)
        {
            e.second += calls;
            return;
        }
    }
    state().nestedCalls.emplace_back(name, calls);
}

inline void report(long steps)
{
    if (!on() || steps <= 0) return;
    double total = 0;
    for (const auto& e : state().phases) total += e.second;
    std::printf("  phase time over %ld steps (BRAE_INTER_PHASE_TIME; synchronised at every boundary):\n", steps);
    for (const auto& e : state().phases)
    {
        std::printf("    %-44s %8.1f ms/step  %5.1f%%\n", e.first.c_str(), 1e3*e.second/steps, 100.0*e.second/total);
    }
    std::printf("    %-44s %8.1f ms/step\n", "ALL PHASES", 1e3*total/steps);
    std::printf("  inside those phases:\n");
    for (std::size_t i = 0; i < state().nested.size(); ++i)
    {
        std::printf("    %-44s %8.1f ms/step  %5.1f%%   (%.1f calls/step)\n",
                    state().nested[i].first.c_str(), 1e3*state().nested[i].second/steps,
                    100.0*state().nested[i].second/total, (double)state().nestedCalls[i].second/steps);
    }
}

}   // namespace interPhase
}   // namespace brae
