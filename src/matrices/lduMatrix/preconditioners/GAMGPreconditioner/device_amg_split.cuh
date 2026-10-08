// BRAE_AMG_PCG_SPLIT=<solve>: where the time of an AMG-PCG solve goes, part by part and grid by grid.
// A solve is one captured graph by default: the host launches it and waits, so nothing inside it can be timed.
// The solve named here -- `pcorr`, `pressure` (p / p_rgh), or `all` -- leaves the graph for the plain loop, with
// the device synchronised after every part and the wall time since the previous one charged to that part. It is
// an INSTRUMENT, never a measured run: every launch waits for the one before, so the parts sum to more than the
// graph's solve takes (the report says by how much). What it gives is the parts' PROPORTIONS and the iteration
// count, which is the graph's -- the same recurrence, the residual read at every iteration.
// The parts are reported in BRAE_INTER_PHASE_TIME's table, which it therefore needs.
#pragma once

#include "inter_phase_time.cuh"
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace brae {
namespace amgSplit {

constexpr int MAX_GRIDS = 40;

// a grid's parts of one V-cycle, in the order the cycle reaches them
enum Part
{
    smoothPre,
    residual,
    restriction,
    coarsest,
    prolongation,
    smoothPost,
    nParts
};

// the Krylov loop's own parts
enum Krylov
{
    prologue,
    matrixCast,
    vectorCasts,
    product,
    updates,
    residualRead,
    nKrylov
};

struct State
{
    bool active = false;
    const char* name = "pressure";
    std::chrono::steady_clock::time_point last;
    double grid[MAX_GRIDS][nParts] = {};
    double krylov[nKrylov] = {};
};

inline State& state()
{
    static State s;
    return s;
}

// The solve the deviceAMGPCG calls inside this scope are -- the split's selection and its labels. `pressure`
// outside any scope.
struct Name
{
    const char* before;

    explicit Name(const char* n)
    :
        before(state().name)
    {
        state().name = n;
    }
    ~Name()
    {
        state().name = before;
    }
};

// is the solve named now the one BRAE_AMG_PCG_SPLIT asks for
inline bool wanted()
{
    static const std::string v = []()
    {
        const char* e = std::getenv("BRAE_AMG_PCG_SPLIT");
        const std::string s = e ? e : "";
        if (s.empty()) return s;
        if (s != "pcorr" && s != "pressure" && s != "all")
        {
            throw std::runtime_error("brae AMG-PCG: BRAE_AMG_PCG_SPLIT=" + s
                                     + " is not one of pcorr, pressure, all.");
        }
        if (!interPhase::on())
        {
            throw std::runtime_error("brae AMG-PCG: BRAE_AMG_PCG_SPLIT is reported in BRAE_INTER_PHASE_TIME's "
                                     "table; set BRAE_INTER_PHASE_TIME=1 with it.");
        }
        return s;
    }();
    if (v.empty()) return false;
    return v == "all" || v == state().name;
}

// the clock starts: everything launched so far is waited for and belongs to nobody
inline void begin()
{
    State& s = state();
    for (int g = 0; g < MAX_GRIDS; ++g)
    {
        for (int p = 0; p < nParts; ++p)
        {
            s.grid[g][p] = 0.0;
        }
    }
    for (int k = 0; k < nKrylov; ++k)
    {
        s.krylov[k] = 0.0;
    }
    cudaDeviceSynchronize();
    s.last = std::chrono::steady_clock::now();
    s.active = true;
}

// the wall time since the previous lap, device work waited for, goes to `slot`
inline void lapInto(double& slot)
{
    State& s = state();
    cudaDeviceSynchronize();
    const auto now = std::chrono::steady_clock::now();
    slot += std::chrono::duration<double>(now - s.last).count();
    s.last = now;
}

// ...to a grid's part: what the V-cycles call. Nothing, and no synchronise, outside a split solve.
inline void lap(
    int g,
    Part p)
{
    if (!state().active) return;
    if (g >= MAX_GRIDS)
    {
        throw std::runtime_error("brae AMG-PCG: BRAE_AMG_PCG_SPLIT holds " + std::to_string(MAX_GRIDS)
                                 + " grids and the hierarchy has more.");
    }
    lapInto(state().grid[g][p]);
}

// ...to one of the Krylov loop's parts
inline void lap(Krylov k)
{
    if (!state().active) return;
    lapInto(state().krylov[k]);
}

}   // namespace amgSplit
}   // namespace brae
