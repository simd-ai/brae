#pragma once
// `brae check --setup`: the case's own solver sets the case up completely and stops before its first step. Each solver's main asks
// this after constructing its solver; brae_cli.cuh's check sets BRAE_CHECK_SETUP and hands the case to that solver.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace brae {
namespace cli {

// true when this run is `brae check --setup`, having said what was set up and the device memory it took
inline bool checkSetupDone(const char* solver)
{
    if (!std::getenv("BRAE_CHECK_SETUP")) return false;
    std::size_t freeB = 0;
    std::size_t totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    std::printf("brae check --setup: %s set the case up completely (dictionaries, mesh, fields, boundary conditions, device "
                "arrays); GPU memory in use %.2f of %.2f GiB; no step taken\n", solver,
                static_cast<double>(totalB - freeB) / (1u << 30), static_cast<double>(totalB) / (1u << 30));
    std::fflush(stdout);
    return true;
}

}  // namespace cli
}  // namespace brae
