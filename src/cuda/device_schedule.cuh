#pragma once
// HOW THE HOST WAITS FOR THE DEVICE. cudaDeviceScheduleAuto takes a blocking wait on this machine (NVIDIA GB10,
// one context, 20 cores): every blocking copy and stream synchronise then returns about 200 us after the device
// has finished. MEASURED on RAS/DTCHull, ten steps of the interFoam device loop, ms a step: auto 570,
// blockingSync 570, yield 490, spin 488 -- the alpha step 107 -> 80 and the pressure correctors 256 -> 225, every
// written file byte-identical. brae spins; BRAE_CUDA_SCHEDULE=auto|yield|block|spin chooses another. It has to
// run before the first CUDA call creates the context, so a binary calls it first thing in main.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>

namespace brae {

inline void setCudaSchedule()
{
    const char* e = std::getenv("BRAE_CUDA_SCHEDULE");
    const std::string v = e ? e : "spin";
    unsigned int flags = cudaDeviceScheduleSpin;
    if (v == "auto")
    {
        flags = cudaDeviceScheduleAuto;
    }
    else if (v == "yield")
    {
        flags = cudaDeviceScheduleYield;
    }
    else if (v == "block")
    {
        flags = cudaDeviceScheduleBlockingSync;
    }
    else if (v != "spin")
    {
        std::fprintf(stderr, "brae: BRAE_CUDA_SCHEDULE=%s is not one of auto, yield, block, spin\n", v.c_str());
        std::exit(1);
    }
    const cudaError_t r = cudaSetDeviceFlags(flags);
    if (r != cudaSuccess)
    {
        std::printf("  CUDA wait mode: %s could not be set (%s); the driver's default stands\n", v.c_str(),
                    cudaGetErrorString(r));
    }
}

} // namespace brae
