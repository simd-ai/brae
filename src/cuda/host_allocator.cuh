#pragma once
// HOW THE HOST'S FREED MEMORY IS KEPT. glibc serves a large malloc with an mmap of its own and unmaps it at the
// free, and it trims the top of the heap back to the system; a time step that allocates large work arrays --
// every host hook, the mesh update, the motion solver -- then takes its page faults again at EVERY step, and
// which arrays do depends on everything else the step allocates. It showed as a hook that tripled, 221 -> 662
// ms a step, when an unrelated part of the step stopped allocating (the wall-distance wave moved to the GPU).
//
// MEASURED on waveMakerPiston refined to 896,000 cells, the interFoam device loop, 30 steps, ms a step, glibc's
// defaults against this setting on the same binary: the step 2,499 -> 1,702; the momentum boundary hook
// 723 -> 190; the pressure correctors 650 -> 289; the device refresh after a mesh move 329 -> 114; the momentum
// matrix 261 -> 85. Peak resident memory 3,425,512 -> 3,457,804 kB. At 224,000 cells 514 -> 409, at 56,000 no
// change (113): its arrays are below glibc's thresholds. RAS/DTCHull, static, 845,536 cells: 770 -> 743.
// Thirteen write-gate cases byte-identical between the two; mixerVesselAMI differs from ITSELF run to run by
// 1e-12 under either, so it says nothing about this.
//
// brae keeps the memory: no mmap for a malloc (M_MMAP_MAX 0), no trimming (M_TRIM_THRESHOLD -1, which mallopt(3)
// documents as "disables trimming completely"), and the heap grown 256 MB at a time (M_TOP_PAD). THE COST:
// memory the process has used once stays with it until it ends.
//
// BRAE_HOST_ALLOCATOR=system leaves glibc's defaults -- the identity gate's other arm. A setting the user has
// already made in the environment (MALLOC_MMAP_MAX_, MALLOC_TRIM_THRESHOLD_, MALLOC_TOP_PAD_, or glibc.malloc
// in GLIBC_TUNABLES) is theirs and is left alone. It has to run before the first large allocation, so a binary
// calls it first thing in main and prints the notice it returns once its header line is out.
//
// THE NOTICE IS A PROOF, not a report of having asked: after the three mallopt calls a 64 MB block -- far above
// glibc's mmap threshold -- is allocated, and the run stops unless it came from the heap (mallinfo2's count of
// mmapped blocks unchanged). BRAE_CONTROL_HOST_ALLOCATOR_UNSET=1 skips the mallopt calls and keeps the probe:
// the gate's control, which must stop.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <string>

namespace brae {

// sets the allocator and returns the notice to print; an unknown BRAE_HOST_ALLOCATOR, or a parameter glibc
// will not take, stops the run by name
inline std::string setHostAllocator()
{
    const char* e = std::getenv("BRAE_HOST_ALLOCATOR");
    const std::string v = e ? e : "keep";
    if (v == "system")
    {
        return "";
    }
    if (v != "keep")
    {
        std::fprintf(stderr, "brae: BRAE_HOST_ALLOCATOR=%s is not one of keep, system\n", v.c_str());
        std::exit(1);
    }
    const char* tunables = std::getenv("GLIBC_TUNABLES");
    if (std::getenv("MALLOC_MMAP_MAX_") || std::getenv("MALLOC_TRIM_THRESHOLD_") || std::getenv("MALLOC_TOP_PAD_")
     || (tunables && std::strstr(tunables, "glibc.malloc.")))
    {
        return "  host memory: the environment sets the allocator itself (MALLOC_*_ or GLIBC_TUNABLES); "
               "brae leaves it as set\n";
    }
    const int topPad = 256*1024*1024;
    const bool unset = std::getenv("BRAE_CONTROL_HOST_ALLOCATOR_UNSET") != nullptr;
    if (!unset
     && (mallopt(M_MMAP_MAX, 0) != 1 || mallopt(M_TRIM_THRESHOLD, -1) != 1 || mallopt(M_TOP_PAD, topPad) != 1))
    {
        std::fprintf(stderr, "brae: glibc refused the host allocator setting (mallopt); "
                             "BRAE_HOST_ALLOCATOR=system runs without it\n");
        std::exit(1);
    }
    const std::size_t mappedBefore = mallinfo2().hblks;
    void* probe = std::malloc(std::size_t(64)*1024*1024);
    const bool mapped = mallinfo2().hblks != mappedBefore;
    std::free(probe);
    if (probe == nullptr || mapped)
    {
        std::fprintf(stderr, "brae: the host allocator setting did not take: a 64 MB block still came from "
                             "mmap. BRAE_HOST_ALLOCATOR=system runs without it\n");
        std::exit(1);
    }
    return "  host memory: freed blocks are kept for reuse (no mmap for a malloc, no trimming); "
           "BRAE_HOST_ALLOCATOR=system restores glibc's defaults\n";
}

} // namespace brae
