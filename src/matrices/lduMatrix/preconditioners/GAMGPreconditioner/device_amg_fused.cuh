#pragma once
// ---------------------------------------------------------------------------------------------------------------
// device_amg_fused.cuh -- the FUSED COARSE-LEVEL FP32 V-CYCLE: every grid of the single-precision cycle below a
// size, and the coarsest solve, in ONE kernel of ONE block (device_amg_fused.cu has the design and the numbers).
// INTERNAL: included by the AMG's own translation units only.
// ---------------------------------------------------------------------------------------------------------------
#include "device_amg.cuh"
#include "device_amg_internal.cuh"

namespace brae {

// Hierarchies with more grids below the threshold fuse only their last FUSED_MAX_LEVELS: the descriptors travel
// as the kernel's arguments (a graph bakes them at capture, as it bakes every other launch's), 152 bytes a grid.
constexpr int FUSED_MAX_LEVELS = 16;

// everything the kernel reads or writes on ONE grid
struct FusedLevel
{
    int n = 0;                           // cells
    int nc = 0;                          // cells of the next coarser grid (0 on the coarsest)
    const float* diag = nullptr;
    // the product: contiguous rows where row != nullptr (amulCsrFK's operands), else the face arrays (amulFK's)
    const label* row = nullptr;
    const label* col = nullptr;
    const float* val = nullptr;
    const float* upper = nullptr;
    const float* lower = nullptr;
    const label* nei = nullptr;
    const label* owner = nullptr;
    const label* ownerStart = nullptr;
    const label* losort = nullptr;
    const label* losortStart = nullptr;
    // restriction to the next grid (the gather's lists) and prolongation from it
    const label* galStart = nullptr;
    const label* galList = nullptr;
    const label* map = nullptr;
    // the cycle's work vectors on this grid (b and x are the caller's on the first fused grid)
    float* b = nullptr;
    float* x = nullptr;
    float* ax = nullptr;
    float* r = nullptr;
};

struct FusedCoarse
{
    int n = 0;
    bool inverse = false;                // the precomputed inverse (with the LU's fallback); else the dense LU
    const scalar* lu = nullptr;          // LU, row-major (coarseLUSolveKernel's operands)
    const int* piv = nullptr;
    const scalar* inv = nullptr;         // inverse: A^-1 row-major, the column-major LU, its gather, the safety flag
    const scalar* fc = nullptr;
    const int* perm = nullptr;
    const int* unsafe = nullptr;
    scalar* rcD = nullptr;               // the FP64 right-hand side and solution (the unfused cycle's vB, vX)
    scalar* xcD = nullptr;
};

struct FusedArgs
{
    int nLev = 0;                        // grids below the coarsest that the kernel walks; lev[nLev] is the coarsest
    int nPre = 0;
    int nPost = 0;
    FusedCoarse coarse;
    FusedLevel lev[FUSED_MAX_LEVELS];
};
// a kernel's arguments may not exceed 4 KB on every architecture the cycle runs on (32 KB from CUDA 12.1 on Volta and later)
static_assert(sizeof(FusedArgs) <= 4096, "the fused V-cycle's descriptors must fit a kernel's argument space");

// what the fused kernel has been launched for in this process (a test's premise: that the arm it compares ran it):
// the launches (a capture counts once, its replays not), and the last one's first fused grid and grid count
struct FusedStats
{
    long long launches = 0;
    int firstGrid = -1;
    int grids = 0;
};
FusedStats amgFusedStats();

// the launch (device_amg_fused.cu): one block of `threads`
void amgFusedVcycleLaunch(const FusedArgs& args, int threads);

// Runs grid g's V-cycle -- and every coarser one -- in the fused kernel when this hierarchy and cycle allow it and
// g is the first grid of the fused range; false (nothing launched) otherwise, and the caller recurses as usual.
// BRAE_AMG_FUSE_BELOW=<cells> (default 1024; 0 switches the fusion off) sets the range: every grid of that many
// cells or fewer.
bool amgFusedVcycleAt(
    int g,
    AMGData& amg,
    const LduF& Ag,
    const float* bg,
    float* xg);

} // namespace brae
