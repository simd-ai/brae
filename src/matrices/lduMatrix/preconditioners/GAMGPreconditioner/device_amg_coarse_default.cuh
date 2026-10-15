// The AMG coarsest grid's direct solve when BRAE_AMG_COARSE_SOLVE is not set (device_amg_coarse.cu,
// coarseSolveKind): the one line a change of default touches. The precomputed inverse, which takes the LU's own
// substitutions wherever its check finds it unsafe (a floored pivot, an entry not finite, the probes); =lu keeps the LU.
// Gated by tests/test_amg_coarse_inverse.cu and tests/interfoam_write/amg_pcg/coarse_inverse.sh; on the GB10 a solve
// 1.2-1.8 us where the LU's is 4.9-82 us (n = 8-128), the step 6-14 % shorter on seven tutorials of 2,268-56,000 cells.
#pragma once

#include "device_amg.cuh"

namespace brae
{

constexpr CoarseSolveKind COARSE_SOLVE_DEFAULT = CoarseSolveKind::Inverse;

} // namespace brae
