// The AMG coarsest grid's direct solve when BRAE_AMG_COARSE_SOLVE is not set (device_amg_coarse.cu,
// coarseSolveKind): the one line a change of default touches.
#pragma once

#include "device_amg.cuh"

namespace brae
{

constexpr CoarseSolveKind COARSE_SOLVE_DEFAULT = CoarseSolveKind::LU;

} // namespace brae
