#pragma once
// ACPP/PCUDA compat shim (BRAE_BACKEND=ACPP only): nvcc predefines a few things PCUDA doesn't (doc/pcuda.md
// upstream). Force-included ahead of every .cu TU as the sole -include, so cuda_runtime.h is pulled in here.
#include <cuda_runtime.h>

#ifndef __forceinline__
#define __forceinline__ inline   // PCUDA has no forced-inline attribute; a hint only, so this is a no-op
#endif

#ifndef cudaStreamPerThread
// PCUDA's default stream already behaves as CUDA's per-thread default (doc/pcuda.md), so map straight to it.
#define cudaStreamPerThread cudaStream_t{}
#endif

// Bit-reinterpret intrinsics, not in PCUDA's device library: used by the atomicCAS-based double atomicMax/Min.
__host__ __device__ inline long long __double_as_longlong(double v)
{
    union { double d; long long l; } u; u.d = v; return u.l;
}
__host__ __device__ inline double __longlong_as_double(long long v)
{
    union { double d; long long l; } u; u.l = v; return u.d;
}

__host__ __device__ inline double __dmul_rn(double a, double b) { return a * b; }   // round-to-nearest-even is the only IEEE-754 mode for `*` anyway
