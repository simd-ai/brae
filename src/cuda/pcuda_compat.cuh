#pragma once
// pcudaParallelFor: portable kernel-launch primitive, replacing kernel<<<grid,block>>>(args) chevron syntax.
// AdaptiveCpp provides it natively (<pcuda.hpp>); under nvcc this builds an equivalent trampoline kernel.
//
// Kernels here must be __device__, not __global__: nvcc refuses to call __global__ except via <<<>>>, even
// from a lambda. PCUDA doesn't care (JIT marks entries via LLVM IR metadata) -- purely an nvcc requirement.
#ifdef BRAE_ACPP
#include <pcuda.hpp>
#else
#include <cuda_runtime.h>

template <class F>
__global__ void braeTrampolineKernel(F f) { f(); }

template <class F>
inline cudaError_t pcudaParallelFor(dim3 grid, dim3 block, size_t sharedMem, cudaStream_t stream, F f)
{
    braeTrampolineKernel<<<grid, block, sharedMem, stream>>>(f);
    return cudaGetLastError();
}
template <class F>
inline cudaError_t pcudaParallelFor(dim3 grid, dim3 block, size_t sharedMem, F f)
{
    return pcudaParallelFor(grid, block, sharedMem, cudaStream_t(0), f);
}
template <class F>
inline cudaError_t pcudaParallelFor(dim3 grid, dim3 block, F f)
{
    return pcudaParallelFor(grid, block, size_t(0), cudaStream_t(0), f);
}
#endif
