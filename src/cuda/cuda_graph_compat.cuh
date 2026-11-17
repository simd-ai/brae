// CUDA 12.4..12.9 vs CUDA 13 spellings of the two graph-construction calls the conditional-graph solvers use.
// CUDA 13 made the edge-data forms the unsuffixed ones; CUDA 12 keeps the unsuffixed names on the edge-data-free
// signatures (the edge-data forms there are *_v2 / *_v3). Neither caller uses edge data, so both wrappers take
// none and pass nullptr to CUDA 13. Conditional nodes (cudaGraphNodeTypeConditional, cudaGraphConditionalHandleCreate,
// cudaGraphSetConditional) exist from CUDA 12.4 with the same signatures as 13.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace brae {

inline cudaError_t braeGraphAddNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
                                    std::size_t nDeps, cudaGraphNodeParams* params)
{
#if CUDART_VERSION >= 13000
    return cudaGraphAddNode(node, graph, deps, nullptr, nDeps, params);
#else
    return cudaGraphAddNode(node, graph, deps, nDeps, params);
#endif
}

// The capture's current dependency frontier (status, graph and edge data are not needed by the callers).
inline cudaError_t braeStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* status,
                                            const cudaGraphNode_t** deps, std::size_t* nDeps)
{
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(stream, status, nullptr, nullptr, deps, nullptr, nDeps);
#else
    return cudaStreamGetCaptureInfo(stream, status, nullptr, nullptr, deps, nDeps);
#endif
}

} // namespace brae
