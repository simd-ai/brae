// brae DeviceHalo, NVSHMEM on-stream halo exchange (see device_halo.cuh). Pack/scatter are plain CUDA
// kernels; the transfer is the host on-stream put + barrier, so no relocatable device code is needed.
#include "device_halo.cuh"
#include "pcuda_compat.cuh"
#ifdef BRAE_WITH_NVSHMEM
#include <nvshmem_host.h>
#endif

namespace brae {
namespace {
constexpr int TPB = 128;

// sendRegion[f] = psi[faceCells[f]]  (gather the interface's own boundary values)
__device__
void packKernel(
    const scalar* __restrict__ psi,
    const label*  __restrict__ faceCells,
    scalar*       __restrict__ sendRegion,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) sendRegion[f] = psi[faceCells[f]];
}

// result[faceCells[f]] -= coeff[f] * recvRegion[f]  (the coupled-interface off-diagonal, OF A.psi convention).
// atomicAdd because a cell may own several faces on the SAME interface (multiple cut faces to one neighbour),
// so distinct threads can target the same result cell -- exactly like device_spmv's cyclicAmulKernel.
__device__
void scatterKernel(
    scalar*       __restrict__ result,
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ coeff,
    const scalar* __restrict__ recvRegion,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < n) atomicAdd(&result[faceCells[f]], -coeff[f] * recvRegion[f]);
}

// bvalOut[f] = w[f]*psi[faceCells[f]] + (1 - w[f])*recvRegion[f]  (the coupled processor-face boundary value).
// bvalOut points at bval + procStart[i], so each interface writes its slice of the flattened boundary array.
__device__
void bvalKernel(
    const scalar* __restrict__ psi,
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ w,
    const scalar* __restrict__ recvRegion,
    scalar*       __restrict__ bvalOut,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const scalar wf = w[f];
    bvalOut[f] = wf * psi[faceCells[f]] + (1.0 - wf) * recvRegion[f];
}
} // namespace

DeviceHalo::DeviceHalo(
    int myPart,
    const std::vector<int>& nbrParts,
    const std::vector<std::vector<label>>& faceCells)
    : myPart_(myPart), nbr_(nbrParts)
{
    const int nI = static_cast<int>(nbr_.size());
    size_.resize(nI);
    recvOffset_.resize(nI);
    remoteOffset_.resize(nI, 0);
    faceCellsD_.resize(nI);
    label off = 0;
    for (int i = 0; i < nI; ++i)
    {
        size_[i] = static_cast<label>(faceCells[i].size());
        recvOffset_[i] = off;
        off += size_[i];
        if (size_[i] > 0) faceCellsD_[i].copyFrom(faceCells[i]);
    }
    total_ = off;

    // Symmetric send/recv buffers sized to the GLOBAL max total (nvshmem_malloc is collective + symmetric, so
    // recvBuf_.data() is the same symmetric address on every PE). Never allocate zero.
    const label maxTotal = Pstream::allReduce(total_, ReduceOp::Max);
    const std::size_t cap = static_cast<std::size_t>(maxTotal > 0 ? maxTotal : 1);
    sendBuf_ = SymBuffer<scalar>(cap);
    recvBuf_ = SymBuffer<scalar>(cap);

    // One-time offset exchange: tell each neighbour where in OUR recv buffer to write; learn where to write in
    // THEIRS. One interface per neighbour partition, so the source rank disambiguates (tag 0 is fine).
    std::vector<scalar> sendOff(nI), recvOff(nI, 0);
    for (int i = 0; i < nI; ++i) sendOff[i] = static_cast<scalar>(recvOffset_[i]);
    for (int i = 0; i < nI; ++i)
    {
        Pstream::irecv(&recvOff[i], 1, nbr_[i], 0);
        Pstream::isend(&sendOff[i], 1, nbr_[i], 0);
    }
    Pstream::waitAll();
    for (int i = 0; i < nI; ++i) remoteOffset_[i] = static_cast<label>(recvOff[i]);
}

void DeviceHalo::postExchange(
    const scalar* psi_d,
    cudaStream_t stream)
{
    const int nI = static_cast<int>(nbr_.size());
    for (int i = 0; i < nI; ++i)
        if (size_[i] > 0)
        {
            const label* fc = faceCellsD_[i].data();
            scalar* sb = sendBuf_.data() + recvOffset_[i];
            const int n = static_cast<int>(size_[i]);
            pcudaParallelFor((n + TPB - 1) / TPB, TPB, 0, stream, [=] __device__ () {
                packKernel(psi_d, fc, sb, n);
            });
        }
#ifdef BRAE_WITH_NVSHMEM
    for (int i = 0; i < nI; ++i)
        if (size_[i] > 0)
            nvshmemx_putmem_on_stream(
                recvBuf_.data() + remoteOffset_[i],           // symmetric addr -> neighbour's recv
                sendBuf_.data() + recvOffset_[i],
                static_cast<std::size_t>(size_[i]) * sizeof(scalar),
                nbr_[i],
                stream);
#endif
}

void DeviceHalo::waitExchange(cudaStream_t stream)
{
#ifdef BRAE_WITH_NVSHMEM
    nvshmemx_barrier_all_on_stream(stream);
#else
    (void)stream;
#endif
}

void DeviceHalo::exchange(
    const scalar* psi_d,
    cudaStream_t stream)
{
    postExchange(psi_d, stream);
    waitExchange(stream);
}

void DeviceHalo::updateInterfaceMatrix(
    int i,
    scalar* result_d,
    const scalar* coeff_d,
    cudaStream_t stream)
{
    if (size_[i] > 0)
    {
        const label* fc = faceCellsD_[i].data();
        const scalar* rb = recvBuf_.data() + recvOffset_[i];
        const int n = static_cast<int>(size_[i]);
        pcudaParallelFor((n + TPB - 1) / TPB, TPB, 0, stream, [=] __device__ () {
            scatterKernel(result_d, fc, coeff_d, rb, n);
        });
    }
}

void DeviceHalo::scatterBoundaryValues(
    const scalar* psi_d,
    const std::vector<DeviceBuffer<scalar>>& weights,
    const std::vector<label>& procStart,
    scalar* bval_d,
    cudaStream_t stream)
{
    const int nI = static_cast<int>(nbr_.size());
    for (int i = 0; i < nI; ++i)
    {
        if (size_[i] <= 0) continue;
        const label* fc = faceCellsD_[i].data();
        const scalar* wd = weights[i].data();
        const scalar* rb = recvBuf_.data() + recvOffset_[i];
        scalar* bo = bval_d + procStart[i];
        const int n = static_cast<int>(size_[i]);
        pcudaParallelFor((n + TPB - 1) / TPB, TPB, 0, stream, [=] __device__ () {
            bvalKernel(psi_d, fc, wd, rb, bo, n);
        });
    }
}

std::vector<scalar> DeviceHalo::neighbourField(
    int i,
    cudaStream_t stream) const
{
    cudaStreamSynchronize(stream);
    std::vector<scalar> out(size_[i]);
    if (size_[i] > 0)
        cudaMemcpy(out.data(), recvBuf_.data() + recvOffset_[i],
                   static_cast<std::size_t>(size_[i]) * sizeof(scalar), cudaMemcpyDeviceToHost);
    return out;
}

} // namespace brae
