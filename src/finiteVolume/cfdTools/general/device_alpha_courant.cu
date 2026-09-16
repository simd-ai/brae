// The VoF Courant number on the device -- see device_alpha_courant.cuh for the mask, the boundary half
// and the unmasked denominator.
#include "device_alpha_courant.cuh"
#include "device_blas.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

void ckC(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae deviceAlphaCourantNo: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// surfaceSum(mag(phi)) per cell, then the nearInterface mask. One gather, no atomics, the same
// ownerStart/losort/bndCellStart walk deviceDiv uses -- so the summation order is deterministic and two
// runs of the same case report the same Courant number.
__global__ void sumPhiKernel(
    int nC,
    const label*  __restrict__ ownerStart,
    const label*  __restrict__ losort, const label* __restrict__ losortStart,
    const label*  __restrict__ bndCellStart, const label* __restrict__ bndPerm,
    const label*  __restrict__ bndIsEmpty,
    const scalar* __restrict__ phiInt, const scalar* __restrict__ phiBnd,
    const scalar* __restrict__ alpha1,
    scalar* __restrict__ out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar s = 0;
    // |phi| to BOTH sides of an internal face -- surfaceSum, not a divergence.
    for (int k = ownerStart[c]; k < ownerStart[c + 1]; ++k) s += fabs(phiInt[k]);
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) s += fabs(phiInt[losort[k]]);
    // ...and to the owner of every boundary face. Dropping this understates Co near inlets.
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
    {
        const int b = bndPerm[k];
        if (bndIsEmpty[b]) continue;              // emptyFvPatch::size() == 0 in OpenFOAM
        s += fabs(phiBnd[b]);
    }

    if (alpha1)
    {
        // pos0(a - 0.01)*pos0(0.99 - a): a 0/1 MASK over the CLOSED band, pos0 being 1 at exactly zero.
        const scalar a  = alpha1[c];
        const scalar lo = a - scalar(0.01);
        const scalar hi = scalar(0.99) - a;
        const scalar mask = ((lo >= scalar(0)) ? scalar(1) : scalar(0))
                          * ((hi >= scalar(0)) ? scalar(1) : scalar(0));
        s *= mask;
    }
    out[c] = s;
}

}   // namespace


DeviceCourantNumbers deviceAlphaCourantNo(
    const DeviceMesh&           dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>* alpha1,
    scalar                      deltaT)
{
    DeviceCourantNumbers c;
    const int nC = dm.nCells;
    if (nC == 0) return c;

    DeviceBuffer<scalar> sumPhi(static_cast<std::size_t>(nC));
    sumPhiKernel<<<nBlocks(nC), TPB>>>(
        nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(),
        phiInt.data(), phiBnd.data(), alpha1 ? alpha1->data() : nullptr, sumPhi.data());
    ckC(cudaGetLastError(), "surfaceSum(mag(phi))");

    // The MAX is what limits the step; deviceMaxRatio is a single pass and skips V <= 0, as the host's
    // courantNo does. The MEAN divides the masked flux sum by the WHOLE mesh volume -- the denominator
    // is deliberately not masked.
    const scalar maxRatio = deviceMaxRatio(sumPhi, dm.V);
    const scalar sPhi     = deviceSumMag(sumPhi);
    const scalar sV       = deviceSumMag(dm.V);
    c.CoNum     = scalar(0.5)*maxRatio*deltaT;
    c.meanCoNum = (sV > scalar(0)) ? scalar(0.5)*(sPhi/sV)*deltaT : scalar(0);
    return c;
}

} // namespace brae
