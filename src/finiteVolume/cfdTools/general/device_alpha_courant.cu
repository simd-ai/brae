// The VoF Courant number on the device -- see device_alpha_courant.cuh for the mask, the boundary half
// and the unmasked denominator.
#include "device_alpha_courant.cuh"
#include "device_blas.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
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
__global__
void sumPhiKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndIsEmpty,
    const scalar* __restrict__ phiInt,
    const scalar* __restrict__ phiBnd,
    const scalar* __restrict__ alpha1,
    // the coupled pair's faces of each cell and their flux; null on a mesh without a pair
    const label* __restrict__ pairCellStart,
    const label* __restrict__ pairPerm,
    const scalar* __restrict__ pairPhi,
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
    // ...and to the cell of every face of a coupled patch, which is a patch like any other to surfaceSum
    if (pairPhi)
    {
        for (int k = pairCellStart[c]; k < pairCellStart[c + 1]; ++k)
        {
            s += fabs(pairPhi[pairPerm[k]]);
        }
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
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>* alpha1,
    scalar deltaT,
    const DeviceCyclic* cyc,
    std::vector<scalar>* sumPhiOut)
{
    DeviceCourantNumbers c;
    const int nC = dm.nCells;
    if (nC == 0) return c;
    static const char* const leftOutEnv = std::getenv("BRAE_CONTROL_COURANT_PAIR_LEFT_OUT");
    static const bool leftOutBandOnly = leftOutEnv && std::string(leftOutEnv) == "interface";
    const bool pairLeftOut = leftOutEnv && (!leftOutBandOnly || alpha1);
    const bool pair = cyc && cyc->n > 0 && !pairLeftOut;
    if (cyc && cyc->n > 0)
    {
        if (cyc->ifCellStart.size() != static_cast<std::size_t>(nC) + 1
         || cyc->phi.size() != static_cast<std::size_t>(cyc->n)
         || cyc->ifPerm.size() != static_cast<std::size_t>(cyc->n))
        {
            throw std::runtime_error(
                "brae deviceAlphaCourantNo: the coupled pair's per-cell map or its flux is not this mesh's ("
                + std::to_string(cyc->ifCellStart.size()) + " starts for " + std::to_string(nC) + " cells, "
                + std::to_string(cyc->phi.size()) + " fluxes and " + std::to_string(cyc->ifPerm.size())
                + " entries for " + std::to_string(cyc->n) + " coupled faces).");
        }
        // said by the call that does it: the notice where the pair is summed, the control's line where it is
        // left out -- a log that holds both is the control that leaves it out of one of the two numbers
        static bool said = false;
        static bool saidLeftOut = false;
        if (pair && !said)
        {
            said = true;
            std::printf("  Courant number: the coupled pair's %d faces are in each cell's flux sum\n", cyc->n);
        }
        if (pairLeftOut && !saidLeftOut)
        {
            saidLeftOut = true;
            std::printf("  *** CONTROL MODE: the coupled pair's faces are left out of the %sCourant number's "
                        "flux sum. This run is deliberately wrong. ***\n", leftOutBandOnly ? "interface " : "");
        }
    }

    DeviceBuffer<scalar> sumPhi(static_cast<std::size_t>(nC));
    sumPhiKernel<<<nBlocks(nC), TPB>>>(
        nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndIsEmpty.data(),
        phiInt.data(), phiBnd.data(), alpha1 ? alpha1->data() : nullptr,
        pair ? cyc->ifCellStart.data() : nullptr,
        pair ? cyc->ifPerm.data() : nullptr,
        pair ? cyc->phi.data() : nullptr,
        sumPhi.data());
    ckC(cudaGetLastError(), "surfaceSum(mag(phi))");
    if (sumPhiOut)
    {
        sumPhi.copyTo(*sumPhiOut);
    }

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
