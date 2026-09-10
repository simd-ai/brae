// generalizedNewtonian (powerLaw) on the device -- see generalizedNewtonian.cuh.
#include "generalizedNewtonian.cuh"
#include "device_divdevreff.cuh"   // deviceBoundaryGradU: gaussGrad's boundary correction, the dev2 term's own
#include "device_kepsilon.cuh"     // deviceCellLimitGradU: the case's grad(U) limiter
#include "device_blas.cuh"         // deviceCopy
#include <cuda_runtime.h>

namespace brae {
namespace gpu {
namespace generalizedNewtonian {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// One entry of nu_ from a gradient packed component-major, (i*3+j)*n + e -- the layout deviceGradU and
// deviceDivDevReff use -- over an arbitrary run of entries, so cells and boundary faces take one kernel.
__global__ void powerLawNuKernel(
    int                        n,
    const scalar* __restrict__ grad,
    const scalar* __restrict__ nu0,
    scalar                     pn,
    scalar                     nuMin,
    scalar                     nuMax,
    scalar* __restrict__       nu)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const tensor t{grad[0 * n + e], grad[1 * n + e], grad[2 * n + e],
                   grad[3 * n + e], grad[4 * n + e], grad[5 * n + e],
                   grad[6 * n + e], grad[7 * n + e], grad[8 * n + e]};
    nu[e] = cpu::generalizedNewtonian::powerLawNu(nu0[e], cpu::generalizedNewtonian::strainRate(t), pn, nuMin,
                                                  nuMax);
}
} // namespace


void correctNu(
    const DeviceMesh&                                dm,
    const DeviceVectorBoundary&                      dbU,
    const DeviceBuffer<scalar>&                      Ux,
    const DeviceBuffer<scalar>&                      Uy,
    const DeviceBuffer<scalar>&                      Uz,
    const DeviceBuffer<scalar>* const*               UbStored,
    scalar                                           gradULimitK,
    const DeviceBuffer<scalar>&                      nu0,
    const DeviceBuffer<scalar>&                      nu0Bnd,
    const cpu::generalizedNewtonian::PowerLawCoeffs& coeffs,
    DeviceBuffer<scalar>&                            nu,
    DeviceBuffer<scalar>&                            nuBnd)
{
    const int nC = dm.nCells;
    const int nB = dm.nBndFaces;
    const DeviceBuffer<scalar>* Uc[3] = {&Ux, &Uy, &Uz};

    // fvc::grad(U): the three component gradients from U's boundary as it STANDS, packed so row i is
    // gaussGrad(U_i) -- the order deviceDivDevReff builds, so the two consumers of grad(U) cannot differ.
    DeviceBuffer<scalar> bvals[3];
    for (int i = 0; i < 3; ++i)
    {
        if (UbStored && UbStored[i] && UbStored[i]->size() == static_cast<std::size_t>(nB))
            deviceCopy(bvals[i], *UbStored[i]);
        else
            deviceBCValue(dbU.comp[i], *Uc[i], bvals[i]);
    }
    DeviceBuffer<scalar> gxs[3], gys[3], gzs[3];
    {
        const DeviceBuffer<scalar>* vol[3] = {Uc[0], Uc[1], Uc[2]};
        const DeviceBuffer<scalar>* bv[3]  = {&bvals[0], &bvals[1], &bvals[2]};
        deviceGaussGradFused(dm, 3, vol, bv, gxs, gys, gzs);
    }
    DeviceBuffer<scalar> gradU(static_cast<std::size_t>(9) * nC);
    for (int i = 0; i < 3; ++i)
    {
        const std::size_t bytes = static_cast<std::size_t>(nC) * sizeof(scalar);
        cudaCheck(cudaMemcpy(gradU.data() + (0 * 3 + i) * nC, gxs[i].data(), bytes, cudaMemcpyDeviceToDevice), "gn g");
        cudaCheck(cudaMemcpy(gradU.data() + (1 * 3 + i) * nC, gys[i].data(), bytes, cudaMemcpyDeviceToDevice), "gn g");
        cudaCheck(cudaMemcpy(gradU.data() + (2 * 3 + i) * nC, gzs[i].data(), bytes, cudaMemcpyDeviceToDevice), "gn g");
    }
    if (gradULimitK > scalar(0))
    {
        deviceCellLimitGradU(dm, dbU, Ux, Uy, Uz, gradU, gradULimitK);
    }
    DeviceBuffer<scalar> gradB;
    deviceBoundaryGradU(dm, dbU, Ux, Uy, Uz, gradU, gradB, UbStored);

    nu.resize(static_cast<std::size_t>(nC));
    nuBnd.resize(static_cast<std::size_t>(nB));
    if (nC > 0)
    {
        powerLawNuKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), nu0.data(), coeffs.n, coeffs.nuMin, coeffs.nuMax,
                                               nu.data());
        cudaCheck(cudaGetLastError(), "gnNuCell");
    }
    if (nB > 0)
    {
        powerLawNuKernel<<<nBlocks(nB), TPB>>>(nB, gradB.data(), nu0Bnd.data(), coeffs.n, coeffs.nuMin,
                                               coeffs.nuMax, nuBnd.data());
        cudaCheck(cudaGetLastError(), "gnNuBnd");
    }
}

} // namespace generalizedNewtonian
} // namespace gpu
} // namespace brae
