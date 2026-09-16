// The alpha sub-cycle -- see device_alpha_subcycle.cuh for the four ways to get it wrong.
#include "device_alpha_subcycle.cuh"
#include "device_blas.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {
namespace {

void ckSub(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae interFoam alpha sub-cycle: ") + what + ": "
                                 + cudaGetErrorString(e));
}

}   // namespace

void deviceAlphaEqnSubCycle(
    int                          nAlphaSubCycles,
    scalar                       totalDeltaT,
    DeviceBuffer<scalar>&        alpha1,
    const DeviceBuffer<scalar>&  alpha1Old,
    DeviceBuffer<scalar>&        rhoPhiInt,
    DeviceBuffer<scalar>&        rhoPhiBnd,
    const DeviceAlphaEqnStep&    step)
{
    if (nAlphaSubCycles < 1)
        throw std::runtime_error(
            "brae interFoam: nAlphaSubCycles must be at least 1; alphaControls.H reads it with "
            "get<label> and the sub-cycle loop runs it that many times.");
    if (totalDeltaT <= scalar(0))
        throw std::runtime_error("brae interFoam: the sub-cycle needs a positive time step.");

    if (nAlphaSubCycles == 1)
    {
        // alphaEqnSubCycle.H:31-34 -- the plain branch. NOT a one-iteration sub-cycle: OpenFOAM does
        // not construct a subCycle at all here, so deltaT is untouched and nothing is saved or
        // restored. Routing this through the loop below would be arithmetically equivalent and would
        // hide that `nAlphaSubCycles 1` is the ordinary path rather than a degenerate case.
        step(alpha1Old, totalDeltaT, alpha1, rhoPhiInt, rhoPhiBnd);
        return;
    }

    // a: Time::subCycle divides deltaT by nSubCycles (Time.C:1006-1009).
    const scalar dtSub  = totalDeltaT / static_cast<scalar>(nAlphaSubCycles);
    const scalar weight = dtSub / totalDeltaT;

    DeviceBuffer<scalar> sumInt, sumBnd, kInt, kBnd;
    bool sumStarted = false;

    // b: each sub-step starts from the PREVIOUS one's result. The first starts from the real old time,
    // and `alpha1Old` is const so it cannot be walked forward by accident -- the carried state is a
    // local buffer, not a rebinding of the caller's.
    DeviceBuffer<scalar> carried;
    deviceCopy(carried, alpha1Old);

    for (int k = 0; k < nAlphaSubCycles; ++k)
    {
        step(carried, dtSub, alpha1, kInt, kBnd);
        deviceCopy(carried, alpha1);

        // c: the TIME-WEIGHTED sum, accumulated per sub-step rather than replaced by the last one.
        if (!sumStarted)
        {
            // zeroed, mirroring the host's assign(0) -- cudaMemset writes +0.0 for an IEEE double,
            // and starting the sum at 0 rather than at w*rhoPhi_0 keeps the first sub-step on the same
            // arithmetic as every other one.
            sumInt.resize(kInt.size());
            sumBnd.resize(kBnd.size());
            if (sumInt.size())
                ckSub(cudaMemset(sumInt.data(), 0, sumInt.size()*sizeof(scalar)), "rhoPhi sum");
            if (sumBnd.size())
                ckSub(cudaMemset(sumBnd.data(), 0, sumBnd.size()*sizeof(scalar)), "rhoPhi sum, boundary");
            sumStarted = true;
        }
        if (kInt.size()) deviceAxpy(weight, kInt, sumInt);
        if (kBnd.size()) deviceAxpy(weight, kBnd, sumBnd);
    }

    deviceCopy(rhoPhiInt, sumInt);
    deviceCopy(rhoPhiBnd, sumBnd);
    // d: alpha1Old was never written; the signature is what enforces it.
}

} // namespace brae
