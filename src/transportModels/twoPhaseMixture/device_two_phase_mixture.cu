// The two-phase mixture on the device -- see device_two_phase_mixture.cuh for the provenance and for
// why the raw/clamped split is the one thing to get wrong here.
#include "device_two_phase_mixture.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

inline void cudaCheckMix(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae deviceMixtureCorrect: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// ON FUSED MULTIPLY-ADD, AND WHY THE GATE FOR THIS FILE IS NOT BIT-FOR-BIT.
//
// Both compilers contract `x*y + z` into an FMA -- nvcc by default, g++ at -O3 on ARM64 -- so each
// rounds ONCE where the written expression rounds twice. That is the more accurate operation and
// neither is wrong. But they do not choose the SAME fusion: for `a*rho1 + (1-a)*rho2` one emits
// fma(a, rho1, (1-a)*rho2) and the other fma((1-a), rho2, a*rho1), and those differ in the last bit.
//
// Forbidding contraction on the device with __dmul_rn/__dadd_rn was tried and made it WORSE: the host
// still fused, so three fields started disagreeing instead of one. Matching source text cannot pin
// this; only forbidding contraction on BOTH sides could, and that gives up accuracy and speed for a
// bit-identity that is not itself a correctness criterion.
//
// So the gate carries a few ULP and says why. What keeps it worth having is that it still
// DISCRIMINATES: the one thing this component can get wrong is taking the CLAMPED alpha for rho, which
// is worth ~50 on an overshooting cell -- fifteen orders above the FMA difference.
__device__ __forceinline__ scalar clamp01d(scalar a)
{
    return a < scalar(0) ? scalar(0) : (a > scalar(1) ? scalar(1) : a);
}

// One pass over alpha1, four fields out. The RAW value feeds rho and the CLAMPED one feeds mu and nu,
// which is the whole content of the model at this level: they differ only where MULES has left alpha
// outside [0,1], and that is exactly where an interface is.
__global__ void mixtureCorrectKernel(
    const scalar* __restrict__ alpha1,
    int                        n,
    scalar rho1, scalar nu1, scalar rho2, scalar nu2,
    scalar* __restrict__ alpha2,
    scalar* __restrict__ rho,
    scalar* __restrict__ mu,
    scalar* __restrict__ nu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar a  = alpha1[i];
    const scalar a2 = scalar(1) - a;          // alpha2 = 1 - alpha1, RAW, as createFields writes it
    const scalar ac = clamp01d(a);            // ...and the clamped one, for mu and nu only

    if (alpha2) alpha2[i] = a2;
    if (rho)    rho[i]    = a*rho1 + a2*rho2;                        // RAW -- both terms

    // mu and nu are computed together whenever either is asked for: nu is mu/rhoClamped, and
    // recomputing mu inside a separate nu kernel is a second chance to write the clamp differently.
    if (mu || nu)
    {
        const scalar m = ac*rho1*nu1 + (scalar(1) - ac)*rho2*nu2;
        if (mu) mu[i] = m;
        if (nu) nu[i] = m / (ac*rho1 + (scalar(1) - ac)*rho2);
    }
}

}   // namespace


void deviceMixtureCorrect(
    const scalar*                alpha1,
    int                          nCells,
    const DevicePhaseProperties& props,
    scalar*                      alpha2,
    scalar*                      rho,
    scalar*                      mu,
    scalar*                      nu)
{
    if (nCells <= 0) return;
    if (!alpha1)
        throw std::runtime_error("brae deviceMixtureCorrect: alpha1 is null.");

    mixtureCorrectKernel<<<nBlocks(nCells), TPB>>>(
        alpha1, nCells, props.rho1, props.nu1, props.rho2, props.nu2, alpha2, rho, mu, nu);
    cudaCheckMix(cudaGetLastError(), "launch");
}

} // namespace brae
