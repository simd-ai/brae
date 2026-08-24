// device_generalized_newtonian.cu -- see the header for the OF definitions this reproduces.
#include "device_generalized_newtonian.cuh"
#include "pcuda_compat.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// OF's SMALL, the floor powerLaw applies to the strain rate before raising it to (n-1).
//
// WHERE IT ACTUALLY BITES, established by mutating it out and watching the tests: n < 1 makes the
// exponent negative, so a zero-strain cell raises 0 to a negative power and gets +inf -- but fmin(nuMax,
// inf) clamps that straight back, so for nu0 > 0 the floor changes nothing. The case that needs it is
// nu0 == 0, where the product is 0*inf = NaN and no clamp can recover. Keeping OF's guard rather than
// relying on the clamp is what makes that corner defined.
constexpr scalar kSmall = 1.0e-15;

__device__
void gnPowerLawMuKernel(
    int nC,
    const scalar* __restrict__ S2,
    const scalar* __restrict__ rho,
    scalar nuMin,
    scalar nuMax,
    scalar nExp,
    scalar* __restrict__ mu)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar r = rho[c];
    if (r <= scalar(0)) return;                      // leave mu alone rather than divide by zero
    const scalar nu0 = mu[c] / r;                    // OF compressible nu() = thermo.mu()/rho
    const scalar sr  = sqrt(fmax(S2[c], scalar(0))); // sqrt(2)*mag(symm(grad U)); S2 is already 2*magSqr
    const scalar nu  = fmax(nuMin, fmin(nuMax, nu0 * pow(fmax(sr, kSmall), nExp - scalar(1))));
    mu[c] = r * nu;                                  // muEff = rho*nuEff, and nuEff() = nu_ (not nu + nu_)
}

}   // namespace

void deviceGeneralizedNewtonianPowerLawMu(
    const DeviceBuffer<scalar>& S2,
    const DeviceBuffer<scalar>& rho,
    scalar nuMin,
    scalar nuMax,
    scalar n,
    DeviceBuffer<scalar>& mu)
{
    const int nC = static_cast<int>(mu.size());
    if (nC == 0) return;
    if (static_cast<int>(S2.size()) != nC || static_cast<int>(rho.size()) != nC) return;
    const scalar* S2d = S2.data();
    const scalar* rhod = rho.data();
    scalar* mud = mu.data();
    pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
        gnPowerLawMuKernel(nC, S2d, rhod, nuMin, nuMax, n, mud);
    });
    cudaCheck(cudaGetLastError(), "generalizedNewtonianPowerLawMu");
}

}   // namespace brae
