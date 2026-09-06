#pragma once
// device_pcg_detail.cuh -- shared BiCGStab machinery: the on-device scalar-recurrence kernels (rho/beta/alpha/omega,
// all guarded NaN-safe per OF checkSingularity) + the BiCGScalars device-scratch singleton. Used by BOTH the single-GPU
// deviceJacobiBiCGStab (device_pcg.cu) and the distributed deviceParallelJacobiBiCGStab (device_pcg_distributed.cu), so
// it lives in a header (kernels static = per-TU internal linkage; bicgScalars() inline = one shared scratch instance).
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include <cuda_runtime.h>

namespace brae {

// ON-DEVICE into a flag `bd` (read only on the K-cadence convergence check, not every iter -> removes the 2 per-iter
// breakdown D2H reads). The guarded recurrence stays NaN-safe between checks; for a NON-breakdown solve nothing trips
// (rho/omega stay >> VSMALL until convergence breaks first) so the iteration is bit-identical to the host-guarded form.
constexpr scalar BICG_VSMALL = 1e-300;   // = OF solveScalar VSMALL (SolverPerformance.C checkSingularity: mag(x) < vsmall_)
static __global__
void omegaK(
    const scalar* __restrict__ ts,
    const scalar* __restrict__ tt,
    scalar* __restrict__ om,
    scalar* __restrict__ negOm,
    scalar* __restrict__ bd)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar t=*tt;
        const scalar o = (t > BICG_VSMALL) ? (*ts)/t : 0.0;
        *om=o;
        *negOm=-o;
        if (fabs(o) < BICG_VSMALL) *bd = 1.0;            // OF: checkSingularity(mag(omega)) (guards next-iter beta)
    }
}
static __global__
void bicgBetaK(
    const scalar* __restrict__ rr,
    const scalar* __restrict__ rrOld,
    const scalar* __restrict__ al,
    const scalar* __restrict__ om,
    scalar* __restrict__ beta,
    scalar* __restrict__ bd)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar ro=*rrOld, o=*om;
        const bool ok = fabs(ro) >= BICG_VSMALL && fabs(o) >= BICG_VSMALL;
        *beta = ok ? (*rr / ro) * (*al / o) : 0.0;       // beta = (rA0rA/rA0rAold)*(alpha/omega), guarded
        if (!ok) *bd = 1.0;
    }
}
static __global__
void bicgRhoSingK(
    const scalar* __restrict__ rr,
    scalar* __restrict__ bd)
{
    if (threadIdx.x==0 && blockIdx.x==0 && fabs(*rr) < BICG_VSMALL) *bd = 1.0;   // OF: checkSingularity(mag(rA0rA))
}
static __global__
void bicgAlphaK(
    const scalar* __restrict__ rr,
    const scalar* __restrict__ r0Ay,
    scalar* __restrict__ al,
    scalar* __restrict__ negAl,
    scalar* __restrict__ bd)
{
    if (threadIdx.x==0 && blockIdx.x==0)
    {
        const scalar d=*r0Ay;
        const bool ok = fabs(d) >= BICG_VSMALL;
        const scalar q = ok ? (*rr)/d : 0.0;
        *al=q;
        *negAl=-q;                                // OF: alpha = rA0rA/rA0AyA (guarded vs 0/0 breakdown)
        if (!ok) *bd = 1.0;
    }
}
// Persistent BiCGStab device scalars (one-time alloc; single solve at a time, sequential across the 3 U components).
struct BiCGScalars
{
    DeviceBuffer<scalar> rr, rrOld, alpha, negAlpha, omega, negOmega, beta, r0Ay, tt, ts, sNorm, rNorm, bd;
    BiCGScalars()
    {
        for (auto* p : {&rr,&rrOld,&alpha,&negAlpha,&omega,&negOmega,&beta,&r0Ay,&tt,&ts,&sNorm,&rNorm,&bd})
            p->resize(1);
    }
};
inline BiCGScalars& bicgScalars() { static BiCGScalars s; return s; }

} // namespace brae
