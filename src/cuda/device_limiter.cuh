#pragma once
// The TVD limiter and the limited face weight, shared by every path that interpolates with one: the
// internal faces (device_fvm.cu), and a COUPLED face, whose neighbour is a cell across the pair rather
// than a patch value (device_alpha_flux.cu). They were local to device_fvm.cu until the cyclic alpha
// flux needed the same arithmetic; a second copy would be a second thing to keep in step, which is the
// defect this project keeps finding.
//
// provenance:
//   openfoam: src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes (NVDTVD::r, vanLeer.H)
//   brae:     src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSchemes_cpp.cuh
#include "cf_types.cuh"

namespace brae {

// THE LIMITER OF r, selected by twoByk as the comment above says. Shared by the scalar kernel and the
// V kernel, so that `vanLeerV` and `vanLeer` are one expression and not two.
__device__ __forceinline__ scalar limiterOfR(
    scalar r,
    scalar twoByk)
{
    scalar limiter;
    if (twoByk > 0.0)
    {
        limiter = twoByk * r;
        limiter = (limiter < 0.0) ? 0.0 : (limiter > 1.0 ? 1.0 : limiter);    // clamp(.,0,1)
    }
    else if (twoByk == kVanLeerTwoByk)
    {
        // OF vanLeer.H:70, not clamped -- and it is NOT bounded by 1, which is the thing to know before
        // anyone "tidies" a clamp in here. psi(r) = 0 for r <= 0, rises through psi(1) = 1, and
        // ASYMPTOTES TO 2 (r=5 -> 1.667, r=100 -> 1.980). That is the Sweby TVD ceiling, not a bug:
        // limitedLinear clamps to [0,1] because its own form would run away, vanLeer does not because
        // its form is already second-order TVD. Verified against OpenFOAM's own expression over 2880
        // (flux, phiP, phiN, gradP, gradN, d) samples: worst difference 0.000e+00.
        const scalar ar = fabs(r);
        limiter = (r + ar) / (1.0 + ar);
    }
    else
    {
        limiter = r * (r + 1.0) / (r*r + 1.0);        // OF vanAlbada: NOT clamped, and it is <= 1 anyway
    }
    return limiter;
}

__device__ __forceinline__ scalar limitedFaceWeight(
    int f, int P, int N, scalar p, scalar cdwF,
    const scalar* __restrict__ field,
    const scalar* __restrict__ gx, const scalar* __restrict__ gy, const scalar* __restrict__ gz,
    const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
    scalar twoByk)
{
    const scalar dx = dOwnX[f] - dNeiX[f], dy = dOwnY[f] - dNeiY[f], dz = dOwnZ[f] - dNeiZ[f];   // d = C[N]-C[P]
    // NVDTVD::r, upwind-cell gradient (strict phi>0) projected on d, vs the face gradient.
    const int U = (p > 0.0) ? P : N;
    const scalar gradcf = dx*gx[U] + dy*gy[U] + dz*gz[U];
    const scalar gradf = field[N] - field[P];
    scalar r;   // sign(s) = (s>=0)?1:-1  (OF Scalar.H)
    if (fabs(gradcf) >= 1000.0 * fabs(gradf))
        r = 2.0 * 1000.0 * ((gradcf >= 0.0) ? 1.0 : -1.0) * ((gradf >= 0.0) ? 1.0 : -1.0) - 1.0;
    else
        r = 2.0 * (gradcf / gradf) - 1.0;
    const scalar limiter = limiterOfR(r, twoByk);
    const scalar pos0 = (p >= 0.0) ? 1.0 : 0.0;
    return limiter * cdwF + (1.0 - limiter) * pos0;
}

} // namespace brae
