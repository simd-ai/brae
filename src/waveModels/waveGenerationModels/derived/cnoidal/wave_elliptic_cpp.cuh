#pragma once
// The complete elliptic integrals K(m), E(m) and the Jacobi functions sn, cn, dn, as the cnoidal wave
// model evaluates them.
//
// provenance:
//   openfoam:  src/waveModels/waveGenerationModels/derived/cnoidal/Elliptic.H:60-163
//
// Both are the arithmetic-geometric mean: K and E from its limit, the amplitude by the descending
// Landen recurrence. The stopping test is |a - g| < SMALL (1e-15) in both, with the amplitude's
// capped at 25 passes. Header-only, as OpenFOAM's is.
#include "cf_types.cuh"
#include <cmath>

namespace brae {
namespace cpu {
namespace waveModels {
namespace elliptic {

inline void ellipticIntegralsKE(
    scalar m,
    scalar& K,
    scalar& E)
{
    const scalar pi = 3.14159265358979323846;
    if (m == 0)
    {
        K = 0.5*pi;
        E = 0.5*pi;
        return;
    }
    scalar a = 1;
    scalar g = std::sqrt(1 - m);
    scalar ga = g*a;
    scalar aux = 1;
    scalar sum = 2 - m;
    while (true)
    {
        const scalar aOld = a;
        const scalar gOld = g;
        ga = gOld*aOld;
        a = 0.5*(gOld + aOld);
        aux += aux;
        sum -= aux*(a*a - ga);
        if (std::fabs(aOld - gOld) < 1.0e-15) break;
        g = std::sqrt(ga);
    }
    K = 0.5*pi/a;
    E = 0.25*pi/a*sum;
}

inline scalar JacobiAmp(
    scalar u,
    scalar mIn)
{
    const scalar pi = 3.14159265358979323846;
    const int ITER = 25;
    scalar a[ITER + 1];
    scalar g[ITER + 1];
    scalar c[ITER + 1];
    const scalar m = std::fabs(mIn);
    if (m == 0) return u;
    if (m == 1)
    {
        return 2*std::atan(std::exp(u)) - pi/2;
    }
    a[0] = 1.0;
    g[0] = std::sqrt(1.0 - m);
    c[0] = std::sqrt(m);
    scalar aux = 1.0;
    int n = 0;
    for (n = 0; n < ITER; n++)
    {
        if (std::fabs(a[n] - g[n]) < 1.0e-15) break;
        aux += aux;
        a[n+1] = 0.5*(a[n] + g[n]);
        g[n+1] = std::sqrt(a[n]*g[n]);
        c[n+1] = 0.5*(a[n] - g[n]);
    }
    scalar amp = aux*a[n]*u;
    for (; n > 0; n--)
    {
        amp = 0.5*(amp + std::asin(c[n]*std::sin(amp)/a[n]));
    }
    return amp;
}

inline void JacobiSnCnDn(
    scalar u,
    scalar m,
    scalar& Sn,
    scalar& Cn,
    scalar& Dn)
{
    const scalar amp = JacobiAmp(u, m);
    Sn = std::sin(amp);
    Cn = std::cos(amp);
    Dn = std::sqrt(1.0 - m*std::sin(amp)*std::sin(amp));
}

} // namespace elliptic
} // namespace waveModels
} // namespace cpu
} // namespace brae
