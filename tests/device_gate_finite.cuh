#pragma once
// requireFinite -- the guard every fmax-based comparison in this tree needs ahead of it.
//
// WHY IT EXISTS, and it is not defensive noise. `std::fmax(a, NaN)` returns `a`: it IGNORES the NaN.
// Every device gate here accumulates a worst-case difference with
//
//     worst = std::fmax(worst, std::fabs(device[i] - host[i]));
//
// so a field that has gone entirely non-finite produces `nan` differences, every one of which fmax
// drops, and the gate reports 0.000e+00 -- INDISTINGUISHABLE FROM PERFECT AGREEMENT.
//
// MEASURED: tests/test_device_inter_dambreak_alpha.cu once printed "alpha 0.0000e+00, U.x 0.0000e+00,
// p_rgh 0.0000e+00" and four green checks on a run whose every one of 2268 cells was NaN. It was only
// caught because exact zeros across three fields solved by two different Krylov solvers is not
// agreement -- it is too good. Nothing in the gate itself objected.
//
// A finiteness check therefore has to come BEFORE any fmax accumulator, not after it, and it has to be
// a `check` the harness counts rather than a printout a reader might skim past.
#include "cf_types.cuh"
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace brae {
namespace gatecheck {

// Returns the number of non-finite entries and prints them when there are any. The caller feeds the
// result to its own `check`, so the failure is counted the same way every other arm is.
template <typename T>
inline int nonFinite(const char* what, const std::vector<T>& v)
{
    if constexpr (!std::is_floating_point<T>::value) { (void)what; (void)v; return 0; }
    else
    {
    int bad = 0;
    for (T x : v) if (!std::isfinite(x)) ++bad;
    if (bad)
        std::printf("  !!    %s: %d of %zu entries are NOT FINITE -- every fmax difference below "
                    "would read 0.000e+00\n", what, bad, v.size());
    return bad;
    }
}

// ...and for a VECTOR field, all three components. Without this overload a U field matches the template
// above, whose non-floating branch returns 0 -- so `nonFinite("U", U.internal)` would compile, check
// nothing, and put the same false confidence one call higher. A non-template overload wins resolution.
inline int nonFinite(
    const char* what,
    const std::vector<vector>& v)
{
    int bad = 0;
    for (const vector& x : v)
    {
        if (!std::isfinite(x.x) || !std::isfinite(x.y) || !std::isfinite(x.z))
        {
            ++bad;
        }
    }
    if (bad)
        std::printf("  !!    %s: %d of %zu vectors are NOT FINITE -- every fmax difference below "
                    "would read 0.000e+00\n", what, bad, v.size());
    return bad;
}

}   // namespace gatecheck
}   // namespace brae
