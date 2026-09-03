#pragma once
// Function1 -- OF src/OpenFOAM/primitives/functions/Function1.
//
// A Function1 is OF's "value that may depend on time": `constant`, `table`, `polynomial`, `csvFile`,
// expressions. Boundary conditions take them for exactly this reason -- uniformTotalPressure's p0,
// uniformFixedValue's uniformValue, flowRateInletVelocity's flowRate.
//
// brae evaluated only the constant forms and refused the rest BY NAME (see fv_patch_field.cuh). This
// adds `table`, which is what OpenFOAM's own transient tutorials actually use:
// pimpleFoam/RAS/TJunction ramps its outlet total pressure with
//     p0  table ((0 10) (1 40));
//
// SEMANTICS, taken from OF and not invented:
//   * interpolation: LINEAR between the bracketing entries (TableBase's default interpolationScheme).
//   * out of range:  CLAMP -- `bounds::repeatableBounding::CLAMP` is TableBase.C's default (line 76),
//                    so t below the first entry gives the first value and t above the last gives the
//                    last. NOT extrapolation, which would invent pressures the case never asked for.
//   * a single entry behaves as a constant.
//
// Only `table` and `constant` are built here. Anything else keeps the existing named refusal rather
// than being approximated -- a polynomial silently run as a table is a different boundary condition.

#include "cf_types.cuh"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace brae {

class Function1
{
public:
    Function1() = default;

    static Function1 constant(scalar v)
    {
        Function1 f;
        f.entries_.emplace_back(scalar(0), v);
        return f;
    }

    // (t, value) pairs, in the order the dictionary lists them. Sorted here so an unordered table
    // still interpolates correctly rather than silently producing nonsense between entries.
    static Function1 table(std::vector<std::pair<scalar, scalar>> pts)
    {
        Function1 f;
        std::sort(pts.begin(), pts.end(),
                  [](const std::pair<scalar,scalar>& a, const std::pair<scalar,scalar>& b)
                  { return a.first < b.first; });
        f.entries_ = std::move(pts);
        return f;
    }

    bool empty() const { return entries_.empty(); }

    // True when every entry carries the same value, so value(t) is the same at every t. A driver that
    // samples the table once (the rhoSimpleFoam mirror seeds p0 at t = 0 and never refreshes it) can run
    // such a table exactly; anything else it must refuse rather than freeze at the first value.
    bool isConstant() const
    {
        for (const auto& e : entries_)
        {
            if (e.second != entries_.front().second) return false;
        }
        return true;
    }

    // OF TableBase::value(x): linear between brackets, clamped outside.
    scalar value(scalar t) const
    {
        if (entries_.empty()) return 0;
        if (entries_.size() == 1) return entries_.front().second;
        if (t <= entries_.front().first) return entries_.front().second;   // CLAMP low
        if (t >= entries_.back().first)  return entries_.back().second;    // CLAMP high
        for (std::size_t i = 1; i < entries_.size(); ++i)
        {
            if (t <= entries_[i].first)
            {
                const scalar t0 = entries_[i-1].first, t1 = entries_[i].first;
                const scalar v0 = entries_[i-1].second, v1 = entries_[i].second;
                const scalar dt = t1 - t0;
                if (dt <= 0) return v1;                       // duplicate abscissae: take the later value
                return v0 + (v1 - v0)*(t - t0)/dt;
            }
        }
        return entries_.back().second;
    }

private:
    std::vector<std::pair<scalar, scalar>> entries_;   // (t, value), ascending in t
};

}   // namespace brae
