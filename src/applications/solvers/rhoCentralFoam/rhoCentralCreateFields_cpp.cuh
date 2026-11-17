#pragma once
// rhoCentralFoam's createFields.H and createFieldRefs.H, with the thermo's construction (heThermo::init and
// hePsiThermo's constructor calculate()), on the host. Both arms start from the state it returns.
#include "rhoCentralControls.cuh"
#include "rhoCentralState.cuh"

#include <vector>

namespace brae {
namespace cpu {
namespace rhoCentral {

using brae::rhoCentral::Case;
using brae::rhoCentral::PatchFields;
using brae::rhoCentral::State;

// Reads p, T and U from the start directory into their patch fields, and builds the conservative state.
State createFields(const Case& cs, PatchFields& pf);

// Evaluates every patch of `f` on `internal` (no coupled patch is accepted, so no exchange is posted) and copies
// the patch values into the flat boundary array, in patch order.
template <typename T>
inline void evaluatePatches(GeometricField<T>& f, const std::vector<T>& internal, const Case& cs, std::vector<T>& flat)
{
    for (auto& b : f.boundary) b->initEvaluate(internal);
    for (auto& b : f.boundary) b->evaluate(internal);
    flat.resize(cs.nB);
    std::size_t k = 0;
    for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
    {
        const std::vector<T>& v = f.boundary[pi]->value();
        for (label i = 0; i < cs.patches[pi].size; ++i) flat[k++] = v[i];
    }
}

} // namespace rhoCentral
} // namespace cpu
} // namespace brae
