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

// Reads p, T and U from the start directory into their patch fields, and builds the conservative state. A cold start and
// a restart are the same code, as in OpenFOAM: rho, rhoU and rhoE are NO_READ (createFields.H), rebuilt from p, T and U,
// and T from he. control: the gate's restartIoExtrapolate (rhoCentralState.cuh).
State createFields(const Case& cs, PatchFields& pf, int control = brae::rhoCentral::controlNone);

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

// inletOutletFvPatchField::updateCoeffs (inletOutletFvPatchField.C:99-113), which every evaluation of an inletOutlet
// patch runs first: valueFraction = 1 - pos0(phip), on the flux `phi` as it stands -- rhoCentralFoam.C:195's, this
// step's, at :232, :265 and :287 alike. phiB is that flux on every boundary face, flat in patch order.
template <typename T>
inline void switchInletOutlet(GeometricField<T>& f, const std::vector<scalar>& phiB, const Case& cs)
{
    std::size_t k0 = 0;
    for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
    {
        const std::size_t n = static_cast<std::size_t>(cs.patches[pi].size);
        if (f.boundary[pi]->isInletOutlet())
            f.boundary[pi]->updateFromFlux(std::vector<scalar>(phiB.begin() + k0, phiB.begin() + k0 + n));
        k0 += n;
    }
}

} // namespace rhoCentral
} // namespace cpu
} // namespace brae
