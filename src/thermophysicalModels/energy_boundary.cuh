#pragma once
// energy_boundary.cuh -- OpenFOAM's ENERGY boundary conditions, evaluated LIVE at the energy assembly.
//
// basicThermo does not give he the boundary conditions the case wrote for T. It DERIVES them
// (basicThermo::heBoundaryTypes, basicThermo.C:197-231), in this order and no other:
//
//     isA<fixedValueFvPatchScalarField>       -> fixedEnergy
//     isA<zeroGradientFvPatchScalarField>
//       || isA<fixedGradientFvPatchScalarField> -> gradientEnergy
//     isA<mixedFvPatchScalarField>            -> mixedEnergy        (inletOutlet is one of these)
//     everything else                         -> T's own type, unchanged
//
// Each of the three re-derives its coefficients from the CURRENT p and T on every updateCoeffs, which
// the matrix assembly calls once per iteration:
//
//     fixedEnergy    .C:95-114 : Tw.evaluate();  operator==(he(pw, Tw, patchi))
//     gradientEnergy .C:95-119 : Tw.evaluate();  gradient()      = Cpv(pw,Tw,patchi)*Tw.snGrad() + Z
//     mixedEnergy    .C:103-131: Tw.evaluate();  valueFraction() = Tw.valueFraction()
//                                                refValue()      = he(pw, Tw.refValue(), patchi)
//                                                refGrad()       = Cpv(pw,Tw,patchi)*Tw.refGrad() + Z
//
// Z is the term this file deliberately does NOT carry:
//
//     Z = patch().deltaCoeffs()*( he(pw, Tw, patchi) - he(pw, Tw, patch().faceCells()) )
//
// Both halves take the SAME pw and the SAME Tw; they differ only in which mixture object heThermo asks
// (heThermo.C:264-296 -- patchFaceMixture vs cellMixture). pureMixture returns the one mixture_ for both
// (pureMixture.H:89-101), so for a single-species thermo -- which is every thermo brae implements -- Z is
// identically zero. It is not an approximation and it is not thermo-dependent: a multi-species mixture
// revives it, and the createFields refusal on `mixture` is what stops one arriving silently.
// See rhoSimpleFoam/REFUSALS.md, the gradientEnergy entry, for the same derivation.
//
// WHAT THIS CHANGES ON THE GAS PATH: nothing, in value. For perfectGas + hConst, he(p,T) is
// p-independent and Cpv is a constant, and T's refValue/refGrad/prescribed gradient are the dictionary's
// own numbers -- so every coefficient this recomputes is arithmetically the number createFields already
// stored once (rhoCreateFields_cpp.cu, the heOf/Cpv mapping block). That is why the nine compressible
// gates cannot move, and it is exactly why the gate for this file compares COEFFICIENTS against
// OpenFOAM's own rather than a converged field: an end-to-end gas run cannot tell a correct
// transcription from a wrong one.
//
// WHAT IT CHANGES ON THE LIQUID PATH: everything. he(p,T) carries -p/rho(T) on the internal-energy form
// and Cpv is a correlation in T, so a boundary whose p or T moves has coefficients that move with it,
// and the static construction-time image is wrong from the second iteration onward. That is the whole
// reason this stage exists (H3.3).
//
// HOST ONLY, for now, and that is not a gap the device arm can fall into silently. The CUDA arm takes its
// boundary projection from these same host fields (rhoCreateFields.cuh), so it inherits the seed; it has
// no per-iteration twin of this file, which is stage H3.6. Today that costs nothing, because the only
// thermo whose coefficients move is the liquid one and createFields refuses `properties liquid` outright
// (stage H3.4 lifts that refusal, and must not be lifted for the device arm before H3.6 lands).
#include "geometric_field.cuh"
#include "liquid_thermo.cuh"
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {

// he's boundary coefficients, rebuilt from the current p and T. Call it AFTER T's boundary has been
// evaluated (that is Tw.evaluate() at the head of all three updateCoeffs) and BEFORE the energy matrix
// is assembled -- the matrix reads exactly these.
//
// `he` must already carry the patch CLASS that heBoundaryTypes maps T's onto; createFields builds it
// that way by copying T's type across. Where it does not, this refuses by name rather than skipping the
// patch: a skipped energy boundary is a wrong boundary condition that still converges.
inline void updateEnergyBoundaryCoeffs(
    GeometricField<scalar>&       he,
    const GeometricField<scalar>& T,
    const GeometricField<scalar>& p,
    const ThermoCoeffs&           thermo,
    const std::vector<FvPatch>&   patches)
{
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const label n = patches[pi].size;
        if (n == 0) continue;

        const fvPatchField<scalar>& Tp = *T.boundary[pi];
        const std::vector<scalar>&  pw = p.boundary[pi]->value();
        const std::vector<scalar>&  Tw = Tp.value();
        if (static_cast<label>(pw.size()) != n || static_cast<label>(Tw.size()) != n)
            throw std::runtime_error(
                "brae: energy boundary update found p or T with the wrong face count on patch '"
                + patches[pi].name + "'.");

        // THE ORDER OF THESE THREE IS OpenFOAM'S, not convenience: fixedValue, then the gradient pair,
        // then mixed (basicThermo.C:203-231). No brae patch class currently satisfies two of the tests at
        // once, so today the order is unobservable -- it is written this way so that one which did would
        // be dispatched the way OpenFOAM dispatches it rather than the way this file happens to read.
        if (dynamic_cast<const FixedValuePatchField<scalar>*>(&Tp))
        {
            // fixedEnergy: `operator==(thermo.he(pw, Tw, patchi))`. operator== on a fixedValue patch
            // replaces the STORED value, not merely the exposed one -- setStoredValues, not setValue, or
            // the next evaluateBoundary reverts it to the construction seed.
            auto* hp = dynamic_cast<FixedValuePatchField<scalar>*>(he.boundary[pi].get());
            if (!hp)
                throw std::runtime_error(
                    "brae: T on patch '" + patches[pi].name + "' is a fixedValue, which OpenFOAM maps to "
                    "fixedEnergy, but he's patch there is not of the fixedValue family. Refusing rather "
                    "than leaving the energy boundary at its construction-time value.");
            std::vector<scalar> hb(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
                hb[static_cast<std::size_t>(i)] = thermoHeOf(pw[i], Tw[i], thermo);
            hp->setStoredValues(std::move(hb));
        }
        else if (dynamic_cast<const FixedGradientPatchField<scalar>*>(&Tp))
        {
            // gradientEnergy, the fixedGradient half. Tp.snGrad() on this class returns the PRESCRIBED
            // gradient (fixedGradientFvPatchField.H), not one re-derived from a stale value.
            auto* hp = dynamic_cast<FixedGradientPatchField<scalar>*>(he.boundary[pi].get());
            if (!hp)
                throw std::runtime_error(
                    "brae: T on patch '" + patches[pi].name + "' is a fixedGradient, which OpenFOAM maps "
                    "to gradientEnergy, but he's patch there is not of the fixedGradient family. Refusing "
                    "rather than running an adiabatic wall under a heat-flux boundary condition.");
            const std::vector<scalar> Tsn = Tp.snGrad(T.internal);
            std::vector<scalar> hGrad(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
                hGrad[static_cast<std::size_t>(i)] = thermoCpvOf(pw[i], Tw[i], thermo) * Tsn[i];
            hp->setGradient(std::move(hGrad));
        }
        else if (dynamic_cast<const MixedPatchField<scalar>*>(&Tp))
        {
            // mixedEnergy. The valueFraction is COPIED from T -- it is a blend factor, dimensionless, and
            // the same in energy space as in temperature space. inletOutlet reaches here too: OpenFOAM
            // gives he a plain mixedEnergy whose fraction is T's flux switch, and brae builds an
            // inletOutlet for he whose updateFromFlux ran off the same phi in the same iteration, so the
            // copy is an identity there and a real assignment on a plain `mixed` wall.
            auto* hp = dynamic_cast<MixedPatchField<scalar>*>(he.boundary[pi].get());
            if (!hp)
                throw std::runtime_error(
                    "brae: T on patch '" + patches[pi].name + "' is a mixed patch, which OpenFOAM maps to "
                    "mixedEnergy, but he's patch there is not of the mixed family. Refusing rather than "
                    "leaving the energy boundary at its construction-time coefficients.");

            const std::vector<scalar>* vf = Tp.valueFractionPtr();
            if (!vf || static_cast<label>(vf->size()) != n)
                throw std::runtime_error(
                    "brae: T on patch '" + patches[pi].name + "' is a mixed patch with no valueFraction. "
                    "OpenFOAM's mixedEnergy copies it verbatim; there is nothing to copy.");

            const std::vector<scalar> Tref = Tp.refValues();
            std::vector<scalar> hRef(static_cast<std::size_t>(n));
            for (label i = 0; i < n; ++i)
                hRef[static_cast<std::size_t>(i)] = thermoHeOf(pw[i], Tref[i], thermo);

            // Cpv is evaluated at Tw -- the patch VALUE, which on a mixed patch is the blend and not the
            // refValue. Using the refValue here would be a different number the moment the blend is not
            // 1, and on the liquid path Cpv varies with T by ~1% per 10 K.
            const std::vector<scalar>* Tgrad = Tp.refGradPtr();
            std::vector<scalar> hGrad(static_cast<std::size_t>(n), scalar(0));
            if (Tgrad)
                for (label i = 0; i < n; ++i)
                    hGrad[static_cast<std::size_t>(i)] =
                        thermoCpvOf(pw[i], Tw[i], thermo) * (*Tgrad)[static_cast<std::size_t>(i)];

            hp->setValueFraction(*vf);
            hp->setRefValues(std::move(hRef));
            hp->setRefGrad(std::move(hGrad));
        }
        // zeroGradient falls through with nothing written, and that is the mirror rather than a gap.
        // OpenFOAM maps it to gradientEnergy as well, whose gradient is Cpv*Tw.snGrad(); zeroGradient's
        // snGrad() is an identically ZERO field (zeroGradientFvPatchField.H), so the prescribed gradient
        // is zero on every iteration and for every thermo. A fixedGradient patch carrying gradient 0 has
        // the same four coefficients as a zeroGradient one (valueInternalCoeffs 1, valueBoundaryCoeffs 0,
        // gradientInternalCoeffs 0, gradientBoundaryCoeffs 0) and the same evaluate, so brae's
        // ZeroGradientPatchField IS gradientEnergy here, exactly.
        //
        // calculated, empty, symmetry, symmetryPlane, wedge and slip fall through too, and so do they in
        // OpenFOAM: none of them isA fixedValue, zeroGradient, fixedGradient or mixed, so heBoundaryTypes
        // leaves he with T's own type and there is no energy condition to update.
    }
}

}   // namespace cpu
}   // namespace brae
