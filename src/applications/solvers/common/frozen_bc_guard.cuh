#pragma once
// refuseFrozenPerStepBC -- the guard for boundary conditions that need a PER-STEP driver update.
//
// The patch-field factory ACCEPTS fixedMean, fanPressure, codedFixedValue and codedMixed on the strength
// of a per-step update its own comment promises -- "the solver recomputes refValue every step" -- and
// only some drivers keep that promise: gpuPimpleFoam maintains all four (collectFixedMean /
// collectFanPressure / setupCodedBCs), gpuSimpleFoam maintains the coded pair. On every other driver the
// patch was built from the file `value` and never touched again: a frozen boundary where the case asked
// for a maintained mean, a fan curve, or a compiled snippet -- the unhonoured-contract pattern this tree
// keeps finding, with the refusal written (collectFixedMean's own message) but unreachable from the
// drivers that needed it. So those drivers call this at their READ sites -- the last place the
// dictionary type still exists -- and refuse by name.
#include "foam_field_reader.cuh"
#include <stdexcept>
#include <string>

namespace brae {

template<class T>
inline void refuseFrozenPerStepBC(
    const FieldData<T>&  fd,
    const std::string&   fieldName,
    const char*          driver,
    bool                 codedMaintained)
{
    for (const auto& b : fd.boundary)
    {
        const bool frozen =
            b.type == "fixedMean"
         || b.type == "fanPressure"
         || (!codedMaintained && (b.type == "codedFixedValue" || b.type == "codedMixed"));
        if (frozen)
            throw std::runtime_error(
                std::string("brae: ") + driver + " builds " + fieldName + " patch '" + b.name
                + "' of type '" + b.type + "' from its file `value` and never updates it, where "
                "OpenFOAM recomputes it every updateCoeffs (fixedMeanFvPatchField / "
                "fanPressureFvPatchScalarField / codedFixedValueFvPatchField). gpuPimpleFoam "
                "maintains these per step, and gpuSimpleFoam maintains the coded pair; this driver "
                "does not. Refusing rather than freezing the boundary.");
    }
}

} // namespace brae
