#pragma once
// The pair of lines fv::limitTemperature prints on every call, and the count behind them.
//
// provenance:
//   openfoam: src/fvOptions/corrections/limitTemperature/limitTemperature.C:200-215
//   brae:     src/finiteVolume/cfdTools/general/fvOptions/limit_temperature_report.cu
//   tests:    tests/limit_temperature_vs_openfoam.sh
//
// OpenFOAM emits these unconditionally from correct(he) -- once for the lower limit, once for the upper
// -- and they are the ONLY visible evidence that the option did anything:
//
//     limitTemperature=limitT, Type=Lower, LimitedCells=0, CellsPercent=0, Tmin=101, UnlimitedTmin=298
//     limitTemperature=limitT, Type=Upper, LimitedCells=0, CellsPercent=0, Tmax=1000, UnlimitedTmax=298
//
// brae applied the clamp on both arms and printed nothing, so a case where the clamp bit looked exactly
// like a case where it did not -- and "how many cells did the option touch" is the first question asked
// when a compressible run goes wrong. It is also what makes a gate on this option non-vacuous: the
// COUNTS are comparable against OpenFOAM's own log line for line, where the converged field is not.
#include "cf_types.cuh"

namespace brae {

// One formatter for the host reference and the CUDA arm, for the same reason bound_report has one: a
// diagnostic whose whole value is being diff-able against OpenFOAM's log must not be printf'd twice.
// `isLower` picks Type=Lower/Upper and the Tmin=/Tmax= + UnlimitedTmin=/UnlimitedTmax= spellings.
void printLimitTemperature(
    const char* optionName,     // the fvOptions dict key -- OF prints name_, not the type
    bool        isLower,
    label       limitedCells,
    label       totalCells,
    scalar      limitValue,     // Tmin_ or Tmax_ as the case wrote it
    scalar      unlimitedT);    // min(T) or max(T) BEFORE this call changed anything

} // namespace brae
