#pragma once
// The line Foam::bound prints whenever it fires, and the reduction behind it.
//
// provenance:
//   openfoam: src/finiteVolume/cfdTools/general/bound/bound.C:38-46
//   brae:     src/finiteVolume/cfdTools/general/bound/bound_report.cu
//   tests:    tests/bound_message_vs_openfoam.sh, tests/test_bound_report.cu
//
// OpenFOAM computes min(vsf) unconditionally -- it is the guard bound() branches on -- and when the
// guard fires it prints the field's pre-bound min, max and average before touching it. brae clamped
// silently on both arms. What that cost is not hypothetical: the whole of item 78 (a substituted
// PBiCGStab preconditioned with `diagonal`, epsilon driven to the bound floor in 201 interior cells,
// nut = Cmu k^2/epsilon reaching 1.5e+17) was found by dumping fields at chosen iterations and
// comparing extremes against OpenFOAM, over several days. Real OpenFOAM on the same case printed TEN
// of these lines in the first eight iterations:
//
//     bounding epsilon, min: -33532.1 max: 1.2543e+07 average: 400769
//     bounding k, min: -2.93301 max: 21582.9 average: 1876.82
//
// The message is the diagnostic for that entire class, and brae was the only one of the two codes not
// emitting it.
#include "cf_types.cuh"

namespace brae {

// ONE formatter for both arms. The host reference and the device path must not drift on a diagnostic
// whose entire value is being line-comparable with OpenFOAM's own log, so neither writes its own
// printf. `%g` reproduces OpenFOAM's default Info precision (6 significant digits): the oracle lines
// above round-trip through it exactly.
void printBounding(
    const char* fieldName,
    scalar      minValue,
    scalar      maxValue,
    scalar      average);

} // namespace brae
