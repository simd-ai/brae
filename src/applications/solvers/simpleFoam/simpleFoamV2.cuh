#pragma once
// DISPATCH for the rebuilt simpleFoam -- the envelope guard that stands in front of it.
//
// The rebuilt path (UEqn.cu + pEqn.cu + simpleFoam.cu, each validated against a _cpp reference that is
// itself validated against OpenFOAM) covers a STRICT SUBSET of what the old solver runs. Making it
// reachable therefore has two halves, and the second is the one that matters:
//
//   1. `application simpleFoam` can select it;
//   2. a case OUTSIDE its envelope is REFUSED, with the reason, and never silently handed to a different
//      algorithm.
//
// Silent fallback is the failure mode this whole rebuild exists to remove. brae has already shipped a
// solver that read MRFProperties, ignored it, converged, and reported nothing wrong. A path that quietly
// degrades to the old solver -- or worse, quietly solves plain SIMPLE for a case asking for SIMPLEC --
// reproduces exactly that, and the user cannot tell from the output which algorithm ran.
//
// Selection is OPT-IN (BRAE_SIMPLEFOAM_V2=1) while the envelope is small. It is a switch on WHICH
// implementation runs, not on whether the answer is checked: when it is off nothing changes, and when it
// is on the case either runs on the new path or stops.
#include <string>
#include "cf_types.cuh"   // scalar
#include <vector>

namespace brae {
namespace gpu {

struct EnvelopeReport
{
    bool supported = false;
    std::vector<std::string> blockers;   // why the case cannot run on the new path
    std::vector<std::string> notices;    // supported, but brae does something the case did not literally ask for
};

// Inspect the case dictionaries and decide whether the rebuilt path can run it FAITHFULLY.
//
// Every check below corresponds to a component the new path does not implement, and each one is a
// defect that would otherwise be silent -- a converged, plausible, wrong answer:
//
//   MRF / fvOptions            change the momentum equation; UEqn.cu and pEqn.cu refuse them
//   fixedFluxPressure on p     pEqn.H resets its gradient through constrainPressure, which is not
//                              ported; brae maps the type to zeroGradient, the same BC only at zero flux
//   ddtSchemes != steadyState  this is the STEADY solver
//   div(phi,U) scheme          the CUDA UEqn implements upwind and linearUpwind; a case asking for
//                              limitedLinear or LUST would get a different discretisation, which is
//                              precisely the class of defect that hid in brae's LUST implicit weights.
//                              linearUpwind's NAMED gradient is checked too: `grad(U) cellLimited ...`
//                              is a different correction and is refused rather than approximated
//   turbulence model           only kEpsilon and laminar are wired
//   coupled patches            cyclic / cyclicAMI / cyclicACMI / processor are not handled by the
//                              rebuilt components
//
// `notices` carries the one thing brae substitutes ON PURPOSE and must still say out loud: a case asking
// for `GAMG` on p gets brae's AMG-preconditioned PCG, which is a different algorithm with a different
// iteration count.
EnvelopeReport simpleFoamV2Envelope(const std::string& caseDir);

// Is the rebuilt path selected for this run? (BRAE_SIMPLEFOAM_V2=1)
bool simpleFoamV2Selected();

// Run the case on the rebuilt path. Throws std::runtime_error listing every blocker if the case is
// outside the envelope -- the caller must NOT catch it and fall back.
// Returns the number of SIMPLE iterations performed.
int runSimpleFoamV2(const std::string& caseDir);

// What `laplacianSchemes` asked for: whether the non-orthogonal correction is on, and the name of a
// scheme that is recognised but not ported (empty when there is none). Hoisted from the .cu so the
// parse is unit-testable -- the `limited 0` conflation lived unseen while nothing could call it.
struct LaplacianScheme
{
    bool        corrected = true;   // OpenFOAM's default when the word is absent
    // `limited <k> corrected`: limitedSnGrad's coefficient. limitCoeff = 0 is this struct's OWN
    // "no limiter" sentinel (what k = 1 reduces to) -- it is NOT what `limited 0` means: the limiter
    // min(k*|orth|/((1-k)*|corr| + SMALL), 1) at k = 0 is identically zero, so `limited 0` suppresses
    // the correction entirely and maps to corrected = false.
    scalar      limitCoeff = 0.0;
    std::string unsupported;
};
LaplacianScheme laplacianScheme(const std::string& caseDir);

} // namespace gpu
} // namespace brae
