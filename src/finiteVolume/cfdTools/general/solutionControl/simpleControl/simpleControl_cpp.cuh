#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's SIMPLE solution control.
//
// provenance:
//   openfoam:
//     symbol: Foam::simpleControl
//     file:   src/finiteVolume/cfdTools/general/solutionControl/simpleControl/simpleControl.C
//     base:   Foam::solutionControl
//             src/finiteVolume/cfdTools/general/solutionControl/solutionControl/solutionControl.C
//   brae:
//     reference: .../simpleControl_cpp.cu
//     cuda:      (host-only by nature -- CONFIGURATION, not GPU work)
//     tests:     tests/test_simple_control_cpp.cu
//
// WHERE THE KEYS COME FROM. solutionControl::dict() is
//
//     mesh_.solutionDict().subOrEmptyDict(algorithmName_)      // solutionControl.C:300-303
//
// i.e. the `SIMPLE` sub-dictionary of system/fvSolution, and subOrEmptyDict means an ABSENT SIMPLE block
// is not an error -- every key then takes its default. The full set, from solutionControl.C:47-57 (and
// confirmed against `ofscan schema solutionControl`):
//
//     nNonOrthogonalCorrectors   label   getOrDefault    0
//     momentumPredictor          bool    getOrDefault    true
//     transonic                  bool    getOrDefault    false
//     consistent                 bool    getOrDefault    false
//     frozenFlow                 bool    getOrDefault    false
//     residualControl                    subOrEmptyDict
//
// residualControl IS PARSED IN THE FLAT FORM for SIMPLE and that is not a simplification:
// simpleControl::read() calls solutionControl::read(true) -- absTolOnly (simpleControl.C:42-46) -- so
//
//     residualControl { p 1e-2; U 1e-3; "(k|epsilon)" 1e-3; }
//
// is name -> absTol, with relTol unused. PIMPLE passes false there and takes the dictionary form
// { tolerance ...; relTol ...; }, which is why this is a per-algorithm fact rather than a global one.
// The keys are matched against field names by REGEX (solutionControl.C:142-157, applyToField with
// useRegEx defaulting true), so `"(k|epsilon)"` legitimately answers for two fields.
//
// CONVERGENCE. criteriaSatisfied (simpleControl.C:49-90):
//   * no residualControl at all              -> never converges, run to endTime;
//   * otherwise every field that MATCHED a control must have its INITIAL residual below absTol,
//     and at least one check must actually have been performed (`checked && achieved`).
// The initial residual is the one from the first solve of that field this iteration, not the final one --
// using the final residual would declare convergence as soon as the linear solver did its job.
//
// The loop order (simpleControl.C:136-160) is:
//     setFirstIterFlag; read(); if (initialised && criteriaSatisfied) writeAndEnd();
//     else { initialised = true; storePrevIterFields(); } return runTime.loop();
// so the criteria are only tested from the SECOND iteration on, and prevIter (which field relaxation
// needs) is stored on every iteration that is not the converged one.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace brae {
namespace cpu {

struct SimpleControlDict
{
    label  nNonOrthogonalCorrectors = 0;
    bool   momentumPredictor = true;
    bool   transonic = false;
    bool   consistent = false;
    bool   frozenFlow = false;
    // name (possibly a regex) -> absTol, in file order. A vector, not a map: OpenFOAM matches in
    // declaration order and returns the FIRST match (applyToField), so order is semantics.
    std::vector<std::pair<std::string, scalar>> residualControl;
};

// Parse the SIMPLE block of fvSolution. `fvSolution` is the whole dictionary; an absent SIMPLE block
// yields all defaults, as subOrEmptyDict does.
SimpleControlDict readSimpleControl(const FoamDict& fvSolution);

class SimpleControl
{
public:
    explicit SimpleControl(SimpleControlDict d) : d_(std::move(d)) {}

    const SimpleControlDict& dict() const { return d_; }

    bool momentumPredictor() const { return d_.momentumPredictor; }
    bool consistent()        const { return d_.consistent; }
    bool transonic()         const { return d_.transonic; }
    bool frozenFlow()        const { return d_.frozenFlow; }

    // Non-orthogonal corrector loop: runs nNonOrthogonalCorrectors + 1 times, then resets.
    // (solutionControlI.H:78-95 -- ++corrNonOrtho_; return corrNonOrtho_ <= nNonOrthCorr_ + 1)
    bool correctNonOrthogonal();
    bool finalNonOrthogonalIter() const { return corrNonOrtho_ == d_.nNonOrthogonalCorrectors + 1; }

    // criteriaSatisfied: `initialResiduals` maps field name -> the initial residual of its first solve
    // this iteration. Returns false when no control is configured, and requires that at least one
    // control actually matched a solved field.
    bool criteriaSatisfied(const std::map<std::string, scalar>& initialResiduals) const;

    // One SIMPLE iteration. Returns false when the run should stop.
    //   iteration  the 1-based iteration about to run
    //   maxIters   endTime/deltaT, i.e. what runTime.loop() bounds
    bool loop(label iteration, label maxIters,
              const std::map<std::string, scalar>& initialResiduals);

    bool converged() const { return converged_; }

private:
    SimpleControlDict d_;
    label corrNonOrtho_ = 0;
    bool  initialised_ = false;
    bool  converged_ = false;
};

// applyToField: index of the first residualControl entry whose (regex) name matches, else -1.
label applyToField(const std::vector<std::pair<std::string, scalar>>& ctrl,
                   const std::string& fieldName);

} // namespace cpu
} // namespace brae
