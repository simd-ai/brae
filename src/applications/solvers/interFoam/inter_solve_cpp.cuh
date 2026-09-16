#pragma once
// interFoam's time step -- the sub-cycle and the order of the PIMPLE loop. Host reference.
//
// provenance:
//   openfoam:
//     file: applications/solvers/multiphase/interFoam/interFoam.C:90-176
//     also: applications/solvers/multiphase/VoF/alphaEqnSubCycle.H:1-36
//           src/OpenFOAM/algorithms/subCycle/subCycle.H   (subCycleField: the old-time save/restore)
//           src/OpenFOAM/db/Time/Time.C:993-1014          (Time::subCycle: deltaT /= nSubCycles)
//   brae:
//     reference: this header
//     cuda:      (pending)
//     tests:     tests/test_inter_solve_cpp.cu
//
// TWO THINGS LIVE HERE, and they fail in different ways.
//
// ============ 1. THE ALPHA SUB-CYCLE ============
//
// 23 of the 44 shipped tutorials set nAlphaSubCycles to 3, eight to 2, one each to 4 and 5. Only 11 run
// without sub-cycling at all. It exists because the interface has a much tighter stability limit than
// the flow, and sub-cycling buys that back without shrinking the whole time step.
//
//     if (nAlphaSubCycles > 1)
//     {
//         totalDeltaT = runTime.deltaT();
//         rhoPhiSum = 0;
//         for (subCycle<volScalarField> alphaSubCycle(alpha1, nAlphaSubCycles); !(++alphaSubCycle).end();)
//         {
//             #include "alphaEqn.H"
//             rhoPhiSum += (runTime.deltaT()/totalDeltaT)*rhoPhi;
//         }
//         rhoPhi = rhoPhiSum;
//     }
//     else { #include "alphaEqn.H" }
//
//     rho == alpha1*rho1 + alpha2*rho2;
//
// FOUR THINGS TO GET RIGHT:
//
//   a. EACH SUB-STEP USES deltaT/nAlphaSubCycles. Time::subCycle divides deltaT_ and deltaT0_ by
//      nSubCycles (Time.C:1006-1009), so alphaEqn's rDeltaT inside the loop is n times larger. Running
//      the sub-steps at the full deltaT advances the interface n times too far and MULES will happily
//      keep it bounded while it does.
//
//   b. EACH SUB-STEP'S OLD TIME IS THE PREVIOUS SUB-STEP'S RESULT, not the time step's start. That is
//      what makes it a sequence rather than n copies of the same step.
//
//   c. rhoPhi IS A TIME-WEIGHTED SUM: sum_k (dt_k/dt_total)*rhoPhi_k. Not the last sub-step's value,
//      and not a plain sum. With uniform sub-steps the weights are all 1/n, so it is the arithmetic
//      mean -- and a port that took the last value instead would be wrong by a factor that looks like
//      a flux and hands UEqn a mass flux inconsistent with the alpha that was actually advected.
//      OpenFOAM writes the weighted form because LTS makes the sub-steps unequal.
//
//   d. alpha1.oldTime() IS RESTORED AFTERWARDS. subCycleField saves a copy on construction and puts it
//      back in its destructor (subCycle.H:40-55). The PIMPLE outer loop runs alphaEqnSubCycle AGAIN on
//      its next iteration, and it must restart from the same old time -- otherwise outer iteration 2
//      advances from iteration 1's result and the field moves two time steps in one.
//
// ============ 2. THE ORDER OF THE PIMPLE LOOP ============
//
// interFoam.C:91-176, with the mesh-motion branch omitted (brae has no moving mesh here):
//
//     CourantNo, alphaCourantNo, setDeltaT     <- BEFORE the time advances
//     ++runTime
//     while (pimple.loop())
//     {
//         alphaControls
//         alphaEqnSubCycle                     <- alpha moves FIRST, before the momentum
//         mixture.correct()                    <- ...and the mixture is refreshed from the new alpha
//         if (pimple.frozenFlow()) continue;
//         UEqn
//         while (pimple.correct()) { pEqn }
//         if (pimple.turbCorr()) turbulence->correct()
//     }
//
// THE ORDER IS THE CONTENT. alpha is advanced before the momentum predictor, so UEqn sees the NEW rho
// and the NEW surface tension; and mixture.correct() runs again after the sub-cycle even though
// alphaEqn already called it per corrector, because the sub-cycle may have moved alpha since. Running
// UEqn first -- the order every single-phase solver uses -- gives a momentum equation built on last
// step's density field, which converges and is wrong by the density ratio at the interface.
//
// The driver below takes the stages as callbacks and calls them in this order, so the order is a
// property of executable code rather than of a comment. The gate injects recording hooks and asserts
// the sequence against interFoam.C.
#include "cf_types.cuh"
#include "fvc.cuh"
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

// One sub-step: advance `alpha` over `dtSub` starting from `alphaOld`, and produce that sub-step's
// rhoPhi. The caller supplies alphaEqn; this component owns only the sequencing and the weighting.
using AlphaEqnStep = std::function<void(const std::vector<scalar>& alphaOld,
                                        scalar                     dtSub,
                                        std::vector<scalar>&       alphaNew,
                                        SurfaceScalarField&        rhoPhi)>;

// alphaEqnSubCycle.H:1-30. `alpha1Old` is the field's old time; it is READ by every sub-step and is
// left UNCHANGED on return -- see note d.
void alphaEqnSubCycle(label                      nAlphaSubCycles,
                      scalar                     totalDeltaT,
                      std::vector<scalar>&       alpha1,
                      const std::vector<scalar>& alpha1Old,
                      SurfaceScalarField&        rhoPhi,
                      const AlphaEqnStep&        step);

// The stages of one interFoam time step, in the order interFoam.C runs them. Used by the driver below
// and by the gate that records them.
enum class Stage
{
    courantNo,
    alphaCourantNo,
    setDeltaT,
    advanceTime,
    alphaControls,
    alphaEqnSubCycle,
    mixtureCorrect,
    UEqn,
    pEqn,
    turbulenceCorrect,
    write
};

const char* stageName(Stage s);

struct SolverHooks
{
    std::function<void(Stage)> run;      // called once per stage, in order
};

struct LoopControls
{
    label nOuterCorrectors = 1;
    label nCorrectors      = 1;          // pressure correctors per outer iteration
    bool  frozenFlow       = false;      // pimple.frozenFlow(): skip momentum and pressure entirely
    // turbulence->correct() runs ONCE per time step by default, on the final outer iteration --
    // turbOnFinalIterOnly is TRUE in OpenFOAM, and a port that advanced the closure every outer
    // corrector runs it nOuterCorrectors times per physical step.
    bool  turbOnFinalIterOnly = true;
};

// interFoam.C:91-176. Calls hooks.run(stage) in OpenFOAM's order; the mesh-motion branch is omitted
// because brae has no moving mesh in a VoF case, and adding one must be a decision taken here.
void runTimeStep(const LoopControls& ctl, const SolverHooks& hooks);

} // namespace interFoam
} // namespace cpu
} // namespace brae
