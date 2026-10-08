#pragma once
// The alpha SUB-CYCLE on the device -- alphaEqnSubCycle.H, which 33 of the 44 shipped interFoam
// tutorials use (23 of them at nAlphaSubCycles 3).
//
// provenance:
//   openfoam:  applications/solvers/multiphase/VoF/alphaEqnSubCycle.H:31-45,
//              OpenFOAM/db/Time/Time.C:1006-1009 (subCycle divides deltaT),
//              OpenFOAM/db/Time/subCycleTime.H (subCycleField restores the old-time field)
//   host:      src/applications/solvers/interFoam/inter_solve_cpp.cu, alphaEqnSubCycle -- the ORACLE.
//   tests:     tests/test_device_alpha_subcycle.cu
//
// WHY IT IS ITS OWN FILE AND ITS OWN GATE. The sub-cycle is four lines of arithmetic and every one of
// the four ways to get it wrong leaves a BOUNDED, PLAUSIBLE alpha field that no boundedness gate and no
// eyeball sees:
//
//   a  RUNNING THE SUB-STEPS AT THE FULL deltaT. Time::subCycle divides it; a port that passes the
//      solver's deltaT through advects n times as far and simply looks like a faster interface.
//
//   b  RESTARTING EACH SUB-STEP FROM THE TIME STEP'S OLD ALPHA instead of the previous sub-step's
//      result. That advects by deltaT/n in total rather than by deltaT -- the opposite error, and just
//      as invisible.
//
//   c  TAKING THE LAST rhoPhi INSTEAD OF THE TIME-WEIGHTED SUM sum_k (dt_k/dt_total)*rhoPhi_k. rhoPhi
//      is what the momentum equation is built on, so this one does not show in alpha at all -- it shows
//      in U, one equation later.
//
//   d  WRITING alpha.oldTime(). subCycleField restores it in its destructor, so the PIMPLE outer loop's
//      second pass starts from the same place as the first. Here `alpha1Old` is const and the signature
//      is what enforces it.
//
// THE WEIGHT IS RECOMPUTED FROM EACH SUB-STEP'S OWN dt rather than assumed to be 1/n, so an LTS path
// with unequal sub-steps needs no change here.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include <functional>

namespace brae {

// One sub-step: advance `alpha` from `alphaOld` over `deltaT`, and leave that sub-step's rhoPhi behind.
// A callback rather than a fixed call because the caller owns what a step IS -- the explicit corrector
// loop, or the MULESCorr pre-solve plus its correctors -- and owns the host-side boundary evaluation
// between correctors that neither can do for itself.
using DeviceAlphaEqnStep = std::function<void(const DeviceBuffer<scalar>& alphaOld,
                                              scalar                     deltaT,
                                              DeviceBuffer<scalar>&      alpha,
                                              DeviceBuffer<scalar>&      rhoPhiInt,
                                              DeviceBuffer<scalar>&      rhoPhiBnd)>;

void deviceAlphaEqnSubCycle(
    int                          nAlphaSubCycles,
    scalar                       totalDeltaT,
    DeviceBuffer<scalar>&        alpha1,
    const DeviceBuffer<scalar>&  alpha1Old,     // never written -- see d
    DeviceBuffer<scalar>&        rhoPhiInt,
    DeviceBuffer<scalar>&        rhoPhiBnd,
    const DeviceAlphaEqnStep&    step);

} // namespace brae
