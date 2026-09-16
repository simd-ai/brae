// interFoam's time step -- see inter_solve_cpp.cuh for the provenance and for the six things in it
// that are not obvious from the code.
#include "inter_solve_cpp.cuh"

namespace brae {
namespace cpu {
namespace interFoam {

void alphaEqnSubCycle(label                      nAlphaSubCycles,
                      scalar                     totalDeltaT,
                      std::vector<scalar>&       alpha1,
                      const std::vector<scalar>& alpha1Old,
                      SurfaceScalarField&        rhoPhi,
                      const AlphaEqnStep&        step)
{
    if (nAlphaSubCycles < 1)
        throw std::runtime_error(
            "brae interFoam: nAlphaSubCycles must be at least 1; alphaControls.H reads it with "
            "get<label> and the sub-cycle loop runs it that many times.");
    if (totalDeltaT <= scalar(0))
        throw std::runtime_error("brae interFoam: the sub-cycle needs a positive time step.");

    if (nAlphaSubCycles == 1)
    {
        // alphaEqnSubCycle.H:31-34 -- the plain branch. NOT a one-iteration sub-cycle: OpenFOAM does
        // not construct a subCycle at all here, so deltaT is untouched and nothing is saved or
        // restored. Routing this through the loop below would be equivalent arithmetically and would
        // hide that `nAlphaSubCycles 1` is the ordinary path, not a degenerate case.
        step(alpha1Old, totalDeltaT, alpha1, rhoPhi);
        return;
    }

    // a: Time::subCycle divides deltaT by nSubCycles (Time.C:1006-1009).
    const scalar dtSub  = totalDeltaT / static_cast<scalar>(nAlphaSubCycles);
    const scalar weight = dtSub / totalDeltaT;

    SurfaceScalarField rhoPhiSum;
    bool               sumStarted = false;

    // b: each sub-step starts from the PREVIOUS sub-step's result. The first starts from the real old
    // time, and `alpha1Old` is const so it cannot be walked forward by accident -- the carried state
    // is a local.
    std::vector<scalar> carried = alpha1Old;

    for (label k = 0; k < nAlphaSubCycles; ++k)
    {
        SurfaceScalarField rhoPhiK;
        step(carried, dtSub, alpha1, rhoPhiK);
        carried = alpha1;

        // c: rhoPhiSum += (deltaT/totalDeltaT)*rhoPhi, accumulated per sub-step. The weight is
        // recomputed from the sub-step's own dt rather than assumed to be 1/n, so an LTS path with
        // unequal sub-steps needs no change here.
        if (!sumStarted)
        {
            rhoPhiSum.internal.assign(rhoPhiK.internal.size(), scalar(0));
            rhoPhiSum.boundary.resize(rhoPhiK.boundary.size());
            for (std::size_t pi = 0; pi < rhoPhiK.boundary.size(); ++pi)
                rhoPhiSum.boundary[pi].assign(rhoPhiK.boundary[pi].size(), scalar(0));
            sumStarted = true;
        }
        for (std::size_t f = 0; f < rhoPhiK.internal.size(); ++f)
            rhoPhiSum.internal[f] += weight * rhoPhiK.internal[f];
        for (std::size_t pi = 0; pi < rhoPhiK.boundary.size(); ++pi)
            for (std::size_t i = 0; i < rhoPhiK.boundary[pi].size(); ++i)
                rhoPhiSum.boundary[pi][i] += weight * rhoPhiK.boundary[pi][i];
    }

    rhoPhi = rhoPhiSum;
    // d: alpha1Old is never written. subCycleField restores the old-time field in its destructor so
    // that the PIMPLE outer loop's next pass starts from the same place; here it is simply const, and
    // the signature is what enforces it.
}


const char* stageName(Stage s)
{
    switch (s)
    {
        case Stage::courantNo:         return "CourantNo";
        case Stage::alphaCourantNo:    return "alphaCourantNo";
        case Stage::setDeltaT:         return "setDeltaT";
        case Stage::advanceTime:       return "++runTime";
        case Stage::alphaControls:     return "alphaControls";
        case Stage::alphaEqnSubCycle:  return "alphaEqnSubCycle";
        case Stage::mixtureCorrect:    return "mixture.correct";
        case Stage::UEqn:              return "UEqn";
        case Stage::pEqn:              return "pEqn";
        case Stage::turbulenceCorrect: return "turbulence.correct";
        case Stage::write:             return "runTime.write";
    }
    return "?";
}


void runTimeStep(const LoopControls& ctl, const SolverHooks& hooks)
{
    if (!hooks.run)
        throw std::runtime_error("brae interFoam: runTimeStep needs a stage hook.");
    if (ctl.nOuterCorrectors < 1 || ctl.nCorrectors < 1)
        throw std::runtime_error(
            "brae interFoam: nOuterCorrectors and nCorrectors must both be at least 1.");

    // The step is chosen from the PREVIOUS step's fluxes, before the time advances (interFoam.C:93-105).
    hooks.run(Stage::courantNo);
    hooks.run(Stage::alphaCourantNo);
    hooks.run(Stage::setDeltaT);
    hooks.run(Stage::advanceTime);

    for (label outer = 0; outer < ctl.nOuterCorrectors; ++outer)
    {
        const bool finalIter = (outer == ctl.nOuterCorrectors - 1);

        hooks.run(Stage::alphaControls);
        hooks.run(Stage::alphaEqnSubCycle);
        // Called AGAIN here even though alphaEqn calls it per corrector: the sub-cycle may have moved
        // alpha since the last one, and UEqn below needs the mixture built from where alpha ended up.
        hooks.run(Stage::mixtureCorrect);

        // pimple.frozenFlow(): alpha keeps advancing on a fixed velocity field. Momentum, pressure AND
        // turbulence are all skipped -- `continue` jumps past the turbulence corrector too, which a
        // port that only guarded UEqn and pEqn would keep running.
        if (ctl.frozenFlow) continue;

        hooks.run(Stage::UEqn);
        for (label corr = 0; corr < ctl.nCorrectors; ++corr) hooks.run(Stage::pEqn);

        if (!ctl.turbOnFinalIterOnly || finalIter) hooks.run(Stage::turbulenceCorrect);
    }

    hooks.run(Stage::write);
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
