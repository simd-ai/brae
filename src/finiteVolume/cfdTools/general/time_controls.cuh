#pragma once
// Courant-limited time-step control -- OF src/finiteVolume/cfdTools/general/include/
// {readTimeControls,CourantNo,setInitialDeltaT,setDeltaT}.H
//
// WHY IT LIVES HERE, and not inside gpuPimpleFoam. In OpenFOAM these are #include files that EVERY
// transient solver pulls in unchanged -- pimpleFoam, rhoPimpleFoam, interFoam, sonicFoam all use the
// same four. None of them reimplements the formula. brae had none of it: gpuPimpleFoam refused
// `adjustTimeStep yes` outright, which is 11 of OpenFOAM's 35 pimpleFoam tutorials and the default for
// transient cases. Putting it here means the compressible transient solver gets it by including one
// header, exactly as rhoPimpleFoam.C does.
//
// EXACT OF SEMANTICS, transcribed rather than approximated:
//
//   CourantNo.H       sumPhi = surfaceSum(mag(phi))            [per cell]
//                     CoNum     = 0.5*gMax(sumPhi/V)*deltaT
//                     meanCoNum = 0.5*(gSum(sumPhi)/gSum(V))*deltaT
//
//   setInitialDeltaT  if (timeIndex == 0 && CoNum > SMALL)
//                         deltaT = min(maxCo*deltaT/CoNum, min(deltaT, maxDeltaT))
//
//   setDeltaT         maxDeltaTFact = maxCo/(CoNum + SMALL)
//                     deltaTFact    = min(min(maxDeltaTFact, 1 + 0.1*maxDeltaTFact), 1.2)
//                     deltaT        = min(deltaTFact*deltaT, maxDeltaT)
//
// The 1.2 cap and the `1 + 0.1*maxDeltaTFact` damping are not arbitrary: OF's own header says
// "Reduction of time-step is immediate, but increase is damped to avoid unstable oscillations". A
// reimplementation that merely scaled by maxCo/CoNum would ring. Both are reproduced.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// OF readTimeControls.H. maxDeltaT defaults to GREAT (i.e. no cap) exactly as OF does.
struct TimeControls
{
    bool   adjustTimeStep = false;
    scalar maxCo          = 1.0;
    scalar maxDeltaT      = 1.0e300;   // OF: GREAT

    static TimeControls read(const FoamDict& controlDict)
    {
        TimeControls tc;
        const std::string a = controlDict.wordOr("adjustTimeStep", "no");
        tc.adjustTimeStep = (a == "yes" || a == "true" || a == "on" || a == "1");
        tc.maxCo     = controlDict.scalarOr("maxCo", 1.0);
        tc.maxDeltaT = controlDict.scalarOr("maxDeltaT", 1.0e300);
        return tc;
    }
};

struct CourantNumbers
{
    scalar CoNum     = 0;
    scalar meanCoNum = 0;
};

// surfaceSum(mag(phi)) per cell -- OF fvc::surfaceSum, which adds |phi| to BOTH the owner and the
// neighbour of every internal face, and to the owner of every boundary face. Missing the boundary half
// understates Co near inlets, which is exactly where the limiting cell usually is.
inline std::vector<scalar> surfaceSumMagPhi(
    const std::vector<label>& owner,
    const std::vector<label>& neighbour,
    const std::vector<scalar>& phiInternal,
    const std::vector<scalar>& phiBoundary,
    label nCells,
    label nInternalFaces)
{
    std::vector<scalar> sumPhi(static_cast<std::size_t>(nCells), 0.0);
    for (label f = 0; f < nInternalFaces && f < (label)phiInternal.size(); ++f)
    {
        const scalar a = std::fabs(phiInternal[f]);
        if (owner[f] >= 0 && owner[f] < nCells)         sumPhi[owner[f]]     += a;
        if (f < (label)neighbour.size() && neighbour[f] >= 0 && neighbour[f] < nCells)
            sumPhi[neighbour[f]] += a;
    }
    for (std::size_t b = 0; b < phiBoundary.size(); ++b)
    {
        const label f = nInternalFaces + static_cast<label>(b);
        if (f < (label)owner.size() && owner[f] >= 0 && owner[f] < nCells)
            sumPhi[owner[f]] += std::fabs(phiBoundary[b]);
    }
    return sumPhi;
}

// OF CourantNo.H. `sumPhi` is surfaceSum(mag(phi)) per cell -- the caller supplies it because the flux
// lives on the device and summing it there is the solver's job, not this header's.
inline CourantNumbers courantNo(
    const std::vector<scalar>& sumPhi,
    const std::vector<scalar>& V,
    scalar deltaT)
{
    CourantNumbers c;
    if (sumPhi.empty() || V.empty()) return c;
    scalar maxRatio = 0, sPhi = 0, sV = 0;
    for (std::size_t i = 0; i < sumPhi.size() && i < V.size(); ++i)
    {
        if (V[i] > 0) maxRatio = std::max(maxRatio, sumPhi[i]/V[i]);
        sPhi += sumPhi[i];
        sV   += V[i];
    }
    c.CoNum     = 0.5*maxRatio*deltaT;
    c.meanCoNum = (sV > 0) ? 0.5*(sPhi/sV)*deltaT : scalar(0);
    return c;
}

// OF setInitialDeltaT.H -- applied ONCE, before the first step, so the run starts at the requested
// Courant number instead of spending steps ramping to it.
inline scalar setInitialDeltaT(scalar deltaT, scalar CoNum, const TimeControls& tc)
{
    if (!tc.adjustTimeStep) return deltaT;
    const scalar kSmall = 1.0e-37;                       // OF SMALL
    if (CoNum <= kSmall) return deltaT;
    return std::min(tc.maxCo*deltaT/CoNum, std::min(deltaT, tc.maxDeltaT));
}

// OF setDeltaT.H -- applied every step. Reduction is immediate; growth is damped and capped at 1.2x.
inline scalar setDeltaT(scalar deltaT, scalar CoNum, const TimeControls& tc)
{
    if (!tc.adjustTimeStep) return deltaT;
    const scalar kSmall = 1.0e-37;
    const scalar maxDeltaTFact = tc.maxCo/(CoNum + kSmall);
    const scalar deltaTFact = std::min(std::min(maxDeltaTFact, scalar(1) + scalar(0.1)*maxDeltaTFact), scalar(1.2));
    return std::min(deltaTFact*deltaT, tc.maxDeltaT);
}

// ---------------------------------------------------------------------------------------------------
// THE VoF ADDITION: a SECOND Courant number, computed only where the interface is.
//
//   provenance: applications/solvers/multiphase/VoF/alphaCourantNo.H:34-54
//               src/transportModels/interfaceProperties/interfaceProperties.C:244-248 (nearInterface)
//               applications/solvers/multiphase/VoF/setDeltaT.H:36-53
//
// WHY A SECOND ONE AT ALL. The ordinary Courant number is a global maximum over every cell, so it is
// set by whatever corner of the domain has the fastest flow -- usually far from the interface. A VoF
// interface has its own, much tighter stability limit, and MULES does not protect against advecting it
// more than a cell per step. interFoam therefore limits the step by BOTH, and the shipped tutorials
// almost always set maxAlphaCo equal to or below maxCo (0.65/0.65 in 12 of them, 0.5/0.5 in 10).
//
// FOUR THINGS TO GET RIGHT:
//
// 1. maxAlphaCo IS MANDATORY. alphaCourantNo.H reads it with get<scalar> -- no default -- where
//    readTimeControls.H gives maxCo a default of 1. A case that turns on adjustTimeStep for interFoam
//    and omits maxAlphaCo is a FatalError, and defaulting it would run the interface unconstrained.
//
// 2. nearInterface() IS A 0/1 MASK, NOT A WEIGHT: pos0(alpha1 - 0.01)*pos0(0.99 - alpha1). pos0 is 1
//    at exactly zero, so the band is the CLOSED interval [0.01, 0.99]. A smooth weight, or pos instead
//    of pos0, changes which cells are counted at the edges of the band.
//
// 3. WITH NO INTERFACE THE ALPHA COURANT NUMBER IS ZERO, and the step is then limited by maxCo alone
//    -- maxAlphaCo/(0 + SMALL) is astronomically large and the min picks the other branch. That is
//    correct and it matters: a case that has not yet developed an interface must not be throttled.
//
// 4. THE TWO LIMITS COMBINE INSIDE maxDeltaTFact, BEFORE THE DAMPING:
//        maxDeltaTFact = min(maxCo/(CoNum + SMALL), maxAlphaCo/(alphaCoNum + SMALL))
//    and the 1.2 cap and the `1 + 0.1*maxDeltaTFact` growth damping are applied to that combined
//    value. Damping each separately and then taking the min is not the same number.

struct VoFTimeControls
{
    TimeControls base;
    scalar       maxAlphaCo = 0;     // MANDATORY -- see note 1

    static VoFTimeControls read(const FoamDict& controlDict)
    {
        VoFTimeControls tc;
        tc.base = TimeControls::read(controlDict);
        // get<scalar>, not getOrDefault: alphaCourantNo.H:34-37.
        tc.maxAlphaCo = controlDict.scalarOr("maxAlphaCo", scalar(-1));
        if (tc.base.adjustTimeStep && tc.maxAlphaCo < 0)
            throw std::runtime_error(
                "brae interFoam: controlDict sets `adjustTimeStep` but has no `maxAlphaCo`. OpenFOAM "
                "reads it with get<scalar> and has NO default (alphaCourantNo.H:34-37), unlike maxCo "
                "which defaults to 1. Defaulting it here would advance the interface with no limit of "
                "its own, which is the one thing the second Courant number exists to prevent.");
        return tc;
    }
};

// nearInterface() = pos0(alpha1 - 0.01)*pos0(0.99 - alpha1) -- interfaceProperties.C:244-248.
// A 0/1 mask over the CLOSED band [0.01, 0.99]; pos0 is 1 at exactly zero, so both ends are included.
inline std::vector<scalar> nearInterface(const std::vector<scalar>& alpha1)
{
    std::vector<scalar> mask(alpha1.size());
    for (std::size_t c = 0; c < alpha1.size(); ++c)
    {
        const scalar lo = alpha1[c] - scalar(0.01);
        const scalar hi = scalar(0.99) - alpha1[c];
        mask[c] = ((lo >= scalar(0)) ? scalar(1) : scalar(0))
                * ((hi >= scalar(0)) ? scalar(1) : scalar(0));
    }
    return mask;
}

// alphaCourantNo.H:42-54 -- the ordinary Courant formula with sumPhi masked to the interface band.
// Shares courantNo() rather than repeating it: the ONLY difference between the two is the mask, and
// writing the formula twice is two chances for the 0.5 or the gSum-of-ratios to drift apart.
inline CourantNumbers alphaCourantNo(
    const std::vector<scalar>& sumPhi,      // surfaceSum(mag(phi)), per cell
    const std::vector<scalar>& alpha1,
    const std::vector<scalar>& V,
    scalar                     deltaT)
{
    const std::vector<scalar> mask = nearInterface(alpha1);
    std::vector<scalar> masked(sumPhi.size());
    for (std::size_t c = 0; c < sumPhi.size() && c < mask.size(); ++c) masked[c] = mask[c]*sumPhi[c];
    // NOTE the denominators are NOT masked: meanAlphaCoNum is gSum(maskedPhi)/gSum(V), over the WHOLE
    // mesh volume, not over the interface cells' volume. It is a domain-average of an interface
    // quantity and reads small; the max is what limits the step.
    return courantNo(masked, V, deltaT);
}

// VoF setDeltaT.H:36-53. Both limits enter maxDeltaTFact BEFORE the damping -- see note 4.
// THE WRITE CADENCE IS PART OF THE TIME STEP, which is not obvious and is why this is here rather
// than wherever brae decides to write fields. Time::setDeltaT(scalar) takes `adjust = true` by
// default (Time.C:1178) and calls Time::adjustDeltaT(), which under `writeControl adjustableRunTime`
// SHORTENS the step so the next write time is landed on exactly. So a solver that reads maxCo and
// maxAlphaCo and ignores writeControl does not reproduce OpenFOAM's deltaT.
//
// MEASURED ON damBreak, whose controlDict says `writeControl adjustableRunTime; writeInterval 0.05`:
// OpenFOAM's first step is 0.000119904 -- that is 0.05/417 -- where brae's unclamped setDeltaTVoF
// gave 0.00012. The gap is in the fourth digit and it compounds: eleven steps later, on `endTime
// 0.004`, OpenFOAM ended at t = 0.00385757 and brae at 0.00385805.
struct WriteCadence
{
    bool   adjustable     = false;   // writeControl adjustableRunTime -- the only mode that adjusts
    scalar writeInterval  = 0;
    label  writeTimeIndex = 0;       // Time::writeTimeIndex_, which advance() below moves

    static WriteCadence read(const FoamDict& controlDict)
    {
        WriteCadence w;
        // BOTH spellings, from Time::writeControlNames (Time.C:53-62), which tabulates
        // wcAdjustableRunTime twice -- as "adjustable" and as "adjustableRunTime". damBreak's own
        // controlDict uses the short one, and matching only the long one made this whole clamp a
        // no-op on the very case it was measured against. Any other value, tabulated or not, leaves
        // adjustDeltaT a no-op exactly as it is in OpenFOAM.
        const std::string wc = controlDict.wordOr("writeControl", "timeStep");
        w.adjustable    = (wc == "adjustable" || wc == "adjustableRunTime");
        w.writeInterval = controlDict.scalarOr("writeInterval", scalar(0));
        if (w.adjustable && !(w.writeInterval > scalar(0)))
            throw std::runtime_error(
                "brae: controlDict says `writeControl adjustableRunTime` but gives no positive "
                "`writeInterval`. Time::adjustDeltaT divides by it (Time.C:1150).");
        return w;
    }

    // Time::operator++ (Time.C:1046-1074), and the two details there both matter: the time is the one
    // AFTER the step, the deltaT is the one that took it, and the index only ever moves FORWARD.
    void advance(scalar tSinceStart, scalar deltaT)
    {
        if (!adjustable) return;
        const label wi =
            static_cast<label>((tSinceStart + scalar(0.5)*deltaT)/writeInterval);
        if (wi > writeTimeIndex) writeTimeIndex = wi;
    }
};

// Time::adjustDeltaT() (Time.C:1136-1170). `tSinceStart` is value() - startTime_, which is what the
// drivers' own clocks already measure.
inline scalar adjustDeltaT(scalar deltaT, scalar tSinceStart, const WriteCadence& w)
{
    if (!w.adjustable) return deltaT;

    const scalar timeToNextWrite =
        std::max(scalar(0), scalar(w.writeTimeIndex + 1)*w.writeInterval - tSinceStart);
    const scalar n = timeToNextWrite/deltaT;

    // "For tiny deltaT the label can overflow" -- OpenFOAM leaves deltaT alone rather than wrapping.
    if (!(n < scalar(2147483647))) return deltaT;

    // nSteps can be < 1 so make sure at least 1
    const label nStepsToNextWrite = std::max(label(1), static_cast<label>(std::lround(n)));
    const scalar newDeltaT = timeToNextWrite/nStepsToNextWrite;

    // Control the increase of the time step to within a factor of 2 and the decrease within 5.
    return (newDeltaT >= deltaT) ? std::min(newDeltaT, scalar(2)*deltaT)
                                 : std::max(newDeltaT, scalar(0.2)*deltaT);
}

// setDeltaT.H, and the adjustDeltaT that Time::setDeltaT performs on the way in. `w` null is a case
// with no adjustable write cadence -- every gate fixture, and any case on `writeControl timeStep` or
// `runTime`, for which OpenFOAM's adjustDeltaT is a no-op anyway.
inline scalar setDeltaTVoF(scalar              deltaT,
                           scalar              CoNum,
                           scalar              alphaCoNum,
                           const VoFTimeControls& tc,
                           scalar              tSinceStart = 0,
                           const WriteCadence* w = nullptr)
{
    if (!tc.base.adjustTimeStep) return deltaT;
    const scalar kSmall = 1.0e-37;
    const scalar maxDeltaTFact = std::min(tc.base.maxCo/(CoNum + kSmall),
                                          tc.maxAlphaCo/(alphaCoNum + kSmall));
    const scalar deltaTFact =
        std::min(std::min(maxDeltaTFact, scalar(1) + scalar(0.1)*maxDeltaTFact), scalar(1.2));
    const scalar dt = std::min(deltaTFact*deltaT, tc.base.maxDeltaT);
    return w ? adjustDeltaT(dt, tSinceStart, *w) : dt;
}

}   // namespace brae
