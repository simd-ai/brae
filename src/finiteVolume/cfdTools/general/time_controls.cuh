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
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// OF readTimeControls.H. maxDeltaT defaults to GREAT (i.e. no cap) exactly as OF does.
// OPENFOAM'S SMALL AND GREAT IN THE DOUBLE BUILD these solvers mirror (etc/bashrc: WM_PRECISION_OPTION=DP;
// scalar.H:120-133): doubleScalarSMALL = 1.0e-15 and doubleScalarGREAT = 1.0e+15 (doubleScalar.H:58-62).
// setDeltaT.H adds SMALL to each Courant number before it divides, setInitialDeltaT.H tests `CoNum > SMALL`, and
// readTimeControls.H defaults maxDeltaT to GREAT.
// THESE WERE 1.0e-37 AND 1.0e300 UNTIL 2026-10-05 -- the FLOAT build's VSMALL (floatScalar.H:64) and the double
// build's VGREAT, each under a comment naming the right constant. CoNum + 1e-37 is CoNum; CoNum + 1e-15 is a
// few ulp above it. So on every step the Courant number limits (deltaTFact below the 1.2 cap) brae's deltaT
// sat about 1e-15/CoNum relative from OpenFOAM's expression. MEASURED against OpenFOAM's own log, a step a row
// (tests/interfoam_write/clock/deltat_openfoam_log.sh): of laminar/damBreak's first 186 steps, 175 were not
// OpenFOAM's bits, up to 1.3e-15 relative, and of 306 with maxAlphaCo 0.5, 298, up to 2.3e-15; with 1e-15 all
// are. The same mistake had been found and fixed in ddtCorr (inter_peqn_cpp.cu, device_inter_peqn.cu).
// THE ONE PRODUCT THAT FEEDS A SUM, `1.0 + 0.1*maxDeltaTFact`, IS A FUSED MULTIPLY-ADD in this machine's
// OpenFOAM: interFoam's own code at 0x24278 (linuxARM64GccDPInt32Opt, v2412) is `fmadd d0, d12, d1, d0` with
// 1.0 and the constant 0.1 -- gcc -O3 contracts it on aarch64. It is written std::fma below, so that brae's
// step does not depend on how brae was compiled. (An x86-64 OpenFOAM built without -mfma rounds the product
// first; there the two differ by an ulp of deltaTFact on about one damped step in sixteen.)
//   BRAE_CONTROL_DELTAT_SMALL_FLOAT=1   a gate's CONTROL, deliberately wrong: 1.0e-37 again
constexpr scalar timeControlGreat = 1.0e+15;
inline scalar timeControlSmall()
{
    static const bool floatBuilds = std::getenv("BRAE_CONTROL_DELTAT_SMALL_FLOAT") != nullptr;
    static bool said = false;
    if (floatBuilds && !said)
    {
        said = true;
        std::printf("  *** CONTROL MODE: the time-step control adds 1e-37 to the Courant number where OpenFOAM "
                    "adds SMALL = 1e-15. This run is deliberately wrong. ***\n");
    }
    return floatBuilds ? scalar(1.0e-37) : scalar(1.0e-15);
}

// setDeltaT.H's cap on a step's growth: deltaTFact = min(min(maxDeltaTFact, 1 + 0.1*maxDeltaTFact), 1.2).
//   BRAE_CONTROL_DELTAT_GROWTH_UNCAPPED=1   a gate's CONTROL, deliberately wrong: the 1.2 is left out, so a
//                                           run that starts at rest takes maxDeltaT at its first step
inline scalar timeControlGrowthCap()
{
    static const bool uncapped = std::getenv("BRAE_CONTROL_DELTAT_GROWTH_UNCAPPED") != nullptr;
    static bool said = false;
    if (uncapped && !said)
    {
        said = true;
        std::printf("  *** CONTROL MODE: the time-step control leaves out setDeltaT.H's cap of 1.2 on a step's "
                    "growth. This run is deliberately wrong. ***\n");
    }
    return uncapped ? timeControlGreat : scalar(1.2);
}

struct TimeControls
{
    bool adjustTimeStep = false;
    scalar maxCo = 1.0;
    scalar maxDeltaT = timeControlGreat;

    static TimeControls read(const FoamDict& controlDict)
    {
        TimeControls tc;
        const std::string a = controlDict.wordOr("adjustTimeStep", "no");
        tc.adjustTimeStep = (a == "yes" || a == "true" || a == "on" || a == "1");
        tc.maxCo     = controlDict.scalarOr("maxCo", 1.0);
        tc.maxDeltaT = controlDict.scalarOr("maxDeltaT", timeControlGreat);
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
    const scalar kSmall = timeControlSmall();
    if (CoNum <= kSmall) return deltaT;
    return std::min(tc.maxCo*deltaT/CoNum, std::min(deltaT, tc.maxDeltaT));
}

// OF setDeltaT.H -- applied every step. Reduction is immediate; growth is damped and capped at 1.2x.
inline scalar setDeltaT(scalar deltaT, scalar CoNum, const TimeControls& tc)
{
    if (!tc.adjustTimeStep) return deltaT;
    const scalar kSmall = timeControlSmall();
    const scalar maxDeltaTFact = tc.maxCo/(CoNum + kSmall);
    const scalar damped = std::fma(scalar(0.1), maxDeltaTFact, scalar(1));
    const scalar deltaTFact = std::min(std::min(maxDeltaTFact, damped), timeControlGrowthCap());
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
// 1. maxAlphaCo IS MANDATORY, AT A FIXED STEP TOO. alphaCourantNo.H reads it with get<scalar> -- no default
//    -- where readTimeControls.H gives maxCo a default of 1, and interFoam.C:100-102 includes it at EVERY
//    step of a run that is not local-time-stepped, whatever adjustTimeStep says: the interface Courant number
//    is printed either way. So a case that omits it is a FatalIOError at its first step. This reader used to
//    refuse only under `adjustTimeStep yes` and its test said "a fixed-step case never reads it"; real
//    interFoam on laminar/damBreak with the entry removed and `adjustTimeStep no` stops with "Entry
//    'maxAlphaCo' not found in dictionary" (tests/interfoam_write/refusal/max_alpha_co.sh runs it).
//    NOT under localEuler: interFoam.C:94-97 runs setRDeltaT.H there instead, which reads its own
//    maxAlphaCo from the PIMPLE dictionary with a default of 0.2 (setRDeltaT.H:11-14).
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
//    value. That is OpenFOAM's order and it is kept as transcribed -- though damping each and then taking
//    the min is the same number (the damping is non-decreasing in its argument, in floating point too), so
//    no test can tell the two apart and none claims to.

struct VoFTimeControls
{
    TimeControls base;
    scalar       maxAlphaCo = 0;     // MANDATORY -- see note 1

    // `localTimeStep`: ddtSchemes `default` is localEuler, where alphaCourantNo.H does not run (note 1)
    static VoFTimeControls read(
        const FoamDict& controlDict,
        bool localTimeStep = false)
    {
        VoFTimeControls tc;
        tc.base = TimeControls::read(controlDict);
        // get<scalar>, not getOrDefault: alphaCourantNo.H:34-37.
        tc.maxAlphaCo = controlDict.scalarOr("maxAlphaCo", scalar(-1));
        if (!localTimeStep && tc.maxAlphaCo < 0)
        {
            throw std::runtime_error(
                "brae interFoam: controlDict has no `maxAlphaCo`. OpenFOAM reads it with get<scalar> and has "
                "NO default (alphaCourantNo.H:34-37), unlike maxCo which defaults to 1, and it reads it at "
                "every step of a run that is not local-time-stepped, at a fixed step too (interFoam.C:100-102): "
                "Entry 'maxAlphaCo' not found in dictionary. Defaulting it here would run a case OpenFOAM "
                "stops on, and under `adjustTimeStep` would advance the interface with no limit of its own.");
        }
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
// MEASURED ON damBreak, whose controlDict says `writeControl adjustable; writeInterval 0.05` and
// `deltaT 0.001`: OpenFOAM's first step is 0.00119047619 -- that is 0.05/42 -- where brae's unclamped
// setDeltaTVoF gave the raw 1.2 x 0.001 = 0.0012. The gap is in the third digit and it compounds. (This note
// carried both numbers a factor of ten too small until 2026-10-05, as the clock gate's control did.)
struct WriteCadence
{
    // writeControl adjustableRunTime -- the only mode that adjusts
    bool adjustable = false;
    // writeControl runTime or adjustableRunTime: the modes whose write time is Time::writeTimeIndex_
    // moving (Time.C:1115-1130). runTime moves it exactly as the adjustable mode does and only never
    // trims deltaT, so advance() tracks it for a writer to read while adjustDeltaT keeps keying on
    // `adjustable` alone.
    bool runTimeIndexed = false;
    scalar writeInterval = 0;
    // Time::writeTimeIndex_, which advance() below moves
    label writeTimeIndex = 0;

    static WriteCadence read(const FoamDict& controlDict)
    {
        WriteCadence w;
        // BOTH spellings, from Time::writeControlNames (Time.C:53-62), which tabulates
        // wcAdjustableRunTime twice -- as "adjustable" and as "adjustableRunTime". damBreak's own
        // controlDict uses the short one, and matching only the long one made this whole clamp a
        // no-op on the very case it was measured against. Any other value, tabulated or not, leaves
        // adjustDeltaT a no-op exactly as it is in OpenFOAM.
        const std::string wc = controlDict.wordOr("writeControl", "timeStep");
        w.adjustable = (wc == "adjustable" || wc == "adjustableRunTime");
        w.runTimeIndexed = w.adjustable || wc == "runTime";
        // `writeInterval`, else its older name `writeFrequency` (TimeIO.C:286-297), as the writer reads it
        // (inter_writer_cpp.cu). This read the first alone until 2026-10-06: a case spelt `writeFrequency` wrote
        // nothing under `runTime`, with no word, and was refused under `adjustable` for an entry it had.
        w.writeInterval = controlDict.scalarOr("writeInterval", controlDict.scalarOr("writeFrequency", scalar(0)));
        if (w.runTimeIndexed && !(w.writeInterval > scalar(0)))
        {
            // runTime's index divides by it as the adjustable mode's does, below
            w.runTimeIndexed = w.adjustable;
        }
        if (w.adjustable && !(w.writeInterval > scalar(0)))
            throw std::runtime_error(
                "brae: controlDict says `writeControl adjustableRunTime` but gives no positive "
                "`writeInterval`. Time::adjustDeltaT divides by it (Time.C:1150).");
        return w;
    }

    // Time::operator++ (Time.C:1046-1074), and the two details there both matter: the time is the one
    // AFTER the step, the deltaT is the one that took it, and the index only ever moves FORWARD. Returns
    // whether it moved, which under runTime and adjustableRunTime IS Time::writeTime_ (Time.C:1115-1130).
    bool advance(
        scalar tSinceStart,
        scalar deltaT)
    {
        if (!runTimeIndexed) return false;
        const label wi =
            static_cast<label>((tSinceStart + scalar(0.5)*deltaT)/writeInterval);
        if (wi > writeTimeIndex)
        {
            writeTimeIndex = wi;
            return true;
        }
        return false;
    }
};

// Time::adjustDeltaT() (Time.C:1136-1170). `tSinceStart` is value() - startTime_, which is what the
// drivers' own clocks already measure.
inline scalar adjustDeltaT(
    scalar deltaT,
    scalar tSinceStart,
    const WriteCadence& w)
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
    if (newDeltaT >= deltaT)
    {
        return std::min(newDeltaT, scalar(2)*deltaT);
    }
    return std::max(newDeltaT, scalar(0.2)*deltaT);
}

// setInitialDeltaT.H AS A SOLVER'S START RUNS IT (interFoam.C:81-85: CourantNo.H then this, before the time
// loop). The statement is reached only at time index 0 with a Courant number above SMALL, and where it is
// reached Time::setDeltaT lands the step on the write cadence as well -- `adjust` defaults to true
// (Time.C:981-990) -- EVEN WHEN THE VALUE IT IS HANDED IS THE deltaT IT ALREADY HOLDS. So a start that has a
// flux enters the loop with deltaT already trimmed to the write interval, and the first step's setDeltaT.H
// grows THAT; a start from rest (Courant number 0) enters with the controlDict's own. The two differ wherever
// deltaT does not divide the write interval: with writeInterval 0.011 and deltaT 0.0025 OpenFOAM's first step
// is 0.011/3 from a start with a flux and 0.011/4 from rest
// (tests/interfoam_write/clock/initial_deltat.sh). `timeIndex` is the start's own (uniform/time on a
// restart, 0 otherwise).
inline scalar setInitialDeltaT(
    scalar deltaT,
    scalar CoNum,
    const TimeControls& tc,
    label timeIndex,
    const WriteCadence& w)
{
    if (!tc.adjustTimeStep || timeIndex != 0 || !(CoNum > timeControlSmall())) return deltaT;
    return adjustDeltaT(setInitialDeltaT(deltaT, CoNum, tc), scalar(0), w);
}

// setDeltaT.H, and the adjustDeltaT that Time::setDeltaT performs on the way in. `w` null is a case
// with no adjustable write cadence -- every gate fixture, and any case on `writeControl timeStep` or
// `runTime`, for which OpenFOAM's adjustDeltaT is a no-op anyway.
inline scalar setDeltaTVoF(
    scalar deltaT,
    scalar CoNum,
    scalar alphaCoNum,
    const VoFTimeControls& tc,
    scalar tSinceStart = 0,
    const WriteCadence* w = nullptr)
{
    if (!tc.base.adjustTimeStep) return deltaT;
    const scalar kSmall = timeControlSmall();
    const scalar maxDeltaTFact = std::min(tc.base.maxCo/(CoNum + kSmall),
                                          tc.maxAlphaCo/(alphaCoNum + kSmall));
    const scalar damped = std::fma(scalar(0.1), maxDeltaTFact, scalar(1));
    const scalar deltaTFact = std::min(std::min(maxDeltaTFact, damped), timeControlGrowthCap());
    const scalar dt = std::min(deltaTFact*deltaT, tc.base.maxDeltaT);
    return w ? adjustDeltaT(dt, tSinceStart, *w) : dt;
}

}   // namespace brae
