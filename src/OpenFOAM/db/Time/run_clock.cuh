#pragma once
// run_clock.cuh -- OpenFOAM's Foam::Time clock, for a solver that owns its time loop and lets deltaT vary.
//
// brae::Time and WriteControl (brae_time.cuh, write_control.cuh) count steps: the time is start + n*deltaT and the
// number of steps is known before the loop. OpenFOAM's is neither. Its time ACCUMULATES (Time::operator++,
// Time.C:1067: setTime(value() + deltaT_)), the loop stops on that accumulated value (Time::run), and under
// adjustTimeStep the step changes every iteration. This is that clock and nothing else:
//
//   run()          Time::run: value() < endTime - 0.5*deltaT, on the deltaT the previous step ended with
//   setDeltaT(dt)  Time::setDeltaT(dt, adjust = true) (Time.C:981-990): Time::adjustDeltaT, i.e. the trimming to
//                  the write times under writeControl adjustable, then each function object's (Time.C:102-143)
//   operator++     Time::operator++ (Time.C:1057-1266): deltaT0/deltaTSave, the accumulated value, the snap to 0,
//                  writeTime_ for timeStep / runTime / adjustable, and the time-name precision raise on write steps
//
// The policy that picks a step stays in time_controls.cuh, shared with interFoam: setDeltaT.H (the caller's, with its
// own Courant number), startDeltaT (a restart's uniform/time deltaT), WriteCadence and adjustDeltaT (Time's write
// cadence and the function objects' trimming). This class only keeps the clock they act on.
//
// Not carried, and refused by name at construction: writeControl clockTime and cpuTime (a write would depend on how
// fast the run went), and an unknown writeControl word (OpenFOAM stops on it, Enum::get).
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "start_time.cuh"      // openFoamTimeName, timeFormatting
#include "time_controls.cuh"   // startDeltaT, WriteCadence, adjustDeltaT

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace brae {

class RunClock
{
public:
    // `startDir` is the resolved start directory (its uniform/time is read as Time::setControls reads it), and
    // `startTime` that directory's time value.
    RunClock(const FoamDict& controlDict, const std::string& startDir, scalar startTime)
      : startTime_(startTime), origin_(startTime), value_(startTime)
    {
        control_ = controlDict.wordOr("writeControl", "timeStep");
        if (control_ == "clockTime" || control_ == "cpuTime")
            throw std::runtime_error("controlDict writeControl `" + control_ + "` is not ported: a write would depend "
                                     "on how fast the run went. timeStep, runTime, adjustable (adjustableRunTime) "
                                     "and none are.");
        if (control_ != "timeStep" && control_ != "runTime" && control_ != "adjustable"
            && control_ != "adjustableRunTime" && control_ != "none")
            throw std::runtime_error("controlDict writeControl `" + control_ + "` is not one of OpenFOAM's: none, "
                                     "timeStep, runTime, adjustable, adjustableRunTime, clockTime, cpuTime.");
        cadence_ = WriteCadence::read(controlDict);
        cadence_.startTime = startTime;
        if (control_ == "timeStep")
        {
            // TimeIO.C:286-297: writeInterval, else writeFrequency; label(writeInterval_) at Time.C:1112
            const scalar w = controlDict.scalarOr("writeInterval", controlDict.scalarOr("writeFrequency", scalar(0)));
            if (!(w >= scalar(1)))
                throw std::runtime_error("controlDict writeControl timeStep needs a writeInterval of 1 or more "
                                         "(TimeIO.C:288-293).");
            stepInterval_ = static_cast<long>(w);
        }
        if (!controlDict.found("endTime"))
            throw std::runtime_error("controlDict has no endTime; OpenFOAM reads it with readEntry.");
        endTime_ = controlDict.scalarOr("endTime", scalar(0));

        // Time::setControls (Time.C:193-195, 279-307): controlDict's deltaT, then, under adjustTimeStep alone, the
        // start's uniform/time deltaT; the time index from uniform/time's `index` whatever adjustTimeStep says
        deltaT_ = startDeltaT(controlDict, startDir, controlDict.scalarOr("deltaT", scalar(0)));
        deltaTSave_ = deltaT0_ = deltaT_;
        const std::string timePath = startDir + "/uniform/time";
        if (std::filesystem::exists(timePath))
        {
            const FoamDict td = readDict(timePath);
            if (td.found("index")) timeIndex_ = static_cast<long>(td.scalarOr("index", scalar(0)));
            // Time.C:301: deltaT0 read whatever adjustTimeStep says; the first operator++ overwrites it before any use
            if (td.found("deltaT0")) deltaT0_ = td.scalarOr("deltaT0", deltaT0_);
        }
        startTimeIndex_ = timeIndex_;
        name_ = openFoamTimeName(static_cast<double>(value_));
    }

    // Time::run(): on the deltaT the previous iteration ended with, before this one's setDeltaT
    bool run() const { return value_ < endTime_ - scalar(0.5)*deltaT_; }

    // Time::setDeltaT(deltaT) with its default adjust = true
    void setDeltaT(scalar dt) { deltaT_ = adjustDeltaT(dt, value_ - origin_, cadence_); }

    // GATE CONTROLS ONLY, the restart's two time-state mistakes: the step count from uniform/time's `index` dropped
    // (timeStep writes then fall off OpenFOAM's global grid), and the runTime/adjustable write grid measured from 0
    // instead of from the start time (Time.C operator++ and adjustDeltaT use value() - startTime_)
    void forgetStartIndex() { timeIndex_ = startTimeIndex_ = 0; }
    void measureWritesFromZero()
    {
        origin_ = 0;
        cadence_.startTime = 0;
    }

    // Time::setDeltaT(deltaT, false): a gate's control only (the write-time trimming left out)
    void setDeltaTNoAdjust(scalar dt) { deltaT_ = dt; }

    // Time::operator++ (Time.C:1057-1266)
    RunClock& operator++()
    {
        deltaT0_ = deltaTSave_;
        deltaTSave_ = deltaT_;
        const scalar oldTimeValue = value_;
        value_ = value_ + deltaT_;
        ++timeIndex_;
        // "If the time is very close to zero reset to zero" (SMALL = 1e-15, doubleScalar.H)
        if (std::fabs(value_) < scalar(10)*scalar(1.0e-15)*deltaT_) value_ = 0;
        name_ = openFoamTimeName(static_cast<double>(value_));

        writeTime_ = false;
        if (control_ == "timeStep")
        {
            writeTime_ = !(timeIndex_ % stepInterval_);
            cadence_.advance(value_ - origin_, deltaT_);   // the function objects' indices
        }
        else
        {
            // runTime and adjustable: writeIndex = ((value - startTime) + 0.5*deltaT)/writeInterval moving forward
            const bool moved = cadence_.advance(value_ - origin_, deltaT_);
            writeTime_ = (control_ != "none") && moved;
        }
        if (writeTime_) raisePrecision(oldTimeValue);
        return *this;
    }

    scalar value() const { return value_; }
    scalar deltaT() const { return deltaT_; }
    scalar deltaT0() const { return deltaT0_; }
    scalar startTime() const { return startTime_; }
    scalar endTime() const { return endTime_; }
    long timeIndex() const { return timeIndex_; }
    long startTimeIndex() const { return startTimeIndex_; }
    bool writeTime() const { return writeTime_; }
    const std::string& timeName() const { return name_; }

private:
    // Time.C:1189-1260: on a write step, the name must read back as the previous time plus the step, to within
    // 10^-precision (or a tenth of the step, or SMALL); otherwise the static precision rises until it does, to
    // maxPrecision_ = 3 - log10(SMALL) = 18
    void raisePrecision(scalar oldTimeValue)
    {
        TimeFormatting& f = timeFormatting();
        const scalar userDeltaT = value_ - (value_ - deltaT_);
        const scalar timeTol =
            std::max(std::min(std::pow(scalar(10), -scalar(f.precision)), scalar(0.1)*userDeltaT), scalar(1.0e-15));
        auto off = [&](const std::string& n)
        {
            const scalar v = static_cast<scalar>(std::strtod(n.c_str(), nullptr));
            return std::fabs(v - oldTimeValue - userDeltaT) > timeTol;
        };
        if (!off(name_)) return;
        const int oldPrecision = f.precision;
        while (f.precision < 18 && off(name_))
        {
            ++f.precision;
            name_ = openFoamTimeName(static_cast<double>(value_));
        }
        if (f.precision != oldPrecision)
        {
            f.raisedPrecision = f.precision;
            std::fprintf(stderr, "brae: increased the timePrecision from %d to %d to distinguish between timeNames at "
                                 "time %s\n", oldPrecision, f.precision, name_.c_str());
        }
    }

    std::string control_;
    long stepInterval_ = 1;
    WriteCadence cadence_;
    scalar startTime_ = 0;
    scalar origin_ = 0;   // what the runTime/adjustable write grid is measured from: the start time (a control: 0)
    scalar endTime_ = 0;
    scalar value_ = 0;
    scalar deltaT_ = 0;
    scalar deltaT0_ = 0;
    scalar deltaTSave_ = 0;
    long timeIndex_ = 0;
    long startTimeIndex_ = 0;
    bool writeTime_ = false;
    std::string name_;
};

}   // namespace brae
