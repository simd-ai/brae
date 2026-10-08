#pragma once
// Foam::timeControl -- WHEN a function object's execute() or write() fires
// (src/OpenFOAM/db/functionObjects/timeControl/timeControl.C).
//
// A function object's dictionary names two of these, `executeControl` and `writeControl`, each with its
// interval (timeControl.C:109-167); functionObjects::timeControl asks them at every step
// (timeControlFunctionObject.C, execute() and write()). The default of both is `timeStep` with no interval:
// every step.
//
// WHAT IT IS ASKED WITH. Time::run() calls the function objects at the TOP of the step after the one that
// produced the fields (Time.C:781-860), before setDeltaT.H: value() is the end of the step just taken,
// deltaTValue() the step that took it, timeIndex() its index and writeTime() its write flag. FunctionObjectTime
// is that instant, filled by the loop that owns the clock.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brae {

struct FunctionObjectTime
{
    // Time::value(), ::startTime() and ::endTime()
    scalar value = 0;
    scalar startTime = 0;
    scalar endTime = 0;
    // Time::deltaTValue(): the step just taken
    scalar deltaT = 0;
    // Time::timeIndex()
    label timeIndex = 0;
    // Time::writeTime() of the step just taken
    bool writeTime = false;
};

class TimeControl
{
public:
    // timeControl::timeControls, by the names timeControl.C:39-51 tabulates
    enum class Mode
    {
        none,
        always,
        timeStep,
        writeTime,
        runTime,
        adjustableRunTime,
        onEnd
    };

    Mode mode = Mode::timeStep;
    label intInterval = 0;
    scalar interval = 0;
    // timeControl::executionIndex_: 0 at construction, whatever time the run starts from (timeControl.C:56-68)
    label executionIndex = 0;

    // timeControl::read (timeControl.C:109-167). `prefix` is "execute" or "write"; `who` names the object
    // in a refusal.
    static TimeControl read(
        const FoamDict& dict,
        const std::string& prefix,
        const std::string& who)
    {
        TimeControl c;
        std::string controlName = prefix + "Control";
        std::string intervalName = prefix + "Interval";
        // "Accept deprecated 'outputControl' instead of 'writeControl'", and then the interval's older
        // name with it (:119-136)
        if (prefix == "write" && dict.found("outputControl"))
        {
            controlName = "outputControl";
            intervalName = "outputInterval";
        }
        const std::string word = dict.wordOr(controlName, "timeStep");
        if (word == "none")
        {
            c.mode = Mode::none;
        }
        else if (word == "always")
        {
            c.mode = Mode::always;
        }
        else if (word == "timeStep")
        {
            c.mode = Mode::timeStep;
        }
        else if (word == "writeTime" || word == "outputTime")
        {
            c.mode = Mode::writeTime;
        }
        else if (word == "runTime")
        {
            c.mode = Mode::runTime;
        }
        else if (word == "adjustable" || word == "adjustableRunTime")
        {
            c.mode = Mode::adjustableRunTime;
        }
        else if (word == "onEnd")
        {
            c.mode = Mode::onEnd;
        }
        else if (word == "clockTime" || word == "cpuTime")
        {
            // timeControl.C:230-258: the index is the machine's elapsed clock or CPU time over the interval,
            // so WHICH steps fire is another answer on every run and every machine
            throw std::runtime_error(
                "brae: function object `" + who + "` has `" + controlName + " " + word + "`. That control "
                "fires by the machine's elapsed time (timeControl.C:230-258), not by the simulation's, and "
                "is not ported.");
        }
        else
        {
            // Enum::getOrDefault on a word the table does not hold: OpenFOAM stops
            throw std::runtime_error(
                "brae: function object `" + who + "` has `" + controlName + " " + word + "`, which is not "
                "one of timeControl.C:39-51's: none, always, timeStep, writeTime, outputTime, runTime, "
                "adjustable, adjustableRunTime, clockTime, cpuTime, onEnd.");
        }
        if (c.mode == Mode::timeStep || c.mode == Mode::writeTime)
        {
            // getOrDefault<label>(intervalName, 0) (:147)
            c.intInterval = static_cast<label>(dict.intOr(intervalName, 0));
            c.interval = scalar(c.intInterval);
        }
        else if (c.mode == Mode::runTime || c.mode == Mode::adjustableRunTime)
        {
            // dict.get<scalar>(intervalName): mandatory (:158)
            if (!dict.found(intervalName))
            {
                throw std::runtime_error(
                    "brae: function object `" + who + "` has `" + controlName + " " + word + "` and no `"
                    + intervalName + "`. OpenFOAM stops on that too: the entry is mandatory under this "
                    "control (timeControl.C:157).");
            }
            c.interval = dict.scalarOr(intervalName, scalar(0));
            if (!(c.interval > scalar(0)))
            {
                throw std::runtime_error(
                    "brae: function object `" + who + "`: `" + intervalName + "` is not positive. "
                    "timeControl::execute divides by it (timeControl.C:213-221).");
            }
        }
        return c;
    }

    // timeControl::execute (timeControl.C:170-278). Not const: writeTime, runTime and adjustableRunTime
    // move the index they answer from.
    bool execute(const FunctionObjectTime& t)
    {
        switch (mode)
        {
            case Mode::none:
            {
                return false;
            }
            case Mode::always:
            {
                return true;
            }
            case Mode::timeStep:
            {
                return intInterval <= 1 || (t.timeIndex % intInterval) == 0;
            }
            case Mode::writeTime:
            {
                if (t.writeTime)
                {
                    ++executionIndex;
                    return intInterval <= 1 || (executionIndex % intInterval) == 0;
                }
                return false;
            }
            case Mode::runTime:
            case Mode::adjustableRunTime:
            {
                const label index = runTimeIndex(t.value - t.startTime, t.deltaT, interval);
                if (index > executionIndex)
                {
                    executionIndex = index;
                    return true;
                }
                return false;
            }
            case Mode::onEnd:
            {
                return t.value > t.endTime - scalar(0.5)*t.deltaT;
            }
        }
        return false;
    }

    // label(((value - startTime) + 0.5*deltaT)/interval) (timeControl.C:213-221): ONE expression for this
    // class and for a check: the clock keeps its own copy of an adjustableRunTime object's index, which the
    // time step is trimmed from (FunctionObjectCadence::advance, time_controls.cuh), and the two must agree
    static label runTimeIndex(
        scalar tSinceStart,
        scalar deltaT,
        scalar interval)
    {
        if (control("noHalfStep"))
        {
            return static_cast<label>(tSinceStart/interval);
        }
        return static_cast<label>((tSinceStart + scalar(0.5)*deltaT)/interval);
    }

    // BRAE_CONTROL_FUNCTION_OBJECT_TIMING names ONE rule of the timing to break -- the gates' controls, not
    // user switches:
    //   noHalfStep  a runTime index without the half step: label((value - startTime)/interval)
    //   window      an object is active whatever its timeStart and timeEnd say
    static bool control(const char* part)
    {
        static const char* const set = std::getenv("BRAE_CONTROL_FUNCTION_OBJECT_TIMING");
        return set && std::string(set) == part;
    }
};

}   // namespace brae
