#pragma once
// functionObjects::timeControl -- the wrapper that decides WHEN a function object is run
// (src/OpenFOAM/db/functionObjects/timeControl/timeControlFunctionObject.C).
//
// functionObjectList::read wraps an object in one of these when its dictionary has any timing entry
// (entriesPresent, :484-501) and hands it over bare otherwise (functionObjectList.C:1066). The list calls
// execute() and then write() on every object at every step; the wrapper passes each on only inside the
// active window and when its control fires:
//
//     execute()   active() && executeControl_.execute()            -> the object's execute()        :504-516
//     write()     active() && writeControl_.execute()              -> the object's execute() if it
//                                                                      has not run at this time index,
//                                                                      then its write()              :531-546
//     end()       active() && (executeControl_ || writeControl_)   -> the object's end()             :549-557
//
// NOT HERE: adjustTimeStep (:560-792), an adjustableRunTime write control trimming the time step. That is the
// clock's (FunctionObjectCadence, time_controls.cuh), which carries it for every object, ported or not. Its
// index and this write control's are one expression of the same numbers; the solver checks they agree.
#include "function_object_cpp.cuh"
#include <memory>
#include <stdexcept>
#include <string>

namespace brae {
namespace functionObjects {

class TimeControlFunctionObject : public FunctionObject
{
public:
    static bool entriesPresent(const FoamDict& dict)
    {
        // Foam::timeControl::entriesPresent(dict, prefix) is dict.found(prefix + "Control")
        // (timeControl.C:86-95); "output" is the older name of "write"
        return dict.found("writeControl") || dict.found("outputControl") || dict.found("executeControl")
            || dict.found("timeStart") || dict.found("timeEnd") || dict.found("triggerStart")
            || dict.found("triggerEnd");
    }

    TimeControlFunctionObject(
        const std::string& name,
        const FunctionObjectRun& run,
        const FoamDict& dict,
        std::unique_ptr<FunctionObject> object)
    :
        FunctionObject(name, run),
        executeControl_(TimeControl::read(dict, "execute", name)),
        writeControl_(TimeControl::read(dict, "write", name)),
        fo_(std::move(object))
    {
        readControls(dict);
    }

    const char* type() const override
    {
        return fo_->type();
    }

    bool read(const FoamDict& dict) override
    {
        return fo_->read(dict);
    }

    bool execute() override
    {
        if (active() && executeControl_.execute(run_.time))
        {
            executeTimeIndex_ = run_.time.timeIndex;
            fo_->execute();
        }
        return true;
    }

    bool write() override
    {
        if (active() && writeControl_.execute(run_.time))
        {
            // "Ensure written results reflect the current state"
            if (executeTimeIndex_ != run_.time.timeIndex)
            {
                executeTimeIndex_ = run_.time.timeIndex;
                fo_->execute();
            }
            fo_->write();
        }
        return true;
    }

    bool end() override
    {
        if (active() && (executeControl_.execute(run_.time) || writeControl_.execute(run_.time)))
        {
            fo_->end();
        }
        return true;
    }

    // :819-834: the mesh's changes reach the object only inside the window
    void movePoints() override
    {
        if (active())
        {
            fo_->movePoints();
        }
    }

    void updateMesh() override
    {
        if (active())
        {
            fo_->updateMesh();
        }
    }

    const TimeControl& writeControl() const
    {
        return writeControl_;
    }

private:
    // readControls (:56-86), as far as the window goes
    void readControls(const FoamDict& dict)
    {
        timeStart_ = dict.scalarOr("timeStart", timeStart_);
        timeEnd_ = dict.scalarOr("timeEnd", timeEnd_);
        const std::string mode = dict.wordOr("controlMode", "time");
        if (mode != "time")
        {
            // trigger, timeOrTrigger, timeAndTrigger: active() then reads functionObjectList::triggerIndex,
            // which other function objects set as they run (runTimeControl)
            throw std::runtime_error(
                "brae: function object `" + name_ + "` has `controlMode " + mode + "`. Its active window is "
                "then decided by the trigger index other function objects set "
                "(timeControlFunctionObject.C:89-138), which is not ported. Ported: `controlMode time`, "
                "with `timeStart` and `timeEnd`.");
        }
    }

    // active() (:89-138) under controlMode time
    bool active() const
    {
        if (TimeControl::control("window"))
        {
            return true;
        }
        return run_.time.value >= timeStart_ - scalar(0.5)*run_.time.deltaT
            && run_.time.value <= timeEnd_ + scalar(0.5)*run_.time.deltaT;
    }

    TimeControl executeControl_;
    TimeControl writeControl_;
    std::unique_ptr<FunctionObject> fo_;
    // the constructor's own (:456-479): never executed, and -VGREAT to VGREAT
    label executeTimeIndex_ = -1;
    scalar timeStart_ = scalar(-1.0e+300);
    scalar timeEnd_ = scalar(1.0e+300);
};

}   // namespace functionObjects
}   // namespace brae
