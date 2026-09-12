#pragma once
// brae::Time -- the OpenFOAM Time/functionObjectList lifecycle, owned centrally instead of per solver.
//
// WHY THIS EXISTS. In OpenFOAM, functionObjects are a member of Time (Time.C:469), and Time drives the
// whole lifecycle itself:
//
//     read()             TimeIO.C:443,484   on construction, and on every controlDict re-read
//     start()            Time.C:820         first time step
//     execute()          Time.C:797,825     every time step
//     end()              Time.C:801         final time step
//     adjustTimeStep()   Time.C:142         transient only
//
// rhoSimpleFoam.C contains NO functionObject code -- only `#include "postProcess.H"` and
// `runTime.write()`. A solver gets functionObjects by having a Time; it cannot forget them and it
// cannot implement them differently from another solver.
//
// brae had no such class. Each solver parsed `functions{}` itself, and in practice only ever looked for
// one hardcoded type (gpuSimpleFoam.cu:744 searches for `forceCoeffs` and nothing else). rhoSimpleFoam
// never parsed it at all, so on a case like gasMixing/injectorPipe -- which ships scalarTransport,
// sampling and abort -- every functionObject was silently skipped. Not refused, not warned: skipped.
// That is the bug class this file removes.
//
// WHAT THIS DELIBERATELY DOES NOT OWN. brae captures CUDA graphs INSIDE the linear solver
// (device_amg_vcycle.cu:315,332 and device_amg_pcg.cu:182), several layers below a time step, so the
// outer loop is free to be centralised without disturbing capture. The line drawn here is OF's:
//
//     PCG / AMG V-cycle          solver internals, graph-captured   -- NOT here
//     SIMPLE / PIMPLE iterations each solver, different shapes       -- NOT here
//     time advance, functionObjects, write cadence                   -- here
//
// Everything in this file is host code executing ONCE PER TIME STEP, against device work of
// milliseconds to seconds per iteration. It must never touch per-iteration device state: the moment it
// does, it becomes a synchronisation point in a loop that is deliberately asynchronous.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "brae_notice.cuh"   // noticeIgnored: the established "never drop an input silently" channel
#include "object_registry.cuh"   // OF: class Time : public objectRegistry (Time.H:74-80)
#include "write_control.cuh"
#include "start_time.cuh"   // OF Time::setControls() startFrom (Time.C:149)   // OF writeControl/writeInterval cadence, which Time owns   // noticeIgnored: the established "never drop an input silently" channel
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace brae {

// One functionObject. Mirrors OF's functionObject virtual interface: a solver never calls these, Time
// does. Returning false from execute()/write() reports failure without aborting the run, as in OF.
class FunctionObject
{
public:
    virtual ~FunctionObject() = default;

    // OF functionObject::name() -- the sub-dictionary key, used in logs and error messages.
    virtual const std::string& name() const = 0;

    // OF functionObject::execute(): called every time step, after the solver has advanced the fields.
    virtual bool execute() = 0;

    // OF functionObject::write(): called on write times only. Split from execute() because OF splits
    // them -- scalarTransport solves every step but writes on writeControl.
    virtual bool write() { return true; }

    // OF functionObject::end(): called once at the final time step.
    virtual bool end() { return true; }
};

// OF functionObjectList. Constructed from controlDict's `functions` entry, owned by Time.
//
// Unknown types are REPORTED, never skipped in silence -- that silence is exactly what let gasMixing's
// scalarTransport disappear. A type brae cannot build is named on stderr so the run states plainly
// which part of the case it is not honouring.
class FunctionObjectList
{
public:
    // OF functionObjectList::read(). Walks controlDict.functions and builds what it recognises.
    //
    // Anything it cannot build goes through noticeIgnored -- the SAME convention the rest of brae uses
    // for "never drop an input silently". Reporting here rather than in each solver is the entire point
    // of centralising: a solver that adopts Time cannot forget to report, in the same way an OF solver
    // cannot forget to run functionObjects. Returns the number not built.
    // `handledOutside` names types the CALLING SOLVER computes on its own, outside this lifecycle.
    // They are reported as APPROXIMATED rather than ignored, because that is what they are: brae's
    // forceCoeffs, for example, is a single post-run print, whereas OF's forceCoeffs is a per-time-step
    // functionObject (execute()/write(), forceCoeffs.H:547,550) that writes a coefficient HISTORY under
    // postProcessing/. Reporting those as "ignored" would be false; reporting nothing would hide a real
    // difference from OF. Neither is acceptable, so they get their own category.
    // The two "outside" lists exist because the SAME type can be faithful in one solver and an
    // approximation in another, and saying so accurately matters more than saying it uniformly:
    //
    //   pimpleFoam  forceCoeffs -> samples on the write cadence and writes
    //                              postProcessing/forceCoeffs/<time>/coefficient.dat, i.e. what OF does
    //   simpleFoam  forceCoeffs -> a single post-run print; OF's runs every step and writes a history
    //
    // Calling both "approximated" would understate pimpleFoam; calling both "applied" would overstate
    // simpleFoam. Each solver declares which it is.
    int read(const FoamDict* functions,
             const std::vector<std::string>& approximatedOutside = {},
             const std::vector<std::string>& appliedOutside = {})
    {
        objects_.clear();
        if (!functions) return 0;
        int missed = 0;
        auto listed = [](const std::vector<std::string>& v, const std::string& t)
        {
            for (const std::string& h : v) if (h == t) return true;
            return false;
        };
        for (const auto& s : functions->subs)
        {
            const std::string type = s.second.wordOr("type", "");
            const bool haveFactory = factories_.count(type) != 0;
            std::unique_ptr<FunctionObject> fo = create(s.first, type, s.second);
            if (fo)
            {
                objects_.push_back(std::move(fo));
                continue;
            }
            // A REGISTERED factory that returns null has declined for a reason of its own and has
            // already said so (e.g. an unsupported diffusivity form, or a missing field). Falling
            // through to the generic "type is not implemented" here would contradict it -- the type IS
            // implemented; this instance of it was declined.
            if (haveFactory) { ++missed; continue; }
            if (listed(appliedOutside, type))
            {
                noticeApplied(
                    "functions/" + s.first,
                    "type '" + type + "' is implemented by the solver outside the functionObject "
                    "lifecycle, on OpenFOAM's own write cadence.");
                continue;
            }
            if (listed(approximatedOutside, type))
            {
                noticeApproximated(
                    "functions/" + s.first,
                    "type '" + type + "' is computed by the solver outside the functionObject "
                    "lifecycle, so it is reported once rather than per time step as OpenFOAM does.");
                continue;
            }
            ++missed;
            noticeIgnored(
                "functions/" + s.first,
                "type '" + type + "' is not implemented, so this functionObject is NOT run. "
                "OpenFOAM would execute it every time step.");
        }
        return missed;
    }

    bool start()   { return all(&FunctionObject::execute); }   // OF calls execute() on the first step
    bool execute() { return all(&FunctionObject::execute); }
    bool write()   { return all(&FunctionObject::write); }
    bool end()     { return all(&FunctionObject::end); }

    std::size_t size() const { return objects_.size(); }

    // OF selects functionObjects through a runtime selection table (addToRunTimeSelectionTable), which
    // lets a type live next to its dependencies rather than inside Time. Same idea here: a concrete
    // functionObject that needs the DEVICE solver cannot be constructed in this file without dragging
    // DeviceSimpleSolver into OpenFOAM/db, so the owning solver registers a factory before read().
    //
    // Register BEFORE read() -- read() is what consults the table.
    using Factory = std::function<std::unique_ptr<FunctionObject>(const std::string&, const FoamDict&)>;

    void registerType(const std::string& type, Factory f) { factories_[type] = std::move(f); }

private:
    std::unique_ptr<FunctionObject> create(
        const std::string& name, const std::string& type, const FoamDict& dict)
    {
        auto it = factories_.find(type);
        if (it == factories_.end()) return nullptr;
        return it->second(name, dict);
    }

    std::map<std::string, Factory> factories_;

    bool all(bool (FunctionObject::*fn)())
    {
        bool ok = true;
        for (auto& o : objects_) ok = ((*o).*fn)() && ok;   // run every one, do not short-circuit
        return ok;
    }

    std::vector<std::unique_ptr<FunctionObject>> objects_;
};

// brae::Time -- OF's Foam::Time, reduced to what brae's solvers actually need.
//
// THE POINT. In OF a solver writes `while (simple.loop())` and `runTime.write()` and gets
// functionObjects for free, because the chain is
//
//     simple.loop()  ->  runTime.loop()             simpleControl.C:160
//     Time::loop()   ->  run() then operator++()    Time.C:863
//     Time::run()    ->  functionObjects_.execute() Time.C:797
//
// Not one of simpleFoam.C / pimpleFoam.C / rhoSimpleFoam.C contains a single functionObject call. That
// is the property being reproduced: a solver cannot forget to run them, because running them is not
// its job. Adding execute()/write()/end() to each brae solver by hand would have reproduced the
// duplication this class exists to remove.
//
// Time owns the run bookkeeping -- iteration index, time value, write cadence, functionObjects -- and
// NOTHING about how an equation is assembled or solved. Each solver keeps its own loop BODY (SIMPLE and
// PIMPLE have genuinely different shapes) and its own convergence test; it just stops owning the
// scaffolding around it. Device work never enters here: this is host code running once per time step
// against device work of milliseconds per iteration, and the moment it touches per-iteration device
// state it becomes a synchronisation point in a deliberately asynchronous loop.
class Time
{
public:
    // WriteControl is OPTIONAL. gpuSimpleFoam carries its own isWriteTime lambda rather than
    // WriteControl -- a separate duplication, and not one to fix by quietly changing a validated
    // solver's write cadence. A solver without WriteControl drives the write side itself by calling
    // write() at its own write point; the LIFECYCLE (start/execute/end) is Time's either way, which is
    // the part that must not be per-solver.
    // OWNING constructor: Time builds its own WriteControl and resolves startFrom, exactly as OF's
    // Time::setControls() does (Time.C:149-188 for startFrom, TimeIO.C:277 for writeControl). Neither is
    // a solver's business -- OF's simpleFoam.C/pimpleFoam.C/rhoSimpleFoam.C mention neither.
    //
    // Before this, all three brae drivers carried their own copies: two inline `startFrom` blocks and
    // three separate `isWriteTime` lambdas. Measured identical before consolidating (startFrom on an
    // adversarial directory; writeControl over 8 cases x 40 steps including runTime float accumulation
    // and a non-zero-startTime restart), so this is pure de-duplication -- but read_surface_field was
    // the same shape and HAD silently diverged, at 10^6x the reproducibility floor.
    Time(const std::string& caseDir, const FoamDict& controlDict,
         const std::vector<std::pair<std::string, FunctionObjectList::Factory>>& types = {},
         const std::vector<std::string>& approximatedOutside = {},
         const std::vector<std::string>& appliedOutside = {})
      : owned_(controlDict), wc_(&owned_), nSteps_(0)
    {
        startName_ = resolveStartTime(caseDir,
                                      controlDict.wordOr("startFrom", "startTime"),
                                      controlDict.wordOr("startTime", "0"));
        owned_.setStartTime(static_cast<scalar>(std::strtod(startName_.c_str(), nullptr)));
        for (const auto& t : types) functionObjects_.registerType(t.first, t.second);
        functionObjects_.read(controlDict.subDict("functions"), approximatedOutside, appliedOutside);
    }

    // Register and read functionObjects AFTER construction. Needed because startFrom must resolve
    // before a functionObject that reads a field from the start directory can even be described --
    // Time knows the start time, and the factory needs it. OF has no such ordering problem: its
    // functionObjects find their fields lazily through the objectRegistry, so they can be built before
    // anything they touch exists. brae's do too, EXCEPT for the field PATH, which is still passed in.
    void readFunctionObjects(
        const FoamDict& controlDict,
        const std::vector<std::pair<std::string, FunctionObjectList::Factory>>& types = {},
        const std::vector<std::string>& approximatedOutside = {},
        const std::vector<std::string>& appliedOutside = {})
    {
        for (const auto& t : types) functionObjects_.registerType(t.first, t.second);
        functionObjects_.read(controlDict.subDict("functions"), approximatedOutside, appliedOutside);
    }

    // The resolved start directory name -- OF's Time::timeName() at construction.
    const std::string& startName() const { return startName_; }
    WriteControl&      writeControl()    { return owned_; }

    Time(const FoamDict& controlDict, WriteControl* wc, int nSteps,
         const std::vector<std::pair<std::string, FunctionObjectList::Factory>>& types = {},
         const std::vector<std::string>& approximatedOutside = {},
         const std::vector<std::string>& appliedOutside = {})
      : owned_(controlDict), wc_(wc ? wc : &owned_), nSteps_(nSteps)
    {
        for (const auto& t : types) functionObjects_.registerType(t.first, t.second);
        functionObjects_.read(controlDict.subDict("functions"), approximatedOutside, appliedOutside);
    }

    // OF Time::loop(): test run(), advance, fire functionObjects. The solver writes
    //     while (time.loop()) { ...its own iteration... }
    // and never mentions functionObjects, exactly as OF's solvers do not.
    bool loop()
    {
        if (iter_ >= nSteps_) return false;
        ++iter_;
        if (iter_ == 1) functionObjects_.start();   // OF Time.C:820
        else            functionObjects_.execute(); // OF Time.C:797,825
        return true;
    }

    int    timeIndex() const { return iter_; }
    scalar timeValue() const { return wc_ ? wc_->timeValue(iter_) : scalar(iter_); }
    // OF Time::deltaTValue(): the controlDict's deltaT, what an expression's deltaT() reads. NaN without a
    // WriteControl, so a consumer that needs it refuses rather than assuming 1.
    scalar deltaT() const { return wc_ ? wc_->deltaT() : std::numeric_limits<scalar>::quiet_NaN(); }
    std::string timeName() const { return WriteControl::timeName(timeValue()); }

    // OF runTime.write(): true when this step is a write time. functionObjects_.write() fires here and
    // not in loop(), because OF splits them -- scalarTransport solves every step, writes on cadence.
    bool writeTime()
    {
        if (iter_ >= nSteps_) return false;   // the final write is the solver's own, as in OF writeAndEnd
        if (!wc_ || !wc_->isWriteTime(iter_, timeValue())) return false;
        functionObjects_.write();
        return true;
    }

    // OF Time.C:790-802, the "Ensure functionObjects execute on last time step" block. run() fires
    // execute() BEFORE the increment, so the top-of-loop firing always sees the PREVIOUS body's
    // results and the final body would otherwise never be seen at all. OF therefore does one more
    // execute() on the way out, then end(). Omitting it silently drops the last time step from every
    // functionObject -- for scalarTransport that is one whole transport step missing from the answer.
    void end()
    {
        if (iter_ > 0) functionObjects_.execute();
        functionObjects_.end();
    }

    // Lets a solver stop early (residualControl) without Time thinking the run was truncated.
    void stop() { nSteps_ = iter_; }

    // The functionObject REPORT must happen at start-up, so a case that refuses on an unsupported model
    // still learns which of its functionObjects brae would not have honoured. But the step count is only
    // known once endTime/startTime/deltaT have been resolved, which is later. So Time is constructed
    // early (reporting fires) and told the count here, before the loop.
    void setSteps(int n) { nSteps_ = n; }

    // For a solver that owns its own write cadence: fire the write side at its write point.
    void write() { functionObjects_.write(); }

    // OF has `class Time : public objectRegistry`. Composition rather than inheritance here, but the
    // property that matters is the same: a functionObject reaches the solver and its fields BY NAME and
    // LATE, so Time can be constructed before the things its functionObjects will need.
    ObjectRegistry&       registry()       { return registry_; }
    const ObjectRegistry& registry() const { return registry_; }

    FunctionObjectList&       functionObjects()       { return functionObjects_; }
    const FunctionObjectList& functionObjects() const { return functionObjects_; }

private:
    WriteControl       owned_;      // used when Time constructs its own; wc_ points here or at the caller's
    WriteControl*      wc_;
    std::string        startName_ = "0";
    ObjectRegistry     registry_;
    FunctionObjectList functionObjects_;
    int                nSteps_;
    int                iter_ = 0;
};

}   // namespace brae
