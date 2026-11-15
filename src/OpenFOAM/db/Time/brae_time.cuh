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
#include "run_timing.cuh"   // BRAE_PHASE_TIME: each functionObject's own wall time (the benchmark protocol's D)
#include "start_time.cuh"   // OF Time::setControls() startFrom (Time.C:149)   // OF writeControl/writeInterval cadence, which Time owns   // noticeIgnored: the established "never drop an input silently" channel
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace brae {

// The time step a functionObject runs for, as OpenFOAM's timeControl reads it off Time (timeControl.C execute()):
// the step just solved, at the top of the next loop (Time::run, Time.C:797-826) or at the end of the run.
struct FoStep
{
    int         timeIndex = 0;      // time_.timeIndex()
    scalar      time = 0;           // time_.value()
    scalar      startTime = 0;      // time_.startTime()
    scalar      endTime = 0;        // time_.endTime()
    scalar      deltaT = 1;         // time_.deltaTValue()
    bool        writeTime = false;  // time_.writeTime(): the solver wrote this step's fields
    std::string timeName;           // Time::timeName(time)
    std::string startName;          // the run's start directory (writeFile::newFileAtStartTime)
};

// One functionObject. Mirrors OF's functionObject virtual interface: a solver never calls these, Time
// does. Returning false from execute()/write() reports failure without aborting the run, as in OF.
class FunctionObject
{
public:
    virtual ~FunctionObject() = default;

    // OF functionObject::name() -- the sub-dictionary key, used in logs and error messages.
    virtual const std::string& name() const = 0;

    // OF functionObject::execute(): called on the steps its executeControl selects, after the solver has
    // advanced the fields.
    virtual bool execute() = 0;

    // OF functionObject::write(): called on the steps its writeControl selects (every step by default, as
    // OpenFOAM's: a functionObject's write is NOT the solver's field write). Split from execute() because OF
    // splits them -- scalarTransport solves every step but writes on writeControl.
    virtual bool write() { return true; }

    // OF functionObject::end(): called once at the final time step.
    virtual bool end() { return true; }

    // The step being run, set by FunctionObjectList before each call (OF's objects read it off time_).
    void bindStep(const FoStep* s) { step_ = s; }
    const FoStep& step() const { return *step_; }
    bool hasStep() const { return step_ != nullptr; }

private:
    const FoStep* step_ = nullptr;
};

// OF timeControl (timeControl.C): one of a functionObject's two controls, `execute` or `write`.
//   <prefix>Control   none | always | timeStep | writeTime (outputTime) | runTime | adjustable(RunTime) | onEnd
//   <prefix>Interval  timeStep and writeTime: an integer, default 0 (every one); runTime: a time, MANDATORY
// clockTime and cpuTime are REFUSED: they fire on the machine's elapsed time, so what a run outputs would depend on how
// fast it ran -- the very quantity a solver-only benchmark compares. OpenFOAM's deprecated outputControl/outputInterval
// is read for `write`, as OpenFOAM reads it.
class FoTimeControl
{
public:
    enum class Mode { None, Always, TimeStep, WriteTime, RunTime, OnEnd };

    // "" when read, else why it is refused (the object is then not run)
    std::string read(
        const FoamDict& d,
        const std::string& prefix)
    {
        std::string controlName = prefix + "Control";
        std::string intervalName = prefix + "Interval";
        if (prefix == "write" && d.found("outputControl"))
        {
            controlName = "outputControl";
            intervalName = "outputInterval";
        }
        const std::string w = d.wordOr(controlName, "timeStep");
        if (w == "none") mode_ = Mode::None;
        else if (w == "always") mode_ = Mode::Always;
        else if (w == "timeStep") mode_ = Mode::TimeStep;
        else if (w == "writeTime" || w == "outputTime") mode_ = Mode::WriteTime;
        else if (w == "runTime" || w == "adjustable" || w == "adjustableRunTime") mode_ = Mode::RunTime;
        else if (w == "onEnd") mode_ = Mode::OnEnd;
        else if (w == "clockTime" || w == "cpuTime")
            return controlName + " " + w + ": fires on the machine's elapsed time, so the output would depend on the run's speed; "
                   "not run (timeStep, writeTime, runTime and onEnd are)";
        else
            return controlName + " '" + w + "' is not one of OpenFOAM's controls (timeControl.C)";
        intInterval_ = 0;
        interval_ = 0;
        if (mode_ == Mode::TimeStep || mode_ == Mode::WriteTime)
        {
            intInterval_ = d.intOr(intervalName, 0);
            interval_ = static_cast<scalar>(intInterval_);
        }
        else if (mode_ == Mode::RunTime)
        {
            if (!d.found(intervalName))
                return controlName + " " + w + " needs " + intervalName + ", which OpenFOAM reads with no default (timeControl.C)";
            interval_ = d.scalarOr(intervalName, 0);
            if (!(interval_ > 0)) return intervalName + " must be positive for " + controlName + " " + w;
        }
        return "";
    }

    // timeControl::execute(): whether this control fires on step s. Not const: writeTime and runTime count
    // their firings, as OpenFOAM's do (executionIndex_)
    bool fires(const FoStep& s)
    {
        switch (mode_)
        {
            case Mode::None:
                return false;
            case Mode::Always:
                return true;
            case Mode::TimeStep:
                return intInterval_ <= 1 || (s.timeIndex % intInterval_) == 0;
            case Mode::WriteTime:
                if (!s.writeTime) return false;
                ++executionIndex_;
                return intInterval_ <= 1 || (executionIndex_ % intInterval_) == 0;
            case Mode::RunTime:
            {
                const long i = static_cast<long>(((s.time - s.startTime) + 0.5 * s.deltaT) / interval_);
                if (i > executionIndex_)
                {
                    executionIndex_ = i;
                    return true;
                }
                return false;
            }
            case Mode::OnEnd:
                return s.time > s.endTime - 0.5 * s.deltaT;
        }
        return false;
    }

private:
    Mode   mode_ = Mode::TimeStep;
    int    intInterval_ = 0;
    scalar interval_ = 0;
    long   executionIndex_ = 0;
};

// OF functionObjectList. Constructed from controlDict's `functions` entry, owned by Time.
//
// Unknown types are REPORTED, never skipped in silence -- that silence is exactly what let gasMixing's
// scalarTransport disappear. A type brae cannot build is named on stderr so the run states plainly
// which part of the case it is not honouring.
//
// THE LIFECYCLE IS OPENFOAM's (functionObjectList.C, timeControlFunctionObject.C), object by object:
//   start()     construction only (functionObjectList::start() is read(), :609-612). brae executed every object here,
//               BEFORE the first step: one execution more than OpenFOAM's, and for scalarTransport one tracer solve on
//               the initial fields that OpenFOAM never makes
//   each step   execute() if the object is active (timeStart/timeEnd) and its executeControl fires; then write() if its
//               writeControl fires, executing first if it has not at this step (timeControl::write). OpenFOAM's
//               default writeControl is timeStep: a forceCoeffs history is written every step, not on the solver's
//               field writes
//   the end     the last step's execute/write, then end() where either control fires (Time.C:790-802)
// `enabled false` is not constructed (functionObjectList.C:1047-1108), and is said so.
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
    // They are reported as APPROXIMATED rather than ignored, because that is what they are. Reporting those
    // as "ignored" would be false; reporting nothing would hide a real difference from OF. Neither is
    // acceptable, so they get their own category. The two "outside" lists exist because the SAME type can be
    // faithful in one solver and an approximation in another, and each solver declares which it is.
    int read(const FoamDict* functions,
             const std::vector<std::string>& approximatedOutside = {},
             const std::vector<std::string>& appliedOutside = {})
    {
        entries_.clear();
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
            const std::string enabled = s.second.wordOr("enabled", "true");
            if (enabled == "false" || enabled == "no" || enabled == "off" || enabled == "n" || enabled == "f")
            {
                noticeApplied("functions/" + s.first, "enabled " + enabled + ": not constructed, as OpenFOAM does not construct it");
                continue;
            }
            // the controls first: an object whose controls brae cannot honour is not run at all, rather than run on
            // another schedule than the case's
            Entry e;
            std::string why = e.exec.read(s.second, "execute");
            if (why.empty()) why = e.write.read(s.second, "write");
            if (why.empty())
            {
                for (const char* k : {"triggerStart", "triggerEnd", "controlMode"})
                    if (s.second.found(k)) why = std::string(k) + ": trigger-controlled execution is not implemented";
            }
            if (!why.empty())
            {
                ++missed;
                noticeIgnored("functions/" + s.first, "type '" + type + "': " + why + ". This functionObject is NOT run.");
                continue;
            }
            e.timeStart = s.second.found("timeStart") ? s.second.scalarOr("timeStart", 0) : -std::numeric_limits<scalar>::max();
            e.timeEnd = s.second.found("timeEnd") ? s.second.scalarOr("timeEnd", 0) : std::numeric_limits<scalar>::max();
            const bool haveFactory = factories_.count(type) != 0;
            std::unique_ptr<FunctionObject> fo = create(s.first, type, s.second);
            if (fo)
            {
                e.fo = std::move(fo);
                entries_.push_back(std::move(e));
                continue;
            }
            // A REGISTERED factory that returns null has declined for a reason of its own and has
            // already said so (e.g. an unsupported diffusivity form, or a missing field). Falling
            // through to the generic "type is not implemented" here would contradict it -- the type IS
            // implemented; this instance of it was declined.
            if (haveFactory)
            {
                ++missed;
                continue;
            }
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

    // OF functionObjectList::start(): read() alone -- the objects are built in read() here, so nothing is left to do
    bool start() { return true; }

    // OF functionObjectList::execute(), for step s: each object's execute() then write(), under its own controls
    bool executeStep(const FoStep& s)
    {
        bool ok = true;
        for (Entry& e : entries_)
        {
            e.fo->bindStep(&s);
            if (!e.active(s)) continue;
            if (e.exec.fires(s))
            {
                e.executedAt = s.timeIndex;
                ok = timed(e.executeSeconds, e.executions, [&]() { return e.fo->execute(); }) && ok;
            }
            if (e.write.fires(s))
            {
                // timeControl::write: written results reflect the current state
                if (e.executedAt != s.timeIndex)
                {
                    e.executedAt = s.timeIndex;
                    ok = timed(e.executeSeconds, e.executions, [&]() { return e.fo->execute(); }) && ok;
                }
                ok = timed(e.writeSeconds, e.writes, [&]() { return e.fo->write(); }) && ok;
            }
        }
        return ok;
    }

    // BRAE_PHASE_TIME: each object's calls and wall seconds over the run, one line an object (run_timing.cuh)
    void reportTiming() const
    {
        if (!RunTiming::on()) return;
        for (const Entry& e : entries_)
            std::printf("brae-timing fo %s execute %ld %.6f write %ld %.6f\n", e.fo->name().c_str(), e.executions, e.executeSeconds, e.writes,
                        e.writeSeconds);
    }

    // OF functionObjectList::end(): end() where either control fires on the last step (timeControl::end)
    bool endRun(const FoStep& s)
    {
        bool ok = true;
        for (Entry& e : entries_)
        {
            e.fo->bindStep(&s);
            const bool a = e.exec.fires(s);
            const bool b = e.write.fires(s);
            if (e.active(s) && (a || b)) ok = e.fo->end() && ok;
        }
        return ok;
    }

    std::size_t size() const { return entries_.size(); }

    // OF selects functionObjects through a runtime selection table (addToRunTimeSelectionTable), which
    // lets a type live next to its dependencies rather than inside Time. Same idea here: a concrete
    // functionObject that needs the DEVICE solver cannot be constructed in this file without dragging
    // DeviceSimpleSolver into OpenFOAM/db, so the owning solver registers a factory before read().
    //
    // Register BEFORE read() -- read() is what consults the table.
    using Factory = std::function<std::unique_ptr<FunctionObject>(const std::string&, const FoamDict&)>;

    void registerType(const std::string& type, Factory f) { factories_[type] = std::move(f); }

private:
    struct Entry
    {
        std::unique_ptr<FunctionObject> fo;
        FoTimeControl exec;
        FoTimeControl write;
        scalar timeStart = 0;
        scalar timeEnd = 0;
        int executedAt = std::numeric_limits<int>::min();
        long executions = 0;
        long writes = 0;
        double executeSeconds = 0;
        double writeSeconds = 0;

        // timeControl::active(): within timeStart and timeEnd, half a step either side
        bool active(const FoStep& s) const
        {
            return s.time >= timeStart - 0.5 * s.deltaT && s.time <= timeEnd + 0.5 * s.deltaT;
        }
    };

    // a call, and under BRAE_PHASE_TIME its wall time between two drains of the device, so that it holds its own work alone
    template<class Call>
    static bool timed(
        double& seconds,
        long& calls,
        Call call)
    {
        ++calls;
        if (!RunTiming::on()) return call();
        RunTiming::drain();
        const double t0 = RunTiming::now();
        const bool ok = call();
        RunTiming::drain();
        seconds += RunTiming::now() - t0;
        return ok;
    }

    std::unique_ptr<FunctionObject> create(
        const std::string& name, const std::string& type, const FoamDict& dict)
    {
        auto it = factories_.find(type);
        if (it == factories_.end()) return nullptr;
        return it->second(name, dict);
    }

    std::map<std::string, Factory> factories_;
    std::vector<Entry> entries_;
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
    // OWNING constructor: Time builds its own WriteControl and resolves startFrom, exactly as OF's
    // Time::setControls() does (Time.C:149-188 for startFrom, TimeIO.C:277 for writeControl). Neither is
    // a solver's business -- OF's simpleFoam.C/pimpleFoam.C/rhoSimpleFoam.C mention neither.
    //
    // Before this, all three brae drivers carried their own copies: two inline `startFrom` blocks and
    // three separate `isWriteTime` lambdas. Measured identical before consolidating (startFrom on an
    // adversarial directory; writeControl over 8 cases x 40 steps including runTime float accumulation
    // and a non-zero-startTime restart), so this is pure de-duplication -- but read_surface_field was
    // the same shape and HAD silently diverged, at 10^6x the reproducibility floor.
    // readFunctions false: a driver that registers its types and reads the functions itself (readFunctionObjects), so that
    // they are read, and reported, once and with its types known
    Time(const std::string& caseDir, const FoamDict& controlDict,
         const std::vector<std::pair<std::string, FunctionObjectList::Factory>>& types = {},
         const std::vector<std::string>& approximatedOutside = {},
         const std::vector<std::string>& appliedOutside = {},
         bool readFunctions = true)
      : owned_(controlDict), wc_(&owned_), nSteps_(0)
    {
        const StartTime start = resolveStartTime(caseDir, controlDict);
        startName_ = start.name;
        owned_.setStartTime(static_cast<scalar>(start.value));
        startValue_ = static_cast<scalar>(start.value);
        endTime_ = controlDict.scalarOr("endTime", std::numeric_limits<scalar>::max());
        for (const auto& t : types) functionObjects_.registerType(t.first, t.second);
        if (readFunctions) functionObjects_.read(controlDict.subDict("functions"), approximatedOutside, appliedOutside);
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

    // The run's resolved start, for a driver that resolves it itself (gpuPimpleFoam): what a functionObject's step reads (its
    // time, its index, its output directory). The driver's own write cadence is not touched.
    void setRunStart(
        const std::string& name,
        scalar value)
    {
        startName_ = name;
        startValue_ = value;
    }
    WriteControl&      writeControl()    { return owned_; }

    Time(const FoamDict& controlDict, WriteControl* wc, int nSteps,
         const std::vector<std::pair<std::string, FunctionObjectList::Factory>>& types = {},
         const std::vector<std::string>& approximatedOutside = {},
         const std::vector<std::string>& appliedOutside = {})
      : owned_(controlDict), wc_(wc ? wc : &owned_), nSteps_(nSteps)
    {
        startValue_ = wc_->startTime();
        endTime_ = controlDict.scalarOr("endTime", std::numeric_limits<scalar>::max());
        for (const auto& t : types) functionObjects_.registerType(t.first, t.second);
        functionObjects_.read(controlDict.subDict("functions"), approximatedOutside, appliedOutside);
    }

    // OF Time::loop(): test run(), fire functionObjects for the step just solved, advance. The solver writes
    //     while (time.loop()) { ...its own iteration... }
    // and never mentions functionObjects, exactly as OF's solvers do not.
    bool loop()
    {
        if (iter_ >= nSteps_) return false;
        if (iter_ == 0) functionObjects_.start();   // OF Time.C:820: construction only
        else            runStep();                  // OF Time.C:797,825: the previous step's
        ++iter_;
        return true;
    }

    int    timeIndex() const { return iter_; }
    scalar timeValue() const { return wc_ ? wc_->timeValue(iter_) : scalar(iter_); }
    // OF Time::deltaTValue(): the controlDict's deltaT, what an expression's deltaT() reads. NaN without a
    // WriteControl, so a consumer that needs it refuses rather than assuming 1.
    scalar deltaT() const { return wc_ ? wc_->deltaT() : std::numeric_limits<scalar>::quiet_NaN(); }
    std::string timeName() const { return WriteControl::timeName(timeValue()); }

    // OF runTime.write(): true when this step is a write time, and noted as one: a functionObject under
    // `writeControl writeTime` fires on the steps whose fields the solver writes.
    bool writeTime()
    {
        if (iter_ >= nSteps_) return false;   // the final write is the solver's own, as in OF writeAndEnd
        if (!wc_ || !wc_->isWriteTime(iter_, timeValue())) return false;
        writtenAt_ = iter_;
        return true;
    }

    // OF Time.C:790-802, the "Ensure functionObjects execute on last time step" block. run() fires
    // execute() BEFORE the increment, so the top-of-loop firing always sees the PREVIOUS body's
    // results and the final body would otherwise never be seen at all. OF therefore does one more
    // execute() on the way out, then end(). Omitting it silently drops the last time step from every
    // functionObject -- for scalarTransport that is one whole transport step missing from the answer.
    void end()
    {
        stepNow_ = stepContext();
        if (iter_ > 0) functionObjects_.executeStep(stepNow_);
        functionObjects_.endRun(stepNow_);
        if (reportTiming_) functionObjects_.reportTiming();
    }

    // whether this process prints the functionObjects' timing (a multi-GPU run's rank 0 alone)
    void setReportTiming(bool on) { reportTiming_ = on; }

    // Lets a solver stop early (residualControl) without Time thinking the run was truncated.
    void stop() { nSteps_ = iter_; }

    // The functionObject REPORT must happen at start-up, so a case that refuses on an unsupported model
    // still learns which of its functionObjects brae would not have honoured. But the step count is only
    // known once endTime/startTime/deltaT have been resolved, which is later. So Time is constructed
    // early (reporting fires) and told the count here, before the loop.
    void setSteps(int n) { nSteps_ = n; }

    // For a solver that owns its own write cadence: this step's fields are written (what writeControl writeTime
    // reads). The functionObjects' own writes are their writeControl's, fired each step.
    void write() { writtenAt_ = iter_; }

    // OF has `class Time : public objectRegistry`. Composition rather than inheritance here, but the
    // property that matters is the same: a functionObject reaches the solver and its fields BY NAME and
    // LATE, so Time can be constructed before the things its functionObjects will need.
    ObjectRegistry&       registry()       { return registry_; }
    const ObjectRegistry& registry() const { return registry_; }

    FunctionObjectList&       functionObjects()       { return functionObjects_; }
    const FunctionObjectList& functionObjects() const { return functionObjects_; }

private:
    // the step just solved, as a functionObject's controls read it. The time index counts from the case's own 0
    // (startTime/deltaT at the start), as OpenFOAM's continues across a restart from the time's uniform/time
    FoStep stepContext() const
    {
        FoStep s;
        const scalar dt = wc_ ? wc_->deltaT() : scalar(1);
        const scalar t0 = startValue_;
        s.deltaT = dt;
        s.startTime = t0;
        s.endTime = endTime_;
        s.time = t0 + static_cast<scalar>(iter_) * dt;
        s.timeIndex = static_cast<int>(std::llround(t0 / dt)) + iter_;
        s.writeTime = writtenAt_ == iter_;
        s.timeName = WriteControl::timeName(s.time);
        s.startName = startName_;
        return s;
    }

    void runStep()
    {
        stepNow_ = stepContext();
        functionObjects_.executeStep(stepNow_);
    }

    WriteControl       owned_;      // used when Time constructs its own; wc_ points here or at the caller's
    WriteControl*      wc_;
    std::string        startName_ = "0";
    scalar             startValue_ = 0;
    scalar             endTime_ = std::numeric_limits<scalar>::max();
    ObjectRegistry     registry_;
    FunctionObjectList functionObjects_;
    FoStep             stepNow_;
    int                nSteps_;
    int                iter_ = 0;
    int                writtenAt_ = -1;
    bool               reportTiming_ = true;
};

}   // namespace brae
