#pragma once
// interFoam's function objects: the list built from the case's controlDict, the fields of interFoam's
// registry that are handed to it, and Time::run()'s calls as the two time loops make them.
//
// WHERE THE CALLS SIT. OpenFOAM runs the list at the TOP of a step, on what the step before left
// (Time::run, Time.C:781-860), and the state dictionary is written there too, into the time directory that
// step's runTime.write() made. Nothing touches a field between the bottom of one step and the top of the next,
// so here the list is executed in the step's write stage, ahead of the writer, which then writes the state
// dictionary with the directory's other files:
//
//     start()      before the first step: the objects are built on the start mesh, nothing is executed
//     execute()    in every step's write stage, written or not
//     end()        after the step that ends the run (Time::run's `!isRunning` branch: execute, then end)
//     movePoints() after the mesh has moved, updateMesh() after it has changed (polyMesh.C:1292,
//                  polyMeshUpdate.C:152)
//
// THE FIELDS HANDED OVER are the four a time directory holds and the gates compare: p, p_rgh, alpha.<phase1>
// and U. An object asking for another is refused by name (the object says which are held).
//
// THE TYPES PORTED: probes. An enabled object of another type is NOT RUN and is said so by name once, at
// start; setTimeStep and setTimeStepFaRegion, which change the solution, are refused by the case reader.
#include "brae_notice.cuh"
#include "function_object_list_cpp.cuh"
#include "probes_cpp.cuh"
#include "time_controls.cuh"
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

// the values of `field` in `cells`: a function object's read of a field that lives on the host
template <class Type>
inline void functionObjectCellsOf(
    const std::vector<Type>& field,
    const std::vector<label>& cells,
    std::vector<Type>& out)
{
    out.resize(cells.size());
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
        out[i] = field[cells[i]];
    }
}

// The registry's fields as a loop registers them: a name, and how the values of a list of cells are read
// where the field lives.
class InterFunctionObjectFields : public FunctionObjectFields
{
public:
    using ScalarCells = std::function<void(const std::vector<label>&, std::vector<scalar>&)>;
    using VectorCells = std::function<void(const std::vector<label>&, std::vector<vector>&)>;

    void add(
        const std::string& name,
        ScalarCells read)
    {
        scalars_[name] = std::move(read);
    }

    void add(
        const std::string& name,
        VectorCells read)
    {
        vectors_[name] = std::move(read);
    }

    bool volScalar(const std::string& name) const override
    {
        return scalars_.count(name) != 0;
    }

    bool volVector(const std::string& name) const override
    {
        return vectors_.count(name) != 0;
    }

    std::vector<std::string> names() const override
    {
        std::vector<std::string> out;
        for (const auto& s : scalars_)
        {
            out.push_back(s.first);
        }
        for (const auto& v : vectors_)
        {
            out.push_back(v.first);
        }
        return out;
    }

    void cellValues(
        const std::string& name,
        const std::vector<label>& cells,
        std::vector<scalar>& out) override
    {
        scalars_.at(name)(cells, out);
    }

    void cellValues(
        const std::string& name,
        const std::vector<label>& cells,
        std::vector<vector>& out) override
    {
        vectors_.at(name)(cells, out);
    }

private:
    std::map<std::string, ScalarCells> scalars_;
    std::map<std::string, VectorCells> vectors_;
};

class InterFunctionObjects
{
public:
    InterFunctionObjectFields fields;

    // functionObjectList::start(): the objects are built. `mesh` and `cellCentres` are the loop's own, read
    // live at every search.
    void start(
        const std::string& caseDir,
        const std::string& startTimeName,
        int writePrecision,
        scalar startTime,
        scalar endTime,
        label startTimeIndex,
        const PrimitiveMesh& mesh,
        const std::vector<vector>& cellCentres,
        bool topologyChanges)
    {
        FunctionObjectRun& run = list_.run();
        run.casePath = caseDir;
        run.startTimeName = startTimeName;
        run.writePrecision = writePrecision;
        run.time.value = startTime;
        run.time.startTime = startTime;
        run.time.endTime = endTime;
        run.time.timeIndex = startTimeIndex;
        run.mesh = &mesh;
        run.cellCentres = &cellCentres;
        run.topologyChanges = topologyChanges;
        run.fields = &fields;
        list_.registerType(
            "probes",
            [](
                const std::string& name,
                const FunctionObjectRun& r,
                const FoamDict& dict)
            {
                return std::unique_ptr<functionObjects::FunctionObject>(new functionObjects::Probes(name, r, dict));
            });
        list_.read(readDict(caseDir + "/system/controlDict"), !stateControl("fresh"));
        for (std::size_t i = 0; i < list_.size(); ++i)
        {
            std::printf("  function object `%s` (%s): run at every step as OpenFOAM's Time::run does\n",
                        list_.object(i).name().c_str(), list_.object(i).type());
        }
        for (const auto& o : list_.notRun())
        {
            noticeIgnored(
                "controlDict functions/" + o.first,
                "type `" + o.second + "` is not ported, so this function object is NOT run: its "
                "postProcessing/ output and the fields it writes are not produced. Ported: probes. (Its "
                "adjustableRunTime write times still trim the time step, as OpenFOAM's do.)");
        }
        propertiesBody_ = list_.propertiesBody();
    }

    // the instant Time::run() would call the list at, after the step that reached `value`
    void setTime(
        scalar value,
        scalar deltaT,
        label timeIndex,
        bool writeTime)
    {
        FunctionObjectRun& run = list_.run();
        run.time.value = value;
        run.time.deltaT = deltaT;
        run.time.timeIndex = timeIndex;
        run.time.writeTime = writeTime;
    }

    // functionObjectList::execute() for the step just taken, and the clock's copy of every
    // adjustableRunTime write index against the object's own
    void execute(const WriteCadence& cadence)
    {
        const std::string before = list_.propertiesBody();
        list_.execute();
        propertiesBody_ = stateControl("stale") ? before : list_.propertiesBody();
        if (std::getenv("BRAE_CONTROL_FUNCTION_OBJECT_STEP") != nullptr
            || std::getenv("BRAE_CONTROL_FUNCTION_OBJECT_TIMING") != nullptr)
        {
            // a gate's control breaks one of the two on purpose
            return;
        }
        for (std::size_t i = 0; i < list_.size(); ++i)
        {
            const auto* timed = dynamic_cast<const functionObjects::TimeControlFunctionObject*>(&list_.object(i));
            if (!timed || timed->writeControl().mode != TimeControl::Mode::adjustableRunTime)
            {
                continue;
            }
            for (const FunctionObjectCadence& fo : cadence.functionObjects)
            {
                if (fo.name == timed->name() && fo.executionIndex != timed->writeControl().executionIndex)
                {
                    throw std::runtime_error(
                        "brae interFoam: function object `" + fo.name + "`: the write index the time step "
                        "was trimmed by (" + std::to_string(fo.executionIndex) + ") is not the one its rows "
                        "were written by (" + std::to_string(timed->writeControl().executionIndex) + "). "
                        "They are one number in OpenFOAM (timeControl::executionIndex_).");
                }
            }
        }
    }

    void end()
    {
        list_.end();
        propertiesBody_ = list_.propertiesBody();
    }

    void movePoints()
    {
        list_.movePoints();
    }

    void updateMesh()
    {
        list_.updateMesh();
    }

    // the body of <time>/uniform/functionObjects/functionObjectProperties as the last execute left it
    const std::string& propertiesBody() const
    {
        return propertiesBody_;
    }

    std::size_t size() const
    {
        return list_.size();
    }

private:
    // BRAE_CONTROL_FUNCTION_OBJECT_STATE names ONE rule of the state dictionary to break -- the gates'
    // controls, not user switches:
    //   stale  a time directory's dictionary is the one the step BEFORE left: written ahead of the execute
    //   fresh  the start time's dictionary is not read
    static bool stateControl(const char* part)
    {
        static const char* const set = std::getenv("BRAE_CONTROL_FUNCTION_OBJECT_STATE");
        return set && std::string(set) == part;
    }

    functionObjects::FunctionObjectList list_;
    std::string propertiesBody_;
};

}   // namespace interFoam
}   // namespace cpu
}   // namespace brae
