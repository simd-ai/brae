#pragma once
// Foam::functionObjectList -- the run's function objects, and where the run calls them
// (src/OpenFOAM/db/functionObjects/functionObjectList/functionObjectList.C).
//
// No solver calls a function object. Time owns the list (Time.C:469) and Time::run(), which every solver's
// loop condition reaches, runs it (Time.C:781-860):
//
//     the first step        start()    = read(): the objects are BUILT, nothing is executed    :609-612
//     every later step      execute()  at the TOP of the step, on the fields the step before left
//     the run is over       execute()  once more, then end()
//
// execute() (:615-792) calls execute() and then write() on EVERY object at EVERY step -- whether anything
// happens is the object's wrapper's business (time_control_function_object_cpp.cuh) -- and then, at a write
// time, writes the state dictionary the objects share (:776-789, function_object_properties_cpp.cuh). Here
// the solver's writer writes that file with the time directory's others, from propertiesBody(): the fields
// are the same before the top of the next step as after the bottom of this one, so the solver executes the
// list first and writes the directory second.
//
// read() (:939-1209) takes controlDict's `functions`, one sub-dictionary an object, in the entry's order: an
// object with `enabled false` is skipped (:1047), one with a timing entry is wrapped (:1066), and the type
// is looked up in a table (functionObject::New). brae's table holds the types that are ported. A type it does
// not hold is NOT RUN and is listed in notRun() for the solver to say so by name; what such an object would
// have done to the SOLUTION is the solver's to refuse.
#include "function_object_cpp.cuh"
#include "time_control_function_object_cpp.cuh"
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brae {
namespace functionObjects {

class FunctionObjectList
{
public:
    using Factory = std::function<std::unique_ptr<FunctionObject>(
        const std::string& name,
        const FunctionObjectRun& run,
        const FoamDict& dict)>;

    FunctionObjectList()
    {
        run_.properties = &properties_;
    }

    // the objects hold a reference to run(): the list stays where it was built
    FunctionObjectList(const FunctionObjectList&) = delete;
    FunctionObjectList& operator=(const FunctionObjectList&) = delete;

    // functionObject's run-time selection table, as far as it is ported. Before read().
    void registerType(
        const std::string& type,
        Factory factory)
    {
        factories_[type] = std::move(factory);
    }

    FunctionObjectRun& run()
    {
        return run_;
    }

    // `readProperties` false is a gate's control: the start time's state dictionary left unread
    void read(
        const FoamDict& controlDict,
        bool readProperties = true)
    {
        // createPropertiesDict (:88-108): READ_IF_PRESENT at the start time, with or without objects
        if (readProperties)
        {
            properties_.read(
                run_.casePath + "/" + run_.startTimeName + "/uniform/functionObjects/functionObjectProperties");
        }
        const FoamDict* fns = controlDict.subDict("functions");
        if (!fns)
        {
            return;
        }
        for (const auto& leaf : fns->leaves)
        {
            if (!leaf.first.empty() && leaf.first[0] == '#')
            {
                // #includeFunc and its kin expand to an object's dictionary in OpenFOAM
                throw std::runtime_error(
                    "brae: controlDict's `functions` has the directive `" + leaf.first + "`, which OpenFOAM "
                    "expands into a function object and brae's reader does not. Write the object's "
                    "dictionary out in `functions`.");
            }
        }
        for (const auto& entry : fns->subs)
        {
            const std::string& key = entry.first;
            const FoamDict& dict = entry.second;
            if (!dict.switchOr("enabled", true))
            {
                continue;
            }
            // functionObject::New: dict.get<word>("type"), mandatory
            if (!dict.found("type"))
            {
                throw std::runtime_error(
                    "brae: controlDict's function object `" + key + "` has no `type`. OpenFOAM stops on that "
                    "too (functionObject::New).");
            }
            const std::string type = dict.wordOr("type", "");
            const auto it = factories_.find(type);
            if (it == factories_.end())
            {
                notRun_.push_back({key, type});
                continue;
            }
            std::unique_ptr<FunctionObject> object = it->second(key, run_, dict);
            if (TimeControlFunctionObject::entriesPresent(dict))
            {
                object.reset(new TimeControlFunctionObject(key, run_, dict, std::move(object)));
            }
            objects_.push_back(std::move(object));
        }
    }

    // execute(writeProperties = true) (:615-792)
    bool execute()
    {
        bool ok = true;
        for (auto& object : objects_)
        {
            ok = object->execute() && ok;
            ok = object->write() && ok;
        }
        return ok;
    }

    // end() (:838-905)
    bool end()
    {
        bool ok = true;
        for (auto& object : objects_)
        {
            ok = object->end() && ok;
        }
        return ok;
    }

    // :1227-1248
    void movePoints()
    {
        for (auto& object : objects_)
        {
            object->movePoints();
        }
    }

    void updateMesh()
    {
        for (auto& object : objects_)
        {
            object->updateMesh();
        }
    }

    // the state dictionary's entries, the body of <time>/uniform/functionObjects/functionObjectProperties
    std::string propertiesBody() const
    {
        return properties_.body();
    }

    std::size_t size() const
    {
        return objects_.size();
    }

    const FunctionObject& object(std::size_t i) const
    {
        return *objects_[i];
    }

    // name and type of every enabled object whose type the table does not hold
    const std::vector<std::pair<std::string, std::string>>& notRun() const
    {
        return notRun_;
    }

private:
    FunctionObjectRun run_;
    FunctionObjectProperties properties_;
    std::map<std::string, Factory> factories_;
    std::vector<std::unique_ptr<FunctionObject>> objects_;
    std::vector<std::pair<std::string, std::string>> notRun_;
};

}   // namespace functionObjects
}   // namespace brae
