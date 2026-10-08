#pragma once
// Foam::functionObject -- the interface every function object is run through
// (src/OpenFOAM/db/functionObjects/functionObject/functionObject.H:222-366), and what one reads of the run.
//
// In OpenFOAM an object holds `time_` and looks the mesh and the fields up in its registry
// (regionFunctionObject, fvMeshFunctionObject). brae has no registry: FunctionObjectRun is what the loop that
// owns the clock and the fields hands every object, and FunctionObjectFields is the registry's part of it --
// the fields by OpenFOAM's own names, read where they live, host vectors on the host loop and a gather of
// the cells asked for on the GPU loop.
//
// NOT brae::FunctionObject of brae_time.cuh: that one is the single-phase solvers' reduced list (name,
// execute, write, end; no timing wrapper, no state dictionary). They move onto this one in a later unit.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "function_object_properties_cpp.cuh"
#include "primitive_mesh.cuh"
#include "time_control_cpp.cuh"
#include <string>
#include <vector>

namespace brae {

// The registry's fields, as far as the loop hands them over. A name it does not hold is refused by the
// object that asked, with names() in the message.
class FunctionObjectFields
{
public:
    virtual ~FunctionObjectFields() = default;

    virtual bool volScalar(const std::string& name) const = 0;
    virtual bool volVector(const std::string& name) const = 0;
    virtual std::vector<std::string> names() const = 0;

    // the values of field `name` in `cells`, as the step just taken left them
    virtual void cellValues(
        const std::string& name,
        const std::vector<label>& cells,
        std::vector<scalar>& out) = 0;
    virtual void cellValues(
        const std::string& name,
        const std::vector<label>& cells,
        std::vector<vector>& out) = 0;
};

struct FunctionObjectRun
{
    // Time::globalPath()
    std::string casePath;
    // Time::timeName(startTime): the instance of every output file (probes.C createProbeFiles,
    // writeFile.C baseTimeDir)
    std::string startTimeName;
    // IOstream::defaultPrecision(), the case's writePrecision
    int writePrecision = 6;
    // the instant the objects are called at; the loop sets it before every call
    FunctionObjectTime time;
    // the mesh as it stands, and its cell centres
    const PrimitiveMesh* mesh = nullptr;
    const std::vector<vector>* cellCentres = nullptr;
    // whether the mesh changes its cells during the run (a refining mesh)
    bool topologyChanges = false;
    FunctionObjectFields* fields = nullptr;
    FunctionObjectProperties* properties = nullptr;
};

namespace functionObjects {

class FunctionObject
{
public:
    FunctionObject(
        const std::string& name,
        const FunctionObjectRun& run)
    :
        name_(name),
        run_(run)
    {}

    virtual ~FunctionObject() = default;

    const std::string& name() const
    {
        return name_;
    }

    virtual const char* type() const = 0;

    // functionObject.H:329-349. execute() and write() are pure there too.
    virtual bool read(const FoamDict& dict) = 0;
    virtual bool execute() = 0;
    virtual bool write() = 0;

    virtual bool end()
    {
        return true;
    }

    // polyMesh::movePoints and ::updateMesh call the list (polyMesh.C:1292, polyMeshUpdate.C:152)
    virtual void movePoints()
    {}

    virtual void updateMesh()
    {}

protected:
    // stateFunctionObject::setResult -> properties::setObjectResult
    void setResult(
        const std::string& typeName,
        const std::string& entryName,
        const std::string& value) const
    {
        run_.properties->setResult(name_, typeName, entryName, value);
    }

    std::string name_;
    const FunctionObjectRun& run_;
};

}   // namespace functionObjects
}   // namespace brae
