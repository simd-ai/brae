#pragma once
// WHICH TIME DIRECTORY A MESH FILE IS READ FROM -- Time::findInstance, and polyMesh's use of it.
//
// brae read the mesh from `constant/polyMesh` and nowhere else, in four places. That is right for a case
// that starts from rest and WRONG for a genuine restart, because OpenFOAM does not look there: it SEARCHES
// for each file, and a run that refined or moved its mesh has written a newer one into a time directory.
//
// WHAT OPENFOAM DOES, and the two halves are separate classes. Time::findInstance (Time.C:731-754) hands an
// IOobject to fileOperation::findInstance (fileOperation.C:1077-1190), which is the rule:
//
//   1. the CURRENT instance first -- `if (exists(io)) return io;` -- so a case whose start directory holds
//      the file needs no search at all
//   2. otherwise take Time::times(), walk BACK to the last instance whose value is <= startValue, and from
//      there walk DOWN to the first instance that has the file
//   3. `stopInstance` bounds that walk: reaching it ends the search (and is a FatalError under MUST_READ)
//   4. `constant` is re-tested at the end for the cases times() cannot express (an empty list, a first
//      entry that is not `constant`, or a NEGATIVE startValue, which makes the backward walk skip it)
//
// and Time::times() is fileOperation::sortTimes (fileOperation.C:221-275): `constant` FIRST with value 0,
// then every directory whose name parses as a scalar, sorted -- and the sort covers [1, end) only, so
// `constant` stays at index 0 whatever the numeric values are.
//
// polyMesh RESOLVES EACH FILE SEPARATELY, and that is the part a single `polyMeshDir` cannot express
// (polyMesh.C:175-245):
//
//   points                            findInstance(meshDir, "points")
//   faces                             findInstance(meshDir, "faces")
//   owner, neighbour                  faces_.instance()            READ_IF_PRESENT
//   boundary                          findInstance(meshDir, "boundary", MUST_READ, stop = faces instance)
//   pointZones, faceZones, cellZones  faces_.instance()            READ_IF_PRESENT
//
// So POINTS AND FACES CAN COME FROM DIFFERENT DIRECTORIES, and on a MOVING mesh they always do: the run
// writes `points` into every time directory and leaves `faces` in `constant`. A REFINING mesh writes both.
// `boundary` is searched independently but never older than the faces it describes -- OpenFOAM's own
// comment there is "allow 'newer' boundary file".
//
// facesInstance() IS faces_.instance() (polyMesh.C) and is what hexRef8 reads cellLevel, pointLevel and
// refinementHistory from (hexRef8.C:1912-1990), which is why this matters beyond the mesh itself: on a
// restart of a refined case the levels live beside the faces, and reading `constant/polyMesh/cellLevel`
// there either finds the START mesh's levels or nothing at all.
#include "cf_types.cuh"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace timePaths {

struct Instant
{
    std::string name;
    scalar      value = 0;
};

// Time::times() -- fileOperation::sortTimes (fileOperation.C:221-275)
inline std::vector<Instant> times(const std::string& caseDir)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<Instant> numeric;
    bool haveConstant = false;
    for (const auto& e : fs::directory_iterator(caseDir, ec))
    {
        if (!e.is_directory()) continue;
        const std::string nm = e.path().filename().string();
        if (nm == "constant")
        {
            haveConstant = true;
            continue;
        }
        char* end = nullptr;
        const double v = std::strtod(nm.c_str(), &end);
        // readScalar's own test: the WHOLE name must parse, so `0.orig` and `system` are not times
        if (end == nm.c_str() || *end != '\0') continue;
        numeric.push_back(Instant{nm, static_cast<scalar>(v)});
    }
    std::sort(numeric.begin(), numeric.end(),
              [](const Instant& a, const Instant& b) { return a.value < b.value; });
    std::vector<Instant> out;
    // `constant` is index 0 with value 0 and is NOT part of the sort, which is OpenFOAM's own ordering
    if (haveConstant) out.push_back(Instant{"constant", scalar(0)});
    out.insert(out.end(), numeric.begin(), numeric.end());
    return out;
}

// IOobject::typeHeaderOk's file test, as findInstance uses it: the file, or its gzipped form. An EMPTY
// name asks about the directory alone, which is findInstance's documented "search for directory only".
inline bool instanceHas(
    const std::string& caseDir,
    const std::string& instance,
    const std::string& local,
    const std::string& name)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = caseDir + "/" + instance + (local.empty() ? "" : ("/" + local));
    if (name.empty()) return fs::is_directory(dir, ec);
    return fs::exists(dir + "/" + name, ec) || fs::exists(dir + "/" + name + ".gz", ec);
}

// Time::findInstance (Time.C:731-754) -> fileOperation::findInstance (fileOperation.C:1077-1190).
// Returns an EMPTY string only when the search failed and `constantFallback` is false, which is the one
// case OpenFOAM also returns an empty word for; a MUST_READ failure is the caller's to refuse, because
// what to say about it depends on the file.
inline std::string findInstance(
    const std::string& caseDir,
    const std::string& timeName,
    scalar             startValue,
    const std::string& local,
    const std::string& name,
    const std::string& stopInstance = std::string(),
    bool               constantFallback = true)
{
    // 1. the current instance, tested before anything is listed
    if (instanceHas(caseDir, timeName, local, name)) return timeName;

    const std::vector<Instant> ts = times(caseDir);
    // 2. back to the last instance at or before startValue...
    long i = static_cast<long>(ts.size()) - 1;
    for (; i >= 0; --i)
    {
        if (ts[static_cast<std::size_t>(i)].value <= startValue) break;
    }
    // ...and down from there
    for (; i >= 0; --i)
    {
        const std::string& inst = ts[static_cast<std::size_t>(i)].name;
        // OpenFOAM's own shortcut: the current instance was tested above
        if (inst == timeName && inst != stopInstance) continue;
        if (instanceHas(caseDir, inst, local, name)) return inst;
        if (!stopInstance.empty() && inst == stopInstance) break;
    }
    // 4. and `constant` for the cases the walk above cannot reach
    const bool constantMissed = ts.empty() || ts.front().name != "constant" || startValue < scalar(0);
    if (constantMissed && instanceHas(caseDir, "constant", local, name)) return "constant";
    return constantFallback ? std::string("constant") : std::string();
}

// polyMesh's own resolution (polyMesh.C:175-245). `points` and `faces` are independent; `boundary` is
// searched separately but stops at the faces instance.
struct MeshInstances
{
    std::string points;
    std::string faces;
    std::string boundary;

    // owner, neighbour, the three zone lists, and hexRef8's cellLevel/pointLevel/refinementHistory all
    // sit beside the FACES -- polyMesh reads every one of them at faces_.instance()
    std::string facesDir(const std::string& caseDir) const
    {
        return caseDir + "/" + faces + "/polyMesh";
    }
    std::string pointsDir(const std::string& caseDir) const
    {
        return caseDir + "/" + points + "/polyMesh";
    }
    std::string boundaryDir(const std::string& caseDir) const
    {
        return caseDir + "/" + boundary + "/polyMesh";
    }
};

inline MeshInstances meshInstances(
    const std::string& caseDir,
    const std::string& timeName,
    scalar             startValue,
    const std::string& meshDir = "polyMesh")
{
    MeshInstances mi;
    // A GATE'S CONTROL, and the strongest kind: it restores exactly what brae SHIPPED before this unit --
    // every mesh file taken from constant/, whatever the time directories hold. On a case that starts from
    // rest it is not a control at all, because that IS the answer; on a genuine restart of a refined mesh it
    // puts the run on the mesh the case started from, and the cell count says so at the first step.
    if (std::getenv("BRAE_CONTROL_MESH_CONSTANT_ONLY") != nullptr)
    {
        mi.points = mi.faces = mi.boundary = "constant";
        return mi;
    }
    mi.points = findInstance(caseDir, timeName, startValue, meshDir, "points");
    mi.faces  = findInstance(caseDir, timeName, startValue, meshDir, "faces");
    mi.boundary = findInstance(caseDir, timeName, startValue, meshDir, "boundary", mi.faces);
    return mi;
}

// ...and the form a solver has to hand: it knows the START DIRECTORY, whose basename IS OpenFOAM's
// timeName() and whose numeric value is timeOutputValue(). One helper so the derivation exists once --
// two call sites deriving the same instance separately is how a setting comes to mean two things.
inline MeshInstances meshInstancesForStartDir(
    const std::string& caseDir,
    const std::string& startDir)
{
    std::string timeName = startDir;
    const std::size_t slash = timeName.find_last_of('/');
    if (slash != std::string::npos) timeName = timeName.substr(slash + 1);
    char* end = nullptr;
    const double v = std::strtod(timeName.c_str(), &end);
    // a non-numeric start directory is not a time: `constant` is the only instance to search from
    const scalar startValue = (end != timeName.c_str() && *end == '\0') ? static_cast<scalar>(v) : scalar(0);
    return meshInstances(caseDir, timeName, startValue);
}

} // namespace timePaths
} // namespace cpu
} // namespace brae
