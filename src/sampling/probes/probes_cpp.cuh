#pragma once
// Foam::probes -- fields sampled at points, a row a write, a file a field
// (src/sampling/probes/probes.H, probes.C, probesTemplates.C).
//
// END TO END, as OpenFOAM does it:
//   read()            probeLocations, fields, fixedLocations (true), includeOutOfBounds (true), verbose,
//                     sampleOnExecute (false), interpolationScheme ("cell"); then findElements and
//                     prepare(ACTION_NONE), which classifies the fields and opens nothing        probes.C:399-429
//   findElements()    elementList_[probei] = mesh.findCell(location), -1 where no cell holds it, with a
//                     warning; the search is poly_mesh_tet_search_cpp.cuh                         probes.C:160-273
//   write()           performAction(ACTION_ALL): the volScalarFields in sorted order, then the
//                     volVectorFields -- sample, storeResults, writeValues                       probes.C:432-466
//   execute()         the same without the row, and only under sampleOnExecute                   probes.C:452-460
//   sample()          the value of the probe's CELL: under fixedLocations interpolation<Type>::New(scheme)
//                     ->interpolate(position, celli), which for `cell` is psi[celli]
//                     (interpolationCell.H:73-81), and psi[celli] directly otherwise; -VGREAT for a probe
//                     with no cell                                                     probesTemplates.C:189-232
//   storeResults()    average(f), min(f), max(f), size(f) of the row into the state dictionary  :100-121
//   writeValues()     the time and each probe's value, each left-justified in writePrecision + 7 columns
//                     and a space between                                                        :127-154
//   createProbeFiles  at the FIRST row: <case>/postProcessing/<name>/<start time>/<field>, a line a probe
//                     and the column heads                                                       probes.C:55-155
//   movePoints()      the cells are searched again, only under fixedLocations                    probes.C:560-568
//   updateMesh()      searched again under fixedLocations, mapped through the change otherwise  probes.C:469-557
//
// NOT PORTED, each refused by name: a surface field; a field the loop does not hand over; a pattern in
// `fields`; `interpolationScheme` other than `cell` under fixedLocations (cellPoint and the rest interpolate
// inside the cell); `fixedLocations false` on a mesh that refines (the probe's cell is then mapped through
// mapPolyMesh::reverseCellMap).
#include "function_object_cpp.cuh"
#include "poly_mesh_tet_search_cpp.cuh"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace functionObjects {

class Probes : public FunctionObject
{
public:
    Probes(
        const std::string& name,
        const FunctionObjectRun& run,
        const FoamDict& dict)
    :
        FunctionObject(name, run)
    {
        read(dict);
    }

    const char* type() const override
    {
        return "probes";
    }

    bool read(const FoamDict& dict) override
    {
        const std::string who = "brae: function object `" + name_ + "` (probes)";
        // dict.readEntry("probeLocations") and ("fields"): both mandatory
        if (!dict.found("probeLocations") || !dict.found("fields"))
        {
            throw std::runtime_error(
                who + " needs `probeLocations` and `fields`. OpenFOAM stops on a missing one too "
                "(probes.C:401-402).");
        }
        std::vector<scalar> xyz = dict.scalarListOr("probeLocations", {});
        // a list may carry its size: N((x y z) ...)
        if (xyz.size() % 3 == 1 && xyz[0] == scalar((xyz.size() - 1)/3))
        {
            xyz.erase(xyz.begin());
        }
        if (xyz.size() % 3 != 0)
        {
            throw std::runtime_error(
                who + ": `probeLocations` is not a list of points that brae's reader can take as numbers "
                "(a macro or an expression inside it is not expanded).");
        }
        locations_.clear();
        for (std::size_t i = 0; i < xyz.size(); i += 3)
        {
            locations_.push_back(vector{xyz[i], xyz[i + 1], xyz[i + 2]});
        }

        fixedLocations_ = dict.switchOr("fixedLocations", true);
        includeOutOfBounds_ = dict.switchOr("includeOutOfBounds", true);
        verbose_ = dict.switchOr("verbose", false);
        onExecute_ = dict.switchOr("sampleOnExecute", false);
        const std::string scheme = dict.wordOr("interpolationScheme", "cell");
        if (scheme != "cell")
        {
            if (fixedLocations_)
            {
                throw std::runtime_error(
                    who + " has `interpolationScheme " + scheme + "`. Under fixedLocations OpenFOAM "
                    "interpolates inside the cell with it (probesTemplates.C:196-213); only `cell`, the "
                    "cell's own value, is ported.");
            }
            // probes.C:404-418: a warning, and the cell's value whatever the entry says
            std::printf("  probes `%s`: only cell interpolation can be applied when not using fixedLocations; "
                        "`interpolationScheme %s` is ignored, as OpenFOAM ignores it\n",
                        name_.c_str(), scheme.c_str());
        }
        if (!fixedLocations_ && run_.topologyChanges)
        {
            throw std::runtime_error(
                who + " has `fixedLocations false` on a mesh that refines. OpenFOAM then carries each "
                "probe's cell through every change with mapPolyMesh::reverseCellMap (probes.C:487-518), "
                "which is not ported. Ported on such a mesh: `fixedLocations true`, the search repeated.");
        }

        // prepare()'s classification (probes.C:276-358): the registry's volScalarFields that `fields`
        // selects, sorted, then its volVectorFields. brae cannot list OpenFOAM's registry, so a name is
        // taken only if the loop hands that field over, and a pattern is refused.
        scalarFields_.clear();
        vectorFields_.clear();
        for (const std::string& field : dict.wordListOr("fields", {}))
        {
            if (field.find_first_of("\"*?|[]") != std::string::npos)
            {
                throw std::runtime_error(
                    who + ": `fields` has the pattern " + field + ". OpenFOAM matches it against every "
                    "field of its registry, which brae cannot list: name the fields.");
            }
            if (run_.fields->volScalar(field))
            {
                scalarFields_.push_back(field);
            }
            else if (run_.fields->volVector(field))
            {
                vectorFields_.push_back(field);
            }
            else
            {
                std::string held;
                for (const std::string& n : run_.fields->names())
                {
                    held += (held.empty() ? "" : ", ") + n;
                }
                throw std::runtime_error(
                    who + " samples `" + field + "`, which this solver does not hand a function object. "
                    "Handed over: " + held + ". (OpenFOAM samples every volume and surface field of its "
                    "registry by that name, and none if there is none.)");
            }
        }
        std::sort(scalarFields_.begin(), scalarFields_.end());
        std::sort(vectorFields_.begin(), vectorFields_.end());

        findElements();
        return true;
    }

    bool execute() override
    {
        if (onExecute_)
        {
            performAction(false);
        }
        return true;
    }

    bool write() override
    {
        performAction(true);
        return true;
    }

    void movePoints() override
    {
        // BRAE_CONTROL_PROBE_SEARCH=once: a gate's control, the cells of the start mesh kept
        if (fixedLocations_ && !PolyMeshTetSearch::control("once"))
        {
            findElements();
        }
    }

    void updateMesh() override
    {
        if (fixedLocations_)
        {
            // polyMesh::updateMesh drops the base points with the old faces (polyMeshUpdate.C:54)
            search_ = PolyMeshTetSearch();
            findElements();
        }
    }

    const std::vector<label>& elements() const
    {
        return elementList_;
    }

private:
    // doubleScalar.H:60
    static constexpr scalar kVGreat = 1.0e+300;

    std::string number(scalar v) const
    {
        std::ostringstream o;
        o.precision(run_.writePrecision);
        o << v;
        return o.str();
    }

    std::string number(const vector& v) const
    {
        return "(" + number(v.x) + " " + number(v.y) + " " + number(v.z) + ")";
    }

    // os << setw(width) << s on a stream with ios_base::left: padded on the right, never cut
    static std::string padded(
        const std::string& s,
        std::size_t width)
    {
        return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
    }

    void findElements()
    {
        const PrimitiveMesh& m = *run_.mesh;
        const std::vector<vector>& C = *run_.cellCentres;
        if (!search_.built())
        {
            for (const std::string& dir : {std::string("constant"), run_.startTimeName})
            {
                const std::string file = run_.casePath + "/" + dir + "/polyMesh/tetBasePtIs";
                std::error_code ec;
                if (std::filesystem::exists(file, ec) || std::filesystem::exists(file + ".gz", ec))
                {
                    throw std::runtime_error(
                        "brae: function object `" + name_ + "` (probes): the mesh has a `tetBasePtIs` "
                        "file (" + file + "). OpenFOAM then fans every face into tets from the file's base "
                        "points (polyMesh.C:153), where brae computes them: not ported.");
                }
            }
            search_.build(m, C);
        }
        elementList_ = search_.findCells(m, C, locations_, "function object `" + name_ + "` (probes)");
        for (std::size_t i = 0; i < locations_.size(); ++i)
        {
            if (elementList_[i] < 0)
            {
                // probes.C:224-231, a warning there too
                std::printf("  probes `%s`: did not find location %s in any cell. Skipping location.\n",
                            name_.c_str(), number(locations_[i]).c_str());
            }
        }
    }

    // sample() of a volume field (probesTemplates.C:189-232): the cell's value, -VGREAT with no cell
    template <class Type>
    std::vector<Type> sample(
        const std::string& field,
        const Type& unset) const
    {
        std::vector<label> cells;
        for (const label celli : elementList_)
        {
            if (celli >= 0)
            {
                cells.push_back(celli);
            }
        }
        std::vector<Type> held;
        if (!cells.empty())
        {
            run_.fields->cellValues(field, cells, held);
        }
        std::vector<Type> values(elementList_.size(), unset);
        std::size_t k = 0;
        for (std::size_t i = 0; i < elementList_.size(); ++i)
        {
            if (elementList_[i] >= 0)
            {
                values[i] = held[k++];
            }
        }
        return values;
    }

    // storeResults (probesTemplates.C:100-121): MinMax and average over EVERY probe's value, a probe with
    // no cell included with its -VGREAT
    void storeResults(
        const std::string& field,
        const std::vector<scalar>& values) const
    {
        // MinMax<T>() starts at (pTraits<T>::max, pTraits<T>::min) (MinMaxI.H:54-57)
        scalar lo = scalar(1.0e+300);
        scalar hi = scalar(-1.0e+300);
        scalar sum = 0;
        for (const scalar v : values)
        {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            sum += v;
        }
        // average(f) = sum(f)/f.size() (FieldFunctions.C:572-584)
        const scalar avg = sum/scalar(values.size());
        setResult("scalar", "average(" + field + ")", FunctionObjectProperties::token(avg));
        setResult("scalar", "min(" + field + ")", FunctionObjectProperties::token(lo));
        setResult("scalar", "max(" + field + ")", FunctionObjectProperties::token(hi));
        setResult("label", "size(" + field + ")", std::to_string(values.size()));
        if (verbose_)
        {
            std::printf("%s : %s\n    avg: %s\n    min: %s\n    max: %s\n\n", name_.c_str(), field.c_str(),
                        number(avg).c_str(), number(lo).c_str(), number(hi).c_str());
        }
    }

    // ...and of vectors: min and max are by component (VectorSpaceI.H:648-669)
    void storeResults(
        const std::string& field,
        const std::vector<vector>& values) const
    {
        vector lo{1.0e+300, 1.0e+300, 1.0e+300};
        vector hi{-1.0e+300, -1.0e+300, -1.0e+300};
        vector sum{0, 0, 0};
        for (const vector& v : values)
        {
            lo = vector{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
            hi = vector{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
            sum += v;
        }
        const vector avg = sum/scalar(values.size());
        setResult("vector", "average(" + field + ")", FunctionObjectProperties::token(avg));
        setResult("vector", "min(" + field + ")", FunctionObjectProperties::token(lo));
        setResult("vector", "max(" + field + ")", FunctionObjectProperties::token(hi));
        setResult("label", "size(" + field + ")", std::to_string(values.size()));
        if (verbose_)
        {
            std::printf("%s : %s\n    avg: %s\n    min: %s\n    max: %s\n\n", name_.c_str(), field.c_str(),
                        number(avg).c_str(), number(lo).c_str(), number(hi).c_str());
        }
    }

    // createProbeFiles (probes.C:55-155), one field's
    std::ofstream& file(const std::string& field)
    {
        const auto it = files_.find(field);
        if (it != files_.end())
        {
            return *it->second;
        }
        // "Use startTime as the instance for output files"; regionName() is empty for the one region
        const std::string dir = run_.casePath + "/postProcessing/" + name_ + "/" + run_.startTimeName;
        std::filesystem::create_directories(dir);
        std::unique_ptr<std::ofstream> os(new std::ofstream(dir + "/" + field, std::ios::binary));
        if (!*os)
        {
            throw std::runtime_error("brae: function object `" + name_ + "` cannot open " + dir + "/" + field);
        }
        const std::size_t width = static_cast<std::size_t>(run_.writePrecision + 7);
        for (std::size_t probei = 0; probei < locations_.size(); ++probei)
        {
            *os << "# Probe " << probei << ' ' << number(locations_[probei]);
            if (elementList_[probei] < 0)
            {
                *os << "  # Not Found";
            }
            *os << '\n';
        }
        *os << padded("# Time", width);
        for (std::size_t probei = 0; probei < locations_.size(); ++probei)
        {
            if (includeOutOfBounds_ || elementList_[probei] >= 0)
            {
                *os << ' ' << padded(std::to_string(probei), width);
            }
        }
        *os << '\n';
        os->flush();
        return *files_.emplace(field, std::move(os)).first->second;
    }

    // writeValues (probesTemplates.C:127-154)
    template <class Type>
    void writeValues(
        const std::string& field,
        const std::vector<Type>& values)
    {
        const std::size_t width = static_cast<std::size_t>(run_.writePrecision + 7);
        std::ofstream& os = file(field);
        os << padded(number(run_.time.value), width);
        for (std::size_t probei = 0; probei < values.size(); ++probei)
        {
            if (includeOutOfBounds_ || elementList_[probei] >= 0)
            {
                os << ' ' << padded(number(values[probei]), width);
            }
        }
        os << '\n';
        os.flush();
    }

    // performAction (probes.C:432-449, probesTemplates.C:158-182)
    void performAction(bool writeRow)
    {
        if (locations_.empty() || (scalarFields_.empty() && vectorFields_.empty()))
        {
            return;
        }
        if (writeRow)
        {
            // prepare(ACTION_WRITE) opens every field's file before the first is sampled (probes.C:351-354)
            for (const std::string& field : scalarFields_)
            {
                file(field);
            }
            for (const std::string& field : vectorFields_)
            {
                file(field);
            }
        }
        for (const std::string& field : scalarFields_)
        {
            const std::vector<scalar> values = sample<scalar>(field, -kVGreat);
            storeResults(field, values);
            if (writeRow)
            {
                writeValues(field, values);
            }
        }
        for (const std::string& field : vectorFields_)
        {
            std::size_t notFound = 0;
            std::size_t firstNotFound = 0;
            for (std::size_t i = 0; i < elementList_.size(); ++i)
            {
                if (elementList_[i] < 0)
                {
                    firstNotFound = notFound == 0 ? i : firstNotFound;
                    ++notFound;
                }
            }
            if (notFound > 0)
            {
                // OBSERVED, OpenFOAM v2412 on laminar/damBreak with U and a location at (5 5 5): the scalar
                // fields' first rows are written, then the run stops in storeResults with "ill defined
                // primitiveEntry starting at keyword 'min(U)'" (primitiveEntryIO.C:243) -- 'average(U)' when
                // NO location has a cell, the average then being the vector of -VGREAT itself
                const std::string keyword =
                    (notFound == elementList_.size() ? "average(" : "min(") + field + ")";
                throw std::runtime_error(
                    "brae: function object `" + name_ + "` (probes) samples the vector field `" + field
                    + "` with the location " + number(locations_[firstNotFound]) + ", which no cell holds. "
                    "OpenFOAM stops there too, at its first sample: a result of the row that is a vector of "
                    "-VGREAT is not an entry its state dictionary can hold (`ill defined primitiveEntry "
                    "starting at keyword '" + keyword + "'`). Remove the location or the field.");
            }
            const std::vector<vector> values = sample<vector>(field, vector{-kVGreat, -kVGreat, -kVGreat});
            storeResults(field, values);
            if (writeRow)
            {
                writeValues(field, values);
            }
        }
    }

    std::vector<vector> locations_;
    std::vector<std::string> scalarFields_;
    std::vector<std::string> vectorFields_;
    bool fixedLocations_ = true;
    bool includeOutOfBounds_ = true;
    bool verbose_ = false;
    bool onExecute_ = false;
    PolyMeshTetSearch search_;
    std::vector<label> elementList_;
    std::map<std::string, std::unique_ptr<std::ofstream>> files_;
};

}   // namespace functionObjects
}   // namespace brae
