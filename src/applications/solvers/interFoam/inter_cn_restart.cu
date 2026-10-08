// inter_cn_restart.cu -- see inter_cn_restart.cuh for the two facts this implements.
#include "inter_cn_restart.cuh"

#include "foam_dict.cuh"           // isCoupledInterfaceType
#include "foam_field_reader.cuh"   // FieldData / readField
#include "read_surface_field.cuh"

#include <filesystem>
#include <stdexcept>

namespace brae {

namespace {

constexpr const char* WHO = "brae interFoam CrankNicolson restart: ";

// brae's clock counts steps from 1, so the run's start index is 0 and OpenFOAM's -2 is -2 here.
// See fact (1) in the header.
constexpr label kStartTimeIndex = 0;
constexpr label kRestartStartTimeIndex = -2;

// One field of the start directory, as its cells (or internal faces) and its patch values. False and
// nothing touched when the file is not there.
template <typename T>
bool readCellsAndPatches(
    const InterCnRestart& r,
    const std::string& name,
    std::size_t nInternal,
    const std::vector<FvPatch>& patches,
    std::vector<T>& internalOut,
    std::vector<std::vector<T>>& boundaryOut)
{
    if (r.dir.empty() || name.empty()) return false;
    const std::string path = r.dir + "/" + name;
    if (!std::filesystem::exists(path)) return false;

    const FieldData<T> fd = readField<T>(path);
    if (fd.internalUniform)
    {
        internalOut.assign(nInternal, fd.internalUniformValue);
    }
    else if (fd.internalField.size() != nInternal)
    {
        throw std::runtime_error(
            std::string(WHO) + path + " has " + std::to_string(fd.internalField.size())
            + " internal values and the mesh wants " + std::to_string(nInternal)
            + " (a decomposed directory, or the wrong mesh).");
    }
    else
    {
        internalOut = fd.internalField;
    }

    // Every patch of the field, by name. A ddt0 field is written `calculated` with an explicit value
    // list, and a constraint patch (empty, wedge, symmetry) carries no value at all -- those stay the
    // zeros lookupOrCreate would have left, which is what the scheme reads there.
    boundaryOut.assign(patches.size(), std::vector<T>());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        boundaryOut[pi].assign(static_cast<std::size_t>(patches[pi].size), T{});
        for (const PatchFieldData<T>& b : fd.boundary)
        {
            if (b.name != patches[pi].name || !b.hasValue) continue;
            if (b.valueUniform)
            {
                std::fill(boundaryOut[pi].begin(), boundaryOut[pi].end(), b.uniformValue);
            }
            else if (b.values.size() == boundaryOut[pi].size())
            {
                boundaryOut[pi] = b.values;
            }
            else
            {
                throw std::runtime_error(
                    std::string(WHO) + path + " patch `" + b.name + "` has "
                    + std::to_string(b.values.size()) + " values and the mesh patch has "
                    + std::to_string(boundaryOut[pi].size()) + ".");
            }
            break;
        }
    }
    return true;
}

}   // namespace


bool seedCnDdt0(
    cpu::fv::CrankNicolsonDdt0<scalar>& ddt0,
    const InterCnRestart& r,
    std::size_t nInternal,
    const std::vector<FvPatch>& patches)
{
    if (!readCellsAndPatches(r, ddt0.name, nInternal, patches, ddt0.internal, ddt0.boundary)) return false;
    ddt0.startTimeIndex = kRestartStartTimeIndex;
    ddt0.timeIndex = kStartTimeIndex;
    ddt0.exists = true;
    return true;
}


bool seedCnDdt0(
    cpu::fv::CrankNicolsonDdt0<vector>& ddt0,
    const InterCnRestart& r,
    std::size_t nInternal,
    const std::vector<FvPatch>& patches)
{
    if (!readCellsAndPatches(r, ddt0.name, nInternal, patches, ddt0.internal, ddt0.boundary)) return false;
    ddt0.startTimeIndex = kRestartStartTimeIndex;
    ddt0.timeIndex = kStartTimeIndex;
    ddt0.exists = true;
    return true;
}


bool readCnOldOld(
    const InterCnRestart& r,
    const std::string& name,
    std::size_t nCells,
    std::vector<scalar>& cells)
{
    std::vector<std::vector<scalar>> unusedPatches;
    return readCellsAndPatches<scalar>(r, name, nCells, {}, cells, unusedPatches);
}


bool readCnOldOld(
    const InterCnRestart& r,
    const std::string& name,
    std::size_t nCells,
    const std::vector<FvPatch>& patches,
    std::vector<vector>& cells,
    std::vector<std::vector<vector>>& patchValues)
{
    return readCellsAndPatches<vector>(r, name, nCells, patches, cells, patchValues);
}


bool readCnOldOldSurface(
    const InterCnRestart& r,
    const std::string& name,
    label nInternalFaces,
    const std::vector<FvPatch>& patches,
    SurfaceScalarField& out)
{
    if (r.dir.empty()) return false;
    const std::string path = r.dir + "/" + name;
    if (!std::filesystem::exists(path)) return false;
    out = readSurfaceField(path, patches, nInternalFaces);
    return true;
}


bool seedCnDdt0(
    DeviceCnDdt0& ddt0,
    const InterCnRestart& r,
    int nComp,
    std::size_t nInternal,
    std::size_t nBoundary,
    const std::vector<FvPatch>& patches)
{
    if (nComp != 1 && nComp != 3)
        throw std::runtime_error(std::string(WHO) + "a ddt0 field has 1 or 3 components.");

    // Read into the host shapes first, then hand each component its own device buffer -- the device
    // field is component-major where the file is value-major.
    std::vector<std::vector<scalar>> comp(static_cast<std::size_t>(nComp));
    std::vector<std::vector<scalar>> bnd(static_cast<std::size_t>(nComp));
    if (nComp == 1)
    {
        cpu::fv::CrankNicolsonDdt0<scalar> host;
        host.name = ddt0.name;
        if (!seedCnDdt0(host, r, nInternal, patches)) return false;
        comp[0] = host.internal;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (isCoupledInterfaceType(patches[pi].type)) continue;
            bnd[0].insert(bnd[0].end(), host.boundary[pi].begin(), host.boundary[pi].end());
        }
    }
    else
    {
        cpu::fv::CrankNicolsonDdt0<vector> host;
        host.name = ddt0.name;
        if (!seedCnDdt0(host, r, nInternal, patches)) return false;
        for (int k = 0; k < 3; ++k) comp[static_cast<std::size_t>(k)].reserve(nInternal);
        for (const vector& v : host.internal)
        {
            comp[0].push_back(v.x); comp[1].push_back(v.y); comp[2].push_back(v.z);
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (isCoupledInterfaceType(patches[pi].type)) continue;
            for (const vector& v : host.boundary[pi])
            {
                bnd[0].push_back(v.x); bnd[1].push_back(v.y); bnd[2].push_back(v.z);
            }
        }
    }

    // ...and the boundary only where the consumer keeps one -- see the header
    if (nBoundary == 0)
    {
        for (std::size_t k = 0; k < bnd.size(); ++k) bnd[k].clear();
    }
    else if (bnd[0].size() != nBoundary)
    {
        throw std::runtime_error(
            std::string(WHO) + ddt0.name + " has " + std::to_string(bnd[0].size())
            + " non-coupled boundary values and the device field wants " + std::to_string(nBoundary) + ".");
    }

    ddt0.nComp = nComp;
    for (int k = 0; k < nComp; ++k)
    {
        ddt0.internal[k].copyFrom(comp[static_cast<std::size_t>(k)]);
        ddt0.boundary[k].copyFrom(bnd[static_cast<std::size_t>(k)]);
    }
    ddt0.startTimeIndex = kRestartStartTimeIndex;
    ddt0.timeIndex = kStartTimeIndex;
    ddt0.exists = true;
    return true;
}

}   // namespace brae
