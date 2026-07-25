#pragma once
// brae::GeometricField<T>, internal field + boundary patch fields. Mirrors OpenFOAM
// GeometricField (the volField pairing of an internal field with a boundaryField).
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"   // compileFoamRegex ((?i) flag) + isConstraintPatchType
#include <memory>
#include <regex>
#include <stdexcept>
#include <vector>

namespace brae {

template <typename T>
class GeometricField
{
public:
    std::vector<T>                                  internal;
    std::vector<std::unique_ptr<fvPatchField<T>>>   boundary;   // one per mesh patch (in order)

    // OpenFOAM correctBoundaryConditions order: init all (processor patches post their
    // exchange) -> wait -> evaluate all. Safe in serial (no requests are posted).
    void evaluateBoundary()
    {
        for (auto& b : boundary)
            b->initEvaluate(internal);
        Pstream::waitAll();
        for (auto& b : boundary)
            b->evaluate(internal);
    }
};

// Build a GeometricField from parsed file data + the mesh patches (matched by name).
template <typename T>
GeometricField<T> buildField(
    const FieldData<T>& fd,
    const std::vector<FvPatch>& patches,
    label nCells)
{
    GeometricField<T> gf;
    if (fd.internalUniform) gf.internal.assign(nCells, fd.internalUniformValue);
    else if (static_cast<label>(fd.internalField.size()) != nCells)   // wrong mesh / decomposed field / hand-edit
        throw std::runtime_error("field internalField has " + std::to_string(fd.internalField.size())
            + " values but the mesh has " + std::to_string(nCells) + " cells (size mismatch) -- refusing to read"
              " out of bounds. Check the field is for THIS mesh (not a decomposed/other-mesh field).");
    else                    gf.internal = fd.internalField;

    for (const FvPatch& p : patches)
    {
        const PatchFieldData<T>* d = nullptr;
        for (const PatchFieldData<T>& b : fd.boundary)              // exact patch-NAME entry wins (GeometricBoundaryField pass 1)
            if (b.name == p.name)
            {
                d = &b;
                break;
            }
        // else group / regex (pass 2 = polyBoundaryMesh::indices(key, useGroups=true)): a LITERAL keyword selects
        // by group membership; a regex keyword matches the patch name OR a group name. (OF: last pattern wins.)
        if (!d)
            for (const PatchFieldData<T>& b : fd.boundary)
            {
                bool hit = false;
                for (const std::string& gname : p.inGroups)        // literal group name
                    if (b.name == gname)
                    {
                        hit = true;
                        break;
                    }
                if (!hit)
                {
                    try
                    {
                        const std::regex re = compileFoamRegex(b.name);
                        if (std::regex_match(p.name, re)) hit = true;                                        // patch-name regex
                        else
                            for (const std::string& gname : p.inGroups)
                                if (std::regex_match(gname, re))   // group-name regex
                                {
                                    hit = true;
                                    break;
                                }
                    }
                    catch (const std::regex_error&) {}
                }
                if (hit) d = &b;
            }
        // Constraint patches (empty/symmetry/wedge/cyclic/cyclicAMI/...) may be omitted from the field file, OF auto-
        // selects the matching constraint fvPatchField from the mesh patch type (what #includeEtc setConstraintTypes
        // defaults). Synthesise that entry here so such cases run unmodified.
        PatchFieldData<T> synth;
        if (!d && isConstraintPatchType(p.type))
        {
            synth.name = p.name;
            synth.type = p.type;
            d = &synth;
        }
        if (!d) throw std::runtime_error("brae: no boundaryField entry for patch " + p.name);
        gf.boundary.push_back(makePatchField<T>(p, *d));
    }
    return gf;
}

} // namespace brae
