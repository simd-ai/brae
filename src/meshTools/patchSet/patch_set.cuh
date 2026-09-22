#pragma once
// polyBoundaryMesh::patchSet(patchNames): the patches a list of names selects, each entry a literal name, a
// group name or a regular expression over either (usePatchGroups = true, the default).
//
// provenance:
//   openfoam: src/OpenFOAM/meshes/polyMesh/polyBoundaryMesh/polyBoundaryMesh.C (patchSet)
//   used by:  the inverseDistance diffusivity (displacement_laplacian_fv_motion_solver_cpp.cu) and, through
//             the wallDist it registers, interFoam's kOmegaSST (inter_case_cpp.cu)
#include "cf_types.cuh"
#include "foam_dict.cuh"   // compileFoamRegex
#include "fv_patch.cuh"
#include <regex>
#include <string>
#include <vector>

namespace brae {

inline bool patchSetMatches(
    const FvPatch& p,
    const std::string& key)
{
    if (key == p.name) return true;
    for (const std::string& grp : p.inGroups)
    {
        if (key == grp) return true;
    }
    try
    {
        const std::regex re = compileFoamRegex(key);
        if (std::regex_match(p.name, re)) return true;
        for (const std::string& grp : p.inGroups)
        {
            if (std::regex_match(grp, re)) return true;
        }
    }
    catch (...)
    {
    }
    return false;
}

// the selected patch indices, in the mesh's patch order
inline std::vector<label> patchSet(
    const std::vector<FvPatch>& patches,
    const std::vector<std::string>& keys)
{
    std::vector<label> ids;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (const std::string& key : keys)
        {
            if (patchSetMatches(patches[pi], key))
            {
                ids.push_back(static_cast<label>(pi));
                break;
            }
        }
    }
    return ids;
}

} // namespace brae
