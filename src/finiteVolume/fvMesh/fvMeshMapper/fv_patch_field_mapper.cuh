#pragma once
// OpenFOAM's fvPatchFieldMapper -- the addressing a field is mapped through when the mesh under it
// changes -- and Field<Type>::autoMap, which is what every patch field's own autoMap starts with.
//
// provenance:
//   openfoam: src/OpenFOAM/lnInclude/faceMapper.C, src/finiteVolume/lnInclude/fvSurfaceMapper.C and
//             fvPatchMapper.C build the addressing; src/OpenFOAM/fields/Fields/Field/Field.C:372-399
//             (the direct map) and :456-470 (the weighted one) apply it
//   brae:     the addressing is BUILT in src/dynamicFvMesh/dynamicRefineFvMesh (unit 7b-2, gated in
//             tests/refine_update_vs_openfoam.sh); this header is only the type it is built into and the
//             two ways of applying it, so that a patch field can be mapped without depending on the
//             dynamic mesh.
//
// WHY IT LIVES HERE. `fvPatchField::autoMap` needs this type, and fv_patch_field.cuh is below the
// dynamic mesh in the include order -- so the mapping is declared at the level that USES it and filled in
// by the level that computes it, exactly as OpenFOAM separates fvPatchFieldMapper from mapPolyMesh.
//
// THREE STATES PER FACE, and the third is the one that matters: a face can map DIRECTLY from one old
// face, INTERPOLATIVELY from several with weights, or not be mapped at all -- a direct entry of -1, or an
// empty interpolative addressing. fvPatchMapper produces the last one for a face whose old face was in a
// DIFFERENT patch (fvPatchMapper.C:106-112), and `hasUnmapped` is what tells a patch field to fill those
// from its own internal field instead.
#include "cf_types.cuh"
#include <vector>

namespace brae {

struct FvPatchFieldMapping
{
    bool                             direct = false;
    // direct: one old index per new face, or -1 for an unmapped one
    std::vector<label>               directAddressing;
    // interpolative: the old indices and their weights, empty for an unmapped face
    std::vector<std::vector<label>>  addressing;
    std::vector<std::vector<scalar>> weights;
    // the faces that came from nowhere at all (a new face with no old face), for the record
    std::vector<label>               insertedFaces;

    label size() const
    {
        return static_cast<label>(direct ? directAddressing.size() : addressing.size());
    }

    // fvPatchMapper::hasUnmapped: is there a face with no source? Derived rather than stored, so it
    // cannot disagree with the addressing it describes.
    bool hasUnmapped() const
    {
        if (direct)
        {
            for (const label a : directAddressing) { if (a < 0) return true; }
            return false;
        }
        for (const std::vector<label>& a : addressing) { if (a.empty()) return true; }
        return false;
    }
};

// Field<Type>::autoMap's own work, WITHOUT the patch field's unmapped-face fill: a direct entry below
// zero leaves the value alone (Field.C:386-396), and an interpolative sum accumulates from ZERO in the
// addressing order.
template <typename T>
void mapFieldThrough(
    std::vector<T>&                  f,
    const FvPatchFieldMapping&       pm,
    const T&                         zero)
{
    const std::vector<T> old = f;
    if (pm.direct)
    {
        f.assign(pm.directAddressing.size(), zero);
        if (old.empty()) return;
        for (std::size_t i = 0; i < f.size(); ++i)
        {
            const label a = pm.directAddressing[i];
            if (a >= 0 && static_cast<std::size_t>(a) < old.size()) f[i] = old[static_cast<std::size_t>(a)];
        }
        return;
    }
    f.assign(pm.addressing.size(), zero);
    if (old.empty()) return;
    for (std::size_t i = 0; i < f.size(); ++i)
    {
        T v = zero;
        for (std::size_t j = 0; j < pm.addressing[i].size(); ++j)
        {
            v = v + pm.weights[i][j]*old[static_cast<std::size_t>(pm.addressing[i][j])];
        }
        f[i] = v;
    }
}

} // namespace brae
