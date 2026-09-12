#pragma once
// patch_entry_lookup.cuh -- find the boundaryField entry that applies to a mesh patch, OF's way.
//
// OpenFOAM resolves a boundaryField key against a patch in three steps: exact name, then literal group
// membership, then REGEX against the patch name or any of its group names, with the LAST matching pattern
// winning. Cases lean on this constantly -- `"(?i).*walls"`, `".*"`, `"(inlet|outlet)"`.
//
// buildField (geometric_field.cuh) has always done this correctly. Several other places did NOT: they
// compared `entry.name == patch.name` and silently fell back to a default when a case used a regex key.
// The result was a plausible run with the wrong coefficient:
//
//   - the per-face Prt for alphatWallFunction reverted to the model default 1.0 instead of the wall
//     function's 0.85, i.e. wall alphat and the wall heat flux ~15% low (squareBend* key their alphat
//     wall entry as "(?i).*walls" while the mesh patch is literally `walls`);
//   - turbulent inlet BCs kept their written `value` placeholder instead of the computed inlet value.
//
// Same rule, one implementation, so a regex-keyed case cannot resolve differently depending on which
// piece of code is asking.

#include "foam_dict.cuh"   // compileFoamRegex ((?i) flag support)
#include "fv_patch.cuh"
#include <regex>
#include <string>
#include <vector>

namespace brae {

// The entry-direction companion of findPatchEntry below: every mesh patch whose OWN resolution lands
// on `e`. An entry's key may be a regex or a group name covering several patches, so "the patch this
// entry is about" is only answerable as a set -- and answerable only by running the patch-direction
// resolution, or exact-name entries would shadow differently than OF's pass order. The guards that
// used to compare `entry.name == patch.name` returned nothing for a regex key and silently skipped it
// (audit finding #16); this is what they resolve through instead.
template <typename Entry, typename Patch>
inline const Entry* findPatchEntry(const std::vector<Entry>& entries, const Patch& p);

template <typename Entry, typename Patch>
inline std::vector<const Patch*> patchesResolvingTo(
    const std::vector<Entry>& entries,
    const Entry&              e,
    const std::vector<Patch>& patches)
{
    std::vector<const Patch*> out;
    for (const Patch& p : patches)
        if (findPatchEntry(entries, p) == &e) out.push_back(&p);
    return out;
}

// The entry that applies to `p`, or nullptr. Entries must expose `.name`; any PatchFieldData<T> does.
template <typename Entry, typename Patch>
inline const Entry* findPatchEntry(const std::vector<Entry>& entries, const Patch& p)
{
    for (const Entry& b : entries)                       // 1. exact name wins outright
        if (b.name == p.name) return &b;

    const Entry* hit = nullptr;                          // 2./3. group, then regex; LAST match wins (OF)
    for (const Entry& b : entries)
    {
        bool match = false;
        for (const std::string& g : p.inGroups)
            if (b.name == g) { match = true; break; }
        if (!match)
        {
            try
            {
                const std::regex re = compileFoamRegex(b.name);
                if (std::regex_match(p.name, re)) match = true;
                else
                    for (const std::string& g : p.inGroups)
                        if (std::regex_match(g, re)) { match = true; break; }
            }
            catch (const std::regex_error&) {}
        }
        if (match) hit = &b;
    }
    return hit;
}

}   // namespace brae
