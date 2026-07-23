#pragma once
// Read an MRF case from files: constant/polyMesh/cellZones + constant/MRFProperties, and OF's
// correctBoundaryVelocity (in-zone moving walls take the frame velocity Omega x r).
#include "cf_types.cuh"
#include "foam_token_reader.cuh"
#include "foam_dict.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "mrf_zone.cuh"
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace brae {

// cellZone name -> cell labels (constant/polyMesh/cellZones). Empty if the file is absent.
inline std::map<std::string, std::vector<label>> readCellZones(const std::string& polyMeshDir)
{
    std::map<std::string, std::vector<label>> zones;
    const std::string path = polyMeshDir + "/cellZones";
    {
        std::ifstream f(path);
        std::ifstream g(path + ".gz");
        if (!f.good() && !g.good()) return zones;
    }
    // A binary polyMesh writes each zone's cellLabels list as a raw blob the ASCII TokenStream
    // cannot tokenize (it mangles the bytes -> "read past end"). Route those to the byte-level reader.
    if (foamFormat(path) == "binary")
    {
        for (auto& zp : readBinaryCellZones(path))
            zones[zp.first] = std::move(zp.second);
        return zones;
    }
    TokenStream ts(path);
    const label nz = ts.nextLabel();
    ts.expect("(");
    for (label z = 0; z < nz; ++z)
    {
        const std::string name = ts.next();
        ts.expect("{");
        std::vector<label> cells;
        while (ts.peek() != "}")
        {
            const std::string key = ts.next();
            if (key == "cellLabels")
            {
                ts.next();                              // "List<label>"
                const label n = ts.nextLabel();
                ts.expect("(");
                cells.resize(n);
                for (label i = 0; i < n; ++i)
                    cells[i] = ts.nextLabel();
                ts.expect(")");
                ts.expect(";");
            }
            else   // skip type etc.
            {
                while (ts.peek() != ";")
                    ts.next();
                ts.expect(";");
            }
        }
        ts.expect("}");
        zones[name] = std::move(cells);
    }
    return zones;
}

struct MRFConfig
{
    bool active = false;
    std::string cellZone;
    scalar omega = 0;
    vector axis{0, 0, 1}, origin{0, 0, 0};
    std::vector<std::string> nonRotatingPatches;
};

// First active MRF entry in constant/MRFProperties. active=false if absent/inactive.
inline MRFConfig readMRFProperties(const std::string& constantDir)
{
    MRFConfig c;
    {
        std::ifstream f(constantDir + "/MRFProperties");
        if (!f.good()) return c;
    }
    const FoamDict d = readDict(constantDir + "/MRFProperties");
    auto v3 = [](const std::vector<scalar>& a, vector dflt) { return a.size() >= 3 ? vector{a[0], a[1], a[2]} : dflt; };
    for (const auto& s : d.subs)
    {
        const FoamDict& mrf = s.second;
        const std::string act = mrf.wordOr("active", "yes");
        if (!(act == "yes" || act == "true" || act == "on" || act == "1")) continue;
        c.active = true;
        c.cellZone = mrf.wordOr("cellZone", "");
        c.omega = mrf.scalarOr("omega", 0.0);
        c.axis = v3(mrf.scalarListOr("axis", {}), vector{0, 0, 1});
        c.origin = v3(mrf.scalarListOr("origin", {}), vector{0, 0, 0});
        c.nonRotatingPatches = mrf.wordListOr("nonRotatingPatches", {});
        break;
    }
    return c;
}

// OF MRF.correctBoundaryVelocity: in-zone, velocity-fixing patches that are NOT nonRotating take the absolute
// frame velocity U = Omega x (Cf - origin) (e.g. a noSlip wall rotating with the frame). Leaves nonRotating
// patches (stationary in the absolute frame) and non-fixed patches untouched.
inline void mrfCorrectBoundaryVelocity(
    GeometricField<vector>& U,
    const MRFZone& z,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::vector<std::string>& nonRotating)
{
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& p = fvp[pi];
        if (p.type == "empty" || p.size == 0 || !U.boundary[pi]->fixesValue()) continue;
        bool skip = false;
        for (const auto& nm : nonRotating)
            if (p.name == nm)
            {
                skip = true;
                break;
            }
        if (skip) continue;
        bool inZone = true;
        for (label i = 0; i < p.size; ++i)
            if (!z.inZone[p.faceCells[i]])
            {
                inZone = false;
                break;
            }
        if (!inZone) continue;
        std::vector<vector> wv(p.size);
        for (label i = 0; i < p.size; ++i)
            wv[i] = cross(z.Omega, g.Cf()[p.start + i] - z.origin);
        U.boundary[pi] = std::make_unique<FixedValuePatchField<vector>>(p, false, vector{0, 0, 0}, wv);
    }
    U.evaluateBoundary();
}

} // namespace brae
