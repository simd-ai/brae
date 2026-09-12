#include "MRF_cpp.cuh"
#include "foam_dict.cuh"
#include "mrf_read.cuh"      // readCellZones: the polyMesh cellZones parser (ASCII + binary)

#include <cmath>
#include <fstream>

namespace brae {
namespace cpu {
namespace MRF {

namespace {

bool isOn(const std::string& s)
{
    return s == "yes" || s == "true" || s == "on" || s == "1";
}

vector asVector(const std::vector<scalar>& a, const vector& dflt)
{
    if (a.size() < 3) return dflt;
    return vector{a[0], a[1], a[2]};
}

// A patch whose faces do NOT move with the frame: OpenFOAM tests pp.coupled() or membership of
// excludedPatchLabels_ (which is what nonRotatingPatches resolves to).
bool isExcludedPatch(const FvPatch& p, const std::vector<std::string>& nonRotating)
{
    if (isCoupledInterfaceType(p.type) || p.type == "processor") return true;
    for (const std::string& n : nonRotating)
    {
        if (n == p.name) return true;
    }
    return false;
}

} // namespace

std::vector<ZoneSpec> readMRFProperties(const std::string& constantDir)
{
    std::vector<ZoneSpec> out;
    {
        std::ifstream probe(constantDir + "/MRFProperties");
        if (!probe.good()) return out;
    }
    const FoamDict d = readDict(constantDir + "/MRFProperties");
    for (const auto& s : d.subs)
    {
        const FoamDict& mrf = s.second;
        ZoneSpec z;
        z.active = isOn(mrf.wordOr("active", "yes"));
        if (!z.active) continue;
        z.cellZone = mrf.wordOr("cellZone", "");
        z.omega    = mrf.scalarOr("omega", 0.0);
        z.axis     = asVector(mrf.scalarListOr("axis", {}), vector{0, 0, 1});
        z.origin   = asVector(mrf.scalarListOr("origin", {}), vector{0, 0, 0});
        z.nonRotatingPatches = mrf.wordListOr("nonRotatingPatches", {});
        out.push_back(z);
    }
    return out;
}

Zone buildZone(
    const ZoneSpec&             spec,
    const std::vector<label>&   zoneCells,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches)
{
    Zone z;
    z.active = spec.active;
    z.origin = spec.origin;

    const scalar am = mag(spec.axis);
    z.Omega = (am > 0.0) ? (spec.omega * (spec.axis / am)) : vector{0, 0, 0};

    z.cells = zoneCells;
    z.inZone.assign(m.nCells(), false);
    for (label c : zoneCells)
    {
        z.inZone[c] = true;
    }

    // faceType: 0 not in zone, 1 moving with the frame, 2 coupled/nonRotating.
    //
    // EITHER side in the zone, not both. The faces between a zone cell and a non-zone cell are the
    // zone's interface, and they are exactly where the frame flux has to be removed.
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    for (label f = 0; f < m.nInternalFaces(); ++f)
    {
        if (z.inZone[own[f]] || z.inZone[nei[f]])
        {
            z.internalFaces.push_back(f);
        }
    }

    z.includedFaces.resize(patches.size());
    z.excludedFaces.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        const bool excluded = isExcludedPatch(p, spec.nonRotatingPatches);

        // An `empty` patch is neither: OpenFOAM skips it entirely, so its faces stay type 0.
        if (!excluded && p.type == "empty") continue;

        for (label i = 0; i < p.size; ++i)
        {
            if (!z.inZone[p.faceCells[i]]) continue;
            if (excluded)
            {
                z.excludedFaces[pi].push_back(i);
            }
            else
            {
                z.includedFaces[pi].push_back(i);
            }
        }
    }
    return z;
}

void correctBoundaryVelocity(
    GeometricField<vector>&     U,
    const std::vector<Zone>&    zones,
    const std::vector<FvPatch>& patches)
{
    for (const Zone& z : zones)
    {
        if (!z.active) continue;
        for (std::size_t pi = 0; pi < patches.size() && pi < z.includedFaces.size(); ++pi)
        {
            if (z.includedFaces[pi].empty()) continue;

            std::vector<vector> pf = U.boundary[pi]->value();
            for (label i : z.includedFaces[pi])
            {
                pf[i] = cross(z.Omega, patches[pi].Cf[i] - z.origin);
            }
            U.boundary[pi]->setValue(pf);
        }
    }
}

void addCoriolis(
    const std::vector<Zone>&    zones,
    const std::vector<vector>&  U,
    const std::vector<scalar>&  V,
    std::vector<vector>&        source)
{
    for (const Zone& z : zones)
    {
        if (!z.active) continue;
        for (label c : z.cells)
        {
            source[c] = source[c] - V[c] * cross(z.Omega, U[c]);
        }
    }
}

void makeRelative(
    SurfaceScalarField&         phi,
    const std::vector<Zone>&    zones,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    for (const Zone& z : zones)
    {
        if (!z.active) continue;

        for (label f : z.internalFaces)
        {
            phi.internal[f] -= dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]);
        }

        // Included faces move WITH the frame, so their relative flux is zero outright -- not the
        // subtraction the internal faces take.
        for (std::size_t pi = 0; pi < patches.size() && pi < z.includedFaces.size(); ++pi)
        {
            for (label i : z.includedFaces[pi])
            {
                phi.boundary[pi][i] = 0.0;
            }
        }
        for (std::size_t pi = 0; pi < patches.size() && pi < z.excludedFaces.size(); ++pi)
        {
            for (label i : z.excludedFaces[pi])
            {
                const label f = patches[pi].start + i;
                phi.boundary[pi][i] -= dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]);
            }
        }
    }
}

} // namespace MRF
} // namespace cpu
} // namespace brae
