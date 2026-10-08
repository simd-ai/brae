#include "MRF_cpp.cuh"
#include <stdexcept>
#include "foam_dict.cuh"
#include "mrf_read.cuh"      // readCellZones: the polyMesh cellZones parser (ASCII + binary)

#include <cmath>
#include <fstream>

namespace brae {
namespace cpu {
namespace MRF {

namespace {

// `origin` and `axis` are read with get<vector> and no default (MRFZone.C:563-564): a zone without one
// stops OpenFOAM. They defaulted to (0 0 0) and (0 0 1) here, so such a zone rotated about z.
vector requiredVector(
    const FoamDict& mrf,
    const std::string& zone,
    const char* name)
{
    const std::vector<scalar> a = mrf.scalarListOr(name, {});
    if (a.size() != 3)
    {
        throw std::runtime_error(
            "brae MRF: zone `" + zone + "` has no `" + name + "` of three components. OpenFOAM reads it "
            "with no default (MRFZone.C:563-564) and stops.");
    }
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
        // a Switch (MRFZone.C:553): read by hand, `active y;` dropped the zone
        z.active = mrf.switchOr("active", true);
        if (!z.active) continue;
        z.cellZone = mrf.wordOr("cellZone", "");
        // omega IS A Function1 (MRFZone.C, `omega_.reset(Function1<scalar>::New("omega", coeffs_,
        // &mesh_))`), mandatory, and a transient case may make it one: interFoam's mixerVessel2D writes
        // `omega constant 6.2831853;`. This took scalarOr, which reads the LAST token -- right for that
        // spelling by accident, a crash on a table's `)` and a silent ZERO for the dictionary form,
        // which is not a leaf at all. A constant is read in each of its three spellings; any other
        // type is refused by name, because a frame whose speed brae holds fixed is a different case.
        {
            const std::vector<std::string>* ov = mrf.find("omega");
            const FoamDict* od = mrf.subDict("omega");
            auto number = [&](const std::string& t)
            {
                std::size_t used = 0;
                scalar v = 0;
                try
                {
                    v = std::stod(t, &used);
                }
                catch (...)
                {
                    used = 0;
                }
                if (used != t.size() || t.empty())
                    throw std::runtime_error(
                        "brae MRF: zone `" + s.first + "` has `omega " + t + "`, which is not a number.");
                return v;
            };
            if (ov && ov->size() == 1)
            {
                z.omega = number((*ov)[0]);
            }
            else if (ov && ov->size() == 2 && (*ov)[0] == "constant")
            {
                z.omega = number((*ov)[1]);
            }
            else if (od && od->wordOr("type", "") == "constant" && od->find("value"))
            {
                z.omega = number(od->find("value")->back());
            }
            else if (ov || od)
            {
                const std::string type = ov ? (ov->empty() ? std::string("<empty>") : (*ov)[0])
                                            : od->wordOr("type", "<no type>");
                throw std::runtime_error(
                    "brae MRF: zone `" + s.first + "` gives omega as a Function1 of type `" + type
                    + "`. brae reads a constant (`omega 6.28;`, `omega constant 6.28;`, or "
                    "`omega { type constant; value 6.28; }`) and holds it for the run; a speed that "
                    "varies in time is not ported, and holding it fixed would solve another case.");
            }
            else
            {
                throw std::runtime_error(
                    "brae MRF: zone `" + s.first + "` has no `omega` entry. OpenFOAM's Function1::New "
                    "is mandatory there (MRFZone.C) and stops; brae used to run the zone at omega 0.");
            }
        }
        z.axis = requiredVector(mrf, s.first, "axis");
        z.origin = requiredVector(mrf, s.first, "origin");
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

    // EVERY nonRotatingPatches ENTRY MUST BE A PLAIN EXISTING PATCH NAME, or this refuses by name.
    // OpenFOAM's excludedPatchNames_ is a `wordRes` (MRFZone.H:92) resolved with
    // boundaryMesh().indices(matcher, useGroups = true) (MRFZone.C:578-579), so it matches REGEXES and
    // PATCH GROUPS; isExcludedPatch below compares exact strings. The two disagree in the worst direction:
    // a patch OpenFOAM matches by regex is EXCLUDED there (the frame flux is subtracted on its faces) and
    // would be INCLUDED here (its flux zeroed outright), silently. No shipped tutorial writes one -- every
    // MRF case in the tree has `nonRotatingPatches ()` or plain names -- so this is refused rather than
    // ported, and it is checked here because buildZone is what a topology change re-runs.
    for (const std::string& n : spec.nonRotatingPatches)
    {
        bool found = false;
        for (const FvPatch& p : patches)
        {
            found = found || p.name == n;
        }
        if (!found)
        {
            throw std::runtime_error(
                "brae MRF: nonRotatingPatches names `" + n + "`, which is not a patch of this mesh. "
                "OpenFOAM would match it as a wordRes -- a regex or a patch GROUP -- and exclude every "
                "patch it matched (MRFZone.H:92, MRFZone.C:578-579, polyBoundaryMesh::indices with "
                "useGroups); this port compares exact names, so such an entry would be INCLUDED here and "
                "EXCLUDED there. Refused rather than run the opposite treatment.");
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

void update(
    std::vector<Zone>&                                zones,
    const std::vector<ZoneSpec>&                      specs,
    const std::map<std::string, std::vector<label>>&  cellZones,
    const PrimitiveMesh&                              m,
    const std::vector<FvPatch>&                       patches)
{
    // The caller keeps one spec per zone, in build order, and OpenFOAM's own update walks the list the
    // same way. A mismatch means a caller built the zones from somewhere else, and rebuilding the wrong
    // spec onto a zone would be silent -- the counts are the only thing that can say so.
    if (zones.size() != specs.size())
    {
        throw std::runtime_error(
            "brae MRF::update: " + std::to_string(zones.size()) + " zone(s) against "
            + std::to_string(specs.size()) + " spec(s). Each zone is rebuilt from the spec it was built "
            "from, so the two lists must stay parallel.");
    }
    for (std::size_t i = 0; i < zones.size(); ++i)
    {
        // OpenFOAM keeps cellZoneID_ and indexes the live cellZones with it; resetZones rebuilds every
        // zone in order through a change, so the INDEX survives and only the labels move. brae keys by
        // name, which is the same statement on a list whose order is its file's.
        const auto it = cellZones.find(specs[i].cellZone);
        if (it == cellZones.end())
        {
            throw std::runtime_error(
                "brae MRF::update: cellZone `" + specs[i].cellZone + "` is not in the mesh's zones after "
                "the change. OpenFOAM stops on a missing MRF cellZone (MRFZone.C:583-590, the "
                "FatalErrorInFunction at :587).");
        }
        zones[i] = buildZone(specs[i], it->second, m, patches);
    }
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
