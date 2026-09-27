// brae's dynamicRefineFvMesh::updateTopology() against REAL OpenFOAM's, stepped -- unit 7 of the
// dynamicRefineFvMesh port.
//
// The oracle is tools/dumpRefineUpdate: OpenFOAM's own class with writes added (of-instrument), driven
// over an ANALYTIC field so that what is compared is the refinement DRIVER and not a solve. Per step it
// writes the cells selected for refinement, the refineCell marker after the map rebuild and after the
// buffer layers, the points selected for unrefinement, both maps, and the mesh, levels and history the
// step leaves behind.
//
// THE FIELD IS A MOVING SPHERE, 1 inside and 0 outside, recomputed from the CELL CENTRES on whatever
// mesh the step starts from. Both sides compute the same expression on the same centres, so the
// selection is exactly comparable; and it moves, so refinement happens ahead of it and unrefinement
// behind, which one step cannot exercise.
//
// ITS OWN DUMP READER, and deliberately not the one in test_hex_ref8_vs_openfoam.cu: this dump repeats
// every key once per step, so it is read into one Dump PER STEP rather than one map.
#include "cf_types.cuh"
#include "dynamic_refine_fv_mesh_cpp.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "hex_ref8_cpp.cuh"
#include "mesh_edges_cpp.cuh"
#include "primitive_mesh.cuh"
#include "primitive_patch_cpp.cuh"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::dynamicRefine;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

struct Dump
{
    std::map<std::string, label> scalars;
    std::map<std::string, std::vector<label>> lists;
    std::map<std::string, std::vector<std::vector<label>>> listLists;
};

// One Dump per step, plus a header Dump at index 0 for what is written before the first `step` line.
std::vector<Dump> readSteps(const std::string& path, std::vector<vector>& sphere)
{
    std::vector<Dump> out(1);
    std::ifstream is(path);
    if (!is)
    {
        std::printf("  FAIL: cannot open %s\n", path.c_str());
        ++failures;
        return out;
    }
    std::string line;
    while (std::getline(is, line))
    {
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        if (key == "step")
        {
            out.emplace_back();
            continue;
        }
        if (key == "sphereCentre" || key == "sphereVelocity")
        {
            vector v{0, 0, 0};
            ls >> v.x >> v.y >> v.z;
            sphere.push_back(v);
            continue;
        }
        if (key == "sphereRadius")
        {
            scalar r = 0;
            ls >> r;
            vector v{r, 0, 0};
            sphere.push_back(v);
            continue;
        }
        if (key == "mode") continue;
        label n = 0;
        if (!(ls >> n)) continue;
        std::string tail;
        std::getline(ls, tail);
        const bool nothingAfterN = (tail.find_first_not_of(" \t\r") == std::string::npos);
        Dump& d = out.back();
        // the same structural rule the hexRef8 harness ended up with: a scalar is `name value` and
        // NOTHING else, so a list whose name looks like a count's cannot be swallowed as one
        if (nothingAfterN && n != 0 && key != "points" && key != "faces" && key != "patches"
         && key != "historyAddedCells")
        {
            d.scalars[key] = n;
            continue;
        }
        if (!nothingAfterN)
        {
            std::vector<label> v(static_cast<std::size_t>(n));
            std::istringstream vs(line);
            std::string k2;
            label n2 = 0;
            vs >> k2 >> n2;
            for (label i = 0; i < n; ++i) vs >> v[static_cast<std::size_t>(i)];
            d.lists[key] = v;
            continue;
        }
        if (n == 0)
        {
            // `name 0` is genuinely ambiguous -- a scalar zero, an empty flat list and an empty block
            // all look the same -- so it is stored as ALL THREE. `nRefinementIterations 0` is a scalar
            // and `fieldOneCells 0` a list, and nothing in the line separates them.
            d.scalars[key] = 0;
            d.lists[key] = std::vector<label>();
            d.listLists[key] = std::vector<std::vector<label>>();
            continue;
        }
        std::vector<std::vector<label>> v;
        v.reserve(static_cast<std::size_t>(n));
        for (label i = 0; i < n; ++i)
        {
            if (!std::getline(is, line)) break;
            std::istringstream rs(line);
            if (key == "patches")
            {
                std::string nm;
                label st = 0, sz = 0;
                rs >> nm >> st >> sz;
                v.push_back({st, sz});
            }
            else if (key == "points")
            {
                v.push_back({});   // positions are not compared as labels; the line is consumed
            }
            else
            {
                label k = 0;
                rs >> k;
                std::vector<label> e(static_cast<std::size_t>(k));
                for (label j = 0; j < k; ++j) rs >> e[static_cast<std::size_t>(j)];
                v.push_back(e);
            }
        }
        d.listLists[key] = v;
    }
    return out;
}

// the real-valued lists unit 7b compares: read separately from the label ones, because a label reader
// would silently truncate every value to 0
std::map<std::string, std::vector<scalar>> readScalars(const std::string& path, label nSteps)
{
    // keyed "<step>/<name>"; the file is read a second time, which is cheap next to the comparison
    std::map<std::string, std::vector<scalar>> out;
    std::ifstream is(path);
    if (!is) return out;
    std::string line;
    label step = 0;
    while (std::getline(is, line))
    {
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        if (key == "step") { ls >> step; continue; }
        // the per-patch blocks: `name <nPatches>` and then one line per patch, `<n> v v ...`. Stored
        // FLATTENED in patch order, which is how the comparison flattens brae's side.
        if (key == "braePhiBnd" || key == "braePhiUBnd" || key == "braePhiFlatBnd"
         || key == "refinePhiUBnd" || key == "unrefinePhiUBnd"
         || key == "caseAlphaBnd" || key == "caseUBnd" || key == "casePrghBnd"
         || key == "caseAlphaRefValue" || key == "caseAlphaValueFraction"
         || key == "caseAlphaGradient" || key == "caseAlphaP0"
         || key == "casePrghRefValue" || key == "casePrghValueFraction"
         || key == "casePrghGradient" || key == "casePrghP0"
         || key == "braeNutBnd" || key == "braeOmegaBnd" || key == "braeKqBnd"
         || key == "braeUwallBnd" || key == "braeUfBnd"
         || key == "braeNutRefValue" || key == "braeNutValueFraction"
         || key == "braeNutGradient" || key == "braeNutP0"
         || key == "braeOmegaRefValue" || key == "braeOmegaValueFraction"
         || key == "braeOmegaGradient" || key == "braeOmegaP0"
         || key == "braeKqRefValue" || key == "braeKqValueFraction"
         || key == "braeKqGradient" || key == "braeKqP0")
        {
            label nPatches = 0;
            ls >> nPatches;
            std::vector<scalar> flat;
            for (label p = 0; p < nPatches; ++p)
            {
                if (!std::getline(is, line)) break;
                std::istringstream ps(line);
                label n = 0;
                ps >> n;
                for (label i = 0; i < n; ++i)
                {
                    scalar v = 0;
                    ps >> v;
                    flat.push_back(v);
                }
            }
            out[std::to_string(step) + "/" + key] = flat;
            continue;
        }
        if (key != "braeScalar" && key != "braeFresh" && key != "braeVector" && key != "V0" && key != "V"
         && key != "braePhi" && key != "braePhiFlat" && key != "braePhiU" && key != "braeUf"
         && key != "refinePhiU" && key != "unrefinePhiU"
         && key != "refineOldCellVolumes" && key != "unrefineOldCellVolumes")
        {
            continue;
        }
        label n = 0;
        ls >> n;
        const label nRead = (key == "braeVector" || key == "braeUf") ? 3*n : n;
        std::vector<scalar> v(static_cast<std::size_t>(nRead));
        for (label i = 0; i < nRead; ++i) ls >> v[static_cast<std::size_t>(i)];
        out[std::to_string(step) + "/" + key] = v;
    }
    (void)nSteps;
    return out;
}

// the worst RELATIVE difference, and where. A merged cell's value is a volume-weighted mean, so it is
// only exact when the weights are -- which is why the gate runs an arm with OpenFOAM's own volumes.
struct Worst
{
    scalar rel = 0;
    std::size_t at = 0;
    scalar mine = 0;
    scalar theirs = 0;
};

Worst worstDiff(const std::vector<scalar>& mine, const std::vector<scalar>& theirs)
{
    Worst w;
    const std::size_t n = mine.size() < theirs.size() ? mine.size() : theirs.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const scalar d = std::fabs(mine[i] - theirs[i]);
        const scalar scale = std::fabs(theirs[i]) > scalar(1e-300) ? std::fabs(theirs[i]) : scalar(1);
        const scalar rel = d/scale;
        if (rel > w.rel) { w.rel = rel; w.at = i; w.mine = mine[i]; w.theirs = theirs[i]; }
    }
    return w;
}

void compareScalars(
    const char* name,
    const std::vector<scalar>& mine,
    const std::map<std::string, std::vector<scalar>>& sc,
    const std::string& key,
    scalar bound)
{
    const auto it = sc.find(key);
    if (it == sc.end()) { std::printf("  FAIL: the dump has no `%s`\n", key.c_str()); ++failures; return; }
    if (mine.size() != it->second.size())
    {
        std::printf("  FAIL: %s has %zu values, OpenFOAM %zu\n", name, mine.size(), it->second.size());
        ++failures;
        return;
    }
    const Worst w = worstDiff(mine, it->second);
    if (w.rel > bound)
    {
        std::printf("  FAIL: %s worst relative difference %.3e at %zu (brae %.17g, OpenFOAM %.17g), "
                    "bound %.1e\n", name, (double)w.rel, w.at, (double)w.mine, (double)w.theirs,
                    (double)bound);
        ++failures;
    }
    else
    {
        std::printf("  ok:   %s is OpenFOAM's (%zu values, worst %.3e)\n", name, mine.size(),
                    (double)w.rel);
    }
}

void compareList(const char* name, const std::vector<label>& mine, const Dump& d, const char* key)
{
    const auto it = d.lists.find(key);
    if (it == d.lists.end()) { std::printf("  FAIL: the dump has no `%s`\n", key); ++failures; return; }
    if (mine.size() != it->second.size())
    {
        std::printf("  FAIL: %s has %zu entries, OpenFOAM %zu\n", name, mine.size(), it->second.size());
        ++failures;
        return;
    }
    for (std::size_t i = 0; i < mine.size(); ++i)
    {
        if (mine[i] != it->second[i])
        {
            std::printf("  FAIL: %s differs first at %zu (brae %d, OpenFOAM %d)\n", name, i,
                        (int)mine[i], (int)it->second[i]);
            ++failures;
            return;
        }
    }
    std::printf("  ok:   %s is OpenFOAM's (%zu entries)\n", name, mine.size());
}

void compareFlags(const char* name, const std::vector<char>& mine, const Dump& d, const char* key)
{
    std::vector<label> asLabels(mine.size());
    for (std::size_t i = 0; i < mine.size(); ++i) asLabels[i] = mine[i] ? 1 : 0;
    compareList(name, asLabels, d, key);
}

void compareListList(
    const char* name,
    const std::vector<std::vector<label>>& mine,
    const Dump& d,
    const char* key)
{
    const auto it = d.listLists.find(key);
    if (it == d.listLists.end()) { std::printf("  FAIL: the dump has no `%s`\n", key); ++failures; return; }
    if (mine.size() != it->second.size())
    {
        std::printf("  FAIL: %s has %zu entries, OpenFOAM %zu\n", name, mine.size(), it->second.size());
        ++failures;
        return;
    }
    for (std::size_t i = 0; i < mine.size(); ++i)
    {
        if (mine[i] != it->second[i])
        {
            std::printf("  FAIL: %s differs at %zu (brae %zu entries, OpenFOAM %zu)\n", name, i,
                        mine[i].size(), it->second[i].size());
            ++failures;
            return;
        }
    }
    std::printf("  ok:   %s is OpenFOAM's (%zu entries)\n", name, mine.size());
}

}   // namespace

int main(int argc, char** argv)
{
    std::printf("== brae dynamicRefineFvMesh::updateTopology vs OpenFOAM's ==\n");
    if (argc < 3)
    {
        std::printf("  SKIP: usage: %s <caseDir> <dump>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string dumpPath = argv[2];

    std::vector<vector> sphere;                 // radius (in .x), centre, velocity -- in write order
    const std::vector<Dump> steps = readSteps(dumpPath, sphere);
    if (failures) return 1;
    if (sphere.size() != 3)
    {
        std::printf("  FAIL: the dump did not carry the sphere (%zu entries)\n", sphere.size());
        return 1;
    }
    const scalar radius = sphere[0].x;
    const vector centre = sphere[1];
    const vector velocity = sphere[2];
    const label nSteps = static_cast<label>(steps.size()) - 1;
    std::printf("  %d steps, sphere radius %g from (%g %g %g) moving (%g %g %g)\n", (int)nSteps,
                (double)radius, (double)centre.x, (double)centre.y, (double)centre.z,
                (double)velocity.x, (double)velocity.y, (double)velocity.z);

    RefineUpdateState s;
    s.m.read(caseDir + "/constant/polyMesh");
    FvGeometry g0;
    g0.build(s.m);
    s.patches = buildPatches(s.m, g0);
    check("the case mesh is the one OpenFOAM started from",
          s.m.nCells() == steps[0].scalars.at("startCells"));

    const FoamDict dmd = readDict(caseDir + "/constant/dynamicMeshDict");
    const RefineControls controls = readRefineControls(dmd);
    std::printf("  controls: refineInterval %d, maxRefinement %d, maxCells %d, nBufferLayers %d, "
                "unrefineLevel %g\n", (int)controls.refineInterval, (int)controls.maxRefinement,
                (int)controls.maxCells, (int)controls.nBufferLayers, (double)controls.unrefineLevel);

    // a mesh that has never been refined starts at level 0 everywhere, and its history is the identity
    s.levels.cellLevel.assign(static_cast<std::size_t>(s.m.nCells()), label(0));
    s.levels.pointLevel.assign(static_cast<std::size_t>(s.m.nPoints()), label(0));
    s.history = cpu::hexRef8::freshHistory(s.m.nCells());
    {
        // dynamicRefineFvMesh::init's own scan, which runs ONCE at construction and is only renumbered
        // afterwards
        const std::vector<std::vector<label>> cells = meshCells(s.m);
        const std::vector<std::vector<label>> pointCells = pointCellsFromCells(s.m, cells);
        s.protectedCell = initProtectedCells(s.levels.cellLevel, s.levels.pointLevel, pointCells, cells,
                                            s.m, s.patches);
        std::printf("  protectedCell at construction: %zu entries\n", s.protectedCell.size());
    }

    // UNIT 7b. The two passive fields the oracle carries: set ONCE to a value that says which cell each
    // came from, then only ever mapped -- so what is compared after three steps is three mappings
    // composed, not one.
    const std::map<std::string, std::vector<scalar>> sc = readScalars(dumpPath, nSteps);
    {
        std::vector<scalar> ps(static_cast<std::size_t>(s.m.nCells()));
        std::vector<vector> pv(static_cast<std::size_t>(s.m.nCells()));
        for (label celli = 0; celli < s.m.nCells(); ++celli)
        {
            ps[static_cast<std::size_t>(celli)] = scalar(celli);
            pv[static_cast<std::size_t>(celli)] =
                vector{scalar(celli), scalar(2*celli), scalar(3*celli)};
        }
        s.cellScalars.push_back(ps);        // carried across all three steps
        s.cellScalars.push_back(ps);        // ...and re-set at the start of every step, below
        s.cellVectors.push_back(pv);
        // UNIT 7b-2: the oriented surface field, set to the face index -- internal faces by their own
        // label, each patch face by its MESH label, which is what the oracle writes
        RefineUpdateState::CarriedSurfaceField phi;
        phi.oriented = true;
        phi.field.resize(static_cast<std::size_t>(s.m.nInternalFaces()));
        for (label f = 0; f < s.m.nInternalFaces(); ++f)
        {
            phi.field[static_cast<std::size_t>(f)] = scalar(f);
        }
        phi.bnd.resize(s.m.patches().size());
        for (std::size_t pi = 0; pi < s.m.patches().size(); ++pi)
        {
            const PatchInfo& pp = s.m.patches()[pi];
            phi.bnd[pi].resize(static_cast<std::size_t>(pp.size));
            for (label i = 0; i < pp.size; ++i)
            {
                phi.bnd[pi][static_cast<std::size_t>(i)] = scalar(pp.start + i);
            }
        }
        s.surfaceScalars.push_back(phi);        // ORIENTED: negated on a flip, averaged as a vector
        phi.oriented = false;
        s.surfaceScalars.push_back(phi);        // ...and the same field without the flag
        // whether the case's correctFluxes names a velocity for each. The dictionary is the authority:
        // `braePhi` gets a correction only where the gate's own arm B puts `(braePhi braeU)` in it, which
        // no tutorial does.
        for (const auto& pair : controls.correctFluxes)
        {
            std::printf("  correctFluxes: (%s %s)\n", pair.first.c_str(), pair.second.c_str());
        }
        const auto velocityFor = [&](const std::string& name)
        {
            for (const auto& pair : controls.correctFluxes)
            {
                if (pair.first == name) return pair.second;
            }
            return std::string();
        };
        s.surfaceScalarVelocity.push_back(velocityFor("braePhi"));
        s.surfaceScalarVelocity.push_back(velocityFor("braePhiFlat"));

        // ...AND A SURFACE VECTOR, the one surface path this gate did not cover and the one Uf takes. The
        // two surface branches are different code: a surface SCALAR's hull average sums one component, a
        // surface VECTOR's sums three. UNORIENTED, because Uf is a velocity and not a flux, so it is not
        // negated on a flipped face.
        //
        // WHY: the restart profile of tests/interfoam_amr_levels_vs_openfoam.sh reads alpha exactly and Uf
        // 2.3274e-10 relative -- 16x what the same case reads with the change switched OFF -- diffuse over
        // about a thousand faces. Every other adaptive fixture's FIRST change happens from rest, where a
        // surface field is zero and a refined cell's hull average is exactly 0 in both codes, so none of
        // them exercises this path at all. This arm is what tells the MAP from the rebuild that follows it.
        dynamicRefine::RefineUpdateState::CarriedSurfaceVectorField uf;
        uf.oriented = false;
        uf.field.resize(static_cast<std::size_t>(s.m.nInternalFaces()));
        for (label f = 0; f < s.m.nInternalFaces(); ++f)
        {
            uf.field[static_cast<std::size_t>(f)] =
                vector{scalar(f), scalar(2*f), scalar(3*f)};
        }
        uf.bnd.resize(s.patches.size());
        for (std::size_t pi = 0; pi < s.patches.size(); ++pi)
        {
            const FvPatch& pp = s.patches[pi];
            uf.bnd[pi].resize(static_cast<std::size_t>(pp.size));
            for (label i = 0; i < pp.size; ++i)
            {
                const label g = pp.start + i;
                uf.bnd[pi][static_cast<std::size_t>(i)] =
                    vector{scalar(g), scalar(2*g), scalar(3*g)};
            }
        }
        s.surfaceVectors.push_back(uf);
        std::printf("  the carried flux's velocity: `%s`\n",
                    s.surfaceScalarVelocity.at(0).empty() ? "(not in the table)"
                                                          : s.surfaceScalarVelocity.at(0).c_str());
    }
    // UNIT 8a: the case's own fields, with the patch TYPES the case wrote -- inletOutlet and zeroGradient
    // on alpha, pressureInletOutletVelocity and uniformFixedValue on U, totalPressure and
    // fixedFluxPressure on p_rgh. They are read and then only MAPPED, so every boundary value and every
    // piece of per-face state below is brae's autoMap against OpenFOAM's.
    GeometricField<scalar> caseAlpha =
        buildField(readField<scalar>(caseDir + "/0/alpha.water"), s.patches, s.m.nCells());
    GeometricField<vector> caseU =
        buildField(readField<vector>(caseDir + "/0/U"), s.patches, s.m.nCells());
    GeometricField<scalar> casePrgh =
        buildField(readField<scalar>(caseDir + "/0/p_rgh"), s.patches, s.m.nCells());
    // OpenFOAM's own field-from-file has its boundary EVALUATED at construction, so a zeroGradient patch
    // holds the cell values and not zeros before anything is mapped. brae's buildField leaves value_ at
    // whatever the constructor set, so it is evaluated once here -- otherwise the comparison measures the
    // starting state and not the mapping. MEASURED: without this, `walls` (zeroGradient on alpha) read 0
    // where OpenFOAM read 1, from the first face of the second patch onwards.
    // OpenFOAM's own field-from-file has its boundary EVALUATED at construction, so a zeroGradient patch
    // holds the cell values and not zeros before anything is mapped. MEASURED: without this, `walls`
    // (zeroGradient on alpha) read 0 where OpenFOAM read 1, from the first face of the second patch on.
    //
    // AND IT COMES BEFORE THE STATE PATTERN BELOW, not after: the oracle writes the pattern into an
    // already-constructed field and does NOT re-evaluate, so a fixedGradient patch keeps the file's value
    // beside a gradient that no longer produced it. Evaluating after the pattern instead read p_rgh's last
    // wall face as 1561.98 against OpenFOAM's 0 -- the value the new gradient WOULD give.
    caseAlpha.evaluateBoundary();
    caseU.evaluateBoundary();
    casePrgh.evaluateBoundary();

    // THE SAME PER-FACE PATTERN THE ORACLE WRITES, for the same reason: the case's own state is uniform,
    // so a comparison of it cannot tell a mapping from a re-assignment. See the note in
    // fv_patch_field.cuh.
    for (std::size_t pi = 0; pi < s.patches.size(); ++pi)
    {
        const label start = s.patches[pi].start;
        const label n = s.patches[pi].size;
        std::vector<scalar> ramp(static_cast<std::size_t>(n));
        std::vector<scalar> frac(static_cast<std::size_t>(n));
        for (label i = 0; i < n; ++i)
        {
            ramp[static_cast<std::size_t>(i)] = scalar(start + i);
            frac[static_cast<std::size_t>(i)] =
                scalar(0.25) + scalar(0.5)*scalar(i % 3)/scalar(3);
        }
        // A PATTERN IN THE VALUES TOO WAS TRIED AND TAKEN BACK OUT. It would make the base's own value
        // mapping discriminating -- the fail-proof on it is GREEN as things stand, because alpha and p_rgh
        // are 0 over most of the boundary and a resize that keeps the old values and zero-fills the new
        // faces gives the same answer as mapping them. Writing a per-face ramp into the values needs the
        // two sides to agree on what they wrote, and they did not: OpenFOAM's went in through the Field
        // base while brae's went through assignValue, and U's walls patch came out 299901 against
        // OpenFOAM's own number. Recorded as unwitnessed rather than left as a green line with a broken
        // arm behind it; the STATE comparisons below carry the pattern and are discriminating.
        if (!caseAlpha.boundary[pi]->mappedRefValues().empty())
        {
            caseAlpha.boundary[pi]->setMappedRefValues(ramp);
            caseAlpha.boundary[pi]->setMappedValueFraction(frac);
        }
        if (!casePrgh.boundary[pi]->mappedGradient().empty())
        {
            casePrgh.boundary[pi]->setMappedGradient(ramp);
        }
        if (!casePrgh.boundary[pi]->mappedP0().empty())
        {
            casePrgh.boundary[pi]->setMappedP0(ramp);
        }
    }

    s.carriedScalarFields.push_back(&caseAlpha);
    s.carriedScalarFields.push_back(&casePrgh);
    s.carriedVectorFields.push_back(&caseU);
    std::printf("  carried whole fields: alpha.water, p_rgh (scalar) and U (vector), %zu patches each\n",
                s.patches.size());

    // UNIT 8b: the patch types the other two adaptive cases need, on fields the gate writes for them.
    // Read only if the files are there, so the gate still runs on a case without them.
    std::vector<std::pair<std::string, GeometricField<scalar>*>> typedScalars;
    std::vector<std::pair<std::string, GeometricField<vector>*>> typedVectors;
    GeometricField<scalar> braeNut, braeOmega, braeKq;
    GeometricField<vector> braeUwall;
    {
        const std::pair<const char*, GeometricField<scalar>*> sc3[3] =
            {{"braeNut", &braeNut}, {"braeOmega", &braeOmega}, {"braeKq", &braeKq}};
        for (const auto& e : sc3)
        {
            const std::string path = caseDir + "/0/" + e.first;
            if (!std::ifstream(path)) continue;
            *e.second = buildField(readField<scalar>(path), s.patches, s.m.nCells());
            e.second->evaluateBoundary();
            typedScalars.emplace_back(e.first, e.second);
            s.carriedScalarFields.push_back(e.second);
        }
        const std::string up = caseDir + "/0/braeUwall";
        if (std::ifstream(up))
        {
            braeUwall = buildField(readField<vector>(up), s.patches, s.m.nCells());
            braeUwall.evaluateBoundary();
            typedVectors.emplace_back("braeUwall", &braeUwall);
            s.carriedVectorFields.push_back(&braeUwall);
        }
        std::printf("  unit 8b typed fields: %zu scalar, %zu vector\n", typedScalars.size(),
                    typedVectors.size());
    }

    // ...and OpenFOAM's OWN old cell volumes, injected per change. brae's FvGeometry::V() agrees with
    // OpenFOAM's to round-off but not bit-for-bit, and a volume-weighted mean carries that into every
    // merged value -- so injecting them is what makes a MAPPER defect separable from brae's volumes.
    // The `BRAE_REFINE_UPDATE_OWN_V` arm below runs without the injection and states its own bound.
    const bool ownVolumes = (std::getenv("BRAE_REFINE_UPDATE_OWN_V") != nullptr);
    std::printf("  old cell volumes for the merge weights: %s\n",
                ownVolumes ? "brae's own FvGeometry::V()" : "OpenFOAM's, injected");

    for (label step = 1; step <= nSteps; ++step)
    {
        const Dump& d = steps[static_cast<std::size_t>(step)];
        if (!ownVolumes)
        {
            const auto rv = sc.find(std::to_string(step) + "/refineOldCellVolumes");
            const auto uv = sc.find(std::to_string(step) + "/unrefineOldCellVolumes");
            s.injectedRefineOldV = (rv == sc.end()) ? std::vector<scalar>() : rv->second;
            s.injectedUnrefineOldV = (uv == sc.end()) ? std::vector<scalar>() : uv->second;
        }
        std::printf("  -- step %d, from %d cells\n", (int)step, (int)s.m.nCells());

        // the field, on THIS step's mesh, at THIS step's centre -- the tool's own expression
        FvGeometry g;
        g.build(s.m);
        const vector c = centre + scalar(step)*velocity;
        std::vector<scalar> field(static_cast<std::size_t>(s.m.nCells()), scalar(0));
        for (label celli = 0; celli < s.m.nCells(); ++celli)
        {
            field[static_cast<std::size_t>(celli)] =
                (mag(g.C()[static_cast<std::size_t>(celli)] - c) < radius) ? scalar(1) : scalar(0);
        }
        // ...and the same field as the cells it is 1 in, which is what the oracle wrote: if this list
        // differs the two sides are not being driven by the same input and nothing below means anything
        {
            std::vector<label> ones;
            for (label celli = 0; celli < s.m.nCells(); ++celli)
            {
                if (field[static_cast<std::size_t>(celli)] > scalar(0.5)) ones.push_back(celli);
            }
            compareList("the driving field's cells", ones, d, "fieldOneCells");
        }

        // the fresh field: written again on THIS step's mesh, so the cells a merge combines hold eight
        // different values and the weighted mean is a real average rather than a constant
        {
            std::vector<scalar>& fresh = s.cellScalars.at(1);
            fresh.resize(static_cast<std::size_t>(s.m.nCells()));
            for (label celli = 0; celli < s.m.nCells(); ++celli)
            {
                fresh[static_cast<std::size_t>(celli)] = scalar(celli);
            }
        }

        // OpenFOAM's own interpolated flux, PER CHANGE, on the mesh that change produced -- which is the
        // field mapFields itself used, computed from the already-mapped braeU. Injected so that brae's
        // surface interpolation is not in the way of the correction's logic. The driver is handed the
        // refine's before the step and the unrefine's is picked up inside it, so both changes of a step
        // get the right one.
        const bool correcting = !s.surfaceScalarVelocity.empty()
                             && !s.surfaceScalarVelocity.at(0).empty()
                             && s.surfaceScalarVelocity.at(0) != "none";
        const auto loadPhiU = [&](const std::string& key)
        {
            const auto pu = sc.find(std::to_string(step) + "/" + key);
            const auto pb = sc.find(std::to_string(step) + "/" + key + "Bnd");
            s.injectedPhiU = (pu == sc.end()) ? std::vector<scalar>() : pu->second;
            s.injectedPhiUBnd.clear();
            if (pb == sc.end()) return;
            // the patch sizes here are the POST-change ones, which the dump's own mesh block carries
            std::size_t at = 0;
            const auto it = d.listLists.find("patches");
            if (it == d.listLists.end()) return;
            for (const std::vector<label>& pp : it->second)
            {
                std::vector<scalar> one;
                for (label i = 0; i < pp[1] && at < pb->second.size(); ++i, ++at)
                {
                    one.push_back(pb->second[at]);
                }
                s.injectedPhiUBnd.push_back(one);
            }
        };
        if (correcting) loadPhiU("refinePhiU");
        s.injectedPhiURefine = s.injectedPhiU;
        s.injectedPhiURefineBnd = s.injectedPhiUBnd;
        if (correcting) loadPhiU("unrefinePhiU");
        s.injectedPhiUUnrefine = s.injectedPhiU;
        s.injectedPhiUUnrefineBnd = s.injectedPhiUBnd;

        const label timeIndex = d.scalars.at("timeIndex");
        const RefineUpdateStep r = refineUpdate(s, controls, field, timeIndex);

        check("hasChanged is OpenFOAM's", r.hasChanged == (d.scalars.at("hasChanged") != 0));
        compareList("cellsToRefine", r.cellsToRefine, d, "cellsToRefine");
        if (r.refined)
        {
            compareList("the refinement's pointMap", r.refineMap.pointMap, d, "refinePointMap");
            compareList("the refinement's faceMap", r.refineMap.faceMap, d, "refineFaceMap");
            compareList("the refinement's cellMap", r.refineMap.cellMap, d, "refineCellMap");
            compareList("the refinement's reversePointMap", r.refineMap.reversePointMap, d,
                        "refineReversePointMap");
            compareList("the refinement's reverseFaceMap", r.refineMap.reverseFaceMap, d,
                        "refineReverseFaceMap");
            compareList("the refinement's reverseCellMap", r.refineMap.reverseCellMap, d,
                        "refineReverseCellMap");
            compareFlags("refineCell after the map rebuild", r.refineCellAfterMap, d, "refineCellAfterMap");
            compareFlags("refineCell after the buffer layers", r.refineCellAfterBuffer, d,
                         "refineCellAfterBuffer");
        }
        // flipFaceFlux, which is what an ORIENTED field is negated on. MEASURED: it is EMPTY on every
        // change of this fixture, so the flip is carried and cannot be witnessed here -- which is said
        // rather than left, and is why the fail-proof on it is green.
        if (r.refined) compareList("the refinement's flipFaceFlux", r.refineMap.flipFaceFlux, d,
                                   "refineFlipFaceFlux");
        compareList("pointsToUnrefine", r.pointsToUnrefine, d, "pointsToUnrefine");
        if (r.unrefined)
        {
            compareList("the unrefinement's pointMap", r.unrefineMap.pointMap, d, "unrefinePointMap");
            compareList("the unrefinement's faceMap", r.unrefineMap.faceMap, d, "unrefineFaceMap");
            compareList("the unrefinement's cellMap", r.unrefineMap.cellMap, d, "unrefineCellMap");
            compareList("the unrefinement's reversePointMap", r.unrefineMap.reversePointMap, d,
                        "unrefineReversePointMap");
            compareList("the unrefinement's reverseFaceMap", r.unrefineMap.reverseFaceMap, d,
                        "unrefineReverseFaceMap");
            compareList("the unrefinement's reverseCellMap", r.unrefineMap.reverseCellMap, d,
                        "unrefineReverseCellMap");
            compareList("the unrefinement's flipFaceFlux", r.unrefineMap.flipFaceFlux, d,
                        "unrefineFlipFaceFlux");
        }

        // the state the step leaves: the mesh, the levels and the history
        check("the mesh has OpenFOAM's counts",
              s.m.nPoints() == d.scalars.at("stepnPoints")
           && s.m.nFaces() == d.scalars.at("stepnFaces")
           && s.m.nInternalFaces() == d.scalars.at("stepnInternalFaces")
           && s.m.nCells() == d.scalars.at("stepnCells"));
        compareList("owner", s.m.owner(), d, "owner");
        compareList("neighbour", s.m.neighbour(), d, "neighbour");
        {
            std::vector<std::vector<label>> faces(static_cast<std::size_t>(s.m.nFaces()));
            for (label f = 0; f < s.m.nFaces(); ++f)
            {
                faces[static_cast<std::size_t>(f)].assign(
                    s.m.faceVerts().begin() + s.m.faceOffsets()[static_cast<std::size_t>(f)],
                    s.m.faceVerts().begin() + s.m.faceOffsets()[static_cast<std::size_t>(f) + 1]);
            }
            compareListList("every face's vertex list", faces, d, "faces");
        }
        compareList("cellLevel", s.levels.cellLevel, d, "cellLevel");
        compareList("pointLevel", s.levels.pointLevel, d, "pointLevel");
        compareList("the history's visibleCells", s.history.visibleCells, d, "historyVisibleCells");
        compareList("the history's parent list", s.history.parent, d, "historyParent");
        compareListList("the history's addedCells", s.history.addedCells, d, "historyAddedCells");
        check("the history was compacted on the same steps OpenFOAM compacts on",
              r.compacted == (d.scalars.at("compactedThisStep") != 0));
        check("nRefinementIterations is OpenFOAM's",
              s.nRefinementIterations == d.scalars.at("nRefinementIterations") + 1);

        // UNIT 7b: the mapped fields and the old-time volumes. The bound is 0 with OpenFOAM's own
        // volumes injected -- the mapping is then pure arithmetic on identical inputs, and anything but
        // exactness is a defect -- and 1e-12 when brae weighs the merges with its own V.
        {
            const scalar bound = ownVolumes ? scalar(1e-12) : scalar(0);
            compareScalars("the mapped scalar field", s.cellScalars.at(0), sc,
                           std::to_string(step) + "/braeScalar", bound);
            compareScalars("the step-fresh scalar field -- the one a merge can average", s.cellScalars.at(1),
                           sc, std::to_string(step) + "/braeFresh", bound);
            std::vector<scalar> flatV(3*s.cellVectors.at(0).size());
            for (std::size_t i = 0; i < s.cellVectors.at(0).size(); ++i)
            {
                flatV[3*i + 0] = s.cellVectors.at(0)[i].x;
                flatV[3*i + 1] = s.cellVectors.at(0)[i].y;
                flatV[3*i + 2] = s.cellVectors.at(0)[i].z;
            }
            compareScalars("the mapped vector field", flatV, sc,
                           std::to_string(step) + "/braeVector", bound);
            // V0 is not a mapped field: it is gathered, the merged parts are SUMMED into the master, and
            // then split and merged cells are overwritten with their own new V. brae's V is its own, so
            // this one is held at 1e-12 on both arms and the worst is printed.
            compareScalars("the old-time volumes V0", s.V0, sc, std::to_string(step) + "/V0",
                           scalar(1e-12));
            // UNIT 7b-2: the oriented surface field, internal and boundary. Pure addressing and a sign,
            // so the bound is ZERO on both arms.
            // UNIT 8a: the typed patch fields. Pure addressing and a zero-gradient fill, so bound 0.
            {
                // AN `empty` PATCH CONTRIBUTES NOTHING TO OpenFOAM'S ARRAY, and that is not a detail of
                // the dump: emptyFvPatchField is constructed ZERO-SIZED on a patch that has faces
                // (emptyFvPatchField.C:41), so OpenFOAM writes no value for any of them. brae's own
                // empty patch field IS sized -- both loops index it per face -- so a flat comparison has
                // to skip those patches or it compares 5,434 values against 208 on a 2-D mesh, which is
                // what pointing this harness at laminar/damBreak first read.
                const auto skipPatch = [&](std::size_t pi)
                {
                    return pi < s.patches.size() && s.patches[pi].type == "empty";
                };
                const auto flatScalarBnd = [&](const GeometricField<scalar>& f)
                {
                    std::vector<scalar> out;
                    for (std::size_t pi = 0; pi < f.boundary.size(); ++pi)
                    {
                        if (skipPatch(pi)) continue;
                        const std::vector<scalar> v = f.boundary[pi]->value();
                        out.insert(out.end(), v.begin(), v.end());
                    }
                    return out;
                };
                compareScalars("alpha.water's patch values", flatScalarBnd(caseAlpha), sc,
                               std::to_string(step) + "/caseAlphaBnd", scalar(0));
                compareScalars("p_rgh's patch values", flatScalarBnd(casePrgh), sc,
                               std::to_string(step) + "/casePrghBnd", scalar(0));
                std::vector<scalar> flatU;
                for (std::size_t pi = 0; pi < caseU.boundary.size(); ++pi)
                {
                    if (skipPatch(pi)) continue;
                    for (const vector& v : caseU.boundary[pi]->value())
                    {
                        flatU.push_back(v.x);
                        flatU.push_back(v.y);
                        flatU.push_back(v.z);
                    }
                }
                compareScalars("U's patch values", flatU, sc, std::to_string(step) + "/caseUBnd",
                               scalar(0));
                // ...and the per-face STATE each type carries, which is what the overrides map: a mixed
                // patch's refValue and valueFraction, a fixedGradient's gradient, a totalPressure's p0.
                // A patch of another type writes an empty list on OpenFOAM's side, so the comparison is
                // over the patches that HAVE the state, in patch order.
                std::vector<scalar> refV, vFrac, grad, p0;
                for (std::size_t pi = 0; pi < caseAlpha.boundary.size(); ++pi)
                {
                    if (skipPatch(pi)) continue;
                    const std::vector<scalar> r = caseAlpha.boundary[pi]->mappedRefValues();
                    refV.insert(refV.end(), r.begin(), r.end());
                    const std::vector<scalar> v = caseAlpha.boundary[pi]->mappedValueFraction();
                    vFrac.insert(vFrac.end(), v.begin(), v.end());
                }
                compareScalars("alpha.water's inletOutlet refValue", refV, sc,
                               std::to_string(step) + "/caseAlphaRefValue", scalar(0));
                compareScalars("alpha.water's inletOutlet valueFraction", vFrac, sc,
                               std::to_string(step) + "/caseAlphaValueFraction", scalar(0));
                for (std::size_t pi = 0; pi < casePrgh.boundary.size(); ++pi)
                {
                    if (skipPatch(pi)) continue;
                    const std::vector<scalar> g = casePrgh.boundary[pi]->mappedGradient();
                    grad.insert(grad.end(), g.begin(), g.end());
                    const std::vector<scalar> q = casePrgh.boundary[pi]->mappedP0();
                    p0.insert(p0.end(), q.begin(), q.end());
                }
                compareScalars("p_rgh's fixedFluxPressure gradient", grad, sc,
                               std::to_string(step) + "/casePrghGradient", scalar(0));
                compareScalars("p_rgh's totalPressure p0", p0, sc,
                               std::to_string(step) + "/casePrghP0", scalar(0));
            }

            // UNIT 8b: the wall-function and movingWallVelocity patches. Their internal fields are the
            // cell index, so a wall function's patch value differs face by face and the comparison cannot
            // pass on a constant.
            for (const auto& e : typedScalars)
            {
                std::vector<scalar> flat;
                for (std::size_t pi = 0; pi < e.second->boundary.size(); ++pi)
                {
                    if (pi < s.patches.size() && s.patches[pi].type == "empty") continue;
                    const std::vector<scalar>& v = e.second->boundary[pi]->value();
                    flat.insert(flat.end(), v.begin(), v.end());
                }
                compareScalars((e.first + "'s patch values").c_str(), flat, sc,
                               std::to_string(step) + "/" + e.first + "Bnd", scalar(0));
            }
            for (const auto& e : typedVectors)
            {
                std::vector<scalar> flat;
                for (std::size_t pi = 0; pi < e.second->boundary.size(); ++pi)
                {
                    if (pi < s.patches.size() && s.patches[pi].type == "empty") continue;
                    for (const vector& v : e.second->boundary[pi]->value())
                    {
                        flat.push_back(v.x);
                        flat.push_back(v.y);
                        flat.push_back(v.z);
                    }
                }
                compareScalars((e.first + "'s patch values").c_str(), flat, sc,
                               std::to_string(step) + "/" + e.first + "Bnd", scalar(0));
            }

            // ORIENTED: the hull average's round trip through an intensive vector puts brae's own Sf
            // into the answer, so this one is held at 1e-12 and the worst is printed. UNORIENTED: pure
            // addressing and an average of the values themselves, so bound 0.
            const char* names[2] = {"braePhi", "braePhiFlat"};
            const scalar bounds[2] = {scalar(1e-12), scalar(0)};
            for (int k = 0; k < 2; ++k)
            {
                const std::string what = std::string("the mapped surface field `") + names[k] + "`";
                compareScalars((what + ", internal").c_str(), s.surfaceScalars.at(k).field, sc,
                               std::to_string(step) + "/" + names[k], bounds[k]);
                std::vector<scalar> flat;
                for (std::size_t pi = 0; pi < s.surfaceScalars.at(k).bnd.size(); ++pi)
                {
                    if (pi < s.patches.size() && s.patches[pi].type == "empty") continue;
                    const std::vector<scalar>& pf = s.surfaceScalars.at(k).bnd[pi];
                    flat.insert(flat.end(), pf.begin(), pf.end());
                }
                compareScalars((what + ", boundary").c_str(), flat, sc,
                               std::to_string(step) + "/" + names[k] + "Bnd", bounds[k]);
            }

            // ...AND THE SURFACE VECTOR. Flattened x,y,z per face, which is the order the dump writes, so a
            // mapper that carried the wrong component or summed the three in a different order reads here
            // and not as a magnitude that happens to agree.
            if (!s.surfaceVectors.empty())
            {
                const auto& uf = s.surfaceVectors.at(0);
                std::vector<scalar> flatUf;
                flatUf.reserve(3*uf.field.size());
                for (const vector& v : uf.field)
                {
                    flatUf.push_back(v.x);
                    flatUf.push_back(v.y);
                    flatUf.push_back(v.z);
                }
                compareScalars("the mapped surface VECTOR `braeUf`, internal", flatUf, sc,
                               std::to_string(step) + "/braeUf", scalar(0));
                std::vector<scalar> flatUfB;
                for (std::size_t pi = 0; pi < uf.bnd.size(); ++pi)
                {
                    if (pi < s.patches.size() && s.patches[pi].type == "empty") continue;
                    for (const vector& v : uf.bnd[pi])
                    {
                        flatUfB.push_back(v.x);
                        flatUfB.push_back(v.y);
                        flatUfB.push_back(v.z);
                    }
                }
                compareScalars("the mapped surface VECTOR `braeUf`, boundary", flatUfB, sc,
                               std::to_string(step) + "/braeUfBnd", scalar(0));
            }
        }
    }

    std::printf("test_refine_update_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
