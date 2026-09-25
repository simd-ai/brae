// brae's cell mapper against REAL OpenFOAM's own, through one refinement step and one unrefinement
// step of laminar/damBreakWithObstacle -- unit 5 of the dynamicRefineFvMesh port.
//
// WHAT IS UNDER TEST: how a field's CELL VALUES move when the mesh changes under it. Given OpenFOAM's
// own mapPolyMesh and the field as it stood BEFORE the change, produce the values OpenFOAM produced
// after it. That is `cellMapper` plus `Field<Type>::map`, and it is the one part of adaptive
// refinement that decides whether the answer is right rather than merely well-shaped.
//
// THE ORACLE is tools/dumpRefineMap, which does not transcribe OpenFOAM: it derives a mesh from
// dynamicRefineFvMesh, overrides the VIRTUAL protected `refine`/`unrefine` with a body that calls the
// base implementation and nothing else, and reads the map out of the autoPtr it returns. It then
// builds the very `cellMapper` that `fvMesh::mapFields` uses and dumps its addressing and weights. So
// the comparison is against OpenFOAM's own mapping operator, on OpenFOAM's own map.
//
// THE WRITTEN <time> FIELDS CANNOT SERVE AS THIS ORACLE, and that is measured rather than assumed:
// the solver advances the fields after `update()` returns, so the post-map state and the written state
// differ on 13,774 of 24,815 cells, by up to 3.12e-01. The oracle prints the post-MAP values.
//
// usage: test_field_map_vs_openfoam <oracleDump> <expected phase: refine|unrefine>
#include "dynamic_refine_fv_mesh_cpp.cuh"
#include "map_poly_mesh_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf("  %s:   %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// `pre <name> <celli> <v...>` / `post <name> <celli> <v...>`, kept per component so one reader serves
// a volScalarField and a volVectorField alike
struct Fields
{
    std::map<std::string, std::vector<std::vector<scalar>>> pre, post;
    label preCells = -1, postCells = -1;
};

Fields readFields(const std::string& path)
{
    Fields f;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.rfind("[brae] ", 0) != 0) continue;
        std::istringstream is(line.substr(7));
        std::string which;
        is >> which;
        const bool isPre = (which == "pre");
        if (!isPre && which != "post") continue;
        std::string tag;
        is >> tag;
        if (tag == "nCells")
        {
            label n = 0;
            is >> n;
            if (isPre) f.preCells = n;
            else       f.postCells = n;
            continue;
        }
        // a patch row carries two indices, not one -- skip it, this unit is internal values only
        if (tag.size() > 5 && tag.compare(tag.size() - 5, 5, "Patch") == 0) continue;
        if (tag == "cellLevel" || tag == "pointLevel"
         || tag == "nPoints" || tag == "nFaces" || tag == "nInternalFaces") continue;
        label celli = 0;
        if (!(is >> celli)) continue;
        std::vector<scalar> comps;
        scalar x = 0;
        while (is >> x) comps.push_back(x);
        if (comps.empty()) continue;
        auto& dst = isPre ? f.pre[tag] : f.post[tag];
        if (static_cast<std::size_t>(celli) >= dst.size())
        {
            dst.resize(static_cast<std::size_t>(celli) + 1);
        }
        dst[static_cast<std::size_t>(celli)] = comps;
    }
    return f;
}

struct Diff
{
    scalar worst = 0;
    scalar refMax = 0;
    std::size_t nAbove = 0;
};

Diff compare(
    const std::vector<std::vector<scalar>>& mine,
    const std::vector<std::vector<scalar>>& of)
{
    Diff d;
    for (std::size_t i = 0; i < mine.size() && i < of.size(); ++i)
    {
        for (std::size_t c = 0; c < mine[i].size() && c < of[i].size(); ++c)
        {
            const scalar w = std::fabs(mine[i][c] - of[i][c]);
            d.worst = std::fmax(d.worst, w);
            d.refMax = std::fmax(d.refMax, std::fabs(of[i][c]));
            if (w > scalar(0)) ++d.nAbove;
        }
    }
    return d;
}

// mapCellField, per component, so one call serves both field ranks
std::vector<std::vector<scalar>> mapComponentwise(
    const std::vector<std::vector<scalar>>& oldF,
    const CellMapper&                       mapper,
    std::size_t                             nNew)
{
    std::size_t nComp = 0;
    for (const auto& v : oldF) nComp = std::max(nComp, v.size());
    std::vector<std::vector<scalar>> out(nNew, std::vector<scalar>(nComp, scalar(0)));
    for (std::size_t c = 0; c < nComp; ++c)
    {
        std::vector<scalar> o(oldF.size(), scalar(0));
        for (std::size_t i = 0; i < oldF.size(); ++i)
        {
            if (c < oldF[i].size()) o[i] = oldF[i][c];
        }
        std::vector<scalar> m;
        mapCellField(m, o, mapper);
        for (std::size_t i = 0; i < nNew && i < m.size(); ++i) out[i][c] = m[i];
    }
    return out;
}

}   // namespace


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <oracleDump> <refine|unrefine>\n", argv[0]);
        return 2;
    }
    const std::string dumpPath = argv[1];
    const std::string wantPhase = argv[2];

    std::printf("== brae cell mapper vs OpenFOAM: %s ==\n", dumpPath.c_str());

    const MapPolyMesh mpm = readMapPolyMesh(dumpPath);
    const Fields f = readFields(dumpPath);

    std::printf("  phase `%s`, %d old cells -> %zu new; OpenFOAM's own mapper was %s\n",
                mpm.phase.c_str(), (int)mpm.nOldCells, mpm.cellMap.size(),
                mpm.openfoamSaysDirect ? "DIRECT" : "INTERPOLATIVE");
    check("the dump is the phase this arm asked for", mpm.phase == wantPhase);
    check("...and its cell map spans the new mesh",
          mpm.cellMap.size() == static_cast<std::size_t>(f.postCells) && f.postCells > 0);
    check("...from the old one", mpm.nOldCells == f.preCells && f.preCells > 0);

    const CellMapper mapper(mpm, static_cast<label>(mpm.cellMap.size()));
    // brae must reach OpenFOAM's OWN verdict on the branch, not merely reproduce the numbers: the two
    // branches are different arithmetic and picking by luck is the defect this unit exists to catch.
    std::printf("  brae's mapper is %s, %zu cells map from nothing\n",
                mapper.direct() ? "DIRECT" : "INTERPOLATIVE", mapper.insertedObjects().size());
    check("brae takes the branch OpenFOAM took", mapper.direct() == mpm.openfoamSaysDirect);
    if (!mapper.direct())
    {
        check("...and the map carried the old cell volumes, so the weights are volume weights",
              mpm.hasOldCellVolumes() && mpm.mapCarriedOldVolumes);
    }

    bool anyField = false;
    for (const auto& entry : f.pre)
    {
        const std::string& name = entry.first;
        const auto post = f.post.find(name);
        if (post == f.post.end()) continue;
        if (entry.second.size() != static_cast<std::size_t>(f.preCells)) continue;
        // V and V0 are GEOMETRY, not mapped solution fields: V is recomputed from the new mesh and V0
        // is the mapped value with the correction applied on top. They are compared in their own block
        // below, not through the cell mapper.
        if (name == "V" || name == "V0") continue;
        anyField = true;
        const std::vector<std::vector<scalar>> mine =
            mapComponentwise(entry.second, mapper, static_cast<std::size_t>(f.postCells));
        const Diff d = compare(mine, post->second);
        std::printf("  %-14s worst %.4e of %.4e (%zu of %d cells differ)\n",
                    name.c_str(), (double)d.worst, (double)d.refMax, d.nAbove, (int)f.postCells);
        check(("brae's mapped `" + name + "` is OpenFOAM's, cell for cell").c_str(),
              d.worst == scalar(0));
    }
    check("the oracle carried a field to map", anyField);

    // THE OLD-TIME VOLUMES. `mapFields` maps V0 with the same cell mapper and then OVERWRITES it on
    // every cell the change touched, with that cell's CURRENT volume. A gate on mapped field values
    // cannot see this at all -- and getting it wrong leaves every field exactly right while the first
    // ddt after a refinement is out by the split ratio.
    {
        const auto postV = f.post.find("V");
        const auto postV0 = f.post.find("V0");
        if (postV != f.post.end() && postV0 != f.post.end() && mpm.hasOldCellVolumes())
        {
            // V0 as the mapper leaves it: the OLD volumes the map carried, mapped
            std::vector<scalar> mappedV0;
            mapCellField(mappedV0, mpm.oldCellVolumes, mapper);
            std::vector<scalar> V(static_cast<std::size_t>(f.postCells), scalar(0));
            for (std::size_t i = 0; i < V.size() && i < postV->second.size(); ++i)
            {
                if (!postV->second[i].empty()) V[i] = postV->second[i][0];
            }
            const std::vector<scalar> mine = dynamicRefine::correctOldVolumes(mpm, mappedV0, V);

            std::vector<std::vector<scalar>> mineRows(mine.size());
            for (std::size_t i = 0; i < mine.size(); ++i) mineRows[i] = {mine[i]};
            const Diff d = compare(mineRows, postV0->second);
            // ...and how much work the correction did, so a green arm cannot be a green no-op
            std::size_t nMoved = 0;
            for (std::size_t i = 0; i < mine.size() && i < mappedV0.size(); ++i)
            {
                if (mine[i] != mappedV0[i]) ++nMoved;
            }
            std::printf("  V0 correction: worst %.4e of %.4e (%zu cells differ); it moved %zu cells "
                        "off the mapped value\n",
                        (double)d.worst, (double)d.refMax, d.nAbove, nMoved);
            check("brae's corrected old-time volumes are OpenFOAM's, cell for cell",
                  d.worst == scalar(0));
            check("...and the correction actually moved cells, so this arm is not a no-op", nMoved > 0);

            // CONTROL: the correction skipped entirely. If the mapped V0 already equalled OpenFOAM's
            // corrected one there would be nothing here to measure.
            const Diff du = compare([&]{
                std::vector<std::vector<scalar>> r(mappedV0.size());
                for (std::size_t i = 0; i < mappedV0.size(); ++i) r[i] = {mappedV0[i]};
                return r;
            }(), postV0->second);
            std::printf("  CONTROL: the MAPPED V0 with no correction: worst %.4e of %.4e (%zu cells "
                        "differ)\n", (double)du.worst, (double)du.refMax, du.nAbove);
            check("...is a different answer, so the correction is what this arm measures",
                  du.worst > scalar(0));
        }
    }

    // CONTROL 1: the identity map. Under REFINEMENT the mapper is a pure gather and the mesh is mostly
    // cells it does not move -- and on a refined VoF field the eight children of an interface cell all
    // take the same parent value -- so `f[i] = old[i]` is right nearly everywhere. Without this control
    // the refinement arm would pass on a port that never read cellMap at all.
    {
        const auto alpha = f.pre.find("alpha.water");
        const auto post = f.post.find("alpha.water");
        if (alpha != f.pre.end() && post != f.post.end())
        {
            std::vector<std::vector<scalar>> ident(static_cast<std::size_t>(f.postCells));
            for (std::size_t i = 0; i < ident.size(); ++i)
            {
                ident[i] = (i < alpha->second.size()) ? alpha->second[i]
                                                      : std::vector<scalar>{scalar(0)};
            }
            const Diff d = compare(ident, post->second);
            std::printf("  CONTROL: the IDENTITY map instead of cellMap: worst %.4e, %zu cells "
                        "differ of %d\n", (double)d.worst, d.nAbove, (int)f.postCells);
            check("...is a different answer, so the cell map is what this arm measures",
                  d.worst > scalar(0));
        }
    }

    // CONTROL 2: uniform weights. Only live on the interpolative arm, and it is the subtle one: the
    // volume weights of eight equal-volume children are 1/8 to within an ulp -- they print as
    // 0.125000000000000083, 0.124999999999999806 -- so a port that hardcodes 1.0/8.0 is one ulp per
    // child out, which is exactly the size of error an AMR run then amplifies.
    if (!mapper.direct())
    {
        MapPolyMesh uniform = mpm;
        uniform.oldCellVolumes.clear();          // no volumes -> the uniform fallback
        const CellMapper uniformMapper(uniform, static_cast<label>(uniform.cellMap.size()));
        const auto alpha = f.pre.find("alpha.water");
        const auto post = f.post.find("alpha.water");
        if (alpha != f.pre.end() && post != f.post.end())
        {
            const std::vector<std::vector<scalar>> mine =
                mapComponentwise(alpha->second, uniformMapper, static_cast<std::size_t>(f.postCells));
            const Diff d = compare(mine, post->second);
            std::printf("  CONTROL: uniform 1/n weights instead of volume weights: worst %.4e, %zu "
                        "cells differ\n", (double)d.worst, d.nAbove);
            check("...is a different answer, so the volume weighting is measured",
                  d.worst > scalar(0));
        }
    }
    else
    {
        std::printf("  (this arm is DIRECT, so there are no weights and the volume-weight control "
                    "cannot witness anything here)\n");
    }

    std::printf("test_field_map_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
