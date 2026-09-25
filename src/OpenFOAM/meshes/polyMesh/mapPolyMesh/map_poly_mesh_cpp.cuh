// OpenFOAM's mapPolyMesh and cellMapper: how a field's CELL VALUES move when the mesh changes under
// it. This is the one part of adaptive refinement that decides whether the answer is right, and both
// its input and its output can be read from OpenFOAM -- so it is gated before the machinery that
// produces the map exists.
//
// OF v2412: src/OpenFOAM/meshes/polyMesh/mapPolyMesh/mapPolyMesh.H
//           src/OpenFOAM/meshes/polyMesh/mapPolyMesh/cellMapper/cellMapper.C:50-265
//           src/OpenFOAM/fields/Fields/Field/Field.C:318-333 (direct) and :370-379 (interpolative)
//
// THE TWO STEPS TAKE DIFFERENT BRANCHES, and that is the whole point of the unit:
//   * REFINEMENT adds cells from a master and leaves every `cellsFrom*Map` empty, so the mapper is
//     DIRECT: a pure gather, `f[i] = old[cellMap[i]]`, no arithmetic. Each of eight children takes its
//     parent's value verbatim -- piecewise constant and NOT conservative. Bit-exact.
//   * UNREFINEMENT fills `cellsFromCellsMap`, so the mapper is INTERPOLATIVE for the WHOLE field, not
//     just the merged cells: an untouched cell maps from itself with weight 1, a merged cell from its
//     masters with VOLUME weights. The accumulation order and the old volumes both set the last digit.
//
// A port that used one branch everywhere would look plausible and be wrong in one direction only.
//
// WHAT THIS IS NOT. `dynamicRefineFvMesh::mapFields` does two further things the cell mapper does not,
// and they are what break a running solver rather than a single mapped field: the V0 correction
// (dynamicRefineFvMesh.C:204-252 -- a split or merged cell takes the CURRENT volume, not the mapped
// old one, or the next ddt is wrong by the split ratio) and the flux correction over faceMap
// (:256-424, `correctFluxes`). Neither is here, and a gate on cell values cannot see either.
#pragma once

#include "cf_types.cuh"
#include <string>
#include <utility>
#include <vector>

namespace brae {

// mapPolyMesh.H, as much of it as a cell mapper reads. A NEGATIVE cellMap entry is an INSERTED cell --
// one that maps from nothing; a negative reverseCellMap entry is a cell that is gone.
struct MapPolyMesh
{
    label                                             nOldCells = 0;
    std::vector<label>                                cellMap;          // new -> old, -1 inserted
    std::vector<label>                                reverseCellMap;   // old -> new
    // cellsFromCellsMap: for each new cell, the OLD cells it was made from, in OpenFOAM's own order --
    // which is the accumulation order of the weighted sum, so it is part of the answer
    std::vector<std::pair<label, std::vector<label>>> cellsFromCells;
    // the old mesh's cell volumes, as the map carried them. EMPTY means the map had none and the
    // weights stay uniform. This must come from the ORACLE and not from brae's own FvGeometry::V():
    // brae's V is validated against OpenFOAM's but is FP-conditioning-limited rather than bit-exact
    // (tests/test_mesh_geometry.cu), and feeding it in here would put brae's volume round-off into the
    // gate's floor and make a mapper defect indistinguishable from it.
    std::vector<scalar>                               oldCellVolumes;
    // what OpenFOAM itself said about this map, for the gate to check brae's own predicate against
    // rather than assume it: `refine` or `unrefine`, and whether ITS cellMapper came out direct
    std::string                                       phase;
    bool                                              mapCarriedOldVolumes = false;
    bool                                              openfoamSaysDirect = false;

    bool hasOldCellVolumes() const
    {
        return !oldCellVolumes.empty();
    }
};

// The oracle's dump, as `[brae] <tag> ...` rows.
MapPolyMesh readMapPolyMesh(const std::string& path);

// cellMapper.C:50-265. Built once from the map; `direct()` decides which of the two addressings the
// field map below reads.
class CellMapper
{
public:
    CellMapper(
        const MapPolyMesh& mpm,
        label              nNewCells);

    // cellMapper.C:285-291 -- ALL FOUR cellsFrom*Map lists empty. brae carries only the cells-from-cells
    // list, because refinement and unrefinement are the only producers here; a map that filled the
    // point, edge or face lists is refused where it is read.
    bool direct() const
    {
        return direct_;
    }
    // the direct branch: cellMap, with every INSERTED cell forced to read old cell 0 (:75)
    const std::vector<label>& directAddressing() const;
    // the interpolative branch
    const std::vector<std::vector<label>>& addressing() const;
    const std::vector<std::vector<scalar>>& weights() const;
    // the cells that map from nothing, ascending
    const std::vector<label>& insertedObjects() const
    {
        return inserted_;
    }

private:
    bool                             direct_ = false;
    std::vector<label>               directAddr_;
    std::vector<std::vector<label>>  addr_;
    std::vector<std::vector<scalar>> wght_;
    std::vector<label>               inserted_;
};

// Field.C:318-333 and :370-379, in OpenFOAM's own operand and accumulation order. The direct branch
// leaves a cell whose address is negative UNTOUCHED (there are none after the mapper forces them to
// 0); the interpolative branch starts each cell at zero and accumulates in addressing order.
template <typename T>
void mapCellField(
    std::vector<T>&       f,
    const std::vector<T>& oldF,
    const CellMapper&     mapper)
{
    if (mapper.direct())
    {
        const std::vector<label>& a = mapper.directAddressing();
        f.assign(a.size(), T{});
        if (oldF.empty()) return;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            const label mapI = a[i];
            if (mapI >= 0)
            {
                f[i] = oldF[static_cast<std::size_t>(mapI)];
            }
        }
        return;
    }
    const std::vector<std::vector<label>>&  a = mapper.addressing();
    const std::vector<std::vector<scalar>>& w = mapper.weights();
    f.assign(a.size(), T{});
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        T v{};
        for (std::size_t j = 0; j < a[i].size(); ++j)
        {
            v = v + w[i][j]*oldF[static_cast<std::size_t>(a[i][j])];
        }
        f[i] = v;
    }
}

}   // namespace brae
