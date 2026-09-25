#include "map_poly_mesh_cpp.cuh"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brae {

namespace {

const char* const WHO = "brae mapPolyMesh: ";

// OF doubleScalar.H:64
constexpr scalar VSMALL = scalar(1.0e-300);

}   // namespace


MapPolyMesh readMapPolyMesh(const std::string& path)
{
    MapPolyMesh mpm;
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error(std::string(WHO) + "cannot open the map dump " + path);
    }
    // tools/dumpRefineMap writes `[brae] map <k> <tag> ...`, one row per entry, so a reader splits on
    // whitespace and never has to model OpenFOAM's `N ( a b ... )`. There is one map per mesh change
    // and the tool reports how many it captured; this reads map 0 and refuses a dump with more.
    std::string line;
    bool sawMap = false;
    label nCaptured = -1;
    while (std::getline(in, line))
    {
        if (line.rfind("[brae] ", 0) != 0) continue;
        std::istringstream is(line.substr(7));
        std::string word;
        is >> word;
        if (word == "nCaptured")
        {
            is >> nCaptured;
            continue;
        }
        if (word != "map") continue;
        label k = -1;
        std::string tag;
        is >> k >> tag;
        if (k != 0) continue;
        sawMap = true;
        if (tag == "nOldCells")
        {
            is >> mpm.nOldCells;
        }
        else if (tag == "phase")
        {
            is >> mpm.phase;
        }
        else if (tag == "hasOldCellVolumes")
        {
            std::string v;
            is >> v;
            mpm.mapCarriedOldVolumes = (v == "true");
        }
        else if (tag == "cellMapperDirect")
        {
            std::string v;
            is >> v;
            mpm.openfoamSaysDirect = (v == "true");
        }
        else if (tag == "cellMap")
        {
            label i = 0, o = 0;
            is >> i >> o;
            if (static_cast<std::size_t>(i) >= mpm.cellMap.size())
            {
                mpm.cellMap.resize(static_cast<std::size_t>(i) + 1, label(-1));
            }
            mpm.cellMap[static_cast<std::size_t>(i)] = o;
        }
        else if (tag == "oldCellVolumes")
        {
            label i = 0;
            scalar v = 0;
            is >> i >> v;
            if (static_cast<std::size_t>(i) >= mpm.oldCellVolumes.size())
            {
                mpm.oldCellVolumes.resize(static_cast<std::size_t>(i) + 1, scalar(0));
            }
            mpm.oldCellVolumes[static_cast<std::size_t>(i)] = v;
        }
        else if (tag == "cellsFromCells")
        {
            // `<newi> <nMasters> <old...>` -- the master ORDER on the row is OpenFOAM's own, and it is
            // the accumulation order of the weighted sum, so it is part of the answer
            label celli = 0, n = 0;
            is >> celli >> n;
            std::vector<label> mo(static_cast<std::size_t>(n));
            for (label& x : mo) is >> x;
            mpm.cellsFromCells.emplace_back(celli, std::move(mo));
        }
        else if (tag == "cellsFromFaces" || tag == "cellsFromEdges" || tag == "cellsFromPoints")
        {
            throw std::runtime_error(
                std::string(WHO) + path + " carries a `" + tag + "` entry. brae's cell mapper models "
                "only the cells-from-cells list, because refinement and unrefinement are its only "
                "producers; a map with any of the other three would silently take the interpolative "
                "branch here with those cells left mapping from nothing.");
        }
    }
    if (!sawMap)
    {
        throw std::runtime_error(
            std::string(WHO) + path + " holds no `map 0` rows: the step it was taken on changed no "
            "topology, so there is no mapping to compare.");
    }
    if (nCaptured > 1)
    {
        throw std::runtime_error(
            std::string(WHO) + path + " captured " + std::to_string(nCaptured) + " mesh changes in one "
            "step. brae reads the first; comparing against a composition of two would be wrong.");
    }
    return mpm;
}


CellMapper::CellMapper(
    const MapPolyMesh& mpm,
    label              nNewCells)
{
    const std::size_t n = static_cast<std::size_t>(nNewCells);
    if (mpm.cellMap.size() != n)
    {
        throw std::runtime_error(
            std::string(WHO) + "the cell map has " + std::to_string(mpm.cellMap.size())
            + " entries where the new mesh has " + std::to_string(nNewCells) + " cells.");
    }

    // cellMapper.C:285-291 -- direct when NO cell was made from several others
    direct_ = mpm.cellsFromCells.empty();

    if (direct_)
    {
        // :54-58 then :70-78 -- an inserted cell reads old cell 0 and is listed
        directAddr_ = mpm.cellMap;
        for (std::size_t i = 0; i < directAddr_.size(); ++i)
        {
            if (directAddr_[i] < 0)
            {
                directAddr_[i] = 0;
                inserted_.push_back(static_cast<label>(i));
            }
        }
        return;
    }

    // :108-112
    addr_.assign(n, std::vector<label>());
    wght_.assign(n, std::vector<scalar>());

    // :122-152 -- UNIFORM weights first, for every cells-from-cells entry
    for (const auto& map : mpm.cellsFromCells)
    {
        const std::size_t celli = static_cast<std::size_t>(map.first);
        const std::vector<label>& mo = map.second;
        if (mo.empty()) continue;                       // :127 safety
        if (!addr_[celli].empty())
        {
            throw std::runtime_error(
                std::string(WHO) + "cell " + std::to_string(map.first) + " is mapped twice. OpenFOAM "
                "aborts on the same thing (cellMapper.C:131-136).");
        }
        addr_[celli] = mo;
        wght_[celli].assign(mo.size(), scalar(1)/static_cast<scalar>(mo.size()));
    }

    // :154-199 -- then OVERWRITTEN with volume weights where the map carried the old volumes. The
    // uniform pass above is not wasted work: it is what sizes wght_ and what the zero-volume
    // exception falls back to.
    if (mpm.hasOldCellVolumes())
    {
        if (static_cast<label>(mpm.oldCellVolumes.size()) != mpm.nOldCells)
        {
            throw std::runtime_error(
                std::string(WHO) + "the map carries " + std::to_string(mpm.oldCellVolumes.size())
                + " old cell volumes where the old mesh had " + std::to_string(mpm.nOldCells)
                + ". OpenFOAM asks the same question and aborts (cellMapper.C:160-168): are the "
                "volumes already mapped?");
        }
        const std::vector<scalar>& V = mpm.oldCellVolumes;
        for (const auto& map : mpm.cellsFromCells)
        {
            const std::size_t celli = static_cast<std::size_t>(map.first);
            const std::vector<label>& mo = map.second;
            if (mo.empty()) continue;
            std::vector<scalar>& w = wght_[celli];
            scalar sumV = 0;
            for (std::size_t ci = 0; ci < mo.size(); ++ci)
            {
                w[ci] = V[static_cast<std::size_t>(mo[ci])];
                sumV += V[static_cast<std::size_t>(mo[ci])];
            }
            if (sumV > VSMALL)
            {
                for (scalar& x : w) x /= sumV;
            }
            else
            {
                // :195-196 -- a zero-volume merge falls back to uniform
                w.assign(mo.size(), scalar(1)/static_cast<scalar>(mo.size()));
            }
        }
    }

    // :205-220 -- and LAST the cells that map from a single old cell, only where nothing set them
    for (std::size_t celli = 0; celli < n; ++celli)
    {
        const label mappedi = mpm.cellMap[celli];
        if (mappedi >= 0 && addr_[celli].empty())
        {
            addr_[celli].assign(1, mappedi);
            wght_[celli].assign(1, scalar(1));
        }
    }

    // :223-256 -- whatever is still empty maps from dummy cell 0
    for (std::size_t celli = 0; celli < n; ++celli)
    {
        if (addr_[celli].empty())
        {
            addr_[celli].assign(1, label(0));
            wght_[celli].assign(1, scalar(1));
            inserted_.push_back(static_cast<label>(celli));
        }
    }
}


const std::vector<label>& CellMapper::directAddressing() const
{
    if (!direct_)
    {
        throw std::runtime_error(
            std::string(WHO) + "directAddressing asked of an INTERPOLATIVE mapper. OpenFOAM aborts "
            "the same way; the two branches are not interchangeable and taking the wrong one is the "
            "defect this unit exists to catch.");
    }
    return directAddr_;
}


const std::vector<std::vector<label>>& CellMapper::addressing() const
{
    if (direct_)
    {
        throw std::runtime_error(
            std::string(WHO) + "interpolative addressing asked of a DIRECT mapper.");
    }
    return addr_;
}


const std::vector<std::vector<scalar>>& CellMapper::weights() const
{
    if (direct_)
    {
        throw std::runtime_error(std::string(WHO) + "weights asked of a DIRECT mapper.");
    }
    return wght_;
}

}   // namespace brae
