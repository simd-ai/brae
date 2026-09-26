// removeFaces::compatibleRemoves. See remove_faces_cpp.cuh.
#include "remove_faces_cpp.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace removeFaces {

namespace {

constexpr const char* WHO = "brae removeFaces::compatibleRemoves: ";

// removeFaces::changeCellRegion (:56-78). OpenFOAM recurses over cellCells; this walks the same
// component with an explicit stack, which reaches the SAME SET of cells: the function assigns one
// fixed newRegion to the connected component of oldRegion that contains celli, so no traversal order
// can change which cells end up in it or what they are set to. The stack is used because the component
// is a merged cell region and OpenFOAM's own comment only hopes it is small ("Can be recursive since
// hopefully only small area of faces removed in one go").
void changeCellRegion(
    const std::vector<std::vector<label>>& cellCells,
    label                                  celli,
    label                                  oldRegion,
    label                                  newRegion,
    std::vector<label>&                    cellRegion)
{
    if (cellRegion[static_cast<std::size_t>(celli)] != oldRegion) return;
    std::vector<label> todo;
    todo.push_back(celli);
    while (!todo.empty())
    {
        const label c = todo.back();
        todo.pop_back();
        if (cellRegion[static_cast<std::size_t>(c)] != oldRegion) continue;
        cellRegion[static_cast<std::size_t>(c)] = newRegion;
        for (const label nbr : cellCells[static_cast<std::size_t>(c)])
        {
            if (cellRegion[static_cast<std::size_t>(nbr)] == oldRegion) todo.push_back(nbr);
        }
    }
}

}   // namespace

label compatibleRemoves(
    const PrimitiveMesh&                   m,
    const std::vector<std::vector<label>>& cellCells,
    const std::vector<label>&              facesToRemove,
    std::vector<label>&                    cellRegion,
    std::vector<label>&                    regionMaster,
    std::vector<label>&                    newFacesToRemove)
{
    if (cellCells.size() != static_cast<std::size_t>(m.nCells()))
        throw std::runtime_error(
            std::string(WHO) + "the cellCells addressing has "
            + std::to_string(cellCells.size()) + " rows, not the mesh's "
            + std::to_string(m.nCells()) + ". The region flood walks it, so a stale one would stop in "
            "the wrong place silently.");

    const std::vector<label>& faceOwner = m.owner();
    const std::vector<label>& faceNeighbour = m.neighbour();

    // :591-596
    cellRegion.assign(static_cast<std::size_t>(m.nCells()), label(-1));
    regionMaster.assign(static_cast<std::size_t>(m.nCells()), label(-1));

    label nRegions = 0;

    // :600-683. One pass over the requested faces, growing and merging regions.
    for (const label facei : facesToRemove)
    {
        if (facei >= m.nInternalFaces())
            throw std::runtime_error(
                std::string(WHO) + "face " + std::to_string(facei) + " is not internal (the mesh has "
                + std::to_string(m.nInternalFaces()) + " internal faces). OpenFOAM FatalErrors here "
                "too (:605-610): only an internal face has two cells to merge.");

        const label own = faceOwner[static_cast<std::size_t>(facei)];
        const label nei = faceNeighbour[static_cast<std::size_t>(facei)];

        const label region0 = cellRegion[static_cast<std::size_t>(own)];
        const label region1 = cellRegion[static_cast<std::size_t>(nei)];

        if (region0 == -1)
        {
            if (region1 == -1)
            {
                // a new region, and its master is the OWNER -- which is the lower-numbered of the two,
                // because that is what owning an internal face means
                cellRegion[static_cast<std::size_t>(own)] = nRegions;
                cellRegion[static_cast<std::size_t>(nei)] = nRegions;
                regionMaster[static_cast<std::size_t>(nRegions)] = own;
                ++nRegions;
            }
            else
            {
                cellRegion[static_cast<std::size_t>(own)] = region1;
                regionMaster[static_cast<std::size_t>(region1)] =
                    std::min(own, regionMaster[static_cast<std::size_t>(region1)]);
            }
        }
        else
        {
            if (region1 == -1)
            {
                // nei is higher numbered than own, so it cannot be below region0's master and the
                // master is left alone -- OpenFOAM says so in its own comment (:643-644)
                cellRegion[static_cast<std::size_t>(nei)] = region0;
            }
            else if (region0 != region1)
            {
                // both sides already have a region: keep the LOWER-numbered region and the lower master
                label freedRegion = -1;
                label keptRegion = -1;
                if (region0 < region1)
                {
                    changeCellRegion(cellCells, nei, region1, region0, cellRegion);
                    keptRegion = region0;
                    freedRegion = region1;
                }
                else if (region1 < region0)
                {
                    changeCellRegion(cellCells, own, region0, region1, cellRegion);
                    keptRegion = region1;
                    freedRegion = region0;
                }
                const label master0 = regionMaster[static_cast<std::size_t>(region0)];
                const label master1 = regionMaster[static_cast<std::size_t>(region1)];
                regionMaster[static_cast<std::size_t>(freedRegion)] = -1;
                regionMaster[static_cast<std::size_t>(keptRegion)] = std::min(master0, master1);
            }
        }
    }

    regionMaster.resize(static_cast<std::size_t>(nRegions));

    // :687-727. OpenFOAM's own two checks, and they are what make the incremental min() above sound
    // rather than hopeful: the master must be the lowest-numbered cell of its region, and a region of
    // one cell means a requested face did not merge anything.
    {
        std::vector<label> nCells(static_cast<std::size_t>(nRegions), label(0));
        for (std::size_t celli = 0; celli < cellRegion.size(); ++celli)
        {
            const label r = cellRegion[celli];
            if (r == -1) continue;
            ++nCells[static_cast<std::size_t>(r)];
            if (static_cast<label>(celli) < regionMaster[static_cast<std::size_t>(r)])
                throw std::runtime_error(
                    std::string(WHO) + "cell " + std::to_string(celli) + " is in region "
                    + std::to_string(r) + " whose master is "
                    + std::to_string(regionMaster[static_cast<std::size_t>(r)])
                    + ", so the master is not the lowest-numbered cell of the region. OpenFOAM "
                    "FatalErrors here too (:699-707).");
        }
        for (std::size_t region = 0; region < nCells.size(); ++region)
        {
            if (nCells[region] == 1)
                throw std::runtime_error(
                    std::string(WHO) + "region " + std::to_string(region) + " has only 1 cell in it. "
                    "OpenFOAM FatalErrors here too (:713-721).");
        }
    }

    // :731-737
    label nUsedRegions = 0;
    for (const label master : regionMaster)
    {
        if (master != -1) ++nUsedRegions;
    }

    // :740-755. THE RECOUNT, and it is a walk over the internal faces rather than over the input: any
    // internal face whose two cells ended in the same region has the same cell on both sides once the
    // merge is played, so it must go too. That also means the answer is in ASCENDING FACE ORDER and can
    // contain faces the caller never asked about.
    std::vector<label> allFacesToRemove;
    allFacesToRemove.reserve(facesToRemove.size());
    for (label facei = 0; facei < m.nInternalFaces(); ++facei)
    {
        const label own = faceOwner[static_cast<std::size_t>(facei)];
        const label nei = faceNeighbour[static_cast<std::size_t>(facei)];
        if (cellRegion[static_cast<std::size_t>(own)] != -1
            && cellRegion[static_cast<std::size_t>(own)] == cellRegion[static_cast<std::size_t>(nei)])
        {
            allFacesToRemove.push_back(facei);
        }
    }
    newFacesToRemove.swap(allFacesToRemove);

    return nUsedRegions;
}

} // namespace removeFaces
} // namespace cpu
} // namespace brae
