// primitiveMesh::calcCellCells. See mesh_cell_cells_cpp.cuh.
#include "mesh_cell_cells_cpp.cuh"

namespace brae {

std::vector<std::vector<label>> buildCellCells(const PrimitiveMesh& m)
{
    // :62-71. Count the internal faces at each cell first, so every row is sized once.
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::size_t nCells = static_cast<std::size_t>(m.nCells());
    std::vector<label> ncc(nCells, label(0));
    for (std::size_t facei = 0; facei < nei.size(); ++facei)
    {
        ++ncc[static_cast<std::size_t>(own[facei])];
        ++ncc[static_cast<std::size_t>(nei[facei])];
    }
    std::vector<std::vector<label>> out(nCells);
    for (std::size_t celli = 0; celli < nCells; ++celli)
    {
        out[celli].resize(static_cast<std::size_t>(ncc[celli]));
        ncc[celli] = 0;                                  // reused as the fill counter, as OpenFOAM does
    }
    // :84-92. One pass over the internal faces, appending the partner at both ends.
    for (std::size_t facei = 0; facei < nei.size(); ++facei)
    {
        const label ownCelli = own[facei];
        const label neiCelli = nei[facei];
        out[static_cast<std::size_t>(ownCelli)]
           [static_cast<std::size_t>(ncc[static_cast<std::size_t>(ownCelli)]++)] = neiCelli;
        out[static_cast<std::size_t>(neiCelli)]
           [static_cast<std::size_t>(ncc[static_cast<std::size_t>(neiCelli)]++)] = ownCelli;
    }
    return out;
}

} // namespace brae
