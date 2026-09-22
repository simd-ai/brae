#include "primitive_patch_cpp.cuh"
#include <unordered_map>

namespace brae {

std::vector<std::vector<label>> meshCells(const PrimitiveMesh& m)
{
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<std::vector<label>> cellFaceAddr(static_cast<std::size_t>(m.nCells()));
    for (std::size_t facei = 0; facei < own.size(); ++facei)
    {
        cellFaceAddr[static_cast<std::size_t>(own[facei])].push_back(static_cast<label>(facei));
    }
    for (std::size_t facei = 0; facei < nei.size(); ++facei)
    {
        cellFaceAddr[static_cast<std::size_t>(nei[facei])].push_back(static_cast<label>(facei));
    }
    return cellFaceAddr;
}

std::vector<std::vector<label>> meshPointFaces(const PrimitiveMesh& m)
{
    std::vector<std::vector<label>> pf(static_cast<std::size_t>(m.nPoints()));
    for (label facei = 0; facei < m.nFaces(); ++facei)
    {
        for (label fp = 0; fp < m.faceSize(facei); ++fp)
        {
            pf[static_cast<std::size_t>(m.faceVert(facei, fp))].push_back(facei);
        }
    }
    return pf;
}

std::vector<std::vector<label>> pointCellsFromPointFaces(
    const PrimitiveMesh& m,
    const std::vector<std::vector<label>>& pointFaces)
{
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const label nIf = m.nInternalFaces();
    std::vector<std::vector<label>> pointCellAddr(pointFaces.size());
    std::vector<char> usedCells(static_cast<std::size_t>(m.nCells()), 0);
    for (std::size_t pointi = 0; pointi < pointFaces.size(); ++pointi)
    {
        std::vector<label>& currCells = pointCellAddr[pointi];
        for (const label facei : pointFaces[pointi])
        {
            // Owner cell - only allow one occurance
            const label o = own[static_cast<std::size_t>(facei)];
            if (!usedCells[static_cast<std::size_t>(o)])
            {
                usedCells[static_cast<std::size_t>(o)] = 1;
                currCells.push_back(o);
            }
            // Neighbour cell - only allow one occurance
            if (facei < nIf)
            {
                const label n = nei[static_cast<std::size_t>(facei)];
                if (!usedCells[static_cast<std::size_t>(n)])
                {
                    usedCells[static_cast<std::size_t>(n)] = 1;
                    currCells.push_back(n);
                }
            }
        }
        for (const label c : currCells)
        {
            usedCells[static_cast<std::size_t>(c)] = 0;
        }
    }
    return pointCellAddr;
}

PrimitivePatchAddressing primitivePatch(
    const PrimitiveMesh& m,
    const std::vector<label>& faces)
{
    PrimitivePatchAddressing pp;
    pp.faces = faces;
    std::unordered_map<label, label> markedPoints;
    markedPoints.reserve(4*faces.size());
    pp.localFaces.resize(faces.size());
    for (std::size_t i = 0; i < faces.size(); ++i)
    {
        const label f = faces[i];
        for (label fp = 0; fp < m.faceSize(f); ++fp)
        {
            const label pointi = m.faceVert(f, fp);
            const auto inserted = markedPoints.emplace(pointi, static_cast<label>(pp.meshPoints.size()));
            if (inserted.second)
            {
                pp.meshPoints.push_back(pointi);
            }
            pp.localFaces[i].push_back(inserted.first->second);
        }
    }
    pp.pointFaces.resize(pp.meshPoints.size());
    for (std::size_t i = 0; i < pp.localFaces.size(); ++i)
    {
        for (const label pointi : pp.localFaces[i])
        {
            pp.pointFaces[static_cast<std::size_t>(pointi)].push_back(static_cast<label>(i));
        }
    }
    return pp;
}

std::vector<label> faceRange(
    label start,
    label size)
{
    std::vector<label> f(static_cast<std::size_t>(size));
    for (label i = 0; i < size; ++i)
    {
        f[static_cast<std::size_t>(i)] = start + i;
    }
    return f;
}

} // namespace brae
