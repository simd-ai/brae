#include "fvc_smooth_cpp.cuh"
#include "primitive_patch_cpp.cuh"   // meshCells: primitiveMesh::cells() in OpenFOAM's order
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace fvc {

namespace {

const char* const WHO = "brae fvc::smooth: ";

constexpr scalar small = 1.0e-15;
constexpr scalar vSmall = 1.0e-300;
constexpr scalar great = 1.0e15;
// FaceCellWaveBase.C:42
constexpr scalar propagationTol = 0.01;

// smoothData
struct SmoothData
{
    scalar value = -great;

    bool valid() const
    {
        return value > -small;
    }
    bool equal(const SmoothData& rhs) const
    {
        return value == rhs.value;
    }
    // smoothDataI.H:31-62
    bool update(
        const SmoothData& svf,
        scalar scale,
        scalar tol)
    {
        if (!valid() || (value < vSmall))
        {
            value = svf.value/scale;
            return true;
        }
        if (svf.value > (1 + tol)*scale*value)
        {
            value = svf.value/scale;
            return true;
        }
        return false;
    }
};

} // namespace


void smooth(
    std::vector<scalar>& field,
    scalar coeff,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches)
{
    for (const FvPatch& p : patches)
    {
        if (p.coupled || p.type == "cyclic" || p.type == "cyclicAMI" || p.type == "cyclicACMI"
         || p.type == "processor")
        {
            throw std::runtime_error(
                std::string(WHO) + "the mesh has the coupled patch '" + p.name + "'. fvcSmooth.C:76-92 "
                "seeds every coupled face and FaceCellWave carries the wave across them "
                "(handleCyclicPatches), which is not ported.");
        }
    }
    const label nC = m.nCells();
    const label nF = m.nFaces();
    const label nIf = m.nInternalFaces();
    if (static_cast<label>(field.size()) != nC)
    {
        throw std::runtime_error(std::string(WHO) + "the field is not the mesh's cells.");
    }
    const std::vector<label>& owner = m.owner();
    const std::vector<label>& neighbour = m.neighbour();
    const std::vector<std::vector<label>> cells = meshCells(m);
    // fvcSmooth.C:50 -- a scalar sum, 1 + coeff
    const scalar maxRatio = 1 + coeff;

    std::vector<SmoothData> faceInfo(static_cast<std::size_t>(nF));
    std::vector<SmoothData> cellInfo(static_cast<std::size_t>(nC));
    std::vector<char> changedFace(static_cast<std::size_t>(nF), 0);
    std::vector<char> changedCell(static_cast<std::size_t>(nC), 0);
    std::vector<label> changedFaces;
    std::vector<label> changedCells;

    // fvcSmooth.C:58-74 and FaceCellWave::setFaceInfo: every internal face across which one side is more
    // than maxRatio times the other, seeded with the LARGER value, in face order
    for (label f = 0; f < nIf; ++f)
    {
        const scalar own = field[static_cast<std::size_t>(owner[f])];
        const scalar nbr = field[static_cast<std::size_t>(neighbour[f])];
        scalar seed = 0;
        if (own > maxRatio*nbr)
        {
            seed = own;
        }
        else if (nbr > maxRatio*own)
        {
            seed = nbr;
        }
        else
        {
            continue;
        }
        faceInfo[static_cast<std::size_t>(f)].value = seed;
        changedFace[static_cast<std::size_t>(f)] = 1;
        changedFaces.push_back(f);
    }

    // fvcSmooth.C:97-103
    for (label c = 0; c < nC; ++c)
    {
        cellInfo[static_cast<std::size_t>(c)].value = field[static_cast<std::size_t>(c)];
    }

    // FaceCellWave::faceToCell: each changed face, owner then neighbour; smoothData::updateCell scales by
    // the tracking data's maxRatio
    auto faceToCell = [&]()
    {
        for (const label f : changedFaces)
        {
            const SmoothData newInfo = faceInfo[static_cast<std::size_t>(f)];
            const label sides[2] = {owner[static_cast<std::size_t>(f)],
                                    f < nIf ? neighbour[static_cast<std::size_t>(f)] : label(-1)};
            for (const label c : sides)
            {
                if (c < 0) continue;
                SmoothData& cur = cellInfo[static_cast<std::size_t>(c)];
                if (cur.equal(newInfo)) continue;
                if (cur.update(newInfo, maxRatio, propagationTol) && !changedCell[static_cast<std::size_t>(c)])
                {
                    changedCell[static_cast<std::size_t>(c)] = 1;
                    changedCells.push_back(c);
                }
            }
            changedFace[static_cast<std::size_t>(f)] = 0;
        }
        changedFaces.clear();
        return static_cast<label>(changedCells.size());
    };
    // FaceCellWave::cellToFace: each changed cell, its faces in mesh.cells() order; updateFace scales by 1
    auto cellToFace = [&]()
    {
        for (const label c : changedCells)
        {
            const SmoothData newInfo = cellInfo[static_cast<std::size_t>(c)];
            for (const label f : cells[static_cast<std::size_t>(c)])
            {
                SmoothData& cur = faceInfo[static_cast<std::size_t>(f)];
                if (cur.equal(newInfo)) continue;
                if (cur.update(newInfo, scalar(1), propagationTol) && !changedFace[static_cast<std::size_t>(f)])
                {
                    changedFace[static_cast<std::size_t>(f)] = 1;
                    changedFaces.push_back(f);
                }
            }
            changedCell[static_cast<std::size_t>(c)] = 0;
        }
        changedCells.clear();
        return static_cast<label>(changedFaces.size());
    };
    // FaceCellWave::iterate, maxIter = mesh.globalData().nTotalCells() (fvcSmooth.C:119); reaching it is
    // FaceCellWave's FatalError
    const label maxIter = nC;
    label iter = 0;
    for (; iter < maxIter; ++iter)
    {
        const label nCells = faceToCell();
        const label nFaces = nCells ? cellToFace() : 0;
        if (!nCells || !nFaces)
        {
            break;
        }
    }
    if (iter >= maxIter)
    {
        throw std::runtime_error(
            std::string(WHO) + "the wave did not settle in nTotalCells sweeps; FaceCellWave stops with "
            "\"Maximum number of iterations reached\" there.");
    }
    // fvcSmooth.C:123-126
    for (label c = 0; c < nC; ++c)
    {
        field[static_cast<std::size_t>(c)] = cellInfo[static_cast<std::size_t>(c)].value;
    }
}

} // namespace fvc
} // namespace cpu
} // namespace brae
