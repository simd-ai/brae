#include "les_delta_cpp.cuh"
#include "primitive_patch_cpp.cuh"   // meshCells: primitiveMesh::cells() in OpenFOAM's order
#include "wedge_patch.cuh"           // wedgePolyPatch::centreNormal
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace LESdelta {

namespace {

const char* const WHO = "brae LESdelta: ";

constexpr scalar small = 1.0e-15;
constexpr scalar vSmall = 1.0e-300;
constexpr scalar rootVSmall = 1.0e-150;
constexpr scalar great = 1.0e15;
// FaceCellWaveBase.C:42
constexpr scalar propagationTol = 0.01;

scalar readDeltaCoeff(const FoamDict& d)
{
    return d.optionalSubDict("cubeRootVolCoeffs")->scalarOr("deltaCoeff", scalar(1));
}

// smoothDelta::deltaData
struct DeltaData
{
    scalar delta = -great;

    bool valid() const
    {
        return delta > -small;
    }
    bool equal(const DeltaData& rhs) const
    {
        return delta == rhs.delta;
    }
    // deltaData::update(w2, scale, tol): take the neighbour's delta over `scale` when mine is unset or
    // the neighbour's is more than (1 + tol)*scale larger
    bool update(
        const DeltaData& w2,
        scalar scale,
        scalar tol)
    {
        if (!valid() || (delta < vSmall))
        {
            delta = w2.delta/scale;
            return true;
        }
        if (w2.delta > (scalar(1) + tol)*scale*delta)
        {
            delta = w2.delta/scale;
            return true;
        }
        return false;
    }
};

} // namespace


Spec read(
    const FoamDict& lesDict,
    const std::string& file)
{
    Spec s;
    const std::string type = lesDict.wordOr("delta", "");
    if (type == "cubeRootVol")
    {
        s.deltaCoeff = readDeltaCoeff(lesDict);
        return s;
    }
    if (type != "smooth")
    {
        throw std::runtime_error(
            std::string(WHO) + file + " asks for LES delta `" + type + "`. cubeRootVol and smooth around "
            "cubeRootVol are ported; every other LESdelta (Prandtl, vanDriest, maxDeltaxyz, IDDESDelta, "
            "...) computes a different filter width and is not substituted.");
    }
    // smoothDelta.C:123-147: the geometric delta is LESdelta::New on `smoothCoeffs` (or the LES
    // dictionary itself), and maxDeltaRatio is read from the same place with no default
    const FoamDict* sc = lesDict.optionalSubDict("smoothCoeffs");
    const std::string inner = sc->wordOr("delta", "");
    if (inner != "cubeRootVol")
    {
        throw std::runtime_error(
            std::string(WHO) + file + " smooths the LES delta `" + inner + "`. smooth is ported around "
            "cubeRootVol only.");
    }
    if (!sc->found("maxDeltaRatio"))
    {
        throw std::runtime_error(
            std::string(WHO) + file + " has `delta smooth` and no `maxDeltaRatio`, which smoothDelta reads "
            "with no default (smoothDelta.C:141).");
    }
    s.smooth = true;
    s.deltaCoeff = readDeltaCoeff(*sc);
    s.maxDeltaRatio = sc->scalarOr("maxDeltaRatio", scalar(0));
    if (!(s.maxDeltaRatio > scalar(0)))
    {
        throw std::runtime_error(std::string(WHO) + file + " has a maxDeltaRatio that is not positive.");
    }
    return s;
}


GeometricDirections geometricDirections(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches)
{
    // polyMesh::calcDirections: solutionD from the EMPTY patches' summed |face area| components, then
    // geometricD = solutionD, OVERWRITTEN wholesale from the WEDGE patches' summed |centreNormal| when there
    // are any. A zero-sized patch of either kind knocks nothing out.
    int solutionD[3] = {1, 1, 1};
    bool hasEmpty = false;
    bool hasWedge = false;
    vector emptyDir{0, 0, 0};
    vector wedgeDir{0, 0, 0};
    for (const FvPatch& p : patches)
    {
        if (p.type == "empty" && p.size > 0)
        {
            hasEmpty = true;
            for (label i = 0; i < p.size; ++i)
            {
                emptyDir.x += std::fabs(p.nf[i].x)*p.magSf[i];
                emptyDir.y += std::fabs(p.nf[i].y)*p.magSf[i];
                emptyDir.z += std::fabs(p.nf[i].z)*p.magSf[i];
            }
        }
        if (p.type == "wedge" && p.size > 0)
        {
            hasWedge = true;
            const vector cn = wedgeGeometry(p).centreNormal;
            wedgeDir.x += std::fabs(cn.x);
            wedgeDir.y += std::fabs(cn.y);
            wedgeDir.z += std::fabs(cn.z);
        }
    }
    auto knockOut = [](const vector& v, int d[3])
    {
        const scalar mv = mag(v);
        const scalar c[3] = {v.x, v.y, v.z};
        for (int k = 0; k < 3; ++k)
        {
            const scalar nk = (mv < rootVSmall) ? scalar(0) : c[k]/mv;
            d[k] = (nk > scalar(1e-6)) ? -1 : 1;
        }
    };
    if (hasEmpty)
    {
        knockOut(emptyDir, solutionD);
    }
    int geometricD[3] = {solutionD[0], solutionD[1], solutionD[2]};
    if (hasWedge)
    {
        knockOut(wedgeDir, geometricD);
    }
    GeometricDirections r;
    r.nD = (geometricD[0] > 0) + (geometricD[1] > 0) + (geometricD[2] > 0);
    if (r.nD == 2)
    {
        // mesh.bounds(): the bounding box of every mesh point
        const std::vector<vector>& pts = m.points();
        vector lo = pts.front();
        vector hi = pts.front();
        for (const vector& q : pts)
        {
            lo = vector{std::fmin(lo.x, q.x), std::fmin(lo.y, q.y), std::fmin(lo.z, q.z)};
            hi = vector{std::fmax(hi.x, q.x), std::fmax(hi.y, q.y), std::fmax(hi.z, q.z)};
        }
        const scalar span[3] = {hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
        for (int k = 0; k < 3; ++k)
        {
            if (geometricD[k] == -1)
            {
                r.thickness = span[k];
                break;
            }
        }
    }
    return r;
}


std::vector<scalar> cubeRootVol(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar deltaCoeff)
{
    const GeometricDirections dirs = geometricDirections(m, patches);
    const std::vector<scalar>& V = g.V();
    std::vector<scalar> delta(V.size());
    if (dirs.nD == 3)
    {
        for (std::size_t c = 0; c < V.size(); ++c)
        {
            delta[c] = deltaCoeff*std::cbrt(V[c]);
        }
        return delta;
    }
    if (dirs.nD == 2)
    {
        if (!(dirs.thickness > scalar(0)))
        {
            throw std::runtime_error(
                std::string(WHO) + "the mesh is 2-D and its bounding box has no span in the knocked-out "
                "direction; cubeRootVolDelta would divide by zero.");
        }
        for (std::size_t c = 0; c < V.size(); ++c)
        {
            delta[c] = deltaCoeff*std::sqrt(V[c]/dirs.thickness);
        }
        return delta;
    }
    // cubeRootVolDelta.C: a 1-D case leaves delta unset (FatalError only under debug); refused here
    throw std::runtime_error(
        std::string(WHO) + "the mesh has " + std::to_string(dirs.nD) + " geometric directions. "
        "cubeRootVolDelta computes a delta for 3-D and 2-D meshes only.");
}


std::vector<scalar> smooth(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const std::vector<scalar>& geometricDelta,
    scalar maxDeltaRatio)
{
    for (const FvPatch& p : patches)
    {
        if (p.coupled || p.type == "cyclic" || p.type == "cyclicAMI" || p.type == "cyclicACMI"
         || p.type == "processor")
        {
            throw std::runtime_error(
                std::string(WHO) + "smooth delta on a mesh with the coupled patch '" + p.name + "' is not "
                "ported: FaceCellWave carries the wave across coupled patches (handleCyclicPatches), and "
                "setChangedFaces seeds every coupled face.");
        }
    }
    const label nC = m.nCells();
    const label nF = m.nFaces();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& owner = m.owner();
    const std::vector<label>& neighbour = m.neighbour();
    const std::vector<std::vector<label>> cells = meshCells(m);

    std::vector<DeltaData> faceInfo(static_cast<std::size_t>(nF));
    std::vector<DeltaData> cellInfo(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        cellInfo[static_cast<std::size_t>(c)].delta = geometricDelta[static_cast<std::size_t>(c)];
    }
    std::vector<char> changedFace(static_cast<std::size_t>(nF), 0);
    std::vector<char> changedCell(static_cast<std::size_t>(nC), 0);
    std::vector<label> changedFaces;
    std::vector<label> changedCells;

    // setChangedFaces + FaceCellWave::setFaceInfo: every internal face across which one side's delta is
    // more than maxDeltaRatio times the other's, seeded with the LARGER one, in face order
    for (label f = 0; f < nIf; ++f)
    {
        const scalar ownDelta = geometricDelta[static_cast<std::size_t>(owner[f])];
        const scalar neiDelta = geometricDelta[static_cast<std::size_t>(neighbour[f])];
        scalar seed = 0;
        if (ownDelta > maxDeltaRatio*neiDelta)
        {
            seed = ownDelta;
        }
        else if (neiDelta > maxDeltaRatio*ownDelta)
        {
            seed = neiDelta;
        }
        else
        {
            continue;
        }
        faceInfo[static_cast<std::size_t>(f)].delta = seed;
        changedFace[static_cast<std::size_t>(f)] = 1;
        changedFaces.push_back(f);
    }

    // FaceCellWave::faceToCell: each changed face, owner then neighbour; deltaData::updateCell scales by
    // the tracking data, which is maxDeltaRatio
    auto faceToCell = [&]()
    {
        for (const label f : changedFaces)
        {
            const DeltaData newInfo = faceInfo[static_cast<std::size_t>(f)];
            const label sides[2] = {owner[static_cast<std::size_t>(f)],
                                    f < nIf ? neighbour[static_cast<std::size_t>(f)] : label(-1)};
            for (const label c : sides)
            {
                if (c < 0) continue;
                DeltaData& cur = cellInfo[static_cast<std::size_t>(c)];
                if (cur.equal(newInfo)) continue;
                if (cur.update(newInfo, maxDeltaRatio, propagationTol) && !changedCell[static_cast<std::size_t>(c)])
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
            const DeltaData newInfo = cellInfo[static_cast<std::size_t>(c)];
            for (const label f : cells[static_cast<std::size_t>(c)])
            {
                DeltaData& cur = faceInfo[static_cast<std::size_t>(f)];
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
    // FaceCellWave::iterate, maxIter = nTotalCells + 1 (smoothDelta.C:106)
    const label maxIter = nC + 1;
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
        throw std::runtime_error(std::string(WHO) + "the smoothing wave did not settle in nCells + 1 sweeps.");
    }
    std::vector<scalar> delta(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        delta[static_cast<std::size_t>(c)] = cellInfo[static_cast<std::size_t>(c)].delta;
    }
    return delta;
}


std::vector<scalar> compute(
    const Spec& s,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const std::vector<scalar> geometric = cubeRootVol(m, g, patches, s.deltaCoeff);
    if (!s.smooth)
    {
        return geometric;
    }
    return smooth(m, patches, geometric, s.maxDeltaRatio);
}

} // namespace LESdelta
} // namespace cpu
} // namespace brae
