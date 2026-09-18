#include "patch_wave_cpp.cuh"
#include "face_cpp.cuh"
#include "foam_dict.cuh"
#include "primitive_patch_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace brae {

namespace {

const char* const WHO = "brae patchWave: ";

// SMALL, GREAT and VGREAT in double precision
constexpr scalar small = 1.0e-15;
constexpr scalar great = 1.0e15;
constexpr scalar vGreat = 1.0e300;

// FaceCellWaveBase::propagationTol_
constexpr scalar propagationTol = 0.01;

// wallPoint: the nearest wall-face centre found so far, and the squared distance to it
struct WallPoint
{
    // wallPoint(): origin_(point::max), distSqr_(-GREAT)
    vector origin{vGreat, vGreat, vGreat};
    scalar distSqr = -great;

    bool valid() const
    {
        return distSqr > -small;
    }
    // wallPoint::equal is operator==, which compares the origins only
    bool equal(const WallPoint& rhs) const
    {
        return origin.x == rhs.origin.x && origin.y == rhs.origin.y && origin.z == rhs.origin.z;
    }
    // wallPoint::update: take w2's origin if it is nearer to pt by more than the tolerance
    bool update(
        const vector& pt,
        const WallPoint& w2,
        scalar tol)
    {
        const scalar dist2 = magSqr(pt - w2.origin);
        if (!valid())
        {
            // current not yet set so use any value
            distSqr = dist2;
            origin = w2.origin;
            return true;
        }
        const scalar diff = distSqr - dist2;
        if (diff < 0)
        {
            // already nearer to pt
            return false;
        }
        if ((diff < small) || ((distSqr > small) && (diff/distSqr < tol)))
        {
            // don't propagate small changes
            return false;
        }
        // update with new values
        distSqr = dist2;
        origin = w2.origin;
        return true;
    }
};

// FaceCellWave<wallPoint>, on a mesh with no coupled patches
class FaceCellWave
{
public:
    FaceCellWave(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<std::vector<label>>& cells)
    :
        m_(m),
        g_(g),
        cells_(cells),
        allFaceInfo_(static_cast<std::size_t>(m.nFaces())),
        allCellInfo_(static_cast<std::size_t>(m.nCells())),
        changedFace_(static_cast<std::size_t>(m.nFaces()), 0),
        changedCell_(static_cast<std::size_t>(m.nCells()), 0)
    {}

    // setFaceInfo(changedFaces, changedFacesInfo)
    void setFaceInfo(
        label facei,
        const WallPoint& faceInfo)
    {
        allFaceInfo_[static_cast<std::size_t>(facei)] = faceInfo;
        // Mark facei as visited and changed (both on list and on face itself)
        changedFace_[static_cast<std::size_t>(facei)] = 1;
        changedFaces_.push_back(facei);
    }

    // iterate(maxIter): the number of iterations taken
    label iterate(label maxIter)
    {
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
        return iter;
    }

    const std::vector<WallPoint>& allFaceInfo() const
    {
        return allFaceInfo_;
    }
    const std::vector<WallPoint>& allCellInfo() const
    {
        return allCellInfo_;
    }

private:
    void updateCell(
        label celli,
        const WallPoint& neighbourInfo,
        scalar tol,
        WallPoint& cellInfo)
    {
        const bool propagate = cellInfo.update(g_.C()[static_cast<std::size_t>(celli)], neighbourInfo, tol);
        if (propagate && !changedCell_[static_cast<std::size_t>(celli)])
        {
            changedCell_[static_cast<std::size_t>(celli)] = 1;
            changedCells_.push_back(celli);
        }
    }

    void updateFace(
        label facei,
        const WallPoint& neighbourInfo,
        scalar tol,
        WallPoint& faceInfo)
    {
        const bool propagate = faceInfo.update(g_.Cf()[static_cast<std::size_t>(facei)], neighbourInfo, tol);
        if (propagate && !changedFace_[static_cast<std::size_t>(facei)])
        {
            changedFace_[static_cast<std::size_t>(facei)] = 1;
            changedFaces_.push_back(facei);
        }
    }

    // Propagate face to cell
    label faceToCell()
    {
        const std::vector<label>& owner = m_.owner();
        const std::vector<label>& neighbour = m_.neighbour();
        const label nInternalFaces = m_.nInternalFaces();
        for (const label facei : changedFaces_)
        {
            // a copy: updating a cell never touches a face, but the reference would be to a vector this
            // loop does not resize either; the copy keeps the two obviously apart
            const WallPoint newInfo = allFaceInfo_[static_cast<std::size_t>(facei)];
            // Owner
            {
                const label celli = owner[static_cast<std::size_t>(facei)];
                WallPoint& currInfo = allCellInfo_[static_cast<std::size_t>(celli)];
                if (!currInfo.equal(newInfo))
                {
                    updateCell(celli, newInfo, propagationTol, currInfo);
                }
            }
            // Neighbour.
            if (facei < nInternalFaces)
            {
                const label celli = neighbour[static_cast<std::size_t>(facei)];
                WallPoint& currInfo = allCellInfo_[static_cast<std::size_t>(celli)];
                if (!currInfo.equal(newInfo))
                {
                    updateCell(celli, newInfo, propagationTol, currInfo);
                }
            }
            // Reset status of face
            changedFace_[static_cast<std::size_t>(facei)] = 0;
        }
        // Handled all changed faces by now
        changedFaces_.clear();
        return static_cast<label>(changedCells_.size());
    }

    // Propagate cell to face
    label cellToFace()
    {
        for (const label celli : changedCells_)
        {
            const WallPoint newInfo = allCellInfo_[static_cast<std::size_t>(celli)];
            // Evaluate all connected faces
            for (const label facei : cells_[static_cast<std::size_t>(celli)])
            {
                WallPoint& currInfo = allFaceInfo_[static_cast<std::size_t>(facei)];
                if (!currInfo.equal(newInfo))
                {
                    updateFace(facei, newInfo, propagationTol, currInfo);
                }
            }
            // Reset status of cell
            changedCell_[static_cast<std::size_t>(celli)] = 0;
        }
        // Handled all changed cells by now
        changedCells_.clear();
        return static_cast<label>(changedFaces_.size());
    }

    const PrimitiveMesh& m_;
    const FvGeometry& g_;
    const std::vector<std::vector<label>>& cells_;
    std::vector<WallPoint> allFaceInfo_;
    std::vector<WallPoint> allCellInfo_;
    std::vector<char> changedFace_;
    std::vector<label> changedFaces_;
    std::vector<char> changedCell_;
    std::vector<label> changedCells_;
};

// cellDistFuncs::smallestDist: the smallest true distance from p to any of wallFaces
scalar smallestDist(
    const PrimitiveMesh& m,
    const vector& p,
    const PrimitivePatchAddressing& wallPatch,
    const std::vector<label>& wallFaces)
{
    scalar minDist = great;
    for (const label patchFacei : wallFaces)
    {
        const scalar d = faceNearestDistance(
            m,
            wallPatch.faces[static_cast<std::size_t>(patchFacei)],
            m.points(),
            p);
        if (d < minDist)
        {
            minDist = d;
        }
    }
    return minDist;
}

// cellDistFuncs::getPointNeighbours: patchFacei and every face of the patch sharing a point with it.
// OpenFOAM lists itself, then its edge neighbours, then its point-only neighbours; smallestDist takes a
// strict minimum over them, which no order changes, so the set is built directly.
void getPointNeighbours(
    const PrimitivePatchAddressing& patch,
    label patchFacei,
    std::vector<label>& neighbours)
{
    neighbours.clear();
    neighbours.push_back(patchFacei);
    for (const label pointi : patch.localFaces[static_cast<std::size_t>(patchFacei)])
    {
        for (const label facei : patch.pointFaces[static_cast<std::size_t>(pointi)])
        {
            if (std::find(neighbours.begin(), neighbours.end(), facei) == neighbours.end())
            {
                neighbours.push_back(facei);
            }
        }
    }
}

// cellDistFuncs::correctBoundaryCells(patchIDs, doPointCells = true, ...), the combined-patch branch
// that useCombinedWallPatch selects by default
void correctBoundaryCells(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const std::vector<label>& sortedPatchIDs,
    std::vector<scalar>& wallDistCorrected)
{
    std::vector<label> faceLabels;
    for (const label patchi : sortedPatchIDs)
    {
        const FvPatch& patch = patches[static_cast<std::size_t>(patchi)];
        for (label i = 0; i < patch.size; ++i)
        {
            faceLabels.push_back(patch.start + i);
        }
    }
    const PrimitivePatchAddressing wallPatch = primitivePatch(m, faceLabels);

    // Correct all cells with face on wall
    const std::vector<vector>& cellCentres = g.C();
    std::unordered_set<label> nearestFace;
    std::vector<label> neighbours;
    label nWalls = 0;
    for (const label patchi : sortedPatchIDs)
    {
        const FvPatch& patch = patches[static_cast<std::size_t>(patchi)];
        for (label patchFacei = 0; patchFacei < patch.size; ++patchFacei)
        {
            getPointNeighbours(wallPatch, nWalls, neighbours);
            const label celli = patch.faceCells[static_cast<std::size_t>(patchFacei)];
            wallDistCorrected[static_cast<std::size_t>(celli)] =
                smallestDist(m, cellCentres[static_cast<std::size_t>(celli)], wallPatch, neighbours);
            // Store wallCell and its nearest neighbour
            nearestFace.insert(celli);
            nWalls++;
        }
    }

    // Correct all cells with a point on the wall. The cells of a point come from primitiveMesh's
    // pointCells(pointi), whose order does not matter here: each cell is corrected once, from the
    // faces of the first wall point met that touches it.
    const std::vector<std::vector<label>> pointFaces = meshPointFaces(m);
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const label nIf = m.nInternalFaces();
    for (std::size_t patchPointi = 0; patchPointi < wallPatch.meshPoints.size(); ++patchPointi)
    {
        const label verti = wallPatch.meshPoints[patchPointi];
        std::vector<label> pointCells;
        for (const label facei : pointFaces[static_cast<std::size_t>(verti)])
        {
            pointCells.push_back(own[static_cast<std::size_t>(facei)]);
            if (facei < nIf)
            {
                pointCells.push_back(nei[static_cast<std::size_t>(facei)]);
            }
        }
        std::sort(pointCells.begin(), pointCells.end());
        pointCells.erase(std::unique(pointCells.begin(), pointCells.end()), pointCells.end());
        for (const label celli : pointCells)
        {
            if (nearestFace.count(celli) == 0)
            {
                wallDistCorrected[static_cast<std::size_t>(celli)] = smallestDist(
                    m,
                    cellCentres[static_cast<std::size_t>(celli)],
                    wallPatch,
                    wallPatch.pointFaces[patchPointi]);
                nearestFace.insert(celli);
            }
        }
    }
}

} // namespace

PatchWave patchWave(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const std::vector<label>& patchIDs,
    bool correctWalls)
{
    for (const FvPatch& p : patches)
    {
        if (isCoupledInterfaceType(p.type))
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + p.name + "` is " + p.type + ". The wave crosses a coupled "
                "patch in FaceCellWave::handleCyclicPatches/handleProcPatches, which are not ported.");
        }
    }
    std::vector<label> sortedIDs = patchIDs;
    std::sort(sortedIDs.begin(), sortedIDs.end());
    sortedIDs.erase(std::unique(sortedIDs.begin(), sortedIDs.end()), sortedIDs.end());

    const std::vector<std::vector<label>> cells = meshCells(m);
    FaceCellWave wave(m, g, cells);

    // setChangedFaces: every face of the patches, in the mesh's patch order, seeded with its centre
    for (std::size_t patchi = 0; patchi < patches.size(); ++patchi)
    {
        if (!std::binary_search(sortedIDs.begin(), sortedIDs.end(), static_cast<label>(patchi))) continue;
        const FvPatch& patch = patches[patchi];
        for (label patchFacei = 0; patchFacei < patch.size; ++patchFacei)
        {
            WallPoint seed;
            seed.origin = g.Cf()[static_cast<std::size_t>(patch.start + patchFacei)];
            seed.distSqr = 0.0;
            wave.setFaceInfo(patch.start + patchFacei, seed);
        }
    }
    // MeshWave<wallPoint>(mesh, changedFaces, faceDist, nTotalCells + 1)
    const label maxIter = m.nCells() + 1;
    const label iter = wave.iterate(maxIter);
    if (iter >= maxIter)
    {
        throw std::runtime_error(std::string(WHO) + "Maximum number of iterations reached. Increase maxIter.");
    }

    // getValues
    PatchWave out;
    const std::vector<WallPoint>& cellInfo = wave.allCellInfo();
    const std::vector<WallPoint>& faceInfo = wave.allFaceInfo();
    out.distance.resize(cellInfo.size());
    for (std::size_t celli = 0; celli < cellInfo.size(); ++celli)
    {
        const scalar dist = cellInfo[celli].distSqr;
        if (cellInfo[celli].valid())
        {
            out.distance[celli] = std::sqrt(dist);
        }
        else
        {
            out.distance[celli] = dist;
            out.nUnset++;
        }
    }
    out.patchDistance.resize(patches.size());
    for (std::size_t patchi = 0; patchi < patches.size(); ++patchi)
    {
        const FvPatch& patch = patches[patchi];
        std::vector<scalar>& patchField = out.patchDistance[patchi];
        patchField.resize(static_cast<std::size_t>(patch.size));
        for (label patchFacei = 0; patchFacei < patch.size; ++patchFacei)
        {
            const WallPoint& w = faceInfo[static_cast<std::size_t>(patch.start + patchFacei)];
            if (w.valid())
            {
                // Adding SMALL to avoid problems with /0 in the turbulence models
                patchField[static_cast<std::size_t>(patchFacei)] = std::sqrt(w.distSqr) + small;
            }
            else
            {
                patchField[static_cast<std::size_t>(patchFacei)] = w.distSqr;
                out.nUnset++;
            }
        }
    }

    // Correct wall cells for true distance
    if (correctWalls)
    {
        correctBoundaryCells(m, g, patches, sortedIDs, out.distance);
    }
    return out;
}

} // namespace brae
