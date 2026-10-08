#pragma once
// polyMesh::findCell(p) in its default mode, CELL_TETS: WHICH CELL HOLDS A POINT
// (src/OpenFOAM/meshes/polyMesh/polyMesh.C:1517-1593, polyMeshTetDecomposition/).
//
// What OpenFOAM does, end to end:
//   findCell(p, CELL_TETS)        -> findCellFacePt -> cellTree().findInside(p)          polyMesh.C:1366-1388
//   indexedOctree::findInside     the leaf whose box holds p, then the leaf's cells IN THEIR ORDER, the first
//                                 one that `contains` p                                   indexedOctree.C:2648
//   treeDataCell::contains        polyMesh::pointInCell(p, celli, CELL_TETS)              treeDataCell.C:279
//   pointInCell, CELL_TETS        polyMeshTetDecomposition::findTet finds a tet of the cell that p is inside
//   findTet                       the cell's faces, each fanned from its BASE POINT into tets whose apex is
//                                 the cell centre (polyMeshTetDecomposition.C:594, tetIndicesI.H:54-147)
//   tetrahedron::inside           p is OUT only if it is more than SMALL beyond one of the four planes, by
//                                 the plane's UNIT normal (tetrahedronI.H:522)
//   polyMesh::tetBasePtIs         the base point of every face: the first face point whose fan makes tets
//                                 of quality above sqr(SMALL) from BOTH cells (findFaceBasePts, :224)
//
// A POINT ON A FACE IS IN BOTH CELLS -- the test's tolerance is on the far side of each plane -- and the
// octree's order decides. laminar/damBreak's usual probe (0.292 0.05 0.0073) is one: x = 0.292 is the plane
// between two blocks. The octree starts from identity(nCells) (indexedOctree.C:2175) and divide() keeps the
// order of what it is handed (:38-64), so a leaf's cells are in ascending order; a leaf holds every cell whose
// box overlaps its own (treeDataCell::overlaps, inclusive: boundBox.H box_box_overlaps), and the leaf found
// for p is one whose box holds p. So findInside returns THE LOWEST-NUMBERED CELL THAT HOLDS p, and that is
// what this file computes, over all cells, with no octree. The two can part only where p is within the test's
// 1e-15 of a leaf's own plane, a plane at a position drawn from Random(261782) (polyMesh.C:950-952).
//
// NOT PORTED, refused by name where a search meets it:
//   * a face on a COUPLED patch with more than three points. Its base point is chosen with the cell centre
//     across the coupling (findFaceBasePts, syncTools::swapBoundaryFacePositions), which brae does not form.
//     Refused only for a point whose candidate cells have such a face.
//   * a `tetBasePtIs` file beside the mesh (polyMesh.C:153): the base points are then the file's. The
//     caller looks for it.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

class PolyMeshTetSearch
{
public:
    // The mesh's cells and tetBasePtIs. ONCE for a mesh's topology: OpenFOAM keeps the base points across
    // movePoints (polyMesh.C:1236-1240, only the cell tree is reset, :1279) and drops them on a topology
    // change (polyMeshUpdate.C:54). `C` is the cell centres of the mesh as it stands at the first search.
    void build(
        const PrimitiveMesh& m,
        const std::vector<vector>& C)
    {
        const label nCells = m.nCells();
        const label nFaces = m.nFaces();
        const label nInternal = m.nInternalFaces();
        // primitiveMesh::calcCells (primitiveMeshCells.C:32-98): a cell's faces are those it owns, then
        // those it neighbours
        cellFaceOffsets_.assign(static_cast<std::size_t>(nCells) + 1, 0);
        for (label f = 0; f < nFaces; ++f)
        {
            ++cellFaceOffsets_[m.owner()[f] + 1];
        }
        for (label f = 0; f < nInternal; ++f)
        {
            ++cellFaceOffsets_[m.neighbour()[f] + 1];
        }
        for (label c = 0; c < nCells; ++c)
        {
            cellFaceOffsets_[c + 1] += cellFaceOffsets_[c];
        }
        cellFaces_.assign(static_cast<std::size_t>(cellFaceOffsets_[nCells]), 0);
        std::vector<label> fill(cellFaceOffsets_.begin(), cellFaceOffsets_.end() - 1);
        for (label f = 0; f < nFaces; ++f)
        {
            cellFaces_[fill[m.owner()[f]]++] = f;
        }
        for (label f = 0; f < nInternal; ++f)
        {
            cellFaces_[fill[m.neighbour()[f]]++] = f;
        }

        // polyMeshTetDecomposition::findFaceBasePts with tol = minTetQuality = sqr(SMALL)
        tetBasePtIs_.assign(static_cast<std::size_t>(nFaces), -1);
        coupledFace_.assign(static_cast<std::size_t>(nFaces), 0);
        for (label f = 0; f < nInternal; ++f)
        {
            tetBasePtIs_[f] = findSharedBasePoint(m, f, C[m.owner()[f]], C[m.neighbour()[f]]);
        }
        for (const PatchInfo& patch : m.patches())
        {
            const bool coupled = coupledType(patch.type);
            for (label f = patch.start; f < patch.start + patch.size; ++f)
            {
                if (coupled)
                {
                    // the base point needs the centre across the coupling: marked, and met in findCell
                    coupledFace_[f] = m.faceSize(f) > 3 ? 1 : 0;
                    tetBasePtIs_[f] = 0;
                }
                else
                {
                    tetBasePtIs_[f] = findBasePoint(m, f, C[m.owner()[f]]);
                }
            }
        }
        built_ = true;
    }

    bool built() const
    {
        return built_;
    }

    // polyMesh::findCell(p) for each of `points`, on the mesh as it stands: -1 where no cell holds it.
    // `who` names the caller in a refusal.
    std::vector<label> findCells(
        const PrimitiveMesh& m,
        const std::vector<vector>& C,
        const std::vector<vector>& points,
        const std::string& who) const
    {
        const label nCells = m.nCells();
        // each cell's box (primitiveMesh::cellBb: its points' bounds), widened by a hundredth of its
        // diagonal: ONLY a first cut before the tet test, far inside OpenFOAM's own, which is a whole
        // octree leaf. A point the tet test takes is within its 1e-15 of the cell.
        std::vector<vector> lo(static_cast<std::size_t>(nCells), vector{1e300, 1e300, 1e300});
        std::vector<vector> hi(static_cast<std::size_t>(nCells), vector{-1e300, -1e300, -1e300});
        for (label c = 0; c < nCells; ++c)
        {
            for (label k = cellFaceOffsets_[c]; k < cellFaceOffsets_[c + 1]; ++k)
            {
                const label f = cellFaces_[k];
                for (label v = 0; v < m.faceSize(f); ++v)
                {
                    const vector& x = m.points()[m.faceVert(f, v)];
                    lo[c] = vector{std::min(lo[c].x, x.x), std::min(lo[c].y, x.y), std::min(lo[c].z, x.z)};
                    hi[c] = vector{std::max(hi[c].x, x.x), std::max(hi[c].y, x.y), std::max(hi[c].z, x.z)};
                }
            }
            const scalar pad = scalar(0.01)*mag(hi[c] - lo[c]);
            lo[c] = lo[c] - vector{pad, pad, pad};
            hi[c] = hi[c] + vector{pad, pad, pad};
        }

        std::vector<label> found(points.size(), -1);
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            const vector& p = points[i];
            if (control("nearestCentre"))
            {
                // primitiveMesh::findNearestCell (primitiveMeshFindCell.C:92-116)
                label nearest = 0;
                for (label c = 1; c < nCells; ++c)
                {
                    if (magSqr(C[c] - p) < magSqr(C[nearest] - p))
                    {
                        nearest = c;
                    }
                }
                found[i] = nearest;
                continue;
            }
            for (label c = 0; c < nCells; ++c)
            {
                if (p.x < lo[c].x || p.x > hi[c].x || p.y < lo[c].y || p.y > hi[c].y
                    || p.z < lo[c].z || p.z > hi[c].z)
                {
                    continue;
                }
                refuseCoupledCandidate(m, c, p, who);
                if (found[i] < 0 && pointInCell(m, C, c, p))
                {
                    // the lowest-numbered cell that holds it; the cells after it are still looked at for
                    // a coupled face, which could have made one of THEM the lowest
                    found[i] = c;
                }
            }
        }
        return found;
    }

    // polyMesh::pointInCell(p, celli, CELL_TETS): findTet finds a tet (polyMeshTetDecomposition.C:594-634)
    bool pointInCell(
        const PrimitiveMesh& m,
        const std::vector<vector>& C,
        label celli,
        const vector& p) const
    {
        for (label k = cellFaceOffsets_[celli]; k < cellFaceOffsets_[celli + 1]; ++k)
        {
            const label facei = cellFaces_[k];
            const label n = m.faceSize(facei);
            for (label tetPti = 1; tetPti < n - 1; ++tetPti)
            {
                // tetIndices::faceTriIs and ::tet (tetIndicesI.H): a face with no base point takes 0
                label faceBasePtI = tetBasePtIs_[facei];
                if (faceBasePtI < 0)
                {
                    faceBasePtI = 0;
                }
                const label facePtI = (tetPti + faceBasePtI) % n;
                const label faceOtherPtI = (facePtI + 1) % n;
                const vector& a = C[celli];
                const vector& b = m.points()[m.faceVert(facei, faceBasePtI)];
                const bool own = m.owner()[facei] == celli;
                // the neighbour's is the flipped face
                const vector& c = m.points()[m.faceVert(facei, own ? facePtI : faceOtherPtI)];
                const vector& d = m.points()[m.faceVert(facei, own ? faceOtherPtI : facePtI)];
                if (inside(a, b, c, d, p))
                {
                    return true;
                }
            }
        }
        return false;
    }

    label tetBasePt(label facei) const
    {
        return tetBasePtIs_[facei];
    }

    // BRAE_CONTROL_PROBE_SEARCH names ONE rule of the search to break -- the gates' controls, not user
    // switches:
    //   nearestCentre  the cell whose centre is nearest, primitiveMesh::findNearestCell: where a point on a
    //                  face goes if the octree's order is not what decides
    static bool control(const char* part)
    {
        static const char* const set = std::getenv("BRAE_CONTROL_PROBE_SEARCH");
        return set && std::string(set) == part;
    }

private:
    // doubleScalar.H:58-65
    static constexpr scalar kSmall = 1.0e-15;
    static constexpr scalar kGreat = 1.0e+15;
    static constexpr scalar kRootVSmall = 1.0e-150;

    // every polyPatch that is a coupledPolyPatch answers coupled() true: the cyclics and the processors
    static bool coupledType(const std::string& type)
    {
        return isCoupledInterfaceType(type) || type.find("cyclic") != std::string::npos
            || type.find("Cyclic") != std::string::npos || type.find("processor") != std::string::npos;
    }

    // triangle::unitNormal (triangleI.H:200-210): the area normal, normalise(ROOTVSMALL)
    static vector unitNormal(
        const vector& p0,
        const vector& p1,
        const vector& p2)
    {
        const vector n = scalar(0.5)*cross(p1 - p0, p2 - p0);
        const scalar s = mag(n);
        if (s < kRootVSmall)
        {
            return vector{0, 0, 0};
        }
        return n/s;
    }

    // tetrahedron::inside (tetrahedronI.H:522-576): "assuming that the point is in the tet unless definitively
    // shown otherwise by obtaining a positive dot product greater than a tolerance of SMALL"
    static bool inside(
        const vector& a,
        const vector& b,
        const vector& c,
        const vector& d,
        const vector& p)
    {
        if (dot(p - b, unitNormal(b, c, d)) > kSmall)
        {
            return false;
        }
        if (dot(p - a, unitNormal(a, d, c)) > kSmall)
        {
            return false;
        }
        if (dot(p - a, unitNormal(a, b, d)) > kSmall)
        {
            return false;
        }
        if (dot(p - a, unitNormal(a, c, b)) > kSmall)
        {
            return false;
        }
        return true;
    }

    // tetrahedron::quality, ::mag and ::circumRadius (tetrahedronI.H:319, :259, :293)
    static scalar quality(
        const vector& a,
        const vector& b,
        const vector& c,
        const vector& d)
    {
        const scalar volume = (scalar(1)/scalar(6))*dot(cross(b - a, c - a), d - a);
        const vector va = b - a;
        const vector vb = c - a;
        const vector vc = d - a;
        const scalar lambda = magSqr(vc) - dot(va, vc);
        const scalar mu = magSqr(vb) - dot(va, vb);
        const vector ba = cross(vb, va);
        const vector ca = cross(vc, va);
        const vector num = lambda*ba - mu*ca;
        const scalar denom = dot(vc, ba);
        scalar radius = kGreat;
        if (!(std::fabs(denom) < kRootVSmall))
        {
            radius = mag(scalar(0.5)*(va + num/denom));
        }
        const scalar r = std::min(radius, kGreat);
        return volume/(scalar(8)/(scalar(9)*std::sqrt(scalar(3)))*r*r*r + kRootVSmall);
    }

    // polyMeshTetDecomposition::minQuality (polyMeshTetDecomposition.C:41-91): the fan of the face from
    // `faceBasePtI`, the least quality of its tets seen from `cC`
    static scalar minQuality(
        const PrimitiveMesh& m,
        const vector& cC,
        label fI,
        bool isOwner,
        label faceBasePtI)
    {
        const label n = m.faceSize(fI);
        const vector& tetBasePt = m.points()[m.faceVert(fI, faceBasePtI)];
        scalar thisBaseMinTetQuality = scalar(1.0e+300);
        for (label tetPtI = 1; tetPtI < n - 1; ++tetPtI)
        {
            const label facePtI = (tetPtI + faceBasePtI) % n;
            const label otherFacePtI = (facePtI + 1) % n;
            const vector& pA = m.points()[m.faceVert(fI, isOwner ? facePtI : otherFacePtI)];
            const vector& pB = m.points()[m.faceVert(fI, isOwner ? otherFacePtI : facePtI)];
            thisBaseMinTetQuality = std::min(thisBaseMinTetQuality, quality(cC, tetBasePt, pA, pB));
        }
        return thisBaseMinTetQuality;
    }

    // findSharedBasePoint (:133-166): the first face point that both cells can fan from
    static label findSharedBasePoint(
        const PrimitiveMesh& m,
        label fI,
        const vector& oCc,
        const vector& nCc)
    {
        for (label faceBasePtI = 0; faceBasePtI < m.faceSize(fI); ++faceBasePtI)
        {
            const scalar ownQuality = minQuality(m, oCc, fI, true, faceBasePtI);
            const scalar neiQuality = minQuality(m, nCc, fI, false, faceBasePtI);
            if (std::min(ownQuality, neiQuality) > kSmall*kSmall)
            {
                return faceBasePtI;
            }
        }
        // "none that can produce a good decomposition"
        return -1;
    }

    // findBasePoint (:188-221), a boundary face: its one cell
    static label findBasePoint(
        const PrimitiveMesh& m,
        label fI,
        const vector& cC)
    {
        for (label faceBasePtI = 0; faceBasePtI < m.faceSize(fI); ++faceBasePtI)
        {
            if (minQuality(m, cC, fI, true, faceBasePtI) > kSmall*kSmall)
            {
                return faceBasePtI;
            }
        }
        return -1;
    }

    void refuseCoupledCandidate(
        const PrimitiveMesh& m,
        label celli,
        const vector& p,
        const std::string& who) const
    {
        for (label k = cellFaceOffsets_[celli]; k < cellFaceOffsets_[celli + 1]; ++k)
        {
            const label facei = cellFaces_[k];
            if (!coupledFace_[facei])
            {
                continue;
            }
            std::string patch;
            for (const PatchInfo& q : m.patches())
            {
                if (facei >= q.start && facei < q.start + q.size)
                {
                    patch = q.name;
                }
            }
            char at[160];
            std::snprintf(at, sizeof(at), "(%.17g %.17g %.17g)", (double)p.x, (double)p.y, (double)p.z);
            throw std::runtime_error(
                "brae: " + who + ": the location " + at + " is at cell " + std::to_string(celli) + ", which has "
                "a face on the coupled patch `" + patch + "`. OpenFOAM fans that face into tets from a base "
                "point it chooses with the cell centre across the coupling (polyMeshTetDecomposition::"
                "findFaceBasePts), and that choice is not ported: move the location off the patch's cells.");
        }
    }

    bool built_ = false;
    std::vector<label> cellFaceOffsets_;
    std::vector<label> cellFaces_;
    std::vector<label> tetBasePtIs_;
    std::vector<char> coupledFace_;
};

}   // namespace brae
