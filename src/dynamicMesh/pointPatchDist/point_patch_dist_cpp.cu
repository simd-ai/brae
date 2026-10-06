#include "point_patch_dist_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace brae {

namespace {

// PointEdgeWaveBase.C:40
const scalar propagationTol = scalar(0.01);
// GREAT (doubleScalar.H:58, chosen by scalar.H:126): what a point the wave never reached holds. (EpePoint's
// own sentinel below carries distSqr 0 where OpenFOAM's default element has GREAT: every read of it is behind
// valid() or an origin comparison an invalid entry cannot pass, so it never reaches a number.)
constexpr scalar great = 1.0e15;

// externalPointEdgePoint: an origin point and the squared distance from it. `valid` is
// origin != point::max, which is the sentinel OpenFOAM constructs the field with.
struct EpePoint
{
    vector origin{std::numeric_limits<scalar>::max(),
                  std::numeric_limits<scalar>::max(),
                  std::numeric_limits<scalar>::max()};
    scalar distSqr = 0;

    bool valid() const { return origin.x != std::numeric_limits<scalar>::max(); }
    bool equal(const EpePoint& r) const
    {
        return origin.x == r.origin.x && origin.y == r.origin.y && origin.z == r.origin.z
            && distSqr == r.distSqr;
    }
};

// externalPointEdgePointI.H:38-78. `pt` is the position being updated -- a point's own position when a
// point takes from an edge, the edge's CENTRE when an edge takes from a point.
bool update(EpePoint& self, const vector& pt, const EpePoint& w2, scalar tol)
{
    const vector d = pt - w2.origin;
    const scalar dist2 = d.x*d.x + d.y*d.y + d.z*d.z;
    if (!self.valid())
    {
        self.distSqr = dist2;
        self.origin = w2.origin;
        return true;
    }
    const scalar diff = self.distSqr - dist2;
    if (diff < 0)
    {
        return false;
    }
    // SMALL is OpenFOAM's 1e-15 for a double build (floatScalar SMALL is 1e-6; scalar is double here)
    const scalar small_ = scalar(1e-15);
    if ((diff < small_) || ((self.distSqr > small_) && (diff/self.distSqr < tol)))
    {
        return false;
    }
    self.distSqr = dist2;
    self.origin = w2.origin;
    return true;
}

}   // namespace


PointPatchDist pointPatchDist(
    const PrimitiveMesh&        m,
    const MeshEdges&            e,
    const std::vector<FvPatch>& patches,
    const std::vector<label>&   patchIDs)
{
    for (const label pi : patchIDs)
    {
        if (pi < 0 || pi >= static_cast<label>(patches.size()))
        {
            throw std::runtime_error("brae pointPatchDist: patch index out of range.");
        }
    }
    for (const FvPatch& q : patches)
    {
        if (q.coupled)
        {
            throw std::runtime_error(
                "brae pointPatchDist: the mesh has the coupled patch `" + q.name + "`. PointEdgeWave "
                "carries the wave across a cyclic or processor pair in handleCyclicPatches and "
                "handleProcPatches, and neither is ported: the distance on the far side would be "
                "whatever the wave reached through the interior instead.");
        }
    }

    const label nPoints = m.nPoints();
    const std::vector<vector>& pts = m.points();
    std::vector<EpePoint> pointInfo(static_cast<std::size_t>(nPoints));
    std::vector<EpePoint> edgeInfo(static_cast<std::size_t>(e.nEdges()));
    std::vector<char> changedPoint(static_cast<std::size_t>(nPoints), 0);
    std::vector<char> changedEdge(static_cast<std::size_t>(e.nEdges()), 0);
    std::vector<label> changedPoints;
    std::vector<label> changedEdges;

    // pointPatchDist.C:70-99: every mesh point of every named patch seeds ITSELF, distance zero, in
    // the order the patch set gives them
    for (const label pi : patchIDs)
    {
        const FvPatch& q = patches[static_cast<std::size_t>(pi)];
        const PrimitivePatchAddressing addr = primitivePatch(m, faceRange(q.start, q.size));
        for (const label meshPointi : addr.meshPoints)
        {
            EpePoint& info = pointInfo[static_cast<std::size_t>(meshPointi)];
            info.origin = pts[static_cast<std::size_t>(meshPointi)];
            info.distSqr = 0;
            if (!changedPoint[static_cast<std::size_t>(meshPointi)])
            {
                changedPoint[static_cast<std::size_t>(meshPointi)] = 1;
                changedPoints.push_back(meshPointi);
            }
        }
    }

    // PointEdgeWave::iterate, with the cyclic and processor transfers left out (refused above). The
    // cap is OpenFOAM's: globalData().nTotalPoints().
    const label maxIter = nPoints;
    for (label iter = 0; iter < maxIter; ++iter)
    {
        // pointToEdge
        for (const label pointi : changedPoints)
        {
            const EpePoint& nbr = pointInfo[static_cast<std::size_t>(pointi)];
            for (const label edgei : e.pointEdges[static_cast<std::size_t>(pointi)])
            {
                EpePoint& cur = edgeInfo[static_cast<std::size_t>(edgei)];
                if (cur.equal(nbr)) continue;
                if (update(cur, e.centre(edgei, pts), nbr, propagationTol)
                 && !changedEdge[static_cast<std::size_t>(edgei)])
                {
                    changedEdge[static_cast<std::size_t>(edgei)] = 1;
                    changedEdges.push_back(edgei);
                }
            }
            changedPoint[static_cast<std::size_t>(pointi)] = 0;
        }
        changedPoints.clear();
        if (changedEdges.empty()) break;

        // edgeToPoint
        for (const label edgei : changedEdges)
        {
            const EpePoint& nbr = edgeInfo[static_cast<std::size_t>(edgei)];
            const label ends[2] = {e.start[static_cast<std::size_t>(edgei)],
                                   e.end[static_cast<std::size_t>(edgei)]};
            for (const label pointi : ends)
            {
                EpePoint& cur = pointInfo[static_cast<std::size_t>(pointi)];
                if (cur.equal(nbr)) continue;
                if (update(cur, pts[static_cast<std::size_t>(pointi)], nbr, propagationTol)
                 && !changedPoint[static_cast<std::size_t>(pointi)])
                {
                    changedPoint[static_cast<std::size_t>(pointi)] = 1;
                    changedPoints.push_back(pointi);
                }
            }
            changedEdge[static_cast<std::size_t>(edgei)] = 0;
        }
        changedEdges.clear();
        if (changedPoints.empty()) break;
    }

    // A POINT THE WAVE NEVER REACHED HOLDS GREAT, not 0: pointPatchDist.C:52 constructs the field at GREAT and
    // :126-136 overwrites the valid points only. rigidBodyMeshMotion.C:181-206 makes a motion scale of exactly
    // 0 of that -- the point stays where it is -- where a 0 here made a scale of exactly 1: the point moved
    // rigidly with the body. Serially a point is unreached when no edge path joins it to the patches (a second
    // mesh region; by reading also when the patches have no face at all, which no gate holds). No shipped
    // tutorial has one; MEASURED on two
    // blocks that share no vertex (tests/point_patch_dist_unreached_vs_openfoam.sh): 45 of 90 points, each
    // 1e15 from OpenFOAM's value with the 0 and its scale 1.0 from OpenFOAM's.
    //   BRAE_CONTROL_POINT_PATCH_DIST_UNSET_ZERO=1: a gate's CONTROL, deliberately wrong -- 0, as before
    static const bool unsetZero = std::getenv("BRAE_CONTROL_POINT_PATCH_DIST_UNSET_ZERO") != nullptr;
    static bool said = false;
    if (unsetZero && !said)
    {
        said = true;
        std::printf("  *** CONTROL MODE: pointPatchDist leaves a point the wave never reached at 0, not GREAT. "
                    "This run is deliberately wrong. ***\n");
    }
    PointPatchDist out;
    out.distance.assign(static_cast<std::size_t>(nPoints), unsetZero ? scalar(0) : great);
    for (label p = 0; p < nPoints; ++p)
    {
        const EpePoint& info = pointInfo[static_cast<std::size_t>(p)];
        if (info.valid())
        {
            out.distance[static_cast<std::size_t>(p)] = std::sqrt(info.distSqr);
        }
        else
        {
            ++out.nUnset;
        }
    }
    return out;
}


std::vector<scalar> rigidBodyMeshMotionScale(
    const std::vector<scalar>& pointDist,
    scalar                     innerDistance,
    scalar                     outerDistance)
{
    if (!(outerDistance > innerDistance))
    {
        throw std::runtime_error(
            "brae rigidBodyMeshMotionScale: outerDistance must exceed innerDistance; OpenFOAM divides "
            "by their difference.");
    }
    const scalar pi = scalar(3.14159265358979323846);
    std::vector<scalar> scale(pointDist.size());
    for (std::size_t i = 0; i < pointDist.size(); ++i)
    {
        const scalar r = (outerDistance - pointDist[i])/(outerDistance - innerDistance);
        const scalar c = std::fmin(std::fmax(r, scalar(0)), scalar(1));
        scale[i] = std::fmin(std::fmax(scalar(0.5) - scalar(0.5)*std::cos(c*pi), scalar(0)), scalar(1));
    }
    return scale;
}

} // namespace brae
