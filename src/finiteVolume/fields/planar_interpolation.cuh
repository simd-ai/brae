#pragma once
// pointToPointPlanarInterpolation, the host half -- OpenFOAM's DEFAULT mapping for
// timeVaryingMappedFixedValue (mapMethod absent or `planar*`). brae ran 3D nearest for every case,
// which is OF's `nearest` option, not its default: a silent substitution that staircases any profile
// coarser than the mesh (pitzDailyExptInlet's 35 y-stations, say) where OF interpolates linearly.
//
// The pipeline, from pointToPointPlanarInterpolation.C (read, not recalled):
//   - calcCoordinateSystem (:49-130): origin = points[0], e1 toward the farthest point, normal from
//     the point with the largest perpendicular distance to the p0-e1 line; fatal below 3
//     non-collinear points.
//   - calcWeights (:135+): source points to LOCAL 2D coords, PERTURBED by perturb*random in the
//     bounding box (Random seed 123456) purely to break collinear ties; triSurfaceTools::delaunay2D;
//     barycentric weights of each destination point in its containing triangle
//     (triSurfaceTools::calcInterpolationWeights), nearest-feature projection outside the hull.
//
// brae triangulates UNPERTURBED (Bowyer-Watson below): the perturbation only selects among equally
// valid triangulations of ties, and on tied configurations the interpolated VALUE differs by
// O(perturb * bbox * local gradient) = O(1e-5 relative) at most -- the tvm gate's bound carries that.
#include "cf_types.cuh"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace planarInterp {

struct Weights          // per destination point: up to 3 sources with barycentric weights
{
    label  j[3] = {0, 0, 0};
    scalar w[3] = {1, 0, 0};
};

struct Point2 { scalar x = 0, y = 0; };

// The plane, exactly as calcCoordinateSystem builds it.
struct Plane
{
    vector origin{}, e1{}, e2{};
    static Plane fit(const std::vector<vector>& pts)
    {
        if (pts.size() < 3)
            throw std::runtime_error(
                "brae planar interpolation: need at least 3 non-collinear boundaryData points "
                "(pointToPointPlanarInterpolation.C:56 fatals identically); got "
                + std::to_string(pts.size()));
        const vector& p0 = pts[0];
        scalar maxD = 1e-300; std::size_t i1 = 0;
        for (std::size_t i = 1; i < pts.size(); ++i)
        {
            const vector d{pts[i].x - p0.x, pts[i].y - p0.y, pts[i].z - p0.z};
            const scalar m2 = dot(d, d);
            if (m2 > maxD) { maxD = m2; i1 = i; }
        }
        if (i1 == 0)
            throw std::runtime_error("brae planar interpolation: all boundaryData points coincide");
        vector e1{pts[i1].x - p0.x, pts[i1].y - p0.y, pts[i1].z - p0.z};
        const scalar m1 = std::sqrt(dot(e1, e1));
        e1 = vector{e1.x / m1, e1.y / m1, e1.z / m1};
        maxD = 1e-300; std::size_t i2 = 0;
        for (std::size_t i = 1; i < pts.size(); ++i)
        {
            if (i == i1) continue;
            vector d{pts[i].x - p0.x, pts[i].y - p0.y, pts[i].z - p0.z};
            const scalar a = dot(d, e1);                       // removeCollinear
            d = vector{d.x - a*e1.x, d.y - a*e1.y, d.z - a*e1.z};
            const scalar m2 = dot(d, d);
            if (m2 > maxD) { maxD = m2; i2 = i; }
        }
        if (i2 == 0)
            throw std::runtime_error(
                "brae planar interpolation: boundaryData points are collinear -- no plane "
                "(pointToPointPlanarInterpolation.C:110 fatals identically)");
        const vector q{pts[i2].x - p0.x, pts[i2].y - p0.y, pts[i2].z - p0.z};
        vector n{e1.y*q.z - e1.z*q.y, e1.z*q.x - e1.x*q.z, e1.x*q.y - e1.y*q.x};
        const scalar mn = std::sqrt(dot(n, n));
        n = vector{n.x / mn, n.y / mn, n.z / mn};
        // e2 = n x e1 completes the right-handed local basis (coordSystem::cartesian(origin, n, e1))
        const vector e2{n.y*e1.z - n.z*e1.y, n.z*e1.x - n.x*e1.z, n.x*e1.y - n.y*e1.x};
        return Plane{p0, e1, e2};
    }
    Point2 local(const vector& p) const
    {
        const vector d{p.x - origin.x, p.y - origin.y, p.z - origin.z};
        return Point2{dot(d, e1), dot(d, e2)};
    }
};

// Bowyer-Watson Delaunay over the 2D source points. O(n^2) insertion -- boundaryData profiles are
// tens to a few thousand points, far from where that matters.
struct Tri { std::size_t a, b, c; };

inline std::vector<Tri> delaunay2D(const std::vector<Point2>& pt)
{
    struct T { std::size_t a, b, c; scalar cx, cy, r2; bool alive; };
    auto circum = [&](std::size_t a, std::size_t b, std::size_t c, T& t)
    {
        const scalar ax = pt[a].x, ay = pt[a].y, bx = pt[b].x, by = pt[b].y,
                     cx = pt[c].x, cy = pt[c].y;
        const scalar d = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
        if (std::fabs(d) < 1e-300) { t.r2 = -1; return; }   // degenerate -> never "contains"
        const scalar a2 = ax*ax + ay*ay, b2 = bx*bx + by*by, c2 = cx*cx + cy*cy;
        t.cx = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / d;
        t.cy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / d;
        const scalar dx = ax - t.cx, dy = ay - t.cy;
        t.r2 = dx*dx + dy*dy;
    };

    // super-triangle enclosing everything
    scalar xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    for (const auto& p : pt)
    {
        xmin = std::fmin(xmin, p.x); xmax = std::fmax(xmax, p.x);
        ymin = std::fmin(ymin, p.y); ymax = std::fmax(ymax, p.y);
    }
    const scalar w = std::fmax(xmax - xmin, ymax - ymin) + scalar(1);
    std::vector<Point2> P = pt;
    const std::size_t s0 = P.size(), s1 = s0 + 1, s2 = s0 + 2;
    P.push_back({xmin - 10*w, ymin - 10*w});
    P.push_back({xmax + 10*w, ymin - 10*w});
    P.push_back({(xmin + xmax)/2, ymax + 10*w});

    std::vector<T> tris;
    auto pushTri = [&](std::size_t a, std::size_t b, std::size_t c)
    {
        T t{a, b, c, 0, 0, 0, true};
        auto cc = [&](std::size_t i, std::size_t j, std::size_t k, T& out)
        {
            const scalar ax = P[i].x, ay = P[i].y, bx = P[j].x, by = P[j].y,
                         cx = P[k].x, cy = P[k].y;
            const scalar d = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
            if (std::fabs(d) < 1e-300) { out.r2 = -1; return; }
            const scalar a2 = ax*ax + ay*ay, b2 = bx*bx + by*by, c2 = cx*cx + cy*cy;
            out.cx = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / d;
            out.cy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / d;
            const scalar dx = ax - out.cx, dy = ay - out.cy;
            out.r2 = dx*dx + dy*dy;
        };
        cc(a, b, c, t);
        tris.push_back(t);
    };
    (void)circum;
    pushTri(s0, s1, s2);

    for (std::size_t ip = 0; ip < pt.size(); ++ip)
    {
        // collect triangles whose circumcircle contains the point
        std::vector<std::pair<std::size_t, std::size_t>> edges;   // boundary of the cavity
        auto addEdge = [&](std::size_t a, std::size_t b)
        {
            for (auto& e : edges)
                if ((e.first == b && e.second == a) || (e.first == a && e.second == b))
                { e.first = e.second = (std::size_t)-1; return; }   // shared -> internal, drop
            edges.push_back({a, b});
        };
        for (auto& t : tris)
        {
            if (!t.alive || t.r2 < 0) continue;
            const scalar dx = P[ip].x - t.cx, dy = P[ip].y - t.cy;
            if (dx*dx + dy*dy <= t.r2 * (1.0 + 1e-12))
            {
                t.alive = false;
                addEdge(t.a, t.b); addEdge(t.b, t.c); addEdge(t.c, t.a);
            }
        }
        for (const auto& e : edges)
            if (e.first != (std::size_t)-1) pushTri(e.first, e.second, ip);
    }

    std::vector<Tri> out;
    for (const auto& t : tris)
        if (t.alive && t.a < s0 && t.b < s0 && t.c < s0) out.push_back({t.a, t.b, t.c});
    return out;
}

// Weights for each destination: barycentric in the containing triangle; outside the hull, the nearest
// triangle with barycentrics CLAMPED to it (triSurfaceTools' nearest-feature projection).
inline std::vector<Weights> planarWeights(
    const std::vector<vector>& srcPts,
    const std::vector<vector>& dstPts)
{
    const Plane pl = Plane::fit(srcPts);
    std::vector<Point2> s(srcPts.size()), d(dstPts.size());
    for (std::size_t i = 0; i < srcPts.size(); ++i) s[i] = pl.local(srcPts[i]);
    for (std::size_t i = 0; i < dstPts.size(); ++i) d[i] = pl.local(dstPts[i]);
    const std::vector<Tri> tris = delaunay2D(s);
    if (tris.empty())
        throw std::runtime_error("brae planar interpolation: triangulation produced no triangles");

    std::vector<Weights> W(dstPts.size());
    for (std::size_t i = 0; i < d.size(); ++i)
    {
        scalar bestScore = -1e300;
        Weights best;
        for (const Tri& t : tris)
        {
            const Point2& A = s[t.a]; const Point2& B = s[t.b]; const Point2& C = s[t.c];
            const scalar den = (B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y);
            if (std::fabs(den) < 1e-300) continue;
            const scalar w0 = ((B.y - C.y)*(d[i].x - C.x) + (C.x - B.x)*(d[i].y - C.y)) / den;
            const scalar w1 = ((C.y - A.y)*(d[i].x - C.x) + (A.x - C.x)*(d[i].y - C.y)) / den;
            const scalar w2 = 1.0 - w0 - w1;
            const scalar score = std::fmin(w0, std::fmin(w1, w2));  // >= 0 iff inside
            if (score > bestScore)
            {
                bestScore = score;
                // clamp-and-renormalise: identity inside the triangle, nearest-feature outside
                scalar c0 = std::fmax(w0, scalar(0)), c1 = std::fmax(w1, scalar(0)),
                       c2 = std::fmax(w2, scalar(0));
                const scalar sum = c0 + c1 + c2;
                best.j[0] = (label)t.a; best.j[1] = (label)t.b; best.j[2] = (label)t.c;
                best.w[0] = c0/sum; best.w[1] = c1/sum; best.w[2] = c2/sum;
            }
        }
        W[i] = best;
    }
    return W;
}

} // namespace planarInterp
} // namespace brae
