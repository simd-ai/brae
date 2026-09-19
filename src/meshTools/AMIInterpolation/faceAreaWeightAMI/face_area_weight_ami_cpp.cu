// OpenFOAM's faceAreaWeightAMI, the host reference -- see face_area_weight_ami_cpp.cuh.
#include "face_area_weight_ami_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace brae {
namespace cpu {
namespace ami {

namespace {

constexpr scalar ROOTVSMALL = 1.0e-150;
constexpr scalar GREAT = 1.0e+15;
constexpr scalar PI = 3.14159265358979323846;
// faceAreaIntersect.C: Foam::scalar Foam::faceAreaIntersect::tol = 1e-6
constexpr scalar INTERSECT_TOL = 1e-6;

// VectorI.H normalise(): zero below ROOTVSMALL, else divided by its magnitude
vector normalised(const vector& v)
{
    const scalar s = mag(v);
    if (s < ROOTVSMALL)
    {
        return vector{0, 0, 0};
    }
    return v/s;
}

scalar clampUnit(scalar c)
{
    return (c < -1) ? scalar(-1) : (1 < c) ? scalar(1) : c;
}

// triangle<>::areaNormal: 0.5*((p1 - p0)^(p2 - p0))
vector triAreaNormal(
    const vector& p0,
    const vector& p1,
    const vector& p2)
{
    return 0.5*cross(p1 - p0, p2 - p0);
}

struct Tri
{
    vector p[3];
};

// triPoints::mag and ::centre are the triangle's
scalar triMag(const Tri& t)
{
    return mag(triAreaNormal(t.p[0], t.p[1], t.p[2]));
}

label fcIndex(
    label i,
    label n)
{
    return (i == n - 1) ? 0 : i + 1;
}

label rcIndex(
    label i,
    label n)
{
    return (i == 0) ? n - 1 : i - 1;
}

// face::calcEdgeVectors
std::vector<vector> edgeVectors(
    const std::vector<label>& f,
    const std::vector<vector>& points)
{
    const label n = static_cast<label>(f.size());
    std::vector<vector> e(static_cast<std::size_t>(n));
    for (label i = 0; i < n; ++i)
    {
        e[i] = normalised(points[f[fcIndex(i, n)]] - points[f[i]]);
    }
    return e;
}

// face::mostConcaveAngle
label mostConcaveAngle(
    const std::vector<label>& f,
    const std::vector<vector>& points,
    const std::vector<vector>& edges,
    scalar& maxAngle)
{
    const vector n = faceAreaNormal(f, points);
    const label nE = static_cast<label>(edges.size());
    label index = 0;
    maxAngle = -GREAT;
    for (label i = 0; i < nE; ++i)
    {
        const vector& leftEdge = edges[rcIndex(i, nE)];
        const vector& rightEdge = edges[i];
        const vector edgeNormal = cross(rightEdge, leftEdge);
        const scalar edgeCos = dot(leftEdge, rightEdge);
        const scalar edgeAngle = std::acos(clampUnit(edgeCos));
        scalar angle;
        if (dot(edgeNormal, n) > 0)
        {
            angle = PI + edgeAngle;
        }
        else
        {
            angle = PI - edgeAngle;
        }
        if (angle > maxAngle)
        {
            maxAngle = angle;
            index = i;
        }
    }
    return index;
}

// faceAreaIntersect::triSliceWithPlane
void triSliceWithPlane(
    const Tri& tri,
    const vector& origin,
    const vector& normal,
    Tri* tris,
    label& nTris,
    scalar len)
{
    scalar d[3];
    label nCoPlanar = 0;
    label nPos = 0;
    label posI = -1;
    label negI = -1;
    label copI = -1;
    for (label i = 0; i < 3; ++i)
    {
        d[i] = dot(tri.p[i] - origin, normal);
        if (std::fabs(d[i]) < INTERSECT_TOL*len)
        {
            nCoPlanar++;
            copI = i;
            d[i] = 0.0;
        }
        else
        {
            if (d[i] > 0)
            {
                nPos++;
                posI = i;
            }
            else
            {
                negI = i;
            }
        }
    }
    // faceAreaIntersectI.H planeIntersection: (dp*t[negI] - dn*t[posI])/(-dn + dp)
    auto planeIntersection = [&](label nI, label pI)
    {
        const scalar dp = d[pI];
        const scalar dn = d[nI];
        return (dp*tri.p[nI] - dn*tri.p[pI])/(-dn + dp);
    };
    auto setTri = [&](const vector& a, const vector& b, const vector& c)
    {
        Tri& t = tris[nTris++];
        t.p[0] = a;
        t.p[1] = b;
        t.p[2] = c;
    };
    if ((nPos == 3) || ((nPos == 2) && (nCoPlanar == 1)) || ((nPos == 1) && (nCoPlanar == 2)))
    {
        tris[nTris++] = tri;
    }
    else if ((nPos == 2) && (nCoPlanar == 0))
    {
        const label i0 = negI;
        const label i1 = fcIndex(i0, 3);
        const label i2 = fcIndex(i1, 3);
        const vector p01 = planeIntersection(i0, i1);
        const vector p02 = planeIntersection(i0, i2);
        setTri(tri.p[i1], tri.p[i2], p02);
        setTri(tri.p[i1], p02, p01);
    }
    else if (nPos == 1)
    {
        const label i0 = posI;
        if (nCoPlanar == 0)
        {
            const label i1 = fcIndex(i0, 3);
            const label i2 = fcIndex(i1, 3);
            const vector p01 = planeIntersection(i1, i0);
            const vector p02 = planeIntersection(i2, i0);
            setTri(tri.p[i0], p01, p02);
        }
        else
        {
            const label i1 = negI;
            const label i2 = copI;
            const vector p01 = planeIntersection(i1, i0);
            if (fcIndex(i0, 3) == i1)
            {
                setTri(tri.p[i0], p01, tri.p[i2]);
            }
            else
            {
                setTri(tri.p[i0], tri.p[i2], p01);
            }
        }
    }
}

// faceAreaIntersect::triangleIntersect, the area half (the centroid is not used by the weights)
void triangleIntersect(
    const Tri& src,
    const vector& tgt0,
    const vector& tgt1,
    const vector& tgt2,
    const vector& n,
    scalar& area)
{
    Tri workTris1[10];
    label nWorkTris1 = 0;
    Tri workTris2[10];
    label nWorkTris2 = 0;

    const scalar srcArea = triMag(src);
    if (srcArea < ROOTVSMALL)
    {
        return;
    }
    const scalar t = std::sqrt(srcArea);
    // edge 0
    {
        const scalar s = mag(tgt1 - tgt0);
        if (s < ROOTVSMALL)
        {
            return;
        }
        const vector n0 = cross(tgt0 - tgt1, (-s)*n);
        const scalar magSqrN0 = magSqr(n0);
        if (magSqrN0 < ROOTVSMALL)
        {
            return;
        }
        triSliceWithPlane(src, tgt0, n0/std::sqrt(magSqrN0), workTris1, nWorkTris1, t);
    }
    if (nWorkTris1 == 0)
    {
        return;
    }
    // edge 1
    {
        const scalar s = mag(tgt2 - tgt1);
        if (s < ROOTVSMALL)
        {
            return;
        }
        const vector n1 = cross(tgt1 - tgt2, (-s)*n);
        const scalar magSqrN1 = magSqr(n1);
        if (magSqrN1 < ROOTVSMALL)
        {
            return;
        }
        const vector nn = n1/std::sqrt(magSqrN1);
        nWorkTris2 = 0;
        for (label i = 0; i < nWorkTris1; ++i)
        {
            triSliceWithPlane(workTris1[i], tgt1, nn, workTris2, nWorkTris2, t);
        }
        if (nWorkTris2 == 0)
        {
            return;
        }
    }
    // edge 2
    {
        const scalar s = mag(tgt2 - tgt0);
        if (s < ROOTVSMALL)
        {
            return;
        }
        const vector n2 = cross(tgt2 - tgt0, (-s)*n);
        const scalar magSqrN2 = magSqr(n2);
        if (magSqrN2 < ROOTVSMALL)
        {
            return;
        }
        const vector nn = n2/std::sqrt(magSqrN2);
        nWorkTris1 = 0;
        for (label i = 0; i < nWorkTris2; ++i)
        {
            triSliceWithPlane(workTris2[i], tgt2, nn, workTris1, nWorkTris1, t);
        }
        for (label i = 0; i < nWorkTris1; ++i)
        {
            area += triMag(workTris1[i]);
        }
    }
}

struct PatchGeom
{
    std::vector<std::vector<std::vector<label>>> tris;
    std::vector<vector> normals;
    std::vector<scalar> magSf;
    // primitivePatch::faceFaces, in PrimitivePatch::calcAddressing's order
    std::vector<std::vector<label>> faceFaces;
};

// advancingFrontAMI::triangulatePatch with tmMesh and areaNormalisationMode project, and the face
// normals primitivePatch::faceNormals gives (face::unitNormal)
PatchGeom geometry(const Patch& p)
{
    const std::vector<vector>& pts = *p.points;
    PatchGeom g;
    const std::size_t n = p.faces.size();
    g.tris.resize(n);
    g.normals.resize(n);
    g.magSf.resize(n);
    for (std::size_t f = 0; f < n; ++f)
    {
        const std::vector<label>& face = p.faces[f];
        faceTriangles(face, pts, g.tris[f]);
        g.normals[f] = normalised(faceAreaNormal(face, pts));
        scalar m = 0;
        for (const std::vector<label>& t : g.tris[f])
        {
            m += dot(triAreaNormal(pts[t[0]], pts[t[1]], pts[t[2]]), g.normals[f]);
        }
        g.magSf[f] = m;
    }

    // PrimitivePatch::calcAddressing (PrimitivePatchAddressing.C). Faces in order; for each edge of a
    // face (edge k = (f[k], f[k+1])) not already registered by a lower face, every HIGHER face of
    // pointFaces[edge start] -- ascending, as calcPointFaces fills it -- carrying the same edge is a
    // neighbour, recorded both ways. The advancing front walks these lists in this order, and the
    // order decides which target faces it reaches.
    std::unordered_map<label, std::vector<label>> pointFaces;
    for (std::size_t f = 0; f < n; ++f)
    {
        for (const label v : p.faces[f])
        {
            pointFaces[v].push_back(static_cast<label>(f));
        }
    }
    auto edgeOf = [&](std::size_t f, std::size_t k)
    {
        const std::vector<label>& face = p.faces[f];
        return std::pair<label, label>(face[k], face[(k + 1) % face.size()]);
    };
    auto sameEdge = [](const std::pair<label, label>& a, const std::pair<label, label>& b)
    {
        return (a.first == b.first && a.second == b.second) || (a.first == b.second && a.second == b.first);
    };
    std::vector<std::vector<char>> detected(n);
    for (std::size_t f = 0; f < n; ++f)
    {
        detected[f].assign(p.faces[f].size(), 0);
    }
    g.faceFaces.assign(n, {});
    for (std::size_t f = 0; f < n; ++f)
    {
        for (std::size_t k = 0; k < p.faces[f].size(); ++k)
        {
            if (detected[f][k]) continue;
            const std::pair<label, label> e = edgeOf(f, k);
            for (const label nb : pointFaces[e.first])
            {
                if (nb <= static_cast<label>(f)) continue;
                const std::size_t nbi = static_cast<std::size_t>(nb);
                for (std::size_t kk = 0; kk < p.faces[nbi].size(); ++kk)
                {
                    if (sameEdge(edgeOf(nbi, kk), e))
                    {
                        g.faceFaces[f].push_back(nb);
                        g.faceFaces[nbi].push_back(static_cast<label>(f));
                        detected[nbi][kk] = 1;
                    }
                }
            }
        }
    }
    return g;
}

// faceAreaWeightAMI::calcInterArea with reverseTarget false
scalar interArea(
    const Patch& src,
    const PatchGeom& sg,
    label s,
    const Patch& tgt,
    const PatchGeom& tg,
    label t)
{
    // advancingFrontAMI::isCandidate, maxDistance2 and minCosAngle at their defaults
    if (sg.magSf[s] < ROOTVSMALL || tg.magSf[t] < ROOTVSMALL)
    {
        return 0;
    }
    vector n = scalar(-1)*sg.normals[s];
    n = n + tg.normals[t];
    const scalar magN = mag(n);
    // faceAreaWeightAMI.C calcInterArea: a pair with no resultant normal is warned about and has no
    // area. The advancing front meets such pairs as it walks, so this is its answer, not a substitute.
    if (!(magN > ROOTVSMALL))
    {
        return 0;
    }
    const vector nHat = n/magN;
    const std::vector<vector>& sp = *src.points;
    const std::vector<vector>& tp = *tgt.points;
    scalar area = 0;
    // faceAreaIntersect::calc: every source triangle against every target triangle, the target's
    // vertices taken 2, 1, 0 when reverseB is false
    for (const std::vector<label>& a : sg.tris[s])
    {
        Tri tA;
        tA.p[0] = sp[a[0]];
        tA.p[1] = sp[a[1]];
        tA.p[2] = sp[a[2]];
        for (const std::vector<label>& b : tg.tris[t])
        {
            triangleIntersect(tA, tp[b[2]], tp[b[1]], tp[b[0]], nHat, area);
        }
    }
    return area;
}

// AMIInterpolation::normaliseWeights with conformal true
void normalise(
    const std::vector<scalar>& magSf,
    std::vector<std::vector<scalar>>& w,
    std::vector<scalar>& wSum)
{
    wSum.assign(w.size(), scalar(0));
    for (std::size_t f = 0; f < w.size(); ++f)
    {
        if (w[f].empty())
        {
            continue;
        }
        scalar s = 0;
        for (const scalar x : w[f])
        {
            s += x;
        }
        const scalar t = s/magSf[f];
        for (scalar& x : w[f])
        {
            x /= s;
        }
        wSum[f] = t;
    }
}

} // namespace


vector faceAreaNormal(
    const std::vector<label>& f,
    const std::vector<vector>& points)
{
    const std::size_t nPoints = f.size();
    if (nPoints == 3)
    {
        return triAreaNormal(points[f[0]], points[f[1]], points[f[2]]);
    }
    vector centrePoint{0, 0, 0};
    for (std::size_t i = 0; i < nPoints; ++i)
    {
        centrePoint = centrePoint + points[f[i]];
    }
    centrePoint = centrePoint/static_cast<scalar>(nPoints);
    vector n{0, 0, 0};
    for (std::size_t i = 0; i < nPoints; ++i)
    {
        const vector& nextPoint = (i < nPoints - 1) ? points[f[i + 1]] : points[f[0]];
        n = n + triAreaNormal(points[f[i]], nextPoint, centrePoint);
    }
    return n;
}


void faceTriangles(
    const std::vector<label>& f,
    const std::vector<vector>& points,
    std::vector<std::vector<label>>& tris)
{
    const label n = static_cast<label>(f.size());
    if (n < 3)
    {
        throw std::runtime_error("brae AMI: asked to split a face with fewer than three vertices.");
    }
    if (n == 3)
    {
        tris.push_back(f);
        return;
    }
    const std::vector<vector> edges = edgeVectors(f, points);
    scalar maxAngle = 0;
    const label startIndex = mostConcaveAngle(f, points, edges, maxAngle);
    if (n == 4)
    {
        // start at the point with the largest internal angle
        const label nextIndex = fcIndex(startIndex, n);
        const label splitIndex = fcIndex(nextIndex, n);
        tris.push_back({f[startIndex], f[nextIndex], f[splitIndex]});
        tris.push_back({f[splitIndex], f[fcIndex(splitIndex, n)], f[startIndex]});
        return;
    }
    // the general case: the opposite point that most nearly bisects the largest angle
    const scalar bisectAngle = maxAngle/2;
    const vector& rightEdge = edges[startIndex];
    label index = fcIndex(fcIndex(startIndex, n), n);
    label minIndex = index;
    scalar minDiff = PI;
    for (label i = 0; i < n - 3; ++i)
    {
        const vector splitEdge = normalised(points[f[index]] - points[f[startIndex]]);
        const scalar splitCos = dot(splitEdge, rightEdge);
        const scalar splitAngle = std::acos(clampUnit(splitCos));
        const scalar angleDiff = std::fabs(splitAngle - bisectAngle);
        if (angleDiff < minDiff)
        {
            minDiff = angleDiff;
            minIndex = index;
        }
        index = fcIndex(index, n);
    }
    const label diff = (minIndex > startIndex) ? minIndex - startIndex : minIndex + n - startIndex;
    const label nPoints1 = diff + 1;
    const label nPoints2 = n - diff + 1;
    std::vector<label> face1(static_cast<std::size_t>(nPoints1));
    index = startIndex;
    for (label i = 0; i < nPoints1; ++i)
    {
        face1[i] = f[index];
        index = fcIndex(index, n);
    }
    std::vector<label> face2(static_cast<std::size_t>(nPoints2));
    index = minIndex;
    for (label i = 0; i < nPoints2; ++i)
    {
        face2[i] = f[index];
        index = fcIndex(index, n);
    }
    faceTriangles(face1, points, tris);
    faceTriangles(face2, points, tris);
}


namespace {

// face::centre (face.C): the area-weighted centre of the fan about the vertex average
vector faceCentre(
    const std::vector<label>& f,
    const std::vector<vector>& pts)
{
    const std::size_t n = f.size();
    if (n == 3)
    {
        return (1.0/3.0)*(pts[f[0]] + pts[f[1]] + pts[f[2]]);
    }
    vector centrePoint{0, 0, 0};
    for (const label v : f)
    {
        centrePoint += pts[v];
    }
    centrePoint = centrePoint/scalar(n);
    scalar sumA = 0;
    vector sumAc{0, 0, 0};
    for (std::size_t i = 0; i < n; ++i)
    {
        const vector& a = pts[f[i]];
        const vector& b = pts[f[(i + 1) % n]];
        const vector ttc = a + b + centrePoint;
        const scalar ta = mag(cross(a - centrePoint, b - centrePoint));
        sumA += ta;
        sumAc += ta*ttc;
    }
    return (sumA > 1e-300) ? sumAc/(3.0*sumA) : centrePoint;
}

// |p - nearest point of triangle abc|: triangle::nearestPointClassify (Ericson's regions)
scalar triDistance(
    const vector& p,
    const vector& a,
    const vector& b,
    const vector& c)
{
    const vector ab = b - a;
    const vector ac = c - a;
    const vector ap = p - a;
    const scalar d1 = dot(ab, ap);
    const scalar d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return mag(p - a);
    const vector bp = p - b;
    const scalar d3 = dot(ab, bp);
    const scalar d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return mag(p - b);
    const scalar vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
    {
        const scalar v = d1/(d1 - d3);
        return mag(p - (a + v*ab));
    }
    const vector cp = p - c;
    const scalar d5 = dot(ab, cp);
    const scalar d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return mag(p - c);
    const scalar vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
    {
        const scalar w = d2/(d2 - d6);
        return mag(p - (a + w*ac));
    }
    const scalar va = d3*d6 - d5*d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    {
        const scalar w = (d4 - d3)/((d4 - d3) + (d5 - d6));
        return mag(p - (b + w*(c - b)));
    }
    const scalar denom = 1/(va + vb + vc);
    const scalar v = vb*denom;
    const scalar w = vc*denom;
    return mag(p - (a + v*ab + w*ac));
}

// face::nearestPoint's distance: the triangle itself, else the fan (f[i], f[i+1], centre)
scalar faceDistance(
    const vector& p,
    const std::vector<label>& f,
    const std::vector<vector>& pts)
{
    if (f.size() == 3)
    {
        return triDistance(p, pts[f[0]], pts[f[1]], pts[f[2]]);
    }
    const vector ctr = faceCentre(f, pts);
    scalar best = GREAT;
    for (std::size_t i = 0; i < f.size(); ++i)
    {
        const scalar d = triDistance(p, pts[f[i]], pts[f[(i + 1) % f.size()]], ctr);
        if (std::fabs(d) < std::fabs(best))
        {
            best = d;
        }
    }
    return best;
}

bool contains(
    const std::vector<label>& list,
    label x)
{
    return std::find(list.begin(), list.end(), x) != list.end();
}

// The advancing front of faceAreaWeightAMI (faceAreaWeightAMI.C, advancingFrontAMI.C), serial
class Walk
{
public:
    Walk(
        const Patch& src,
        const Patch& tgt)
        : src_(src),
          tgt_(tgt),
          sg_(geometry(src)),
          tg_(geometry(tgt))
    {}

    const PatchGeom& sg() const { return sg_; }
    const PatchGeom& tg() const { return tg_; }

    // calcInterArea
    scalar area(
        label s,
        label t) const
    {
        return interArea(src_, sg_, s, tgt_, tg_, t);
    }

    // overlaps: faceAreaIntersect::overlaps stops as soon as the running area passes the threshold,
    // and the areas only accumulate, so it is the full area's test
    bool overlaps(
        label s,
        label t,
        scalar threshold) const
    {
        return area(s, t) > threshold;
    }

    // advancingFrontAMI::findTargetFace: the target face nearest the source face's bounding-box centre
    // (or one of its points), excluding `exclude`. The octree's search radius only prunes, and a miss
    // falls back to GREAT, so this is the nearest over all faces; strict < keeps the first of a tie.
    label findTargetFace(
        label s,
        const std::vector<label>& exclude,
        label srcFacePti = -1) const
    {
        const std::vector<vector>& sp = *src_.points;
        const std::vector<label>& f = src_.faces[static_cast<std::size_t>(s)];
        vector sample;
        if (srcFacePti == -1)
        {
            vector lo = sp[f[0]];
            vector hi = sp[f[0]];
            for (const label v : f)
            {
                lo = vector{std::min(lo.x, sp[v].x), std::min(lo.y, sp[v].y), std::min(lo.z, sp[v].z)};
                hi = vector{std::max(hi.x, sp[v].x), std::max(hi.y, sp[v].y), std::max(hi.z, sp[v].z)};
            }
            sample = 0.5*(lo + hi);
        }
        else
        {
            sample = sp[f[static_cast<std::size_t>(srcFacePti)]];
        }
        label best = -1;
        scalar bestD2 = GREAT*GREAT;
        for (std::size_t t = 0; t < tgt_.faces.size(); ++t)
        {
            if (contains(exclude, static_cast<label>(t))) continue;
            const scalar d = faceDistance(sample, tgt_.faces[t], *tgt_.points);
            const scalar d2 = d*d;
            if (d2 < bestD2)
            {
                bestD2 = d2;
                best = static_cast<label>(t);
            }
        }
        // ...and isCandidate, which with the default maxDistance2 and minCosAngle rejects only a face of
        // no area
        if (best >= 0 && (sg_.magSf[static_cast<std::size_t>(s)] < ROOTVSMALL
                       || tg_.magSf[static_cast<std::size_t>(best)] < ROOTVSMALL))
        {
            best = -1;
        }
        return best;
    }

    // advancingFrontAMI::appendNbrFaces: the unvisited, unqueued neighbours of face t within 89 degrees
    void appendNbrFaces(
        label t,
        const std::vector<label>& visited,
        std::vector<label>& queue) const
    {
        static const scalar thetaCos = std::cos(89.0*PI/180.0);
        for (const label nb : tg_.faceFaces[static_cast<std::size_t>(t)])
        {
            if (contains(visited, nb) || contains(queue, nb)) continue;
            if (dot(tg_.normals[static_cast<std::size_t>(t)], tg_.normals[static_cast<std::size_t>(nb)]) > thetaCos)
            {
                queue.push_back(nb);
            }
        }
    }

    // faceAreaWeightAMI::processSourceFace: the target faces around the start, last in first out,
    // kept when the intersection passes faceAreaIntersect::tolerance() of the source face's area
    bool processSourceFace(
        label s,
        label tStart,
        std::vector<label>& queue,
        std::vector<label>& visited,
        Weights& W) const
    {
        if (tStart == -1)
        {
            return false;
        }
        queue.push_back(tStart);
        appendNbrFaces(tStart, visited, queue);
        bool processed = false;
        while (!queue.empty())
        {
            const label t = queue.back();
            queue.pop_back();
            visited.push_back(t);
            const scalar a = area(s, t);
            if (a/sg_.magSf[static_cast<std::size_t>(s)] > INTERSECT_TOL)
            {
                W.srcAddress[static_cast<std::size_t>(s)].push_back(t);
                W.srcWeights[static_cast<std::size_t>(s)].push_back(a);
                W.tgtAddress[static_cast<std::size_t>(t)].push_back(s);
                W.tgtWeights[static_cast<std::size_t>(t)].push_back(a);
                appendNbrFaces(t, visited, queue);
                processed = true;
            }
        }
        return processed;
    }

    // faceAreaWeightAMI::setNextFaces
    bool setNextFaces(
        label& startSeedi,
        label& s,
        label& t,
        const std::vector<char>& mapFlag,
        label nFlagged,
        std::vector<label>& seedFaces,
        const std::vector<label>& visited) const
    {
        if (nFlagged == 0)
        {
            return false;
        }
        const label n = static_cast<label>(mapFlag.size());
        auto findNext = [&](label i)
        {
            for (label k = i + 1; k < n; ++k)
            {
                if (mapFlag[static_cast<std::size_t>(k)]) return k;
            }
            return label(-1);
        };
        const std::vector<label>& srcNbrFaces = sg_.faceFaces[static_cast<std::size_t>(s)];
        t = -1;
        bool valuesSet = false;
        const label s0 = s;
        (void)s0;
        for (const label faceS : std::vector<label>(srcNbrFaces))
        {
            if (mapFlag[static_cast<std::size_t>(faceS)] && seedFaces[static_cast<std::size_t>(faceS)] == -1)
            {
                for (const label faceT : visited)
                {
                    const scalar threshold = sg_.magSf[static_cast<std::size_t>(faceS)]*INTERSECT_TOL;
                    if (overlaps(faceS, faceT, threshold))
                    {
                        seedFaces[static_cast<std::size_t>(faceS)] = faceT;
                        if (!valuesSet)
                        {
                            s = faceS;
                            t = faceT;
                            valuesSet = true;
                        }
                    }
                }
            }
        }
        if (valuesSet)
        {
            return true;
        }
        label facei = startSeedi;
        if (!mapFlag[static_cast<std::size_t>(startSeedi)])
        {
            facei = findNext(facei);
        }
        const label startSeedi0 = facei;
        bool foundNextSeed = false;
        while (facei != -1)
        {
            if (!foundNextSeed)
            {
                startSeedi = facei;
                foundNextSeed = true;
            }
            if (seedFaces[static_cast<std::size_t>(facei)] != -1)
            {
                s = facei;
                t = seedFaces[static_cast<std::size_t>(facei)];
                return true;
            }
            facei = findNext(facei);
        }
        // the front stalled: search for a new target face
        facei = startSeedi0;
        while (facei != -1)
        {
            s = facei;
            t = findTargetFace(s, visited);
            if (t >= 0)
            {
                return true;
            }
            facei = findNext(facei);
        }
        // errorOnNotFound: requireMatch with no low-weight correction
        throw std::runtime_error(
            "brae AMI: unable to set a target face for source face " + std::to_string(s) + ". OpenFOAM stops "
            "on it (faceAreaWeightAMI::setNextFaces, requireMatch).");
    }

private:
    const Patch& src_;
    const Patch& tgt_;
    PatchGeom sg_;
    PatchGeom tg_;
};

} // namespace


Weights faceAreaWeight(
    const Patch& src,
    const Patch& tgt)
{
    const Walk walk(src, tgt);
    const std::size_t nS = src.faces.size();
    const std::size_t nT = tgt.faces.size();
    Weights W;
    W.srcAddress.resize(nS);
    W.srcWeights.resize(nS);
    W.tgtAddress.resize(nT);
    W.tgtWeights.resize(nT);
    W.srcMagSf = walk.sg().magSf;
    W.tgtMagSf = walk.tg().magSf;
    if (nS == 0 || nT == 0)
    {
        throw std::runtime_error("brae AMI: a side of the pair has no faces.");
    }

    // faceAreaWeightAMI::calculate hands initialiseWalk srcFacei = tgtFacei = 0, which are not -1, so no
    // search is made: the walk starts from source face 0 seeded with target face 0
    label s = 0;
    label t = 0;

    // calcAddressing
    std::vector<label> queue;
    std::vector<label> visited;
    std::vector<label> seedFaces(nS, -1);
    seedFaces[0] = 0;
    std::vector<char> mapFlag(nS, 1);
    label nFlagged = static_cast<label>(nS);
    label startSeedi = 0;
    bool continueWalk = true;
    do
    {
        queue.clear();
        visited.clear();
        walk.processSourceFace(s, t, queue, visited, W);
        if (mapFlag[static_cast<std::size_t>(s)])
        {
            mapFlag[static_cast<std::size_t>(s)] = 0;
            --nFlagged;
        }
        continueWalk = walk.setNextFaces(startSeedi, s, t, mapFlag, nFlagged, seedFaces, visited);
    } while (continueWalk);

    // restartUncoveredSourceFace (on by default): a face less than 0.95 covered is searched again from
    // the target face nearest each of its points, its partners so far excluded
    const scalar minWeight = 0.95;
    for (std::size_t si = 0; si < nS; ++si)
    {
        scalar sum = 0;
        for (const scalar a : W.srcWeights[si])
        {
            sum += a;
        }
        if (sum/W.srcMagSf[si] >= minWeight) continue;
        const std::vector<label>& f = src.faces[si];
        for (std::size_t fpi = 0; fpi < f.size(); ++fpi)
        {
            const label ti = walk.findTargetFace(static_cast<label>(si), W.srcAddress[si], static_cast<label>(fpi));
            if (ti != -1)
            {
                queue.clear();
                visited = W.srcAddress[si];
                walk.processSourceFace(static_cast<label>(si), ti, queue, visited, W);
            }
        }
    }

    normalise(W.srcMagSf, W.srcWeights, W.srcWeightsSum);
    normalise(W.tgtMagSf, W.tgtWeights, W.tgtWeightsSum);
    for (std::size_t si = 0; si < nS; ++si)
    {
        if (W.srcAddress[si].empty())
        {
            throw std::runtime_error(
                "brae AMI: source face " + std::to_string(si) + " overlaps no target face. With requireMatch "
                "(the cyclicAMI default) OpenFOAM stops on it; brae refuses.");
        }
    }
    return W;
}

} // namespace ami
} // namespace cpu
} // namespace brae
