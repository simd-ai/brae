#include "face_cpp.cuh"
#include <cmath>

namespace brae {

namespace {

// ROOTVSMALL and VSMALL in double precision
constexpr scalar rootVSmall = 1.0e-150;
constexpr scalar vSmall = 1.0e-300;

// triangle::nearestPointClassify(p).distance(), after Ericson, Real-time collision detection
scalar triangleNearestDistance(
    const vector& a,
    const vector& b,
    const vector& c,
    const vector& p)
{
    // Check if P in vertex region outside A
    const vector ab = b - a;
    const vector ac = c - a;
    const vector ap = p - a;

    const scalar d1 = dot(ab, ap);
    const scalar d2 = dot(ac, ap);

    if (d1 <= 0.0 && d2 <= 0.0)
    {
        return mag(a - p);
    }

    // Check if P in vertex region outside B
    const vector bp = p - b;
    const scalar d3 = dot(ab, bp);
    const scalar d4 = dot(ac, bp);

    if (d3 >= 0.0 && d4 <= d3)
    {
        return mag(b - p);
    }

    // Check if P in edge region of AB, if so return projection of P onto AB
    const scalar vc = d1*d4 - d3*d2;

    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        if ((d1 - d3) < rootVSmall)
        {
            // Degenerate triangle, for d1 = d3, a_ and b_ are likely coincident
            return mag(a - p);
        }
        const scalar v = d1/(d1 - d3);
        const vector nearPt = a + v*ab;
        return mag(nearPt - p);
    }

    // Check if P in vertex region outside C
    const vector cp = p - c;
    const scalar d5 = dot(ab, cp);
    const scalar d6 = dot(ac, cp);

    if (d6 >= 0.0 && d5 <= d6)
    {
        return mag(c - p);
    }

    // Check if P in edge region of AC, if so return projection of P onto AC
    const scalar vb = d5*d2 - d1*d6;

    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        if ((d2 - d6) < rootVSmall)
        {
            // Degenerate triangle, for d2 = d6, a_ and c_ are likely coincident
            return mag(a - p);
        }
        const scalar w = d2/(d2 - d6);
        const vector nearPt = a + w*ac;
        return mag(nearPt - p);
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    const scalar va = d3*d6 - d5*d4;

    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        if (((d4 - d3) + (d5 - d6)) < rootVSmall)
        {
            // Degenerate triangle, for (d4 - d3) = (d6 - d5), b_ and c_ are likely coincident
            return mag(b - p);
        }
        const scalar w = (d4 - d3)/((d4 - d3) + (d5 - d6));
        const vector nearPt = b + w*(c - b);
        return mag(nearPt - p);
    }

    // P inside face region
    if ((va + vb + vc) < rootVSmall)
    {
        // Degenerate triangle, return the centre because no edge or points are closest
        const vector nearPt = (1.0/3.0)*(a + b + c);
        return mag(nearPt - p);
    }

    const scalar denom = 1.0/(va + vb + vc);
    const scalar v = vb*denom;
    const scalar w = vc*denom;

    // = u*a + v*b + w*c, u = va*denom = 1.0 - v - w
    const vector nearPt = a + ab*v + ac*w;
    return mag(nearPt - p);
}

} // namespace

vector faceCentreOfPoints(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& points)
{
    const label nPoints = m.faceSize(f);
    // If the face is a triangle, do a direct calculation
    if (nPoints == 3)
    {
        return (1.0/3.0)*(points[m.faceVert(f, 0)] + points[m.faceVert(f, 1)] + points[m.faceVert(f, 2)]);
    }
    vector centrePoint{0, 0, 0};
    for (label pI = 0; pI < nPoints; ++pI)
    {
        centrePoint += points[m.faceVert(f, pI)];
    }
    centrePoint = centrePoint/scalar(nPoints);

    scalar sumA = 0;
    vector sumAc{0, 0, 0};
    for (label pI = 0; pI < nPoints; ++pI)
    {
        const vector& thisPoint = points[m.faceVert(f, pI)];
        const vector& nextPoint = points[m.faceVert(f, (pI + 1)%nPoints)];
        // 3*triangle centre
        const vector ttc = thisPoint + nextPoint + centrePoint;
        // 2*triangle area
        const scalar ta = mag(cross(thisPoint - centrePoint, nextPoint - centrePoint));
        sumA += ta;
        sumAc += ta*ttc;
    }
    if (sumA > vSmall)
    {
        return sumAc/(3.0*sumA);
    }
    return centrePoint;
}

vector faceAverage(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& points,
    const std::vector<vector>& fld)
{
    const label nPoints = m.faceSize(f);
    // If the face is a triangle, do a direct calculation
    if (nPoints == 3)
    {
        return (1.0/3.0)*(fld[m.faceVert(f, 0)] + fld[m.faceVert(f, 1)] + fld[m.faceVert(f, 2)]);
    }

    vector centrePoint{0, 0, 0};
    vector cf{0, 0, 0};
    for (label pI = 0; pI < nPoints; ++pI)
    {
        centrePoint += points[m.faceVert(f, pI)];
        cf += fld[m.faceVert(f, pI)];
    }
    centrePoint = centrePoint/scalar(nPoints);
    cf = cf/scalar(nPoints);

    scalar sumA = 0;
    vector sumAf{0, 0, 0};
    for (label pI = 0; pI < nPoints; ++pI)
    {
        const label thisI = m.faceVert(f, pI);
        const label nextI = m.faceVert(f, (pI + 1)%nPoints);
        // Calculate 3*triangle centre field value
        const vector ttcf = fld[thisI] + fld[nextI] + cf;
        // Calculate 2*triangle area
        const scalar ta = mag(cross(points[thisI] - centrePoint, points[nextI] - centrePoint));
        sumA += ta;
        sumAf += ta*ttcf;
    }
    if (sumA > vSmall)
    {
        return sumAf/(3*sumA);
    }
    return cf;
}

scalar faceNearestDistance(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& points,
    const vector& p)
{
    const label nPoints = m.faceSize(f);
    // If the face is a triangle, do a direct calculation
    if (nPoints == 3)
    {
        return triangleNearestDistance(
            points[m.faceVert(f, 0)],
            points[m.faceVert(f, 1)],
            points[m.faceVert(f, 2)],
            p);
    }

    const vector ctr = faceCentreOfPoints(m, f, points);

    // Initialize to miss, distance=GREAT
    scalar nearest = 1.0e15;
    for (label pI = 0; pI < nPoints; ++pI)
    {
        // Note: for best accuracy, centre point always comes last
        const scalar curHit = triangleNearestDistance(
            points[m.faceVert(f, pI)],
            points[m.faceVert(f, (pI + 1)%nPoints)],
            ctr,
            p);
        if (std::fabs(curHit) < std::fabs(nearest))
        {
            nearest = curHit;
        }
    }
    return nearest;
}

} // namespace brae
