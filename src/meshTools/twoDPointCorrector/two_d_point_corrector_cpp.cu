#include "two_d_point_corrector_cpp.cuh"
#include "solution_directions.cuh"
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace brae {

namespace {

const char* const WHO = "brae twoDPointCorrector: ";

// twoDPointCorrector::edgeOrthogonalityTol
constexpr scalar edgeOrthogonalityTol = 1.0 - 1e-4;

// VSMALL in double precision
constexpr scalar vSmall = 1.0e-300;

} // namespace

void TwoDPointCorrector::build(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    for (const FvPatch& p : patches)
    {
        if (p.type == "wedge")
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + p.name + "` is a wedge. The corrector then snaps each "
                "point to the wedge plane (snapToWedge), which is not ported.");
        }
    }
    // polyMesh::geometricD(): without wedges, the directions the empty patches knock out
    const SolutionDirections sd = solutionDirections(patches);
    int nGeometricD = 0;
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        emptyDir_[cmpt] = !sd.valid(cmpt);
        if (sd.valid(cmpt))
        {
            nGeometricD++;
        }
    }
    required_ = (nGeometricD == 2);
    normalEdgeStart_.clear();
    normalEdgeEnd_.clear();
    if (!required_) return;

    // Try to find an empty patch with faces
    vector pn{0, 0, 0};
    for (const FvPatch& p : patches)
    {
        if (p.type == "empty" && p.size > 0)
        {
            pn = g.Sf()[static_cast<std::size_t>(p.start)];
            break;
        }
    }
    if (mag(pn) < vSmall)
    {
        throw std::runtime_error(std::string(WHO) + "Cannot determine normal vector from patches.");
    }
    pn = pn/mag(pn);
    planeNormal_ = pn;

    // Select edges to be included in check: primitiveMesh::edges() are the distinct point pairs of
    // the faces, and each is corrected on its own, so their order is not needed
    std::unordered_set<unsigned long long> seen;
    const std::vector<vector>& meshPoints = m.points();
    for (label f = 0; f < m.nFaces(); ++f)
    {
        const label n = m.faceSize(f);
        for (label fp = 0; fp < n; ++fp)
        {
            const label a = m.faceVert(f, fp);
            const label b = m.faceVert(f, (fp + 1)%n);
            const unsigned long long lo = static_cast<unsigned long long>(a < b ? a : b);
            const unsigned long long hi = static_cast<unsigned long long>(a < b ? b : a);
            if (!seen.insert((lo << 32) | hi).second) continue;
            // edge::unitVec: (end - start)/mag, or the zero vector below ROOTVSMALL
            const vector d = meshPoints[static_cast<std::size_t>(b)] - meshPoints[static_cast<std::size_t>(a)];
            const scalar s = mag(d);
            const vector edgeVector = (s < 1.0e-150) ? vector{0, 0, 0} : d/s;
            if (std::fabs(dot(edgeVector, pn)) > edgeOrthogonalityTol)
            {
                normalEdgeStart_.push_back(a);
                normalEdgeEnd_.push_back(b);
            }
        }
    }
    if (2*normalEdgeStart_.size() != meshPoints.size())
    {
        throw std::runtime_error(
            std::string(WHO) + "the number of points in the mesh is not twice the number of edges normal "
            "to the plane. OpenFOAM warns and corrects the edges it found; a point on no normal edge is "
            "then left where the motion put it, which brae does not reproduce.");
    }
}

void TwoDPointCorrector::correctPoints(
    const std::vector<vector>& meshPoints,
    std::vector<vector>& p) const
{
    if (!required_) return;

    // mesh.bounds()
    vector bbMin = meshPoints.front();
    vector bbMax = meshPoints.front();
    for (const vector& q : meshPoints)
    {
        bbMin = vector{std::fmin(bbMin.x, q.x), std::fmin(bbMin.y, q.y), std::fmin(bbMin.z, q.z)};
        bbMax = vector{std::fmax(bbMax.x, q.x), std::fmax(bbMax.y, q.y), std::fmax(bbMax.z, q.z)};
    }
    const scalar minC[3] = {bbMin.x, bbMin.y, bbMin.z};
    const scalar maxC[3] = {bbMax.x, bbMax.y, bbMax.z};

    const vector& pn = planeNormal_;
    for (std::size_t edgei = 0; edgei < normalEdgeStart_.size(); ++edgei)
    {
        vector& pStart = p[static_cast<std::size_t>(normalEdgeStart_[edgei])];
        vector& pEnd = p[static_cast<std::size_t>(normalEdgeEnd_[edgei])];

        // calculate average point position
        vector A = 0.5*(pStart + pEnd);
        // meshTools::constrainToMeshCentre
        scalar a[3] = {A.x, A.y, A.z};
        for (int cmpt = 0; cmpt < 3; ++cmpt)
        {
            if (emptyDir_[cmpt])
            {
                a[cmpt] = 0.5*(minC[cmpt] + maxC[cmpt]);
            }
        }
        A = vector{a[0], a[1], a[2]};

        // correct point locations
        pStart = A + pn*dot(pn, pStart - A);
        pEnd = A + pn*dot(pn, pEnd - A);
    }
}

} // namespace brae
