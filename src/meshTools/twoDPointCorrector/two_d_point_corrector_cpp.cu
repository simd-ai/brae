#include "two_d_point_corrector_cpp.cuh"
#include "face_area_weight_ami_cpp.cuh"   // face::areaNormal(points)
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
    // wedgePolyPatch::calcGeometry for every wedge patch with faces: n_ = gAverage(faceNormals()), the face unit normals
    // face::unitNormal(points) (PrimitivePatch::calcFaceNormals), summed in face order and divided by their number;
    // centreNormal_ = sign(n_k)*(max(|n_k|, 0.5) - 0.5) per component, normalised; cosAngle_ = centreNormal_ & n_;
    // axis_ = centreNormal_ ^ n_, normalised
    struct WedgeCalc { vector cn, axis; scalar cosAngle; };
    auto wedgeCalc = [&](const FvPatch& p) {
        vector n{0, 0, 0};
        std::vector<label> f;
        for (label i = 0; i < p.size; ++i)
        {
            const label fi = p.start + i;
            f.resize(static_cast<std::size_t>(m.faceSize(fi)));
            for (label k = 0; k < m.faceSize(fi); ++k) f[static_cast<std::size_t>(k)] = m.faceVert(fi, k);
            const vector a = cpu::ami::faceAreaNormal(f, m.points());
            const scalar s = mag(a);
            n += (s < 1.0e-150) ? vector{0, 0, 0} : a/s;   // vector::normalise: zero below ROOTVSMALL
        }
        n = n/static_cast<scalar>(p.size);
        auto sgn = [](scalar v) { return v >= 0 ? scalar(1) : scalar(-1); };   // Foam::sign: >= 0 is 1
        vector cn{sgn(n.x)*(std::fmax(std::fabs(n.x), 0.5) - 0.5), sgn(n.y)*(std::fmax(std::fabs(n.y), 0.5) - 0.5),
                  sgn(n.z)*(std::fmax(std::fabs(n.z), 0.5) - 0.5)};
        const scalar mc = mag(cn);
        cn = (mc < 1.0e-150) ? vector{0, 0, 0} : cn/mc;
        // wedgePolyPatch.C:96-122: OpenFOAM stops on a centre plane that is not a coordinate plane and on a wedge plane
        // that lies on one (SMALL = 1e-15)
        if (std::fabs(cn.x + cn.y + cn.z) < 1 - 1e-15)
            throw std::runtime_error(std::string(WHO) + "wedge `" + p.name + "`: its centre plane does not align with "
                                     "a coordinate plane (OpenFOAM stops here too, wedgePolyPatch.C:96-105).");
        vector ax = cross(cn, n);
        const scalar magAxis = mag(ax);
        if (magAxis < 1e-15)
            throw std::runtime_error(std::string(WHO) + "wedge `" + p.name + "`: the plane aligns with a coordinate "
                                     "plane; it should make a small angle with it (OpenFOAM stops here too, "
                                     "wedgePolyPatch.C:108-122).");
        ax = ax/magAxis;
        return WedgeCalc{cn, ax, dot(cn, n)};
    };

    // polyMesh::calcDirections: the empty patches' summed |face areas| and the wedges' |centre normals|, each
    // normalised; a component above 1e-6 is knocked out of solutionD (empty) and of geometricD (empty and wedge)
    vector emptyDirVec{0, 0, 0}, wedgeDirVec{0, 0, 0};
    bool hasEmpty = false, hasWedge = false;
    isWedge_ = false;
    for (const FvPatch& p : patches)
    {
        if (p.type == "empty" && p.size > 0)
        {
            hasEmpty = true;
            for (label i = 0; i < p.size; ++i)
            {
                const vector& a = g.Sf()[static_cast<std::size_t>(p.start + i)];
                emptyDirVec += vector{std::fabs(a.x), std::fabs(a.y), std::fabs(a.z)};
            }
        }
        else if (p.type == "wedge" && p.size > 0)
        {
            hasWedge = true;
            const WedgeCalc w = wedgeCalc(p);
            wedgeDirVec += vector{std::fabs(w.cn.x), std::fabs(w.cn.y), std::fabs(w.cn.z)};
            if (!isWedge_)   // calcAddressing: the first wedge patch gives the normal, axis and angle
            {
                isWedge_ = true;
                planeNormal_ = w.cn;
                wedgeAxis_ = w.axis;
                wedgeAngle_ = std::fabs(std::acos(w.cosAngle));
            }
        }
    }
    int geometricD[3] = {1, 1, 1};
    if (hasEmpty)
    {
        emptyDirVec = emptyDirVec/mag(emptyDirVec);
        const scalar e[3] = {emptyDirVec.x, emptyDirVec.y, emptyDirVec.z};
        for (int cmpt = 0; cmpt < 3; ++cmpt) geometricD[cmpt] = (e[cmpt] > 1e-6) ? -1 : 1;
    }
    if (hasWedge)
    {
        wedgeDirVec = wedgeDirVec/mag(wedgeDirVec);
        const scalar w[3] = {wedgeDirVec.x, wedgeDirVec.y, wedgeDirVec.z};
        for (int cmpt = 0; cmpt < 3; ++cmpt) if (w[cmpt] > 1e-6) geometricD[cmpt] = -1;
    }
    int nGeometricD = 0;
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        emptyDir_[cmpt] = (geometricD[cmpt] == -1);
        if (geometricD[cmpt] == 1) nGeometricD++;
    }
    // the corrector before a wedge was ported took solutionDirections(patches); without a wedge the two agree
    if (!hasWedge)
    {
        const SolutionDirections sd = solutionDirections(patches);
        for (int cmpt = 0; cmpt < 3; ++cmpt)
            if (emptyDir_[cmpt] != !sd.valid(cmpt))
                throw std::runtime_error(std::string(WHO) + "the empty directions disagree with solutionDirections.");
    }
    required_ = (nGeometricD == 2);
    normalEdgeStart_.clear();
    normalEdgeEnd_.clear();
    if (!required_) return;

    // the normal: a wedge's (above), else the first empty patch with faces
    vector pn = isWedge_ ? planeNormal_ : vector{0, 0, 0};
    if (!isWedge_)
    {
        for (const FvPatch& p : patches)
        {
            if (p.type == "empty" && p.size > 0)
            {
                pn = g.Sf()[static_cast<std::size_t>(p.start)];
                break;
            }
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
    if (!isWedge_ && 2*normalEdgeStart_.size() != meshPoints.size())
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

        if (isWedge_)
        {
            // snapToWedge(n, A, p): ADash = |A - axis*(axis & A)|, pDash = ADash*tan(wedgeAngle)*planeNormal,
            // p = A + sign(n & p)*pDash (twoDPointCorrector.C:184-193)
            const vector axA = wedgeAxis_*dot(wedgeAxis_, A);
            const scalar ADash = mag(A - axA);
            const vector pDash = (ADash*std::tan(wedgeAngle_))*pn;
            auto sgn = [](scalar v) { return v >= 0 ? scalar(1) : scalar(-1); };
            pStart = A + sgn(dot(pn, pStart))*pDash;
            pEnd = A + sgn(dot(pn, pEnd))*pDash;
        }
        else
        {
            // correct point locations
            pStart = A + pn*dot(pn, pStart - A);
            pEnd = A + pn*dot(pn, pEnd - A);
        }
    }
}

} // namespace brae
