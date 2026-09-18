#include "dynamic_motion_solver_fv_mesh_cpp.cuh"
#include "foam_dict.cuh"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace brae {

namespace {

const char* const WHO = "brae dynamicFvMesh: ";

// triangle::sweptVol: this triangle (a, b, c) swept to t (ta, tb, tc)
scalar triangleSweptVol(
    const vector& a,
    const vector& b,
    const vector& c,
    const vector& ta,
    const vector& tb,
    const vector& tc)
{
    return (1.0/12.0)*
    (
        dot(ta - a, cross(b - a, c - a))
      + dot(tb - b, cross(c - b, ta - b))
      + dot(c - tc, cross(tb - tc, ta - tc))

      + dot(ta - a, cross(b - a, c - a))
      + dot(b - tb, cross(ta - tb, tc - tb))
      + dot(c - tc, cross(b - tc, ta - tc))
    );
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
    // VSMALL
    if (sumA > scalar(1e-300)) return sumAc/(3.0*sumA);
    return centrePoint;
}

scalar faceSweptVolume(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& oldPoints,
    const std::vector<vector>& newPoints)
{
    scalar sv = 0;
    // a central decomposition, the centre point first in every triangle
    const vector centreOldPoint = faceCentreOfPoints(m, f, oldPoints);
    const vector centreNewPoint = faceCentreOfPoints(m, f, newPoints);
    const label nPoints = m.faceSize(f);
    for (label pi = 0; pi < nPoints - 1; ++pi)
    {
        sv += triangleSweptVol(
            centreOldPoint,
            oldPoints[m.faceVert(f, pi)],
            oldPoints[m.faceVert(f, pi + 1)],
            centreNewPoint,
            newPoints[m.faceVert(f, pi)],
            newPoints[m.faceVert(f, pi + 1)]);
    }
    sv += triangleSweptVol(
        centreOldPoint,
        oldPoints[m.faceVert(f, nPoints - 1)],
        oldPoints[m.faceVert(f, 0)],
        centreNewPoint,
        newPoints[m.faceVert(f, nPoints - 1)],
        newPoints[m.faceVert(f, 0)]);
    return sv;
}

std::unique_ptr<DynamicMotionSolverFvMesh> DynamicMotionSolverFvMesh::New(
    const std::string& caseDir,
    const std::string& startDir)
{
    const std::string path = caseDir + "/constant/dynamicMeshDict";
    // dynamicFvMeshNew.C: no dictionary, a static mesh
    if (!std::filesystem::exists(path)) return nullptr;
    const FoamDict d = readDict(path);
    const std::string meshType = d.wordOr("dynamicFvMesh", "");
    if (meshType.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "constant/dynamicMeshDict has no `dynamicFvMesh` entry. OpenFOAM reads it "
            "with get<word> and stops without one.");
    }
    if (meshType == "staticFvMesh") return nullptr;
    if (meshType != "dynamicMotionSolverFvMesh")
    {
        throw std::runtime_error(
            std::string(WHO) + "constant/dynamicMeshDict asks for `dynamicFvMesh " + meshType + "`. Only "
            "dynamicMotionSolverFvMesh is ported (and staticFvMesh, which is no motion): a mesh that "
            "refines or changes topology is a different mesh at every step, and running the case on "
            "the mesh as written would solve a different problem.");
    }

    // motionSolver::New reads the name with getCompat<word>("motionSolver", {{"solver", -1612}})
    std::string solver = d.wordOr("motionSolver", "");
    if (solver.empty())
    {
        solver = d.wordOr("solver", "");
    }
    if (solver != "solidBody")
    {
        throw std::runtime_error(
            std::string(WHO) + "constant/dynamicMeshDict asks for `motionSolver " + solver + "`. Only "
            "solidBody is ported -- a prescribed rigid transformation of the points. The others SOLVE "
            "for the point motion (displacementLaplacian and its kin, from point boundary conditions) "
            "or integrate a body's equations of motion (rigidBodyMotion, sixDoFRigidBodyMotion).");
    }

    // motionSolver::coeffDict(): optionalSubDict(typeName + "Coeffs")
    const FoamDict& coeffs = *d.optionalSubDict("solidBodyCoeffs");
    for (const char* key : {"cellZone", "cellSet"})
    {
        const std::string name = coeffs.wordOr(key, "");
        // zoneMotion.C:50, :70: `none` is a placeholder for "no selection"
        if (!name.empty() && name != "none")
        {
            throw std::runtime_error(
                std::string(WHO) + "the solidBody motion names `" + key + " " + name + "`. Only the "
                "motion of the ENTIRE mesh is ported: moving part of one deforms the cells around it "
                "or slides it on a coupled interface, and neither is here.");
        }
    }
    for (const std::string& dir : {caseDir + "/constant/polyMesh", startDir + "/polyMesh"})
    {
        if (std::filesystem::exists(dir + "/points0") || std::filesystem::exists(dir + "/points0.gz"))
        {
            throw std::runtime_error(
                std::string(WHO) + dir + "/points0 exists. points0MotionSolver then transforms THOSE "
                "points rather than the mesh's own (points0MotionSolver.C:42-76); reading them is not "
                "ported.");
        }
    }
    const std::filesystem::path constantDir = std::filesystem::path(caseDir)/"constant";
    const bool startIsConstant =
        std::filesystem::weakly_canonical(startDir) == std::filesystem::weakly_canonical(constantDir);
    if (!startIsConstant
        && (std::filesystem::exists(startDir + "/polyMesh/points")
            || std::filesystem::exists(startDir + "/polyMesh/points.gz")))
    {
        throw std::runtime_error(
            std::string(WHO) + startDir + "/polyMesh/points exists: this is a restart of a mesh that has "
            "already moved. OpenFOAM starts from those points and from the meshPhi and Uf written "
            "beside them; brae reads the mesh from constant/polyMesh only.");
    }

    std::unique_ptr<DynamicMotionSolverFvMesh> mesh(new DynamicMotionSolverFvMesh());
    mesh->SBMF_ = SolidBodyMotionFunction::New(coeffs, caseDir);
    mesh->motionType_ = mesh->SBMF_->type();
    return mesh;
}

void DynamicMotionSolverFvMesh::attach(
    PrimitiveMesh& m,
    FvGeometry& g,
    std::vector<FvPatch>& patches)
{
    m_ = &m;
    g_ = &g;
    patches_ = &patches;
    points0_ = m.points();
    oldPoints_ = m.points();
    meshPhi_.internal.assign(static_cast<std::size_t>(m.nInternalFaces()), scalar(0));
    meshPhi_.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        meshPhi_.boundary[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(0));
    }
}

void DynamicMotionSolverFvMesh::update(
    scalar time,
    scalar deltaT,
    label timeIndex)
{
    if (!attached())
    {
        throw std::runtime_error(std::string(WHO) + "update() before attach(): no mesh to move.");
    }
    PrimitiveMesh& m = *m_;
    FvGeometry& g = *g_;

    // fvMesh::movePoints: grab old time volumes if the time has been incremented
    if (!haveTimeIndex_ || curTimeIndex_ < timeIndex)
    {
        V0_ = g.V();
        curTimeIndex_ = timeIndex;
    }
    // polyMesh::movePoints: pick up old points
    moving_ = true;
    if (!haveTimeIndex_ || curMotionTimeIndex_ != timeIndex)
    {
        oldPoints_ = m.points();
        curMotionTimeIndex_ = timeIndex;
    }
    haveTimeIndex_ = true;

    // solidBodyMotionSolver::curPoints, moveAllCells
    std::vector<vector> newPoints = transformPoints(SBMF_->transformation(time), points0_);

    // fvGeometryScheme::setMeshPhi
    const scalar rdt = 1.0/deltaT;
    const label nIf = m.nInternalFaces();
    for (label facei = 0; facei < nIf; ++facei)
    {
        meshPhi_.internal[static_cast<std::size_t>(facei)] =
            faceSweptVolume(m, facei, oldPoints_, newPoints)*rdt;
    }
    for (std::size_t pi = 0; pi < patches_->size(); ++pi)
    {
        const FvPatch& p = (*patches_)[pi];
        // Empty patches
        if (p.type == "empty") continue;
        for (label i = 0; i < p.size; ++i)
        {
            meshPhi_.boundary[pi][static_cast<std::size_t>(i)] =
                faceSweptVolume(m, p.start + i, oldPoints_, newPoints)*rdt;
        }
    }

    // ...and the geometry from the new points, each patch's copy in place
    m.movePoints(std::move(newPoints));
    g.build(m);
    const std::vector<FvPatch> rebuilt = buildPatches(m, g);
    if (rebuilt.size() != patches_->size())
    {
        throw std::runtime_error(std::string(WHO) + "the patch list changed size under a point motion.");
    }
    for (std::size_t pi = 0; pi < rebuilt.size(); ++pi)
    {
        (*patches_)[pi] = rebuilt[pi];
    }
}

std::vector<scalar> DynamicMotionSolverFvMesh::Vsc(const SubCycleTimeState& ts) const
{
    const std::vector<scalar>& V = g_->V();
    if (moving_ && ts.subCycling)
    {
        const scalar tFrac = (ts.value - (ts.value0 - ts.deltaT0))/ts.deltaT0;
        // SMALL
        if (tFrac < (1 - scalar(1e-15)))
        {
            std::vector<scalar> v(V.size());
            for (std::size_t c = 0; c < V.size(); ++c)
            {
                v[c] = V0_[c] + tFrac*(V[c] - V0_[c]);
            }
            return v;
        }
    }
    return V;
}

std::vector<scalar> DynamicMotionSolverFvMesh::Vsc0(const SubCycleTimeState& ts) const
{
    const std::vector<scalar>& V = g_->V();
    if (moving_ && ts.subCycling)
    {
        const scalar t0Frac = ((ts.value - ts.deltaT) - (ts.value0 - ts.deltaT0))/ts.deltaT0;
        // SMALL
        if (t0Frac > scalar(1e-15))
        {
            std::vector<scalar> v(V.size());
            for (std::size_t c = 0; c < V.size(); ++c)
            {
                v[c] = V0_[c] + t0Frac*(V[c] - V0_[c]);
            }
            return v;
        }
    }
    return V0_;
}

} // namespace brae
