#include "dynamic_motion_solver_fv_mesh_cpp.cuh"
#include "foam_dict.cuh"
#include "mrf_read.cuh"
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
    if (solver != "solidBody" && solver != "displacementLaplacian")
    {
        throw std::runtime_error(
            std::string(WHO) + "constant/dynamicMeshDict asks for `motionSolver " + solver + "`. Only "
            "solidBody -- a prescribed rigid transformation of the points -- and displacementLaplacian "
            "are ported. The others solve for the point motion another way (velocityLaplacian, "
            "displacementSBRStress, ...) or integrate a body's equations of motion (rigidBodyMotion, "
            "sixDoFRigidBodyMotion).");
    }

    // motionSolver::coeffDict(): optionalSubDict(typeName + "Coeffs")
    const FoamDict& coeffs = *d.optionalSubDict(solver + "Coeffs");
    // zoneMotion.C:47-94: a cellSet, a cellZone or neither; `none` is a placeholder for "no selection"
    std::string cellZone;
    for (const char* key : {"cellZone", "cellSet"})
    {
        std::string name = coeffs.wordOr(key, "");
        if (name == "none")
        {
            name.clear();
        }
        if (name.empty()) continue;
        if (std::string(key) == "cellSet" || solver != "solidBody")
        {
            throw std::runtime_error(
                std::string(WHO) + "the " + solver + " motion names `" + key + " " + name + "`. Only a "
                "solidBody motion of a cellZone is ported: reading a cellSet from constant/polyMesh/sets "
                "is not, and a displacement solver restricted to part of the mesh is not.");
        }
        // zoneMotion.C:84: cellZones().indices(wordRe) matches a regular expression and zone groups
        if (name.find_first_of(".*+?|[](){}^$\\") != std::string::npos)
        {
            throw std::runtime_error(
                std::string(WHO) + "`cellZone " + name + "` looks like a regular expression. OpenFOAM "
                "matches it as a wordRe against every zone and zone group; brae matches one zone by its "
                "literal name only.");
        }
        cellZone = name;
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
    if (solver == "displacementLaplacian")
    {
        mesh->displacement_ = DisplacementLaplacianFvMotionSolver::New(coeffs, caseDir, startDir);
        mesh->motionType_ = solver;
        return mesh;
    }
    if (!cellZone.empty())
    {
        const std::map<std::string, std::vector<label>> zones = readCellZones(caseDir + "/constant/polyMesh");
        const auto z = zones.find(cellZone);
        // zoneMotion.C:86-96
        if (z == zones.end())
        {
            throw std::runtime_error(
                std::string(WHO) + "No matching cellZones: " + cellZone + " in constant/polyMesh/cellZones.");
        }
        mesh->zoneCells_ = z->second;
        mesh->cellZone_ = cellZone;
    }
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
    // zoneMotion.C:99-122: every point of every face of every zone cell, in ascending order. The
    // syncPointList there exchanges across processor boundaries; serial, it marks nothing more
    // (MEASURED on mixerVesselAMI at 82,510 cells: brae's moved points are OpenFOAM's written ones).
    pointIDs_.clear();
    if (!cellZone_.empty())
    {
        // mesh.cells()[celli] is every face whose owner or neighbour is celli
        std::vector<char> inZone(static_cast<std::size_t>(m.nCells()), 0);
        for (const label celli : zoneCells_)
        {
            inZone[static_cast<std::size_t>(celli)] = 1;
        }
        std::vector<char> movePts(m.points().size(), 0);
        for (label facei = 0; facei < m.nFaces(); ++facei)
        {
            const bool own = inZone[static_cast<std::size_t>(m.owner()[facei])];
            const bool nei = facei < m.nInternalFaces() && inZone[static_cast<std::size_t>(m.neighbour()[facei])];
            if (!own && !nei) continue;
            for (label j = 0; j < m.faceSize(facei); ++j)
            {
                movePts[static_cast<std::size_t>(m.faceVert(facei, j))] = 1;
            }
        }
        for (std::size_t pointi = 0; pointi < movePts.size(); ++pointi)
        {
            if (movePts[pointi])
            {
                pointIDs_.push_back(static_cast<label>(pointi));
            }
        }
    }
    if (displacement_)
    {
        displacement_->attach(m, g, patches);
    }
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
    label timeIndex,
    bool finalIteration,
    GamgAgglomerationCache* agglomeration)
{
    if (!attached())
    {
        throw std::runtime_error(std::string(WHO) + "update() before attach(): no mesh to move.");
    }
    PrimitiveMesh& m = *m_;
    FvGeometry& g = *g_;

    // motionSolver::newPoints(), evaluated before fvMesh::movePoints is entered: on the mesh as it
    // stands. A displacement solver's GAMG hierarchy is the MESH's, shared with every other GAMG solve
    // of the run, so the caller hands in the one it keeps.
    std::vector<vector> newPoints;
    if (displacement_)
    {
        if (!agglomeration)
        {
            throw std::runtime_error(
                std::string(WHO) + "the displacementLaplacian solve needs the run's GAMG agglomeration "
                "cache: the hierarchy is a MeshObject of the mesh, and every GAMG solve shares it.");
        }
        newPoints = displacement_->newPoints(time, finalIteration, m, g, *patches_, *agglomeration);
    }
    else
    {
        // solidBodyMotionSolver::curPoints: moveAllCells() is pointIDs_.empty() (zoneMotion.C:127)
        if (pointIDs_.empty())
        {
            newPoints = transformPoints(SBMF_->transformation(time), points0_);
        }
        else
        {
            // the mesh's CURRENT points, with the zone's points transformed from points0
            newPoints = m.points();
            std::vector<vector> zonePoints0(pointIDs_.size());
            for (std::size_t i = 0; i < pointIDs_.size(); ++i)
            {
                zonePoints0[i] = points0_[static_cast<std::size_t>(pointIDs_[i])];
            }
            const std::vector<vector> moved = transformPoints(SBMF_->transformation(time), zonePoints0);
            for (std::size_t i = 0; i < pointIDs_.size(); ++i)
            {
                newPoints[static_cast<std::size_t>(pointIDs_[i])] = moved[i];
            }
        }
    }

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
    // meshObject::movePoints: GAMGAgglomeration::movePoints sets requireUpdate_ whenever the time index
    // is a multiple of updateInterval, which is 1, and the next GAMGAgglomeration::New builds the
    // hierarchy again on the moved mesh, from wherever the static pairing direction was left
    // (GAMGAgglomeration.C:311-330, :498-516)
    if (agglomeration)
    {
        agglomeration->built = false;
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
