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

// motionSolver::New(mesh, dict) on ONE motion dictionary: the whole dynamicMeshDict for
// dynamicMotionSolverFvMesh, or one sub-dictionary of `solvers` for a list (listForm).
std::unique_ptr<DynamicMotionSolverFvMesh> DynamicMotionSolverFvMesh::fromMotionDict(
    const FoamDict&    d,
    const std::string& caseDir,
    const std::string& startDir,
    bool               listForm)
{
    std::unique_ptr<DynamicMotionSolverFvMesh> mesh(new DynamicMotionSolverFvMesh());
    // motionSolver::New reads the name with getCompat<word>("motionSolver", {{"solver", -1612}})
    std::string solver = d.wordOr("motionSolver", "");
    if (solver.empty())
    {
        solver = d.wordOr("solver", "");
    }
    // ...and under refinement, ONE solidBody motion of the whole mesh (see NewForRefine)
    if (listForm && solver != "solidBody")
    {
        throw std::runtime_error(
            std::string(WHO) + "a refining mesh asks for `motionSolver " + solver + "`. Only solidBody is "
            "ported beside refinement: a displacement solver's point field, or a rigid body's blend, would "
            "have to be carried through every topology change.");
    }
    if (solver != "solidBody" && solver != "displacementLaplacian" && solver != "rigidBodyMotion")
    {
        throw std::runtime_error(
            std::string(WHO) + "constant/dynamicMeshDict asks for `motionSolver " + solver + "`. Only "
            "solidBody -- a prescribed rigid transformation of the points -- displacementLaplacian and "
            "rigidBodyMotion are ported. The others solve for the point motion another way "
            "(velocityLaplacian, displacementSBRStress, ...) or integrate a body's equations of motion "
            "another way (sixDoFRigidBodyMotion).");
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
        if (listForm)
        {
            throw std::runtime_error(
                std::string(WHO) + "a refining mesh's solidBody motion names `" + key + " " + name + "`. "
                "zoneMotion builds its point list once, in its constructor, and has no updateMesh "
                "(zoneMotion.H:76-105), so after a topology change OpenFOAM itself moves stale point labels; "
                "only a motion of the whole mesh is ported beside refinement.");
        }
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

    if (solver == "displacementLaplacian")
    {
        mesh->displacement_ = DisplacementLaplacianFvMotionSolver::New(coeffs, caseDir, startDir);
        mesh->motionType_ = solver;
        return mesh;
    }
    if (solver == "rigidBodyMotion")
    {
        mesh->rigidBody_ = RigidBodyMeshMotion::New(caseDir, startDir);
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
    mesh->listForm_ = listForm;
    return mesh;
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

    return fromMotionDict(d, caseDir, startDir, false);
}

std::unique_ptr<DynamicMotionSolverFvMesh> DynamicMotionSolverFvMesh::NewForRefine(
    const std::string& caseDir,
    const std::string& startDir)
{
    const std::string path = caseDir + "/constant/dynamicMeshDict";
    if (!std::filesystem::exists(path)) return nullptr;
    const FoamDict d = readDict(path);
    if (d.wordOr("dynamicFvMesh", "") != "dynamicRefineFvMesh")
    {
        throw std::runtime_error(
            std::string(WHO) + "NewForRefine is the motion of a dynamicRefineFvMesh; this dictionary asks for `"
            + d.wordOr("dynamicFvMesh", "") + "`.");
    }
    const FoamDict* solvers = d.subDict("solvers");
    if (!solvers || solvers->subs.empty()) return nullptr;
    if (solvers->subs.size() > 1)
    {
        std::string names;
        for (const std::pair<std::string, FoamDict>& sub : solvers->subs)
        {
            names += (names.empty() ? "" : ", ") + sub.first;
        }
        throw std::runtime_error(
            std::string(WHO) + "constant/dynamicMeshDict names " + std::to_string(solvers->subs.size())
            + " motion solvers under refinement (" + names + "). dynamicMotionSolverListFvMesh sums their "
            "displacements (dynamicMotionSolverListFvMesh.C:176-183); one is ported.");
    }
    return fromMotionDict(solvers->subs.front().second, caseDir, startDir, true);
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
    if (rigidBody_)
    {
        // rigidBodyMeshMotion.C:172-206: the blend is built ONCE, on points0
        rigidBody_->attach(m, g, patches, points0_);
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
    GamgAgglomerationCache* agglomeration,
    const BodyLoad* load)
{
    if (!attached())
    {
        throw std::runtime_error(std::string(WHO) + "update() before attach(): no mesh to move.");
    }
    PrimitiveMesh& m = *m_;
    FvGeometry& g = *g_;
    // THE SIZES A TOPOLOGY CHANGE MUST HAVE CARRIED. The swept-volume loops below index meshPhi_ and the
    // new points by the mesh's CURRENT face and point counts, and movePoints' own size check comes after
    // them -- so a change this object was not told about (topoChanged) would write past the end first.
    if (points0_.size() != m.points().size()
     || meshPhi_.internal.size() != static_cast<std::size_t>(m.nInternalFaces())
     || meshPhi_.boundary.size() != patches_->size())
    {
        throw std::runtime_error(
            std::string(WHO) + "the mesh has " + std::to_string(m.points().size()) + " points and "
            + std::to_string(m.nInternalFaces()) + " internal faces; points0 has "
            + std::to_string(points0_.size()) + " and the mesh flux " + std::to_string(meshPhi_.internal.size())
            + ". A topology change reached the mesh without reaching its motion (topoChanged).");
    }

    // motionSolver::newPoints(), evaluated before fvMesh::movePoints is entered: on the mesh as it
    // stands. A displacement solver's GAMG hierarchy is the MESH's, shared with every other GAMG solve
    // of the run, so the caller hands in the one it keeps.
    std::vector<vector> newPoints;
    if (rigidBody_)
    {
        if (!load)
        {
            throw std::runtime_error(
                std::string(WHO) + "a rigidBodyMotion mesh was asked to move without the fluid load. "
                "The driver must hand the body the pressure and the shear on its patches: a body "
                "moved without them is a body with no weight on it.");
        }
        newPoints = rigidBody_->newPoints(time, deltaT, timeIndex, m, g, *patches_, *load);
    }
    else if (displacement_)
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

    // dynamicMotionSolverListFvMesh::update (:176-183): the list sums each solver's displacement from
    // the CURRENT points and moves to points() + disp -- fl(p + fl(q - p)), not q. The two agree whenever
    // q - p is exact, and differ by an ulp near zero; every step of a refining mesh with a motion takes
    // this form, and the swept volumes below are taken to THESE points.
    if (listForm_)
    {
        const std::vector<vector>& p = m.points();
        for (std::size_t i = 0; i < newPoints.size(); ++i)
        {
            newPoints[i] = p[i] + (newPoints[i] - p[i]);
        }
    }

    // fvMesh::movePoints: grab old time volumes if the time has been incremented
    if (!haveTimeIndex_ || curTimeIndex_ < timeIndex)
    {
        // ...and the level BEFORE that, which fvMesh::V00() keeps (fvMesh.C:1015-1040) and the
        // CrankNicolson ddt's moving branch weights its ddt0 by. OpenFOAM creates V00 lazily, as a
        // copy of V0 the first time it is asked for, and rotates it from then on -- so the step that
        // creates it reads V0 in its place, which is what an unmoved mesh would give anyway.
        if (V00Exists_) V00_ = V0_;
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
    // ...WITH THE cyclicACMI EXEMPTION THE CALLER ALREADY HELD. buildPatches refuses a cyclicACMI
    // unless it is told the caller is the OF-mirror interFoam loop, which couples the pair itself
    // (fv_patch.cu, `mirrorACMI`). A list that already HOLDS a cyclicACMI patch was built with that
    // flag -- it could not exist otherwise -- so rebuilding it without one made the move throw the
    // refusal the start had legitimately passed, with the mesh already moved.
    bool mirrorACMI = false;
    for (const FvPatch& q : *patches_)
    {
        mirrorACMI = mirrorACMI || q.type == "cyclicACMI";
    }
    const std::vector<FvPatch> rebuilt = buildPatches(m, g, mirrorACMI);
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

void DynamicMotionSolverFvMesh::topoChanged(
    std::vector<scalar> V0,
    label               timeIndex)
{
    if (!attached())
    {
        throw std::runtime_error(std::string(WHO) + "topoChanged() before attach(): no mesh.");
    }
    const PrimitiveMesh& m = *m_;
    if (V0.size() != static_cast<std::size_t>(m.nCells()) || points0_.size() != m.points().size())
    {
        throw std::runtime_error(
            std::string(WHO) + "after the topology change the mesh has " + std::to_string(m.nCells())
            + " cells and " + std::to_string(m.points().size()) + " points; the change's V0 has "
            + std::to_string(V0.size()) + " and points0 " + std::to_string(points0_.size())
            + ". The refiner carries both (RefineUpdateState::V0, ::points0).");
    }
    // fvMesh::updateMesh: storeOldVol (once per index), then the mapped and corrected V0
    V0_ = std::move(V0);
    curTimeIndex_ = timeIndex;
    haveTimeIndex_ = true;
    // meshPhi recreated as zero on the new mesh, with no old-time level (fvMesh.C:1057-1077)
    meshPhi_.internal.assign(static_cast<std::size_t>(m.nInternalFaces()), scalar(0));
    meshPhi_.boundary.resize(patches_->size());
    for (std::size_t pi = 0; pi < patches_->size(); ++pi)
    {
        meshPhi_.boundary[pi].assign(static_cast<std::size_t>((*patches_)[pi].size), scalar(0));
    }
    // polyMesh::updateMesh maps oldPoints, and the move that follows overwrites them from the refined
    // points (curMotionTimeIndex_ != timeIndex): sized here only so nothing reads a stale length
    oldPoints_ = m.points();
    // dynamicRefineFvMesh::refine/unrefine and updateTopology: moving(false) at a change (:460, :1461)
    moving_ = false;
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
