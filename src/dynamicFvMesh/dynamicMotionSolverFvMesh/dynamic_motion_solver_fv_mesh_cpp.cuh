#pragma once
// OpenFOAM's dynamicMotionSolverFvMesh with the solidBody motion solver, the host reference: what
// mesh.update() does to the mesh, and everything a solver then reads from a mesh that has moved.
//
// provenance:
//   openfoam:  src/dynamicFvMesh/dynamicFvMesh/dynamicFvMeshNew.C:65-128 (no dictionary: static)
//              src/dynamicFvMesh/dynamicMotionSolverFvMesh/dynamicMotionSolverFvMesh.C:101-114 (update)
//              src/dynamicMesh/motionSolvers/displacement/solidBody/solidBodyMotionSolver.C:70-91
//              src/dynamicMesh/motionSolvers/displacement/displacement/zoneMotion.C:37-127
//              src/dynamicMesh/motionSolvers/displacement/points0/points0MotionSolver.C:42-76
//              src/finiteVolume/fvMesh/fvMesh.C (movePoints), fvMeshGeometry.C (storeOldVol, V0,
//                  Vsc, Vsc0, phi)
//              src/finiteVolume/fvMesh/fvGeometryScheme/fvGeometryScheme/fvGeometryScheme.C:60-117
//                  (setMeshPhi)
//              src/OpenFOAM/meshes/polyMesh/polyMesh.C (movePoints, oldPoints)
//              src/OpenFOAM/meshes/meshShapes/face/face.C:513-563 (centre), :656-730 (sweptVol)
//              src/OpenFOAM/meshes/primitiveShapes/triangle/triangleI.H:386-401 (sweptVol)
//   tests:     tests/test_mesh_motion_vs_openfoam.cu, against the polyMesh/points and meshPhi
//              OpenFOAM writes per time step and the V and C its postProcess writes from them
//
// THE ORDER INSIDE ONE update(), which is fvMesh::movePoints':
//   1. the volumes the mesh HAS become V0 -- once per time index, so a second update in the same step
//      (moveMeshOuterCorrectors) does not overwrite them;
//   2. the points the mesh HAS become oldPoints, under the same once-per-index rule;
//   3. the points are replaced by the motion solver's, which are an absolute function of the NEW time
//      applied to points0 -- never an increment;
//   4. meshPhi = sweptVol(oldPoints, newPoints)*(1.0/deltaT) on every face -- a product with the
//      reciprocal, not a quotient, and brae compares it to the last digit;
//   5. the geometry is recomputed from the new points: Sf, magSf, Cf, C, V, and the interpolation
//      weights and deltaCoeffs with them, and each FvPatch's copy IN PLACE, because every patch field
//      holds a reference to its FvPatch.
//
// face::sweptVol fans each face about face::centre() at both times, and face::centre is NOT the
// mesh's Cf: primitiveMesh computes Cf with its own sums (primitiveMeshFaceCentresAndAreas.C), and
// the two differ in the last digits on any face that is not a triangle. src/OpenFOAM/motion/
// swept_volume.cuh uses a third arrangement of the same cross product; this file transcribes face.C.
//
// NOT PORTED, refused by name where the dictionary is read: every dynamicFvMesh but
// dynamicMotionSolverFvMesh (and staticFvMesh, which is no motion); every motionSolver but solidBody;
// a `cellZone` or `cellSet` (the motion of part of a mesh deforms the cells around it, and the
// interFoam tutorial that asks for one slides it on an AMI); a `points0` file; and a start from a
// time directory that carries its own polyMesh/points.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "primitive_mesh.cuh"
#include "solid_body_motion_function_cpp.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {

// face::centre(points) and face::sweptVol(oldPoints, newPoints), for face f of the mesh's topology
vector faceCentreOfPoints(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& points);

scalar faceSweptVolume(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& oldPoints,
    const std::vector<vector>& newPoints);

// Time::subCycle's two TimeStates, as fvMesh::Vsc and Vsc0 read them
struct SubCycleTimeState
{
    bool subCycling = false;
    // the sub-cycle's own time and step
    scalar value = 0;
    scalar deltaT = 0;
    // ...and the step it divides (prevTimeState)
    scalar value0 = 0;
    scalar deltaT0 = 0;
};

class DynamicMotionSolverFvMesh
{
public:
    // dynamicFvMesh::New for the case, from its dictionaries alone. Returns null for a mesh that does
    // not move -- no constant/dynamicMeshDict, or `dynamicFvMesh staticFvMesh` -- and throws, by
    // name, for every motion that is not ported. Reading is separate from attach() so that a case
    // reader with no mutable mesh in hand still refuses what it must.
    static std::unique_ptr<DynamicMotionSolverFvMesh> New(
        const std::string& caseDir,
        const std::string& startDir);

    // The mesh, its geometry and its patches this motion MOVES IN PLACE: the caller's, which must
    // outlive this object. Takes points0 from the mesh as it stands.
    void attach(
        PrimitiveMesh& m,
        FvGeometry& g,
        std::vector<FvPatch>& patches);
    bool attached() const
    {
        return m_ != nullptr;
    }

    // mesh.update() at the new time
    void update(
        scalar time,
        scalar deltaT,
        label timeIndex);

    // polyMesh::moving(): false until the first update
    bool moving() const
    {
        return moving_;
    }
    const std::string& motionType() const
    {
        return motionType_;
    }
    const std::vector<vector>& points0() const
    {
        return points0_;
    }
    const std::vector<vector>& oldPoints() const
    {
        return oldPoints_;
    }
    // fvMesh::phi(): the mesh-motion flux of the current step. Empty patches carry zeros here;
    // OpenFOAM's have no faces.
    const SurfaceScalarField& meshPhi() const
    {
        return meshPhi_;
    }
    const std::vector<scalar>& V0() const
    {
        return V0_;
    }
    // fvMesh::Vsc() and Vsc0(): V and V0 outside a sub-cycle, and inside one the volumes at the end
    // and at the start of THAT sub-cycle, linear in time between V0 and V
    std::vector<scalar> Vsc(const SubCycleTimeState& ts) const;
    std::vector<scalar> Vsc0(const SubCycleTimeState& ts) const;

private:
    DynamicMotionSolverFvMesh() = default;

    PrimitiveMesh* m_ = nullptr;
    FvGeometry* g_ = nullptr;
    std::vector<FvPatch>* patches_ = nullptr;
    std::unique_ptr<SolidBodyMotionFunction> SBMF_;
    std::string motionType_;
    std::vector<vector> points0_;
    std::vector<vector> oldPoints_;
    std::vector<scalar> V0_;
    SurfaceScalarField meshPhi_;
    bool moving_ = false;
    // fvMesh::curTimeIndex_ and polyMesh::curMotionTimeIndex_
    label curTimeIndex_ = -1;
    label curMotionTimeIndex_ = -1;
    bool haveTimeIndex_ = false;
};

} // namespace brae
