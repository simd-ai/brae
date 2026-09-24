// OpenFOAM's rigidBodyMeshMotion: a body whose equations of motion the fluid load drives, and the
// mesh that blends between moving with it and standing still.
//
// OF v2412: src/rigidBodyMeshMotion/rigidBodyMeshMotion/rigidBodyMeshMotion.C:88-389
//           src/rigidBodyDynamics/rigidBodyMotion/rigidBodyMotion.C:151-197 (solve, forwardDynamics)
//           src/rigidBodyDynamics/rigidBodySolvers/Newmark/Newmark.C:76-100
//
// The five pieces this assembles are each gated on their own against real OpenFOAM: the point-patch
// distance and the cosine scale (rigid_body_dynamics unit 1), the joint chain and the slerped point
// transform (unit 2), the fluid force on the body (unit 3), the Newmark integrator and the
// acceleration relaxation (unit 4a) and the articulated-body algorithm (unit 4b). What is new here is
// the ASSEMBLY: the once-per-time-index state roll, the state the dynamics is evaluated at when a
// step solves more than once, and the displacement the mesh is then moved by.
//
// NOT PORTED, refused by name where the dictionary is read (see RBD::readMotionSpec): `ramp` (it
// scales gravity AND the fluid force), `test`, `nIter`, `cOfGdisplacement`, `bodyIdCofG`,
// `restraints`, more than one body, a body that is not a `cuboid`, a joint other than Py or Ry, a
// `mergeWith` body and a parent that is not `root`. Here, additionally: `rho rhoInf` (a reference
// density in place of the live field) and a patch entry that is a regular expression.
#pragma once

#include "cf_types.cuh"
#include "forces.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "primitive_mesh.cuh"
#include "rigid_body_motion_cpp.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {

// What the fluid puts on the body, as the driver has it at the moment the mesh is moved. The pressure
// is the TOTAL pressure p, not p_rgh: OpenFOAM's `forces` looks up `p` and this case's `rho` is the
// live mixture field, so the buoyancy the body floats on is in the pressure it is handed.
struct BodyLoad
{
    const GeometricField<vector>*           U = nullptr;
    // per patch, per face
    const std::vector<std::vector<scalar>>* p = nullptr;
    const std::vector<std::vector<scalar>>* rho = nullptr;
    const std::vector<std::vector<scalar>>* nuEff = nullptr;
};

class RigidBodyMeshMotion
{
public:
    // From the case's dictionaries alone: constant/dynamicMeshDict for the body and the chain,
    // constant/g for gravity, and <startTime>/uniform/rigidBodyMotionState for the state to start
    // from -- which OpenFOAM reads only if that file is there and otherwise takes from the coeffs
    // (rigidBodyMeshMotion.C:93-118), where the tutorial writes none, so a start from rest.
    static std::unique_ptr<RigidBodyMeshMotion> New(
        const std::string& caseDir,
        const std::string& startDir);

    // The mesh as it stands, which is the mesh points0 is taken from. The point-patch distance and the
    // scale are computed ONCE here, on points0, as OpenFOAM computes them once in its constructor
    // (rigidBodyMeshMotion.C:172-205) -- recomputing them on the moved mesh would make the blend drift
    // with the body.
    void attach(
        const PrimitiveMesh&        m,
        const FvGeometry&           g,
        const std::vector<FvPatch>& patches,
        const std::vector<vector>&  points0);

    // motionSolver::newPoints(): solve() at this instant, then curPoints() = points0 + displacement.
    // Called ONCE PER OUTER CORRECTOR under moveMeshOuterCorrectors, and the state roll inside is what
    // makes the three of them an iteration rather than three sub-steps.
    std::vector<vector> newPoints(
        scalar                      time,
        scalar                      deltaT,
        label                       timeIndex,
        const PrimitiveMesh&        m,
        const FvGeometry&           g,
        const std::vector<FvPatch>& patches,
        const BodyLoad&             load);

    const RBD::MotionSpec& spec() const
    {
        return spec_;
    }
    const RBD::ModelState& state() const
    {
        return state_;
    }
    const std::vector<vector>& pointDisplacement() const
    {
        return pointDisplacement_;
    }
    // the cosine blend, one value per mesh point: 1 inside innerDistance, 0 beyond outerDistance
    const std::vector<scalar>& weight() const
    {
        return weight_;
    }
    // the spatial force the last solve was handed, (angular, linear) about the global origin
    const RBD::SpatialVector& lastForce() const
    {
        return lastForce_;
    }
    bool report() const
    {
        return spec_.report;
    }

private:
    RBD::MotionSpec      spec_;
    // motionState_ and motionState0_: the current joint state and the one at the start of the step
    RBD::ModelState      state_;
    RBD::ModelState      state0_;
    std::vector<scalar>  weight_;
    std::vector<vector>  points0_;
    std::vector<vector>  pointDisplacement_;
    std::vector<label>   bodyPatches_;
    // the field file the point-patch TYPES are read from, for the constraint that ends every solve
    std::string          pointDisplacementPath_;
    vector               g_{0, 0, 0};
    RBD::SpatialVector   lastForce_;
    label                curTimeIndex_ = -1;
    bool                 haveTimeIndex_ = false;
    bool                 attached_ = false;
};

}   // namespace brae
