#pragma once
// OpenFOAM's rigidBodyDynamics, the part rigidBodyMeshMotion uses to MOVE POINTS: the spatial
// transform of each body in the chain at a given joint state, and the septernion-slerp interpolation
// that weights it by the mesh-motion scale. The host reference.
//
// provenance:
//   openfoam: src/OpenFOAM/primitives/spatialVectorAlgebra/SpatialTensor/spatialTransform/
//                 spatialTransformI.H:47-56 (the constructor), :106-109 (inv), :137-143 (operator&),
//                 :172-178 (transformPoint), :256-283 (Xrx, Xry, Xrz, Xt)
//             src/OpenFOAM/primitives/transform/transform.H:88-109 (Rx, Ry)
//             src/rigidBodyDynamics/joints/Py/Py.C:41-53 (S_, jcalc),
//                 joints/Ry/Ry.C:41-53, joints/composite/compositeJoint.C:38-52 (jcalc is last())
//             src/rigidBodyDynamics/rigidBodyModel/rigidBodyModel.C:118-160 (join_, XT_ and lambda_),
//                 :162-215 (join for a composite: one jointBody per joint but the last),
//                 :380-392 (X0), forwardDynamics.C:76-96 (Xlambda_ = J.X & XT_, X0_ = Xlambda_ & X0_)
//             src/rigidBodyDynamics/rigidBodyMotion/rigidBodyMotion.C:253-290 (transformPoints with a
//                 weight -- the septernion slerp, and the solid-body branch where the weight is 1)
//             src/OpenFOAM/primitives/septernion/septernionI.H:59-63 (septernion(spatialTransform)),
//                 septernion.C:63-71 (slerp), quaternion.C:81-96 (slerp), :143-175 (pow),
//                 quaternionI.H:206-280 (quaternion from a rotation tensor)
//   brae:
//     reference: this header
//     tests:     tests/test_rigid_body_transform_vs_openfoam.cu, against the pointDisplacement
//                OpenFOAM writes for RAS/floatingObject and the joint state in
//                <time>/uniform/rigidBodyMotionState
//
// WHAT THIS IS NOT. It does not integrate the body's equations of motion: `q` is an input here, taken
// from OpenFOAM's own state. The Newmark solver, the body's inertia and the fluid force and moment are
// the units after this one, and the motion solver refuses the case until they exist.
//
// THE CHAIN A `composite` JOINT MAKES is not one body. rigidBodyModel::join adds a massless jointBody
// for every joint but the last, so `composite (Py Ry)` on one body is two links: root -> jointBody
// through Py with the body's `transform`, then jointBody -> the body through Ry with the identity.
// Reading it as a single body with two degrees of freedom gives a different X0 and a different mesh.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {
namespace RBD {

// spatialTransform: a rotation E and an offset r. transformPoint is E & (p - r), NOT E & p + r.
struct SpatialTransform
{
    tensor E{1, 0, 0, 0, 1, 0, 0, 0, 1};
    vector r{0, 0, 0};
};

SpatialTransform operator&(const SpatialTransform& a, const SpatialTransform& b);
SpatialTransform inv(const SpatialTransform& x);
vector transformPoint(const SpatialTransform& x, const vector& p);
SpatialTransform Xt(const vector& r);
SpatialTransform Xry(scalar omega);

// The joints this unit carries. OpenFOAM has seventeen; the two RAS/floatingObject names are ported
// and every other one is refused where the chain is built, by the name the dictionary gave it.
enum class JointType
{
    Py,     // prismatic along y:  X = Xt(S.l()*q), S = (0 0 0  0 1 0)
    Ry      // revolute about y:   X = Xry(q),      S = (0 1 0  0 0 0)
};

// One link of the chain: the joint that reaches this body, the q it reads, the fixed transform from
// the parent's frame (XT_) and which body is the parent (lambda_).
struct Link
{
    JointType joint = JointType::Py;
    label     qIndex = 0;
    SpatialTransform XT;
    label     lambda = 0;
};

struct Model
{
    // links[i] is body i + 1; body 0 is the root, whose X0 is the identity
    std::vector<Link> links;
    // the body the mesh motion moves -- the last link, which carries the real body
    label bodyID() const { return static_cast<label>(links.size()); }
    label nDoF() const { return static_cast<label>(links.size()); }

    // rigidBodyModel::X0, through forwardDynamics' Xlambda_ = J.X & XT_ and X0_ = Xlambda_ & X0_[lambda]
    std::vector<SpatialTransform> X0(const std::vector<scalar>& q) const;

    // rigidBodyMotion::transformPoints(bodyID, weight, points0): the transform from the INITIAL state
    // in the global frame to the current one, applied whole where the weight is 1 and slerped where it
    // is between. A weight at or below SMALL leaves the point exactly where it was.
    std::vector<vector> transformPoints(
        const std::vector<scalar>& q,
        const std::vector<scalar>& weight,
        const std::vector<vector>& points0) const;
};

// The one body of a `rigidBodyMotionCoeffs` dictionary, its joint chain and its two mesh distances.
struct MotionSpec
{
    Model model;
    std::string bodyName;
    std::vector<std::string> patches;
    scalar innerDistance = 0;
    scalar outerDistance = 0;
};

// pointConstraints::constrainDisplacement, which rigidBodyMeshMotion::solve runs on the displacement
// it has just written: pf.correctBoundaryConditions() evaluates every POINT patch field, and a
// valuePointPatchField's evaluate WRITES its value into the shared point field
// (valuePointPatchField.C: setInInternalField). So a `fixedValue` point patch pins every point it
// owns, including points the body's transform moved -- on RAS/floatingObject the tank's own walls sit
// inside the body's outerDistance and OpenFOAM holds 291 of them at zero where the blend would have
// moved them. Patches are evaluated in patch order, so a point on two of them takes the LAST one's.
//
// Reads the field file for the patch types. Refuses any point-patch type that is neither `fixedValue`
// (which pins) nor `calculated` (which does not evaluate at all).
void constrainPointDisplacement(
    const std::string&          fieldPath,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    std::vector<vector>&        displacement);

// Read constant/dynamicMeshDict. Refuses anything this unit does not carry, by name: a motionSolver
// that is not rigidBodyMotion, more than one body, a joint other than Py or Ry, a `mergeWith` body,
// and a parent that is not `root`.
MotionSpec readMotionSpec(const std::string& dictPath);

} // namespace RBD
} // namespace brae
