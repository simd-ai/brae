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
#include "function1.cuh"
#include "spatial_algebra_cpp.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
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

// THE FOUR ACTIONS a spatialTransform has, which are four different pieces of arithmetic and not one
// with a flag (spatialTransformI.H). Reading a force through the motion action, or a motion through
// the dual, is silent -- both give a six-vector of plausible size.
//     X & v        motion:  ( E & v.w,  E & (v.l - (r ^ v.w)) )                        :146-156
//     *X & f       dual:    ( E & (f.w - (r ^ f.l)),  E & f.l )                         :216-226
//     X.T() & f    transpose, with ETfl = E^T & f.l:                                    :191-203
//                           ( (E^T & f.w) + (r ^ ETfl),  ETfl )
//     spatialTensor(X)     = [ E, 0 ; -Erx(), E ]     with Erx() = E & (*r)             :120-127
//     spatialTensor(X.T()) = [ E^T, -(Erx())^T ; 0, E^T ]                               :181-188
SpatialVector motionAction(const SpatialTransform& X, const SpatialVector& v);
SpatialVector dualAction(const SpatialTransform& X, const SpatialVector& f);
SpatialVector transposeAction(const SpatialTransform& X, const SpatialVector& f);
SpatialTensor motionTensor(const SpatialTransform& X);
SpatialTensor transposeTensor(const SpatialTransform& X);

// rigidBodyInertia (rigidBodyInertiaI.H): the mass, the centre of mass IN THE BODY FRAME, and the
// inertia ABOUT THE CENTRE OF MASS.
struct RigidBodyInertia
{
    scalar m = 0;
    vector c{0, 0, 0};
    symmTensor Ic{0, 0, 0, 0, 0, 0};
};

// Io = Ic + m*(I*|c|^2 - c (x) c)   (rigidBodyInertiaI.H:35-42, :119-122). REBUILT ON EVERY CALL,
// as OpenFOAM does: caching it changes the rounding of the line that reads it.
symmTensor inertiaIo(const RigidBodyInertia& rbi);

// The 6x6 form (rigidBodyInertiaI.H:127-136): [ Io, m*skew(c) ; -m*skew(c), m*I ].
SpatialTensor inertiaTensor(const RigidBodyInertia& rbi);

// I & v, the SPECIALISED overload (rigidBodyInertiaI.H:192-206) -- NOT the 6x6 product, which is
// algebraically the same and arithmetically different:
//     ( (Io & w) + m*(c ^ l),   m*l - m*(c ^ w) )
// Note `m*l - m*(c ^ w)`: two separate multiplications by m, not m*(l - c^w).
SpatialVector inertiaAction(const RigidBodyInertia& rbi, const SpatialVector& v);

// cuboid (bodies/cuboid/cuboidI.H:30-47): Ic = diag(m/12 * (Ly^2 + Lz^2), ...). The dictionary's
// `rho` and `Lx`/`Ly`/`Lz` are NOT read by the body -- they exist to feed the #eval for `mass`.
RigidBodyInertia cuboidInertia(scalar mass, const vector& centreOfMass, const vector& L);

// One link of the chain: the joint that reaches this body, the q it reads, the fixed transform from
// the parent's frame (XT_) and which body is the parent (lambda_).
struct Link
{
    JointType joint = JointType::Py;
    label     qIndex = 0;
    SpatialTransform XT;
    label     lambda = 0;
    // the body this joint reaches. A composite joint's intermediate links are massless `jointBody`s
    // (bodies/jointBody/jointBodyI.H), which contribute nothing but a link and a degree of freedom.
    RigidBodyInertia inertia;
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

    // rigidBodyModel::forwardDynamics (forwardDynamics.C:56-206) -- the articulated-body algorithm,
    // three passes over the chain. `fx` is per BODY (index 0 is the root and is never read), a spatial
    // force in the GLOBAL frame about the global origin, which is where the mesh motion's `forces`
    // object puts it. Gravity enters as a base acceleration a[0] = (0, -g) and nowhere else.
    std::vector<scalar> forwardDynamics(
        const std::vector<scalar>&       q,
        const std::vector<scalar>&       qDot,
        const std::vector<SpatialVector>& fx,
        const vector&                    g) const;

    // rigidBodyMotion::transformPoints(bodyID, weight, points0): the transform from the INITIAL state
    // in the global frame to the current one, applied whole where the weight is 1 and slerped where it
    // is between. A weight at or below SMALL leaves the point exactly where it was.
    std::vector<vector> transformPoints(
        const std::vector<scalar>& q,
        const std::vector<scalar>& weight,
        const std::vector<vector>& points0) const;
};

// rigidBodyModelState (rigidBodyModelState.H:63-81): the joint position, velocity and acceleration,
// and the clock the state was integrated to. `deltaT` is the step that REACHED this state, which is
// the previous step's when it is read as `motionState0_`.
struct ModelState
{
    std::vector<scalar> q;
    std::vector<scalar> qDot;
    std::vector<scalar> qDdot;
    // rigidBodyModelState.C:46-71 -- every entry is optional and these are its defaults
    scalar t = -1;
    scalar deltaT = 0;
};

// Newmark (rigidBodySolvers/Newmark/Newmark.C:56-64). `beta` is CLAMPED FROM BELOW by gamma, which is
// what makes the scheme unconditionally stable; the dictionary's beta is a floor, not the value.
struct NewmarkCoeffs
{
    scalar gamma = scalar(0.5);
    scalar beta = scalar(0.25);
};

inline NewmarkCoeffs newmarkCoeffs(scalar gamma, scalar betaEntry)
{
    NewmarkCoeffs c;
    c.gamma = gamma;
    const scalar g = gamma + scalar(0.5);
    c.beta = std::fmax(scalar(0.25)*g*g, betaEntry);
    return c;
}

// Newmark.C:86-104. BOTH updates read qDdot at the new state AND at the old one -- and `s0` is the
// state at the START OF THE TIME STEP, not the previous corrector's, so re-solving within a step is
// an iteration and not a sub-step. At the defaults gamma 0.5 / beta 0.25 the two weights of each pair
// are equal, so this case cannot tell them apart; the gate's `gamma 0.9` arm can.
void newmarkSolve(
    const ModelState&    s0,
    const NewmarkCoeffs& c,
    ModelState&          s);

// rigidBodyMotion::forwardDynamics (rigidBodyMotion.C:150-170): the acceleration the dynamics just
// produced is relaxed IN PLACE against the one the state carried at entry -- which is the PREVIOUS
// CORRECTOR's relaxed value, not the start-of-step one. Measured on floatingObject: with
// accelerationRelaxation 0.7 the written qDdot is 0.7 of the value an otherwise identical run with 1.0
// leaves, to a relative difference of 0.
void relaxAcceleration(
    std::vector<scalar>&       qDdot,
    const std::vector<scalar>& qDdotPrev,
    scalar                     aRelax,
    scalar                     aDamp);

// The one body of a `rigidBodyMotionCoeffs` dictionary, its joint chain, its two mesh distances and
// the solver that integrates it.
struct MotionSpec
{
    Model model;
    std::string bodyName;
    std::vector<std::string> patches;
    scalar innerDistance = 0;
    scalar outerDistance = 0;
    // `solver { type Newmark; gamma; beta; }` -- the only integrator ported; the others are refused
    // by name where the dictionary is read.
    NewmarkCoeffs newmark;
    // rigidBodyMotion.C:80-82 -- `accelerationRelaxation` (default 1) and `accelerationDamping`
    // (default 1). The tutorial writes the first as a table that is ZERO until t = 4, which is why a
    // gate that ran the shipped case would be measuring a body that never moves.
    Function1 accelerationRelaxation;
    scalar accelerationDamping = 1;
    // `report` -- the status block the log carries, which is the cheapest oracle for a body's state
    bool report = false;
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
