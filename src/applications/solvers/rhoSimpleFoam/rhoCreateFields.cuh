#pragma once
// The DEVICE projection of rhoSimpleFoam's createFields.H -- the device twin of rhoCreateFields_cpp.
//
// provenance:
//   openfoam:  applications/solvers/compressible/rhoSimpleFoam/createFields.H
//     also:    .../rhoSimpleFoam/createFieldRefs.H  (psi, and the thermo references)
//   reference: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu  (gated against
//              OpenFOAM's own createFields by tests/rho_createfields_vs_openfoam.sh)
//   cuda:      src/applications/solvers/rhoSimpleFoam/rhoCreateFields.cu
//   tests:     tests/test_rho_createfields_cuda.cu
//
// WHAT THIS IS AND IS NOT. createFields is classified HOST_ONLY and stays that way: reading the case,
// constructing the thermo, resolving pressureControl and deciding whether rho came from disk or from the
// equation of state are dictionary operations with no device work in them, and the _cpp reference already
// does all of it against a gate. What this file adds is the PROJECTION -- uploading that field set and
// deriving the device-side addressing the solver needs. It calls the reference; it does not re-read the
// case, because two readers of the same dictionary is exactly the drift this port exists to avoid.
//
// WHY IT IS A MODULE AND NOT LEFT TO EACH CALLER. Every piece below was hand-rolled inside
// test_rho_simple_step_cuda before this file existed, and hand-rolling it is how the two masks get
// conflated: constrainHbyA asks `assignable` while adjustPhi asks `fixesValue() && !isInletOutlet()`,
// and the two are complements on most patches but not all. MEASURED, not assumed: sbMatched's
// inletOutlet outlet reads assignable=1, fixesValue=1, isInletOutlet=1, so both rules give the same
// answer there -- it is SLIP that separates them, being non-assignable without fixing a value. A caller
// that derives one mask from the other is therefore correct on every patch except the ones that matter,
// which is the hardest kind of wrong to notice.
//
// THE WALL PREDICATE IS THE BC's, NOT THE PATCH TYPE's. wfPatch[pi] comes from
// epsilon.boundary[pi]->isTurbulenceWallFunction(), never from `fvp[pi].type == "wall"` alone: a `wall`
// carrying a plain zeroGradient epsilon is not constrained by OpenFOAM, and the 4-arg
// buildDeviceWallData overload -- which falls back to the type -- is the one that once pinned 545 cells
// on a far-field wall and drove nut 1028x high.
//
// WHAT IS STATIC AND WHAT IS NOT. The mesh, the boundary objects, the two masks and the wall geometry
// are functions of the CASE and are built once. nu, mu, rho and the transport are functions of the
// current temperature and move every iteration -- they are the caller's, refreshed per step, and
// deliberately not cached here where they would go stale silently.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"      // DeviceWallData, buildDeviceWallData
#include "rhoCreateFields_cpp.cuh"  // the host field set this projects
#include "rhoSimpleFoam.cuh"        // RhoSolverFields -- the state the driver mutates
#include <vector>

namespace brae {
namespace gpu {
namespace rhoSimple {

// Everything the driver and the closure need that is a function of the CASE rather than of the
// iteration. Held together so a caller cannot construct half of it.
struct RhoDeviceFields
{
    DeviceMesh           dm;
    DeviceVectorBoundary dbU;
    DeviceBoundary       dbP, dbHe, dbT;
    DeviceBoundary       dbK, dbEps;      // empty on a laminar case
    RhoSolverFields      f;               // the mutable solution state, seeded from the host's

    // constrainHbyA's mask and adjustPhi's mask. They answer DIFFERENT questions -- see the header.
    DeviceBuffer<label>  takeUAtBoundary;
    DeviceBuffer<label>  adjustable;

    // The closure's static geometry. Empty on a laminar case.
    DeviceWallData       wall;
    DeviceBuffer<label>  wfBndMask;      // per BOUNDARY FACE: carries a turbulence wall function
    DeviceBuffer<scalar> wallYBndFace;   // nearWallDist y, per boundary face (0 off a wall-function patch)
    std::vector<label>   wfFaceOfBnd;    // wall-face order -> boundary-face index, for deviceGatherWallNu

    // The TURBULENT INLETS, per boundary face. OpenFOAM recomputes these every updateCoeffs:
    //   turbulentIntensityKineticEnergyInlet       k_b   = 1.5*I^2*magSqr(U_b)
    //   turbulentMixingLengthDissipationRateInlet  eps_b = (Cmu^0.75/L)*k_b^1.5
    // so they move with the solution and cannot be seeded once.
    //
    // Nothing in this lineage produced them. The device closure takes them as MASKS, and a null mask is
    // NOT "this case has no turbulent inlet" -- it is silently no turbulent inlet at all, on a case whose
    // 0/k and 0/epsilon ask for one. sbMatched and squareBend, the two main compressible fixtures, both
    // carry the pair. Built here so the closure can be given them, and so a caller that cannot supply
    // them can refuse rather than run a fabricated inlet.
    DeviceBuffer<label>  turbInletKMask,   turbInletEpsMask;
    DeviceBuffer<scalar> turbInletKInt,    turbInletEpsLen;
    bool                 hasTurbulentInlet = false;

    // compressible::alphatWallFunction, per boundary face, with that patch's OWN Prt (default 0.85 --
    // NOT the model's 1.0). The closure writes alphat_b = rho_b*nut_b/Prt on these faces, which is
    // EddyDiffusivity.C:38's alphat_.correctBoundaryConditions(). Projected from the per-patch
    // hf.alphatWallFn / hf.alphatPrt the host reader already gathers.
    DeviceBuffer<label>  alphatWallMask;
    DeviceBuffer<scalar> alphatPrtFace;

    // updateCoeffs() metadata for the boundary conditions whose coefficients move with the SOLUTION.
    // The device boundary objects are a pre-baked snapshot -- bcType, refValue and valueFraction are
    // fixed at build time -- so anything OpenFOAM recomputes inside updateCoeffs has to be recomputed
    // here, per iteration, by the driver. rhoUEqn.cuh:64-83 states that contract; nothing satisfied it.
    //
    // hasMixed covers the freestream family (freestreamVelocity/freestreamPressure), whose valueFraction
    // OpenFOAM rebuilds from the current flow ANGLE. hasFlowRate covers flowRateInletVelocity, which
    // needs the per-patch magSf mask and the face normals to turn a prescribed mass flow into a velocity
    // against the LIVE boundary density -- feeding it the wrong rho is the angledDuct defect.
    bool                              hasMixed    = false;
    bool                              hasFlowRate = false;
    std::vector<DeviceBuffer<scalar>> frMagSf;      // one per flowRate patch: magSf there, 0 elsewhere
    std::vector<scalar>               frMdot;       // matching prescribed mass flow, same order
    DeviceBuffer<scalar>              frNx, frNy, frNz;   // boundary-face normals, all faces

    bool turbulent = false;
    int  nBndFaces = 0;
};

// Flatten a per-patch host array into the boundary-face order DeviceMesh uses. Exposed because every
// caller that supplies a per-iteration boundary field (nu, mu, rho, alphaEff) needs exactly this, and
// three private copies of it is how the padding convention drifts.
std::vector<scalar> flattenBoundary(
    const std::vector<std::vector<scalar>>& perPatch,
    const std::vector<FvPatch>&             patches,
    int                                     nBndFaces,
    scalar                                  pad);

// ...and the same for a GeometricField's own patch values.
std::vector<scalar> flattenFieldBoundary(
    const GeometricField<scalar>& f,
    const std::vector<FvPatch>&   patches,
    int                           nBndFaces,
    scalar                        pad);

// The wall-face-ordered gather of a per-boundary-face array, which the closure needs for nu at the wall.
// Kept here because the ORDER has to match the one buildDeviceWallData used, and that order is decided
// here.
std::vector<scalar> gatherWallFaces(
    const std::vector<scalar>& bndFaceValues,
    const std::vector<label>&  wfFaceOfBnd);

// Build the device projection of an already-constructed host field set.
//
// REFUSES a mesh with coupled patches by name: buildDeviceMesh keeps cyclic/AMI/processor faces out of
// the LDU, so every equation would silently lose their contribution.
RhoDeviceFields createDeviceFields(
    const cpu::rhoSimple::RhoSimpleFields& hf,
    const PrimitiveMesh&                   m,
    const FvGeometry&                      g,
    const std::vector<FvPatch>&            patches);

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
