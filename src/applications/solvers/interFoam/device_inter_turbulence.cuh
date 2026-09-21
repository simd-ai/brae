#pragma once
// interFoam's turbulence ON THE DEVICE: gpu::kEpsilonRAS::correct, handed what inter_turbulence_cpp.cu
// hands the host reference.
//
// provenance:
//   openfoam:  src/phaseSystemModels/twoPhaseInter/incompressibleInterPhaseTransportModel/
//                  incompressibleInterPhaseTransportModel.C:46-110, :134-144
//              src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C:214-296
//   host:      src/applications/solvers/interFoam/inter_turbulence_cpp.cuh -- the ORACLE. It says what
//              the two lineages are and why validate() runs in one of them only.
//   closure:   src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cuh, already gated against
//              its own host reference under rhoSimpleFoam (tests/test_rho_kepsilon_cuda.cu)
//   tests:     tests/interfoam_ras_dambreak_vs_openfoam.sh -- against real OpenFOAM, and against the
//              SAME device loop with the host closure in this one's place
//
// THIS FILE OWNS NO ARITHMETIC. The closure is the one rhoSimpleFoam runs; what differs is what each
// of its inputs IS in interFoam, and every one of these was measured on the host first:
//
//   the equation's flux   rhoPhi under `density variable`, phi otherwise
//   divU's flux           phi, ALWAYS -- the closure's `phiByRho` slot, which here is not a quotient
//   the patches' flux     phi, ALWAYS (KEpsilonInput::bcPhiBnd). rhoPhi is the alpha step's flux, built
//                         from the phi the time step STARTED on; at step one that is zero on every
//                         atmosphere face while phi is not
//   rho                   the mixture's under `density variable`; ONES otherwise. The closure has no
//                         rho-free path, and multiplying by 1.0 changes no bit
//   rho's patch values    the device step's own blend, which carries alpha2's one-pass-older patch values
//   nu                    the mixture's, cells and patches, in both lineages
//   validate()            the host's, at buildInterFields: nut arrives here already validated or not
#include "cf_types.cuh"
#include "device_boundary.cuh"
#include "device_buffer.cuh"
#include "device_kepsilon.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "inter_solve_record.cuh"
#include "inter_turbulence_cpp.cuh"
#include "kEpsilon.cuh"
#include "kOmegaSST.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

struct DeviceInterTurbulence
{
    // the state, which outlives every step
    DeviceBuffer<scalar> k;
    DeviceBuffer<scalar> epsilon;
    DeviceBuffer<scalar> nut;
    DeviceBuffer<scalar> nutBnd;

    DeviceBoundary dbK;
    DeviceBoundary dbEps;
    DeviceWallData wall;
    // per BOUNDARY face: does it carry a turbulence wall function, its near-wall distance, and the nut
    // wall function's own kind and coefficients
    DeviceBuffer<label> wfBndMask;
    DeviceBuffer<scalar> wallYBndFace;
    DeviceBuffer<label> nutWfKindBnd;
    DeviceBuffer<scalar> nutWfCmu25Bnd;
    DeviceBuffer<scalar> nutWfKappaBnd;
    DeviceBuffer<scalar> nutWfEBnd;
    DeviceBuffer<scalar> nutWfYplLamBnd;
    // wall-face order -> boundary-face index, the order buildDeviceWallData decided
    DeviceBuffer<label> wallFaceOfBnd;
    int nWallFaces = 0;

    // rho = 1 for the uniform lineage, cells and boundary faces
    // the LES filter width, one per cell, computed once on the host by LESdelta::compute and uploaded
    // here: kEqn's nut and its destruction term both read it. Empty on a RAS case.
    DeviceBuffer<scalar> lesDelta;

    DeviceBuffer<scalar> onesCell;
    DeviceBuffer<scalar> onesBnd;

    // kOmegaSST's own, when the case names it: the CELL wall distance F1 and F2 read (wallDist::New's y,
    // not the wall functions' near-wall face distance), the faces where F1 is 1 by construction, and
    // which nut faces the closure's field assignment fills. The second field lives in InterTurbulence::
    // omega there, not ::epsilon, and `epsilon` below carries it.
    DeviceBuffer<scalar> yCell;
    DeviceBuffer<label>  f1OneMask;
    DeviceBuffer<label>  nutCalcMask;

    // nut's OWN boundary, for the flux-conditional patches the closure's field assignment does not decide:
    // correctBoundaryConditions evaluates an inletOutlet nut against the flux (valueFraction = neg(phi)),
    // inflow faces taking the inletValue and outflow faces the cell nut just written. `nutIoFaces` is how
    // many such faces the case has; zero leaves nut's boundary exactly as the closure left it.
    DeviceBoundary dbNut;
    int nutIoFaces = 0;
    // ...and which faces correctBoundaryConditions evaluates at all: every patch that is not a wall, not
    // `empty` and not `calculated` (kOmegaSST_cpp.cu:946-984)
    DeviceBuffer<label> nutEvalMask;
    int nutEvalFaces = 0;

    // The TURBULENT INLETS, which OpenFOAM recomputes at every updateCoeffs from U's current patch
    // values: turbulentIntensityKineticEnergyInlet (k = 1.5*(I*|U|)^2) and the mixing-length pair
    // (epsilon = Cmu^0.75*k^1.5/L, omega = sqrt(k)/(Cmu^0.25*L)). The mask CARRIES the kind, as
    // rhoCreateFields.cu:478-495 builds it: 1 = epsilon, 2 = omega.
    DeviceBuffer<label>  turbInletKMask, turbInletEpsMask;
    DeviceBuffer<scalar> turbInletKInt,  turbInletEpsLen;
    bool hasTurbulentInlet = false;

    // per-step scratch
    DeviceBuffer<scalar> nuWall;
    DeviceBuffer<scalar> nutBndIn;
    DeviceBuffer<scalar> nutWallIn;
    gpu::kEpsilonRAS::KEpsilonStages stages;
};

// Uploads the host turbulence block -- its fields AS THEY STAND, so validate() has or has not run --
// and builds the wall data. Refuses what the device closure refuses: a wall function on a patch that
// is not a `wall`.
DeviceInterTurbulence buildDeviceInterTurbulence(
    const cpu::interFoam::InterTurbulence& t,
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

struct DeviceInterTurbulenceStepInput
{
    const DeviceBuffer<scalar>* Ux = nullptr;
    const DeviceBuffer<scalar>* Uy = nullptr;
    const DeviceBuffer<scalar>* Uz = nullptr;
    const DeviceBuffer<scalar>* phiInt = nullptr;
    const DeviceBuffer<scalar>* phiBnd = nullptr;
    const DeviceBuffer<scalar>* rhoPhiInt = nullptr;
    const DeviceBuffer<scalar>* rhoPhiBnd = nullptr;
    const DeviceBuffer<scalar>* rho = nullptr;
    const DeviceBuffer<scalar>* rhoBnd = nullptr;
    // A PERIODIC PAIR, and the two fluxes on its own faces: the VOLUMETRIC one (phiByRho, which
    // divU takes) and the EQUATION's (which the transport matrix and divPhi take). They are one
    // field in the uniform lineage and two in the variable-density one, exactly as phiInt and
    // phiByRhoInt are. Null = a mesh with no pair.
    DeviceCyclic*               cyc          = nullptr;
    const DeviceBuffer<scalar>* cycPhi       = nullptr;   // volumetric, on the pair
    const DeviceBuffer<scalar>* cycRhoPhi    = nullptr;   // mass, on the pair
    // rho.oldTime(): the density the time step STARTED on
    const DeviceBuffer<scalar>* rhoOld = nullptr;
    const DeviceBuffer<scalar>* nu = nullptr;
    const DeviceBuffer<scalar>* nuBnd = nullptr;
    scalar deltaT = 0;
    // every solve of the run, in order, for the solver-log gate
    std::vector<cpu::interFoam::LinearSolveRecord>* epsilonLog = nullptr;
    std::vector<cpu::interFoam::LinearSolveRecord>* kLog = nullptr;
};

// turbulence->correct(), interFoam.C:171.
void deviceCorrectInterTurbulence(
    DeviceInterTurbulence& d,
    const cpu::interFoam::InterTurbulence& t,
    const DeviceInterTurbulenceStepInput& in,
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU);

// The device's answer back into the host block: k, epsilon and nut on cells, their patches
// re-evaluated, and nut's patch values as the closure left them.
void downloadDeviceInterTurbulence(
    const DeviceInterTurbulence& d,
    cpu::interFoam::InterTurbulence& t,
    const std::vector<FvPatch>& patches);

} // namespace brae
