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
#include "device_fvoptions.cuh"   // DeviceMangroves
#include "device_dilu.cuh"
#include "device_crank_nicolson_ddt.cuh"
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
    // nutkRoughWallFunction's per-face Ks, Cs and sqrt(sqrt(Cmu)) (zero off a rough patch), uploaded only when
    // a face is rough, and the gate's no-history control
    bool hasRoughWall = false;
    DeviceBuffer<scalar> nutWfKsBnd;
    DeviceBuffer<scalar> nutWfCsBnd;
    DeviceBuffer<scalar> nutWfRoughCmu25Bnd;
    bool nutkRoughNoHistory = false;
    // wall-face order -> boundary-face index, the order buildDeviceWallData decided
    DeviceBuffer<label> wallFaceOfBnd;
    int nWallFaces = 0;

    // rho = 1 for the uniform lineage, cells and boundary faces
    // the LES filter width, one per cell, computed once on the host by LESdelta::compute and uploaded
    // here: kEqn's nut and its destruction term both read it. Empty on a RAS case.
    DeviceBuffer<scalar> lesDelta;

    DeviceBuffer<scalar> onesCell;
    DeviceBuffer<scalar> onesBnd;
    // the mesh's DILU level schedule, built only when kFinal/epsilonFinal name `PBiCG` with `DILU`
    // (waves/mangroveInteraction): the closure then runs OpenFOAM's PBiCG (device_pbicg.cuh) and not
    // the smoothSolver sweep every other turbulent tutorial names
    DeviceDilu dilu;
    // kCoeff and epsilonCoeff of multiphaseMangrovesTurbulenceModel at this step's U, per cell
    // one per active multiphaseMangrovesTurbulenceModel, in the option list's order
    std::vector<DeviceBuffer<scalar>> mangroveK;
    std::vector<DeviceBuffer<scalar>> mangroveEps;
    // CrankNicolson's state (the host closure's InterTurbulenceCrankNicolson): the two ddt0 fields and
    // the old-old level of each field, rotated once per time index as storeOldTimes does
    DeviceCnDdt0 cnDdt0K;
    DeviceCnDdt0 cnDdt0Eps;
    // psi.oldTime() FOR THE CLOSURE, per TIME INDEX -- the device twin of InterTurbulence::kOldStep. These
    // were cnKEntry/cnEpsEntry, filled only under CrankNicolson; every ddt scheme reads the old-TIME level
    // and only the old-OLD level (cnKOO/cnEpsOO) is CN's, so they are advanced unconditionally now.
    DeviceBuffer<scalar> kOldStep;
    DeviceBuffer<scalar> epsOldStep;
    DeviceBuffer<scalar> cnKOO;
    DeviceBuffer<scalar> cnEpsOO;
    label oldStepTimeIndex = -1;

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

// THE MESH HAS MOVED: every distance the closure holds was measured on the mesh as it stood before,
// and each one is a MeshObject OpenFOAM recomputes on a move. Call it after fvMesh::movePoints and
// after the host block's own moveInterTurbulence, which is what refills t.yCell:
//   wallDist::New(mesh).y()   -- per cell, F1 and F2 (wallDist is an UpdateableMeshObject,
//                                wallDist.C:193-221)
//   nearWallDist              -- per boundary face, every turbulence wall function's y
//                                (nearWallDist::correct, called from fvMesh::movePoints)
//   DeviceWallData            -- the wall faces' y, deltaCoeffs and wall velocity
// The topology is fixed by a move, so every array keeps its length and its face order; this asserts
// that rather than trusting it.
void refreshDeviceInterTurbulenceGeometry(
    DeviceInterTurbulence& d,
    const cpu::interFoam::InterTurbulence& t,
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

struct DeviceInterTurbulenceStepInput
{
    // LOCALEULER: the per-cell rDeltaT, which kOmegaSST's two fvm::ddts take (KOmegaSSTInput::
    // rDeltaTCells); the other closures refuse it. Null == 1/deltaT.
    const DeviceBuffer<scalar>* rDeltaT = nullptr;
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
    // A MOVING MESH, which the closures' ddt and divU both need: the volumes the cells had BEFORE this
    // step's move (the Euler source takes V0 while the diagonal keeps V) and the mesh flux (divU is
    // the divergence of the ABSOLUTE flux, phi + meshPhi). Null on a mesh that does not move. The HOST
    // closure has carried the pair since it was ported (InterTurbulenceStepInput::V0/meshPhi, gated on
    // waves/waveMakerPiston `pistonSST`); the device arm refused a moving RAS closure until it did too.
    const DeviceBuffer<scalar>* V0 = nullptr;
    const DeviceBuffer<scalar>* meshPhiInt = nullptr;
    const DeviceBuffer<scalar>* meshPhiBnd = nullptr;
    // every solve of the run, in order, for the solver-log gate
    std::vector<cpu::interFoam::LinearSolveRecord>* epsilonLog = nullptr;
    std::vector<cpu::interFoam::LinearSolveRecord>* kLog = nullptr;
    // CrankNicolson: the scheme's clock and rho.oldTime().oldTime() (read in the `density variable`
    // lineage; null in the other, where the ones vector stands in). Null runs the closure's ddt as Euler.
    const cpu::fv::CrankNicolsonClock* cn = nullptr;
    const DeviceBuffer<scalar>* rhoOO = nullptr;
    // fvOptions(k) and fvOptions(epsilon): multiphaseMangrovesTurbulenceModel's -Sp(Cx*Cd*a*N*|U|, .),
    // with U the closure's own -- OpenFOAM looks `U` up when the equation is built. Under kEpsilon
    // only; the case reader refuses the option under any other closure. Null = no such option.
    const DeviceMangroves* mangroves = nullptr;
    // THE STEP'S INDEX and WHETHER THIS IS THE FINAL OUTER CORRECTOR, unset sentinels that the entry
    // refuses -- the closure keys its old-time snapshot on the first and picks <field>Final by the second.
    label timeIndex = -1;
    int   finalIter = -1;
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
