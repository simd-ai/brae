#pragma once
// ONE WHOLE interFoam TIME STEP on the device: the alpha sub-cycle, mixture.correct(), the momentum
// equation, and the pressure corrector.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/interFoam/interFoam.C:88-140
//   host:      src/applications/solvers/interFoam/inter_solve_cpp.cu, runTimeStep -- the ORACLE, and
//              the path that reaches alpha 3.4346e-09, p_rgh 2.29e-06 and U 3.31e-06 relative on
//              damBreak against real OpenFOAM.
//   tests:     tests/test_device_inter_step.cu
//
// Everything under this is separately landed and separately gated. What this file owns is THE LOOP
// ORDER, and interFoam's order is not numerics -- it is which field each equation sees:
//
//   ALPHA ADVANCES FIRST, before the momentum predictor, and mixture.correct() runs between them. So
//   UEqn is built on the NEW density. The single-phase order -- momentum first -- converges perfectly
//   well, on last step's density, which at a water/air interface is wrong by a factor of 1000.
//
//   rhoPhi COMES OUT OF THE ALPHA EQUATION and is what fvm::div in UEqn is built on. It is the
//   MULES-limited MASS flux, not the volumetric phi, and not rho*phi rebuilt afterwards.
//
//   THE MOMENTUM MATRIX IS ASSEMBLED AND RELAXED EVEN WHEN momentumPredictor IS OFF, because pEqn
//   needs A() and H(). damBreak sets `momentumPredictor no`, so on the case this port is gated against
//   the predictor never solves and the matrix is still the whole basis of the pressure equation.
//
//   THE PRESSURE CORRECTOR IS THE LAST THING, and the fields it writes -- phi, U, p_rgh, p -- are what
//   the NEXT step's alpha equation advects with.
//
// WHAT STAYS ON THE HOST is what has stayed on the host throughout: the boundary conditions. alpha's
// patch values and its contact angle, fvm::div's per-patch coefficients for the alpha pre-solve, U's
// boundary for the momentum matrix, and p_rgh's for the pressure laplacian. All four arrive through
// `hooks`. Everything that scales with the CELL COUNT runs on the device.
#include "device_crank_nicolson_ddt.cuh"
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_inter_alpha_step.cuh"
#include "device_inter_pressure_step.cuh"
#include "device_inter_ueqn.cuh"
#include "UEqn.cuh"
#include "pEqn.cuh"
#include "grad_u_memo.cuh"   // GradUMemo
#include <functional>

namespace brae {

// The step calls the U-boundary hook at three points, and they are NOT one operation in OpenFOAM.
enum class DeviceUBoundaryCall
{
    // The fvMatrix constructor's U.boundaryFieldRef().updateCoeffs() at the momentum assembly
    // (fvMatrix.C:396): the patches' COEFFICIENTS move to the flux and the phase fraction as they
    // stand, and the STORED patch values do not -- updateCoeffs is not an evaluate, except in a class
    // whose updateCoeffs ends in one (pressureInletOutletVelocity). fvc::grad(U) inside divDevRhoReff
    // reads the stored values.
    assembly,
    // U.correctBoundaryConditions(): updateCoeffs where the patch is not updated(), then the evaluate.
    evaluate,
    // ...the same, at the ONE call where OpenFOAM's patches are still updated(): after the first
    // pressure corrector of a pass that ran no momentum predictor. The constructor's updateCoeffs set
    // the flag and nothing has evaluated U since, so the evaluate skips updateCoeffs and blends with
    // the ASSEMBLY-TIME valueFraction (mixedFvPatchField.C). The hook must keep the new flux from
    // U's flux-conditional patches, as the host pEqn does (inter_peqn_cpp.cu, uPatchesUpdatedAtEntry,
    // with what it cost).
    evaluateStillUpdated,
    // ...and the predictor solve's own U.correctBoundaryConditions() (fvMatrix::solve): the patches are
    // STILL updated() from the assembly, so no updateCoeffs runs. phi has not moved since, so a
    // flux-conditional patch reads the same either way and the hook treats this as `evaluate`; a patch
    // that updates from U's CELLS -- outletPhaseMeanVelocity -- would not, and must skip it, as the host
    // loop's momentumPredictor does (it updates only at the assembly and in the pressure correctors).
    evaluateAfterPredictor
};

struct DeviceInterStepHooks
{
    DeviceInterAlphaHooks    alpha;      // alpha's patch values + contact angle, and fvm::div's coeffs
    DeviceInterPressureHooks pressure;   // p_rgh's laplacian boundary coefficients, after constrainPressure

    // After EACH pressure corrector, with the flux it left: continuityErrs.H, which pEqn.H includes
    // there. Only a writer reads what it accumulates (uniform/cumulativeContErr). Null does nothing.
    std::function<void(const DeviceBuffer<scalar>& phiInt,
                       const DeviceBuffer<scalar>& phiBnd)> correctorDone;

    // U's boundary, rebuilt from whatever the device last wrote. It is refreshed once per step rather
    // than per corrector because interFoam's PIMPLE loop evaluates it there.
    // `UbStored` are U's EVALUATED patch values, one buffer per component, as the host's evaluate
    // left them. They are NOT the same as re-deriving them on the device with deviceBCValue: at a
    // flux-conditional patch like damBreak's pressureInletOutletVelocity atmosphere the two differ,
    // and fvc::grad(U) inside divDevRhoReff reads the STORED ones. MEASURED with the re-derived value
    // standing in: the dev2 term's contribution was 100% wrong on that patch's 46 cells (2.24e-06 of
    // 2.24e-06) while the three walls were between exact and 0.8%.
    // `call` says WHICH of OpenFOAM's operations the call stands for -- see DeviceUBoundaryCall.
    std::function<void(const DeviceBuffer<scalar>& Ux,
                       const DeviceBuffer<scalar>& Uy,
                       const DeviceBuffer<scalar>& Uz,
                       DeviceVectorBoundary&       dbU,
                       DeviceBuffer<scalar>*       UbStored,
                       DeviceUBoundaryCall         call)> updateUBoundary;

    // The face fields that depend on the NEW alpha, over the mesh's FULL face array:
    // surfaceTensionForce() = interpolate(sigma*K)*snGrad(alpha1), and snGrad(rho). Both are rebuilt
    // AFTER the alpha step and BEFORE the momentum equation, because both equations read them and the
    // interface has just moved. nuEff is the mixture's, which mixture.correct() has just written.
    std::function<void(const DeviceBuffer<scalar>& alpha1,
                       const DeviceBuffer<scalar>& K,
                       const DeviceBuffer<scalar>& rho,
                       // rho's PATCH values, boundary-face order. snGrad(rho) on a patch is
                       // deltaCoeffs*(rho_b - rho_cell): rho's patches are `calculated`, and a
                       // zeroGradient copy zeroes a term that is 100% of the pressure source on an
                       // inflow patch -- see rhoWithPatchValues in inter_case_cpp.cuh.
                       const DeviceBuffer<scalar>& rhoBnd,
                       DeviceBuffer<scalar>&       stf,
                       DeviceBuffer<scalar>&       snGradRho,
                       DeviceBuffer<scalar>&       nuEffCell,
                       DeviceBuffer<scalar>&       nuEffBnd,
                       // snGrad(p_rgh) over the full face array, needed ONLY when the case runs a
                       // momentum predictor: it is explicit there and implicit in the pressure
                       // equation, and its boundary half is per-patch dispatch like everything else.
                       DeviceBuffer<scalar>&       snGradPrgh)> interfaceForces;
};

// Intermediates the step is willing to hand back, for a gate that has to localise a disagreement
// rather than only measure one. A whole step is a dozen operators and a final U that is 60% out cannot
// say which of them did it; these can. Null costs nothing. This is the of-instrument approach applied
// to brae's own code -- the same reason tools/dumpInterFoam exists for OpenFOAM's.
struct DeviceInterStepTaps
{
    DeviceBuffer<scalar> rAU;
    DeviceBuffer<scalar> HbyA[3];
    // H()'s two halves, x component: before the pair's off-diagonal, and the pair's own contribution
    DeviceBuffer<scalar> HNoPairX;
    DeviceBuffer<scalar> HPairX;
    DeviceBuffer<scalar> phiHbyAInt;      // AFTER the two interFoam terms
    DeviceBuffer<scalar> phiHbyABnd;      // ...and its BOUNDARY, which fvc::div sums too
    DeviceBuffer<scalar> UEqnDiag;        // relaxed, as A() takes it
    DeviceBuffer<scalar> UEqnSourceX;
    DeviceBuffer<scalar> UEqnUpper, UEqnLower;
    DeviceBuffer<scalar> UEqnIC, UEqnBC;   // component 0
    // the p_rgh system, first corrector: the raw LDU, the extensive source, and the boundary
    // coefficients the fold uses. Source and matrix are separate questions and a solved field cannot
    // tell them apart.
    DeviceBuffer<scalar> pDiag, pUpper, pLower, pSource, pIC, pBC;
    DeviceBuffer<scalar> rAUfAllTap;
    DeviceBuffer<scalar> pSolved;         // p_rgh after the FIRST corrector's solve
    DeviceBuffer<scalar> ddtRhoOld;       // rho.oldTime(), the field the ddt source is built on
    // phig, rAUf, phig-flux and the flux itself on the periodic pair. Unlike the matrix taps above
    // these are taken on EVERY corrector, so they hold the LAST -- which is the state the host's own
    // taps hold and the only one the two arms can be compared in.
    DeviceBuffer<scalar> phigIf, rAUfIf, ffIf, phiIf, phiHbyAIfPrePhig, cycJumpTap;
    // phiHbyA on the internal faces BEFORE phig -- the host's PressureTaps::phiHbyA is at that point
    DeviceBuffer<scalar> nonOrthSource;
    DeviceBuffer<scalar> phigIntTap;
    DeviceBuffer<scalar> phigBndTap;
    DeviceBuffer<scalar> divPhiHbyA;
    // which pressure corrector these came from -- this arm fills them at corr == 0, the FIRST
    int tapCorrector = -1;
    DeviceBuffer<scalar> phiHbyAIntPrePhig;
    DeviceBuffer<scalar> phiHbyABndPrePhig;
    std::vector<std::vector<scalar>> jumpHistory;
    DeviceBuffer<scalar> uEqnCycIfCoeff;   // UEqn's interface off-diagonal on the pair
    // ...and the ALPHA step's own two on the pair, taken as it leaves: alphaPhi10 and rhoPhi there
    DeviceBuffer<scalar> alphaPhiIfTap, rhoPhiIfTap, alphaAfterAlphaStep;
    // ...and the implicit pre-solve's own two, before the corrector loop
    DeviceBuffer<scalar> preSolveAlpha, preSolveAlphaPhiIf;
};

// CRANKNICOLSON on the device loop (device_crank_nicolson_ddt.cuh), owned by the driver because every
// piece of it outlives a step: the scheme's clock, the three ddt0 fields the step's equations keep --
// the momentum's "ddt0(rho,U)", ddtCorr's "ddtCorrDdt0(U)" and "ddtCorrDdt0(phi)" -- and the old-old
// levels they read. alpha1's old-old level stands for rho.oldTime().oldTime(): the step rebuilds that
// density from it exactly as it rebuilds rho.oldTime() from alpha1.oldTime(). phi's is the driver's to
// create when OpenFOAM creates it (inter_driver_cpp.cu, phiOOExists). alphaEqn.H's own blend and the
// un-blend of the end-of-step alpha flux travel here too, with alphaPhi10's old level and where to
// leave the new one. The turbulence closure keeps its own (DeviceInterTurbulence).
struct DeviceInterCrankNicolson
{
    const cpu::fv::CrankNicolsonClock* clock = nullptr;
    DeviceCnDdt0 ddt0RhoU;
    DeviceCnDdt0 ddtCorrU;
    DeviceCnDdt0 ddtCorrPhi;
    const DeviceBuffer<scalar>* UOO[3] = {nullptr, nullptr, nullptr};
    const DeviceBuffer<scalar>* UOOBnd[3] = {nullptr, nullptr, nullptr};
    const DeviceBuffer<scalar>* alpha1OO = nullptr;
    const DeviceBuffer<scalar>* phiOOInt = nullptr;
    const DeviceBuffer<scalar>* phiOOBnd = nullptr;
    // ...and the PAIR's own old-old flux and its dphidt0 level, which ddtCorr needs there
    const DeviceBuffer<scalar>* phiOOIf = nullptr;
    DeviceCnDdt0 ddtCorrPhiIf;
    // ON A MOVING MESH ddtCorr is fvcDdtUfCorr, a different operator: its flux side is Uf.oldTime()
    // and its second ddt0 is a SURFACE VECTOR field over the same faces (CrankNicolsonDdtScheme.C:
    // 1201-1257). Uf's two old levels are the driver's, as phi's are, three components each over the
    // internal faces and the boundary faces.
    DeviceCnDdt0 ddtCorrUf;
    const DeviceBuffer<scalar>* UfOld[3] = {nullptr, nullptr, nullptr};
    const DeviceBuffer<scalar>* UfOldBnd[3] = {nullptr, nullptr, nullptr};
    const DeviceBuffer<scalar>* UfOO[3] = {nullptr, nullptr, nullptr};
    const DeviceBuffer<scalar>* UfOOBnd[3] = {nullptr, nullptr, nullptr};
    // alphaEqn.H:18-56, :91-97, :236-262 -- the off-centring coefficient the scheme constructed for
    // ddt(alpha) gives on THIS step (0 before the scheme is warm), its cnCoeff, and alphaPhi10's levels
    scalar ocAlpha = 0;
    scalar cnAlpha = 1;
    // ...and whether phi HAS an old-time level yet. GeometricField::oldTime() creates it on the first
    // request, as a copy of the field as it stands then; on a MOVING mesh ddtCorr reads Uf.oldTime()
    // instead and never asks, so the alpha step's own blend is the first request and is inert for
    // that one step. False here means "the level does not exist yet, blend with phi itself" -- see
    // the note in inter_driver_cpp.cu at the host's offCentredFlux call.
    bool phiOldExists = false;
    const DeviceBuffer<scalar>* alphaPhiOldInt = nullptr;   // null with ocAlpha > 0: created by this step
    const DeviceBuffer<scalar>* alphaPhiOldBnd = nullptr;
    DeviceBuffer<scalar>* alphaPhiOutInt = nullptr;
    DeviceBuffer<scalar>* alphaPhiOutBnd = nullptr;
    DeviceBuffer<scalar>* alphaPhiCreatedInt = nullptr;
    DeviceBuffer<scalar>* alphaPhiCreatedBnd = nullptr;
    // ...and the same three on the PAIR's own faces, which are in neither of the arrays above
    const DeviceBuffer<scalar>* alphaPhiOldIf = nullptr;
    DeviceBuffer<scalar>*       alphaPhiOutIf = nullptr;
    DeviceBuffer<scalar>*       alphaPhiCreatedIf = nullptr;
};

// fvSolution's `cache { grad(U); }` on the device loop -- the host's GradUCache (inter_turbulence_cpp.cuh), whose
// comment has what OpenFOAM does and why it is not the uncached answer. The registry's grad(U) in the three forms
// the assembly's sites read (MomentumInput::gradUGivenMemo): per component for linearUpwind's correction, the V
// schemes' limiter and the corrected laplacian; packed for the dev2 term, with the boundary gaussGrad corrected
// WHEN THE FIELD WAS FORMED. Filled at the host loop's instants -- validate() (uploaded from the host, which runs
// it) and each kOmegaSST correct (deviceStoreGradU) -- and consumed at the assembly while `valid`.
struct DeviceGradUCache
{
    bool on = false;
    bool valid = false;
    GradUMemo memo;
    DeviceBuffer<scalar> tensor;   // 9*nC, deviceDivDevReff's packing [(d*3+i)*nC + c]
    DeviceBuffer<scalar> bnd;      // 9*nBndFaces, gradBKernel's packing [q*nB + b]
    // assemblies that took it: the gate's witness that the arm consumed the cache at all
    long consumed = 0;
};

// Form the cache from U as it stands and U's STORED patch values -- the host's storeGradU after kOmegaSST's
// correct: the unlimited Gauss gradient (deviceGaussGradFused, bit-identical to the assembly's own) and its
// boundary through deviceBoundaryGradU against dbU's snGrad now. Sets `valid`.
void deviceStoreGradU(
    DeviceGradUCache& c,
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>* const* UbStored);

// ...or take one formed on the HOST (validate() runs there on this loop): `tensor9` 9*nC and `bnd9` 9*nBndFaces,
// both in the packings above. Sets `valid`.
void deviceSetGradU(
    DeviceGradUCache& c,
    const std::vector<scalar>& tensor9,
    const std::vector<scalar>& bnd9,
    int nC);

struct DeviceInterStepControls
{
    // CrankNicolson, or null for Euler -- see DeviceInterCrankNicolson
    DeviceInterCrankNicolson* cn = nullptr;
    // OUT, on a write step only: the alpha flux the step's last alpha solve leaves (alphaEqn.H's
    // alphaPhi10, after the CrankNicolson un-blend), which OpenFOAM writes as alphaPhi0.<phase1>.
    // Internal faces, the uncoupled boundary faces, and a periodic pair's faces. Null copies nothing.
    DeviceBuffer<scalar>* alphaPhiWriteInt = nullptr;
    DeviceBuffer<scalar>* alphaPhiWriteBnd = nullptr;
    DeviceBuffer<scalar>* alphaPhiWriteIf  = nullptr;
    // LOCALEULER: the per-cell rDeltaT fvm::ddt(rho, U) reads (MomentumAssemblyInput::ddtRDeltaT). Null ==
    // the scalar step. Separate from alphaInput's so a gate can switch one consumer off at a time.
    const DeviceBuffer<scalar>* rDeltaTUEqn = nullptr;
    // ...and ddtCorr's face field, interpolate(rDeltaT) on internal and boundary faces. Both or neither.
    const DeviceBuffer<scalar>* rDeltaTfInt = nullptr;
    const DeviceBuffer<scalar>* rDeltaTfBnd = nullptr;
    // fvSolution's cached grad(U) (null when the case caches nothing), whether the mesh changes this step
    // (gradScheme bypasses and deletes the registry field then, gradScheme.C:132-142), and the gate's control
    // BRAE_CONTROL_GRADU_UNCACHED (every site forms its own, OpenFOAM's UNCACHED answer). The step consumes
    // the cache at its assembly and invalidates it once U has changed (after the correctors).
    DeviceGradUCache* gradUCache = nullptr;
    bool gradUMeshChanging = false;
    bool gradUUncachedControl = false;
    // ...and BRAE_CONTROL_GRADU_BND_LIVE: the cached cells, the dev2 boundary rebuilt at the assembly. WRONG.
    bool gradUBndLiveControl = false;
    // the cell volumes the mesh had BEFORE this step's move, for the ddt's old-time term
    // (OF EulerDdtScheme: rho.oldTime()*U.oldTime()*Vsc0()). Null on a static mesh.
    const DeviceBuffer<scalar>* V0 = nullptr;
    // ...and the volumes TWO steps back (fvMesh::V00), which only CrankNicolson's moving branch reads
    // (CrankNicolsonDdtScheme.C:1029-1047). Required with V0 under that scheme, null under Euler.
    const DeviceBuffer<scalar>* V00 = nullptr;
    // ...and the mesh flux over the FULL face array, for fvc::makeRelative(phi, U) at the end of the
    // pressure corrector. Null on a static mesh, where OpenFOAM's makeRelative is a no-op.
    const DeviceBuffer<scalar>* meshPhiAll = nullptr;
    // (Sf & Uf.oldTime()) per internal face, which ddtCorr takes in phi.oldTime()'s place on a moving
    // mesh (fvcDdt.C:220-227 -> EulerDdtScheme's fvcDdtUfCorr). Null on a static mesh.
    const DeviceBuffer<scalar>* phiUfOldInt = nullptr;
    // ...and where the pressure step leaves the ABSOLUTE flux for the driver's fvc::correctUf.
    DeviceBuffer<scalar>* phiAbsIntOut = nullptr;
    DeviceBuffer<scalar>* phiAbsBndOut = nullptr;
    // ...and rAU, which is not this loop's to keep but the NEXT mesh update's: under `correctPhi`
    // CorrectPhi is solved with fvc::interpolate(rAU) of the LAST corrector (correctPhi.H, and
    // interFoam.C:138), and that call is a host one on either arm. Null when the caller does not
    // need it, which is every case with a mesh that does not move.
    DeviceBuffer<scalar>* rAUOut = nullptr;

    // The case's MRF zones, built once by the driver (buildDeviceMRFZone). interFoam touches them in
    // three places -- UEqn.H:6 MRF.DDt(rho, U), pEqn.H:18 MRF.zeroFilter(ddtCorr term) and pEqn.H:19
    // MRF.makeRelative(phiHbyA) -- and the fourth, MRF.correctBoundaryVelocity(U) at UEqn.H:1, is the
    // driver's because it writes the HOST field the boundary snapshot is taken from.
    const std::vector<DeviceMRFZone>* mrf = nullptr;
    // fvOptions' explicitPorositySource/DarcyForchheimer, built by the driver from the host's own
    // OptionList (its transformed D and F tensors). Every other option type stays host-only and the
    // driver refuses it by name.
    const DevicePorosity* porosity = nullptr;
    // ...and multiphaseMangrovesSource, the drag and added mass of waves/mangroveInteraction, which the
    // shared assembler adds in the same slot from this step's rho, U.oldTime() and deltaT.
    const DeviceMangroves* mangroves = nullptr;
    // THE MESH'S PERIODIC PAIR. Every piece it needs is gated against the host on its own
    // (tests/test_device_cyclic_laplacian_vs_host.cu, test_device_mules_cyclic_vs_host.cu,
    // test_device_inter_peqn_cyclic_vs_host.cu). `cyc->phi` must hold the pair's CURRENT flux -- the
    // momentum coupling reads it for its convective half -- which is why the alpha step has to fill it
    // before the momentum matrix is assembled.
    DeviceCyclic* cyc = nullptr;
    // ...and the two per-face arrays the pair needs that nothing else owns: the alpha flux the
    // correctors carry between them, and the mass flux the momentum equation reads. The DRIVER owns
    // both, because they outlive a step the way phi does.
    DeviceBuffer<scalar>* alphaPhiIf = nullptr;
    DeviceBuffer<scalar>* rhoPhiIf   = nullptr;
    // the pair's flux at the TOP of the step -- phi.oldTime() there, which fvc::ddtCorr compares with
    // the flux of U.oldTime(). The driver snapshots it beside dPhiOI/dPhiOB, before anything writes
    // cyc->phi. Null on a mesh with no pair.
    const DeviceBuffer<scalar>* phiOldIf = nullptr;
    // nHatf on the pair, kept BY THE DRIVER across steps -- see DeviceInterAlphaControls::nHatfIf
    DeviceBuffer<scalar>* nHatfIf = nullptr;
    // ...and the three fields phig is built from ON THE PAIR. The device's face arrays exclude coupled
    // patches (device_mesh.cuh:41-44), so these carry what the boundary half of stf, ghf and
    // snGrad(rho) would otherwise hold there. The interfaceForces hook fills the first and the third
    // -- they change with alpha every step -- and ghf is the mesh's, built once.
    const DeviceBuffer<scalar>* stfIf       = nullptr;
    const DeviceBuffer<scalar>* ghfIf       = nullptr;
    const DeviceBuffer<scalar>* snGradRhoIf = nullptr;
    DeviceInterAlphaControls  alpha;
    DeviceMulesControls       mules;
    // The alpha equation's per-step settings -- cAlpha, deltaN and the two flux schemes. The flux
    // pointers and deltaT in it are IGNORED and filled by the step, because the sub-cycle owns the
    // time step and the caller owns the flux.
    DeviceAlphaStepInput      alphaInput;

    DeviceAlphaSolverControls momentum;      // the case's fvSolution entry for U
    // ...and for p_rgh, on every corrector but the last
    DeviceAlphaSolverControls pressure;
    // ...and p_rghFinal on the last -- pEqn.H:50's select(finalInnerIter()); see InterFields::pSolve
    DeviceAlphaSolverControls pressureFinal;
    // whether each entry names `solver PCG; preconditioner DIC;`, and the level schedule both share
    bool pressurePcgDIC = false;
    bool pressureFinalPcgDIC = false;
    DeviceDilu* dic = nullptr;
    // ...or `solver GAMG;`, each entry with its own controls, and the mesh's hierarchy both share
    const GamgControls* pressureGamg = nullptr;
    const GamgControls* pressureFinalGamg = nullptr;
    // ...and the same two for `solver PCG; preconditioner { preconditioner GAMG; ... }`
    const GamgPreconditionerControls* pressurePcgGamg = nullptr;
    const GamgPreconditionerControls* pressureFinalPcgGamg = nullptr;
    DeviceGamgCache* gamgCache = nullptr;
    GamgSolveLog* gamgLog = nullptr;
    // every p_rgh solve's own report, appended in order; null = not kept
    std::vector<DeviceSolverPerf>* pressureSolveLog = nullptr;
    // ...and the momentum predictor's: an array of THREE logs, [0] Ux, [1] Uy, [2] Uz; null = not kept
    std::vector<DeviceSolverPerf>* momentumSolveLog = nullptr;
    // pimple.correct() -- fvSolution's PIMPLE/nCorrectors. THE WHOLE OF pEqn.H REPEATS, not just the
    // solve: interFoam.C wraps `#include "pEqn.H"` in `while (pimple.correct())`, so rAU, HbyA,
    // phiHbyA, phig, the solve, U and phi are all rebuilt each pass, each from the U and phi the last
    // one left. damBreak asks for THREE and this port ran ONE, which is a real omission -- but it is
    // NOT the cause of the 60% gap in U that this file is under diagnosis for: MEASURED, going from
    // one corrector to three moved U from 1.6039e-01 to 1.5848e-01 of a field whose max is 2.65e-01,
    // about one per cent of the gap. Written here because the first version of this comment claimed
    // the opposite before the measurement came back.
    int    nCorrectors       = 1;
    // fvSolution's PIMPLE/nNonOrthogonalCorrectors, and laplacianSchemes' `corrected` (and `limited`
    // coefficient, 0 = unlimited) for the p_rgh laplacian -- see DeviceInterPressureInput
    int    nNonOrthogonalCorrectors = 0;
    bool   correctedLaplacian = false;
    // ...and grad(p_rgh)'s own gradSchemes entry, which that correction is built from: the two numbers a
    // GradChoice holds, forwarded to DeviceInterPressureInput below and taken there by deviceGradOf.
    bool   prghGradLeastSquares = false;
    scalar prghGradCellLimitK = 0;
    bool   nonOrthCoeffs = false;   // nonOrthDeltaCoeffs without the correction -- inter_ueqn_cpp.cuh:181
    scalar snGradLimitCoeff = 0;
    bool   momentumPredictor = true;         // damBreak sets this OFF
    // `frozenFlow yes` in the PIMPLE dict: OpenFOAM `continue`s past the momentum, the pressure AND
    // the turbulence corrector for the whole outer iteration (interFoam.C:156-158). The ALPHA
    // equation and mixture.correct() still run -- they are above that branch, at :152 and :154 --
    // so alpha keeps advancing on a velocity field nothing touches.
    bool   frozenFlow        = false;
    scalar relaxU            = 1;
    bool   relaxEquationU    = false;        // the case NAMES a factor -- see gpu::MomentumInput
    bool   needReference     = false;
    int    pRefCell          = 0;
    scalar pRefValue         = 0;
    // div(rhoPhi,U), as the case's fvSchemes NAMES it. THIS WAS HARDCODED TO upwind, and damBreak asks
    // for `Gauss linearUpwind grad(U)` -- which is the one scheme whose MATRIX is pure upwind while the
    // whole of it lives in a DEFERRED SOURCE CORRECTION. So the diagonal matched the host exactly, to
    // 1.5e-16 relative, while the source was 0.76% out: precisely the signature that took five other
    // candidates to eliminate. 24 of the 44 shipped tutorials name linearUpwind here.
    brae::cpu::DivScheme divScheme = brae::cpu::DivScheme::upwind;
    scalar divSchemeCoeff          = 1;
    // the `k` of `grad(U) cellLimited Gauss linear <k>`. gradULimitK is the gradient linearUpwind
    // NAMES; gradUSchemeLimitK is the gradSchemes `grad(U)` ENTRY, which divDevRhoReff's dev2 term
    // takes. On interFoam's tutorials they are the same word twice, but they are separate lookups.
    scalar gradULimitK             = 0;
    scalar gradUSchemeLimitK       = 0;
    bool   gradUSchemeLeastSq      = false;   // that entry's base scheme; see gpu::MomentumInput

    // polyMesh::solutionD() -- which coordinate directions the vector equation is SOLVED in. On a 2-D
    // case the empty direction is knocked out, and fvMatrix::H()'s validComponents block skips it. This
    // port hardcoded all three as valid, which on damBreak -- empty front and back -- told H() to solve
    // a direction OpenFOAM never does.
    int    solutionD[3]      = {1, 1, 1};

    // THE EDDY VISCOSITY, when the closure lives on the device: nuEff = nut + nu (eddyViscosity::nuEff),
    // cells and boundary faces. The interfaceForces hook hands back the MIXTURE's nu and these are added
    // to it on the device, so nut never crosses to the host. Null = the hook's value is nuEff already --
    // a laminar case, or the host closure, which adds its own nut inside the hook.
    const DeviceBuffer<scalar>* nutCell = nullptr;
    const DeviceBuffer<scalar>* nutBnd = nullptr;

    // constrainHbyA's per-face `assignable` mask, which the shared pressure predictor requires.
    // assignable() is NOT fixesValue(): slip and inletOutlet are non-assignable without fixing one,
    // and damBreak's atmosphere is pressureInletOutletVelocity.
    const DeviceBuffer<int>*  takeUAtBoundary = nullptr;
    // THE FLUX U's SWITCHES READ, per boundary face: 1 where U's patch names `phi rhoPhi`. OpenFOAM's
    // inletOutlet and pressureInletOutletVelocity LOOK UP the field their `phi` entry names at every
    // updateCoeffs (lookupPatchField<surfaceScalarField>(phiName_)), and rhoPhi is written by the alpha
    // step alone -- so every U evaluate of a step reads the alpha step's mass flux while phi moves with
    // each corrector. MEASURED on laminar/damBreak with the atmosphere naming rhoPhi: at step one's three
    // correctors the two disagree in SIGN on 46, 46 and 19 of the 46 atmosphere faces (rhoPhi is zero at
    // rest, phi is inflow). Null = no U patch names rhoPhi, and every switch reads phi as before.
    // NOT DISCRIMINATED by the damBreak gate: the one switch whose result reaches the fields there is the
    // assembly's, where rhoPhi and phi agree in sign (both the alpha step's flux), and the corrector-site
    // switches that disagree are re-switched at the next assembly before anything reads them -- withholding
    // this mask leaves `rhophi` and `rhophiU` bitwise unchanged. It is here because OpenFOAM reads the named
    // flux at every one of those updateCoeffs, and a momentum predictor or a U-reading closure between the
    // correctors would read the difference.
    const DeviceBuffer<int>*  uFluxIsRhoPhi = nullptr;
};

// Every field is in and out: this is a time step, and the next one starts from what this leaves.
// `alpha1Old`, `UOld` and `phiOld` are the previous time level and are not written.
void deviceInterStep(
    const DeviceMesh&                dm,
    scalar                           deltaT,
    const DeviceInterStepControls&   ctl,
    const DevicePhaseProperties&     props,
    const DeviceInterStepHooks&      hooks,
    // geometry that does not change
    const DeviceBuffer<scalar>&      gh,
    const DeviceBuffer<scalar>&      ghf,        // full face array
    const DeviceBuffer<scalar>&      magSf,      // full face array
    // the state
    DeviceBuffer<scalar>&            alpha1,
    const DeviceBuffer<scalar>&      alpha1Old,
    DeviceBuffer<scalar>&            UX,
    DeviceBuffer<scalar>&            UY,
    DeviceBuffer<scalar>&            UZ,
    const DeviceBuffer<scalar>&      UOldX,
    const DeviceBuffer<scalar>&      UOldY,
    const DeviceBuffer<scalar>&      UOldZ,
    // ...and U.oldTime()'s PATCH values, which ddtCorr's boundary half interpolates with
    const DeviceBuffer<scalar>&      UOldBndX,
    const DeviceBuffer<scalar>&      UOldBndY,
    const DeviceBuffer<scalar>&      UOldBndZ,
    DeviceBuffer<scalar>&            phiInt,
    DeviceBuffer<scalar>&            phiBnd,
    // phi.oldTime(), and U's per-face fixesValue mask. Both are fvc::ddtCorr's, which pEqn.H adds to
    // phiHbyA: phi and U are separate state, so after a step the stored flux and the flux you would
    // get by interpolating the stored velocity DO NOT agree, and that difference is real information
    // the pressure equation needs. A solver that drops it decouples pressure and velocity slowly.
    const DeviceBuffer<scalar>&      phiOldInt,
    const DeviceBuffer<scalar>&      phiOldBnd,
    const DeviceBuffer<int>&         bndUFixesValue,
    DeviceBuffer<scalar>&            p_rgh,
    DeviceBuffer<scalar>&            p,
    // carried across steps so the calculateK pass count matches OpenFOAM's
    DeviceBuffer<scalar>&            nHatfInt,
    DeviceBuffer<scalar>&            nHatfBnd,
    DeviceBuffer<scalar>&            alpha1Bnd,
    DeviceBuffer<scalar>&            K,
    const DeviceBuffer<int>&         bndAlphaFixesValue,
    const DeviceBuffer<int>&         bndAlphaFlag,
    DeviceVectorBoundary&            dbU,
    // what the step leaves for the next one to read
    DeviceBuffer<scalar>&            rho,
    DeviceBuffer<scalar>&            mu,
    DeviceBuffer<scalar>&            nu,
    DeviceBuffer<scalar>&            rhoPhiInt,
    DeviceBuffer<scalar>&            rhoPhiBnd,
    DeviceInterStepTaps*             taps = nullptr);

} // namespace brae
