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
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_inter_alpha_step.cuh"
#include "device_inter_pressure_step.cuh"
#include "device_inter_ueqn.cuh"
#include "UEqn.cuh"
#include "pEqn.cuh"
#include <functional>

namespace brae {

struct DeviceInterStepHooks
{
    DeviceInterAlphaHooks    alpha;      // alpha's patch values + contact angle, and fvm::div's coeffs
    DeviceInterPressureHooks pressure;   // p_rgh's laplacian boundary coefficients, after constrainPressure

    // U's boundary, rebuilt from whatever the device last wrote. It is refreshed once per step rather
    // than per corrector because interFoam's PIMPLE loop evaluates it there.
    // `UbStored` are U's EVALUATED patch values, one buffer per component, as the host's evaluate
    // left them. They are NOT the same as re-deriving them on the device with deviceBCValue: at a
    // flux-conditional patch like damBreak's pressureInletOutletVelocity atmosphere the two differ,
    // and fvc::grad(U) inside divDevRhoReff reads the STORED ones. MEASURED with the re-derived value
    // standing in: the dev2 term's contribution was 100% wrong on that patch's 46 cells (2.24e-06 of
    // 2.24e-06) while the three walls were between exact and 0.8%.
    std::function<void(const DeviceBuffer<scalar>& Ux,
                       const DeviceBuffer<scalar>& Uy,
                       const DeviceBuffer<scalar>& Uz,
                       DeviceVectorBoundary&       dbU,
                       DeviceBuffer<scalar>*       UbStored)> updateUBoundary;

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

struct DeviceInterStepControls
{
    // the cell volumes the mesh had BEFORE this step's move, for the ddt's old-time term
    // (OF EulerDdtScheme: rho.oldTime()*U.oldTime()*Vsc0()). Null on a static mesh.
    const DeviceBuffer<scalar>* V0 = nullptr;
    // ...and the mesh flux over the FULL face array, for fvc::makeRelative(phi, U) at the end of the
    // pressure corrector. Null on a static mesh, where OpenFOAM's makeRelative is a no-op.
    const DeviceBuffer<scalar>* meshPhiAll = nullptr;
    // (Sf & Uf.oldTime()) per internal face, which ddtCorr takes in phi.oldTime()'s place on a moving
    // mesh (fvcDdt.C:220-227 -> EulerDdtScheme's fvcDdtUfCorr). Null on a static mesh.
    const DeviceBuffer<scalar>* phiUfOldInt = nullptr;

    // The case's MRF zones, built once by the driver (buildDeviceMRFZone). interFoam touches them in
    // three places -- UEqn.H:6 MRF.DDt(rho, U), pEqn.H:18 MRF.zeroFilter(ddtCorr term) and pEqn.H:19
    // MRF.makeRelative(phiHbyA) -- and the fourth, MRF.correctBoundaryVelocity(U) at UEqn.H:1, is the
    // driver's because it writes the HOST field the boundary snapshot is taken from.
    const std::vector<DeviceMRFZone>* mrf = nullptr;
    // fvOptions' explicitPorositySource/DarcyForchheimer, built by the driver from the host's own
    // OptionList (its transformed D and F tensors). Every other option type stays host-only and the
    // driver refuses it by name.
    const DevicePorosity* porosity = nullptr;
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
    scalar snGradLimitCoeff = 0;
    bool   momentumPredictor = true;         // damBreak sets this OFF
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
