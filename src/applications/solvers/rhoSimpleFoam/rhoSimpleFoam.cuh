#pragma once
// CUDA DRIVER -- one rhoSimpleFoam SIMPLE iteration, device-resident, composed of the ported components.
//
// provenance:
//   openfoam:  applications/solvers/compressible/rhoSimpleFoam/rhoSimpleFoam.C:64-100 (the loop body)
//     tail:    .../rhoSimpleFoam/pEqn.H:70-110 and pcEqn.H:83-123 (the post-solve tail)
//   reference: src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam_cpp.cu  (gated end-to-end against
//              real OpenFOAM by tests/rho_simple_end_to_end_vs_openfoam.sh)
//   cuda:      src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam.cu
//   tests:     tests/test_rho_simple_step_cuda.cu
//
// THE DRIVER OWNS NO NUMERICS. Every term comes from a component with its own OpenFOAM provenance and its
// own gate:
//
//     gpu::rhoSimple::assembleUEqn                rhoUEqn.cu    (fvm::div + divDevRhoReff + relax)
//     gpu::rhoSimple::addPressureGradient         rhoUEqn.cu    (-fvc::grad(p))
//     gpu::rhoSimple::assembleEEqn                rhoEEqn.cu    (he transport + the kinetic-energy source)
//     gpu::rhoSimple::pressurePredictor           rhoPEqn.cu    (rAU, HbyA, phiHbyA, adjustPhi)
//     gpu::rhoSimple::consistentPressurePredictor rhoPcEqn.cu   (...and rAtU, when `consistent`)
//     gpu::kEpsilonRAS::correct                   kEpsilon.cu   (via the caller's hook -- see below)
//
// What this file adds is the ORDER, the linear solves and the post-solve tail -- and the order is exactly
// the part no single component's gate can check.
//
// ORDER, AND WHY EACH POSITION MATTERS (rhoSimpleFoam.C:64-100):
//   1. UEqn   solved first, on a COPY with -grad(p) added: the pressure equation needs the ORIGINAL
//             matrix for A(), H() and H1(), and adding grad(p) to it would change rAU and HbyA.
//   2. EEqn   AFTER the momentum solve, so its kinetic-energy source uses the just-solved U. It ends in
//             thermo.correct(), which moves T and therefore psi -- so everything below sees a newer
//             thermodynamic state than the iteration started with.
//   3. p      pcEqn.H when `consistent`, pEqn.H otherwise. pcEqn.H additionally OPENS with
//             `rho = thermo.rho()`; pEqn.H does not, so the SIMPLEC pressure equation is built from a
//             density that already reflects the just-solved temperature and the plain SIMPLE one is not.
//   4. p is relaxed AFTER the flux correction and BEFORE the velocity correction, so phi is built from
//             the unrelaxed pressure and U from the relaxed one.
//   5. turbulence->correct() LAST, so iteration n's momentum equation uses the closure from n-1. A
//             LAGGED coupling; correcting before UEqn is a different algorithm that still converges to
//             something plausible.
//
// THE FLUX CORRECTION ADDS, IT DOES NOT SUBTRACT. rhoSimpleFoam writes its pressure equation as
// `fvc::div(phiHbyA) - fvm::laplacian(...) == 0` and rhoPEqn_cpp negates the whole assembled matrix to
// match, where the incompressible solver writes `fvm::laplacian(...) == fvc::div(phiHbyA)`. So
// `phi = phiHbyA + pEqn.flux()` here against `phiHbyA - pEqn.flux()` there, and gpu::correctFlux
// (pEqn.cu) is NOT reusable on this path despite the note in rhoPEqn.cuh saying it is -- it subtracts,
// and it takes the incompressible PressureStages. This driver carries its own.
//
// SOLVER SELECTION IS A CORRECTNESS SWITCH, NOT A PERFORMANCE ONE. The subsonic pressure equation's only
// implicit term is fvm::laplacian, so upper == lower and the matrix is SYMMETRIC: AMG-PCG applies, and
// the cached hierarchy with it. The transonic branch adds fvm::div(phid, p), whose lower = -w*phi and
// upper = lower + phi, so the matrix is ASYMMETRIC and must go to BiCGStab. Running CG on it burned the
// full 3000-iteration cap and stalled before printing iteration 1.
//
// THE AMG HIERARCHY IS A FUNCTION OF THE MESH, NOT OF THE EQUATION. Only the STRUCTURE is cached --
// cDiag/cUpper/cLower are Galerkin-rebuilt every step -- so a hierarchy a simpleFoam run wrote for this
// mesh is valid here. It is built ONLY on the subsonic branch, so a transonic case does not pay for an
// agglomeration it never uses.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_amg.cuh"
#include "rhoUEqn.cuh"
#include "rhoEEqn.cuh"
#include "rhoPEqn.cuh"
#include "rhoPcEqn.cuh"
#include "device_fvoptions.cuh"   // DevicePorosity
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace brae {
namespace gpu {
namespace rhoSimple {

// The device-resident solution state the loop carries between iterations. Every boundary-face twin is
// held beside its cell field because the modules take the PATCH value, not the owner cell's -- that
// distinction is load-bearing in the diffusivity of all four equations.
struct RhoSolverFields
{
    DeviceBuffer<scalar> Ux, Uy, Uz;
    // U's boundary values, one buffer per component: DeviceVectorBoundary is three DeviceBoundary
    // objects, so there is no single vector-valued patch array to point at. The energy equation reads
    // them because fvc::div(phi, Ekp) evaluates Ekp on boundary faces too.
    DeviceBuffer<scalar> UxBnd, UyBnd, UzBnd;
    DeviceBuffer<scalar> p,   pBnd;
    DeviceBuffer<scalar> he,  heBnd;
    DeviceBuffer<scalar> T,   TBnd;
    DeviceBuffer<scalar> psi, psiBnd;
    DeviceBuffer<scalar> rho, rhoBnd;
    DeviceBuffer<scalar> phiInt, phiBnd;
    // The closure's state. Owned here rather than by the turbulence hook so that the hook stays a
    // callback and does not need to reach back into the driver for its own fields.
    DeviceBuffer<scalar> k, epsilon, nut, nutBnd, alphat;
    // alphat at boundary FACES. The energy equation's diffusivity on a patch is the WALL FUNCTION's,
    // never the adjacent cell's -- at a fixed-temperature wall with compressible::alphatWallFunction the
    // two differ by the whole of alphat, and falling back to the cell value silently removes it.
    DeviceBuffer<scalar> alphatBnd;

    // The THERMO's own density, cells and boundary. NOT the solver's rho above: heRhoThermo::calculate()
    // rewrites this inside thermo.correct(), and rhoThermo::rho() then returns it -- so on a heRhoThermo
    // case the density the solver carries lags the pressure by one outer iteration, which is the whole
    // point of keeping the two apart. Conflating them let calculate() overwrite a RELAXED rho
    // mid-iteration: measured on angledDuct at iteration 2, OpenFOAM's rho spread was 0.002 and brae's
    // 0.94. Held here rather than inside a thermo object because the driver's thermo is a hook, and a
    // device-resident hook and a host one must be able to write the same state.
    DeviceBuffer<scalar> rhoThermo, rhoThermoBnd;

    // fvc::domainIntegrate(psi*p) at the start of the run, for the closed-volume correction.
    double initialMass = 0.0;
};

struct RhoStepInput
{
    // --- algorithm ---
    bool consistent = false;   // simple.consistent() -> pcEqn.H rather than pEqn.H
    bool transonic  = false;   // simple.transonic()  -> the convective pressure branch

    // --- the effective transport for THIS iteration, supplied by the caller because it comes from the
    //     thermo and the closure, both of which the caller owns:
    //       laminar     muEff = mu(T)             alphaEff = CpByCpv*alpha(T)
    //       turbulent   muEff = mu(T) + rho*nut   alphaEff = CpByCpv*(alpha(T) + alphat)
    const DeviceBuffer<scalar>* muEffCell       = nullptr;
    const DeviceBuffer<scalar>* muEffBndFace    = nullptr;
    const DeviceBuffer<scalar>* alphaEffCell    = nullptr;
    const DeviceBuffer<scalar>* alphaEffBndFace = nullptr;

    // --- schemes ---
    cpu::rhoSimple::DivScheme schemeU  = cpu::rhoSimple::DivScheme::upwind;
    cpu::rhoSimple::DivScheme schemeHe = cpu::rhoSimple::DivScheme::upwind;
    cpu::rhoSimple::DivScheme schemeKE = cpu::rhoSimple::DivScheme::upwind;
    bool   boundedU = false, boundedHe = false, boundedKE = false;
    scalar schemeCoeffU = 1.0;
    scalar gradULimitK = 0.0, gradHeLimitK = 0.0, gradKELimitK = 0.0;
    bool   correctedLaplacian = false;
    scalar snGradLimitCoeff   = 0.0;
    bool   isE = true;                    // he == "e" selects Ekp, "h" selects K

    // --- relaxation. Each carries the "the case NAMES a factor" sentinel for the same reason the
    //     modules do: fvMatrix::relax early-returns only on alpha <= 0, so relax(1.0) is not identity.
    bool   relaxEquationU = false;  scalar relaxU  = 1.0;
    bool   relaxEquationHe = false; scalar relaxHe = 1.0;
    scalar relaxP    = 1.0;              // relaxationFactors/FIELDS p   -- p.relax() after the solve
    scalar relaxPEqn = 1.0;              // relaxationFactors/EQUATIONS p -- pEqn.relax(), transonic only
    bool   relaxPEqnSpecified = false;
    scalar relaxRho  = 1.0;              // rho.relax(), applied only when NOT transonic

    // --- linear solver ---
    scalar tolU = 1e-12, relTolU = 0.0;
    scalar tolHe = 1e-12, relTolHe = 0.0;
    scalar tolP = 1e-12, relTolP = 0.0;
    int    maxIter = 2000;
    bool   uSymGaussSeidel = false;
    bool   captureVcycle = true;
    int    pcgCheckEvery = 1;
    // Reuse the AMG hierarchy STRUCTURE across runs. The agglomeration is the build cost and depends only
    // on the mesh, so it serialises next to constant/polyMesh. Empty = no caching -- and empty is what
    // the compressible application passed until now, which is why every compressible run started cold
    // even though buildOrLoadAMG would have loaded a hierarchy any simpleFoam run had left there.
    std::string amgCacheDir;

    label  pRefCell  = -1;
    scalar pRefValue = 0.0;

    // pressureControl::limit(p) -- pEqn.H/pcEqn.H apply it AFTER the velocity correction, and a clip
    // requires p's boundary values to be re-evaluated afterwards. The CUDA driver used to drop this
    // entirely and refuse nothing, so a case naming pMin/pMax in its SIMPLE dict got them on the host
    // reference and not on the device -- the silent substitution this project exists to catch, in the
    // driver rather than in a module. rhoBox names `pMin 1000`, which never binds there, which is
    // exactly why the driver's own gate did not notice.
    bool   limitMaxP  = false;
    bool   limitMinP  = false;
    scalar pMaxLimit  = 0.0;
    scalar pMinLimit  = 0.0;
    label  nNonOrthogonalCorrectors = 0;

    const DeviceBuffer<label>* adjustable      = nullptr;   // adjustPhi mask     (!fixesValue)
    const DeviceBuffer<label>* takeUAtBoundary = nullptr;   // constrainHbyA mask (!assignable)

    // --- the fvOptions this port DOES implement -------------------------------------------------
    // explicitPorositySource (DarcyForchheimer / fixedCoeff). UEqn.H applies fvOptions(rho, U), and for a
    // porosity that is `eqn -= porosityEqn` with the resistance the model builds. rhoUEqn.cu has taken
    // this since it was written; the DRIVER had no field to carry it and never set uin.porosity, so a
    // porous case ran with the porosity silently absent -- it drives the duct at the wrong speed and
    // still converges. validation/angledDuct and OpenFOAM's own angledDuctExplicitFixedCoeff are exactly
    // that case. Null means the case has none; `hasFvOptions` continues to mean an UNPORTED one.
    const DevicePorosity* porosity = nullptr;

    // limitTemperature (fvOptions/corrections/limitTemperature). A CORRECTION, not a source: it has no
    // addSup and no constrain, so no assembly changes -- it acts only as fvOptions.correct(he) AFTER the
    // energy solve and BEFORE thermo.correct(), which is what makes it show up in T at all. The bounds
    // arrive already converted to ENERGY because the conversion needs the thermo, and the thermo is the
    // caller's here for the same reason thermoCorrect and updateRho are hooks.
    bool   limitHe = false;
    scalar heMin   = 0.0;
    scalar heMax   = 0.0;

    // --- updateCoeffs() for the boundary conditions whose coefficients move with the solution ------
    // OpenFOAM's fvMatrix constructor calls updateCoeffs() at every assembly, so a patch that switches
    // on the flux, blends on the flow angle, or holds a mass flow against the live density is refreshed
    // before any coefficient is read. The device boundary objects are a PRE-BAKED snapshot instead, so
    // the driver has to do it explicitly -- rhoUEqn.cuh:64-83 spells the contract out and calls it "not
    // advisory". The driver satisfied none of it: every such patch kept its seeded coefficients for the
    // whole run. The driver's own gate could not see that, because rhoBox carries only fixedValue,
    // zeroGradient, noSlip and empty -- not one patch of any of these three kinds. sbMatched carries
    // four inletOutlet and one flowRateInletVelocity, and is what the gate's second arm runs on.
    //
    // Supplied by the caller rather than derived here because it is CASE geometry, gathered once by
    // createFields; the driver is handed the same objects every iteration.
    bool hasMixed = false;
    // One magSf mask per flowRateInletVelocity patch, with the prescribed mass flows in the same order.
    const std::vector<DeviceBuffer<scalar>>* frMagSf = nullptr;
    const std::vector<scalar>*               frMdot  = nullptr;
    const DeviceBuffer<scalar>*              frNx    = nullptr;
    const DeviceBuffer<scalar>*              frNy    = nullptr;
    const DeviceBuffer<scalar>*              frNz    = nullptr;

    // --- refusals, each thrown by the component that owns the term ---
    bool hasMRF = false, hasFvOptions = false, hasCoupledPatches = false;
    std::string fvOptionUnsupported;

    // thermo.correct(): recover T from he, then psi from that T. A hook because it is a THERMO operation
    // and the thermo is the caller's -- the driver must not learn to be a thermodynamic model.
    std::function<void()> thermoCorrect;
    // rho = thermo.rho(), cells AND boundary. Separate from thermoCorrect because pcEqn.H calls it at its
    // OPENING and both branches call it again in the tail, and those are different moments.
    std::function<void()> updateRho;
    // turbulence->correct(), at the END of the iteration exactly where rhoSimpleFoam.C:97 calls it. A
    // hook rather than a hard dependency: the model is runtime-selected, and a driver that hard-codes
    // kEpsilon is the god object starting over.
    std::function<void()> correct;
};

using Residuals = std::map<std::string, scalar>;

// Scratch that persists ACROSS iterations. Kept out of RhoStepInput so the loop cannot rebuild it by
// accident: allocating the pressure buffers fresh each iteration gives the persistent AMG state a
// different fine matrix every time, and the cached V-cycle and PCG graphs are keyed on that matrix. That
// defect was exact at iteration 1 and 1.3e-01 wrong at iteration 2, with every per-stage gate green
// throughout -- which is the whole reason this driver has an end-to-end test of its own.
struct RhoSolverWorkspace
{
    AMGData amg;
    bool    amgBuilt = false;
    DeviceBuffer<scalar> ones;
    PressureMatrix       P;
    DeviceBuffer<scalar> diagC, b;
};

// ONE rhoSimpleFoam SIMPLE iteration, in place on `f`. Returns the INITIAL residual of each field's first
// solve this iteration, which is what simpleControl's residualControl compares against.
// updateCoeffs() for the boundary conditions whose coefficients are a function of the SOLUTION -- the
// flux switch (inletOutlet/outletInlet), the freestream flow-angle blend, and flowRateInletVelocity's
// velocity from the live boundary density. rhoSimpleStep calls this at the top of every iteration;
// it is exposed because it has a contract testable on its own, and because the gate needs to withhold it.
void updateBoundaryCoeffs(
    RhoSolverFields&      f,
    DeviceVectorBoundary& dbU,
    DeviceBoundary&       dbP,
    DeviceBoundary&       dbHe,
    DeviceBoundary&       dbT,
    const RhoStepInput&   in);

Residuals rhoSimpleStep(
    RhoSolverFields&            f,
    RhoSolverWorkspace&         w,
    const DeviceMesh&           dm,
    // NON-const: the updateCoeffs() block at the top of the step rewrites refValue and valueFraction on
    // the patches that switch on the flux, blend on the flow angle, or carry a prescribed mass flow.
    DeviceVectorBoundary&       dbU,
    DeviceBoundary&             dbP,
    DeviceBoundary&             dbHe,
    DeviceBoundary&             dbT,
    const RhoStepInput&         in);

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
