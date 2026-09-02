#pragma once
// _cpp REFERENCE -- host transcription of rhoSimpleFoam's driver: one full SIMPLE iteration, composed
// from the six components this port has already gated individually.
//
// provenance:
//   openfoam:
//     file: applications/solvers/compressible/rhoSimpleFoam/rhoSimpleFoam.C:55-100
//     also: applications/solvers/compressible/rhoSimpleFoam/pEqn.H:70-110  (the post-solve tail)
//           applications/solvers/compressible/rhoSimpleFoam/pcEqn.H:83-123 (the same tail, rAtU)
//           src/thermophysicalModels/basic/heThermo/heThermo.C  (alphahe = CpByCpv*alpha)
//   brae:
//     reference: src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam_cpp.cu
//     cuda:      (pending -- this file is what has to run end-to-end BEFORE any .cu, see PORT.md)
//     tests:     tests/test_rho_simple_step_cpp.cu
//
// OpenFOAM, verbatim:
//
//     while (simple.loop())
//     {
//         #include "UEqn.H"
//         #include "EEqn.H"
//         if (simple.consistent()) { #include "pcEqn.H" }
//         else                     { #include "pEqn.H"  }
//         turbulence->correct();
//         runTime.write();
//     }
//
// THE DRIVER OWNS NO NUMERICS. Every term comes from a component with its own OpenFOAM provenance and its
// own gate against real OpenFOAM: createFields, UEqn, EEqn, pEqn, pcEqn. What this file adds is the
// ORDER, the linear solves, `thermo.correct()`, and the post-solve tail -- and the order is the part that
// cannot be checked by looking at any one component.
//
// ORDER, AND WHY EACH POSITION MATTERS:
//   1. UEqn      solved first; its matrix is what pEqn's rAU and H() are taken from.
//   2. EEqn      AFTER the momentum solve, so its kinetic-energy source uses the just-solved U. It ends
//                with thermo.correct(), which moves T -- and therefore psi, mu and alpha -- so everything
//                downstream sees a newer thermodynamic state than the iteration started with.
//   3. p         pcEqn.H when `consistent`, pEqn.H otherwise. pcEqn.H additionally opens with
//                `rho = thermo.rho()`; pEqn.H does not.
//   4. turbulence->correct() LAST, so the momentum equation of iteration n uses the closure from n-1.
//                A LAGGED coupling: correcting before UEqn instead is a different algorithm that still
//                converges to something plausible.
//
// TURBULENCE: TWO CLOSURES ARE PORTED, EVERY OTHER MODEL IS REFUSED BY NAME. kEpsilon and kOmegaSST run
// in full, each with its own gate; anything else -- realizableKE, RNGkEpsilon, LaunderSharmaKE,
// SpalartAllmaras, LES -- throws and names itself rather than running as kEpsilon or as laminar, either
// of which converges to a smooth, plausible, wrong field.
//
// This paragraph used to say the closure was "not ported" and that ANY RAS case was refused. That was
// true when written; it stopped being true when the closure landed, and a contract paragraph that
// contradicts the code twenty lines below it is worse than none -- a caller reads it, concludes every
// turbulent case throws, and neither supplies the closure's inputs nor treats a turbulent run as the
// supported path it is.
//
// Laminar runs in full too: muEff is the laminar mu(T) and alphaEff is heThermo's CpByCpv*alpha(T), both
// from the transport model, both recomputed from the current T each iteration.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "rhoCreateFields_cpp.cuh"
#include "fvOptions_cpp.cuh"
#include "komega_sst_coeffs.cuh"   // KOmegaSSTCoeffs, for a case whose RASModel is kOmegaSST
#include "rhoUEqn_cpp.cuh"
#include "rhoEEqn_cpp.cuh"
#include "rhoPEqn_cpp.cuh"
#include "rhoPcEqn_cpp.cuh"
#include "kEpsilon_cpp.cuh"
#include <map>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace rhoSimple {

struct StepInput
{
    // --- algorithm ---
    bool consistent = false;   // simple.consistent() -> pcEqn.H rather than pEqn.H
    bool transonic  = false;   // simple.transonic()  -> the convective pressure branch
    // `while (simple.correctNonOrthogonal())`: the pressure equation is re-assembled and re-solved
    // nNonOrthogonalCorrectors + 1 times (solutionControlI.H:78-95), each pass recomputing the deferred
    // non-orthogonal correction from the just-solved p; only the final pass's matrix feeds pEqn.flux().
    // A trajectory feature, not a fixed-point one: at convergence every pass re-solves an unchanged
    // system, so a converged comparison cannot see the count -- the gate compares at matched EARLY
    // iterations. This step used to solve exactly once whatever the case named.
    label nNonOrthogonalCorrectors = 0;

    // --- schemes ---
    DivScheme schemeU   = DivScheme::upwind;
    DivScheme schemeHe  = DivScheme::upwind;
    DivScheme schemeKE  = DivScheme::upwind;
    bool      boundedU  = false;
    bool      boundedHe = false;
    bool      boundedKE = false;
    scalar    schemeCoeffU     = 1.0;
    scalar    gradULimitK      = 0.0;
    scalar    gradHeLimitK     = 0.0;
    scalar    gradKELimitK     = 0.0;
    bool      correctedLaplacian = false;
    scalar    snGradLimitCoeff   = 0.0;

    // --- relaxation ---
    scalar relaxU  = 1.0;
    // Whether fvSolution NAMES a factor for U / he. NOT the same question as `< 1`:
    // fvMatrix::relax(1.0) still applies the dominance clamp, so a case naming 1 relaxes and a case
    // naming nothing does not. relaxPEqnSpecified below has always carried this for p.
    bool   relaxEquationU  = false;
    bool   relaxEquationHe = false;
    // ...and the same for the closure, where the old behaviour was the opposite error: it relaxed
    // unconditionally, applying a clamp OpenFOAM does not apply when the case names nothing.
    bool   relaxEquationK   = true;
    bool   relaxEquationEps = true;
    scalar relaxHe = 1.0;
    scalar relaxP  = 1.0;          // relaxationFactors/FIELDS p, for p.relax() after the solve
    scalar relaxPEqn = 1.0;        // relaxationFactors/EQUATIONS p, for pEqn.relax()
    bool   relaxPEqnSpecified = false;
    scalar relaxRho = 1.0;         // rho.relax(), applied only when NOT transonic
    // Diagnostic only: the pressure floor a GATE forced, so it can report how many cells sit on it.
    // The solver reads pMin/pMax from PressureControl, never from here.
    scalar pMinProbe = 0.0;
    scalar heMinProbe = 0.0;   // likewise, for the energy floor a gate forced

    // --- linear solver ---
    scalar tolU = 1e-12, relTolU = 0.0;
    scalar tolHe = 1e-12, relTolHe = 0.0;
    scalar tolP = 1e-12, relTolP = 0.0;
    // Per equation, as OF reads them (lduMatrixSolver.C:204-205): a cap decides where a solve stops,
    // and one cap for four equations meant p's cap on every one of them.
    int    maxIterU = 2000, maxIterP = 2000, maxIterHe = 2000, maxIterTurb = 2000;
    int    minIterU = 0,    minIterP = 0,    minIterHe = 0,    minIterTurb = 0;

    // --- turbulence ---
    // kEpsilon, the compressible instantiation. Anything else is refused in createFields by name. The
    // model runs LAST in the iteration, so the momentum equation of iteration n uses the closure from
    // n-1 -- OpenFOAM's lagged coupling, and correcting before UEqn is a different algorithm.
    KEpsilonCoeffs keCoeffs{};
    // kOmegaSST's, for a case whose RASModel names it. Its scheme flags are separate because kOmegaSST
    // takes them as arguments rather than in its coefficients, and the SST tutorials ask for
    // `bounded Gauss limitedLinear 1` on both scalars where the kEpsilon ones ask for plain upwind.
    KOmegaSSTCoeffs sstCoeffs{};
    scalar relaxOmega        = 1.0;
    scalar gradKLimitK       = 0.0;   // gradSchemes/grad(k) and grad(omega), for CDkOmega and F1
    // ONE scheme for div(phi,k) and div(phi,epsilon|omega), whichever model the case names -- the
    // harness refuses a case whose two entries disagree. The coefficient is the RAW k of
    // `limitedLinear k` (the weights functions compute twoByk themselves, scheme_parse.cuh).
    bool   limitedLinearTurb = false;
    scalar turbLimiterCoeff  = 1.0;
    bool   linearUpwindTurb  = false; // kOmegaSST assembles it; kEpsilon does not -- refused upstream
    scalar relaxK = 1.0, relaxEpsilon = 1.0;
    scalar tolTurb = 1e-12, relTolTurb = 0.0;
    bool   boundedTurb = false;         // `bounded Gauss <scheme>` on div(phi,k) and div(phi,epsilon)
    // Non-empty = the CASE names a convection scheme on div(phi,k)/div(phi,epsilon|omega) that the
    // compressible closure does not assemble (Gauss upwind and Gauss limitedLinear, each with or
    // without `bounded`, are ported; linearUpwind and entries that disagree between the two scalars
    // are not).
    // The step REFUSES before the closure runs; the device twin carries the same refusal
    // (kEpsilon.cu, hasNonUpwindDivScheme) and both used to be set only by fail-proofs -- the flag was
    // hardcoded in the harness and the case's own fvSchemes never reached either arm.
    std::string turbDivUnsupported;
    scalar Prt = 1.0;                   // EddyDiffusivity: alphat = rho*nut/Prt

    // --- refusals ---
    bool hasMRF              = false;
    bool hasFvOptions        = false;
    std::string fvOptionUnsupported;   // which one, so the refusal names it
    // The IMPLEMENTED fvOptions, read from the case. hasFvOptions above stays TRUE only for the ones
    // this port does not implement, so a case carrying both an explicitPorositySource and, say, an
    // explicit heat source is still refused by name rather than run with half its options applied.
    const cpu::fvOptions::OptionList* fvOpts = nullptr;
    // limitTemperature (fvOptions/corrections/limitTemperature). It is a CORRECTION, not a source: it has
    // no addSup and no constrain, so the momentum, pressure and energy ASSEMBLIES are untouched and it
    // acts only as fvOptions.correct(he) after the energy solve, clamping he between he(p,Tmin) and
    // he(p,Tmax). That is why a case whose only option is this one need not set hasFvOptions -- any
    // OTHER option still does, and is still refused by name.
    bool   limitT    = false;
    scalar limitTmin = 0.0;
    scalar limitTmax = 0.0;
};

// field name -> the INITIAL residual of its first solve this iteration, which is what simpleControl's
// residualControl compares against.
using Residuals = std::map<std::string, scalar>;

// thermo.correct(): recover T from he, then psi from that T. EEqn.H calls it at its end, and every
// consumer downstream of that point -- pEqn's psi, the next iteration's mu and alpha -- reads the result.
// Separated out because it is a thermo operation appearing inside an energy file, and because the gate
// needs to be able to run it on its own.
// rho = thermo.rho(), internal and boundary, from the CURRENT p and T. WHICH rho that is depends on the
// thermo type: for hePsiThermo it is p*psi recomputed live, for heRhoThermo it is the STORED rho_ that
// heRhoThermo::calculate() last filled inside thermo.correct() -- from the pressure BEFORE the pressure
// equation ran. Exported because the CUDA driver takes it as a hook and its gate must give both sides
// the same one.
void updateRho(
    RhoSimpleFields&            f,
    const std::vector<FvPatch>& patches);

void thermoCorrect(
    RhoSimpleFields&            f,
    const std::vector<FvPatch>& patches);

// The effective transport, recomputed from the CURRENT state:
//     laminar     muEff = mu(T)              alphaEff = CpByCpv*alpha(T)
//     turbulent   muEff = mu(T) + rho*nut    alphaEff = CpByCpv*(alpha(T) + alphat)
// which is heThermo::alphaEff(alphat) and the compressible model's muEff. These two lines are the ONLY
// place the closure enters the momentum and energy equations -- everything else about a turbulent case is
// the same solver -- which is why they are one function rather than scattered through the step.
void effectiveTransport(
    const RhoSimpleFields&            f,
    const std::vector<FvPatch>&       patches,
    std::vector<scalar>&              muEff,
    std::vector<std::vector<scalar>>& muEffBnd,
    std::vector<scalar>&              alphaEff,
    std::vector<std::vector<scalar>>& alphaEffBnd);

// ONE SIMPLE iteration, in place on f. Returns the initial residual of each field's first solve.
Residuals rhoSimpleStep(
    RhoSimpleFields&            f,
    const StepInput&            in,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
