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
// TURBULENCE IS REFUSED, NOT SKIPPED. The compressible turbulence closure is its own manifest component
// and is not ported. A case whose momentumTransport names RAS or LES is refused BY NAME here rather than
// run as if it were laminar -- which would converge to a smooth, plausible, wrong field. Laminar runs in
// full: muEff is the laminar mu(T) and alphaEff is heThermo's CpByCpv*alpha(T), both from the transport
// model, both recomputed from the current T each iteration.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "rhoCreateFields_cpp.cuh"
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
    scalar relaxHe = 1.0;
    scalar relaxP  = 1.0;          // relaxationFactors/FIELDS p, for p.relax() after the solve
    scalar relaxPEqn = 1.0;        // relaxationFactors/EQUATIONS p, for pEqn.relax()
    bool   relaxPEqnSpecified = false;
    scalar relaxRho = 1.0;         // rho.relax(), applied only when NOT transonic

    // --- linear solver ---
    scalar tolU = 1e-12, relTolU = 0.0;
    scalar tolHe = 1e-12, relTolHe = 0.0;
    scalar tolP = 1e-12, relTolP = 0.0;
    int    maxIter = 2000;

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
    bool   sstLimitedLinear  = false;
    scalar sstLimiterCoeff   = 1.0;
    bool   sstLinearUpwind   = false;
    scalar relaxK = 1.0, relaxEpsilon = 1.0;
    scalar tolTurb = 1e-12, relTolTurb = 0.0;
    bool   boundedTurb = false;         // `bounded Gauss <scheme>` on div(phi,k) and div(phi,epsilon)
    scalar Prt = 1.0;                   // EddyDiffusivity: alphat = rho*nut/Prt

    // --- refusals ---
    bool hasMRF              = false;
    bool hasFvOptions        = false;
    // limitTemperature (fvOptions/corrections/limitTemperature). It is a CORRECTION, not a source: it has
    // no addSup and no constrain, so the momentum, pressure and energy ASSEMBLIES are untouched and it
    // acts only as fvOptions.correct(he) after the energy solve, clamping he between he(p,Tmin) and
    // he(p,Tmax). That is why a case whose only option is this one need not set hasFvOptions -- any
    // OTHER option still does, and is still refused by name.
    bool   limitT    = false;
    scalar limitTmin = 0.0;
    scalar limitTmax = 0.0;
    bool hasFixedFluxPressure = false;
};

// field name -> the INITIAL residual of its first solve this iteration, which is what simpleControl's
// residualControl compares against.
using Residuals = std::map<std::string, scalar>;

// thermo.correct(): recover T from he, then psi from that T. EEqn.H calls it at its end, and every
// consumer downstream of that point -- pEqn's psi, the next iteration's mu and alpha -- reads the result.
// Separated out because it is a thermo operation appearing inside an energy file, and because the gate
// needs to be able to run it on its own.
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
