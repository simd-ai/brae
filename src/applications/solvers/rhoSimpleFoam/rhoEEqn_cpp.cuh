#pragma once
// _cpp REFERENCE -- host transcription of rhoSimpleFoam's energy equation.
//
// provenance:
//   openfoam:
//     file: applications/solvers/compressible/rhoSimpleFoam/EEqn.H:1-31
//     also: src/finiteVolume/finiteVolume/convectionSchemes/boundedConvectionScheme/
//           boundedConvectionScheme.C  (fvmDiv AND fvcDiv -- the bounded term has BOTH forms)
//   brae:
//     reference: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu
//     cuda:      (pending -- the whole case runs in _cpp first, see PORT.md)
//     tests:     tests/test_rho_eeqn_cpp.cu
//
// OpenFOAM, verbatim:
//
//     volScalarField& he = thermo.he();
//
//     fvScalarMatrix EEqn
//     (
//         fvm::div(phi, he)
//       + (
//             he.name() == "e"
//           ? fvc::div(phi, volScalarField("Ekp", 0.5*magSqr(U) + p/rho))
//           : fvc::div(phi, volScalarField("K", 0.5*magSqr(U)))
//         )
//       - fvm::laplacian(turbulence->alphaEff(), he)
//      ==
//         fvOptions(rho, he)
//     );
//
//     if (MRF.active())
//     {
//         EEqn += fvc::div(MRF.phi(), p);
//     }
//
//     EEqn.relax();
//     fvOptions.constrain(EEqn);
//     EEqn.solve();
//     fvOptions.correct(he);
//     thermo.correct();
//
// THE BRANCH IS THE COMPONENT. `he.name()` is "e" or "h" depending on the thermo's `energy` entry, and the
// two arms carry DIFFERENT kinetic-energy sources:
//
//     e  ->  Ekp = 0.5|U|^2 + p/rho        internal energy: the flow work p/rho is carried explicitly
//     h  ->  K   = 0.5|U|^2                enthalpy: p/rho is already inside h, so adding it would double it
//
// These are not a rounding apart. On the validation fixture p ~ 1.1e5 and rho ~ 0.38, so p/rho ~ 2.9e5
// while 0.5|U|^2 is order 10 -- the wrong arm is wrong by four orders of magnitude on the dominant term.
// It also cannot be caught by looking at a converged temperature field alone, because a solver that
// consistently uses the wrong arm simply converges to a different, smooth, plausible answer. So
// kineticEnergy() below is exposed as its own function and gated directly against OpenFOAM's
// `stage_Ekp`, rather than being an unnamed sub-expression inside the assembly.
//
// A PORT THAT PICKS ONE ARM IS WRONG FOR THE OTHER THERMO, and both appear in OpenFOAM's own tutorials:
// squareBend and its variants are sensibleInternalEnergy (e), while the enthalpy cases select h and carry
// `div(phi,K)` in fvSchemes instead of `div(phi,Ekp)`. This is exactly the kind of branch that
// case-by-case porting reaches one side of.
//
// TWO BOUNDED TERMS, TWO FORMS. `div(phi,e)` and `div(phi,Ekp)` are both `bounded Gauss ...` in every
// rhoSimpleFoam tutorial that names them, and boundedConvectionScheme has a DIFFERENT expression for each
// use:
//     fvmDiv:  scheme.fvmDiv(phi,vf) - fvm::Sp(surfaceIntegrate(phi), vf)     -> the matrix diagonal
//     fvcDiv:  scheme.fvcDiv(phi,vf) - surfaceIntegrate(phi)*vf               -> an explicit field
// The implicit one is applied to `he`, the explicit one to the kinetic-energy field. Applying the matrix
// form to both, or neither to the explicit term, changes the answer only away from convergence -- which
// is precisely where it is hardest to notice.
//
// NOT COVERED HERE, and refused rather than skipped: MRF (`EEqn += fvc::div(MRF.phi(), p)`) and fvOptions.
// Separate manifest components.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvc.cuh"
#include "rhoUEqn_cpp.cuh"      // DivScheme -- the same scheme set, named once
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace rhoSimple {

struct EnergyInput
{
    // MASS flux, kg/s.
    const std::vector<scalar>*              phi         = nullptr;
    const std::vector<std::vector<scalar>>* phiBnd      = nullptr;

    // turbulence->alphaEff(), the effective thermal diffusivity. Injected rather than computed here: the
    // compressible turbulence closure is its own manifest component.
    const std::vector<scalar>*              alphaEff    = nullptr;
    const std::vector<std::vector<scalar>>* alphaEffBnd = nullptr;

    // "e" or "h". Anything else is refused -- thermo.validate(.., "h", "e") refuses the same set.
    std::string heName;

    scalar    relaxHe            = 1.0;
    // See RhoMomentumInput::relaxEquationU -- the guard is whether the case NAMES a factor, not whether
    // the factor is below 1.
    bool      relaxEquationHe    = false;
    // `bounded` on div(phi,he) and on div(phi,Ekp|K) INDEPENDENTLY: they are separate fvSchemes entries
    // and a case may bound one and not the other.
    bool      boundedHe          = false;
    bool      boundedKE          = false;
    DivScheme schemeHe           = DivScheme::upwind;
    DivScheme schemeKE           = DivScheme::upwind;
    // The `k` of the gradient each linearUpwind statement NAMES.
    scalar    gradHeLimitK       = 0.0;
    scalar    gradKELimitK       = 0.0;
    bool      correctedLaplacian = false;
    scalar    snGradLimitCoeff   = 0.0;
    bool      hasMRF             = false;
    bool      hasFvOptions       = false;
};

// EEqn.H's kinetic-energy source, per cell. THE branch:
//     "e" -> 0.5|U|^2 + p/rho        "h" -> 0.5|U|^2
// Exposed on its own so it can be gated directly against OpenFOAM's stage_Ekp instead of only through
// the assembled matrix.
std::vector<scalar> kineticEnergy(
    const std::string&            heName,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho);

// The same on boundary faces, from the patch values of U, p and rho.
std::vector<std::vector<scalar>> kineticEnergyBoundary(
    const std::string&            heName,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho,
    const std::vector<FvPatch>&   patches);

// EEqn.H's explicit kinetic-energy divergence, `fvc::div(phi, Ekp|K)`, returned EXTENSIVE (V*div) --
// which is how an fvMatrix consumes an added field (`source -= V*field`). Exposed so the branch's EFFECT
// on the equation, not only the field it builds, can be measured on its own.
std::vector<scalar> kineticEnergyDivergence(
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho,
    const EnergyInput&            in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// EEqn.H up to and including EEqn.relax(): the implicit convection and diffusion of he, plus the explicit
// kinetic-energy divergence in the source. Returned before solve(), which is the state OpenFOAM's
// stage_eD / stage_eSrc harness dumps.
FvScalarMatrix assembleEEqn(
    const GeometricField<scalar>& he,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho,
    const EnergyInput&            in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
