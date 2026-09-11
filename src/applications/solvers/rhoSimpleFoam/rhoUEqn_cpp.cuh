#pragma once
// _cpp REFERENCE -- host transcription of rhoSimpleFoam's momentum predictor.
//
// provenance:
//   openfoam:
//     file: applications/solvers/compressible/rhoSimpleFoam/UEqn.H:1-21
//     also: src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress.C:107-117
//           (divDevRhoReff -- the COMPRESSIBLE overload, taking rho)
//           src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C  (DDt(rho, U) == rho*DDt(U))
//   brae:
//     reference: src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu
//     cuda:      (pending -- the whole case runs in _cpp first, see PORT.md)
//     tests:     tests/test_rho_ueqn_cpp.cu
//
// OpenFOAM, verbatim:
//
//     MRF.correctBoundaryVelocity(U);
//
//     tmp<fvVectorMatrix> tUEqn
//     (
//         fvm::div(phi, U)
//       + MRF.DDt(rho, U)
//       + turbulence->divDevRhoReff(U)
//      ==
//         fvOptions(rho, U)
//     );
//     fvVectorMatrix& UEqn = tUEqn.ref();
//
//     UEqn.relax();
//
//     fvOptions.constrain(UEqn);
//
//     solve(UEqn == -fvc::grad(p));
//
//     fvOptions.correct(U);
//
// THE ONE THING THAT MAKES THIS DIFFERENT FROM simpleFoam's UEqn, and it is not the notation.
//
// linearViscousStress.C defines ONE operator and OpenFOAM reaches it by two routes:
//
//     divDevRhoReff(U)      = -fvc::div((alpha*rho*nuEff)*dev2(T(grad U))) - fvm::laplacian(alpha*rho*nuEff, U)
//     divDevReff(U)         the INCOMPRESSIBLE lineage, where alpha == 1 and rho == 1
//
// so the compressible momentum equation carries the DYNAMIC viscosity mu_eff = rho*nu_eff where the
// incompressible one carries the KINEMATIC nu_eff. Everything else about the expression is identical.
// That single factor is the whole difference, and it is a factor of rho -- order 1 for air at ambient
// conditions and order 10 across a compressible duct, so a port that reuses the incompressible form is
// wrong by a factor that LOOKS plausible and varies with the solution. It cannot be caught by inspecting
// a converged field; it has to be gated against OpenFOAM's own matrix, which is what
// tests/rho_ueqn_vs_openfoam.sh does.
//
// The same factor applies to `phi`. Here phi is the MASS flux (kg/s) from compressibleCreatePhi.H, not
// the volumetric flux, so fvm::div(phi,U) is already rho-weighted and needs no change -- the operator is
// the same, the field it is given is not. Passing a volumetric phi would be the same class of error.
//
// MRF.DDt(rho, U) is MRFZoneList::DDt(rho,U) == rho*DDt(U): the Coriolis acceleration, rho-weighted.
// Refused rather than approximated when a case declares MRF, for the reason in the incompressible
// reference: brae has shipped a solver that read MRFProperties, ignored it, converged, and reported
// nothing wrong.
//
// ORDER IS OpenFOAM's:
//   1. assemble div + MRF + divDevRhoReff   (divDevRhoReff contributes to BOTH matrix and source)
//   2. relax                                (fvMatrix::relax is asymmetric)
//   3. add -fvc::grad(p)                    (an explicit source, added AFTER relaxation)
// Relaxing after adding grad(p) would relax the pressure gradient too, which OpenFOAM does not do:
// `solve(UEqn == -fvc::grad(p))` builds a new equation from the ALREADY-RELAXED UEqn.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvOptions_cpp.cuh"
#include "MRF_cpp.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvc.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace rhoSimple {

// The div(phi,U) scheme, as named by fvSchemes. Same set as the incompressible reference; listed here
// rather than shared so that adding one to this solver is a decision taken against rhoSimpleFoam's own
// tutorials rather than inherited silently.
enum class DivScheme
{
    upwind,
    linearUpwind,
    limitedLinear,
    limitedLinearV,
    LUST,
    linearUpwindV
};

struct RhoMomentumInput
{
    // MASS flux, kg/s (compressibleCreatePhi.H). NOT the volumetric flux.
    const std::vector<scalar>*              phi      = nullptr;
    const std::vector<std::vector<scalar>>* phiBnd   = nullptr;

    // rho and the KINEMATIC nuEff, kept separate rather than pre-multiplied, because the caller has both
    // and the multiplication is part of what this component is responsible for getting right. Passing a
    // pre-made muEff would move the one decision that distinguishes this solver out of the file that
    // documents it.
    const std::vector<scalar>*              rho      = nullptr;   // cells
    const std::vector<std::vector<scalar>>* rhoBnd   = nullptr;   // [patch][face]
    const std::vector<scalar>*              nuEff    = nullptr;   // cells
    const std::vector<std::vector<scalar>>* nuEffBnd = nullptr;   // [patch][face]

    // ALTERNATIVE to rho/nuEff: the dynamic viscosity directly. When set, these are used verbatim and
    // rho/nuEff are ignored. This exists for the gate, which injects OpenFOAM's own turbulence->muEff()
    // so the ASSEMBLY can be measured without a ported compressible turbulence model in the way.
    const std::vector<scalar>*              muEff    = nullptr;   // cells
    const std::vector<std::vector<scalar>>* muEffBnd = nullptr;   // [patch][face]

    // MRF.DDt(rho, U), UEqn.H:8. Null with hasMRF set is a REFUSAL, not a no-op.
    const std::vector<MRF::Zone>*           mrf      = nullptr;

    scalar    relaxU             = 1.0;
    // Whether fvSolution NAMES a relaxation factor for U. Not the same question as relaxU < 1:
    // fvMatrix::relax(1.0) still applies the dominance clamp, so a case naming 1 relaxes and a case
    // naming nothing does not. Mirrors PressureInput::relaxPSpecified.
    bool      relaxEquationU     = false;
    bool      bounded            = false;
    bool      linearUpwind       = false;
    scalar    gradULimitK        = 0.0;
    // U's PATCH VALUES AS THEY STOOD BEFORE this iteration's updateCoeffs equivalents. The only consumer
    // is the div(phi,U) limiter's gradient: gaussConvectionScheme::fvmDiv takes
    // tinterpScheme_().weights(vf) (.C:84) and only THEN builds the fvMatrix (:89), whose constructor
    // calls updateCoeffs (fvMatrix.C:396) -- so LimitedScheme::calcLimiter's fvc::grad(lPhi) reads the
    // boundary the PREVIOUS iteration left, while divDevRhoReff, internalCoeffs and boundaryCoeffs all
    // legitimately read the refreshed one. Measured on angledDuctExplicitFixedCoeff restarted from
    // OpenFOAM's iteration 5: that inlet is 46.9869516954335 before and 46.7633991650534 after, and
    // feeding the limiter the refreshed value puts the momentum matrix 1.300e-03 off OpenFOAM's own
    // against 4.405e-13 with the stored one, the Gauss upwind control unmoved at 2.0e-15.
    const std::vector<std::vector<vector>>* UPreUpdateBnd = nullptr;
    // grad(magSqr(U))'s OWN gradSchemes entry. `Gauss limitedLinear` on div(phi,U) is instantiated for a
    // vector as NVDTVD + limitFuncs::magSqr (LimitedScheme.H:188-189), so LimitedScheme::calcLimiter's
    // lPhi is Foam::magSqr(phi) (LimitFuncs.C:34-39), a volScalarField NAMED `magSqr(U)` -- and
    // fvc::grad(lPhi) resolves gradSchemes under the key `grad(magSqr(U))`, which almost always falls to
    // `default`. This took a hardcoded Gauss gradient whatever the case said; gasMixing/injectorPipe's
    // default is leastSquares, and the momentum matrix read 4.63e-02 off OpenFOAM's own there.
    bool      gradMagSqrULeastSq = false;
    scalar    gradMagSqrULimitK   = 0.0;
    // grad(U)'s and grad(p)'s BASE scheme: leastSquares, or Gauss linear. The momentum's three grad(U)
    // consumers (the limitedLinearV weights, linearUpwind's and linearUpwindV's corrections) and
    // -fvc::grad(p) resolve through the case's gradSchemes in OpenFOAM (fvcGrad.C:149).
    bool      gradULeastSq       = false;
    bool      gradPLeastSq       = false;
    // The gradient linearUpwind/linearUpwindV NAMES in div(phi,U), for the convection correction only.
    // -1 = not resolved by the caller; the correction then takes gradULimitK (see StepInput).
    scalar    gradULULimitK      = -1.0;
    DivScheme scheme             = DivScheme::upwind;
    scalar    schemeCoeff        = 1.0;
    bool      correctedLaplacian = false;
    scalar    snGradLimitCoeff   = 0.0;
    bool      hasMRF             = false;   // declared by the case -> must refuse until ported
    bool      hasFvOptions       = false;   // an UNIMPLEMENTED option -> must refuse
    // WHICH option, so the refusal names it. "Refuse rather than silently substitute" is only useful if
    // the message says what to implement next; a generic "the case declares fvOptions" sends the reader
    // back to the dictionary to work it out.
    std::string fvOptionUnsupported;
    // The IMPLEMENTED ones. UEqn.H applies fvOptions(rho, U), and for explicitPorositySource that is
    // eqn -= porosityEqn with the resistance built by the porosity model. Null = no options to apply.
    const cpu::fvOptions::OptionList* fvOpts = nullptr;
    // The LAMINAR dynamic viscosity per cell, for DarcyForchheimer's Darcy term -- OF resolves
    // "thermo:mu" (DarcyForchheimer.C:64), NOT the effective viscosity. Required whenever fvOpts
    // carries a porosity that is not fixedCoeff; addSup refuses without it.
    const std::vector<scalar>*        muLaminar = nullptr;
};

// mu_eff = rho*nu_eff, cells and boundary. Exposed because pEqn.H needs the same product and two
// derivations of it is two chances to disagree.
std::vector<scalar> dynamicViscosity(
    const std::vector<scalar>& rho,
    const std::vector<scalar>& nuEff);

std::vector<std::vector<scalar>> dynamicViscosityBoundary(
    const std::vector<std::vector<scalar>>& rhoBnd,
    const std::vector<std::vector<scalar>>& nuEffBnd);

// Steps 1+2 of UEqn.H: fvm::div(phi,U) + MRF.DDt(rho,U) + turbulence->divDevRhoReff(U), then UEqn.relax().
//
// Returns the relaxed momentum matrix WITHOUT the pressure gradient, because that is the object pEqn.H
// needs: rAU = 1/UEqn.A() and UEqn.H() are both taken from the relaxed matrix before -grad(p) is applied.
FvVectorMatrix assembleUEqn(
    const GeometricField<vector>& U,
    const RhoMomentumInput&       in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// Step 3: the right-hand side of `solve(UEqn == -fvc::grad(p))`, added to source as -grad(p)*V after
// relaxation. p here is the ABSOLUTE pressure the thermo owns, not a kinematic p/rho.
void addPressureGradient(
    FvVectorMatrix&               UEqn,
    const GeometricField<scalar>& p,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    bool                          leastSquares = false);

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
