#pragma once
// interFoam's alpha equation -- the FLUX ASSEMBLY, the host reference.
//
// provenance:
//   openfoam:
//     file: applications/solvers/multiphase/VoF/alphaEqn.H
//     also: applications/solvers/multiphase/interFoam/alphaControls.H  (the dictionary keys)
//           applications/solvers/multiphase/interFoam/alphaSuSp.H      (Su, Sp, divU -- see below)
//           src/finiteVolume/finiteVolume/fvc/fvcFluxTemplates.C:80-90 (fvc::flux(phi, vf, name))
//           src/finiteVolume/finiteVolume/convectionSchemes/gaussConvectionScheme/
//             gaussConvectionScheme.C:64-73  (flux(faceFlux, vf) == faceFlux*interpolate(vf))
//           src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/vanLeer/vanLeer.H:85
//   brae:
//     reference: this header
//     cuda:      (pending)
//     tests:     tests/test_alpha_eqn_cpp.cu
//
// WHAT IS IN THIS FILE AND WHAT IS NOT. alphaEqn.H is two things bolted together: the assembly of the
// advective and compressive fluxes, and MULES, which limits them. MULES is an explicit iterative
// bound-preserving limiter -- not a matrix assembly -- and it needs an instrumented OpenFOAM for its
// oracle, so the manifest budgets it as its own stage. THIS FILE IS EVERYTHING ELSE: the controls, the
// off-centring, the compression flux, alphaPhiUn, and rhoPhi. It hands MULES a face flux and takes one
// back.
//
// Su, Sp AND divU ARE IDENTICALLY ZERO HERE. interFoam/alphaSuSp.H is three lines --
// `zeroField Su; zeroField Sp; zeroField divU;` -- so alphaEqn.H's source terms collapse for THIS
// solver. They are live in interPhaseChangeFoam, which is a different solver with its own alphaSuSp.H.
// Carrying them would be carrying machinery this port does not have, and the places they would enter
// are named at each site below rather than left implicit.
//
// FOUR THINGS WORTH NAMING, none of them arithmetic:
//
// 1. phic IS ZEROED ON EVERY NON-COUPLED BOUNDARY PATCH (alphaEqn.H:79-89). Interface compression is an
//    anti-diffusion term; applied at an inlet or an outlet it pulls interface INTO the domain through a
//    boundary that has no interface. OpenFOAM's comment is "Do not compress interface at non-coupled
//    boundary faces". A port that builds phic from the boundary flux and stops there runs a physically
//    different problem at every open boundary, and it looks like an inflow condition problem.
//
// 2. alphaPhiUn'S SECOND TERM IS A NESTED FLUX WITH TWO MINUS SIGNS (alphaEqn.H:171-176):
//
//        fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme), alpha1, alpharScheme)
//
//    fvc::flux(psi, vf, scheme) is psi*interpolate(vf) where the INTERPOLATION IS UPWINDED BY psi. So
//    alpha2 is interpolated against -phir and alpha1 against the result. Dropping either minus leaves a
//    flux of the same magnitude, upwinded from the wrong side -- which on a limited scheme differs only
//    at the interface, i.e. only where the answer is decided.
//
// 3. rhoPhi's TWO BRANCHES DIFFER IN WHICH FLUX MULTIPLIES rho2f (alphaEqn.H:248 vs 260): phiCN on the
//    Euler/localEuler branch, phi on the other. With ocCoeff == 0 the two are the same field, so on
//    every Euler case -- 42 of the 44 shipped tutorials -- the difference is unobservable. It is exactly
//    the one CrankNicolson case that would show it.
//
// 4. rho1f - rho2f IS AN ORDER. rhoPhi = alphaPhi10*(rho1f - rho2f) + phi*rho2f is a linear blend that
//    reduces to phi*rho1f where alpha1 is 1 and phi*rho2f where it is 0. Swapped, it still has the right
//    dimensions and the right magnitude and inverts the mixture.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "interface_properties_cpp.cuh"
#include "mules_cpp.cuh"
#include "inter_solve_record.cuh"
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

// fvSolution's solvers/<alpha1> entry. nAlphaCorr and nAlphaSubCycles are get<label> in
// alphaControls.H -- NO default, so a case omitting either is a FatalError in OpenFOAM and a refusal
// here. The rest are getOrDefault.
struct AlphaControls
{
    label  nAlphaCorr         = 0;       // MANDATORY
    label  nAlphaSubCycles    = 0;       // MANDATORY
    bool   MULESCorr          = false;
    bool   alphaApplyPrevCorr = false;
    scalar icAlpha            = 0;       // isotropic compression; 0 in every shipped tutorial
    scalar scAlpha            = 0;       // shear compression;     absent in every shipped tutorial
};

AlphaControls readAlphaControls(const FoamDict& fvSolution, const std::string& alphaFieldName);

// ddtSchemes `ddt(alpha)`. alphaEqn.H:18-52 accepts EXACTLY Euler, localEuler and CrankNicolson, and
// FatalErrors on anything else -- so `backward`, which the rest of brae supports, is refused here by
// OpenFOAM itself rather than by a limitation of this port.
enum class AlphaDdt { Euler, localEuler, CrankNicolson, other };

// The off-centring coefficient, alphaEqn.H:6-53.
//
//   Euler / localEuler                          -> 0
//   CrankNicolson, first step of a cold start   -> 0   (the scheme has no old-time ddt to off-centre)
//   CrankNicolson, thereafter                   -> the scheme's own ocCoeff
//   CrankNicolson with nAlphaSubCycles > 1      -> FatalError
//
// `warmedUp` is OpenFOAM's `alphaRestart || timeIndex > startTimeIndex + 1`, decided by the caller
// because it is a property of the run, not of the schemes.
scalar offCentringCoeff(AlphaDdt scheme,
                        label    nAlphaSubCycles,
                        scalar   schemeOcCoeff,
                        bool     warmedUp);

// cnCoeff = 1/(1 + ocCoeff), alphaEqn.H:56. 1 for Euler.
inline scalar blendingCoeff(scalar ocCoeff) { return scalar(1) / (scalar(1) + ocCoeff); }

// phic, alphaEqn.H:59-89. The optional isotropic and shear contributions are included because they are
// cheap and because leaving them out would mean silently ignoring icAlpha/scAlpha on a case that sets
// them -- but note that NO shipped interFoam tutorial sets either to a non-zero value, so nothing in
// the tutorial set exercises them. `magU` and `shear` may be empty when the corresponding coefficient
// is zero; they are required when it is not.
//
// THE BOUNDARY IS ZEROED on every non-coupled patch, and LEFT ALONE on a coupled one (FvPatch::coupled),
// where the interface does pass through: RAS/damBreakPorousBaffle's cyclic baffle, with the free surface
// staged across it, reads alpha 7.9e-03 out when the pair is zeroed like a wall
// (tests/interfoam_baffle_vs_openfoam.sh).
void compressionFlux(scalar                       cAlpha,
                     const SurfaceScalarField&    phi,
                     const std::vector<scalar>&   magSf,          // the mesh's full face array
                     const std::vector<FvPatch>&  patches,
                     scalar                       icAlpha,
                     const std::vector<scalar>&   magUf,          // internal; icAlpha > 0 only
                     scalar                       scAlpha,
                     const std::vector<scalar>&   shearf,         // internal; scAlpha > 0 only
                     SurfaceScalarField&          phic);

// phiCN, alphaEqn.H:91-97. Returns phi itself when ocCoeff == 0, which is what OpenFOAM's tmp does.
void offCentredFlux(const SurfaceScalarField& phi,
                    const SurfaceScalarField& phiOld,
                    scalar                    cnCoeff,
                    scalar                    ocCoeff,
                    SurfaceScalarField&       phiCN);

// The interpolation scheme named by a divSchemes entry for the alpha fluxes. `Gauss vanLeer` is
// div(phi,alpha) in all 42 shipped tutorials; div(phirb,alpha) is `Gauss linear` in 31, `Gauss vanLeer`
// in 7 and `Gauss interfaceCompression` in 4. The last is a separate scheme, not a variant: a PhiScheme
// whose limiter reads the two cell values alone (limitedSchemes_cpp.cuh, interfaceCompressionWeights).
enum class AlphaFluxScheme { vanLeer, linear, upwind, interfaceCompression };

// fvc::flux(psi, vf, scheme) == psi*interpolate(vf), with the interpolation UPWINDED BY psi
// (gaussConvectionScheme.C:64-73). `psi` is the flux that both scales the result and picks the upwind
// side, which is why the two are one argument and not two.
void fluxWithScheme(const SurfaceScalarField&     psi,
                    const GeometricField<scalar>& vf,
                    AlphaFluxScheme               scheme,
                    const PrimitiveMesh&          m,
                    const FvGeometry&             g,
                    const std::vector<FvPatch>&   patches,
                    SurfaceScalarField&           out,
                    // the limiter's fvc::grad(vf): gradSchemes' `grad(<vf's name>)`, then `default`
                    const GradChoice&             gradVf = GradChoice{});

// alphaPhiUn, alphaEqn.H:164-176 -- the advective flux plus the compressive one. See note 2.
void alphaPhiUn(const SurfaceScalarField&     phi,
                const SurfaceScalarField&     phir,
                const GeometricField<scalar>& alpha1,
                const GeometricField<scalar>& alpha2,
                AlphaFluxScheme               alphaScheme,
                AlphaFluxScheme               alpharScheme,
                const PrimitiveMesh&          m,
                const FvGeometry&             g,
                const std::vector<FvPatch>&   patches,
                SurfaceScalarField&           out,
                // the limiters' gradients: grad(alpha.<phase1>) and grad(alpha.<phase2>)
                const GradChoice&             gradAlpha1 = GradChoice{},
                const GradChoice&             gradAlpha2 = GradChoice{});

// rhoPhi = alphaPhi10*(rho1f - rho2f) + phiForRho2*rho2f, alphaEqn.H:248 / :260. `phiForRho2` is phiCN
// on the Euler/localEuler branch and phi on the other -- see note 3; the caller picks, and this refuses
// to guess.
void massFlux(const SurfaceScalarField& alphaPhi10,
              const SurfaceScalarField& phiForRho2,
              scalar                    rho1,
              scalar                    rho2,
              SurfaceScalarField&       rhoPhi);

// ---------------------------------------------------------------------------------------------------
// THE WHOLE alphaEqn.H, assembled from the pieces above plus MULES. This is what alphaEqnSubCycle
// calls once per sub-step.
//
//   for (aCorr = 0; aCorr < nAlphaCorr; ++aCorr)
//   {
//       phir       = phic*mixture.nHatf();
//       alphaPhiUn = fvc::flux(phi, alpha1, alphaScheme)
//                  + fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme), alpha1, alpharScheme);
//       ... MULES ...
//       alpha2 = 1 - alpha1;
//       mixture.correct();
//   }
//   rhoPhi = alphaPhi10*(rho1 - rho2) + phiCN*rho2;
//
// TWO THINGS THE LOOP DOES THAT ARE EASY TO HOIST OUT OF IT AND WRONG TO:
//
//   * mixture.correct() INSIDE the corrector loop recomputes nHatf from the alpha MULES has just
//     produced, so phir is DIFFERENT on the second corrector. Hoisting the curvature out of the loop
//     makes every corrector compress the interface towards where it was at the start of the step --
//     which converges, to a slightly stale interface, and nAlphaCorr is 2 or 3 in 11 of the 44
//     shipped tutorials.
//   * alpha1 is updated IN PLACE by each corrector. The correctors are a sequence, not an average.
struct AlphaStepInput
{
    // OUT, optional: alpha2's patch values as `alpha2 = 1.0 - alpha1` leaves them (alphaEqn.H:223),
    // which is BEFORE the corrector's mixture.correct() moves alpha1's. See InterFields::alpha2Bnd.
    std::vector<std::vector<scalar>>* alpha2BndOut = nullptr;

    const SurfaceScalarField* phi      = nullptr;   // the volumetric flux
    const SurfaceScalarField* phiCN    = nullptr;   // off-centred; == phi for Euler

    scalar  cAlpha   = 0;
    label   nAlphaCorr = 1;
    scalar  icAlpha  = 0;
    scalar  scAlpha  = 0;
    scalar  rho1 = 0, rho2 = 0;
    scalar  deltaT = 0;
    AlphaFluxScheme alphaScheme  = AlphaFluxScheme::vanLeer;
    AlphaFluxScheme alpharScheme = AlphaFluxScheme::linear;

    // MULES
    bool    MULESCorr = false;
    // alphaApplyPrevCorr: reuse the PREVIOUS time step's compression flux as a first guess. 3 of the
    // 44 shipped tutorials set it. `prevCorr` carries that cached flux in and out; empty means there
    // is none yet, which is what the first step and a topology change both look like.
    bool    alphaApplyPrevCorr = false;
    // the case's own linear-solver controls for the implicit upwind pre-solve (MULESCorr only)
    scalar  tolAlpha = 1e-8, relTolAlpha = 0;
    int     maxIterAlpha = 1000;
    // `minIter`: the sweeps the solve takes even when its initial residual is already under tolerance
    // (smoothSolver.C: `if (minIter_ > 0 || !converged)`)
    int     minIterAlpha = 0;
    // the limiters' gradSchemes entries, grad(alpha.<phase1>) and grad(alpha.<phase2>)
    GradChoice gradAlpha1;
    GradChoice gradAlpha2;
    // `solver smoothSolver; smoother symGaussSeidel|GaussSeidel;` -- run brae::smoothSolver, which is
    // OpenFOAM's, in place of the DILU-PBiCGStab this step grew up on. See smooth_solver_cpp.cuh.
    bool smoothSolver = false;
    bool symmetric = true;
    int nSweeps = 1;
    // appended to, one record per pre-solve; null = not kept
    std::vector<LinearSolveRecord>* solveLog = nullptr;
    // A GATE'S CONTROL, never set by a solver: hand the previous-correction limiter the VOLUMETRIC flux
    // phiCN as its outlet-test argument, which is what this code did before it was measured, in place
    // of the alpha flux OpenFOAM passes. tests/interfoam_dambreak_vs_openfoam.sh's `outflow` profile
    // runs it to show the gate fails when that argument is wrong: 5.5e-03 of alpha against 4.6e-13.
    bool controlPrevCorrOutletOnPhiCN = false;

    // A MOVING MESH: mesh.Vsc() and mesh.Vsc0() at this (sub-)step -- see MULES::Fields. Null on a
    // mesh that does not move. The pre-solve's fvm::ddt takes Vsc on the diagonal and Vsc0 in the
    // source (EulerDdtScheme.C:383-392), and both go to MULES.
    const std::vector<scalar>* Vsc = nullptr;
    const std::vector<scalar>* Vsc0 = nullptr;

    // alpha1's boundaryField().updateCoeffs(), for the conditions whose value a MODEL supplies --
    // waveAlpha. Called where OpenFOAM's first one of the pass fires: at the pre-solve's matrix
    // construction under MULESCorr, and otherwise at the correctBoundaryConditions() that OPENS
    // MULES::explicitSolve (MULESTemplates.C:168) -- AFTER the high-order flux has been built on the
    // values the last update left, and BEFORE the limiter builds its bounded flux on these. Null on a
    // case with no such patch.
    std::function<void()> updateModelledBoundary;
    // Called once per call, after phic is formed and before anything else: where OpenFOAM's first
    // interpolation across a cyclicACMI lands in the step, and so where its lazy rescale does (the
    // driver makes it a no-op after the step's first).
    std::function<void()> geometryUpdate;
};

// One alphaEqn.H. `alpha1` carries the field AND its boundary conditions and is advanced in place;
// `alpha1Old` is the sub-step's starting value and is not written.
void alphaEqnStep(GeometricField<scalar>&                 alpha1,
                  const std::vector<scalar>&              alpha1Old,
                  const AlphaStepInput&                   in,
                  const interfaceProps::InterfaceCoeffs&  ic,
                  const MULES::Controls&                  mulesCtl,
                  const PrimitiveMesh&                    m,
                  const FvGeometry&                       g,
                  const std::vector<FvPatch>&             patches,
                  SurfaceScalarField&                     alphaPhi10,
                  SurfaceScalarField&                     rhoPhi,
                  // mixture.nHatf(): read at the TOP of each corrector and rewritten at its BOTTOM by
                  // mixture.correct(), exactly as alphaEqn.H:162 and :225 do. Passing it in and out
                  // rather than recomputing it here is what makes the number of calculateK passes
                  // match OpenFOAM's -- and the curvature is a fixed point in those passes.
                  SurfaceScalarField&                     nHatf,
                  std::vector<scalar>&                    K,
                  SurfaceScalarField*                     prevCorr = nullptr);

} // namespace interFoam
} // namespace cpu
} // namespace brae
