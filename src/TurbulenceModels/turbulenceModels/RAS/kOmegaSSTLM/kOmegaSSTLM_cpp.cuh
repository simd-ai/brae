#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's kOmegaSSTLM (Langtry-Menter gamma-ReThetat).
//
// provenance:
//   openfoam:
//     file: src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM.C
//     symbols: F1 / Pk / epsilonByk / Fthetat / ReThetac / Flength / ReThetat0 / Fonset /
//              correctReThetatGammaInt / correct
//   brae:
//     reference: src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM_cpp.cu
//     tests:     tests/test_simple_lm_cpp.cu, tests/lm_cpp_vs_openfoam.sh
//
// kOmegaSSTLM IS kOmegaSST plus two more transport equations and three overrides. The overrides live in
// kOmegaSST_cpp (LMHooks) so the base model stays one model; everything here is the part that is new.
//
// EVERY EXPRESSION BELOW IS TRANSCRIBED, NOT RE-DERIVED. In OpenFOAM's own notation, all on INTERNAL
// fields (the model works on volScalarField::Internal throughout):
//
//   Omega  = sqrt(2*magSqr(skew(gradU)))            vorticity magnitude
//   S      = sqrt(2*magSqr(symm(gradU)))            strain magnitude
//   Us     = max(mag(U), deltaU),  deltaU = SMALL = 1e-15 (OpenFOAM doubleScalarSMALL,
//                                  NOT VSMALL 1e-300 -- the two differ by 285 orders and
//                                  Us appears squared in a denominator)
//   dUsds  = (U & (U & gradU))/sqr(Us)              streamwise acceleration
//
//   Fthetat = min(max(Fwake*exp(-(y/delta)^4),
//                     1 - ((gammaInt - 1/ce2)/(1 - 1/ce2))^2), 1)
//             delta  = max(375*Omega*nu*ReThetat*y/sqr(Us), SMALL)
//             ReOmega = sqr(y)*omega/nu,  Fwake = exp(-(ReOmega/1e5)^2)
//
//   ReThetat equation:   ddt + div(phi,ReThetat) - laplacian(DReThetatEff, ReThetat)
//                        == Pthetat*ReThetat0 - Sp(Pthetat, ReThetat)
//             t       = 500*nu/sqr(Us),  Pthetat = (cThetat/t)*(1 - Fthetat)
//             DReThetatEff = sigmaThetat*(nut + nu)
//             then bound(ReThetat, 0)
//
//   ReThetac = ReThetat <= 1870
//              ? ReThetat - 396.035e-2 + 120.656e-4*ReThetat - 868.230e-6*ReThetat^2
//                          + 696.506e-9*ReThetat^3 - 174.105e-12*ReThetat^4
//              : ReThetat - 593.11 - 0.482*(ReThetat - 1870)
//
//   Rev  = sqr(y)*S/nu,   RT = k/(nu*omega)
//   Fonset1 = Rev/(2.193*ReThetac)
//   Fonset2 = min(max(Fonset1, Fonset1^4), 2)
//   Fonset3 = max(1 - (RT/2.5)^3, 0)
//   Fonset  = max(Fonset2 - Fonset3, 0)
//   Fturb   = exp(-(0.25*RT)^4)
//
//   gammaInt equation:   ddt + div(phi,gammaInt) - laplacian(DgammaIntEff, gammaInt)
//                        == Pgamma - Sp(ce1*Pgamma, gammaInt) + Egamma - Sp(ce2*Egamma, gammaInt)
//             Pgamma = ca1*Flength*S*sqrt(gammaInt*Fonset)
//             Egamma = ca2*Omega*Fturb*gammaInt
//             DgammaIntEff = nut + nu        <-- NOT nut/sigmaGamma; there is no sigmaGamma in this model
//             then bound(gammaInt, 0)
//
//   Freattach = exp(-(RT/20)^4)
//   gammaSep  = min(2*max(Rev/(3.235*ReThetac) - 1, 0)*Freattach, 2)*Fthetat
//   gammaIntEff = max(gammaInt, gammaSep)
//
// THE ORDER MATTERS AND IS NOT OBVIOUS. correct() runs kOmegaSST::correct() FIRST, so k and omega are
// advanced using the gammaIntEff computed at the END of the PREVIOUS outer iteration -- the transition
// state is lagged by one iteration, exactly as OpenFOAM lags it. Computing gammaIntEff first would be a
// different (and more implicit) scheme.
//
// EQUALLY NON-OBVIOUS: ReThetat is solved BEFORE ReThetac/Rev/RT are formed, so ReThetac sees the NEW
// ReThetat; and gammaInt is solved before gammaSep is formed, so gammaSep sees the NEW gammaInt. But
// Fthetat is computed ONCE, before either solve, from the OLD ReThetat and the OLD gammaInt, and the
// same value is reused in gammaSep at the end.
//
// ReThetat0 IS A FIXED-POINT ITERATION PER CELL, not a closed form: lambda starts at 0 and is iterated
// against the momentum-thickness correlation until |lambda - lambda0| <= lambdaErr (1e-6). OpenFOAM
// warns past maxLambdaIter (10) but does NOT stop; the loop is `do { } while (lambdaErr > lambdaErr_)`
// with no iteration cap at all. That is transcribed as written, warning included -- capping it would be
// a different algorithm on exactly the cells where the correlation is hardest.
//
// REFUSED, not ignored: MRF and fvOptions on the two transition equations.
#include "cf_types.cuh"
#include "komega_sst_coeffs.cuh"
#include "kOmegaSST_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"

#include <vector>

namespace brae {
namespace cpu {
namespace kOmegaSSTLM {

// kOmegaSSTLM.C's coeffDict defaults. sigmaThetat is the ReThetat diffusivity multiplier; there is no
// corresponding coefficient on gammaInt -- DgammaIntEff is nut + nu unscaled.
struct Coeffs
{
    scalar ca1         = 2.0;
    scalar ca2         = 0.06;
    scalar ce1         = 1.0;
    scalar ce2         = 50.0;
    scalar cThetat     = 0.03;
    scalar sigmaThetat = 2.0;
    scalar lambdaErr   = 1e-6;
    int    maxLambdaIter = 10;
};

void readCoeffs(const void* rasDict, Coeffs& co);   // rasDict is a const FoamDict*; void to keep the header light

struct Residuals { scalar ReThetat = 0, gammaInt = 0; };

// OF ReThetac(): the transition-onset momentum-thickness Reynolds number correlation. Per cell, from
// ReThetat alone.
std::vector<scalar> ReThetac(const std::vector<scalar>& ReThetat);

// OF Flength(): the transition-length correlation, blended into 40 inside the viscous sublayer.
std::vector<scalar> Flength(const std::vector<scalar>& ReThetat,
                            const std::vector<scalar>& omega,
                            const std::vector<scalar>& y,
                            scalar                     nu);

// OF ReThetat0(): the per-cell lambda fixed point. `maxIterUsed`, when non-null, receives the worst
// iteration count so a caller can reproduce OpenFOAM's maxLambdaIter warning.
std::vector<scalar> ReThetat0(const std::vector<scalar>& Us,
                              const std::vector<scalar>& dUsds,
                              const std::vector<scalar>& k,
                              scalar                     nu,
                              const Coeffs&              co,
                              int*                       maxIterUsed = nullptr);

// OF Fthetat().
std::vector<scalar> Fthetat(const std::vector<scalar>& Us,
                            const std::vector<scalar>& Omega,
                            const std::vector<scalar>& omega,
                            const std::vector<scalar>& y,
                            const std::vector<scalar>& ReThetat,
                            const std::vector<scalar>& gammaInt,
                            scalar                     nu,
                            const Coeffs&              co);

// The three CELL-LOCAL stages, exposed at exactly the boundaries the device kernels use
// (lmReThetatPrepKernel / lmGammaPrepKernel / lmGammaEffKernel) so the CUDA port can be compared one
// module at a time rather than as a single fused answer. Each takes the same inputs its kernel does and
// returns the same outputs, so a disagreement names the stage instead of the model.
//
// The reaction convention matches the device's lmAddReactionKernel and OpenFOAM's own:
//     diag += V*sp     source += V*su      i.e. the RHS is  su - Sp(sp, psi)
struct StrainState
{
    std::vector<scalar> S, Omega, Us, dUsds;
};
StrainState strain(const std::vector<tensor>& gradU,
                   const std::vector<vector>& U,
                   scalar                     deltaU);

struct ReThetatPrep
{
    std::vector<scalar> Fthetat, sp, su;
};
ReThetatPrep reThetatPrep(const StrainState&         st,
                          const std::vector<scalar>& k,
                          const std::vector<scalar>& omega,
                          const std::vector<scalar>& y,
                          const std::vector<scalar>& ReThetat,
                          const std::vector<scalar>& gammaInt,
                          scalar                     nu,
                          const Coeffs&              co);

struct GammaPrep
{
    std::vector<scalar> sp, su;
};
GammaPrep gammaPrep(const StrainState&         st,
                    const std::vector<scalar>& k,
                    const std::vector<scalar>& omega,
                    const std::vector<scalar>& y,
                    const std::vector<scalar>& ReThetat,   // the NEW ReThetat, post-solve
                    const std::vector<scalar>& gammaInt,
                    scalar                     nu,
                    const Coeffs&              co);

// gammaIntEff = max(gammaInt, gammaSep), from the NEW ReThetat and NEW gammaInt but the LAGGED Fthetat.
std::vector<scalar> gammaEff(const StrainState&         st,
                             const std::vector<scalar>& k,
                             const std::vector<scalar>& omega,
                             const std::vector<scalar>& y,
                             const std::vector<scalar>& ReThetat,
                             const std::vector<scalar>& gammaInt,
                             const std::vector<scalar>& Fthetat,
                             scalar                     nu);

// OF correctReThetatGammaInt(): the two transport equations and gammaIntEff. Updates ReThetat, gammaInt
// and gammaIntEff in place.
void correctReThetatGammaInt(
    const GeometricField<vector>&  U,
    const GeometricField<scalar>&  k,
    const GeometricField<scalar>&  omega,
    const GeometricField<scalar>&  nutField,
    GeometricField<scalar>&        ReThetat,
    GeometricField<scalar>&        gammaInt,
    std::vector<scalar>&           gammaIntEff,
    const SurfaceScalarField&      phi,
    const std::vector<scalar>&     y,          // CELL wall distance
    scalar                         nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    scalar                         relaxReThetat,
    scalar                         relaxGammaInt,
    scalar                         tol,
    scalar                         relTol,
    int                            maxIter,
    const Coeffs&                  co,
    Residuals*                     res = nullptr,
    bool                           bounded = false,
    bool                           limitedLinear = false,
    bool                           linearUpwind = false,
    scalar                         limiterCoeff = 1.0,
    // The case's laplacianScheme, applied to the two transition equations as OpenFOAM applies it to
    // every laplacian in the case.
    bool                           correctedLaplacian = false,
    scalar                         snGradLimitCoeff = 0.0);

// One kOmegaSSTLM::correct(): kOmegaSST::correct with the LM overrides, then correctReThetatGammaInt.
void correct(
    const GeometricField<vector>&  U,
    GeometricField<scalar>&        k,
    GeometricField<scalar>&        omega,
    GeometricField<scalar>&        nutField,
    GeometricField<scalar>&        ReThetat,
    GeometricField<scalar>&        gammaInt,
    std::vector<scalar>&           gammaIntEff,
    const SurfaceScalarField&      phi,
    const std::vector<scalar>&     y,
    scalar                         nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    scalar                         relaxOmega,
    scalar                         relaxK,
    scalar                         tol,
    scalar                         relTol,
    int                            maxIter,
    const KOmegaSSTCoeffs&         sstCo,
    const Coeffs&                  co,
    kOmegaSST::SSTResiduals*       sstRes = nullptr,
    Residuals*                     lmRes  = nullptr,
    bool                           bounded = false,
    bool                           limitedLinear = false,
    bool                           linearUpwind = false,
    scalar                         limiterCoeff = 1.0,
    bool                           correctedLaplacian = false,
    scalar                         snGradLimitCoeff = 0.0);

} // namespace kOmegaSSTLM
} // namespace cpu
} // namespace brae
