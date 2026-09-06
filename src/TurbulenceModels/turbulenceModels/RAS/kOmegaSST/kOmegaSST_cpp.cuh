#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's kOmegaSST.
//
// provenance:
//   openfoam:
//     file: src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSSTBase.C
//     symbols: kOmegaSSTBase::correct / F1 / F2 / F23 / GbyNu0 / GbyNu / Pk / epsilonByk / correctNut
//   brae:
//     reference: src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST_cpp.cu
//     cuda:      src/cuda/device_komega_sst.cu   (deviceKOmegaSSTCorrect)
//     tests:     tests/test_komegasst_cpp.cu, tests/komegasst_vs_openfoam.sh
//
// WHY A HOST REFERENCE AT ALL, when a validated CUDA kOmegaSST already exists: the same reason every other
// component of this port has one. The device model is a single fused entry point; a disagreement with
// OpenFOAM in it is one number covering production, the two blending functions, the eddy-viscosity
// limiter, two transport equations and two wall functions. The reference is stage-addressable, so a
// disagreement names the stage. kEpsilon was wired into the rebuilt solver WITHOUT one, which is a gap in
// this port and not a precedent to follow.
//
// EVERY EXPRESSION BELOW IS TRANSCRIBED, NOT RE-DERIVED. In OpenFOAM's own notation:
//
//   S2       = 2*magSqr(symm(gradU))                                      (kOmegaSSTBase.C:137)
//   GbyNu0   = gradU && devTwoSymm(gradU)                                 (:168-181)
//   G        = nut*GbyNu0                                                 (:525)
//   CDkOmega = (2*alphaOmega2)*(grad(k) & grad(omega))/omega              (:530)
//   F1       = tanh(pow4(arg1)),
//              arg1 = min(min(max((1/betaStar)*sqrt(k)/(omega*y),
//                                 500*nu/(sqr(y)*omega)),
//                             (4*alphaOmega2)*k/(max(CDkOmega,1e-10)*sqr(y))), 10)
//   F2       = tanh(sqr(arg2)),
//              arg2 = min(max((2/betaStar)*sqrt(k)/(omega*y), 500*nu/(sqr(y)*omega)), 100)
//   F23      = F2 (times F3 when the F3 switch is on; F3 is not ported and is refused)
//   blend    = F1*(psi1 - psi2) + psi2                                    (kOmegaSSTBase.H:blend)
//   GbyNu    = min(GbyNu0, (c1/a1)*betaStar*omega*max(a1*omega, b1*F23*sqrt(S2)))
//   Pk(G)    = min(G, (c1*betaStar)*k*omega)
//   epsilonByk = betaStar*omega                                           (:157-165)
//   nut      = a1*k/max(a1*omega, b1*F23*sqrt(S2))                        (:117-126)
//   DkEff    = alphaK(F1)*nut + nu,  DomegaEff = alphaOmega(F1)*nut + nu
//
// y IS THE CELL WALL DISTANCE, at every cell -- wallDist::New(mesh).y() -- NOT the near-wall face distance
// the wall functions use. OpenFOAM writes both as `y` and they are different fields; F1/F2 need the former.
//
// REFUSED, not ignored: the F3 near-wall switch, decayControl (with kInf/omegaInf), MRF and fvOptions.
// ofscan lists 17 keys on kOmegaSSTBase; the three this reference does not implement are exactly those.
#include "cf_types.cuh"
#include "komega_sst_coeffs.cuh"
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
namespace kOmegaSST {

// S2 = 2*magSqr(symm(gradU)).
std::vector<scalar> S2(const std::vector<tensor>& gradU);

// GbyNu0 = gradU && devTwoSymm(gradU). devTwoSymm(t) = (t + t^T) - (2/3)*tr(t)*I.
std::vector<scalar> GbyNu0(const std::vector<tensor>& gradU);

// CDkOmega = (2*alphaOmega2)*(grad(k) & grad(omega))/omega.
std::vector<scalar> CDkOmega(const std::vector<vector>& gradK,
                             const std::vector<vector>& gradOmega,
                             const std::vector<scalar>& omega,
                             const KOmegaSSTCoeffs& co);

// The two blending functions. `y` is the CELL wall distance; `nu` the laminar kinematic viscosity.
// PER-CELL nu: the compressible lineage's is mu(T)/rho, a field, and both blenders use it in their
// viscous cross-over term. The scalar forms below forward to these, so every existing caller keeps the
// constant it already passes rather than being silently rebound.
std::vector<scalar> F1(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                       const std::vector<scalar>& y, const std::vector<scalar>& CDkOmega,
                       const std::vector<scalar>& nuC, const KOmegaSSTCoeffs& co);
std::vector<scalar> F2(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                       const std::vector<scalar>& y, const std::vector<scalar>& nuC,
                       const KOmegaSSTCoeffs& co);

inline std::vector<scalar> F1(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                              const std::vector<scalar>& y, const std::vector<scalar>& CDkOmega,
                              scalar nu, const KOmegaSSTCoeffs& co)
{
    return F1(k, omega, y, CDkOmega, std::vector<scalar>(k.size(), nu), co);
}
inline std::vector<scalar> F2(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                              const std::vector<scalar>& y, scalar nu, const KOmegaSSTCoeffs& co)
{
    return F2(k, omega, y, std::vector<scalar>(k.size(), nu), co);
}

// nut = a1*k/max(a1*omega, b1*F23*sqrt(S2)).
// F1 ON THE BOUNDARY FACES, as the field expression behind DkEff(F1) = alphaK(F1)*nut + nu and
// DomegaEff(F1) evaluates it there (kOmegaSSTBase.H:342-358, F1 at kOmegaSSTBase.C): from the patch k,
// omega and nu, the owner cell's y (wallDist's y is zeroGradient on every non-wall patch,
// patchDistMethod::patchTypes) and CDkOmega's own boundary value -- the Gauss gradients' patch-corrected
// boundary values, gb = gc + n*(snGrad - n&gc). On a wall y is 0, every term of arg1 is unbounded and
// min(., 10) makes F1 exactly 1, which is what OpenFOAM's field carries there. The diffusivities used to
// blend the OWNER CELL's F1 on every patch face: on rhoSST's inlet OpenFOAM's patch F1 is 0.5220 against
// the cells' 0.5291 (the inlet's fixed k and omega evaluate a different blend), and that 1% of the blend
// in the laplacian's boundary coefficient was the seed of a residual that kOmegaSST alone showed from
// iteration 2 -- the first iteration at which k and omega differ between the patch and its cell.
std::vector<std::vector<scalar>> F1Boundary(
    const GeometricField<scalar>&           k,
    const GeometricField<scalar>&           omega,
    const std::vector<scalar>&              y,
    const std::vector<vector>&              gradK,
    const std::vector<vector>&              gradOmega,
    scalar                                  nu,
    const std::vector<std::vector<scalar>>* nuBnd,
    const std::vector<FvPatch>&             patches,
    const KOmegaSSTCoeffs&                  co);

std::vector<scalar> correctNut(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                               const std::vector<scalar>& F23, const std::vector<scalar>& S2,
                               const KOmegaSSTCoeffs& co);

// The initial residuals of the two assembled transport equations, in OpenFOAM's normalisation, WITHOUT
// solving anything. This is the honest oracle for "does our discretisation agree with OpenFOAM": at
// OpenFOAM's converged state both must be small, and that is a statement about the equations rather than
// about how far a solve happens to move a field. (The obvious test -- run one correct() and check the
// field does not move -- is NOT valid here: OpenFOAM itself stops on a residual plateau, not at an exact
// fixed point, so solving from its state to 1e-12 moves the field by however much that plateau is worth.)
struct Compressible;   // defined below

// correctNut(S2) as kOmegaSSTBase runs it, boundary and EddyDiffusivity included: nut =
// a1*k/max(a1*omega, b1*F23*sqrt(S2)) on cells; nut.correctBoundaryConditions() -- wall patches through
// nutkWallFunction, calculated patches from the boundary k/omega/S2/F2 (a field assignment writes the
// boundary too); alphat = rho*nut/Prt when comp carries one. Called at the end of correct() and, on its
// own, for turbulence->validate() at construction (eddyViscosity::validate -> correctNut()), which used
// to be skipped on this model so the first momentum solve ran on the case file's nut.
void correctNutField(
    const GeometricField<vector>&           U,
    const GeometricField<scalar>&           k,
    const GeometricField<scalar>&           omega,
    GeometricField<scalar>&                 nutField,
    const std::vector<tensor>&              gradU,     // S2 = 2|symm(gradU)|^2, boundary S2 from its patch correction
    const std::vector<scalar>&              y,         // cell wall distance (F2)
    const std::vector<std::vector<scalar>>& yWall,     // near-wall distance per wall face (nutkWallFunction)
    scalar                                  nu,        // the incompressible scalar; 0 with comp->nu/nuBnd set
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches,
    const KOmegaSSTCoeffs&                  co,
    const Compressible*                     comp);

struct SSTResiduals
{
    scalar omega = 0, k = 0;

    // OPT-IN diagnostics, compared against tools/dumpKOmegaSST's stage_sst* writes. The solver asks for
    // the residuals every outer iteration and would otherwise pay to copy every intermediate with them.
    bool captureStages = false;
    std::vector<scalar> divU, s2, gbyNu0, G, CD, f1, f23;
    std::vector<std::vector<scalar>> f1Bnd;   // F1 on the boundary faces, per patch
    std::vector<tensor> gradU;
    // the assembled systems, before relax and after, plus the off-diagonals a per-cell view misses
    std::vector<scalar> omD0, omSrc0, omD, omSrc, omUpper, omLower;
    std::vector<scalar> kD0,  kSrc0,  kD,  kSrc,  kUpper,  kLower;
};

// THE COMPRESSIBLE INSTANTIATION, exactly as it was done for kEpsilon. OpenFOAM has ONE
// kOmegaSSTBase.C templated on the lineage; what the compressible one supplies is alpha = 1, rho = the
// density field, alphaRhoPhi = the MASS flux, and a nu that varies with temperature. Every term below is
// already OpenFOAM's -- this struct is what turns the incompressible reading of them into the
// compressible one, so there is one transcription rather than two.
//
// TWO FLUXES, NOT ONE, and it is the same trap. `fvm::div` and the `bounded` Sp take the MASS flux,
// while divU = fvc::div(fvc::absolute(this->phi(), U)) takes the VOLUMETRIC one, because
// compressibleTurbulenceModel::phi() returns phi_/fvc::interpolate(rho) when the stored flux has mass
// dimensions. In the incompressible lineage the two are the same field and the distinction is invisible.
struct Compressible
{
    const std::vector<scalar>*              rho      = nullptr;   // cell density
    const std::vector<std::vector<scalar>>* rhoBnd   = nullptr;   // its patch values
    const std::vector<scalar>*              nu       = nullptr;   // laminar KINEMATIC viscosity, per cell
    const std::vector<std::vector<scalar>>* nuBnd    = nullptr;   // per boundary face
    const SurfaceScalarField*               phiByRho = nullptr;   // the VOLUMETRIC flux, for divU alone
    std::vector<scalar>*                    alphat   = nullptr;   // out: rho*nut/Prt (EddyDiffusivity)
    scalar                                  Prt      = 1.0;
};

// kOmegaSSTLM's three virtual overrides of this model, supplied by the DERIVED model rather than
// branched on here -- OpenFOAM's kOmegaSSTLM overrides F1, Pk and epsilonByk and inherits everything
// else (kOmegaSSTLM.C:43-76):
//
//   F1         = max(kOmegaSST::F1, F3),  F3 = exp(-(Ry/120)^8),  Ry = y*sqrt(k)/nu
//   Pk(G)      = gammaIntEff*kOmegaSST::Pk(G)
//   epsilonByk = clamp(gammaIntEff, 0.1, 1)*kOmegaSST::epsilonByk
//
// That F3 is NOT the base model's F3 near-wall switch, which this reference refuses: kOmegaSSTBase::F3
// is `1 - tanh(pow4(min(150*nu/(omega*y^2), 10)))` and multiplies F23. Two different functions, both
// called F3, in a base and its derived class.
struct LMHooks
{
    // Per cell. Null leaves this model exactly as it was: plain kOmegaSST.
    const std::vector<scalar>* gammaIntEff = nullptr;
};

// One kOmegaSST::correct(): the whole model, updating k, omega and nut in place.
//
// Refuses (throws) rather than silently approximating: F3, decayControl, and a case with no wall.
void correct(
    const GeometricField<vector>&  U,
    GeometricField<scalar>&        k,
    GeometricField<scalar>&        omega,
    GeometricField<scalar>&        nutField,
    const SurfaceScalarField&      phi,
    const std::vector<scalar>&     y,          // CELL wall distance
    scalar                         nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    scalar                         relaxOmega,
    scalar                         relaxK,
    scalar                         tol,
    scalar                         relTol,
    int                            maxIter,
    const KOmegaSSTCoeffs&         co = {},
    SSTResiduals*                  res = nullptr,    // the two INITIAL residuals, in OF's normalisation
    // div(phi,k) and div(phi,omega). The SST tutorials ask for `bounded Gauss limitedLinear 1` on both,
    // which is a DIFFERENT matrix from upwind -- limitedLinear supplies the convection weights, and
    // `bounded` subtracts Sp(fvc::div(phi), var). pitzDaily's kEpsilon asks for plain `Gauss upwind`,
    // so the defaults keep that and every existing call site is unchanged.
    bool                           bounded = false,
    bool                           limitedLinear = false,
    scalar                         limiterCoeff = 1.0,
    // `Gauss linearUpwind grad(<var>)` on k and omega. T3A asks for it on every turbulence scalar, and
    // it is a different matrix from both upwind and limitedLinear: upwind's, plus a deferred gradient
    // correction on the source. Ignoring it ran a scheme the case did not name.
    bool                           linearUpwind = false,
    // OpenFOAM applies the case's laplacianScheme to EVERY laplacian, the turbulence equations included.
    // `Gauss linear corrected` changes two things: the implicit face coefficient becomes
    // nonOrthDeltaCoeffs, AND an explicit deferred source is subtracted. Running the orthogonal form on
    // k and omega where the case says corrected is a different discretisation under the case's own
    // scheme name -- worth 5.2e-04 on T3A's ReThetat, whose mesh reaches 43.8 degrees.
    bool                           correctedLaplacian = false,
    scalar                         snGradLimitCoeff = 0.0,
    // kOmegaSSTLM. Null (the default) is plain kOmegaSST and nothing below changes.
    const LMHooks*                 lm = nullptr,
    // LAST, and deliberately: every existing caller passes these positionally, and inserting a parameter
    // ahead of them silently rebinds their arguments. Null throughout is the incompressible reading and
    // reproduces the previous arithmetic exactly.
    const Compressible*            comp = nullptr,
    // fvSolution solvers/<field>/minIter -- see kEpsilon_cpp.cuh. Last, after comp, for the same reason.
    int                            minIter = 0,
    // Whether fvSolution NAMES a factor for omega / k. OpenFOAM reaches relax() through fvMatrix::relax(),
    // which does nothing unless relaxEquation(name) holds (fvMatrix.C:1250-1263); brae's relaxMatrix at
    // 1.0 still applies the dominance clamp. Defaulted true so every positional caller keeps its
    // arithmetic; the compressible driver passes what the case says. Same shape as kEpsilon_cpp.
    bool                           relaxEquationOmega = true,
    bool                           relaxEquationK = true);

} // namespace kOmegaSST
} // namespace cpu
} // namespace brae
