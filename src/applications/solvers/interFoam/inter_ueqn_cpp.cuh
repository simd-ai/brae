#pragma once
// interFoam's momentum predictor -- the host reference.
//
// provenance:
//   openfoam:
//     file: applications/solvers/multiphase/interFoam/UEqn.H:1-33
//     also: src/finiteVolume/finiteVolume/ddtSchemes/EulerDdtScheme/EulerDdtScheme.C:434-470
//             (fvmDdt(volScalarField rho, vf) -- THE OLD-TIME rho, see below)
//           src/phaseSystemModels/twoPhaseInter/incompressibleInterPhaseTransportModel/
//             incompressibleInterPhaseTransportModel.C:117-131  (which divDevRhoReff is reached)
//           src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress.C:119-131
//             (divDevRhoReff(rho, U) -- the rho-weighted overload)
//           src/transportModels/interfaceProperties/interfaceProperties.C:237-241
//             (surfaceTensionForce() = interpolate(sigma*K)*snGrad(alpha1))
//   brae:
//     reference: this header
//     cuda:      src/applications/solvers/interFoam/device_inter_ueqn.cu -- the two pieces that are
//                NOT rhoSimpleFoam's (the two-rho ddt and the face force); fvm::div(rhoPhi,U) and
//                divDevRhoReff are the device operators that solver already carries.
//     tests:     tests/test_inter_ueqn_cpp.cu
//
// OpenFOAM, verbatim:
//
//     MRF.correctBoundaryVelocity(U);
//
//     fvVectorMatrix UEqn
//     (
//         fvm::ddt(rho, U) + fvm::div(rhoPhi, U)
//       + MRF.DDt(rho, U)
//       + turbulence->divDevRhoReff(rho, U)
//      ==
//         fvOptions(rho, U)
//     );
//
//     UEqn.relax();
//     fvOptions.constrain(UEqn);
//
//     if (pimple.momentumPredictor())
//     {
//         solve(UEqn == fvc::reconstruct((mixture.surfaceTensionForce()
//                                       - ghf*fvc::snGrad(rho)
//                                       - fvc::snGrad(p_rgh)) * mesh.magSf()));
//         fvOptions.correct(U);
//     }
//
// THREE THINGS THAT ARE NOT IN rhoSimpleFoam's UEqn, and none of them is notation.
//
// 1. ddt's SOURCE CARRIES rho.oldTime(), THE DIAGONAL CARRIES rho.
//
//    EulerDdtScheme.C:455-467:
//        fvm.diag()   = rDeltaT*rho.primitiveField()*Vsc
//        fvm.source() = rDeltaT*rho.oldTime().primitiveField()*vf.oldTime().primitiveField()*Vsc
//
//    Two different rho fields in the same term. On rhoSimpleFoam this never arises -- steady, no ddt --
//    and on a weakly compressible transient the two rhos differ by a per-cent, so reusing the new one
//    looks like a rounding choice. HERE THEY DIFFER BY A FACTOR OF 1000: rho is the water/air blend, and
//    any cell the interface crossed during the alpha solve has rho_old = 1 and rho = 1000 or the reverse.
//    Writing rho in the source multiplies that cell's old momentum by 1000, exactly at the interface,
//    which is the only place a VoF solution is decided. tests/test_inter_ueqn_cpp.cu arm 1 carries the
//    control that separates the two; on a converged smooth field they agree and no ordinary gate can.
//
// 2. THE VISCOSITY IS rho*nuEff AND rho IS NOT THE ONE INSIDE nu.
//
//    interFoam's turbulence is incompressibleInterPhaseTransportModel; divDevRhoReff(rho, U) forwards to
//    incTurbulence_->divDevRhoReff(rho, U) (.C:129), i.e. linearViscousStress's rho-weighted overload:
//
//        -fvc::div((rho*nuEff)*dev2(T(grad U))) - fvm::laplacian(rho*nuEff, U)
//
//    so the operator is the one brae already has (linearViscousStress_cpp.cuh), given mu_eff = rho*nuEff.
//    What is worth naming is that rho*nuEff is NOT mu: the mixture's rho takes the RAW alpha while nu's
//    denominator takes the CLAMPED one (two_phase_mixture_cpp.cuh), so wherever MULES has left alpha
//    outside [0,1] the two differ. OpenFOAM forms rho*nuEff, so this forms rho*nuEff.
//
// 3. THE MOMENTUM SOURCE IS A RECONSTRUCTED FACE FLUX, NOT A CELL GRADIENT.
//
//    rhoSimpleFoam adds -fvc::grad(p). interFoam builds a surface field, reconstructs it, and adds that.
//    The two are not interchangeable: the face form is what makes gravity and surface tension balance
//    p_rgh face-by-face, which is the whole reason the p_rgh formulation exists. A cell-gradient
//    substitute leaves a spurious current at the interface that looks like physics.
//
//    Sign: `solve(UEqn == R)` is fvMatrix::operator==, i.e. source += V*R (fvMatrix.C:1855-1862 with the
//    double negative of operator-=). rhoSimpleFoam's addPressureGradient carries the same convention with
//    R = -grad(p); it is written out again here because getting it backwards still converges.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fv_matrix_ops.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fvc_reconstruct_cpp.cuh"
#include "limitedSchemes_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "linearViscousStress_cpp.cuh"
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

// ddtSchemes.default. interFoam's 44 shipped tutorials are Euler in 42 of them, `CrankNicolson 0.5` in
// one and `localEuler` (LTS) in one. The two non-Euler forms are REFUSED by name rather than silently
// run as Euler: localEuler is a per-cell pseudo-time that changes what the solution means, and
// CrankNicolson is additionally refused by alphaEqn.H:44-50 when sub-cycling, so accepting it here would
// let the momentum and alpha equations disagree about the time scheme.
enum class DdtScheme { Euler, backward, CrankNicolson, localEuler, steadyState };

// div(rhoPhi,U), as named by fvSchemes. The shipped tutorials ask for linearUpwind grad(U) (24 files),
// vanLeerV (8), upwind (6), linear (3) and limitedLinear 0.2 (1). vanLeerV is the V-variant of the
// limiter added for div(phi,alpha) and is NOT the same object -- it limits the component along the
// direction of steepest change rather than each component separately -- so it is refused until ported.
enum class DivScheme { upwind, linear, limitedLinear, limitedLinearV, linearUpwind, linearUpwindV, LUST };

struct InterMomentumInput
{
    // rhoPhi -- the MULES-limited MASS flux out of alphaEqn, kg/s. Not the volumetric phi.
    const std::vector<scalar>*              rhoPhi    = nullptr;
    const std::vector<std::vector<scalar>>* rhoPhiBnd = nullptr;

    // The mixture density at THIS time level, and at the previous one. Both are required: the ddt
    // diagonal takes the first and the ddt source takes the second (EulerDdtScheme.C:455-467).
    const std::vector<scalar>*              rho       = nullptr;   // cells
    const std::vector<scalar>*              rhoOld    = nullptr;   // cells
    const std::vector<std::vector<scalar>>* rhoBnd    = nullptr;   // [patch][face]

    // U at the previous time level, for the ddt source.
    const std::vector<vector>*              UOld      = nullptr;   // cells

    // The mixture KINEMATIC effective viscosity, nu + nut. Multiplied by rho here, because the
    // multiplication is the one decision this component owns (see note 2 in the header).
    const std::vector<scalar>*              nuEff     = nullptr;   // cells
    const std::vector<std::vector<scalar>>* nuEffBnd  = nullptr;   // [patch][face]

    // ALTERNATIVE to rho/nuEff for the stress term only: the dynamic viscosity directly, so a gate can
    // inject OpenFOAM's own and measure the ASSEMBLY without a ported mixture turbulence model in the
    // way. Mirrors RhoMomentumInput::muEff.
    const std::vector<scalar>*              muEff     = nullptr;
    const std::vector<std::vector<scalar>>* muEffBnd  = nullptr;

    scalar    deltaT             = 0.0;          // > 0 always; interFoam has no steady path
    DdtScheme ddtScheme          = DdtScheme::Euler;

    DivScheme scheme             = DivScheme::upwind;
    scalar    schemeCoeff        = 1.0;          // limitedLinear's k
    scalar    gradULimitK        = 0.0;          // cellLimited coefficient on grad(U), 0 = unlimited
    bool      gradULeastSq       = false;
    // linearUpwind NAMES its own gradient (`linearUpwind grad(U)` -> gradSchemes entry `grad(U)`); -1
    // means the caller did not resolve one and gradULimitK stands in, as in rhoSimpleFoam.
    scalar    gradULULimitK      = -1.0;

    // THE GUARD IS "THE CASE NAMES A FACTOR", NOT "THE FACTOR IS BELOW 1". damBreak's fvSolution says
    // `equations { ".*" 1; }`, which relaxEquation() FINDS (solution.C:330-334), and relax(1) still runs
    // the diagonal-dominance clamp D = max(|D|, sumOff)/alpha (fvMatrix.C:1102-1107). A port that skipped
    // relax on alpha == 1 would differ from OpenFOAM on every shipped interFoam tutorial.
    bool      relaxEquationU     = false;
    scalar    relaxU             = 1.0;

    bool      correctedLaplacian = false;
    scalar    snGradLimitCoeff   = 0.0;

    bool      hasMRF             = false;        // declared by the case -> refuse until ported
    bool      hasFvOptions       = false;        // declared by the case -> refuse until ported
    std::string fvOptionUnsupported;
};

// mu_eff = rho*nu_eff. Shared with rhoSimpleFoam's derivation rather than re-derived: the product is the
// same one, gated by tests/rho_ueqn_vs_openfoam.sh, and two copies are two chances to disagree.
inline std::vector<scalar> dynamicViscosity(const std::vector<scalar>& rho,
                                            const std::vector<scalar>& nuEff)
{
    if (rho.size() != nuEff.size())
        throw std::runtime_error("brae interFoam UEqn: rho and nuEff differ in length.");
    std::vector<scalar> mu(rho.size());
    for (std::size_t c = 0; c < rho.size(); ++c) mu[c] = rho[c] * nuEff[c];
    return mu;
}

inline std::vector<std::vector<scalar>> dynamicViscosityBoundary(
    const std::vector<std::vector<scalar>>& rhoBnd,
    const std::vector<std::vector<scalar>>& nuEffBnd)
{
    if (rhoBnd.size() != nuEffBnd.size())
        throw std::runtime_error("brae interFoam UEqn: rhoBnd and nuEffBnd differ in patch count.");
    std::vector<std::vector<scalar>> mu(rhoBnd.size());
    for (std::size_t pi = 0; pi < rhoBnd.size(); ++pi)
    {
        if (rhoBnd[pi].size() != nuEffBnd[pi].size())
            throw std::runtime_error("brae interFoam UEqn: rhoBnd and nuEffBnd differ on a patch.");
        mu[pi].resize(rhoBnd[pi].size());
        for (std::size_t i = 0; i < rhoBnd[pi].size(); ++i) mu[pi][i] = rhoBnd[pi][i] * nuEffBnd[pi][i];
    }
    return mu;
}

// fvm::ddt(rho, U), Euler, on a fixed mesh (Vsc == Vsc0 == V).
//
//     diag[c]   += rDeltaT*rho[c]*V[c]
//     source[c] += rDeltaT*rhoOld[c]*UOld[c]*V[c]
//
// rhoOld is a SEPARATE argument from rho on purpose: passing the same vector twice is the defect this
// signature exists to make visible at the call site.
inline void addEulerDdtRhoU(FvVectorMatrix&            M,
                            const std::vector<scalar>& rho,
                            const std::vector<scalar>& rhoOld,
                            const std::vector<vector>& UOld,
                            const std::vector<scalar>& V,
                            scalar                     deltaT)
{
    if (deltaT <= scalar(0))
        throw std::runtime_error("brae interFoam UEqn: deltaT must be positive; interFoam has no steady path.");
    const std::size_t nC = rho.size();
    if (rhoOld.size() != nC || UOld.size() != nC || V.size() != nC || M.diag.size() != nC)
        throw std::runtime_error("brae interFoam UEqn: ddt field lengths disagree.");
    const scalar rDeltaT = scalar(1) / deltaT;
    for (std::size_t c = 0; c < nC; ++c)
    {
        M.diag[c] += rDeltaT * rho[c] * V[c];
        const scalar w = rDeltaT * rhoOld[c] * V[c];
        M.source[c].x += w * UOld[c].x;
        M.source[c].y += w * UOld[c].y;
        M.source[c].z += w * UOld[c].z;
    }
}

// (surfaceTensionForce() - ghf*snGrad(rho) - snGrad(p_rgh)) * magSf, per face.
//
// All four inputs are surface fields the caller has already formed, so this is the sign and the magSf
// factor and nothing else -- which is precisely the part that cannot be checked by looking at a plot.
inline void momentumSourceFlux(const std::vector<scalar>& surfaceTensionForce,
                               const std::vector<scalar>& ghf,
                               const std::vector<scalar>& snGradRho,
                               const std::vector<scalar>& snGradPrgh,
                               const std::vector<scalar>& magSf,   // the mesh's full face array
                               std::vector<scalar>&       out)
{
    // The three face fields set the length; magSf is the MESH'S FULL FACE ARRAY -- internal faces
    // first, then the boundary patches -- so it indexes straight through and is only required to be
    // long enough. compressionFlux carries the same convention, and getting it wrong here cost a run
    // on damBreak that failed with a length mismatch rather than a wrong number.
    const std::size_t n = surfaceTensionForce.size();
    if (ghf.size() != n || snGradRho.size() != n || snGradPrgh.size() != n)
        throw std::runtime_error("brae interFoam UEqn: momentum source face fields differ in length.");
    if (magSf.size() < n)
        throw std::runtime_error(
            "brae interFoam UEqn: magSf is shorter than the face fields it scales.");
    out.resize(n);
    for (std::size_t f = 0; f < n; ++f)
        out[f] = (surfaceTensionForce[f] - ghf[f]*snGradRho[f] - snGradPrgh[f]) * magSf[f];
}

// fvc::reconstruct over the mesh, from the face flux above. Delegates every face to the validated
// accumulation in fvc_reconstruct_cpp.cuh, so the same-sign surfaceSum lives in one place.
inline std::vector<vector> reconstruct(const SurfaceScalarField&   ssf,
                                       const PrimitiveMesh&        m,
                                       const FvGeometry&           g,
                                       const std::vector<FvPatch>& patches)
{
    using namespace cpu::fvcReconstruct;
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<vector>& Sf  = g.Sf();

    std::vector<tensor> T(static_cast<std::size_t>(nC), tensor{0,0,0,0,0,0,0,0,0});
    std::vector<vector> v(static_cast<std::size_t>(nC), vector{0,0,0});
    for (label f = 0; f < nIf; ++f)
    {
        accumulate(Sf[f], ssf.internal[f], T[own[f]], v[own[f]]);
        accumulate(Sf[f], ssf.internal[f], T[nei[f]], v[nei[f]]);
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        for (label i = 0; i < fp.size; ++i)
        {
            const label c = own[fp.start + i];
            accumulate(Sf[fp.start + i], ssf.boundary[pi][i], T[c], v[c]);
        }
    }
    std::vector<vector> out(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c) out[c] = dot(inv(T[c]), v[c]);
    return out;
}

// Steps 1-3 of UEqn.H: ddt + div + divDevRhoReff, then UEqn.relax().
//
// Returned WITHOUT the momentum-predictor source, because that is the object pEqn.H needs: rAU = 1/A()
// and H() are both taken from the relaxed matrix before the face forces are applied.
FvVectorMatrix assembleUEqn(
    const GeometricField<vector>& U,
    const InterMomentumInput&     in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// Step 4: `solve(UEqn == fvc::reconstruct(flux))`. source += V*reconstruct(flux) -- see note 3.
void addMomentumPredictorSource(
    FvVectorMatrix&             UEqn,
    const SurfaceScalarField&   flux,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

// ---------------------------------------------------------------------------------------------------
// UEqn.H end to end, on a case: assemble, add the face-force source, solve.
//
// WHAT THE SOURCE LOOKS LIKE ON A REAL CASE, and it is the clearest statement of what the p_rgh
// formulation IS. The momentum predictor's only body force is
//
//     fvc::reconstruct((surfaceTensionForce - ghf*snGrad(rho) - snGrad(p_rgh)) * magSf)
//
// and on a hydrostatic start -- p_rgh uniform, alpha sharp -- every one of those three terms is
// IDENTICALLY ZERO except at the interface: snGrad(p_rgh) because p_rgh is uniform, snGrad(rho)
// because rho is piecewise constant, and the surface tension because snGrad(alpha) is. So the bulk of
// each phase feels NOTHING from the momentum predictor, and the motion comes from the pressure solve.
//
// That is the whole point of solving for p_rgh rather than p: gravity is absorbed into the pressure
// variable and appears only where the density actually varies. A port that put rho*g in the momentum
// source instead -- the obvious reading of "add gravity" -- would accelerate the entire water column
// and then have the pressure solve cancel it, which is a much worse-conditioned problem and gives a
// different answer at the interface. tests/test_inter_case_cpp.cu asserts the zero directly.
struct MomentumSolveControls
{
    scalar tolU    = 1e-7;
    scalar relTolU = 0;
    int    maxIterU = 1000;
};

// Assemble UEqn, add the reconstructed face force, and solve for U. `faceForce` is the surface field
// (surfaceTensionForce - ghf*snGrad(rho) - snGrad(p_rgh)) * magSf, formed by the caller so that the
// three terms are visible where they are chosen.
void momentumPredictor(GeometricField<vector>&     U,
                       const InterMomentumInput&   in,
                       const SurfaceScalarField&   faceForce,
                       const MomentumSolveControls& sc,
                       const PrimitiveMesh&        m,
                       const FvGeometry&           g,
                       const std::vector<FvPatch>& patches,
                       // pimple.momentumPredictor(). FALSE means ASSEMBLE AND RELAX BUT DO NOT SOLVE --
                       // the matrix is still needed for rAU = 1/A() and H(), and U is left for the
                       // pressure corrector. damBreak sets `momentumPredictor no`, so this is the
                       // canonical case's own setting, not an exotic one.
                       bool                        solveMomentum,
                       FvVectorMatrix&             UEqnOut);

} // namespace interFoam
} // namespace cpu
} // namespace brae
