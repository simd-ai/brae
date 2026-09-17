#pragma once
// interFoam's createFields -- the case on disk turned into the fields the solver runs on.
//
// provenance:
//   openfoam:
//     file: applications/solvers/multiphase/interFoam/createFields.H
//     also: applications/solvers/multiphase/VoF/createAlphaFluxes.H
//           src/finiteVolume/cfdTools/general/include/readGravitationalAcceleration.H
//   brae:
//     reference: this header
//     cuda:      (pending)
//     tests:     tests/test_inter_case_cpp.cu, tests/interfoam_createfields_vs_openfoam.sh
//
// WHY THIS FILE IS SEPARATE FROM THE DRIVER, and it is the lesson rhoSimpleFoam's mirror wrote down:
// the harness and the solver must SHARE the case-to-fields translation. A private copy in the driver
// is the defect this project keeps finding one level up -- the gate proves the step, the driver feeds
// it something else, and nothing compares the two. buildInterFields is the one translation, and both
// the gate and brae_interFoam call it.
//
// WHAT createFields.H ACTUALLY BUILDS, in order, and the three places a port goes wrong:
//
//   1. p_rgh and U are READ. alpha1 is read as "alpha." + phase1Name -- damBreak's is alpha.water, and
//      the NAME comes from `phases (water air)` in transportProperties, not from a convention. Reading
//      a hard-coded "alpha1" finds nothing on any shipped case.
//
//   2. alpha2 = 1 - alpha1, then the mixture: rho from the RAW alpha, mu and nu from the CLAMPED one.
//      (two_phase_mixture_cpp.cuh carries that split and its gate.)
//
//   3. phi COMES FROM createAlphaFluxes.H, NOT from fvc::flux(U). OpenFOAM READS `phi` from the start
//      directory if it is there and only computes linearInterpolate(U) & Sf when it is not. A restart
//      therefore continues from the written flux, and recomputing it from U is a different field --
//      U and phi are not consistent to round-off after a solve, and the difference is the continuity
//      error the pressure corrector has just driven down.
//
//   4. gh and ghf need g AND hRef, and ghRef carries g's SIGN (inter_create_fields_cpp.cuh).
//      p = p_rgh + rho*gh is written and never solved.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "time_controls.cuh"
#include "two_phase_mixture_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include "alpha_eqn_cpp.cuh"
#include "inter_ueqn_cpp.cuh"
#include "inter_solve_cpp.cuh"
#include "inter_create_fields_cpp.cuh"
#include "mules_cpp.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

struct InterFields
{
    // --- read from the start directory
    GeometricField<scalar> alpha1;
    GeometricField<vector> U;
    GeometricField<scalar> p_rgh;
    SurfaceScalarField     phi;            // read if present, else linearInterpolate(U) & Sf

    // --- derived
    std::vector<scalar> alpha2, rho, mu, nu, gh, p;
    // alpha2's PATCH values, which are NOT 1 - alpha1's. `alpha2 = 1.0 - alpha1` (alphaEqn.H:223) runs
    // one line ABOVE the corrector's mixture.correct(), and at a contact-angle wall that call rewrites
    // alpha1's gradient and re-evaluates its patch. `rho == alpha1*rho1 + alpha2*rho2` then blends the
    // NEW alpha1 patch value with the OLD alpha2 one. Empty until the first alpha step, where
    // 1 - alpha1's patch value is what OpenFOAM's createFields leaves too.
    std::vector<std::vector<scalar>> alpha2Bnd;
    // THE MIXTURE ON THE BOUNDARY, built from alpha's PATCH VALUES and not from the face cell's.
    // Those are different fields at a contact-angle wall: alpha's patch value is
    // patchInternalField + gradient/deltaCoeffs, and the contact angle's gradient is what pulls the
    // interface up the wall -- 9681 over a deltaCoeffs of 20000 is +0.48 of alpha. Taking the cell
    // value instead gives an AIR viscosity on a face the interface has climbed, and
    // divDevRhoReff's laplacian is built from exactly that. Measured on capillaryRise against
    // OpenFOAM's own UEqn.A(): exact in all 3200 water cells, up to 56% low in the air cells at the
    // wall, which is where the contact line is.
    std::vector<std::vector<scalar>> rhoBnd, muBnd, nuBnd;
    // THE INTERFACE NORMAL AND CURVATURE ARE STATE, not a derived quantity recomputed on demand.
    // calculateK reads alpha's WALL GRADIENT, which the previous calculateK wrote through
    // correctContactAngle -- so it is a fixed-point iteration, and running it a different number of
    // times than OpenFOAM gives a different curvature. Measured on capillaryRise against OpenFOAM's
    // own K: one pass is 18% low at the contact line, two passes agree to 4e-07 relative, three
    // overshoot by 8%. interfaceProperties' CONSTRUCTOR runs the first pass, which is why these are
    // filled by buildInterFields and not left empty.
    SurfaceScalarField  nHatf;
    std::vector<scalar> K;
    std::vector<scalar> ghfInternal;                 // gh on the internal faces
    // ...AND ON THE BOUNDARY. The face forces are surfaceScalarFields, boundary included, and at a
    // contact-angle wall snGrad(alpha) is the contact angle's own gradient -- so zeroing the boundary
    // of the surface-tension force removes exactly the term the contact angle exists to apply. That
    // was measured on capillaryRise: it is worth 6% of the velocity at step 1.
    std::vector<std::vector<scalar>> ghfBoundary;
    SurfaceScalarField  rhoPhi;

    // --- the case's own settings
    cpu::twoPhase::MixtureSpec   mixture;
    interfaceProps::InterfaceCoeffs interface;
    AlphaControls                alphaCtl;
    MULES::Controls              mulesCtl;
    VoFTimeControls              timeCtl;
    // Part of the TIME STEP, not of the output: Time::setDeltaT calls adjustDeltaT, which under
    // `writeControl adjustableRunTime` trims deltaT to land on the next write time. See
    // time_controls.cuh, which carries the damBreak measurement.
    WriteCadence writeCadence;
    DivScheme                    divRhoPhiU     = DivScheme::upwind;
    scalar                       divRhoPhiUCoeff = 1.0;
    AlphaFluxScheme              divPhiAlpha    = AlphaFluxScheme::vanLeer;
    AlphaFluxScheme              divPhirbAlpha  = AlphaFluxScheme::linear;
    AlphaDdt                     ddtAlpha       = AlphaDdt::Euler;
    DdtScheme                    ddtU           = DdtScheme::Euler;

    // fvSolution's PIMPLE block. READ, not assumed: damBreak sets `momentumPredictor no`, which means
    // UEqn is ASSEMBLED AND NEVER SOLVED -- the matrix exists so pEqn can take A() and H() from it,
    // and the velocity is left entirely to the pressure corrector. A driver that always solves the
    // momentum equation runs a different algorithm on the canonical case and converges anyway.
    LoopControls pimple;
    label   nNonOrthogonalCorrectors = 0;
    bool    momentumPredictorOn = true;

    // THE CASE'S OWN p_rgh SOLVE, read from fvSolution's `solvers` block. It was hardcoded to 1e-9 in
    // the driver, with a BRAE_PTOL override -- so every tutorial ran to a tolerance nobody chose, and
    // a case asking for a LOOSER one (damBreak says `tolerance 1e-07; relTol 0.05`, five per cent of
    // the initial residual) was solved far tighter than OpenFOAM solves it. That is not a free
    // improvement: OpenFOAM's answer IS the loosely-solved one, and a gate comparing against it
    // measures the two stopping points.
    //
    // TWO ENTRIES, CHOSEN PER CORRECTOR. pEqn.H:50 solves with p_rgh.select(pimple.finalInnerIter()),
    // and finalInnerIter() (pimpleControlI.H:98-111) is true only on the LAST corrector's last
    // non-orthogonal pass -- so damBreak's three correctors solve to `relTol 0.05` twice and to
    // p_rghFinal's `relTol 0` once. This used to read relTol from the Final entry and apply it to all
    // three, which over-solved the first two and handed the last a different starting guess.
    //
    // AND THE SOLVER, where brae has OpenFOAM's own. It is not cosmetic here: damBreak's over-1 alpha
    // excursion is dt x the div(phi) this solve leaves (interFoam's alphaSuSp.H has no divU), and
    // with PBiCGStab standing in for the case's PCG+DIC it ran anywhere from 0.12x to 3.18x
    // OpenFOAM's from step to step. Tightening ONLY p_rgh removed all of it on both codes.
    struct PressureLinearSolve
    {
        std::string solver;
        std::string preconditioner;
        scalar tol = 1e-7;
        scalar relTol = 0;
        // lduMatrix::defaultMaxIter, lduMatrix.H:125
        int maxIter = 1000;
        // brae::pcg is lduMatrix PCG + DICPreconditioner, gated in tests/test_pcg.cu
        bool pcgDIC() const { return solver == "PCG" && preconditioner == "DIC"; }
    };
    // THE CASE'S alpha SOLVE, which only a MULESCorr case performs (the implicit upwind pre-solve,
    // alphaEqn.H:103-149). It used to run at a struct default of 1e-8 on the host and a hardcoded
    // 1e-12 on the device, neither read from the case. Defaults are lduMatrix::solver's own.
    struct AlphaLinearSolve
    {
        std::string solver;
        std::string smoother;
        scalar tol = 1e-6;
        scalar relTol = 0;
        int maxIter = 1000;
        int nSweeps = 1;
        bool gaussSeidel() const
        {
            return solver == "smoothSolver" && (smoother == "symGaussSeidel" || smoother == "GaussSeidel");
        }
    };
    AlphaLinearSolve aSolve;
    // solvers/p_rgh
    PressureLinearSolve pSolve;
    // solvers/p_rghFinal
    PressureLinearSolve pSolveFinal;
    // relaxationFactors/equations. THE QUESTION IS "DOES THE CASE NAME ONE", not "is it below 1":
    // fvMatrix::relax() is `if (mesh.relaxEquation(name, coeff)) relax(coeff)` and relaxEquation is
    // `found(name) || found("default")` (solution.C:330-334), so a case naming 1 relaxes -- the
    // dominance clamp still runs -- and a case naming nothing does not. damBreak says `".*" 1`;
    // capillaryRise has no relaxationFactors block at all, and this was hardcoded to true until that
    // case was run.
    bool    relaxEquationU = false;
    scalar  relaxU = 1.0;

    vector  g{0, 0, 0};
    scalar  hRef = 0;
    scalar  ghRefValue = 0;
    scalar  deltaT = 0;
    std::string alphaName;                 // "alpha." + phase1Name
    bool    phiWasRead = false;            // see note 3
};

// rho AS OpenFOAM HOLDS IT: the cell values plus CALCULATED patch values, for fvc::snGrad(rho).
// createFields.H builds rho from `alpha1*rho1 + alpha2*rho2`, so its patches are `calculated` and their
// snGrad() is the base class's, deltaCoeffs*(rho_b - rho_cell) (fvPatchField.C:220-223) -- NOT zero.
// brae took snGrad(rho) from a zeroGradient copy, which is zero on every patch. On a fixedFluxPressure
// wall that cancels through constrainPressure, and on both shipped tutorials alpha's patch value equals
// the cell's everywhere else, so nothing showed: the manifest carried it as LATENT. It is not small.
// Where a value-fixing pressure patch takes in one phase over a cell holding the other -- measured on
// damBreak with the atmosphere's inletValue set to 1 -- OpenFOAM's boundary snGrad(rho) is 1.57e+05 on
// 19 of 46 faces and phig there is 1.66e-02 against a phiHbyA of 2.5e-06. brae had 0, and alpha, p_rgh
// and U were each 100% out from the second step on.
GeometricField<scalar> rhoWithPatchValues(
    const std::vector<scalar>& rhoCells,
    const std::vector<std::vector<scalar>>& rhoBnd,
    const std::vector<FvPatch>& patches);

// Tell every flux-conditional patch of U, p_rgh and alpha1 the current phi -- see the definition.
// Call it whenever phi changes, before the next boundary evaluation reads it.
void pushFluxToPatches(
    InterFields& f,
    const std::vector<FvPatch>& patches);

// The case's dictionaries and fields -> InterFields. Throws, by name, on anything not ported.
// Rebuild the boundary blends from alpha's current patch values. Called wherever mixture.correct()
// is -- the patch values move with the contact angle every calculateK.
void updateMixtureBoundary(InterFields& f, const std::vector<FvPatch>& patches);

InterFields buildInterFields(const std::string&          caseDir,
                             const std::string&          startDir,
                             const PrimitiveMesh&        m,
                             const FvGeometry&           g,
                             const std::vector<FvPatch>& patches);

} // namespace interFoam
} // namespace cpu
} // namespace brae
