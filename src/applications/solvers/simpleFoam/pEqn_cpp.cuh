#pragma once
// _cpp REFERENCE -- host transcription of simpleFoam's pressure corrector.
//
// provenance:
//   openfoam:
//     file: applications/solvers/incompressible/simpleFoam/pEqn.H:1-50
//   brae:
//     reference: src/applications/solvers/simpleFoam/pEqn_cpp.cu
//     cuda:      (pending)
//     tests:     tests/test_peqn_cpp.cu
//
// OpenFOAM, verbatim:
//
//     volScalarField rAU(1.0/UEqn.A());
//     volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
//     surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
//     MRF.makeRelative(phiHbyA);
//     adjustPhi(phiHbyA, U, p);
//
//     tmp<volScalarField> rAtU(rAU);
//
//     if (simple.consistent())
//     {
//         rAtU = 1.0/(1.0/rAU - UEqn.H1());
//         phiHbyA += fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p)*mesh.magSf();
//         HbyA -= (rAU - rAtU())*fvc::grad(p);
//     }
//
//     tUEqn.clear();
//     constrainPressure(p, U, phiHbyA, rAtU(), MRF);
//
//     while (simple.correctNonOrthogonal())
//     {
//         fvScalarMatrix pEqn(fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA));
//         pEqn.setReference(pRefCell, pRefValue);
//         pEqn.solve();
//         if (simple.finalNonOrthogonalIter()) { phi = phiHbyA - pEqn.flux(); }
//     }
//
//     #include "continuityErrs.H"
//     p.relax();
//     U = HbyA - rAtU()*fvc::grad(p);
//     U.correctBoundaryConditions();
//     fvOptions.correct(U);
//
// WHY THE STAGES ARE EXPOSED. Every intermediate above is returned rather than kept local, because the
// method for this rebuild is "find the FIRST divergent intermediate", not "compare the final residual".
// A pressure corrector that disagrees with OpenFOAM can be wrong at rAU, at HbyA, at the flux, at the
// Laplacian, at the reference cell or at the flux correction, and a single number over `p` cannot say
// which. brae has already spent whole investigations on that distinction -- one of them ended at
// `phi = phiHbyA - pEqn.flux()` leaving a growing divergence, which is stage 7 here.
//
// REFUSED, not ignored -- same contract as UEqn_cpp:
//   * MRF          (pEqn.H:5 makeRelative, and inside constrainPressure)
//   * fvOptions    (pEqn.H:49 correct(U))
//   * fixedFluxPressure on a p patch -- pEqn.H:21 updates it through constrainPressure, not ported
// `consistent` (SIMPLEC, pEqn.H:8-16) IS implemented: matrixH1 and fvc::snGrad were added for it.
// A case that asks for any of them gets an error, not a quietly different equation.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "MRF_cpp.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvc.cuh"
#include <vector>

namespace brae {
namespace cpu {

struct PressureInput
{
    // MRF.makeRelative(phiHbyA), pEqn.H:5 -- between fvc::flux(HbyA) and adjustPhi, because adjustPhi
    // balances the flux it is given.
    const std::vector<MRF::Zone>* mrf = nullptr;

    scalar relaxP = 1.0;        // relaxationFactors/fields p
    label  pRefCell = -1;       // setRefCell; -1 => the case does not need a reference
    scalar pRefValue = 0.0;
    // SIMPLEC. Implemented: rAtU = 1/(1/rAU - UEqn.H1()), with the phiHbyA and HbyA corrections.
    bool   consistent = false;
    // A pressure patch of type fixedFluxPressure -- pEqn.H reaches it through constrainPressure, which
    // is NOT ported, so it must be refused rather than left with a stale gradient.
    bool   correctedLaplacian = false;   // `corrected` laplacianSchemes
    scalar snGradLimitCoeff = 0.0;       // `limited <k> corrected` (OF limitedSnGrad)
    bool   hasMRF = false;      // refused
    bool   hasFvOptions = false;// refused
};

// Every intermediate of pEqn.H, in the order OpenFOAM produces them.
struct PressureStages
{
    std::vector<scalar> rAU;        // 1/UEqn.A()
    // SIMPLEC's rAtU = 1/(1/rAU - UEqn.H1()). EQUAL to rAU when `consistent` is off, and everything
    // downstream (the laplacian diffusivity, the U corrector) uses rAtU unconditionally -- pEqn.H does
    // the same with `tmp<volScalarField> rAtU(rAU)`. Keeping two names that are usually the same object
    // is OpenFOAM's own structure and it is what makes the SIMPLEC branch a two-line diff rather than a
    // second code path.
    std::vector<scalar> rAtU;
    std::vector<vector> HbyA;       // constrainHbyA(rAU*UEqn.H(), U, p)
    SurfaceScalarField  phiHbyA;    // fvc::flux(HbyA), after adjustPhi
    bool                phiAdjusted = false;   // did adjustPhi actually scale anything
    FvScalarMatrix      pEqn;       // fvm::laplacian(rAU,p) == fvc::div(phiHbyA), with the reference set
    SurfaceScalarField  phi;        // phiHbyA - pEqn.flux()   (only after correctFlux)
    std::vector<vector> U;          // HbyA - rAU*grad(p)      (only after correctVelocity)
};

// Stages 1-3: rAU, HbyA, phiHbyA (including adjustPhi). Does not solve anything.
PressureStages pressurePredictor(
    const FvVectorMatrix&         UEqn,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const PressureInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// Stage 4-5: assemble fvm::laplacian(rAU, p) == fvc::div(phiHbyA) and apply setReference.
//
// The reference is applied here rather than by the caller because OpenFOAM applies it to the ASSEMBLED
// matrix and its effect (source[cell] += diag[cell]*value; diag[cell] += diag[cell]) is not recoverable
// afterwards -- it is not a boundary condition, it is a rank fix on a singular operator.
FvScalarMatrix assemblePEqn(
    const PressureStages&         st,
    const GeometricField<scalar>& p,
    const PressureInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// Stage 7: phi = phiHbyA - pEqn.flux(), with `p` the SOLVED pressure.
SurfaceScalarField correctFlux(
    const PressureStages&         st,
    const FvScalarMatrix&         pEqn,
    const std::vector<scalar>&    pSolved,      // internal field only -- see matrixFlux
    const PrimitiveMesh&          m,
    const std::vector<FvPatch>&   patches);

// Stage 9: U = HbyA - rAU*fvc::grad(p), with `p` the RELAXED pressure.
//
// The order in pEqn.H is p.relax() and THEN the momentum corrector, so the velocity correction uses the
// relaxed pressure while the flux correction above used the unrelaxed one. Swapping them is a real and
// silent error, which is why the two stages take different arguments instead of sharing one `p`.
std::vector<vector> correctVelocity(
    const PressureStages&         st,
    const GeometricField<scalar>& pRelaxed,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// GeometricField::relax: p = pPrev + alpha*(p - pPrev)   (OF GeometricField.C:1089-1095)
void relaxField(
    std::vector<scalar>&       p,
    const std::vector<scalar>& pPrev,
    scalar                     alpha);

} // namespace cpu
} // namespace brae
