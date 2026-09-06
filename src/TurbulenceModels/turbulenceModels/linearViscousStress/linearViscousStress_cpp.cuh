#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's linear viscous stress divergence.
//
// provenance:
//   openfoam:
//     symbol: Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
//     file:   src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress.C:107-117
//     via:    Foam::IncompressibleTurbulenceModel<TransportModel>::divDevReff
//             src/TurbulenceModels/incompressible/IncompressibleTurbulenceModel/
//             IncompressibleTurbulenceModel.C:123-129   ( divDevReff(U) -> divDevRhoReff(U) )
//     reached_from: applications/solvers/incompressible/simpleFoam/UEqn.H:9
//   brae:
//     reference: this file
//     cuda:      src/cuda/device_divdevreff.cu
//     tests:     tests/test_divdevreff_cpp.cu
//
// OpenFOAM, verbatim:
//
//     return
//     (
//       - fvc::div((this->alpha_*this->rho_*this->nuEff())*dev2(T(fvc::grad(U))))
//       - fvm::laplacian(this->alpha_*this->rho_*this->nuEff(), U)
//     );
//
// For the INCOMPRESSIBLE lineage alpha_ == 1 and rho_ == 1 (nuEff is kinematic), so this reduces to
//
//     divDevReff(U) = -fvc::div(nuEff*dev2(T(grad(U)))) - fvm::laplacian(nuEff, U)
//
// and UEqn.H ADDS it, giving the momentum operator
//
//     fvm::div(phi,U) - fvm::laplacian(nuEff,U) - fvc::div(nuEff*dev2(T(grad U)))
//
// Both halves matter. Dropping the explicit dev2 term leaves a Laplacian-only stress that converges to a
// slightly different answer and reports nothing -- it is not a stability problem, it is a wrong answer.
//
// dev2 is OpenFOAM's, not a paraphrase:
//     src/OpenFOAM/primitives/Tensor/TensorI.H:832   dev2(t) = t - 2*sph(t),  sph(t) = (1/3)*tr(t)*I
// so  dev2(t) = t - (2/3)*tr(t)*I.
//
// This is the REFERENCE implementation: readable, host-only, allocation-per-step, no performance claim.
// Its single job is to be obviously the same as the OpenFOAM text above so that the CUDA path can be
// diffed against it.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvc.cuh"
#include "fvm.cuh"
#include <vector>

// `transpose`, `dev2` and `scalar*tensor` are NOT redefined here: brae already has them in
// src/OpenFOAM/primitives/cf_types.cuh (transpose:103, operator*:108, dev2:113, the last already carrying
// the OpenFOAM TensorI.H citation). A second local copy would be a second thing to keep correct, and the
// rule for this rebuild is to move working code, not to rewrite it.
namespace brae {
namespace cpu {

// nuEff interpolated to faces, the way OpenFOAM's fvm::laplacian(volScalarField, U) does it.
//
// Internal faces are the linear (weight-based) interpolation. BOUNDARY faces take the boundary field of
// nuEff -- NOT the owner cell's value. On a wall with a nut wall function those differ by the whole of
// nut_wall, and using the cell value silently under-predicts wall shear; brae has been bitten by exactly
// this before (the boundary_mu_eff gate). Public because UEqn_cpp assembles the same implicit laplacian
// and must use the same face viscosity -- two copies of this rule is two chances to get it wrong.
SurfaceScalarField effectiveFaceViscosity(
    const std::vector<scalar>& nuEff,
    const std::vector<std::vector<scalar>>& nuEffBnd,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

// The EXPLICIT half: -fvc::div(nuEff*dev2(T(grad(U)))), returned as a per-cell vector (already divided by
// V by fvc::div, exactly as OpenFOAM's surfaceIntegrate does).
//
// The boundary tensors come from fvc::gradUBoundary, which applies OpenFOAM's
// gaussGrad::correctBoundaryConditions -- the wall-normal component of the boundary gradient is replaced
// by snGrad(U). Using the raw cell gradient on boundary faces instead is a silent wall-shear error.
std::vector<vector> divDevReffExplicit(
    const GeometricField<vector>& U,
    const std::vector<scalar>&    nuEff,      // nCells
    const std::vector<std::vector<scalar>>& nuEffBnd,   // [patch][face]
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    // gradSchemes/grad(U): 0 = the unlimited base scheme, >0 = cellLimited with that coefficient.
    scalar                        gradULimitK = 0.0);

// The full operator as it appears in UEqn.H: the IMPLICIT laplacian assembled into the matrix, and the
// EXPLICIT dev2 term added to the source.
//
// Sign convention follows OpenFOAM exactly. UEqn.H writes `+ turbulence->divDevReff(U)` and divDevReff
// returns MINUS both terms, so this adds -laplacian to the matrix and -div(...) to the source. Since
// brae's FvMatrix keeps `source` on the right-hand side (M.psi = source), an explicit term that OpenFOAM
// places on the left with a minus sign enters `source` with a PLUS sign; that single sign flip is the one
// place a transcription of this function can go wrong without failing loudly, so it is asserted by
// tests/test_divdevreff_cpp.cu against an OpenFOAM dump rather than reasoned about here.
void addDivDevReff(
    FvVectorMatrix&               UEqn,
    const GeometricField<vector>& U,
    const std::vector<scalar>&    nuEff,
    const std::vector<std::vector<scalar>>& nuEffBnd,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    bool                          correctedLaplacian = false,
    // `limited <k> corrected` -- caps the non-orth correction against the orthogonal part. 0 = uncapped.
    scalar                        snGradLimitCoeff = 0.0,
    // The case's gradScheme for grad(U). divDevRhoReff's explicit term is
    // fvc::div((rho*nuEff)*dev2(T(fvc::grad(U)))), and OpenFOAM resolves that grad against gradSchemes --
    // aerofoilNACA0012 asks for `cellLimited Gauss linear 1` on grad(U) while squareBend asks for plain
    // `Gauss linear`, so a fixture with only the latter cannot tell the two apart. 0 = unlimited, which
    // is what every existing caller passes and what `Gauss linear` means.
    scalar                        gradULimitK = 0.0);

} // namespace cpu
} // namespace brae
