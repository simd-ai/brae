#pragma once
// brae::fvc, explicit finite-volume operators (mirrors OpenFOAM finiteVolume/fvc).
// Gauss linear gradient of a volScalarField -> volVectorField (internal field).
//   grad[c] = (1/V)[ sum_internal Sf*p_f (+own/-nei) + sum_boundary Sf_bf*p_bf ]
//   p_f = w*p_own + (1-w)*p_nei  (Gauss linear interpolation)
// The field's boundary must be evaluated first (p.evaluateBoundary()).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include <vector>

namespace brae {

// A surfaceScalarField (face flux): internal faces + per-patch boundary faces.
struct SurfaceScalarField {
    std::vector<scalar>              internal;   // nInternalFaces
    std::vector<std::vector<scalar>> boundary;   // [patch][face]
};

namespace fvc {

std::vector<vector> gaussGrad(const GeometricField<scalar>& p,
                              const PrimitiveMesh& m,
                              const FvGeometry& g,
                              const std::vector<FvPatch>& patches);

// Array form, for a scalar field that is not a GeometricField. limitedLinear on a VECTOR needs
// fvc::grad(magSqr(U)) (LimitedScheme.C::calcLimiter with limitFuncs::magSqr), and magSqr(U) is a derived
// field with no patch objects of its own -- only values. Building a synthetic GeometricField for it would
// mean inventing boundary types it does not have.
std::vector<vector> gaussGrad(const std::vector<scalar>& internal,
                              const std::vector<std::vector<scalar>>& boundary,
                              const PrimitiveMesh& m,
                              const FvGeometry& g,
                              const std::vector<FvPatch>& patches);

// Gauss gradient of a volVectorField -> volTensorField (grad(U)_ij = sum Sf_i U_j / V).
std::vector<tensor> gaussGrad(const GeometricField<vector>& U,
                              const PrimitiveMesh& m,
                              const FvGeometry& g,
                              const std::vector<FvPatch>& patches);

// flux(U) = interpolate(U) & Sf  (linear interpolation, face-normal flux).
// Array form: for a vector field that is not a GeometricField (HbyA in pEqn.H, whose boundary values
// constrainHbyA partly takes from U). The GeometricField overload delegates to this one.
SurfaceScalarField flux(const std::vector<vector>& internal,
                        const std::vector<std::vector<vector>>& boundary,
                        const PrimitiveMesh& m, const FvGeometry& g,
                        const std::vector<FvPatch>& patches);

SurfaceScalarField flux(const GeometricField<vector>& U,
                        const PrimitiveMesh& m, const FvGeometry& g,
                        const std::vector<FvPatch>& patches);

// rhoFlux(rho, U) = interpolate(rho) * flux(U)  -- the MASS flux, kg/s, for the compressible solvers.
//
// Deliberately composed from interpolate() and flux() rather than fused into one loop. Both already
// reproduce OpenFOAM's linear face weights, and OF builds its compressible flux the same way
// (linearInterpolate(rho)*(linearInterpolate(U) & Sf)). Fusing would mean re-deriving those weights here,
// and any disagreement would surface much later as a fourth-digit mismatch in the pressure equation that
// looks like a p-equation bug rather than an interpolation one.
SurfaceScalarField rhoFlux(const std::vector<scalar>& rho,
                           const GeometricField<vector>& U,
                           const PrimitiveMesh& m, const FvGeometry& g,
                           const std::vector<FvPatch>& patches);

// fvc::snGrad(vf) -- the explicit surface-normal gradient, needed by SIMPLEC's phiHbyA correction.
//
// provenance: snGradScheme.C (snGrad(vf, deltaCoeffs)) and correctedSnGrad.C (the correction).
//
//   internal:  dc[f]*(vf[nei] - vf[own]),  dc = nonOrthDeltaCoeffs when `corrected`, else deltaCoeffs
//              plus, when corrected, corrVecs[f] & interpolate(grad(vf))[f]
//   boundary:  pvf.snGrad() on UNCOUPLED patches -- the patch's OWN deltaCoeffs, NOT the corrected ones
//              (snGradScheme.C passes tdeltaCoeffs only in the pvf.coupled() branch), and no correction,
//              since the non-orthogonal correction vectors are zero there.
//
// The boundary value is assembled from the patch's gradient coefficients rather than from a separate
// snGrad() method: snGrad == gradientInternalCoeffs*psi_c + gradientBoundaryCoeffs by construction, so
// reusing them keeps this consistent with fvm::laplacian's boundary treatment by definition instead of
// by coincidence -- zeroGradient gives 0 from both, fixedValue gives dc*(value - psi_c) from both.
SurfaceScalarField snGrad(
    const GeometricField<scalar>& vf,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    bool                          corrected);

// interpolate a volScalarField (cell array) to faces: linear internal; boundary = cell value
// (zeroGradient/extrapolated, as for rAU = 1/A()).
SurfaceScalarField interpolate(const std::vector<scalar>& vol,
                               const PrimitiveMesh& m, const FvGeometry& g,
                               const std::vector<FvPatch>& patches);

// div(phi) = surfaceIntegrate(phi) = (1/V)[ sum_internal phi (+own/-nei) + sum_boundary phi ].
std::vector<scalar> div(const SurfaceScalarField& phi,
                        const PrimitiveMesh& m, const FvGeometry& g,
                        const std::vector<FvPatch>& patches);

// Boundary face gradient tensors of U (per patch), built from the cell gradient and corrected so
// the wall-normal component equals snGrad(U) (OpenFOAM gaussGrad::correctBoundaryConditions):
//   gradU_b = gradU_cell + n (x) ( snGrad(U) - n & gradU_cell ),  n = Sf/|Sf|.
std::vector<std::vector<tensor>> gradUBoundary(const GeometricField<vector>& U,
                                               const std::vector<tensor>& gradUcell,
                                               const PrimitiveMesh& m, const FvGeometry& g,
                                               const std::vector<FvPatch>& patches);

// fvc::div of a volTensorField (cell tensors + per-patch boundary face tensors) -> volVectorField:
//   div(T)[c] = (1/V)[ sum_internal (Sf & T_f)(+own/-nei) + sum_boundary (Sf & T_b) ],  linear interp.
std::vector<vector> div(const std::vector<tensor>& tCell,
                        const std::vector<std::vector<tensor>>& tBnd,
                        const PrimitiveMesh& m, const FvGeometry& g,
                        const std::vector<FvPatch>& patches);

} // namespace fvc
} // namespace brae
