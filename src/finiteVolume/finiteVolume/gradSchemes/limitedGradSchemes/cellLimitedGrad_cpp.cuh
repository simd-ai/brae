#pragma once
// cellLimitedGrad<Type, minmod> -- the cell-limited gradient, host reference.
//
// provenance:
//   openfoam: src/finiteVolume/finiteVolume/gradSchemes/limitedGradSchemes/cellLimitedGrad/
//             cellLimitedGrad.C, cellLimitedGrad.H (limitFace/limitFaceCmpt),
//             gradientLimiters/minmodGradientLimiter.H
//   brae:     .../cellLimitedGrad_cpp.cu
//   tests:    tests/test_celllimited_cpp.cu, tests/celllimited_vs_openfoam.sh
//
// `cellLimited Gauss linear <k>` takes the base Gauss gradient and scales it, per cell and PER COMPONENT,
// so that extrapolating the cell value to any of its own face centres cannot overshoot the range of the
// cell's neighbours. It is not a smoothing: the limiter is <= 1 and only ever shrinks the gradient.
//
//   maxDelta/minDelta   over the cell's neighbours -- INCLUDING the boundary faces, which contribute the
//                       patch VALUE (a coupled patch contributes its neighbour field instead) -- then
//                       taken relative to the cell's own value.
//   k < 1               widens the admissible band by (1/k - 1)*(maxDelta - minDelta) on each side.
//                       k = 1 leaves it alone, and k < SMALL disables the scheme entirely.
//   limitFaceCmpt       r = maxDelta/extrapolate if extrapolate > SMALL, minDelta/extrapolate if
//                       < -SMALL, and NO CONTRIBUTION AT ALL otherwise -- an early return, not r = 1,
//                       which is the easiest line here to get wrong. minmod's limiter is min(r, 1).
//   limitGradient       scalar: g *= limiter. vector: the limiter is itself a vector and scales
//                       grad(U_j) by limiter[j] -- i.e. COLUMN j of the tensor, because OpenFOAM's
//                       grad(U)_ij is d(U_j)/d(x_i).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"

#include <vector>

namespace brae {
namespace cpu {

// Limits a gradient that has ALREADY been computed by the base scheme, in place.
// The same limiter on a field given as VALUES -- the internal field and the patch values -- for a
// caller that has no GeometricField: the energy equation's kinetic-energy term builds K = 0.5|U|^2 (or
// Ekp) on the fly with its boundary values in a separate vector. The GeometricField overload below
// delegates here. It used to be the other way round, and the caller handed a boundary-less shim: the
// first case that named `grad(Ekp) cellLimited` dereferenced a null patch inside limitPass (SIGSEGV),
// which the never-forwarded limiter coefficient had hidden -- see rhoSimpleFoamDriver_cpp.cu.
void cellLimitGrad(
    std::vector<vector>&                    grad,
    const std::vector<scalar>&              vsf,
    const std::vector<std::vector<scalar>>& vsfBnd,
    scalar                                  k,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches);

void cellLimitGrad(
    std::vector<vector>&          grad,   // in/out, per cell
    const GeometricField<scalar>& vsf,
    scalar                        k,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

void cellLimitGrad(
    std::vector<tensor>&          grad,   // in/out, per cell
    const GeometricField<vector>& vsf,
    scalar                        k,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

} // namespace cpu
} // namespace brae
