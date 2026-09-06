#pragma once
// Foam::bound -- the lower-bounding every RAS model applies to k, epsilon and omega after its solve.
//
// provenance:
//   openfoam: src/finiteVolume/cfdTools/general/bound/bound.C
//   brae:     src/finiteVolume/cfdTools/general/bound/bound_cpp.cu
//   tests:    tests/test_bound_cpp.cu
//
// THIS IS NOT A CLAMP, and the difference is not cosmetic. OpenFOAM writes
//
//     vsf = max(max(vsf, average(max(vsf, lowerBound))*pos0(-vsf)), lowerBound)
//
// so a cell whose solve came out NEGATIVE is replaced by the area-weighted average of its own
// neighbours, and only a cell that is merely small is floored. Flooring a negative omega to SMALL
// instead puts ~1e-15 in the denominator of the very next iteration's (F1 - 1)*CDkOmega/omega term,
// which is ~1e15 -- measured on pitzDailySST, omega went negative in 4 cells on iteration 1 and the
// run reached 1e+46 by iteration 200. Upwind convection never produces the negative cell, which is
// why a floor survives there and only fails once a limited scheme is selected.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include <vector>

namespace brae {
namespace cpu {

// Returns the pre-bound minimum, so a caller can report it the way OpenFOAM's Info line does.
scalar bound(
    GeometricField<scalar>&     vsf,
    scalar                      lowerBound,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

} // namespace cpu
} // namespace brae
