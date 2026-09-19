#pragma once
// OpenFOAM's one-equation LES model kEqn, in the incompressible lineage, the host reference.
//
// provenance:
//   openfoam:  src/TurbulenceModels/turbulenceModels/LES/kEqn/kEqn.C:44-52 (correctNut), :97-106
//                  (constructor: bound(k, kMin)), :130-189 (correct)
//              .../LES/kEqn/kEqn.H:152-157 (DkEff = nut + nu, no sigma)
//              .../LES/LESModel/LESModel.C:57-117 (Ce, kMin read from the LES dictionary itself, with
//                  defaults 1.048 and SMALL), :249-253 (correct: delta.correct() first)
//              .../eddyViscosity/eddyViscosity.C (validate: correctNut)
//   brae:      les_delta_cpp.cuh computes delta; this file assembles and solves k
//   tests:     tests/interfoam_les_vs_openfoam.sh against real OpenFOAM on LES/nozzleFlow2D
//
// THE EQUATION, with alpha = rho = 1 (incompressible::turbulenceModel, which is what interFoam builds when
// the case does not say `density variable`):
//
//     ddt(k) + div(phi, k) - laplacian(nut + nu, k)
//         == nut*(grad(U) && devTwoSymm(grad(U))) - SuSp(2/3*div(phi), k) - Sp(Ce*sqrt(k)/delta, k)
//
// then relax, solve, bound(k, kMin), and correctNut: nut = Ck*sqrt(k)*delta, nut's patches evaluated.
// G is taken with the nut the previous correctNut left; the Sp coefficient with k BEFORE the solve.
//
// NOT PORTED, refused by the caller that reads the case: the `density variable` lineage, fvOptions on k,
// a convection scheme on k other than upwind and limitedLinear, a gradient scheme other than Gauss linear.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "geometric_field.cuh"
#include "pcg.cuh"                  // SolverPerformance
#include "primitive_mesh.cuh"
#include "smooth_solver_cpp.cuh"    // LinearSolverChoice
#include <vector>

namespace brae {
namespace cpu {
namespace LESkEqn {

struct Coeffs
{
    // kEqnCoeffs, or the LES dictionary itself when there is none (coeffDict_ is optionalSubDict)
    scalar Ck = 0.094;
    // LESModel's, read from the LES dictionary and NOT from kEqnCoeffs
    scalar Ce = 1.048;
    scalar kMin = 1.0e-15;
    // `div(phi,k)`: upwind, or limitedLinear with its coefficient
    bool limitedLinear = false;
    scalar limitedLinearCoeff = 1;
    // laplacianSchemes and snGradSchemes, as the caller resolves them for every equation
    bool correctedLaplacian = false;
    scalar snGradLimitCoeff = 0;
};

// eddyViscosity::validate() -> kEqn::correctNut
void correctNut(
    const GeometricField<scalar>& k,
    const std::vector<scalar>& delta,
    const Coeffs& co,
    GeometricField<scalar>& nut);

struct Solve
{
    LinearSolverChoice which;
    scalar tol = 1e-6;
    scalar relTol = 0;
    int maxIter = 1000;
    int minIter = 0;
    // fvMatrix::relax() for kFinal: `on` false means OpenFOAM does not call relax at all
    bool relaxOn = false;
    scalar relax = 1;
};

// The stages tools/dumpKEqn writes from OpenFOAM's own kEqn, for a gate to compare one by one: G and
// divU, the convection matrix's off-diagonals, the diffusivity on the faces, the assembled matrix's
// diagonal and source (before relax), and k after the solve and before bound.
struct Taps
{
    std::vector<scalar> G;
    std::vector<scalar> divU;
    std::vector<scalar> divUpper;
    std::vector<scalar> divLower;
    std::vector<scalar> DkEfff;
    std::vector<scalar> diag;
    std::vector<scalar> source;
    std::vector<scalar> kSolved;
};

// kEqn::correct(). phi is the volumetric flux; nu and nuBnd the mixture's laminar viscosity on cells and
// patches; deltaT the time step (k.oldTime() is k as it enters).
SolverPerformance correct(
    const GeometricField<vector>& U,
    GeometricField<scalar>& k,
    GeometricField<scalar>& nut,
    const SurfaceScalarField& phi,
    const std::vector<scalar>& nu,
    const std::vector<std::vector<scalar>>& nuBnd,
    const std::vector<scalar>& delta,
    scalar deltaT,
    const Coeffs& co,
    const Solve& solve,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    Taps* taps = nullptr);

} // namespace LESkEqn
} // namespace cpu
} // namespace brae
