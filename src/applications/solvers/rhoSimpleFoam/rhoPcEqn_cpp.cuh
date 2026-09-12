#pragma once
// _cpp REFERENCE -- host transcription of rhoSimpleFoam's SIMPLEC pressure equation, BOTH branches.
//
// provenance:
//   openfoam:
//     file: applications/solvers/compressible/rhoSimpleFoam/pcEqn.H:1-123
//     also: src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixATmul.C  (lduMatrix::H1)
//           src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C             (fvMatrix::H1 over it)
//   brae:
//     reference: src/applications/solvers/rhoSimpleFoam/rhoPcEqn_cpp.cu
//     cuda:      (pending -- the whole case runs in _cpp first, see PORT.md)
//     tests:     tests/test_rho_pceqn_cpp.cu
//
// SELECTED BY `consistent yes`. The driver picks this file over pEqn.H, and the two are NOT the same
// equation with a different diffusivity. Against pEqn.H, pcEqn.H differs in six places:
//
//   1. `rho = thermo.rho()` runs at the TOP. pEqn.H updates rho only at the END. So the SIMPLEC pressure
//      equation is built from a density that already reflects the just-solved p and T, and the plain
//      SIMPLE one is not.
//   2. rAtU = 1/(1/rAU - UEqn.H1()) -- the consistent term, and the whole point of SIMPLEC.
//   3. The laplacian diffusivity is `rhorAtU` = rho*rAtU as a VOLUME field, where pEqn.H passes the
//      surface field `rhorAUf` = fvc::interpolate(rho*rAU). Two differences in one line: rAtU for rAU,
//      and a volume field that fvm::laplacian interpolates itself.
//   4. BOTH branches add the SIMPLEC flux correction
//          phiHbyA += fvc::interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)*mesh.magSf()
//      which has no counterpart in pEqn.H at all.
//   5. BOTH branches correct HbyA: `HbyA -= (rAU - rAtU)*fvc::grad(p)`.
//   6. The velocity corrector afterwards uses rAtU, not rAU.
//
// ORDER IS LOAD-BEARING IN BOTH BRANCHES, and differently:
//
//   transonic:  phid is built from the UNCORRECTED phiHbyA, and only then does phiHbyA receive the
//               SIMPLEC correction and the psi*p subtraction -- in ONE statement, so there is no
//               intermediate state where only one of them has been applied.
//   subsonic:   adjustPhi runs FIRST and the SIMPLEC correction is added AFTER it. adjustPhi balances the
//               flux it is handed, so correcting first would have it balance a different flux and scale
//               the outflow by a different factor.
//
// THE SIMPLEC CORRECTION IS THE SHAPE THAT COST REAL TIME IN simpleFoam, where getting `rAtU = 1/(1/rAU -
// H1())` right but omitting the matching `(rAtU - rAU)` flux correction made the converged velocity WORSE
// than doing neither. The two halves are one change; this file carries both or neither.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvc.cuh"
#include "rhoPEqn_cpp.cuh"      // PressureInput / adjustPhi -- the same inputs, the same utility
#include <vector>

namespace brae {
namespace cpu {
namespace rhoSimple {

// Every intermediate of pcEqn.H, in the order OpenFOAM produces them.
struct ConsistentPressureStages
{
    std::vector<scalar> rAU;        // 1/UEqn.A()
    std::vector<scalar> rAtU;       // 1/(1/rAU - UEqn.H1())   -- SIMPLEC
    std::vector<scalar> rhorAtU;    // rho*rAtU, a VOLUME field: fvm::laplacian interpolates it
    std::vector<vector> HbyA0;      // constrainHbyA(rAU*UEqn.H(), U, p), BEFORE its correction
    std::vector<vector> HbyA;       // after  HbyA -= (rAU - rAtU)*fvc::grad(p)
    SurfaceScalarField  phiHbyA0;   // fvc::interpolate(rho)*fvc::flux(HbyA0), BEFORE either branch
    SurfaceScalarField  phiHbyA;    // after the branch's corrections
    SurfaceScalarField  phid;       // transonic only, built from the UNCORRECTED phiHbyA
    bool                closedVolume = false;   // subsonic only
    bool                transonic    = false;
};

// UEqn.H1() -- the row sum of the off-diagonal coefficients, per unit volume. Exposed because rAtU is
// meaningless without it and because it is the one quantity in SIMPLEC with no counterpart in SIMPLE.
std::vector<scalar> momentumH1(
    const FvVectorMatrix&       UEqn,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

// Stages 1-5 of pcEqn.H: rAU, rAtU, rhorAtU, HbyA, phiHbyA, and the branch's own corrections. Solves
// nothing. `rho` is expected to be the field pcEqn.H starts from -- that is, AFTER its opening
// `rho = thermo.rho()`, which is the caller's business because it is a thermo call, not a pressure one.
ConsistentPressureStages consistentPressurePredictor(
    const FvVectorMatrix&         UEqn,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const PressureInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// Stage 6: assemble fvc::div(phiHbyA) [+ fvm::div(phid,p)] - fvm::laplacian(rhorAtU, p), relax on the
// transonic branch only, and apply setReference.
FvScalarMatrix assemblePcEqn(
    const ConsistentPressureStages& st,
    const GeometricField<scalar>&   p,
    const PressureInput&            in,
    const PrimitiveMesh&            m,
    const FvGeometry&               g,
    const std::vector<FvPatch>&     patches);

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
