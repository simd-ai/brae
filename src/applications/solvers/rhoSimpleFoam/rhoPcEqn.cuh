#pragma once
// CUDA implementation of rhoSimpleFoam's SIMPLEC pressure equation -- the device twin of rhoPcEqn_cpp.
//
// provenance:
//   openfoam:  applications/solvers/compressible/rhoSimpleFoam/pcEqn.H:1-123
//     also:    src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixATmul.C  (lduMatrix::H1)
//   reference: src/applications/solvers/rhoSimpleFoam/rhoPcEqn_cpp.cu  (gated by
//              tests/rho_pceqn_vs_openfoam.sh against OpenFOAM's own dumps, both branches)
//   cuda:      src/applications/solvers/rhoSimpleFoam/rhoPcEqn.cu
//   tests:     tests/test_rho_pceqn_cuda.cu
//
// SELECTED BY `consistent yes`, and it is NOT pEqn.H with a different diffusivity. The reference header
// enumerates six differences; the four that shape this file are:
//
//   1. rAtU = 1/(1/rAU - UEqn.H1()) -- the consistent term, and the whole point of SIMPLEC.
//   2. The laplacian diffusivity is rho*rAtU, and its BOUNDARY is rho's PATCH value times rAtU's
//      OWNER-CELL value -- the same mixed form rhoPEqn's rhorAUf has, so the FACE boundary-coefficient
//      variant is required, not the cell one.
//   3. BOTH branches add the SIMPLEC flux correction
//          phiHbyA += fvc::interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)*mesh.magSf()
//      which has no counterpart in pEqn.H at all. It is built here the way the incompressible twin
//      builds it: as the FLUX of a laplacian whose gamma is rho*(rAtU - rAU), because gamma_f*snGrad(p)*
//      magSf IS that flux -- one code path instead of a second snGrad.
//   4. BOTH branches then correct HbyA: HbyA -= (rAU - rAtU)*fvc::grad(p).
//
// ORDER IS LOAD-BEARING, AND DIFFERENTLY IN EACH BRANCH:
//
//   transonic:  phid is built from the UNCORRECTED phiHbyA, and only then does phiHbyA receive the
//               SIMPLEC correction and the psi*p subtraction -- in ONE statement, so no intermediate
//               state exists where only one of them has been applied.
//   subsonic:   adjustPhi runs FIRST and the correction is added AFTER. adjustPhi balances the flux it is
//               handed, so correcting first would have it balance a different flux and scale the
//               adjustable outflow by a different factor.
//
// THE TWO HALVES OF SIMPLEC ARE ONE CHANGE. In simpleFoam, getting rAtU right while omitting the matching
// (rAtU - rAU) flux correction made the converged velocity WORSE than doing neither. This file carries
// both or neither, and the gate asserts each moves the answer.
//
// SOLVER, as in rhoPEqn.cuh and for the same reason: the subsonic branch is symmetric (its only implicit
// term is fvm::laplacian) so AMG-PCG and the cached hierarchy apply; the transonic branch adds
// fvm::div(phid, p) and is asymmetric, so it must go to BiCGStab. That is a correctness switch.
//
// REFUSED, not ignored: MRF.makeRelative, any unported fvOptions, fixedFluxPressure, coupled patches.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "pEqn.cuh"        // PressureMatrix, and the shared correctFlux/correctVelocity stages
#include "rhoUEqn.cuh"     // MomentumMatrix -- pcEqn.H consumes UEqn.A(), UEqn.H() and UEqn.H1()
#include "rhoPEqn.cuh"     // RhoPressureInput -- the SAME inputs; SIMPLEC changes the equation, not them
#include <string>

namespace brae {
namespace gpu {
namespace rhoSimple {

// Every intermediate of pcEqn.H, kept separate so a gate can name the first divergent stage.
struct ConsistentPressureStages
{
    DeviceBuffer<scalar> rAU;          // 1/UEqn.A()
    DeviceBuffer<scalar> rAtU;         // 1/(1/rAU - UEqn.H1())  -- SIMPLEC
    DeviceBuffer<scalar> rhorAtU;      // rho*rAtU, cells
    DeviceBuffer<scalar> rhorAtUf;     // interpolated to internal faces -- the laplacian diffusivity
    DeviceBuffer<scalar> rhorAtUfBnd;  // rhoBnd*rAtU[faceCell] -- the mixed boundary form, see the header
    DeviceBuffer<scalar> HbyA0[3];     // constrainHbyA(rAU*UEqn.H(), U, p), BEFORE its correction
    DeviceBuffer<scalar> HbyA[3];      // after HbyA -= (rAU - rAtU)*grad(p)
    DeviceBuffer<scalar> HbyAb[3];     // boundary faces, after constrainHbyA
    DeviceBuffer<scalar> phiHbyA0Int, phiHbyA0Bnd;   // BEFORE either branch
    DeviceBuffer<scalar> phiHbyAInt,  phiHbyABnd;    // after the branch's corrections
    DeviceBuffer<scalar> phidInt,     phidBnd;       // transonic only, from the UNCORRECTED phiHbyA

    bool   closedVolume = false;   // subsonic only
    bool   transonic    = false;
    scalar massCorr     = 1.0;
};

// Stages 1-5 of pcEqn.H: rAU, rAtU, rhorAtU, HbyA, phiHbyA and the branch's own corrections. Solves
// nothing.
//
// `rho` is expected to be the field pcEqn.H starts from -- that is, AFTER its opening
// `rho = thermo.rho()`. pEqn.H updates rho only at the END, so the SIMPLEC pressure equation is built
// from a density reflecting the just-solved p and T and the plain SIMPLE one is not. That call is the
// caller's business because it is a thermo operation, not a pressure one.
void consistentPressurePredictor(
    ConsistentPressureStages&   st,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary&       dbP,
    const MomentumMatrix&       UEqn,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& p,
    const RhoPressureInput&     in);

// Stage 6: fvc::div(phiHbyA) [+ fvm::div(phid,p)] - fvm::laplacian(rhorAtU, p), relaxed on the transonic
// branch only, then setReference.
void assemblePcEqn(
    PressureMatrix&                 P,
    const ConsistentPressureStages& st,
    const DeviceMesh&               dm,
    const DeviceBoundary&           dbP,
    const DeviceBuffer<scalar>&     p,
    const RhoPressureInput&         in);

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
