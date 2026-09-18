#pragma once
// The IMPLICIT UPWIND PRE-SOLVE on the device -- alphaEqn.H:103-155, the half of the alpha equation
// that only exists when `MULESCorr yes`.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/VoF/alphaEqn.H:103-155
//   host:      src/applications/solvers/interFoam/alpha_eqn_cpp.cu, the MULESCorr block in
//              alphaEqnStep -- the ORACLE, and part of the path that reaches 3.4346e-09 on damBreak
//              against real OpenFOAM (tests/interfoam_dambreak_vs_openfoam.sh).
//   tests:     tests/test_device_alpha_presolve.cu
//
// WHAT IT IS. OpenFOAM builds
//
//     fvScalarMatrix alpha1Eqn
//     (
//         EulerDdtScheme<scalar>(mesh).fvmDdt(alpha1)
//       + gaussConvectionScheme<scalar>(mesh, phiCN, upwind<scalar>(mesh, phiCN)).fvmDiv(phiCN, alpha1)
//      == Su + fvm::Sp(Sp + divU, alpha1)
//     );
//
// and solves it. THE CONVECTION IS UPWIND, NAMED EXPLICITLY IN OPENFOAM'S OWN CODE -- it is NOT the
// case's div(phi,alpha). That is the entire point of the split: the implicit half is first-order and
// unconditionally bounded, and every bit of the case's scheme lives in the correction CMULES then
// limits. Running the case's vanLeer here would make the pre-solve itself unbounded and leave CMULES
// correcting towards a field that had already overshot. Su, Sp and divU are all zeroField for
// interFoam (interFoam/alphaSuSp.H is three of them), so the right-hand side vanishes.
//
// alphaPhi10 IS THE MATRIX'S OWN FLUX, alpha1Eqn.flux(). What that does and does not mean, MEASURED in
// tests/test_device_alpha_presolve.cu rather than asserted:
//
//   ON THE INTERNAL FACES it is the same field as the upwind flux of the SOLVED alpha, bit for bit, and
//   necessarily so: faceH = upper*psi[nei] - lower*psi[own] with upwind's upper = min(phi,0) and
//   lower = -max(phi,0) collapses to phi*psi[upwind]. There is nothing to tell apart there. This file
//   said otherwise at first.
//
//   ON THE BOUNDARY it is not, wherever a patch contributes more than (1, 0): flux_b is
//   internalCoeffs*psi[faceCell] - boundaryCoeffs, so at a fixedValue patch it carries the PRESCRIBED
//   value where the field's own upwind flux would carry the cell's. Measured at 2.046e-03 on a flux of
//   scale 1.664e-03.
//
//   WHAT IT REALLY PINS is the closure alpha == alphaOld - (dt/V)*sum_f alphaPhi10, which holds to the
//   LINEAR SOLVER's residual and not to round-off -- 1.3e-12 at tolerance 1e-12 and 3.0e-06 at 1e-6,
//   tracking it. A flux that were not the matrix's leaves a DISCRETISATION-sized closure error that
//   does not move with the tolerance: the flux of the field the solve started from breaks it by
//   5.1e-03 and a central flux of the solved field by 2.3e-02.
//
// WHAT THE CALLER SUPPLIES, and it is the same split as everywhere else in this solver's device half:
// `iC` and `bC` are the matrix's internalCoeffs and boundaryCoeffs, flattened in boundary-face order.
// Those come from OpenFOAM's per-patch valueInternalCoeffs/valueBoundaryCoeffs -- branchy dispatch over
// inletOutlet, zeroGradient, fixedValue and empty over a few hundred faces -- so they are built on the
// host, exactly as alpha1's patch values are for deviceAlphaCorrector. Everything that scales with the
// CELL COUNT, which is the assembly of the internal LDU and the linear solve itself, runs here.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_pcg.cuh"   // DeviceSolverPerf

namespace brae {

// The case's own fvSolution entry for alpha. These are read, not defaulted: a case that asks for
// MULESCorr and omits them is not this file's to guess at.
struct DeviceAlphaSolverControls
{
    scalar tol     = 1e-8;
    scalar relTol  = 0;
    int    maxIter = 1000;
    // `solver smoothSolver; smoother symGaussSeidel|GaussSeidel;` -- what every MULESCorr tutorial names
    // for alpha. Read by the alpha pre-solve only; the momentum and pressure entries that share this
    // struct ignore it. See deviceAlphaPreSolve for why it is not interchangeable with BiCGStab.
    bool smoothSolver = false;
    // symGaussSeidel (ascending then descending) against GaussSeidel (ascending only)
    bool symmetric = true;
    // smoothSolver.C:78 -- sweeps between residual evaluations
    int nSweeps = 1;
};

// `alpha1` goes in as the initial guess -- the sub-step's old time, which is what OpenFOAM hands the
// solver -- and comes out solved. `alphaPhi10Int`/`alphaPhi10Bnd` come out as alpha1Eqn.flux().
//
// Returns the solver's final normalised residual, so a caller can refuse a pre-solve that did not
// converge rather than carry a half-solved alpha into the corrector.
scalar deviceAlphaPreSolve(
    const DeviceMesh&             dm,
    DeviceBuffer<scalar>&         alpha1,
    const DeviceBuffer<scalar>&   alpha1Old,
    const DeviceBuffer<scalar>&   phiCNInt,
    const DeviceBuffer<scalar>&   iC,          // internalCoeffs, boundary-face order
    const DeviceBuffer<scalar>&   bC,          // boundaryCoeffs
    scalar                        deltaT,
    const DeviceAlphaSolverControls& sc,
    DeviceBuffer<scalar>&         alphaPhi10Int,
    DeviceBuffer<scalar>&         alphaPhi10Bnd,
    // the solver's own report -- initial and final residual and the iteration count; null = not kept
    DeviceSolverPerf*             perfOut = nullptr);

} // namespace brae
