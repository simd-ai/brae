#pragma once
// Coarse-grid single-launch solvers -- INTERNAL interface (the two public deviceCoarse* are in device_amg.cuh).
// The V-cycle (device_amg.cu) dispatches to these at the coarsest level; the definitions live in device_amg_coarse.cu.
#include "device_ldu.cuh"     // DeviceLduView
#include "device_buffer.cuh"  // DeviceBuffer
#include "cf_types.cuh"       // scalar

namespace brae {
void deviceCoarsePCG(const DeviceLduView& cv, const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc, int nIters);
// The ASYMMETRIC twin of deviceCoarsePCG, for a coarsest level whose operator has upper != lower (the
// transonic pressure equation's fvm::div(phid, p)). CG is not a solve there -- its alpha = (r.z)/(p.Ap)
// presumes p.Ap is the A-norm of p, which only holds for a symmetric A -- so this runs BiCGStab instead,
// which is OpenFOAM's own choice for an asymmetric coarsest level (GAMGSolver.C:299-328 constructs a
// PBiCGStab for `matrixLevels_[coarsestLevel].asymmetric()`). Same single-launch shape as
// deviceCoarsePCG: one block, shared-memory vectors, a fixed iteration count, no host sync.
void deviceCoarseBiCGStab(const DeviceLduView& cv, const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc, int nIters);
// DIRECT coarsest solve: dense LU with partial pivoting, OpenFOAM's own `directSolveCoarsest`
// alternative at this level (GAMGSolver.C:266-278 -> LUscalarMatrix). The factorisation is done once per
// Galerkin update (amgGalerkin, the one point where the coarse coefficients change) and each V-cycle then
// pays only the two substitutions. Being exact, it makes the V-cycle a fixed linear operator by
// construction -- the property the iterative twins have to reach COARSE_REL_TOL to earn.
void deviceCoarseLUFactor(const DeviceLduView& cv, DeviceBuffer<scalar>& lu, DeviceBuffer<int>& piv);
void deviceCoarseLUSolve(int nC, const DeviceBuffer<scalar>& lu, const DeviceBuffer<int>& piv,
                         const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc);
void deviceCoarseJacobiSingleBlock(const DeviceLduView& cv, const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc, int nSweeps);
} // namespace brae
