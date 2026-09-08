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
void deviceCoarseJacobiSingleBlock(const DeviceLduView& cv, const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc, int nSweeps);
} // namespace brae
