#pragma once
// Coarse-grid single-launch solvers -- INTERNAL interface (the two public deviceCoarse* are in device_amg.cuh).
// The V-cycle (device_amg.cu) dispatches to these at the coarsest level; the definitions live in device_amg_coarse.cu.
#include "device_ldu.cuh"     // DeviceLduView
#include "device_buffer.cuh"  // DeviceBuffer
#include "cf_types.cuh"       // scalar

namespace brae {
void deviceCoarsePCG(const DeviceLduView& cv, const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc, int nIters);
void deviceCoarseJacobiSingleBlock(const DeviceLduView& cv, const DeviceBuffer<scalar>& rc, DeviceBuffer<scalar>& xc, int nSweeps);
} // namespace brae
