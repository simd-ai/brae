#pragma once
// OpenFOAM's PBiCG with the DILU preconditioner, on the device.
//
// provenance:
//   openfoam: src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C (solve)
//             src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/DILUPreconditioner.C
//                 (precondition, preconditionT)
//   brae:     src/OpenFOAM/matrices/pbicg.cu -- the host reference this is transcribed from, loop for
//             loop; tests/interfoam_mangrove_vs_openfoam.sh holds that one to OpenFOAM's log
//   tests:    tests/test_device_pbicg.cu -- against the host reference
//
// NOT PBiCGStab (device_pcg.cuh's deviceJacobiBiCGStab). PBiCG carries a TRANSPOSE system beside the
// direct one -- Tmul and preconditionT -- and at the same tolerance the two stop at different iterates.
//
// THE TRANSPOSE IS A VIEW, not a second set of kernels. lduMatrix::Tmul is Amul with upper and lower
// exchanged (lduMatrixATmul.C), and DILUPreconditioner::preconditionT is precondition with the same
// exchange (DILUPreconditioner.C:126-170); calcReciprocalD reads only the product upper*lower, so the
// one rD serves both. So the transpose system runs deviceAmul and diluApply on a DeviceLduView whose
// upper and lower pointers are swapped.
//
// A COUPLED INTERFACE IS REFUSED, as the host reference refuses it: an interface's transpose
// coefficient is the OTHER side's, and nothing here exchanges them.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_dilu.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"   // DeviceSolverPerf

namespace brae {

// `dilu` must carry the level schedule of A's mesh (buildDeviceDilu); its rD is recomputed here from A,
// as OpenFOAM constructs the preconditioner inside the solve.
DeviceSolverPerf devicePBiCGDilu(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter,
    DeviceDilu& dilu);

} // namespace brae
