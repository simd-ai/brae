#pragma once
// The INTERIOR of displacementLaplacian's equation assembled on the device -- the host assembly
// (displacement_laplacian_fv_motion_solver_cpp.cu), in parallel, to the bit.
//
// provenance:
//   openfoam:  src/fvMotionSolver/fvMotionSolvers/displacement/laplacian/displacementLaplacianFvMotionSolver.C
//              (solve: fvm::laplacian(diffusivity, cellDisplacement, "laplacian(diffusivity,cellDisplacement)"))
//              src/finiteVolume/finiteVolume/laplacianSchemes/gaussLaplacianScheme/gaussLaplacianScheme.C
//              src/finiteVolume/finiteVolume/gradSchemes/gaussGrad/gaussGrad.C
//   host:      fvm.cuh (laplacian, laplacianCorrFlux, laplacianNonOrthSource) and fvc.cu (gaussGrad) -- the ORACLE
//   tests:     tests/interfoam_write/identity/motion_assembly_device.sh
//
// WHAT IS HERE: the face coefficients and the diagonal they sum to, Gauss-linear grad(cellDisplacement), the
// non-orthogonal correction's face flux and its per-cell sum. WHAT STAYS ON THE HOST: the patches' coefficients
// (fvm::laplacianBoundaryCoeffs, the same code fvm::laplacian runs), the source's sign, and the solve -- the
// case's own solver for cellDisplacement, which is not a pressure.
//
// THE ORDER IS THE HOST'S. The host walks the internal faces once, adding to the owner and to the neighbour, and
// then the patches; a cell therefore receives its faces in ASCENDING FACE NUMBER, the two sides interleaved, and
// its boundary faces after them. Each cell here folds its own faces in that order (the device mesh's ownerStart
// and losort lists, merged), so every sum is the host's sum, addend by addend.
//
// THE FUSED PRODUCTS ARE THE HOST'S, read off the host binary (aarch64 GCC contracts a*b + c):
//   gaussGrad          Uf = fma(w, Do - Dn, Dn); grad += Sf (x) Uf is fmla, a component at a time, and the
//                      neighbour's subtraction fmls; the division by V is a division
//   laplacianCorrFlux  gf = fma(w, gradO, (1 - w)*gradN); corr_j = fma(cz, gf_zj, fma(cx, gf_xj, cy*gf_yj));
//                      (gamma*magSf)*corr is two products
//   laplacian          (dc*gamma)*magSf is two products and the diagonal's subtraction is not fused
// Written with explicit fma calls, and the unfused ones kept apart in memory so the device compiler cannot fuse
// them either. The identity gate says they are still the host's bits after a compiler change.
//
// MEASURED on waveMakerPiston refined to 896,000 cells: the host's matrix, gradient and correction are 94 ms a
// step; this is 5.3, its three uploads and three downloads included. With BRAE_CONTROL_MOTION_ASSEMBLY_CHECK=1
// (the host assembles too and the motion solver compares bytes) not one bit differs over 50 steps of
// waveMakerFlap, nor over 5 of the 3-D waveMakerMultiPaddleFlap at 448,000 cells.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "fv_patch.cuh"
#include <vector>

namespace brae {

// the work arrays, kept across calls
struct DeviceDisplacementAssembly
{
    // the face diffusivity on the internal faces
    DeviceBuffer<scalar> gamma;
    // cellDisplacement, 3 a cell, and its boundary values, 3 a boundary face in the device mesh's order
    DeviceBuffer<scalar> D;
    DeviceBuffer<scalar> Db;
    DeviceBuffer<scalar> upper;
    // grad(cellDisplacement), 9 a cell
    DeviceBuffer<scalar> grad;
    // the correction's face flux, 3 an internal face, and its per-cell sum, 3 a cell
    DeviceBuffer<scalar> ffc;
    DeviceBuffer<scalar> corr;
    DeviceBuffer<scalar> diag;
};

// `gamma` is the diffusivity on the internal faces, `D` the cell field, `boundary` its values patch by patch (an
// empty patch's are not read). Returns fvm::laplacian's upper (= lower) and diag before the patches' diagonal,
// and laplacianNonOrthSource's per-cell sum, which the caller subtracts from the source. `dm` must hold the
// geometry the host assembly would read: the device loop refreshes it after every mesh move. Refuses a coupled
// patch -- the motion solver has none, and the correction's coupled faces are not here.
// BRAE_CONTROL_MOTION_ASSEMBLY_UNFUSED=1 rounds the gradient's products before adding them -- the identity
// gate's control: one rounding's difference, and the files differ.
void deviceDisplacementAssembly(
    const DeviceMesh& dm,
    const std::vector<FvPatch>& patches,
    const std::vector<scalar>& gamma,
    const std::vector<vector>& D,
    const std::vector<std::vector<vector>>& boundary,
    DeviceDisplacementAssembly& w,
    std::vector<scalar>& upper,
    std::vector<scalar>& diag,
    std::vector<vector>& corr);

} // namespace brae
