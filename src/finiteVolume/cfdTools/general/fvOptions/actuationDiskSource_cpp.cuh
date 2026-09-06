#pragma once
// fv::actuationDiskSource, Froude variant -- host reference.
//
// provenance:
//   openfoam: src/fvOptions/sources/derived/actuationDiskSource/actuationDiskSource.C
//             .../actuationDiskSourceTemplates.C (calcFroudeMethod)
//   brae:     src/finiteVolume/cfdTools/general/fvOptions/actuationDiskSource_cpp.cu
//   tests:    tests/actuationdisk_vs_openfoam.sh
//
// The whole model, per turbine, per iteration:
//
//   Uref = mean(U) over the MONITOR cells (the cells containing the case's upstream points)
//   Ct   = sink*Ct,  Cp = sink*Cp        sink `true` -> +1, `false` -> -1
//   a    = 1 - Cp/Ct                      the axial induction factor
//   T    = 2*rhoRef*diskArea*(Uref & diskDir)^2 * a*(1 - a)
//   source[c] += (V[c]/Vdisk) * T * diskDir      for each DISK cell
//
// so the thrust is distributed by cell volume over the disk and points ALONG diskDir.
//
// THE SIGN, and it is NOT the one this function writes. `source` here is the OPTION matrix's source, the
// one calcFroudeMethod fills -- not the momentum equation's. simpleFoam then assembles
// `UEqn == fvOptions(U)`, and the free operator== is `UEqn - fvOptions(U)` (fvMatrix.C), so the MOMENTUM
// source LOSES what is added here. That minus is what makes the disk a turbine rather than a propeller:
// T is positive and diskDir points downstream. Applying it the other way still converges -- turbineSiting
// simply settles on a wind field ~15% too fast at the turbines -- so the caller (UEqn / actuation_disk.cu)
// carries the minus explicitly and tests/turbinesiting_cuda_vs_openfoam.sh is what holds it there.
//
// EACH TURBINE IS INDEPENDENT: its own monitor cells, its own Uref, its own thrust, its own disk cells.
// turbineSiting ships two. Sharing one mask between them would apply both thrusts to the union of their
// cells and converge to a plausible but wrong wind field.
//
// SCOPE: the Froude variant with constant Cp/Ct. OpenFOAM also has `variableScaling` and Function1
// Cp/Ct tables, which are refused rather than approximated.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_options.cuh"   // FvOptionsData::ActuationDisk, shared with the device path

#include <vector>

namespace brae {
namespace cpu {

// What one turbine contributes this iteration, kept so a gate can compare it against the numbers
// OpenFOAM writes to postProcessing/.../actuationDiskSource.dat.
struct ActuationDiskState
{
    vector Uref{0, 0, 0};
    scalar a = 0, T = 0;
};

// OF calcFroudeMethod. `source` is the momentum equation's extensive source.
ActuationDiskState addSupFroude(
    const FvOptionsData::ActuationDisk& d,
    const std::vector<vector>&          U,
    const std::vector<scalar>&          V,
    std::vector<vector>&                source);

} // namespace cpu
} // namespace brae
