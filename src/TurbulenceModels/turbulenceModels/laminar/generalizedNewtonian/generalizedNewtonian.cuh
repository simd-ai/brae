#pragma once
// generalizedNewtonian (powerLaw) on the device -- the CUDA twin of generalizedNewtonian_cpp.cuh, which
// carries the OpenFOAM provenance and the semantics. Same formula through the same BRAE_HD lines
// (powerLawNu, strainRate); what differs is only where the gradient comes from.
//
// provenance:
//   openfoam: src/TurbulenceModels/turbulenceModels/laminar/generalizedNewtonian/generalizedNewtonian.C
//             (see generalizedNewtonian_cpp.cuh)
//   brae:
//     reference: generalizedNewtonian_cpp.cu
//     consumer:  src/applications/solvers/rhoSimpleFoam/rhoSimpleFoamDriver.cu (turbulence->correct())
//     tests:     tests/rho_generalized_newtonian_vs_openfoam.sh (ARMS="1 cuda")
//
// THE GRADIENT OpenFOAM's strainRate() takes is fvc::grad(U) with U's patches as they STAND -- the values
// the last evaluate left, which on the mirror arm are the stored boundary arrays (RhoSolverFields::U*Bnd),
// not a fresh deviceBCValue. Its patch values then come from gaussGrad::correctBoundaryConditions, the
// same boundary tensor deviceDivDevReff builds for the dev2 term (deviceBoundaryGradU).
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "generalizedNewtonian_cpp.cuh"   // PowerLawCoeffs, powerLawNu, strainRate

namespace brae {
namespace gpu {
namespace generalizedNewtonian {

// nu_ = viscosityModel_->nu(this->nu(), strainRate()), cells (nC) and every boundary face in the device's
// flat layout (nBndFaces).
//   UbStored     U's stored boundary values, one buffer per component; null re-derives with deviceBCValue
//   gradULimitK  cellLimited coefficient of gradSchemes/grad(U) (0 = Gauss linear)
//   nu0/nu0Bnd   this->nu() = mu/rho_ per cell and per boundary face
void correctNu(
    const DeviceMesh&                                dm,
    const DeviceVectorBoundary&                      dbU,
    const DeviceBuffer<scalar>&                      Ux,
    const DeviceBuffer<scalar>&                      Uy,
    const DeviceBuffer<scalar>&                      Uz,
    const DeviceBuffer<scalar>* const*               UbStored,
    scalar                                           gradULimitK,
    const DeviceBuffer<scalar>&                      nu0,
    const DeviceBuffer<scalar>&                      nu0Bnd,
    const cpu::generalizedNewtonian::PowerLawCoeffs& coeffs,
    DeviceBuffer<scalar>&                            nu,
    DeviceBuffer<scalar>&                            nuBnd);

} // namespace generalizedNewtonian
} // namespace gpu
} // namespace brae
