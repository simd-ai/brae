#pragma once
// OpenFOAM's one-equation LES model kEqn on the device -- a TRANSCRIPTION of the host reference,
// les_kEqn_cpp.cu, statement by statement and in its term order.
//
// provenance:
//   openfoam:  src/TurbulenceModels/turbulenceModels/LES/kEqn/kEqn.C:44-52 (correctNut), :130-189
//                  (correct); kEqn.H:152-157 (DkEff = nut + nu, no sigma)
//   host:      les_kEqn_cpp.cu -- the ORACLE, gated against real OpenFOAM on LES/nozzleFlow2D
//   tests:     tests/interfoam_les_vs_openfoam.sh, the device arm
//
// WHAT IS SHARED WITH THE RAS CLOSURES rather than written again here: the convection and diffusion
// half (turbulence::assembleScalarTransport, the same call kEpsilon's k equation makes), the
// production gradU && devTwoSymm(gradU) (deviceGByNuFromGradU), the smoothSolver sweep and the
// bounding. What is kEqn's own is the DIFFUSIVITY (nut + nu, with no sigma), the DESTRUCTION
// coefficient Ce*sqrt(k)/delta, and nut = Ck*sqrt(k)*delta.
//
// THE FILTER WIDTH is the host's: `delta` is computed once by LESdelta::compute on a mesh that does
// not move and uploaded by the caller. A moving mesh would have to recompute it, and the driver
// refuses that combination by name.
#include "cf_types.cuh"
#include "device_boundary.cuh"
#include "device_buffer.cuh"
#include "device_kepsilon.cuh"     // DeviceSolverPerf, deviceGByNuFromGradU, deviceGradUShared
#include "device_mesh.cuh"
#include "les_kEqn_cpp.cuh"        // cpu::LESkEqn::Coeffs -- one struct for both arms

namespace brae {
namespace gpu {
namespace LESkEqn {

struct Input
{
    const DeviceBuffer<scalar>* Ux = nullptr;
    const DeviceBuffer<scalar>* Uy = nullptr;
    const DeviceBuffer<scalar>* Uz = nullptr;
    const DeviceBuffer<scalar>* phiInt = nullptr;   // the VOLUMETRIC flux: alpha = rho = 1 here
    const DeviceBuffer<scalar>* phiBnd = nullptr;
    const DeviceBuffer<scalar>* nu = nullptr;       // the mixture's laminar viscosity, cells
    const DeviceBuffer<scalar>* nuBnd = nullptr;    // ...and patch faces
    const DeviceBuffer<scalar>* nutBnd = nullptr;   // nut's patch values, for DkEff's boundary half
    const DeviceBuffer<scalar>* delta = nullptr;    // the LES filter width, one per cell
    const DeviceBuffer<scalar>* kOld = nullptr;     // k.oldTime(), for fvm::ddt under Euler
    scalar rDeltaT = 0;

    cpu::LESkEqn::Coeffs co;

    // the smoothSolver the case names for k, and fvMatrix::relax when it asks for one
    bool   symmetric = true;
    int    nSweeps = 1;
    scalar tol = 1e-6;
    scalar relTol = 0;
    int    maxIter = 1000;
    int    minIter = 0;
    bool   relaxOn = false;
    scalar relax = 1;
};

// kEqn::correctNut: nut = Ck*sqrt(k)*delta. The PATCH values are the caller's -- nut's conditions are
// host objects, as they are for every closure on this loop.
void correctNut(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& delta,
    const cpu::LESkEqn::Coeffs& co,
    DeviceBuffer<scalar>&       nut);

// kEqn::correct(). `k` goes in as k.oldTime()'s field and comes out solved, bounded and with nut
// rebuilt from it.
DeviceSolverPerf correct(
    const DeviceMesh&           dm,
    DeviceBoundary&             dbK,
    const DeviceVectorBoundary& dbU,
    DeviceBuffer<scalar>&       k,
    DeviceBuffer<scalar>&       nut,
    const Input&                in);

} // namespace LESkEqn
} // namespace gpu
} // namespace brae
