#pragma once
// rhoTurbulenceHook.cuh -- the compressible closure's per-iteration inputs, built ON THE DEVICE.
//
// gpu::rhoSimple::rhoSimpleStep takes turbulence as a hook (StepInput::correct) because the model runs
// LAST in the iteration and the driver should not know which one it is. Building that hook's INPUTS was
// left to the caller, and the only caller was the CUDA test harness -- which built them on the host:
// seven device-to-host copies (T, rho, TBnd, rhoBnd, phi, phiBnd, nutBnd) and six host-to-device copies
// (nu cells, nu faces, nu walls, phiByRho int/bnd, the nut snapshot), EVERY ITERATION, on a case whose
// whole point is that it never leaves the GPU. A driver built on that would round-trip 13 arrays per
// iteration to run a solver that otherwise does not.
//
// So the inputs are built here, on the device, and the hook is shared: the harness and the runnable
// driver call the SAME function, for the same reason buildStepInput is shared on the host path -- a
// private copy in the driver means the gate measures one closure input set and the solver runs another.
//
// WHAT THE CLOSURE NEEDS, and where each comes from (all transcribed from the host driver, which is the
// gated reference -- rhoSimpleFoam_cpp.cu):
//
//   nu (cells, boundary faces)  the LAMINAR kinematic viscosity mu(T)/rho, which varies cell by cell on
//                               a compressible case where the incompressible lineage has one number.
//   nuWall (wall faces)         the same values gathered into WALL-face order, which is not
//                               boundary-face order (RhoDeviceFields::wfFaceOfBnd is the map).
//   phiByRho                    compressibleTurbulenceModel::phi(), the VOLUMETRIC flux: internal faces
//                               divided by fvc::interpolate(rho), boundary faces by the PATCH rho --
//                               effectiveFaceViscosity's semantics, which replaces the interpolated
//                               boundary with the field's own boundary values.
//   nutBndIn                    a SNAPSHOT of the entering wall viscosity. Not nutBnd itself: nutBnd is
//                               also the output correct() overwrites, so aliasing them makes the closure
//                               read a value it has already replaced partway through.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_boundary.cuh"
#include "device_mesh.cuh"
#include "kepsilon_coeffs.cuh"
#include "kEpsilon.cuh"
#include "rhoCreateFields.cuh"
#include "rhoSimpleFoam.cuh"
#include "thermo_types.cuh"
#include <string>

namespace brae {
namespace gpu {
namespace rhoSimple {

// The scratch the hook writes into, owned by the CALLER so it survives the loop: allocating six device
// buffers per iteration is a cudaMalloc storm on a solver that otherwise allocates nothing per step.
struct TurbulenceHookBuffers
{
    DeviceBuffer<scalar> nuCell, nuBnd, nuWall;
    DeviceBuffer<scalar> phiByRhoInt, phiByRhoBnd;
    DeviceBuffer<scalar> nutBndIn;
    DeviceBuffer<scalar> rhoFace;        // fvc::interpolate(rho) on the internal faces
    DeviceBuffer<label>  wallFaceOfBnd;  // uploaded once, on the first call
    kEpsilonRAS::KEpsilonStages stages;
};

// The case's turbulence settings, as the closure takes them. Filled from the SAME parse the host step
// uses (cpu::rhoSimple::StepInput), so a scheme or relaxation the case names cannot reach one arm only.
struct TurbulenceHookOptions
{
    KEpsilonCoeffs co{};
    scalar         Prt = 1.0;
    bool           bounded = false;
    bool           correctedLaplacian = false;
    std::string    divSchemeUnsupported;   // non-empty -> the closure refuses by name
    bool           relaxEquationK = true,  relaxEquationEps = true;
    scalar         relaxK = 1.0, relaxEps = 1.0;
    scalar         tol = 1e-12, relTol = 0.0;
    int            maxIter = 2000;
    int            minIter = 0;
    // fvOptions.constrain(kEqn)/(epsEqn) -- kEpsilon.C calls it on both. A scalarFixedValueConstraint
    // naming k or epsilon pins those cells with fvMatrix::setValues, which is NOT the same as writing
    // the field afterwards: setValues also removes the coupling from the neighbours' equations.
    const DeviceBuffer<label>*  fvoKMask   = nullptr;
    const DeviceBuffer<scalar>* fvoKVal    = nullptr;
    const DeviceBuffer<label>*  fvoEpsMask = nullptr;
    const DeviceBuffer<scalar>* fvoEpsVal  = nullptr;
};

// One turbulence correction, inputs and all, without leaving the device.
void correctTurbulence(
    RhoSolverFields&              f,
    const RhoDeviceFields&        dev,
    const DeviceMesh&             dm,
    DeviceVectorBoundary&         dbU,
    const ThermoCoeffs&           thermo,
    const TurbulenceHookOptions&  opt,
    TurbulenceHookBuffers&        buf);

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
