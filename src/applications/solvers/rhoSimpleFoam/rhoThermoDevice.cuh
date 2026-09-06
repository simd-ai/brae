#pragma once
// rhoThermoDevice.cuh -- the three per-iteration operations the CUDA driver takes as hooks, implemented
// so they never leave the device.
//
// WHY THIS FILE EXISTS. rhoSimpleStep takes thermoCorrect, updateRho and turbulence->correct() as
// std::function hooks, and the argument for that is sound: `rho = thermo.rho()` is a thermo operation
// and a driver that hard-codes one has learned to be a thermodynamic model. But a hook is only as
// device-resident as its implementation, and the only implementations that existed were the driver
// gate's, which copy the whole field set to the host, run the _cpp reference, and copy it back. That is
// correct and is deliberately what the gate wants -- both sides must share one thermo, or it would be
// comparing two thermos as well as two drivers -- but it is four field round-trips per iteration, and a
// solver built on it would spend the run on PCIe rather than in the kernels.
//
// So these are the same three operations against the device state directly. They are NOT a second
// thermodynamic model: every one of them evaluates the SAME BRAE_HD inline functions the host reference
// calls -- hConstHeToT, perfectGasPsi, perfectGasRho, transportMu, transportAlpha, thermoCpByCpv, all
// declared __host__ __device__ in thermophysicalModels/. There is one equation of state in the tree and
// both paths call it; what differs is only where the loop runs. That is the property the gate below
// leans on, and the reason a device/host disagreement here would be a wiring defect rather than a
// physics one.
//
// WHAT IS REFUSED. perfectGas + hConst + (const | sutherland) only, which is the scope the reference
// implements and the scope ThermoCoeffs describes. ThermoModel::liquidH2O replaces Cp, mu, kappa and rho
// with per-cell NSRDS correlations and inverts he -> T by Newton; device_thermo.cu already carries that
// path for the legacy solver, but wiring it here without a compressible liquid fixture to gate it would
// be exactly the silent substitution this project keeps finding. It throws and names itself.

#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_boundary.cuh"
#include "thermo_types.cuh"
#include "rhoSimpleFoam.cuh"

namespace brae {
namespace gpu {
namespace rhoSimple {

// basicThermo::correct() -- EEqn.H's closing line. T from the energy variable, then psi from THAT T, then
// the thermo's own rho from the p and T it sees at correct() time (heRhoThermo.C:88). Doing only the
// first half leaves the pressure equation carrying a psi that belongs to the previous iteration.
//
// The BOUNDARY is not recomputed from he: T's patch values come from T's own boundary conditions, which
// is what basicThermo::correct() leaves in place -- it recalculates the INTERNAL field and lets the
// conditions stand. psi and the thermo rho on the boundary follow that T. dbT is therefore evaluated
// here, and it is why the driver's updateBoundaryCoeffs refreshes dbT's flux switch even though nothing
// in the driver itself reads it.
void thermoCorrect(
    RhoSolverFields&      f,
    const DeviceBoundary& dbT,
    const ThermoCoeffs&   c);

// `rho = thermo.rho()`, cells AND boundary, and WHICH rho that is depends on the thermo type:
//   hePsiThermo::rho()  ->  p_*psi_   recomputed with the pressure that was just solved
//   heRhoThermo::rho()  ->  rho_      the STORED field, from before the pressure equation ran
// Recomputing it live for heRhoThermo is wrong wherever p moves appreciably in one iteration; the
// reference's own comment carries the angledDuct measurement that established it.
void updateRho(
    RhoSolverFields&    f,
    const ThermoCoeffs& c);

// muEff and alphaEff, cells and boundary faces, for the iteration about to run.
//
//   laminar     muEff = mu(T)              alphaEff = CpByCpv*alpha(T)
//   turbulent   muEff = mu(T) + rho*nut    alphaEff = CpByCpv*(alpha(T) + alphat)
//
// alphaEff carries CpByCpv -- gamma for sensibleInternalEnergy, 1 for sensibleEnthalpy -- because OF's
// heThermo::alphaEff does; dropping it is a 40% error in the energy diffusivity for air, and one that
// converges quietly to the wrong temperature.
//
// transportAlpha takes the VISCOSITY, not the temperature: constTransport is mu/Pr and sutherland is the
// Eucken kappa/Cp built from mu. The boundary values are the PATCH's nut and alphat, never the adjacent
// cell's -- that distinction is the entire reason a wall function exists.
void effectiveTransport(
    const RhoSolverFields& f,
    const ThermoCoeffs&    c,
    bool                   turbulent,
    DeviceBuffer<scalar>&  muEff,
    DeviceBuffer<scalar>&  muEffBnd,
    DeviceBuffer<scalar>&  alphaEff,
    DeviceBuffer<scalar>&  alphaEffBnd);

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
