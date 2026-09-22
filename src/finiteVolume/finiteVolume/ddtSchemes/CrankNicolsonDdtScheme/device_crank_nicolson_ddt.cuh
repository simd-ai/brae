#pragma once
// OpenFOAM's CrankNicolson ddt scheme on the device: fvm::ddt(rho, vf) and fvc::ddtCorr(U, phi) on a
// mesh that does not move, transcribed from the host reference (crank_nicolson_ddt_scheme_cpp.cuh,
// which carries the provenance and the algorithm) term for term. The ddt0 state lives on the device;
// its two time indices and the coefficients they select live on the host, as the reference has them.
#include "cf_types.cuh"
#include "crank_nicolson_ddt_scheme_cpp.cuh"   // CrankNicolsonClock
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include <string>

namespace brae {

// DDt0Field<GeoField> on the device: a scalar (one component) or a vector (three), its cells -- or
// internal faces, for a surface field -- and its boundary faces
struct DeviceCnDdt0
{
    std::string name;
    int nComp = 1;
    DeviceBuffer<scalar> internal[3];
    DeviceBuffer<scalar> boundary[3];
    label startTimeIndex = 0;
    label timeIndex = 0;
    bool exists = false;

    void lookupOrCreate(
        const cpu::fv::CrankNicolsonClock& clock,
        int nComponents,
        std::size_t nInternal,
        std::size_t nBoundary);
    bool evaluate(const cpu::fv::CrankNicolsonClock& clock)
    {
        const bool evaluated = (timeIndex != clock.timeIndex);
        timeIndex = clock.timeIndex;
        return evaluated;
    }
    scalar coef(const cpu::fv::CrankNicolsonClock& clock) const
    {
        return (clock.timeIndex > startTimeIndex) ? scalar(1) + clock.ocCoeff : scalar(1);
    }
    scalar coef0(const cpu::fv::CrankNicolsonClock& clock) const
    {
        return (clock.timeIndex > startTimeIndex + 1) ? scalar(1) + clock.ocCoeff : scalar(1);
    }
    scalar rDtCoef(const cpu::fv::CrankNicolsonClock& clock) const { return coef(clock)/clock.deltaT; }
    scalar rDtCoef0(const cpu::fv::CrankNicolsonClock& clock) const { return coef0(clock)/clock.deltaT0; }
};

// fvm::ddt(rho, vf), static mesh, added INTO diag and the per-component sources. `rho`, `rhoOld` and
// `rhoOO` null together is fvm::ddt(vf). `nComp` is 1 (k, epsilon) or 3 (U); the arrays are indexed
// by component, and unused slots may be null.
void deviceCnFvmDdt(
    const cpu::fv::CrankNicolsonClock& clock,
    DeviceCnDdt0& ddt0,
    const DeviceBuffer<scalar>* rho,
    const DeviceBuffer<scalar>* rhoOld,
    const DeviceBuffer<scalar>* rhoOO,
    int nComp,
    const DeviceBuffer<scalar>* const* vfOld,
    const DeviceBuffer<scalar>* const* vfOO,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>* const* src);

// fvc::ddtCorr(U, phi), static mesh, no coupled patch (the device's boundary arrays hold none, and the
// caller refuses a pair under CrankNicolson by name). `bndUFixesValue` is 1 on every boundary face
// whose patch fixes U. `UOldBnd`/`UOOBnd` are the STORED patch values of U's two old levels.
void deviceCnDdtCorr(
    const DeviceMesh& dm,
    const cpu::fv::CrankNicolsonClock& clock,
    DeviceCnDdt0& ddt0,
    DeviceCnDdt0& dphidt0,
    const DeviceBuffer<scalar>* const* UOld,
    const DeviceBuffer<scalar>* const* UOO,
    const DeviceBuffer<scalar>* const* UOldBnd,
    const DeviceBuffer<scalar>* const* UOOBnd,
    const DeviceBuffer<scalar>& phiOldInt,
    const DeviceBuffer<scalar>& phiOldBnd,
    const DeviceBuffer<scalar>& phiOOInt,
    const DeviceBuffer<scalar>& phiOOBnd,
    const DeviceBuffer<int>& bndUFixesValue,
    scalar ddtPhiCoeff,
    DeviceBuffer<scalar>& outInt,
    DeviceBuffer<scalar>& outBnd);

}   // namespace brae
