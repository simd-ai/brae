#pragma once
// OpenFOAM's CrankNicolson ddt scheme on the device: fvm::ddt(rho, vf) and fvc::ddtCorr(U, phi) on a
// mesh that does not move, transcribed from the host reference (crank_nicolson_ddt_scheme_cpp.cuh,
// which carries the provenance and the algorithm) term for term. The ddt0 state lives on the device;
// its two time indices and the coefficients they select live on the host, as the reference has them.
#include "cf_types.cuh"
#include "crank_nicolson_ddt_scheme_cpp.cuh"   // CrankNicolsonClock
#include "device_buffer.cuh"
#include "device_cyclic.cuh"
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

// THE PATCH HALF of fvm::ddt's ddt0, on the device's flat boundary-face arrays (the host's
// fv::CrankNicolsonDdt0Operands): the operands' STORED patch values at the two old levels. ddt0 is a whole
// GeometricField and OpenFOAM advances its patches beside its cells (CrankNicolsonDdtScheme.C:1040-1047
// moving, :1069-1073 static) -- the same expression on both branches, no volumes:
//     ddt0_b <- rDtCoef0*(rhoOld_b*vfOld_b - rhoOO_b*vfOO_b) - offCentre(ddt0_b)
// Nothing in the solve reads them; the field is WRITTEN with them. `rhoOld`/`rhoOO` null together is rho = 1.
struct DeviceCnDdt0PatchOperands
{
    const DeviceBuffer<scalar>* rhoOld = nullptr;
    const DeviceBuffer<scalar>* rhoOO = nullptr;
    const DeviceBuffer<scalar>* vfOld[3] = {nullptr, nullptr, nullptr};
    const DeviceBuffer<scalar>* vfOO[3] = {nullptr, nullptr, nullptr};
};

// fvm::ddt(rho, vf), added INTO diag and the per-component sources. `rho`, `rhoOld` and
// `rhoOO` null together is fvm::ddt(vf). `nComp` is 1 (k, epsilon) or 3 (U); the arrays are indexed
// by component, and unused slots may be null.
//
// V0 and V00 TOGETHER take the scheme's MOVING branch (CrankNicolsonDdtScheme.C:1029-1065), which the
// host reference carries and this transcribes: ddt0 weighted by the volume each old level belongs to
// and the source by V0 rather than V. The diagonal takes V either way -- that is the volume the new
// field lives in. Null together is the static branch.
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
    DeviceBuffer<scalar>* const* src,
    const DeviceBuffer<scalar>* V0 = nullptr,
    const DeviceBuffer<scalar>* V00 = nullptr,
    // the field's patches advanced beside its cells, when the caller keeps them (the writer's callers)
    const DeviceCnDdt0PatchOperands* patchOperands = nullptr);

// fvc::ddtCorr(U, Uf) on a MOVING mesh (fvcDdtUfCorr, CrankNicolsonDdtScheme.C:1201-1257),
// transcribed from the host reference. A DIFFERENT OPERATOR from the static twin below, not a variant
// of it:
//   * the flux side is `Sf & Uf.oldTime()`, not phi.oldTime() -- on a moving mesh those are two
//     numbers, because phi carries the mesh flux and Uf does not;
//   * the second ddt0 is a SURFACE VECTOR field (ddtCorrDdt0(Uf)), so it needs Uf.oldTime().oldTime();
//   * the interpolation is subtracted as a VECTOR and dotted with Sf once, rather than each term
//     being dotted separately -- the same arithmetic, in OpenFOAM's order.
// A periodic pair is refused by name: those faces are in neither array, and no moving case with one
// is gated on this arm.
void deviceCnDdtUfCorr(
    const DeviceMesh& dm,
    const cpu::fv::CrankNicolsonClock& clock,
    DeviceCnDdt0& ddt0,
    DeviceCnDdt0& dUfdt0,
    const DeviceBuffer<scalar>* const* UOld,
    const DeviceBuffer<scalar>* const* UOO,
    const DeviceBuffer<scalar>* const* UOldBnd,
    const DeviceBuffer<scalar>* const* UOOBnd,
    const DeviceBuffer<scalar>* const* UfOld,      // three components, internal faces
    const DeviceBuffer<scalar>* const* UfOldBnd,   // ...and boundary faces
    const DeviceBuffer<scalar>* const* UfOO,
    const DeviceBuffer<scalar>* const* UfOOBnd,
    const DeviceBuffer<int>& bndUFixesValue,
    scalar ddtPhiCoeff,
    DeviceBuffer<scalar>& outInt,
    DeviceBuffer<scalar>& outBnd,
    const DeviceCyclic* cyc = nullptr);

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
    DeviceBuffer<scalar>& outBnd,
    // ...and THE PERIODIC PAIR, whose faces are in neither array: its own dphidt0 level, phi's two
    // old levels there, and where to write. All null is a mesh with no pair; a pair with any of them
    // missing is refused, because the Euler twin would otherwise stand in silently.
    const DeviceCyclic* cyc = nullptr,
    DeviceCnDdt0* dphidt0If = nullptr,
    const DeviceBuffer<scalar>* phiOldIf = nullptr,
    const DeviceBuffer<scalar>* phiOOIf = nullptr,
    DeviceBuffer<scalar>* outIf = nullptr);

}   // namespace brae
