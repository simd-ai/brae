#pragma once
// Device DarcyForchheimer porosity (explicitPorositySource), evaluated each iteration from the current U.
// OF (incompressible) porosityModels::DarcyForchheimer::apply, mu=nu_laminar, rho=1:
//   Cd = nu*diag(d) + |U|*diag(0.5 f);  isoCd = tr(Cd)
//   diag[c]            += V*isoCd                       (implicit, isotropic part)
//   relaxSrc[comp][c]  -= V*((Cd - I*isoCd).U)[comp]    (explicit, anisotropic remainder)
// Diagonal d/f only (identity coordinateSystem). Cells are a cellZone (unique) -> no atomics needed.
#include "cf_types.cuh"
#include "device_boundary.cuh"   // DeviceBoundary, for the limitTemperature boundary clamp
#include "device_buffer.cuh"
#include "device_mesh.cuh"      // DeviceMesh -- deviceSetValues walks the ldu addressing

namespace brae {

struct DevicePorosity {
    bool                active = false;
    DeviceBuffer<label> cells;          // porous cellZone cells
    vector              d{0,0,0}, f{0,0,0};   // adjusted Darcy d + Forchheimer f
    // porosityModels::fixedCoeff. A DIFFERENT model, not a variant of the above: its coefficients are
    // fixed (alpha [1/s], beta [1/m]) rather than derived from the viscosity, and OF stores them as FULL
    // TENSORS because calcTransformModelData() rotates diag(alpha) into the coordinateSystem's frame
    // (fixedCoeff.C: alpha_[zonei] = csys().transform(alphaCoeff)). Row-major 9 components.
    bool                fixed = false;
    scalar              fa[9] = {0,0,0,0,0,0,0,0,0};   // transformed alpha tensor
    scalar              fb[9] = {0,0,0,0,0,0,0,0,0};   // transformed beta  tensor
    scalar              rhoRef = 1.0;                  // fixedCoeff::correct: read only when the eqn is in force units
};

// diag[c] += V*isoCd for the porous cells.  Call once (mDiag) before rAU.
void deviceFvoPorosityDiag(const DevicePorosity& por, scalar nu, const DeviceBuffer<scalar>& V,
                           const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                           DeviceBuffer<scalar>& diag);
// relaxSrc[c] += V*(isoCd - c_comp)*U_comp for the porous cells (= the explicit -V*((Cd-I*isoCd).U)[comp]).
void deviceFvoPorositySource(const DevicePorosity& por, int comp, scalar nu, const DeviceBuffer<scalar>& V,
                             const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                             DeviceBuffer<scalar>& src);

// limitVelocity: clamp |U| <= max on the given cells (OF fv::limitVelocity::correct, U *= sqrt(max^2/|U|^2) where it
// exceeds max, preserving direction).
void deviceFvoLimitVelocity(const DeviceBuffer<label>& cells, scalar maxU,
                            DeviceBuffer<scalar>& Ux, DeviceBuffer<scalar>& Uy, DeviceBuffer<scalar>& Uz);

// limitTemperature (OF fv::limitTemperature::correct(he)): clamp the ENERGY variable into [heMin, heMax] on
// the given cells. The caller converts the T limits with the case thermo -- OF does the same
// (heMin = thermo.he(p, Tmin, cells)) and clamps he rather than T, because he is what the equation solved.
void deviceFvoLimitEnergy(const DeviceBuffer<label>& cells, scalar heMin, scalar heMax,
                          DeviceBuffer<scalar>& he);

// The he BOUNDARY half of the same clamp, applied only where the patch does NOT fix a value and only when
// the selection is the whole mesh (limitTemperature.C: `if (!cellSetOption::useSubMesh())`).
void deviceFvoLimitEnergyBoundary(const DeviceBoundary& dbHe, scalar heMin, scalar heMax,
                                  DeviceBuffer<scalar>& heBnd);

// velocityDampingConstraint (OF fv::velocityDampingConstraint::addDamping, called from constrain(eqn) after relax):
// for cells where |U| > UMax, diag[c] += C*V[c]^(2/3)*(|U|-UMax) (implicit-only diagonal sink; all 3 components share
// the matrix diagonal). U is lagged (eqn.psi()). Add to the RELAXED diagonal so it feeds the predictor AND rAU/HbyA.
void deviceFvoVelocityDamping(const DeviceBuffer<label>& cells, scalar UMax, scalar C,
                              const DeviceBuffer<scalar>& V, const DeviceBuffer<scalar>& Ux,
                              const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                              DeviceBuffer<scalar>& diag);

// fvMatrix::setValues, the matrix manipulation OpenFOAM's fvOptions CONSTRAINTS and the turbulence wall
// functions both end in. Shared: the kEpsilon closure and the energy equation apply the same one.
void deviceSetValues(
    const DeviceMesh&           dm,
    const DeviceBuffer<label>&  mask,     // per CELL, non-zero = pinned
    const DeviceBuffer<scalar>& value,    // per CELL
    DeviceBuffer<scalar>&       diag,
    DeviceBuffer<scalar>&       upper,
    DeviceBuffer<scalar>&       lower,
    DeviceBuffer<scalar>&       source,
    DeviceBuffer<scalar>&       internalCoeffs,
    DeviceBuffer<scalar>&       boundaryCoeffs,
    DeviceBuffer<scalar>&       psi);

} // namespace brae
