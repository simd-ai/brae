#pragma once
// fv::rotorDiskSource -- the Froude blade-element-momentum rotor, host reference.
//
// provenance:
//   openfoam: src/fvOptions/sources/derived/rotorDiskSource/rotorDiskSource.C
//             .../rotorDiskSourceTemplates.C (calculate), bladeModel/, profileModel/
//   brae:     src/finiteVolume/cfdTools/general/fvOptions/rotorDiskSource_cpp.cu
//   tests:    tests/rotordisk_vs_openfoam.sh
//
// Per disk cell, in the cylindrical frame (e1 radial, e2 azimuthal, e3 = axis):
//
//   Uc      = (e1.U, e2.U, e3.U), then Uc.x = 0 and Uc.y = r*omega - Uc.y   (blade-relative)
//   alpha   = (theta0 + twist(r)) - atan2(-Uc.z, Uc.y),  wrapped into [-pi, pi]
//   Cd, Cl  = profile lookup at alpha
//   f       = 0.5*|Uc|^2 * chord(r) * nBlades * area / r / 2pi
//   force   = -f*Cd * e2  +  tipFactor*f*Cl * e3,   tipFactor = 1 iff r/rMax < tipEffect
//
// SIGN. OpenFOAM's addSup does `eqn -= force` with force carrying eqn's dimensions PER VOLUME (calculate
// divides by V), and fvMatrix::operator-= is `source() += V*su` -- so the extensive source gains the RAW
// force. Writing it as a subtraction is the easy mistake, and it is invisible in a converged residual
// because the rotor is a body force, not a constraint: it just thrusts the wrong way.
//
// SCOPE, refused rather than approximated: geometryMode `specified` only, fixedTrim only, no coning
// (Rcone = I), one lookup profile, inletFlowType local|fixed. OpenFOAM supports more of each.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "rotor_disk.cuh"   // RotorDiskParams, shared with the device build

#include <vector>

namespace brae {
namespace cpu {

// The per-cell geometry the force loop needs, resolved once against the mesh.
struct RotorDisk
{
    bool   active = false;
    vector axis{0, 1, 0};
    scalar omega = 0, nBlades = 0, tipEffect = 1, rMax = 0, theta0 = 0;
    bool   localInflow = true;
    vector inletVel{0, 0, 0};

    std::vector<label>  cells;
    std::vector<vector> e2;                       // azimuthal unit vector per disk cell
    std::vector<scalar> radius, area, twist, chord;
    std::vector<scalar> pAlpha, pCd, pCl;         // profile table, alpha ascending [rad]
};

RotorDisk buildRotorDisk(
    const RotorDiskParams& p,
    const PrimitiveMesh&   m,
    const FvGeometry&      g);

// OF addSup: source += V * (force/V), i.e. the raw force, on the disk cells. `source` is the momentum
// equation's extensive source.
void addSup(
    const RotorDisk&           rd,
    const std::vector<vector>& U,
    std::vector<vector>&       source);

// The raw per-cell force itself, for gating against OpenFOAM's own rotorForce field.
void rotorForce(
    const RotorDisk&           rd,
    const std::vector<vector>& U,
    std::vector<vector>&       force);   // sized nCells, zero off the disk

} // namespace cpu
} // namespace brae
