#pragma once
// MRFZone / MRFZoneList -- the multiple reference frame, as simpleFoam reaches it.
//
// provenance:
//   openfoam: src/finiteVolume/cfdTools/general/MRF/MRFZone.C
//             src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C
//             src/finiteVolume/cfdTools/general/MRF/MRFZoneTemplates.C
//   brae:     src/finiteVolume/cfdTools/general/MRF/MRF_cpp.cu
//   tests:    tests/mrf_cpp_vs_openfoam.sh, tests/mrf_cuda_vs_openfoam.sh, tests/interfoam_mrf_vs_openfoam.sh, tests/mrf_cpp_vs_openfoam.sh
//
// simpleFoam reaches it in exactly three places that do arithmetic (UEqn.H:3,8 and pEqn.H:5):
//
//   MRF.correctBoundaryVelocity(U)   included patch faces take U_b = Omega x (Cf - origin)
//   + MRF.DDt(U)                     Omega x U on the zone cells; the field is added to the equation's
//                                    LHS, so it lands as  source -= V*(Omega x U).  EXPLICIT in U --
//                                    OpenFOAM builds a volVectorField from the CURRENT U rather than
//                                    an implicit Coriolis operator, so it is lagged like any other
//                                    deferred term.
//   MRF.makeRelative(phiHbyA)        the frame flux is removed from the zone's faces
//
// pEqn.H:21's constrainPressure(p, U, phiHbyA, rAtU(), MRF) does nothing unless a patch carries
// fixedFluxPressure, which is outside this path's envelope and refused there.
//
// THE FACE CLASSIFICATION IS THE SUBSTANCE. MRFZone::setMRFFaces builds three lists, and which list a
// face lands in decides whether its flux is zeroed, corrected, or left alone:
//
//   internalFaces   internal faces with EITHER side in the zone -- not both. The zone INTERFACE faces
//                   are the ones that carry the frame flux, so taking only the interior of the zone
//                   silently drops the correction exactly where it does the most work.
//   includedFaces   boundary faces owned by the zone on a patch that is neither coupled, nor named in
//                   nonRotatingPatches, nor empty. These MOVE with the frame: flux zeroed outright.
//   excludedFaces   boundary faces owned by the zone on a coupled or nonRotating patch. These do NOT
//                   move with the frame, so they take the same subtraction the internal faces take.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"          // SurfaceScalarField

#include <map>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace MRF {

// One MRFProperties entry, before it is resolved against a mesh.
struct ZoneSpec
{
    std::string              cellZone;
    bool                     active = true;
    vector                   origin{0, 0, 0};
    vector                   axis{0, 0, 1};
    scalar                   omega = 0;
    std::vector<std::string> nonRotatingPatches;
};

// The same entry resolved against a mesh: Omega, the zone cells, and the three face lists.
struct Zone
{
    bool   active = true;
    vector Omega{0, 0, 0};
    vector origin{0, 0, 0};

    std::vector<bool>  inZone;         // per cell
    std::vector<label> cells;
    std::vector<label> internalFaces;                  // mesh-global internal face ids
    std::vector<std::vector<label>> includedFaces;     // [patch] -> patch-local face ids
    std::vector<std::vector<label>> excludedFaces;     // [patch] -> patch-local face ids
};

// EVERY active entry, not just the first -- MRFZoneList is a list and OpenFOAM applies all of them.
std::vector<ZoneSpec> readMRFProperties(const std::string& constantDir);

Zone buildZone(
    const ZoneSpec&             spec,
    const std::vector<label>&   zoneCells,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches);

// MRFZoneList::update() (MRFZoneList.C:441-450) -> MRFZone::update() (MRFZone.C:598-604), which calls
// setMRFFaces() AND NOTHING ELSE: it does not re-read the dictionary, does not re-look-up cellZoneID_ and
// does not re-evaluate omega. Everything buildZone computes beyond the spec's three scalars IS
// setMRFFaces, and the spec cannot change, so the faithful update is a rebuild from the SAME spec against
// the LIVE cellZone -- which a topology change has renumbered by giving each child its parent's zone id.
// The two lists stay parallel by construction (one zone per spec, built in order).
void update(
    std::vector<Zone>&                                zones,
    const std::vector<ZoneSpec>&                      specs,
    const std::map<std::string, std::vector<label>>&  cellZones,
    const PrimitiveMesh&                              m,
    const std::vector<FvPatch>&                       patches);

// MRFZoneList::correctBoundaryVelocity -- included patch faces take the frame velocity.
void correctBoundaryVelocity(
    GeometricField<vector>&     U,
    const std::vector<Zone>&    zones,
    const std::vector<FvPatch>& patches);

// MRFZoneList::DDt, as UEqn.H adds it: source -= V*(Omega x U) on the zone cells.
void addCoriolis(
    const std::vector<Zone>&    zones,
    const std::vector<vector>&  U,
    const std::vector<scalar>&  V,
    std::vector<vector>&        source);

// MRFZoneList::makeRelative(phi).
void makeRelative(
    SurfaceScalarField&         phi,
    const std::vector<Zone>&    zones,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

} // namespace MRF
} // namespace cpu
} // namespace brae
