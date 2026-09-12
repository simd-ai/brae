#pragma once
// Device MRF, built from the VALIDATED cpu::MRF::Zone rather than from a second face classification.
//
// provenance:
//   openfoam: src/finiteVolume/cfdTools/general/MRF/MRFZone.C (setMRFFaces, makeRelativeRhoFlux)
//   brae:     src/finiteVolume/cfdTools/general/MRF/device_MRF.cu
//   tests:    tests/mrf_cuda_vs_openfoam.sh
//
// WHY NOT THE SHIPPED src/finiteVolume/cfdTools/MRF/device_mrf.cuh: it classifies internal faces as those
// with BOTH cells in the zone (OpenFOAM uses EITHER) and applies the same subtraction to every in-zone
// boundary face. OpenFOAM instead ZEROES the included faces -- the ones that move with the frame -- and
// subtracts only on the excluded ones. On mixerVessel2D the included faces carry a frame flux up to
// 1.28e-04, so the two are not the same arithmetic and the difference is measurable.
//
// The zone geometry is static and Omega is constant, so the per-face frame flux is precomputed once.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "MRF_cpp.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"

#include <vector>

namespace brae {

struct DeviceMRFZone
{
    vector Omega{0, 0, 0};
    DeviceBuffer<label>  zoneCell;       // 1 in the zone, 0 outside (Coriolis)
    DeviceBuffer<scalar> frameFluxInt;   // (Omega x (Cf - origin)) & Sf, 0 off the zone's internal faces
    DeviceBuffer<scalar> frameFluxBnd;   // the same, on EXCLUDED boundary faces only
    DeviceBuffer<label>  zeroBnd;        // 1 on INCLUDED boundary faces: OF sets these to zero outright
    bool active = false;
};

DeviceMRFZone buildDeviceMRFZone(
    const cpu::MRF::Zone&       z,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

// MRFZoneList::DDt as UEqn.H adds it: src[c] -= V[c]*(Omega x U)_kk on the zone cells, per component.
void deviceMrfCoriolisZone(
    const std::vector<DeviceMRFZone>& zones,
    const DeviceBuffer<scalar>&       V,
    const DeviceBuffer<scalar>&       Ux,
    const DeviceBuffer<scalar>&       Uy,
    const DeviceBuffer<scalar>&       Uz,
    int                               cmpt,
    DeviceBuffer<scalar>&             src);

// MRFZoneList::makeRelative(phi): subtract the frame flux on internal and excluded faces, ZERO the
// included ones.
void deviceMrfMakeRelative(
    const std::vector<DeviceMRFZone>& zones,
    DeviceBuffer<scalar>&             phiInt,
    DeviceBuffer<scalar>&             phiBnd);

} // namespace brae
