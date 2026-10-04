#pragma once
// The mesh flux of a point motion on the device: every face's swept volume over the time step -- the host's
// faceSweptVolume (dynamic_motion_solver_fv_mesh_cpp.cu), in parallel, to the bit.
//
// provenance:
//   openfoam:  src/finiteVolume/fvMesh/fvGeometryScheme/fvGeometryScheme/fvGeometryScheme.C (setMeshPhi)
//              src/OpenFOAM/meshes/meshShapes/face/face.C (sweptVol, centre)
//              src/OpenFOAM/meshes/primitiveShapes/triangle/triangleI.H (sweptVol)
//   host:      dynamic_motion_solver_fv_mesh_cpp.cu (faceSweptVolume, triangleSweptVol) and face_cpp.cu
//              (faceCentreOfPoints) -- the ORACLE
//   tests:     tests/interfoam_write/identity/swept_volume_device.sh
//
// A face is independent of every other, so there is no order to keep between faces; inside a face the
// triangles of the central decomposition are added in the host's order, and inside a triangle the six terms.
//
// THE FUSED PRODUCTS ARE THE HOST'S, read off the host binary (aarch64 GCC), instruction by instruction:
//   cross(v, w)           each component its FIRST product fused, the second rounded:
//                         x = fma(v.y, w.z, -(v.z*w.y)), y = fma(v.z, w.x, -(v.x*w.z)), z = fma(v.x, w.y, -(v.y*w.x))
//   dot(u, c)             fma(u.z, c.z, fma(u.x, c.x, u.y*c.y))
//   triangle::sweptVol    six such terms added left to right (the first and the fourth are one value), then a
//                         plain product by 1/12; the face's sum of triangles is plain additions
//   face centre           sum of points, a division by their number, then per triangle
//                         ta = sqrt(fma(c.z, c.z, fma(c.x, c.x, c.y*c.y))), sumAc = fma(ta, ttc, sumAc); the
//                         result sumAc/(3*sumA), a plain product and three divisions; a triangle's is
//                         (1/3)*((p0 + p1) + p2)
// Every product the host leaves unfused is __dmul_rn here, so the device compiler cannot fuse it into the add
// that follows -- the host's are across a function call (faceCentreOfPoints and triangleSweptVol are not
// inlined), the kernel's would not be.
//
// MEASURED on waveMakerPiston refined to 896,000 cells: the host's face loop 97 ms a step, this 8.8 with the two
// point uploads and the download. With BRAE_CONTROL_SWEPT_VOLUME_CHECK=1 (the host's loop runs too and the mesh
// compares bytes) not one bit differs over 25 steps of waveMakerFlap, sloshingTank3D6DoF, floatingObject and
// DTCHullMovingCoarse, whose faces are triangles, quadrilaterals and polygons.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

// the faces' point lists on the device and the work arrays, kept across calls
struct DeviceSweptVolumes
{
    bool built = false;
    // the caller's count of topology changes when the lists went up
    unsigned long long topology = 0;
    label nF = 0;
    label nP = 0;
    DeviceBuffer<label> faceOffsets;
    DeviceBuffer<label> faceVerts;
    // the points before and after, 3 a point
    DeviceBuffer<scalar> oldPoints;
    DeviceBuffer<scalar> newPoints;
    DeviceBuffer<scalar> out;
};

// meshPhi = sweptVol/deltaT on every internal face and on the faces of every patch that is not `empty`, as
// DynamicMotionSolverFvMesh::update fills it (an empty patch's entries are left as they are). `topology` is a
// count the caller raises whenever the faces change; the lists go up again when it moves.
// BRAE_CONTROL_SWEPT_VOLUME_UNFUSED=1 rounds the dot product's last term before adding it -- the identity
// gate's control: one rounding's difference, and the caller's byte check names the face.
void deviceSweptVolumes(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const std::vector<vector>& oldPoints,
    const std::vector<vector>& newPoints,
    scalar rdt,
    unsigned long long topology,
    DeviceSweptVolumes& w,
    SurfaceScalarField& meshPhi);

} // namespace brae
