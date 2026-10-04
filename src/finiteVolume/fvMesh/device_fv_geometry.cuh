#pragma once
// FvGeometry::build on the device -- the host's three routines (fv_geometry.cu), in parallel, to the bit.
//
// provenance:
//   openfoam:  src/OpenFOAM/meshes/primitiveMesh/primitiveMeshCheck/primitiveMeshTools.C
//              (updateFaceCentresAndAreas, updateCellCentresAndVols)
//              src/finiteVolume/fvMesh/fvGeometryScheme/basic/basicFvGeometryScheme.C
//              (weights, deltaCoeffs, nonOrthDeltaCoeffs, nonOrthCorrectionVectors)
//   host:      src/finiteVolume/fvMesh/fv_geometry.cu -- the ORACLE
//   tests:     tests/interfoam_write/identity/geometry_device.sh
//
// THREE KERNELS. A face's centre and area need its points only. A cell's centre and volume fold its faces in
// the host's order -- the faces it owns, ascending, then the ones it neighbours, ascending: the host's owner
// pass over all faces and its neighbour pass over the internal ones, which is primitiveMesh::cells()'s order --
// twice, once for the estimated centre and once for the pyramids. An internal face's interpolation factors
// need its two cells.
//
// THE FUSED PRODUCTS ARE THE HOST'S, read off the host binary (aarch64 GCC), instruction by instruction:
//   faces   a triangle: Cf = ((a + b) + c)*(1/3), Sf = cross(b - a, c - a)*0.5. Otherwise the points summed
//           from the first, divided by their number; per edge n = cross(next - this, centre - this),
//           a = sqrt(fma(n.z, n.z, fma(n.x, n.x, n.y*n.y))), sumAc = fma(a, (this + next) + centre, sumAc);
//           Cf = (sumAc*(1/3))/sumA, Sf = sumN*0.5; |Sf| = sqrt(fma(z, z, fma(x, x, y*y)))
//   cross   each component its first product fused, its second rounded (fnmsub)
//   cells   pyr3Vol = fma(Sf.z, d.z, fma(Sf.x, d.x, Sf.y*d.y)); pc = fma(Cf, 0.75, 0.25*cEst);
//           C = fma(pc, pyr3Vol, C); V += pyr3Vol; then C/V and V*(1/3)
//   factors the two |Sf . d| by the same fused dot; w = SfdNei/(SfdOwn + SfdNei);
//           |delta| by the fused magnitude; unitArea = Sf/|Sf|, three divisions;
//           nonOrthDeltaCoeffs = 1/fmax(fused dot, 0.05*|delta|); corr = fma(-nonOrthDeltaCoeffs, delta, unitArea)
// Every product the host leaves unfused is __dmul_rn, so the device compiler cannot fuse it.
//
// MEASURED on waveMakerPiston refined to 896,000 cells: the host's build 87 ms a step; this 13.7 with the points
// up and the nine arrays down. With BRAE_CONTROL_GEOMETRY_CHECK=1 (the host builds too and the mesh compares
// the nine arrays' bytes) not one bit differs on ten moving-mesh cases, 25 steps on four of them.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

// the mesh's addressing on the device, the points, and the geometry they give -- kept across calls
struct DeviceFvGeometry
{
    bool built = false;
    // the caller's count of topology changes when the addressing went up
    unsigned long long topology = 0;
    label nF = 0;
    label nIf = 0;
    label nC = 0;
    label nP = 0;
    DeviceBuffer<label> faceOffsets;
    DeviceBuffer<label> faceVerts;
    DeviceBuffer<label> own;
    DeviceBuffer<label> nei;
    // primitiveMesh::cells() as one array: per cell the faces it owns, ascending, then those it neighbours
    DeviceBuffer<label> cellStart;
    DeviceBuffer<label> cellFaces;
    // 3 a point
    DeviceBuffer<scalar> points;
    // the geometry: 3 a face for Cf and Sf, 3 a cell for C, 3 an internal face for the correction vectors
    DeviceBuffer<scalar> Cf;
    DeviceBuffer<scalar> Sf;
    DeviceBuffer<scalar> magSf;
    DeviceBuffer<scalar> C;
    DeviceBuffer<scalar> V;
    DeviceBuffer<scalar> weights;
    DeviceBuffer<scalar> deltaCoeffs;
    DeviceBuffer<scalar> nonOrthDeltaCoeffs;
    DeviceBuffer<scalar> nonOrthCorr;
    // how many times the geometry above has been computed
    unsigned long long count = 0;
    // the host geometry's generation() right after it took these arrays: while it still says so, the device's
    // copy IS the host's geometry
    unsigned long long hostGeneration = 0;
    // for refreshDeviceMeshFromDeviceGeometry: the mesh face of every slot of the device mesh's boundary
    // numbering, and the addressing it was made for
    DeviceBuffer<label> bndFace;
    unsigned long long bndFaceAddressing = 0;
    // the host arrays the last adopt handed back, the right sizes for the next download
    FvGeometry::Built host;
};

// g.build(m) with the arithmetic on the device: the mesh's points go up, the nine arrays come down and g takes
// them (FvGeometry::adopt), and `w` keeps them on the device for whoever needs them there next. `topology` is a
// count the caller raises whenever the faces change.
// BRAE_CONTROL_GEOMETRY_UNFUSED=1 rounds the pyramid volume's fused products before adding them -- the identity
// gate's control: one rounding's difference in the cell centres and volumes.
void deviceFvGeometry(
    const PrimitiveMesh& m,
    unsigned long long topology,
    DeviceFvGeometry& w,
    FvGeometry& g);

// refreshDeviceMeshGeometry(dm, m, g, fvp) WITHOUT THE HOST: the device mesh's geometric buffers formed on the
// device from the geometry deviceFvGeometry left there -- copies, the split of Sf and the correction vectors
// into components, and the face-to-cell offsets Cf - C, which are plain subtractions on either side. Only
// while `w` is still g's geometry (w.hostGeneration == g.generation(): a cyclicACMI rescale, or any other
// write to g after the move, makes the caller take the host's refresh). Returns false, touching nothing, when
// it is not, or when the sizes are not the device mesh's.
// MEASURED on waveMakerPiston refined to 896,000 cells: the host's refresh rebuilds the whole device mesh, its
// addressing included, and uploads it -- 52 ms a step.
bool refreshDeviceMeshFromDeviceGeometry(
    DeviceMesh& dm,
    DeviceFvGeometry& w,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp);

} // namespace brae
