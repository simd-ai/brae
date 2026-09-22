#pragma once
// The LES filter width -- OpenFOAM's LESdelta family, the host reference. Two of them: cubeRootVol, and
// smooth around it.
//
// provenance:
//   openfoam:  src/TurbulenceModels/turbulenceModels/LES/LESdeltas/LESdelta/LESdelta.C (New: the `delta`
//                  keyword of the dictionary it is handed)
//              .../LESdeltas/cubeRootVolDelta/cubeRootVolDelta.C:36-78 (calcDelta), :85-100 (deltaCoeff,
//                  from `cubeRootVolCoeffs`, default 1)
//              .../LESdeltas/smoothDelta/smoothDelta.C:38-77 (setChangedFaces), :80-117 (calcDelta),
//                  :123-147 (constructor: the geometric delta from `smoothCoeffs`, maxDeltaRatio MUST_READ)
//              .../LESdeltas/smoothDelta/smoothDeltaDeltaDataI.H (update, updateCell, updateFace, valid)
//              src/meshTools/algorithms/MeshWave/FaceCellWave.C (setFaceInfo, faceToCell, cellToFace,
//                  iterate, updateCell, updateFace), FaceCellWaveBase.C:42 (propagationTol_ = 0.01)
//              src/OpenFOAM/meshes/polyMesh/polyMesh.C calcDirections (nGeometricD: empty AND wedge
//                  patches knock directions out)
//
// cubeRootVol:  3-D   delta = deltaCoeff*cbrt(V)
//               2-D   delta = deltaCoeff*sqrt(V/thickness), thickness = the mesh's bounding-box span in the
//                     first direction geometricD knocks out -- on an AXISYMMETRIC WEDGE that is the wedge's
//                     normal direction, and the span is the wedge's full width at its largest radius
//
// smooth:       the geometric delta, then a FaceCellWave that raises a cell's delta to its neighbour's
//               divided by maxDeltaRatio. THE ORDER IS PART OF THE RESULT: a cell or face takes a new
//               value only when it is more than propagationTol (1%) larger than what it holds, so a cell
//               reached by a large value late keeps a slightly smaller one it took earlier. This file
//               keeps FaceCellWave's order: the seeded faces in face order, each face's owner before its
//               neighbour, each changed cell's faces in mesh.cells() order (owned, then neighboured).
//
// NOT PORTED, refused where the case names them: every other LESdelta (Prandtl, vanDriest, maxDeltaxyz,
// IDDESDelta, ...), smooth around anything but cubeRootVol, a mesh with coupled patches (the wave crosses
// them in handleCyclicPatches/handleProcPatches, which are not here), and a mesh that moves (smoothDelta
// recomputes only when mesh.changing()).
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace LESdelta {

struct Spec
{
    // `smooth` wraps a geometric delta; `cubeRootVol` is that geometric delta on its own
    bool smooth = false;
    scalar deltaCoeff = 1;
    scalar maxDeltaRatio = 0;
};

// Reads the LES dictionary's `delta` and the coefficient sub-dictionaries LESdelta::New hands on.
// Refuses every delta type but cubeRootVol and smooth-around-cubeRootVol, by name.
Spec read(
    const FoamDict& lesDict,
    const std::string& file);

// polyMesh::nGeometricD() and the knocked-out direction's bounding-box span, as cubeRootVolDelta uses
// them: `thickness` is 0 unless the mesh is 2-D.
struct GeometricDirections
{
    int nD = 3;
    scalar thickness = 0;
};
GeometricDirections geometricDirections(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches);

// cubeRootVolDelta::calcDelta -- per cell
std::vector<scalar> cubeRootVol(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar deltaCoeff);

// smoothDelta::calcDelta around a given geometric delta -- per cell
std::vector<scalar> smooth(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const std::vector<scalar>& geometricDelta,
    scalar maxDeltaRatio);

// the delta the spec names, per cell
std::vector<scalar> compute(
    const Spec& s,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches);

} // namespace LESdelta
} // namespace cpu
} // namespace brae
