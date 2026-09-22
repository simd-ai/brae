#pragma once
// OpenFOAM's face and triangle geometry on a face of the mesh's topology, for points that are not
// necessarily the mesh's own: the host reference.
//
// provenance:
//   openfoam: src/OpenFOAM/meshes/meshShapes/face/face.C:513-563 (centre)
//             src/OpenFOAM/meshes/meshShapes/face/faceTemplates.C:52-117 (average)
//             src/OpenFOAM/meshes/meshShapes/face/faceIntersection.C:193-300 (nearestPoint,
//                 nearestPointClassify)
//             src/OpenFOAM/meshes/primitiveShapes/triangle/triangleI.H:747-900 (nearestPointClassify)
//
// face::centre IS NOT THE MESH'S Cf. primitiveMesh computes Cf with its own sums
// (primitiveMeshFaceCentresAndAreas.C) and the two differ in the last digits on any face that is not a
// triangle. Every caller here -- the swept volume, the cellMotion boundary's face average, the wall
// distance's nearest point -- is OpenFOAM code that calls face::centre, so it is face::centre that is
// transcribed.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

// face::centre(points), for face f of the mesh's topology
vector faceCentreOfPoints(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& points);

// face::average(points, fld): the area-weighted average of a point field over face f
vector faceAverage(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& points,
    const std::vector<vector>& fld);

// face::nearestPoint(p, points).distance(): the distance from p to face f, over the triangles of its
// central decomposition, or to the face itself when it is a triangle
scalar faceNearestDistance(
    const PrimitiveMesh& m,
    label f,
    const std::vector<vector>& points,
    const vector& p);

} // namespace brae
