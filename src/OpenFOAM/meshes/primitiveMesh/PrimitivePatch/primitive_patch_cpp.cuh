#pragma once
// OpenFOAM's topological addressing, in OpenFOAM's ORDER: a mesh's cells(), pointFaces() and
// pointCells(), and a PrimitivePatch's meshPoints(), localFaces() and pointFaces(). The host reference.
//
// provenance:
//   openfoam: src/OpenFOAM/meshes/primitiveMesh/primitiveMeshCells.C (calcCells)
//             src/OpenFOAM/meshes/primitiveMesh/primitiveMeshPointFaces.C:33-48 (pointFaces)
//             src/OpenFOAM/meshes/primitiveMesh/primitiveMeshPointCells.C (calcPointCells)
//             src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchMeshData.C (calcMeshData)
//             src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchPointAddressing.C
//                 (calcPointFaces)
//
// WHY THE ORDER IS THE CONTENT. Every list here is a SET in the mathematics and a SEQUENCE in the code
// that reads it: a wall-distance wave visits a cell's faces in cells() order and keeps the first of two
// equally near origins; volPointInterpolation sums a point's cell weights in pointCells() order. A
// different order is a different last digit, or a different nearest wall.
//
// pointCells() HAS THREE ORDERS in OpenFOAM, chosen by what the mesh has already computed: the inverse of
// cellPoints() when that exists (ascending), else the walk over pointFaces() when THAT exists (not
// sorted), else the walk over cells() (ascending). pointCellsFromPointFaces is the second, which is the
// one a displacement motion solver meets: its wall distance asks for pointFaces() before anything asks
// for pointCells().
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

// primitiveMesh::cells(): each cell's faces, those it owns first, then those it neighbours, each in
// ascending face order
std::vector<std::vector<label>> meshCells(const PrimitiveMesh& m);

// primitiveMesh::pointFaces(): invertManyToMany of faces(), so each point's faces ascending
std::vector<std::vector<label>> meshPointFaces(const PrimitiveMesh& m);

// primitiveMesh::calcPointCells' pointFaces branch: for each face of the point, its owner and then its
// neighbour, each cell once, in the order met
std::vector<std::vector<label>> pointCellsFromPointFaces(
    const PrimitiveMesh& m,
    const std::vector<std::vector<label>>& pointFaces);

// A PrimitivePatch over a list of the mesh's faces
struct PrimitivePatchAddressing
{
    // the mesh faces the patch is made of, in the patch's order
    std::vector<label> faces;
    // meshPoints(): the mesh point of each local point, numbered by first appearance face by face
    std::vector<label> meshPoints;
    // localFaces(): each face in local point labels
    std::vector<std::vector<label>> localFaces;
    // pointFaces(): each local point's faces, ascending
    std::vector<std::vector<label>> pointFaces;
};

PrimitivePatchAddressing primitivePatch(
    const PrimitiveMesh& m,
    const std::vector<label>& faces);

// the faces [start, start + size) of the mesh, as a polyPatch is
std::vector<label> faceRange(
    label start,
    label size);

} // namespace brae
