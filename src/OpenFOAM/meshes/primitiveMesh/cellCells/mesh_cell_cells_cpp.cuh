#pragma once
// OpenFOAM's primitiveMesh::cellCells(): per cell, the cells across its INTERNAL faces. The host
// reference.
//
// provenance:
//   openfoam: src/OpenFOAM/meshes/primitiveMesh/primitiveMeshCellCells.C:32-95 (calcCellCells)
//   brae:
//     reference: this header
//     tests:     tests/hex_ref8_vs_openfoam.sh, through removeFaces::compatibleRemoves, which floods
//                regions over this addressing
//
// THE ORDER IS FACE ORDER, not cell order: OpenFOAM counts internal faces per cell, sizes each row,
// then fills by walking the internal faces once and appending the partner at both ends. So a row is
// ordered by the FACE label that produced the entry, and a cell appears once per shared internal face
// -- twice if two cells share two faces, which a refined mesh does have.
//
// A BOUNDARY FACE CONTRIBUTES NOTHING, coupled or not. OpenFOAM's loop is over faceNeighbour(), which
// exists only for internal faces, so a cyclic pair's two cells are NOT neighbours here. That is what
// makes the region flood in compatibleRemoves stop at a processor or cyclic boundary, and it is
// OpenFOAM's behaviour rather than an omission.
#include "cf_types.cuh"
#include "compact_list_list.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

std::vector<std::vector<label>> buildCellCells(const PrimitiveMesh& m);
// ...and compact (compact_list_list.cuh): the same rows, one array of values and one of offsets
CompactListList compactCellCells(const PrimitiveMesh& m);

} // namespace brae
