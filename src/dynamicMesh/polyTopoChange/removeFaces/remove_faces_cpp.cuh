#pragma once
// OpenFOAM's removeFaces: merging cells by removing the internal faces between them. Unit 6b of the
// dynamicRefineFvMesh port, and the half of hexRef8::setUnrefinement that changes the MESH -- the
// levels and the history are unit 6b-1, in hex_ref8_cpp.cuh.
//
// provenance:
//   openfoam: src/dynamicMesh/polyTopoChange/polyTopoChange/removeFaces.C
//               compatibleRemoves (:581-745), changeCellRegion (:56-78)
//             src/dynamicMesh/polyTopoChange/hexRef8/hexRef8.C:5606-5804 (setUnrefinement, the caller:
//                 it builds the face set from the split points' pointFaces and hands it over)
//   brae:
//     addressing: src/OpenFOAM/meshes/primitiveMesh/cellCells/mesh_cell_cells_cpp.cuh
//     tests:      tests/hex_ref8_vs_openfoam.sh, against tools/dumpHexRef8's own call of OpenFOAM's
//                 compatibleRemoves on the same face set
//
// WHAT compatibleRemoves IS FOR. The caller hands it the faces it would LIKE removed; it answers with
// the cell REGIONS those removals imply, the master cell of each region, and -- the point of the
// function -- the faces that must ALSO go for the answer to be a valid mesh. Removing the six faces
// around one split point merges eight cells into one, and every internal face between any two of those
// eight then has the same cell on both sides, so it has to go too.
//
// AND ON hexRef8's OWN FACE SET THE RECOUNT ADDS NOTHING, which is measured and not assumed: the twelve
// internal faces of a 2x2x2 block of children are exactly the twelve faces that meet at the split
// point, so `pointFaces` already contains every face the recount would find. MEASURED on all three
// unrefinement arms -- 1152 faces in and the SAME LIST out on `unrefine` (96 split points x 12), 384 on
// `unrefine3`, 3072 on `unrefineTwice`. So a gate driven only by hexRef8's own call cannot tell the
// recount from `newFacesToRemove = facesToRemove`, and tools/dumpHexRef8 calls OpenFOAM's
// compatibleRemoves a SECOND time on a deliberately reduced set (one face per block dropped) to give it
// something to find. The region flood is live either way: the block stays connected through the other
// eleven faces.
//
// THE MASTER IS THE LOWEST-NUMBERED CELL OF THE REGION, and OpenFOAM asserts it rather than sorting for
// it: the first face of a new region makes its OWNER the master, which is the lower of the pair because
// an internal face is owned by the lower cell; every later merge takes min() of the two masters. The
// final check walks every cell and FatalErrors if one is below its region's master. brae throws there
// and names the cell, because that check is what makes the incremental min() sound.
//
// A REGION OF ONE CELL IS AN ERROR, not a no-op: it would mean a face was requested whose two sides
// ended up in different regions, which cannot happen through the loop above. Transcribed as a throw.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace removeFaces {

// removeFaces::compatibleRemoves. Returns the number of USED regions (the ones whose master survived
// the merges), and fills:
//   cellRegion       per cell, its region or -1. Sized nCells.
//   regionMaster     per region, its lowest-numbered cell, or -1 where the region was freed by a merge.
//                    Sized nRegions -- which counts the freed ones, so it is not the return value.
//   newFacesToRemove the recount: every internal face whose two cells ended in the same region, in
//                    ASCENDING face order (OpenFOAM walks the internal faces, not the input list).
label compatibleRemoves(
    const PrimitiveMesh&                   m,
    const std::vector<std::vector<label>>& cellCells,
    const std::vector<label>&              facesToRemove,
    std::vector<label>&                    cellRegion,
    std::vector<label>&                    regionMaster,
    std::vector<label>&                    newFacesToRemove);

} // namespace removeFaces
} // namespace cpu
} // namespace brae
