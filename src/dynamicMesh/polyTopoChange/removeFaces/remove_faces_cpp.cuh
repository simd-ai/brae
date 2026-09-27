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
#include "mesh_edges_cpp.cuh"
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

// ----------------------------------------------------------------------------------------------
// removeFaces::setRefinement, unit 6b-3a: THE DECISIONS. Which edges go, which faces merge with which,
// which points go, and which faces are touched at all.
//
// provenance:
//   openfoam: removeFaces.C:762-1400 of setRefinement, plus changeFaceRegion (:81-131) and
//             getFacesAffected (:142-196)
//   oracle:   tools/dumpHexRef8/removeFacesDump.C -- a copy of OpenFOAM's own class with WRITES ONLY,
//             because every one of these is a LOCAL of setRefinement (the of-instrument pattern)
//
// WHY THE EDGE COUNT DECIDES EVERYTHING. `nFacesPerEdge` starts at -1 and is counted DOWN from
// edgeFaces().size()-1 once per face being removed, so after the two loops an edge carries the number of
// faces that will still use it: 0 means nothing uses the edge any more, 2 means the two survivors are
// coplanar halves of one face and must be MERGED, and 3 or more means the edge stays. -1 is the "not
// touched, and only two faces use it" case, which is an interior edge of a face and is left alone.
//
// AND THE THREE PLACES A 2 IS PUT BACK TO 3, which is the whole subtlety of the function:
//   two boundary faces of DIFFERENT patches      never merged across a patch boundary
//   two boundary faces at an angle below minCos   OpenFOAM's own feature-angle guard
//   two internal faces between DIFFERENT cell pairs, as UNORDERED pairs of regions (Foam::edge
//     comparison is symmetric, edgeI.H:499 -> PairI.H) -- merging those would put two faces between one
//     cell pair, which is what OpenFOAM's comment calls an upper-triangular ordering problem
struct RemoveFacesView
{
    const PrimitiveMesh*                   m = nullptr;
    const MeshEdges*                       edges = nullptr;        // edges() and pointEdges()
    const std::vector<std::vector<label>>* faceEdges = nullptr;
    const std::vector<std::vector<label>>* edgeFaces = nullptr;
    const std::vector<std::vector<label>>* cells = nullptr;        // getFacesAffected's cell faces
    const std::vector<std::vector<label>>* pointFaces = nullptr;   // ...and its point faces
    // polyPatch::faceNormals(), which is the face AREA VECTOR normalised -- only the minCos filter
    // reads it, and only for an edge whose two survivors are both on the boundary
    const std::vector<vector>*             faceAreas = nullptr;
};

struct RemoveFacesDecisions
{
    // per edge, the number of faces that will still use it, after the filter puts some 2s back to 3
    std::vector<label> nFacesPerEdge;
    std::vector<label> edgesToRemove;                  // ascending
    // per face: -1 removed or unvisited, -2 a region of ONE face (so nothing to merge), else its region
    std::vector<label> faceRegion;
    label              nFaceRegions = 0;
    std::vector<label> pointsToRemove;                 // ascending
    std::vector<char>  affectedFace;
    // invertOneToMany(nFaceRegions, faceRegion): each region's faces in ASCENDING face order
    std::vector<std::vector<label>> regionToFaces;
};

RemoveFacesDecisions setRefinementDecisions(
    const RemoveFacesView&    v,
    const std::vector<label>& facesToRemove,
    const std::vector<label>& cellRegion,
    const std::vector<label>& cellRegionMaster,
    scalar                    minCos);

} // namespace removeFaces
} // namespace cpu
} // namespace brae
