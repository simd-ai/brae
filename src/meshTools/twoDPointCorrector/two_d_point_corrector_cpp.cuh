#pragma once
// OpenFOAM's twoDPointCorrector: after a motion, the two points of every edge normal to a 2-D mesh's
// plane are put back on one line normal to it, about the mid-plane. The host reference.
//
// provenance:
//   openfoam: src/meshTools/twoDPointCorrector/twoDPointCorrector.C:44 (edgeOrthogonalityTol),
//                 :49-175 (calcAddressing), :200-212 (required_: nGeometricD() == 2),
//                 :270-307 (correctPoints)
//             src/meshTools/meshTools/meshTools.C:628-646 (constrainToMeshCentre)
//             src/OpenFOAM/meshes/primitiveMesh/primitiveMeshEdges.C (the edges it walks)
//             src/dynamicMesh/motionSolvers/motionSolver/motionSolver.C:207-210 (twoDCorrectPoints)
//
// THE PLANE NORMAL IS TAKEN ONCE, from the first face of the first non-empty `empty` patch, when the
// corrector is first used, and kept: the corrector is a MeshObject whose movePoints() keeps its
// addressing. The mid-plane is not kept: constrainToMeshCentre reads the mesh's bounds as they are.
//
// NOT PORTED: a wedge (snapToWedge); refused where the corrector is built.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

class TwoDPointCorrector
{
public:
    // twoDPointCorrector::New(mesh) and its calcAddressing, on the mesh as it stands
    void build(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches);

    bool required() const
    {
        return required_;
    }

    // correctPoints(p), with the bounds of the mesh whose points are `meshPoints`
    void correctPoints(
        const std::vector<vector>& meshPoints,
        std::vector<vector>& p) const;

private:
    bool required_ = false;
    vector planeNormal_{0, 0, 0};
    // polyMesh::geometricD() == -1, per component
    bool emptyDir_[3] = {false, false, false};
    // the edges normal to the plane, as (start, end) point pairs
    std::vector<label> normalEdgeStart_;
    std::vector<label> normalEdgeEnd_;
};

} // namespace brae
