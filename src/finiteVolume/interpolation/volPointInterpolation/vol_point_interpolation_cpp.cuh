#pragma once
// OpenFOAM's volPointInterpolation: a cell field to the mesh points, by inverse distance. The host
// reference.
//
// provenance:
//   openfoam: src/finiteVolume/interpolation/volPointInterpolation/volPointInterpolation.C:65-167
//                 (calcBoundaryAddressing), :169-256 (makeInternalWeights, makeBoundaryWeights),
//                 :349-472 (makeWeights), :500-505 (movePoints: the weights are remade)
//             src/finiteVolume/interpolation/volPointInterpolation/volPointInterpolate.C:129-164
//                 (interpolateInternalField), :226-322 (flatBoundaryField, interpolateBoundaryField)
//
// TWO KINDS OF POINT, and the split is the whole of the scheme:
//   a point on no face of a real patch   the cells around it, weighted 1/|p - C|
//   a point on a face of a real patch    the BOUNDARY FACE values around it, weighted 1/|p - Cf|, and
//                                        no cell at all
// "Real" is neither empty nor coupled. A point on only an empty patch -- every point of a 2-D mesh --
// is interior. The weights are normalised by their sum once, when they are made, and the value is the
// plain sum of weight times value in pointCells order: dividing afterwards would be a different last
// digit.
//
// NOT HERE: the constraint step that follows in volPointInterpolation::interpolate (pointConstraints::
// constrain -- the point patch fields' evaluate, then corner constraints). It belongs to the point field,
// and the caller that owns the point boundary conditions applies it. Coupled and separated patches
// (syncUntransformedData, addSeparated, the normalisation they bring) are refused by the caller.
#include "cf_types.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include "primitive_patch_cpp.cuh"
#include <vector>

namespace brae {

class VolPointInterpolation
{
public:
    // makeWeights on the mesh as it stands. pointCells is the mesh's, in the order it has (see
    // primitive_patch_cpp.cuh).
    void makeWeights(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& patches,
        const std::vector<std::vector<label>>& pointCells);

    // interpolateInternalField: every point that is not a patch point, from the cells
    void interpolateInternalField(
        const std::vector<vector>& vf,
        std::vector<vector>& pf) const;

    // interpolateBoundaryField, without the constraint step: every patch point, from the values of the
    // boundary faces of real patches. boundaryField is per patch, per face, as the fvPatchFields hold it.
    void interpolateBoundaryField(
        const std::vector<std::vector<vector>>& boundaryField,
        std::vector<vector>& pf) const;

    bool isPatchPoint(label pointi) const
    {
        return isPatchPoint_[static_cast<std::size_t>(pointi)] != 0;
    }

private:
    const std::vector<std::vector<label>>* pointCells_ = nullptr;
    label nInternalFaces_ = 0;
    std::vector<label> patchStart_;
    std::vector<label> patchSize_;
    // boundaryPtr_: every boundary face of the mesh as one patch
    PrimitivePatchAddressing boundary_;
    std::vector<char> boundaryIsPatchFace_;
    std::vector<char> isPatchPoint_;
    std::vector<std::vector<scalar>> pointWeights_;
    std::vector<std::vector<scalar>> boundaryPointWeights_;
};

} // namespace brae
