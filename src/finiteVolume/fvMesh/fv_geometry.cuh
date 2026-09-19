#pragma once
// brae::FvGeometry, cell/face geometry computed EXACTLY per OpenFOAM v2412
// (primitiveMeshTools::makeFaceCentresAndAreas / makeCellCentresAndVols and
// basicFvGeometryScheme weights/deltaCoeffs/nonOrth*), in fp64.
//
// Face quantities (Cf, Sf, magSf) are sized nFaces. The cell-centre estimate + pyramid
// decomposition gives C, V. Interpolation quantities (weights, deltaCoeffs, nonOrth*) are
// defined on internal faces [0, nInternalFaces).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <vector>
#include <utility>
#include <unordered_map>

namespace brae {

class FvGeometry {
public:
    void build(const PrimitiveMesh& m);

    // ---- the cyclicACMI seam -------------------------------------------------------------------
    // A cyclicACMI patch shares its faces with a coincident nonOverlapPatch wall: the SAME face exists
    // twice in the mesh. Their areas must be split by the overlap mask BEFORE cell volumes are computed,
    // or the shared face is counted twice and every adjacent cell gets the wrong volume. OF does exactly
    // this and says why (cyclicACMIPolyPatch.C:452): "Initialise the AMI early to make sure we adapt the
    // face areas before the cell centre calculation gets triggered."
    //
    // build() == buildFaceGeometry() + buildCellGeometry(), unchanged, so every non-ACMI caller is
    // unaffected. An ACMI caller splits them and scales in between.
    void buildFaceGeometry(const PrimitiveMesh& m);
    void buildCellGeometry(const PrimitiveMesh& m);

    // Multiply Sf/magSf of the given faces. Scaling is always FROM RAW: buildFaceGeometry recomputes the
    // areas from the point positions and clears the flag, and a second scale without that rebuild is a
    // hard error rather than a silent double-scale (OF keeps the raw areas cached for the same reason,
    // cyclicACMIPolyPatch.C:392).
    void applyAreaScaling(const std::vector<std::pair<label, scalar>>& faceScale);

    // cyclicACMIPolyPatch::scalePatchFaceAreas (cyclicACMIPolyPatch.C:230-258) for a scale that moves
    // with time: the face's area set from its RAW area times the mask, and |Sf| taken from the result
    // (cyclicACMIFvPatch::resetPatchAreas, magSf = mag(faceAreas)). Used by cyclic_acmi_cpp every step.
    void setFaceArea(label f, const vector& Sf);

    // primitiveMeshTools::updateCellCentresAndVols after the rescale: the centres and volumes from the
    // current face areas, and NOTHING else -- OpenFOAM's weights, deltaCoeffs and non-orthogonal vectors
    // are cached from the first time they were asked for and a static mesh never clears them. It
    // recomputes every cell where OpenFOAM recomputes the ACMI patches' face cells only; the rest are
    // unchanged inputs, and the per-cell sum runs in OpenFOAM's order either way (owned faces ascending,
    // then neighbour faces ascending -- primitiveMesh::calcCells, and makeCellCentresAndVols' two passes).
    void updateCellCentresAndVols(const PrimitiveMesh& m);

    // Pre-scaling |Sf| of the faces applyAreaScaling touched (empty on a mesh without cyclicACMI).
    // The AMI normalises its overlap by the RAW area: dividing by an already-scaled area returns 1 for
    // every face and erases the mask. Stored sparsely -- only ACMI faces are ever scaled -- so a mesh
    // without one pays nothing, and it is consulted once per source face, not per candidate pair.
    const std::unordered_map<label, scalar>& rawArea() const { return rawArea_; }
    scalar rawMagSf(label f) const
    {
        const auto it = rawArea_.find(f);
        return it == rawArea_.end() ? magSf_[f] : it->second;
    }

    const std::vector<vector>& Cf()    const { return Cf_; }
    const std::vector<vector>& Sf()    const { return Sf_; }
    const std::vector<scalar>& magSf() const { return magSf_; }
    const std::vector<vector>& C()     const { return C_; }
    const std::vector<scalar>& V()     const { return V_; }
    const std::vector<scalar>& weights()            const { return weights_; }
    const std::vector<scalar>& deltaCoeffs()        const { return deltaCoeffs_; }
    const std::vector<scalar>& nonOrthDeltaCoeffs() const { return nonOrthDeltaCoeffs_; }
    const std::vector<vector>& nonOrthCorrectionVectors() const { return nonOrthCorr_; }

private:
    void makeFaceCentresAndAreas(const PrimitiveMesh& m);
    void makeCellCentresAndVols(const PrimitiveMesh& m);
    void makeInterpolation(const PrimitiveMesh& m);

    std::vector<vector> Cf_, Sf_, C_, nonOrthCorr_;
    std::vector<scalar> magSf_, V_, weights_, deltaCoeffs_, nonOrthDeltaCoeffs_;
    bool areaScaled_ = false;   // guards against scaling already-scaled areas
    std::unordered_map<label, scalar> rawArea_;   // pre-scaling |Sf|, ACMI faces only
};

} // namespace brae
