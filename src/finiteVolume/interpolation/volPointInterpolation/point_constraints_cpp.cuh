#pragma once
// OpenFOAM's pointConstraints::constrainDisplacement, the last line of rigidBodyMeshMotion::solve
// (rigidBodyMeshMotion.C:385-388): the point patch fields evaluated in patch order, then every point on
// more than one constraint patch -- or on a constraint patch's perimeter -- transformed by its combined
// constraint. The host reference.
//
// provenance:
//   openfoam: src/finiteVolume/interpolation/volPointInterpolation/pointConstraints.C:46-146 (the
//                 accumulation over every non-empty, non-coupled face patch's boundaryPoints, in patch
//                 order), :310-340 (only non-zero constraints kept), :395-429 (constrainDisplacement)
//             src/finiteVolume/interpolation/volPointInterpolation/pointConstraintsTemplates.C:112-126
//                 (constrainCorners: pf[p] = transform(T, pf[p]))
//             src/OpenFOAM/fields/pointPatchFields/pointPatchField/pointConstraint/pointConstraintI.H:
//                 60-86 (applyConstraint), :130-148 (constraintTransformation)
//             src/OpenFOAM/meshes/polyMesh/polyPatches/constraint/symmetryPlane/symmetryPlanePolyPatch.C:
//                 46-95 (the normal: gSum(faceAreas).normalise(ROOTVSMALL), once, and the planarity check)
//             src/OpenFOAM/fields/pointPatchFields/constraint/symmetryPlane/symmetryPlanePointPatchField.C:
//                 103-123 (evaluate: (pif + transform(I - 2.0*sqr(n), pif))/2.0)
//             src/OpenFOAM/fields/pointPatchFields/basic/value/valuePointPatchField.C:203-211 (fixedValue
//                 writes its values into the shared point field)
//   brae:
//     reference: this header
//     tests:     tests/point_constraints_vs_openfoam.sh, against tools/dumpPointConstraints (OpenFOAM's own
//                class on the same input)
//
// WHAT DECIDES A PATCH'S CONSTRAINT IS THE MESH PATCH'S TYPE, not the field file's entry:
// pointPatchField::New replaces an entry whose constraintType differs from the patch's with the patch's own
// type (pointPatchFieldNew.C:143-161), and the corner constraint comes from the pointPatch. So a
// `symmetryPlane` mesh patch is a symmetryPlane point patch whatever its entry says.
//
// THE NORMAL IS FROZEN at construction (symmetryPlanePolyPatch.C:48: only while n_ is still rootMax), from
// the face areas the mesh had then -- on a moving mesh the construction-time plane, never the live one.
//
// NOT PORTED, refused by name at build: a point patch other than fixedValue (uniform), calculated and a
// symmetryPlane mesh patch; any other constraint mesh patch (symmetry, wedge, empty, cyclic*, processor --
// their applyConstraint differs or they are coupled); a mesh with fewer than three geometric directions
// (twoDPointCorrector acts there, pointConstraints.C:419-423).
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {

struct PointPatchConstraint
{
    enum class Kind
    {
        calculated,
        fixedValue,
        symmetryPlane
    };
    std::string        name;
    Kind               kind = Kind::calculated;
    // fixedValue's uniform value
    vector             value{0, 0, 0};
    // symmetryPlane's frozen normal
    vector             n{0, 0, 0};
    // primitivePatch meshPoints, in first-appearance order
    std::vector<label> meshPoints;
};

class PointConstraints
{
public:
    // Built ONCE: the entries of the pointDisplacement file at fieldPath, the mesh patch types, and every
    // symmetryPlane normal from Sf as it stands now (the construction-time face areas).
    static PointConstraints build(
        const std::string&          fieldPath,
        const PrimitiveMesh&        m,
        const std::vector<FvPatch>& patches,
        const std::vector<vector>&  Sf);

    // pf.correctBoundaryConditions(): every patch evaluated in patch order, each writing into the shared
    // point field
    void evaluate(std::vector<vector>& d) const;

    // constrainCorners: d[p] = T & d[p] at every constrained point
    void constrainCorners(std::vector<vector>& d) const;

    // constrainDisplacement with overrideFixedValue false: evaluate, then constrainCorners (the coupled
    // sync and twoDPointCorrector are no-ops on the meshes build() accepts)
    void constrainDisplacement(std::vector<vector>& d) const;

    const std::vector<PointPatchConstraint>& patchConstraints() const { return patches_; }
    // patchPatchPointConstraintPoints_ / _Tensors_ and each point's constraint count, in OpenFOAM's order
    const std::vector<label>&  cornerPoints() const { return cornerPoints_; }
    const std::vector<tensor>& cornerTensors() const { return cornerTensors_; }
    const std::vector<label>&  cornerCounts() const { return cornerCounts_; }

private:
    std::vector<PointPatchConstraint> patches_;
    std::vector<label>                cornerPoints_;
    std::vector<tensor>               cornerTensors_;
    std::vector<label>                cornerCounts_;
};

} // namespace brae
