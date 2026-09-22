#pragma once
// A cyclicAMI pair on the host, for the OF-mirror interFoam loop: the AMI weights between the two sides,
// the coupled patch geometry built from them, and both again after every mesh move.
//
// provenance:
//   openfoam: src/meshTools/AMIInterpolation/patches/cyclicAMI/cyclicAMIPolyPatch/cyclicAMIPolyPatch.C
//               :369-445 resetAMI: the AMI of the current points, owner = src, neighbour = tgt
//               :511-545 initMovePoints: marks the AMI out of date; the next AMI() recomputes it (:989)
//             cyclicAMIPolyPatchTemplates.C interpolateUntransformed: the owner reads interpolateToSource,
//               the neighbour interpolateToTarget
//             src/meshTools/AMIInterpolation/AMIInterpolation/AMIInterpolationTemplates.C weightedSum
//             src/finiteVolume/fvMesh/fvPatches/constraint/cyclicAMI/cyclicAMIFvPatch.C
//               makeWeights: w = |dn|/(|d| + |dn|), d = nf & (Cf - Cn), dn the AMI interpolate of the
//                            neighbour's own
//               delta:       (Cf - Cn) - interpolate(neighbour's Cf - Cn), untransformed on a parallel pair
//               makeDeltaCoeffs, makeNonOrthoDeltaCoeffs, makeNonOrthoCorrVectors: no correction, so
//                            basicFvGeometryScheme's coupled forms stand: 1/|delta|,
//                            1/max(nf & delta, 0.05|delta|), nf - delta*nonOrthDeltaCoeffs
//             src/finiteVolume/fields/fvPatchFields/constraint/cyclicAMI/cyclicAMIFvPatchField.C
//               patchNeighbourField, updateInterfaceMatrix: the AMI interpolate of the neighbour's cells
//   brae:     face_area_weight_ami_cpp is the AMI; FvPatch's amiOffsets/amiNbrCells/amiWeights carry it to
//             every operator through patchNeighbourValue
//   tests:    tests/test_face_area_weight_ami.cu (the AMI, the motion), tests/interfoam_ami_vs_openfoam.sh
//             (RAS/mixerVesselAMI against real OpenFOAM)
//
// NOT PORTED, refused by name: a transform (anything but `transform noOrdering`), a periodic AMI,
// lowWeightCorrection, requireMatch false, an AMIMethod other than faceAreaWeightAMI, createAMIFaces,
// and every other AMI keyword this does not read.
#include "cf_types.cuh"
#include "face_area_weight_ami_cpp.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace cyclicAMIFvPatch {

struct Pair
{
    // the owner (the lower patch index, cyclicAMIPolyPatch::owner) is the AMI's source
    label src = -1;
    label tgt = -1;
    ami::Weights weights;
};

class Interfaces
{
public:
    bool empty() const
    {
        return pairs_.empty();
    }
    const std::vector<Pair>& pairs() const
    {
        return pairs_;
    }

    // resetAMI on the points as they stand, and the coupled half of both patches from it. The mesh
    // update rebuilds every FvPatch from the moved geometry, uncoupled; this couples them again.
    void update(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        std::vector<FvPatch>& patches);

private:
    friend Interfaces setup(
        const std::string& polyMeshDir,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        std::vector<FvPatch>& patches);
    std::vector<Pair> pairs_;
};

// Every cyclicAMI pair of the mesh, checked against its polyMesh/boundary entry and coupled. Empty when
// the mesh has none.
Interfaces setup(
    const std::string& polyMeshDir,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    std::vector<FvPatch>& patches);

} // namespace cyclicAMIFvPatch
} // namespace cpu
} // namespace brae
