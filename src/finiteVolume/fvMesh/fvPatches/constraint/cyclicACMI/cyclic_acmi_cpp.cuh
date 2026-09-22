#pragma once
// A COINCIDENT cyclicACMI pair on the host, for the OF-mirror interFoam loop: the coupling, the overlap
// mask with its `scale`, and the face areas the mask splits between the coupled patch and its
// non-overlap patch, re-evaluated every time step.
//
// provenance:
//   openfoam: src/meshTools/AMIInterpolation/patches/cyclicACMI/cyclicACMIPolyPatch/cyclicACMIPolyPatch.C
//               :47-138  updateAreas: once per time index, scaledMask = min(1 - tol, max(tol,
//                        scale(timeOutputValue)*mask)) on each side, the neighbour's scale a clone of the
//                        owner's evaluated on the NEIGHBOUR patch (:107-110)
//               :187-297 scalePatchFaceAreas: non-overlap Sf = raw*(1 - min(max(mask, tol), 1 - tol)),
//                        coupled Sf = raw*max(tol, mask), then the face cells' centres and volumes
//               :306-351 resetAMI: mask = clamp(AMI weight sum, 0, 1)
//             src/finiteVolume/fvMesh/fvPatches/constraint/cyclicACMI/cyclicACMIFvPatch.C
//               :46-91   updateAreas/resetPatchAreas: the fvPatch Sf, Cf, magSf taken from the new areas
//               :94-137  makeWeights: w = dn_nbr/(dn + dn_nbr), as a cyclic's
//             src/finiteVolume/fields/fvPatchFields/constraint/cyclicACMI/cyclicACMIFvPatchField.C
//               patchNeighbourField: the AMI interpolate of the neighbour's cells; updateCoeffs hands
//               (1 - mask) to the non-overlap patch field, which for symmetry does nothing
//   brae:     acmi_area_scaling.cuh is the device drivers' ACMI, one scale value per step; this is the
//             host OF-mirror one, which also evaluates a per-face coded scale (codedPatchFunction1.cuh)
//   tests:    tests/interfoam_leakage_vs_openfoam.sh against real OpenFOAM on RAS/damBreakLeakage
//
// WHAT "COINCIDENT" BUYS. createBaffles turns internal faces into the pair, so every face of one side
// lies exactly on one face of the other. OpenFOAM's AMI then gives each face ONE partner, its twin, with
// weight 1 after the renormalisation scalePatchFaceAreas applies -- measured on damBreakLeakage, all 13
// faces `1(i)` with weight `1(1)`. The AMI interpolate is then a copy, and cyclicACMIFvPatchField is the
// cyclic's CoupledCyclicPatchField on the scaled areas. Anything else -- faces that slide, overlap in
// part, or pair in another order -- makes the weights fractions, and is refused by name.
//
// WHEN, IN THE STEP -- and it is part of the answer. OpenFOAM rescales lazily, in two levels: the
// polyPatch areas and the face cells' volumes when the alpha pre-solve's matrix is constructed (its
// updateCoeffs asks for the mask), the fvPatch |Sf| at the first interpolation across the pair, inside
// that solve. Everything before it in the step saw the OLD areas, and one thing is formed there:
// alphaEqn.H's phic = cAlpha*|phi/magSf| (:59). On the face that just opened it is |phi| over the
// CLOSED area -- 41.7 on damBreakLeakage, where the open area gives 4e-9 -- and it multiplies the nHatf
// mixture.correct() then builds on the OPEN area. MEASURED with OpenFOAM's alphaEqn instrumented: the
// limited correction cancels the pre-solve's flux on the receiving side exactly, and rescaling at the
// top of the step instead put 5e-10 of water into the air cell that OpenFOAM leaves dry. So the driver
// rescales through alphaEqnStep's geometryUpdate, after phic and before the pre-solve, once per step.
//
// NOT PORTED, refused: a pair that is not coincident one to one, a rotational pair, a mesh that also
// moves (OpenFOAM re-runs the AMI and rescales the mesh flux in movePoints), a moving scale under the
// explicit MULES path, icAlpha, scAlpha or alpha sub-cycling (each moves the rescale point), and the
// device loop.
#include "cf_types.cuh"
#include "codedPatchFunction1.cuh"
#include "function1.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace cyclicACMI {

// cyclicAMIPolyPatch.C:48
constexpr scalar tolerance = 1e-10;

// One side of a pair: its coupled patch, its non-overlap patch, their raw areas, and its scale.
struct Side
{
    label patch = -1;
    label nonOverlap = -1;
    std::vector<vector> rawSf;
    std::vector<vector> rawNonOverlapSf;
    // clamp(AMI weight sum, 0, 1); 1 on a coincident pair (see the header)
    std::vector<scalar> mask;
    // the mask the scale made, as the last rescale left it (the unscaled mask when there is no scale)
    std::vector<scalar> scaledMask;
    // `scale`: none, one Function1 of time, or a coded PatchFunction1 (the owner's, cloned)
    Function1 scale;
    std::shared_ptr<CodedPatchFunction1> coded;
};

class Interfaces
{
public:
    bool empty() const { return sides_.empty(); }
    // true when some pair has a `scale`, and so must be rescaled every time step
    bool scaled() const;
    const std::vector<Side>& sides() const { return sides_; }

    // cyclicACMIPolyPatch::updateAreas + cyclicACMIFvPatch::updateAreas, once per time index, at time
    // `t` (OpenFOAM's timeOutputValue; brae has no user time). A no-op without a scale, as OpenFOAM's is.
    void rescale(
        scalar t,
        const PrimitiveMesh& m,
        FvGeometry& g,
        std::vector<FvPatch>& patches);

    // for setup() only
    std::vector<Side>& sidesRef() { return sides_; }

private:
    std::vector<Side> sides_;
};

// Couples every cyclicACMI pair of the mesh, after checking it is coincident, and applies the masks at
// the start time t0 (time index 0). `g` is rebuilt from the scaled areas -- centres, volumes AND the
// interpolation factors, which OpenFOAM computes from the geometry after the first rescale -- and
// `patches` rebuilt from `g`. Returns an empty set on a mesh without one. Call it BEFORE
// attachCyclicCoupling, which couples the plain cyclics of the rebuilt patches.
Interfaces setup(
    const PrimitiveMesh& m,
    FvGeometry& g,
    std::vector<FvPatch>& patches,
    scalar t0);

} // namespace cyclicACMI
} // namespace cpu
} // namespace brae
