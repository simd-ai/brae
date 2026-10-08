#pragma once
// cyclicACMI area scaling -- OF cyclicACMIPolyPatch::scalePatchFaceAreas (cyclicACMIPolyPatch.C:187).
//
// THE MECHANISM. A cyclicACMI interface is TWO coincident patches over the same faces: the coupled
// patch and a wall (its `nonOverlapPatch`). OF's blockMeshDict literally lists the same block face
// under both -- verified on the generated mesh of pimpleFoam/RAS/oscillatingInletACMI2D, where the two
// agree to |dCf| = 0 and |dArea| = 0. The overlap mask splits the area between them:
//
//     coupled  Sf = Sf_raw * max(tol, mask)                        (cyclicACMIPolyPatch.C:252)
//     wall     Sf = Sf_raw * (1 - min(max(mask, tol), 1 - tol))    (cyclicACMIPolyPatch.C:226)
//
// with tol = cyclicAMIPolyPatch::tolerance_ = 1e-10 (cyclicAMIPolyPatch.C:48). The tolerance only keeps
// either side from reaching exactly zero area; the two scales sum to 1 to within it.
//
// So a face that has slid off its neighbour (mask -> 0) hands essentially all of its area to the wall
// and becomes solid, instead of staying coupled to nothing and leaking. That is the whole point of
// ACMI, and it is carried entirely by the geometry -- no special discretisation.
//
// WHY THE ORDER MATTERS, in OF's own words (cyclicACMIPolyPatch.C:452):
//
//     "Initialise the AMI early to make sure we adapt the face areas before the cell centre
//      calculation gets triggered."
//
// The same face is in the mesh twice. Cell volumes come from a pyramid decomposition over a cell's
// faces, so before scaling that face contributes TWICE and every adjacent cell has the wrong volume.
// Scaling first makes the pair contribute once in total. Hence the FvGeometry seam: face geometry ->
// mask -> scale -> cell geometry.
//
// FROM RAW, ALWAYS. OF rescales from the primitivePatch areas -- "using primitivePatch face areas
// since these are based on the raw point locations (not affected by ACMI scaling)" -- never from an
// already-scaled value, which would compound the mask on every mesh move until the interface vanished.
// Here that invariant is enforced by FvGeometry::applyAreaScaling, which refuses a second scale unless
// buildFaceGeometry has recomputed the areas from the points in between.
//
// REFUSALS, not warnings. OF warns and returns when the coupled and non-overlap patches disagree in
// size ("This is OK for decomposition but should be considered fatal at run-time"). At run time brae
// refuses: a mismatched or non-coincident pair means the coincidence assumption this whole mechanism
// rests on is false, and the alternative is a silently mis-split interface.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ami_interface.cuh"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brae {

// OF cyclicAMIPolyPatch.C:48. Not a blend width -- purely a guard against exactly-zero face areas.
constexpr scalar ACMI_TOLERANCE = 1e-10;

// Scale the coupled/non-overlap face areas of every cyclicACMI interface in `amis`.
// Call between FvGeometry::buildFaceGeometry and FvGeometry::buildCellGeometry. A no-op (and no
// scaling flag set) when there is no ACMI interface, so non-ACMI meshes are untouched.
// `t` is the current time, for the optional per-patch `scale` Function1 (OF cyclicACMIPolyPatch's
// srcScalePtr_). With no scale entry the value is 1 and this is the geometric mask alone.
inline void applyACMIAreaScaling(
    const PrimitiveMesh& m,
    FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::vector<AMIInterface>& amis,
    scalar t = 0)
{
    std::vector<std::pair<label, scalar>> faceScale;

    for (const AMIInterface& ai : amis)
    {
        if (!ai.acmi) continue;

        const FvPatch& cpl = fvp[ai.patch];
        const std::string& wallName = m.patches()[ai.patch].nonOverlapPatch;
        if (wallName.empty())
            throw std::runtime_error(
                "brae: cyclicACMI patch '" + cpl.name + "' has no nonOverlapPatch. ACMI splits each "
                "face between the coupled patch and a coincident wall; without that wall the uncovered "
                "area has nowhere to go and the interface would leak wherever it stops overlapping.");

        label wallIdx = -1;
        for (std::size_t p = 0; p < fvp.size(); ++p)
            if (fvp[p].name == wallName) { wallIdx = static_cast<label>(p); break; }
        if (wallIdx < 0)
            throw std::runtime_error(
                "brae: cyclicACMI patch '" + cpl.name + "' names nonOverlapPatch '" + wallName +
                "', which is not in the mesh.");

        const FvPatch& wall = fvp[wallIdx];
        if (wall.size != cpl.size)
            throw std::runtime_error(
                "brae: cyclicACMI '" + cpl.name + "' has " + std::to_string(cpl.size) + " faces but its "
                "nonOverlapPatch '" + wallName + "' has " + std::to_string(wall.size) + ". They must be "
                "the SAME faces -- the area of each is split between them by the overlap mask -- so a "
                "size mismatch means the pair is not the coincident pair ACMI requires.");
        if (static_cast<label>(ai.mask.size()) != cpl.size)
            throw std::runtime_error(
                "brae: cyclicACMI '" + cpl.name + "' mask has " + std::to_string(ai.mask.size()) +
                " entries for " + std::to_string(cpl.size) + " faces.");

        // OF cyclicACMIPolyPatch::updateAreas:
        //     scaledMask = min(1 - tol, max(tol, scale(t)*mask))
        // evaluated once per patch per time step. Note the clamp is applied to the SCALED mask, so a
        // scale of 0 leaves tol (not 0) coupled and 1 - tol blocked -- the interface closes but the
        // patch never becomes degenerate.
        const PatchInfo& pinfo = m.patches()[ai.patch];
        // a per-face coded scale is evaluated by the interFoam host loop only (cyclic_acmi_cpp); here
        // an empty acmiScale would read as "no scale" and the interface would run fully open
        if (pinfo.acmiScaleCoded)
            throw std::runtime_error(
                "brae: cyclicACMI '" + cpl.name + "' has `scale { type coded; }`, a per-face PatchFunction1. "
                "This solver path evaluates a scale that is one number per time step (`constant`, "
                "`table`); the coded one is carried by interFoam's host loop only. Refused rather than "
                "run with the interface fully open.");
        const scalar sc = pinfo.acmiScale.empty() ? scalar(1) : pinfo.acmiScale.value(t);

        for (label i = 0; i < cpl.size; ++i)
        {
            const label fc = cpl.start + i;
            const label fw = wall.start + i;

            // The pair must be face-for-face coincident: OF indexes the mask with the same face index
            // on both patches. If the ordering ever differed, the mask would be applied to the wrong
            // face and the interface would be mis-split in a way no residual would reveal.
            const vector d = g.Cf()[fc] - g.Cf()[fw];
            const scalar tolCf = 1e-10*std::sqrt(std::max(g.magSf()[fc], scalar(1e-300)));
            if (mag(d) > std::max(tolCf, scalar(1e-12)))
                throw std::runtime_error(
                    "brae: cyclicACMI '" + cpl.name + "' face " + std::to_string(i) + " is not coincident "
                    "with '" + wallName + "' face " + std::to_string(i) + " (centres differ by " +
                    std::to_string(mag(d)) + "). ACMI splits ONE face's area between the two patches, so "
                    "they must be the same face in the same order.");

            const scalar mk = pinfo.acmiScale.empty()
                            ? ai.mask[i]
                            : std::min(scalar(1) - ACMI_TOLERANCE,
                                       std::max(ACMI_TOLERANCE, sc*ai.mask[i]));
            faceScale.emplace_back(fc, std::max(ACMI_TOLERANCE, mk));
            faceScale.emplace_back(
                fw, scalar(1) - std::min(std::max(mk, ACMI_TOLERANCE), scalar(1) - ACMI_TOLERANCE));
        }
    }

    if (std::getenv("BRAE_AMI_REPORT"))
        for (const AMIInterface& ai : amis)
        {
            if (!ai.acmi) continue;
            scalar lo = 1e300, hi = -1e300, sum = 0;
            for (const scalar mk : ai.mask) { lo = std::min(lo, mk); hi = std::max(hi, mk); sum += mk; }
            std::fprintf(stderr, "ACMI: patch:%s mask min:%g max:%g average:%g  scale(t):%g\n",
                         fvp[ai.patch].name.c_str(), (double)lo, (double)hi,
                         (double)(ai.mask.empty() ? 0 : sum/(scalar)ai.mask.size()),
                         (double)(m.patches()[ai.patch].acmiScale.empty()
                                  ? scalar(1) : m.patches()[ai.patch].acmiScale.value(t)));
        }
    if (!faceScale.empty()) g.applyAreaScaling(faceScale);
}

// True if any patch is a cyclicACMI -- lets a caller take the split-geometry path only when needed.
inline bool hasACMI(const std::vector<FvPatch>& fvp)
{
    for (const FvPatch& p : fvp) if (p.type == "cyclicACMI") return true;
    return false;
}

inline bool meshHasACMI(const PrimitiveMesh& m)
{
    for (const PatchInfo& p : m.patches()) if (p.type == "cyclicACMI") return true;
    return false;
}

// Geometry + patches + AMI, in the ONE order that is correct for cyclicACMI. Every caller that builds
// these three together should use this rather than open-coding the sequence, because two of the steps
// are order-critical in opposite directions and getting either wrong fails silently:
//
//   * the MASK and the WEIGHTS must be computed from RAW areas. Normalising the overlap by an
//     already-scaled area returns 1 for every face -- the mask erases itself and the interface reverts
//     to fully coupled.
//   * the CELL VOLUMES must be computed from SCALED areas. A cyclicACMI face is in the mesh twice
//     (coupled patch + coincident wall), so before scaling it is counted twice and every adjacent cell
//     is wrong -- 14% on the two-block fixture.
//
// Without an ACMI patch this is exactly the old three-line sequence, so non-ACMI meshes are unchanged
// and pay nothing.
// Rebuild geometry for a MOVED mesh with the ACMI split re-applied, from raw areas, in the right
// order. fvp is only read (to locate the patches), never rebuilt -- the moving-mesh path already
// rebuilds the device buffers from a fresh buildDeviceMesh. On a mesh without cyclicACMI this is
// exactly g.build(m).
//
// Re-scaling every step is required, not an optimisation: the mask changes as the zone slides (OF
// re-runs resetAMI + scalePatchFaceAreas from initMovePoints, cyclicACMIPolyPatch.C:470-487). Holding
// the first step's mask would freeze the interface in its t=0 state -- fully open, in the
// oscillatingInletACMI2D case, so the channel would never close.
inline void rebuildGeometryWithACMI(
    const PrimitiveMesh& m,
    FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    scalar t = 0)
{
    g.build(m);
    if (!meshHasACMI(m)) return;
    const std::vector<AMIInterface> raw = buildAMIInterfaces(m, g, fvp);
    g.buildFaceGeometry(m);
    applyACMIAreaScaling(m, g, fvp, raw, t);
    g.buildCellGeometry(m);
}

// True if any cyclicACMI carries a `scale` -- the interface then changes with TIME even on a mesh that
// never moves, so the geometry has to be rebuilt every step (TJunctionSwitching is static).
inline bool hasACMITimeScale(const PrimitiveMesh& m)
{
    for (const PatchInfo& p : m.patches())
        if (p.type == "cyclicACMI" && (!p.acmiScale.empty() || p.acmiScaleCoded)) return true;
    return false;
}

inline void buildGeometryPatchesAndAMI(
    const PrimitiveMesh& m,
    FvGeometry& g,
    std::vector<FvPatch>& fvp,
    std::vector<AMIInterface>& amis,
    scalar t = 0)
{
    g.build(m);
    fvp = buildPatches(m, g);

    if (!meshHasACMI(m))
    {
        amis = buildAMIInterfaces(m, g, fvp);
        return;
    }

    // Pass 1: raw geometry. Cell volumes are wrong here (the duplicated face is double-counted) and the
    // patch deltaCoeffs with them, but the MASK depends only on the face polygons and is already right.
    const std::vector<AMIInterface> raw = buildAMIInterfaces(m, g, fvp);

    // Split the duplicated faces' areas, then recompute cell geometry from the scaled areas.
    g.buildFaceGeometry(m);
    applyACMIAreaScaling(m, g, fvp, raw, t);   // stashes the pre-scale areas in g for the weights below
    g.buildCellGeometry(m);

    // Pass 2: everything downstream of the corrected cell centres -- patch deltaCoeffs, and the AMI's
    // own deltas and flux areas -- rebuilt. The weights are still normalised by the RAW areas, so the
    // mask survives and the weight sums stay on OF's ACMI (non-conformal) branch.
    fvp  = buildPatches(m, g);
    amis = buildAMIInterfaces(m, g, fvp);
}

}   // namespace brae
