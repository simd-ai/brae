#pragma once
// brae::FvPatch, per-boundary-patch finite-volume addressing/geometry. Mirrors OpenFOAM
// fvPatch: faceCells (owner cell of each boundary face) and boundary deltaCoeffs
// (1/|Cf - C_own|, basicFvGeometryScheme). Boundary face Sf/magSf/Cf are reached via the
// global face index (start + i) from FvGeometry.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include <string>
#include <vector>

namespace brae {

struct FvPatch {
    std::string         name;
    std::string         type;
    std::vector<std::string> inGroups;  // groups this patch belongs to (boundaryField group-keyword match)
    label               start = 0;
    label               size  = 0;
    std::vector<label>  faceCells;    // owner cell of each boundary face
    std::vector<scalar> deltaCoeffs;  // 1/|Cf - C[faceCell]|
    std::vector<vector> nf;           // unit face normal Sf/|Sf| (for slip/symmetry projection)
    std::vector<scalar> magSf;        // |Sf| face area (flowRateInletVelocity: gSum(rho*magSf))
    std::vector<vector> Cf;           // face centre (timeVaryingMapped boundaryData -> face mapping)
    // boundBox(patch.localPoints()).min() -- the componentwise minimum of the patch's OWN points. OF's
    // atmBoundaryLayer measures its profile height from `zDir & ppMin`, not from z = 0, so on a case
    // whose terrain sits at a real elevation this is the difference between a boundary layer and a
    // logarithm of the altitude. It is the patch POINTS, not the face centres: the lowest point of a
    // patch is below the lowest face centre on it.
    vector              ppMin{0, 0, 0};

    // THE COUPLED HALF of a `cyclic` patch, filled by attachCyclicCoupling() and left empty by
    // buildPatches(). It is opt-in because the drivers outside the OF-mirror tree couple their cyclics
    // through CyclicInterface and their own assembly, and an operator that suddenly branched on
    // `coupled` would change what they compute. A driver that calls attachCyclicCoupling() is saying
    // that every operator it uses treats a coupled face as OpenFOAM does: as a face between TWO
    // cells, interpolated from the cells and never from a stored patch value.
    //
    // cyclicFvPatch.C: delta = (Cf - Cn) - (Cf - Cn)_nbr;  w = dn_nbr/(dn + dn_nbr) with dn = nf & (Cf - Cn).
    // basicFvGeometryScheme.C: on a coupled patch deltaCoeffs = 1/|delta| (which REPLACES the
    // uncoupled 1/(nf & (Cf - Cn)) above), nonOrthDeltaCoeffs = 1/max(nf & delta, 0.05|delta|), and
    // nonOrthCorrectionVectors = nf - delta*nonOrthDeltaCoeffs -- zero on every uncoupled patch.
    bool                coupled = false;
    label               nbrPatch = -1;
    // cyclicPolyPatch::owner(): index() < neighbPatchID(). A jump is stored on the owner and read,
    // negated, by the other side.
    bool                owner = false;
    // Coupled through an AMI (a cyclicACMI pair, cyclic_acmi_cpp): the interpolation is the cyclic's,
    // but syncTools::syncBoundaryFaceList exchanges values across processor and cyclicPolyPatch only
    // (syncToolsTemplates.C:1223), and cyclicAMIPolyPatch derives from coupledPolyPatch. So whatever
    // OpenFOAM syncs with syncFaceList -- MULES' limiter -- is NOT synced across this pair.
    bool                ami = false;
    // a translational cyclic (and a coincident ACMI): the neighbour patch's face cells, face for face.
    // EMPTY on a cyclicAMI, whose faces meet their neighbours through the stencil below -- read the
    // neighbour's value through patchNeighbourValue, never through this.
    std::vector<label>  nbrFaceCells;
    // cyclicAMI (cyclic_ami_cpp): face i's neighbour value is the AMI's weighted sum over the neighbour
    // patch's face cells, slots amiOffsets[i] .. amiOffsets[i + 1] (AMIInterpolation::weightedSum with
    // multiplyWeightedOp: the result starts at zero and takes += w*value, slot by slot). The owner side
    // holds the AMI's src addressing, the other side its tgt addressing (interpolateToSource/-Target).
    std::vector<label>  amiOffsets;
    std::vector<label>  amiNbrFaces;
    std::vector<label>  amiNbrCells;
    std::vector<scalar> amiWeights;
    std::vector<scalar> weights;
    std::vector<vector> delta;
    std::vector<scalar> nonOrthDeltaCoeffs;
    std::vector<vector> nonOrthCorrectionVectors;
};

// The value OpenFOAM's linear scheme puts on a coupled face: w*pif + (1 - w)*pnf, from the two CELLS
// (surfaceInterpolationScheme.C, the pLambda branch under vf.boundaryField()[pi].coupled()). Never from a
// stored patch value -- HbyA's cyclic patch holds interpolate(rAU)*interpolate(H), and fvc::flux(HbyA)
// does not read it.
// patchNeighbourField at one face: the neighbour cell's value across a cyclic, the AMI's weighted sum of
// the neighbour patch's cells across a cyclicAMI
template <typename T>
inline T patchNeighbourValue(
    const FvPatch& p,
    label i,
    const std::vector<T>& cells)
{
    const std::size_t k = static_cast<std::size_t>(i);
    if (p.amiOffsets.empty())
    {
        return cells[p.nbrFaceCells[k]];
    }
    T r{};
    for (label s = p.amiOffsets[k]; s < p.amiOffsets[k + 1]; ++s)
    {
        r += p.amiWeights[static_cast<std::size_t>(s)]*cells[p.amiNbrCells[static_cast<std::size_t>(s)]];
    }
    return r;
}

// ...and of a field held on the NEIGHBOUR PATCH's faces (cyclicAMIPolyPatch::interpolate of a patch
// field, as cyclicAMIFvPatch::delta and makeWeights take it). A cyclic's face i meets the neighbour's
// face i.
template <typename T>
inline T patchNeighbourFaceValue(
    const FvPatch& p,
    label i,
    const std::vector<T>& nbrFaces)
{
    const std::size_t k = static_cast<std::size_t>(i);
    if (p.amiOffsets.empty())
    {
        return nbrFaces[k];
    }
    T r{};
    for (label s = p.amiOffsets[k]; s < p.amiOffsets[k + 1]; ++s)
    {
        r += p.amiWeights[static_cast<std::size_t>(s)]*nbrFaces[p.amiNbrFaces[static_cast<std::size_t>(s)]];
    }
    return r;
}

template <typename T>
inline T coupledLinear(
    const FvPatch& p,
    label i,
    const std::vector<T>& cells)
{
    const std::size_t k = static_cast<std::size_t>(i);
    return p.weights[k]*cells[p.faceCells[k]] + (scalar(1) - p.weights[k])*patchNeighbourValue(p, i, cells);
}

// Fill the coupled half of `p` against its neighbour `q` (index `nbr`), a translational pair whose
// faces meet face for face: the weights, delta and non-orthogonal vectors of cyclicFvPatch.C and
// basicFvGeometryScheme.C. attachCyclicCoupling uses it for every `cyclic`; cyclic_acmi_cpp for a
// coincident cyclicACMI pair, whose AMI maps each face onto its twin with weight 1.
void coupleTranslationalPair(
    FvPatch& p,
    const FvPatch& q,
    label nbr,
    bool owner,
    const FvGeometry& g);

// Fill the coupled half of every `cyclic` patch. Refuses a rotational pair by name: the vector
// transform is not carried through the operators that branch on `coupled`.
void attachCyclicCoupling(
    std::vector<FvPatch>& patches,
    const PrimitiveMesh& m,
    const FvGeometry& g);

// mirrorACMI: the caller is the OF-mirror interFoam host loop, which couples a coincident cyclicACMI
// pair itself (cpu::cyclicACMI::setup); every other caller has the refusal below.
std::vector<FvPatch> buildPatches(const PrimitiveMesh& m, const FvGeometry& g, bool mirrorACMI = false);

} // namespace brae
