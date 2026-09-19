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
    std::vector<label>  nbrFaceCells;
    std::vector<scalar> weights;
    std::vector<vector> delta;
    std::vector<scalar> nonOrthDeltaCoeffs;
    std::vector<vector> nonOrthCorrectionVectors;
};

// The value OpenFOAM's linear scheme puts on a coupled face: w*pif + (1 - w)*pnf, from the two CELLS
// (surfaceInterpolationScheme.C, the pLambda branch under vf.boundaryField()[pi].coupled()). Never from a
// stored patch value -- HbyA's cyclic patch holds interpolate(rAU)*interpolate(H), and fvc::flux(HbyA)
// does not read it.
template <typename T>
inline T coupledLinear(
    const FvPatch& p,
    label i,
    const std::vector<T>& cells)
{
    const std::size_t k = static_cast<std::size_t>(i);
    return p.weights[k]*cells[p.faceCells[k]] + (scalar(1) - p.weights[k])*cells[p.nbrFaceCells[k]];
}

// Fill the coupled half of every `cyclic` patch. Refuses a rotational pair by name: the vector
// transform is not carried through the operators that branch on `coupled`.
void attachCyclicCoupling(
    std::vector<FvPatch>& patches,
    const PrimitiveMesh& m,
    const FvGeometry& g);

std::vector<FvPatch> buildPatches(const PrimitiveMesh& m, const FvGeometry& g);

} // namespace brae
