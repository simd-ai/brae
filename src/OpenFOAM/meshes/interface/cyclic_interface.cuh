#pragma once
// brae::CyclicInterface, the cyclic (periodic) lduInterface. Structurally identical to the processor
// interface (result[faceCells] -= coeff * psi[nbrFaceCells]) but the neighbour cells are LOCAL (same
// mesh, no MPI) and a transform is applied to vectors/tensors (identity for translational periodicity,
// a rotation for rotational). Mirrors OpenFOAM cyclicPolyPatch / cyclicFvPatch / cyclicFvPatchField.
//
// Geometry (cyclicFvPatch::delta / makeWeights), translational (transform = I):
//   patchD = Cf_own - C_own;  nbrD = Cf_nbr - C_nbr;  delta = patchD - nbrD
//   deltaCoeffs = 1 / max(nf_own & delta, 0.05*mag(delta))
//   w = (nf_nbr & nbrD) / (nf_own & patchD + nf_nbr & nbrD)
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cf_types.cuh"
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

struct CyclicInterface
{
    label patch = -1, nbrPatch = -1;        // this cyclic patch and its paired neighbour (fv patch indices)
    std::vector<label>  faceCells;          // owner cells of this patch's faces
    std::vector<label>  nbrFaceCells;       // owner cells of the matched neighbour faces (coupling target)
    std::vector<scalar> deltaCoeffs;        // coupled 1/(nf & delta)
    std::vector<scalar> weights;            // this-side interp weight: face = w*own + (1-w)*nbr
    std::vector<vector> dOwn;               // Cf - C[own] (this-side face delta, for linearUpwind reconstruction)
    std::vector<vector> dNbr;               // Cf_nbr - C[nbr] (neighbour-side face delta, UN-rotated)
    // fvPatch::delta() -- dOwn minus the TRANSFORMED neighbour delta, which is what deltaCoeffs and
    // corrVec are already built from. Stored rather than left for each consumer to re-derive: the TVD
    // limiter at a coupled face needs the same vector, and re-deriving it means re-deriving the rotation.
    std::vector<vector> delta;
    std::vector<vector> corrVec;            // non-orth correction vector nf - delta*deltaCoeffs (laplacian "corrected")
    bool                translational = true;
    vector              separation{0, 0, 0};   // translational: period vector Cf_nbr - Cf_own
    tensor              forwardT{1, 0, 0, 0, 1, 0, 0, 0, 1};   // rotational: nbr->own rotation (identity if translational)
};

// Rodrigues rotation tensor R about a unit-normalised axis by `angle` (R & v rotates v right-handed).
inline tensor rotationTensor(const vector& axis, scalar angle)
{
    const vector a = axis / mag(axis);
    const scalar c = std::cos(angle), s = std::sin(angle), t = 1.0 - c;
    return { c + t*a.x*a.x,     t*a.x*a.y - s*a.z, t*a.x*a.z + s*a.y,
             t*a.x*a.y + s*a.z, c + t*a.y*a.y,     t*a.y*a.z - s*a.x,
             t*a.x*a.z - s*a.y, t*a.y*a.z + s*a.x, c + t*a.z*a.z };
}

// Build the cyclic interfaces from the mesh. Faces are matched by stored order (OpenFOAM orders cyclic
// patches so face i pairs with neighbour face i); the constant separation vector is recorded so callers
// can verify the match geometrically.
inline std::vector<CyclicInterface> buildCyclicInterfaces(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    std::map<std::string, label> nameToIdx;
    for (label pi = 0; pi < (label)fvp.size(); ++pi) nameToIdx[fvp[pi].name] = pi;
    const std::vector<PatchInfo>& pinfo = m.patches();

    std::vector<CyclicInterface> out;
    for (label pi = 0; pi < (label)fvp.size(); ++pi)
    {
        if (fvp[pi].type != "cyclic") continue;
        CyclicInterface ci;
        ci.patch = pi;
        const std::string nbrName = pinfo[pi].neighbourPatch;
        const auto it = nameToIdx.find(nbrName);
        if (it == nameToIdx.end()) throw std::runtime_error("cyclic: neighbourPatch '" + nbrName + "' not found");
        ci.nbrPatch = it->second;
        ci.translational = (pinfo[pi].transform != "rotational");
        const FvPatch& P = fvp[pi];
        const FvPatch& N = fvp[ci.nbrPatch];
        if (P.size != N.size) throw std::runtime_error("cyclic: patch/neighbour face-count mismatch");
        ci.faceCells = P.faceCells;
        ci.nbrFaceCells = N.faceCells;
        ci.deltaCoeffs.resize(P.size);
        ci.weights.resize(P.size);
        ci.dOwn.resize(P.size);
        ci.dNbr.resize(P.size);
        ci.delta.resize(P.size);
        ci.corrVec.resize(P.size);

        // Rotational: the nbr->own rotation tensor forwardT, angle from a matched face pair about the axis.
        if (!ci.translational)
        {
            const vector a = pinfo[pi].rotationAxis / mag(pinfo[pi].rotationAxis);
            const vector ctr = pinfo[pi].rotationCentre;
            const vector po = g.Cf()[P.start] - ctr, pn = g.Cf()[N.start] - ctr;
            const vector perpO = po - dot(po, a) * a, perpN = pn - dot(pn, a) * a;   // components perpendicular to axis
            const scalar angle = std::atan2(dot(cross(perpN, perpO), a), dot(perpN, perpO));  // signed nbr->own angle
            ci.forwardT = rotationTensor(a, angle);
        }

        for (label i = 0; i < P.size; ++i)
        {
            const vector nfo = g.Sf()[P.start + i] / g.magSf()[P.start + i];   // own outward unit normal
            const vector nfn = g.Sf()[N.start + i] / g.magSf()[N.start + i];   // neighbour outward unit normal
            const vector patchD = g.Cf()[P.start + i] - g.C()[P.faceCells[i]];
            const vector nbrD   = g.Cf()[N.start + i] - g.C()[N.faceCells[i]];
            const vector nbrDt  = ci.translational ? nbrD : dot(nbrD, transpose(ci.forwardT));  // transform(forwardT, nbrD)
            const vector delta  = patchD - nbrDt;                              // delta = patchD - R*nbrD
            const scalar di = dot(nfo, patchD), dni = dot(nfn, nbrD);
            ci.weights[i]     = dni / (di + dni);
            ci.deltaCoeffs[i] = 1.0 / std::fmax(dot(nfo, delta), 0.05 * mag(delta));
            ci.corrVec[i] = nfo - delta * ci.deltaCoeffs[i];   // laplacian non-orth correction vector (k = nf - delta*dc)
            ci.dOwn[i] = patchD;
            ci.dNbr[i] = nbrD;   // linearUpwind face deltas (Cf-C_own ; Cf_nbr-C_nbr, un-rotated)
            ci.delta[i] = delta;
            if (i == 0) ci.separation = g.Cf()[N.start + i] - g.Cf()[P.start + i];
        }
        out.push_back(std::move(ci));
    }
    return out;
}

} // namespace brae
