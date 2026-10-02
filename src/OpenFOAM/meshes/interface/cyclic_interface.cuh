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
    // An AMI-family pair (cyclicAMI, cyclicACMI). ONE thing reads it: MULES' limiter sync, which is
    // syncTools::syncFaceList and covers processor and cyclicPolyPatch ONLY (syncToolsTemplates.C:
    // 1225-1234; cyclicAMIPolyPatch derives from coupledPolyPatch, cyclicAMIPolyPatch.H:70-72), so the
    // two sides of such a pair keep their OWN lambda. The host carries the same flag as FvPatch::ami.
    bool                ami = false;
    // A cyclicAMI pair: face i's neighbour value is the AMI's weighted sum over the neighbour patch's
    // face cells, slots amiOffsets[i] .. amiOffsets[i + 1] -- FvPatch's own stencil (fv_patch.cuh,
    // patchNeighbourValue), copied so the device reads the numbers the host does. Empty on a cyclic and
    // on a coincident ACMI, whose faces meet one neighbour cell (nbrFaceCells); nbrFaceCells is then -1.
    std::vector<label>  amiOffsets;
    std::vector<label>  amiNbrCells;
    std::vector<scalar> amiWeights;
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
//
// `includeCoupledACMI`: also take a cyclicACMI pair the caller has ALREADY COUPLED as a coincident
// translational pair (cpu::cyclicACMI::setup -- every face lies on one twin, AMI weight 1, so the AMI
// interpolate is a copy and the pair is a cyclic on the mask-scaled areas). OPT-IN, because the legacy
// drivers call this same function and couple an ACMI through AMIInterface/DeviceAMI instead; taking it
// here as well would couple it twice there.
inline std::vector<CyclicInterface> buildCyclicInterfaces(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    bool includeCoupledACMI = false,
    bool includeCoupledAMI = false)
{
    std::map<std::string, label> nameToIdx;
    for (label pi = 0; pi < (label)fvp.size(); ++pi) nameToIdx[fvp[pi].name] = pi;
    const std::vector<PatchInfo>& pinfo = m.patches();

    std::vector<CyclicInterface> out;
    for (label pi = 0; pi < (label)fvp.size(); ++pi)
    {
        const bool acmi = includeCoupledACMI && fvp[pi].type == "cyclicACMI" && fvp[pi].coupled;
        // `includeCoupledAMI`: a cyclicAMI pair the caller has ALREADY COUPLED (cpu::cyclicAMIFvPatch::
        // setup). Its geometry is NOT re-derived here: the weights, delta, delta coefficients and
        // correction vectors are the ones cyclic_ami_cpp put on the patch from the AMI, taken as they
        // stand so the device and the host cannot drift apart. OPT-IN for the reason ACMI is.
        if (includeCoupledAMI && fvp[pi].type == "cyclicAMI" && fvp[pi].coupled)
        {
            const FvPatch& P = fvp[pi];
            const std::size_t n = static_cast<std::size_t>(P.size);
            if (P.amiOffsets.size() != n + 1 || P.weights.size() != n || P.delta.size() != n
                || P.nonOrthDeltaCoeffs.size() != n || P.nonOrthCorrectionVectors.size() != n)
            {
                throw std::runtime_error(
                    "cyclicAMI: patch `" + P.name + "` is marked coupled but carries no AMI stencil or "
                    "coupled geometry; cpu::cyclicAMIFvPatch::setup has to run before the pair is built.");
            }
            CyclicInterface ci;
            ci.patch = pi;
            ci.nbrPatch = P.nbrPatch;
            ci.ami = true;
            ci.translational = true;
            ci.faceCells = P.faceCells;
            ci.nbrFaceCells.assign(n, label(-1));
            ci.amiOffsets = P.amiOffsets;
            ci.amiNbrCells = P.amiNbrCells;
            ci.amiWeights = P.amiWeights;
            ci.weights = P.weights;
            ci.deltaCoeffs = P.nonOrthDeltaCoeffs;
            ci.delta = P.delta;
            ci.corrVec = P.nonOrthCorrectionVectors;
            ci.dOwn.resize(n);
            ci.dNbr.resize(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                ci.dOwn[i] = g.Cf()[P.start + static_cast<label>(i)] - g.C()[P.faceCells[i]];
                // linearUpwind's neighbour half is (Cf - d - C[own]) & gradNbr (linearUpwind.C, the
                // coupled branch of correction()), d the patch's delta: with one neighbour cell that
                // is the neighbour's own Cf - C, and here it is whatever the AMI's delta leaves
                ci.dNbr[i] = ci.dOwn[i] - ci.delta[i];
            }
            out.push_back(std::move(ci));
            continue;
        }
        if (fvp[pi].type != "cyclic" && !acmi) continue;
        CyclicInterface ci;
        ci.patch = pi;
        ci.ami = acmi;
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
