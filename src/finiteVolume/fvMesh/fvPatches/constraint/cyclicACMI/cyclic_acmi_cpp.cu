// A coincident cyclicACMI pair on the host -- see cyclic_acmi_cpp.cuh.
#include "cyclic_acmi_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace cyclicACMI {

namespace {

label patchIndex(
    const std::vector<FvPatch>& patches,
    const std::string& name)
{
    for (std::size_t i = 0; i < patches.size(); ++i)
    {
        if (patches[i].name == name)
        {
            return static_cast<label>(i);
        }
    }
    return -1;
}

// The two faces are one face: the same centre, to round-off of the patch's size.
bool sameFace(
    const vector& a,
    const vector& b,
    scalar area)
{
    return mag(a - b) <= std::fmax(scalar(1e-10)*std::sqrt(area), scalar(1e-14));
}

// scalePatchFaceAreas (cyclicACMIPolyPatch.C:218-258) for one side, from its raw areas
void applySide(
    const Side& s,
    FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const FvPatch& cpl = patches[static_cast<std::size_t>(s.patch)];
    const FvPatch& no = patches[static_cast<std::size_t>(s.nonOverlap)];
    const scalar maxTol = scalar(1) - tolerance;
    for (label i = 0; i < cpl.size; ++i)
    {
        const std::size_t k = static_cast<std::size_t>(i);
        const scalar mk = s.scaledMask[k];
        // non-overlap: scale = 1 - min(max(mask, tol), 1 - tol)
        const scalar noScale = scalar(1) - std::min(std::max(mk, tolerance), maxTol);
        g.setFaceArea(no.start + i, s.rawNonOverlapSf[k]*noScale);
        // coupled: scale = max(tol, mask)
        g.setFaceArea(cpl.start + i, s.rawSf[k]*std::max(tolerance, mk));
    }
}

// cyclicACMIFvPatch::resetPatchAreas: the fvPatch's magSf (and nf = Sf/magSf, which OpenFOAM forms on
// every call) from the new areas
void resetPatchAreas(
    FvPatch& p,
    const FvGeometry& g)
{
    for (label i = 0; i < p.size; ++i)
    {
        const label f = p.start + i;
        p.magSf[static_cast<std::size_t>(i)] = g.magSf()[f];
        p.nf[static_cast<std::size_t>(i)] = g.Sf()[f]/g.magSf()[f];
    }
}

// updateAreas' mask: min(1 - tol, max(tol, scale(t)*mask)), per face
void scaleMask(
    Side& s,
    scalar t,
    const std::vector<FvPatch>& patches)
{
    const FvPatch& cpl = patches[static_cast<std::size_t>(s.patch)];
    const std::size_t n = static_cast<std::size_t>(cpl.size);
    std::vector<scalar> sc(n, scalar(1));
    if (s.coded)
    {
        sc = s.coded->value(t, cpl.name, cpl.Cf, t, t);
    }
    else if (!s.scale.empty())
    {
        std::fill(sc.begin(), sc.end(), s.scale.value(t));
    }
    s.scaledMask.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        s.scaledMask[i] = std::min(scalar(1) - tolerance, std::max(tolerance, sc[i]*s.mask[i]));
    }
}

} // namespace


bool Interfaces::scaled() const
{
    for (const Side& s : sides_)
    {
        if (s.coded || !s.scale.empty())
        {
            return true;
        }
    }
    return false;
}


void Interfaces::rescale(
    scalar t,
    const PrimitiveMesh& m,
    FvGeometry& g,
    std::vector<FvPatch>& patches)
{
    if (!scaled())
    {
        return;
    }
    for (Side& s : sides_)
    {
        scaleMask(s, t, patches);
        applySide(s, g, patches);
    }
    g.updateCellCentresAndVols(m);
    for (const Side& s : sides_)
    {
        resetPatchAreas(patches[static_cast<std::size_t>(s.patch)], g);
        resetPatchAreas(patches[static_cast<std::size_t>(s.nonOverlap)], g);
    }
}


Interfaces setup(
    const PrimitiveMesh& m,
    FvGeometry& g,
    std::vector<FvPatch>& patches,
    scalar t0)
{
    Interfaces acmi;
    const std::vector<PatchInfo>& info = m.patches();
    std::vector<std::pair<label, label>> pairs;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "cyclicACMI")
        {
            continue;
        }
        const std::string who = "brae interFoam: cyclicACMI '" + patches[pi].name + "'";
        const label nbr = patchIndex(patches, info[pi].neighbourPatch);
        if (nbr < 0 || patches[static_cast<std::size_t>(nbr)].type != "cyclicACMI"
         || info[static_cast<std::size_t>(nbr)].neighbourPatch != patches[pi].name)
        {
            throw std::runtime_error(
                who + " names the neighbourPatch '" + info[pi].neighbourPatch + "', which is not a "
                "cyclicACMI naming it back.");
        }
        if (info[pi].transform == "rotational")
        {
            throw std::runtime_error(
                who + " is rotational. The host loop couples a translational pair only.");
        }
        if (static_cast<label>(pi) < nbr)
        {
            pairs.emplace_back(static_cast<label>(pi), nbr);
        }
    }
    if (pairs.empty())
    {
        return acmi;
    }

    for (const auto& pr : pairs)
    {
        for (const label pi : {pr.first, pr.second})
        {
            const FvPatch& p = patches[static_cast<std::size_t>(pi)];
            const label other = (pi == pr.first) ? pr.second : pr.first;
            const FvPatch& q = patches[static_cast<std::size_t>(other)];
            const std::string who = "brae interFoam: cyclicACMI '" + p.name + "'";
            const std::string& noName = info[static_cast<std::size_t>(pi)].nonOverlapPatch;
            const label no = patchIndex(patches, noName);
            if (noName.empty() || no < 0)
            {
                throw std::runtime_error(
                    who + " names no nonOverlapPatch the mesh has ('" + noName + "'). ACMI splits each face's "
                    "area between the coupled patch and that one; without it the covered area has nowhere "
                    "to go.");
            }
            const FvPatch& w = patches[static_cast<std::size_t>(no)];
            if (q.size != p.size || w.size != p.size)
            {
                throw std::runtime_error(
                    who + " has " + std::to_string(p.size) + " faces, its neighbour '" + q.name + "' " +
                    std::to_string(q.size) + " and its nonOverlapPatch '" + w.name + "' " +
                    std::to_string(w.size) + ". The host loop couples a COINCIDENT pair only, face for "
                    "face; any other pairing gives OpenFOAM's AMI fractional weights, which are not ported.");
            }
            Side s;
            s.patch = pi;
            s.nonOverlap = no;
            s.rawSf.resize(static_cast<std::size_t>(p.size));
            s.rawNonOverlapSf.resize(static_cast<std::size_t>(p.size));
            for (label i = 0; i < p.size; ++i)
            {
                const std::size_t k = static_cast<std::size_t>(i);
                const label f = p.start + i;
                const label fq = q.start + i;
                const label fw = w.start + i;
                const scalar a = g.magSf()[f];
                if (!sameFace(g.Cf()[f], g.Cf()[fq], a) || std::fabs(g.magSf()[fq] - a) > scalar(1e-10)*a)
                {
                    throw std::runtime_error(
                        who + " face " + std::to_string(i) + " does not coincide with face " +
                        std::to_string(i) + " of '" + q.name + "' (centres " + std::to_string(mag(g.Cf()[f] -
                        g.Cf()[fq])) + " apart). The host loop couples a COINCIDENT pair only, face for "
                        "face -- a baffle createBaffles made; a sliding or partly overlapping interface "
                        "gives OpenFOAM's AMI fractional weights, which are not ported.");
                }
                if (!sameFace(g.Cf()[f], g.Cf()[fw], a) || std::fabs(g.magSf()[fw] - a) > scalar(1e-10)*a)
                {
                    throw std::runtime_error(
                        who + " face " + std::to_string(i) + " does not coincide with face " +
                        std::to_string(i) + " of its nonOverlapPatch '" + w.name + "'. ACMI splits ONE "
                        "face's area between the two patches, so they must be the same face in the same "
                        "order.");
                }
                s.rawSf[k] = g.Sf()[f];
                s.rawNonOverlapSf[k] = g.Sf()[fw];
            }
            // resetAMI: clamp(weight sum, 0, 1). A coincident face's single partner covers it, so 1.
            s.mask.assign(static_cast<std::size_t>(p.size), scalar(1));
            s.scaledMask = s.mask;
            // the owner's scale; the neighbour evaluates a clone on its own faces (propagateACMIScale
            // copied the owner's entry across)
            const PatchInfo& pinfo = info[static_cast<std::size_t>(pi)];
            if (pinfo.acmiScaleCoded)
            {
                s.coded = std::make_shared<CodedPatchFunction1>(pinfo.acmiScaleCodedSpec);
            }
            else if (!pinfo.acmiScale.empty())
            {
                s.scale = pinfo.acmiScale;
            }
            acmi.sidesRef().push_back(std::move(s));
        }
    }

    // the areas at the start time: the scale at t0 where there is one, the unscaled mask where there is
    // not (the constructor's scalePatchFaceAreas, :372-416)
    for (Side& s : acmi.sidesRef())
    {
        if (s.coded || !s.scale.empty())
        {
            scaleMask(s, t0, patches);
        }
        else
        {
            s.scaledMask = s.mask;
        }
        applySide(s, g, patches);
    }
    // centres, volumes and the interpolation factors from the split areas, then the patches from them
    g.buildCellGeometry(m);
    patches = buildPatches(m, g, /*mirrorACMI=*/true);
    for (const auto& pr : pairs)
    {
        FvPatch& p = patches[static_cast<std::size_t>(pr.first)];
        FvPatch& q = patches[static_cast<std::size_t>(pr.second)];
        coupleTranslationalPair(p, q, pr.second, true, g);
        coupleTranslationalPair(q, p, pr.first, false, g);
        p.ami = true;
        q.ami = true;
    }
    return acmi;
}

} // namespace cyclicACMI
} // namespace cpu
} // namespace brae
