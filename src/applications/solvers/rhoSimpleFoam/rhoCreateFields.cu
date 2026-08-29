// The device projection of rhoSimpleFoam's createFields.H. See rhoCreateFields.cuh for the contract and
// for what deliberately stays on the host.
#include "rhoCreateFields.cuh"
#include "near_wall_dist.cuh"
#include <stdexcept>

namespace brae {
namespace gpu {
namespace rhoSimple {

std::vector<scalar> flattenBoundary(
    const std::vector<std::vector<scalar>>& perPatch,
    const std::vector<FvPatch>&             patches,
    int                                     nBndFaces,
    scalar                                  pad)
{
    std::vector<scalar> out;
    out.reserve(static_cast<std::size_t>(nBndFaces));
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            out.push_back(pi < perPatch.size() && i < (label)perPatch[pi].size() ? perPatch[pi][i] : pad);
        }
    }
    // PADDED, not truncated. DeviceMesh's boundary gather SKIPS cyclic and processor faces, so nBndFaces
    // can exceed the sum of the patch sizes this loop walked; a short array would then be read past its
    // end by any kernel indexing on the device's own boundary count.
    out.resize(static_cast<std::size_t>(nBndFaces), pad);
    return out;
}


std::vector<scalar> flattenFieldBoundary(
    const GeometricField<scalar>& f,
    const std::vector<FvPatch>&   patches,
    int                           nBndFaces,
    scalar                        pad)
{
    std::vector<std::vector<scalar>> v(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) v[pi] = f.boundary[pi]->value();
    return flattenBoundary(v, patches, nBndFaces, pad);
}


std::vector<scalar> gatherWallFaces(
    const std::vector<scalar>& bndFaceValues,
    const std::vector<label>&  wfFaceOfBnd)
{
    std::vector<scalar> out(wfFaceOfBnd.size());
    for (std::size_t i = 0; i < wfFaceOfBnd.size(); ++i)
    {
        out[i] = bndFaceValues[static_cast<std::size_t>(wfFaceOfBnd[i])];
    }
    return out;
}


RhoDeviceFields createDeviceFields(
    const cpu::rhoSimple::RhoSimpleFields& hf,
    const PrimitiveMesh&                   m,
    const FvGeometry&                      g,
    const std::vector<FvPatch>&            patches)
{
    for (const FvPatch& p : patches)
    {
        if (p.type == "cyclic" || p.type == "cyclicAMI" || p.type == "cyclicACMI" || p.type == "processor")
        {
            throw std::runtime_error(
                "rhoSimpleFoam createFields(cuda): the mesh has a coupled patch ('" + p.name + "', type "
                + p.type + "). buildDeviceMesh keeps those faces out of the LDU, so every equation this "
                "field set feeds would lose their contribution silently. Refusing rather than building a "
                "device projection that is missing part of the mesh.");
        }
    }

    RhoDeviceFields d;
    const label nC = m.nCells();

    d.dm = buildDeviceMesh(m, g, patches);
    d.nBndFaces = d.dm.nBndFaces;
    d.turbulent = hf.turbulent;

    d.dbU  = buildDeviceVectorBoundary(hf.U, patches, g);
    d.dbP  = buildDeviceBoundary(hf.p, patches, g);
    d.dbHe = buildDeviceBoundary(hf.he, patches, g);
    d.dbT  = buildDeviceBoundary(hf.T, patches, g);

    // ---- the solution state ------------------------------------------------------------------
    {
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        {
            ux[c] = hf.U.internal[c].x;
            uy[c] = hf.U.internal[c].y;
            uz[c] = hf.U.internal[c].z;
        }
        d.f.Ux.copyFrom(ux);
        d.f.Uy.copyFrom(uy);
        d.f.Uz.copyFrom(uz);

        std::vector<std::vector<scalar>> bx(patches.size()), by(patches.size()), bz(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<vector>& b = hf.U.boundary[pi]->value();
            bx[pi].resize(b.size());
            by[pi].resize(b.size());
            bz[pi].resize(b.size());
            for (std::size_t i = 0; i < b.size(); ++i)
            {
                bx[pi][i] = b[i].x;
                by[pi][i] = b[i].y;
                bz[pi][i] = b[i].z;
            }
        }
        d.f.UxBnd.copyFrom(flattenBoundary(bx, patches, d.nBndFaces, 0.0));
        d.f.UyBnd.copyFrom(flattenBoundary(by, patches, d.nBndFaces, 0.0));
        d.f.UzBnd.copyFrom(flattenBoundary(bz, patches, d.nBndFaces, 0.0));
    }

    d.f.p.copyFrom(hf.p.internal);
    d.f.pBnd.copyFrom(flattenFieldBoundary(hf.p, patches, d.nBndFaces, 0.0));
    d.f.he.copyFrom(hf.he.internal);
    d.f.heBnd.copyFrom(flattenFieldBoundary(hf.he, patches, d.nBndFaces, 0.0));
    d.f.T.copyFrom(hf.T.internal);
    d.f.TBnd.copyFrom(flattenFieldBoundary(hf.T, patches, d.nBndFaces, 0.0));
    d.f.rho.copyFrom(hf.rho.internal);
    // rho pads with 1, not 0: a padded face that reached a divisor would produce an infinity rather than
    // a number that is merely wrong, and the two are not equally easy to notice.
    d.f.rhoBnd.copyFrom(flattenFieldBoundary(hf.rho, patches, d.nBndFaces, 1.0));
    d.f.psi.copyFrom(hf.psi);
    d.f.psiBnd.copyFrom(flattenBoundary(hf.psiBnd, patches, d.nBndFaces, 0.0));
    d.f.phiInt.copyFrom(hf.phi.internal);
    d.f.phiBnd.copyFrom(flattenBoundary(hf.phi.boundary, patches, d.nBndFaces, 0.0));
    d.f.initialMass = hf.initialMass;

    // ---- the two masks, and they are NOT the same question ------------------------------------
    {
        std::vector<label> takeU, adjustable;
        takeU.reserve(static_cast<std::size_t>(d.nBndFaces));
        adjustable.reserve(static_cast<std::size_t>(d.nBndFaces));
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (label i = 0; i < patches[pi].size; ++i)
            {
                // constrainHbyA: `assignable`. slip and inletOutlet are non-assignable WITHOUT fixing a
                // value, so this cannot be recovered from fixesValue below.
                takeU.push_back(hf.U.boundary[pi]->assignable() ? 0 : 1);
                // adjustPhi: `fixesValue() && !isInletOutlet()`, and BOTH halves matter --
                // mixedFvPatchField::fixesValue() is TRUE and inletOutlet inherits it, so testing
                // fixesValue alone marks an inletOutlet OUTLET as fixed outflow and leaves adjustPhi
                // nothing adjustable to balance the inflow against.
                const bool fixed =
                    hf.U.boundary[pi]->fixesValue() && !hf.U.boundary[pi]->isInletOutlet();
                adjustable.push_back(fixed ? 0 : 1);
            }
        }
        takeU.resize(static_cast<std::size_t>(d.nBndFaces), 0);
        adjustable.resize(static_cast<std::size_t>(d.nBndFaces), 0);
        d.takeUAtBoundary.copyFrom(takeU);
        d.adjustable.copyFrom(adjustable);
    }

    // ---- the closure's static geometry --------------------------------------------------------
    if (hf.turbulent && !hf.epsilon.internal.empty())
    {
        d.dbK   = buildDeviceBoundary(hf.k, patches, g);
        d.dbEps = buildDeviceBoundary(hf.epsilon, patches, g);

        d.f.k.copyFrom(hf.k.internal);
        d.f.epsilon.copyFrom(hf.epsilon.internal);
        d.f.nut.copyFrom(hf.nut.internal);
        d.f.nutBnd.copyFrom(flattenFieldBoundary(hf.nut, patches, d.nBndFaces, 0.0));
        if (!hf.alphat.internal.empty()) d.f.alphat.copyFrom(hf.alphat.internal);

        // The predicate is the BC's, not the patch type's -- see the header.
        std::vector<char> wfPatch(patches.size(), 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            wfPatch[pi] = hf.epsilon.boundary[pi]->isTurbulenceWallFunction() ? 1 : 0;
        }
        d.wall = buildDeviceWallData(m, g, patches, hf.U, wfPatch);

        // The per-FACE question, which is not the per-cell one DeviceWallData answers: a cell can touch a
        // wall and an inlet at once, and correctNut must give those two faces different values.
        const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, patches);
        std::vector<label>  mask;
        std::vector<scalar> yBnd;
        label bndIdx = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const bool isWF = isTurbWallPatch(patches, pi, wfPatch);
            for (label i = 0; i < patches[pi].size; ++i, ++bndIdx)
            {
                mask.push_back(isWF ? 1 : 0);
                yBnd.push_back(isWF ? yW[pi][i] : scalar(0.0));
                // The wall-face ORDER has to be the one buildDeviceWallData built, because
                // deviceGatherWallNu indexes into it -- so it is recorded here, where that order is
                // decided, rather than reconstructed by a caller walking the patches again.
                if (isWF) d.wfFaceOfBnd.push_back(bndIdx);
            }
        }
        mask.resize(static_cast<std::size_t>(d.nBndFaces), 0);
        yBnd.resize(static_cast<std::size_t>(d.nBndFaces), scalar(0.0));
        d.wfBndMask.copyFrom(mask);
        d.wallYBndFace.copyFrom(yBnd);
    }

    return d;
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
