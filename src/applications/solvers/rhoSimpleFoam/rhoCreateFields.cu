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

    // ---- updateCoeffs() metadata, for the boundary conditions whose coefficients move with the
    //      solution. See the block comment on RhoDeviceFields: the device boundary objects are a
    //      snapshot, so the driver has to recompute per iteration what OpenFOAM recomputes inside
    //      updateCoeffs(). This gathers what those updates need and cannot derive themselves.
    {
        // The freestream family. A wedge also reports category 5 and must NOT count -- its valueFraction
        // is geometry, fixed for the run, not a flow angle -- so an axisymmetric case has nothing to
        // update and should not pay for a kernel that would rewrite a geometric blend every iteration.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if ((hf.U.boundary[pi]->bcCategory() == 5 && !hf.U.boundary[pi]->wedgeFaceT())
             || hf.p.boundary[pi]->bcCategory() == 5)
            {
                d.hasMixed = true;
                break;
            }
        }

        // flowRateInletVelocity, mass form (category 9). The mask is magSf on this patch and 0 on every
        // other face, so dot(rhoBnd, mask) is exactly OpenFOAM's gSum(rho*magSf) over the patch. The walk
        // is the SAME one flattenBoundary uses -- every patch in order, no skipping -- which is
        // unambiguous here only because this function has already refused any coupled patch above.
        std::vector<scalar> nx, ny, nz;
        nx.reserve(static_cast<std::size_t>(d.nBndFaces));
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (label i = 0; i < patches[pi].size; ++i)
            {
                nx.push_back(patches[pi].nf[i].x);
                ny.push_back(patches[pi].nf[i].y);
                nz.push_back(patches[pi].nf[i].z);
            }
        }
        nx.resize(static_cast<std::size_t>(d.nBndFaces), 0.0);
        ny.resize(static_cast<std::size_t>(d.nBndFaces), 0.0);
        nz.resize(static_cast<std::size_t>(d.nBndFaces), 0.0);

        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (hf.U.boundary[pi]->bcCategory() != 9) continue;
            d.hasFlowRate = true;
            std::vector<scalar> mask(static_cast<std::size_t>(d.nBndFaces), 0.0);
            label bi = 0;
            for (std::size_t pj = 0; pj < patches.size(); ++pj)
            {
                for (label i = 0; i < patches[pj].size; ++i, ++bi)
                {
                    if (pj == pi && bi < d.nBndFaces) mask[static_cast<std::size_t>(bi)] = patches[pj].magSf[i];
                }
            }
            d.frMagSf.emplace_back();
            d.frMagSf.back().copyFrom(mask);
            // OpenFOAM re-reads flowRate_->value(t) at every updateCoeffs; steady with a constant entry
            // makes that the seeded value, which the patch object already holds.
            d.frMdot.push_back(hf.U.boundary[pi]->flowRateValue());
        }
        if (d.hasFlowRate)
        {
            d.frNx.copyFrom(nx);
            d.frNy.copyFrom(ny);
            d.frNz.copyFrom(nz);
        }
    }

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
    // The thermo's OWN density, which is not the solver's rho -- see RhoSolverFields. basicThermo's
    // constructor runs calculate() before any solving, so it starts equal to rho, which is exactly what
    // the reference's createFields leaves in hf.rhoThermo.
    d.f.rhoThermo.copyFrom(hf.rhoThermo);
    d.f.rhoThermoBnd.copyFrom(flattenBoundary(hf.rhoThermoBnd, patches, d.nBndFaces, 1.0));
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
        if (!hf.alphat.internal.empty())
        {
            d.f.alphat.copyFrom(hf.alphat.internal);
            // The BOUNDARY alphat too. A device-resident alphaEff reads it on every patch face, and a
            // wall carrying compressible::alphatWallFunction has a patch value the adjacent cell does
            // not: leaving this unseeded would drop the turbulent half of the energy diffusivity at
            // exactly the wall where it is largest.
            d.f.alphatBnd.copyFrom(flattenFieldBoundary(hf.alphat, patches, d.nBndFaces, 0.0));

            // ...and WHICH faces the closure must recompute, with that patch's own Prt. Same walk.
            std::vector<label>  am(static_cast<std::size_t>(d.nBndFaces), 0);
            std::vector<scalar> ap(static_cast<std::size_t>(d.nBndFaces), scalar(0.85));
            label abi = 0;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const bool wf = pi < hf.alphatWallFn.size() && hf.alphatWallFn[pi] != 0;
                const scalar pr = pi < hf.alphatPrt.size() ? hf.alphatPrt[pi] : scalar(0.85);
                for (label i = 0; i < patches[pi].size; ++i, ++abi)
                {
                    if (abi >= d.nBndFaces) break;
                    if (wf) { am[abi] = 1; ap[abi] = pr; }
                }
            }
            d.alphatWallMask.copyFrom(am);
            d.alphatPrtFace.copyFrom(ap);
        }

        // The TURBULENT INLETS, per boundary face. OF's turbulentIntensityKineticEnergyInlet and
        // turbulentMixingLengthDissipationRateInlet recompute their value every updateCoeffs, so the
        // closure needs the per-face intensity / mixing length rather than the seeded k and epsilon.
        // The walk is flattenBoundary's -- every patch in order -- which is unambiguous here because
        // this function has already refused any coupled patch.
        {
            std::vector<label>  km(static_cast<std::size_t>(d.nBndFaces), 0);
            std::vector<label>  em(static_cast<std::size_t>(d.nBndFaces), 0);
            std::vector<scalar> ki(static_cast<std::size_t>(d.nBndFaces), scalar(0));
            std::vector<scalar> el(static_cast<std::size_t>(d.nBndFaces), scalar(0));
            label bi = 0;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                // Kind 0 is intensityK, 1 is mixingLengthEpsilon (TurbulentInletPatchField::Kind); -1 on
                // every other patch type, which is what lets this tell "no such patch" from "a patch
                // whose coefficient happens to be zero".
                const int  kk = hf.k.boundary[pi]->turbulentInletKind();
                const int  ek = hf.epsilon.boundary[pi]->turbulentInletKind();
                const scalar kc = hf.k.boundary[pi]->turbulentInletCoefficient();
                const scalar ec = hf.epsilon.boundary[pi]->turbulentInletCoefficient();
                for (label i = 0; i < patches[pi].size; ++i, ++bi)
                {
                    if (bi >= d.nBndFaces) break;
                    if (kk == 0) { km[bi] = 1; ki[bi] = kc; d.hasTurbulentInlet = true; }
                    if (ek == 1) { em[bi] = 1; el[bi] = ec; d.hasTurbulentInlet = true; }
                }
            }
            if (d.hasTurbulentInlet)
            {
                d.turbInletKMask.copyFrom(km);
                d.turbInletEpsMask.copyFrom(em);
                d.turbInletKInt.copyFrom(ki);
                d.turbInletEpsLen.copyFrom(el);
            }
        }

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
