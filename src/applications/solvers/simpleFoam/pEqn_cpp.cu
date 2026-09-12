// _cpp REFERENCE implementation -- see pEqn_cpp.cuh for the OpenFOAM provenance and the refusal contract.
#include "pEqn_cpp.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include <cmath>
#include <stdexcept>

namespace brae {
namespace cpu {

namespace {

constexpr scalar VSMALL_ = 1e-300;
constexpr scalar SMALL_  = 1e-15;

void refuseUnsupported(const PressureInput& in)
{
    if (in.hasMRF && !in.mrf)
        throw std::runtime_error(
            "pEqn_cpp: the case declares MRF, which pEqn.H applies via MRF.makeRelative(phiHbyA) "
            "(pEqn.H:5) and inside constrainPressure (pEqn.H:21). No zones were supplied; refusing "
            "rather than solving a different equation.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "pEqn_cpp: the case declares fvOptions, which pEqn.H applies as fvOptions.correct(U) "
            "(pEqn.H:49). Refusing rather than solving a different equation.");
}

} // namespace


PressureStages pressurePredictor(
    const FvVectorMatrix&         UEqn,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const PressureInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);
    PressureStages st;

    // rAU = 1/UEqn.A()
    const std::vector<scalar> A = matrixA<vector>(UEqn, m, g, patches);
    st.rAU.resize(A.size());
    for (std::size_t c = 0; c < A.size(); ++c) st.rAU[c] = 1.0 / A[c];

    // HbyA = rAU*UEqn.H()
    const std::vector<vector> H = matrixH(UEqn, U, m, g, patches);
    st.HbyA.resize(H.size());
    for (std::size_t c = 0; c < H.size(); ++c)
        st.HbyA[c] = {st.rAU[c] * H[c].x, st.rAU[c] * H[c].y, st.rAU[c] * H[c].z};

    // constrainHbyA(HbyA, U, p) -- constrainHbyA.C:
    //     if (!U.boundaryField()[patchi].assignable() && !isA<fixedFluxExtrapolatedPressure>(p...))
    //         HbyAbf[patchi] = U.boundaryField()[patchi];
    // Everywhere else HbyA keeps the extrapolated boundary value it inherits from fvMatrix::H(), whose
    // boundary type is extrapolatedCalculated -- i.e. the owner cell's value.
    //
    // assignable() is NOT fixesValue(): slip and inletOutlet are non-assignable without fixing a value.
    // See fv_patch_field.cuh, where assignable() was added for exactly this call.
    std::vector<std::vector<vector>> HbyAb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        HbyAb[pi].resize(fp.size);
        const bool takeU = !U.boundary[pi]->assignable();
        const std::vector<vector>& ub = U.boundary[pi]->value();
        for (label i = 0; i < fp.size; ++i)
            HbyAb[pi][i] = takeU ? ub[i] : st.HbyA[fp.faceCells[i]];
    }

    // phiHbyA = fvc::flux(HbyA)
    st.phiHbyA = fvc::flux(st.HbyA, HbyAb, m, g, patches);

    // MRF.makeRelative(phiHbyA) -- pEqn.H:5, BEFORE adjustPhi.
    if (in.mrf) MRF::makeRelative(st.phiHbyA, *in.mrf, g, patches);

    // adjustPhi(phiHbyA, U, p) -- adjustPhi.C. Only when p needs a reference, i.e. no patch fixes p.
    // Scales the ADJUSTABLE outflow so that global continuity closes before the pressure solve; without
    // it a closed-outlet case has an inconsistent right-hand side and the singular pressure operator has
    // no solution at all.
    if (in.pRefCell >= 0)
    {
        scalar massIn = 0.0, fixedMassOut = 0.0, adjustableMassOut = 0.0, totalFlux = VSMALL_;
        // totalFlux = VSMALL + sum(mag(phi)), and Foam::sum() of a GeometricField is
        // gSum(f1.primitiveField()) (GeometricFieldFunctions.C:470-497) -- INTERNAL faces only. The
        // boundary loop that used to sit here inflated the normaliser behind every relative test below.
        for (const auto& v : st.phiHbyA.internal) totalFlux += std::fabs(v);

        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // adjustPhi branches on Up.fixesValue() && !isA<inletOutlet>(Up) -- fixesValue here, NOT
            // assignable; the two questions are different and both appear in this one file.
            // `fixesValue() && !isA<inletOutlet>(Up)`, both halves. mixedFvPatchField::fixesValue() is
            // TRUE and inletOutlet inherits it, so the exclusion is what keeps an inletOutlet outlet
            // ADJUSTABLE -- which is the whole point of adjustPhi having something to scale.
            const bool fixed = U.boundary[pi]->fixesValue() && !U.boundary[pi]->isInletOutlet();
            for (scalar v : st.phiHbyA.boundary[pi])
            {
                if (v < 0.0)        massIn -= v;
                else if (fixed)     fixedMassOut += v;
                else                adjustableMassOut += v;
            }
        }

        scalar massCorr = 1.0;
        const scalar magAdj = std::fabs(adjustableMassOut);
        if (magAdj > VSMALL_ && magAdj / totalFlux > SMALL_)
        {
            massCorr = (massIn - fixedMassOut) / adjustableMassOut;
        }
        else if (std::fabs(fixedMassOut - massIn) / totalFlux > 1e-8)
        {
            // OpenFOAM makes this a FatalError. Reproduce the refusal: the alternative is a pressure
            // equation with no solution, solved anyway.
            throw std::runtime_error(
                "pEqn_cpp: adjustPhi -- continuity error cannot be removed by adjusting the outflow. "
                "Check the velocity boundary conditions (adjustPhi.C).");
        }

        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // THE SAME predicate as the sum loop above -- `fixesValue() && !isInletOutlet()`. This
            // loop tested fixesValue alone, so an inletOutlet outflow was counted ADJUSTABLE in the
            // sums and then skipped by the scaler: massCorr was computed for one face set and applied
            // to another, and global continuity missed by exactly the inletOutlet share.
            if (U.boundary[pi]->fixesValue() && !U.boundary[pi]->isInletOutlet()) continue;
            for (scalar& v : st.phiHbyA.boundary[pi])
                if (v > 0.0) v *= massCorr;
        }
        st.phiAdjusted = (massCorr != 1.0);
    }

    // ---- SIMPLEC (pEqn.H:8-16) -----------------------------------------------------------------
    //
    //     tmp<volScalarField> rAtU(rAU);
    //     if (simple.consistent())
    //     {
    //         rAtU = 1.0/(1.0/rAU - UEqn.H1());
    //         phiHbyA += fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p)*mesh.magSf();
    //         HbyA -= (rAU - rAtU())*fvc::grad(p);
    //     }
    //
    // AFTER adjustPhi, which is where OpenFOAM puts it -- the correction is not part of the flux that
    // global continuity is enforced on. Moving it earlier would rescale it.
    st.rAtU = st.rAU;
    if (in.consistent)
    {
        const std::vector<scalar> H1 = matrixH1<vector>(UEqn, m, g, patches);
        for (std::size_t c = 0; c < st.rAtU.size(); ++c)
            st.rAtU[c] = 1.0 / (1.0 / st.rAU[c] - H1[c]);

        // phiHbyA += interpolate(rAtU - rAU)*snGrad(p)*magSf, on internal AND boundary faces.
        std::vector<scalar> dR(st.rAU.size());
        for (std::size_t c = 0; c < dR.size(); ++c) dR[c] = st.rAtU[c] - st.rAU[c];
        const SurfaceScalarField dRf  = fvc::interpolate(dR, m, g, patches);
        const SurfaceScalarField snGp = fvc::snGrad(p, m, g, patches, in.correctedLaplacian);
        const std::vector<scalar>& magSf = g.magSf();
        for (std::size_t f = 0; f < st.phiHbyA.internal.size(); ++f)
            st.phiHbyA.internal[f] += dRf.internal[f] * snGp.internal[f] * magSf[f];
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
                st.phiHbyA.boundary[pi][i] +=
                    dRf.boundary[pi][i] * snGp.boundary[pi][i] * magSf[patches[pi].start + i];

        // HbyA -= (rAU - rAtU)*fvc::grad(p). Internal cells only: HbyA's boundary is consumed by
        // fvc::flux(HbyA) above, which has already run, and by U's boundary, which correctBoundaryConditions
        // overwrites.
        const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, patches);
        for (std::size_t c = 0; c < st.HbyA.size(); ++c)
        {
            st.HbyA[c].x -= (st.rAU[c] - st.rAtU[c]) * gradP[c].x;
            st.HbyA[c].y -= (st.rAU[c] - st.rAtU[c]) * gradP[c].y;
            st.HbyA[c].z -= (st.rAU[c] - st.rAtU[c]) * gradP[c].z;
        }
    }

    // constrainPressure(p, U, phiHbyA, rAtU(), MRF) -- pEqn.H:21, AFTER adjustPhi (:6) and the SIMPLEC
    // correction (:8-16), so the snGrad is consistent with the flux the pressure equation actually
    // sees. Incompressible: rho is geometricOneField, and rAtU's boundary is the owner cell's
    // (calculated). MRF is refused above, so relative() is the identity. See rhoPEqn_cpp.cu for why
    // the numerator cancels exactly at every non-assignable-U patch.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!p.boundary[pi]->updateableSnGrad()) continue;
        const std::vector<vector> Ub = U.boundary[pi]->value();
        std::vector<scalar> snGrad(static_cast<std::size_t>(patches[pi].size));
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const scalar sfU = patches[pi].magSf[i] * dot(patches[pi].nf[i], Ub[i]);
            snGrad[i] = (st.phiHbyA.boundary[pi][i] - sfU)
                      / (patches[pi].magSf[i] * st.rAtU[patches[pi].faceCells[i]]);
        }
        p.boundary[pi]->updateSnGrad(snGrad);
    }

    return st;
}


FvScalarMatrix assemblePEqn(
    const PressureStages&         st,
    const GeometricField<scalar>& p,
    const PressureInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    // fvm::laplacian(rAtU, p). rAtU == rAU here because `consistent` is refused above.
    //
    // The face diffusivity is fvc::interpolate(rAU), whose boundary value is the owner cell's -- rAU is an
    // extrapolatedCalculated field (fvMatrix::A() sets that type), so this is the correct boundary value
    // rather than a convenience.
    const SurfaceScalarField rAUf = fvc::interpolate(st.rAtU, m, g, patches);
    FvScalarMatrix pEqn = fvm::laplacian<scalar>(rAUf, p, m, g, patches, in.correctedLaplacian);
    if (in.correctedLaplacian)
    {
        const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, patches);
        const std::vector<scalar> corr =
            fvm::laplacianNonOrthSource<scalar, vector>(rAUf, p, gradP, m, g, patches,
                                                        in.snGradLimitCoeff);
        for (std::size_t c = 0; c < corr.size(); ++c) pEqn.source[c] -= corr[c];
        // ...and STORE the face flux, which fvMatrix::flux() adds back (fvMatrix.C:1688). Without it
        // `phi = phiHbyA - pEqn.flux()` drops the correction that the source above put in, and phi stops
        // being conservative on a non-orthogonal mesh.
        pEqn.faceFluxCorrection =
            fvm::laplacianCorrFlux<scalar, vector>(rAUf, gradP, m, g, in.snGradLimitCoeff, &p);
    }

    // == fvc::div(phiHbyA). brae's FvMatrix solves M.psi = source, and fvc::div returns a per-volume
    // quantity, so the right-hand side enters `source` multiplied by V.
    const std::vector<scalar> divPhiHbyA = fvc::div(st.phiHbyA, m, g, patches);
    const std::vector<scalar>& V = g.V();
    for (std::size_t c = 0; c < divPhiHbyA.size(); ++c) pEqn.source[c] += divPhiHbyA[c] * V[c];

    // pEqn.setReference(pRefCell, pRefValue) -- fvMatrix.C:1011-1023, verbatim:
    //     source()[celli] += diag()[celli]*value;
    //     diag()[celli]   += diag()[celli];
    // Note it DOUBLES the diagonal rather than setting it; a "fix the cell to a value" reading of this
    // would produce a different matrix.
    if (in.pRefCell >= 0)
    {
        pEqn.source[in.pRefCell] += pEqn.diag[in.pRefCell] * in.pRefValue;
        pEqn.diag[in.pRefCell]   += pEqn.diag[in.pRefCell];
    }
    return pEqn;
}


SurfaceScalarField correctFlux(
    const PressureStages&         st,
    const FvScalarMatrix&         pEqn,
    const std::vector<scalar>&    pSolved,
    const PrimitiveMesh&          m,
    const std::vector<FvPatch>&   patches)
{
    // phi = phiHbyA - pEqn.flux(). This is the step that makes phi discretely conservative; a past
    // investigation traced a growing divergence to exactly this line, so it is its own stage.
    const SurfaceScalarField f = matrixFlux(pEqn, pSolved, m, patches);
    SurfaceScalarField phi = st.phiHbyA;
    for (std::size_t i = 0; i < phi.internal.size(); ++i) phi.internal[i] -= f.internal[i];
    for (std::size_t pi = 0; pi < phi.boundary.size(); ++pi)
        for (std::size_t i = 0; i < phi.boundary[pi].size(); ++i)
            phi.boundary[pi][i] -= f.boundary[pi][i];
    return phi;
}


std::vector<vector> correctVelocity(
    const PressureStages&         st,
    const GeometricField<scalar>& pRelaxed,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    // U = HbyA - rAtU*fvc::grad(p), with the RELAXED p (pEqn.H relaxes before this line).
    const std::vector<vector> gradP = fvc::gaussGrad(pRelaxed, m, g, patches);
    std::vector<vector> U(st.HbyA.size());
    for (std::size_t c = 0; c < U.size(); ++c)
        U[c] = {st.HbyA[c].x - st.rAtU[c] * gradP[c].x,
                st.HbyA[c].y - st.rAtU[c] * gradP[c].y,
                st.HbyA[c].z - st.rAtU[c] * gradP[c].z};
    return U;
}


void relaxField(std::vector<scalar>& p, const std::vector<scalar>& pPrev, scalar alpha)
{
    // GeometricField::relax -- operator==(prevIter() + alpha*(*this - prevIter()))
    if (alpha >= 1.0 || alpha <= 0.0) return;
    for (std::size_t c = 0; c < p.size(); ++c) p[c] = pPrev[c] + alpha * (p[c] - pPrev[c]);
}

} // namespace cpu
} // namespace brae
