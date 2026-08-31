// _cpp REFERENCE implementation -- see pcEqn_cpp.cuh for the OpenFOAM provenance and the refusal contract.
#include "rhoPcEqn_cpp.cuh"
#include "fvm.cuh"
#include "fvc.cuh"   // gaussGrad, for the laplacian non-orthogonal correction
#include "fv_matrix_ops.cuh"
#include "linearViscousStress_cpp.cuh"   // effectiveFaceViscosity: linear inside, BOUNDARY field on faces
#include <cmath>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace rhoSimple {

namespace {

// fvc::interpolate of a volScalarField: linear inside, the field's own patch value on boundary faces.
SurfaceScalarField interp(
    const std::vector<scalar>&              vol,
    const std::vector<std::vector<scalar>>& bnd,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches)
{
    return effectiveFaceViscosity(vol, bnd, m, g, patches);
}

void refuseUnsupported(const PressureInput& in)
{
    if (!in.rho || !in.rhoBnd)
        throw std::runtime_error("rhoSimpleFoam pcEqn_cpp: rho was not supplied.");
    if (in.transonic && (!in.psi || !in.psiBnd))
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn_cpp: the transonic branch builds phid from psi (pcEqn.H:16-20) and "
            "subtracts fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho); no psi was supplied.");
    if (in.hasMRF)
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn_cpp: the case declares MRF, which pcEqn.H applies as "
            "MRF.makeRelative(fvc::interpolate(rho), phiHbyA) (pcEqn.H:9) and again inside "
            "constrainPressure. Not implemented; refusing.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn_cpp: the case declares fvOptions, which pcEqn.H puts on the right-hand "
            "side as fvOptions(psi, p, rho.name()). Not implemented; refusing.");
    if (in.snGradLimitCoeff != 0.0)
        throw std::runtime_error(
            "rhoSimpleFoam pcEqn_cpp: a `limited <k> corrected` laplacian was asked for; brae implements "
            "`corrected` (uncapped) here, which is what all five rhoSimpleFoam tutorials set. Refusing.");
}

} // namespace


std::vector<scalar> momentumH1(
    const FvVectorMatrix&       UEqn,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    return matrixH1<vector>(UEqn, m, g, patches);
}


ConsistentPressureStages consistentPressurePredictor(
    const FvVectorMatrix&         UEqn,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const PressureInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);
    const label nC = m.nCells();

    ConsistentPressureStages st;
    st.transonic = in.transonic;

    // rAU = 1/UEqn.A(), and rAtU = 1/(1/rAU - UEqn.H1()). Written in OpenFOAM's own form rather than
    // simplified to A/(A - H1): the two agree in exact arithmetic, and keeping the written form means a
    // reader comparing this against pcEqn.H is comparing the same expression.
    const std::vector<scalar> A  = matrixA<vector>(UEqn, m, g, patches);
    const std::vector<scalar> H1 = matrixH1<vector>(UEqn, m, g, patches);
    st.rAU.resize(nC);
    st.rAtU.resize(nC);
    st.rhorAtU.resize(nC);
    for (label c = 0; c < nC; ++c)
    {
        st.rAU[c]  = 1.0 / A[c];
        st.rAtU[c] = 1.0 / (1.0 / st.rAU[c] - H1[c]);
        st.rhorAtU[c] = (*in.rho)[c] * st.rAtU[c];
    }

    // HbyA = constrainHbyA(rAU*UEqn.H(), U, p). Note rAU, not rAtU -- the consistent factor enters
    // through the CORRECTION below, not through H.
    const std::vector<vector> H = matrixH(UEqn, U, m, g, patches);
    st.HbyA0.resize(nC);
    for (label c = 0; c < nC; ++c)
        st.HbyA0[c] = vector{ st.rAU[c]*H[c].x, st.rAU[c]*H[c].y, st.rAU[c]*H[c].z };
    std::vector<std::vector<vector>> HbyAb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const bool takeU = !U.boundary[pi]->assignable();
        const std::vector<vector>& ub = U.boundary[pi]->value();
        HbyAb[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            HbyAb[pi][i] = takeU ? ub[i] : st.HbyA0[patches[pi].faceCells[i]];
    }

    // phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA) -- the FACTORS interpolated, as in pEqn.H.
    const SurfaceScalarField fluxHbyA = fvc::flux(st.HbyA0, HbyAb, m, g, patches);
    const SurfaceScalarField rhof     = interp(*in.rho, *in.rhoBnd, m, g, patches);
    st.phiHbyA0.internal.resize(fluxHbyA.internal.size());
    for (std::size_t f = 0; f < fluxHbyA.internal.size(); ++f)
        st.phiHbyA0.internal[f] = rhof.internal[f] * fluxHbyA.internal[f];
    st.phiHbyA0.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        st.phiHbyA0.boundary[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            st.phiHbyA0.boundary[pi][i] = rhof.boundary[pi][i] * fluxHbyA.boundary[pi][i];
    }
    st.phiHbyA = st.phiHbyA0;

    // constrainPressure(p, rho, U, phiHbyA, rhorAtU, MRF) -- pcEqn.H:16 ("Update the pressure BCs to
    // ensure flux consistency"), on the RAW phiHbyA before the SIMPLEC correction and both branches.
    // rhorAtU is the VOL field rho*rAtU, so its boundary is rho's patch value times the owner cell's
    // rAtU (calculated). See rhoPEqn_cpp.cu for the cancellation note.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!p.boundary[pi]->updateableSnGrad()) continue;
        const std::vector<vector> Ub = U.boundary[pi]->value();
        std::vector<scalar> snGrad(static_cast<std::size_t>(patches[pi].size));
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const scalar sfU = patches[pi].magSf[i] * dot(patches[pi].nf[i], Ub[i]);
            const scalar rhorAtUb = rhof.boundary[pi][i] * st.rAtU[patches[pi].faceCells[i]];
            snGrad[i] = (st.phiHbyA.boundary[pi][i] - rhof.boundary[pi][i] * sfU)
                      / (patches[pi].magSf[i] * rhorAtUb);
        }
        p.boundary[pi]->updateSnGrad(snGrad);
    }

    // The SIMPLEC flux correction, shared by both branches:
    //     fvc::interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)*mesh.magSf()
    // rho*(rAtU - rAU) is formed per cell first, then interpolated -- the product, not the factors.
    std::vector<scalar> drho(nC);
    for (label c = 0; c < nC; ++c) drho[c] = (*in.rho)[c] * (st.rAtU[c] - st.rAU[c]);
    std::vector<std::vector<scalar>> drhoB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        drhoB[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            drhoB[pi][i] = (*in.rhoBnd)[pi][i] * (st.rAtU[c] - st.rAU[c]);
        }
    }
    const SurfaceScalarField drhof   = interp(drho, drhoB, m, g, patches);
    const SurfaceScalarField snGradP = fvc::snGrad(p, m, g, patches, in.correctedLaplacian);

    SurfaceScalarField simplecCorr;
    simplecCorr.internal.resize(st.phiHbyA.internal.size());
    for (std::size_t f = 0; f < st.phiHbyA.internal.size(); ++f)
        simplecCorr.internal[f] = drhof.internal[f] * snGradP.internal[f] * g.magSf()[f];
    simplecCorr.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        simplecCorr.boundary[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            simplecCorr.boundary[pi][i] =
                drhof.boundary[pi][i] * snGradP.boundary[pi][i] * patches[pi].magSf[i];
    }

    if (in.transonic)
    {
        // phid is built from the UNCORRECTED phiHbyA -- pcEqn.H forms it before the += statement below.
        const SurfaceScalarField psif = interp(*in.psi, *in.psiBnd, m, g, patches);
        st.phid.internal.resize(st.phiHbyA.internal.size());
        for (std::size_t f = 0; f < st.phiHbyA.internal.size(); ++f)
            st.phid.internal[f] = (psif.internal[f] / rhof.internal[f]) * st.phiHbyA.internal[f];
        st.phid.boundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            st.phid.boundary[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                st.phid.boundary[pi][i] =
                    (psif.boundary[pi][i] / rhof.boundary[pi][i]) * st.phiHbyA.boundary[pi][i];
        }

        // ONE statement in OpenFOAM, so both terms use the SAME pre-correction phiHbyA on the right-hand
        // side. Applying them in sequence would feed the psi*p term a phiHbyA that already carried the
        // SIMPLEC correction, which is a different number.
        std::vector<scalar> psip(nC);
        for (label c = 0; c < nC; ++c) psip[c] = (*in.psi)[c] * p.internal[c];
        std::vector<std::vector<scalar>> psipB(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& pb = p.boundary[pi]->value();
            psipB[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                psipB[pi][i] = (*in.psiBnd)[pi][i] * pb[i];
        }
        const SurfaceScalarField psipf = interp(psip, psipB, m, g, patches);
        for (std::size_t f = 0; f < st.phiHbyA.internal.size(); ++f)
            st.phiHbyA.internal[f] = st.phiHbyA0.internal[f] + simplecCorr.internal[f]
                - psipf.internal[f] * st.phiHbyA0.internal[f] / rhof.internal[f];
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
                st.phiHbyA.boundary[pi][i] = st.phiHbyA0.boundary[pi][i] + simplecCorr.boundary[pi][i]
                    - psipf.boundary[pi][i] * st.phiHbyA0.boundary[pi][i] / rhof.boundary[pi][i];
    }
    else
    {
        // adjustPhi FIRST, then the correction. adjustPhi balances the flux it is handed, so adding the
        // SIMPLEC term first would have it scale the outflow by a different factor.
        st.closedVolume = adjustPhi(st.phiHbyA, U, p, m, patches);
        for (std::size_t f = 0; f < st.phiHbyA.internal.size(); ++f)
            st.phiHbyA.internal[f] += simplecCorr.internal[f];
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
                st.phiHbyA.boundary[pi][i] += simplecCorr.boundary[pi][i];
    }

    // HbyA -= (rAU - rAtU)*fvc::grad(p), both branches.
    const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, patches);
    st.HbyA = st.HbyA0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar d = st.rAU[c] - st.rAtU[c];
        st.HbyA[c].x -= d * gradP[c].x;
        st.HbyA[c].y -= d * gradP[c].y;
        st.HbyA[c].z -= d * gradP[c].z;
    }
    return st;
}


FvScalarMatrix assemblePcEqn(
    const ConsistentPressureStages& st,
    const GeometricField<scalar>&   p,
    const PressureInput&            in,
    const PrimitiveMesh&            m,
    const FvGeometry&               g,
    const std::vector<FvPatch>&     patches)
{
    const label nC = m.nCells();

    // - fvm::laplacian(rhorAtU, p). rhorAtU is a VOLUME field in pcEqn.H, so its face value is the linear
    // interpolation with the field's own patch values on boundary faces -- which is what fvm::laplacian
    // does to a volScalarField gamma, and what pEqn.H achieves by passing an already-interpolated surface
    // field. The two routes agree; the difference from pEqn.H is rAtU in place of rAU.
    std::vector<std::vector<scalar>> rhorAtUb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        rhorAtUb[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            rhorAtUb[pi][i] = (*in.rhoBnd)[pi][i] * st.rAtU[c];
        }
    }
    const SurfaceScalarField gammaf =
        effectiveFaceViscosity(st.rhorAtU, rhorAtUb, m, g, patches);

    FvScalarMatrix M = fvm::laplacian<scalar>(gammaf, p, m, g, patches, in.correctedLaplacian);
    // `corrected` HAS TWO HALVES AND THIS FILE ONLY HAD ONE. correctedLaplacian selects
    // nonOrthDeltaCoeffs for the implicit coefficients; the explicit
    // source -= V*div(gamma*magSf*(corrVec & interpolate(grad p))) is a separate term that
    // gaussLaplacianScheme adds whenever the snGrad scheme is corrected. pEqn.H's transcription carries
    // it and EEqn.H's does; this one did not, so on a mesh with real non-orthogonality its source was
    // short by the whole correction while its DIAGONAL stayed exact -- which is why
    // rho_pceqn_vs_openfoam.sh passed: sbMatched is near-orthogonal enough that the source barely
    // notices, and every comparison of D() agreed regardless.
    //
    // Found by the CUDA port: test_rho_pceqn_cuda read `pcEqn source` 2.450e-01 against this reference
    // with rAU, rAtU, rhorAtU, HbyA, phiHbyA and every matrix coefficient at 1e-16. The device had the
    // term and the reference did not.
    if (in.correctedLaplacian)
    {
        std::vector<std::vector<scalar>> pb(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) pb[pi] = p.boundary[pi]->value();
        const std::vector<vector> gradP = fvc::gaussGrad(p.internal, pb, m, g, patches);
        const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
            gammaf, p, gradP, m, g, patches, in.snGradLimitCoeff);
        for (label c = 0; c < nC; ++c) M.source[c] -= corr[c];
    }
    for (label c = 0; c < nC; ++c)
    {
        M.diag[c]   = -M.diag[c];
        M.source[c] = -M.source[c];
    }
    for (std::size_t f = 0; f < M.upper.size(); ++f)
    {
        M.upper[f] = -M.upper[f];
        M.lower[f] = -M.lower[f];
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
        {
            M.internalCoeffs[pi][i] = -M.internalCoeffs[pi][i];
            M.boundaryCoeffs[pi][i] = -M.boundaryCoeffs[pi][i];
        }
    for (std::size_t f = 0; f < M.faceFluxCorrection.size(); ++f)
        M.faceFluxCorrection[f] = -M.faceFluxCorrection[f];

    // + fvm::div(phid, p) -- transonic only, exactly as in pEqn.H.
    if (st.transonic)
    {
        addEqual(M, fvm::div(st.phid.internal, st.phid.boundary, p, m, patches), 1.0);
    }

    // + fvc::div(phiHbyA).
    const std::vector<scalar> divPhiHbyA = fvc::div(st.phiHbyA, m, g, patches);
    for (label c = 0; c < nC; ++c) M.source[c] -= divPhiHbyA[c] * g.V()[c];

    // pEqn.relax() -- transonic only here too. The guard is "the case NAMES a factor", not "the factor is
    // below 1": relax(1.0) still runs the diagonal-dominance step. See PressureInput::relaxPSpecified.
    if (st.transonic && in.relaxPSpecified && in.relaxP > 0.0)
    {
        relaxMatrix<scalar>(M, p, m, patches, in.relaxP);
    }

    // pEqn.setReference(refCell, refValue) -- doubles the diagonal, it does not set it.
    if (in.pRefCell >= 0)
    {
        M.source[in.pRefCell] += M.diag[in.pRefCell] * in.pRefValue;
        M.diag[in.pRefCell]   += M.diag[in.pRefCell];
    }
    return M;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
