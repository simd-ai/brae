// _cpp REFERENCE implementation -- see pEqn_cpp.cuh for the OpenFOAM provenance and the refusal contract.
#include "rhoPEqn_cpp.cuh"
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

const scalar kVSmall = 1.0e-300;
const scalar kSmall  = 1.0e-15;

void refuseUnsupported(const PressureInput& in)
{
    if (!in.rho || !in.rhoBnd)
        throw std::runtime_error("rhoSimpleFoam pEqn_cpp: rho was not supplied.");
    if (in.transonic && (!in.psi || !in.psiBnd))
        throw std::runtime_error(
            "rhoSimpleFoam pEqn_cpp: the transonic branch builds phid from psi "
            "((fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA, pEqn.H:14-18) and none was supplied.");
    if (in.hasMRF)
        throw std::runtime_error(
            "rhoSimpleFoam pEqn_cpp: the case declares MRF, which pEqn.H applies as "
            "MRF.makeRelative(fvc::interpolate(rho), phiHbyA) (pEqn.H:10) and again inside "
            "constrainPressure. Not implemented; refusing rather than solving a different equation.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "rhoSimpleFoam pEqn_cpp: the case declares fvOptions, which pEqn.H puts on the right-hand "
            "side as fvOptions(psi, p, rho.name()). Not implemented; refusing.");
    if (in.snGradLimitCoeff != 0.0)
        throw std::runtime_error(
            "rhoSimpleFoam pEqn_cpp: a `limited <k> corrected` laplacian was asked for; brae implements "
            "`corrected` (uncapped) here, which is what all five rhoSimpleFoam tutorials set. Refusing "
            "rather than running the uncapped form under the limited name.");
}

// A volScalarField's face interpolation where the BOUNDARY value is the field's own patch value, which is
// what fvc::interpolate does. Reused from the viscous stress rather than re-derived: two implementations
// of OpenFOAM's face interpolation is two chances to disagree with it.
SurfaceScalarField interp(
    const std::vector<scalar>&              vol,
    const std::vector<std::vector<scalar>>& bnd,
    const PrimitiveMesh&                    m,
    const FvGeometry&                       g,
    const std::vector<FvPatch>&             patches)
{
    return effectiveFaceViscosity(vol, bnd, m, g, patches);
}

} // namespace


bool adjustPhi(
    SurfaceScalarField&           phi,
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const PrimitiveMesh&          m,
    const std::vector<FvPatch>&   patches)
{
    // adjustPhi.C:37 -- the whole function is a no-op unless the pressure needs a reference. A case with
    // a fixed-pressure outlet has its own mass balance and OpenFOAM leaves the flux alone.
    bool needsRef = true;
    for (const auto& b : p.boundary)
        if (b->fixesValue()) { needsRef = false; break; }
    if (!needsRef) return false;

    scalar massIn = 0.0, fixedMassOut = 0.0, adjustableMassOut = 0.0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // adjustPhi branches on `Up.fixesValue() && !isA<inletOutlet>(Up)` -- fixesValue, NOT assignable.
        // The two questions appear within a few lines of each other in pEqn.H and are not the same one.
        //
        // BOTH halves are needed, and the second is not decoration: mixedFvPatchField::fixesValue() is
        // TRUE (mixedFvPatchField.H:197) and inletOutlet inherits it, so without the exclusion an
        // inletOutlet outlet counts as a FIXED outflow and adjustPhi has nothing adjustable left to
        // balance the inflow against -- it would then reach the "continuity error cannot be removed"
        // fatal error on a case OpenFOAM solves.
        const bool fixed = U.boundary[pi]->fixesValue() && !U.boundary[pi]->isInletOutlet();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const scalar v = phi.boundary[pi][i];
            if (v < 0.0)      massIn -= v;
            else if (fixed)   fixedMassOut += v;
            else              adjustableMassOut += v;
        }
    }

    scalar totalFlux = kVSmall;
    for (scalar v : phi.internal) totalFlux += std::fabs(v);
    for (const auto& b : phi.boundary)
        for (scalar v : b) totalFlux += std::fabs(v);

    scalar massCorr = 1.0;
    const scalar magAdj = std::fabs(adjustableMassOut);
    if (magAdj > kVSmall && magAdj / totalFlux > kSmall)
    {
        massCorr = (massIn - fixedMassOut) / adjustableMassOut;
    }
    else if (std::fabs(fixedMassOut - massIn) / totalFlux > 1e-8)
    {
        // OpenFOAM's FatalError, kept fatal. A continuity error that cannot be removed by adjusting the
        // outflow means the velocity boundary conditions do not admit a solution; scaling something
        // anyway would hide that behind a converged-looking run.
        throw std::runtime_error(
            "rhoSimpleFoam pEqn_cpp: continuity error cannot be removed by adjusting the outflow "
            "(adjustPhi.C:96-110). Check the velocity boundary conditions, or run potentialFoam to "
            "initialise the outflow. Specified mass inflow " + std::to_string((double)massIn)
            + ", specified mass outflow " + std::to_string((double)fixedMassOut)
            + ", adjustable mass outflow " + std::to_string((double)adjustableMassOut)
            + ", total flux " + std::to_string((double)totalFlux) + ".");
    }

    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const bool fixed = U.boundary[pi]->fixesValue() && !U.boundary[pi]->isInletOutlet();
        if (fixed) continue;
        for (label i = 0; i < patches[pi].size; ++i)
            if (phi.boundary[pi][i] > 0.0) phi.boundary[pi][i] *= massCorr;
    }

    // closedVolume: every boundary flux negligible against the total. Not "the domain has no inlet" --
    // it is measured from the fluxes, and it is what decides whether p gets the psi-weighted mass
    // correction after the solve.
    return std::fabs(massIn) / totalFlux < kSmall
        && std::fabs(fixedMassOut) / totalFlux < kSmall
        && std::fabs(adjustableMassOut) / totalFlux < kSmall;
}


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
    const label nC = m.nCells();

    PressureStages st;
    st.transonic = in.transonic;

    // rAU = 1/UEqn.A().
    const std::vector<scalar> A = matrixA<vector>(UEqn, m, g, patches);
    st.rAU.resize(nC);
    for (label c = 0; c < nC; ++c) st.rAU[c] = 1.0 / A[c];

    // rhorAUf = fvc::interpolate(rho*rAU). rho*rAU is a volScalarField, so its BOUNDARY is rho's patch
    // value times rAU's -- and rAU is a calculated field whose patch value is the owner cell's.
    std::vector<scalar> rhorAU(nC);
    for (label c = 0; c < nC; ++c) rhorAU[c] = (*in.rho)[c] * st.rAU[c];
    std::vector<std::vector<scalar>> rhorAUb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        rhorAUb[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            rhorAUb[pi][i] = (*in.rhoBnd)[pi][i] * st.rAU[patches[pi].faceCells[i]];
    }
    st.rhorAUf = interp(rhorAU, rhorAUb, m, g, patches);

    // HbyA = constrainHbyA(rAU*UEqn.H(), U, p). constrainHbyA replaces HbyA by U on a patch whose U is
    // NOT assignable -- assignable() is not fixesValue(): slip and inletOutlet are non-assignable without
    // fixing a value, and the distinction is why fv_patch_field carries both.
    const std::vector<vector> H = matrixH(UEqn, U, m, g, patches);
    st.HbyA.resize(nC);
    for (label c = 0; c < nC; ++c)
        st.HbyA[c] = vector{ st.rAU[c]*H[c].x, st.rAU[c]*H[c].y, st.rAU[c]*H[c].z };
    std::vector<std::vector<vector>> HbyAb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const bool takeU = !U.boundary[pi]->assignable();
        const std::vector<vector>& ub = U.boundary[pi]->value();
        HbyAb[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            HbyAb[pi][i] = takeU ? ub[i] : st.HbyA[c];
        }
    }

    // phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA). THE FACTORS, interpolated separately -- unlike
    // compressibleCreatePhi.H one file earlier, which interpolates the product rho*U. Both forms are
    // OpenFOAM's and each belongs to exactly one place.
    const SurfaceScalarField fluxHbyA = fvc::flux(st.HbyA, HbyAb, m, g, patches);
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

    // constrainPressure(p, rho, U, phiHbyA, rhorAUf, MRF) -- pEqn.H:12, on the RAW phiHbyA, BEFORE
    // either branch and before adjustPhi. Each p patch that updates its own snGrad (fixedFluxPressure)
    // is handed
    //     snGrad = (phiHbyA_b - rho_b*(Sf_b & U_b)) / (magSf_b * rhorAUf_b)
    // (constrainPressure.C:60-77; MRF is refused above, so relative() is the identity). At a
    // NON-assignable-U patch constrainHbyA above set HbyA_b = U_b, so the numerator cancels EXACTLY --
    // X - X, not a small residual -- which is why the old zeroGradient substitution survived every
    // fixed-velocity fixture and why the gate for this lives on an assignable-U (inletOutlet) outlet.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!p.boundary[pi]->updateableSnGrad()) continue;
        const std::vector<vector> Ub = U.boundary[pi]->value();
        std::vector<scalar> snGrad(static_cast<std::size_t>(patches[pi].size));
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const scalar sfU = patches[pi].magSf[i] * dot(patches[pi].nf[i], Ub[i]);
            snGrad[i] = (st.phiHbyA.boundary[pi][i] - rhof.boundary[pi][i] * sfU)
                      / (patches[pi].magSf[i] * st.rhorAUf.boundary[pi][i]);
        }
        p.boundary[pi]->updateSnGrad(snGrad);
    }

    if (in.transonic)
    {
        // phid = (fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA, built from phiHbyA BEFORE the
        // subtraction below -- the order in pEqn.H is not incidental.
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

        // phiHbyA -= fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho).
        // interpolate(psi*p) is the interpolation of the PRODUCT, so psi*p is formed per cell first.
        std::vector<scalar> psip(nC);
        for (label c = 0; c < nC; ++c) psip[c] = (*in.psi)[c] * p.internal[c];
        std::vector<std::vector<scalar>> psipb(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& pb = p.boundary[pi]->value();
            psipb[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                psipb[pi][i] = (*in.psiBnd)[pi][i] * pb[i];
        }
        const SurfaceScalarField psipf = interp(psip, psipb, m, g, patches);
        for (std::size_t f = 0; f < st.phiHbyA.internal.size(); ++f)
            st.phiHbyA.internal[f] -=
                psipf.internal[f] * st.phiHbyA.internal[f] / rhof.internal[f];
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
                st.phiHbyA.boundary[pi][i] -=
                    psipf.boundary[pi][i] * st.phiHbyA.boundary[pi][i] / rhof.boundary[pi][i];

        // closedVolume stays false: pEqn.H never runs adjustPhi on this branch.
    }
    else
    {
        st.closedVolume = adjustPhi(st.phiHbyA, U, p, m, patches);
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
    const label nC = m.nCells();

    // - fvm::laplacian(rhorAUf, p), the term both branches share.
    //
    // The SAME two halves as the energy equation's laplacian, and the same half was missing here.
    // correctedLaplacian selects nonOrthDeltaCoeffs for the implicit coefficients; the explicit
    // source -= V*div(gamma*magSf*(corrVecs & interpolate(grad(p)))) is a separate term that
    // gaussLaplacianScheme adds whenever the snGrad scheme is corrected -- independently of
    // nNonOrthogonalCorrectors, which controls how many times the pressure equation is re-solved and
    // not whether the correction exists. Added before the sign flip below so it is negated with the
    // rest of the term.
    FvScalarMatrix M = fvm::laplacian<scalar>(st.rhorAUf, p, m, g, patches, in.correctedLaplacian);
    if (in.correctedLaplacian)
    {
        std::vector<std::vector<scalar>> pb(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) pb[pi] = p.boundary[pi]->value();
        const std::vector<vector> gradP = fvc::gaussGrad(p.internal, pb, m, g, patches);
        const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
            st.rhorAUf, p, gradP, m, g, patches, in.snGradLimitCoeff);
        for (label c = 0; c < nC; ++c) M.source[c] -= corr[c];
        // ...and the SAME correction as a face flux, or `phi = phiHbyA + pEqn.flux()` drops what the
        // source above put in and phi stops being conservative on a non-orthogonal mesh (OF stores it
        // in gaussLaplacianScheme whenever corrected && fluxRequired(p), which createFields.H:43 sets
        // unconditionally; fvMatrix::flux() adds it back at fvMatrix.C:1516-1518). This line existed
        // on the incompressible twin (pEqn_cpp.cu) and NOT here -- the rho driver negated an empty
        // vector for as long as the census took to notice. Identical function + identical arguments
        // as the source term, so the two cannot drift.
        M.faceFluxCorrection = fvm::laplacianCorrFlux<scalar, vector>(
            st.rhorAUf, gradP, m, g, in.snGradLimitCoeff, &p);
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

    // + fvm::div(phid, p) -- TRANSONIC ONLY. This is what makes the pressure equation convective, and it
    // is the entire structural difference between the two branches' matrices.
    if (st.transonic)
    {
        addEqual(M, fvm::div(st.phid.internal, st.phid.boundary, p, m, patches), 1.0);
    }

    // + fvc::div(phiHbyA). An explicit field on the LEFT of the equation, so `source -= V*div` -- and
    // fvc::div returns the per-volume divergence, so the V multiplies back out.
    const std::vector<scalar> divPhiHbyA = fvc::div(st.phiHbyA, m, g, patches);
    for (label c = 0; c < nC; ++c) M.source[c] -= divPhiHbyA[c] * g.V()[c];

    // pEqn.relax() -- TRANSONIC ONLY. pEqn.H relaxes the pressure equation on that branch and not on the
    // subsonic one; relaxing both, or neither, changes the iteration path without changing the converged
    // answer, which is exactly the kind of difference a converged-field comparison cannot see.
    // NOTE the guard is "the case names a factor", not "the factor is below 1" -- see relaxPSpecified.
    if (st.transonic && in.relaxPSpecified && in.relaxP > 0.0)
    {
        relaxMatrix<scalar>(M, p, m, patches, in.relaxP);
    }

    // pEqn.setReference(refCell, refValue) -- fvMatrix.C, verbatim:
    //     source()[celli] += diag()[celli]*value;
    //     diag()[celli]   += diag()[celli];
    // It DOUBLES the diagonal rather than setting it. A "pin the cell to a value" reading of this line
    // gives a different matrix that still solves.
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
