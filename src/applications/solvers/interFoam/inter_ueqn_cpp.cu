// interFoam's momentum predictor -- see inter_ueqn_cpp.cuh for the provenance and for the three things
// that are not in rhoSimpleFoam's UEqn.
#include "inter_ueqn_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "solve_vector.cuh"

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

void refuseUnsupported(const InterMomentumInput& in)
{
    if (in.ddtScheme != DdtScheme::Euler)
    {
        const char* name = (in.ddtScheme == DdtScheme::backward)      ? "backward"
                         : (in.ddtScheme == DdtScheme::CrankNicolson) ? "CrankNicolson"
                         : (in.ddtScheme == DdtScheme::localEuler)    ? "localEuler"
                                                                      : "steadyState";
        throw std::runtime_error(
            std::string("brae interFoam UEqn: ddtSchemes asks for `") + name + "`, which is not ported. "
            "Running it as Euler would make the momentum equation disagree with alphaEqn about the time "
            "scheme (alphaEqn.H:44-50 accepts only Euler and CrankNicolson, and refuses CrankNicolson "
            "under sub-cycling), so the two would advance the same field by different rules.");
    }
    if (in.hasMRF && (!in.mrf || in.mrf->empty()))
        throw std::runtime_error(
            "brae interFoam UEqn: the case declares MRF, and UEqn.H adds MRF.DDt(rho, U). brae has "
            "shipped a solver that read MRFProperties, ignored it, converged and reported nothing "
            "wrong; this refuses instead. Hand the resolved zones over in InterMomentumInput::mrf.");
    if (in.hasFvOptions && (!in.fvOptions || in.fvOptions->empty()))
        throw std::runtime_error(
            "brae interFoam UEqn: the case declares fvOptions `" + in.fvOptionUnsupported
            + "`, which UEqn.H applies as fvOptions(rho, U) and constrains the matrix with. Not ported.");
    if (!in.rhoPhi || !in.rhoPhiBnd)
        throw std::runtime_error("brae interFoam UEqn: rhoPhi is required (fvm::div(rhoPhi, U)).");
    if (!in.rho || !in.rhoOld || !in.UOld)
        throw std::runtime_error(
            "brae interFoam UEqn: fvm::ddt(rho, U) needs rho, rho.oldTime() AND U.oldTime(). The "
            "diagonal takes rho and the source takes rho.oldTime() (EulerDdtScheme.C:455-467); they are "
            "separate arguments because on this solver they differ by the density ratio.");
    if (!in.muEff && (!in.nuEff || !in.nuEffBnd || !in.rhoBnd))
        throw std::runtime_error(
            "brae interFoam UEqn: divDevRhoReff needs either muEff directly or rho and nuEff on both "
            "cells and boundary, from which mu_eff = rho*nuEff is formed.");
}

// grad(U) for whichever of the three consumers asks for one, resolved through the case's gradSchemes.
std::vector<tensor> gradU(const GeometricField<vector>& U,
                          const InterMomentumInput&     in,
                          scalar                        limitK,
                          const PrimitiveMesh&          m,
                          const FvGeometry&             g,
                          const std::vector<FvPatch>&   patches)
{
    std::vector<tensor> gU = in.gradULeastSq ? fvc::leastSquaresGrad(U, m, g, patches)
                                             : fvc::gaussGrad(U, m, g, patches);
    cellLimitGrad(gU, U, limitK, m, g, patches);
    return gU;
}

FvVectorMatrix divWithScheme(const GeometricField<vector>& U,
                             const InterMomentumInput&     in,
                             const PrimitiveMesh&          m,
                             const FvGeometry&             g,
                             const std::vector<FvPatch>&   patches)
{
    const std::vector<scalar>& phi = *in.rhoPhi;
    switch (in.scheme)
    {
        case DivScheme::linear:
            // `Gauss linear` on div(rhoPhi,U) -- central differencing, weights = the mesh's own. Three
            // shipped tutorials ask for it. It is not upwind with a correction; it is different weights.
            return fvm::div<vector>(phi, *in.rhoPhiBnd, U, g.weights(), m, patches);

        case DivScheme::limitedLinear:
            // limitedLinear on a VECTOR is NVDTVD + limitFuncs::magSqr (LimitedScheme.H:188-189): the
            // limiter is built on the scalar field magSqr(U), whose gradient resolves through the case's
            // `grad(magSqr(U))` entry. rhoSimpleFoam carries that lookup; this solver's parser does not
            // yet, and defaulting it to a Gauss gradient put rhoSimpleFoam's momentum matrix 4.63e-02 off
            // OpenFOAM's own on a case whose default was leastSquares. One shipped tutorial asks for it.
            throw std::runtime_error(
                "brae interFoam UEqn: `Gauss limitedLinear` on div(rhoPhi,U) is not ported -- its limiter "
                "needs the case's grad(magSqr(U)) scheme, which this solver does not resolve yet.");

        case DivScheme::limitedLinearV:
        {
            const std::vector<tensor> gU = gradU(U, in, in.gradULimitK, m, g, patches);
            const std::vector<scalar> w =
                limitedSchemes::limitedLinearVWeights(phi, U, gU, in.schemeCoeff, m, g);
            return fvm::div<vector>(phi, *in.rhoPhiBnd, U, w, m, patches);
        }

        case DivScheme::vanLeerV:
        {
            // the limiter's gradient is fvc::grad(U) through the case's grad(U) entry
            // (LimitedScheme::calcLimiter), as limitedLinearV's above
            const std::vector<tensor> gU = gradU(U, in, in.gradULimitK, m, g, patches);
            const std::vector<scalar> w = limitedSchemes::vanLeerVWeights(phi, U, gU, m, g);
            return fvm::div<vector>(phi, *in.rhoPhiBnd, U, w, m, patches);
        }

        case DivScheme::LUST:
            return fvm::div<vector>(phi, *in.rhoPhiBnd, U, limitedSchemes::lustWeights(phi, g), m, patches);

        case DivScheme::upwind:
        case DivScheme::linearUpwind:
        case DivScheme::linearUpwindV:
        default:
            // linearUpwind and linearUpwindV DERIVE from upwind: same weights, correction only. The
            // correction is added by the caller, after this.
            return fvm::div<vector>(phi, *in.rhoPhiBnd, U, m, patches);
    }
}

}   // namespace


FvVectorMatrix assembleUEqn(
    const GeometricField<vector>& U,
    const InterMomentumInput&     in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);

    // fvm::div(rhoPhi, U)
    FvVectorMatrix M = divWithScheme(U, in, m, g, patches);

    // linearUpwind's / linearUpwindV's deferred correction. OpenFOAM applies it inside fvm::div
    // (gaussConvectionScheme.C:112-115) and `fvm += ...` on an fvMatrix is `source -= V*...`
    // (fvMatrix.C:1855-1862), so the caller SUBTRACTS what these return. The correction's gradient is
    // the one the SCHEME names -- `linearUpwind grad(U)` in 24 of the shipped tutorials -- not grad(U)'s
    // own entry, which divDevRhoReff below takes; they coincide here only because interFoam's tutorials
    // name the same one twice.
    const scalar luGradK = (in.gradULULimitK >= scalar(0)) ? in.gradULULimitK : in.gradULimitK;
    if (in.scheme == DivScheme::linearUpwind || in.scheme == DivScheme::linearUpwindV
     || in.scheme == DivScheme::LUST)
    {
        const std::vector<tensor> gU = gradU(U, in, luGradK, m, g, patches);
        const std::vector<vector> corr =
            (in.scheme == DivScheme::linearUpwindV)
                ? limitedSchemes::linearUpwindVCorrection(*in.rhoPhi, U, gU, m, g)
                : fvm::linearUpwindCorrection<vector, tensor>(*in.rhoPhi, gU, m, g);
        // LUST carries a QUARTER of linearUpwind's correction (LUST.H overrides both weights and
        // correction; taking one without the other is a different scheme).
        const scalar fac = (in.scheme == DivScheme::LUST) ? scalar(0.25) : scalar(1);
        for (std::size_t c = 0; c < corr.size(); ++c)
        {
            M.source[c].x -= fac * corr[c].x;
            M.source[c].y -= fac * corr[c].y;
            M.source[c].z -= fac * corr[c].z;
        }
    }

    // fvm::ddt(rho, U). Added to the SAME matrix, before relax, exactly as the constructor's `+` does.
    addEulerDdtRhoU(M, *in.rho, *in.rhoOld, *in.UOld, g.V(), in.deltaT, in.V0);

    // + MRF.DDt(rho, U), UEqn.H:6. MRFZoneList::DDt(rho, U) is rho*DDt(U) (MRFZoneList.C), and DDt(U)
    // the volVectorField Omega x U on the zone's cells, built from the CURRENT U -- explicit, lagged
    // like any deferred term. `fvMatrix + volField` is source -= V*field (fvMatrix.C:1855-1862), so
    // the cell takes  source -= V*rho*(Omega x U).
    if (in.mrf && !in.mrf->empty())
    {
        std::vector<vector> acc(static_cast<std::size_t>(m.nCells()), vector{0, 0, 0});
        // addCoriolis subtracts V*(Omega x U) from what it is handed
        MRF::addCoriolis(*in.mrf, U.internal, g.V(), acc);
        const std::vector<scalar>& rho = *in.rho;
        for (label c = 0; c < m.nCells(); ++c)
        {
            M.source[c].x += rho[c] * acc[c].x;
            M.source[c].y += rho[c] * acc[c].y;
            M.source[c].z += rho[c] * acc[c].z;
        }
    }

    // turbulence->divDevRhoReff(rho, U). incompressibleInterPhaseTransportModel.C:129 forwards to
    // linearViscousStress's rho-weighted overload, so the operator is brae's existing one given
    // mu_eff = rho*nuEff. Note that rho*nuEff is NOT the mixture mu wherever alpha is outside [0,1]:
    // rho takes the raw alpha, nu's denominator the clamped one. OpenFOAM forms rho*nuEff.
    std::vector<scalar>              muEffOwned;
    std::vector<std::vector<scalar>> muEffBndOwned;
    const std::vector<scalar>*              muEff    = in.muEff;
    const std::vector<std::vector<scalar>>* muEffBnd = in.muEffBnd;
    if (!muEff)
    {
        muEffOwned    = dynamicViscosity(*in.rho, *in.nuEff);
        muEffBndOwned = dynamicViscosityBoundary(*in.rhoBnd, *in.nuEffBnd);
        muEff    = &muEffOwned;
        muEffBnd = &muEffBndOwned;
    }
    addDivDevReff(M, U, *muEff, *muEffBnd, m, g, patches, in.correctedLaplacian, in.snGradLimitCoeff,
                  in.gradULimitK, in.gradULeastSq);

    // == fvOptions(rho, U), UEqn.H:9. explicitPorositySource builds a porosityEqn and does
    // `eqn -= porosityEqn`, and `UEqn == options` subtracts that again, so the NET effect is the
    // porosity equation as written: diag += V*tr(Cd)/3, source -= V*((Cd - I*tr(Cd)/3) & U), with
    //     Cd = mu*D + (rho*|U|)*F,   mu = rho*nu                  (DarcyForchheimerTemplates.C:53)
    // fvOptions_cpp carries that arithmetic and both negations; this hands it interFoam's fields.
    if (in.fvOptions && !in.fvOptions->empty())
    {
        if (!in.nuLaminar)
            throw std::runtime_error(
                "brae interFoam UEqn: fvOptions(rho, U) needs the mixture's LAMINAR nu -- "
                "DarcyForchheimer's mu is rho*nu (DarcyForchheimer.C:214-217), not rho*nuEff.");
        const std::vector<scalar>& rho = *in.rho;
        std::vector<scalar> mu(rho.size());
        for (std::size_t c = 0; c < rho.size(); ++c)
        {
            mu[c] = rho[c] * (*in.nuLaminar)[c];
        }
        fvOptions::addSup(*in.fvOptions, M, U, scalar(0), g, /*forceDimensions=*/true, &rho, &mu);
    }

    // UEqn.relax(). damBreak names `".*" 1`, which relaxEquation() finds, and relax(1) still applies the
    // diagonal-dominance clamp -- see InterMomentumInput::relaxEquationU.
    if (in.relaxEquationU && in.relaxU > scalar(0))
        relaxMatrix<vector>(M, U, m, patches, in.relaxU);

    return M;
}


void addMomentumPredictorSource(
    FvVectorMatrix&             UEqn,
    const SurfaceScalarField&   flux,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    // `solve(UEqn == R)` is fvMatrix::operator==, which is source += V*R. The sign is written out rather
    // than reasoned about at each call site: PLUS here, where rhoSimpleFoam's grad(p) term carries the
    // minus inside R itself.
    const std::vector<vector> R = reconstruct(flux, m, g, patches);
    const std::vector<scalar>& V = g.V();
    for (label c = 0; c < m.nCells(); ++c)
    {
        UEqn.source[c].x += R[c].x * V[c];
        UEqn.source[c].y += R[c].y * V[c];
        UEqn.source[c].z += R[c].z * V[c];
    }
}


void momentumPredictor(GeometricField<vector>&     U,
                       const InterMomentumInput&   in,
                       const SurfaceScalarField&   faceForce,
                       const MomentumSolveControls& sc,
                       const PrimitiveMesh&        m,
                       const FvGeometry&           g,
                       const std::vector<FvPatch>& patches,
                       bool                        solveMomentum,
                       FvVectorMatrix&             UEqnOut)
{
    // ORDER IS OpenFOAM's: assemble and RELAX first, then add the face force. `solve(UEqn == R)`
    // builds a new equation from the ALREADY-RELAXED matrix, so relaxing after adding R would relax
    // the buoyancy and surface tension too -- which OpenFOAM does not do.
    UEqnOut = assembleUEqn(U, in, m, g, patches);

    // rAU and H() are taken from the relaxed matrix BEFORE the face force, which is why the matrix is
    // handed back to the caller here rather than after.
    // UEqn.H:17-31 wraps the solve in `if (pimple.momentumPredictor())`. With it off the matrix is
    // still assembled and relaxed -- pEqn needs A() and H() -- and U is untouched.
    if (!solveMomentum) return;
    FvVectorMatrix solved = UEqnOut;
    addMomentumPredictorSource(solved, faceForce, m, g, patches);
    SolverPerformance perf[3];
    solveVector(solved, U, m, patches, sc.tolU, sc.relTolU, sc.maxIterU, 0, sc.solutionD, perf, &sc.which);
    // fvMatrix::solve() ends with psi.correctBoundaryConditions(); solveVector's evaluateBoundary() is
    // only half of that for a flux-conditional patch -- see updateVelocityPatchesFromCells.
    updateVelocityPatchesFromCells(U, patches);
    if (sc.solveLog)
    {
        for (int k = 0; k < 3; ++k)
        {
            if (sc.solutionD && !sc.solutionD->valid(k)) continue;
            sc.solveLog[k].push_back(
                LinearSolveRecord{perf[k].initialResidual, perf[k].finalResidual, perf[k].nIterations});
        }
    }
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
