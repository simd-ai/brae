// interFoam's momentum predictor -- see inter_ueqn_cpp.cuh for the provenance and for the three things
// that are not in rhoSimpleFoam's UEqn.
#include "inter_ueqn_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "solve_vector.cuh"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

void refuseUnsupported(const InterMomentumInput& in)
{
    if ((in.gradUCached == nullptr) != (in.gradUBndCached == nullptr))
        throw std::runtime_error(
            "brae interFoam UEqn: a cached grad(U) needs its cells AND the boundary gaussGrad corrected when "
            "it was formed; the caller gave one without the other.");
    if (in.ddtScheme == DdtScheme::localEuler && (!in.rDeltaT || in.V0))
        throw std::runtime_error(
            "brae interFoam UEqn: ddtSchemes asks for localEuler, whose fvm::ddt(rho, U) reads the local "
            "rDeltaT setRDeltaT.H forms (localEulerDdtScheme.C:282-308); the caller supplied none, or the "
            "mesh moves, which the localEuler port does not carry.");
    if (in.ddtScheme != DdtScheme::Euler && in.ddtScheme != DdtScheme::CrankNicolson
     && in.ddtScheme != DdtScheme::localEuler)
    {
        const char* name = (in.ddtScheme == DdtScheme::backward)      ? "backward"
                         : (in.ddtScheme == DdtScheme::localEuler)    ? "localEuler"
                                                                      : "steadyState";
        throw std::runtime_error(
            std::string("brae interFoam UEqn: ddtSchemes asks for `") + name + "`, which is not ported. "
            "Running it as Euler would make the momentum equation disagree with alphaEqn about the time "
            "scheme (alphaEqn.H:44-50 accepts only Euler and CrankNicolson, and refuses CrankNicolson "
            "under sub-cycling), so the two would advance the same field by different rules.");
    }
    if (in.ddtScheme == DdtScheme::CrankNicolson && (!in.cn || !in.cnDdt0 || !in.rhoOO || !in.UOO))
        throw std::runtime_error(
            "brae interFoam UEqn: ddtSchemes asks for CrankNicolson, whose fvm::ddt(rho, U) reads the "
            "scheme's clock, its own ddt0 field, rho.oldTime().oldTime() and U.oldTime().oldTime(); the "
            "caller supplied fewer. Refused rather than run Euler under the scheme's name.");
    // CrankNicolson ON A MOVING MESH takes the scheme's moving branch, which fv::fvmDdt carries now --
    // ddt0 weighted by V0 and V00 and the source by V0 (CrankNicolsonDdtScheme.C:1029-1065). What is
    // still refused is HALF a moving mesh: V0 without V00 means the caller did not ask the mesh for
    // its second old level, and the static form would then be run under the scheme's name.
    if (in.ddtScheme == DdtScheme::CrankNicolson && in.V0 && !in.V00)
        throw std::runtime_error(
            "brae interFoam UEqn: CrankNicolson's fvm::ddt on a moving mesh needs mesh().V00() as well "
            "as V0 -- the scheme's moving branch weights the two old levels by their own volumes. The "
            "caller gave V0 alone.");
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
    if (in.gradUCached)
    {
        // the registry's field is grad(U)'s, formed through grad(U)'s own entry; a site whose gradient
        // resolved to a different limiter would be asking for another name, which the reader refuses
        if (limitK != in.gradULimitK)
            throw std::runtime_error(
                "brae interFoam UEqn: a cached grad(U) was offered to a site whose gradient resolves to "
                "another entry. Only sites asking for the name grad(U) take the cache.");
        return *in.gradUCached;
    }
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
        {
            // limitedLinear on a VECTOR is NVDTVD + limitFuncs::magSqr (LimitedScheme.H:188-189,
            // LimitFuncs.C:34-39): NOT per component and NOT the V form. The limiter is the scalar one,
            // built on lPhi = magSqr(U) and fvc::grad(lPhi) -- `grad(magSqr(U))`, which this solver's
            // gradSchemes reader holds to Gauss linear with every other entry -- and the ONE weight it
            // gives each face carries all three components. lPhi's patch values are magSqr of U's
            // stored ones: the limiter is built inside gaussConvectionScheme::fvmDiv (.C:84), before the
            // fvMatrix constructor runs updateCoeffs; the driver has refreshed U's patches by then, and
            // on a pressureInletOutletVelocity -- eulerianInjection's sides -- the refresh reproduces
            // the value the last U.correctBoundaryConditions() left, because phi has not moved since.
            const label nC = m.nCells();
            std::vector<scalar> mag2(static_cast<std::size_t>(nC));
            for (label c = 0; c < nC; ++c)
            {
                const vector& u = U.internal[c];
                mag2[c] = u.x*u.x + u.y*u.y + u.z*u.z;
            }
            std::vector<std::vector<scalar>> mag2b(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const std::vector<vector>& ub = U.boundary[pi]->value();
                mag2b[pi].resize(ub.size());
                for (std::size_t i = 0; i < ub.size(); ++i)
                {
                    mag2b[pi][i] = ub[i].x*ub[i].x + ub[i].y*ub[i].y + ub[i].z*ub[i].z;
                }
            }
            const std::vector<vector> gradM = fvc::gaussGrad(mag2, mag2b, m, g, patches);
            GeometricField<scalar> lPhi;          // limitedLinearWeights reads only .internal
            lPhi.internal = mag2;
            const std::vector<scalar> w =
                limitedSchemes::limitedLinearWeights(phi, lPhi, gradM, in.schemeCoeff, m, g);
            for (const FvPatch& q : patches)
            {
                if (q.coupled)
                {
                    throw std::runtime_error(
                        "brae interFoam UEqn: `Gauss limitedLinear` on div(rhoPhi,U) across the coupled patch `"
                        + q.name + "` is not ported.");
                }
            }
            return fvm::div<vector>(phi, *in.rhoPhiBnd, U, w, m, patches);
        }

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

// Instrument: BRAE_STAGE_DUMP_DIR=<dir> (+ BRAE_STAGE_DUMP_ITER=n, default 1) writes UEqn's SOURCE
// after each term is added, at ONE call, in the same shape tools/dumpInterFoam writes OpenFOAM's
// (UEqnSourceDiv/Ddt/Dev/PreRelax.dump). The assembled source is the only term of UEqn.H() still out
// on permeable-moving -- 3.82e-04 relative -- and the terms behind it are three orders apart in size
// (OpenFOAM at step two: div EXACTLY 0, divDevRhoReff 2.227e-09, ddt 1.259e-06, and relax adds
// another 4.5e-07), so the assembled number cannot say which one carries it.
//
// THE DIAGONAL EITHER SIDE OF relax(), because the relaxation contribution is (D - D0)*psi and the
// case reaches it even at `"U.*" 1`: relax runs max(D, D, sumOff) before dividing by alpha.
struct UEqnStageDump
{
    std::string dir;
    bool on = false;

    void vectors(const char* name, const std::vector<vector>& v) const
    {
        if (!on) return;
        std::ofstream o(dir + "/" + name);
        o.precision(17);
        for (const vector& x : v) o << x.x << " " << x.y << " " << x.z << "\n";
    }

    void scalars(const char* name, const std::vector<scalar>& v) const
    {
        if (!on) return;
        std::ofstream o(dir + "/" + name);
        o.precision(17);
        for (const scalar x : v) o << x << "\n";
    }

    void tensors(const char* name, const std::vector<tensor>& v) const
    {
        if (!on) return;
        std::ofstream o(dir + "/" + name);
        o.precision(17);
        for (const tensor& t : v)
        {
            o << t.xx << " " << t.xy << " " << t.xz << " "
              << t.yx << " " << t.yy << " " << t.yz << " "
              << t.zx << " " << t.zy << " " << t.zz << "\n";
        }
    }
};


UEqnStageDump openStageDump()
{
    UEqnStageDump d;
    const char* dd = std::getenv("BRAE_STAGE_DUMP_DIR");
    if (!dd) return d;
    static int calls = 0;
    const char* it = std::getenv("BRAE_STAGE_DUMP_ITER");
    if (++calls != (it && *it ? std::atoi(it) : 1)) return d;
    std::error_code ec;
    std::filesystem::create_directories(dd, ec);
    d.dir = dd;
    d.on = !ec;
    return d;
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
        std::vector<vector> corr =
            (in.scheme == DivScheme::linearUpwindV)
                ? limitedSchemes::linearUpwindVCorrection(*in.rhoPhi, U, gU, m, g)
                : fvm::linearUpwindCorrection<vector, tensor>(*in.rhoPhi, gU, m, g);
        for (const FvPatch& q : patches)
        {
            if (q.coupled && in.scheme != DivScheme::linearUpwind)
                throw std::runtime_error(
                    "brae interFoam UEqn: linearUpwindV and LUST carry no explicit correction across "
                    "the coupled patch `" + q.name + "`; only linearUpwind's is ported.");
        }
        fvm::addLinearUpwindCorrectionCoupled<vector, tensor>(corr, *in.rhoPhiBnd, gU, g, patches);
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

    const UEqnStageDump stage = openStageDump();
    stage.vectors("ueqnSrcDiv", M.source);

    // fvm::ddt(rho, U). Added to the SAME matrix, before relax, exactly as the constructor's `+` does.
    if (in.ddtScheme == DdtScheme::CrankNicolson)
    {
        // ...with the mesh's OLD volumes when it moves: the scheme's moving branch weights ddt0 by V0
        // and V00 and its source by V0, where the static one uses V throughout.
        fv::fvmDdt(*in.cn, *in.cnDdt0, in.rho, in.rhoOld, in.rhoOO, *in.UOld, *in.UOO, g.V(), M,
                   in.V0, in.V00);
    }
    else if (in.ddtScheme == DdtScheme::localEuler)
    {
        addLocalEulerDdtRhoU(M, *in.rho, *in.rhoOld, *in.UOld, g.V(), *in.rDeltaT);
    }
    else
    {
        addEulerDdtRhoU(M, *in.rho, *in.rhoOld, *in.UOld, g.V(), in.deltaT, in.V0);
    }

    stage.vectors("ueqnSrcDdt", M.source);

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
                  in.gradULimitK, in.gradULeastSq, in.nonOrthCoeffs, in.gradUCached, in.gradUBndCached);

    stage.vectors("ueqnSrcDev", M.source);
    // ...and the three things divDevRhoReff's explicit half is built from, because that half is where
    // the permeable-moving source gap sits (43 per cent of the term at step two). All three calls are
    // pure functions of arguments already in hand, so re-evaluating them here moves nothing.
    if (stage.on)
    {
        stage.scalars("ueqnMuEff", *muEff);
        // ...and U's PATCH values as the assembly sees them, which is the only input gaussGrad has on
        // a boundary cell that the internal field does not give it.
        {
            std::vector<vector> ub;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    ub.push_back(U.boundary[pi]->value()[static_cast<std::size_t>(i)]);
                }
            }
            stage.vectors("ueqnUbnd", ub);
            std::vector<vector> rv;
            std::vector<scalar> vf;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const std::vector<vector> r = U.boundary[pi]->refValues();
                const std::vector<scalar>* f = U.boundary[pi]->valueFractionPtr();
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const std::size_t k = static_cast<std::size_t>(i);
                    rv.push_back(k < r.size() ? r[k] : vector{0, 0, 0});
                    vf.push_back((f && k < f->size()) ? (*f)[k] : scalar(-1));
                }
            }
            stage.vectors("ueqnUbndRef", rv);
            stage.scalars("ueqnUbndVf", vf);
        }
        stage.tensors("ueqnGradU", in.gradUCached ? *in.gradUCached
                                   : in.gradULeastSq ? fvc::leastSquaresGrad(U, m, g, patches)
                                                     : fvc::gaussGrad(U, m, g, patches));
        stage.vectors("ueqnDivDevExpl",
                      divDevReffExplicit(U, *muEff, *muEffBnd, m, g, patches,
                                         in.gradULimitK, in.gradULeastSq,
                                         in.gradUCached, in.gradUBndCached));
    }

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
        // ...and the mangroves' drag and added mass, whose ddt(U) reads U.oldTime() and the step
        fvOptions::addSup(*in.fvOptions, M, U, scalar(0), g, /*forceDimensions=*/true, &rho, &mu,
                          in.UOld, in.deltaT);
    }

    // UEqn.relax(). damBreak names `".*" 1`, which relaxEquation() finds, and relax(1) still applies the
    // diagonal-dominance clamp -- see InterMomentumInput::relaxEquationU.
    stage.vectors("ueqnSrcPreRelax", M.source);
    stage.scalars("ueqnDiagPreRelax", M.diag);
    if (in.relaxEquationU && in.relaxU > scalar(0))
        relaxMatrix<vector>(M, U, m, patches, in.relaxU);
    stage.vectors("ueqnSrcRelax", M.source);
    stage.scalars("ueqnDiagRelax", M.diag);

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
    solveVector(solved, U, m, patches, sc.tolU, sc.relTolU, sc.maxIterU, sc.minIterU, sc.solutionD,
                perf, &sc.which);
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
