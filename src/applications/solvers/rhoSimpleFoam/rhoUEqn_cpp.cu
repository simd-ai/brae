// _cpp REFERENCE implementation -- see UEqn_cpp.cuh for the OpenFOAM provenance and the refusal contract.
#include "rhoUEqn_cpp.cuh"
#include "fvOptions_cpp.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "linearViscousStress_cpp.cuh"
#include "limitedSchemes_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include <stdexcept>

namespace brae {
namespace cpu {
namespace rhoSimple {

namespace {

// The convection operator for the requested scheme. `phi` here is the MASS flux, so the operator itself
// is unchanged from the incompressible one -- it is the field it is handed that carries the rho.
FvVectorMatrix divWithScheme(
    const GeometricField<vector>& U,
    const RhoMomentumInput&       in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    namespace ls = limitedSchemes;
    switch (in.scheme)
    {
        case DivScheme::LUST:
            return fvm::div(*in.phi, *in.phiBnd, U, ls::lustWeights(*in.phi, g), m, patches);

        case DivScheme::limitedLinearV:
        {
            const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
            return fvm::div(*in.phi, *in.phiBnd, U,
                            ls::limitedLinearVWeights(*in.phi, U, gradU, in.schemeCoeff, m, g),
                            m, patches);
        }

        case DivScheme::limitedLinear:
        {
            // NOT per-component and NOT the V form: LimitedScheme.H instantiates limitedLinear for a
            // vector as NVDTVD + limitFuncs::magSqr, so the limiter is built on the SCALAR magSqr(U).
            std::vector<scalar> mag2(m.nCells());
            for (label c = 0; c < m.nCells(); ++c)
            {
                const vector& u = U.internal[c];
                mag2[c] = u.x*u.x + u.y*u.y + u.z*u.z;
            }
            std::vector<std::vector<scalar>> mag2b(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const std::vector<vector>& ub = U.boundary[pi]->value();
                mag2b[pi].resize(patches[pi].size);
                for (label i = 0; i < patches[pi].size; ++i)
                    mag2b[pi][i] = ub[i].x*ub[i].x + ub[i].y*ub[i].y + ub[i].z*ub[i].z;
            }
            const std::vector<vector> gradM = fvc::gaussGrad(mag2, mag2b, m, g, patches);
            GeometricField<scalar> shim;      // limitedLinearWeights reads only .internal
            shim.internal = mag2;
            return fvm::div(*in.phi, *in.phiBnd, U,
                            ls::limitedLinearWeights(*in.phi, shim, gradM, in.schemeCoeff, m, g),
                            m, patches);
        }

        case DivScheme::upwind:
        case DivScheme::linearUpwind:
        case DivScheme::linearUpwindV:
        default:
            // linearUpwind DERIVES from upwind: identical weights, its whole effect being the deferred
            // correction applied below.
            return fvm::div(*in.phi, *in.phiBnd, U, m, patches);
    }
}

// How much of linearUpwind's deferred correction this scheme carries. LUST.H overrides correction() as
// 0.25*linearUpwind::correction, so it is a FACTOR and not a flag.
scalar correctionFactor(const RhoMomentumInput& in)
{
    if (in.scheme == DivScheme::linearUpwind || in.linearUpwind) return 1.0;
    if (in.scheme == DivScheme::LUST)                            return 0.25;
    return 0.0;
}

void refuseUnsupported(const RhoMomentumInput& in)
{
    if (in.hasMRF && !in.mrf)
        throw std::runtime_error(
            "rhoSimpleFoam UEqn_cpp: the case declares MRF, which UEqn.H applies via "
            "MRF.correctBoundaryVelocity(U) and MRF.DDt(rho, U) (rhoSimpleFoam/UEqn.H:3,8). No zones were "
            "supplied to this assembly; refusing rather than silently solving a different equation.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "rhoSimpleFoam UEqn_cpp: the case declares an fvOption this port does not implement"
            + (in.fvOptionUnsupported.empty() ? std::string()
                                              : std::string(" -- '") + in.fvOptionUnsupported + "'")
            + ". UEqn.H applies fvOptions(rho, U), fvOptions.constrain(UEqn) and fvOptions.correct(U) "
              "(rhoSimpleFoam/UEqn.H:11,17,21). explicitPorositySource (DarcyForchheimer and fixedCoeff) "
              "IS implemented; refusing rather than silently solving a different equation.");
    const bool haveMu  = in.muEff && in.muEffBnd;
    const bool haveRho = in.rho && in.rhoBnd && in.nuEff && in.nuEffBnd;
    if (!haveMu && !haveRho)
        throw std::runtime_error(
            "rhoSimpleFoam UEqn_cpp: divDevRhoReff needs the DYNAMIC viscosity rho*nuEff "
            "(linearViscousStress.C:107-117). Supply either muEff/muEffBnd directly, or rho/rhoBnd with "
            "nuEff/nuEffBnd so it can be formed. Neither was given.");
}

} // namespace


std::vector<scalar> dynamicViscosity(
    const std::vector<scalar>& rho,
    const std::vector<scalar>& nuEff)
{
    if (rho.size() != nuEff.size())
        throw std::runtime_error("rhoSimpleFoam UEqn_cpp: rho and nuEff differ in length.");
    std::vector<scalar> muEff(rho.size());
    for (std::size_t c = 0; c < rho.size(); ++c) muEff[c] = rho[c] * nuEff[c];
    return muEff;
}


std::vector<std::vector<scalar>> dynamicViscosityBoundary(
    const std::vector<std::vector<scalar>>& rhoBnd,
    const std::vector<std::vector<scalar>>& nuEffBnd)
{
    if (rhoBnd.size() != nuEffBnd.size())
        throw std::runtime_error("rhoSimpleFoam UEqn_cpp: rhoBnd and nuEffBnd differ in patch count.");
    std::vector<std::vector<scalar>> muEffBnd(rhoBnd.size());
    for (std::size_t pi = 0; pi < rhoBnd.size(); ++pi)
    {
        if (rhoBnd[pi].size() != nuEffBnd[pi].size())
            throw std::runtime_error("rhoSimpleFoam UEqn_cpp: rhoBnd and nuEffBnd differ on a patch.");
        muEffBnd[pi].resize(rhoBnd[pi].size());
        for (std::size_t i = 0; i < rhoBnd[pi].size(); ++i)
            muEffBnd[pi][i] = rhoBnd[pi][i] * nuEffBnd[pi][i];
    }
    return muEffBnd;
}


FvVectorMatrix assembleUEqn(
    const GeometricField<vector>& U,
    const RhoMomentumInput&       in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);

    // fvm::div(phi, U), with phi the MASS flux.
    FvVectorMatrix M = divWithScheme(U, in, m, g, patches);

    // linearUpwind's deferred correction. OpenFOAM applies it INSIDE fvm::div
    // (gaussConvectionScheme.C:112-115), so it lands here, before everything else -- and it is
    // SUBTRACTED, because `fvm += ...` on an fvMatrix means `source -= V*...` (fvMatrix.C:1855-1862).
    // linearUpwindV carries a DIFFERENT correction, not a scaled one, so it branches before the factor.
    if (in.scheme == DivScheme::linearUpwindV)
    {
        std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
        cellLimitGrad(gradU, U, in.gradULimitK, m, g, patches);
        const std::vector<vector> corr =
            limitedSchemes::linearUpwindVCorrection(*in.phi, U, gradU, m, g);
        for (std::size_t c = 0; c < corr.size(); ++c)
        {
            M.source[c].x -= corr[c].x;
            M.source[c].y -= corr[c].y;
            M.source[c].z -= corr[c].z;
        }
    }
    const scalar corrFac = correctionFactor(in);
    if (corrFac != 0.0)
    {
        std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
        cellLimitGrad(gradU, U, in.gradULimitK, m, g, patches);
        const std::vector<vector> corr =
            fvm::linearUpwindCorrection<vector, tensor>(*in.phi, gradU, m, g);
        for (std::size_t c = 0; c < corr.size(); ++c)
        {
            M.source[c].x -= corrFac * corr[c].x;
            M.source[c].y -= corrFac * corr[c].y;
            M.source[c].z -= corrFac * corr[c].z;
        }
    }

    // `bounded Gauss <scheme>`: - fvm::Sp(fvc::div(phi), U). Applied BEFORE relax, as OpenFOAM does. On
    // this solver div(phi) is the MASS imbalance, so the term vanishes as the continuity error does --
    // the same way, for the same reason, as in the incompressible solver.
    if (in.bounded)
    {
        SurfaceScalarField phis;
        phis.internal = *in.phi;
        phis.boundary = *in.phiBnd;
        const std::vector<scalar> divPhi = fvc::div(phis, m, g, patches);
        for (label c = 0; c < m.nCells(); ++c) M.diag[c] -= divPhi[c] * g.V()[c];
    }

    // turbulence->divDevRhoReff(U) -- THE compressible difference. linearViscousStress.C defines one
    // operator, and what makes it the compressible one is that the viscosity handed to it is the DYNAMIC
    // mu_eff = rho*nu_eff rather than the kinematic nu_eff. Both halves (the implicit laplacian into the
    // matrix and the explicit dev2 term into the source) come from the one shared call, so they cannot
    // drift apart, and that call is the SAME validated component the incompressible solver uses -- it is
    // parameterised by the viscosity, not specialised to a lineage.
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
                  in.gradULimitK);

    // == fvOptions(rho, U). rhoSimpleFoam's momentum equation is in FORCE units, which is what selects
    // fixedCoeff's rhoRef branch over the kinematic one.
    if (in.fvOpts && !in.fvOpts->empty())
    {
        cpu::fvOptions::addSup(*in.fvOpts, M, U, /*nu (unused with muLaminar)=*/0.0, g,
                               /*forceDimensions=*/true, in.rho, in.muLaminar);
    }

    // + MRF.DDt(rho, U), UEqn.H:8. MRFZoneList::DDt(rho,U) is rho*DDt(U), so the Coriolis acceleration is
    // formed first and then rho-weighted per cell. Weighting AFTER is not a rearrangement: addCoriolis
    // already returns the extensive (V-multiplied) source, and rho is a cell field, so rho*(a*V) and
    // (rho*a)*V are the same number -- but the rho must be there, and an unweighted Coriolis term would
    // be wrong by exactly the density.
    if (in.mrf)
    {
        if (!in.rho)
            throw std::runtime_error(
                "rhoSimpleFoam UEqn_cpp: MRF.DDt(rho, U) needs rho itself (MRFZoneList.C -- DDt(rho,U) is "
                "rho*DDt(U)); only a pre-formed muEff was supplied, from which rho cannot be recovered.");
        std::vector<vector> acc(m.nCells(), vector{0.0, 0.0, 0.0});
        MRF::addCoriolis(*in.mrf, U.internal, g.V(), acc);
        const std::vector<scalar>& rho = *in.rho;
        for (label c = 0; c < m.nCells(); ++c)
        {
            M.source[c].x += rho[c] * acc[c].x;
            M.source[c].y += rho[c] * acc[c].y;
            M.source[c].z += rho[c] * acc[c].z;
        }
    }

    // THE GUARD IS "THE CASE NAMES A FACTOR", NOT "THE FACTOR IS BELOW 1".
    //
    // OpenFOAM's fvMatrix::relax() looks the name up -- `if (mesh.relaxEquation(name, coeff)) relax(coeff)`
    // (fvMatrix.C:1250-1263), and relaxEquation is `eqnRelaxDict_.found(name) || found("default")`
    // (solution.C:330-334). relax(alpha) itself returns early ONLY on alpha <= 0 (fvMatrix.C:1102-1107),
    // so a factor of exactly 1 still runs the whole body: the diagonal-dominance clamp
    // D = max(|D|, sumOff)/alpha and the matching source term S += (D - D0)*psi.
    //
    // The previous guard `relaxU > 0 && relaxU < 1` therefore skipped a case that NAMES 1 -- and
    // validation/sbMatched names `p 1` while validation/kEpsCorrect names `k 1` and `epsilon 1`, so this
    // is live rather than theoretical. rhoPEqn_cpp and rhoPcEqn_cpp already carry the correct form as
    // relaxPSpecified; U and he simply never got it, and the device path (rhoUEqn.cu's relaxEquationU)
    // has had it all along -- so this brings the reference up to the twin it is the oracle for.
    if (in.relaxEquationU && in.relaxU > 0.0)
    {
        relaxMatrix<vector>(M, U, m, patches, in.relaxU);
    }
    return M;
}


void addPressureGradient(
    FvVectorMatrix&               UEqn,
    const GeometricField<scalar>& p,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    // solve(UEqn == -fvc::grad(p)). The right-hand side of an fvMatrix equation is its source, and
    // fvc::grad returns a per-volume quantity, so the extensive form is -grad(p)*V. p is the ABSOLUTE
    // pressure here, not the kinematic p/rho the incompressible solver carries, which is why this term
    // needs no rho: it is already a force per unit volume.
    const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, patches);
    for (label c = 0; c < m.nCells(); ++c)
    {
        UEqn.source[c].x -= gradP[c].x * g.V()[c];
        UEqn.source[c].y -= gradP[c].y * g.V()[c];
        UEqn.source[c].z -= gradP[c].z * g.V()[c];
    }
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
