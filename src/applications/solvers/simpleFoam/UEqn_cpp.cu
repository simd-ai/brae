// _cpp REFERENCE implementation -- see UEqn_cpp.cuh for the OpenFOAM provenance and the refusal contract.
#include "UEqn_cpp.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "linearViscousStress_cpp.cuh"
#include "limitedSchemes_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include <stdexcept>

namespace brae {
namespace cpu {

namespace {

// The convection operator for the requested scheme. Three KINDS, and the branch is the whole point:
// weights only, deferred correction only, or both -- see DivScheme in UEqn_cpp.cuh.
FvVectorMatrix divWithScheme(
    const GeometricField<vector>& U,
    const MomentumInput&          in,
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
            // vector as NVDTVD + limitFuncs::magSqr, so the limiter is built on the SCALAR magSqr(U) and
            // its Gauss gradient (LimitedScheme.C::calcLimiter).
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
            // correction applied by the callers below.
            return fvm::div(*in.phi, *in.phiBnd, U, m, patches);
    }
}

// How much of linearUpwind's deferred correction this scheme carries. LUST is why this is a FACTOR and
// not a flag: LUST.H overrides correction() as 0.25*linearUpwind::correction, so a port treating LUST as
// "weights only" would drop a quarter of the scheme silently.
scalar correctionFactor(const MomentumInput& in)
{
    if (in.scheme == DivScheme::linearUpwind || in.linearUpwind) return 1.0;
    if (in.scheme == DivScheme::LUST)                            return 0.25;
    return 0.0;
}

void refuseUnsupported(const MomentumInput& in)
{
    // UEqn.H reaches MRF twice (correctBoundaryVelocity, DDt) and fvOptions three times (the source, the
    // constraint, the correction). Neither is optional when the case declares it. Refusing here is the
    // point: brae has already shipped a compressible path that read MRFProperties, ignored it, converged,
    // and reported nothing wrong.
    if (in.hasMRF && !in.mrf)
        throw std::runtime_error(
            "UEqn_cpp: the case declares MRF, which UEqn.H applies via MRF.correctBoundaryVelocity(U) and "
            "MRF.DDt(U) (simpleFoam/UEqn.H:3,8). No zones were supplied to this assembly; refusing rather "
            "than silently solving a different equation.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "UEqn_cpp: the case declares fvOptions, which UEqn.H applies as fvOptions(U), "
            "fvOptions.constrain(UEqn) and fvOptions.correct(U) (simpleFoam/UEqn.H:11,17,23). The _cpp "
            "reference does not implement it; refusing rather than silently solving a different equation.");
}

} // namespace


FvVectorMatrix momentumCore(
    const GeometricField<vector>& U,
    const MomentumInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);

    // fvm::div(phi, U) -- the convection operator. Its implicit weights come from the div SCHEME, which is
    // why the scheme is a first-class part of the port manifest rather than a detail.
    FvVectorMatrix M = divWithScheme(U, in, m, g, patches);

    // linearUpwind's deferred correction. OpenFOAM applies it INSIDE fvm::div (gaussConvectionScheme.C:
    // 112-115), so it lands here, before everything else -- and it is SUBTRACTED, because `fvm += ...`
    // on an fvMatrix means `source -= V*...` (fvMatrix.C:1855-1862). The gradient is the one the scheme
    // NAMES (`linearUpwind grad(U)`), resolved through gradSchemes by the caller's envelope check.
    // linearUpwindV carries a DIFFERENT correction, not a scaled one, so it branches before the factor.
    if (in.scheme == DivScheme::linearUpwindV)
    {
        std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
        // `linearUpwind <name>` where <name> resolves to `cellLimited Gauss linear <k>`:
        // the correction uses the LIMITED gradient. k = 0 leaves the plain Gauss one.
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
        // `linearUpwind <name>` where <name> resolves to `cellLimited Gauss linear <k>`:
        // the correction uses the LIMITED gradient. k = 0 leaves the plain Gauss one.
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

    // - fvm::laplacian(nuEff, U), the implicit half of divDevReff. Face nuEff takes the BOUNDARY field on
    // boundary faces, not the owner cell value; see interpolateEff in linearViscousStress_cpp.cu.
    addEqual(M, fvm::laplacian<vector>(
                    effectiveFaceViscosity(*in.nuEff, *in.nuEffBnd, m, g, patches), U, m, g, patches),
             -1.0);
    return M;
}


FvVectorMatrix assembleUEqn(
    const GeometricField<vector>& U,
    const MomentumInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);

    FvVectorMatrix M = divWithScheme(U, in, m, g, patches);

    // linearUpwind's deferred correction. OpenFOAM applies it INSIDE fvm::div (gaussConvectionScheme.C:
    // 112-115), so it lands here, before everything else -- and it is SUBTRACTED, because `fvm += ...`
    // on an fvMatrix means `source -= V*...` (fvMatrix.C:1855-1862). The gradient is the one the scheme
    // NAMES (`linearUpwind grad(U)`), resolved through gradSchemes by the caller's envelope check.
    // linearUpwindV carries a DIFFERENT correction, not a scaled one, so it branches before the factor.
    if (in.scheme == DivScheme::linearUpwindV)
    {
        std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
        // `linearUpwind <name>` where <name> resolves to `cellLimited Gauss linear <k>`:
        // the correction uses the LIMITED gradient. k = 0 leaves the plain Gauss one.
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
        // `linearUpwind <name>` where <name> resolves to `cellLimited Gauss linear <k>`:
        // the correction uses the LIMITED gradient. k = 0 leaves the plain Gauss one.
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

    // `bounded`: - fvm::Sp(fvc::div(phi), U). Applied BEFORE relax, as OpenFOAM does -- it is part of the
    // matrix the relaxation then acts on, not a correction bolted on afterwards.
    if (in.bounded)
    {
        SurfaceScalarField phis;
        phis.internal = *in.phi;
        phis.boundary = *in.phiBnd;
        const std::vector<scalar> divPhi = fvc::div(phis, m, g, patches);
        const std::vector<scalar>& V = g.V();
        for (std::size_t c = 0; c < M.diag.size(); ++c) M.diag[c] -= divPhi[c] * V[c];
    }

    // turbulence->divDevReff(U): implicit -laplacian(nuEff,U) into the matrix AND the explicit
    // -div(nuEff*dev2(T(grad U))) into the source. Both halves, one call, so they cannot drift apart.
    addDivDevReff(M, U, *in.nuEff, *in.nuEffBnd, m, g, patches, in.correctedLaplacian,
                  in.snGradLimitCoeff);

    // == fvOptions(U). BEFORE relax, as UEqn.H has it: the source is part of the matrix relaxation then
    // acts on. The double negation (`eqn -= porosityEqn` inside, `UEqn == fvOptions(U)` outside) cancels,
    // so the porosity enters as written -- see fvOptions_cpp.cuh.
    // + MRF.DDt(U), UEqn.H:8. Part of the LHS expression, so it is in the matrix BEFORE relax, exactly
    // as fvOptions' source is. Explicit in U: OpenFOAM builds a volVectorField from the current U.
    if (in.mrf) MRF::addCoriolis(*in.mrf, U.internal, g.V(), M.source);

    if (in.options) fvOptions::addSup(*in.options, M, U, in.nuLaminar, g);

    // UEqn.relax(). OpenFOAM guards this with if(relaxEquation(name)); a factor of 1 is the identity, and
    // relaxMatrix already early-returns on alpha <= 0.
    if (in.relaxU > 0.0 && in.relaxU < 1.0)
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
    // fvc::grad returns a per-volume quantity, so the extensive form is -grad(p)*V.
    const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, patches);
    const std::vector<scalar>& V = g.V();
    for (std::size_t c = 0; c < gradP.size(); ++c)
    {
        UEqn.source[c].x -= gradP[c].x * V[c];
        UEqn.source[c].y -= gradP[c].y * V[c];
        UEqn.source[c].z -= gradP[c].z * V[c];
    }
}

} // namespace cpu
} // namespace brae
