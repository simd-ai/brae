#include "kOmegaSSTLM_cpp.cuh"
#include "bound_cpp.cuh"
#include "limitedSchemes_cpp.cuh"
#include "foam_dict.cuh"
#include "pbicgstab.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace kOmegaSSTLM {

namespace {

// sqrt(2*magSqr(skew(t))). skew(t)_ij = 0.5*(t_ij - t_ji): the diagonal is zero and the off-diagonals
// pair up, so magSqr = 2*(a^2 + b^2 + c^2) over the three independent entries.
inline scalar OmegaOf(const tensor& t)
{
    const scalar a = 0.5 * (t.xy - t.yx);
    const scalar b = 0.5 * (t.xz - t.zx);
    const scalar c = 0.5 * (t.yz - t.zy);
    return std::sqrt(2.0 * (2.0 * (a * a + b * b + c * c)));
}

// The convection matrix for one transition scalar. Same three schemes the turbulence scalars elsewhere
// support: linearUpwind derives from upwind, so its MATRIX is upwind's and the scheme is entirely the
// deferred source correction the caller applies.
FvScalarMatrix divWithScheme(
    const SurfaceScalarField&     phi,
    const GeometricField<scalar>& vf,
    const std::vector<vector>&    gradVf,
    bool                          limitedLinear,
    scalar                        coeff,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    if (limitedLinear)
    {
        return fvm::div(phi.internal, phi.boundary, vf,
                        limitedSchemes::limitedLinearWeights(phi.internal, vf, gradVf, coeff, m, g),
                        m, patches);
    }
    return fvm::div(phi.internal, phi.boundary, vf, m, patches);
}

// A volScalarField diffusivity's BOUNDARY comes from its operands' boundary fields, not from
// interpolating the cell value out to the face -- the same rule kOmegaSST_cpp's effectiveDiffusivity
// carries, and for the same reason: nut's `calculated` patch value differs from the adjacent cell's.
SurfaceScalarField diffusivity(
    const std::vector<scalar>&    DCell,
    const GeometricField<scalar>& nutField,
    scalar                        mult,      // sigmaThetat for ReThetat, 1 for gammaInt
    scalar                        nu,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    SurfaceScalarField sf = fvc::interpolate(DCell, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& nb = nutField.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            sf.boundary[pi][i] = mult * (nb[i] + nu);
        }
    }
    return sf;
}

std::vector<vector> boundaryGrad(
    const GeometricField<scalar>& vf,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    std::vector<std::vector<scalar>> vfb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) vfb[pi] = vf.boundary[pi]->value();
    return fvc::gaussGrad(vf.internal, vfb, m, g, patches);
}

}   // namespace


void readCoeffs(const void* rasDictV, Coeffs& co)
{
    const FoamDict* ras = static_cast<const FoamDict*>(rasDictV);
    if (!ras) return;
    const FoamDict* d = ras->subDict("kOmegaSSTLMCoeffs");
    if (!d) d = ras;
    co.ca1           = d->scalarOr("ca1",           co.ca1);
    co.ca2           = d->scalarOr("ca2",           co.ca2);
    co.ce1           = d->scalarOr("ce1",           co.ce1);
    co.ce2           = d->scalarOr("ce2",           co.ce2);
    co.cThetat       = d->scalarOr("cThetat",       co.cThetat);
    co.sigmaThetat   = d->scalarOr("sigmaThetat",   co.sigmaThetat);
    co.lambdaErr     = d->scalarOr("lambdaErr",     co.lambdaErr);
    co.maxLambdaIter = d->intOr   ("maxLambdaIter", co.maxLambdaIter);
}


std::vector<scalar> ReThetac(const std::vector<scalar>& ReThetat)
{
    std::vector<scalar> out(ReThetat.size());
    for (std::size_t c = 0; c < ReThetat.size(); ++c)
    {
        const scalar r = ReThetat[c];
        if (r <= 1870)
        {
            const scalar r2 = r * r, r3 = r2 * r, r4 = r2 * r2;
            out[c] = r
                   - 396.035e-2
                   + 120.656e-4 * r
                   - 868.230e-6 * r2
                   + 696.506e-9 * r3
                   - 174.105e-12 * r4;
        }
        else
        {
            out[c] = r - 593.11 - 0.482 * (r - 1870);
        }
    }
    return out;
}


std::vector<scalar> Flength(
    const std::vector<scalar>& ReThetat,
    const std::vector<scalar>& omega,
    const std::vector<scalar>& y,
    const scalar               nu)
{
    std::vector<scalar> out(ReThetat.size());
    for (std::size_t c = 0; c < ReThetat.size(); ++c)
    {
        const scalar r = ReThetat[c];
        scalar F;
        if (r < 400)
        {
            F = 398.189e-1 - 119.270e-4 * r - 132.567e-6 * r * r;
        }
        else if (r < 596)
        {
            F = 263.404 - 123.939e-2 * r + 194.548e-5 * r * r - 101.695e-8 * r * r * r;
        }
        else if (r < 1200)
        {
            F = 0.5 - 3e-4 * (r - 596);
        }
        else
        {
            F = 0.3188;
        }
        // Inside the viscous sublayer the correlation is replaced by 40, blended by Fsublayer. This is
        // what keeps the model from producing transition on the first cell off the wall.
        const scalar a = y[c] * y[c] * omega[c] / (200.0 * nu);
        const scalar Fsublayer = std::exp(-(a * a));
        out[c] = F * (1.0 - Fsublayer) + 40.0 * Fsublayer;
    }
    return out;
}


std::vector<scalar> ReThetat0(
    const std::vector<scalar>& Us,
    const std::vector<scalar>& dUsds,
    const std::vector<scalar>& k,
    const scalar               nu,
    const Coeffs&              co,
    int*                       maxIterUsed)
{
    std::vector<scalar> out(Us.size());
    int maxIter = 0;
    for (std::size_t c = 0; c < Us.size(); ++c)
    {
        const scalar Tu = std::fmax(100.0 * std::sqrt((2.0/3.0) * k[c]) / Us[c], 0.027);

        // OpenFOAM initialises lambda to zero every call and notes in a comment that caching it between
        // time steps would converge faster. Transcribed as written: caching would change the iterate
        // sequence and so the fixed point it lands on to within lambdaErr.
        scalar lambda = 0, thetat = 0, lambdaErr;
        int    iter = 0;
        do
        {
            const scalar lambda0 = lambda;
            const scalar Flambda =
                dUsds[c] <= 0
              ? 1.0 - (-12.986 * lambda - 123.66 * lambda * lambda - 405.689 * lambda * lambda * lambda)
                      * std::exp(-std::pow(Tu / 1.5, 1.5))
              : 1.0 + 0.275 * (1.0 - std::exp(-35.0 * lambda))
                      * std::exp(Tu <= 1.3 ? -Tu / 0.5 : -2.0 * Tu);

            thetat = (Tu <= 1.3)
                   ? (1173.51 - 589.428 * Tu + 0.2196 / (Tu * Tu)) * Flambda * nu / Us[c]
                   : 331.50 * std::pow(Tu - 0.5658, -0.671) * Flambda * nu / Us[c];

            lambda = thetat * thetat / nu * dUsds[c];
            lambda = (lambda < -0.1) ? -0.1 : (lambda > 0.1 ? 0.1 : lambda);

            lambdaErr = std::fabs(lambda - lambda0);
            if (++iter > maxIter) maxIter = iter;
        }
        while (lambdaErr > co.lambdaErr);

        out[c] = std::fmax(thetat * Us[c] / nu, 20.0);
    }
    if (maxIterUsed) *maxIterUsed = maxIter;
    // OpenFOAM warns and carries on -- the loop itself has no cap. Matching that means a pathological
    // cell costs iterations, not correctness.
    if (maxIter > co.maxLambdaIter)
        std::fprintf(stderr, "brae WARNING: kOmegaSSTLM lambda iterations (%d) exceed maxLambdaIter (%d)\n",
                     maxIter, co.maxLambdaIter);
    return out;
}


std::vector<scalar> Fthetat(
    const std::vector<scalar>& Us,
    const std::vector<scalar>& Omega,
    const std::vector<scalar>& omega,
    const std::vector<scalar>& y,
    const std::vector<scalar>& ReThetat,
    const std::vector<scalar>& gammaInt,
    const scalar               nu,
    const Coeffs&              co)
{
    // dimensionedScalar("deltaMin", dimLength, SMALL). OpenFOAM's SMALL is 1e-15, not VSMALL 1e-300 --
    // and delta sits under y in y/delta, so the two are not interchangeable at a wall cell.
    const scalar deltaMin = 1e-15;
    const scalar invCe2   = 1.0 / co.ce2;
    std::vector<scalar> out(Us.size());
    for (std::size_t c = 0; c < Us.size(); ++c)
    {
        const scalar delta = std::fmax(375.0 * Omega[c] * nu * ReThetat[c] * y[c] / (Us[c] * Us[c]),
                                       deltaMin);
        const scalar ReOmega = y[c] * y[c] * omega[c] / nu;
        const scalar w       = ReOmega / 1e5;
        const scalar Fwake   = std::exp(-(w * w));
        const scalar yd      = y[c] / delta;
        const scalar yd2     = yd * yd;
        const scalar t       = (gammaInt[c] - invCe2) / (1.0 - invCe2);
        out[c] = std::fmin(std::fmax(Fwake * std::exp(-(yd2 * yd2)), 1.0 - t * t), 1.0);
    }
    return out;
}


StrainState strain(
    const std::vector<tensor>& gradU,
    const std::vector<vector>& U,
    const scalar               deltaU)
{
    const std::size_t nC = gradU.size();
    StrainState st;
    st.S.resize(nC); st.Omega.resize(nC); st.Us.resize(nC); st.dUsds.resize(nC);
    for (std::size_t c = 0; c < nC; ++c)
    {
        const tensor& t = gradU[c];
        // 2*magSqr(symm) and 2*magSqr(skew) from the same nine components.
        scalar symSq = 0, skSq = 0;
        const scalar tv[9] = {t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                const scalar sy = 0.5 * (tv[i*3+j] + tv[j*3+i]);
                const scalar sk = 0.5 * (tv[i*3+j] - tv[j*3+i]);
                symSq += sy * sy;
                skSq  += sk * sk;
            }
        st.S[c]     = std::sqrt(2.0 * symSq);
        st.Omega[c] = std::sqrt(2.0 * skSq);
        // deltaU is a FLOOR on |U| (dimensionedScalar("deltaU", dimVelocity, SMALL)), not a blend.
        st.Us[c]    = std::fmax(mag(U[c]), deltaU);
        // dUsds = (U & (U & gradU))/sqr(Us) = U_i U_j gradU_ij / Us^2. OpenFOAM's `v & T` contracts the
        // FIRST index and fvc::grad gives gradU_ij = d(U_j)/d(x_i), so the pair reads out the
        // streamwise acceleration.
        const scalar Uv[3] = {U[c].x, U[c].y, U[c].z};
        scalar num = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                num += Uv[i] * Uv[j] * tv[i*3+j];
        st.dUsds[c] = num / (st.Us[c] * st.Us[c]);
    }
    return st;
}


ReThetatPrep reThetatPrep(
    const StrainState&         st,
    const std::vector<scalar>& k,
    const std::vector<scalar>& omega,
    const std::vector<scalar>& y,
    const std::vector<scalar>& ReThetatF,
    const std::vector<scalar>& gammaInt,
    const scalar               nu,
    const Coeffs&              co)
{
    const std::size_t nC = st.Us.size();
    ReThetatPrep out;
    out.Fthetat = Fthetat(st.Us, st.Omega, omega, y, ReThetatF, gammaInt, nu, co);
    const std::vector<scalar> rt0 = ReThetat0(st.Us, st.dUsds, k, nu, co, nullptr);
    out.sp.resize(nC);
    out.su.resize(nC);
    for (std::size_t c = 0; c < nC; ++c)
    {
        const scalar t = 500.0 * nu / (st.Us[c] * st.Us[c]);
        const scalar Pthetat = (co.cThetat / t) * (1.0 - out.Fthetat[c]);
        out.sp[c] = Pthetat;                 // - Sp(Pthetat, ReThetat)
        out.su[c] = Pthetat * rt0[c];        // == Pthetat*ReThetat0
    }
    return out;
}


GammaPrep gammaPrep(
    const StrainState&         st,
    const std::vector<scalar>& k,
    const std::vector<scalar>& omega,
    const std::vector<scalar>& y,
    const std::vector<scalar>& ReThetatF,
    const std::vector<scalar>& gammaInt,
    const scalar               nu,
    const Coeffs&              co)
{
    const std::size_t nC = st.Us.size();
    const std::vector<scalar> reThetac = ReThetac(ReThetatF);
    const std::vector<scalar> fLength  = Flength(ReThetatF, omega, y, nu);
    GammaPrep out;
    out.sp.resize(nC);
    out.su.resize(nC);
    for (std::size_t c = 0; c < nC; ++c)
    {
        const scalar Rev = y[c] * y[c] * st.S[c] / nu;
        const scalar RT  = k[c] / (nu * omega[c]);

        const scalar Fonset1 = Rev / (2.193 * reThetac[c]);
        const scalar f1sq    = Fonset1 * Fonset1;
        const scalar Fonset2 = std::fmin(std::fmax(Fonset1, f1sq * f1sq), 2.0);
        const scalar rt      = RT / 2.5;
        const scalar Fonset3 = std::fmax(1.0 - rt * rt * rt, 0.0);
        const scalar Fonset  = std::fmax(Fonset2 - Fonset3, 0.0);

        const scalar Pgamma = co.ca1 * fLength[c] * st.S[c] * std::sqrt(gammaInt[c] * Fonset);

        const scalar q     = 0.25 * RT;
        const scalar q2    = q * q;
        const scalar Fturb = std::exp(-(q2 * q2));
        const scalar Egamma = co.ca2 * st.Omega[c] * Fturb * gammaInt[c];

        out.su[c] = Pgamma + Egamma;
        out.sp[c] = co.ce1 * Pgamma + co.ce2 * Egamma;
    }
    return out;
}


std::vector<scalar> gammaEff(
    const StrainState&         st,
    const std::vector<scalar>& k,
    const std::vector<scalar>& omega,
    const std::vector<scalar>& y,
    const std::vector<scalar>& ReThetatF,
    const std::vector<scalar>& gammaInt,
    const std::vector<scalar>& FthetatF,
    const scalar               nu)
{
    const std::size_t nC = st.Us.size();
    const std::vector<scalar> reThetac = ReThetac(ReThetatF);
    std::vector<scalar> out(nC);
    for (std::size_t c = 0; c < nC; ++c)
    {
        const scalar Rev = y[c] * y[c] * st.S[c] / nu;
        const scalar RT  = k[c] / (nu * omega[c]);
        const scalar r   = RT / 20.0;
        const scalar r2  = r * r;
        const scalar Freattach = std::exp(-(r2 * r2));
        const scalar gammaSep =
            std::fmin(2.0 * std::fmax(Rev / (3.235 * reThetac[c]) - 1.0, 0.0) * Freattach, 2.0)
          * FthetatF[c];
        out[c] = std::fmax(gammaInt[c], gammaSep);
    }
    return out;
}


void correctReThetatGammaInt(
    const GeometricField<vector>&  U,
    const GeometricField<scalar>&  k,
    const GeometricField<scalar>&  omega,
    const GeometricField<scalar>&  nutField,
    GeometricField<scalar>&        ReThetat,
    GeometricField<scalar>&        gammaInt,
    std::vector<scalar>&           gammaIntEff,
    const SurfaceScalarField&      phi,
    const std::vector<scalar>&     y,
    const scalar                   nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    const scalar                   relaxReThetat,
    const scalar                   relaxGammaInt,
    const scalar                   tol,
    const scalar                   relTol,
    const int                      maxIter,
    const Coeffs&                  co,
    Residuals*                     res,
    const bool                     bounded,
    const bool                     limitedLinear,
    const bool                     linearUpwind,
    const scalar                   limiterCoeff,
    const bool                     correctedLaplacian,
    const scalar                   snGradLimitCoeff)
{
    const label nC = m.nCells();
    const std::vector<scalar>& V = g.V();

    // ---- fields derived from the velocity gradient -----------------------------------------------
    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    // deltaU is dimensionedScalar("deltaU", dimVelocity, SMALL) -- OpenFOAM's SMALL, 1e-15.
    const StrainState st = strain(gradU, U.internal, 1e-15);

    const std::vector<scalar> divU = fvc::div(phi, m, g, patches);

    // Fthetat is formed ONCE, from the OLD ReThetat and the OLD gammaInt, and the SAME values are
    // reused in gammaSep after both solves. Recomputing it there would be a different model.
    const ReThetatPrep rp =
        reThetatPrep(st, k.internal, omega.internal, y, ReThetat.internal, gammaInt.internal, nu, co);
    const std::vector<scalar>& fThetat = rp.Fthetat;

    // ---- ReThetat equation -----------------------------------------------------------------------
    {
        std::vector<scalar> DCell(nC);
        for (label c = 0; c < nC; ++c) DCell[c] = co.sigmaThetat * (nutField.internal[c] + nu);
        const SurfaceScalarField Df =
            diffusivity(DCell, nutField, co.sigmaThetat, nu, m, g, patches);

        const std::vector<vector> gradRt = boundaryGrad(ReThetat, m, g, patches);
        FvScalarMatrix M =
            divWithScheme(phi, ReThetat, gradRt, limitedLinear, limiterCoeff, m, g, patches);
        {
            FvScalarMatrix L = fvm::laplacian(Df, ReThetat, m, g, patches, correctedLaplacian);
            if (correctedLaplacian)
            {
                const std::vector<vector> gradVf = boundaryGrad(ReThetat, m, g, patches);
                const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                    Df, ReThetat, gradVf, m, g, patches, snGradLimitCoeff);
                for (label c = 0; c < nC; ++c) L.source[c] -= corr[c];
            }
            addEqual(M, L, -1.0);
        }
        for (label c = 0; c < nC; ++c)
        {
            M.source[c] += rp.su[c] * V[c];                  // == Pthetat*ReThetat0
            M.diag[c]   += rp.sp[c] * V[c];                  // - Sp(Pthetat, ReThetat)
            if (bounded) M.diag[c] -= divU[c] * V[c];        // `bounded` -> - Sp(fvc::div(phi), .)
        }
        if (linearUpwind)
        {
            const std::vector<scalar> corr =
                fvm::linearUpwindCorrection<scalar, vector>(phi.internal, gradRt, m, g);
            for (label c = 0; c < nC; ++c) M.source[c] -= corr[c];
        }
        relaxMatrix(M, ReThetat, m, patches, relaxReThetat);
        const SolverPerformance p =
            pbicgstab(M, ReThetat.internal, m, patches, tol, relTol, maxIter);
        if (res) res->ReThetat = p.initialResidual;
        ReThetat.evaluateBoundary();
        bound(ReThetat, 0.0, m, g, patches);
    }

    // The gamma prep is formed AFTER the ReThetat solve, so ReThetac, Flength and Fonset all see the NEW
    // ReThetat -- and gammaSep below sees the NEW gammaInt. Only Fthetat is lagged.
    const GammaPrep gp =
        gammaPrep(st, k.internal, omega.internal, y, ReThetat.internal, gammaInt.internal, nu, co);

    // ---- intermittency equation ------------------------------------------------------------------
    {
        std::vector<scalar> DCell(nC);
        // DgammaIntEff = nut + nu. There is no sigmaGamma in this model -- the intermittency diffuses
        // at the full effective viscosity.
        for (label c = 0; c < nC; ++c) DCell[c] = nutField.internal[c] + nu;
        const SurfaceScalarField Df = diffusivity(DCell, nutField, 1.0, nu, m, g, patches);

        const std::vector<vector> gradGi = boundaryGrad(gammaInt, m, g, patches);
        FvScalarMatrix M =
            divWithScheme(phi, gammaInt, gradGi, limitedLinear, limiterCoeff, m, g, patches);
        {
            FvScalarMatrix L = fvm::laplacian(Df, gammaInt, m, g, patches, correctedLaplacian);
            if (correctedLaplacian)
            {
                const std::vector<vector> gradVf = boundaryGrad(gammaInt, m, g, patches);
                const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                    Df, gammaInt, gradVf, m, g, patches, snGradLimitCoeff);
                for (label c = 0; c < nC; ++c) L.source[c] -= corr[c];
            }
            addEqual(M, L, -1.0);
        }
        for (label c = 0; c < nC; ++c)
        {
            M.source[c] += gp.su[c] * V[c];                  // == Pgamma + Egamma
            M.diag[c]   += gp.sp[c] * V[c];                  // - Sp(ce1*Pgamma + ce2*Egamma, gammaInt)
            if (bounded) M.diag[c] -= divU[c] * V[c];
        }
        if (linearUpwind)
        {
            const std::vector<scalar> corr =
                fvm::linearUpwindCorrection<scalar, vector>(phi.internal, gradGi, m, g);
            for (label c = 0; c < nC; ++c) M.source[c] -= corr[c];
        }
        relaxMatrix(M, gammaInt, m, patches, relaxGammaInt);
        const SolverPerformance p =
            pbicgstab(M, gammaInt.internal, m, patches, tol, relTol, maxIter);
        if (res) res->gammaInt = p.initialResidual;
        gammaInt.evaluateBoundary();
        bound(gammaInt, 0.0, m, g, patches);
    }

    // ---- separation-induced transition -----------------------------------------------------------
    gammaIntEff = gammaEff(st, k.internal, omega.internal, y,
                           ReThetat.internal, gammaInt.internal, fThetat, nu);
}


void correct(
    const GeometricField<vector>&  U,
    GeometricField<scalar>&        k,
    GeometricField<scalar>&        omega,
    GeometricField<scalar>&        nutField,
    GeometricField<scalar>&        ReThetat,
    GeometricField<scalar>&        gammaInt,
    std::vector<scalar>&           gammaIntEff,
    const SurfaceScalarField&      phi,
    const std::vector<scalar>&     y,
    const scalar                   nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    const scalar                   relaxOmega,
    const scalar                   relaxK,
    const scalar                   tol,
    const scalar                   relTol,
    const int                      maxIter,
    const KOmegaSSTCoeffs&         sstCo,
    const Coeffs&                  co,
    kOmegaSST::SSTResiduals*       sstRes,
    Residuals*                     lmRes,
    const bool                     bounded,
    const bool                     limitedLinear,
    const bool                     linearUpwind,
    const scalar                   limiterCoeff,
    const bool                     correctedLaplacian,
    const scalar                   snGradLimitCoeff)
{
    const label nC = m.nCells();
    // OpenFOAM constructs gammaIntEff_ as ZERO and kOmegaSST::correct() runs FIRST, so the very first
    // outer iteration advances k with no turbulent production at all and with the destruction clamped
    // to a tenth. That is not a warm-up hack, it is the model as written -- seeding gammaIntEff to 1
    // instead would make the first iteration fully turbulent and change where the transition lands.
    if (gammaIntEff.size() != static_cast<std::size_t>(nC)) gammaIntEff.assign(nC, 0.0);

    kOmegaSST::LMHooks hooks;
    hooks.gammaIntEff = &gammaIntEff;

    kOmegaSST::correct(U, k, omega, nutField, phi, y, nu, m, g, patches,
                       relaxOmega, relaxK, tol, relTol, maxIter, sstCo, sstRes,
                       bounded, limitedLinear, limiterCoeff, linearUpwind,
                       correctedLaplacian, snGradLimitCoeff, &hooks);

    correctReThetatGammaInt(U, k, omega, nutField, ReThetat, gammaInt, gammaIntEff, phi, y, nu,
                            m, g, patches, relaxOmega, relaxOmega, tol, relTol, maxIter, co, lmRes,
                            bounded, limitedLinear, linearUpwind, limiterCoeff,
                            correctedLaplacian, snGradLimitCoeff);
}

} // namespace kOmegaSSTLM
} // namespace cpu
} // namespace brae
