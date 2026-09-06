#include "SpalartAllmaras_cpp.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "pbicgstab.cuh"
#include "limitedSchemes_cpp.cuh"
#include "bound_cpp.cuh"
#include "nut_wall_function.cuh"

#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace brae {
namespace cpu {
namespace SA {

namespace {

namespace ls = limitedSchemes;

constexpr scalar SMALL_ = 1e-15;

// Omega = sqrt(2)*mag(skew(gradU)). skew(t) = (t - t^T)/2; mag() is the Frobenius norm over all nine
// components. This is the VORTICITY magnitude -- the k-epsilon family's S2 uses symm() instead, and
// swapping them silently produces a plausible but different model.
scalar omegaOf(const tensor& t)
{
    const scalar a = 0.5 * (t.xy - t.yx);
    const scalar b = 0.5 * (t.xz - t.zx);
    const scalar c = 0.5 * (t.yz - t.zy);
    // The diagonal of a skew tensor is zero and the off-diagonals pair up, so magSqr = 2*(a^2+b^2+c^2).
    const scalar magSqrSkew = 2.0 * (a * a + b * b + c * c);
    return std::sqrt(2.0) * std::sqrt(magSqrSkew);
}

FvScalarMatrix divWithScheme(
    const SurfaceScalarField&     phi,
    const GeometricField<scalar>& vf,
    const std::vector<vector>&    gradVf,
    const DivScheme&              sch,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    if (sch.limitedLinear)
    {
        return fvm::div(phi.internal, phi.boundary, vf,
                        ls::limitedLinearWeights(phi.internal, vf, gradVf, sch.coeff, m, g),
                        m, patches);
    }
    // linearUpwind derives from upwind, so the MATRIX is upwind's either way; the whole scheme is the
    // deferred source correction applied by the caller.
    return fvm::div(phi.internal, phi.boundary, vf, m, patches);
}

} // namespace

void readCoeffs(const FoamDict* ras, Coeffs& c)
{
    if (!ras) return;
    const FoamDict* d = ras->subDict("SpalartAllmarasCoeffs");
    if (!d) return;
    c.sigmaNut = d->scalarOr("sigmaNut", c.sigmaNut);
    c.kappa    = d->scalarOr("kappa",    c.kappa);
    c.Cb1      = d->scalarOr("Cb1",      c.Cb1);
    c.Cb2      = d->scalarOr("Cb2",      c.Cb2);
    c.Cw2      = d->scalarOr("Cw2",      c.Cw2);
    c.Cw3      = d->scalarOr("Cw3",      c.Cw3);
    c.Cv1      = d->scalarOr("Cv1",      c.Cv1);
    c.Cs       = d->scalarOr("Cs",       c.Cs);
    c.Ct3      = d->scalarOr("Ct3",      c.Ct3);
    c.Ct4      = d->scalarOr("Ct4",      c.Ct4);
    const std::string sw = d->wordOr("ft2", c.ft2 ? "true" : "false");
    c.ft2 = (sw == "true" || sw == "yes" || sw == "on" || sw == "1");
}

void correct(
    const GeometricField<vector>& U,
    GeometricField<scalar>&       nuTilda,
    GeometricField<scalar>&       nutField,
    const SurfaceScalarField&     phi,
    const std::vector<scalar>&    y,
    scalar                        nu,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches,
    scalar                        relaxNuTilda,
    scalar                        tol,
    scalar                        relTol,
    int                           maxIter,
    const Coeffs&                 co,
    const DivScheme&              sch,
    Residuals*                    res)
{
    const label nC = m.nCells();
    const scalar Cw1 = co.Cw1();
    const scalar kappa2 = co.kappa * co.kappa;
    const scalar Cv1cubed = co.Cv1 * co.Cv1 * co.Cv1;
    const scalar Cw3pow6 = std::pow(co.Cw3, 6.0);

    // chi, fv1, fv2, ft2
    std::vector<scalar> chi(nC), fv1(nC), fv2(nC), ft2(nC, 0.0);
    for (label c = 0; c < nC; ++c)
    {
        chi[c] = nuTilda.internal[c] / nu;
        const scalar chi3 = chi[c] * chi[c] * chi[c];
        fv1[c] = chi3 / (chi3 + Cv1cubed);
        fv2[c] = 1.0 - chi[c] / (1.0 + chi[c] * fv1[c]);
        if (co.ft2) ft2[c] = co.Ct3 * std::exp(-co.Ct4 * chi[c] * chi[c]);
    }

    // Stilda = max(Omega + fv2*nuTilda/sqr(kappa*dTilda), Cs*Omega), with dTilda = y for RAS SA.
    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    std::vector<scalar> Stilda(nC);
    for (label c = 0; c < nC; ++c)
    {
        const scalar Om = omegaOf(gradU[c]);
        const scalar kd = co.kappa * y[c];
        const scalar add = (kd > 0.0) ? fv2[c] * nuTilda.internal[c] / (kd * kd) : 0.0;
        Stilda[c] = std::fmax(Om + add, co.Cs * Om);
    }

    // r = min(nuTilda/(max(Stilda, SMALL)*sqr(kappa*dTilda)), 10), then g = r + Cw2*(r^6 - r) and
    // fw = g*((1 + Cw3^6)/(g^6 + Cw3^6))^(1/6). r's BOUNDARY is zero in OpenFOAM, but fw is only ever
    // evaluated on the internal field (it is a volScalarField::Internal), so only cells are built here.
    std::vector<scalar> fw(nC);
    for (label c = 0; c < nC; ++c)
    {
        const scalar kd = co.kappa * y[c];
        const scalar den = std::fmax(Stilda[c], SMALL_) * kd * kd;
        const scalar r = (den > 0.0) ? std::fmin(nuTilda.internal[c] / den, 10.0) : 10.0;
        const scalar r6 = std::pow(r, 6.0);
        const scalar gg = r + co.Cw2 * (r6 - r);
        const scalar g6 = std::pow(gg, 6.0);
        fw[c] = gg * std::pow((1.0 + Cw3pow6) / (g6 + Cw3pow6), 1.0 / 6.0);
    }

    // grad(nuTilda): the Cb2 source needs it, and linearUpwind's correction needs it too.
    std::vector<std::vector<scalar>> ntB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        ntB[pi] = nuTilda.boundary[pi]->value();
    }
    const std::vector<vector> gradNuTilda = fvc::gaussGrad(nuTilda.internal, ntB, m, g, patches);

    // DnuTildaEff = (nuTilda + nu)/sigmaNut. A volScalarField, so its BOUNDARY comes from nuTilda's own
    // boundary -- the same distinction that put 90% of the kEpsilon epsilon residual on one patch.
    std::vector<scalar> DEff(nC);
    for (label c = 0; c < nC; ++c)
    {
        DEff[c] = (nuTilda.internal[c] + nu) / co.sigmaNut;
    }
    SurfaceScalarField Df = fvc::interpolate(DEff, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            Df.boundary[pi][i] = (ntB[pi][i] + nu) / co.sigmaNut;
        }
    }

    // Cell-level term dump, same columns as the device's BRAE_DUMP_SA, so the two paths can be diffed
    // term by term rather than argued about.
    if (const char* dp = std::getenv("BRAE_DUMP_SA_CPP"))
    {
        static bool dumped = false;
        if (!dumped)
        {
            dumped = true;
            std::FILE* fp = std::fopen(dp, "w");
            std::fprintf(fp, "cell,nuTilda,y,Omega,Stilda,fw,gradNt2,P,D,nut\n");
            for (label c = 0; c < nC; ++c)
            {
                const scalar Om = omegaOf(gradU[c]);
                const scalar y2 = (y[c] * y[c] > 1e-300) ? y[c] * y[c] : 1e-300;
                const scalar g2 = magSqr(gradNuTilda[c]);
                const scalar P = co.Cb1 * Stilda[c] * nuTilda.internal[c] + (co.Cb2 / co.sigmaNut) * g2;
                const scalar D = Cw1 * fw[c] * nuTilda.internal[c] / y2;
                std::fprintf(fp, "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                             static_cast<int>(c), nuTilda.internal[c], y[c], Om, Stilda[c], fw[c],
                             g2, P, D, nuTilda.internal[c] * fv1[c]);
            }
            std::fclose(fp);
        }
    }

    FvScalarMatrix M = divWithScheme(phi, nuTilda, gradNuTilda, sch, m, g, patches);
    addEqual(M, fvm::laplacian(Df, nuTilda, m, g, patches), -1.0);

    const std::vector<scalar> divPhi =
        sch.bounded ? fvc::div(phi, m, g, patches) : std::vector<scalar>(nC, 0.0);

    for (label c = 0; c < nC; ++c)
    {
        const scalar V = g.V()[c];

        // - Cb2/sigmaNut*magSqr(grad(nuTilda)) sits on the LHS, so the source GAINS it.
        M.source[c] += (co.Cb2 / co.sigmaNut) * magSqr(gradNuTilda[c]) * V;

        // == Cb1*Stilda*nuTilda*(1 - ft2)
        M.source[c] += co.Cb1 * Stilda[c] * nuTilda.internal[c] * (1.0 - ft2[c]) * V;

        // - Sp((Cw1*fw - Cb1/kappa^2*ft2)*nuTilda/dTilda^2, nuTilda)
        const scalar d2 = y[c] * y[c];
        if (d2 > 0.0)
        {
            const scalar sp = (Cw1 * fw[c] - co.Cb1 / kappa2 * ft2[c]) * nuTilda.internal[c] / d2;
            M.diag[c] += sp * V;
        }

        // `bounded`: - Sp(fvc::div(phi), nuTilda)
        if (sch.bounded) M.diag[c] -= divPhi[c] * V;
    }

    // linearUpwind's deferred correction. The caller SUBTRACTS what linearUpwindCorrection returns --
    // see the sign note in fvm.cuh.
    if (sch.linearUpwind)
    {
        const std::vector<scalar> corr =
            fvm::linearUpwindCorrection<scalar, vector>(phi.internal, gradNuTilda, m, g);
        for (label c = 0; c < nC; ++c)
        {
            M.source[c] -= corr[c];
        }
    }

    relaxMatrix(M, nuTilda, m, patches, relaxNuTilda);
    const SolverPerformance p = pbicgstab(M, nuTilda.internal, m, patches, tol, relTol, maxIter);
    if (res) res->nuTilda = p.initialResidual;

    // bound(nuTilda_, 0) -- the lower bound is ZERO here, not the SMALL that k/epsilon/omega use.
    nuTilda.evaluateBoundary();
    bound(nuTilda, 0.0, m, g, patches);

    // correctNut: nut_ = nuTilda*fv1 is a FIELD assignment, so the boundary takes nuTilda's own boundary
    // -- and then correctBoundaryConditions() runs, which is NOT a no-op. A wall carrying
    // nutUSpaldingWallFunction OVERWRITES what the assignment just put there with the Spalding wall
    // viscosity, and nuTilda is fixedValue ZERO at a wall, so the assignment alone leaves nut_wall = 0.
    // OpenFOAM converges airFoil2D's wall nut to ~4.5e-03; taking the assignment's zero instead removes
    // the wall's entire eddy viscosity, and with it the wall shear -- 25% of this case's momentum
    // residual sat on those 78 faces.
    for (label c = 0; c < nC; ++c)
    {
        const scalar ch = nuTilda.internal[c] / nu;
        const scalar chi3 = ch * ch * ch;
        nutField.internal[c] = nuTilda.internal[c] * (chi3 / (chi3 + Cv1cubed));
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& nb = nuTilda.boundary[pi]->value();
        std::vector<scalar> vals(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const scalar ch = nb[i] / nu;
            const scalar chi3 = ch * ch * ch;
            vals[i] = nb[i] * (chi3 / (chi3 + Cv1cubed));
        }

        // correctBoundaryConditions(): a velocity-based wall function computes its own value.
        if (nutField.boundary[pi]->isNutUSpalding())
        {
            const std::vector<scalar>& seed = nutField.boundary[pi]->value();
            const std::vector<vector>& uw = U.boundary[pi]->value();
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const scalar magUp = mag(U.internal[c] - uw[i]);
                const scalar magGradU = magUp * patches[pi].deltaCoeffs[i];
                const scalar yw = (patches[pi].deltaCoeffs[i] > 0.0) ? 1.0 / patches[pi].deltaCoeffs[i] : 0.0;
                vals[i] = spaldingNutValue(magUp, magGradU, yw, nu, co.nutKappa, co.E,
                                           i < static_cast<label>(seed.size()) ? seed[i] : 0.0);
            }
        }
        nutField.boundary[pi]->setValue(vals);
    }
}

} // namespace SA
} // namespace cpu
} // namespace brae
