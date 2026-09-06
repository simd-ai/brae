// _cpp REFERENCE implementation -- see realizableKE_cpp.cuh for the OpenFOAM provenance.
#include "realizableKE_cpp.cuh"
#include "nut_wall_function.cuh"
#include "near_wall_dist.cuh"
#include "pbicgstab.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace brae {
namespace cpu {
namespace realizableKE {

namespace {

// devSymm(t) = symm(t) - (1/3)*tr(t)*I, returned as a full 3x3 for the contractions below.
inline void devSymm(const tensor& t, scalar S[9])
{
    const scalar tr3 = (t.xx + t.yy + t.zz) / 3.0;
    S[0] = t.xx - tr3;
    S[1] = 0.5*(t.xy + t.yx);
    S[2] = 0.5*(t.xz + t.zx);
    S[3] = S[1];
    S[4] = t.yy - tr3;
    S[5] = 0.5*(t.yz + t.zy);
    S[6] = S[2];
    S[7] = S[5];
    S[8] = t.zz - tr3;
}

} // namespace


std::vector<scalar> S2(const std::vector<tensor>& gradU)
{
    std::vector<scalar> out(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        scalar S[9];
        devSymm(gradU[c], S);
        scalar m = 0;
        for (int q = 0; q < 9; ++q) m += S[q]*S[q];
        out[c] = 2.0 * m;
    }
    return out;
}


std::vector<scalar> rCmu(const std::vector<tensor>& gradU,
                         const std::vector<scalar>& s2,
                         const std::vector<scalar>& k,
                         const std::vector<scalar>& eps,
                         const RealizableKECoeffs& co)
{
    constexpr scalar SMALL = 1.0e-15;
    std::vector<scalar> out(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        scalar S[9];
        devSymm(gradU[c], S);
        const scalar magS = std::sqrt(s2[c]);

        // ((S&S)&&S): the inner product of S*S with S, i.e. sum_ij (S*S)_ij S_ij.
        scalar SS[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                scalar v = 0;
                for (int q = 0; q < 3; ++q) v += S[i*3+q]*S[q*3+j];
                SS[i*3+j] = v;
            }
        scalar SSS = 0;
        for (int q = 0; q < 9; ++q) SSS += SS[q]*S[q];

        const scalar W = (2.0*std::sqrt(2.0)) * SSS / (magS*s2[c] + SMALL);
        scalar arg = std::sqrt(6.0)*W;
        arg = arg < -1.0 ? -1.0 : (arg > 1.0 ? 1.0 : arg);
        const scalar phis = (1.0/3.0)*std::acos(arg);
        const scalar As = std::sqrt(6.0)*std::cos(phis);

        // Us = sqrt(S2/2 + magSqr(skew(gradU))); skew(t) = 0.5*(t - t^T)
        const tensor& t = gradU[c];
        const scalar w01 = 0.5*(t.xy - t.yx), w02 = 0.5*(t.xz - t.zx), w12 = 0.5*(t.yz - t.zy);
        const scalar magSqrSkew = 2.0*(w01*w01 + w02*w02 + w12*w12);   // the skew part has zero diagonal
        const scalar Us = std::sqrt(0.5*s2[c] + magSqrSkew);

        out[c] = 1.0 / (co.A0 + As*Us*k[c]/eps[c]);
    }
    return out;
}


void correct(
    const GeometricField<vector>&  U,
    GeometricField<scalar>&        k,
    GeometricField<scalar>&        eps,
    GeometricField<scalar>&        nutField,
    const SurfaceScalarField&      phi,
    scalar                         nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    scalar                         relaxEps,
    scalar                         relaxK,
    scalar                         tol,
    scalar                         relTol,
    int                            maxIter,
    const RealizableKECoeffs&      co,
    RKEResiduals*                  res)
{
    const label nC = m.nCells();
    const scalar Cmu25 = std::pow(co.Cmu, 0.25), Cmu75 = std::pow(co.Cmu, 0.75);
    std::vector<scalar>& nutF = nutField.internal;

    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    const std::vector<scalar> s2 = S2(gradU);
    const std::vector<scalar> divU = fvc::div(phi, m, g, patches);

    // G = nut*(gradU && devTwoSymm(gradU)) -- the SAME production as kEpsilon (devTwoSymm, not devSymm).
    std::vector<scalar> G(nC);
    for (label c = 0; c < nC; ++c)
    {
        const tensor& t = gradU[c];
        const scalar tr = t.xx + t.yy + t.zz;
        const scalar d[9] = {
            2*t.xx - (2.0/3.0)*tr,  t.xy + t.yx,            t.xz + t.zx,
            t.yx + t.xy,            2*t.yy - (2.0/3.0)*tr,  t.yz + t.zy,
            t.zx + t.xz,            t.zy + t.yz,            2*t.zz - (2.0/3.0)*tr };
        const scalar gg[9] = { t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz };
        scalar s = 0;
        for (int q = 0; q < 9; ++q) s += gg[q]*d[q];
        G[c] = nutF[c] * s;
    }

    // epsilonWallFunction: the near-wall epsilon constraint and the G override. Cmu is the WALL
    // function's 0.09 here, not the model's variable rCmu.
    std::vector<label> nw(nC, 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (patches[pi].type == "wall")
            for (label i = 0; i < patches[pi].size; ++i) ++nw[patches[pi].faceCells[i]];

    const std::vector<std::vector<scalar>> yWall = nearWallDist(m, g, patches);
    std::vector<scalar> eps0(nC, 0.0), G0(nC, 0.0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        const FvPatch& wp = patches[pi];
        const std::vector<scalar>& yw = yWall[pi];
        const std::vector<scalar> nutw = nutkWallFunction(wp, yw, k.internal, nu, co.Cmu, co.kappa, co.E);
        const std::vector<vector>& Uw = U.boundary[pi]->value();
        for (label i = 0; i < wp.size; ++i)
        {
            const label c = wp.faceCells[i];
            const scalar w = 1.0/nw[c], kc = k.internal[c];
            const scalar magGradUw = mag((Uw[i] - U.internal[c]) * wp.deltaCoeffs[i]);
            eps0[c] += w * Cmu75 * std::pow(kc, 1.5) / (co.kappa * yw[i]);
            G0[c]   += w * (nutw[i] + nu) * magGradUw * Cmu25 * std::sqrt(kc) / (co.kappa * yw[i]);
        }
    }
    std::vector<label> wallCells;
    std::vector<scalar> epsVals;
    for (label c = 0; c < nC; ++c)
        if (nw[c] > 0)
        {
            G[c] = G0[c];
            eps.internal[c] = eps0[c];
            wallCells.push_back(c);
            epsVals.push_back(eps0[c]);
        }

    // ---- epsilon equation. NOT kEpsilon's with different constants -------------------------------
    //   == C1*magS*eps - Sp(C2*eps/(k + sqrt(nuLimited*eps)), eps),  C1 = max(eta/(5+eta), 0.43)
    {
        std::vector<scalar> DepsEff(nC);
        for (label c = 0; c < nC; ++c) DepsEff[c] = nutF[c]/co.sigmaEps + nu;
        const SurfaceScalarField Df = fvc::interpolate(DepsEff, m, g, patches);

        FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, eps, m, patches);
        addEqual(M, fvm::laplacian(Df, eps, m, g, patches), -1.0);
        const scalar nuLimited = std::fmax(nu, 0.0);
        for (label c = 0; c < nC; ++c)
        {
            const scalar V = g.V()[c];
            const scalar magS = std::sqrt(s2[c]);
            const scalar eta = magS * k.internal[c] / eps.internal[c];
            const scalar C1 = std::fmax(eta/(5.0 + eta), 0.43);
            M.source[c] += C1 * magS * eps.internal[c] * V;
            M.diag[c]   += co.C2 * eps.internal[c]
                         / (k.internal[c] + std::sqrt(nuLimited*eps.internal[c])) * V;
        }
        relaxMatrix(M, eps, m, patches, relaxEps);
        setValues(M, eps.internal, m, patches, wallCells, epsVals);
        const SolverPerformance pe = pbicgstab(M, eps.internal, m, patches, tol, relTol, maxIter);
        if (res) res->epsilon = pe.initialResidual;
        for (label c = 0; c < nC; ++c) eps.internal[c] = std::fmax(eps.internal[c], 1e-15);
        eps.evaluateBoundary();
    }

    // ---- k equation: kEpsilon's shape -----------------------------------------------------------
    {
        std::vector<scalar> DkEff(nC);
        for (label c = 0; c < nC; ++c) DkEff[c] = nutF[c]/co.sigmak + nu;
        const SurfaceScalarField Df = fvc::interpolate(DkEff, m, g, patches);

        FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, k, m, patches);
        addEqual(M, fvm::laplacian(Df, k, m, g, patches), -1.0);
        for (label c = 0; c < nC; ++c)
        {
            const scalar V = g.V()[c];
            M.source[c] += G[c] * V;
            const scalar sp = (2.0/3.0) * divU[c];
            M.diag[c]   += V * std::fmax(sp, 0.0);
            M.source[c] -= V * std::fmin(sp, 0.0) * k.internal[c];
            M.diag[c]   += eps.internal[c] / k.internal[c] * V;
        }
        relaxMatrix(M, k, m, patches, relaxK);
        const SolverPerformance pk = pbicgstab(M, k.internal, m, patches, tol, relTol, maxIter);
        if (res) res->k = pk.initialResidual;
        for (label c = 0; c < nC; ++c) k.internal[c] = std::fmax(k.internal[c], 1e-15);
        k.evaluateBoundary();
    }

    // ---- correctNut with the VARIABLE Cmu, from the NEW k/epsilon and the OLD gradU ---------------
    {
        const std::vector<scalar> rc = rCmu(gradU, s2, k.internal, eps.internal, co);
        for (label c = 0; c < nC; ++c) nutF[c] = rc[c] * k.internal[c]*k.internal[c] / eps.internal[c];
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        nutField.boundary[pi]->setValue(
            nutkWallFunction(patches[pi], yWall[pi], k.internal, nu, co.Cmu, co.kappa, co.E));
    }
}

} // namespace realizableKE
} // namespace cpu
} // namespace brae
