#include "les_kEqn_cpp.cuh"
#include "bound_cpp.cuh"
#include "fv_matrix_ops.cuh"
#include "fvm.cuh"
#include "limitedSchemes_cpp.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace LESkEqn {

namespace {

const char* const WHO = "brae kEqn: ";

// tgradU() && devTwoSymm(tgradU()), in TensorI.H's own term order (devTwoSymm :788-798, Tensor &&
// SymmTensor :1266-1276)
scalar gradUDevTwoSymm(const tensor& t)
{
    const scalar sph = (scalar(2)/scalar(3))*(t.xx + t.yy + t.zz);
    const scalar sxx = 2*t.xx - sph;
    const scalar sxy = t.xy + t.yx;
    const scalar sxz = t.xz + t.zx;
    const scalar syy = 2*t.yy - sph;
    const scalar syz = t.yz + t.zy;
    const scalar szz = 2*t.zz - sph;
    return t.xx*sxx + t.xy*sxy + t.xz*sxz
         + t.yx*sxy + t.yy*syy + t.yz*syz
         + t.zx*sxz + t.zy*syz + t.zz*szz;
}

} // namespace


void correctNut(
    const GeometricField<scalar>& k,
    const std::vector<scalar>& delta,
    const Coeffs& co,
    GeometricField<scalar>& nut)
{
    if (delta.size() != k.internal.size())
    {
        throw std::runtime_error(std::string(WHO) + "delta and k differ in length.");
    }
    // kEqn.C:46-47: nut = Ck*sqrt(k)*delta; nut.correctBoundaryConditions()
    for (std::size_t c = 0; c < k.internal.size(); ++c)
    {
        nut.internal[c] = co.Ck*std::sqrt(k.internal[c])*delta[c];
    }
    nut.evaluateBoundary();
}


SolverPerformance correct(
    const GeometricField<vector>& U,
    GeometricField<scalar>& k,
    GeometricField<scalar>& nut,
    const SurfaceScalarField& phi,
    const std::vector<scalar>& nu,
    const std::vector<std::vector<scalar>>& nuBnd,
    const std::vector<scalar>& delta,
    scalar deltaT,
    const Coeffs& co,
    const Solve& solve,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    Taps* taps)
{
    if (!(deltaT > scalar(0)))
    {
        throw std::runtime_error(std::string(WHO) + "correct needs a positive deltaT.");
    }
    const label nC = m.nCells();
    const std::vector<scalar>& V = g.V();
    const std::vector<scalar> kOld = k.internal;

    // divU = fvc::div(fvc::absolute(phi, U)): a static mesh, so phi itself
    const std::vector<scalar> divU = fvc::div(phi, m, g, patches);

    // G = nut*(gradU && devTwoSymm(gradU)), with the nut the previous correctNut left
    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    std::vector<scalar> G(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        G[c] = nut.internal[c]*gradUDevTwoSymm(gradU[c]);
    }

    // k's patches learn the flux (inletOutlet) at the assembly, where fvMatrix's constructor runs
    // updateCoeffs
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        k.boundary[pi]->updateFromFlux(phi.boundary[pi]);
    }
    k.evaluateBoundary();

    // fvm::div(phi, k)
    FvScalarMatrix M;
    if (co.limitedLinear)
    {
        std::vector<std::vector<scalar>> kb(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            kb[pi] = k.boundary[pi]->value();
        }
        // LimitedScheme::calcLimiter takes fvc::grad(k) through the grad(k) scheme (Gauss linear here)
        const std::vector<vector> gradK = fvc::gaussGrad(k.internal, kb, m, g, patches);
        const std::vector<scalar> w = limitedSchemes::limitedLinearWeights(phi.internal, k, gradK,
                                                                             co.limitedLinearCoeff, m, g);
        M = fvm::div<scalar>(phi.internal, phi.boundary, k, w, m, patches);
    }
    else
    {
        M = fvm::div<scalar>(phi.internal, phi.boundary, k, m, patches);
    }
    if (taps)
    {
        taps->G = G;
        taps->divU = divU;
        taps->divUpper = M.upper;
        taps->divLower = M.lower;
    }

    // - fvm::laplacian(DkEff, k), DkEff = nut + nu interpolated linearly, its patch values nut_b + nu_b
    {
        std::vector<scalar> D(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            D[c] = nut.internal[c] + nu[c];
        }
        SurfaceScalarField Df = fvc::interpolate(D, m, g, patches);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].coupled)
            {
                continue;
            }
            const std::vector<scalar>& nb = nut.boundary[pi]->value();
            for (label i = 0; i < patches[pi].size; ++i)
            {
                Df.boundary[pi][i] = nb[i] + nuBnd[pi][i];
            }
        }
        if (taps)
        {
            taps->DkEfff = Df.internal;
        }
        FvScalarMatrix L = fvm::laplacian(Df, k, m, g, patches, co.correctedLaplacian);
        if (co.correctedLaplacian)
        {
            std::vector<std::vector<scalar>> kb(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                kb[pi] = k.boundary[pi]->value();
            }
            const std::vector<vector> gradK = fvc::gaussGrad(k.internal, kb, m, g, patches);
            const std::vector<scalar> corr = fvm::laplacianNonOrthSource<scalar, vector>(
                Df, k, gradK, m, g, patches, co.snGradLimitCoeff);
            for (label c = 0; c < nC; ++c)
            {
                L.source[c] -= corr[c];
            }
        }
        addEqual(M, L, scalar(-1));
    }

    const scalar rDeltaT = scalar(1)/deltaT;
    for (label c = 0; c < nC; ++c)
    {
        // fvm::ddt(k), Euler
        M.diag[c] += rDeltaT*V[c];
        M.source[c] += rDeltaT*kOld[c]*V[c];
        // == G
        M.source[c] += G[c]*V[c];
        // - fvm::SuSp((2/3)*divU, k)
        const scalar sp = (scalar(2)/scalar(3))*divU[c];
        M.diag[c] += V[c]*std::fmax(sp, scalar(0));
        M.source[c] -= V[c]*std::fmin(sp, scalar(0))*k.internal[c];
        // - fvm::Sp(Ce*sqrt(k)/delta, k), with k as it stands before the solve
        M.diag[c] += V[c]*(co.Ce*std::sqrt(k.internal[c])/delta[c]);
    }

    if (taps)
    {
        taps->diag = M.diag;
        taps->source = M.source;
    }
    if (solve.relaxOn)
    {
        relaxMatrix(M, k, m, patches, solve.relax);
    }
    if (!solve.which.smoothSolver)
    {
        throw std::runtime_error(
            std::string(WHO) + "k is solved with OpenFOAM's smoothSolver only; the case names another solver.");
    }
    const SolverPerformance perf = smoothSolver(M, k.internal, m, patches, solve.which.symmetric, solve.tol,
                                                solve.relTol, solve.maxIter, solve.minIter, solve.which.nSweeps);
    k.evaluateBoundary();
    if (taps)
    {
        taps->kSolved = k.internal;
    }
    bound(k, co.kMin, m, g, patches, "k");
    correctNut(k, delta, co, nut);
    return perf;
}

} // namespace LESkEqn
} // namespace cpu
} // namespace brae
