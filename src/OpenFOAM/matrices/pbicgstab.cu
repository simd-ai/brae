#include "pbicgstab.cuh"
#include <cmath>

namespace brae {

SolverPerformance pbicgstab(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter)
{
    refuseCoupledPatches(patches, "PBiCGStab");
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& upper = M.upper;
    const std::vector<scalar>& lower = M.lower;

    std::vector<scalar> diagC = M.diag, b = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            diagC[c] += M.internalCoeffs[pi][i];
            b[c]     += M.boundaryCoeffs[pi][i];
        }

    auto Amul = [&](const std::vector<scalar>& x, std::vector<scalar>& Ax)
    {
        for (label c = 0; c < nC; ++c) Ax[c] = diagC[c] * x[c];
        for (label f = 0; f < nIf; ++f)
        {
            Ax[nei[f]] += lower[f] * x[own[f]];
            Ax[own[f]] += upper[f] * x[nei[f]];
        }
    };
    auto sumMag = [&](const std::vector<scalar>& x)
    {
        scalar s = 0;
        for (label c = 0; c < nC; ++c) s += std::fabs(x[c]);
        return s;
    };
    auto sumProd = [&](const std::vector<scalar>& a, const std::vector<scalar>& c)
    {
        scalar s = 0;
        for (label i = 0; i < nC; ++i) s += a[i] * c[i];
        return s;
    };
    auto sumSqr = [&](const std::vector<scalar>& a)
    {
        scalar s = 0;
        for (label i = 0; i < nC; ++i) s += a[i] * a[i];
        return s;
    };

    // DILU reciprocal diagonal.
    std::vector<scalar> rD = diagC;
    for (label f = 0; f < nIf; ++f) rD[nei[f]] -= upper[f] * lower[f] / rD[own[f]];
    for (label c = 0; c < nC; ++c) rD[c] = 1.0 / rD[c];
    auto precondition = [&](std::vector<scalar>& wA, const std::vector<scalar>& rA)
    {
        for (label c = 0; c < nC; ++c) wA[c] = rD[c] * rA[c];
        for (label f = 0; f < nIf; ++f)      wA[nei[f]] -= rD[nei[f]] * lower[f] * wA[own[f]];
        for (label f = nIf - 1; f >= 0; --f) wA[own[f]] -= rD[own[f]] * upper[f] * wA[nei[f]];
    };

    std::vector<scalar> yA(nC, 0.0), rA(nC);
    Amul(psi, yA);
    for (label c = 0; c < nC; ++c) rA[c] = b[c] - yA[c];

    // normFactor (lduMatrixSolver::normFactor)
    std::vector<scalar> sumA(nC);
    for (label c = 0; c < nC; ++c) sumA[c] = diagC[c];
    for (label f = 0; f < nIf; ++f)
    {
        sumA[nei[f]] += lower[f];
        sumA[own[f]] += upper[f];
    }
    scalar xRef = 0.0;
    for (label c = 0; c < nC; ++c) xRef += psi[c];
    xRef /= nC;
    scalar normFactor = 0.0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar t = sumA[c] * xRef;
        normFactor += std::fabs(yA[c] - t) + std::fabs(b[c] - t);
    }
    normFactor += 1e-20;

    SolverPerformance perf;
    perf.initialResidual = sumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr)
    {
        return (fr < tolerance) || (relTol > 0.0 && fr < relTol * perf.initialResidual);
    };

    if (minIter > 0 || !converged(perf.finalResidual))
    {
        std::vector<scalar> pA(nC, 0.0), AyA(nC, 0.0), sA(nC), zA(nC), tA(nC);
        const std::vector<scalar> rA0 = rA;
        scalar rA0rA = 0, alpha = 0, omega = 0;
        int nIter = 0;
        do
        {
            const scalar rA0rAold = rA0rA;
            rA0rA = sumProd(rA0, rA);
            if (std::fabs(rA0rA) < 1e-300) break;

            if (nIter == 0) pA = rA;
            else
            {
                if (std::fabs(omega) < 1e-300) break;
                const scalar beta = (rA0rA / rA0rAold) * (alpha / omega);
                for (label c = 0; c < nC; ++c) pA[c] = rA[c] + beta * (pA[c] - omega * AyA[c]);
            }

            precondition(yA, pA);                 // yA = M^-1 pA
            Amul(yA, AyA);                         // AyA = A yA
            alpha = rA0rA / sumProd(rA0, AyA);
            for (label c = 0; c < nC; ++c) sA[c] = rA[c] - alpha * AyA[c];

            perf.finalResidual = sumMag(sA) / normFactor;
            if (nIter >= minIter && converged(perf.finalResidual))
            {
                for (label c = 0; c < nC; ++c) psi[c] += alpha * yA[c];
                ++nIter;
                perf.nIterations = nIter;
                return perf;
            }

            precondition(zA, sA);                 // zA = M^-1 sA
            Amul(zA, tA);                          // tA = A zA
            omega = sumProd(tA, sA) / sumSqr(tA);
            for (label c = 0; c < nC; ++c)
            {
                psi[c] += alpha * yA[c] + omega * zA[c];
                rA[c] = sA[c] - omega * tA[c];
            }
            perf.finalResidual = sumMag(rA) / normFactor;
            ++nIter;
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
        perf.nIterations = nIter;
    }
    return perf;
}

} // namespace brae
