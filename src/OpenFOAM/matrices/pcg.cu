#include "pcg.cuh"
#include <cmath>

namespace brae {

SolverPerformance pcg(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter,
    const CoupledJumps* jumps)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& upper = M.upper;
    const std::vector<scalar>& lower = M.lower;

    // Complete the matrix: addBoundaryDiag / addBoundarySource (scatter to faceCells).
    std::vector<scalar> diagC = M.diag;
    std::vector<scalar> b     = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            diagC[c] += M.internalCoeffs[pi][i];
            // a coupled patch's boundaryCoeffs are interface coefficients, not a source
            if (!patches[pi].coupled)
            {
                b[c] += M.boundaryCoeffs[pi][i];
            }
        }

    auto Amul = [&](const std::vector<scalar>& x, std::vector<scalar>& Ax, bool isField)
    {
        for (label c = 0; c < nC; ++c)
            Ax[c] = diagC[c] * x[c];
        for (label f = 0; f < nIf; ++f)
        {
            Ax[nei[f]] += lower[f] * x[own[f]];
            Ax[own[f]] += upper[f] * x[nei[f]];
        }
        updateCoupledInterfaces(M, patches, x, Ax, scalar(-1), isField, jumps);
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

    std::vector<scalar> pA(nC, 0.0), wA(nC, 0.0), rA(nC);

    Amul(psi, wA, true);
    for (label c = 0; c < nC; ++c) rA[c] = b[c] - wA[c];

    // normFactor = sum(|A.psi - sumA*xRef| + |b - sumA*xRef|) + small
    std::vector<scalar> sumA(nC);
    for (label c = 0; c < nC; ++c) sumA[c] = diagC[c];
    for (label f = 0; f < nIf; ++f)
    {
        sumA[nei[f]] += lower[f];
        sumA[own[f]] += upper[f];
    }
    // lduMatrix::sumA: sumA[faceCell] -= interfaceBouCoeffs
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!patches[pi].coupled)
        {
            continue;
        }
        for (label i = 0; i < patches[pi].size; ++i)
        {
            sumA[patches[pi].faceCells[i]] -= M.boundaryCoeffs[pi][i];
        }
    }
    scalar xRef = 0.0;
    for (label c = 0; c < nC; ++c) xRef += psi[c];
    xRef /= nC;
    scalar normFactor = 0.0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar t = sumA[c] * xRef;
        normFactor += std::fabs(wA[c] - t) + std::fabs(b[c] - t);
    }
    normFactor += 1e-20;   // solverPerformance::small_

    SolverPerformance perf;
    perf.initialResidual = sumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;

    // DIC reciprocal diagonal
    std::vector<scalar> rD = diagC;
    for (label f = 0; f < nIf; ++f) rD[nei[f]] -= upper[f] * upper[f] / rD[own[f]];
    for (label c = 0; c < nC; ++c) rD[c] = 1.0 / rD[c];

    auto converged = [&](scalar fr)
    {
        return (fr < tolerance) || (relTol > 0.0 && fr < relTol * perf.initialResidual);
    };

    scalar wArA = 1e300, wArAold;
    int nIter = 0;
    if (minIter > 0 || !converged(perf.finalResidual))
    {
        do
        {
            wArAold = wArA;

            // DIC precondition: wA = M^-1 rA  (forward then backward sweep)
            for (label c = 0; c < nC; ++c) wA[c] = rD[c] * rA[c];
            for (label f = 0; f < nIf; ++f)      wA[nei[f]] -= rD[nei[f]] * upper[f] * wA[own[f]];
            for (label f = nIf - 1; f >= 0; --f) wA[own[f]] -= rD[own[f]] * upper[f] * wA[nei[f]];

            wArA = sumProd(wA, rA);
            if (nIter == 0) pA = wA;
            else
            {
                const scalar beta = wArA / wArAold;
                for (label c = 0; c < nC; ++c) pA[c] = wA[c] + beta * pA[c];
            }

            Amul(pA, wA, false);
            const scalar wApA  = sumProd(wA, pA);
            const scalar alpha = wArA / wApA;
            for (label c = 0; c < nC; ++c)
            {
                psi[c] += alpha * pA[c];
                rA[c] -= alpha * wA[c];
            }

            perf.finalResidual = sumMag(rA) / normFactor;
            ++nIter;
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
    }
    perf.nIterations = nIter;
    return perf;
}

} // namespace brae
