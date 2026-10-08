// OpenFOAM's PBiCG with DILU -- see pbicg.cuh.
#include "pbicg.cuh"
#include <cmath>
#include <stdexcept>

namespace brae {

SolverPerformance pbicgDILU(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter)
{
    refuseCoupledPatches(patches, "PBiCG");
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& upper = M.upper;
    const std::vector<scalar>& lower = M.lower;

    // fvMatrix::solveSegregated: addBoundaryDiag, addBoundarySource
    std::vector<scalar> diagC = M.diag;
    std::vector<scalar> b = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            diagC[c] += M.internalCoeffs[pi][i];
            b[c] += M.boundaryCoeffs[pi][i];
        }
    }

    // lduMatrix::Amul and Tmul (lduMatrixATmul.C)
    auto Amul = [&](std::vector<scalar>& Ax, const std::vector<scalar>& x)
    {
        for (label c = 0; c < nC; ++c)
        {
            Ax[c] = diagC[c]*x[c];
        }
        for (label f = 0; f < nIf; ++f)
        {
            Ax[nei[f]] += lower[f]*x[own[f]];
            Ax[own[f]] += upper[f]*x[nei[f]];
        }
    };
    auto Tmul = [&](std::vector<scalar>& Tx, const std::vector<scalar>& x)
    {
        for (label c = 0; c < nC; ++c)
        {
            Tx[c] = diagC[c]*x[c];
        }
        for (label f = 0; f < nIf; ++f)
        {
            Tx[nei[f]] += upper[f]*x[own[f]];
            Tx[own[f]] += lower[f]*x[nei[f]];
        }
    };
    auto sumMag = [&](const std::vector<scalar>& x)
    {
        scalar s = 0;
        for (label c = 0; c < nC; ++c)
        {
            s += std::fabs(x[c]);
        }
        return s;
    };
    auto sumProd = [&](const std::vector<scalar>& a, const std::vector<scalar>& c)
    {
        scalar s = 0;
        for (label i = 0; i < nC; ++i)
        {
            s += a[i]*c[i];
        }
        return s;
    };

    // DILUPreconditioner::calcReciprocalD
    std::vector<scalar> rD = diagC;
    for (label f = 0; f < nIf; ++f)
    {
        rD[nei[f]] -= upper[f]*lower[f]/rD[own[f]];
    }
    for (label c = 0; c < nC; ++c)
    {
        rD[c] = 1.0/rD[c];
    }
    // DILUPreconditioner::precondition
    auto precondition = [&](std::vector<scalar>& wA, const std::vector<scalar>& rA)
    {
        for (label c = 0; c < nC; ++c)
        {
            wA[c] = rD[c]*rA[c];
        }
        for (label f = 0; f < nIf; ++f)
        {
            wA[nei[f]] -= rD[nei[f]]*lower[f]*wA[own[f]];
        }
        for (label f = nIf - 1; f >= 0; --f)
        {
            wA[own[f]] -= rD[own[f]]*upper[f]*wA[nei[f]];
        }
    };
    // DILUPreconditioner::preconditionT: the same sweeps with upper and lower exchanged
    auto preconditionT = [&](std::vector<scalar>& wT, const std::vector<scalar>& rT)
    {
        for (label c = 0; c < nC; ++c)
        {
            wT[c] = rD[c]*rT[c];
        }
        for (label f = 0; f < nIf; ++f)
        {
            wT[nei[f]] -= rD[nei[f]]*upper[f]*wT[own[f]];
        }
        for (label f = nIf - 1; f >= 0; --f)
        {
            wT[own[f]] -= rD[own[f]]*lower[f]*wT[nei[f]];
        }
    };

    std::vector<scalar> pA(nC);
    std::vector<scalar> wA(nC);
    Amul(wA, psi);
    std::vector<scalar> rA(nC);
    for (label c = 0; c < nC; ++c)
    {
        rA[c] = b[c] - wA[c];
    }

    // lduMatrix::solver::normFactor: sumA*gAverage(psi), then sum(|Apsi - t| + |source - t|) + small
    std::vector<scalar> sumA(nC);
    for (label c = 0; c < nC; ++c)
    {
        sumA[c] = diagC[c];
    }
    for (label f = 0; f < nIf; ++f)
    {
        sumA[nei[f]] += lower[f];
        sumA[own[f]] += upper[f];
    }
    scalar xRef = 0;
    for (label c = 0; c < nC; ++c)
    {
        xRef += psi[c];
    }
    xRef /= nC;
    scalar normFactor = 0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar t = sumA[c]*xRef;
        normFactor += std::fabs(wA[c] - t) + std::fabs(b[c] - t);
    }
    normFactor += 1e-20;

    SolverPerformance perf;
    perf.initialResidual = sumMag(rA)/normFactor;
    perf.finalResidual = perf.initialResidual;
    // solverPerformance::checkConvergence
    auto converged = [&](scalar fr)
    {
        return (fr < tolerance) || (relTol > 1e-20 && fr < relTol*perf.initialResidual);
    };

    if (minIter > 0 || !converged(perf.finalResidual))
    {
        std::vector<scalar> pT(nC, scalar(0));
        std::vector<scalar> wT(nC);
        Tmul(wT, psi);
        std::vector<scalar> rT(nC);
        for (label c = 0; c < nC; ++c)
        {
            rT[c] = b[c] - wT[c];
        }
        scalar wArT = 0;
        int nIter = 0;
        do
        {
            const scalar wArTold = wArT;
            precondition(wA, rA);
            preconditionT(wT, rT);
            wArT = sumProd(wA, rT);
            if (nIter == 0)
            {
                for (label c = 0; c < nC; ++c)
                {
                    pA[c] = wA[c];
                    pT[c] = wT[c];
                }
            }
            else
            {
                const scalar beta = wArT/wArTold;
                for (label c = 0; c < nC; ++c)
                {
                    pA[c] = wA[c] + beta*pA[c];
                    pT[c] = wT[c] + beta*pT[c];
                }
            }
            Amul(wA, pA);
            Tmul(wT, pT);
            const scalar wApT = sumProd(wA, pT);
            // solverPerformance::checkSingularity: |wApT|/normFactor < VSMALL
            if (std::fabs(wApT)/normFactor < 1e-300)
            {
                break;
            }
            const scalar alpha = wArT/wApT;
            for (label c = 0; c < nC; ++c)
            {
                psi[c] += alpha*pA[c];
                rA[c] -= alpha*wA[c];
                rT[c] -= alpha*wT[c];
            }
            perf.finalResidual = sumMag(rA)/normFactor;
        } while ((++nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
        perf.nIterations = nIter;
    }
    return perf;
}

} // namespace brae
