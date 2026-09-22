// brae::smoothSolver -- see smooth_solver_cpp.cuh for the provenance and the three things to get right.
// The loops below follow OpenFOAM's files in order so the two can be read side by side.
#include "smooth_solver_cpp.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {

std::vector<label> lduOwnerStart(
    const std::vector<label>& owner,
    const std::vector<label>& nei,
    label nCells)
{
    const std::size_t nIf = nei.size();
    std::vector<label> ownStart(static_cast<std::size_t>(nCells) + 1, 0);
    for (std::size_t f = 0; f < nIf; ++f)
    {
        if (owner[f] >= nei[f] || (f > 0 && owner[f] < owner[f - 1]))
        {
            throw std::runtime_error(
                "brae smoothSolver: internal face " + std::to_string(f) + " breaks OpenFOAM's "
                "upper-triangular order (faces sorted by owner, owner < neighbour). The Gauss-Seidel "
                "smoothers walk the faces each cell OWNS through ownerStartAddr, which only exists "
                "for a mesh ordered that way; renumberMesh or checkMesh will say the same.");
        }
        ++ownStart[static_cast<std::size_t>(owner[f]) + 1];
    }
    for (label c = 0; c < nCells; ++c)
    {
        ownStart[static_cast<std::size_t>(c) + 1] += ownStart[static_cast<std::size_t>(c)];
    }
    return ownStart;
}

namespace {

void gaussSeidelSweeps(
    const std::vector<label>& ownStart,
    const std::vector<label>& nei,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    int nSweeps,
    bool symmetric,
    const FvScalarMatrix* M,
    const std::vector<FvPatch>* patches,
    const CoupledJumps* jumps);

} // namespace

void gaussSeidelSmoothFolded(
    const std::vector<label>& ownStart,
    const std::vector<label>& nei,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    int nSweeps,
    bool symmetric)
{
    gaussSeidelSweeps(ownStart, nei, diag, upper, lower, b, psi, nSweeps, symmetric, nullptr, nullptr, nullptr);
}

void gaussSeidelSmoothFolded(
    const std::vector<label>& ownStart,
    const std::vector<label>& nei,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    int nSweeps,
    bool symmetric,
    const FvScalarMatrix& M,
    const std::vector<FvPatch>& patches,
    const CoupledJumps* jumps)
{
    gaussSeidelSweeps(ownStart, nei, diag, upper, lower, b, psi, nSweeps, symmetric, &M, &patches, jumps);
}

namespace {

void gaussSeidelSweeps(
    const std::vector<label>& ownStart,
    const std::vector<label>& nei,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower,
    const std::vector<scalar>& b,
    std::vector<scalar>& psi,
    int nSweeps,
    bool symmetric,
    const FvScalarMatrix* M,
    const std::vector<FvPatch>* patches,
    const CoupledJumps* jumps)
{
    const label nCells = static_cast<label>(psi.size());
    std::vector<scalar> bPrime(static_cast<std::size_t>(nCells));

    for (int sweep = 0; sweep < nSweeps; ++sweep)
    {
        // bPrime = source, and then the coupled interfaces, Jacobi: bPrime += coeff*pnf from psi as the
        // sweep finds it (GaussSeidelSmoother.C, updateMatrixInterfaces with add = false, which the
        // patch field turns into an addition)
        bPrime = b;
        if (M && patches)
        {
            updateCoupledInterfaces(*M, *patches, psi, bPrime, scalar(1), true, jumps);
        }

        for (label celli = 0; celli < nCells; ++celli)
        {
            const label fStart = ownStart[static_cast<std::size_t>(celli)];
            const label fEnd = ownStart[static_cast<std::size_t>(celli) + 1];

            // Get the accumulated neighbour side
            scalar psii = bPrime[static_cast<std::size_t>(celli)];

            // Accumulate the owner product side
            for (label facei = fStart; facei < fEnd; ++facei)
            {
                psii -= upper[static_cast<std::size_t>(facei)]
                      * psi[static_cast<std::size_t>(nei[static_cast<std::size_t>(facei)])];
            }

            // Finish current psi
            psii /= diag[static_cast<std::size_t>(celli)];

            // Distribute the neighbour side using current psi
            for (label facei = fStart; facei < fEnd; ++facei)
            {
                bPrime[static_cast<std::size_t>(nei[static_cast<std::size_t>(facei)])] -=
                    lower[static_cast<std::size_t>(facei)]*psii;
            }

            psi[static_cast<std::size_t>(celli)] = psii;
        }

        // GaussSeidelSmoother.C stops here.
        if (!symmetric) continue;

        for (label celli = nCells - 1; celli >= 0; --celli)
        {
            const label fStart = ownStart[static_cast<std::size_t>(celli)];
            const label fEnd = ownStart[static_cast<std::size_t>(celli) + 1];

            // Get the accumulated neighbour side -- what the ascending walk LEFT there
            scalar psii = bPrime[static_cast<std::size_t>(celli)];

            // Accumulate the owner product side
            for (label facei = fStart; facei < fEnd; ++facei)
            {
                psii -= upper[static_cast<std::size_t>(facei)]
                      * psi[static_cast<std::size_t>(nei[static_cast<std::size_t>(facei)])];
            }

            // Note: do not need to distribute the neighbour side since these will not be revisited
            psii /= diag[static_cast<std::size_t>(celli)];
            psi[static_cast<std::size_t>(celli)] = psii;
        }
    }
}

} // namespace

SolverPerformance smoothSolver(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    bool symmetric,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter,
    int nSweeps,
    const CoupledJumps* jumps)
{
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    // fvMatrix::solve: addBoundaryDiag / addBoundarySource, scattered to the face cells.
    std::vector<scalar> diagC = M.diag;
    std::vector<scalar> b = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const std::size_t c = static_cast<std::size_t>(patches[pi].faceCells[i]);
            diagC[c] += M.internalCoeffs[pi][static_cast<std::size_t>(i)];
            // a coupled patch's boundaryCoeffs are interface coefficients, not a source
            if (!patches[pi].coupled)
            {
                b[c] += M.boundaryCoeffs[pi][static_cast<std::size_t>(i)];
            }
        }
    }

    const std::vector<label> ownInternal(own.begin(), own.begin() + nIf);
    const std::vector<label> ownStart = lduOwnerStart(ownInternal, nei, nC);

    SolverPerformance perf;

    // If nSweeps is negative do a fixed number of sweeps -- and report no residual at all.
    if (nSweeps < 0)
    {
        gaussSeidelSmoothFolded(ownStart, nei, diagC, M.upper, M.lower, b, psi, -nSweeps, symmetric, M, patches, jumps);
        perf.nIterations -= nSweeps;
        return perf;
    }

    auto amul = [&](
        const std::vector<scalar>& x,
        std::vector<scalar>& Ax)
    {
        for (label c = 0; c < nC; ++c)
        {
            Ax[static_cast<std::size_t>(c)] = diagC[static_cast<std::size_t>(c)]*x[static_cast<std::size_t>(c)];
        }
        for (label f = 0; f < nIf; ++f)
        {
            const std::size_t o = static_cast<std::size_t>(own[static_cast<std::size_t>(f)]);
            const std::size_t n = static_cast<std::size_t>(nei[static_cast<std::size_t>(f)]);
            Ax[n] += M.lower[static_cast<std::size_t>(f)]*x[o];
            Ax[o] += M.upper[static_cast<std::size_t>(f)]*x[n];
        }
        // the only operand is psi, the solution field: a jump applies
        updateCoupledInterfaces(M, patches, x, Ax, scalar(-1), true, jumps);
    };

    std::vector<scalar> Apsi(static_cast<std::size_t>(nC));
    amul(psi, Apsi);

    // lduMatrix::solver::normFactor: sum(|A.psi - sumA*avg(psi)| + |b - sumA*avg(psi)|) + small
    std::vector<scalar> sumA = diagC;
    for (label f = 0; f < nIf; ++f)
    {
        sumA[static_cast<std::size_t>(nei[static_cast<std::size_t>(f)])] += M.lower[static_cast<std::size_t>(f)];
        sumA[static_cast<std::size_t>(own[static_cast<std::size_t>(f)])] += M.upper[static_cast<std::size_t>(f)];
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
            sumA[static_cast<std::size_t>(patches[pi].faceCells[i])] -= M.boundaryCoeffs[pi][static_cast<std::size_t>(i)];
        }
    }
    scalar xRef = 0;
    for (label c = 0; c < nC; ++c)
    {
        xRef += psi[static_cast<std::size_t>(c)];
    }
    xRef /= nC;
    scalar normFactor = 0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar t = sumA[static_cast<std::size_t>(c)]*xRef;
        normFactor += std::fabs(Apsi[static_cast<std::size_t>(c)] - t) + std::fabs(b[static_cast<std::size_t>(c)] - t);
    }
    // solverPerformance::small_
    normFactor += 1e-20;

    // lduMatrix::residual (lduMatrixATmul.C:268-340), and NOT b - A.psi: it starts from
    // source - diag*psi and subtracts the off-diagonal terms face by face. Same number in exact
    // arithmetic, a different rounding -- and a symGaussSeidel solve of an upwind matrix ends at 1e-13,
    // where the rounding IS the residual.
    std::vector<scalar> rA(static_cast<std::size_t>(nC));
    auto residual = [&]()
    {
        for (label c = 0; c < nC; ++c)
        {
            rA[static_cast<std::size_t>(c)] =
                b[static_cast<std::size_t>(c)] - diagC[static_cast<std::size_t>(c)]*psi[static_cast<std::size_t>(c)];
        }
        for (label f = 0; f < nIf; ++f)
        {
            const std::size_t o = static_cast<std::size_t>(own[static_cast<std::size_t>(f)]);
            const std::size_t n = static_cast<std::size_t>(nei[static_cast<std::size_t>(f)]);
            rA[n] -= M.lower[static_cast<std::size_t>(f)]*psi[o];
            rA[o] -= M.upper[static_cast<std::size_t>(f)]*psi[n];
        }
        // lduMatrix::residual: the interfaces ADD, coeff*pnf
        updateCoupledInterfaces(M, patches, psi, rA, scalar(1), true, jumps);
        scalar s = 0;
        for (label c = 0; c < nC; ++c)
        {
            s += std::fabs(rA[static_cast<std::size_t>(c)]);
        }
        return s/normFactor;
    };
    // SolverPerformance::checkConvergence
    auto converged = [&](scalar fr)
    {
        return (fr < tolerance) || (relTol > scalar(1e-20) && fr < relTol*perf.initialResidual);
    };

    // THE INITIAL residual is source - Apsi, off the A.psi the normFactor was built from
    // (smoothSolver.C:122-150); only the loop's goes through lduMatrix::residual.
    {
        scalar s0 = 0;
        for (label c = 0; c < nC; ++c)
        {
            s0 += std::fabs(b[static_cast<std::size_t>(c)] - Apsi[static_cast<std::size_t>(c)]);
        }
        perf.initialResidual = s0/normFactor;
    }
    perf.finalResidual = perf.initialResidual;

    // Check convergence, solve if not converged
    if (minIter > 0 || !converged(perf.finalResidual))
    {
        // Smoothing loop
        do
        {
            gaussSeidelSmoothFolded(ownStart, nei, diagC, M.upper, M.lower, b, psi, nSweeps, symmetric, M, patches, jumps);
            perf.finalResidual = residual();
        } while
        (
            ((perf.nIterations += nSweeps) < maxIter && !converged(perf.finalResidual))
         || perf.nIterations < minIter
        );
    }
    return perf;
}

} // namespace brae
