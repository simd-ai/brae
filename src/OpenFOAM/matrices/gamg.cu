#include "gamg.cuh"
#include <algorithm>
#include <cmath>
#include <map>

namespace brae {
namespace {

// One multigrid level: lduMatrix in upper-triangular addressing + GS ownerStart, plus the
// cell restriction to the next coarser level.
struct Level
{
    label nCells = 0, nFaces = 0;
    std::vector<label>  lowerAddr, upperAddr;   // owner, neighbour (size nFaces)
    std::vector<label>  ownerStart;             // size nCells+1 (GaussSeidel traversal)
    std::vector<scalar> diag, upper, lower;
    std::vector<scalar> faceWeights;            // for agglomeration (faceAreaPair = magSf)
    std::vector<label>  restrict;               // this-level cell -> next-coarser cell
    label               nCoarse = 0;
};

std::vector<label> buildOwnerStart(const std::vector<label>& lowerAddr, label nCells)
{
    std::vector<label> os(nCells + 1, 0);
    for (label f = 0; f < (label)lowerAddr.size(); ++f) os[lowerAddr[f] + 1]++;
    for (label c = 0; c < nCells; ++c) os[c + 1] += os[c];
    return os;
}

void Amul(const Level& L, const std::vector<scalar>& x, std::vector<scalar>& Ax)
{
    for (label c = 0; c < L.nCells; ++c) Ax[c] = L.diag[c] * x[c];
    for (label f = 0; f < L.nFaces; ++f)
    {
        Ax[L.upperAddr[f]] += L.lower[f] * x[L.lowerAddr[f]];
        Ax[L.lowerAddr[f]] += L.upper[f] * x[L.upperAddr[f]];
    }
}

// pairGAMGAgglomerate: greedy pairing by largest face weight, alternating sweep direction.
std::vector<label> agglomeratePair(const Level& L, bool forward, label& nCoarse)
{
    const label nF = L.nCells;
    std::vector<label> nNbr(nF, 0);
    for (label f = 0; f < L.nFaces; ++f)
    {
        nNbr[L.upperAddr[f]]++;
        nNbr[L.lowerAddr[f]]++;
    }
    std::vector<label> off(nF + 1, 0);
    for (label c = 0; c < nF; ++c) off[c + 1] = off[c] + nNbr[c];
    std::vector<label> cellFaces(off[nF]);
    std::fill(nNbr.begin(), nNbr.end(), 0);
    for (label f = 0; f < L.nFaces; ++f) cellFaces[off[L.upperAddr[f]] + nNbr[L.upperAddr[f]]++] = f;
    for (label f = 0; f < L.nFaces; ++f) cellFaces[off[L.lowerAddr[f]] + nNbr[L.lowerAddr[f]]++] = f;

    std::vector<label> map(nF, -1);
    nCoarse = 0;
    for (label cf = 0; cf < nF; ++cf)
    {
        const label c = forward ? cf : nF - cf - 1;
        if (map[c] >= 0) continue;
        label matchFace = -1;
        scalar maxW = -1e300;
        for (label o = off[c]; o < off[c + 1]; ++o)
        {
            const label f = cellFaces[o];
            if (map[L.upperAddr[f]] < 0 && map[L.lowerAddr[f]] < 0 && L.faceWeights[f] > maxW)
            {
                matchFace = f;
                maxW = L.faceWeights[f];
            }
        }
        if (matchFace >= 0)
        {
            map[L.upperAddr[matchFace]] = nCoarse;
            map[L.lowerAddr[matchFace]] = nCoarse;
            ++nCoarse;
        }
        else
        {
            label cm = -1;
            scalar cMax = -1e300;
            for (label o = off[c]; o < off[c + 1]; ++o)
            {
                const label f = cellFaces[o];
                if (L.faceWeights[f] > cMax)
                {
                    cm = f;
                    cMax = L.faceWeights[f];
                }
            }
            if (cm >= 0) map[c] = std::max(map[L.upperAddr[cm]], map[L.lowerAddr[cm]]);
        }
    }
    for (label cf = 0; cf < nF; ++cf)
    {
        const label c = forward ? cf : nF - cf - 1;
        if (map[c] < 0) map[c] = nCoarse++;
    }
    if (!forward)
    {
        --nCoarse;
        for (auto& v : map) v = nCoarse - v;
        ++nCoarse;
    }
    return map;
}

// Build the next coarser level (addressing + agglomerated matrix + restricted face weights).
Level coarsen(const Level& F)
{
    Level C;
    C.nCells = F.nCoarse;
    const std::vector<label>& map = F.restrict;

    // Coarse faces = unique (min,max) coarse-cell pairs. Index them in SORTED order (std::map
    // iterates sorted) so lowerAddr is non-decreasing -> upper-triangular + valid ownerStart/GS.
    std::map<std::pair<label, label>, label> pairIdx;
    for (label f = 0; f < F.nFaces; ++f)
    {
        const label co = map[F.lowerAddr[f]], cn = map[F.upperAddr[f]];
        if (co != cn) pairIdx[{std::min(co, cn), std::max(co, cn)}] = 0;
    }
    {
        label idx = 0;
        for (auto& kv : pairIdx) kv.second = idx++;
    }
    C.nFaces = (label)pairIdx.size();
    C.lowerAddr.resize(C.nFaces);
    C.upperAddr.resize(C.nFaces);
    for (const auto& kv : pairIdx)
    {
        C.lowerAddr[kv.second] = kv.first.first;
        C.upperAddr[kv.second] = kv.first.second;
    }

    std::vector<label> faceRestrict(F.nFaces);   // >=0: coarse face; <0: -1-coarseCell (internal)
    std::vector<char>  faceFlip(F.nFaces, 0);
    for (label f = 0; f < F.nFaces; ++f)
    {
        const label co = map[F.lowerAddr[f]], cn = map[F.upperAddr[f]];
        if (co == cn) faceRestrict[f] = -1 - co;
        else
        {
            faceRestrict[f] = pairIdx[{std::min(co, cn), std::max(co, cn)}];
            faceFlip[f] = (co > cn) ? 1 : 0;
        }
    }

    C.diag.assign(C.nCells, 0.0);
    C.upper.assign(C.nFaces, 0.0);
    C.lower.assign(C.nFaces, 0.0);
    C.faceWeights.assign(C.nFaces, 0.0);
    for (label c = 0; c < F.nCells; ++c) C.diag[map[c]] += F.diag[c];
    for (label f = 0; f < F.nFaces; ++f)
    {
        const label cf = faceRestrict[f];
        if (cf >= 0)
        {
            if (!faceFlip[f]) { C.upper[cf] += F.upper[f]; C.lower[cf] += F.lower[f]; }
            else              { C.upper[cf] += F.lower[f]; C.lower[cf] += F.upper[f]; }
            C.faceWeights[cf] += F.faceWeights[f];
        }
        else
        {
            C.diag[-1 - cf] += F.upper[f] + F.lower[f];
        }
    }
    C.ownerStart = buildOwnerStart(C.lowerAddr, C.nCells);
    return C;
}

void restrictField(std::vector<scalar>& coarse, const std::vector<scalar>& fine, const std::vector<label>& map, label nCoarse)
{
    coarse.assign(nCoarse, 0.0);
    for (label c = 0; c < (label)fine.size(); ++c) coarse[map[c]] += fine[c];
}
void prolongField(std::vector<scalar>& fine, const std::vector<scalar>& coarse, const std::vector<label>& map)
{
    fine.resize(map.size());
    for (label c = 0; c < (label)map.size(); ++c) fine[c] = coarse[map[c]];   // injection
}

// GaussSeidel forward sweep (OF GaussSeidelSmoother): bPrime=source, per cell subtract owned
// upper contributions, divide by diag, propagate lower to neighbours.
void gaussSeidel(const Level& L, std::vector<scalar>& psi, const std::vector<scalar>& source, int nSweeps)
{
    std::vector<scalar> bPrime(L.nCells);
    for (int s = 0; s < nSweeps; ++s)
    {
        bPrime = source;
        label fEnd = L.ownerStart[0];
        for (label c = 0; c < L.nCells; ++c)
        {
            const label fStart = fEnd;
            fEnd = L.ownerStart[c + 1];
            scalar psii = bPrime[c];
            for (label fp = fStart; fp < fEnd; ++fp) psii -= L.upper[fp] * psi[L.upperAddr[fp]];
            psii /= L.diag[c];
            for (label fp = fStart; fp < fEnd; ++fp) bPrime[L.upperAddr[fp]] -= L.lower[fp] * psii;
            psi[c] = psii;
        }
    }
}

void scaleCorrection(const Level& L, std::vector<scalar>& field, const std::vector<scalar>& source)
{
    std::vector<scalar> Acf(L.nCells);
    Amul(L, field, Acf);
    scalar num = 0.0, den = 0.0;
    for (label c = 0; c < L.nCells; ++c)
    {
        num += field[c] * source[c];
        den += field[c] * Acf[c];
    }
    const scalar sf = num / ((den >= 0 ? std::fmax(den, 1e-300) : std::fmin(den, -1e-300)));
    for (label c = 0; c < L.nCells; ++c) field[c] = sf * field[c] + (source[c] - sf * Acf[c]) / L.diag[c];
}

// Dense Gaussian elimination for the (small) coarsest level.
void denseSolve(const Level& L, std::vector<scalar>& x, const std::vector<scalar>& b)
{
    const label n = L.nCells;
    std::vector<scalar> A(n * n, 0.0), rhs(b);
    for (label c = 0; c < n; ++c) A[c * n + c] = L.diag[c];
    for (label f = 0; f < L.nFaces; ++f)
    {
        A[L.lowerAddr[f] * n + L.upperAddr[f]] += L.upper[f];
        A[L.upperAddr[f] * n + L.lowerAddr[f]] += L.lower[f];
    }
    for (label k = 0; k < n; ++k)
    {
        label piv = k;
        scalar best = std::fabs(A[k * n + k]);
        for (label i = k + 1; i < n; ++i)
            if (std::fabs(A[i * n + k]) > best)
            {
                best = std::fabs(A[i * n + k]);
                piv = i;
            }
        if (piv != k)
        {
            for (label j = 0; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]);
            std::swap(rhs[k], rhs[piv]);
        }
        const scalar dkk = A[k * n + k];
        for (label i = k + 1; i < n; ++i)
        {
            const scalar fct = A[i * n + k] / dkk;
            if (fct == 0.0) continue;
            for (label j = k; j < n; ++j) A[i * n + j] -= fct * A[k * n + j];
            rhs[i] -= fct * rhs[k];
        }
    }
    x.assign(n, 0.0);
    for (label i = n - 1; i >= 0; --i)
    {
        scalar s = rhs[i];
        for (label j = i + 1; j < n; ++j) s -= A[i * n + j] * x[j];
        x[i] = s / A[i * n + i];
    }
}

} // namespace

SolverPerformance gamg(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int nCellsInCoarsest,
    int nPreSweeps,
    int nPostSweeps,
    int nFinestSweeps,
    int minIter)
{
    (void)nPreSweeps;
    const label nC = m.nCells();

    // Finest level (level 0): completed matrix (diag/source include boundary), faceWeights=magSf.
    std::vector<Level> levels(1);
    levels.reserve(64);            // keep references valid across coarsening push_backs
    Level& L0 = levels[0];
    L0.nCells = nC;
    L0.nFaces = m.nInternalFaces();
    L0.lowerAddr.assign(m.owner().begin(), m.owner().begin() + L0.nFaces);
    L0.upperAddr = m.neighbour();
    L0.diag = M.diag;
    L0.upper = M.upper;
    L0.lower = M.lower;
    std::vector<scalar> b = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            L0.diag[c] += M.internalCoeffs[pi][i];
            b[c]        += M.boundaryCoeffs[pi][i];
        }
    L0.ownerStart = buildOwnerStart(L0.lowerAddr, nC);
    L0.faceWeights.assign(L0.nFaces, 0.0);
    for (label f = 0; f < L0.nFaces; ++f) L0.faceWeights[f] = g.magSf()[f];

    // Build the agglomeration hierarchy.
    bool forward = true;
    for (int lev = 0; lev < 50; ++lev)
    {
        label nCoarse = 0;
        std::vector<label> map = agglomeratePair(levels[lev], forward, nCoarse);
        forward = !forward;
        if (nCoarse <= nCellsInCoarsest || nCoarse >= levels[lev].nCells || nCoarse < 2) break;
        levels[lev].restrict = std::move(map);
        levels[lev].nCoarse  = nCoarse;
        levels.push_back(coarsen(levels[lev]));
    }
    const int coarsest = (int)levels.size() - 1;

    auto sumMag = [&](const std::vector<scalar>& x)
    {
        scalar s = 0;
        for (scalar v : x) s += std::fabs(v);
        return s;
    };

    // normFactor (as PCG / lduMatrixSolver::normFactor).
    std::vector<scalar> Apsi(nC);
    Amul(L0, psi, Apsi);
    std::vector<scalar> sumA(nC);
    for (label c = 0; c < nC; ++c) sumA[c] = L0.diag[c];
    for (label f = 0; f < L0.nFaces; ++f)
    {
        sumA[L0.upperAddr[f]] += L0.lower[f];
        sumA[L0.lowerAddr[f]] += L0.upper[f];
    }
    scalar xRef = 0.0;
    for (label c = 0; c < nC; ++c) xRef += psi[c];
    xRef /= nC;
    scalar normFactor = 0.0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar t = sumA[c] * xRef;
        normFactor += std::fabs(Apsi[c] - t) + std::fabs(b[c] - t);
    }
    normFactor += 1e-20;

    std::vector<scalar> finestResidual(nC);
    for (label c = 0; c < nC; ++c) finestResidual[c] = b[c] - Apsi[c];

    SolverPerformance perf;
    perf.initialResidual = sumMag(finestResidual) / normFactor;
    perf.finalResidual   = perf.initialResidual;

    auto converged = [&](scalar fr)
    {
        return (fr < tolerance) || (relTol > 0.0 && fr < relTol * perf.initialResidual);
    };

    if (minIter > 0 || !converged(perf.finalResidual))
    {
        std::vector<std::vector<scalar>> cSource(levels.size()), cCorr(levels.size());
        int nIter = 0;
        do
        {
            // Restrict residual down.
            restrictField(cSource[1], finestResidual, levels[0].restrict, levels[1].nCells);
            for (int k = 1; k < coarsest; ++k)
                restrictField(cSource[k + 1], cSource[k], levels[k].restrict, levels[k + 1].nCells);

            // Coarsest direct solve.
            denseSolve(levels[coarsest], cCorr[coarsest], cSource[coarsest]);

            // Prolong + scale + post-smooth up.
            for (int k = coarsest - 1; k >= 1; --k)
            {
                prolongField(cCorr[k], cCorr[k + 1], levels[k].restrict);
                scaleCorrection(levels[k], cCorr[k], cSource[k]);
                gaussSeidel(levels[k], cCorr[k], cSource[k], nPostSweeps);
            }

            // Finest correction.
            std::vector<scalar> finestCorr;
            prolongField(finestCorr, cCorr[1], levels[0].restrict);
            scaleCorrection(levels[0], finestCorr, finestResidual);
            for (label c = 0; c < nC; ++c) psi[c] += finestCorr[c];
            gaussSeidel(levels[0], psi, b, nFinestSweeps);

            // Residual + convergence.
            Amul(L0, psi, Apsi);
            for (label c = 0; c < nC; ++c) finestResidual[c] = b[c] - Apsi[c];
            perf.finalResidual = sumMag(finestResidual) / normFactor;
            ++nIter;
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
        perf.nIterations = nIter;
    }
    return perf;
}

} // namespace brae
