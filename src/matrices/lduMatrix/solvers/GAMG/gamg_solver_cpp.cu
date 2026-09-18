#include "gamg_solver_cpp.cuh"
#include "foam_dict.cuh"
#include "smooth_solver_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brae {

namespace {

// solverPerformance::small_ and vsmall_
constexpr scalar perfSmall = 1e-20;
constexpr scalar perfVSmall = 1e-300;
// SolverPerformance.H: great_
constexpr scalar perfGreat = 1e+20;

// a symmetric lduMatrix on one level's addressing
struct LduLevel
{
    const GamgLduAddressing* addr = nullptr;
    std::vector<scalar> diag;
    std::vector<scalar> upper;
};

// lduMatrix::Amul, symmetric and with no interfaces
void amul(
    const LduLevel& A,
    const std::vector<scalar>& psi,
    std::vector<scalar>& Apsi)
{
    const std::vector<label>& u = A.addr->upperAddr;
    const std::vector<label>& l = A.addr->lowerAddr;
    const std::size_t nCells = A.diag.size();
    for (std::size_t cell = 0; cell < nCells; ++cell)
    {
        Apsi[cell] = A.diag[cell]*psi[cell];
    }
    for (std::size_t face = 0; face < A.upper.size(); ++face)
    {
        const std::size_t uf = static_cast<std::size_t>(u[face]);
        const std::size_t lf = static_cast<std::size_t>(l[face]);
        Apsi[uf] += A.upper[face]*psi[lf];
        Apsi[lf] += A.upper[face]*psi[uf];
    }
}

// lduMatrix::residual. rA may be the source itself, as the pre-smoothing branch calls it.
void residual(
    const LduLevel& A,
    std::vector<scalar>& rA,
    const std::vector<scalar>& psi,
    const std::vector<scalar>& source)
{
    const std::vector<label>& u = A.addr->upperAddr;
    const std::vector<label>& l = A.addr->lowerAddr;
    const std::size_t nCells = A.diag.size();
    for (std::size_t cell = 0; cell < nCells; ++cell)
    {
        rA[cell] = source[cell] - A.diag[cell]*psi[cell];
    }
    for (std::size_t face = 0; face < A.upper.size(); ++face)
    {
        const std::size_t uf = static_cast<std::size_t>(u[face]);
        const std::size_t lf = static_cast<std::size_t>(l[face]);
        rA[uf] -= A.upper[face]*psi[lf];
        rA[lf] -= A.upper[face]*psi[uf];
    }
}

// lduMatrix::solver::normFactor, L1 scaled
scalar normFactor(
    const LduLevel& A,
    const std::vector<scalar>& psi,
    const std::vector<scalar>& source,
    const std::vector<scalar>& Apsi,
    std::vector<scalar>& tmpField)
{
    const std::vector<label>& u = A.addr->upperAddr;
    const std::vector<label>& l = A.addr->lowerAddr;
    const std::size_t nCells = A.diag.size();
    for (std::size_t cell = 0; cell < nCells; ++cell)
    {
        tmpField[cell] = A.diag[cell];
    }
    for (std::size_t face = 0; face < A.upper.size(); ++face)
    {
        tmpField[static_cast<std::size_t>(u[face])] += A.upper[face];
        tmpField[static_cast<std::size_t>(l[face])] += A.upper[face];
    }
    scalar average = 0;
    for (std::size_t cell = 0; cell < nCells; ++cell)
    {
        average += psi[cell];
    }
    average /= static_cast<scalar>(nCells);
    scalar sum = 0;
    for (std::size_t cell = 0; cell < nCells; ++cell)
    {
        const scalar t = tmpField[cell]*average;
        sum += std::fabs(Apsi[cell] - t) + std::fabs(source[cell] - t);
    }
    return sum + perfSmall;
}

scalar sumMag(const std::vector<scalar>& x)
{
    scalar s = 0;
    for (const scalar v : x)
    {
        s += std::fabs(v);
    }
    return s;
}

scalar sumProd(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    scalar s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        s += a[i]*b[i];
    }
    return s;
}

// SolverPerformance::checkConvergence
bool converged(
    const SolverPerformance& perf,
    scalar tolerance,
    scalar relTol)
{
    return perf.finalResidual < tolerance
        || (relTol > perfSmall && perf.finalResidual < relTol*perf.initialResidual);
}

// DICPreconditioner::calcReciprocalD
std::vector<scalar> dicReciprocalD(const LduLevel& A)
{
    const std::vector<label>& u = A.addr->upperAddr;
    const std::vector<label>& l = A.addr->lowerAddr;
    std::vector<scalar> rD = A.diag;
    for (std::size_t face = 0; face < A.upper.size(); ++face)
    {
        rD[static_cast<std::size_t>(u[face])] -=
            A.upper[face]*A.upper[face]/rD[static_cast<std::size_t>(l[face])];
    }
    for (scalar& v : rD)
    {
        v = 1.0/v;
    }
    return rD;
}

// the two substitution passes DICSmoother::smooth and DICPreconditioner::precondition share,
// on a field that already holds rD*r
void dicSubstitute(
    const LduLevel& A,
    const std::vector<scalar>& rD,
    std::vector<scalar>& w)
{
    const std::vector<label>& u = A.addr->upperAddr;
    const std::vector<label>& l = A.addr->lowerAddr;
    const std::size_t nFaces = A.upper.size();
    for (std::size_t face = 0; face < nFaces; ++face)
    {
        const std::size_t uf = static_cast<std::size_t>(u[face]);
        w[uf] -= rD[uf]*A.upper[face]*w[static_cast<std::size_t>(l[face])];
    }
    for (std::size_t face = nFaces; face-- > 0;)
    {
        const std::size_t lf = static_cast<std::size_t>(l[face]);
        w[lf] -= rD[lf]*A.upper[face]*w[static_cast<std::size_t>(u[face])];
    }
}

// lduMatrix::smoother::New on one level: DIC, a Gauss-Seidel, or DIC's sweeps then Gauss-Seidel's
struct Smoother
{
    const LduLevel* A = nullptr;
    bool dic = false;
    bool gaussSeidel = false;
    bool symGaussSeidel = false;
    std::vector<scalar> rD;
    std::vector<label> ownStart;

    Smoother(
        const LduLevel& level,
        const std::string& name)
    :
        A(&level)
    {
        dic = (name == "DIC" || name == "DICGaussSeidel");
        gaussSeidel = (name == "GaussSeidel" || name == "DICGaussSeidel");
        symGaussSeidel = (name == "symGaussSeidel");
        if (dic)
        {
            rD = dicReciprocalD(level);
        }
        if (gaussSeidel || symGaussSeidel)
        {
            ownStart = lduOwnerStart(level.addr->lowerAddr, level.addr->upperAddr, level.addr->nCells);
        }
    }

    void smooth(
        std::vector<scalar>& psi,
        const std::vector<scalar>& source,
        int nSweeps) const
    {
        if (dic)
        {
            // DICSmoother::smooth
            std::vector<scalar> rA(rD.size());
            for (int sweep = 0; sweep < nSweeps; ++sweep)
            {
                residual(*A, rA, psi, source);
                for (std::size_t i = 0; i < rA.size(); ++i)
                {
                    rA[i] *= rD[i];
                }
                dicSubstitute(*A, rD, rA);
                for (std::size_t i = 0; i < rA.size(); ++i)
                {
                    psi[i] += rA[i];
                }
            }
        }
        if (gaussSeidel || symGaussSeidel)
        {
            gaussSeidelSmoothFolded(
                ownStart,
                A->addr->upperAddr,
                A->diag,
                A->upper,
                A->upper,
                source,
                psi,
                nSweeps,
                symGaussSeidel);
        }
    }
};

// PCG::scalarSolve with DICPreconditioner, as GAMGSolver builds it for the coarsest level:
// `solver PCG; preconditioner DIC;` and the GAMG's tolerance and relTol, every other control default
SolverPerformance pcgDic(
    const LduLevel& A,
    std::vector<scalar>& psi,
    const std::vector<scalar>& source,
    scalar tolerance,
    scalar relTol)
{
    const int maxIter = 1000;
    const std::size_t nCells = psi.size();
    std::vector<scalar> pA(nCells);
    std::vector<scalar> wA(nCells);
    scalar wArA = perfGreat;
    scalar wArAold = wArA;

    amul(A, psi, wA);
    std::vector<scalar> rA(nCells);
    for (std::size_t cell = 0; cell < nCells; ++cell)
    {
        rA[cell] = source[cell] - wA[cell];
    }
    const scalar nf = normFactor(A, psi, source, wA, pA);

    SolverPerformance perf;
    perf.initialResidual = sumMag(rA)/nf;
    perf.finalResidual = perf.initialResidual;
    if (converged(perf, tolerance, relTol)) return perf;

    const std::vector<scalar> rD = dicReciprocalD(A);
    do
    {
        wArAold = wArA;
        for (std::size_t cell = 0; cell < nCells; ++cell)
        {
            wA[cell] = rD[cell]*rA[cell];
        }
        dicSubstitute(A, rD, wA);
        wArA = sumProd(wA, rA);
        if (perf.nIterations == 0)
        {
            pA = wA;
        }
        else
        {
            const scalar beta = wArA/wArAold;
            for (std::size_t cell = 0; cell < nCells; ++cell)
            {
                pA[cell] = wA[cell] + beta*pA[cell];
            }
        }
        amul(A, pA, wA);
        const scalar wApA = sumProd(wA, pA);
        // checkSingularity
        if (std::fabs(wApA)/nf < perfVSmall) break;
        const scalar alpha = wArA/wApA;
        for (std::size_t cell = 0; cell < nCells; ++cell)
        {
            psi[cell] += alpha*pA[cell];
            rA[cell] -= alpha*wA[cell];
        }
        perf.finalResidual = sumMag(rA)/nf;
    } while (++perf.nIterations < maxIter && !converged(perf, tolerance, relTol));
    return perf;
}

// GAMGSolver::scale
void scale(
    std::vector<scalar>& field,
    std::vector<scalar>& Acf,
    const LduLevel& A,
    const std::vector<scalar>& source)
{
    amul(A, field, Acf);
    const std::size_t nCells = field.size();
    scalar sfNum = 0;
    scalar sfDen = 0;
    for (std::size_t i = 0; i < nCells; ++i)
    {
        sfNum += field[i]*source[i];
        sfDen += field[i]*Acf[i];
    }
    // stabilise(x, vsmall)
    const scalar sf = sfNum/(sfDen >= 0 ? sfDen + perfVSmall : sfDen - perfVSmall);
    for (std::size_t i = 0; i < nCells; ++i)
    {
        field[i] = sf*field[i] + (source[i] - sf*Acf[i])/A.diag[i];
    }
}

// GAMGAgglomeration::restrictField
void restrictField(
    std::vector<scalar>& cf,
    const std::vector<scalar>& ff,
    const std::vector<label>& fineToCoarse)
{
    std::fill(cf.begin(), cf.end(), scalar(0));
    for (std::size_t i = 0; i < ff.size(); ++i)
    {
        cf[static_cast<std::size_t>(fineToCoarse[i])] += ff[i];
    }
}

// GAMGAgglomeration::prolongField: injection, and it OVERWRITES ff
void prolongField(
    std::vector<scalar>& ff,
    const std::vector<scalar>& cf,
    const std::vector<label>& fineToCoarse)
{
    for (std::size_t i = 0; i < fineToCoarse.size(); ++i)
    {
        ff[i] = cf[static_cast<std::size_t>(fineToCoarse[i])];
    }
}

// GAMGSolver::agglomerateMatrix, symmetric branch
LduLevel agglomerateMatrix(
    const LduLevel& fineMatrix,
    const GamgAgglomeration& agglomeration,
    label fineLevelIndex)
{
    const std::size_t li = static_cast<std::size_t>(fineLevelIndex);
    LduLevel coarse;
    coarse.addr = &agglomeration.meshLevels[li];
    coarse.diag.resize(static_cast<std::size_t>(agglomeration.nCells[li]));
    restrictField(coarse.diag, fineMatrix.diag, agglomeration.restrictAddressing[li]);
    coarse.upper.assign(static_cast<std::size_t>(agglomeration.nFaces[li]), scalar(0));
    const std::vector<label>& faceRestrictAddr = agglomeration.faceRestrictAddressing[li];
    for (std::size_t fineFacei = 0; fineFacei < faceRestrictAddr.size(); ++fineFacei)
    {
        const label cFace = faceRestrictAddr[fineFacei];
        if (cFace >= 0)
        {
            coarse.upper[static_cast<std::size_t>(cFace)] += fineMatrix.upper[fineFacei];
        }
        else
        {
            // Add the fine face coefficient into the diagonal.
            coarse.diag[static_cast<std::size_t>(-1 - cFace)] += 2*fineMatrix.upper[fineFacei];
        }
    }
    return coarse;
}

} // namespace

SolverPerformance gamgCoarsestPcgDic(
    const GamgLduAddressing& addr,
    const std::vector<scalar>& diag,
    const std::vector<scalar>& upper,
    std::vector<scalar>& psi,
    const std::vector<scalar>& source,
    scalar tolerance,
    scalar relTol)
{
    LduLevel A;
    A.addr = &addr;
    A.diag = diag;
    A.upper = upper;
    return pcgDic(A, psi, source, tolerance, relTol);
}

bool gamgSmootherPorted(const std::string& smoother)
{
    return smoother == "DIC"
        || smoother == "DICGaussSeidel"
        || smoother == "GaussSeidel"
        || smoother == "symGaussSeidel";
}

namespace {

// GAMGSolver's constructor and initVcycle, for one matrix: the folded fine level, the coarse matrices
// built from it, the work fields and one smoother per level. Held by pointer from the smoothers, so it
// is built in place and never copied.
struct GamgHierarchy
{
    const GamgAgglomeration* agglomeration = nullptr;
    LduLevel fine;
    std::vector<LduLevel> matrixLevels;
    std::vector<std::vector<scalar>> coarseCorrFields;
    std::vector<std::vector<scalar>> coarseSources;
    std::vector<Smoother> smoothers;
    // the sub-fields OpenFOAM carves out of Apsi and finestCorrection
    std::vector<scalar> ACf;
    std::vector<scalar> preSmoothedCoarseCorrField;

    GamgHierarchy(const GamgHierarchy&) = delete;
    GamgHierarchy& operator=(const GamgHierarchy&) = delete;
    GamgHierarchy() = default;

    // `fineDiag` and `fineUpper` are the matrix as the solver is handed it, boundary folded in
    void build(
        const GamgAgglomeration& a,
        const std::vector<scalar>& fineDiag,
        const std::vector<scalar>& fineUpper,
        const std::string& smoother)
    {
        agglomeration = &a;
        fine.addr = &a.fineMesh;
        fine.diag = fineDiag;
        fine.upper = fineUpper;
        const label nLevels = a.size();
        matrixLevels.clear();
        matrixLevels.reserve(static_cast<std::size_t>(nLevels));
        for (label fineLevelIndex = 0; fineLevelIndex < nLevels; ++fineLevelIndex)
        {
            const LduLevel& fineMatrix =
                fineLevelIndex == 0 ? fine : matrixLevels[static_cast<std::size_t>(fineLevelIndex) - 1];
            matrixLevels.push_back(agglomerateMatrix(fineMatrix, a, fineLevelIndex));
        }
        coarseCorrFields.assign(static_cast<std::size_t>(nLevels), std::vector<scalar>());
        coarseSources.assign(static_cast<std::size_t>(nLevels), std::vector<scalar>());
        smoothers.clear();
        smoothers.reserve(static_cast<std::size_t>(nLevels) + 1);
        smoothers.emplace_back(fine, smoother);
        for (label leveli = 0; leveli < nLevels; ++leveli)
        {
            const std::size_t li = static_cast<std::size_t>(leveli);
            coarseSources[li].resize(matrixLevels[li].diag.size());
            coarseCorrFields[li].resize(matrixLevels[li].diag.size());
            smoothers.emplace_back(matrixLevels[li], smoother);
        }
    }
};

// GAMGSolver::Vcycle (GAMGSolverSolve.C:170-463). psi is corrected in place; Apsi is scratch, and on
// return holds A*finestCorrection when the correction was scaled.
void vCycle(
    GamgHierarchy& h,
    const GamgControls& controls,
    std::vector<scalar>& psi,
    const std::vector<scalar>& source,
    std::vector<scalar>& Apsi,
    std::vector<scalar>& finestCorrection,
    const std::vector<scalar>& finestResidual,
    GamgSolveLog* log)
{
    const GamgAgglomeration& agglomeration = *h.agglomeration;
    const label coarsestLevel = static_cast<label>(h.matrixLevels.size()) - 1;
    std::vector<std::vector<scalar>>& coarseCorrFields = h.coarseCorrFields;
    std::vector<std::vector<scalar>>& coarseSources = h.coarseSources;

    // Restrict finest grid residual for the next level up.
    restrictField(coarseSources[0], finestResidual, agglomeration.restrictAddressing[0]);

    // Residual restriction (going to coarser levels)
    for (label leveli = 0; leveli < coarsestLevel; ++leveli)
    {
        const std::size_t li = static_cast<std::size_t>(leveli);
        // the optional pre-smoothing sweeps
        if (controls.nPreSweeps)
        {
            std::fill(coarseCorrFields[li].begin(), coarseCorrFields[li].end(), scalar(0));
            h.smoothers[li + 1].smooth(
                coarseCorrFields[li],
                coarseSources[li],
                std::min(
                    controls.nPreSweeps + controls.preSweepsLevelMultiplier*leveli,
                    controls.maxPreSweeps));
            // but not on the coarsest level because it evaluates to 1
            if (controls.scaleCorrection && leveli < coarsestLevel - 1)
            {
                h.ACf.resize(coarseCorrFields[li].size());
                scale(coarseCorrFields[li], h.ACf, h.matrixLevels[li], coarseSources[li]);
            }
            // Correct the residual with the new solution
            residual(h.matrixLevels[li], coarseSources[li], coarseCorrFields[li], coarseSources[li]);
        }
        // Residual is equal to source
        restrictField(coarseSources[li + 1], coarseSources[li], agglomeration.restrictAddressing[li + 1]);
    }

    // solveCoarsestLevel
    {
        const std::size_t lc = static_cast<std::size_t>(coarsestLevel);
        std::fill(coarseCorrFields[lc].begin(), coarseCorrFields[lc].end(), scalar(0));
        const SolverPerformance coarsePerf = pcgDic(
            h.matrixLevels[lc],
            coarseCorrFields[lc],
            coarseSources[lc],
            controls.tolerance,
            controls.relTol);
        if (log)
        {
            log->coarsest.push_back(coarsePerf);
        }
    }

    // Smoothing and prolongation of the coarse correction fields (going to finer levels)
    for (label leveli = coarsestLevel - 1; leveli >= 0; --leveli)
    {
        const std::size_t li = static_cast<std::size_t>(leveli);
        // Only store the preSmoothedCoarseCorrField if pre-smoothing is used
        if (controls.nPreSweeps)
        {
            h.preSmoothedCoarseCorrField = coarseCorrFields[li];
        }
        prolongField(coarseCorrFields[li], coarseCorrFields[li + 1], agglomeration.restrictAddressing[li + 1]);

        // Scale coarse-grid correction field
        // but not on the coarsest level because it evaluates to 1
        if (controls.scaleCorrection && leveli < coarsestLevel - 1)
        {
            h.ACf.resize(coarseCorrFields[li].size());
            scale(coarseCorrFields[li], h.ACf, h.matrixLevels[li], coarseSources[li]);
        }
        if (controls.nPreSweeps)
        {
            for (std::size_t i = 0; i < coarseCorrFields[li].size(); ++i)
            {
                coarseCorrFields[li][i] += h.preSmoothedCoarseCorrField[i];
            }
        }
        h.smoothers[li + 1].smooth(
            coarseCorrFields[li],
            coarseSources[li],
            std::min(
                controls.nPostSweeps + controls.postSweepsLevelMultiplier*leveli,
                controls.maxPostSweeps));
    }

    // Prolong the finest level correction
    prolongField(finestCorrection, coarseCorrFields[0], agglomeration.restrictAddressing[0]);
    if (controls.scaleCorrection)
    {
        scale(finestCorrection, Apsi, h.fine, finestResidual);
    }
    for (std::size_t i = 0; i < psi.size(); ++i)
    {
        psi[i] += finestCorrection[i];
    }
    h.smoothers[0].smooth(psi, source, controls.nFinestSweeps);
}

void checkGamgInputs(
    const FvScalarMatrix& M,
    const PrimitiveMesh& m,
    const GamgAgglomeration& agglomeration,
    const GamgControls& controls)
{
    if (!gamgSmootherPorted(controls.smoother))
    {
        throw std::runtime_error(
            "brae GAMG: smoother `" + controls.smoother + "` is not ported. GAMGSolver selects it through "
            "lduMatrix::smoother::New; brae has DIC, DICGaussSeidel, GaussSeidel and symGaussSeidel.");
    }
    if (M.upper != M.lower)
    {
        throw std::runtime_error(
            "brae GAMG: the matrix is asymmetric. GAMGSolver then agglomerates upper and lower separately, "
            "turns scaleCorrection off, and solves the coarsest level with PBiCGStab and DILU "
            "(GAMGSolver.C:83, :314; GAMGSolverAgglomerateMatrix.C:134); only the symmetric branch is ported.");
    }
    if (agglomeration.size() == 0)
    {
        throw std::runtime_error(
            "brae GAMG: no coarse levels created, either matrix too small for GAMG or nCellsInCoarsestLevel "
            "too large. OpenFOAM stops on the same condition (GAMGSolver.C:333).");
    }
    if (agglomeration.fineMesh.nCells != m.nCells())
    {
        throw std::runtime_error("brae GAMG: the agglomeration was built for a different mesh.");
    }
}

// fvMatrix::solveSegregated: addBoundaryDiag, addBoundarySource
void foldBoundary(
    const FvScalarMatrix& M,
    const std::vector<FvPatch>& patches,
    std::vector<scalar>& diag,
    std::vector<scalar>& source)
{
    diag = M.diag;
    source = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const std::size_t c = static_cast<std::size_t>(patches[pi].faceCells[i]);
            diag[c] += M.internalCoeffs[pi][static_cast<std::size_t>(i)];
            source[c] += M.boundaryCoeffs[pi][static_cast<std::size_t>(i)];
        }
    }
}

} // namespace

GamgControls readGamgControls(
    const FoamDict& d,
    scalar tol,
    scalar relTol,
    int maxIter,
    const std::string& who)
{
    auto switchOr = [&](const std::string& key, bool def)
    {
        if (!d.found(key)) return def;
        const std::string w = d.wordOr(key, "");
        if (w == "yes" || w == "true" || w == "on" || w == "y" || w == "t" || w == "1") return true;
        if (w == "no" || w == "false" || w == "off" || w == "n" || w == "f" || w == "0") return false;
        throw std::runtime_error(who + "has `" + key + " " + w + "`, which is not a Switch.");
    };

    GamgControls c;
    c.smoother = d.wordOr("smoother", "");
    c.tolerance = tol;
    c.relTol = relTol;
    c.maxIter = maxIter;
    c.minIter = d.intOr("minIter", 0);
    c.nPreSweeps = d.intOr("nPreSweeps", c.nPreSweeps);
    c.preSweepsLevelMultiplier = d.intOr("preSweepsLevelMultiplier", c.preSweepsLevelMultiplier);
    c.maxPreSweeps = d.intOr("maxPreSweeps", c.maxPreSweeps);
    c.nPostSweeps = d.intOr("nPostSweeps", c.nPostSweeps);
    c.postSweepsLevelMultiplier = d.intOr("postSweepsLevelMultiplier", c.postSweepsLevelMultiplier);
    c.maxPostSweeps = d.intOr("maxPostSweeps", c.maxPostSweeps);
    c.nFinestSweeps = d.intOr("nFinestSweeps", c.nFinestSweeps);
    c.scaleCorrection = switchOr("scaleCorrection", true);
    c.nCellsInCoarsestLevel = static_cast<label>(d.intOr("nCellsInCoarsestLevel", 10));

    if (c.smoother.empty())
    {
        throw std::runtime_error(
            who + "names no `smoother`. lduMatrix::smoother::New reads it with a mandatory lookup and "
            "OpenFOAM stops without it.");
    }
    if (!gamgSmootherPorted(c.smoother))
    {
        throw std::runtime_error(
            who + "asks for `smoother " + c.smoother + "`, which is not ported. brae's GAMG has DIC, "
            "DICGaussSeidel, GaussSeidel and symGaussSeidel (gamg_solver_cpp.cuh).");
    }
    const std::string agglomerator = d.wordOr("agglomerator", "faceAreaPair");
    if (agglomerator != "faceAreaPair")
    {
        throw std::runtime_error(
            who + "asks for `agglomerator " + agglomerator + "`. Only faceAreaPair is ported "
            "(pair_gamg_agglomeration_cpp.cuh); another agglomerator is another hierarchy.");
    }
    if (d.intOr("mergeLevels", 1) != 1)
    {
        throw std::runtime_error(
            who + "asks for `mergeLevels " + std::to_string(d.intOr("mergeLevels", 1)) + "`. "
            "pairGAMGAgglomeration then folds pairs of levels into one (combineLevels, "
            "pairGAMGAgglomerate.C:138), which is not ported.");
    }
    if (d.intOr("updateInterval", 1) != 1)
    {
        throw std::runtime_error(
            who + "sets `updateInterval`. faceAreaPairGAMGAgglomeration then weighs faces by magSf "
            "rather than by the perturbed area vector whenever the time index is not a multiple of it "
            "(faceAreaPairGAMGAgglomeration.C:66-99); only the default's branch is ported.");
    }
    if (!switchOr("cacheAgglomeration", true))
    {
        throw std::runtime_error(
            who + "sets `cacheAgglomeration no`. The hierarchy is then rebuilt at every solve, each "
            "time from wherever the static pairing direction was left; brae builds it once.");
    }
    if (switchOr("interpolateCorrection", false))
    {
        throw std::runtime_error(
            who + "sets `interpolateCorrection yes` (GAMGSolverInterpolate.C), which is not ported.");
    }
    if (switchOr("directSolveCoarsest", false))
    {
        throw std::runtime_error(
            who + "sets `directSolveCoarsest yes`. The coarsest level is then LU-factored rather than "
            "solved by PCG to the entry's tolerance (GAMGSolver.C:303), which is not ported.");
    }
    if (d.subDict("coarsestLevelCorr"))
    {
        throw std::runtime_error(
            who + "carries a `coarsestLevelCorr` sub-dictionary. GAMGSolver then builds the coarsest "
            "level's solver from it (GAMGSolver.C:318) instead of PCG with DIC at the entry's own "
            "tolerance, which is the one brae runs.");
    }
    if (d.found("processorAgglomerator"))
    {
        throw std::runtime_error(
            who + "names a `processorAgglomerator`. This is a serial solver and the entry selects a "
            "different coarse hierarchy in OpenFOAM's parallel runs.");
    }
    return c;
}

SolverPerformance gamgSolve(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const GamgAgglomeration& agglomeration,
    const GamgControls& controls,
    GamgSolveLog* log)
{
    checkGamgInputs(M, m, agglomeration, controls);
    std::vector<scalar> diag;
    std::vector<scalar> source;
    foldBoundary(M, patches, diag, source);

    // the constructor: one coarse matrix per agglomeration level, each from the one above
    GamgHierarchy h;
    h.build(agglomeration, diag, M.upper, controls.smoother);
    const LduLevel& fine = h.fine;

    const std::size_t nCells = psi.size();
    std::vector<scalar> Apsi(nCells);
    amul(fine, psi, Apsi);
    std::vector<scalar> finestCorrection(nCells);
    const scalar nf = normFactor(fine, psi, source, Apsi, finestCorrection);
    std::vector<scalar> finestResidual(nCells);
    for (std::size_t i = 0; i < nCells; ++i)
    {
        finestResidual[i] = source[i] - Apsi[i];
    }

    SolverPerformance perf;
    perf.initialResidual = sumMag(finestResidual)/nf;
    perf.finalResidual = perf.initialResidual;
    if (controls.minIter <= 0 && converged(perf, controls.tolerance, controls.relTol)) return perf;

    do
    {
        vCycle(h, controls, psi, source, Apsi, finestCorrection, finestResidual, log);

        // Calculate finest level residual field
        amul(fine, psi, Apsi);
        for (std::size_t i = 0; i < nCells; ++i)
        {
            finestResidual[i] = source[i];
            finestResidual[i] -= Apsi[i];
        }
        perf.finalResidual = sumMag(finestResidual)/nf;
    } while
    (
        (++perf.nIterations < controls.maxIter && !converged(perf, controls.tolerance, controls.relTol))
     || perf.nIterations < controls.minIter
    );
    return perf;
}

SolverPerformance pcgGamgSolve(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const GamgAgglomeration& agglomeration,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter,
    const GamgPreconditionerControls& precond,
    GamgSolveLog* log)
{
    checkGamgInputs(M, m, agglomeration, precond.gamg);
    if (precond.nVcycles < 1)
    {
        throw std::runtime_error("brae GAMG preconditioner: nVcycles must be at least 1.");
    }
    std::vector<scalar> diag;
    std::vector<scalar> source;
    foldBoundary(M, patches, diag, source);

    // PCG::scalarSolve (PCG.C:67-215)
    LduLevel A;
    A.addr = &agglomeration.fineMesh;
    A.diag = diag;
    A.upper = M.upper;
    const std::size_t nCells = psi.size();
    std::vector<scalar> pA(nCells);
    std::vector<scalar> wA(nCells);
    scalar wArA = perfGreat;
    scalar wArAold = wArA;

    amul(A, psi, wA);
    std::vector<scalar> rA(nCells);
    for (std::size_t cell = 0; cell < nCells; ++cell)
    {
        rA[cell] = source[cell] - wA[cell];
    }
    const scalar nf = normFactor(A, psi, source, wA, pA);

    SolverPerformance perf;
    perf.initialResidual = sumMag(rA)/nf;
    perf.finalResidual = perf.initialResidual;
    if (minIter <= 0 && converged(perf, tolerance, relTol)) return perf;

    // the preconditioner, constructed on first use: a GAMGSolver on the same matrix, with the
    // `preconditioner` sub-dictionary as its controls (lduMatrixPreconditioner.C, New)
    GamgHierarchy h;
    h.build(agglomeration, diag, M.upper, precond.gamg.smoother);
    std::vector<scalar> AwA(nCells);
    std::vector<scalar> finestCorrection(nCells);
    std::vector<scalar> finestResidual(nCells);
    // GAMGPreconditioner::precondition (GAMGPreconditioner.C:79-150): nVcycles V-cycles from zero
    auto precondition = [&](std::vector<scalar>& w, const std::vector<scalar>& r)
    {
        std::fill(w.begin(), w.end(), scalar(0));
        finestResidual = r;
        for (int cycle = 0; cycle < precond.nVcycles; ++cycle)
        {
            vCycle(h, precond.gamg, w, r, AwA, finestCorrection, finestResidual, log);
            if (cycle < precond.nVcycles - 1)
            {
                // Calculate finest level residual field
                amul(A, w, AwA);
                finestResidual = r;
                for (std::size_t i = 0; i < nCells; ++i)
                {
                    finestResidual[i] -= AwA[i];
                }
            }
        }
    };

    do
    {
        wArAold = wArA;
        precondition(wA, rA);
        wArA = sumProd(wA, rA);
        if (perf.nIterations == 0)
        {
            pA = wA;
        }
        else
        {
            const scalar beta = wArA/wArAold;
            for (std::size_t cell = 0; cell < nCells; ++cell)
            {
                pA[cell] = wA[cell] + beta*pA[cell];
            }
        }
        amul(A, pA, wA);
        const scalar wApA = sumProd(wA, pA);
        // checkSingularity
        if (std::fabs(wApA)/nf < perfVSmall) break;
        const scalar alpha = wArA/wApA;
        for (std::size_t cell = 0; cell < nCells; ++cell)
        {
            psi[cell] += alpha*pA[cell];
            rA[cell] -= alpha*wA[cell];
        }
        perf.finalResidual = sumMag(rA)/nf;
    } while
    (
        (++perf.nIterations < maxIter && !converged(perf, tolerance, relTol))
     || perf.nIterations < minIter
    );
    return perf;
}

} // namespace brae
