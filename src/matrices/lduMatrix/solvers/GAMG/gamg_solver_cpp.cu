#include "gamg_solver_cpp.cuh"
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

SolverPerformance gamgSolve(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const GamgAgglomeration& agglomeration,
    const GamgControls& controls,
    GamgSolveLog* log)
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

    // fvMatrix::solveSegregated: addBoundaryDiag, addBoundarySource
    LduLevel fine;
    fine.addr = &agglomeration.fineMesh;
    fine.diag = M.diag;
    fine.upper = M.upper;
    std::vector<scalar> source = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const std::size_t c = static_cast<std::size_t>(patches[pi].faceCells[i]);
            fine.diag[c] += M.internalCoeffs[pi][static_cast<std::size_t>(i)];
            source[c] += M.boundaryCoeffs[pi][static_cast<std::size_t>(i)];
        }
    }

    // the constructor: one coarse matrix per agglomeration level, each from the one above
    const label nLevels = agglomeration.size();
    std::vector<LduLevel> matrixLevels;
    matrixLevels.reserve(static_cast<std::size_t>(nLevels));
    for (label fineLevelIndex = 0; fineLevelIndex < nLevels; ++fineLevelIndex)
    {
        const LduLevel& fineMatrix =
            fineLevelIndex == 0 ? fine : matrixLevels[static_cast<std::size_t>(fineLevelIndex) - 1];
        matrixLevels.push_back(agglomerateMatrix(fineMatrix, agglomeration, fineLevelIndex));
    }
    const label coarsestLevel = nLevels - 1;

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

    // initVcycle
    std::vector<std::vector<scalar>> coarseCorrFields(static_cast<std::size_t>(nLevels));
    std::vector<std::vector<scalar>> coarseSources(static_cast<std::size_t>(nLevels));
    std::vector<Smoother> smoothers;
    smoothers.reserve(static_cast<std::size_t>(nLevels) + 1);
    smoothers.emplace_back(fine, controls.smoother);
    for (label leveli = 0; leveli < nLevels; ++leveli)
    {
        const std::size_t li = static_cast<std::size_t>(leveli);
        coarseSources[li].resize(matrixLevels[li].diag.size());
        coarseCorrFields[li].resize(matrixLevels[li].diag.size());
        smoothers.emplace_back(matrixLevels[li], controls.smoother);
    }
    // the sub-fields OpenFOAM carves out of Apsi and finestCorrection
    std::vector<scalar> ACf;
    std::vector<scalar> preSmoothedCoarseCorrField;

    do
    {
        // Vcycle. Restrict finest grid residual for the next level up.
        restrictField(coarseSources[0], finestResidual, agglomeration.restrictAddressing[0]);

        // Residual restriction (going to coarser levels)
        for (label leveli = 0; leveli < coarsestLevel; ++leveli)
        {
            const std::size_t li = static_cast<std::size_t>(leveli);
            // the optional pre-smoothing sweeps
            if (controls.nPreSweeps)
            {
                std::fill(coarseCorrFields[li].begin(), coarseCorrFields[li].end(), scalar(0));
                smoothers[li + 1].smooth(
                    coarseCorrFields[li],
                    coarseSources[li],
                    std::min(
                        controls.nPreSweeps + controls.preSweepsLevelMultiplier*leveli,
                        controls.maxPreSweeps));
                // but not on the coarsest level because it evaluates to 1
                if (controls.scaleCorrection && leveli < coarsestLevel - 1)
                {
                    ACf.resize(coarseCorrFields[li].size());
                    scale(coarseCorrFields[li], ACf, matrixLevels[li], coarseSources[li]);
                }
                // Correct the residual with the new solution
                residual(matrixLevels[li], coarseSources[li], coarseCorrFields[li], coarseSources[li]);
            }
            // Residual is equal to source
            restrictField(coarseSources[li + 1], coarseSources[li], agglomeration.restrictAddressing[li + 1]);
        }

        // solveCoarsestLevel
        {
            const std::size_t lc = static_cast<std::size_t>(coarsestLevel);
            std::fill(coarseCorrFields[lc].begin(), coarseCorrFields[lc].end(), scalar(0));
            const SolverPerformance coarsePerf = pcgDic(
                matrixLevels[lc],
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
                preSmoothedCoarseCorrField = coarseCorrFields[li];
            }
            prolongField(coarseCorrFields[li], coarseCorrFields[li + 1], agglomeration.restrictAddressing[li + 1]);

            // Scale coarse-grid correction field
            // but not on the coarsest level because it evaluates to 1
            if (controls.scaleCorrection && leveli < coarsestLevel - 1)
            {
                ACf.resize(coarseCorrFields[li].size());
                scale(coarseCorrFields[li], ACf, matrixLevels[li], coarseSources[li]);
            }
            if (controls.nPreSweeps)
            {
                for (std::size_t i = 0; i < coarseCorrFields[li].size(); ++i)
                {
                    coarseCorrFields[li][i] += preSmoothedCoarseCorrField[i];
                }
            }
            smoothers[li + 1].smooth(
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
            scale(finestCorrection, Apsi, fine, finestResidual);
        }
        for (std::size_t i = 0; i < nCells; ++i)
        {
            psi[i] += finestCorrection[i];
        }
        smoothers[0].smooth(psi, source, controls.nFinestSweeps);

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

} // namespace brae
