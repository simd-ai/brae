#include "device_gamg_solver.cuh"
#include "device_blas.cuh"
#include "device_sym_gauss_seidel.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

constexpr int TPB = 256;
// solverPerformance::small_ and vsmall_
constexpr scalar perfSmall = 1e-20;
constexpr scalar perfVSmall = 1e-300;

inline int nBlocks(int n)
{
    return (n + TPB - 1)/TPB;
}

void ckG(
    cudaError_t e,
    const char* what)
{
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string("brae device GAMG: ") + what + ": " + cudaGetErrorString(e));
    }
}

void launchCheck(const char* what)
{
    ckG(cudaGetLastError(), what);
}

} // namespace

namespace gamgKernels {

// GAMGSolver::agglomerateMatrix's diagonal: the fine cells' diagonals, then twice each interior
// face's coefficient, summed per coarse cell in the host's order
__global__ void galerkinDiagK(
    int nCoarse,
    const label* cellStart,
    const label* cellList,
    const label* innerFaceStart,
    const label* innerFaceList,
    const scalar* fineDiag,
    const scalar* fineUpper,
    scalar* coarseDiag)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nCoarse) return;
    scalar s = 0;
    for (label k = cellStart[c]; k < cellStart[c + 1]; ++k)
    {
        s += fineDiag[cellList[k]];
    }
    for (label k = innerFaceStart[c]; k < innerFaceStart[c + 1]; ++k)
    {
        s += 2*fineUpper[innerFaceList[k]];
    }
    coarseDiag[c] = s;
}

// ...and its upper: each coarse face sums the fine faces that map to it, ascending
__global__ void gatherSumK(
    int nCoarse,
    const label* start,
    const label* list,
    const scalar* fine,
    scalar* coarse)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nCoarse) return;
    scalar s = 0;
    for (label k = start[c]; k < start[c + 1]; ++k)
    {
        s += fine[list[k]];
    }
    coarse[c] = s;
}

// GAMGAgglomeration::prolongField: injection
__global__ void prolongK(
    int nFine,
    const label* fineToCoarse,
    const scalar* cf,
    scalar* ff)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nFine) return;
    ff[i] = cf[fineToCoarse[i]];
}

// out = a - b; out may be a
__global__ void subtractK(
    int n,
    const scalar* a,
    const scalar* b,
    scalar* out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] - b[i];
}

// GAMGSolver::scale's last loop
__global__ void scaleK(
    int n,
    scalar sf,
    const scalar* source,
    const scalar* Acf,
    const scalar* diag,
    scalar* field)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    field[i] = sf*field[i] + (source[i] - sf*Acf[i])/diag[i];
}

} // namespace gamgKernels

namespace {

using namespace gamgKernels;

void gatherSum(
    const DeviceBuffer<label>& start,
    const DeviceBuffer<label>& list,
    const scalar* fine,
    DeviceBuffer<scalar>& coarse)
{
    const int n = static_cast<int>(coarse.size());
    if (n == 0) return;
    gatherSumK<<<nBlocks(n), TPB>>>(n, start.data(), list.data(), fine, coarse.data());
    launchCheck("gather sum");
}

void prolong(
    const DeviceBuffer<label>& fineToCoarse,
    const DeviceBuffer<scalar>& cf,
    DeviceBuffer<scalar>& ff)
{
    const int n = static_cast<int>(ff.size());
    prolongK<<<nBlocks(n), TPB>>>(n, fineToCoarse.data(), cf.data(), ff.data());
    launchCheck("prolong");
}

void subtract(
    const DeviceBuffer<scalar>& a,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& out)
{
    const int n = static_cast<int>(out.size());
    subtractK<<<nBlocks(n), TPB>>>(n, a.data(), b.data(), out.data());
    launchCheck("subtract");
}

void zero(DeviceBuffer<scalar>& x)
{
    if (x.size() == 0) return;
    ckG(cudaMemset(x.data(), 0, sizeof(scalar)*x.size()), "zero");
}

// GAMGSolver::scale
void scale(
    DeviceBuffer<scalar>& field,
    DeviceBuffer<scalar>& Acf,
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& source)
{
    deviceAmul(A, field, Acf);
    const scalar sfNum = deviceDot(field, source);
    const scalar sfDen = deviceDot(field, Acf);
    // stabilise(x, vsmall)
    const scalar sf = sfNum/(sfDen >= 0 ? sfDen + perfVSmall : sfDen - perfVSmall);
    const int n = static_cast<int>(field.size());
    scaleK<<<nBlocks(n), TPB>>>(n, sf, source.data(), Acf.data(), A.diag, field.data());
    launchCheck("scale");
}

// DICSmoother::smooth: each sweep applies the incomplete-Cholesky solve to the TRUE residual. The
// smoother multiplies by rD and substitutes; the preconditioner does the same two things to the same
// field, so diluApply is the sweep.
void dicSmooth(
    const DeviceLduView& A,
    const DeviceDilu& dic,
    DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& source,
    DeviceBuffer<scalar>& rA,
    DeviceBuffer<scalar>& wA,
    int nSweeps)
{
    for (int sweep = 0; sweep < nSweeps; ++sweep)
    {
        // psi IS the solution field here, so a jump cyclic subtracts its jump -- see
        // DeviceLduView::cycJump. Every product below that takes psi does the same.
        deviceAmul(A, psi, rA, /*onField=*/true);
        subtract(source, rA, rA);
        diluApply(A, dic, rA, wA);
        deviceAxpy(scalar(1), wA, psi);
    }
}

// GAMGSolver's smoother, as the host's LduLevel::smooth dispatches it (gamg_solver_cpp.cu): DIC's sweeps
// first, then Gauss-Seidel's, so `DICGaussSeidel` runs both -- which is what OpenFOAM's
// DICGaussSeidelSmoother does. `symmetric` picks which of OpenFOAM's two Gauss-Seidel smoothers this is:
// symGaussSeidel walks up and back down, GaussSeidel only up (GaussSeidelSmoother.C has no second half).
// The sweep is the exact, level-scheduled one the smoothSolver already runs on the device
// (device_sym_gauss_seidel.cuh, held to the host's in tests/gs_ladder.cu), so nothing new is computed
// here -- only the dispatch the host file makes.
struct GamgSmootherKind
{
    bool dic = false;
    bool gaussSeidel = false;
    bool symGaussSeidel = false;
};

GamgSmootherKind gamgSmootherKind(const std::string& smoother)
{
    GamgSmootherKind k;
    k.dic = (smoother == "DIC" || smoother == "DICGaussSeidel");
    k.gaussSeidel = (smoother == "GaussSeidel" || smoother == "DICGaussSeidel");
    k.symGaussSeidel = (smoother == "symGaussSeidel");
    return k;
}

void smoothLevel(
    const DeviceLduView& A,
    const DeviceDilu& dic,
    DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& source,
    DeviceBuffer<scalar>& rA,
    DeviceBuffer<scalar>& wA,
    int nSweeps,
    const GamgSmootherKind& kind)
{
    if (kind.dic)
    {
        dicSmooth(A, dic, psi, source, rA, wA, nSweeps);
    }
    if (kind.gaussSeidel || kind.symGaussSeidel)
    {
        const DeviceGaussSeidelLevels& lv = gsLevelsFor(A);
        for (int sweep = 0; sweep < nSweeps; ++sweep)
        {
            deviceSymGaussSeidelSweepExact(A, source, psi, lv, kind.symGaussSeidel);
        }
    }
}

// SolverPerformance::checkConvergence
bool converged(
    const DeviceSolverPerf& perf,
    scalar tolerance,
    scalar relTol)
{
    return perf.finalResidual < tolerance
        || (relTol > perfSmall && perf.finalResidual < relTol*perf.initialResidual);
}

// the inverse of a fine-to-coarse map as CSR over the coarse entity, fine entries ASCENDING.
// `key(i)` is the coarse index of fine entity i, or negative to leave it out.
template <class Key>
void invertMap(
    std::size_t nFine,
    std::size_t nCoarse,
    Key key,
    DeviceBuffer<label>& startD,
    DeviceBuffer<label>& listD)
{
    std::vector<label> start(nCoarse + 1, 0);
    for (std::size_t i = 0; i < nFine; ++i)
    {
        const label c = key(i);
        if (c >= 0)
        {
            ++start[static_cast<std::size_t>(c) + 1];
        }
    }
    for (std::size_t c = 0; c < nCoarse; ++c)
    {
        start[c + 1] += start[c];
    }
    std::vector<label> list(static_cast<std::size_t>(start[nCoarse]));
    std::vector<label> pos(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < nFine; ++i)
    {
        const label c = key(i);
        if (c >= 0)
        {
            list[static_cast<std::size_t>(pos[static_cast<std::size_t>(c)]++)] = static_cast<label>(i);
        }
    }
    startD.copyFrom(start);
    listD.copyFrom(list);
}


// WHAT GAMGSolver's CONSTRUCTOR AND initVcycle DO, shared by the solver below and by the GAMG
// PRECONDITIONER: the checks, the coarse matrices by Galerkin from the fine one, the coarsest level's
// matrix brought down once for the host solve, and the smoothers' diagonals per level. Split out
// verbatim -- every expression and every order is the one the solve ran before it moved, which is
// what tests/interfoam_gamg_vs_openfoam.sh's nineteen profiles hold.
struct GamgSetup
{
    DeviceLduView A;
    label coarsestLevel = 0;
    std::vector<scalar> coarsestDiag;
    std::vector<scalar> coarsestUpper;
    const GamgLduAddressing* coarsestAddr = nullptr;
    GamgSmootherKind kind;
};

GamgSetup gamgSetup(
    const DeviceLduView& Ain,
    DeviceDilu& fineDic,
    DeviceGamgHierarchy& h,
    const GamgControls& controls)
{
    GamgSetup S;
    if (!deviceGamgSmootherPorted(controls.smoother))
    {
        throw std::runtime_error(
            "brae device GAMG: smoother `" + controls.smoother + "` is not ported. OpenFOAM's GAMG has "
            "DIC, DICGaussSeidel, GaussSeidel, symGaussSeidel and others; this one has the first four.");
    }
    if (!h.host || h.level.empty())
    {
        throw std::runtime_error(
            "brae device GAMG: no coarse levels created, either matrix too small for GAMG or "
            "nCellsInCoarsestLevel too large. OpenFOAM stops on the same condition (GAMGSolver.C:333).");
    }
    const int nCells = Ain.nCells;
    const int nFaces = Ain.nInternalFaces;
    if (h.host->fineMesh.nCells != nCells)
    {
        throw std::runtime_error("brae device GAMG: the hierarchy was built for a different mesh.");
    }
    // only the symmetric branch of GAMGSolver is ported, and nothing downstream would notice
    if (Ain.lower != Ain.upper && nFaces > 0)
    {
        DeviceBuffer<scalar> d(static_cast<std::size_t>(nFaces));
        subtractK<<<nBlocks(nFaces), TPB>>>(nFaces, Ain.upper, Ain.lower, d.data());
        launchCheck("symmetry check");
        if (deviceSumMag(d) != scalar(0))
        {
            throw std::runtime_error(
                "brae device GAMG: the matrix is asymmetric. GAMGSolver then agglomerates upper and lower "
                "separately, turns scaleCorrection off, and solves the coarsest level with PBiCGStab and "
                "DILU; only the symmetric branch is ported.");
        }
    }
    DeviceLduView A = Ain;
    A.lower = A.upper;

    // the constructor: one coarse matrix per level, each from the one above
    const label nLevels = static_cast<label>(h.level.size());
    const label coarsestLevel = nLevels - 1;
    for (label leveli = 0; leveli < nLevels; ++leveli)
    {
        DeviceGamgLevel& L = h.level[static_cast<std::size_t>(leveli)];
        const scalar* fineDiag = leveli == 0 ? A.diag : h.level[static_cast<std::size_t>(leveli) - 1].A.diag.data();
        const scalar* fineUpper = leveli == 0 ? A.upper : h.level[static_cast<std::size_t>(leveli) - 1].A.upper.data();
        const int nCoarse = L.A.nCells;
        galerkinDiagK<<<nBlocks(nCoarse), TPB>>>(
            nCoarse,
            L.cellStart.data(),
            L.cellList.data(),
            L.innerFaceStart.data(),
            L.innerFaceList.data(),
            fineDiag,
            fineUpper,
            L.A.diag.data());
        launchCheck("coarse diagonal");
        gatherSum(L.faceStart, L.faceList, fineUpper, L.A.upper);
    }
    // the coarsest level is solved on the host: its matrix comes down once per solve
    DeviceGamgLevel& LC = h.level[static_cast<std::size_t>(coarsestLevel)];
    S.coarsestDiag = LC.A.diag.host();
    S.coarsestUpper = LC.A.upper.host();
    S.coarsestAddr = &h.host->meshLevels[static_cast<std::size_t>(coarsestLevel)];

    // initVcycle: the smoothers. DIC's reciprocal diagonal, per level, for THIS matrix -- refreshed even
    // under a Gauss-Seidel smoother, which does not read it, because DICGaussSeidel runs both.
    S.kind = gamgSmootherKind(controls.smoother);
    diluUpdate(A, fineDic);
    for (DeviceGamgLevel& L : h.level)
    {
        diluUpdate(L.view(), L.dic);
    }
    S.A = A;
    S.coarsestLevel = coarsestLevel;
    return S;
}

// ONE V-CYCLE, GAMGSolver::Vcycle. The caller sets h.finestResidual to the residual this cycle is to
// reduce and psi to the field it corrects; `controls` is the entry the cycle runs under -- the
// solver's own, or the PRECONDITIONER's sub-dictionary, which is what decides the coarsest solve's
// tolerance (GAMGSolver.C, and the `pcgGamgTol` profile of the gate).
void vCycle(
    const GamgSetup& S,
    DeviceGamgHierarchy& h,
    DeviceDilu& fineDic,
    DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& b,
    const GamgControls& controls,
    GamgSolveLog* log)
{
    const DeviceLduView& A = S.A;
    const label coarsestLevel = S.coarsestLevel;
    const GamgSmootherKind& kind = S.kind;
    DeviceGamgLevel& LC = h.level[static_cast<std::size_t>(coarsestLevel)];
    // Vcycle. Restrict finest grid residual for the next level up.
    gatherSum(h.level[0].cellStart, h.level[0].cellList, h.finestResidual.data(), h.level[0].source);

    // Residual restriction (going to coarser levels)
    for (label leveli = 0; leveli < coarsestLevel; ++leveli)
    {
        DeviceGamgLevel& L = h.level[static_cast<std::size_t>(leveli)];
        DeviceGamgLevel& next = h.level[static_cast<std::size_t>(leveli) + 1];
        if (controls.nPreSweeps)
        {
            zero(L.corr);
            smoothLevel(
                L.view(),
                L.dic,
                L.corr,
                L.source,
                L.rA,
                L.wA,
                std::min(
                    controls.nPreSweeps + controls.preSweepsLevelMultiplier*leveli,
                    controls.maxPreSweeps),
                kind);
            // but not on the coarsest level because it evaluates to 1
            if (controls.scaleCorrection && leveli < coarsestLevel - 1)
            {
                scale(L.corr, L.ACf, L.view(), L.source);
            }
            // Correct the residual with the new solution
            deviceAmul(L.view(), L.corr, L.ACf);
            subtract(L.source, L.ACf, L.source);
        }
        // Residual is equal to source
        gatherSum(next.cellStart, next.cellList, L.source.data(), next.source);
    }

    // solveCoarsestLevel, on the host, from zero
    {
        const std::vector<scalar> coarsestSource = LC.source.host();
        std::vector<scalar> coarsestCorr(coarsestSource.size(), scalar(0));
        const SolverPerformance coarsePerf = gamgCoarsestPcgDic(
            *S.coarsestAddr,
            S.coarsestDiag,
            S.coarsestUpper,
            coarsestCorr,
            coarsestSource,
            controls.tolerance,
            controls.relTol);
        LC.corr.copyFrom(coarsestCorr);
        if (log)
        {
            log->coarsest.push_back(coarsePerf);
        }
    }

    // Smoothing and prolongation of the coarse correction fields (going to finer levels)
    for (label leveli = coarsestLevel - 1; leveli >= 0; --leveli)
    {
        DeviceGamgLevel& L = h.level[static_cast<std::size_t>(leveli)];
        DeviceGamgLevel& next = h.level[static_cast<std::size_t>(leveli) + 1];
        // Only store the preSmoothedCoarseCorrField if pre-smoothing is used
        if (controls.nPreSweeps)
        {
            deviceCopy(L.preSmoothed, L.corr);
        }
        prolong(next.restrictMap, next.corr, L.corr);
        // but not on the coarsest level because it evaluates to 1
        if (controls.scaleCorrection && leveli < coarsestLevel - 1)
        {
            scale(L.corr, L.ACf, L.view(), L.source);
        }
        if (controls.nPreSweeps)
        {
            deviceAxpy(scalar(1), L.preSmoothed, L.corr);
        }
        smoothLevel(
            L.view(),
            L.dic,
            L.corr,
            L.source,
            L.rA,
            L.wA,
            std::min(
                controls.nPostSweeps + controls.postSweepsLevelMultiplier*leveli,
                controls.maxPostSweeps),
            kind);
    }

    // Prolong the finest level correction
    prolong(h.level[0].restrictMap, h.level[0].corr, h.finestCorrection);
    if (controls.scaleCorrection)
    {
        scale(h.finestCorrection, h.Apsi, A, h.finestResidual);
    }
    deviceAxpy(scalar(1), h.finestCorrection, psi);
    smoothLevel(A, fineDic, psi, b, h.rA, h.wA, controls.nFinestSweeps, kind);
}

} // namespace

DeviceGamgHierarchy& DeviceGamgCache::get(label nCellsInCoarsestLevel)
{
    // A MESH THAT HAS MOVED: OpenFOAM's GAMGAgglomeration::movePoints sets requireUpdate_, and the
    // next GAMGAgglomeration::New checks the object out and builds the hierarchy again on the moved
    // mesh (GAMGAgglomeration.C:311-330, :498-516). The host cache says so by going un-built, and
    // this upload is stale with it -- the coarse levels' face areas and the pairing they decide are
    // the OLD mesh's. The test is the BUILD COUNT, not `built`: a host GAMG solve between the move and
    // this call (CorrectPhi's pcorr) rebuilds the host hierarchy and leaves it built, and an upload
    // keyed on `built` alone was kept across it -- GamgAgglomerationCache::buildCount has the
    // measurement.
    if (uploaded && host && host->built && host->buildCount == uploadedBuild) return device;
    if (!mesh || !geometry || !host)
    {
        throw std::runtime_error(
            "brae device GAMG: the hierarchy cache was handed no host mesh. It is built on the host, from "
            "the mesh's own addressing and face areas, by the first GAMG solve of the run.");
    }
    const GamgAgglomeration& a = host->get(*mesh, *geometry, nCellsInCoarsestLevel);
    device.host = &a;
    device.level.resize(static_cast<std::size_t>(a.size()));
    for (label leveli = 0; leveli < a.size(); ++leveli)
    {
        const std::size_t li = static_cast<std::size_t>(leveli);
        DeviceGamgLevel& L = device.level[li];
        const GamgLduAddressing& coarse = a.meshLevels[li];
        const std::size_t nCoarse = static_cast<std::size_t>(coarse.nCells);
        const std::size_t nCoarseFaces = coarse.upperAddr.size();
        const std::vector<label>& cellMap = a.restrictAddressing[li];
        const std::vector<label>& faceMap = a.faceRestrictAddressing[li];

        // the coefficients are placeholders: Galerkin refills them at every solve. `lower` stays
        // empty, because the level's view aliases it to upper.
        L.A = buildDeviceLdu(
            std::vector<scalar>(nCoarse, scalar(0)),
            std::vector<scalar>(nCoarseFaces, scalar(0)),
            std::vector<scalar>(),
            coarse.lowerAddr,
            coarse.upperAddr,
            static_cast<int>(nCoarse));
        L.dic = buildDeviceDilu(coarse.lowerAddr, coarse.upperAddr, coarse.nCells);
        L.restrictMap.copyFrom(cellMap);
        auto coarseCellOf = [&](std::size_t i)
        {
            return cellMap[i];
        };
        auto coarseFaceOf = [&](std::size_t f)
        {
            return faceMap[f];
        };
        // a fine face both of whose cells merged is stored as -(coarse cell + 1)
        auto enclosingCellOf = [&](std::size_t f)
        {
            return faceMap[f] < 0 ? -1 - faceMap[f] : label(-1);
        };
        invertMap(cellMap.size(), nCoarse, coarseCellOf, L.cellStart, L.cellList);
        invertMap(faceMap.size(), nCoarseFaces, coarseFaceOf, L.faceStart, L.faceList);
        invertMap(faceMap.size(), nCoarse, enclosingCellOf, L.innerFaceStart, L.innerFaceList);
        L.corr.resize(nCoarse);
        L.source.resize(nCoarse);
        L.ACf.resize(nCoarse);
        L.preSmoothed.resize(nCoarse);
        L.rA.resize(nCoarse);
        L.wA.resize(nCoarse);
    }
    const std::size_t nFine = static_cast<std::size_t>(a.fineMesh.nCells);
    device.Apsi.resize(nFine);
    device.finestCorrection.resize(nFine);
    device.finestResidual.resize(nFine);
    device.rA.resize(nFine);
    device.wA.resize(nFine);
    uploaded = true;
    uploadedBuild = host->buildCount;
    return device;
}

bool deviceGamgSmootherPorted(const std::string& smoother)
{
    // the host's four: DIC, and the two Gauss-Seidel smoothers with DIC's combination of them
    return smoother == "DIC" || smoother == "DICGaussSeidel" || smoother == "GaussSeidel"
        || smoother == "symGaussSeidel";
}

DeviceSolverPerf deviceGamgSolve(
    const DeviceLduView& Ain,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    DeviceDilu& fineDic,
    DeviceGamgHierarchy& h,
    const GamgControls& controls,
    GamgSolveLog* log)
{
    const GamgSetup S = gamgSetup(Ain, fineDic, h, controls);
    const DeviceLduView& A = S.A;
    const int nCells = A.nCells;

    deviceAmul(A, psi, h.Apsi, /*onField=*/true);
    const scalar nf = deviceNormFactor(A, psi, b, deviceOnes(nCells));
    subtract(b, h.Apsi, h.finestResidual);

    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(h.finestResidual)/nf;
    perf.finalResidual = perf.initialResidual;
    if (controls.minIter <= 0 && converged(perf, controls.tolerance, controls.relTol)) return perf;

    do
    {
        vCycle(S, h, fineDic, psi, b, controls, log);

        // Calculate finest level residual field
        deviceAmul(A, psi, h.Apsi, /*onField=*/true);
        subtract(b, h.Apsi, h.finestResidual);
        perf.finalResidual = deviceSumMag(h.finestResidual)/nf;
    } while
    (
        (++perf.nIterations < controls.maxIter && !converged(perf, controls.tolerance, controls.relTol))
     || perf.nIterations < controls.minIter
    );
    return perf;
}


DeviceSolverPerf devicePcgGamgSolve(
    const DeviceLduView& Ain,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    DeviceDilu& fineDic,
    DeviceGamgHierarchy& h,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter,
    const GamgPreconditionerControls& precond,
    GamgSolveLog* log)
{
    if (precond.nVcycles < 1)
    {
        throw std::runtime_error("brae device GAMG preconditioner: nVcycles must be at least 1.");
    }
    const GamgSetup S = gamgSetup(Ain, fineDic, h, precond.gamg);
    const DeviceLduView& A = S.A;
    const int nCells = A.nCells;

    // PCG::scalarSolve's normFactor, taken on the matrix the solve runs on and the psi it starts from
    DeviceBuffer<scalar> Apsi(nCells);
    deviceAmul(A, psi, Apsi, /*onField=*/true);
    const scalar nf = deviceNormFactor(A, psi, b, deviceOnes(nCells));

    // GAMGPreconditioner::precondition, GAMGPreconditioner.C:79-150. w starts at ZERO on every
    // application -- not at the last one's -- and between cycles the residual is recomputed from the
    // w this one left, which is what makes the second cycle reduce anything.
    DevicePreconApply apply =
        [&](DeviceBuffer<scalar>& w, const DeviceBuffer<scalar>& r)
    {
        zero(w);
        deviceCopy(h.finestResidual, r);
        for (int cycle = 0; cycle < precond.nVcycles; ++cycle)
        {
            vCycle(S, h, fineDic, w, r, precond.gamg, log);
            if (cycle < precond.nVcycles - 1)
            {
                deviceAmul(A, w, h.Apsi, /*onField=*/true);
                subtract(r, h.Apsi, h.finestResidual);
            }
        }
    };
    return deviceJacobiPCG(A, b, psi, nf, tolerance, relTol, maxIter, minIter, nullptr, &apply);
}

} // namespace brae
