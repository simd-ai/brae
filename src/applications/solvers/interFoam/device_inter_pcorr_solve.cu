#include "inter_phase_time.cuh"
#include "device_inter_pcorr_solve.cuh"
#include "device_inter_pressure_step.cuh"   // deviceAmgPcgHierarchy
#include "device_blas.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"   // nextDeviceAddressingId
#include "device_pcg.cuh"    // deviceNormFactor
#include <optional>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace brae {

bool DevicePcorrSolver::solve(
    const std::string& asked,
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    SolverPerformance& perf)
{
    for (const FvPatch& p : patches)
    {
        if (p.coupled)
        {
            static bool announced = false;
            if (!announced)
            {
                announced = true;
                std::printf("  pcorr: the mesh has the coupled patch '%s'; pcorr keeps the case's own solver, "
                            "which carries the pair\n", p.name.c_str());
            }
            return false;
        }
    }
    if (M.upper != M.lower)
    {
        throw std::runtime_error(
            "brae interFoam device pcorr solve: the pcorr matrix is asymmetric, and a conjugate gradient needs "
            "a symmetric operator. fvm::laplacian(rAUf, pcorr) is symmetric; reaching here is a defect.");
    }
    static bool announced = false;
    if (!announced)
    {
        announced = true;
        std::printf("  pcorr: system/fvSolution asks for %s; brae runs its AMG-preconditioned PCG instead -- "
                    "same operator, different Krylov method and iteration count\n", asked.c_str());
    }
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    std::optional<interPhase::Nested> timedPart;
    timedPart.emplace("pcorr: the addressing check, the fold and the uploads");
    const bool same = nC == nCells
                   && static_cast<std::size_t>(nIf) == owner.size()
                   && std::equal(owner.begin(), owner.end(), m.owner().begin())
                   && std::equal(neighbour.begin(), neighbour.end(), m.neighbour().begin());
    if (!same)
    {
        // the gather addressing buildDeviceMesh derives, over the internal faces
        owner.assign(m.owner().begin(), m.owner().begin() + nIf);
        neighbour.assign(m.neighbour().begin(), m.neighbour().begin() + nIf);
        nCells = nC;
        std::vector<label> ownerStart(static_cast<std::size_t>(nC) + 1, 0);
        std::vector<label> losortStart(static_cast<std::size_t>(nC) + 1, 0);
        for (label f = 0; f < nIf; ++f)
        {
            ++ownerStart[static_cast<std::size_t>(owner[static_cast<std::size_t>(f)]) + 1];
            ++losortStart[static_cast<std::size_t>(neighbour[static_cast<std::size_t>(f)]) + 1];
        }
        for (label c = 0; c < nC; ++c)
        {
            ownerStart[static_cast<std::size_t>(c) + 1] += ownerStart[static_cast<std::size_t>(c)];
            losortStart[static_cast<std::size_t>(c) + 1] += losortStart[static_cast<std::size_t>(c)];
        }
        std::vector<label> losort(static_cast<std::size_t>(nIf));
        std::vector<label> next(losortStart.begin(), losortStart.end() - 1);
        for (label f = 0; f < nIf; ++f)
        {
            const std::size_t n = static_cast<std::size_t>(neighbour[static_cast<std::size_t>(f)]);
            losort[static_cast<std::size_t>(next[n]++)] = f;
        }
        dOwner.copyFrom(owner);
        dNeighbour.copyFrom(neighbour);
        dOwnerStart.copyFrom(ownerStart);
        dLosort.copyFrom(losort);
        dLosortStart.copyFrom(losortStart);
        // the hierarchy the p_rgh solve takes (deviceAmgPcgHierarchy), from the disk cache on the start mesh
        const bool firstBuild = addressingId == 0;
        addressingId = nextDeviceAddressingId();
        // which hierarchy: see fixedTopology in the header
        static const int mode = []()
        {
            const char* e = std::getenv("BRAE_PCORR_AMG");
            const std::string v = e ? e : "";
            if (v.empty())
            {
                return 0;
            }
            if (v == "plain")
            {
                return 1;
            }
            if (v == "sa")
            {
                return 2;
            }
            throw std::runtime_error("brae interFoam device pcorr solve: BRAE_PCORR_AMG=" + v
                                     + " is not one of plain, sa.");
        }();
        if (mode == 2 || (mode == 0 && fixedTopology && firstBuild))
        {
            static bool said = false;
            if (!said)
            {
                said = true;
                std::printf("  pcorr: its AMG hierarchy is a smoothed-aggregation one of its own, built once "
                            "(the mesh keeps its topology); BRAE_PCORR_AMG=plain takes the one p_rgh uses\n");
            }
            const std::vector<scalar> w(g.magSf().begin(), g.magSf().begin() + nIf);
            const bool smoothed = true;
            interPhase::Nested timedBuild("pcorr: the smoothed-aggregation hierarchy (build)");
            amg = buildAMG(owner, neighbour, w, nC, &smoothed);
        }
        else
        {
            amg = deviceAmgPcgHierarchy(m, g, caseDir, firstBuild, amgMemo);
        }
    }
    // fvMatrix::solveSegregated: addBoundaryDiag, addBoundarySource (no coupled patch reaches here)
    std::vector<scalar> diag = M.diag;
    std::vector<scalar> source = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const std::size_t c = static_cast<std::size_t>(patches[pi].faceCells[i]);
            diag[c] += M.internalCoeffs[pi][static_cast<std::size_t>(i)];
            source[c] += M.boundaryCoeffs[pi][static_cast<std::size_t>(i)];
        }
    }
    dDiag.copyFrom(diag);
    dUpper.copyFrom(M.upper);
    dSource.copyFrom(source);
    dPsi.copyFrom(psi);
    DeviceLduView A{nC, nIf, dDiag.data(), dUpper.data(), dUpper.data(), dOwner.data(), dNeighbour.data(),
                    dOwnerStart.data(), dLosort.data(), dLosortStart.data(), 0, nullptr, nullptr, nullptr};
    A.addressingId = addressingId;
    timedPart.emplace("pcorr: the hierarchy's coarse matrices (Galerkin) and the norm factor");
    amgGalerkin(amg, dDiag, dUpper, dUpper);
    const scalar nf = deviceNormFactor(A, dPsi, dSource, deviceOnes(nC));
    // BRAE_CONTROL_AMG_PCG_ONE_ITERATION=1, the AMG-PCG gates' control, as the p_rgh solve takes it; and
    // BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION=1, which stops pcorr's alone -- the pcorr gate's control
    static const bool oneIteration = std::getenv("BRAE_CONTROL_AMG_PCG_ONE_ITERATION") != nullptr
                                  || std::getenv("BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION") != nullptr;
    timedPart.emplace("pcorr: the AMG-PCG iterations");
    // the fast path's knobs, the ones the p_rgh solve takes (amgPcgKnobs). By default deviceAMGPCG runs the
    // whole PCG loop from its captured graph whatever they say (BRAE_PCG_DEVICE); they matter when that is off
    // or coarse-correction scaling is asked for, and then both pressure entries now take the same ones.
    const AmgPcgKnobs& knobs = amgPcgKnobs();
    const DeviceSolverPerf r = deviceAMGPCG(A, amg, dSource, dPsi, nf, tol, relTol, oneIteration ? 1 : maxIter,
                                            knobs.graph, knobs.checkEvery, knobs.corrScaling, minIter);
    timedPart.emplace("pcorr: the download");
    dPsi.copyTo(psi);
    timedPart.reset();
    perf.initialResidual = r.initialResidual;
    perf.finalResidual = r.finalResidual;
    perf.nIterations = r.nIterations;
    return true;
}

} // namespace brae
