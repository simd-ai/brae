#include "device_inter_pcorr_solve.cuh"
#include "device_inter_pressure_step.cuh"   // deviceAmgPcgHierarchy
#include "device_blas.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"   // nextDeviceAddressingId
#include "device_pcg.cuh"    // deviceNormFactor
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
        amg = deviceAmgPcgHierarchy(m, g, caseDir, firstBuild);
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
    DeviceBuffer<scalar> dDiag(diag);
    DeviceBuffer<scalar> dUpper(M.upper);
    DeviceBuffer<scalar> dSource(source);
    DeviceBuffer<scalar> dPsi(psi);
    DeviceLduView A{nC, nIf, dDiag.data(), dUpper.data(), dUpper.data(), dOwner.data(), dNeighbour.data(),
                    dOwnerStart.data(), dLosort.data(), dLosortStart.data(), 0, nullptr, nullptr, nullptr};
    A.addressingId = addressingId;
    amgGalerkin(amg, dDiag, dUpper, dUpper);
    const scalar nf = deviceNormFactor(A, dPsi, dSource, deviceOnes(nC));
    // BRAE_CONTROL_AMG_PCG_ONE_ITERATION=1, the AMG-PCG gates' control, as the p_rgh solve takes it; and
    // BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION=1, which stops pcorr's alone -- the pcorr gate's control
    static const bool oneIteration = std::getenv("BRAE_CONTROL_AMG_PCG_ONE_ITERATION") != nullptr
                                  || std::getenv("BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION") != nullptr;
    const DeviceSolverPerf r = deviceAMGPCG(A, amg, dSource, dPsi, nf, tol, relTol, oneIteration ? 1 : maxIter,
                                            false, 1, false, minIter);
    dPsi.copyTo(psi);
    perf.initialResidual = r.initialResidual;
    perf.finalResidual = r.finalResidual;
    perf.nIterations = r.nIterations;
    return true;
}

} // namespace brae
