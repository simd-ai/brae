// OpenFOAM's PBiCG with DILU on the device -- see device_pbicg.cuh.
#include "device_pbicg.cuh"
#include "device_blas.cuh"
#include <cmath>
#include <stdexcept>

namespace brae {

DeviceSolverPerf devicePBiCGDilu(
    const DeviceLduView& A,
    const DeviceBuffer<scalar>& b,
    DeviceBuffer<scalar>& psi,
    scalar normFactor,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter,
    DeviceDilu& dilu)
{
    if (A.nCyc > 0)
    {
        throw std::runtime_error(
            "brae devicePBiCGDilu: the matrix carries a coupled interface. PBiCG's transpose system "
            "takes the OTHER side's interface coefficient there (lduMatrix::Tmul), which this solver "
            "does not exchange; the host reference refuses the same (pbicg.cu).");
    }
    if (!dilu.valid || dilu.nCells != A.nCells)
    {
        throw std::runtime_error(
            "brae devicePBiCGDilu: the DILU level schedule was not built for this mesh "
            "(buildDeviceDilu, once, from the mesh addressing).");
    }
    const int nC = A.nCells;
    // the TRANSPOSE system: the same addressing, upper and lower exchanged
    DeviceLduView T = A;
    T.upper = A.lower;
    T.lower = A.upper;

    // DILUPreconditioner::calcReciprocalD, which reads upper*lower only and so serves both systems
    diluUpdate(A, dilu);

    DeviceBuffer<scalar> pA(nC), wA(nC), rA(nC);
    deviceAmul(A, psi, wA, /*onField=*/true);
    deviceCopy(rA, b);
    deviceAxpy(-1.0, wA, rA);

    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA)/normFactor;
    perf.finalResidual = perf.initialResidual;
    // solverPerformance::checkConvergence
    auto converged = [&](scalar fr)
    {
        return (fr < tolerance) || (relTol > 1e-20 && fr < relTol*perf.initialResidual);
    };

    if (minIter > 0 || !converged(perf.finalResidual))
    {
        DeviceBuffer<scalar> pT(nC), wT(nC), rT(nC);
        deviceAmul(T, psi, wT, /*onField=*/true);
        deviceCopy(rT, b);
        deviceAxpy(-1.0, wT, rT);

        scalar wArT = 0;
        int nIter = 0;
        do
        {
            const scalar wArTold = wArT;
            diluApply(A, dilu, rA, wA);                     // precondition
            diluApply(T, dilu, rT, wT);                     // preconditionT
            wArT = deviceDot(wA, rT);
            if (nIter == 0)
            {
                deviceCopy(pA, wA);
                deviceCopy(pT, wT);
            }
            else
            {
                const scalar beta = wArT/wArTold;
                deviceScale(pA, beta);                      // pA = wA + beta*pA
                deviceAxpy(1.0, wA, pA);
                deviceScale(pT, beta);                      // pT = wT + beta*pT
                deviceAxpy(1.0, wT, pT);
            }
            deviceAmul(A, pA, wA);
            deviceAmul(T, pT, wT);
            const scalar wApT = deviceDot(wA, pT);
            // solverPerformance::checkSingularity, |wApT|/normFactor < VSMALL: the loop is LEFT, before
            // the count moves
            if (std::fabs(wApT)/normFactor < 1e-300)
            {
                break;
            }
            const scalar alpha = wArT/wApT;
            deviceAxpy(alpha, pA, psi);
            deviceAxpy(-alpha, wA, rA);
            deviceAxpy(-alpha, wT, rT);
            perf.finalResidual = deviceSumMag(rA)/normFactor;
        } while ((++nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
        perf.nIterations = nIter;
    }
    return perf;
}

} // namespace brae
