// One pass of pEqn.H -- see device_inter_pressure_step.cuh for the three placements it owns.
#include "device_inter_pressure_step.cuh"
#include "device_fvc_reconstruct.cuh"
#include "device_blas.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_mesh.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

void ckS(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("brae interFoam device pressure step: ") + what + ": "
                                 + cudaGetErrorString(e));
}

__global__ void subKernel(const scalar* __restrict__ a, const scalar* __restrict__ b,
                          int n, scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

}   // namespace


scalar deviceInterPressureStep(
    const DeviceMesh&                  dm,
    const DeviceInterPressureInput&    in,
    const DeviceInterPressureHooks&    hooks,
    const DeviceBuffer<scalar>&        rAU,
    const DeviceBuffer<scalar>&        HbyAX,
    const DeviceBuffer<scalar>&        HbyAY,
    const DeviceBuffer<scalar>&        HbyAZ,
    DeviceBuffer<scalar>&              phiHbyAInt,
    DeviceBuffer<scalar>&              phiHbyABnd,
    DeviceBuffer<scalar>&              p_rgh,
    DeviceBuffer<scalar>&              phiInt,
    DeviceBuffer<scalar>&              phiBnd,
    DeviceBuffer<scalar>&              UX,
    DeviceBuffer<scalar>&              UY,
    DeviceBuffer<scalar>&              UZ,
    DeviceBuffer<scalar>&              p,
    DevicePressureTaps*                taps)
{
    if (!in.stf || !in.ghf || !in.snGradRho || !in.magSf || !in.rAUfAll || !in.rho || !in.gh)
        throw std::runtime_error("brae interFoam device pEqn: a required field is missing.");
    if (!hooks.pressureCoeffs)
        throw std::runtime_error(
            "brae interFoam device pEqn: p_rgh's boundary coefficients hook is required. A "
            "fixedFluxPressure patch's gradient is PRESCRIBED from phiHbyA by constrainPressure, and "
            "assembling one that has not been set is the defect this hook exists to prevent.");

    const int nC = dm.nCells, nIf = dm.nInternalFaces, nBf = dm.nBndFaces;
    const int nFaces = nIf + nBf;

    // phig = (stf - ghf*snGrad(rho))*rAUf*magSf over the WHOLE face array, so the boundary half exists.
    DeviceBuffer<scalar> phigAll;
    deviceBuoyancyFlux(nFaces, *in.stf, *in.ghf, *in.snGradRho, *in.rAUfAll, *in.magSf, phigAll);

    // split it: the internal faces come first in the mesh's face order, the boundary patches after.
    DeviceBuffer<scalar> phigInt(static_cast<std::size_t>(nIf)), phigBnd(static_cast<std::size_t>(nBf));
    if (nIf > 0)
        ckS(cudaMemcpy(phigInt.data(), phigAll.data(), sizeof(scalar)*nIf,
                       cudaMemcpyDeviceToDevice), "phig, internal");
    if (nBf > 0)
        ckS(cudaMemcpy(phigBnd.data(), phigAll.data() + nIf, sizeof(scalar)*nBf,
                       cudaMemcpyDeviceToDevice), "phig, boundary");

    // interpolate(rho*rAU) -- the PRODUCT, interpolated once; see device_inter_peqn.cuh note 2.
    DeviceBuffer<scalar> rhoRAUf;
    deviceRhoRAUf(dm, *in.rho, rAU, rhoRAUf);

    DeviceBuffer<scalar> zeroIf(static_cast<std::size_t>(nIf));
    if (nIf > 0) ckS(cudaMemset(zeroIf.data(), 0, sizeof(scalar)*nIf), "zero");
    deviceInterAddPhiHbyATerms(dm, rhoRAUf,
                               in.ddtCorrInt ? *in.ddtCorrInt : zeroIf,
                               phigInt, phigBnd, in.ddtCorrInt != nullptr,
                               phiHbyAInt, phiHbyABnd, in.mrf,
                               in.ddtCorrBnd, in.rhoBndFace, &rAU, in.bndUFixesValue);

    // rAUf on the internal faces -- the head of the full array the caller passed.
    DeviceBuffer<scalar> rAUfInt(static_cast<std::size_t>(nIf));
    if (nIf > 0)
        ckS(cudaMemcpy(rAUfInt.data(), in.rAUfAll->data(), sizeof(scalar)*nIf,
                       cudaMemcpyDeviceToDevice), "rAUf, internal");

    if (in.correctedLaplacian && !hooks.boundaryValues)
        throw std::runtime_error(
            "brae interFoam device pEqn: the corrected laplacian's correction takes grad(p_rgh) from p_rgh's "
            "stored patch values, and no boundaryValues hook was handed in.");

    // THE NON-ORTHOGONAL LOOP (the host's pressureCorrector, inter_peqn_cpp.cu): each pass the fvMatrix
    // constructor's updateCoeffs (pressureCoeffs), the laplacian, its explicit correction from the p_rgh
    // the last pass left, the solve, and p_rgh.correctBoundaryConditions(). The LAST pass's matrix and
    // correction flux make p_rghEqn.flux().
    DevicePressureMatrix P;
    DeviceBuffer<scalar> iC, bC;
    DeviceBuffer<scalar> ffc;
    DeviceSolverPerf perf;
    for (int pass = 0; pass <= in.nNonOrthogonalCorrectors; ++pass)
    {
        const bool lastPass = (pass == in.nNonOrthogonalCorrectors);

        // constrainPressure, and the patches' updateCoeffs at this assembly
        hooks.pressureCoeffs(phiHbyAInt, phiHbyABnd, *in.rAUfAll, iC, bC);

        // the explicit correction: gradOf(p_rgh), laplacianCorrFlux, laplacianNonOrthSource
        DeviceBuffer<scalar> corrSource;
        if (in.correctedLaplacian)
        {
            DeviceBuffer<scalar> bval, gx, gy, gz;
            hooks.boundaryValues(bval);
            deviceGaussGrad(dm, p_rgh, bval, gx, gy, gz);
            // `limited <k>` for 0 < k < 1 (fvm.cuh laplacianCorrFlux); otherwise the correction unlimited
            if (in.snGradLimitCoeff > scalar(0) && in.snGradLimitCoeff < scalar(1))
            {
                deviceLaplacianCorrFluxLimited(dm, rAUfInt, p_rgh, gx, gy, gz, in.snGradLimitCoeff, ffc);
            }
            else
            {
                deviceLaplacianCorrFlux(dm, rAUfInt, gx, gy, gz, ffc);
            }
            deviceFaceDivSource(dm, ffc, corrSource);
        }

        deviceInterAssemblePEqn(dm, rAUfInt, phiHbyAInt, phiHbyABnd,
                                in.needReference, in.pRefCell, &p_rgh, P,
                                in.correctedLaplacian, in.correctedLaplacian ? &corrSource : nullptr);

        if (taps && pass == 0)
        {
            deviceCopy(taps->diag,   P.diag);
            deviceCopy(taps->upper,  P.upper);
            deviceCopy(taps->lower,  P.lower);
            deviceCopy(taps->source, P.source);
            deviceCopy(taps->iC,     iC);
            deviceCopy(taps->bC,     bC);
        }

        // fold the boundary in and solve, with the Final entry on the last pass only
        const DeviceAlphaSolverControls& sv = lastPass ? in.solve : in.solveInner;
        const bool pcgDIC = lastPass ? in.pcgDIC : in.pcgDICInner;
        const GamgControls* gamg = lastPass ? in.gamg : in.gamgInner;
        DeviceBuffer<scalar> diagC, b;
        deviceFold(dm, P.diag, P.source, iC, bC, diagC, b);
        const DeviceLduView A = deviceLduView(dm, diagC, P.upper, P.lower);
        if (gamg)
        {
            if (!in.dic || !in.gamgCache)
            {
                throw std::runtime_error(
                    "brae interFoam device pressure step: the case asks for GAMG and the caller handed in no "
                    "fine-level DIC schedule or no hierarchy cache. Build the first with buildDeviceDilu; "
                    "the second is a DeviceGamgCache that outlives the step.");
            }
            DeviceGamgHierarchy& hierarchy = in.gamgCache->get(gamg->nCellsInCoarsestLevel);
            perf = deviceGamgSolve(A, b, p_rgh, *in.dic, hierarchy, *gamg, in.gamgLog);
        }
        else if (pcgDIC)
        {
            if (!in.dic)
            {
                throw std::runtime_error(
                    "brae interFoam device pressure step: the case asks for PCG with DIC and no level "
                    "schedule was handed in. Build one from the mesh with buildDeviceDilu.");
            }
            const scalar nf = deviceNormFactor(A, p_rgh, b, deviceOnes(nC));
            perf = deviceDICPCG(A, b, p_rgh, nf, sv.tol, sv.relTol, sv.maxIter, 0, *in.dic);
        }
        else
        {
            DeviceBuffer<scalar> dNf;
            deviceNormFactorInto(A, p_rgh, b, deviceOnes(nC), dNf);
            perf = deviceJacobiBiCGStab(A, b, p_rgh, dNf.data(), sv.tol, sv.relTol, sv.maxIter);
        }
        if (in.solveLog)
        {
            in.solveLog->push_back(perf);
        }
        // p_rgh.correctBoundaryConditions() after every solve; after the last one the caller does it
        if (!lastPass && hooks.updateBoundary)
        {
            hooks.updateBoundary(p_rgh);
        }
    }

    // phi = phiHbyA - p_rghEqn.flux(). The flux is taken from the UNFOLDED matrix, because
    // fvMatrix::flux() reads upper/lower and the boundary coefficients, not the folded diagonal.
    DeviceBuffer<scalar> fluxInt, fluxBnd;
    deviceInterPEqnFlux(dm, P, iC, bC, p_rgh, fluxInt, fluxBnd, in.correctedLaplacian ? &ffc : nullptr);
    phiInt.resize(static_cast<std::size_t>(nIf));
    phiBnd.resize(static_cast<std::size_t>(nBf));
    if (nIf > 0)
    {
        subKernel<<<nBlocks(nIf), TPB>>>(phiHbyAInt.data(), fluxInt.data(), nIf, phiInt.data());
        ckS(cudaGetLastError(), "phi = phiHbyA - flux");
    }
    if (nBf > 0)
    {
        subKernel<<<nBlocks(nBf), TPB>>>(phiHbyABnd.data(), fluxBnd.data(), nBf, phiBnd.data());
        ckS(cudaGetLastError(), "phi = phiHbyA - flux, boundary");
    }

    // U = HbyA + rAU*reconstruct((phig - flux)/rAUf) -- the SAME phig, and the flux BEFORE the
    // division. rAUf at an uncoupled patch is the face cell's rAU, which is what the caller's full
    // array holds there.
    DeviceBuffer<scalar> ffInt(static_cast<std::size_t>(nIf)), ffBnd(static_cast<std::size_t>(nBf));
    DeviceBuffer<scalar> rAUfBnd(static_cast<std::size_t>(nBf));
    if (nIf > 0)
    {
        subKernel<<<nBlocks(nIf), TPB>>>(phigInt.data(), fluxInt.data(), nIf, ffInt.data());
        ckS(cudaGetLastError(), "phig - flux");
    }
    if (nBf > 0)
    {
        subKernel<<<nBlocks(nBf), TPB>>>(phigBnd.data(), fluxBnd.data(), nBf, ffBnd.data());
        ckS(cudaGetLastError(), "phig - flux, boundary");
        ckS(cudaMemcpy(rAUfBnd.data(), in.rAUfAll->data() + nIf, sizeof(scalar)*nBf,
                       cudaMemcpyDeviceToDevice), "rAUf, boundary");
    }
    deviceCorrectVelocity(dm, HbyAX, HbyAY, HbyAZ, rAU, ffInt, rAUfInt, ffBnd, rAUfBnd, UX, UY, UZ);

    // p = p_rgh + rho*gh, rebuilt from the SOLVED p_rgh and never carried.
    deviceStaticPressure(nC, p_rgh, *in.rho, *in.gh, p);

    // ...and on a case that needs a reference, p's LEVEL is then set and p_rgh rebuilt from it
    // (pEqn.H:74-83). The caller re-evaluates p_rgh's boundary after this step, as the host arm does.
    if (in.needReference)
    {
        deviceInterPressureReference(nC, in.pRefCell, in.pRefValue, *in.rho, *in.gh, p, p_rgh);
    }

    return perf.finalResidual;
}

} // namespace brae
