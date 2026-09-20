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

// phiHbyA += phig on the pair's faces, the interface twin of addPhigBoundaryKernel.
__global__ void addPhigIfKernel(const scalar* __restrict__ phig, int n, scalar* __restrict__ phiHbyA)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) phiHbyA[i] += phig[i];
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
    // THE PAIR, if the mesh has one, needs its own phiHbyA before anything here is meaningful: the
    // pressure source is fvc::div(phiHbyA), which sums a coupled face like any other (fvc.cu:548-550).
    if (in.cyc && in.cyc->n > 0 && !in.phiHbyAIf)
    {
        throw std::runtime_error(
            "brae interFoam device pEqn: the mesh has a periodic pair and no phiHbyA was given on it. "
            "gpu::pressurePredictor builds HbyA and its flux on the internal and boundary faces only; "
            "until it carries the pair, the pressure source would be missing those faces entirely. The "
            "host loop (no -device) carries it.");
    }

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

    // ...AND ON THE PAIR, the same expression on its own faces. rAUf there is fvc::interpolate(rAU) on
    // a coupled patch -- w*rAU[own] + (1-w)*rAU[nbr] -- which is what the host reference computes with
    // coupledLinear (inter_peqn_cpp.cu:624, fv_patch.cuh:119-126), and deviceCyclicFaceValue is that
    // same arithmetic. phig itself goes through deviceBuoyancyFlux, so the pair cannot drift from the
    // faces around it: one kernel, three arrays.
    const bool havePair = in.cyc && in.cyc->n > 0;
    DeviceBuffer<scalar> rAUfIf, phigIf, phiHbyAIfAll;
    if (havePair)
    {
        if (!in.stfIf || !in.ghfIf || !in.snGradRhoIf)
        {
            throw std::runtime_error(
                "brae interFoam device pEqn: the mesh has a periodic pair and its surface-tension force, "
                "gh or snGrad(rho) was not handed in. phig is a whole-surfaceScalarField expression "
                "(pEqn.H:28-36) and a coupled patch carries it like any other; without those three the "
                "pair's phiHbyA would be missing the buoyancy every other face has.");
        }
        deviceCyclicFaceValue(*in.cyc, rAU, rAUfIf);
        deviceBuoyancyFlux(in.cyc->n, *in.stfIf, *in.ghfIf, *in.snGradRhoIf, rAUfIf, in.cyc->magSf, phigIf);
        // phiHbyA += phig on the pair, as addPhigBoundaryKernel does it on the other patches. The
        // caller's array is const and is the predictor's own, so the sum lives here, like the host's
        // local phiHbyA.
        phiHbyAIfAll.resize(static_cast<std::size_t>(in.cyc->n));
        ckS(cudaMemcpy(phiHbyAIfAll.data(), in.phiHbyAIf->data(), sizeof(scalar)*in.cyc->n,
                       cudaMemcpyDeviceToDevice), "phiHbyA, interface");
        addPhigIfKernel<<<nBlocks(in.cyc->n), TPB>>>(phigIf.data(), in.cyc->n, phiHbyAIfAll.data());
        ckS(cudaGetLastError(), "phiHbyA += phig, interface");
        if (taps)
        {
            deviceCopy(taps->phigIf, phigIf);
            deviceCopy(taps->rAUfIf, rAUfIf);
        }
    }

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
        hooks.pressureCoeffs(phiHbyAInt, phiHbyABnd, *in.rAUfAll, rAU, iC, bC);

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
                                in.correctedLaplacian, in.correctedLaplacian ? &corrSource : nullptr,
                                in.cyc, &rAU, havePair ? &phiHbyAIfAll : nullptr);

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
        // ...and the periodic pair's off-diagonal, which deviceAmul applies as
        // Apsi[own] += ifCoeff*psi[nbr] (device_ldu.cuh:28-33). Without it the solve is a different
        // operator from the matrix that was assembled.
        const DeviceLduView A = (in.cyc && in.cyc->n > 0)
            ? deviceLduViewCyclic(dm, diagC, P.upper, P.lower, in.cyc->n, in.cyc->ownCell.data(),
                                  in.cyc->nbrCell.data(), in.cyc->ifCoeff.data())
            : deviceLduView(dm, diagC, P.upper, P.lower);
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
    // ...and on the PAIR: phi = phiHbyA - p_rghEqn.flux(), the same line, with the flux the solved
    // pressure leaves there (gated in tests/test_device_inter_peqn_cyclic_vs_host.cu). cyc->phi is the
    // pair's flux for everything downstream -- the alpha step reads it as phiCN -- so this is where it
    // is rewritten, exactly as phiInt and phiBnd are above.
    DeviceBuffer<scalar> ffIf;
    if (havePair)
    {
        ckS(cudaMemcpy(in.cyc->phi.data(), phiHbyAIfAll.data(),
                       sizeof(scalar)*in.cyc->n, cudaMemcpyDeviceToDevice), "phi = phiHbyA, interface");
        deviceCyclicCorrectFlux(*in.cyc, p_rgh);

        // ...and (phig - p_rghEqn.flux()) on the same faces, for the reconstruction. The flux is the
        // value deviceCyclicCorrectFlux just subtracted, taken through the shared arithmetic rather
        // than differenced back out of cyc->phi.
        DeviceBuffer<scalar> fluxIf;
        deviceCyclicPressureFlux(*in.cyc, p_rgh, fluxIf);
        ffIf.resize(static_cast<std::size_t>(in.cyc->n));
        subKernel<<<nBlocks(in.cyc->n), TPB>>>(phigIf.data(), fluxIf.data(), in.cyc->n, ffIf.data());
        ckS(cudaGetLastError(), "phig - flux, interface");
        if (taps)
        {
            deviceCopy(taps->ffIf, ffIf);
            deviceCopy(taps->phiIf, in.cyc->phi);
        }
    }

    deviceCorrectVelocity(dm, HbyAX, HbyAY, HbyAZ, rAU, ffInt, rAUfInt, ffBnd, rAUfBnd, UX, UY, UZ,
                          havePair ? in.cyc : nullptr,
                          havePair ? &ffIf : nullptr,
                          havePair ? &rAUfIf : nullptr);

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
