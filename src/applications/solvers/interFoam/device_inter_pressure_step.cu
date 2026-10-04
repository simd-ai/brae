// One pass of pEqn.H -- see device_inter_pressure_step.cuh for the three placements it owns.
#include "inter_phase_time.cuh"
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
#include <cstring>
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

// phiHbyA += phig on the pair's faces, the interface twin of addPhigBoundaryKernel -- and `a += b` on any
// face array, which is all fvc::makeAbsolute is.
__global__ void addPhigIfKernel(const scalar* __restrict__ phig, int n, scalar* __restrict__ phiHbyA)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) phiHbyA[i] += phig[i];
}

}   // namespace


// WHICH SIDE SOLVES A GAMG SYSTEM in the device loop. The case's smoother is one of OpenFOAM's sequential
// ones (DIC, the Gauss-Seidels), and the device runs it level-scheduled: one step per level of the mesh's
// dependency chain, on every multigrid level, and a coarse level's chain is nearly as long as the fine one's.
// MEASURED on RAS/DTCHull (845,536 cells, 891 dependency levels; coarse level 1 has 422,498 cells and 1252),
// two sweeps of the smoother: finest level 23.8 ms on the device and 11.4 ms on one CPU core, coarse level 1
// 23.8 and 6.5, the thirteen levels below it 93 and 16.7. The whole solve: 1548 ms a step on the device, 508
// on the host, the same 214 V-cycles over 25 steps and the same distance from OpenFOAM (p_rgh 2.8e-07).
// So the host solves it below GAMG_HOST_MAX_CELLS. ABOVE it the device does: the host's sweep grows with the
// cells and the device's with the chain, roughly their cube root, so the two cross -- the size is an
// estimate from that one measurement and the scaling, not a measured crossover. A coupled pair stays on the
// device, whose view carries the pair. BRAE_DEVICE_GAMG=host|device forces either.
constexpr label GAMG_HOST_MAX_CELLS = 2000000;

bool gamgSolveOnHost(const DeviceLduView& A)
{
    static const char* const forced = std::getenv("BRAE_DEVICE_GAMG");
    const bool pair = A.nCyc > 0 || A.nAmi > 0;
    bool host = !pair && A.nCells < GAMG_HOST_MAX_CELLS;
    if (forced && std::string(forced) == "device")
    {
        host = false;
    }
    if (forced && std::string(forced) == "host")
    {
        if (pair)
        {
            throw std::runtime_error(
                "brae interFoam (device): BRAE_DEVICE_GAMG=host on a mesh with a coupled pair. The host route "
                "takes a folded system with no pair; the device's GAMG carries it.");
        }
        host = true;
    }
    static bool announced = false;
    if (!announced)
    {
        announced = true;
        std::printf("  GAMG: solved on the %s (%ld cells; the case's smoother is sequential); "
                    "BRAE_DEVICE_GAMG=host|device forces either\n",
                    host ? "host" : "device", (long)A.nCells);
    }
    return host;
}

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
        phiHbyAIfAll.resize(static_cast<std::size_t>(in.cyc->n));
        ckS(cudaMemcpy(phiHbyAIfAll.data(), in.phiHbyAIf->data(), sizeof(scalar)*in.cyc->n,
                       cudaMemcpyDeviceToDevice), "phiHbyA, interface");

        // ...+ interpolate(rho*rAU)*fvc::ddtCorr(U, phi) ON THE PAIR, pEqn.H:16-18, which goes in
        // BEFORE phig exactly as it does on every other face. It is identically zero at the first
        // step of a case at rest -- U.oldTime() and phi.oldTime() are both zero -- and live from the
        // second: MEASURED on validation/interFoamCyclic without it, p_rgh 2.1e+01 of 1.6e+03 at step
        // two with alpha already exact to 4.2e-11, and 6.1e-06 at step one.
        // interpolate(rho*rAU) on a coupled face is the two cells' PRODUCT interpolated, which is what
        // rhoRAUfKernel does on an internal face (device_inter_peqn.cuh note 2).
        if (in.ddtCorrIf && static_cast<int>(in.ddtCorrIf->size()) == in.cyc->n)
        {
            DeviceBuffer<scalar> rhoRAUCell(static_cast<std::size_t>(nC));
            deviceHadamard(rhoRAUCell, *in.rho, rAU);
            DeviceBuffer<scalar> rhoRAUfIf;
            deviceCyclicFaceValue(*in.cyc, rhoRAUCell, rhoRAUfIf);
            DeviceBuffer<scalar> term(static_cast<std::size_t>(in.cyc->n));
            deviceHadamard(term, rhoRAUfIf, *in.ddtCorrIf);
            addPhigIfKernel<<<nBlocks(in.cyc->n), TPB>>>(term.data(), in.cyc->n, phiHbyAIfAll.data());
            ckS(cudaGetLastError(), "phiHbyA += rhoRAUf*ddtCorr, interface");
        }

        if (taps) deviceCopy(taps->phiHbyAIfPrePhig, phiHbyAIfAll);

        // phiHbyA += phig on the pair, as addPhigBoundaryKernel does it on the other patches. The
        // caller's array is const and is the predictor's own, so the sum lives here, like the host's
        // local phiHbyA.
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
    // pEqn.H:13-19: + interpolate(rho*rAU)*ddtCorr, then MRF.makeRelative
    deviceInterAddDdtCorrTerms(dm, rhoRAUf,
                               in.ddtCorrInt ? *in.ddtCorrInt : zeroIf, in.ddtCorrInt != nullptr,
                               phiHbyAInt, phiHbyABnd, in.mrf,
                               in.ddtCorrBnd, in.rhoBndFace, &rAU, in.bndUFixesValue);

    // pEqn.H:21-26 -- `if (p_rgh.needReference()) { fvc::makeRelative(phiHbyA, U);
    // adjustPhi(phiHbyA, U, p_rgh); fvc::makeAbsolute(phiHbyA, U); }` -- transcribed from the host's
    // pressureCorrector, the round trip included: fvc::makeRelative/makeAbsolute act on a MOVING mesh
    // only, on every face (fvcMeshPhi.C:76-86, :115-125), and (a - m) + m is not a no-op in floating
    // point, so it runs even where adjustPhi finds nothing to scale. This used to be REFUSED by name
    // wherever a U patch could leave adjustPhi something to weigh, and silently skipped elsewhere --
    // the balance test OpenFOAM aborts on (adjustPhi.C:108) included.
    // NOT DISCRIMINATED: this block's place BEFORE phig. On every fixture that reaches it phig is exactly
    // zero on the boundary faces adjustPhi sums (their alpha is zeroGradient, or inletOutlet over pure
    // air), and phiHbyAInt enters only totalFlux, orders from its thresholds -- so a block moved after
    // deviceInterAddPhig would pass them all. The order is OpenFOAM's (pEqn.H:21-26, then :36) and the
    // host's (inter_peqn_cpp.cu).
    if (in.needReference)
    {
        if (!hooks.adjustPhi)
            throw std::runtime_error(
                "brae interFoam device pEqn: p_rgh needs a reference and no adjustPhi hook was handed in. "
                "pEqn.H:21-26 balances the boundary flux there, or stops the run where it cannot.");
        const int nFaceAll = nIf + nBf;
        if (in.meshPhiAll && static_cast<int>(in.meshPhiAll->size()) != nFaceAll)
            throw std::runtime_error(
                "brae interFoam device pEqn: the mesh flux must cover the mesh's FULL face array "
                "(internal faces then the boundary patches in order), as phiHbyA does.");
        // the pair's phiHbyA would need the round trip too; a moving mesh with a pair is refused by the
        // driver, so this only states where that stops
        if (in.meshPhiAll && havePair)
            throw std::runtime_error(
                "brae interFoam device pEqn: a moving mesh with a periodic pair under a pressure "
                "reference -- the pair's phiHbyA takes no makeRelative/makeAbsolute round trip here.");
        if (in.meshPhiAll)
        {
            if (nIf > 0)
            {
                subKernel<<<nBlocks(nIf), TPB>>>(phiHbyAInt.data(), in.meshPhiAll->data(), nIf,
                                                 phiHbyAInt.data());
                ckS(cudaGetLastError(), "makeRelative(phiHbyA), internal");
            }
            if (nBf > 0)
            {
                subKernel<<<nBlocks(nBf), TPB>>>(phiHbyABnd.data(), in.meshPhiAll->data() + nIf, nBf,
                                                 phiHbyABnd.data());
                ckS(cudaGetLastError(), "makeRelative(phiHbyA), boundary");
            }
        }
        hooks.adjustPhi(phiHbyAInt, phiHbyABnd);
        if (in.meshPhiAll)
        {
            if (nIf > 0)
            {
                addPhigIfKernel<<<nBlocks(nIf), TPB>>>(in.meshPhiAll->data(), nIf, phiHbyAInt.data());
                ckS(cudaGetLastError(), "makeAbsolute(phiHbyA), internal");
            }
            if (nBf > 0)
            {
                addPhigIfKernel<<<nBlocks(nBf), TPB>>>(in.meshPhiAll->data() + nIf, nBf,
                                                       phiHbyABnd.data());
                ckS(cudaGetLastError(), "makeAbsolute(phiHbyA), boundary");
            }
        }
    }

    // phiHbyA BEFORE phig, which is where the host reference's tap is taken (PressureTaps::phiHbyA,
    // "BEFORE `phiHbyA += phig`") -- after ddtCorr, MRF and adjustPhi, as the host's is. It was taken
    // before ddtCorr here, so the two taps compared different quantities on any step past the first.
    if (taps)
    {
        deviceCopy(taps->phigIntTap, phigInt);
        // ...and phig's BOUNDARY half, the last term of div(phiHbyA + phig) this dump had not
        // compared. A 2D case's largest faces are its EMPTY ones, so a divergence that sums them
        // differs from one that does not by a lot.
        deviceCopy(taps->phigBndTap, phigBnd);
        deviceCopy(taps->phiHbyAIntPrePhig, phiHbyAInt);
        // ...AND ITS BOUNDARY HALF. The p_rgh source is div(phiHbyA), which sums the boundary faces
        // too, so comparing only the internal ones can show agreement while the source differs -- on
        // a moving mesh the wall flux is not zero, the wall moves.
        deviceCopy(taps->phiHbyABndPrePhig, phiHbyABnd);
    }
    // pEqn.H:36
    deviceInterAddPhig(dm, phigInt, phigBnd, phiHbyAInt, phiHbyABnd);

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
    // p_rgh's jump on the pair, refreshed by the hook at every assembly; empty = no jump there
    DeviceBuffer<scalar> cycJump;
    DeviceBuffer<scalar> ffc;
    // ...and the pair's share of it, kept for the flux after the solve as `ffc` is
    DeviceBuffer<scalar> ffcIf;
    const bool noPairNonOrth = std::getenv("BRAE_CONTROL_DEVICE_PAIR_NO_NONORTH") != nullptr;
    DeviceSolverPerf perf;
    for (int pass = 0; pass <= in.nNonOrthogonalCorrectors; ++pass)
    {
        const bool lastPass = (pass == in.nNonOrthogonalCorrectors);

        // constrainPressure, and the patches' updateCoeffs at this assembly
        hooks.pressureCoeffs(phiHbyAInt, phiHbyABnd, *in.rAUfAll, rAU, iC, bC, cycJump);
        const bool haveJump = havePair && static_cast<int>(cycJump.size()) == in.cyc->n;
        if (taps && haveJump)
        {
            std::vector<scalar> jh;
            cycJump.copyTo(jh);
            taps->jumpHistory.push_back(jh);
        }

        // the explicit correction: gradOf(p_rgh), laplacianCorrFlux, laplacianNonOrthSource
        DeviceBuffer<scalar> corrSource;
        if (in.correctedLaplacian)
        {
            DeviceBuffer<scalar> bval, gx, gy, gz;
            hooks.boundaryValues(bval);
            // THE CASE'S OWN gradSchemes ENTRY, through deviceGradOf. A leastSquares fit takes the pair
            // INSIDE it (deviceLeastSquaresGrad's `cyc` argument, leastSquaresVectors.C:131-140) where the
            // Gauss path adds the pair's faces AFTERWARDS -- so the two are wired differently below, and a
            // JUMP cyclic under leastSquares is refused by name rather than fitted without its jump.
            if (in.prghGradLeastSquares && havePair && haveJump)
            {
                throw std::runtime_error(
                    "brae interFoam (device): grad(p_rgh) is `leastSquares` and the mesh carries a JUMP "
                    "cyclic. The least-squares fit takes the neighbour cell's value directly, and there is "
                    "no place in it for the pair's jump, which the Gauss path subtracts face by face. "
                    "Refused rather than fit across the pair without its jump.");
            }
            deviceGradOf(dm, p_rgh, bval, in.prghGradLeastSquares, in.prghGradCellLimitK, gx, gy, gz,
                         (in.prghGradLeastSquares && havePair) ? in.cyc : nullptr);
            // ...and the PAIR's faces, which fvc::grad sums like any other patch's. On a jump cyclic
            // the neighbour value is the cell across LESS the jump, as everywhere else it appears.
            // ONLY ON THE GAUSS PATH: the least-squares fit above already took the pair through its own
            // argument, so adding it again here would count those faces twice.
            if (havePair && !in.prghGradLeastSquares)
            {
                deviceCyclicAddGrad(*in.cyc, p_rgh, dm.V, gx, gy, gz,
                                    haveJump ? &cycJump : nullptr);
            }
            // ...and the LIMITER across a pair, which deviceGradOf cannot do: cellLimitedGrad needs the
            // pair's faces as a CellLimitInterface list (device_mesh.cuh) and deviceGradOf calls the
            // no-interface overload, so a limited gradient on a cell touching the pair would be limited
            // against its own cells only. Refused by name rather than limited on a partial neighbourhood.
            if (havePair && in.prghGradCellLimitK > scalar(0))
            {
                throw std::runtime_error(
                    "brae interFoam (device): grad(p_rgh) is `cellLimited` and the mesh carries a cyclic "
                    "pair. The limiter needs the pair's faces in its neighbourhood (CellLimitInterface) "
                    "and this site limits without them. Refused rather than limit against a partial "
                    "neighbourhood.");
            }
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
            // ...AND ON THE PAIR'S FACES (fvm.cuh, laplacianCorrFluxCoupled; inter_peqn_cpp.cu:949-953):
            // the source takes them like any owner-side face -- NEGATED, as deviceFaceDivSource leaves
            // the internal faces' (the host's `src[own] += ffc`, subtracted) -- and the flux below
            // takes them back
            if (havePair && !noPairNonOrth)
            {
                const bool jumpHere = static_cast<int>(cycJump.size()) == in.cyc->n;
                deviceCyclicLapCorrFlux(*in.cyc, rAU, p_rgh, gx, gy, gz, in.snGradLimitCoeff,
                                        jumpHere ? &cycJump : nullptr, ffcIf);
                deviceCyclicAddToOwner(*in.cyc, ffcIf, scalar(-1), corrSource);
            }
            if (taps && pass == 0) deviceCopy(taps->nonOrthSource, corrSource);
        }

        deviceInterAssemblePEqn(dm, rAUfInt, phiHbyAInt, phiHbyABnd,
                                in.needReference, in.pRefCell, &p_rgh, P,
                                in.correctedLaplacian, in.correctedLaplacian ? &corrSource : nullptr,
                                in.cyc, &rAU, havePair ? &phiHbyAIfAll : nullptr,
                                (taps && pass == 0) ? &taps->divPhiHbyA : nullptr,
                                in.nonOrthCoeffs);

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
        const GamgPreconditionerControls* pcgGamg = lastPass ? in.pcgGamg : in.pcgGamgInner;
        DeviceBuffer<scalar> diagC, b;
        deviceFold(dm, P.diag, P.source, iC, bC, diagC, b);
        // ...and the periodic pair's off-diagonal, which deviceAmul applies as
        // Apsi[own] += ifCoeff*psi[nbr] (device_ldu.cuh:28-33). Without it the solve is a different
        // operator from the matrix that was assembled.
        const DeviceLduView A = (in.cyc && in.cyc->n > 0)
            ? deviceLduViewPair(dm, diagC, P.upper, P.lower, *in.cyc,
                                haveJump ? cycJump.data() : nullptr)
            : deviceLduView(dm, diagC, P.upper, P.lower);
        // THE PRESSURE RULE: brae's AMG-PCG whatever the entry names, with the entry's own stopping controls.
        // BRAE_PRESSURE_CASE_SOLVER=1 runs the case's own entry as ported instead -- what the exact gates
        // against OpenFOAM run (CMakeLists.txt sets it for every test; the user's decisions, 2026-10-03), so
        // they keep checking everything else to 1e-11 while the AMG-PCG is held to its own measured bounds
        // (tests/interfoam_write/amg_pcg/). BRAE_CONTROL_AMG_PCG_ONE_ITERATION=1 stops each AMG-PCG solve
        // after one iteration -- those gates' control.
        static const bool caseSolver = std::getenv("BRAE_PRESSURE_CASE_SOLVER") != nullptr;
        static const bool oneIteration = std::getenv("BRAE_CONTROL_AMG_PCG_ONE_ITERATION") != nullptr;
        if (!caseSolver)
        {
            if (!in.amgPcg)
            {
                throw std::runtime_error(
                    "brae interFoam device pressure step: p_rgh runs brae's AMG-preconditioned PCG and the "
                    "caller handed in no DeviceAmgPcgCache.");
            }
            static bool announced = false;
            if (!announced)
            {
                announced = true;
                const char* asked = gamg ? "GAMG" : pcgGamg ? "PCG with a GAMG preconditioner"
                                  : pcgDIC ? "PCG with DIC" : "its own solver";
                std::printf("  p_rgh: system/fvSolution asks for %s; brae runs its AMG-preconditioned PCG "
                            "instead -- same operator, different Krylov method and iteration count\n", asked);
            }
            interPhase::Nested timed("pressure: AMG-PCG solve");
            AMGData& amg = in.amgPcg->get(dm.addressingId);
            amgGalerkin(amg, diagC, P.upper, P.lower);
            const scalar nf = deviceNormFactor(A, p_rgh, b, deviceOnes(nC));
            const scalar tol = gamg ? gamg->tolerance : sv.tol;
            scalar relTol = gamg ? gamg->relTol : sv.relTol;
            const int maxIter = oneIteration ? 1 : (gamg ? gamg->maxIter : sv.maxIter);
            const int minIter = gamg ? gamg->minIter : 0;
            // the fast path's knobs (amgPcgKnobs); the graph not across a coupled pair
            const AmgPcgKnobs& knobs = amgPcgKnobs();
            // a non-final solve the Final one follows, on an entry that names PCG with DIC, stops at the knob's
            // relTol where its own is looser (AmgPcgKnobs::dicInnerRelTol has the measurement); the Final
            // entry is never touched. BRAE_CONTROL_DIC_INNER_EVERY_CORRECTOR=1 caps every non-final solve --
            // what the cap was first measured as, kept to re-measure what the narrower scope saves.
            static const bool everyCorrector = std::getenv("BRAE_CONTROL_DIC_INNER_EVERY_CORRECTOR") != nullptr;
            const bool finalPass = lastPass && in.finalEntry;
            const bool beforeFinal = in.finalEntry ? !lastPass : in.correctorBeforeFinal;
            if (!finalPass && (beforeFinal || everyCorrector) && pcgDIC
             && knobs.dicInnerRelTol >= scalar(0) && relTol > knobs.dicInnerRelTol)
            {
                static bool capAnnounced = false;
                if (!capAnnounced)
                {
                    capAnnounced = true;
                    std::printf("  p_rgh: the solve before the Final one names PCG with DIC at relTol %g; the "
                                "AMG-PCG stops it at relTol %g -- stopped at the entry's own it leaves its error "
                                "at the interface, where it becomes air velocity; "
                                "BRAE_PRESSURE_DIC_INNER_RELTOL=case keeps the entry's own\n",
                                static_cast<double>(relTol), static_cast<double>(knobs.dicInnerRelTol));
                }
                relTol = knobs.dicInnerRelTol;
            }
            const bool graph = knobs.graph && !(in.cyc && in.cyc->n > 0);
            perf = deviceAMGPCG(A, amg, b, p_rgh, nf, tol, relTol, maxIter, graph, knobs.checkEvery,
                                knobs.corrScaling, minIter);
        }
        else if (gamg)
        {
            if (!in.dic || !in.gamgCache)
            {
                throw std::runtime_error(
                    "brae interFoam device pressure step: the case asks for GAMG and the caller handed in no "
                    "fine-level DIC schedule or no hierarchy cache. Build the first with buildDeviceDilu; "
                    "the second is a DeviceGamgCache that outlives the step.");
            }
            DeviceGamgHierarchy& hierarchy = in.gamgCache->get(gamg->nCellsInCoarsestLevel);
            interPhase::Nested timed("pressure: GAMG solve");
            // WHERE THE SOLVE RUNS. GAMG's smoothers here are OpenFOAM's sequential ones, and a sequential
            // sweep on the device is one step per level of the mesh's dependency chain whatever the level's
            // size -- see gamgSolveOnHost. On the host route the folded system comes down, the host's GAMG
            // (the reference the device's is held to) solves it, and p_rgh goes back.
            if (gamgSolveOnHost(A))
            {
                std::vector<scalar> hDiag;
                std::vector<scalar> hUpper;
                std::vector<scalar> hSource;
                std::vector<scalar> hPsi;
                {
                    interPhase::Nested timedDown("host gamg: bring the system down");
                    hDiag = diagC.host();
                    hUpper = P.upper.host();
                    hSource = b.host();
                    hPsi = p_rgh.host();
                }
                const SolverPerformance hp =
                    gamgSolveFolded(*hierarchy.host, hDiag, hUpper, hSource, hPsi, *gamg, in.gamgLog);
                p_rgh.copyFrom(hPsi);
                perf.initialResidual = hp.initialResidual;
                perf.finalResidual = hp.finalResidual;
                perf.nIterations = hp.nIterations;
            }
            else
            {
                perf = deviceGamgSolve(A, b, p_rgh, *in.dic, hierarchy, *gamg, in.gamgLog);
            }
        }
        else if (pcgGamg)
        {
            if (!in.dic || !in.gamgCache)
            {
                throw std::runtime_error(
                    "brae interFoam device pressure step: the case asks for PCG with a GAMG "
                    "preconditioner and the caller handed in no fine-level DIC schedule or no "
                    "hierarchy cache. Build the first with buildDeviceDilu; the second is a "
                    "DeviceGamgCache that outlives the step.");
            }
            // the hierarchy's coarsest level comes from the SUB-DICTIONARY's entry, as the host's
            // pcgGamgSolve reads it there -- and, like every other GAMG solve of the run, it is the
            // mesh's one hierarchy and is built by whichever solve needs it first
            DeviceGamgHierarchy& hierarchy =
                in.gamgCache->get(pcgGamg->gamg.nCellsInCoarsestLevel);
            perf = devicePcgGamgSolve(A, b, p_rgh, *in.dic, hierarchy, sv.tol, sv.relTol, sv.maxIter,
                                      0, *pcgGamg, in.gamgLog);
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
        const bool jumpNow = static_cast<int>(cycJump.size()) == in.cyc->n;
        deviceCyclicCorrectFlux(*in.cyc, p_rgh, jumpNow ? &cycJump : nullptr);

        // ...and (phig - p_rghEqn.flux()) on the same faces, for the reconstruction. The flux is the
        // value deviceCyclicCorrectFlux just subtracted, taken through the shared arithmetic rather
        // than differenced back out of cyc->phi.
        DeviceBuffer<scalar> fluxIf;
        deviceCyclicPressureFlux(*in.cyc, p_rgh, fluxIf, jumpNow ? &cycJump : nullptr);
        // fvMatrix::flux() adds faceFluxCorrection on the coupled patches too (fv_matrix_ops.cuh:66-70)
        if (in.correctedLaplacian && static_cast<int>(ffcIf.size()) == in.cyc->n)
        {
            deviceAxpy(scalar(1), ffcIf, fluxIf);
            deviceAxpy(scalar(-1), ffcIf, in.cyc->phi);
        }
        ffIf.resize(static_cast<std::size_t>(in.cyc->n));
        subKernel<<<nBlocks(in.cyc->n), TPB>>>(phigIf.data(), fluxIf.data(), in.cyc->n, ffIf.data());
        ckS(cudaGetLastError(), "phig - flux, interface");
        if (taps)
        {
            deviceCopy(taps->ffIf, ffIf);
            deviceCopy(taps->phiIf, in.cyc->phi);
            deviceCopy(taps->cycJumpTap, cycJump);
        }
    }

    deviceCorrectVelocity(dm, HbyAX, HbyAY, HbyAZ, rAU, ffInt, rAUfInt, ffBnd, rAUfBnd, UX, UY, UZ,
                          havePair ? in.cyc : nullptr,
                          havePair ? &ffIf : nullptr,
                          havePair ? &rAUfIf : nullptr);

    // THE ABSOLUTE FLUX, handed back before the line below makes it relative. pEqn.H runs
    // fvc::correctUf(Uf, U, phi) at :66 and fvc::makeRelative(phi, U) at :69, in that order, so the
    // normal component correctUf writes into Uf is the ABSOLUTE flux's. Uf is a host field, so the
    // driver makes that call -- and it must not read the flux this step leaves behind.
    if (in.phiAbsIntOut) deviceCopy(*in.phiAbsIntOut, phiInt);
    if (in.phiAbsBndOut) deviceCopy(*in.phiAbsBndOut, phiBnd);
    if (in.phiAbsIfOut && havePair) deviceCopy(*in.phiAbsIfOut, in.cyc->phi);

    // fvc::makeRelative(phi, U) ON A MOVING MESH -- phi -= meshPhi, fvcMeshPhi.C:76, at the point the
    // host reference does it (inter_peqn_cpp.cu:980): after the velocity correction and before p is
    // rebuilt. The flux that leaves here is the RELATIVE one, which is what the next alpha equation
    // convects with and what the continuity error is measured on. Without it this loop's phi stayed
    // ABSOLUTE: measured on sloshingTank2D, worst |div(phi)| 1.573e-01 against the host's 9.173e-07.
    if (in.meshPhiAll)
    {
        const int nFaceAll = nIf + nBf;
        if (static_cast<int>(in.meshPhiAll->size()) != nFaceAll)
            throw std::runtime_error(
                "brae interFoam device pEqn: the mesh flux must cover the mesh's FULL face array "
                "(internal faces then the boundary patches in order), as phi does.");
        if (nIf > 0)
        {
            subKernel<<<nBlocks(nIf), TPB>>>(phiInt.data(), in.meshPhiAll->data(), nIf, phiInt.data());
            ckS(cudaGetLastError(), "makeRelative(phi), internal");
        }
        if (nBf > 0)
        {
            subKernel<<<nBlocks(nBf), TPB>>>(phiBnd.data(), in.meshPhiAll->data() + nIf, nBf,
                                             phiBnd.data());
            ckS(cudaGetLastError(), "makeRelative(phi), boundary");
        }
        // ...and on the PAIR, whose faces move with the mesh like any other: a cyclicAMI's rotor side
        // sweeps a volume wherever the snapped surface is not the surface of revolution it approximates
        if (havePair)
        {
            if (!in.meshPhiIf || static_cast<int>(in.meshPhiIf->size()) != in.cyc->n)
                throw std::runtime_error(
                    "brae interFoam device pEqn: a moving mesh with a coupled pair and no mesh flux on the "
                    "pair's faces. fvc::makeRelative subtracts it there as on every other face.");
            if (std::getenv("BRAE_CONTROL_DEVICE_PAIR_PHI_ABSOLUTE") == nullptr)
            {
                deviceAxpy(scalar(-1), *in.meshPhiIf, in.cyc->phi);
            }
        }
    }

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

const AmgPcgKnobs& amgPcgKnobs()
{
    static const AmgPcgKnobs knobs = []()
    {
        AmgPcgKnobs k;
        const char* every = std::getenv("BRAE_PCG_CHECK_EVERY");
        if (every && std::atoi(every) >= 1)
        {
            k.checkEvery = std::atoi(every);
        }
        const char* graph = std::getenv("BRAE_USE_GRAPH");
        if (graph)
        {
            k.graph = std::atoi(graph) != 0;
        }
        const char* scaling = std::getenv("BRAE_CORR_SCALING");
        if (scaling)
        {
            k.corrScaling = std::atoi(scaling) != 0;
        }
        const char* inner = std::getenv("BRAE_PRESSURE_DIC_INNER_RELTOL");
        if (inner && std::string(inner) == "case")
        {
            k.dicInnerRelTol = scalar(-1);
        }
        else if (inner)
        {
            char* end = nullptr;
            const double cap = std::strtod(inner, &end);
            if (end == inner || *end != '\0' || !(cap >= 0.0))
            {
                throw std::runtime_error(
                    "brae interFoam: BRAE_PRESSURE_DIC_INNER_RELTOL is `" + std::string(inner)
                    + "`; it is `case` or a relative tolerance that is not negative.");
            }
            k.dicInnerRelTol = static_cast<scalar>(cap);
        }
        return k;
    }();
    return knobs;
}

AMGData deviceAmgPcgHierarchy(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::string& caseDir,
    bool disk,
    AmgHierarchyMemo* memo,
    bool smoothed)
{
    interPhase::Nested timed("pressure: AMG hierarchy (load, build or copy)");
    static const bool cacheOff = []()
    {
        const char* e = std::getenv("BRAE_AMG_CACHE");
        return e && std::atoi(e) == 0;
    }();
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    const bool useDisk = disk && !cacheOff && !caseDir.empty();
    const std::vector<label> own(m.owner().begin(), m.owner().begin() + nIf);
    const std::vector<label> nei(m.neighbour().begin(), m.neighbour().begin() + nIf);
    const std::vector<scalar> w(g.magSf().begin(), g.magSf().begin() + nIf);
    if (useDisk)
    {
        // THE START MESH'S HIERARCHY, FROM THE CASE'S CACHE where the file there is this mesh's and this build's
        // (buildOrLoadAMG, keyed on the owner, neighbour and weights' content) -- the plain one and the smoothed
        // one each in a file of its own. Built and written otherwise.
        AMGCacheRead how = AMGCacheRead::absent;
        const std::string dir = caseDir + "/constant/polyMesh";
        AMGData A = buildOrLoadAMG(own, nei, w, nC, dir, true, smoothed ? &smoothed : nullptr, &how);
        const char* kind = smoothed ? "smoothed-aggregation " : "plain ";
        if (how == AMGCacheRead::loaded)
        {
            std::printf("  AMG hierarchy: the %sone is read from %s (this mesh's, written by this build); "
                        "BRAE_AMG_CACHE=0 builds it\n", kind, amgCachePath(dir, smoothed).c_str());
        }
        else if (how == AMGCacheRead::otherMeshOrBuild)
        {
            std::printf("  AMG hierarchy: the %sone at %s is another mesh's or another build's: built here and "
                        "the file rewritten\n", kind, amgCachePath(dir, smoothed).c_str());
        }
        else if (how == AMGCacheRead::unreadable)
        {
            std::printf("  AMG hierarchy: the %sone at %s could not be read whole: built here and the file "
                        "rewritten\n", kind, amgCachePath(dir, smoothed).c_str());
        }
        return A;
    }
    // ONE BUILD A CHANGED MESH (AmgHierarchyMemo): the second asker takes a copy of the first one's structure
    static const bool rebuilt = std::getenv("BRAE_CONTROL_AMG_HIERARCHY_REBUILT") != nullptr;
    static const bool check = std::getenv("BRAE_CONTROL_AMG_HIERARCHY_CHECK") != nullptr;
    static const bool stale = std::getenv("BRAE_CONTROL_AMG_HIERARCHY_STALE") != nullptr;
    // (the memo is the plain hierarchy's: pcorr alone asks for a smoothed one)
    const bool remember = memo && !smoothed && !rebuilt;
    if (remember && memo->held
     && (stale || (memo->nCells == nC && memo->owner == own && memo->neighbour == nei
                && memo->weights.size() == w.size()
                && std::memcmp(memo->weights.data(), w.data(), w.size()*sizeof(scalar)) == 0)))
    {
        static bool said = false;
        if (!said)
        {
            said = true;
            std::printf("  AMG hierarchy: a changed mesh's is built once and copied to the second solve that "
                        "asks; BRAE_CONTROL_AMG_HIERARCHY_REBUILT=1 builds it for each\n");
            if (stale)
            {
                std::printf("  *** CONTROL MODE: the held hierarchy is handed out without asking whether it "
                            "is this mesh's. This run is deliberately wrong. ***\n");
            }
        }
        // THE SECOND ASKER TAKES THE KEPT COPY ITSELF: two solves ask after a change, pcorr's and p_rgh's, so
        // the copy made for the first one is the only one a change needs. A third asker would build.
        interPhase::Nested timedCopy("pressure: AMG hierarchy handed to the second solve");
        AMGData copy = std::move(memo->structure);
        memo->structure = AMGData{};
        memo->held = false;
        if (check)
        {
            const AMGData fresh = buildAMG(own, nei, w, nC, smoothed ? &smoothed : nullptr);
            const char* what = firstAMGDifference(copy, fresh);
            if (what)
            {
                throw std::runtime_error(
                    std::string("brae interFoam: BRAE_CONTROL_AMG_HIERARCHY_CHECK: of the copied hierarchy, ")
                    + what + " is not what building it for this mesh gives.");
            }
        }
        return copy;
    }
    AMGData built = buildAMG(own, nei, w, nC, smoothed ? &smoothed : nullptr);
    if (remember)
    {
        interPhase::Nested timedKeep("hierarchy: the copy kept for the second solve (cloneAMG)");
        memo->structure = cloneAMG(built);
        memo->held = true;
        memo->nCells = nC;
        memo->owner = own;
        memo->neighbour = nei;
        memo->weights = w;
    }
    return built;
}

AMGData& DeviceAmgPcgCache::get(unsigned long long id)
{
    if (!mesh || !geometry)
    {
        throw std::runtime_error("brae DeviceAmgPcgCache: the mesh and geometry were not set before the first solve.");
    }
    if (!built || addressingId != id)
    {
        // the disk cache is the START mesh's: a rebuild after a refinement is another mesh
        amg = deviceAmgPcgHierarchy(*mesh, *geometry, caseDir, !built, memo);
        built = true;
        addressingId = id;
    }
    return amg;
}

} // namespace brae
