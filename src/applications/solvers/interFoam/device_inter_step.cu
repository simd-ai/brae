// One whole interFoam time step -- see device_inter_step.cuh for the loop order and what it decides.
#include "device_inter_step.cuh"
#include "device_alpha_flux.cuh"
#include "device_blas.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"   // deviceSymGaussSeidel
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace brae {
namespace {

// A STAGE PROBE, off unless BRAE_INTER_STEP_CHECK is set. One whole step is a dozen operators and a
// field that has gone non-finite at the first of them looks exactly like one that went at the last; a
// gate downstream sees only the end. This names the first stage whose output is not finite, which is
// what turns "damBreak gives NaN" into a line number. It is the of-instrument approach applied to
// brae's own code rather than to OpenFOAM's.
bool stepCheckOn()
{
    static const bool on = (std::getenv("BRAE_INTER_STEP_CHECK") != nullptr);
    return on;
}

void probe(const char* stage, const DeviceBuffer<scalar>& b)
{
    if (!stepCheckOn() || b.size() == 0) return;
    std::vector<scalar> h;
    b.copyTo(h);
    int bad = 0;
    scalar mx = 0;
    for (scalar v : h)
    {
        if (!std::isfinite(v)) ++bad;
        else mx = std::fmax(mx, std::fabs(v));
    }
    std::fprintf(stderr, "  [step] %-22s n=%-7zu non-finite=%-6d max|.|=%.6g\n",
                 stage, h.size(), bad, (double)mx);
}

}   // namespace

void deviceInterStep(
    const DeviceMesh&                dm,
    scalar                           deltaT,
    const DeviceInterStepControls&   ctl,
    const DevicePhaseProperties&     props,
    const DeviceInterStepHooks&      hooks,
    const DeviceBuffer<scalar>&      gh,
    const DeviceBuffer<scalar>&      ghf,
    const DeviceBuffer<scalar>&      magSf,
    DeviceBuffer<scalar>&            alpha1,
    const DeviceBuffer<scalar>&      alpha1Old,
    DeviceBuffer<scalar>&            UX,
    DeviceBuffer<scalar>&            UY,
    DeviceBuffer<scalar>&            UZ,
    const DeviceBuffer<scalar>&      UOldX,
    const DeviceBuffer<scalar>&      UOldY,
    const DeviceBuffer<scalar>&      UOldZ,
    DeviceBuffer<scalar>&            phiInt,
    DeviceBuffer<scalar>&            phiBnd,
    const DeviceBuffer<scalar>&      phiOldInt,
    const DeviceBuffer<scalar>&      phiOldBnd,
    const DeviceBuffer<int>&         bndUFixesValue,
    DeviceBuffer<scalar>&            p_rgh,
    DeviceBuffer<scalar>&            p,
    DeviceBuffer<scalar>&            nHatfInt,
    DeviceBuffer<scalar>&            nHatfBnd,
    DeviceBuffer<scalar>&            alpha1Bnd,
    DeviceBuffer<scalar>&            K,
    const DeviceBuffer<int>&         bndAlphaFixesValue,
    const DeviceBuffer<int>&         bndAlphaFlag,
    DeviceVectorBoundary&            dbU,
    DeviceBuffer<scalar>&            rho,
    DeviceBuffer<scalar>&            mu,
    DeviceBuffer<scalar>&            nu,
    DeviceBuffer<scalar>&            rhoPhiInt,
    DeviceBuffer<scalar>&            rhoPhiBnd,
    DeviceInterStepTaps*             taps)
{
    if (!hooks.updateUBoundary || !hooks.interfaceForces)
        throw std::runtime_error(
            "brae interFoam device step: the U-boundary and interface-force hooks are both required. "
            "Both are per-patch host work -- see device_inter_step.cuh -- and running without them "
            "would build the momentum equation on a boundary and a surface tension frozen at the "
            "start of the run.");

    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nBf = dm.nBndFaces;
    const int nFaces = nIf + nBf;

    // 1. THE ALPHA EQUATION, FIRST
    // interFoam.C:96-104. It leaves rhoPhi behind, and mixture.correct() at its end leaves rho, mu and
    // nu -- so the momentum equation below is built on the NEW density, not on last step's.
    // the caller's cAlpha, deltaN and schemes; the flux and the time step are the step's own.
    DeviceAlphaStepInput ain = ctl.alphaInput;
    ain.phiInt   = &phiInt;
    ain.phiBnd   = &phiBnd;
    ain.phiCNInt = &phiInt;                 // Euler: phiCN IS phi
    ain.phiCNBnd = &phiBnd;
    ain.rho1 = props.rho1;
    ain.rho2 = props.rho2;

    DeviceBuffer<scalar> alpha2;
    DeviceBuffer<scalar> alpha2Bnd;
    DeviceInterAlphaControls actl = ctl.alpha;
    actl.alpha2BndOut = &alpha2Bnd;
    deviceInterAlphaStep(dm, alpha1, alpha1Old, deltaT, ain, ctl.mules, actl, props, hooks.alpha,
                         alpha1Bnd, nHatfBnd, bndAlphaFixesValue, bndAlphaFlag,
                         nHatfInt, K, rhoPhiInt, rhoPhiBnd, alpha2, rho, mu, nu);

    probe("alpha", alpha1);
    probe("rho", rho);
    probe("rhoPhi", rhoPhiInt);

    // 2. THE INTERFACE FORCES, from the field the alpha step just left
    // surfaceTensionForce() and snGrad(rho) both read the NEW alpha, and both equations below read
    // them. Building them before the alpha step would apply last step's interface.
    // THE BOUNDARY MIXTURE COMES FROM ALPHA'S PATCH VALUES, not from the face cell's. Those are
    // different fields at a contact-angle wall -- alpha's patch value is patchInternalField +
    // gradient/deltaCoeffs, and the contact angle's gradient is what pulls the interface up it. Taking
    // the cell value instead gives an AIR viscosity on a face the interface has climbed, and
    // divDevRhoReff's laplacian is built from exactly that. Measured on capillaryRise against
    // OpenFOAM's own UEqn.A(): exact in all 3200 water cells, up to 56% low in the air cells AT THE
    // WALL, which is where the contact line is. deviceMixtureCorrect is the same fused kernel the
    // cells use, applied to the patch values.
    // ...and from ALPHA2's patch values for the rho2 half, which are one contact-angle pass older than
    // alpha1's -- see deviceBoundaryRho.
    DeviceBuffer<scalar> rhoBnd(static_cast<std::size_t>(nBf));
    if (nBf > 0)
    {
        if (alpha2Bnd.size() != static_cast<std::size_t>(nBf))
        {
            throw std::runtime_error(
                "brae interFoam device step: the alpha step assigned no alpha2 patch values, so rho's "
                "boundary cannot be blended. nAlphaCorr must be at least 1.");
        }
        deviceBoundaryRho(alpha1Bnd.data(), alpha2Bnd.data(), nBf, props, rhoBnd.data());
    }

    // rhoBnd is built ABOVE the hook because the hook reads it too: snGrad(rho) on a patch is
    // deltaCoeffs*(rho_b - rho_cell), and rho_b is this.
    DeviceBuffer<scalar> stf, snGradRho, nuEffCell, nuEffBnd, snGradPrgh;
    hooks.interfaceForces(alpha1, K, rho, rhoBnd, stf, snGradRho, nuEffCell, nuEffBnd, snGradPrgh);
    // nuEff = nut + nu where the closure is the device's -- see DeviceInterStepControls::nutCell
    if (ctl.nutCell || ctl.nutBnd)
    {
        if (!ctl.nutCell || !ctl.nutBnd || ctl.nutCell->size() != nuEffCell.size()
         || ctl.nutBnd->size() != nuEffBnd.size())
        {
            throw std::runtime_error(
                "brae interFoam device step: nut must be given on cells AND boundary faces, each the "
                "size of the viscosity it is added to.");
        }
        deviceAxpy(scalar(1), *ctl.nutCell, nuEffCell);
        deviceAxpy(scalar(1), *ctl.nutBnd, nuEffBnd);
    }

    probe("stf", stf);
    probe("snGradRho", snGradRho);
    probe("nuEffCell", nuEffCell);

    // 3. THE MOMENTUM MATRIX
    DeviceBuffer<scalar> ub[3];
    hooks.updateUBoundary(UX, UY, UZ, dbU, ub);
    // pressureInletOutletVelocity's updateCoeffs, from the flux registered NOW: an inflow face fixes
    // its tangential components, an outflow face is zeroGradient. The hook rebuilds dbU from the host
    // patches' categories, which carry no flux, so without this every such face stayed zeroGradient for
    // the whole run -- which agreed with the host only while the host made the same omission.
    // ...and inletOutlet's, FIRST, in the order rhoSimpleFoam's device arm takes them
    // (rhoUEqn.cuh:72-79): the io switch off the flux registered now, then the switches that read this
    // iteration's cell velocity. The hook rebuilds dbU from the host patches, and buildDeviceVectorBoundary
    // seeds an inletOutlet as fixedValue at its inletValue on EVERY face (device_boundary.cuh, category 3),
    // so without this the patch is a wall at the inlet value for the whole run.
    deviceUpdateInletOutlet(dbU, phiBnd);
    deviceUpdatePressureInletOutletVelocity(dbU, phiBnd, UX, UY, UZ, /*directionMixed=*/true);

    DeviceBuffer<scalar> muCell, muFace, muBndFace;
    deviceInterMuEff(dm, rho, nuEffCell, rhoBnd, nuEffBnd, muCell, muFace, muBndFace);

    DeviceBuffer<scalar> rhoOld;
    deviceCopy(rhoOld, rho);
    {
        // rho.oldTime() is the mixture at the PREVIOUS alpha, which is what the ddt source takes. It is
        // rebuilt here from alpha1Old rather than carried, because the two differ by the density ratio
        // in every cell the interface crossed and a carried copy is one more thing to get stale.
        rhoOld.resize(static_cast<std::size_t>(nC));
        deviceMixtureCorrect(alpha1Old.data(), nC, props, nullptr, rhoOld.data(), nullptr, nullptr);
    }

    // U's STORED boundary values, filled by the hook from the host's evaluate -- see the hook's own
    // comment for why deviceBCValue is not a substitute.
    const DeviceBuffer<scalar>* ubPtr[3] = {&ub[0], &ub[1], &ub[2]};

    gpu::MomentumInput uin;
    uin.phiInt        = &rhoPhiInt;        // the MASS flux out of the alpha equation
    uin.phiBnd        = &rhoPhiBnd;
    uin.nuEffCell     = &muCell;           // mu = rho*nuEff, the rho-weighted overload
    uin.nuEffFace     = &muFace;
    uin.nuEffBndFace  = &muBndFace;
    uin.scheme            = ctl.divScheme;
    uin.schemeCoeff       = ctl.divSchemeCoeff;
    uin.linearUpwind      = (ctl.divScheme == brae::cpu::DivScheme::linearUpwind);
    uin.gradULimitK       = ctl.gradULimitK;
    // the viscous laplacian under the case's laplacianSchemes, as the host's assembleUEqn hands
    // addDivDevReff (inter_ueqn_cpp.cu): nonOrthDeltaCoeffs and the deferred correction when `corrected`,
    // capped per face under `limited`. The shared assembler carries both; interFoam's step left them off.
    uin.correctedLaplacian = ctl.correctedLaplacian;
    uin.snGradLimitCoeff   = ctl.snGradLimitCoeff;
    uin.gradUSchemeLimitK = ctl.gradUSchemeLimitK;
    uin.relaxU        = ctl.relaxU;
    uin.relaxEquation = ctl.relaxEquationU;
    uin.ddtRho        = &rho;
    uin.ddtRhoOld     = &rhoOld;
    uin.ddtUOld[0]    = &UOldX;
    uin.ddtUOld[1]    = &UOldY;
    uin.ddtUOld[2]    = &UOldZ;
    uin.ddtDeltaT     = deltaT;
    uin.UbStored      = ubPtr;

    probe("muCell", muCell);
    probe("muFace", muFace);
    probe("rhoOld", rhoOld);
    if (taps) deviceCopy(taps->ddtRhoOld, rhoOld);

    gpu::MomentumMatrix UEqn;
    gpu::assembleUEqn(UEqn, dm, dbU, UX, UY, UZ, uin);
    probe("UEqn.diag", UEqn.relaxed ? UEqn.relaxedDiag : UEqn.diag);
    probe("UEqn.upper", UEqn.upper);
    probe("UEqn.source", UEqn.source[0]);
    probe("UEqn.iC", UEqn.iC[0]);
    if (taps)
    {
        deviceCopy(taps->UEqnDiag, UEqn.relaxed ? UEqn.relaxedDiag : UEqn.diag);
        deviceCopy(taps->UEqnSourceX, UEqn.source[0]);
        deviceCopy(taps->UEqnUpper, UEqn.upper);
        deviceCopy(taps->UEqnLower, UEqn.lower);
        deviceCopy(taps->UEqnIC, UEqn.iC[0]);
        deviceCopy(taps->UEqnBC, UEqn.bC[0]);
    }

    // 4. THE MOMENTUM PREDICTOR, if the case asks for one
    // damBreak sets `momentumPredictor no`. The matrix above is still assembled and relaxed either
    // way, because the pressure corrector is built on its A() and H().
    if (ctl.momentumPredictor)
    {
        if (static_cast<int>(snGradPrgh.size()) != nFaces)
            throw std::runtime_error(
                "brae interFoam device step: `momentumPredictor yes` needs snGrad(p_rgh) over the full "
                "face array from the interface-force hook. It is EXPLICIT in UEqn and IMPLICIT in the "
                "pressure equation, and carrying it into both would count the pressure gradient twice.");
        DeviceBuffer<scalar> force;
        deviceMomentumSourceFlux(nFaces, stf, ghf, snGradRho, snGradPrgh, magSf, force);

        DeviceBuffer<scalar> fInt(static_cast<std::size_t>(nIf)), fBnd(static_cast<std::size_t>(nBf));
        if (nIf > 0)
            cudaMemcpy(fInt.data(), force.data(), sizeof(scalar)*nIf, cudaMemcpyDeviceToDevice);
        if (nBf > 0)
            cudaMemcpy(fBnd.data(), force.data() + nIf, sizeof(scalar)*nBf, cudaMemcpyDeviceToDevice);

        // ON A COPY of the source: rAU and H() come from the relaxed matrix BEFORE the force.
        DeviceBuffer<scalar> sx, sy, sz;
        deviceCopy(sx, UEqn.source[0]);
        deviceCopy(sy, UEqn.source[1]);
        deviceCopy(sz, UEqn.source[2]);
        deviceAddMomentumPredictorSource(dm, fInt, fBnd, sx, sy, sz);

        const DeviceLduView A = UEqn.view(dm);
        DeviceBuffer<scalar>* Uk[3] = {&UX, &UY, &UZ};
        DeviceBuffer<scalar>* Sk[3] = {&sx, &sy, &sz};
        for (int k = 0; k < 3; ++k)
        {
            // fvMatrixSolve.C:162-164: a component the mesh does not solve is SKIPPED, not solved to
            // zero -- on a 2-D case that is the empty direction, and OpenFOAM's log has no Uz line.
            if (ctl.solutionD[k] == -1) continue;
            DeviceBuffer<scalar> diagC, b;
            deviceFold(dm, UEqn.relaxed ? UEqn.relaxedDiag : UEqn.diag, *Sk[k],
                       UEqn.iC[k], UEqn.bC[k], diagC, b);
            const DeviceLduView Ak = deviceLduView(dm, diagC, UEqn.upper, UEqn.lower);
            DeviceBuffer<scalar> dNf;
            deviceNormFactorInto(Ak, *Uk[k], b, deviceOnes(nC), dNf);
            // the case's own smoother where it names one -- see deviceAlphaPreSolve for what a
            // substituted solver at the same tolerance costs
            DeviceSolverPerf perf;
            if (ctl.momentum.smoothSolver)
            {
                deviceSymGaussSeidel(Ak, b, *Uk[k], dNf.data(), ctl.momentum.tol, ctl.momentum.relTol,
                                     ctl.momentum.maxIter, &perf, /*minIter=*/0, ctl.momentum.nSweeps,
                                     ctl.momentum.symmetric);
            }
            else
            {
                perf = deviceJacobiBiCGStab(Ak, b, *Uk[k], dNf.data(), ctl.momentum.tol,
                                            ctl.momentum.relTol, ctl.momentum.maxIter);
            }
            if (ctl.momentumSolveLog)
            {
                ctl.momentumSolveLog[k].push_back(perf);
            }
        }
        hooks.updateUBoundary(UX, UY, UZ, dbU, ub);
        deviceUpdateInletOutlet(dbU, phiBnd);
        deviceUpdatePressureInletOutletVelocity(dbU, phiBnd, UX, UY, UZ, /*directionMixed=*/true);
        (void)A;
    }

    // 5. THE PRESSURE CORRECTOR, last, and nCorrectors TIMES
    // interFoam.C:118-121 wraps the WHOLE of pEqn.H in `while (pimple.correct())`, so every pass
    // rebuilds rAU, HbyA, phiHbyA and phig from the U and phi the previous one left -- it is not a
    // repeated solve of one system. The momentum matrix is the same throughout, which is why it is
    // assembled above the loop.
    if (ctl.nCorrectors < 1)
        throw std::runtime_error(
            "brae interFoam device step: nCorrectors must be at least 1; fvSolution's PIMPLE block "
            "reads it and damBreak asks for three.");

    for (int corr = 0; corr < ctl.nCorrectors; ++corr)
    {
        gpu::PressureStages st;
        gpu::PressureInput pin;
        if (!ctl.takeUAtBoundary)
            throw std::runtime_error(
                "brae interFoam device step: constrainHbyA needs the per-face `assignable` mask. "
                "assignable() is NOT fixesValue() -- see DeviceInterStepControls.");
        pin.takeUAtBoundary = ctl.takeUAtBoundary;
        for (int k = 0; k < 3; ++k) pin.solutionD[k] = ctl.solutionD[k];
        gpu::pressurePredictor(st, dm, dbU, UEqn, UX, UY, UZ, pin, nullptr, nullptr);
        probe("rAU", st.rAU);
        probe("HbyA.x", st.HbyA[0]);
        probe("phiHbyA", st.phiHbyAInt);
        if (taps && corr == 0)
        {
            deviceCopy(taps->rAU, st.rAU);
            for (int k = 0; k < 3; ++k) deviceCopy(taps->HbyA[k], st.HbyA[k]);
        }

        // rAUf over the FULL face array: interpolate(rAU) internally, and the face cell's rAU at an
        // uncoupled patch, which is what fvc::interpolate gives there.
        DeviceBuffer<scalar> rAUfAll;
        deviceInterpolateFull(dm, st.rAU, rAUfAll);

        // fvc::ddtCorr(U, phi), Euler. The coefficient is OpenFOAM's DEFAULT LIMITER
        // (ddtPhiCoeff_ = -1), not a constant: it switches the correction off where it is large
        // compared with the flux, and it is zero on every patch where U fixes a value.
        DeviceBuffer<scalar> ddtCorrI, ddtCorrB;
        deviceDdtCorr(dm, phiOldInt, phiOldBnd, UOldX, UOldY, UOldZ, bndUFixesValue,
                      /*ddtPhiCoeff=*/scalar(-1), deltaT, ddtCorrI, ddtCorrB);

        DeviceInterPressureInput pi;
        pi.stf = &stf;
        pi.ghf = &ghf;
        pi.snGradRho = &snGradRho;
        pi.magSf = &magSf;
        pi.rAUfAll = &rAUfAll;
        pi.rho = &rho;
        pi.gh  = &gh;
        pi.ddtCorrInt = &ddtCorrI;
        pi.needReference = ctl.needReference;
        pi.pRefCell = ctl.pRefCell;
        pi.pRefValue = ctl.pRefValue;
        const bool finalCorr = (corr == ctl.nCorrectors - 1);
        pi.solve = finalCorr ? ctl.pressureFinal : ctl.pressure;
        pi.pcgDIC = finalCorr ? ctl.pressureFinalPcgDIC : ctl.pressurePcgDIC;
        pi.dic = ctl.dic;
        pi.gamg = finalCorr ? ctl.pressureFinalGamg : ctl.pressureGamg;
        pi.gamgCache = ctl.gamgCache;
        pi.gamgLog = ctl.gamgLog;
        pi.solveLog = ctl.pressureSolveLog;
        // the non-orthogonal passes before the last take the plain entry, whichever corrector this is
        pi.nNonOrthogonalCorrectors = ctl.nNonOrthogonalCorrectors;
        pi.solveInner = ctl.pressure;
        pi.pcgDICInner = ctl.pressurePcgDIC;
        pi.gamgInner = ctl.pressureGamg;
        pi.correctedLaplacian = ctl.correctedLaplacian;
        pi.snGradLimitCoeff = ctl.snGradLimitCoeff;

        DevicePressureTaps pt;
        deviceInterPressureStep(dm, pi, hooks.pressure, st.rAU, st.HbyA[0], st.HbyA[1], st.HbyA[2],
                                st.phiHbyAInt, st.phiHbyABnd, p_rgh, phiInt, phiBnd,
                                UX, UY, UZ, p, (taps && corr == 0) ? &pt : nullptr);
        if (taps && corr == 0)
        {
            deviceCopy(taps->pDiag, pt.diag);
            deviceCopy(taps->pUpper, pt.upper);
            deviceCopy(taps->pLower, pt.lower);
            deviceCopy(taps->pSource, pt.source);
            deviceCopy(taps->pIC, pt.iC);
            deviceCopy(taps->pBC, pt.bC);
            deviceCopy(taps->rAUfAllTap, rAUfAll);
            deviceCopy(taps->pSolved, p_rgh);
        }

        if (taps && corr == 0)
        {
            deviceCopy(taps->phiHbyAInt, st.phiHbyAInt);
            deviceCopy(taps->phiHbyABnd, st.phiHbyABnd);
        }

        // p_rgh.correctBoundaryConditions() at the end of pEqn.H, and U's with it: the next pass's
        // laplacian, its flux and its HbyA all read them.
        if (hooks.pressure.updateBoundary) hooks.pressure.updateBoundary(p_rgh);
        hooks.updateUBoundary(UX, UY, UZ, dbU, ub);
        deviceUpdateInletOutlet(dbU, phiBnd);
        deviceUpdatePressureInletOutletVelocity(dbU, phiBnd, UX, UY, UZ, /*directionMixed=*/true);
    }
    probe("p_rgh", p_rgh);
    probe("phi", phiInt);
    probe("U.x", UX);
}

} // namespace brae
