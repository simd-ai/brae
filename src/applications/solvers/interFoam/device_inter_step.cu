// One whole interFoam time step -- see device_inter_step.cuh for the loop order and what it decides.
#include "inter_phase_time.cuh"
#include "device_inter_step.cuh"
#include "inter_peqn_cpp.cuh"   // tapCorrectorWanted
#include "device_alpha_flux.cuh"
#include "device_blas.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"   // deviceSymGaussSeidel
#include "device_divdevreff.cuh"   // deviceBoundaryGradU
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace brae {
namespace {

// the flux U's switches read, face by face: rhoPhi where the patch names it, phi elsewhere
__global__ void namedUFluxKernel(
    const int*    __restrict__ isRhoPhi,
    const scalar* __restrict__ rhoPhi,
    const scalar* __restrict__ phi,
    int                        n,
    scalar*       __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = isRhoPhi[i] ? rhoPhi[i] : phi[i];
}

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

// BRAE_CONTROL_DEVICE_HBYA_REEVALUATED=1: constrainHbyA re-evaluating U's patches, as it did
bool hbyaReevaluatedControl()
{
    static const bool on = []
    {
        const bool v = std::getenv("BRAE_CONTROL_DEVICE_HBYA_REEVALUATED") != nullptr;
        if (v)
        {
            std::printf("  *** CONTROL MODE: constrainHbyA re-evaluates U's patches instead of taking their "
                        "stored values. This run is deliberately wrong. ***\n");
        }
        return v;
    }();
    return on;
}

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
    const DeviceBuffer<scalar>&      UOldBndX,
    const DeviceBuffer<scalar>&      UOldBndY,
    const DeviceBuffer<scalar>&      UOldBndZ,
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
    // ...and under CrankNicolson the off-centred flux, alphaEqn.H:91-97: phiCN = cnCoeff*phi +
    // (1 - cnCoeff)*phi.oldTime() once the scheme is warm, phi itself before (ocAlpha 0). The host
    // driver forms it the same way (offCentredFlux, inter_driver_cpp.cu).
    DeviceBuffer<scalar> phiCNInt, phiCNBnd, phiCNIf;
    const bool offCentred = ctl.cn && ctl.cn->ocAlpha > scalar(0);
    if (offCentred)
    {
        // ...blended with the level that EXISTS: on the step where phi.oldTime() is created the level
        // is a copy of phi itself, so the blend is inert (DeviceInterCrankNicolson::phiOldExists)
        const DeviceBuffer<scalar>& phiOldI = ctl.cn->phiOldExists ? phiOldInt : phiInt;
        const DeviceBuffer<scalar>& phiOldB = ctl.cn->phiOldExists ? phiOldBnd : phiBnd;
        deviceOffCentredFlux(nIf, ctl.cn->cnAlpha, phiInt, phiOldI, phiCNInt);
        deviceOffCentredFlux(nBf, ctl.cn->cnAlpha, phiBnd, phiOldB, phiCNBnd);
        ain.phiCNInt = &phiCNInt;
        ain.phiCNBnd = &phiCNBnd;
    }
    // THE PAIR, whose faces are in neither array above. Its volumetric flux lives in cyc->phi and is
    // the caller's to keep current -- the pressure corrector rewrites it at the end of every pass, as
    // it rewrites phi -- and phiCN is that same flux under Euler, exactly as above.
    ain.cyc        = ctl.cyc;
    ain.phiCNIf    = ctl.cyc ? &ctl.cyc->phi : nullptr;
    // ...and OFF-CENTRED on the pair too when the scheme is warm. The host arm blends phi's coupled
    // patch with every other one (offCentredFlux walks phi.boundary whole), and the pair's faces are
    // in neither of the two arrays above -- they are cyc->phi, with ctl.phiOldIf its old level.
    if (offCentred && ctl.cyc && ctl.cyc->n > 0)
    {
        if (!ctl.phiOldIf)
            throw std::runtime_error(
                "brae interFoam device step: CrankNicolson's off-centred flux on a coupled pair needs "
                "the pair's phi.oldTime(); the caller gave none.");
        deviceOffCentredFlux(ctl.cyc->n, ctl.cn->cnAlpha, ctl.cyc->phi,
                             ctl.cn->phiOldExists ? *ctl.phiOldIf : ctl.cyc->phi, phiCNIf);
        ain.phiCNIf = &phiCNIf;
    }
    ain.alphaPhiIf = ctl.alphaPhiIf;
    ain.rho1 = props.rho1;
    ain.rho2 = props.rho2;

    DeviceBuffer<scalar> alpha2;
    DeviceBuffer<scalar> alpha2Bnd;
    DeviceInterAlphaControls actl = ctl.alpha;
    actl.alpha2BndOut = &alpha2Bnd;
    actl.rhoPhiIf     = ctl.rhoPhiIf;
    actl.nHatfIf      = ctl.nHatfIf;
    actl.preSolveAlphaOut      = taps ? &taps->preSolveAlpha : nullptr;
    actl.preSolveAlphaPhiIfOut = taps ? &taps->preSolveAlphaPhiIf : nullptr;
    actl.alphaPhiWriteInt = ctl.alphaPhiWriteInt;
    actl.alphaPhiWriteBnd = ctl.alphaPhiWriteBnd;
    actl.alphaPhiWriteIf  = ctl.alphaPhiWriteIf;
    actl.alphaSubCycleBndWrite = ctl.alphaSubCycleBndWrite;
    if (ctl.cn)
    {
        // alphaEqn.H:236-262: ddt(rho,U) is not Euler, so rhoPhi takes phi beside rho2 and the
        // end-of-step alpha flux is un-blended when the scheme is warm
        actl.rhoPhiFromPhi = true;
        actl.cnCoeffUnblend = (ctl.cn->ocAlpha > scalar(0)) ? ctl.cn->cnAlpha : scalar(1);
        actl.alphaPhiOldInt = ctl.cn->alphaPhiOldInt;
        actl.alphaPhiOldBnd = ctl.cn->alphaPhiOldBnd;
        actl.alphaPhiOutInt = ctl.cn->alphaPhiOutInt;
        actl.alphaPhiOutBnd = ctl.cn->alphaPhiOutBnd;
        actl.alphaPhiOldIf = ctl.cn->alphaPhiOldIf;
        actl.alphaPhiOutIf = ctl.cn->alphaPhiOutIf;
        actl.alphaPhiCreatedIf = ctl.cn->alphaPhiCreatedIf;
        actl.alphaPhiCreatedInt = ctl.cn->alphaPhiCreatedInt;
        actl.alphaPhiCreatedBnd = ctl.cn->alphaPhiCreatedBnd;
    }
    deviceInterAlphaStep(dm, alpha1, alpha1Old, deltaT, ain, ctl.mules, actl, props, hooks.alpha,
                         alpha1Bnd, nHatfBnd, bndAlphaFixesValue, bndAlphaFlag,
                         nHatfInt, K, rhoPhiInt, rhoPhiBnd, alpha2, rho, mu, nu);

    if (taps)
    {
        deviceCopy(taps->alphaAfterAlphaStep, alpha1);
        if (ctl.alphaPhiIf) deviceCopy(taps->alphaPhiIfTap, *ctl.alphaPhiIf);
        if (ctl.rhoPhiIf)   deviceCopy(taps->rhoPhiIfTap,   *ctl.rhoPhiIf);
    }

    probe("alpha", alpha1);
    probe("rho", rho);
    probe("rhoPhi", rhoPhiInt);

    interPhase::mark("1 alpha step");
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
    // THE MIXTURE'S LAMINAR nu, kept before nut is added to it: fvOptions(rho, U)'s DarcyForchheimer
    // takes mu = rho*nu and NOT rho*nuEff (DarcyForchheimer.C:214-217 looks the field named "nu" up),
    // which is the host arm's own refusal if it is handed the wrong one (inter_ueqn_cpp.cu:246-250).
    DeviceBuffer<scalar> nuLamCell;
    if (ctl.porosity)
    {
        deviceCopy(nuLamCell, nuEffCell);
    }
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

    // FROZEN FLOW STOPS HERE. Stages 1 and 2 above are OpenFOAM's alpha equation and the interface
    // properties mixture.correct() leaves, both of which sit BEFORE `if (pimple.frozenFlow())` at
    // interFoam.C:156; everything below is UEqn.H (:161) and the pEqn.H corrector loop (:164-167),
    // which the `continue` skips whole. The U boundary replay at the end of the pressure corrector
    // goes with them: with no momentum and no pressure solve there is no new flux to switch an
    // inletOutlet face on, and OpenFOAM leaves those patches exactly as the last solved step did.

    interPhase::mark("2 interface forces and mixture");
    if (ctl.frozenFlow) return;

    // 3. THE MOMENTUM MATRIX
    DeviceBuffer<scalar> ub[3];
    hooks.updateUBoundary(UX, UY, UZ, dbU, ub, DeviceUBoundaryCall::assembly);
    // pressureInletOutletVelocity's updateCoeffs, from the flux registered NOW: an inflow face fixes
    // its tangential components, an outflow face is zeroGradient. The hook rebuilds dbU from the host
    // patches' categories, which carry no flux, so without this every such face stayed zeroGradient for
    // the whole run -- which agreed with the host only while the host made the same omission.
    // ...and inletOutlet's, FIRST, in the order rhoSimpleFoam's device arm takes them
    // (rhoUEqn.cuh:72-79): the io switch off the flux registered now, then the switches that read this
    // iteration's cell velocity. The hook rebuilds dbU from the host patches, and buildDeviceVectorBoundary
    // seeds an inletOutlet as fixedValue at its inletValue on EVERY face (device_boundary.cuh, category 3),
    // so without this the patch is a wall at the inlet value for the whole run.
    // ...THE FLUX EACH PATCH NAMES, not phi: see DeviceInterStepControls::uFluxIsRhoPhi. Built into its
    // own buffer at each switch, because phi moves with every corrector and rhoPhi does not.
    DeviceBuffer<scalar> uNamedFlux;
    auto namedUFlux = [&](const DeviceBuffer<scalar>& phiB) -> const DeviceBuffer<scalar>&
    {
        if (!ctl.uFluxIsRhoPhi) return phiB;
        const int n = static_cast<int>(phiB.size());
        if (static_cast<int>(ctl.uFluxIsRhoPhi->size()) != n || static_cast<int>(rhoPhiBnd.size()) != n)
            throw std::runtime_error(
                "brae interFoam device step: a U patch names rhoPhi and the mask or the mass flux is not one "
                "value per boundary face.");
        uNamedFlux.resize(static_cast<std::size_t>(n));
        if (n > 0)
        {
            namedUFluxKernel<<<(n + 255)/256, 256>>>(ctl.uFluxIsRhoPhi->data(), rhoPhiBnd.data(), phiB.data(),
                                                     n, uNamedFlux.data());
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("brae interFoam device step: the named U flux: ")
                                         + cudaGetErrorString(e));
        }
        return uNamedFlux;
    };
    deviceUpdateInletOutlet(dbU, namedUFlux(phiBnd));
    // ...and the flux that switch read, kept for the first corrector of a pass with no predictor: the
    // patch is still updated() there and its evaluate keeps THIS valueFraction (the corrector loop
    // below has the rule and the measurement).
    DeviceBuffer<scalar> phiBndAtUpdateCoeffs;
    deviceCopy(phiBndAtUpdateCoeffs, namedUFlux(phiBnd));
    deviceUpdatePressureInletOutletVelocity(dbU, namedUFlux(phiBnd), UX, UY, UZ, /*directionMixed=*/true);
    // ...and symmetry's, which is the same sequence's third step (rhoUEqn.cuh:76-78): a symmetry or slip
    // patch's refValue is U - n(n & U) at THIS iteration's cell velocity, and the builder seeded it from
    // the host field's last evaluate. MEASURED on RAS/angledDuct, whose `porosityWall` is a slip patch
    // tilted 45 degrees, with the HOST closure in the device loop and the porosity off so neither can be
    // the cause: U 2.7e-02 against OpenFOAM where the host loop is 4.1e-15, and that patch the only one
    // whose values differed from the host's.
    deviceUpdateSymmetry(dbU, UX, UY, UZ);
    // ...AND THE WEDGE'S, the fourth step of that sequence and the one this loop never took. The
    // builder seeds a wedge face's refValue with the host patch VALUE, which is not what the mixed
    // kernels need: they blend d*ref + (1 - d)*U_cell, and the ref that reproduces OpenFOAM's
    // transform(faceT, U_cell) is deviceUpdateWedge's, from THIS iteration's cell velocity
    // (device_boundary.cuh). Every other driver in the tree calls the two as a pair. While U is
    // zero the two refs agree, which is why it hid: MEASURED on LES/nozzleFlow2D's laminar twin,
    // ONE step -- the first corrector's two p_rgh solves OpenFOAM's to every digit, the second
    // corrector's initial residual 6.5e-03 for 3.1e-04, p_rgh 5.3e+08 of 8.9e+08, |U| 17%.
    deviceUpdateWedge(dbU, UX, UY, UZ);

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
    // ...and rho.oldTime().oldTime() for CrankNicolson, from alpha1's old-old level the same way
    DeviceBuffer<scalar> rhoOO;
    if (ctl.cn)
    {
        if (!ctl.cn->clock || !ctl.cn->alpha1OO || ctl.cn->alpha1OO->size() != static_cast<std::size_t>(nC))
            throw std::runtime_error(
                "brae interFoam device step: CrankNicolson needs the scheme's clock and alpha1's old-old "
                "level, one value per cell, for rho.oldTime().oldTime().");
        rhoOO.resize(static_cast<std::size_t>(nC));
        deviceMixtureCorrect(ctl.cn->alpha1OO->data(), nC, props, nullptr, rhoOO.data(), nullptr, nullptr);
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
    uin.nonOrthCoeffs = ctl.nonOrthCoeffs;
    uin.snGradLimitCoeff   = ctl.snGradLimitCoeff;
    uin.gradUSchemeLimitK = ctl.gradUSchemeLimitK;
    uin.gradUSchemeLeastSq = ctl.gradUSchemeLeastSq;
    uin.relaxU        = ctl.relaxU;
    uin.relaxEquation = ctl.relaxEquationU;
    uin.interOrder    = true;
    uin.ddtRho        = &rho;
    uin.ddtRhoOld     = &rhoOld;
    uin.ddtV0         = ctl.V0;
    uin.ddtV00        = ctl.V00;
    uin.ddtUOld[0]    = &UOldX;
    uin.ddtUOld[1]    = &UOldY;
    uin.ddtUOld[2]    = &UOldZ;
    uin.ddtDeltaT     = deltaT;
    uin.ddtRDeltaT    = ctl.rDeltaTUEqn;
    if (ctl.cn)
    {
        uin.ddtCn = ctl.cn->clock;
        uin.ddtCnDdt0 = &ctl.cn->ddt0RhoU;
        uin.ddtCnPatch = ctl.cn->ddt0RhoUPatch;
        uin.ddtRhoOO = &rhoOO;
        for (int k = 0; k < 3; ++k) uin.ddtUOO[k] = ctl.cn->UOO[k];
    }
    uin.UbStored      = ubPtr;
    // + MRF.DDt(rho, U), UEqn.H:6, rho-weighted as MRFZoneList::DDt(rho, U) is (MRFZoneList.C:210-217)
    // and as the host arm forms it (inter_ueqn_cpp.cu:207-221). The shared assembler puts it in the
    // source before relax, which is where the fvMatrix expression has it.
    // == fvOptions(rho, U), UEqn.H:9, in the source and diagonal before relax -- the slot the host arm
    // puts it in (inter_ueqn_cpp.cu:245-260). mu is the mixture's rho*nu, formed per cell here because
    // it spans three orders across the interface.
    DeviceBuffer<scalar> porMu;
    if (ctl.porosity)
    {
        deviceHadamard(porMu, rho, nuLamCell);
        uin.porosity    = ctl.porosity;
        uin.porosityMu  = &porMu;
        uin.porosityRho = &rho;
    }
    // ...and the mangroves' drag and added mass, on the same rho: multiphaseMangrovesSource's
    // addSup(rho, eqn) takes the field UEqn is weighted with (inter_ueqn_cpp.cu hands it `rho` too)
    if (ctl.mangroves)
    {
        uin.mangroves    = ctl.mangroves;
        uin.mangrovesRho = &rho;
    }
    uin.cyc          = ctl.cyc;
    uin.cycCorrected = ctl.correctedLaplacian;
    uin.cycNonOrth = ctl.nonOrthCoeffs;
    // fvm::div(rhoPhi, U) on the pair too -- the MASS flux, as uin.phiInt above is
    uin.cycConvFlux  = ctl.rhoPhiIf;
    uin.mrf    = ctl.mrf;
    uin.mrfRho = ctl.mrf ? &rho : nullptr;

    probe("muCell", muCell);
    probe("muFace", muFace);
    probe("rhoOld", rhoOld);
    if (taps) deviceCopy(taps->ddtRhoOld, rhoOld);

    // fvSolution's cached grad(U), at the host loop's rule (inter_driver_cpp.cu's UEqn stage): MRF's
    // correctBoundaryVelocity moves U's eventNo (boundaryFieldRef), a changing mesh bypasses the registry, and
    // where U has changed since the closure formed it OpenFOAM's first request of the assembly would form and
    // store it for the others, in an operand order not modelled -- refused, in the host's own words.
    if (ctl.gradUCache && ctl.gradUCache->on)
    {
        DeviceGradUCache& gc = *ctl.gradUCache;
        if ((ctl.mrf && !ctl.mrf->empty()) || ctl.gradUMeshChanging)
        {
            gc.valid = false;
        }
        if (!ctl.gradUUncachedControl && !ctl.gradUMeshChanging)
        {
            if (!gc.valid)
                throw std::runtime_error(
                    "brae interFoam: fvSolution caches grad(U), and at this UEqn assembly U has "
                    "changed since grad(U) was last formed (or nothing formed it: only kOmegaSST's "
                    "validate and correct do). OpenFOAM's first request in the assembly would form "
                    "and store it for the others, in an operand order brae does not model.");
            uin.gradUGivenMemo   = &gc.memo;
            uin.gradUGivenTensor = &gc.tensor;
            uin.gradBGiven       = &gc.bnd;
            ++gc.consumed;
            if (ctl.gradUBndLiveControl)
            {
                // the gate's control: the boundary gaussGrad would give NOW, from the patches as the assembly
                // left them, in place of the one formed with the field
                static DeviceBuffer<scalar> liveB;
                deviceBoundaryGradU(dm, dbU, UX, UY, UZ, gc.tensor, liveB, ubPtr);
                uin.gradBGiven = &liveB;
            }
        }
    }

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
        deviceCopy(taps->uEqnCycIfCoeff, UEqn.cycIfCoeff);
    }

    interPhase::mark("3 momentum matrix");
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
        // ...and on the pair, the same four fields on its own faces. MEASURED on RAS/mixerVesselAMI's
        // first step without it, host against device: the predicted U 3.7e-01 out in the pair's cells
        // (0.183 against -0.189), U 5.9e-05 after the corrector.
        DeviceBuffer<scalar> fPair;
        const bool pairForce = ctl.cyc && ctl.cyc->n > 0
                            && std::getenv("BRAE_CONTROL_DEVICE_PREDICTOR_NO_PAIR_FORCE") == nullptr;
        if (pairForce)
        {
            if (!ctl.stfIf || !ctl.ghfIf || !ctl.snGradRhoIf || !ctl.snGradPrghIf
             || static_cast<int>(ctl.snGradPrghIf->size()) != ctl.cyc->n)
                throw std::runtime_error(
                    "brae interFoam device step: `momentumPredictor yes` on a mesh with a coupled pair "
                    "needs stf, ghf, snGrad(rho) and snGrad(p_rgh) on the pair's faces.");
            deviceMomentumSourceFlux(ctl.cyc->n, *ctl.stfIf, *ctl.ghfIf, *ctl.snGradRhoIf,
                                     *ctl.snGradPrghIf, ctl.cyc->magSf, fPair);
        }
        deviceAddMomentumPredictorSource(dm, fInt, fBnd, sx, sy, sz, pairForce ? ctl.cyc : nullptr,
                                         pairForce ? &fPair : nullptr);

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
            // ...with the pair's off-diagonal, so the solve applies the operator that was assembled
            const DeviceLduView Ak = (ctl.cyc && ctl.cyc->n > 0)
                ? deviceLduViewPair(dm, diagC, UEqn.upper, UEqn.lower, *ctl.cyc, nullptr, &UEqn.cycIfCoeff)
                : deviceLduView(dm, diagC, UEqn.upper, UEqn.lower);
            DeviceBuffer<scalar> dNf;
            deviceNormFactorInto(Ak, *Uk[k], b, deviceOnes(nC), dNf);
            // the case's own smoother where it names one -- see deviceAlphaPreSolve for what a
            // substituted solver at the same tolerance costs
            DeviceSolverPerf perf;
            if (ctl.momentum.smoothSolver)
            {
                deviceSymGaussSeidel(Ak, b, *Uk[k], dNf.data(), ctl.momentum.tol, ctl.momentum.relTol,
                                     ctl.momentum.maxIter, &perf, ctl.momentum.minIter,
                                     ctl.momentum.nSweeps, ctl.momentum.symmetric);
            }
            else
            {
                // minIter REACHES THIS BRANCH TOO. `deviceJacobiBiCGStab` has taken it on both overloads
                // all along (device_pcg.cuh:128-138, `int minIter = 0`), and this site simply never passed
                // it -- so the refusal added beside it in the momentum-minIter unit, which said the
                // branch "takes no iteration floor", was wrong about its own code. `checkEvery` keeps its
                // default 1, which is what the argument before minIter is.
                perf = deviceJacobiBiCGStab(Ak, b, *Uk[k], dNf.data(), ctl.momentum.tol,
                                            ctl.momentum.relTol, ctl.momentum.maxIter,
                                            /*checkEvery=*/1, ctl.momentum.minIter);
            }
            if (ctl.momentumSolveLog)
            {
                ctl.momentumSolveLog[k].push_back(perf);
            }
        }
        // the predictor's solve ends in U.correctBoundaryConditions(), which clears updated()
        hooks.updateUBoundary(UX, UY, UZ, dbU, ub, DeviceUBoundaryCall::evaluateAfterPredictor);
        deviceUpdateInletOutlet(dbU, namedUFlux(phiBnd));
        deviceUpdatePressureInletOutletVelocity(dbU, namedUFlux(phiBnd), UX, UY, UZ, /*directionMixed=*/true);
        deviceUpdateSymmetry(dbU, UX, UY, UZ);
        deviceUpdateWedge(dbU, UX, UY, UZ);
        (void)A;
    }

    interPhase::mark("4 momentum predictor");
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
        // constrainHbyA takes U's STORED patch values (constrainHbyA.C:67), which the hook's host evaluate
        // left in `ub`; re-evaluated on the device they carry the coefficients the momentum assembly's
        // updateCoeffs set, one corrector before OpenFOAM's U.correctBoundaryConditions() applies them.
        // MEASURED on RAS/DTCHull (no predictor, outletPhaseMeanVelocity), iteration two, first corrector:
        // the outlet's phiHbyA 1.06e-09 from the host's of 633.84, the solved p_rgh one smooth shift of
        // 5e-09, and the written fields 3.1e-05 from OpenFOAM by iteration 25 where the host reads 2.0e-07.
        // BRAE_CONTROL_DEVICE_HBYA_REEVALUATED=1 puts the re-evaluation back -- the gate's control.
        for (int k = 0; k < 3; ++k)
        {
            pin.UbStored[k] = hbyaReevaluatedControl() ? nullptr : ubPtr[k];
        }
        pin.cyc = ctl.cyc;
        for (int k = 0; k < 3; ++k) pin.solutionD[k] = ctl.solutionD[k];
        if (taps && corr == cpu::interFoam::tapCorrectorWanted())
        {
            pin.hNoPairTap = &taps->HNoPairX;
            pin.hPairTap   = &taps->HPairX;
        }
        gpu::pressurePredictor(st, dm, dbU, UEqn, UX, UY, UZ, pin, nullptr, nullptr);
        probe("rAU", st.rAU);
        // the next mesh update's CorrectPhi reads this on the host; see DeviceInterStepControls
        if (ctl.rAUOut) deviceCopy(*ctl.rAUOut, st.rAU);
        probe("HbyA.x", st.HbyA[0]);
        probe("phiHbyA", st.phiHbyAInt);
        if (taps && corr == cpu::interFoam::tapCorrectorWanted())
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
        DeviceBuffer<scalar> ddtCorrI, ddtCorrB, ddtCorrIf;
        if (ctl.cn)
        {
            const DeviceBuffer<scalar>* uo[3] = {&UOldX, &UOldY, &UOldZ};
            const DeviceBuffer<scalar>* uob[3] = {&UOldBndX, &UOldBndY, &UOldBndZ};
            // ON A DYNAMIC MESH the scheme resolves ddtCorr(U, Uf) to fvcDdtUfCorr, a DIFFERENT member
            // function from the static ddtCorr(U, phi) (CrankNicolsonDdtScheme.C:1201-1257) -- the
            // same distinction the Euler branch below makes with phiUfOldInt, and the host arm with
            // `in.UfOld` (inter_peqn_cpp.cu:231). Uf's two old levels come from the driver.
            //
            // THE TEST IS THOSE LEVELS, NOT V0. fvcDdt.C:219 routes on mesh.dynamic() -- moving OR
            // topo-changing -- while V0 is given only on a mesh that MOVES, so a REFINING mesh took the
            // phi branch here and created ddtCorrDdt0(phi), a level OpenFOAM never makes on such a case
            // and the host arm does not have. The driver sets these levels on f.meshIsDynamic, exactly as
            // it sets phiUfOldInt for the Euler branch, so asking for them is asking mesh.dynamic().
            if (ctl.cn->UfOld[0])
            {
                if (!ctl.cn->UfOO[0] || !ctl.cn->UfOldBnd[0] || !ctl.cn->UfOOBnd[0])
                    throw std::runtime_error(
                        "brae interFoam device step: CrankNicolson's ddtCorr on a dynamic mesh is "
                        "fvcDdtUfCorr, which needs Uf.oldTime() and Uf.oldTime().oldTime() on the "
                        "internal and the boundary faces, three components each; the caller gave fewer.");
                deviceCnDdtUfCorr(dm, *ctl.cn->clock, ctl.cn->ddtCorrU, ctl.cn->ddtCorrUf,
                                  uo, ctl.cn->UOO, uob, ctl.cn->UOOBnd,
                                  ctl.cn->UfOld, ctl.cn->UfOldBnd, ctl.cn->UfOO, ctl.cn->UfOOBnd,
                                  bndUFixesValue, /*ddtPhiCoeff=*/scalar(-1), ddtCorrI, ddtCorrB,
                                  ctl.cyc);
            }
            else
            {
                // CrankNicolson's fvcDdtPhiCorr, with its two ddt0 fields, transcribed from the host
                // reference
                if (!ctl.cn->phiOOInt || !ctl.cn->phiOOBnd)
                    throw std::runtime_error(
                        "brae interFoam device step: CrankNicolson's ddtCorr needs phi.oldTime().oldTime().");
                deviceCnDdtCorr(dm, *ctl.cn->clock, ctl.cn->ddtCorrU, ctl.cn->ddtCorrPhi, uo, ctl.cn->UOO,
                                uob, ctl.cn->UOOBnd, phiOldInt, phiOldBnd, *ctl.cn->phiOOInt,
                                *ctl.cn->phiOOBnd,
                                bndUFixesValue, /*ddtPhiCoeff=*/scalar(-1), ddtCorrI, ddtCorrB,
                                // ...and the PAIR's half, in the SAME call: the Euler form below runs
                                // only when the scheme is Euler, and it silently stood in for this one
                                ctl.cyc, &ctl.cn->ddtCorrPhiIf, ctl.phiOldIf, ctl.cn->phiOOIf, &ddtCorrIf);
            }
        }
        else
        {
            deviceDdtCorr(dm, phiOldInt, phiOldBnd, UOldX, UOldY, UOldZ, bndUFixesValue,
                          /*ddtPhiCoeff=*/scalar(-1), deltaT, ddtCorrI, ddtCorrB,
                          &UOldBndX, &UOldBndY, &UOldBndZ,
                          // on a moving mesh (Sf & Uf.oldTime()) takes phi.oldTime()'s place
                          ctl.phiUfOldInt,
                          // ...and under localEuler the face rDeltaT takes 1/deltaT's
                          ctl.rDeltaTfInt, ctl.rDeltaTfBnd);
        }

        // MRF.zeroFilter(interpolate(rho*rAU)*fvc::ddtCorr(U, phi)), pEqn.H:18. MRFZone::zero sets the
        // flux to Zero on the zone's internal faces and on its included AND excluded boundary faces
        // (MRFZoneTemplates.C:213-247); the host zeroes the correction itself before the weighting,
        // which is the same number either way (inter_peqn_cpp.cu:495-520). The correction compares
        // phi.oldTime() with the flux of U.oldTime(), and inside the zone the first is relative to the
        // frame and the second is not, so their difference there is the frame flux, not a correction.
        if (ctl.mrf && !ctl.mrf->empty())
        {
            deviceMrfZeroFilter(*ctl.mrf, ddtCorrI, ddtCorrB);
        }

        // ...and the PAIR's half, which pEqn.H:16-18 builds with the rest of the surfaceScalarField.
        // MRF.zeroFilter does not reach it: a zone's faces are the mesh's own, and a periodic pair is
        // refused on an MRF case before this (the driver's MRF refusal).
        if (!ctl.cn && ctl.cyc && ctl.cyc->n > 0 && ctl.phiOldIf)
        {
            deviceInterDdtCorrCyclic(*ctl.cyc, *ctl.phiOldIf, UOldX, UOldY, UOldZ,
                                     /*ddtPhiCoeff=*/scalar(-1), deltaT, ddtCorrIf);
        }

        DeviceInterPressureInput pi;
        pi.stf = &stf;
        pi.ghf = &ghf;
        pi.snGradRho = &snGradRho;
        // ...and their halves on the periodic pair, which the same three expressions cover there
        pi.ddtCorrIf   = (ctl.cyc && ctl.cyc->n > 0 && ctl.phiOldIf) ? &ddtCorrIf : nullptr;
        pi.stfIf       = ctl.stfIf;
        pi.ghfIf       = ctl.ghfIf;
        pi.snGradRhoIf = ctl.snGradRhoIf;
        pi.magSf = &magSf;
        pi.rAUfAll = &rAUfAll;
        // the mesh flux on a moving mesh, for fvc::makeRelative(phi, U) at the end of pEqn
        pi.meshPhiAll = ctl.meshPhiAll;
        pi.phiAbsIntOut = ctl.phiAbsIntOut;
        pi.phiAbsBndOut = ctl.phiAbsBndOut;
        pi.meshPhiIf = ctl.meshPhiIf;
        pi.phiAbsIfOut = ctl.phiAbsIfOut;
        pi.rho = &rho;
        pi.gh  = &gh;
        pi.ddtCorrInt = &ddtCorrI;
        // ddtCorr's boundary half, which pEqn.H:16-17 adds as part of a whole surfaceScalarField and
        // this step used to drop (inter_peqn_cpp.cu:522-530 has what dropping it cost)
        pi.ddtCorrBnd = &ddtCorrB;
        pi.rhoBndFace = &rhoBnd;
        pi.bndUFixesValue = &bndUFixesValue;
        pi.mrf = ctl.mrf;
        pi.cyc = ctl.cyc;
        pi.phiHbyAIf = (ctl.cyc && ctl.cyc->n > 0) ? &st.phiHbyAIf : nullptr;
        pi.needReference = ctl.needReference;
        pi.pRefCell = ctl.pRefCell;
        pi.pRefValue = ctl.pRefValue;
        const bool finalCorr = (corr == ctl.nCorrectors - 1);
        pi.solve = finalCorr ? ctl.pressureFinal : ctl.pressure;
        pi.pcgDIC = finalCorr ? ctl.pressureFinalPcgDIC : ctl.pressurePcgDIC;
        pi.dic = ctl.dic;
        pi.gamg = finalCorr ? ctl.pressureFinalGamg : ctl.pressureGamg;
        pi.pcgGamg = finalCorr ? ctl.pressureFinalPcgGamg : ctl.pressurePcgGamg;
        pi.gamgCache = ctl.gamgCache;
        pi.amgPcg = ctl.amgPcg;
        pi.gamgLog = ctl.gamgLog;
        pi.solveLog = ctl.pressureSolveLog;
        // the non-orthogonal passes before the last take the plain entry, whichever corrector this is
        pi.nNonOrthogonalCorrectors = ctl.nNonOrthogonalCorrectors;
        pi.solveInner = ctl.pressure;
        pi.pcgDICInner = ctl.pressurePcgDIC;
        pi.gamgInner = ctl.pressureGamg;
        pi.pcgGamgInner = ctl.pressurePcgGamg;
        pi.correctedLaplacian = ctl.correctedLaplacian;
        pi.prghGradLeastSquares = ctl.prghGradLeastSquares;
        pi.prghGradCellLimitK   = ctl.prghGradCellLimitK;
        pi.nonOrthCoeffs = ctl.nonOrthCoeffs;
        pi.snGradLimitCoeff = ctl.snGradLimitCoeff;

        DevicePressureTaps pt;
        deviceInterPressureStep(dm, pi, hooks.pressure, st.rAU, st.HbyA[0], st.HbyA[1], st.HbyA[2],
                                st.phiHbyAInt, st.phiHbyABnd, p_rgh, phiInt, phiBnd,
                                UX, UY, UZ, p, taps ? &pt : nullptr);
        if (taps)
        {
            for (const std::vector<scalar>& jh : pt.jumpHistory)
            {
                taps->jumpHistory.push_back(jh);
            }
            deviceCopy(taps->phigIf, pt.phigIf);
            deviceCopy(taps->rAUfIf, pt.rAUfIf);
            deviceCopy(taps->ffIf, pt.ffIf);
            deviceCopy(taps->phiIf, pt.phiIf);
            deviceCopy(taps->phiHbyAIfPrePhig, pt.phiHbyAIfPrePhig);
            // not nonOrthSource: it is the selected corrector's, copied below. Copied here as well, the
            // LAST corrector's overwrote it -- on pistonLES a converged p_rgh's 3.5e-08 against the host
            // tap's corrector 0, where the hydrostatic start leaves the correction at 1e-300.
            deviceCopy(taps->cycJumpTap, pt.cycJumpTap);
        }
        if (taps && corr == cpu::interFoam::tapCorrectorWanted())
        {
            // EVERY pressure tap from the SAME corrector. These were split across two blocks with
            // different guards -- pSource from corrector 0 and the rest from the last -- and on a case
            // with nCorrectors 2 that made a dump compare corrector 0's source against corrector 1's
            // correction. It read as a 6.9430e+00 defect in a source that is in fact exact.
            taps->tapCorrector = corr;
            deviceCopy(taps->phigIntTap, pt.phigIntTap);
            deviceCopy(taps->phigBndTap, pt.phigBndTap);
            deviceCopy(taps->divPhiHbyA, pt.divPhiHbyA);
            deviceCopy(taps->phiHbyAIntPrePhig, pt.phiHbyAIntPrePhig);
            deviceCopy(taps->phiHbyABndPrePhig, pt.phiHbyABndPrePhig);
            deviceCopy(taps->nonOrthSource, pt.nonOrthSource);
            deviceCopy(taps->pDiag, pt.diag);
            deviceCopy(taps->pUpper, pt.upper);
            deviceCopy(taps->pLower, pt.lower);
            deviceCopy(taps->pSource, pt.source);
            deviceCopy(taps->pIC, pt.iC);
            deviceCopy(taps->pBC, pt.bC);
            deviceCopy(taps->rAUfAllTap, rAUfAll);
            deviceCopy(taps->pSolved, p_rgh);
        }

        if (taps && corr == cpu::interFoam::tapCorrectorWanted())
        {
            deviceCopy(taps->phiHbyAInt, st.phiHbyAInt);
            deviceCopy(taps->phiHbyABnd, st.phiHbyABnd);
        }

        // p_rgh.correctBoundaryConditions() at the end of pEqn.H, and U's with it: the next pass's
        // laplacian, its flux and its HbyA all read them.
        if (hooks.pressure.updateBoundary) hooks.pressure.updateBoundary(p_rgh);
        // ...U's with the ASSEMBLY-TIME coefficients in the first corrector of a pass that ran no
        // predictor: its patches are still updated() there, as the host loop's
        // `uPatchesUpdatedAtEntry = (c == 0) && !momentumPredictor` has it (inter_driver_cpp.cu).
        const bool stillUpdated = (corr == 0) && !ctl.momentumPredictor;
        hooks.updateUBoundary(UX, UY, UZ, dbU, ub,
                              stillUpdated ? DeviceUBoundaryCall::evaluateStillUpdated
                                           : DeviceUBoundaryCall::evaluate);
        // The io switch IS inletOutlet's valueFraction (neg(phi), inletOutletFvPatchField.C:114-127,
        // outletInlet's pos0, outletInletFvPatchField.C:124), and a still-updated() patch keeps the one the assembly set:
        // mixedFvPatchField::evaluate runs updateCoeffs only `if (!this->updated())`
        // (mixedFvPatchField.C:234-237), and neither class evaluates inside updateCoeffs. The hook has
        // just rebuilt dbU with every such face seeded fixedValue, so the switch has to be replayed
        // here -- off the flux the ASSEMBLY read on that one corrector, off the new flux otherwise.
        // Switching on the new flux here was the lag the host loop carries and this loop did not: the
        // device kOmegaSST closure reads U's outlet through dbU for its grad(U) (device_komega_sst.cu),
        // where OpenFOAM's tgradU reads the stored, lagged value. pressureInletOutletVelocity is
        // exempt, as on the host: its updateCoeffs ends in evaluate() and clears the flag.
        // ON A MOVING MESH THE FLUX THESE SWITCHES READ IS THE ABSOLUTE ONE. pEqn.H runs
        // U.correctBoundaryConditions() at :61, fvc::correctUf at :66 and fvc::makeRelative(phi, U) at :69, in
        // that order, so the patches' updateCoeffs look phi up while it is still absolute -- and the
        // pressure step above has already made this loop's phiBnd relative. The host loop keeps the same
        // order and says so (inter_peqn_cpp.cu, "U's patches are NOT re-told here"). MEASURED on
        // RAS/electrostaticDeposition, one step, device against OpenFOAM: on `side-02` the relative flux
        // is +3.56e-04 on all 225 faces (the patch moves with the mesh) while the absolute one is inflow,
        // so the device left the patch on its outflow branch -- U's patch value 1.19e-06 off in the
        // tangential components, U and Uf 1.5e-05 in the written files, with every cell at 1e-16.
        const bool uReadsAbsolute = ctl.phiAbsBndOut
                                 && ctl.phiAbsBndOut->size() == phiBnd.size()
                                 && std::getenv("BRAE_CONTROL_DEVICE_U_PATCH_RELATIVE") == nullptr;
        const DeviceBuffer<scalar>& phiBndForU = uReadsAbsolute ? *ctl.phiAbsBndOut : phiBnd;
        deviceUpdateInletOutlet(dbU, stillUpdated ? phiBndAtUpdateCoeffs : namedUFlux(phiBndForU));
        deviceUpdatePressureInletOutletVelocity(dbU, namedUFlux(phiBndForU), UX, UY, UZ, /*directionMixed=*/true);
        deviceUpdateSymmetry(dbU, UX, UY, UZ);
        deviceUpdateWedge(dbU, UX, UY, UZ);
        if (hooks.correctorDone)
        {
            hooks.correctorDone(phiInt, phiBnd);
        }
    }
    probe("p_rgh", p_rgh);
    probe("phi", phiInt);
    probe("U.x", UX);
    // the predictor's solve and every corrector's `U = HbyA + ...` moved U's eventNo: the cached grad(U) is out of
    // date until the closure forms it again (the host loop invalidates at the same two points; nothing consumes
    // in between)
    if (ctl.gradUCache)
    {
        ctl.gradUCache->valid = false;
    }
}


void deviceStoreGradU(
    DeviceGradUCache& c,
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>* const* UbStored)
{
    if (!c.on) return;
    const int nC = dm.nCells;
    const int nB = dm.nBndFaces;
    for (int i = 0; i < 3; ++i)
    {
        if (!UbStored || !UbStored[i] || UbStored[i]->size() != static_cast<std::size_t>(nB))
            throw std::runtime_error(
                "brae interFoam (device): the cached grad(U) is formed from U's STORED patch values, as OpenFOAM's "
                "fvc::grad(U) reads them; the caller supplied none for every boundary face.");
    }
    // the unlimited Gauss gradient, as the assembly's own sites form it (deviceDivDevReff's fused call)
    const DeviceBuffer<scalar>* vol[3] = {&Ux, &Uy, &Uz};
    deviceGaussGradFused(dm, 3, vol, UbStored, c.memo.gx, c.memo.gy, c.memo.gz);
    c.memo.nC = nC;
    c.tensor.resize(static_cast<std::size_t>(9) * nC);
    for (int i = 0; i < 3; ++i)
    {
        const DeviceBuffer<scalar>* g[3] = {&c.memo.gx[i], &c.memo.gy[i], &c.memo.gz[i]};
        for (int d = 0; d < 3; ++d)
        {
            cudaCheck(cudaMemcpyAsync(c.tensor.data() + static_cast<std::size_t>(d*3 + i)*nC, g[d]->data(),
                                      nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread),
                      "grad(U) cache pack");
        }
    }
    // ...and its boundary, against the patches' snGrad NOW -- the host's fvc::gradUBoundary in storeGradU
    deviceBoundaryGradU(dm, dbU, Ux, Uy, Uz, c.tensor, c.bnd, UbStored);
    c.valid = true;
}


void deviceSetGradU(
    DeviceGradUCache& c,
    const std::vector<scalar>& tensor9,
    const std::vector<scalar>& bnd9,
    int nC)
{
    if (!c.on) return;
    if (tensor9.size() != static_cast<std::size_t>(9) * nC || bnd9.size() % 9 != 0)
        throw std::runtime_error("brae interFoam (device): a host grad(U) to cache has the wrong size.");
    c.tensor.copyFrom(tensor9);
    c.bnd.copyFrom(bnd9);
    c.memo.nC = nC;
    for (int i = 0; i < 3; ++i)
    {
        DeviceBuffer<scalar>* g[3] = {&c.memo.gx[i], &c.memo.gy[i], &c.memo.gz[i]};
        for (int d = 0; d < 3; ++d)
        {
            g[d]->copyFrom(std::vector<scalar>(tensor9.begin() + static_cast<std::ptrdiff_t>(d*3 + i)*nC,
                                               tensor9.begin() + static_cast<std::ptrdiff_t>(d*3 + i + 1)*nC));
        }
    }
    c.valid = true;
}

} // namespace brae
