// THE DEVICE ALPHA STEP ON damBreak'S OWN MESH AND CASE, against the host solver.
//
// Everything before this ran on synthetic fixtures -- a rotating blob on a uniform box, built so each
// arm could discriminate. This one runs the real thing: damBreak's blockMesh mesh, its own 0/ fields,
// its own fvSolution, its inletOutlet atmosphere, its EMPTY frontAndBack patches, and the controls the
// tutorial actually sets -- nAlphaSubCycles 1, nAlphaCorr 2, MULESCorr yes, nLimiterIter 5.
//
// WHY IT IS A SEPARATE GATE FROM interfoam_dambreak_vs_openfoam.sh. That one holds the HOST solver
// against real OpenFOAM and reaches alpha 3.4346e-09. This one holds the DEVICE against that host, so
// the two together are a chain to OpenFOAM with a measured link at each end. Comparing the device
// straight to OpenFOAM would fold both differences into one number and make neither readable.
//
// THE MULESCorr PATH RUNS A LINEAR SOLVE, so the two sides use different Krylov solvers and land on
// the same field to about the pre-solve tolerance, not to round-off. The synthetic gate measured that
// at 6.9e-11 over six steps; this one says what it is on the real mesh.
#include "primitive_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "interface_properties_cpp.cuh"
#include "inter_case_cpp.cuh"
#include "solution_directions.cuh"
#include "inter_driver_cpp.cuh"
#include "inter_solve_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "pbicgstab.cuh"
#include "inter_ueqn_cpp.cuh"
#include "alpha_eqn_cpp.cuh"
#include "device_inter_alpha_step.cuh"
#include "device_inter_step.cuh"
#include "two_phase_mixture_cpp.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

using namespace brae;
using namespace brae::cpu::interFoam;
namespace ip = brae::cpu::interfaceProps;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
std::vector<scalar> flatten(const std::vector<std::vector<scalar>>& b)
{
    std::vector<scalar> v;
    for (const auto& p : b) v.insert(v.end(), p.begin(), p.end());
    return v;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== the device alpha step on damBreak's own case ==\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }
    if (argc < 5)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <nSteps> <deltaT>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1], startDir = argv[2];
    const int nSteps = std::atoi(argv[3]);
    const scalar dt  = static_cast<scalar>(std::atof(argv[4]));

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    // TWO independent reads of the case, so the device run's boundary objects are its own and cannot
    // be advanced by the host run.
    InterFields hostF = buildInterFields(caseDir, startDir, m, g, fvp);
    InterFields devF  = buildInterFields(caseDir, startDir, m, g, fvp);

    // WARM THE CASE UP FIRST, and this is not a convenience. damBreak STARTS FROM REST: U is zero
    // everywhere, so phi is zero, and the alpha equation on its own advects NOTHING -- the flux is
    // made by the pressure corrector out of gravity. Run from 0/ the first version of this gate
    // reported |device - host| = 0 with the interface moved 0 and rhoPhi's scale 0, three green
    // numbers on a comparison of two fields that had not changed. The warm-up runs the WHOLE host
    // solver, so what follows starts from a developed damBreak: a real flux, a real interface, and
    // MULES actually limiting.
    const int nWarm = std::max(nSteps, 10);
    (void)nWarm;
    {
        const RunReport w = runInterFoam(caseDir, startDir, m, g, fvp, nWarm, /*verbose=*/false, &hostF);
        check("the host solver warmed the case up", w.steps == nWarm);
        // the device run starts from the SAME developed state, through its own boundary objects
        devF.alpha1.internal = hostF.alpha1.internal;
        devF.alpha1.evaluateBoundary();
        devF.phi   = hostF.phi;
        devF.nHatf = hostF.nHatf;
        devF.K     = hostF.K;
        scalar maxPhi = 0;
        for (scalar v : devF.phi.internal) maxPhi = std::fmax(maxPhi, std::fabs(v));
        std::printf("  after %d warm-up steps: max |phi| = %.4e\n", nWarm, (double)maxPhi);
        check("the warm-up produced a real flux, which a start from rest does not", maxPhi > scalar(1e-9));
    }

    std::printf("  damBreak: %ld cells, %ld internal faces, %ld boundary faces\n",
                (long)nC, (long)nIf, (long)nBf);
    std::printf("  the case: nAlphaSubCycles %ld, nAlphaCorr %ld, MULESCorr %s, nLimiterIter %ld, "
                "cAlpha %.2f\n",
                (long)devF.alphaCtl.nAlphaSubCycles, (long)devF.alphaCtl.nAlphaCorr,
                devF.alphaCtl.MULESCorr ? "yes" : "no", (long)devF.mulesCtl.nLimiterIter,
                (double)devF.interface.cAlpha);
    {
        int nEmpty = 0;
        for (const FvPatch& q : fvp) if (q.type == "empty") nEmpty += q.size;
        std::printf("  ...and %d of its %ld boundary faces are EMPTY, which MULES must skip\n",
                    nEmpty, (long)nBf);
        check("the real case brings empty patches, which no synthetic box fixture here had",
              nEmpty > 0);
    }

    const std::vector<scalar> a0 = hostF.alpha1.internal;   // the DEVELOPED state, not 0/

    // ---- the HOST alpha half, N steps ------------------------------------------------------------
    AlphaStepInput hin;
    hin.phi = &hostF.phi;
    hin.phiCN = &hostF.phi;
    hin.cAlpha = hostF.interface.cAlpha;
    hin.nAlphaCorr = static_cast<label>(hostF.alphaCtl.nAlphaCorr);
    hin.icAlpha = hostF.alphaCtl.icAlpha;
    hin.scAlpha = hostF.alphaCtl.scAlpha;
    hin.rho1 = hostF.mixture.phases.rho1;
    hin.rho2 = hostF.mixture.phases.rho2;
    hin.MULESCorr = hostF.alphaCtl.MULESCorr;
    hin.alphaScheme  = hostF.divPhiAlpha;
    hin.alpharScheme = hostF.divPhirbAlpha;
    hin.tolAlpha = scalar(1e-12);
    hin.relTolAlpha = 0;
    hin.maxIterAlpha = 2000;

    SurfaceScalarField hAlphaPhi, hRhoPhi;
    AlphaEqnStep hstep =
        [&](const std::vector<scalar>& old, scalar dtSub, std::vector<scalar>& out,
            SurfaceScalarField& rp)
    {
        hin.deltaT = dtSub;
        alphaEqnStep(hostF.alpha1, old, hin, hostF.interface, hostF.mulesCtl, m, g, fvp,
                     hAlphaPhi, rp, hostF.nHatf, hostF.K, nullptr);
        out = hostF.alpha1.internal;
    };
    std::vector<scalar> hostAlpha = a0;
    for (int s = 0; s < nSteps; ++s)
    {
        const std::vector<scalar> old = hostAlpha;
        alphaEqnSubCycle(static_cast<label>(hostF.alphaCtl.nAlphaSubCycles), dt,
                         hostAlpha, old, hRhoPhi, hstep);
        hostF.alpha1.internal = hostAlpha;
        hostF.alpha1.evaluateBoundary();
    }

    // ---- the DEVICE alpha half, the same N steps -------------------------------------------------
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<int> fixes, flag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const int fx = devF.alpha1.boundary[pi]->fixesValue() ? 1 : 0;
        const int fl = (fvp[pi].type == "empty") ? 1 : ((fvp[pi].type == "wedge") ? 2 : 0);
        for (label i = 0; i < fvp[pi].size; ++i) { fixes.push_back(fx); flag.push_back(fl); }
    }
    DeviceBuffer<int> dFixes(fixes), dFlag(flag);
    DeviceBuffer<scalar> dPhiI(devF.phi.internal), dPhiB(flatten(devF.phi.boundary));

    auto patchValues = [&](const GeometricField<scalar>& f)
    {
        std::vector<scalar> v;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& b = f.boundary[pi]->value();
            v.insert(v.end(), b.begin(), b.end());
        }
        return v;
    };

    DeviceInterAlphaHooks hooks;
    hooks.updateBoundary =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aBnd, DeviceBuffer<scalar>& nBnd)
    {
        a.copyTo(devF.alpha1.internal);
        devF.alpha1.evaluateBoundary();
        aBnd.copyFrom(patchValues(devF.alpha1));
        SurfaceScalarField nHb;
        std::vector<scalar> Kb;
        ip::calculateK(devF.alpha1, devF.interface, m, g, fvp, false, nHb, Kb);
        nBnd.copyFrom(flatten(nHb.boundary));
    };
    hooks.divCoeffs =
        [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
    {
        a.copyTo(devF.alpha1.internal);
        devF.alpha1.evaluateBoundary();
        FvScalarMatrix M = fvm::div<scalar>(devF.phi.internal, devF.phi.boundary, devF.alpha1, m, fvp);
        std::vector<scalar> i2, b2;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(M.internalCoeffs[pi][i]); b2.push_back(M.boundaryCoeffs[pi][i]); }
        iC.copyFrom(i2);
        bC.copyFrom(b2);
    };

    DeviceAlphaStepInput din;
    din.phiInt = &dPhiI;
    din.phiBnd = &dPhiB;
    din.phiCNInt = &dPhiI;
    din.phiCNBnd = &dPhiB;
    din.cAlpha = devF.interface.cAlpha;
    din.rho1 = devF.mixture.phases.rho1;
    din.rho2 = devF.mixture.phases.rho2;
    din.deltaN = ip::deltaN(g.V());
    din.alphaScheme  = (devF.divPhiAlpha == AlphaFluxScheme::linear)
                     ? DeviceAlphaScheme::linear
                     : (devF.divPhiAlpha == AlphaFluxScheme::upwind ? DeviceAlphaScheme::upwind
                                                                    : DeviceAlphaScheme::vanLeer);
    din.alpharScheme = (devF.divPhirbAlpha == AlphaFluxScheme::vanLeer)
                     ? DeviceAlphaScheme::vanLeer
                     : (devF.divPhirbAlpha == AlphaFluxScheme::upwind ? DeviceAlphaScheme::upwind
                                                                      : DeviceAlphaScheme::linear);

    DeviceMulesControls dmc;
    dmc.nLimiterIter         = devF.mulesCtl.nLimiterIter;
    dmc.smoothLimiter        = devF.mulesCtl.smoothLimiter;
    dmc.extremaCoeff         = devF.mulesCtl.extremaCoeff;
    dmc.boundaryExtremaCoeff = devF.mulesCtl.boundaryExtremaCoeff;

    DeviceInterAlphaControls dctl;
    dctl.nAlphaSubCycles = static_cast<int>(devF.alphaCtl.nAlphaSubCycles);
    dctl.nAlphaCorr      = static_cast<int>(devF.alphaCtl.nAlphaCorr);
    dctl.MULESCorr       = devF.alphaCtl.MULESCorr;
    dctl.preSolve.tol    = scalar(1e-12);
    dctl.preSolve.relTol = 0;
    dctl.preSolve.maxIter = 2000;

    DevicePhaseProperties props{devF.mixture.phases.rho1, devF.mixture.phases.nu1,
                                devF.mixture.phases.rho2, devF.mixture.phases.nu2};

    DeviceBuffer<scalar> dA(a0), dAOld(a0), dABnd(patchValues(devF.alpha1));
    DeviceBuffer<scalar> dNHatf(devF.nHatf.internal), dNHatfB(flatten(devF.nHatf.boundary));
    DeviceBuffer<scalar> dK(devF.K), rpI, rpB, a2, rho, mu, nu;

    for (int s = 0; s < nSteps; ++s)
    {
        std::vector<scalar> cur;
        dA.copyTo(cur);
        failures += brae::gatecheck::nonFinite("cur", cur);
        dAOld.copyFrom(cur);
        deviceInterAlphaStep(dm, dA, dAOld, dt, din, dmc, dctl, props, hooks,
                             dABnd, dNHatfB, dFixes, dFlag, dNHatf, dK, rpI, rpB, a2, rho, mu, nu);
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }

    std::vector<scalar> devAlpha, devRhoPhi;
    dA.copyTo(devAlpha);
    failures += brae::gatecheck::nonFinite("devAlpha", devAlpha);
    rpI.copyTo(devRhoPhi);
    failures += brae::gatecheck::nonFinite("devRhoPhi", devRhoPhi);

    // ---- the comparison ---------------------------------------------------------------------------
    {
        scalar worst = 0, moved = 0, exc = 0;
        label iw = -1;
        for (label c = 0; c < nC; ++c)
        {
            const scalar e = std::fabs(devAlpha[c] - hostAlpha[c]);
            if (e > worst) { worst = e; iw = c; }
            moved = std::fmax(moved, std::fabs(devAlpha[c] - a0[c]));
            exc   = std::fmax(exc, std::fmax(-devAlpha[c], devAlpha[c] - scalar(1)));
        }
        std::printf("  after %d steps of dt %g: worst |device - host| = %.4e", nSteps, (double)dt,
                    (double)worst);
        if (iw >= 0) std::printf("  (cell %ld: %.17g vs %.17g)", (long)iw,
                                 (double)devAlpha[iw], (double)hostAlpha[iw]);
        std::printf("\n  ...the interface moved %.4e, worst excursion from [0,1] %.3e\n",
                    (double)moved, (double)exc);
        check("the device alpha half tracks the host on damBreak's real mesh", worst < scalar(1e-8));
        check("...having actually advected the interface", moved > scalar(1e-5));
        check("...and alpha is bounded to 1e-6, which is CMULES' fixed-point iteration and the "
              "pre-solve's residual, both measured on the synthetic gate", exc <= scalar(1e-6));

        scalar rw = 0, rs = 0;
        for (label f = 0; f < nIf; ++f)
        {
            rw = std::fmax(rw, std::fabs(devRhoPhi[f] - hRhoPhi.internal[f]));
            rs = std::fmax(rs, std::fabs(hRhoPhi.internal[f]));
        }
        // SCALED BY THE DENSITY RATIO, not by rhoPhi's own magnitude. rhoPhi is
        // alphaPhi*(rho1 - rho2) + phiCN*rho2, so its max is set by the phiCN*rho2 term while its
        // ERROR is the alpha flux's, multiplied by rho1 - rho2 = 999. Comparing the gap to the max
        // therefore measures the ratio of two unrelated things: the arm read 1.2e-10 relative on one
        // warm state and 1.6e-08 on another, from the same code, because the SCALE moved. Dividing
        // the gap by the density ratio recovers the alpha-flux error it actually is, and that has to
        // sit under alpha's own agreement.
        const scalar rhoRatio = devF.mixture.phases.rho1 - devF.mixture.phases.rho2;
        std::printf("  rhoPhi: worst %.4e of %.4e;  /(rho1 - rho2) = %.4e, against alpha's %.4e\n",
                    (double)rw, (double)rs, (double)(rw/rhoRatio), (double)worst);
        check("...and so does the rhoPhi it leaves for the momentum equation -- its error is the "
              "alpha flux's, times the density ratio", rw/rhoRatio < worst);
    }

    // ---- THE WHOLE STEP on damBreak's own mesh and case ------------------------------------------
    // THROUGH runInterFoamDevice -- the function brae_interFoam -device calls, not a copy of it. The
    // hooks used to live here, and a device path whose boundary evaluation is written inside a gate is
    // the defect braeInterFoam.cu's own header names: the gate proves the step and the binary feeds it
    // something else. Both paths below are the shipped ones.
    //
    // WHAT GETTING HERE COST, every one found by running this arm and each worth keeping:
    //   fvc::reconstruct needs OpenFOAM's safeInv -- a 2-D mesh makes the tensor singular and the plain
    //     cofactor inverse gave NaN in all 2268 cells, which std::fmax then hid as 0.000e+00.
    //   the case must be forced to `adjustTimeStep no` -- the host read damBreak's own controlDict and
    //     grew dt while the device loop did not, so the two sat at different physical times and alpha
    //     read 9.57e-01 out. The device loop now adapts too; the arm at the bottom of this file runs
    //     both drivers on the case's own clock, and this one stays fixed so a field difference here
    //     cannot be a clock.
    //   the step computed no fvc::ddtCorr at all, ran ONE pressure corrector where the case asks for
    //     three, hardcoded div(rhoPhi,U) as upwind where the case says `Gauss linearUpwind grad(U)`,
    //     and hardcoded solutionD as all-valid on a 2-D mesh.
    //   UbStored must be U's STORED patch values; filling it with deviceBCValue re-derives them and is
    //     a no-op, which left the dev2 term 100% wrong on the atmosphere alone.
    //   interFoam's pressureCorrector never called updateFromPatchVelocity, so the flux-conditional
    //     velocity patches kept their written seed and UEqn's boundaryCoeffs came out 3.34e-06 where
    //     OpenFOAM's own -- dumped with tools/dumpInterFoam -- are 0.
    //   and this gate's own hooks called mixtureNu(alpha1, alpha2, ...) where the second argument is
    //     mu. Both sides used the same nonsense, so every coefficient agreed to round-off while the
    //     host driver, which calls it correctly, did not. That one was worth 58% of U and it was in the
    //     measurement, not in the code under test.
    //
    // DeviceInterStepTaps is still in the production header for the next such hunt: it hands back rAU,
    // HbyA, phiHbyA and both assembled systems, which is what turned "U is 60% out" into a line.
    {
        InterFields hostF, devF;
        const RunReport rh = runInterFoam(caseDir, startDir, m, g, fvp, nSteps, false, &hostF);
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, fvp, nSteps, false, &devF);
        check("both paths ran the requested steps", rh.steps == nSteps && rd.steps == nSteps);

        std::vector<scalar> dux(static_cast<std::size_t>(nC)), hux(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        { dux[c] = devF.U.internal[c].x; hux[c] = hostF.U.internal[c].x; }
        failures += brae::gatecheck::nonFinite("device alpha", devF.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device U.x", dux);
        failures += brae::gatecheck::nonFinite("device p_rgh", devF.p_rgh.internal);

        scalar wa = 0, wu = 0, wp = 0, sa = 0, su = 0, sp = 0, exc = 0, moved = 0;
        for (label c = 0; c < nC; ++c)
        {
            wa = std::fmax(wa, std::fabs(devF.alpha1.internal[c] - hostF.alpha1.internal[c]));
            wu = std::fmax(wu, std::fabs(dux[c] - hux[c]));
            wp = std::fmax(wp, std::fabs(devF.p_rgh.internal[c] - hostF.p_rgh.internal[c]));
            sa = std::fmax(sa, std::fabs(hostF.alpha1.internal[c]));
            su = std::fmax(su, std::fabs(hux[c]));
            sp = std::fmax(sp, std::fabs(hostF.p_rgh.internal[c]));
            exc = std::fmax(exc, std::fmax(-devF.alpha1.internal[c],
                                           devF.alpha1.internal[c] - scalar(1)));
            moved = std::fmax(moved, std::fabs(devF.alpha1.internal[c] - a0[c]));
        }
        std::printf("  WHOLE STEP through the shipped drivers, %d steps: alpha %.4e of %.3f, "
                    "U.x %.4e of %.4e, p_rgh %.4e of %.4e;  moved %.4e, excursion %.3e\n",
                    nSteps, (double)wa, (double)sa, (double)wu, (double)su,
                    (double)wp, (double)sp, (double)moved, (double)exc);
        check("the device driver tracks the host driver on damBreak", wa < scalar(1e-8));
        check("...in the velocity it leaves", wu < scalar(1e-7)*std::fmax(su, scalar(1e-30)));
        check("...and in the pressure", wp < scalar(1e-8)*std::fmax(sp, scalar(1e-30)));
        check("...with alpha bounded", exc <= scalar(1e-6));
        // damBreak starts FROM REST, so the alpha equation alone advects nothing -- the flux is made by
        // the pressure corrector out of gravity. That the interface moved at all is what says the
        // whole loop ran and not just its alpha half.
        check("...having advected the interface, which a start from rest only does through the "
              "pressure corrector", moved > scalar(1e-4));
    }

    // THE SAME STEP ON THE CASE'S OWN CLOCK.
    // Every arm above runs at a FIXED dt, because the host driver read damBreak's `adjustTimeStep yes`
    // and grew its step while the device loop took whatever it was handed -- two runs at different
    // physical times, which read as alpha 9.57e-01 out and look like a discretisation error.
    //
    // That is now a wiring question rather than a missing feature: the device loop computes both
    // Courant numbers off the flux the last step left behind and runs them through the same
    // setDeltaTVoF the host calls. This arm hands both drivers a copy of the case with its own
    // `adjustTimeStep yes` restored and asks whether they walk the same clock.
    //
    // THE CONTROL IS THAT THE COURANT NUMBER REACHES dt AT ALL. setDeltaTVoF takes
    // min(maxCo/Co, 1 + 0.1*maxCo/Co, 1.2), so below maxCo/Co = 2 the answer is the constant 1.2 and
    // dt rides the cap whatever Co says. At damBreak's own maxCo 1 that is what happens for its first
    // twenty-three steps, and the first draft of this arm did exactly that: host and device both grew
    // dt by 1.2^5 to 2.488320e-04, agreed to 0.000e+00, and every check below passed on a pair of runs
    // in which the ported Courant number was never consulted. The shell therefore hands this arm a
    // case at maxCo 0.0015, where dt is Co's own -- and the check below refuses the cap trajectory by
    // name, so the arm cannot quietly go vacuous again if the case or the schedule changes.
    if (argc > 5)
    {
        const std::string adaptDir = argv[5];
        // TWELVE steps, not this file's five, and the reason is nearInterface(). damBreak's setFields
        // leaves a SHARP interface -- every cell exactly 0 or exactly 1 -- so the [0.01, 0.99] band is
        // EMPTY and alphaCoNum is identically zero. Measured: it first goes nonzero at step 11. At five
        // steps the VoF Courant check below read 0.0000e+00 on both sides and passed on nothing.
        const label nAdapt = nSteps > 12 ? nSteps : 12;
        InterFields hostF, devF;
        const RunReport rh = runInterFoam(adaptDir, adaptDir + "/0", m, g, fvp, nAdapt, false, &hostF);
        const RunReport rd = runInterFoamDevice(adaptDir, adaptDir + "/0", m, g, fvp, nAdapt, false, &devF);

        scalar wa = 0, wu = 0, sa = 0, su = 0;
        for (label c = 0; c < nC; ++c)
        {
            wa = std::fmax(wa, std::fabs(devF.alpha1.internal[c] - hostF.alpha1.internal[c]));
            wu = std::fmax(wu, std::fabs(devF.U.internal[c].x - hostF.U.internal[c].x));
            sa = std::fmax(sa, std::fabs(hostF.alpha1.internal[c]));
            su = std::fmax(su, std::fabs(hostF.U.internal[c].x));
        }
        const scalar dtRel = std::fabs(rd.deltaT - rh.deltaT)/std::fmax(rh.deltaT, scalar(1e-30));
        const scalar tRel = std::fabs(rd.time - rh.time)/std::fmax(rh.time, scalar(1e-30));
        const scalar coRel = std::fabs(rd.CoNum - rh.CoNum)/std::fmax(rh.CoNum, scalar(1e-30));
        const scalar acRel = std::fabs(rd.alphaCoNum - rh.alphaCoNum)
                           / std::fmax(rh.alphaCoNum, scalar(1e-30));
        scalar capped = dt;
        for (label k = 0; k < nAdapt; ++k)
        {
            capped *= scalar(1.2);
        }

        std::printf("  THE CASE'S OWN CLOCK, %d steps: host dt %.6e -> t %.6e, device dt %.6e -> "
                    "t %.6e  (dt %.3e, t %.3e);  Co %.4e/%.4e, alphaCo %.4e/%.4e;  "
                    "alpha %.4e of %.3f, U.x %.4e of %.4e\n",
                    nAdapt, (double)rh.deltaT, (double)rh.time, (double)rd.deltaT, (double)rd.time,
                    (double)dtRel, (double)tRel, (double)rh.CoNum, (double)coRel,
                    (double)rh.alphaCoNum, (double)acRel, (double)wa, (double)sa,
                    (double)wu, (double)su);
        check("dt is the Courant number's and not setDeltaT's 1.2 cap, so this arm is not vacuous",
              std::fabs(rh.deltaT - capped) > scalar(0.01)*rh.deltaT);
        // Both Courant numbers are compared outright, not just their effect on dt: the alpha one does
        // not bind here -- from rest the interface band carries almost no flux -- so nothing else in
        // this arm would notice if deviceAlphaCourantNo's mask were wrong.
        check("...and the device computed it itself", rh.CoNum > scalar(0) && coRel < scalar(1e-9));
        check("...along with the VoF one, which does not bind here and so is checked outright",
              rh.alphaCoNum > scalar(0) && acRel < scalar(1e-9));
        check("the device picks the host's time step from its own Courant numbers", dtRel < scalar(1e-9));
        check("...so both land at the same physical time", tRel < scalar(1e-9));
        check("...and on the same interface", wa < scalar(1e-8));
        check("...and the same velocity", wu < scalar(1e-7)*std::fmax(su, scalar(1e-30)));
    }
    else
    {
        std::printf("  (no adaptive-clock case given; skipping the adjustTimeStep arm)\n");
    }

    std::printf("test_device_inter_dambreak_alpha: %d failures\n", failures);
    return failures ? 1 : 0;
}
