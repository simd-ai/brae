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
        dAOld.copyFrom(cur);
        deviceInterAlphaStep(dm, dA, dAOld, dt, din, dmc, dctl, props, hooks,
                             dABnd, dNHatfB, dFixes, dFlag, dNHatf, dK, rpI, rpB, a2, rho, mu, nu);
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }

    std::vector<scalar> devAlpha, devRhoPhi;
    dA.copyTo(devAlpha);
    rpI.copyTo(devRhoPhi);

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
    // Every one of interFoam's halves, on the case the host solver is itself gated against real
    // OpenFOAM on: fixedFluxPressure on three walls whose gradient constrainPressure PRESCRIBES from
    // phiHbyA, totalPressure at the atmosphere, noSlip walls, a pressureInletOutletVelocity outlet,
    // and 4536 EMPTY faces. None of that appears on a box fixture.
    //
    // WHAT GETTING HERE COST, every one found by running this arm and each worth keeping:
    //   fvc::reconstruct needs OpenFOAM's safeInv -- a 2-D mesh makes the tensor singular and the plain
    //     cofactor inverse gave NaN in all 2268 cells, which std::fmax then hid as 0.000e+00.
    //   the case must be forced to `adjustTimeStep no` -- the host reads damBreak's own controlDict and
    //     grows dt, so the two sat at different physical times and alpha read 9.57e-01 out.
    //   the step computed no fvc::ddtCorr at all, ran ONE pressure corrector where the case asks for
    //     three, hardcoded div(rhoPhi,U) as upwind where the case says linearUpwind, and hardcoded
    //     solutionD as all-valid on a 2-D mesh.
    //   UbStored must be U's STORED patch values; filling it with deviceBCValue re-derives them and is
    //     a no-op, which left the dev2 term 100% wrong on the atmosphere alone.
    //   interFoam's pressureCorrector never called updateFromPatchVelocity, so the flux-conditional
    //     velocity patches kept their written seed and UEqn's boundaryCoeffs came out 3.34e-06 where
    //     OpenFOAM's own -- dumped with tools/dumpInterFoam -- are 0.
    //   and THIS FILE's own hooks called mixtureNu(alpha1, alpha2, ...) where its second argument is
    //     mu: nu = mu/(clamped blend), not a complement. Both sides used the same nonsense, so they
    //     agreed with each other while runInterFoam, which calls it correctly, did not. That one was
    //     worth 58% of U and it was in the gate, not in the code under test.
    //
    // BRAE_INTER_STEP_CHECK prints the first stage whose output is not finite; the taps below compare
    // the momentum and pressure systems coefficient by coefficient.
    {
        InterFields ref = buildInterFields(caseDir, startDir, m, g, fvp);
        const RunReport w0 = runInterFoam(caseDir, startDir, m, g, fvp, nWarm + nSteps,
                                          /*verbose=*/false, &ref);
        check("the host solver ran the warm-up plus the compared steps", w0.steps == nWarm + nSteps);

        InterFields dv = buildInterFields(caseDir, startDir, m, g, fvp);
        {
            InterFields warm = buildInterFields(caseDir, startDir, m, g, fvp);
            runInterFoam(caseDir, startDir, m, g, fvp, nWarm, /*verbose=*/false, &warm);
            dv.alpha1.internal = warm.alpha1.internal;
            dv.U.internal      = warm.U.internal;
            dv.p_rgh.internal  = warm.p_rgh.internal;
            dv.phi = warm.phi;  dv.nHatf = warm.nHatf;  dv.K = warm.K;  dv.rho = warm.rho;
            dv.alpha1.evaluateBoundary();
            dv.U.evaluateBoundary();
            dv.p_rgh.evaluateBoundary();
        }
        auto pv2 = [&](const GeometricField<scalar>& f)
        { std::vector<scalar> v;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          { const auto& b = f.boundary[pi]->value(); v.insert(v.end(), b.begin(), b.end()); }
          return v; };
        const label nFacesAll = static_cast<label>(g.magSf().size());
        std::vector<scalar> ghfAll;
        auto fullFace = [&](const SurfaceScalarField& f)
        { std::vector<scalar> v(f.internal);
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
              v.insert(v.end(), f.boundary[pi].begin(), f.boundary[pi].end());
          return v; };

        DeviceInterStepHooks H;
        H.alpha.updateBoundary =
            [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& aB, DeviceBuffer<scalar>& nB)
        { a.copyTo(dv.alpha1.internal); dv.alpha1.evaluateBoundary(); aB.copyFrom(pv2(dv.alpha1));
          SurfaceScalarField nHb; std::vector<scalar> Kb;
          ip::calculateK(dv.alpha1, dv.interface, m, g, fvp, false, nHb, Kb);
          nB.copyFrom(flatten(nHb.boundary)); };
        H.alpha.divCoeffs =
            [&](const DeviceBuffer<scalar>& a, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
        { a.copyTo(dv.alpha1.internal); dv.alpha1.evaluateBoundary();
          FvScalarMatrix M = fvm::div<scalar>(dv.phi.internal, dv.phi.boundary, dv.alpha1, m, fvp);
          std::vector<scalar> i2, b2;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(M.internalCoeffs[pi][i]); b2.push_back(M.boundaryCoeffs[pi][i]); }
          iC.copyFrom(i2); bC.copyFrom(b2); };
        H.updateUBoundary =
            [&](const DeviceBuffer<scalar>& ax, const DeviceBuffer<scalar>& ay,
                const DeviceBuffer<scalar>& az, DeviceVectorBoundary& db,
                DeviceBuffer<scalar>* ubOut)
        { std::vector<scalar> x, y, z; ax.copyTo(x); ay.copyTo(y); az.copyTo(z);
          for (label c = 0; c < nC; ++c) dv.U.internal[c] = vector{x[c], y[c], z[c]};
          dv.U.evaluateBoundary();
          // the flux-conditional velocity patches, as the solver's pressureCorrector does -- see the
          // note there. Both sides of this comparison must resolve them the same way or the dev2 term
          // reads a different U at the patch.
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          {
              std::vector<vector> Uc(static_cast<std::size_t>(fvp[pi].size), vector{0, 0, 0});
              for (label i = 0; i < fvp[pi].size; ++i)
                  Uc[i] = dv.U.internal[fvp[pi].faceCells[i]];
              dv.U.boundary[pi]->updateFromPatchVelocity(dv.U.boundary[pi]->value(), Uc, {});
          }
          db = buildDeviceVectorBoundary(dv.U, fvp, g);
          // U's STORED patch values, per component -- what fvc::grad(U) inside divDevRhoReff reads.
          if (ubOut)
          {
              std::vector<scalar> bx, by, bz;
              for (std::size_t pi = 0; pi < fvp.size(); ++pi)
              {
                  const std::vector<vector>& v = dv.U.boundary[pi]->value();
                  for (const vector& u : v) { bx.push_back(u.x); by.push_back(u.y); bz.push_back(u.z); }
              }
              ubOut[0].copyFrom(bx);
              ubOut[1].copyFrom(by);
              ubOut[2].copyFrom(bz);
          } };
        H.interfaceForces =
            [&](const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& Kd,
                const DeviceBuffer<scalar>& rd, DeviceBuffer<scalar>& stf,
                DeviceBuffer<scalar>& snRho, DeviceBuffer<scalar>& nuC, DeviceBuffer<scalar>& nuB,
                DeviceBuffer<scalar>& snP)
        { a.copyTo(dv.alpha1.internal); dv.alpha1.evaluateBoundary();
          Kd.copyTo(dv.K); rd.copyTo(dv.rho);
          std::vector<scalar> sK; ip::sigmaK(dv.K, dv.interface.sigma, sK);
          const SurfaceScalarField sKf = fvc::interpolate(sK, m, g, fvp);
          const SurfaceScalarField snA = fvc::snGrad(dv.alpha1, m, g, fvp, false);
          SurfaceScalarField t;
          t.internal.resize(static_cast<std::size_t>(nIf));
          for (label f = 0; f < nIf; ++f) t.internal[f] = sKf.internal[f]*snA.internal[f];
          t.boundary.resize(fvp.size());
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
              t.boundary[pi].push_back(sKf.boundary[pi][i]*snA.boundary[pi][i]);
          stf.copyFrom(fullFace(t));
          GeometricField<scalar> rhoF;
          rhoF.internal = dv.rho;
          for (const FvPatch& q : fvp)
            rhoF.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
          rhoF.evaluateBoundary();
          snRho.copyFrom(fullFace(fvc::snGrad(rhoF, m, g, fvp, false)));
          // mixtureNu's SECOND argument is mu, not alpha2 (two_phase_mixture_cpp.cuh:130):
          // nu = mu/(clamped blend). Passing the complement there gives nonsense, and because both
          // sides of this comparison used the same nonsense they agreed with each other while
          // runInterFoam -- which calls it correctly -- did not.
          std::vector<scalar> muv, nuv;
          brae::cpu::twoPhase::mixtureMu(dv.alpha1.internal, dv.mixture.phases, muv);
          brae::cpu::twoPhase::mixtureNu(dv.alpha1.internal, muv, dv.mixture.phases, nuv);
          nuC.copyFrom(nuv);
          // nuEff AT ALPHA'S PATCH VALUES, not at the face cell's -- the same rule the boundary rho
          // follows, and for the same reason: at a contact-angle wall they are different fields.
          std::vector<scalar> nb, abv;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          { const std::vector<scalar>& v = dv.alpha1.boundary[pi]->value();
            abv.insert(abv.end(), v.begin(), v.end()); }
          { std::vector<scalar> abMu;
            brae::cpu::twoPhase::mixtureMu(abv, dv.mixture.phases, abMu);
            brae::cpu::twoPhase::mixtureNu(abv, abMu, dv.mixture.phases, nb); }
          nuB.copyFrom(nb);
          snP.resize(0); };
        H.pressure.pressureCoeffs =
            [&](const DeviceBuffer<scalar>&, const DeviceBuffer<scalar>& phiHB,
                const DeviceBuffer<scalar>& rAUfAll, DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC)
        { std::vector<scalar> hB, rA; phiHB.copyTo(hB); rAUfAll.copyTo(rA);
          // rAUf PER FACE, from the full array -- the internal faces first, then the patches in order.
          // Standing in rA[0] for every boundary face put U 60% out on this case.
          label off = 0;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          { const FvPatch& q = fvp[pi];
            if (dv.p_rgh.boundary[pi]->updateableSnGrad())
            { std::vector<scalar> sn(static_cast<std::size_t>(q.size));
              for (label i = 0; i < q.size; ++i)
                sn[i] = (hB[off + i] - dv.phi.boundary[pi][i]) / (q.magSf[i] * rA[nIf + off + i]);
              dv.p_rgh.boundary[pi]->updateSnGrad(sn); }
            off += q.size; }
          SurfaceScalarField rf2;
          rf2.internal.assign(rA.begin(), rA.begin() + nIf);
          rf2.boundary.resize(fvp.size());
          { label o2 = nIf;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
              for (label i = 0; i < fvp[pi].size; ++i) rf2.boundary[pi].push_back(rA[o2++]); }
          FvScalarMatrix pe = fvm::laplacian<scalar>(rf2, dv.p_rgh, m, g, fvp, false);
          std::vector<scalar> i2, b2;
          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { i2.push_back(pe.internalCoeffs[pi][i]); b2.push_back(pe.boundaryCoeffs[pi][i]); }
          iC.copyFrom(i2); bC.copyFrom(b2); };
        H.pressure.updateBoundary =
            [&](const DeviceBuffer<scalar>& pr)
        { pr.copyTo(dv.p_rgh.internal); dv.p_rgh.evaluateBoundary(); };

        std::vector<int> takeU, uFixes;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
          for (label i = 0; i < fvp[pi].size; ++i)
          { takeU.push_back(dv.U.boundary[pi]->assignable() ? 0 : 1);
            uFixes.push_back(dv.U.boundary[pi]->fixesValue() ? 1 : 0); }
        DeviceBuffer<int> dTakeU(takeU), dUFixes(uFixes);

        DeviceInterStepControls C;
        C.alpha = dctl;  C.mules = dmc;  C.alphaInput = din;
        C.momentum.tol = scalar(1e-12);
        C.pressure.tol = scalar(1e-12);
        C.pressure.maxIter = 4000;
        C.nCorrectors = static_cast<int>(dv.pimple.nCorrectors);
        // the case's own div(rhoPhi,U). damBreak says `Gauss linearUpwind grad(U)`; the two enums are
        // separate types with the same members, so the mapping is written out rather than cast.
        switch (dv.divRhoPhiU)
        {
            case DivScheme::linearUpwind:
                C.divScheme = brae::cpu::DivScheme::linearUpwind;  break;
            case DivScheme::linearUpwindV:
                C.divScheme = brae::cpu::DivScheme::linearUpwindV; break;
            case DivScheme::limitedLinear:
                C.divScheme = brae::cpu::DivScheme::limitedLinear; break;
            case DivScheme::limitedLinearV:
                C.divScheme = brae::cpu::DivScheme::limitedLinearV; break;
            case DivScheme::LUST:
                C.divScheme = brae::cpu::DivScheme::LUST;          break;
            default:
                C.divScheme = brae::cpu::DivScheme::upwind;        break;
        }
        C.divSchemeCoeff = dv.divRhoPhiUCoeff;
        std::printf("  div(rhoPhi,U) is %s\n",
                    dv.divRhoPhiU == DivScheme::linearUpwind ? "Gauss linearUpwind grad(U)" : "another scheme");
        { const SolutionDirections sd = solutionDirections(fvp);
          for (int k = 0; k < 3; ++k) C.solutionD[k] = sd.d[k];
          std::printf("  solutionD = (%d %d %d) -- damBreak is 2-D, so one direction is knocked out\n",
                      sd.d[0], sd.d[1], sd.d[2]); }
        C.momentumPredictor = dv.momentumPredictorOn;
        C.relaxU = dv.relaxU;
        C.relaxEquationU = dv.relaxEquationU;
        C.takeUAtBoundary = &dTakeU;

        std::vector<scalar> x0(nC), y0(nC), z0(nC);
        for (label c = 0; c < nC; ++c)
        { x0[c] = dv.U.internal[c].x; y0[c] = dv.U.internal[c].y; z0[c] = dv.U.internal[c].z; }
        DeviceBuffer<scalar> A2(dv.alpha1.internal), AO(dv.alpha1.internal);
        DeviceBuffer<scalar> Ux(x0), Uy(y0), Uz(z0), Uox(x0), Uoy(y0), Uoz(z0);
        DeviceBuffer<scalar> PhI(dv.phi.internal), PhB(flatten(dv.phi.boundary));
        DeviceBuffer<scalar> Prgh(dv.p_rgh.internal), Pf;
        DeviceBuffer<scalar> NH(dv.nHatf.internal), NHB(flatten(dv.nHatf.boundary));
        DeviceBuffer<scalar> ABnd(pv2(dv.alpha1)), Kd2(dv.K), Gh(dv.gh), Ghf, MagSf(g.magSf());
        { SurfaceScalarField gf; gf.internal = dv.ghfInternal; gf.boundary = dv.ghfBoundary;
          ghfAll = fullFace(gf);
          Ghf.copyFrom(ghfAll); }
        DeviceBuffer<scalar> Rho2, Mu2, Nu2, RpI2, RpB2;
        DeviceVectorBoundary db2 = buildDeviceVectorBoundary(dv.U, fvp, g);
        DevicePhaseProperties pr2{dv.mixture.phases.rho1, dv.mixture.phases.nu1,
                                  dv.mixture.phases.rho2, dv.mixture.phases.nu2};

        // the warm state, kept BEFORE the run: the hooks write dv.alpha1.internal every corrector, so
        // by the end it holds the device's own answer and cannot serve as a reference point.
        const std::vector<scalar> warmAlpha = dv.alpha1.internal;
        const std::vector<scalar> warmPrgh  = dv.p_rgh.internal;

        // ---- ONE step with the taps open, against the host's own UEqn from the SAME state ---------
        // A final U that is 60% out cannot say which of a dozen operators did it. These can.
        {
            DeviceInterStepTaps taps;
            DeviceBuffer<scalar> a1(warmAlpha), a1o(warmAlpha);
            DeviceBuffer<scalar> ux(x0), uy(y0), uz(z0), uox(x0), uoy(y0), uoz(z0);
            DeviceBuffer<scalar> phI(dv.phi.internal), phB(flatten(dv.phi.boundary));
            DeviceBuffer<scalar> phOI(dv.phi.internal), phOB(flatten(dv.phi.boundary));
            DeviceBuffer<scalar> pr(dv.p_rgh.internal), pf2;
            DeviceBuffer<scalar> nh(dv.nHatf.internal), nhb(flatten(dv.nHatf.boundary));
            DeviceBuffer<scalar> ab(pv2(dv.alpha1)), kk(dv.K);
            DeviceBuffer<scalar> rr, mm, nn, rpi, rpb;
            DeviceVectorBoundary dbt = buildDeviceVectorBoundary(dv.U, fvp, g);
            deviceInterStep(dm, dt, C, pr2, H, Gh, Ghf, MagSf, a1, a1o, ux, uy, uz, uox, uoy, uoz,
                            phI, phB, phOI, phOB, dUFixes, pr, pf2, nh, nhb, ab, kk,
                            dFixes, dFlag, dbt, rr, mm, nn, rpi, rpb, &taps);
            cudaDeviceSynchronize();

            // The HOST's momentum matrix from the same post-alpha state, so rAU and HbyA compare.
            std::vector<scalar> devRho, rpi_host_int;
            rr.copyTo(devRho);
            rpi.copyTo(rpi_host_int);

            std::vector<scalar> dRAU, dHx, dDiag;
            taps.rAU.copyTo(dRAU);
            taps.HbyA[0].copyTo(dHx);
            taps.UEqnDiag.copyTo(dDiag);

            scalar mnR = dRAU.empty() ? 0 : dRAU[0], mxR = mnR;
            for (scalar v : dRAU) { mnR = std::fmin(mnR, v); mxR = std::fmax(mxR, v); }
            scalar mxH = 0;
            for (scalar v : dHx) mxH = std::fmax(mxH, std::fabs(v));
            scalar mnD = dDiag.empty() ? 0 : dDiag[0], mxD = mnD;
            for (scalar v : dDiag) { mnD = std::fmin(mnD, v); mxD = std::fmax(mxD, v); }
            std::printf("  [taps] rAU in [%.4e, %.4e];  max|HbyA.x| %.4e;  UEqn.diag in [%.4e, %.4e]\n",
                        (double)mnR, (double)mxR, (double)mxH, (double)mnD, (double)mxD);

            // rho*V/dt is what the ddt alone puts on the diagonal, so A() = diag/V should be at least
            // rho/dt everywhere and rAU at most dt/rho. A rAU far above that means the diagonal is
            // missing the ddt, or the relax, or the boundary fold.
            scalar worstBound = 0;
            for (label c = 0; c < nC; ++c)
            {
                const scalar rAUmax = dt / devRho[c];
                worstBound = std::fmax(worstBound, dRAU[c] / rAUmax);
            }
            std::printf("  [taps] worst rAU/(dt/rho) = %.4f  (must be <= 1: the ddt alone puts "
                        "rho*V/dt on the diagonal)\n", (double)worstBound);
            check("rAU is bounded by dt/rho, so the ddt is on the diagonal", worstBound <= scalar(1.0));

            // ---- the HOST's own UEqn from the SAME post-alpha state ------------------------------
            // Both sides are handed the DEVICE's rhoPhi, rho and alpha, so what is compared is the
            // momentum assembly and A()/H() alone -- the alpha half is already gated at 6.9e-11 and
            // folding it in again would only blur this.
            {
                std::vector<scalar> devRhoPhiB, devAlphaNow;
                rpb.copyTo(devRhoPhiB);
                a1.copyTo(devAlphaNow);

                std::vector<std::vector<scalar>> rpBndH(fvp.size()), rhoBndH(fvp.size()),
                                                 nuBndH(fvp.size());
                { label o3 = 0;
                  for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                  { for (label i = 0; i < fvp[pi].size; ++i) rpBndH[pi].push_back(devRhoPhiB[o3 + i]);
                    o3 += fvp[pi].size; } }

                std::vector<scalar> muH, nuH, rhoOldH;
                brae::cpu::twoPhase::mixtureMu(devAlphaNow, dv.mixture.phases, muH);
                brae::cpu::twoPhase::mixtureNu(devAlphaNow, muH, dv.mixture.phases, nuH);
                { std::vector<scalar> a2o(static_cast<std::size_t>(nC));
                  for (label c = 0; c < nC; ++c) a2o[c] = scalar(1) - warmAlpha[c];
                  brae::cpu::twoPhase::mixtureRho(warmAlpha, a2o, dv.mixture.phases, rhoOldH); }
                // the host reference's boundary mixture, from ALPHA'S PATCH VALUES too -- InterFields
                // builds rhoBnd/nuBnd that way and it was worth 12.8% on capillaryRise.
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    const std::vector<scalar>& av = dv.alpha1.boundary[pi]->value();
                    for (label i = 0; i < fvp[pi].size; ++i)
                    {
                        const scalar a1b = av[i];
                        const scalar a2b = scalar(1) - a1b;
                        rhoBndH[pi].push_back(a1b*dv.mixture.phases.rho1 + a2b*dv.mixture.phases.rho2);
                        std::vector<scalar> one{a1b}, mu1, outv;
                        brae::cpu::twoPhase::mixtureMu(one, dv.mixture.phases, mu1);
                        brae::cpu::twoPhase::mixtureNu(one, mu1, dv.mixture.phases, outv);
                        nuBndH[pi].push_back(outv[0]);
                        (void)a2b;
                    }
                }

                std::vector<vector> UOldH(static_cast<std::size_t>(nC));
                for (label c = 0; c < nC; ++c) UOldH[c] = vector{x0[c], y0[c], z0[c]};

                GeometricField<vector> Uh2;
                Uh2.internal = UOldH;
                InterFields tmpf = buildInterFields(caseDir, startDir, m, g, fvp);
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    Uh2.boundary.push_back(std::move(tmpf.U.boundary[pi]));
                Uh2.evaluateBoundary();
                // ...and the flux-conditional velocity patches, as the solver's pressureCorrector now
                // does. Without it the REFERENCE carries the written seed at
                // pressureInletOutletVelocity and its boundaryCoeffs come out 3.34e-06 where OpenFOAM's
                // own are 0 -- so the comparison would be against a matrix brae's solver does not
                // build either. MEASURED with tools/dumpInterFoam: OpenFOAM |bC| = 0 on that patch.
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    std::vector<vector> Uc(static_cast<std::size_t>(fvp[pi].size), vector{0, 0, 0});
                    for (label i = 0; i < fvp[pi].size; ++i)
                        Uc[i] = Uh2.internal[fvp[pi].faceCells[i]];
                    Uh2.boundary[pi]->updateFromPatchVelocity(Uh2.boundary[pi]->value(), Uc, {});
                }

                InterMomentumInput hm;
                hm.rhoPhi = &rpi_host_int;
                hm.rhoPhiBnd = &rpBndH;
                hm.rho = &devRho;
                hm.rhoOld = &rhoOldH;
                hm.rhoBnd = &rhoBndH;
                hm.UOld = &UOldH;
                hm.nuEff = &nuH;
                hm.nuEffBnd = &nuBndH;
                hm.deltaT = dt;
                hm.scheme = dv.divRhoPhiU;
                hm.schemeCoeff = dv.divRhoPhiUCoeff;
                hm.relaxEquationU = dv.relaxEquationU;
                hm.relaxU = dv.relaxU;
                const FvVectorMatrix hUEqn = assembleUEqn(Uh2, hm, m, g, fvp);

                const std::vector<scalar> Ah = matrixA(hUEqn, m, g, fvp);
                const std::vector<vector> Hh = matrixH(hUEqn, Uh2, m, g, fvp);

                // rho.oldTime(), measured rather than argued about: with viscosity and relax both
                // ruled out the source IS the ddt, and the ddt has exactly three inputs.
                {
                    std::vector<scalar> dRhoOld;
                    taps.ddtRhoOld.copyTo(dRhoOld);
                    scalar wRo = 0, sRo = 0;
                    label iw2 = -1;
                    for (label c = 0; c < nC; ++c)
                    {
                        const scalar e = std::fabs(dRhoOld[c] - rhoOldH[c]);
                        if (e > wRo) { wRo = e; iw2 = c; }
                        sRo = std::fmax(sRo, std::fabs(rhoOldH[c]));
                    }
                    std::printf("  [bisect] rho.oldTime(): %.4e of %.4e", (double)wRo, (double)sRo);
                    if (iw2 >= 0)
                        std::printf("  (cell %ld: device %.17g, host %.17g; alpha_old there %.17g)",
                                    (long)iw2, (double)dRhoOld[iw2], (double)rhoOldH[iw2],
                                    (double)warmAlpha[iw2]);
                    std::printf("\n");
                }

                std::vector<scalar> dSrc;
                taps.UEqnSourceX.copyTo(dSrc);
                scalar wS = 0, sS = 0;
                for (label c = 0; c < nC; ++c)
                {
                    wS = std::fmax(wS, std::fabs(dSrc[c] - hUEqn.source[c].x));
                    sS = std::fmax(sS, std::fabs(hUEqn.source[c].x));
                }
                std::printf("  [taps vs host] UEqn.source.x %.4e of %.4e\n", (double)wS, (double)sS);

                // BISECT THE SOURCE. Assemble the host matrix again with the viscosity ZEROED: the
                // ddt source and the relax contribution survive, the explicit dev2(T(grad U)) term
                // does not. If the device's own zero-viscosity source then agrees, the gap is the
                // dev2 term; if it does not, it is the ddt or the relax.
                {
                    std::vector<scalar> zeroNu(static_cast<std::size_t>(nC), scalar(0));
                    std::vector<std::vector<scalar>> zeroNuB(fvp.size());
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                        zeroNuB[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
                    InterMomentumInput hm0 = hm;
                    hm0.nuEff = &zeroNu;
                    hm0.nuEffBnd = &zeroNuB;
                    const FvVectorMatrix h0 = assembleUEqn(Uh2, hm0, m, g, fvp);

                    DeviceInterStepTaps t0;
                    DeviceInterStepControls C0 = C;
                    DeviceBuffer<scalar> b1(warmAlpha), b1o(warmAlpha);
                    DeviceBuffer<scalar> vx(x0), vy(y0), vz(z0), vox(x0), voy(y0), voz(z0);
                    DeviceBuffer<scalar> qI(dv.phi.internal), qB(flatten(dv.phi.boundary));
                    DeviceBuffer<scalar> qOI(dv.phi.internal), qOB(flatten(dv.phi.boundary));
                    DeviceBuffer<scalar> qr(dv.p_rgh.internal), qp;
                    DeviceBuffer<scalar> qn(dv.nHatf.internal), qnb(flatten(dv.nHatf.boundary));
                    DeviceBuffer<scalar> qab(pv2(dv.alpha1)), qk(dv.K);
                    DeviceBuffer<scalar> qrho, qmu, qnu, qri, qrb;
                    DeviceVectorBoundary db0 = buildDeviceVectorBoundary(dv.U, fvp, g);
                    // the hook that zeroes nuEff on the device side too
                    DeviceInterStepHooks H0 = H;
                    H0.interfaceForces =
                        [&](const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& Kd,
                            const DeviceBuffer<scalar>& rd, DeviceBuffer<scalar>& stf,
                            DeviceBuffer<scalar>& snRho, DeviceBuffer<scalar>& nuC,
                            DeviceBuffer<scalar>& nuB, DeviceBuffer<scalar>& snP)
                    { H.interfaceForces(a, Kd, rd, stf, snRho, nuC, nuB, snP);
                      nuC.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nC), scalar(0)));
                      nuB.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nBf), scalar(0))); };
                    deviceInterStep(dm, dt, C0, pr2, H0, Gh, Ghf, MagSf, b1, b1o, vx, vy, vz,
                                    vox, voy, voz, qI, qB, qOI, qOB, dUFixes, qr, qp, qn, qnb,
                                    qab, qk, dFixes, dFlag, db0, qrho, qmu, qnu, qri, qrb, &t0);
                    cudaDeviceSynchronize();
                    std::vector<scalar> s0;
                    t0.UEqnSourceX.copyTo(s0);
                    scalar w0 = 0, x0s = 0;
                    for (label c = 0; c < nC; ++c)
                    {
                        w0  = std::fmax(w0, std::fabs(s0[c] - h0.source[c].x));
                        x0s = std::fmax(x0s, std::fabs(h0.source[c].x));
                    }
                    std::printf("  [bisect] with nuEff ZEROED: source.x %.4e of %.4e\n",
                                (double)w0, (double)x0s);

                    // THE dev2 CONTRIBUTION ALONE, both sides, by difference. Split interior from
                    // boundary-adjacent, which is the split that has named every boundary defect in
                    // this tree: exact inside and wrong at the wall is a boundary treatment, not a
                    // scheme.
                    std::vector<char> touchesPatch(static_cast<std::size_t>(nC), 0);
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        if (fvp[pi].type == "empty") continue;
                        for (label i = 0; i < fvp[pi].size; ++i)
                            touchesPatch[fvp[pi].faceCells[i]] = 1;
                    }
                    std::vector<scalar> dFull;
                    taps.UEqnSourceX.copyTo(dFull);
                    scalar wIn = 0, sIn = 0, wBd = 0, sBd = 0;
                    int nIn = 0, nBd = 0;
                    for (label c = 0; c < nC; ++c)
                    {
                        const scalar dDev = dFull[c] - s0[c];
                        const scalar hDev = hUEqn.source[c].x - h0.source[c].x;
                        const scalar e = std::fabs(dDev - hDev);
                        if (touchesPatch[c]) { wBd = std::fmax(wBd, e); sBd = std::fmax(sBd, std::fabs(hDev)); ++nBd; }
                        else                 { wIn = std::fmax(wIn, e); sIn = std::fmax(sIn, std::fabs(hDev)); ++nIn; }
                    }
                    std::printf("  [dev2 alone] interior (%d cells) %.4e of %.4e;  "
                                "patch-adjacent (%d cells) %.4e of %.4e\n",
                                nIn, (double)wIn, (double)sIn, nBd, (double)wBd, (double)sBd);

                    // ...and BY PATCH, with each patch's declared BC type beside it. Which condition
                    // is at fault is the whole question, and a single patch-adjacent number cannot
                    // say. damBreak brings three noSlip walls, a pressureInletOutletVelocity
                    // atmosphere and two EMPTY patches.
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        scalar wp = 0, sp = 0;
                        for (label i = 0; i < fvp[pi].size; ++i)
                        {
                            const label c = fvp[pi].faceCells[i];
                            const scalar dDev = dFull[c] - s0[c];
                            const scalar hDev = hUEqn.source[c].x - h0.source[c].x;
                            wp = std::fmax(wp, std::fabs(dDev - hDev));
                            sp = std::fmax(sp, std::fabs(hDev));
                        }
                        std::printf("      %-14s %-8s %4ld faces:  %.4e of %.4e\n",
                                    fvp[pi].name.c_str(), fvp[pi].type.c_str(),
                                    (long)fvp[pi].size, (double)wp, (double)sp);
                    }
                }

                // BISECT #3: UPWIND ON BOTH SIDES. The matrix is identical either way -- linearUpwind
                // derives its weights from upwind -- so this removes ONLY the deferred source
                // correction. If the gap collapses to round-off, the whole of the remaining 1.1e-05
                // is in that correction and its gradient; if it survives, it is somewhere else.
                {
                    InterMomentumInput hm2 = hm;
                    hm2.scheme = DivScheme::upwind;
                    const FvVectorMatrix h2 = assembleUEqn(Uh2, hm2, m, g, fvp);

                    DeviceInterStepTaps t2;
                    DeviceInterStepControls C2 = C;
                    C2.divScheme = brae::cpu::DivScheme::upwind;
                    DeviceBuffer<scalar> e1(warmAlpha), e1o(warmAlpha);
                    DeviceBuffer<scalar> ex(x0), ey(y0), ez(z0), eox(x0), eoy(y0), eoz(z0);
                    DeviceBuffer<scalar> eI(dv.phi.internal), eB(flatten(dv.phi.boundary));
                    DeviceBuffer<scalar> eOI(dv.phi.internal), eOB(flatten(dv.phi.boundary));
                    DeviceBuffer<scalar> epr(dv.p_rgh.internal), epf;
                    DeviceBuffer<scalar> en(dv.nHatf.internal), enb(flatten(dv.nHatf.boundary));
                    DeviceBuffer<scalar> eab(pv2(dv.alpha1)), ek(dv.K);
                    DeviceBuffer<scalar> erho, emu, enu, eri, erb;
                    DeviceVectorBoundary db2b = buildDeviceVectorBoundary(dv.U, fvp, g);
                    deviceInterStep(dm, dt, C2, pr2, H, Gh, Ghf, MagSf, e1, e1o, ex, ey, ez,
                                    eox, eoy, eoz, eI, eB, eOI, eOB, dUFixes, epr, epf, en, enb,
                                    eab, ek, dFixes, dFlag, db2b, erho, emu, enu, eri, erb, &t2);
                    cudaDeviceSynchronize();
                    std::vector<scalar> s2;
                    t2.UEqnSourceX.copyTo(s2);
                    scalar w2 = 0, x2s = 0;
                    for (label c = 0; c < nC; ++c)
                    {
                        w2  = std::fmax(w2, std::fabs(s2[c] - h2.source[c].x));
                        x2s = std::fmax(x2s, std::fabs(h2.source[c].x));
                    }
                    std::printf("  [bisect] UPWIND on both:     source.x %.4e of %.4e\n",
                                (double)w2, (double)x2s);
                }

                // BISECT #2: RELAX OFF on both sides. damBreak says `equations { ".*" 1; }`, which
                // relaxEquation() FINDS, so relax(1) runs and adds (relaxedDiag - rawDiag)*psi to the
                // source -- on the synthetic gate that clamp moved the diagonal by 1.2e+03 of 1.0e+06.
                // If the gap vanishes without it, the relax source is the fault; if it survives, the
                // ddt assembly is.
                {
                    InterMomentumInput hm1 = hm;
                    hm1.relaxEquationU = false;
                    const FvVectorMatrix h1 = assembleUEqn(Uh2, hm1, m, g, fvp);

                    DeviceInterStepTaps t1;
                    DeviceInterStepControls C1 = C;
                    C1.relaxEquationU = false;
                    DeviceBuffer<scalar> c1(warmAlpha), c1o(warmAlpha);
                    DeviceBuffer<scalar> wx(x0), wy(y0), wz(z0), wox(x0), woy(y0), woz(z0);
                    DeviceBuffer<scalar> rI2(dv.phi.internal), rB2(flatten(dv.phi.boundary));
                    DeviceBuffer<scalar> rOI(dv.phi.internal), rOB(flatten(dv.phi.boundary));
                    DeviceBuffer<scalar> rp3(dv.p_rgh.internal), rp4;
                    DeviceBuffer<scalar> rn(dv.nHatf.internal), rnb(flatten(dv.nHatf.boundary));
                    DeviceBuffer<scalar> rab(pv2(dv.alpha1)), rk(dv.K);
                    DeviceBuffer<scalar> zrho, zmu, znu, zri, zrb;
                    DeviceVectorBoundary db1 = buildDeviceVectorBoundary(dv.U, fvp, g);
                    deviceInterStep(dm, dt, C1, pr2, H, Gh, Ghf, MagSf, c1, c1o, wx, wy, wz,
                                    wox, woy, woz, rI2, rB2, rOI, rOB, dUFixes, rp3, rp4, rn, rnb,
                                    rab, rk, dFixes, dFlag, db1, zrho, zmu, znu, zri, zrb, &t1);
                    cudaDeviceSynchronize();
                    std::vector<scalar> s1;
                    t1.UEqnSourceX.copyTo(s1);
                    scalar w1 = 0, x1s = 0;
                    for (label c = 0; c < nC; ++c)
                    {
                        w1  = std::fmax(w1, std::fabs(s1[c] - h1.source[c].x));
                        x1s = std::fmax(x1s, std::fabs(h1.source[c].x));
                    }
                    std::printf("  [bisect] with RELAX OFF:     source.x %.4e of %.4e\n",
                                (double)w1, (double)x1s);
                }

                scalar wR = 0, sR = 0, wH = 0, sH = 0, wD = 0, sD = 0;
                for (label c = 0; c < nC; ++c)
                {
                    const scalar rh = scalar(1)/Ah[c];
                    wR = std::fmax(wR, std::fabs(dRAU[c] - rh));
                    sR = std::fmax(sR, std::fabs(rh));
                    const scalar hh = rh*Hh[c].x;
                    wH = std::fmax(wH, std::fabs(dHx[c] - hh));
                    sH = std::fmax(sH, std::fabs(hh));
                    wD = std::fmax(wD, std::fabs(dDiag[c] - hUEqn.diag[c]));
                    sD = std::fmax(sD, std::fabs(hUEqn.diag[c]));
                }
                std::printf("  [taps vs host] UEqn.diag %.4e of %.4e;  rAU %.4e of %.4e;  "
                            "HbyA.x %.4e of %.4e\n",
                            (double)wD, (double)sD, (double)wR, (double)sR, (double)wH, (double)sH);

                // H() takes the OFF-DIAGONALS and the BOUNDARY coefficients too, and with the source
                // and the diagonal both exact those are what is left.
                {
                    std::vector<scalar> du, dl, dic, dbc;
                    taps.UEqnUpper.copyTo(du);
                    taps.UEqnLower.copyTo(dl);
                    taps.UEqnIC.copyTo(dic);
                    taps.UEqnBC.copyTo(dbc);
                    scalar wu2 = 0, su2 = 0, wl2 = 0, sl2 = 0;
                    for (label f = 0; f < nIf; ++f)
                    {
                        wu2 = std::fmax(wu2, std::fabs(du[f] - hUEqn.upper[f]));
                        su2 = std::fmax(su2, std::fabs(hUEqn.upper[f]));
                        wl2 = std::fmax(wl2, std::fabs(dl[f] - hUEqn.lower[f]));
                        sl2 = std::fmax(sl2, std::fabs(hUEqn.lower[f]));
                    }
                    scalar wi2 = 0, si2 = 0, wb2 = 0, sb2 = 0;
                    label off6 = 0;
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        for (label i = 0; i < fvp[pi].size; ++i)
                        {
                            wi2 = std::fmax(wi2, std::fabs(dic[off6 + i] - hUEqn.internalCoeffs[pi][i].x));
                            si2 = std::fmax(si2, std::fabs(hUEqn.internalCoeffs[pi][i].x));
                            wb2 = std::fmax(wb2, std::fabs(dbc[off6 + i] - hUEqn.boundaryCoeffs[pi][i].x));
                            sb2 = std::fmax(sb2, std::fabs(hUEqn.boundaryCoeffs[pi][i].x));
                        }
                        off6 += fvp[pi].size;
                    }
                    std::printf("  [taps vs host] upper %.4e of %.4e;  lower %.4e of %.4e;  "
                                "iC.x %.4e of %.4e;  bC.x %.4e of %.4e\n",
                                (double)wu2, (double)su2, (double)wl2, (double)sl2,
                                (double)wi2, (double)si2, (double)wb2, (double)sb2);

                    // WHICH SIDE is near zero, per patch, with the BC type beside it. bC's error
                    // equals its own magnitude, so one of the two is essentially nothing.
                    { label o7 = 0;
                      for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                      {
                          scalar mxD = 0, mxH = 0;
                          for (label i = 0; i < fvp[pi].size; ++i)
                          {
                              mxD = std::fmax(mxD, std::fabs(dbc[o7 + i]));
                              mxH = std::fmax(mxH, std::fabs(hUEqn.boundaryCoeffs[pi][i].x));
                          }
                          std::printf("      bC %-14s %-8s device %.4e  host %.4e\n",
                                      fvp[pi].name.c_str(), fvp[pi].type.c_str(),
                                      (double)mxD, (double)mxH);
                          o7 += fvp[pi].size;
                      } }
                }

                // ---- phiHbyA, built on the host the way pEqn.H builds it -------------------------
                // The momentum half is now right to 1.1e-05; this asks whether the flux the pressure
                // equation is actually SOLVED on agrees. It is four things -- fvc::flux(HbyA) with
                // constrainHbyA, interpolate(rho*rAU)*ddtCorr, and phig on BOTH sides -- and the tap
                // holds all of them.
                {
                    std::vector<vector> HbyAh(static_cast<std::size_t>(nC));
                    std::vector<scalar> rAUh(static_cast<std::size_t>(nC));
                    for (label c = 0; c < nC; ++c)
                    {
                        rAUh[c] = scalar(1)/Ah[c];
                        HbyAh[c] = vector{rAUh[c]*Hh[c].x, rAUh[c]*Hh[c].y, rAUh[c]*Hh[c].z};
                    }
                    // constrainHbyA: on a patch whose U BC is NOT assignable, HbyA's boundary value is
                    // U's. assignable() is not fixesValue().
                    std::vector<std::vector<vector>> HbyAb(fvp.size());
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        const bool takeUb = !dv.U.boundary[pi]->assignable();
                        const std::vector<vector>& ub = dv.U.boundary[pi]->value();
                        for (label i = 0; i < fvp[pi].size; ++i)
                            HbyAb[pi].push_back(takeUb ? ub[i] : HbyAh[fvp[pi].faceCells[i]]);
                    }
                    SurfaceScalarField phiHh = fvc::flux(HbyAh, HbyAb, m, g, fvp);

                    // interpolate(rho*rAU)*ddtCorr, internal faces
                    std::vector<scalar> rrAUf;
                    rhoRAUf(devRho, rAUh, m, g, rrAUf);
                    SurfaceScalarField phiOldF;
                    phiOldF.internal = dv.phi.internal;
                    phiOldF.boundary = dv.phi.boundary;
                    DdtCorrInput dci;
                    dci.phiOld = &phiOldF;
                    dci.UOld = &UOldH;
                    dci.ddtPhiCoeff = -1;
                    dci.deltaT = dt;
                    SurfaceScalarField dcorr;
                    ddtCorr(dci, Uh2, m, g, fvp, dcorr);
                    for (label f = 0; f < nIf; ++f)
                        phiHh.internal[f] += rrAUf[f]*dcorr.internal[f];

                    // phig on BOTH sides, from the same stf/snGradRho the hook built
                    std::vector<scalar> stfH, snRhoH, rAUfAllH(nFacesAll);
                    { std::vector<scalar> sK;
                      ip::sigmaK(dv.K, dv.interface.sigma, sK);
                      const SurfaceScalarField sKf = fvc::interpolate(sK, m, g, fvp);
                      const SurfaceScalarField snA = fvc::snGrad(dv.alpha1, m, g, fvp, false);
                      stfH.resize(static_cast<std::size_t>(nFacesAll));
                      for (label f = 0; f < nIf; ++f) stfH[f] = sKf.internal[f]*snA.internal[f];
                      { label o4 = nIf;
                        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                          for (label i = 0; i < fvp[pi].size; ++i)
                            stfH[o4++] = sKf.boundary[pi][i]*snA.boundary[pi][i]; }
                      GeometricField<scalar> rhoFh;
                      rhoFh.internal = devRho;
                      for (const FvPatch& q : fvp)
                        rhoFh.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
                      rhoFh.evaluateBoundary();
                      const SurfaceScalarField sr = fvc::snGrad(rhoFh, m, g, fvp, false);
                      snRhoH.assign(sr.internal.begin(), sr.internal.end());
                      for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                        snRhoH.insert(snRhoH.end(), sr.boundary[pi].begin(), sr.boundary[pi].end()); }
                    for (label f = 0; f < nIf; ++f)
                    {
                        const scalar w = g.weights()[f];
                        rAUfAllH[f] = w*rAUh[m.owner()[f]] + (scalar(1) - w)*rAUh[m.neighbour()[f]];
                    }
                    { label o5 = nIf;
                      for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                        for (label i = 0; i < fvp[pi].size; ++i)
                          rAUfAllH[o5++] = rAUh[fvp[pi].faceCells[i]]; }
                    std::vector<scalar> phigH;
                    buoyancyFlux(stfH, ghfAll, snRhoH, rAUfAllH, g.magSf(), phigH);
                    for (label f = 0; f < nIf; ++f) phiHh.internal[f] += phigH[f];

                    std::vector<scalar> dPhiH;
                    taps.phiHbyAInt.copyTo(dPhiH);
                    scalar wP = 0, sP = 0;
                    for (label f = 0; f < nIf; ++f)
                    {
                        wP = std::fmax(wP, std::fabs(dPhiH[f] - phiHh.internal[f]));
                        sP = std::fmax(sP, std::fabs(phiHh.internal[f]));
                    }
                    // ...AND ITS BOUNDARY, which fvc::div(phiHbyA) sums. Comparing the internal
                    // faces alone leaves the pressure equation's source half unmeasured.
                    std::vector<scalar> dPhiHB;
                    taps.phiHbyABnd.copyTo(dPhiHB);
                    scalar wPB = 0, sPB = 0;
                    label o8 = 0;
                    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    {
                        for (label i = 0; i < fvp[pi].size; ++i)
                        {
                            wPB = std::fmax(wPB, std::fabs(dPhiHB[o8 + i] - phiHh.boundary[pi][i]));
                            sPB = std::fmax(sPB, std::fabs(phiHh.boundary[pi][i]));
                        }
                        o8 += fvp[pi].size;
                    }
                    std::printf("  [taps vs host] phiHbyA %.4e of %.4e;  BOUNDARY %.4e of %.4e\n",
                                (double)wP, (double)sP, (double)wPB, (double)sPB);
                    // ---- THE p_rgh MATRIX, coefficient by coefficient ------------------------
                    // The source is exact on both sides now, so a p_rgh that is 7.5e+01 out has to be
                    // the matrix. Built on the host from the DEVICE's own rAUf and p_rgh boundary, so
                    // what is compared is the assembly.
                    {
                        std::vector<scalar> rAUfDev;
                        taps.rAUfAllTap.copyTo(rAUfDev);
                        SurfaceScalarField rfD;
                        rfD.internal.assign(rAUfDev.begin(), rAUfDev.begin() + nIf);
                        rfD.boundary.resize(fvp.size());
                        { label oa = nIf;
                          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                            for (label i = 0; i < fvp[pi].size; ++i) rfD.boundary[pi].push_back(rAUfDev[oa++]); }
                        FvScalarMatrix peH = fvm::laplacian<scalar>(rfD, dv.p_rgh, m, g, fvp, false);
                        const std::vector<scalar> dvH = fvc::div(phiHh, m, g, fvp);
                        for (label c = 0; c < nC; ++c) peH.source[c] += dvH[c]*g.V()[c];

                        std::vector<scalar> pd, pu, pl, ps, pic, pbc;
                        taps.pDiag.copyTo(pd);   taps.pUpper.copyTo(pu);  taps.pLower.copyTo(pl);
                        taps.pSource.copyTo(ps); taps.pIC.copyTo(pic);    taps.pBC.copyTo(pbc);
                        scalar wpd=0,spd=0,wpu=0,spu=0,wps=0,sps=0,wpi=0,spi2=0,wpb=0,spb=0;
                        for (label c = 0; c < nC; ++c)
                        { wpd=std::fmax(wpd,std::fabs(pd[c]-peH.diag[c])); spd=std::fmax(spd,std::fabs(peH.diag[c]));
                          wps=std::fmax(wps,std::fabs(ps[c]-peH.source[c])); sps=std::fmax(sps,std::fabs(peH.source[c])); }
                        for (label f = 0; f < nIf; ++f)
                        { wpu=std::fmax(wpu,std::fabs(pu[f]-peH.upper[f])); spu=std::fmax(spu,std::fabs(peH.upper[f])); }
                        { label ob = 0;
                          for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                          { for (label i = 0; i < fvp[pi].size; ++i)
                            { wpi=std::fmax(wpi,std::fabs(pic[ob+i]-peH.internalCoeffs[pi][i])); spi2=std::fmax(spi2,std::fabs(peH.internalCoeffs[pi][i]));
                              wpb=std::fmax(wpb,std::fabs(pbc[ob+i]-peH.boundaryCoeffs[pi][i])); spb=std::fmax(spb,std::fabs(peH.boundaryCoeffs[pi][i])); }
                            ob += fvp[pi].size; } }
                        std::printf("  [pEqn vs host] diag %.4e of %.4e;  upper %.4e of %.4e;  "
                                    "source %.4e of %.4e;  iC %.4e of %.4e;  bC %.4e of %.4e\n",
                                    (double)wpd,(double)spd,(double)wpu,(double)spu,
                                    (double)wps,(double)sps,(double)wpi,(double)spi2,
                                    (double)wpb,(double)spb);
                        (void)pl;

                        // THE DECISIVE ONE. Matrix exact, source exact, solved field 6.4e+01 apart --
                        // so either the two SOLVES differ, or `ref`/`r1` is not solving this system at
                        // all and my reconstruction of the host's state is what is off. Solving THIS
                        // matrix on the host, from the same starting p_rgh, separates the two: if it
                        // lands on the device's answer, the device is right and the reference is the
                        // problem.
                        std::vector<scalar> pHost = warmPrgh;
                        pbicgstab(peH, pHost, m, fvp, scalar(1e-12), scalar(0), 4000);
                        std::vector<scalar> pDev;
                        taps.pSolved.copyTo(pDev);
                        scalar wSame = 0, sSame = 0;
                        for (label c = 0; c < nC; ++c)
                        {
                            wSame = std::fmax(wSame, std::fabs(pDev[c] - pHost[c]));
                            sSame = std::fmax(sSame, std::fabs(pHost[c]));
                        }
                        std::printf("  [pEqn vs host] SAME system solved on the host: %.4e of %.4e\n",
                                    (double)wSame, (double)sSame);
                    }

                    { label o9 = 0;
                      for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                      {
                          scalar wq = 0, sq = 0;
                          for (label i = 0; i < fvp[pi].size; ++i)
                          {
                              wq = std::fmax(wq, std::fabs(dPhiHB[o9 + i] - phiHh.boundary[pi][i]));
                              sq = std::fmax(sq, std::fabs(phiHh.boundary[pi][i]));
                          }
                          std::printf("      phiHbyA_b %-14s %-8s %.4e of %.4e\n",
                                      fvp[pi].name.c_str(), fvp[pi].type.c_str(),
                                      (double)wq, (double)sq);
                          o9 += fvp[pi].size;
                      } }
                }

                // ---- ONE STEP, end to end, against the host at the SAME step count ---------------
                // phiHbyA is right to 2.5e-04 and U after five steps is 58% out. Either the error
                // amplifies -- in which case one step is small -- or the comparison is not
                // like-for-like, in which case one step is already large. The two look nothing alike.
                {
                    InterFields r1 = buildInterFields(caseDir, startDir, m, g, fvp);
                    runInterFoam(caseDir, startDir, m, g, fvp, nWarm + 1, /*verbose=*/false, &r1);
                    std::vector<scalar> u1, p1, a1v;
                    ux.copyTo(u1);
                    pr.copyTo(p1);
                    a1.copyTo(a1v);
                    scalar wu1 = 0, su1 = 0, wp1 = 0, sp1 = 0, wa1 = 0;
                    for (label c = 0; c < nC; ++c)
                    {
                        wu1 = std::fmax(wu1, std::fabs(u1[c] - r1.U.internal[c].x));
                        su1 = std::fmax(su1, std::fabs(r1.U.internal[c].x));
                        wp1 = std::fmax(wp1, std::fabs(p1[c] - r1.p_rgh.internal[c]));
                        sp1 = std::fmax(sp1, std::fabs(r1.p_rgh.internal[c]));
                        wa1 = std::fmax(wa1, std::fabs(a1v[c] - r1.alpha1.internal[c]));
                    }
                    std::printf("  [one step] alpha %.4e;  U.x %.4e of %.4e;  p_rgh %.4e of %.4e\n",
                                (double)wa1, (double)wu1, (double)su1, (double)wp1, (double)sp1);

                    // IS THE p_rgh ERROR A CONSTANT? A pressure equation with no value-fixing patch is
                    // singular and setReference pins one cell; if the two sides pin it differently the
                    // whole field is offset by a constant and every derived quantity that reads a
                    // GRADIENT is untouched. Subtracting the mean separates that from a real shape
                    // difference in one number.
                    scalar mean = 0;
                    for (label c = 0; c < nC; ++c) mean += (p1[c] - r1.p_rgh.internal[c]);
                    mean /= static_cast<scalar>(nC);
                    scalar spread = 0;
                    for (label c = 0; c < nC; ++c)
                        spread = std::fmax(spread, std::fabs((p1[c] - r1.p_rgh.internal[c]) - mean));
                    std::printf("  [one step] p_rgh error: mean %.4e, spread about it %.4e "
                                "(a pure offset would leave the spread at round-off)\n",
                                (double)mean, (double)spread);

                    // WHERE is it? Interior against each patch's own cells, with the BC type beside
                    // it -- the split that named the dev2 defect in one run.
                    {
                        std::vector<char> tp(static_cast<std::size_t>(nC), 0);
                        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                        {
                            if (fvp[pi].type == "empty") continue;
                            for (label i = 0; i < fvp[pi].size; ++i) tp[fvp[pi].faceCells[i]] = 1;
                        }
                        scalar wIn2 = 0;
                        int nIn2 = 0;
                        for (label c = 0; c < nC; ++c)
                            if (!tp[c]) { wIn2 = std::fmax(wIn2, std::fabs(p1[c] - r1.p_rgh.internal[c])); ++nIn2; }
                        std::printf("      p_rgh interior (%d cells) %.4e\n", nIn2, (double)wIn2);
                        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                        {
                            if (fvp[pi].type == "empty") continue;
                            scalar wp3 = 0;
                            for (label i = 0; i < fvp[pi].size; ++i)
                            {
                                const label c = fvp[pi].faceCells[i];
                                wp3 = std::fmax(wp3, std::fabs(p1[c] - r1.p_rgh.internal[c]));
                            }
                            std::printf("      p_rgh %-14s %-8s %.4e   (BC %s)\n",
                                        fvp[pi].name.c_str(), fvp[pi].type.c_str(), (double)wp3,
                                        dv.p_rgh.boundary[pi]->updateableSnGrad()
                                            ? "prescribes its gradient" : "does not");
                        }
                    }
                }
                check("the device's momentum diagonal matches the host's on damBreak",
                      wD < scalar(1e-10)*sD);
                check("...and so does rAU", wR < scalar(1e-10)*sR);
                check("...and so does HbyA", wH < scalar(1e-8)*sH);
            }
        }

        for (int s2 = 0; s2 < nSteps; ++s2)
        {
            std::vector<scalar> ca, cx, cy, cz;
            A2.copyTo(ca);  AO.copyFrom(ca);
            Ux.copyTo(cx);  Uy.copyTo(cy);  Uz.copyTo(cz);
            Uox.copyFrom(cx); Uoy.copyFrom(cy); Uoz.copyFrom(cz);
            // phi.oldTime() is the flux this step STARTS from -- ddtCorr's whole content is the
            // disagreement between it and the flux interpolate(U.oldTime()) would give.
            std::vector<scalar> poi, pob;
            PhI.copyTo(poi);  PhB.copyTo(pob);
            DeviceBuffer<scalar> PhOI(poi), PhOB(pob);
            deviceInterStep(dm, dt, C, pr2, H, Gh, Ghf, MagSf, A2, AO, Ux, Uy, Uz, Uox, Uoy, Uoz,
                            PhI, PhB, PhOI, PhOB, dUFixes, Prgh, Pf, NH, NHB, ABnd, Kd2,
                            dFixes, dFlag, db2, Rho2, Mu2, Nu2, RpI2, RpB2);
        }
        if (cudaDeviceSynchronize() != cudaSuccess)
        { std::printf("  FAIL: whole-step kernels did not complete\n"); return 1; }

        std::vector<scalar> fa2, fux, fprgh;
        A2.copyTo(fa2);
        Ux.copyTo(fux);
        Prgh.copyTo(fprgh);

        // FINITE FIRST, and this is not defensive noise. std::fmax(a, NaN) returns a -- it IGNORES the
        // NaN -- so every |device - host| accumulator below reads 0.000e+00 for a field that has gone
        // non-finite, which is indistinguishable from perfect agreement. That is exactly what this
        // gate reported before the check existed: alpha, U and p_rgh all "0.0000e+00" and four green
        // arms, on a run whose every cell was NaN. Any worst-difference loop built on fmax needs this
        // ahead of it.
        int nBadA = 0, nBadU = 0, nBadP = 0;
        for (label c = 0; c < nC; ++c)
        {
            if (!std::isfinite(fa2[c]))   ++nBadA;
            if (!std::isfinite(fux[c]))   ++nBadU;
            if (!std::isfinite(fprgh[c])) ++nBadP;
        }
        std::printf("  non-finite cells after %d whole steps: alpha %d, U.x %d, p_rgh %d of %ld\n",
                    nSteps, nBadA, nBadU, nBadP, (long)nC);
        check("the whole device step leaves every field finite", nBadA + nBadU + nBadP == 0);

        scalar devMoved = 0;
        for (label c = 0; c < nC; ++c)
            devMoved = std::fmax(devMoved, std::fabs(fa2[c] - warmAlpha[c]));
        check("...and it advanced the field", devMoved > scalar(1e-9));

        scalar wa = 0, wu = 0, wp = 0, sa = 0, su = 0, sp = 0, exc = 0;
        for (label c = 0; c < nC; ++c)
        {
            wa = std::fmax(wa, std::fabs(fa2[c]   - ref.alpha1.internal[c]));
            wu = std::fmax(wu, std::fabs(fux[c]   - ref.U.internal[c].x));
            wp = std::fmax(wp, std::fabs(fprgh[c] - ref.p_rgh.internal[c]));
            sa = std::fmax(sa, std::fabs(ref.alpha1.internal[c]));
            su = std::fmax(su, std::fabs(ref.U.internal[c].x));
            sp = std::fmax(sp, std::fabs(ref.p_rgh.internal[c]));
            exc = std::fmax(exc, std::fmax(-fa2[c], fa2[c] - scalar(1)));
        }
        std::printf("  WHOLE STEP, %d steps: alpha %.4e of %.3f, U.x %.4e of %.4e, "
                    "p_rgh %.4e of %.4e;  the device moved alpha %.4e, excursion %.3e\n",
                    nSteps, (double)wa, (double)sa, (double)wu, (double)su,
                    (double)wp, (double)sp, (double)devMoved, (double)exc);
        // MEASURED over five steps: alpha 7.3e-11, U 6.8e-10 of 2.7e-01 (2.6e-09 relative), p_rgh
        // 2.0e-07 of 2.8e+03 (6.9e-11). The bounds are two orders above those and far below anything
        // a real defect has produced here -- the smallest of the six found through this arm was 1%.
        check("the whole device step tracks the host on damBreak", wa < scalar(1e-8));
        check("...in the velocity it leaves", wu < scalar(1e-7)*std::fmax(su, scalar(1e-30)));
        check("...and in the pressure", wp < scalar(1e-8)*std::fmax(sp, scalar(1e-30)));
        check("...with alpha still bounded", exc <= scalar(1e-6));
    }

    std::printf("test_device_inter_dambreak_alpha: %d failures\n", failures);
    return failures ? 1 : 0;
}
