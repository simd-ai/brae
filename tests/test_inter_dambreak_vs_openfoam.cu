// brae's interFoam against REAL OpenFOAM's, on damBreak, field by field.
//
// THE ORACLE IS OpenFOAM'S OWN WRITTEN STATE after exactly N identical time steps. The shell gate
// rewrites the case with `adjustTimeStep no` and a fixed deltaT before running either solver, because
// an adaptive step makes the two take DIFFERENT steps the moment their Courant numbers differ at all,
// and then every field is compared at a different physical time -- a disagreement that looks like a
// discretisation error and is actually a clock.
//
// WHAT AGREEMENT CAN AND CANNOT MEAN HERE. brae and OpenFOAM do not share a linear solver: OpenFOAM's
// p_rgh runs PCG/DIC at tolerance 1e-07 relTol 0.05, and brae's host path runs DILU-preconditioned
// BiCGStab. So the pressure fields differ by the two solvers' residuals whatever the discretisation
// does, and a tolerance chosen tighter than that would be gating the solver rather than the port. The
// bounds below are therefore stated per field, with the reason each one is what it is.
//
// alpha IS THE FIELD THAT MATTERS. It is bounded, it is conserved, and it is what a VoF solver is for;
// it is also the field least affected by the linear-solver difference, because MULES is explicit.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include "inter_solve_log.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

struct Diff { scalar linf = 0; scalar l2 = 0; scalar refMax = 0; };

Diff compare(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    Diff d;
    scalar s2 = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const scalar e = std::fabs(a[i] - b[i]);
        d.linf = std::fmax(d.linf, e);
        s2 += e*e;
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    d.l2 = std::sqrt(s2 / std::fmax(scalar(1), static_cast<scalar>(a.size())));
    return d;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: damBreak ==\n");
    if (argc < 5)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1], startDir = argv[2], ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    // `bigstep`: the staging script's second run, at dt 5e-3, where the interface moves 0.69 rather
    // than 3.7e-03 and the alpha pre-solve is hard enough for its solver log to discriminate. See the
    // script's header for why the two things this gate measures want opposite fixtures.
    // `prevcorr`: the big step again, with `alphaApplyPrevCorr yes`. It takes every big-step bound and
    // arm, and adds a control on the oracle (below).
    // `prevcorrsub` is `prevcorr` with nAlphaSubCycles 2: the cache has to cross a sub-cycle boundary.
    // Its control reads the un-sub-cycled run, so what it proves live is the sub-cycle count.
    const bool prevCorrSub = argc > 7 && std::string(argv[7]) == "prevcorrsub";
    const bool prevCorr = (argc > 7 && std::string(argv[7]) == "prevcorr") || prevCorrSub;
    // THREE PIMPLE CONTROLS damBreak does not use, each at the big step: nOuterCorrectors 2,
    // nNonOrthogonalCorrectors 1, momentumPredictor yes. The device REFUSES the first two.
    const std::string profileName = argc > 7 ? argv[7] : "";
    const bool nOuter = profileName == "nouter", nonOrth = profileName == "nonorth";
    const bool momPred = profileName == "mompred";
    // `rhophi`: the atmosphere's three flux-conditional conditions all name `phi rhoPhi;`. The device
    // runs U's switch itself and reads phi there, so it refuses the case.
    const bool namedFlux = profileName == "rhophi";
    // `vanleerv` and `linear`: div(rhoPhi,U) as two schemes the tutorial does not name -- the V-limited
    // vanLeer that the closed-tank tutorials use, and central differencing, which the device mapped to
    // upwind through a switch's `default` until this profile existed. Both run on both paths.
    const bool vanLeerV = profileName == "vanleerv";
    const bool linear = profileName == "linear";
    // `compression`: div(phirb,alpha) as `Gauss interfaceCompression`, the PhiScheme four waveMaker
    // tutorials name, on a mesh that does not move -- the still fixture for a scheme whose four
    // tutorials all move their mesh. It was a REFUSAL arm here while the device had no such scheme;
    // the device runs it now (device_fvm.cu, interfaceCompressionWeightsKernel) and it is compared
    // like any other profile.
    const bool compression = profileName == "compression";
    // `alphaminiter`: the alpha entry names `minIter 1`. At the big step OpenFOAM's first pre-solve starts
    // under its tolerance and takes no sweep; minIter forces one. The device's pre-solve does not honour
    // minIter and refuses the case.
    const bool alphaMinIter = profileName == "alphaminiter";
    // `sheared`: the big step on damBreak with its upper blocks sheared six degrees, so the corrected
    // laplacian and snGrad carry a non-zero correction, and with the momentum predictor on, so the
    // predictor's snGrad(p_rgh) is read. It is `gradLsqLimited`'s control run. The device assembles
    // orthogonal and refuses it.
    const bool sheared = profileName == "sheared";
    // `gradLsqLimited`: every gradient `cellLimited leastSquares 1` but nHat, which names `leastSquares`,
    // on the sheared mesh with the predictor on. The device's operators are Gauss linear and refuse it.
    const bool gradLsqLimited = profileName == "gradLsqLimited";
    // `nHatLimited`: the interface normal alone `cellLimited Gauss linear 1`, at the SMALL step -- the
    // big step's MULES leaves alpha 1 +- 1e-7 in the bulk, where the limiter turns round-off into
    // 1e-06 of p_rgh (see the script). The device refuses it.
    const bool nHatLimited = profileName == "nHatLimited";
    const bool pimpleProfile = nOuter || nonOrth || momPred || namedFlux || vanLeerV || linear || compression
                            || alphaMinIter || gradLsqLimited;
    // `nonorth` RUNS on the device: its pressure step carries the non-orthogonal loop (transcribed from
    // the host's pressureCorrector), and the device arm below holds it to OpenFOAM
    // `sheared` RUNS on the device too: the non-orthogonal correction is on that path now, and the
    // device arm below holds it to OpenFOAM on a mesh that is not orthogonal
    // `nouter` RUNS on the device now: nOuterCorrectors is the device driver's own PIMPLE loop
    // (inter_driver_device.cu), measured against OpenFOAM on validation/interFoamCyclic's `outer`
    // profile with the one-corrector answer as its control. It is compared here like any other
    // profile, on a case whose every other control the device already runs.
    const bool deviceRefuses = namedFlux || alphaMinIter || gradLsqLimited
                            || nHatLimited;
    const bool bigStep = (argc > 7 && std::string(argv[7]) == "bigstep") || prevCorr || pimpleProfile || sheared;
    // `inflow`: the atmosphere's inletValue set to 1, so water enters over air cells and rho's patch
    // value differs from the cell's on a patch where p_rgh fixes a value. It is the only fixture here
    // on which fvc::snGrad(rho) is non-zero on a boundary that does not cancel it. See the script.
    const bool inflow = argc > 7 && std::string(argv[7]) == "inflow";
    // `outflow`: the water column raised to the atmosphere with alphaApplyPrevCorr on, at dt 1e-2 --
    // the one fixture on which the previous-correction limiter's flux argument changes the answer.
    const bool outflow = argc > 7 && std::string(argv[7]) == "outflow";
    std::printf("  profile: %s\n",
                prevCorrSub ? "prevcorrsub -- alphaApplyPrevCorr yes across TWO sub-cycles"
              : prevCorr ? "prevcorr -- alphaApplyPrevCorr yes, at the big step"
              : nOuter ? "nouter -- nOuterCorrectors 2, at the big step"
              : nonOrth ? "nonorth -- nNonOrthogonalCorrectors 1, at the big step"
              : momPred ? "mompred -- momentumPredictor yes, at the big step"
              : namedFlux ? "rhophi -- the atmosphere's conditions all name phi rhoPhi, at the big step"
              : compression ? "compression -- div(phirb,alpha) Gauss interfaceCompression, at the big step"
              : alphaMinIter ? "alphaminiter -- the alpha pre-solve names minIter 1, at the big step"
              : gradLsqLimited ? "gradLsqLimited -- every gradient but nHat cellLimited leastSquares 1, sheared, at the big step"
              : sheared ? "sheared -- the upper blocks sheared six degrees, at the big step"
              : nHatLimited ? "nHatLimited -- nHat cellLimited Gauss linear 1, at the small step"
              : bigStep ? "bigstep -- the solver logs discriminate here"
              : inflow  ? "inflow -- snGrad(rho) is live on the atmosphere here"
              : outflow ? "outflow -- water leaves through the atmosphere with alphaApplyPrevCorr on"
                        : "small step -- the tight field bounds live here");

    if (!std::filesystem::exists(ofDir + "/alpha.water"))
    {
        std::printf("  SKIP: OpenFOAM wrote no alpha.water in %s\n", ofDir.c_str());
        return 77;
    }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    // brae, the same N steps, keeping the fields.
    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);

    // THE SOLVER'S OWN LOG AGAINST OpenFOAM'S. The field bounds below tightened by four orders when the
    // host took the case's PCG with DIC and its per-corrector p_rgh/p_rghFinal selection; this arm is
    // the direct statement of why -- every solve takes OpenFOAM's iteration count.
    const std::vector<PressureSolveRecord> ofSolves =
        argc > 5 ? brae::gatecheck::readOfPressureSolves(argv[5]) : std::vector<PressureSolveRecord>{};
    if (argc > 5)
    {
        // the FAIL-PROOF: nothing in compareSolves can pass on an empty parse
        check("OpenFOAM's log gave a p_rgh solve for every corrector of every step",
              !ofSolves.empty() && ofSolves.size() % static_cast<std::size_t>(nSteps) == 0);
        failures += brae::gatecheck::compareSolves("host", r.pSolves, ofSolves, nSteps);
    }

    // ...AND THE alpha PRE-SOLVE'S, which is the arm that would have caught the device running
    // Jacobi-BiCGStab for the case's symGaussSeidel: OpenFOAM's smoother takes TWO sweeps to 1e-13 on
    // this near-triangular upwind matrix, and no Krylov solver stopping at 1e-8 reports that.
    const std::vector<LinearSolveRecord> ofAlphaSolves =
        argc > 5 ? brae::gatecheck::readOfSolves(argv[5], fin.alphaName) : std::vector<LinearSolveRecord>{};
    if (argc > 5)
    {
        // one pre-solve per SUB-CYCLE per OUTER CORRECTOR per step: alphaEqnSubCycle.H sits inside the
        // PIMPLE loop, so a second outer corrector solves alpha again
        check("OpenFOAM's log gave an alpha solve for every sub-cycle of every outer corrector of every step",
              ofAlphaSolves.size() == static_cast<std::size_t>(nSteps)
                                    * static_cast<std::size_t>(fin.alphaCtl.nAlphaSubCycles)
                                    * static_cast<std::size_t>(fin.pimple.nOuterCorrectors));
        // THE FINAL RESIDUAL IS THE ARM THAT TELLS SOLVERS APART, and only on a fixture where the solve
        // is hard. At dt 1e-4 it is not: the smoother takes ONE sweep, so would a Krylov solver, and
        // the control below read 2.6e-03 from OpenFOAM's residuals beside the device's honest 1.5e-03
        // -- an arm that cannot separate those is decoration, so there it is printed and not asserted.
        // Under `bigstep` OpenFOAM takes 0, 5, 2, 2, 2 sweeps, the host leaves its final residuals to
        // 9.1e-10 (lduMatrix::residual's operation order is transcribed too) and the bound is 1e-7.
        failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofAlphaSolves, nSteps,
                                                   fin.alphaName.c_str(), scalar(1e-10), scalar(1e-9),
                                                   bigStep ? scalar(1e-7) : scalar(-1));
        // ...AND ITS CONTROL: the same run with the alpha entry renamed to PBiCGStab, which is what the
        // host ran until it had a smoothSolver. Measured under `bigstep`: two iteration counts of five
        // and final residuals 100% out.
        if (argc > 6 && bigStep)
        {
            const RunReport rc = runInterFoam(argv[6], std::string(argv[6]) + "/0", m, g, patches,
                                              nSteps, /*verbose=*/false);
            scalar wFinal = 0;
            std::size_t nSame = 0;
            for (std::size_t k = 0; k < rc.alphaSolves.size() && k < ofAlphaSolves.size(); ++k)
            {
                if (rc.alphaSolves[k].nIterations == ofAlphaSolves[k].nIterations)
                {
                    ++nSame;
                }
            }
            std::printf("  CONTROL, the alpha entry renamed to PBiCGStab:\n");
            brae::gatecheck::compareSolves("control", rc.alphaSolves, ofAlphaSolves, nSteps,
                                           fin.alphaName.c_str(), scalar(1e300), scalar(1e300),
                                           scalar(-1), &wFinal, /*assertArms=*/false);
            std::printf("  CONTROL: %zu of %zu iteration counts equal, final residuals %.3e (relative) out\n",
                        nSame, ofAlphaSolves.size(), (double)wFinal);
            check("...and PBiCGStab does NOT take OpenFOAM's sweep counts", nSame < ofAlphaSolves.size());
            check("...nor leave its final residuals: more than 20% out, against a bound of 1e-7",
                  wFinal > scalar(0.2));
        }
    }
    else
    {
        std::printf("  (no OpenFOAM log given: the solver-log arms are skipped)\n");
    }

    auto readCells = [&](const std::string& path)
    {
        const FieldData<scalar> fd = readField<scalar>(path);
        std::vector<scalar> v;
        if (fd.internalUniform) v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        else                    v = fd.internalField;
        return v;
    };

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> a0      = readCells(startDir + "/alpha.water");
    check("OpenFOAM's alpha has one value per cell", ofAlpha.size() == static_cast<std::size_t>(nC));

    // HOW FAR THE FIELD MOVED is the yardstick. An absolute tolerance on alpha would be met by a
    // solver that did nothing at all -- 2268 cells of which only a few dozen are near the interface,
    // and five steps of 1e-4 s move it by a few thousandths. The comparison is therefore stated as a
    // fraction of OpenFOAM's OWN change over the same interval.
    scalar ofMoved = 0;
    for (label c = 0; c < nC; ++c) ofMoved = std::fmax(ofMoved, std::fabs(ofAlpha[c] - a0[c]));
    std::printf("  OpenFOAM's alpha moved by up to %.6f over %ld steps\n", (double)ofMoved, (long)nSteps);
    check("OpenFOAM's own run actually moved the interface, so there is something to compare",
          ofMoved > scalar(1e-6));

    // BEFORE any fmax below: compare() accumulates Linf with std::fmax, which DROPS a NaN, so a brae
    // field gone non-finite would read Linf 0 and pass every bound in this file. See
    // tests/device_gate_finite.cuh for the run that printed four green checks on 2268 NaN cells.
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    const Diff dAlpha = compare(fin.alpha1.internal, ofAlpha);
    std::printf("  alpha:  Linf %.4e  L2 %.4e   (OpenFOAM moved %.4e, so Linf is %.2f%% of the change)\n",
                (double)dAlpha.linf, (double)dAlpha.l2, (double)ofMoved,
                (double)(100*dAlpha.linf/ofMoved));

    // THE CONTROL: doing nothing would differ from OpenFOAM by exactly ofMoved. Any claim about alpha
    // is worthless unless brae is much closer than that.
    const Diff dNothing = compare(a0, ofAlpha);
    std::printf("  ...against %.4e for a solver that did NOTHING\n", (double)dNothing.linf);
    check("brae's alpha is far closer to OpenFOAM's than the initial field is",
          dAlpha.linf < scalar(0.05) * dNothing.linf);
    // MEASURED 2.2337e-12. It was 3.43e-09, then 1.24e-08 once the case's own p_rgh tolerance was
    // read, and every one of those was the PRESSURE SOLVER: brae ran PBiCGStab where damBreak names
    // PCG with DIC, and applied p_rghFinal's relTol to all three correctors where pEqn.H selects
    // p_rgh (relTol 0.05) for the first two. With brae::pcg and the per-corrector selection it is
    // 2.2e-12 -- four orders. Then 1.3e-12 with capillaryRise's three boundary fixes, and 3.6e-14 once
    // the atmosphere's totalPressure was given its dynamic term (p0 - 0.5*rho*neg(phi)*|U|^2, which
    // interFoam never applied; see updatePressurePatchesFromVelocity). The bound follows at about 30x.
    // A gate that would pass at 5% is not measuring the discretisation, it is measuring that something
    // happened; this one would not pass at the 5e-11 it carried one fix ago.
    // ...and under `bigstep`, where the interface moves 0.69 instead of 3.7e-03 and carries the error
    // with it, MEASURED 1.2e-12 and bounded at 5e-11.
    // under `outflow` MEASURED 4.6e-13 on the host and 4.7e-13 on the device, bounded at 5e-11 like
    // the other profiles on which the interface travels
    const scalar alphaBound = (bigStep || outflow) ? scalar(5e-11) : scalar(1e-12);
    // ...and the DEVICE's on `nouter`, which is the stopping point compounded -- see pBoundDev
    const scalar alphaBoundDev = nOuter ? scalar(3e-5) : alphaBound;
    std::printf("  (alpha bound for this profile: %.0e)\n", (double)alphaBound);
    check("...and agrees with it absolutely, which is the discretisation and not the control",
          dAlpha.linf < alphaBound);

    // p_rgh and U. BOTH codes now run the case's own PCG with DIC -- brae::pcg is a transcription of
    // lduMatrix PCG + DICPreconditioner, gated in tests/test_pcg.cu -- and both select p_rgh for the
    // first two correctors and p_rghFinal for the last. So the two stop at the same residual, and what
    // is left is floating point amplified through a solve that stops at relTol 0.05, not two solvers'
    // different stopping points. The bounds are relative to each field's own scale.
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    std::printf("  p_rgh:  Linf %.4e  L2 %.4e   (|p_rgh| up to %.4e)\n",
                (double)dP.linf, (double)dP.l2, (double)dP.refMax);
    std::printf("          relative %.3e\n", (double)(dP.linf/std::fmax(dP.refMax, scalar(1e-30))));
    // MEASURED 8.585e-12 relative; 7.2e-10 before totalPressure's dynamic term and 3.35e-06 on
    // PBiCGStab. The old comment here said agreement much below 2.3e-06 "would be luck rather than a
    // claim" -- that was the substituted solver talking.
    // Under `inflow` MEASURED 4.3e-11 on the host and 6.9e-10 on the device, and neither moves when
    // every solve is tightened on both codes -- so it is not the tolerance ball. It is round-off
    // through a transient in which |U| goes 0.19 -> 20 m/s in one step across a 1000:1 boundary
    // density: device against host under tight solves is 1e-12 after two steps and 7e-10 after three.
    // Before rho's patch values reached snGrad(rho) all three fields were 100% out, so 1e-8 is eight
    // orders inside the defect it guards.
    const scalar pBoundHost = inflow ? scalar(1e-9) : scalar(2e-10);
    // `nouter` IS LOOSER ON THE DEVICE, and the reason is the stopping point and not the loop. Two
    // PIMPLE outer correctors run damBreak's every solve twice per step, each starting where the last
    // one stopped, and this tutorial's tolerances are loose -- alpha 1e-8, p_rgh 1e-07 -- so the slack
    // compounds. MEASURED against the HOST at the case's own tolerances: alpha 8.7e-07, p_rgh 2.6e-03,
    // U 8.5e-05 after five steps; with every solve pinned at 1e-16 the SAME comparison reads alpha
    // 3.1e-15, p_rgh 5.9e-12, U 1.0e-13, which is what says the outer loop is right. The SOLVE RECORDS
    // are not compared on this profile for the same reason: at 1e-07 over two correctors the device
    // stops an iteration either side of OpenFOAM.
    const scalar pBoundDev = nOuter ? scalar(2e-2) : (inflow ? scalar(1e-8) : scalar(2e-10));
    const scalar uBoundHost = inflow ? scalar(2e-11) : scalar(5e-10);
    const scalar uBoundDev = nOuter ? scalar(1e-2) : (inflow ? scalar(1e-8) : scalar(5e-10));
    check("p_rgh agrees with OpenFOAM's relatively, both running the case's PCG+DIC",
          dP.linf < pBoundHost * std::fmax(dP.refMax, scalar(1e-12)));

    // THE MOMENTUM PREDICTOR'S SOLVES, component by component, under `mompred`. OpenFOAM logs "Solving
    // for Ux" and "Solving for Uy" and NO Uz -- the empty direction is skipped, not solved to zero
    // (fvMatrixSolve.C:162-164) -- and takes the `UFinal` entry, because with one outer corrector that
    // corrector is the final one.
    std::vector<LinearSolveRecord> ofUx, ofUy, ofUz;
    // ...and under the two sheared profiles, which run the predictor too
    if ((momPred || sheared || gradLsqLimited) && argc > 5)
    {
        ofUx = brae::gatecheck::readOfSolves(argv[5], "Ux");
        ofUy = brae::gatecheck::readOfSolves(argv[5], "Uy");
        ofUz = brae::gatecheck::readOfSolves(argv[5], "Uz");
        check("OpenFOAM logged a Ux and a Uy solve for every step, and no Uz on this 2-D case",
              ofUx.size() == static_cast<std::size_t>(nSteps) && ofUy.size() == ofUx.size() && ofUz.empty());
        check("...and brae did not solve Uz either", r.uSolves[2].empty());
        failures += brae::gatecheck::compareSolves("host", r.uSolves[0], ofUx, nSteps, "Ux",
                                                   scalar(1e-10), scalar(1e-9), scalar(1e-6));
        failures += brae::gatecheck::compareSolves("host", r.uSolves[1], ofUy, nSteps, "Uy",
                                                   scalar(1e-10), scalar(1e-9), scalar(1e-6));
    }

    const FieldData<vector> ofUfd = readField<vector>(ofDir + "/U");
    std::vector<vector> ofU;
    if (ofUfd.internalUniform) ofU.assign(static_cast<std::size_t>(nC), ofUfd.internalUniformValue);
    else                       ofU = ofUfd.internalField;
    scalar uLinf = 0, uRef = 0;
    for (label c = 0; c < nC; ++c)
    {
        const vector& a = fin.U.internal[c];
        const vector& b = ofU[c];
        uLinf = std::fmax(uLinf, std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z)));
        uRef  = std::fmax(uRef,  std::sqrt(b.x*b.x + b.y*b.y + b.z*b.z));
    }
    std::printf("  U:      Linf %.4e            (|U| up to %.4e)\n", (double)uLinf, (double)uRef);
    std::printf("          relative %.3e\n", (double)(uLinf/std::fmax(uRef, scalar(1e-30))));
    // MEASURED 2.286e-11 relative; 2.0e-08 before totalPressure's dynamic term and 9.76e-06 on
    // PBiCGStab. U is rebuilt from the pressure flux, so it carries whatever p_rgh's solve leaves.
    check("U agrees with OpenFOAM's relatively, for the same reason",
          uLinf < uBoundHost * std::fmax(uRef, scalar(1e-12)));

    // THE `prevcorr` CONTROLS, on the oracle as the inflow one is. argv[8] is real OpenFOAM's alpha from
    // the run this profile differs from in ONE setting: for `prevcorr` the same run without the switch,
    // for `prevcorrsub` the same run without the second sub-cycle. If the two agreed to round-off that
    // setting would be doing nothing on this fixture, and a brae that ignored it would pass every
    // bound above -- as the device did, 1.07e-02 out, until it was given the switch at all.
    if (prevCorr || pimpleProfile || nHatLimited)
    {
        const char* what = prevCorrSub ? "nAlphaSubCycles 2"
                         : nOuter ? "nOuterCorrectors 2"
                         : nonOrth ? "nNonOrthogonalCorrectors 1"
                         : momPred ? "momentumPredictor yes"
                         : namedFlux ? "phi rhoPhi"
                         : vanLeerV ? "div(rhoPhi,U) Gauss vanLeerV"
                         : linear ? "div(rhoPhi,U) Gauss linear"
                         : compression ? "div(phirb,alpha) Gauss interfaceCompression"
                         : gradLsqLimited ? "default cellLimited leastSquares 1"
                         : nHatLimited ? "nHat cellLimited Gauss linear 1"
                                  : "alphaApplyPrevCorr yes";
        check("the control was given OpenFOAM's answer without the setting under test", argc > 8);
        if (argc > 8 && alphaMinIter)
        {
            // minIter DOES NOT MOVE damBreak's FIELDS: the one pre-solve that starts under tolerance is the
            // first, and it starts on an exact solution, so the forced sweep changes nothing (measured at
            // alpha tolerances 1e-8, 1e-5, 1e-4 and 1e-3: alpha identical to the last digit). What it moves
            // is the SWEEP COUNT OpenFOAM logs -- which the host's alpha solves above are held to -- so the
            // control is that count: OpenFOAM's run without minIter must log a different one.
            const std::string offLog = std::filesystem::path(argv[8]).parent_path().string() + "/log.interFoam";
            const std::vector<LinearSolveRecord> offSolves = brae::gatecheck::readOfSolves(offLog, fin.alphaName);
            const std::vector<LinearSolveRecord> onSolves = brae::gatecheck::readOfSolves(argv[5], fin.alphaName);
            std::size_t differ = 0;
            for (std::size_t k = 0; k < offSolves.size() && k < onSolves.size(); ++k)
            {
                differ += (offSolves[k].nIterations != onSolves[k].nIterations) ? 1 : 0;
            }
            std::printf("  CONTROL: without `minIter 1` OpenFOAM logs %zu of %zu alpha sweep counts differently\n",
                        differ, onSolves.size());
            check("...so the sweep counts brae is held to can tell minIter from its absence",
                  !onSolves.empty() && offSolves.size() == onSolves.size() && differ > 0);
        }
        else if (argc > 8)
        {
            const std::vector<scalar> offAlpha = readCells(std::string(argv[8]) + "/alpha.water");
            const Diff dSwitch = compare(offAlpha, ofAlpha);
            std::printf("  CONTROL: `%s` moves OpenFOAM's own alpha by %.4e; brae is %.4e from OpenFOAM "
                        "with it\n", what, (double)dSwitch.linf, (double)dAlpha.linf);
            check("...which is more than 1000x brae's distance from the oracle",
                  dSwitch.linf > scalar(1000)*dAlpha.linf && dSwitch.linf > scalar(1e-8));
        }
    }

    // THE `outflow` CONTROL, and this one is on BRAE: the host runs again with the limiter reading
    // phiCN, which is what it did before this profile measured it. Everything else in the run is the
    // same code on the same case. MEASURED 5.5e-03 of alpha; the arm asks for four orders less than
    // that, against a main-run bound of 5e-11.
    if (outflow)
    {
        setenv("BRAE_CONTROL_PREVCORR_PHICN", "1", 1);
        InterFields wrongF;
        const RunReport rw = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/false, &wrongF);
        unsetenv("BRAE_CONTROL_PREVCORR_PHICN");
        const Diff dWrong = compare(wrongF.alpha1.internal, ofAlpha);
        std::printf("  CONTROL: with phiCN in the outlet test brae is %.4e of alpha from OpenFOAM; with "
                    "alphaPhi10, %.4e\n", (double)dWrong.linf, (double)dAlpha.linf);
        check("...the control ran the same number of steps", rw.steps == nSteps);
        check("...and the WRONG argument is caught: more than 1e-6 out, against 5e-11 allowed",
              dWrong.linf > scalar(1e-6));
    }

    // THE `inflow` CONTROL: the ORACLE's answer has to depend on the term, or agreeing with it proves
    // nothing about the term. argv[8] is real OpenFOAM's U on the STANDARD case at the same instant;
    // the two differ only in the atmosphere's inletValue, and in three steps almost no water has
    // entered -- what moves the flow is rho_b on the inflow faces, through snGrad(rho) in phig and
    // through totalPressure's 0.5*rho_b*|U_b|^2. Without rho's patch values brae sat 100% from this
    // oracle, i.e. near the standard answer.
    if (inflow)
    {
        check("the inflow control was given the standard case's OpenFOAM answer", argc > 8);
        if (argc > 8)
        {
            const FieldData<vector> sfd = readField<vector>(std::string(argv[8]) + "/U");
            scalar apart = 0, stdMax = 0;
            for (label c = 0; c < nC && sfd.internalField.size() == static_cast<std::size_t>(nC); ++c)
            {
                const vector& a = sfd.internalField[static_cast<std::size_t>(c)];
                const vector& b = ofU[static_cast<std::size_t>(c)];
                apart = std::fmax(apart, std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z)));
                stdMax = std::fmax(stdMax, std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z));
            }
            std::printf("  CONTROL: OpenFOAM's max|U| is %.4e here and %.4e on the standard case; the two "
                        "answers are %.4e apart\n", (double)uRef, (double)stdMax, (double)apart);
            check("...OpenFOAM's own answer moves by more than 10x the standard case's whole velocity",
                  apart > scalar(10)*stdMax && stdMax > scalar(0));
        }
    }

    // ...and the conserved quantity, which neither solver's linear tolerance can move.
    scalar ofMass = 0, a0Mass = 0;
    for (label c = 0; c < nC; ++c) { ofMass += ofAlpha[c]*g.V()[c]; a0Mass += a0[c]*g.V()[c]; }
    std::printf("  water volume: initial %.10e, OpenFOAM %.10e, brae %.10e\n",
                (double)a0Mass, (double)ofMass, (double)r.alphaMass);
    if (inflow || outflow)
    {
        // water CROSSES the atmosphere on these profiles, so the domain is not closed and the
        // conservation arm has nothing to assert; what both codes let through is compared instead
        check("brae lets through the water OpenFOAM lets through, to 1e-12",
              std::fabs(r.alphaMass - ofMass)/ofMass < scalar(1e-12));
    }
    else
    {
        check("OpenFOAM conserves the water too (a closed domain)",
              std::fabs(ofMass - a0Mass)/a0Mass < scalar(1e-6));
        check("brae's water volume agrees with OpenFOAM's to 1e-9 -- both conserve, so this IS exact",
              std::fabs(r.alphaMass - ofMass)/ofMass < scalar(1e-9));
    }

    // THE DEVICE DRIVER, against OpenFOAM directly, at the case's own tolerances -- which is what
    // `brae_interFoam -device` runs on this tutorial. interfoam_dambreak_device_vs_host.sh compares
    // device with host under TIGHTENED solves, where the choice of pressure solver cannot show; here it
    // can, and until the device took the case's PCG with DIC it would have.
    {
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess)
        {
            cudaGetLastError();
            nDev = 0;
        }
        if (nDev <= 0)
        {
            std::printf("  (no CUDA device: the device arms are skipped)\n");
        }
        else
        {
            if (deviceRefuses)
            {
                // The device loop runs one outer corrector and no non-orthogonal pass. Running this
                // case at 1 and 0 would be a silent substitution, so what is asserted here is that it
                // does NOT run -- and that the message names the control the case asked for.
                bool threw = false;
                std::string why;
                try
                {
                    InterFields unused;
                    runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &unused);
                }
                catch (const std::exception& e)
                {
                    threw = true;
                    why = e.what();
                }
                const char* named = nonOrth ? "nNonOrthogonalCorrectors"
                                  : alphaMinIter ? "minIter"
                                  : (gradLsqLimited || nHatLimited) ? "cellLimited"
                    
                                            : "names the flux";
                std::printf("  DEVICE: %s\n", threw ? why.substr(0, 140).c_str() : "RAN -- it must not");
                check("the DEVICE refuses this case rather than run it at a smaller count",
                      threw && why.find(named) != std::string::npos);
                std::printf("test_inter_dambreak_vs_openfoam: %d failures\n", failures);
                return failures ? 1 : 0;
            }
            InterFields dev;
            const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
            check("the device driver ran the same number of steps", rd.steps == nSteps);
            failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
            failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
            failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
            if (argc > 5)
            {
                if (!nOuter)
                {
                    failures += brae::gatecheck::compareSolves("device", rd.pSolves, ofSolves, nSteps);
                }
                if (momPred)
                {
                    check("...and the device did not solve Uz either", rd.uSolves[2].empty());
                    failures += brae::gatecheck::compareSolves("device", rd.uSolves[0], ofUx, nSteps, "Ux",
                                                               scalar(1e-10), scalar(1e-9), scalar(1e-5));
                    failures += brae::gatecheck::compareSolves("device", rd.uSolves[1], ofUy, nSteps, "Uy",
                                                               scalar(1e-10), scalar(1e-9), scalar(1e-5));
                }
                // Under `bigstep` the device leaves OpenFOAM's final residuals to 5.0e-09 (the host
                // 9.1e-10: its reductions sum in OpenFOAM's order and the device's do not). Bound 1e-6.
                if (!nOuter)
                {
                    failures += brae::gatecheck::compareSolves("device", rd.alphaSolves, ofAlphaSolves, nSteps,
                                                               dev.alphaName.c_str(), scalar(1e-10),
                                                               scalar(1e-9), bigStep ? scalar(1e-6) : scalar(-1));
                }
            }
            const Diff da = compare(dev.alpha1.internal, ofAlpha);
            const Diff dp = compare(dev.p_rgh.internal, ofPrgh);
            scalar du = 0;
            for (label c = 0; c < nC; ++c)
            {
                const vector& a = dev.U.internal[c];
                const vector& b = ofU[c];
                du = std::fmax(du, std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z)));
            }
            std::printf("  DEVICE vs OpenFOAM: alpha %.4e, p_rgh %.3e relative, U %.3e relative\n",
                        (double)da.linf, (double)(dp.linf/std::fmax(dp.refMax, scalar(1e-30))),
                        (double)(du/std::fmax(uRef, scalar(1e-30))));
            // THE HOST'S BOUNDS, because the device now reads the host's numbers to three digits:
            // 3.6e-14, 8.6e-12, 2.3e-11. It did not when this arm was written -- alpha was 1.6e-10,
            // four thousand times the host's -- and the reason was the alpha pre-solve: the driver ran
            // Jacobi-BiCGStab to a hardcoded 1e-12 where the case names symGaussSeidel at 1e-8. At the
            // case's OWN 1e-8 that substitution read alpha 3.3e-06 and U 1.2e-03.
            check("the DEVICE's alpha agrees with OpenFOAM's to the host's bound", da.linf < alphaBoundDev);
            check("...its p_rgh to the device's bound for this profile",
                  dp.linf < pBoundDev*std::fmax(dp.refMax, scalar(1e-12)));
            check("...and its U", du < uBoundDev*std::fmax(uRef, scalar(1e-12)));
        }
    }

    std::printf("test_inter_dambreak_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
