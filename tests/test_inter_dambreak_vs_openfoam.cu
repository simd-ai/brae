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
    const bool bigStep = argc > 7 && std::string(argv[7]) == "bigstep";
    std::printf("  profile: %s\n", bigStep ? "bigstep -- the solver logs discriminate here" : "small step -- the tight field bounds live here");

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
        check("OpenFOAM's log gave an alpha solve for every step",
              ofAlphaSolves.size() == static_cast<std::size_t>(nSteps));
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
    const scalar alphaBound = bigStep ? scalar(5e-11) : scalar(1e-12);
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
    check("p_rgh agrees with OpenFOAM's to 2e-10 relative, both running the case's PCG+DIC",
          dP.linf < scalar(2e-10) * std::fmax(dP.refMax, scalar(1e-12)));

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
    check("U agrees with OpenFOAM's to 5e-10 relative, for the same reason",
          uLinf < scalar(5e-10) * std::fmax(uRef, scalar(1e-12)));

    // ...and the conserved quantity, which neither solver's linear tolerance can move.
    scalar ofMass = 0, a0Mass = 0;
    for (label c = 0; c < nC; ++c) { ofMass += ofAlpha[c]*g.V()[c]; a0Mass += a0[c]*g.V()[c]; }
    std::printf("  water volume: initial %.10e, OpenFOAM %.10e, brae %.10e\n",
                (double)a0Mass, (double)ofMass, (double)r.alphaMass);
    check("OpenFOAM conserves the water too (a closed domain)",
          std::fabs(ofMass - a0Mass)/a0Mass < scalar(1e-6));
    check("brae's water volume agrees with OpenFOAM's to 1e-9 -- both conserve, so this IS exact",
          std::fabs(r.alphaMass - ofMass)/ofMass < scalar(1e-9));

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
            InterFields dev;
            const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
            check("the device driver ran the same number of steps", rd.steps == nSteps);
            failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
            failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
            failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
            if (argc > 5)
            {
                failures += brae::gatecheck::compareSolves("device", rd.pSolves, ofSolves, nSteps);
                // Under `bigstep` the device leaves OpenFOAM's final residuals to 5.0e-09 (the host
                // 9.1e-10: its reductions sum in OpenFOAM's order and the device's do not). Bound 1e-6.
                failures += brae::gatecheck::compareSolves("device", rd.alphaSolves, ofAlphaSolves, nSteps,
                                                           dev.alphaName.c_str(), scalar(1e-10),
                                                           scalar(1e-9), bigStep ? scalar(1e-6) : scalar(-1));
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
            check("the DEVICE's alpha agrees with OpenFOAM's to the host's bound", da.linf < alphaBound);
            check("...its p_rgh to 2e-10 relative", dp.linf < scalar(2e-10)*std::fmax(dp.refMax, scalar(1e-12)));
            check("...and its U to 5e-10 relative", du < scalar(5e-10)*std::fmax(uRef, scalar(1e-12)));
        }
    }

    std::printf("test_inter_dambreak_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
