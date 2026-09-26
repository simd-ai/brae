// brae's interFoam turbulence against REAL OpenFOAM's, on RAS/damBreak, in BOTH of interFoam's lineages.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, as in
// test_inter_dambreak_vs_openfoam.cu, plus its own solver log for epsilon and k.
//
// THREE PROFILES -- tests/interfoam_ras_dambreak_vs_openfoam.sh says why each exists and carries the
// measurement of what each one can see:
//   variable   the tutorial as shipped (`density variable`): kEpsilon weighted by the mixture rho,
//              convecting with rhoPhi, and NO validate() -- the first UEqn runs on the file's nut = 0
//   uniform    the same case without that line: the ordinary single-phase model, validate() at
//              construction
//   custom     uniform with its own kEpsilonCoeffs, equation relaxation 0.7 and a tolerance at which
//              `minIter 1` is what forces each sweep -- three settings the shipped case cannot see
//
// THE DEVICE LOOP RUNS EVERY PROFILE TOO, twice: with the device closure (what `-device` runs), held to
// OpenFOAM at the host's bounds, and with the HOST closure in its place (BRAE_INTER_HOST_CLOSURE), held
// to the first. Against OpenFOAM a disagreement could be the loop's or the closure's; between those two
// it can only be the closure's. Each arm asserts WHICH closure it ran.
//
// TWO CONTROLS ON THE ORACLE, each OpenFOAM's own answer at the same instant:
//   laminar     the case with `simulationType laminar`. If the turbulent oracle sat on top of it, a
//               brae that ignored turbulence would pass every field bound here.
//   the other   the run this profile differs from in its one setting: the other lineage for
//               `variable` and `uniform`, the plain uniform run for `custom`.
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

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

struct Diff
{
    scalar linf = 0;
    scalar refMax = 0;
    scalar rel() const { return linf/std::fmax(refMax, scalar(1e-300)); }
};

Diff compare(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.linf = std::fmax(d.linf, std::fabs(a[i] - b[i]));
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    return d;
}

Diff compare(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const vector e{a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z};
        d.linf = std::fmax(d.linf, mag(e));
        d.refMax = std::fmax(d.refMax, mag(b[i]));
    }
    return d;
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/damBreak (kEpsilon) ==\n");
    if (argc < 10)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> "
                    "<variable|uniform|custom> <lineage> <laminarOfTimeDir> <otherOfTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string profile = argv[6];
    const bool variable = (std::string(argv[7]) == "variable");
    const std::string laminarDir = argv[8];
    const std::string otherDir = argv[9];
    const bool custom = (profile == "custom");
    // nutAtmosphere: the uniform lineage with the atmosphere's nut an inletOutlet, which
    // correctBoundaryConditions evaluates after the model's field assignment
    const bool nutAtmosphere = (profile == "nutAtmosphere");
    // `sst`: RAS/damBreak made kOmegaSST, whose second field is omega and whose closure runs on the
    // device through device_inter_turbulence's SST branch
    const bool sst = (profile == "sst");
    const bool outer = (profile == "outer");
    // `frozen*`: `RAS { turbulence off; }` -- a model that is CONSTRUCTED and VALIDATED and then never
    // corrected again (kEpsilon.C:216-219 gates correct() alone; eddyViscosity.C:119-122 is
    // `correctNut();` with no turbulence_ test). NOT laminar and NOT "keep the file's nut".
    const bool frozen = (profile.rfind("frozen", 0) == 0);
    // ...with kMin and epsilonMin set ABOVE the case's own 0.1, so the CONSTRUCTOR's bound
    // (kEpsilon.C:182-183) is the only thing in the run that touches either field
    const bool frozenFloored = (profile == "frozenFloored");
    const bool frozenSST = (profile == "frozenSST");
    // `flowFrozen`: `frozenFlow yes` in the PIMPLE dict. NOT a `frozen*` profile -- those switch the
    // turbulence MODEL off and `frozen` matches by prefix, so the name deliberately does not start
    // with it. Here the model is live and the whole outer iteration's momentum, pressure and
    // turbulence corrector are skipped instead (interFoam.C:156-158).
    const bool flowFrozen = (profile == "flowFrozen");
    // `splitSolve*`: the second equation carries its OWN `<field>Final` solver entry, which
    // fvMatrix::solve() looks up by FIELD name. The closure took k's for both until this profile.
    const bool splitSolveSST = (profile == "splitSolveSST");
    const bool splitSolve = (profile == "splitSolve") || splitSolveSST;
    // `splitDiv*`: the two equations carry DIFFERENT convection schemes, which fvm::div resolves by the
    // FIELD's name. The closure took k's for both until this profile; the device still refuses.
    const bool splitDivSST = (profile == "splitDivSST");
    const bool splitDiv = (profile == "splitDiv") || splitDivSST;
    // THE DEVICE LOOP'S kOmegaSST FLOOR, which every SST profile is held to and which the bounds below
    // were measured against: nut carries the device loop's U difference through k, omega and F2, so an
    // SST profile sits three orders above a kEpsilon one. Named ONCE rather than spelled out per
    // bound: `splitDivSST` was added later and fell into the kEpsilon branch by omission, reading the
    // documented SST floor (omega 3.08e-11 where `sst` itself reads 3.09e-11) against a 1e-11 bound.
    const bool sstFloor = sst || splitSolveSST || splitDivSST;
    // `lowRe`: the epsilon wall patches name `lowReCorrection true`, which switches epsilon to the
    // resolved form on faces with y+ < yPlusLam and drops their wall production. Parsed by brae's reader
    // all along and read by nothing under interFoam.
    const bool lowRe = (profile == "lowRe") || (profile == "lowReOff");
    // `turbOuter*`: `turbOnFinalIterOnly no` with nOuterCorrectors 2, so the closure runs on EVERY outer
    // corrector -- TWICE per step. `turbOuterSplit` adds literal kFinal/epsilonFinal naming PBiCG beside
    // the case's regex, so corrector 1 runs the smoothSolver and corrector 2 runs PBiCG.
    const bool turbOuterSplit = (profile == "turbOuterSplit");
    const bool turbOuter = (profile == "turbOuter") || turbOuterSplit;
    // how many closure solves OpenFOAM logs per STEP: one per corrector the closure runs on
    const label closurePerStep = turbOuter ? 2 : 1;
    const bool lowReOn = (profile == "lowRe");
    const char* secondName = (sst || frozenSST || splitSolveSST || splitDivSST) ? "omega" : "epsilon";
    std::printf("  profile: %s\n",
                lowRe ? "lowRe -- epsilonWallFunction with lowReCorrection true"
              : splitDivSST ? "splitDivSST -- div(phi,omega) limitedLinear beside div(phi,k) upwind"
              : splitDiv ? "splitDiv -- div(phi,epsilon) limitedLinear beside div(phi,k) upwind"
              : splitSolveSST ? "splitSolveSST -- omegaFinal its own entry, tolerance 1e-12 and 2 sweeps"
              : splitSolve ? "splitSolve -- epsilonFinal its own entry, tolerance 1e-12 and 2 sweeps"
              : frozenFloored ? "frozenFloored -- turbulence off, with floors above the case's own fields"
              : frozenSST ? "frozenSST -- kOmegaSST with turbulence off"
              : flowFrozen ? "flowFrozen -- `frozenFlow yes`: alpha advances, nothing else is solved"
              : frozen ? "frozen -- turbulence off: constructed, validated, never corrected"
              : outer ? "outer -- variable, with nOuterCorrectors 2"
              : nutAtmosphere ? "nutAtmosphere -- uniform, the atmosphere's nut an inletOutlet"
              : custom ? "custom -- its own coefficients, relaxation 0.7, and minIter forcing each sweep"
              : variable ? "variable -- `density variable`, as the tutorial ships"
                         : "uniform -- the ordinary single-phase lineage");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    // THE PATH, not only the answer: a laminar run of this case also reaches the end
    check("brae ran the case turbulent", fin.turbulence.on);
    check("...in the lineage this profile names", fin.turbulence.variableDensity == variable);
    // `on` STAYS TRUE under `turbulence off` -- nuEff is still nut + nu. `frozen` is the flag, and a
    // brae that read the switch as laminar would fail the first of these, not the second.
    check(frozen ? "...with the model FROZEN by `turbulence off`, which is not laminar"
                 : "...with the model live", fin.turbulence.frozen == frozen);
    if (custom)
    {
        // what was READ, beside the answer it produces: the oracle control below says the three
        // settings move OpenFOAM, and these say brae picked each of them up from the case
        const InterTurbulence& t = fin.turbulence;
        check("brae read the case's six kEpsilonCoeffs",
              t.coeffs.Cmu == scalar(0.12) && t.coeffs.C1 == scalar(1.5) && t.coeffs.C2 == scalar(1.8)
           && t.coeffs.C3 == scalar(0.2) && t.coeffs.sigmaK == scalar(1.1)
           && t.coeffs.sigmaEps == scalar(1.2));
        check("...the 0.7 relaxation under the names kFinal and epsilonFinal",
              t.kRelaxFinal.on && t.kRelaxFinal.factor == scalar(0.7)
           && t.epsRelaxFinal.on && t.epsRelaxFinal.factor == scalar(0.7));
        check("...and minIter 1 at tolerance 0.5",
              t.kSolveFinal.minIter == 1 && t.kSolveFinal.tol == scalar(0.5)
           && t.epsSolveFinal.minIter == 1);
    }

    if (splitSolve)
    {
        const InterTurbulence& t = fin.turbulence;
        const SmoothLinearSolve& ks = t.kSolveFinal;
        const SmoothLinearSolve& es = splitSolveSST ? t.omegaSolveFinal : t.epsSolveFinal;
        std::printf("  SPLIT: kFinal tol %.3g nSweeps %d   |   %sFinal tol %.3g nSweeps %d\n",
                    (double)ks.tol, ks.nSweeps, secondName, (double)es.tol, es.nSweeps);
        // THE PROFILE IS NOT GATING ONE SETTING TWICE. Without this the arm would pass on a brae that
        // resolved BOTH names through the case's `"(U|k|epsilon).*"` regex and never saw the literal.
        check("brae read the case's own entry for kFinal", ks.tol == scalar(1e-06) && ks.nSweeps == 1);
        check("...and the second field's SEPARATE entry, which the regex beside it also matches",
              es.tol == scalar(1e-12) && es.nSweeps == 2);
        check("...so the two differ, which is what this profile exists to run",
              ks.tol != es.tol && ks.nSweeps != es.nSweeps);
    }

    if (splitDiv)
    {
        const InterTurbulence& t = fin.turbulence;
        std::printf("  SPLIT DIV: k limitedLinear %d coeff %.3g   |   %s limitedLinear %d coeff %.3g\n",
                    (int)t.kDiv.limitedLinear, (double)t.kDiv.limiterCoeff, secondName,
                    (int)t.secondDiv.limitedLinear, (double)t.secondDiv.limiterCoeff);
        // THE PROFILE IS NOT CHANGING BOTH. k keeps the case's own `Gauss upwind`, so exactly one
        // equation's matrix moves and the arm measures one difference.
        check("brae read div(phi,k) as the case's own Gauss upwind", !t.kDiv.limitedLinear);
        check("...and the second field's as Gauss limitedLinear 1",
              t.secondDiv.limitedLinear && t.secondDiv.limiterCoeff == scalar(1));
    }

    if (lowRe)
    {
        // THE FLAG, in brae's own struct: without this the arm would pass on a brae that parsed the entry
        // and threaded nothing, which is exactly the defect it exists to hold.
        check(lowReOn ? "brae read `lowReCorrection` off the epsilon wall patches"
                      : "...and the control run has it OFF, so the pair differs in that entry alone",
              fin.turbulence.coeffs.epsLowRe == lowReOn);
    }

    if (turbOuter)
    {
        const InterTurbulence& t = fin.turbulence;
        std::printf("  TURB EVERY CORRECTOR: kFinal solver `%s`/`%s`, k solver `%s`/`%s`\n",
                    t.kSolveFinal.solver.c_str(), t.kSolveFinal.preconditioner.c_str(),
                    t.kSolve.solver.c_str(), t.kSolve.preconditioner.c_str());
        check("brae read the case's NON-Final closure solver entry", !t.kSolve.solver.empty());
        if (turbOuterSplit)
        {
            // THE TWO ENTRIES MUST DIFFER IN BRAE'S OWN STRUCTS, or the arm would pass on a brae that
            // resolved both names through the case's regex and never saw the literal kFinal.
            check("...and the Final entry is the PBiCG the profile names, the non-Final the regex's "
                  "smoothSolver", t.kSolveFinal.pbicgDILU() && t.kSolve.gaussSeidel());
            // ...and OpenFOAM's own log carries one of each per step, which is this arm's categorical
            // witness: the solver NAME is printed on the solve line (SolverPerformance.C:99).
            std::size_t gs = 0, pb = 0;
            std::ifstream lf(logPath);
            for (std::string ln; std::getline(lf, ln); )
            {
                if (ln.find("Solving for k,") == std::string::npos) continue;
                gs += (ln.find("smoothSolver:") != std::string::npos) ? 1 : 0;
                pb += (ln.find("DILUPBiCG:") != std::string::npos) ? 1 : 0;
            }
            std::printf("  OpenFOAM logged %zu smoothSolver and %zu DILUPBiCG k solves\n", gs, pb);
            check("OpenFOAM ran the smoothSolver on one corrector and PBiCG on the other, every step",
                  gs == static_cast<std::size_t>(nSteps) && pb == static_cast<std::size_t>(nSteps));
        }
    }

    auto readCells = [&](const std::string& path)
    {
        const FieldData<scalar> fd = readField<scalar>(path);
        std::vector<scalar> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };
    auto readVectorCells = [&](const std::string& path)
    {
        const FieldData<vector> fd = readField<vector>(path);
        std::vector<vector> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };

    // the solver logs: p_rgh and alpha as on the laminar case, and the closure's two
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofA = brae::gatecheck::readOfSolves(logPath, fin.alphaName);
    const std::vector<LinearSolveRecord> ofE = brae::gatecheck::readOfSolves(logPath, secondName);
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    // the FAIL-PROOF: nothing in compareSolves can pass on an empty parse
    if (frozen)
    {
        // THE DIRECT EVIDENCE that correct() never ran, in OpenFOAM's own log: a frozen model prints no
        // `Solving for k` line at all. This is the one assertion here that an empty parse passes, so it
        // is paired with the pressure and alpha solves below, which must still be nSteps long.
        check("OpenFOAM solved NEITHER closure field -- correct() returned immediately every step",
              ofE.empty() && ofK.empty());
        check("...and brae solved neither either",
              r.kSolves.empty() && r.epsilonSolves.empty() && r.omegaSolves.empty());
    }
    else
    {
        // ...and with `turbOnFinalIterOnly no` that is one per CORRECTOR, not one per step: a brae still
        // running the closure once per step logs half as many and compareSolves mismatches on length.
        check("OpenFOAM's log gave one solve of the closure's second field and one k solve per corrector",
              flowFrozen ? (ofE.empty() && ofK.empty())
                         : (ofE.size() == static_cast<std::size_t>(nSteps*closurePerStep)
                            && ofK.size() == ofE.size()));
    }
    if (flowFrozen)
    {
        // MEASURED in OpenFOAM's own log: 0 lines of `Solving for p_rgh`, `for k` or `for epsilon`, and 5
        // of `Solving for alpha.water` -- one per step. So the assertion is not a count comparison but an
        // EMPTINESS, which is the stronger statement and the one the `continue` actually makes.
        check("brae solved NO p_rgh -- the pressure corrector loop never ran", r.pSolves.empty());
        check("...and no k", r.kSolves.empty());
        check("...and no second closure field", r.epsilonSolves.empty() && r.omegaSolves.empty());
        check("OpenFOAM logged none either", ofP.empty() && ofK.empty() && ofE.empty());
        check("...and both still solved alpha once per step",
              r.alphaSolves.size() == static_cast<std::size_t>(nSteps) && ofA.size() == r.alphaSolves.size());
    }
    else
    {
        failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps);
    }
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               scalar(1e-10), scalar(1e-9));
    // MEASURED: initial residuals 1.8e-13 (epsilon) and 1.3e-13 (k) from OpenFOAM's over the run, so
    // both bounds are 1e-10. The FINAL residual's worst is 8.8e-07 relative and that is round-off, not
    // the solver: it is step one's k solve, which ends at 4.140e-10, and the two codes are 3.6e-16
    // apart there -- a few units in the last place of a normalised residual. PBiCGStab in the same
    // seat takes 1 of 5 iteration counts and leaves final residuals 100% out.
    if (!frozen && !flowFrozen)
    {
        failures += brae::gatecheck::compareSolves("host", (sst || frozenSST || splitSolveSST || splitDivSST) ? r.omegaSolves : r.epsilonSolves,
                                                   ofE, nSteps*closurePerStep, secondName,
                                                   scalar(1e-10), scalar(1e-10), scalar(1e-5));
        failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps*closurePerStep, "k",
                                                   scalar(1e-10), scalar(1e-10), scalar(1e-5));
    }

    // BEFORE any fmax: std::fmax drops a NaN -- see tests/device_gate_finite.cuh
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
    const std::vector<scalar>& braeSecond = (sst || frozenSST || splitSolveSST || splitDivSST) ? fin.turbulence.omega.internal
                                                : fin.turbulence.epsilon.internal;
    failures += brae::gatecheck::nonFinite("brae second field", braeSecond);
    // THE FAIL-PROOF. `braeSecond` selects omega or epsilon by profile, and the other of the two is
    // EMPTY -- compare() of an empty field reads Linf 0 and refMax 0, so it passes every bound and every
    // "is exactly the start value" assertion below. frozenSST selected epsilon under kOmegaSST once and
    // five assertions passed on nothing at all.
    check("brae's second closure field has one value per cell -- not the other model's empty one",
          braeSecond.size() == static_cast<std::size_t>(nC));
    failures += brae::gatecheck::nonFinite("brae nut", fin.turbulence.nut.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const std::vector<scalar> ofKf = readCells(ofDir + "/k");
    const std::vector<scalar> ofEf = readCells(ofDir + "/" + std::string(secondName));
    const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
    check("OpenFOAM's fields have one value per cell",
          ofAlpha.size() == static_cast<std::size_t>(nC) && ofKf.size() == ofAlpha.size()
       && ofNut.size() == ofAlpha.size());

    if (flowFrozen)
    {
        // WHAT `frozenFlow yes` MEANS, asserted term by term. Agreement with the oracle alone would not
        // say WHICH quantities were frozen, and the case as shipped cannot say it at all: from rest every
        // field stays put and a brae that skipped the alpha equation too would read the same numbers. The
        // start directory here is a DEVELOPED state (max|U| 0.7926), so the two halves separate.
        check("brae read `frozenFlow` from the PIMPLE dict", fin.pimple.frozenFlow);
        const std::vector<vector> u0 = readVectorCells(startDir + "/U");
        const std::vector<scalar> p0 = readCells(startDir + "/p_rgh");
        const std::vector<scalar> k0 = readCells(startDir + "/k");
        const std::vector<scalar> e0 = readCells(startDir + "/" + std::string(secondName));
        const std::vector<scalar> n0 = readCells(startDir + "/nut");
        const std::vector<scalar> a0 = readCells(startDir + "/" + fin.alphaName);
        const Diff fU = compare(fin.U.internal, u0);
        const Diff fP = compare(fin.p_rgh.internal, p0);
        const Diff fK = compare(fin.turbulence.k.internal, k0);
        const Diff fE = compare(braeSecond, e0);
        const Diff fN = compare(fin.turbulence.nut.internal, n0);
        const Diff fA = compare(fin.alpha1.internal, a0);
        std::printf("  FROZEN FLOW: against the start directory -- U %.4e, p_rgh %.4e, k %.4e, %s %.4e, "
                    "nut %.4e, and alpha MOVED %.4e\n", (double)fU.linf, (double)fP.linf, (double)fK.linf,
                    secondName, (double)fE.linf, (double)fN.linf, (double)fA.linf);
        // THE THREE THINGS THE `continue` SKIPS, one assertion each, absolute rather than bounded
        check("U is EXACTLY the start directory's -- no momentum and no pressure solve touched it",
              fU.linf == scalar(0));
        check("...and p_rgh is", fP.linf == scalar(0));
        check("...and k is -- the `continue` skips turbulence->correct() too, not only UEqn and pEqn",
              fK.linf == scalar(0));
        check("...and the second closure field is", fE.linf == scalar(0));
        // nut IS THE ONE FIELD THAT IS NOT BIT-IDENTICAL, on EITHER side, and it is held at round-off
        // rather than at zero for that reason. MEASURED on this arm: brae 1.1449e-16 from the start
        // directory, OpenFOAM 9.2e-16, brae against OpenFOAM 9.2404e-16, on a nut of order 9.4e-03. Both
        // codes re-form nut from k and epsilon rather than keeping the stored field, and since k and
        // epsilon are EXACTLY unchanged (asserted above) the re-formed value can only differ in the last
        // bits. It is the four TRANSPORTED quantities that carry this profile's claim, and those are
        // exact. A standalone run of the same case read 0.0 here, which this gate does not reproduce and
        // which is NOT chased: an assertion of exactness that cannot be explained is not one to ship.
        check("...and nut is the start value to round-off -- both codes re-form it from k and epsilon",
              fN.linf < scalar(1e-14));
        check("...and OpenFOAM's nut is too, by the same amount", compare(ofNut, n0).linf < scalar(1e-14));
        // ...AND THE ONE IT DOES NOT. Without this the arm passes on a brae that skips the whole step.
        check("alpha DID advance on the frozen velocity field -- the alpha equation is above the "
              "`continue` (interFoam.C:152) and must still run", fA.linf > scalar(1e-3));
        // OpenFOAM's own answer has to agree on both halves, or brae is being held against a run that
        // did not freeze what this profile is about
        check("OpenFOAM's U is EXACTLY its start value too",
              compare(ofU, u0).linf == scalar(0));
        check("...and OpenFOAM's second field is", compare(ofEf, e0).linf == scalar(0));
        check("...and OpenFOAM's alpha advanced as well", compare(ofAlpha, a0).linf > scalar(1e-3));
    }
    if (frozen)
    {
        // WHAT `turbulence off` MEANS, asserted term by term rather than only as agreement with the
        // oracle -- because OpenFOAM and a brae that froze the wrong quantity could still both be
        // "unchanged" and differ in which quantity that was.
        const std::vector<scalar> k0 = readCells(startDir + "/k");
        const std::vector<scalar> e0 = readCells(startDir + "/" + std::string(secondName));
        const std::vector<scalar> n0 = readCells(startDir + "/nut");
        const Diff fK = compare(fin.turbulence.k.internal, k0);
        const Diff fE = compare(braeSecond, e0);
        // the two transported scalars never move: no solve touched them, and with the floors below the
        // case's own values the constructor's bound did not fire either
        std::printf("  FROZEN: k against the start directory's %.4e, %s %.4e\n",
                    (double)fK.linf, secondName, (double)fE.linf);
        if (!frozenFloored)
        {
            check("k is EXACTLY the start directory's -- nothing solved it", fK.linf == scalar(0));
            check("...and so is the second field", fE.linf == scalar(0));
            // OpenFOAM's own, the same way: the oracle has to agree that the fields are frozen, or this
            // arm is measuring brae against a run that did advance them
            check("OpenFOAM's k is EXACTLY its start value too", compare(ofKf, k0).linf == scalar(0));
            check("...and OpenFOAM's second field is too", compare(ofEf, e0).linf == scalar(0));
        }
        else
        {
            // the FLOORS, and they are the only thing that touched either field. bound() replaces a
            // NEGATIVE cell by its neighbours' average and merely FLOORS a small one (bound.C:48-58);
            // 0.1 is small and positive, so both fields come out at the floor exactly.
            const scalar floorV = scalar(0.5);
            scalar kMin = fin.turbulence.k.internal[0];
            scalar eMin = braeSecond[0];
            for (const scalar v : fin.turbulence.k.internal) kMin = std::fmin(kMin, v);
            for (const scalar v : braeSecond) eMin = std::fmin(eMin, v);
            std::printf("  FROZEN+FLOORED: brae min(k) %.17g, min(%s) %.17g   (floor %.17g)\n",
                        (double)kMin, secondName, (double)eMin, (double)floorV);
            // THE WITNESS for the constructor's bound: without it k stays at the case's 0.1, which is
            // 5x below the floor, and nut comes out 0.009 instead of 0.045.
            check("the CONSTRUCTOR's bound floored k -- brae's interFoam reader did this in its LES "
                  "branch only until this arm", kMin == floorV);
            check("...and the second field", eMin == floorV);
            check("...and it is NOT the case's own 0.1", k0[0] == scalar(0.1) && kMin != k0[0]);
            check("OpenFOAM floored both the same way", compare(ofKf, fin.turbulence.k.internal).linf == scalar(0)
               && compare(ofEf, braeSecond).linf == scalar(0));
        }
        // AND THE DISCRIMINATOR BETWEEN THE THREE READINGS OF THE SWITCH. The case ships nut uniform 0.
        // laminar leaves it 0; "keep the file's nut" leaves it 0; validate() rebuilds it. So a non-zero
        // nut is the whole of the evidence that validate() is not gated on turbulence_.
        scalar nutMax = 0;
        for (const scalar v : ofNut) nutMax = std::fmax(nutMax, std::fabs(v));
        std::printf("  FROZEN: the start directory's nut was %.17g, OpenFOAM's is up to %.17g\n",
                    (double)n0[0], (double)nutMax);
        check("the start directory's nut is 0, so this fixture can tell the readings apart",
              n0[0] == scalar(0));
        check("OpenFOAM's nut is NOT 0 -- validate() ran although `turbulence` is off",
              nutMax > scalar(1e-6));
        if (!frozenSST)
        {
            // kEpsilon::correctNut is Cmu*sqr(k)/epsilon exactly (kEpsilon.C:75), and with both fields
            // frozen it is one number over the whole field: 0.09*0.01/0.1 = 0.009 unfloored,
            // 0.09*0.25/0.5 = 0.045 floored. Asserting the VALUE, not just that it moved.
            const scalar kv = frozenFloored ? scalar(0.5) : scalar(0.1);
            const scalar want = scalar(0.09) * kv * kv / kv;
            std::printf("  FROZEN: Cmu*k^2/epsilon = %.17g\n", (double)want);
            check("...and it is exactly Cmu*k^2/epsilon at the frozen fields",
                  std::fabs(nutMax - want) < scalar(1e-14) * want);
        }
    }

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    const Diff dK = compare(fin.turbulence.k.internal, ofKf);
    const Diff dE = compare(braeSecond, ofEf);
    const Diff dN = compare(fin.turbulence.nut.internal, ofNut);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
    std::printf("  %-8s relative %.4e   (%s up to %.4e)\n", (std::string(secondName) + ":").c_str(),
                (double)dE.rel(), secondName, (double)dE.refMax);
    std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dN.rel(), (double)dN.refMax);

    // MEASURED, worst of the three profiles: alpha 4.9e-13, p_rgh 7.1e-13, U 1.3e-11, k 2.8e-13,
    // epsilon 2.4e-13, nut 4.8e-13. Each bound is about 30x its measurement. The smallest thing this
    // port could get wrong and still run -- handing inletOutlet rhoPhi where it looks up phi -- is
    // 4.9e-04 of U, six orders outside.
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < scalar(2e-11));
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < scalar(2e-11));
    check("U agrees with OpenFOAM's relatively", dU.rel() < scalar(5e-10));
    check("k agrees with OpenFOAM's relatively", dK.rel() < scalar(1e-11));
    check("the closure's second field agrees with OpenFOAM's relatively", dE.rel() < scalar(1e-11));
    check("nut agrees with OpenFOAM's relatively", dN.rel() < scalar(2e-11));

    // THE CONTROLS, on the oracle
    const Diff dLamU = compare(readVectorCells(laminarDir + "/U"), ofU);
    const Diff dOtherU = compare(readVectorCells(otherDir + "/U"), ofU);
    const Diff dOtherNut = compare(readCells(otherDir + "/nut"), ofNut);
    std::printf("  CONTROL: OpenFOAM laminar against OpenFOAM turbulent, U relative %.4e\n",
                (double)dLamU.rel());
    std::printf("  CONTROL: OpenFOAM without this profile's setting against OpenFOAM with it, U relative "
                "%.4e, nut %.4e\n", (double)dOtherU.rel(), (double)dOtherNut.rel());
    // MEASURED 1.33 (laminar), 0.29 (the other lineage) and 3.7e-02 (custom against plain uniform),
    // beside a U bound of 5e-10
    check(flowFrozen ? "solving the flow moves OpenFOAM's own U more than 10% against freezing it -- "
                       "`argv[8]` is the same developed start WITHOUT `frozenFlow yes`, so the one thing "
                       "between the two runs is that entry. MEASURED: U 3.2493e-01, p_rgh 9.0768e-02, "
                       "epsilon 1.4379e-01, alpha 1.4387e-02, over all 2268 cells"
          : frozen ? "the FROZEN nut moves OpenFOAM's own U more than 10% from its laminar answer -- so a "
                   "brae that read `turbulence off` as laminar, or as `keep the file's nut`, would be "
                   "this far out and could not pass the field bounds above"
                 : "turbulence moves OpenFOAM's own U by more than 10%",
          dLamU.rel() > scalar(0.1));
    // MEASURED for nutAtmosphere against plain uniform: U 2.7e-03, nut 4.7e-02 -- 20 of the atmosphere's
    // 46 faces take air IN at t = 0.005, where the inletValue stands in for the cell's nut
    // ...and the second outer corrector against one: MEASURED U 4.6e-02 at t = 0.005
    check(turbOuterSplit ? "...and the Final solver entry alone moves OpenFOAM's own U -- small, because it "
                           "is a stopping point; the SOLVER NAMES and sweep counts above are what this arm "
                           "really gates"
          : turbOuter ? "...and running the closure on every outer corrector moves OpenFOAM's own U by more "
                        "than 1% against running it once per step"
          : lowRe ? "...and `lowReCorrection` moves OpenFOAM's own U by more than 1% against the log law. At "
                  "the tutorial's own nu = 1e-6 it moves it by EXACTLY ZERO -- no face has y+ under "
                  "yPlusLam -- which is why this profile raises the water viscosity to 1e-2"
          : splitDiv ? "...and limitedLinear on the second equation alone moves OpenFOAM's U against upwind "
                     "on both -- a different matrix, not a different stopping point"
          : splitSolve ? "...and the second equation's own solver entry moves OpenFOAM's U at all against "
                       "one entry for both -- it is a stopping point, so the move is small and the "
                       "ITERATION COUNTS above are what this profile really gates"
          : frozenFloored ? "...and the FLOORS alone move it by more than 1%, against the same case frozen "
                          "at the default floors -- so the constructor's bound is visible here"
          : frozenSST ? "...and switching kOmegaSST off moves it by more than 10%"
          : flowFrozen ? "...and the same run again as the second control: there is only one setting "
                         "between frozen and solved, so both controls are the same run by design"
          : frozen ? "...and switching kEpsilon off moves it by more than 10%"
          : outer ? "...and the second outer corrector moves it by more than 1%"
          : nutAtmosphere ? "...and the atmosphere's inletOutlet nut moves it by more than 1e-3"
          : custom ? "...and the custom settings move it by more than 1%"
                   : "...and the lineage moves it by more than 10%, so `density` is live on this fixture",
          dOtherU.rel() > (flowFrozen ? scalar(0.1)
                         : turbOuterSplit ? scalar(1e-7)
                         : turbOuter ? scalar(0.01)
                         : lowRe ? scalar(0.01)
                         : nutAtmosphere ? scalar(1e-3)
                         // MEASURED: kEpsilon 1.5e-04 of U (nut 4.5e-03), kOmegaSST 2.4e-05 (nut
                         // 2.2e-04) -- the SST closure's omega is far less sensitive to its convection
                         // scheme here than kEpsilon's epsilon is. brae is 1.3e-14 and 1.1e-12 from
                         // OpenFOAM on the two, so both bounds keep at least four orders of margin.
                         : splitDivSST ? scalar(1e-5)
                         : splitDiv ? scalar(1e-4)
                         : splitSolve ? scalar(1e-9)
                         : (custom || outer || frozenFloored) ? scalar(0.01) : scalar(0.1)));

    // THE DEVICE LOOP, against OpenFOAM directly and at the case's own tolerances -- what
    // `brae_interFoam -device` runs on this tutorial.
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device arm is skipped)\n");
    }
    // NO DEVICE REFUSAL OF THE CONVECTION SPLIT ANY MORE. The closures take each equation's own div
    // entry (KEpsilonInput::epsDiv, KOmegaSSTInput::omegaDiv, read by assembleTransport/schemeOf), so
    // `splitDiv` and `splitDivSST` fall through to the ordinary device arm below and are held to its
    // bounds -- exactly as `splitSolve` and `splitSolveSST` already did once the solver split landed.
    else
    {
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        check("the device driver ran the same number of steps", rd.steps == nSteps);
        check("...turbulent, in the lineage this profile names",
              dev.turbulence.on && dev.turbulence.variableDensity == variable);
        // THE PATH: the device loop can carry either closure, and "agrees with OpenFOAM" is true of both
        check("...with the closure ON THE DEVICE", rd.turbulenceOnDevice);
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        failures += brae::gatecheck::nonFinite("device k", dev.turbulence.k.internal);
        const std::vector<scalar>& devSecond = (sst || frozenSST || splitSolveSST || splitDivSST) ? dev.turbulence.omega.internal
                                                   : dev.turbulence.epsilon.internal;
        failures += brae::gatecheck::nonFinite("device second field", devSecond);
        failures += brae::gatecheck::nonFinite("device nut", dev.turbulence.nut.internal);
        if (flowFrozen)
        {
            // as the host half: OpenFOAM logs no p_rgh, k or second-field solve at all under
            // `frozenFlow yes`, so emptiness is the assertion and it is the stronger one
            check("the DEVICE solved no p_rgh either", rd.pSolves.empty());
            check("...and no k", rd.kSolves.empty());
            check("...and no second closure field",
                  rd.epsilonSolves.empty() && rd.omegaSolves.empty());
            check("...and still solved alpha once per step",
                  rd.alphaSolves.size() == static_cast<std::size_t>(nSteps));
        }
        else
        {
            failures += brae::gatecheck::compareSolves("device", rd.pSolves, ofP, nSteps);
        }
        // `outer`: the second pass's alpha pre-solve starts from the first pass's alpha and its initial
        // residual is 1e-5 of the normFactor, so the device's 1e-12 in phi (its VoF floor) reads as
        // 1e-07 relative there -- see tests/test_inter_cn_vs_openfoam.cu, where it was measured
        failures += brae::gatecheck::compareSolves("device", rd.alphaSolves, ofA, nSteps,
                                                   dev.alphaName.c_str(), scalar(1e-10),
                                                   outer ? scalar(1e-5) : scalar(1e-9));
        // MEASURED: initial residuals 1.8e-13 (epsilon) and 4.0e-13 (k) from OpenFOAM's over the run.
        // THE CONTROL IS ON `custom`: with the wall laplacian coefficient left out of relax() -- what
        // the device closure did until this gate -- epsilon's fields do not move (9.8e-14) and its
        // initial residuals are 1.1e-04 out in every step. 1e-10 is six orders inside that.
        if (frozen || flowFrozen)   // no closure solve to compare: the model is off, or the `continue` skipped it
        {
            // the device closure is BUILT (turbulenceOnDevice above) and advances nothing, so there is
            // no solve to compare -- the assertion is that it ran none, the same one the host arm makes
            check("the DEVICE closure solved neither field either -- it took the same early return",
                  rd.kSolves.empty() && rd.epsilonSolves.empty() && rd.omegaSolves.empty());
        }
        else
        {
            // `splitDivSST` GETS ITS OWN RESIDUAL BOUND, and the reason is the FIRST call, not the
            // discretisation. At call one omega is the uniform initial field, so limitedLinear's limiter
            // is min(k*|snGrad|/((1-k)*|corr| + SMALL), 1) on a gradient that is identically zero -- a
            // 0/0 ratio the last bit decides. MEASURED: this arm reads 1.275e-07 on that first solve's
            // initial residual and 1.3e-13 or below on every later one, while `sst` and `splitSolveSST`
            // (omega upwind, no limiter) read 3.599e-15 and the HOST on this same profile reads
            // 2.742e-15 -- the same coin-flip landing the other way.
            //
            // WHY THIS IS NOT A LOOSENED CLAIM: the fields are at the SST device floor (omega 3.08e-11
            // against `sst`'s own 3.09e-11), the device closure agrees with the HOST closure to 3.98e-15
            // in omega on this very profile, every iteration count is OpenFOAM's, and the ASSEMBLED
            // system is gated directly -- tests/interfoam_sst_assembly_vs_openfoam.sh arms F/G hold the
            // split system against OpenFOAM's own at 3.4e-14, with H and I as controls that each miss by
            // 0.5-1.0 on exactly one equation. The assembly gate carries the claim; this bound only
            // declines to measure a 0/0 with a field comparison.
            const scalar secondStepOne = splitDivSST ? scalar(2e-7) : scalar(1e-10);
            failures += brae::gatecheck::compareSolves("device", (sst || frozenSST || splitSolveSST || splitDivSST) ? rd.omegaSolves : rd.epsilonSolves,
                                                       ofE, nSteps*closurePerStep, secondName, secondStepOne, secondStepOne,
                                                       scalar(1e-5));
            failures += brae::gatecheck::compareSolves("device", rd.kSolves, ofK, nSteps*closurePerStep, "k",
                                                       scalar(1e-10), scalar(1e-10), scalar(1e-5));
        }
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        const Diff eK = compare(dev.turbulence.k.internal, ofKf);
        const Diff eE = compare(devSecond, ofEf);
        const Diff eN = compare(dev.turbulence.nut.internal, ofNut);
        std::printf("  DEVICE vs OpenFOAM: alpha %.4e, p_rgh %.4e, U %.4e, k %.4e, %s %.4e, nut %.4e\n",
                    (double)eA.linf, (double)eP.rel(), (double)eU.rel(), (double)eK.rel(), secondName,
                    (double)eE.rel(), (double)eN.rel());
        // THE HOST'S BOUNDS. MEASURED, worst of the three profiles: alpha 4.9e-13, p_rgh 7.1e-13,
        // U 3.3e-11, k 2.8e-13, epsilon 2.5e-13, nut 4.8e-13. Each device-side decision was broken in
        // turn, once: the patches handed rhoPhi for phi 4.9e-04 of U; the wall nut recomputed instead
        // of read from the stored patch 6.1e-06; rho.oldTime() taken as rho 5.6%; divU from the mass
        // flux 53%; BiCGStab for the case's symGaussSeidel 1 of 5 epsilon iteration counts.
        check("the DEVICE's alpha agrees with OpenFOAM's to the host's bound", eA.linf < scalar(2e-11));
        check("...its p_rgh", eP.rel() < scalar(2e-11));
        check("...its U", eU.rel() < scalar(5e-10));
        // `sst` HAS ITS OWN BOUNDS, and the reason is the device LOOP, not the closure. MEASURED with
        // every solve tightened on both codes -- which does not move them -- device against OpenFOAM:
        // k 5.2e-12, omega 3.1e-11, nut 1.9e-10, U 5.5e-11, where the HOST loop on the same case is
        // 6.7e-15, 1.9e-15, 1.3e-13 and 1.1e-13 and the device closure is the host closure's to 5e-15 /
        // 8e-16 / 3.7e-14 (the arm below). The device loop's U is about 5e-11 from OpenFOAM on the
        // kEpsilon profiles too; under kOmegaSST nut carries it through k/omega and F2, which is the
        // 1.9e-10. Bounds at about 30x the measurement.
        check("...its k", eK.rel() < (sstFloor ? scalar(2e-10) : scalar(1e-11)));
        check("...its second closure field", eE.rel() < (sstFloor ? scalar(1e-9) : scalar(1e-11)));
        check("...and its nut", eN.rel() < (sstFloor ? scalar(5e-9) : scalar(2e-11)));

        // THE SAME DEVICE LOOP WITH THE HOST CLOSURE IN THE DEVICE ONE'S PLACE -- the `_cpp` reference
        // as the in-repo oracle, with everything around it held fixed. Against OpenFOAM a disagreement
        // could be the loop's or the closure's; against this it can only be the closure's.
        setenv("BRAE_INTER_HOST_CLOSURE", "1", 1);
        InterFields mix;
        const RunReport rm = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &mix);
        unsetenv("BRAE_INTER_HOST_CLOSURE");
        check("the mixed run took the HOST closure", !rm.turbulenceOnDevice && rm.steps == nSteps);
        failures += brae::gatecheck::nonFinite("mixed k", mix.turbulence.k.internal);
        const std::vector<scalar>& mixSecond = (sst || frozenSST || splitSolveSST || splitDivSST) ? mix.turbulence.omega.internal
                                                   : mix.turbulence.epsilon.internal;
        failures += brae::gatecheck::nonFinite("mixed second field", mixSecond);
        failures += brae::gatecheck::nonFinite("mixed nut", mix.turbulence.nut.internal);
        failures += brae::gatecheck::nonFinite("mixed U", mix.U.internal);
        const Diff mU = compare(dev.U.internal, mix.U.internal);
        const Diff mK = compare(dev.turbulence.k.internal, mix.turbulence.k.internal);
        const Diff mE = compare(devSecond, mixSecond);
        const Diff mN = compare(dev.turbulence.nut.internal, mix.turbulence.nut.internal);
        std::printf("  DEVICE closure vs HOST closure, same device loop: U %.4e, k %.4e, %s %.4e, "
                    "nut %.4e\n", (double)mU.rel(), (double)mK.rel(), secondName, (double)mE.rel(),
                    (double)mN.rel());
        // MEASURED, worst of the three profiles: U 3.3e-14, k 1.1e-15, epsilon 1.6e-15, nut 1.9e-15
        // -- round-off between two implementations that share no kernel. Bounds at about 30x.
        // THE MODULE'S OWN CLAIM: the device closure is the host reference's. MEASURED under `sst` at the
        // case's own tolerances: U 8.6e-13, k 1.2e-13, omega 6.6e-14, nut 9.5e-13, and with every solve
        // tightened 1.4e-13 / 5.3e-15 / 7.8e-16 / 3.7e-14 -- round-off between two implementations that
        // share no kernel. kEpsilon's are 3.3e-14 / 1.1e-15 / 1.6e-15 / 1.9e-15. Bounds at about 30x.
        check("the device closure agrees with the host closure in U", mU.rel() < (sstFloor ? scalar(3e-11) : scalar(1e-12)));
        check("...in k", mK.rel() < (sstFloor ? scalar(5e-12) : scalar(5e-14)));
        check("...in the second closure field", mE.rel() < (sstFloor ? scalar(5e-12) : scalar(5e-14)));
        check("...and in nut", mN.rel() < (sstFloor ? scalar(5e-11) : scalar(5e-14)));
        // the two closures took the same sweeps, solve for solve
        std::size_t sameE = 0;
        std::size_t sameK = 0;
        const std::vector<LinearSolveRecord>& devSecondSolves = (sst || frozenSST || splitSolveSST || splitDivSST) ? rd.omegaSolves : rd.epsilonSolves;
        const std::vector<LinearSolveRecord>& mixSecondSolves = (sst || frozenSST || splitSolveSST || splitDivSST) ? rm.omegaSolves : rm.epsilonSolves;
        for (std::size_t q = 0; q < devSecondSolves.size() && q < mixSecondSolves.size(); ++q)
        {
            if (devSecondSolves[q].nIterations == mixSecondSolves[q].nIterations)
            {
                ++sameE;
            }
        }
        for (std::size_t q = 0; q < rd.kSolves.size() && q < rm.kSolves.size(); ++q)
        {
            if (rd.kSolves[q].nIterations == rm.kSolves[q].nIterations)
            {
                ++sameK;
            }
        }
        // `flowFrozen` reaches the same emptiness as `frozen`, by a different route: there the MODEL is
        // off, here the `continue` skips turbulence->correct() whole. Either way neither closure solves.
        check((frozen || flowFrozen) ? "...and BOTH closures took no sweep at all"
                     : "...and took the host closure's sweep counts, solve for solve",
              (frozen || flowFrozen) ? (devSecondSolves.empty() && mixSecondSolves.empty()
                        && rd.kSolves.empty() && rm.kSolves.empty())
                     // one per CORRECTOR the closure runs on, not one per step
                     : (sameE == static_cast<std::size_t>(nSteps*closurePerStep)
                        && sameK == static_cast<std::size_t>(nSteps*closurePerStep)));
    }

    std::printf("test_inter_ras_dambreak_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
