// interFoam's time step: the alpha sub-cycle, and the order of the PIMPLE loop.
//
// TWO THINGS, and they fail differently.
//
// THE SUB-CYCLE is numerics. 33 of the 44 shipped tutorials use it (23 at nAlphaSubCycles 3), and the
// three ways to get it wrong all leave a bounded, plausible alpha field: running the sub-steps at the
// full deltaT, restarting each from the time-step's old value instead of the previous sub-step's, and
// taking the last rhoPhi instead of the time-weighted sum. MULES keeps the field in [0,1] through all
// three, so no boundedness gate sees any of them. The oracle here is a sub-step function whose exact
// behaviour is known -- a pure translation -- so each can be measured rather than argued about.
//
// THE LOOP ORDER is not numerics at all: it is which field each equation sees. alpha is advanced
// BEFORE the momentum predictor, so UEqn is built on the new rho and the new surface tension. Running
// UEqn first is what every single-phase solver does and it converges perfectly well -- to a momentum
// equation carrying last step's density, which at a water/air interface is wrong by 1000. The driver
// takes its stages as callbacks so the order is executable, and this file records them.
#include "inter_solve_cpp.cuh"
#include <cmath>
#include <cstdio>
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

void checkNum(const char* what, scalar got, scalar want, scalar tol = scalar(1e-12))
{
    const bool ok = std::fabs(got - want) <= tol * std::fmax(scalar(1), std::fabs(want));
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s (got %.17g want %.17g)\n", what, (double)got, (double)want);
    if (!ok) ++failures;
}

std::string join(const std::vector<Stage>& v)
{
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) { if (i) s += " -> "; s += stageName(v[i]); }
    return s;
}
}   // namespace

int main()
{
    std::printf("== interFoam time step ==\n");

    // ---- 1. the sub-cycle, against a sub-step whose behaviour is exactly known --------------------
    // The oracle: alpha advances by dtSub*rate from whatever it is GIVEN, and that sub-step's rhoPhi
    // is its own index. A pure translation, so the three quantities under test -- the step size, the
    // field each sub-step starts from, and the weighting of rhoPhi -- are each readable in the answer.
    {
        const scalar rate = scalar(2);
        const scalar dt   = scalar(0.6);
        const std::vector<scalar> alphaOld{scalar(0.1), scalar(0.4)};

        label calls = 0;
        std::vector<scalar> seenDt;
        std::vector<scalar> seenStart;
        auto step = [&](const std::vector<scalar>& in, scalar dtSub,
                        std::vector<scalar>& out, SurfaceScalarField& rhoPhi)
        {
            seenDt.push_back(dtSub);
            seenStart.push_back(in[0]);
            out.resize(in.size());
            for (std::size_t c = 0; c < in.size(); ++c) out[c] = in[c] + dtSub*rate;
            rhoPhi.internal.assign(2, static_cast<scalar>(calls + 1));   // 1, 2, 3, ... per sub-step
            rhoPhi.boundary.assign(1, std::vector<scalar>(1, static_cast<scalar>(calls + 1)));
            ++calls;
        };

        // (a) nAlphaSubCycles = 3
        {
            calls = 0; seenDt.clear(); seenStart.clear();
            std::vector<scalar> alpha;
            SurfaceScalarField rhoPhi;
            alphaEqnSubCycle(3, dt, alpha, alphaOld, rhoPhi, step);

            checkNum("three sub-cycles call alphaEqn three times", scalar(calls), scalar(3));
            for (scalar d : seenDt)
                checkNum("...each at deltaT/nAlphaSubCycles", d, dt/scalar(3));
            checkNum("the FIRST sub-step starts from the time-step's old alpha", seenStart[0], alphaOld[0]);
            checkNum("the second starts from the first's RESULT",
                     seenStart[1], alphaOld[0] + (dt/scalar(3))*rate);
            checkNum("...and the third from the second's",
                     seenStart[2], alphaOld[0] + scalar(2)*(dt/scalar(3))*rate);
            checkNum("so alpha has advanced by the FULL deltaT in total",
                     alpha[0], alphaOld[0] + dt*rate);

            // rhoPhi = sum (dt_k/dt_total)*rhoPhi_k = (1 + 2 + 3)/3 = 2
            checkNum("rhoPhi is the TIME-WEIGHTED sum over the sub-steps", rhoPhi.internal[0], scalar(2));
            check("...which is NOT the last sub-step's value", std::fabs(rhoPhi.internal[0] - scalar(3)) > scalar(0.5));
            check("...and NOT the plain sum either", std::fabs(rhoPhi.internal[0] - scalar(6)) > scalar(0.5));
            checkNum("the boundary is weighted the same way", rhoPhi.boundary[0][0], scalar(2));

            // THE RESTORE: alpha1Old must be exactly as it was, or the PIMPLE outer loop's next pass
            // would advance from this pass's result and move the field two steps in one.
            checkNum("alpha.oldTime() is untouched, so the next outer corrector restarts from it",
                     alphaOld[0], scalar(0.1));
        }

        // (b) THE CONTROLS: what each of the three mistakes would have produced. Computed here so the
        //     file shows the wrong answers rather than asserting that they are wrong.
        {
            const scalar fullStepPerSub = alphaOld[0] + scalar(3)*dt*rate;   // every sub-step at dt
            const scalar allFromOld     = alphaOld[0] + (dt/scalar(3))*rate; // never carried forward
            std::printf("  had each sub-step run at the full deltaT: alpha = %.4f (correct: %.4f)\n",
                        (double)fullStepPerSub, (double)(alphaOld[0] + dt*rate));
            std::printf("  had every sub-step restarted from the old alpha: alpha = %.4f\n",
                        (double)allFromOld);
            check("running at the full deltaT would advance 3x too far",
                  std::fabs(fullStepPerSub - (alphaOld[0] + dt*rate)) > scalar(1));
            check("restarting from the old alpha would advance 3x too little",
                  std::fabs(allFromOld - (alphaOld[0] + dt*rate)) > scalar(0.5));
        }

        // (c) nAlphaSubCycles = 1 -- the plain branch, and 11 of the 44 tutorials take it.
        {
            calls = 0; seenDt.clear(); seenStart.clear();
            std::vector<scalar> alpha;
            SurfaceScalarField rhoPhi;
            alphaEqnSubCycle(1, dt, alpha, alphaOld, rhoPhi, step);
            checkNum("nAlphaSubCycles 1 calls alphaEqn once", scalar(calls), scalar(1));
            checkNum("...at the FULL deltaT, undivided", seenDt[0], dt);
            checkNum("...and rhoPhi is that call's own, unweighted", rhoPhi.internal[0], scalar(1));
        }

        // (d) refusals
        {
            std::vector<scalar> alpha;
            SurfaceScalarField rhoPhi;
            bool threw = false;
            try { alphaEqnSubCycle(0, dt, alpha, alphaOld, rhoPhi, step); }
            catch (const std::exception&) { threw = true; }
            check("nAlphaSubCycles 0 is refused", threw);
            threw = false;
            try { alphaEqnSubCycle(3, scalar(0), alpha, alphaOld, rhoPhi, step); }
            catch (const std::exception&) { threw = true; }
            check("a zero time step is refused", threw);
        }
    }

    // ---- 2. the order of the PIMPLE loop -----------------------------------------------------------
    {
        auto record = [](const LoopControls& ctl)
        {
            std::vector<Stage> seen;
            SolverHooks h;
            h.run = [&](Stage s) { seen.push_back(s); };
            runTimeStep(ctl, h);
            return seen;
        };

        // (a) the ordinary PISO-mode case: one outer corrector, two pressure correctors.
        {
            LoopControls ctl;
            ctl.nOuterCorrectors = 1;
            ctl.nCorrectors      = 2;
            const std::vector<Stage> seen = record(ctl);
            const std::vector<Stage> want{
                Stage::courantNo, Stage::alphaCourantNo, Stage::setDeltaT, Stage::advanceTime,
                Stage::meshUpdate, Stage::alphaControls, Stage::alphaEqnSubCycle, Stage::mixtureCorrect,
                Stage::UEqn, Stage::pEqn, Stage::pEqn, Stage::turbulenceCorrect,
                Stage::write};
            std::printf("  %s\n", join(seen).c_str());
            check("the stages run in interFoam.C's order", seen == want);
        }

        // (b) THE ORDER THAT MATTERS: alpha moves BEFORE the momentum predictor, and the mixture is
        //     refreshed in between. UEqn must see the new rho, not last step's.
        {
            LoopControls ctl;
            const std::vector<Stage> seen = record(ctl);
            auto at = [&](Stage s)
            {
                for (std::size_t i = 0; i < seen.size(); ++i) if (seen[i] == s) return (int)i;
                return -1;
            };
            check("alphaEqnSubCycle runs BEFORE UEqn", at(Stage::alphaEqnSubCycle) < at(Stage::UEqn));
            check("mixture.correct runs between them, so UEqn is built on the NEW rho",
                  at(Stage::alphaEqnSubCycle) < at(Stage::mixtureCorrect)
               && at(Stage::mixtureCorrect)   < at(Stage::UEqn));
            check("UEqn runs before pEqn",  at(Stage::UEqn) < at(Stage::pEqn));
            check("the step is chosen BEFORE the time advances",
                  at(Stage::setDeltaT) < at(Stage::advanceTime));
            check("...and both Courant numbers are computed before it",
                  at(Stage::courantNo)      < at(Stage::setDeltaT)
               && at(Stage::alphaCourantNo) < at(Stage::setDeltaT));
        }

        // (c) turbOnFinalIterOnly: the closure advances ONCE per time step, not once per outer
        //     corrector. Defaults to true in OpenFOAM; with nOuterCorrectors 3 the difference is 3x.
        {
            LoopControls ctl;
            ctl.nOuterCorrectors = 3;
            auto count = [](const std::vector<Stage>& v, Stage s)
            {
                int n = 0; for (Stage t : v) if (t == s) ++n; return n;
            };
            const std::vector<Stage> onFinal = record(ctl);
            checkNum("alphaEqnSubCycle runs once per OUTER corrector",
                     scalar(count(onFinal, Stage::alphaEqnSubCycle)), scalar(3));
            checkNum("turbulence.correct runs ONCE per time step (turbOnFinalIterOnly)",
                     scalar(count(onFinal, Stage::turbulenceCorrect)), scalar(1));
            check("...and it is the LAST outer corrector that runs it",
                  onFinal[onFinal.size() - 2] == Stage::turbulenceCorrect);

            ctl.turbOnFinalIterOnly = false;
            const std::vector<Stage> everyIter = record(ctl);
            checkNum("...while turbOnFinalIterOnly no advances it every outer corrector (control)",
                     scalar(count(everyIter, Stage::turbulenceCorrect)), scalar(3));
        }

        // (d) frozenFlow: alpha keeps advancing on a fixed velocity field. `continue` skips the
        //     turbulence corrector too, which a port that only guarded UEqn and pEqn would keep
        //     running -- advancing a closure whose velocity field is not moving.
        {
            LoopControls ctl;
            ctl.frozenFlow = true;
            ctl.nOuterCorrectors = 2;
            const std::vector<Stage> seen = record(ctl);
            std::printf("  frozenFlow: %s\n", join(seen).c_str());
            bool anyMomentum = false;
            for (Stage s : seen)
                anyMomentum = anyMomentum || s == Stage::UEqn || s == Stage::pEqn
                                          || s == Stage::turbulenceCorrect;
            check("frozenFlow skips UEqn, pEqn AND the turbulence corrector", !anyMomentum);
            int nAlpha = 0;
            for (Stage s : seen) if (s == Stage::alphaEqnSubCycle) ++nAlpha;
            checkNum("...while alpha still advances, once per outer corrector",
                     scalar(nAlpha), scalar(2));
        }

        // (e) refusals
        {
            LoopControls ctl; ctl.nOuterCorrectors = 0;
            bool threw = false;
            try { (void)record(ctl); } catch (const std::exception&) { threw = true; }
            check("nOuterCorrectors 0 is refused", threw);
            SolverHooks empty;
            threw = false;
            try { runTimeStep(LoopControls{}, empty); } catch (const std::exception&) { threw = true; }
            check("a driver with no stage hook is refused", threw);
        }
    }

    std::printf("test_inter_solve_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
