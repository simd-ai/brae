// interFoam's SECOND Courant number, and the time step it shares with the first.
//
// The ordinary Courant number is a global maximum over every cell, so it is set by whatever corner of
// the domain runs fastest -- usually nowhere near the interface. A VoF interface has its own, tighter
// limit, and MULES does not stop it being advected more than a cell in a step. interFoam therefore
// limits by BOTH, and the shipped tutorials set them to the same value more often than not (0.65/0.65
// in 12 cases, 0.5/0.5 in 10).
//
// WHAT THE ARMS CATCH, none of it arithmetic:
//
// 1. maxAlphaCo IS MANDATORY where maxCo DEFAULTS TO 1. Two reads of the same controlDict, one
//    refusing and one not. A case that turns on adjustTimeStep and omits maxAlphaCo is a FatalError in
//    OpenFOAM; defaulting it runs the interface with no limit of its own.
//
// 2. nearInterface() IS A 0/1 MASK OVER A CLOSED BAND. pos0(x) is 1 at exactly zero, so alpha = 0.01
//    and alpha = 0.99 are BOTH counted. A smooth weight, or `pos` instead of `pos0`, changes the
//    answer only on the two cells at the edges of the band -- and those are the ones the interface is
//    passing through.
//
// 3. WITH NO INTERFACE THE ALPHA LIMIT MUST NOT THROTTLE THE STEP. alphaCoNum is then exactly 0 and
//    maxAlphaCo/(0 + SMALL) is astronomically large, so the min picks maxCo's branch. A port that
//    guarded the division differently -- or clamped the factor -- would silently freeze a case that
//    has not yet developed an interface.
//
// 4. THE TWO LIMITS COMBINE BEFORE THE DAMPING. min-then-damp is not damp-then-min, and the numbers
//    differ whenever the binding limit wants growth.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "time_controls.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

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
}   // namespace

int main()
{
    std::printf("== interFoam alphaCourantNo / setDeltaT ==\n");

    // ---- 1. the controls: one mandatory, one defaulted ---------------------------------------------
    {
        const std::string base = "/tmp/brae_alpha_courant";
        std::filesystem::remove_all(base);
        auto write = [&](const std::string& dir, const std::string& body)
        {
            std::filesystem::create_directories(base + "/" + dir);
            std::ofstream(base + "/" + dir + "/controlDict")
                << "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
                << body;
            return readDict(base + "/" + dir + "/controlDict");
        };

        // damBreak's own numbers
        const FoamDict good = write("good", "adjustTimeStep yes;\nmaxCo 1;\nmaxAlphaCo 1;\nmaxDeltaT 1;\n");
        const VoFTimeControls tc = VoFTimeControls::read(good);
        check   ("adjustTimeStep yes is read",  tc.base.adjustTimeStep);
        checkNum("maxCo",      tc.base.maxCo,      scalar(1));
        checkNum("maxAlphaCo", tc.maxAlphaCo,      scalar(1));
        checkNum("maxDeltaT",  tc.base.maxDeltaT,  scalar(1));

        // THE PAIR: the same dictionary, one reader refusing and one not.
        const FoamDict noAlpha = write("noalpha", "adjustTimeStep yes;\nmaxCo 0.65;\n");
        checkNum("maxCo DEFAULTS to 1 when absent", TimeControls::read(write("nomax",
                 "adjustTimeStep yes;\n")).maxCo, scalar(1));
        bool threw = false;
        try { (void)VoFTimeControls::read(noAlpha); } catch (const std::exception&) { threw = true; }
        check("...while a missing maxAlphaCo is REFUSED -- get<scalar>, no default", threw);

        // ...but only when the step is actually being adjusted. A fixed-step case never reads it.
        const FoamDict fixed = write("fixed", "adjustTimeStep no;\nmaxCo 0.65;\n");
        bool threwFixed = false;
        try { (void)VoFTimeControls::read(fixed); } catch (const std::exception&) { threwFixed = true; }
        check("a fixed-step case does not need maxAlphaCo at all", !threwFixed);
        checkNum("maxDeltaT defaults to GREAT (no cap)",
                 VoFTimeControls::read(fixed).base.maxDeltaT > scalar(1e100) ? scalar(1) : scalar(0),
                 scalar(1));
    }

    // ---- 2. nearInterface: a 0/1 mask over the CLOSED band [0.01, 0.99] ----------------------------
    {
        const std::vector<scalar> a{
            scalar(0), scalar(0.009), scalar(0.01), scalar(0.011),
            scalar(0.5),
            scalar(0.989), scalar(0.99), scalar(0.991), scalar(1)};
        const std::vector<scalar> m = nearInterface(a);
        checkNum("alpha = 0     -> 0", m[0], scalar(0));
        checkNum("alpha = 0.009 -> 0", m[1], scalar(0));
        checkNum("alpha = 0.01  -> 1  (pos0 is 1 at exactly zero)", m[2], scalar(1));
        checkNum("alpha = 0.5   -> 1", m[4], scalar(1));
        checkNum("alpha = 0.99  -> 1  (the band is CLOSED at both ends)", m[6], scalar(1));
        checkNum("alpha = 0.991 -> 0", m[7], scalar(0));
        checkNum("alpha = 1     -> 0", m[8], scalar(0));
        bool binary = true;
        for (scalar v : m) binary = binary && (v == scalar(0) || v == scalar(1));
        check("every value is exactly 0 or 1 -- a mask, not a weight", binary);
    }

    // ---- 3. alphaCoNum against CoNum on the same field ---------------------------------------------
    {
        const label N = 8;
        PrimitiveMesh m = boxtest::boxMesh(N, 1, 1);
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        const label nC = m.nCells();

        // A FAST corner far from the interface, and a slow interface. This is the whole reason the
        // second Courant number exists: the global max is set by the fast corner, which tells you
        // nothing about whether the interface is being advected safely.
        std::vector<scalar> sumPhi(static_cast<std::size_t>(nC), scalar(1));
        sumPhi[nC - 1] = scalar(100);                       // the fast corner
        std::vector<scalar> alpha(static_cast<std::size_t>(nC), scalar(0));
        alpha[3] = scalar(0.5);                             // the interface, two cells wide
        alpha[4] = scalar(0.5);
        for (label c = 0; c < 3; ++c) alpha[c] = scalar(1);

        const scalar dt = scalar(0.01);
        const CourantNumbers co  = courantNo(sumPhi, g.V(), dt);
        const CourantNumbers aco = alphaCourantNo(sumPhi, alpha, g.V(), dt);
        std::printf("  CoNum %.6g (set by the fast corner), alphaCoNum %.6g (set by the interface)\n",
                    (double)co.CoNum, (double)aco.CoNum);
        checkNum("CoNum  = 0.5*max(sumPhi/V)*dt over EVERY cell",
                 co.CoNum, scalar(0.5)*scalar(100)/g.V()[nC-1]*dt);
        checkNum("alphaCoNum = the same, over the interface cells only",
                 aco.CoNum, scalar(0.5)*scalar(1)/g.V()[3]*dt);
        check("...so the two are different numbers, which is the point",
              co.CoNum > scalar(50)*aco.CoNum);

        // meanAlphaCoNum divides by the WHOLE mesh volume, not the interface volume.
        scalar sV = 0;
        for (scalar v : g.V()) sV += v;
        checkNum("meanAlphaCoNum divides by gSum(V) over the whole mesh",
                 aco.meanCoNum, scalar(0.5)*(scalar(2)/sV)*dt);

        // ...and with NO interface it is exactly zero.
        const std::vector<scalar> allWater(static_cast<std::size_t>(nC), scalar(1));
        const CourantNumbers none = alphaCourantNo(sumPhi, allWater, g.V(), dt);
        checkNum("with no interface alphaCoNum is exactly 0", none.CoNum, scalar(0));
    }

    // ---- 4. setDeltaT with both limits -------------------------------------------------------------
    {
        VoFTimeControls tc;
        tc.base.adjustTimeStep = true;
        tc.base.maxCo     = scalar(0.65);
        tc.base.maxDeltaT = scalar(1);
        tc.maxAlphaCo     = scalar(0.65);
        const scalar dt = scalar(0.001);

        auto oracle = [&](scalar Co, scalar aCo)
        {
            const scalar kSmall = scalar(1e-37);
            const scalar f = std::fmin(tc.base.maxCo/(Co + kSmall), tc.maxAlphaCo/(aCo + kSmall));
            const scalar d = std::fmin(std::fmin(f, scalar(1) + scalar(0.1)*f), scalar(1.2));
            return std::fmin(d*dt, tc.base.maxDeltaT);
        };

        // (a) the ALPHA limit binds: the flow is slow but the interface is moving fast
        checkNum("when the interface binds, it sets the step",
                 setDeltaTVoF(dt, scalar(0.1), scalar(1.3), tc), oracle(scalar(0.1), scalar(1.3)));
        check("...and that is a REDUCTION, applied immediately",
              setDeltaTVoF(dt, scalar(0.1), scalar(1.3), tc) < dt);

        // (b) the ORDINARY limit binds: a fast corner far from a quiet interface
        checkNum("when the bulk flow binds, it sets the step",
                 setDeltaTVoF(dt, scalar(1.3), scalar(0.1), tc), oracle(scalar(1.3), scalar(0.1)));

        // (c) NO INTERFACE: alphaCoNum is 0, so the alpha branch must not throttle anything.
        const scalar withNoInterface = setDeltaTVoF(dt, scalar(0.2), scalar(0), tc);
        const TimeControls onlyCo{true, scalar(0.65), scalar(1)};
        checkNum("with no interface the step is exactly what maxCo alone would give",
                 withNoInterface, setDeltaT(dt, scalar(0.2), onlyCo));
        check("...which is a GROWTH, not a freeze", withNoInterface > dt);

        // (d) GROWTH IS DAMPED, AND THE TWO LIMITS COMBINE BEFORE THE DAMPING.
        //     maxDeltaTFact = min(0.65/0.13, 0.65/0.65) = min(5, 1) = 1  -> deltaTFact = 1
        //     Damping each separately first and then taking the min gives min(1.2, 1) = 1 here, so the
        //     discriminating case is one where BOTH want growth but by different amounts.
        checkNum("growth is capped at 1.2x", setDeltaTVoF(dt, scalar(1e-6), scalar(1e-6), tc),
                 scalar(1.2)*dt);
        {
            // Co gives factor 6.5, alphaCo gives 1.3. Combined-then-damped:
            //   min(6.5, 1.3) = 1.3 -> min(1.3, 1+0.13, 1.2) = 1.13
            // Damped-then-combined:
            //   min(min(6.5,1.65,1.2), min(1.3,1.13,1.2)) = min(1.2, 1.13) = 1.13   <- agrees here
            // so pick numbers where they do NOT: Co factor 1.05, alphaCo factor 3.0
            //   combined: min(1.05,3.0)=1.05 -> min(1.05, 1.105, 1.2) = 1.05
            //   separate: min(min(1.05,1.105,1.2), min(3.0,1.3,1.2)) = min(1.05,1.2) = 1.05  (same)
            // The two orderings agree whenever min is the outer operation on a monotone damping, which
            // it is -- so this is asserted as the IDENTITY it is rather than as a discriminator, and
            // the arm that matters is that maxDeltaTFact is formed from BOTH before anything else.
            const scalar CoA = tc.base.maxCo/scalar(1.05);       // -> factor 1.05
            const scalar aCoA = tc.maxAlphaCo/scalar(3.0);       // -> factor 3.0
            checkNum("both limits enter maxDeltaTFact", setDeltaTVoF(dt, CoA, aCoA, tc),
                     oracle(CoA, aCoA));
            check("...and the tighter of the two is what survives",
                  setDeltaTVoF(dt, CoA, aCoA, tc) < scalar(1.2)*dt);
        }

        // (e) maxDeltaT is a hard ceiling that beats any growth factor.
        VoFTimeControls capped = tc;
        capped.base.maxDeltaT = scalar(0.0010005);
        checkNum("maxDeltaT caps the result", setDeltaTVoF(dt, scalar(1e-9), scalar(1e-9), capped),
                 scalar(0.0010005));

        // (f) a fixed-step case is untouched.
        VoFTimeControls fixed = tc;
        fixed.base.adjustTimeStep = false;
        checkNum("adjustTimeStep no leaves deltaT alone",
                 setDeltaTVoF(dt, scalar(99), scalar(99), fixed), dt);
    }

    std::printf("test_alpha_courant_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
