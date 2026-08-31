// deviceAdjustPhi against OpenFOAM's own two guards.
//
// OF adjustPhi.C:90-119 is not one clause but three:
//
//     scalar totalFlux = VSMALL + sum(mag(phi)).value();          // the WHOLE surface field
//     if (magAdjustableMassOut > VSMALL && magAdjustableMassOut/totalFlux > SMALL)
//         massCorr = (massIn - fixedMassOut)/adjustableMassOut;
//     else if (mag(fixedMassOut - massIn)/totalFlux > 1e-8)
//         FatalErrorInFunction << "Continuity error cannot be removed by adjusting the outflow..."
//
// brae's device kernel had only `if (fabs(adjOut) > 1e-300) massCorr = (massIn - fixedOut)/adjOut;`
// and no else. That differs from OpenFOAM twice:
//
//   1. OF's test is RELATIVE, against totalFlux. An absolute 1e-300 accepts an adjustable outflow that
//      is negligible beside the flux already in the domain and then DIVIDES BY IT -- precisely the
//      uninitialised-outflow case whose remedy OF's message spells out.
//   2. OF REFUSES on the other branch. brae carried on with massCorr = 1.0 and handed the pressure
//      equation a right-hand side that does not satisfy global continuity, which converges to something
//      plausible and wrong.
//
// Both HOST twins already implemented the full clause -- rhoPEqn_cpp.cu:99-122 and
// simpleFoam/pEqn_cpp.cu:92-125 -- so the device kernel was the one of the three that did not, while
// being the one all six device call sites reach (rhoPEqn.cu, rhoPcEqn.cu, simpleFoam/pEqn.cu and three
// in device_simple_foam.cu, i.e. the shipped incompressible solver as well as the OF-mirror one).
//
// NO FIXTURE IS NEEDED and none would be better. The guards are arithmetic on three flux sums, so the
// inputs can be constructed exactly -- including the cases a real mesh would only reach by accident.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_simple.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

static int failures = 0;

static void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-58s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

// One call, reporting whether it threw and what it returned. `adjOut` is placed on an adjustable face,
// `fixedOut` on a non-adjustable one, and `massIn` as a negative (inflow) face.
struct Result { bool threw = false; scalar massCorr = 0; std::string msg; };

static Result run(scalar massIn, scalar fixedOut, scalar adjOut, scalar internalFlux)
{
    // Three boundary faces: inflow, fixed outflow, adjustable outflow.
    const std::vector<scalar> phiB = { -massIn, fixedOut, adjOut };
    const std::vector<label>  adj  = { 0, 0, 1 };
    DeviceBuffer<scalar> dPhiB(phiB);
    DeviceBuffer<label>  dAdj(adj);
    // The internal half of totalFlux. OF sums mag(phi) over the whole surface field, so a domain with a
    // large interior flux makes a given adjustable outflow relatively smaller -- which is the entire
    // point of the relative guard and cannot be expressed without this term.
    DeviceBuffer<scalar> dPhiInt(std::vector<scalar>{ internalFlux });

    Result r;
    try { r.massCorr = deviceAdjustPhi(dAdj, dPhiB, &dPhiInt); }
    catch (const std::exception& e) { r.threw = true; r.msg = e.what(); }
    return r;
}

int main()
{
    std::printf("deviceAdjustPhi: OpenFOAM's two guards (adjustPhi.C:90-119)\n");

    // ---- 1. THE ORDINARY CASE still behaves, and is the negative control for everything below. ----
    // massIn 3, fixedOut 0.5, adjOut 2 against an interior flux of 10: the adjustable outflow is a
    // healthy fraction of totalFlux, so OF takes the first branch.
    //
    // (3 - 0.5)/2 = 1.25, deliberately NOT 1. massCorr's default when neither branch fires is exactly
    // 1.0, so a case whose answer happens to be 1 cannot distinguish "computed it" from "left it alone" --
    // the first version of this check used 2/0.5/1.5, which is 1.0, and would have passed against a
    // kernel that did no arithmetic at all.
    {
        const Result r = run(3.0, 0.5, 2.0, 10.0);
        const double want = (3.0 - 0.5) / 2.0;
        check("a well-posed case does NOT throw (negative control)", !r.threw);
        std::printf("       %-56s %.12g (want %.12g)\n", "(massCorr)", (double)r.massCorr, want);
        check("...and returns (massIn - fixedMassOut)/adjustableMassOut",
              std::fabs((double)r.massCorr - want) < 1e-12);
        check("...a value that is NOT the untouched default of 1 (non-vacuous)",
              std::fabs(want - 1.0) > 0.1);
    }

    // ---- 2. THE REFUSAL: no adjustable outflow, and a continuity error that needs one. ----
    // massIn 2, fixedOut 0.5, adjOut 0. OF's first branch fails on magAdjustableMassOut > VSMALL, and
    // |fixedMassOut - massIn|/totalFlux = 1.5/12.5 is far above 1e-8, so OF exits fatally.
    // brae previously returned massCorr = 1.0 and scaled nothing, silently.
    {
        const Result r = run(2.0, 0.5, 0.0, 10.0);
        check("no adjustable outflow + a real continuity error REFUSES", r.threw);
        check("...and the refusal names the OpenFOAM remedy",
              r.msg.find("potentialFoam") != std::string::npos);
        check("...and reports the four numbers OpenFOAM reports",
              r.msg.find("total flux") != std::string::npos
           && r.msg.find("adjustable mass outflow") != std::string::npos);
    }

    // ---- 3. THE RELATIVE GUARD, which is the clause an absolute test cannot express. ----
    // adjOut = 1e-20 is comfortably above the old `fabs(adjOut) > 1e-300`, so the previous code took the
    // first branch and computed massCorr = 1.5/1e-20 = 1.5e+20 -- scaling a vanishing outflow by twenty
    // orders of magnitude. OF's magAdjustableMassOut/totalFlux > SMALL (1e-15) rejects it: 1e-20/12.5 is
    // 8e-22. This is the case the two forms disagree on, and it is why the test is relative.
    {
        const Result r = run(2.0, 0.5, 1.0e-20, 10.0);
        check("a NEGLIGIBLE adjustable outflow is refused, not divided by", r.threw);
        if (!r.threw)
            std::printf("       %-56s %.6g\n", "(massCorr the absolute test would have produced)",
                        (double)r.massCorr);
    }

    // ---- 4. NO adjustable outflow and NO continuity error: OF does neither, and nor must brae. ----
    // massIn == fixedMassOut, so the second branch's test is 0 > 1e-8, false. OF falls through with
    // massCorr = 1.0. A refusal here would be a false positive on a perfectly balanced case.
    {
        const Result r = run(2.0, 2.0, 0.0, 10.0);
        check("a BALANCED case with no adjustable outflow does not throw", !r.threw);
        check("...and leaves massCorr at 1", !r.threw && std::fabs((double)r.massCorr - 1.0) < 1e-15);
    }

    // ---- 5. THE NORMALISER IS THE WHOLE FIELD, not the boundary slice. ----
    // Same three boundary faces both times; only the interior flux differs. With a small interior the
    // adjustable outflow is a large fraction of totalFlux and OF adjusts; with a large one it is below
    // SMALL and OF refuses. A kernel normalising on the boundary alone -- which is all deviceAdjustPhi
    // is handed unless the caller passes the internal flux -- cannot tell these two apart.
    {
        const Result small = run(2.0, 0.5, 1.0e-13, 1.0e-3);
        const Result large = run(2.0, 0.5, 1.0e-13, 1.0e+6);
        std::printf("       %-56s small interior: %s, large interior: %s\n",
                    "(same boundary, different totalFlux)",
                    small.threw ? "refused" : "adjusted", large.threw ? "refused" : "adjusted");
        check("totalFlux includes the INTERNAL faces (the two cases differ)",
              small.threw != large.threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
