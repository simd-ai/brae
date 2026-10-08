// meanVelocityForce: the fvOptions source that DRIVES a periodic channel, and the only thing holding
// its flow rate at the prescribed Ubar.
//
// It is a closed-loop controller, not a source term you can check by reading it once. Each call measures
// the current volume-averaged velocity, computes the pressure-gradient increment that would close the
// gap, applies it to U directly, and banks it so the next momentum solve carries it. Get any part of
// that loop wrong and it does not fail -- it oscillates, and a controller whose gain is effectively
// doubled oscillates with GROWING amplitude. LES/periodicPlaneChannel came out at a mean velocity of
// 1.476 against a prescribed Ubar of 0.1335 and OpenFOAM's 0.129: eleven times too fast, L2rel(U) of
// 1.00e+01, in a case that ran to completion and wrote plausible-looking fields.
//
// TWO INDEPENDENT DEFECTS, and each alone was enough to break it. Both are structural, and neither is
// visible in a single call:
//
//   1. ONE ACCUMULATOR INSTEAD OF TWO. OF keeps gradP0_ (banked) and dGradP_ (the pending increment).
//      correct() ASSIGNS dGradP_, so a second call against the same matrix REPLACES the first rather
//      than adding to it, and only constrain() -- once per momentum-matrix build -- folds it into
//      gradP0_ and zeroes it (meanVelocityForce.C:170, 246-247). brae did gradP += dGradP on every
//      call. OF's own call pattern is two correct(U) per outer corrector, so brae banked two increments
//      where OF banks one: the drive compounded faster than the flow could respond.
//
//   2. THE CORRECTION RAN BEFORE HbyA. OF's last fvOptions.correct(U) is the tail of pEqn.H, after
//      U = HbyA - rAU*grad(p). brae ran it before H()/HbyA was formed, so the corrector immediately
//      overwrote it and the step ended wherever the pressure solve left the mean -- never at Ubar. The
//      source then kept driving to close a gap the correction had already closed and lost.
//
// SO THE TEST IS A CLOSED-LOOP TEST. Leg 2 is the load-bearing one: it does not check that the mean is
// right once, it checks that it STAYS right and that the excursion SHRINKS. Suppressing the velocity
// correction -- which is what defect 2 amounted to, since the corrector overwrote it -- sends the mean
// to 238 against Ubar = 0.5, and Leg 2's excursion goes 1.4e-03 -> 4.2e+02.
//
// WHAT THIS FIXTURE DOES NOT COVER, stated plainly rather than implied. It pins defect 2. It does NOT
// discriminate defect 1: reinstating the accumulate leaves every leg passing, and not because the legs
// are weak. In a laminar duct that settles, each correction closes the gap exactly, so dGradP -> 0 and
// a doubled bank has nothing left to double. The defect needs a flow whose mean keeps moving between
// corrections -- LES/periodicPlaneChannel, where dGradP never reaches zero and the drive stays
// permanently twice what the flow can absorb. That case is pinned in the sweep gate
// (validation/pimplefoam/verdict.py), which is the test that covers defect 1; measured there, the
// accumulate alone accounts for 1.00e+01 -> 1.91e+00 and the position for 1.91e+00 -> 5.02e-02.
#include "box_mesh.cuh"
#include "device_simple_foam.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "geometric_field.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

const scalar UBAR = scalar(0.5);

// A streamwise-periodic duct: cyclic in x, walls top and bottom. Nothing drives it except the source,
// which is the point -- with the source removed wall drag can only slow it down (Leg 3). nu is set high
// enough that the drag is substantial over the run, so the controller has real work to do.
PrimitiveMesh channel()
{
    PrimitiveMesh base = boxtest::boxMesh(8, 6, 3);
    std::vector<PatchInfo> pp = base.patches();
    for (PatchInfo& q : pp)
    {
        if (q.name == "inlet")  { q.type = "cyclic"; q.neighbourPatch = "outlet"; q.transform = "unknown"; }
        if (q.name == "outlet") { q.type = "cyclic"; q.neighbourPatch = "inlet";  q.transform = "unknown"; }
    }
    PrimitiveMesh m;
    m.assign(base.points(), base.faceVerts(), base.faceOffsets(), base.owner(), base.neighbour(),
             std::move(pp), base.nCells());
    return m;
}

struct Run { std::vector<scalar> meanUx; bool finite = true; };

// March the channel and record the volume-averaged streamwise velocity after every step.
Run march(bool withSource, int nSteps)
{
    const PrimitiveMesh m = channel();
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    FieldData<vector> Ufd;
    Ufd.internalUniform = true;
    Ufd.internalUniformValue = vector{UBAR, 0, 0};   // start AT Ubar: any drift is the controller's doing
    FieldData<scalar> pfd;
    pfd.internalUniform = true; pfd.internalUniformValue = 0;
    for (const FvPatch& q : fvp)
    {
        PatchFieldData<vector> bu; bu.name = q.name; bu.type = (q.type == "cyclic") ? "cyclic" : "fixedValue";
        bu.hasValue = true; bu.valueUniform = true; bu.uniformValue = vector{0,0,0};
        Ufd.boundary.push_back(bu);
        PatchFieldData<scalar> bp; bp.name = q.name; bp.type = (q.type == "cyclic") ? "cyclic" : "zeroGradient";
        bp.hasValue = true; bp.valueUniform = true; bp.uniformValue = 0;
        pfd.boundary.push_back(bp);
    }

    DeviceSimpleControls c;
    c.nu = 0.2; c.relaxU = 0.7; c.relaxP = 0.3; c.tolU = 1e-8; c.tolP = 1e-8; c.turbulent = false;
    c.pRefCell = 0; c.pRefValue = 0;

    GeometricField<vector> U = buildField<vector>(Ufd, fvp, nC); U.evaluateBoundary();
    GeometricField<scalar> p = buildField<scalar>(pfd, fvp, nC); p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
    DeviceSimpleSolver s(m, g, fvp, U, p, phi, c);
    s.setDdtScheme(DdtScheme::Euler);

    if (withSource)
    {
        FvOptionsData fo;
        fo.mvfActive = true;
        fo.mvfRelax  = 1.0;
        fo.mvfUbar   = vector{UBAR, 0, 0};
        s.setFvOptions(fo);                          // selectionMode all -> empty mvfCells
    }

    Run r;
    const std::vector<scalar>& V = g.V();
    scalar vtot = 0;
    for (label ci = 0; ci < nC; ++ci) vtot += V[ci];
    for (int it = 0; it < nSteps; ++it)
    {
        s.pimpleStep(/*deltaT*/1e-2, /*nOuter*/1, /*nCorr*/2);
        const std::vector<vector> u = s.U();
        scalar acc = 0;
        for (label ci = 0; ci < nC; ++ci)
        {
            if (!std::isfinite(u[ci].x)) r.finite = false;
            acc += u[ci].x*V[ci];
        }
        r.meanUx.push_back(acc/vtot);
    }
    return r;
}
} // namespace

// this binary again as a child with the control set: the undriven mean its `child` line prints. A child that
// never said it ran the control has not run it, and comes back as a NaN.
scalar childUndrivenMean(
    const std::string& self,
    const std::string& envSwitch)
{
    const std::string cmd = envSwitch + "=1 '" + self + "' child 2>&1";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p)
    {
        return std::nan("");
    }
    char line[2048];
    scalar mean = std::nan("");
    bool controlSaid = false;
    while (std::fgets(line, sizeof(line), p))
    {
        const std::string l(line);
        controlSaid = controlSaid || l.find("CONTROL MODE (" + envSwitch + ")") != std::string::npos;
        double v = 0;
        if (std::sscanf(line, "CHILD undrivenMean=%lf", &v) == 1)
        {
            mean = static_cast<scalar>(v);
        }
    }
    pclose(p);
    return controlSaid ? mean : std::nan("");
}

int main(
    int argc,
    char** argv)
{
    const int N = 50;
    if (argc > 1 && std::string(argv[1]) == "child")
    {
        const Run alone = march(/*withSource*/false, N);
        std::printf("CHILD undrivenMean=%.17g\n", (double)alone.meanUx.back());
        return 0;
    }
    std::printf("== meanVelocityForce holds the prescribed flow rate ==\n");

    const Run on = march(/*withSource*/true, N);
    const Run off = march(/*withSource*/false, N);

    // ---- Leg 1: it reaches Ubar at all ---------------------------------------------------------------
    {
        check(on.finite, "the driven channel stays finite");
        const scalar last = on.meanUx.back();
        check(std::fabs(last - UBAR) < scalar(0.02)*UBAR,
              "after the run the mean streamwise velocity is at Ubar");
        std::printf("        (Ubar %.4f, reached %.6f)\n", (double)UBAR, (double)last);
    }

    // ---- Leg 2: IT STAYS THERE, and the excursion does not grow -------------------------------------
    // The defect this file exists for. A doubled-gain controller is exactly right on its first
    // correction and wrong only in how the error EVOLVES, so a converged-value check cannot see it.
    // Comparing the second half's worst excursion against the first half's is what distinguishes a
    // controller that settles from one that is winding up.
    {
        scalar first = 0, second = 0;
        for (int i = 0; i < N; ++i)
        {
            const scalar e = std::fabs(on.meanUx[(std::size_t)i] - UBAR);
            if (i < N/2) first = std::fmax(first, e);
            else         second = std::fmax(second, e);
        }
        check(second <= first,
              "the excursion from Ubar SHRINKS over the run -- the loop settles, it does not wind up");
        check(second < scalar(0.05)*UBAR, "...and the late-run mean stays within 5% of Ubar");
        std::printf("        (worst |mean-Ubar|: first half %.3e, second half %.3e)\n",
                    (double)first, (double)second);
    }

    // ---- Leg 3: the source is what does it (vacuity guard) ------------------------------------------
    // Without it the channel has nothing driving it and must decay. If this passed, Legs 1-2 would be
    // measuring an initial condition that happened to sit at Ubar rather than a controller holding it.
    // DECAYS, and no more than that: the mean stays between rest and where it started. The guard was one-sided
    // (below 0.9 Ubar) until 2026-10-08, and a pressure solve that had diverged passed it at -36.3.
    {
        const scalar lastOff = off.meanUx.back();
        check(off.finite && lastOff > scalar(0) && lastOff < scalar(0.9)*UBAR,
              "vacuity guard: with the source removed the flow DECAYS, so Legs 1-2 measure the source");
        std::printf("        (undriven mean after %d steps: %.6f)\n", N, (double)lastOff);
    }

    // ---- Leg 4: the periodic pair is in the pressure hierarchy ONCE -----------------------------------
    // This solver's hierarchy is agglomerated over its interface edges and its coarse matrices hold the
    // pair (device_simple_foam.cu); the pressure solver's own way of carrying a pair on every grid, made for
    // interFoam, must then add nothing (AMGData::pairInCoarseMatrices). CONTROL, this binary again with
    // BRAE_CONTROL_AMG_PAIR_TWICE=1: the pair on the coarse grids twice, two pressure solves diverge, and
    // the undriven mean leaves the band above. MEASURED 2026-10-08: 0.436919 as it is, 11116.38 twice.
    {
        const scalar twice = childUndrivenMean(argv[0], "BRAE_CONTROL_AMG_PAIR_TWICE");
        check(std::isfinite(twice) && !(twice > scalar(0) && twice < scalar(0.9)*UBAR),
              "CONTROL: with the pair carried on the coarse grids a second time the undriven flow does NOT decay");
        std::printf("        (undriven mean with the pair twice: %.6f)\n", (double)twice);
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
