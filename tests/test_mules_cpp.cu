// MULES -- held against the property it exists to guarantee, not against a stored field.
//
// THE ORACLE IS BOUNDEDNESS, AND IT IS AN ASSERTION RATHER THAN A TOLERANCE. MULES has one job: blend
// a high-order face flux towards first-order upwind by exactly as much as it takes to keep every cell
// inside [psiMin, psiMax], and no more. That is a postcondition, it holds on any mesh and any flux
// field, and it needs no instrumented OpenFOAM to check -- which matters, because a stock run never
// writes the limiter and a stored-field oracle for lambda does not exist yet.
//
// A BOUNDEDNESS GATE ALONE WOULD BE TRIVIAL TO PASS: lambda = 0 everywhere is perfectly bounded and is
// also first-order upwind, i.e. a solver that has thrown away the scheme the case asked for. So the
// gate is a PAIR of arms that bracket the answer from both sides:
//
//   arm 1  the limited run stays in [0,1] exactly, where the unlimited one does NOT  (not too weak)
//   arm 2  on a smooth, well-resolved field lambda is 1 on every face                (not too strong)
//
// Neither alone is worth anything. Arm 1's control is the same advection with lambda == 1, run in this
// file, so the overshoot is measured rather than asserted to exist.
//
// THE FIXTURE is one-dimensional advection of a step at constant velocity on a divergence-free flux --
// the smallest problem on which a limiter has something to do. Central differencing on a step
// overshoots; that is the whole point of choosing it.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "mules_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace brae;
namespace mules = brae::cpu::MULES;

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

constexpr label kN     = 20;
constexpr scalar kU    = 1.0;       // advection velocity, +x
constexpr scalar kDt   = 0.25;      // Courant 0.25 on unit cells
constexpr label kSteps = 40;

// The fixture, rebuilt for each arm so no arm can leave state behind.
struct Case
{
    PrimitiveMesh        m;
    FvGeometry           g;
    std::vector<FvPatch> fvp;
    SurfaceScalarField   phi;       // divergence-free, +x
};

Case makeCase()
{
    Case c;
    c.m = boxtest::boxMesh(kN, 1, 1);
    c.g.build(c.m);
    c.fvp = buildPatches(c.m, c.g);

    // phi = U & Sf. Cells are unit cubes, so every x-face has area 1 and the internal flux is U.
    // The two end patches carry the SAME flux with the signs their outward normals give, which is what
    // makes div(phi) identically zero -- without that the alpha field would gain or lose mass at the
    // ends and a boundedness failure would be the fixture's, not MULES's.
    c.phi.internal.assign(static_cast<std::size_t>(c.m.nInternalFaces()), kU);
    c.phi.boundary.resize(c.fvp.size());
    for (std::size_t pi = 0; pi < c.fvp.size(); ++pi)
    {
        const FvPatch& q = c.fvp[pi];
        scalar v = 0;
        if (q.name == "inlet")  v = -kU;      // outward -x, so an inflow is negative
        if (q.name == "outlet") v =  kU;
        c.phi.boundary[pi].assign(static_cast<std::size_t>(q.size), v);
    }
    return c;
}

// alpha = 1 upstream of the step, 0 downstream. fixedValue 1 at the inlet, zeroGradient everywhere else.
GeometricField<scalar> makeAlpha(const Case& c, label stepAt)
{
    GeometricField<scalar> a;
    a.internal.resize(static_cast<std::size_t>(c.m.nCells()));
    for (label i = 0; i < c.m.nCells(); ++i) a.internal[i] = (i < stepAt) ? scalar(1) : scalar(0);
    for (const FvPatch& q : c.fvp)
    {
        if (q.name == "inlet")
            a.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(
                q, true, scalar(1), std::vector<scalar>{}));
        else
            a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    }
    a.evaluateBoundary();
    return a;
}

// The HIGH-ORDER flux the case is taken to have asked for: plain central differencing. Chosen because
// it is the simplest flux that genuinely overshoots on a step, so arm 1's control has something to
// measure.
SurfaceScalarField centralFlux(const Case& c, const GeometricField<scalar>& a)
{
    SurfaceScalarField f;
    f.internal.resize(static_cast<std::size_t>(c.m.nInternalFaces()));
    for (label i = 0; i < c.m.nInternalFaces(); ++i)
    {
        const scalar w = c.g.weights()[i];
        f.internal[i] = c.phi.internal[i]
                      * (w*a.internal[c.m.owner()[i]] + (scalar(1) - w)*a.internal[c.m.neighbour()[i]]);
    }
    f.boundary.resize(c.fvp.size());
    for (std::size_t pi = 0; pi < c.fvp.size(); ++pi)
    {
        const std::vector<scalar>& av = a.boundary[pi]->value();
        f.boundary[pi].resize(av.size());
        for (std::size_t i = 0; i < av.size(); ++i)
            f.boundary[pi][i] = c.phi.boundary[pi][i] * av[i];
    }
    return f;
}

scalar totalMass(const GeometricField<scalar>& a, const FvGeometry& g)
{
    scalar s = 0;
    for (std::size_t i = 0; i < a.internal.size(); ++i) s += a.internal[i] * g.V()[i];
    return s;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== MULES ==\n");
    const std::string tut = (argc > 1) ? argv[1] : "";

    // ---- 0. the controls ---------------------------------------------------------------------------
    if (!tut.empty() && std::filesystem::exists(tut + "/system/fvSolution"))
    {
        const FoamDict fvSolution = readDict(tut + "/system/fvSolution");
        const mules::Controls c = mules::readControls(fvSolution, "alpha.water");
        // damBreak sets nLimiterIter 5, NOT the default 3 -- a port that hardcoded the default would
        // run a different limiter on the canonical case.
        checkNum("damBreak nLimiterIter is 5, not the default 3", scalar(c.nLimiterIter), scalar(5));
        checkNum("smoothLimiter absent -> 0", c.smoothLimiter, scalar(0));
        checkNum("extremaCoeff absent -> 0",  c.extremaCoeff,  scalar(0));
    }
    else
    {
        std::printf("  SKIP: OpenFOAM's damBreak tutorial not found at \"%s\"\n", tut.c_str());
        return 77;
    }

    // ---- 1. BOUNDEDNESS, and the control that shows it is not free ---------------------------------
    scalar worstLimited = 0, worstUnlimited = 0, massDrift = 0;
    {
        const Case c = makeCase();
        mules::Fields f;                       // all null: rho == 1, Sp == Su == 0, bounds [0,1]
        mules::Controls ctl;
        ctl.nLimiterIter = 5;                  // damBreak's

        GeometricField<scalar> lim = makeAlpha(c, kN/3);
        GeometricField<scalar> raw = makeAlpha(c, kN/3);
        const scalar mass0 = totalMass(lim, c.g);
        scalar influx = 0;

        for (label step = 0; step < kSteps; ++step)
        {
            // limited
            {
                const std::vector<scalar> old = lim.internal;
                SurfaceScalarField phiPsi = centralFlux(c, lim);
                mules::explicitSolveLimited(scalar(1)/kDt, lim, old, c.phi, phiPsi, f, ctl,
                                            c.m, c.g, c.fvp);
                for (scalar v : lim.internal)
                {
                    worstLimited = std::fmax(worstLimited, std::fmax(-v, v - scalar(1)));
                }
                // what crossed the boundary this step, for the conservation arm
                for (std::size_t pi = 0; pi < c.fvp.size(); ++pi)
                    for (std::size_t i = 0; i < phiPsi.boundary[pi].size(); ++i)
                        influx -= phiPsi.boundary[pi][i] * kDt;
            }
            // THE CONTROL: the same advection with the correction unlimited, i.e. lambda == 1.
            {
                const std::vector<scalar> old = raw.internal;
                const SurfaceScalarField phiPsi = centralFlux(c, raw);
                mules::explicitSolve(scalar(1)/kDt, raw.internal, old, phiPsi, f, c.m, c.g, c.fvp);
                raw.evaluateBoundary();
                for (scalar v : raw.internal)
                    worstUnlimited = std::fmax(worstUnlimited, std::fmax(-v, v - scalar(1)));
            }
        }
        massDrift = std::fabs(totalMass(lim, c.g) - (mass0 + influx));

        std::printf("  %d steps of 1D advection, Co = %.2f:\n", (int)kSteps, (double)(kU*kDt));
        std::printf("    limited   worst excursion outside [0,1] = %.3e\n", (double)worstLimited);
        std::printf("    UNLIMITED worst excursion outside [0,1] = %.3e\n", (double)worstUnlimited);
        check("MULES keeps alpha in [0,1] -- an assertion, not a tolerance", worstLimited <= scalar(1e-14));
        check("...and the unlimited flux does NOT, so this arm discriminates",
              worstUnlimited > scalar(1e-3));
        std::printf("    mass drift vs the boundary flux = %.3e\n", (double)massDrift);
        check("MULES is conservative: the interior gains exactly what crossed the boundary",
              massDrift <= scalar(1e-12) * std::fmax(scalar(1), std::fabs(totalMass(lim, c.g))));
    }

    // ---- 2. WHAT MULES LIMITS, AND WHAT IT LEAVES ALONE --------------------------------------------
    // The other half of the bracket. Without it, lambda == 0 everywhere would pass arm 1 -- and that is
    // first-order upwind, i.e. the scheme the case asked for silently discarded.
    //
    // THE PREMISE I STARTED WITH WAS WRONG, and the measurement corrected it. "Smooth and in bounds
    // implies lambda == 1" is false: MULES is LOCAL-EXTREMUM-DIMINISHING, so psiMaxn is the maximum
    // over a cell's NEIGHBOURS, never its own value. A smooth interior peak already exceeds both of its
    // neighbours, so its budget is negative and MULES clamps it to zero. That is the scheme, not a
    // defect -- and `extremaCoeff` exists precisely to buy the slack back, which is what the second
    // pair below asserts.
    {
        const Case c = makeCase();
        mules::Fields f;
        mules::Controls ctl;

        auto lambdaOn = [&](const std::vector<scalar>& cells, scalar extremaCoeff)
        {
            GeometricField<scalar> a;
            a.internal = cells;
            for (const FvPatch& q : c.fvp)
                a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
            a.evaluateBoundary();
            SurfaceScalarField phiPsi = centralFlux(c, a);
            mules::Controls k = ctl; k.extremaCoeff = extremaCoeff;
            mules::Limiter lam;
            const std::vector<scalar> old = a.internal;
            mules::limit(scalar(1)/kDt, a, old, c.phi, phiPsi, f, k, c.m, c.g, c.fvp, &lam);
            return lam;
        };
        auto countLimited = [&](const mules::Limiter& l)
        {
            int n = 0;
            for (scalar v : l.internal) if (v < scalar(1) - scalar(1e-12)) ++n;
            return n;
        };

        // (a) A MONOTONE ramp, well inside [0,1]. Every interior cell lies between its two neighbours,
        //     so nothing needs limiting -- EXCEPT at the two ends, where a zeroGradient patch
        //     contributes no extrema at all (MULESTemplates.C:345-360 takes the `else` branch) and the
        //     end cell therefore sees a ONE-SIDED neighbourhood it is already outside of.
        std::vector<scalar> ramp(static_cast<std::size_t>(c.m.nCells()));
        for (label i = 0; i < c.m.nCells(); ++i)
            ramp[i] = scalar(0.2) + scalar(0.6) * c.g.C()[i].x / scalar(kN);
        const mules::Limiter lr = lambdaOn(ramp, scalar(0));
        const int nr = countLimited(lr);
        std::printf("  monotone ramp: %d of %d faces limited\n", nr, (int)lr.internal.size());
        check("on a monotone in-bounds ramp MULES limits only the two domain-end faces", nr == 2);
        bool endsOnly = true;
        for (label fi = 0; fi < c.m.nInternalFaces(); ++fi)
        {
            const bool touchesEnd = (c.m.owner()[fi] == 0) || (c.m.neighbour()[fi] == c.m.nCells() - 1);
            if (!touchesEnd) endsOnly = endsOnly && (lr.internal[fi] >= scalar(1) - scalar(1e-12));
        }
        check("...and every face away from the ends keeps lambda == 1", endsOnly);

        // (b) A SMOOTH COSINE, which has a peak and a trough inside the domain. MULES limits at both,
        //     because a local extremum is exactly what an extremum-diminishing limiter diminishes.
        std::vector<scalar> hump(static_cast<std::size_t>(c.m.nCells()));
        for (label i = 0; i < c.m.nCells(); ++i)
        {
            const scalar x = c.g.C()[i].x / scalar(kN);
            hump[i] = scalar(0.5) + scalar(0.3)*std::cos(scalar(2)*scalar(M_PI)*x);
        }
        const int nh0 = countLimited(lambdaOn(hump, scalar(0)));
        const int nh2 = countLimited(lambdaOn(hump, scalar(0.2)));
        std::printf("  smooth cosine: %d faces limited at extremaCoeff 0, %d at extremaCoeff 0.2\n",
                    nh0, nh2);
        check("a smooth field with interior EXTREMA is limited there -- MULES is extremum-diminishing",
              nh0 > 0);
        check("...and extremaCoeff buys exactly that slack back", nh2 == 0);

        // (c) THE STEP, for contrast: limited nearly everywhere. Without this the two arms above could
        //     be satisfied by a limiter that never acts.
        std::vector<scalar> step(static_cast<std::size_t>(c.m.nCells()));
        for (label i = 0; i < c.m.nCells(); ++i) step[i] = (i < kN/3) ? scalar(1) : scalar(0);
        const mules::Limiter ls = lambdaOn(step, scalar(0));
        scalar sMin = 1;
        for (scalar l : ls.internal) sMin = std::fmin(sMin, l);
        std::printf("  sharp step:    %d faces limited, smallest lambda %.6f\n",
                    countLimited(ls), (double)sMin);
        check("...while a step is limited hard (control)", sMin < scalar(0.5));
    }

    // ---- 3. lambda in [0,1], and monotone in nLimiterIter ------------------------------------------
    {
        const Case c = makeCase();
        mules::Fields f;

        auto lambdaAfter = [&](label iters)
        {
            GeometricField<scalar> a = makeAlpha(c, kN/3);
            SurfaceScalarField phiPsi = centralFlux(c, a);
            mules::Controls ctl; ctl.nLimiterIter = iters;
            mules::Limiter lam;
            const std::vector<scalar> old = a.internal;
            mules::limit(scalar(1)/kDt, a, old, c.phi, phiPsi, f, ctl, c.m, c.g, c.fvp, &lam);
            return lam;
        };

        const mules::Limiter l1 = lambdaAfter(1);
        const mules::Limiter l3 = lambdaAfter(3);
        const mules::Limiter l9 = lambdaAfter(9);

        bool inRange = true;
        for (scalar l : l3.internal) inRange = inRange && (l >= scalar(0) && l <= scalar(1));
        check("lambda is in [0,1] on every face", inRange);

        // nLimiterIter is a MONOTONE TIGHTENING, not a convergence loop: lambda starts at 1 and never
        // grows. Worth stating precisely, because the obvious explanation is wrong. It is NOT the
        // running `min(lambda, ...)` that makes it so -- both accumulators are built FROM lambda, so
        // they shrink as it does and the new value is already below the old one. Replacing the min
        // with a plain assignment was tried here and produced a bit-identical field; the min is
        // defensive, not load-bearing, and this gate does not pretend to catch its removal.
        bool mono = true;
        for (std::size_t i = 0; i < l1.internal.size(); ++i)
            mono = mono && (l3.internal[i] <= l1.internal[i] + scalar(1e-15))
                        && (l9.internal[i] <= l3.internal[i] + scalar(1e-15));
        check("lambda is monotone non-increasing in nLimiterIter", mono);

        // ON ONE-DIMENSIONAL ADVECTION THE FIRST PASS IS ALREADY THE FIXED POINT: every face is either
        // unlimited or driven straight to zero, so nothing is left for a second pass to tighten. That
        // makes this fixture blind to an implementation that ASSIGNS instead of taking the running min
        // -- the two agree when the answer never moves. Measured and reported, not asserted away.
        int changed = 0;
        for (std::size_t i = 0; i < l1.internal.size(); ++i)
            if (std::fabs(l9.internal[i] - l1.internal[i]) > scalar(1e-15)) ++changed;
        std::printf("  1D: faces whose lambda changed between 1 and 9 iterations: %d of %d\n",
                    changed, (int)l1.internal.size());
    }

    // ---- 3b. THE ITERATION, on a case where it has something to do ---------------------------------
    // nLimiterIter only earns its keep when limiting one face changes the budget available to another,
    // which needs a cell whose faces are limited by DIFFERENT neighbours at once. One dimension cannot
    // produce that -- each cell has one inflow and one outflow -- so this arm is two-dimensional with
    // a flux deliberately NOT aligned with the mesh, and the interface smooth enough that the first
    // pass leaves lambda strictly between 0 and 1 on a handful of faces.
    //
    // Without this arm the gate cannot tell `lambda = min(lambda, ...)` from `lambda = ...`, which is
    // the difference between a monotone tightening and a loop whose answer depends on the parity of
    // the iteration count.
    {
        const label N2 = 12;
        PrimitiveMesh m2 = boxtest::boxMesh(N2, N2, 1);
        FvGeometry g2; g2.build(m2);
        const std::vector<FvPatch> fvp2 = buildPatches(m2, g2);
        const vector Udiag{scalar(0.8), scalar(0.6), scalar(0)};

        SurfaceScalarField phi2;
        phi2.internal.resize(static_cast<std::size_t>(m2.nInternalFaces()));
        for (label f = 0; f < m2.nInternalFaces(); ++f)
        {
            const vector& S = g2.Sf()[f];
            phi2.internal[f] = Udiag.x*S.x + Udiag.y*S.y + Udiag.z*S.z;
        }
        phi2.boundary.resize(fvp2.size());
        for (std::size_t pi = 0; pi < fvp2.size(); ++pi)
        {
            const FvPatch& q = fvp2[pi];
            phi2.boundary[pi].resize(static_cast<std::size_t>(q.size));
            for (label i = 0; i < q.size; ++i)
            {
                const vector& S = g2.Sf()[q.start + i];
                phi2.boundary[pi][i] = Udiag.x*S.x + Udiag.y*S.y + Udiag.z*S.z;
            }
        }

        auto sumLambda2 = [&](label iters)
        {
            GeometricField<scalar> a;
            a.internal.resize(static_cast<std::size_t>(m2.nCells()));
            for (label c = 0; c < m2.nCells(); ++c)
            {
                const vector& C = g2.C()[c];
                const scalar r = std::hypot(C.x - scalar(N2)/2, C.y - scalar(N2)/2);
                a.internal[c] = scalar(0.5)*(scalar(1) + std::tanh(scalar(N2)/5 - r));
            }
            for (const FvPatch& q : fvp2)
                a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
            a.evaluateBoundary();

            SurfaceScalarField pp;
            pp.internal.resize(static_cast<std::size_t>(m2.nInternalFaces()));
            for (label i = 0; i < m2.nInternalFaces(); ++i)
            {
                const scalar w = g2.weights()[i];
                pp.internal[i] = phi2.internal[i]
                    * (w*a.internal[m2.owner()[i]] + (scalar(1) - w)*a.internal[m2.neighbour()[i]]);
            }
            pp.boundary.resize(fvp2.size());
            for (std::size_t pi = 0; pi < fvp2.size(); ++pi)
            {
                const std::vector<scalar>& av = a.boundary[pi]->value();
                pp.boundary[pi].resize(av.size());
                for (std::size_t i = 0; i < av.size(); ++i)
                    pp.boundary[pi][i] = phi2.boundary[pi][i] * av[i];
            }

            mules::Fields f2; mules::Controls c2; c2.nLimiterIter = iters;
            mules::Limiter lam;
            const std::vector<scalar> old = a.internal;
            mules::limit(scalar(1)/scalar(0.2), a, old, phi2, pp, f2, c2, m2, g2, fvp2, &lam);
            scalar sum = 0; int partial = 0;
            for (scalar l : lam.internal)
            {
                sum += l;
                if (l > scalar(1e-9) && l < scalar(1) - scalar(1e-9)) ++partial;
            }
            return std::pair<scalar,int>{sum, partial};
        };

        const auto a1 = sumLambda2(1);
        const auto a2 = sumLambda2(2);
        const auto a4 = sumLambda2(4);
        std::printf("  2D diagonal flux: sum(lambda) 1 iter %.10f, 2 iters %.10f, 4 iters %.10f"
                    "  (%d faces partially limited)\n",
                    (double)a1.first, (double)a2.first, (double)a4.first, a1.second);
        check("the fixture leaves faces PARTIALLY limited, which is what makes the iteration matter",
              a1.second > 0);
        check("a second pass tightens lambda further -- the iteration is not a no-op",
              a2.first < a1.first - scalar(1e-12));
        check("...and it keeps tightening, monotonically", a4.first <= a2.first + scalar(1e-15));
    }

    // ---- 4. the boundary flux is never limited -----------------------------------------------------
    // phiBD is overwritten with phiPsi on every non-coupled patch, so phiCorr is identically zero
    // there. A prescribed inflow crosses whatever the limiter decides in the interior.
    {
        const Case c = makeCase();
        GeometricField<scalar> a = makeAlpha(c, kN/3);
        const SurfaceScalarField phiPsi = centralFlux(c, a);
        SurfaceScalarField phiBD;
        mules::boundedDonorFlux(c.phi, a, phiPsi, c.m, c.fvp, phiBD);

        scalar worstBnd = 0;
        for (std::size_t pi = 0; pi < c.fvp.size(); ++pi)
            for (std::size_t i = 0; i < phiBD.boundary[pi].size(); ++i)
                worstBnd = std::fmax(worstBnd, std::fabs(phiBD.boundary[pi][i] - phiPsi.boundary[pi][i]));
        check("phiBD == phiPsi on every non-coupled patch, so phiCorr is zero there", worstBnd == scalar(0));

        // CONTROL: the upwind flux at the inlet is NOT the same number, so the overwrite is doing work.
        // The inlet carries alpha = 1 by fixedValue while the first cell also holds 1 here, so the
        // discriminating patch is the OUTLET, where the upwind value is the cell's and phiPsi's is the
        // zeroGradient face value -- identical again. The honest control is therefore the internal
        // faces, where phiBD and phiPsi genuinely differ.
        scalar worstInt = 0;
        for (std::size_t i = 0; i < phiBD.internal.size(); ++i)
            worstInt = std::fmax(worstInt, std::fabs(phiBD.internal[i] - phiPsi.internal[i]));
        std::printf("  internal: worst |phiBD - phiPsi| = %.3e (the correction the limiter acts on)\n",
                    (double)worstInt);
        check("...and on internal faces the two DIFFER, so phiCorr is not trivially zero",
              worstInt > scalar(1e-3));
    }

    // ---- 5. a wedge patch is zeroed outright -------------------------------------------------------
    // 2D axisymmetric cases are wedges. MULESTemplates.C:530-533 sets lambda = 0 on them before any
    // coupled logic, and nothing else in this file would produce a zero there.
    {
        Case c = makeCase();
        for (FvPatch& q : c.fvp)
            if (q.name == "wallZmin" || q.name == "wallZmax") q.type = "wedge";

        GeometricField<scalar> a = makeAlpha(c, kN/3);
        SurfaceScalarField phiPsi = centralFlux(c, a);
        mules::Fields f; mules::Controls ctl;
        mules::Limiter lam;
        const std::vector<scalar> old = a.internal;
        mules::limit(scalar(1)/kDt, a, old, c.phi, phiPsi, f, ctl, c.m, c.g, c.fvp, &lam);

        bool wedgeZero = true, otherOne = true;
        for (std::size_t pi = 0; pi < c.fvp.size(); ++pi)
            for (scalar l : lam.boundary[pi])
            {
                if (c.fvp[pi].type == "wedge") wedgeZero = wedgeZero && (l == scalar(0));
                else                           otherOne  = otherOne  && (l == scalar(1));
            }
        check("a wedge patch gets lambda = 0", wedgeZero);
        check("...and a non-wedge patch does not (control)", otherOne);
    }

    // ---- 6. the controls that have a non-obvious default -------------------------------------------
    {
        const std::string base = "/tmp/brae_mules";
        std::filesystem::remove_all(base);
        auto write = [&](const std::string& dir, const std::string& body)
        {
            std::filesystem::create_directories(base + "/" + dir);
            std::ofstream(base + "/" + dir + "/fvSolution")
                << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
                << "solvers\n{\n    \"alpha.water.*\"\n    {\n" << body << "    }\n}\n";
            return readDict(base + "/" + dir + "/fvSolution");
        };
        const mules::Controls d = mules::readControls(write("def", "        nAlphaCorr 1;\n"), "alpha.water");
        checkNum("nLimiterIter defaults to 3", scalar(d.nLimiterIter), scalar(3));
        // boundaryExtremaCoeff defaults to extremaCoeff, NOT to 0.
        const mules::Controls e =
            mules::readControls(write("ext", "        extremaCoeff 0.2;\n"), "alpha.water");
        checkNum("extremaCoeff read", e.extremaCoeff, scalar(0.2));
        checkNum("boundaryExtremaCoeff defaults to extremaCoeff, not to 0",
                 e.boundaryExtremaCoeff, scalar(0.2));
        const mules::Controls b =
            mules::readControls(write("both", "        extremaCoeff 0.2; boundaryExtremaCoeff 0.5;\n"),
                                "alpha.water");
        checkNum("...and is read when given", b.boundaryExtremaCoeff, scalar(0.5));

        bool threw = false;
        try { (void)mules::readControls(readDict(base + "/def/fvSolution"), "alpha.oil"); }
        catch (const std::exception&) { threw = true; }
        check("no solvers entry for the field is refused", threw);
    }

    std::printf("test_mules_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
