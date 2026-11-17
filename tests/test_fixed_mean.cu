// fixedMean: a fixedValue whose face values ARE the adjacent cell values, shifted (or scaled) so their
// AREA-WEIGHTED mean equals a prescribed one.
//
// OF fixedMeanFvPatchField::updateCoeffs():
//     psi  = patchInternalField()
//     mean = sum(magSf*psi)/sum(magSf)
//     if (|target| > SMALL && |mean| > 0.5*|target|) psi *= |target|/|mean|
//     else                                           psi += (target - mean)
// brae builds it as a plain fixedValue and recomputes refValue every step, the same way codedFixedValue
// and fanPressure are handled -- so what has to be asserted is the CONTRACT on the values the solver
// actually imposed, not a formula copied twice.
//
// Leg 1 is that contract: after a step, the area-weighted mean of the patch must BE the target. Leg 2
// pins which branch ran -- with target = 0 the shift branch is forced (|target| is not > SMALL), and the
// imposed values must then be the cell values plus a single constant, i.e. their SPREAD is unchanged.
// A scale-instead-of-shift implementation satisfies leg 1 and fails leg 2.
#include "box_mesh.cuh"
#include "device_simple_foam.cuh"
#include "cyclic_field.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "fan_pressure.cuh"   // applyFixedMean: the formula the solver uses
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
}

int main()
{
    const label N = 6;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // a non-uniform p, so the patch values vary and a shift is distinguishable from a scale
    std::vector<scalar> p0(nC);
    for (label c = 0; c < nC; ++c) p0[c] = 3.0 + 2.0*std::sin(0.6*c);
    std::vector<vector> U0(nC, vector{1.0, 0.0, 0.0});

    GeometricField<vector> U = buildCyclicField<vector>(U0, fvp, {}, /*wallNoSlip*/true);  U.evaluateBoundary();
    GeometricField<scalar> p = buildCyclicField<scalar>(p0, fvp, {});                      p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    DeviceSimpleControls ctl;
    ctl.nu = 1e-3;
    ctl.turbulent = false;
    DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl);

    // the DeviceBoundary face range of the first non-coupled patch (that ordering skips coupled patches)
    label start = 0, count = 0;
    std::string pname;
    for (const FvPatch& q : fvp)
    {
        if (isCyclicFamilyType(q.type)) continue;
        if (q.size > 0) { count = q.size; pname = q.name; break; }
        start += q.size;
    }
    if (!count) { std::printf("  FAIL vacuous: no usable patch in the fixture\n"); return 1; }

    const scalar target = 0.0;                     // forces OF's SHIFT branch
    solver.setFixedMean({{start, count, target}});
    solver.step();

    const std::vector<scalar> rv  = solver.pressureBoundary().refValue.host();
    const std::vector<scalar> sf  = solver.pressureBoundary().magSf.host();
    const std::vector<label>  fc  = solver.pressureBoundary().faceCell.host();
    const std::vector<scalar> pc  = solver.p();

    // ---- 1. the area-weighted mean over the patch equals the target ----
    {
        scalar sA = 0, sV = 0;
        for (label i = 0; i < count; ++i) { const label f = start + i; sA += sf[f]; sV += sf[f]*rv[f]; }
        const scalar mean = sV / sA;
        std::printf("  patch '%s' (%d faces): area-weighted mean = %+.3e  (target %+.3g)\n",
                    pname.c_str(), (int)count, (double)mean, (double)target);
        if (std::fabs(mean - target) > 1e-10)
        { std::printf("  FAIL fixedMean did not hold its prescribed mean\n"); ++failures; }
    }

    // ---- 2. the formula itself, both branches, against a hand-worked example ----
    // The end-to-end leg above cannot see WHICH branch ran: `solver.p()` is the state AFTER the step,
    // not the patchInternalField the boundary condition saw mid-step. So the formula has one home
    // (applyFixedMean, which the solver calls) and is asserted here directly.
    {
        const std::vector<scalar> a  = {1.0, 3.0, 2.0, 2.0};   // areas: deliberately non-uniform
        const std::vector<scalar> ps = {1.0, 2.0, 5.0, 4.0};
        std::vector<scalar> out(4);
        const scalar sA = 8.0;
        const scalar mean = (1.0*1.0 + 3.0*2.0 + 2.0*5.0 + 2.0*4.0)/sA;   // = 25/8 = 3.125

        // target 0 -> SHIFT (|target| is not > SMALL), so every value moves by the SAME constant
        applyFixedMean(0.0, a.data(), ps.data(), 4, out.data());
        scalar sv = 0, lo = 1e300, hi = -1e300;
        for (int i = 0; i < 4; ++i) { sv += a[i]*out[i]; const scalar d = out[i] - ps[i]; lo = std::fmin(lo,d); hi = std::fmax(hi,d); }
        std::printf("  shift branch (target 0): mean = %+.3e, offset spread = %.3e (offset %+.4f, mean was %.4f)\n",
                    (double)(sv/sA), (double)(hi - lo), (double)lo, (double)mean);
        if (std::fabs(sv/sA) > 1e-12)
        { std::printf("  FAIL the shift branch does not reach the target mean\n"); ++failures; }
        if (hi - lo > 1e-12)
        { std::printf("  FAIL the shift branch is not a single constant offset\n"); ++failures; }
        if (std::fabs(lo + mean) > 1e-12)
        { std::printf("  FAIL the offset is not (target - mean)\n"); ++failures; }

        // target 2 -> SCALE (|mean| = 3.125 > 0.5*2), matching the MAGNITUDE and keeping psi's sign
        applyFixedMean(2.0, a.data(), ps.data(), 4, out.data());
        sv = 0;
        scalar rlo = 1e300, rhi = -1e300;
        for (int i = 0; i < 4; ++i) { sv += a[i]*out[i]; const scalar r = out[i]/ps[i]; rlo = std::fmin(rlo,r); rhi = std::fmax(rhi,r); }
        std::printf("  scale branch (target 2): mean = %+.6f, ratio spread = %.3e (ratio %.6f, want %.6f)\n",
                    (double)(sv/sA), (double)(rhi - rlo), (double)rlo, (double)(2.0/mean));
        if (std::fabs(sv/sA - 2.0) > 1e-12)
        { std::printf("  FAIL the scale branch does not reach the target mean\n"); ++failures; }
        if (rhi - rlo > 1e-12)
        { std::printf("  FAIL the scale branch is not a single constant ratio\n"); ++failures; }
        if (std::fabs(rlo - 2.0/mean) > 1e-12)
        { std::printf("  FAIL the ratio is not |target|/|mean|\n"); ++failures; }

        // ...and the branches must actually DIFFER, or the two blocks above test one behaviour twice
        std::vector<scalar> sh(4);
        applyFixedMean(0.0, a.data(), ps.data(), 4, sh.data());
        scalar w = 0;
        for (int i = 0; i < 4; ++i) w = std::fmax(w, std::fabs(sh[i] - out[i]));
        if (w < 1e-6)
        { std::printf("  FAIL vacuous: shift and scale coincide on this fixture\n"); ++failures; }
    }

    // ---- 3. a NON-ZERO target end to end: OF matches the MAGNITUDE, not the sign (it uses mag()) ----
    {
        const scalar t2 = 2.0;
        solver.setFixedMean({{start, count, t2}});
        solver.step();
        const std::vector<scalar> r2 = solver.pressureBoundary().refValue.host();
        scalar sA = 0, sV = 0;
        for (label i = 0; i < count; ++i) { const label f = start + i; sA += sf[f]; sV += sf[f]*r2[f]; }
        std::printf("  target %+.3g end to end: area-weighted mean = %+.6f (|mean| is what OF matches)\n",
                    (double)t2, (double)(sV/sA));
        if (std::fabs(std::fabs(sV/sA) - t2) > 1e-8)
        { std::printf("  FAIL a non-zero target is not held in magnitude\n"); ++failures; }
    }

    std::printf("fixed_mean: %d failures\n", failures);
    return failures ? 1 : 0;
}
