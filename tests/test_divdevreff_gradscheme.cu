// divDevReff must use the case's NAMED grad(U) scheme, not a bare Gauss one.
//
// OF's linearViscousStress::divDevReff is
//     - fvc::div(nuEff*dev2(T(fvc::grad(U)))) - fvm::laplacian(nuEff, U)
// and `fvc::grad(U)` resolves gradSchemes `grad(U)`. A case saying `grad(U) cellLimited Gauss linear 1`
// therefore gets a LIMITED gradient inside the stress term. brae built a plain Gauss gradient there and
// ignored the entry.
//
// WHY IT HID FOR SO LONG. The whole term carries a factor of nuEff, so at a laminar water viscosity it
// is invisible. Measured on pimpleFoam/RAS/oscillatingInletACMI2D as a FREE run -- no restart, no probe,
// nothing but the two solvers and their written fields -- laminar, 10 steps:
//
//     nu = 1e-6 (the tutorial) : 6.5e-07 either way
//     nu = 1e-3 (a turbulent nut's size) : 4.7e-04 unlimited, 6.7e-08 limited
//
// a factor of 7000. On the tutorial itself the fix took the static laminar case from 6.5e-07 to 1.4e-07
// and the moving one from 9.7e-07 to 3.4e-07.
//
// NOT TO BE CONFUSED WITH the non-orthogonal laplacian correction, which brae deliberately leaves
// UNLIMITED and which is correct: OF's correctedSnGrad<vector>::correction loops the components and
// calls fullGradCorrection(vf.component(cmpt)), so the scheme lookup is for "grad(U.component(0))",
// which no case defines and which therefore falls back to gradSchemes `default`. The comment at the
// gradient split in solveMomentumPredictor says so, and checking it against v2412 confirmed it.
//
// WHAT THIS ASSERTS. The limiter is a property, not a formula: cellLimitedGrad clamps a cell's gradient
// so the value it extrapolates to each face stays inside the range of the cell and its neighbours. So
//   - on a field the gradient already reproduces exactly (linear in x), limiting must be a NO-OP
//   - on a field with a sharp jump, it must BITE
// A version that ignored the coefficient would pass the first and fail the second, and one that limited
// unconditionally would do the reverse. Both legs together pin it.
#include "box_mesh.cuh"
#include "device_divdevreff.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cyclic_field.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

// V*fvc::div(nuEff*dev2(T(grad(U)))) for a given U, with and without the named limiter.
std::vector<scalar> ddr(const DeviceMesh& dm, const DeviceVectorBoundary& dbU, const FvGeometry& g,
                        const std::vector<vector>& Ucells, scalar nu, scalar kc, int comp)
{
    const int nC = dm.nCells;
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (int c = 0; c < nC; ++c) { ux[c] = Ucells[c].x; uy[c] = Ucells[c].y; uz[c] = Ucells[c].z; }
    DeviceBuffer<scalar> Ux, Uy, Uz, nuC, nuB, sx, sy, sz;
    Ux.copyFrom(ux); Uy.copyFrom(uy); Uz.copyFrom(uz);
    nuC.copyFrom(std::vector<scalar>(nC, nu));
    nuB.copyFrom(std::vector<scalar>(static_cast<std::size_t>(dm.nBndFaces), nu));
    deviceDivDevReff(dm, dbU, Ux, Uy, Uz, nuC, nuB, sx, sy, sz, nullptr, nullptr, nullptr, kc);
    return (comp == 0 ? sx : (comp == 1 ? sy : sz)).host();
}

scalar worstDiff(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar w = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) w = std::fmax(w, std::fabs(a[i] - b[i]));
    return w;
}

}   // namespace

int main()
{
    const label N = 8;
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const int nC = static_cast<int>(m.nCells());

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    GeometricField<vector> Uf = buildCyclicField<vector>(std::vector<vector>(nC), fvp, {}, /*wallNoSlip*/true);
    Uf.evaluateBoundary();
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(Uf, fvp, g);

    const scalar nu = 1e-3;   // the size that makes the term visible at all

    // The field has to vary along the direction the term differentiates, or the whole thing is zero and
    // both legs pass on nothing: U = (f(x), 0, 0) gives sigma_xx = (nu/3)*dUx/dx and div_x = d(sigma_xx)/dx.
    // A step in Y instead leaves sigma_yx = dUy/dx = 0 and the x-component identically zero -- which is
    // exactly what the first draft of this test did, and its vacuity guard caught it.
    // boxMesh is UNIT-SPACED, not a unit cube: centres run 0.5 .. N-0.5. Put the step in the MIDDLE of
    // the domain, or it lands against the wall and the leg tests the boundary rather than the limiter.
    auto stepX   = [&](int c) { return g.C()[c].x > 0.5*N ? 1.0 : 0.0; };
    auto linearX = [&](int c) { return 2.0*g.C()[c].x; };

    // TWO rings in from the wall, not one. cellLimitedGrad reads boundary FACE values into its min/max,
    // and a no-slip wall sets Ux = 0 where the linear profile says 2x -- so the gradient in the wall row
    // is genuinely wrong and genuinely limited. divDevReff at a cell then reads its NEIGHBOURS' gradients
    // through the Gauss divergence, so the first row in from the wall inherits it: measured -1.67e-04 vs
    // -3.33e-04 at (6.5, 1.5). Two rings in it is 0.000e+00 over all 16 cells, exactly.
    std::vector<int> interior;
    for (int c = 0; c < nC; ++c)
    {
        const int i = c % N, j = (c/N) % N;
        if (i >= 2 && i <= N-3 && j >= 2 && j <= N-3) interior.push_back(c);
    }

    // ---- 1. a LINEAR field, away from the walls: the Gauss gradient is exact, so limiting is a NO-OP ----
    {
        std::vector<vector> U(nC);
        for (int c = 0; c < nC; ++c) U[c] = vector{linearX(c), 0, 0};
        const std::vector<scalar> unlim = ddr(dm, dbU, g, U, nu, 0.0, 0);
        const std::vector<scalar> lim   = ddr(dm, dbU, g, U, nu, 1.0, 0);
        scalar w = 0;
        for (int c : interior) w = std::fmax(w, std::fabs(unlim[c] - lim[c]));
        std::printf("  linear field   : |limited - unlimited| on %zu wall-free cells = %.3e (must be 0)\n",
                    interior.size(), (double)w);
        if (interior.empty())
        { std::printf("  FAIL vacuous: no wall-free cells, so the no-op claim is untested\n"); ++failures; }
        if (w > 1e-18)
        {
            std::printf("  FAIL the limiter changed a gradient that was already exact -- cellLimitedGrad must be\n"
                        "       a no-op where the extrapolated face values already lie inside the neighbour range\n");
            ++failures;
        }
    }

    // ---- 2. a SHARP field: the limiter must bite, and the coefficient must be read ----
    {
        std::vector<vector> U(nC);
        for (int c = 0; c < nC; ++c) U[c] = vector{stepX(c), 0, 0};
        const std::vector<scalar> unlim = ddr(dm, dbU, g, U, nu, 0.0, 0);
        const std::vector<scalar> lim   = ddr(dm, dbU, g, U, nu, 1.0, 0);
        const scalar w = worstDiff(unlim, lim);
        scalar scale = 0;
        for (scalar v : unlim) scale = std::fmax(scale, std::fabs(v));
        std::printf("  step field     : |limited - unlimited| = %.3e   (unlimited peak %.3e)\n",
                    (double)w, (double)scale);
        if (w <= 1e-18)
        {
            std::printf("  FAIL the coefficient was IGNORED: a step profile must be limited, so the two results\n"
                        "       cannot be identical. This is the defect the test exists for.\n");
            ++failures;
        }
        // VACUITY GUARD: the difference has to be a real fraction of the term, not round-off dressed up,
        // and the term itself has to be non-zero -- a field the divergence is blind to proves nothing.
        if (scale <= 0 || w/scale < 1e-3)
        {
            std::printf("  FAIL vacuous: unlimited peak %.3e, limiter moved it by %.3e of its own size -- this\n"
                        "       fixture does not exercise limiting\n",
                        (double)scale, (double)(scale > 0 ? w/scale : 0));
            ++failures;
        }
    }

    // ---- 3. kc = 0 must mean UNLIMITED, not "limit with 0" (the default every non-limited case takes)
    {
        std::vector<vector> U(nC);
        for (int c = 0; c < nC; ++c) U[c] = vector{stepX(c), 0, 0};
        const std::vector<scalar> a = ddr(dm, dbU, g, U, nu, 0.0, 0);
        const std::vector<scalar> b = ddr(dm, dbU, g, U, nu, 0.0, 0);
        if (worstDiff(a, b) != scalar(0))
        { std::printf("  FAIL kc=0 is not deterministic\n"); ++failures; }
        std::printf("  kc = 0         : unlimited path reproducible\n");
    }

    std::printf("divdevreff_gradscheme: %d failures\n", failures);
    return failures ? 1 : 0;
}
