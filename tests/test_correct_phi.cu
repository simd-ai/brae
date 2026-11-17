// CorrectPhi -- OF finiteVolume/cfdTools/general/CorrectPhi, as pimpleFoam calls it after a mesh move.
//
//     phi = mesh.Sf() & Uf();                       // rebuild the ABSOLUTE flux from the face velocity
//     CorrectPhi(U, phi, p, rAUf = 1, divU = 0, pimple);
//     fvc::makeRelative(phi, U);
//
// The middle line solves laplacian(1, pcorr) == div(phi) and subtracts pcorrEqn.flux(), i.e. it PROJECTS
// the remapped flux back onto the divergence-free space. Everything the projection is worth is in that
// postcondition, so that is what Leg 2 asserts directly rather than re-deriving the algebra.
//
// WHAT MAKES IT SUBTLE is which faces the correction is allowed to leave through. pcorr's boundary is
// built from p's types, NOT U's: fixedValue 0 wherever p fixes a value, zeroGradient everywhere else.
// So at a wall or a prescribed inlet -- where p is zeroGradient -- pcorrEqn.flux() is identically zero
// and the prescribed flux SURVIVES the projection untouched (Leg 3). At a pressure outlet it does not,
// and must not (Leg 4): that is the only place the spurious divergence can go. An implementation that
// built pcorr's boundary from U's types instead would pin the outlet and open the inlet -- the exact
// inverse -- and would still produce a divergence-free field, passing Leg 2 and failing Legs 3 and 4.
//
// The fixture uses a deliberately NON-SOLENOIDAL U so the projection has real work to do; Leg 1 is the
// vacuity guard that says so, because on a divergence-free start every other leg would pass trivially.
#include "box_mesh.cuh"
#include "device_simple_foam.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
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
} // namespace

int main()
{
    std::printf("== CorrectPhi (flux projection after a mesh move) ==\n");

    const label N = 6;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N);
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // A velocity with a real divergence: u_x grows with x, and nothing compensates it. fvc::flux of this
    // is exactly the "flux that does not close" a mesh move produces, without needing to move a mesh.
    const std::vector<vector>& C = g.C();
    GeometricField<vector> U;
    U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = vector{1.0 + 2.0*C[c].x, 0.3*C[c].y, 0.0};

    GeometricField<scalar> p;
    p.internal.assign(static_cast<std::size_t>(nC), scalar(0));

    // patch 0: U fixedValue (a prescribed inlet), p zeroGradient   -> pcorr zeroGradient, flux PINNED
    // patch 1: U zeroGradient,                    p fixedValue     -> pcorr fixedValue,   flux OPEN
    // the rest: U fixedValue walls,               p zeroGradient   -> pinned
    const std::size_t inletPatch = 0, outletPatch = 1;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& q = fvp[pi];
        if (pi == outletPatch)
        {
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
            p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, true, scalar(0), std::vector<scalar>{}));
        }
        else
        {
            const vector uv = (pi == inletPatch) ? vector{1.0, 0.0, 0.0} : vector{0.0, 0.0, 0.0};
            U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, uv, std::vector<vector>{}));
            p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        }
    }
    U.evaluateBoundary();
    p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    DeviceSimpleControls ctl;
    ctl.nu = 1e-3;
    ctl.turbulent = false;
    ctl.needRef = false;          // p fixes a value on the outlet, so pcorr does too
    ctl.tolPcorr = 1e-13;         // OF's tutorials use 0.02; tightened here so Leg 2 can be a real number
    ctl.relTolPcorr = 0.0;
    DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl);
    solver.enableUf();            // Uf = interpolate(U): what CorrectPhi rebuilds the flux FROM

    // boundary face order = patch order, coupled patches skipped (the DeviceBoundary convention)
    std::vector<label> patchOf, cellOf;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCyclicFamilyType(fvp[pi].type)) continue;
        for (label i = 0; i < fvp[pi].size; ++i) { patchOf.push_back((label)pi); cellOf.push_back(fvp[pi].faceCells[i]); }
    }

    auto divergence = [&](const std::vector<scalar>& pi_, const std::vector<scalar>& pb)
    {
        std::vector<scalar> d(static_cast<std::size_t>(nC), scalar(0));
        for (label f = 0; f < m.nInternalFaces(); ++f)
        {
            d[m.owner()[f]]     += pi_[f];
            d[m.neighbour()[f]] -= pi_[f];
        }
        for (std::size_t i = 0; i < pb.size(); ++i) d[cellOf[i]] += pb[i];
        const std::vector<scalar>& V = g.V();
        for (label c = 0; c < nC; ++c) d[c] /= V[c];
        return d;
    };
    auto maxAbs = [](const std::vector<scalar>& v)
    {
        scalar mx = 0;
        for (const scalar x : v) mx = std::max(mx, std::fabs(x));
        return mx;
    };

    const std::vector<scalar> pi0 = solver.phiInternal();
    const std::vector<scalar> pb0 = solver.phiBoundary();
    const scalar divBefore = maxAbs(divergence(pi0, pb0));

    // ---- Leg 1: vacuity guard -- the starting flux really is divergent -----------------------------
    check(divBefore > 1.0, "vacuity guard: the un-projected flux has a large divergence to remove");

    solver.correctPhi(fvp, 0);

    const std::vector<scalar> pi1 = solver.phiInternal();
    const std::vector<scalar> pb1 = solver.phiBoundary();
    const scalar divAfter = maxAbs(divergence(pi1, pb1));

    // ---- Leg 2: the postcondition the whole routine exists for -------------------------------------
    check(divAfter < 1e-9*std::max(divBefore, scalar(1)),
          "after CorrectPhi the flux is divergence-free in every cell");
    std::printf("        (max|div phi|  %.4e  ->  %.4e)\n", (double)divBefore, (double)divAfter);

    // ---- Leg 3: the prescribed flux survives, because pcorr is zeroGradient there -------------------
    {
        scalar moved = 0;
        int n = 0;
        for (std::size_t i = 0; i < pb0.size(); ++i)
            if (patchOf[i] != (label)outletPatch) { moved = std::max(moved, std::fabs(pb1[i] - pb0[i])); ++n; }
        check(n > 0, "vacuity guard: there are pinned (p-zeroGradient) boundary faces");
        check(moved == 0.0, "the flux through every p-zeroGradient patch is EXACTLY unchanged (inlet + walls)");
    }

    // ---- Leg 4: ...and the correction does leave through the pressure outlet ------------------------
    {
        scalar moved = 0;
        int n = 0;
        for (std::size_t i = 0; i < pb0.size(); ++i)
            if (patchOf[i] == (label)outletPatch) { moved = std::max(moved, std::fabs(pb1[i] - pb0[i])); ++n; }
        check(n > 0, "vacuity guard: the fixture has a p-fixedValue outlet");
        check(moved > 1e-6, "discrimination: the outlet IS corrected -- pcorr's BCs come from p, not from U");
    }

    // ---- Leg 5: global conservation ----------------------------------------------------------------
    // The projection moves flux around; it must not create any. With one open patch, everything the
    // domain gains through the pinned patches has to leave through it.
    {
        scalar net = 0;
        for (const scalar x : pb1) net += x;
        scalar scale = 0;
        for (const scalar x : pb1) scale += std::fabs(x);
        check(std::fabs(net) < 1e-10*std::max(scale, scalar(1)),
              "the corrected flux balances globally: sum of the boundary flux is zero");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
