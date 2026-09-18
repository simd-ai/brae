// interfaceProperties::calculateK -- interface curvature, and the contact angle that bends it.
//
// THE ORACLE PROBLEM. A stock OpenFOAM run never writes K_, so there is no stored field to compare
// against without instrumenting OpenFOAM's own class. The manifest carried this component as the one
// that needed of-instrument for exactly that reason.
//
// IT DOES NOT, BECAUSE CURVATURE IS GEOMETRY. K has known values that no solver has to supply:
//
//   * a FLAT interface has K = 0, exactly, at any resolution. That is an identity, not a tolerance,
//     and it is what catches spurious curvature -- the thing that drives the parasitic currents a VoF
//     code is judged on.
//   * a SPHERE of radius R has K = 2/R, approached as the mesh refines. That pins the MAGNITUDE and,
//     more importantly, the SIGN: positive for a drop of phase 1, negative for a bubble of it. A sign
//     error leaves every surface-tension force in the solver pointing the wrong way with the right
//     magnitude, and a flat-interface test cannot see it because -0 is 0.
//   * the CONTACT ANGLE has an exact postcondition of its own: after correctContactAngle,
//     acos(nHat & nf) IS theta. That is what the 2x2 solve is for.
//
// The convergence arm is what makes the sphere's tolerance principled rather than chosen: the error is
// measured at two resolutions and required to FALL, so the bound is a statement about the scheme
// rather than about this mesh.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "interface_properties_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace brae;
using namespace brae::cpu::interfaceProps;

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

struct Case
{
    PrimitiveMesh        m;
    FvGeometry           g;
    std::vector<FvPatch> fvp;
};

Case makeCase(label N, scalar h)
{
    Case c;
    c.m = boxtest::boxMesh(N, N, N, scalar(0), h, h, h);
    c.g.build(c.m);
    c.fvp = buildPatches(c.m, c.g);
    return c;
}

// alpha as a smooth function of the cells, with zeroGradient walls.
GeometricField<scalar> fieldFrom(const Case& c, const std::vector<scalar>& cells)
{
    GeometricField<scalar> a;
    a.internal = cells;
    for (const FvPatch& q : c.fvp)
        a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    a.evaluateBoundary();
    return a;
}
}   // namespace

int main()
{
    std::printf("== interfaceProperties: curvature ==\n");

    // ---- 0. the raw-value Gauss gradient reproduces fvc::gaussGrad exactly -------------------------
    // The smoothed path needs a gradient built from values rather than from a GeometricField. This arm
    // is what stops the two drifting: they must agree to round-off on an unsmoothed field, so a change
    // to either that does not match the other fails here rather than diverging silently.
    {
        const Case c = makeCase(6, scalar(1));
        std::vector<scalar> v(static_cast<std::size_t>(c.m.nCells()));
        for (label i = 0; i < c.m.nCells(); ++i)
            v[i] = scalar(0.3)*c.g.C()[i].x - scalar(0.2)*c.g.C()[i].y + scalar(0.7)*c.g.C()[i].z;
        const GeometricField<scalar> a = fieldFrom(c, v);

        const std::vector<vector> ref = fvc::gaussGrad(a, c.m, c.g, c.fvp);
        std::vector<std::vector<scalar>> ab(c.fvp.size());
        for (std::size_t pi = 0; pi < c.fvp.size(); ++pi) ab[pi] = a.boundary[pi]->value();
        const std::vector<vector> got = gaussGradFromValues(v, ab, c.m, c.g, c.fvp);

        scalar w = 0;
        for (label i = 0; i < c.m.nCells(); ++i)
            w = std::fmax(w, std::fmax(std::fabs(got[i].x - ref[i].x),
                std::fmax(std::fabs(got[i].y - ref[i].y), std::fabs(got[i].z - ref[i].z))));
        std::printf("  worst |gaussGradFromValues - fvc::gaussGrad| = %.3e\n", (double)w);
        check("the two gradients agree exactly, so the smoothed path cannot drift", w <= scalar(1e-14));
    }

    // ---- 1. A FLAT INTERFACE GENERATES NO MEANINGFUL CURVATURE -- AND "no meaningful" IS THE CLAIM.
    //
    // This arm was first written as `K == 0 to round-off` and it failed at 3.6e-07. The code was right
    // and the claim was wrong: nHatfv is gradAlphaf/(mag(gradAlphaf) + deltaN), which is NOT a unit
    // vector -- its magnitude is mag/(mag + deltaN), and that varies across the interface profile. So
    // div of it cannot vanish identically, and what is left is the deltaN stabiliser's own residue.
    //
    // THE RIGHT STATEMENT IS THEREFORE A RATIO AND A SCALING, both measured here:
    //   * the plane's spurious curvature is orders of magnitude below a real one on the same mesh --
    //     that is the parasitic-current claim a VoF code is judged on;
    //   * and it tracks deltaN, which is what identifies it as the stabiliser rather than a defect.
    //     deltaN is 1e-8/cbrt(average V), so COARSENING the mesh SHRINKS it.
    {
        auto flatWorstK = [&](scalar h)
        {
            const Case c = makeCase(8, h);
            std::vector<scalar> v(static_cast<std::size_t>(c.m.nCells()));
            for (label i = 0; i < c.m.nCells(); ++i)
            {
                const scalar y = c.g.C()[i].y / h;          // in cell units, so the profile is fixed
                v[i] = scalar(0.5)*(scalar(1) + std::tanh((y - scalar(4)) / scalar(1.2)));
            }
            const GeometricField<scalar> a = fieldFrom(c, v);

            InterfaceCoeffs ic;
            ic.cAlpha = scalar(1);
            SurfaceScalarField nHatf;
            std::vector<scalar> K;
            calculateK(a, ic, c.m, c.g, c.fvp, /*gradLeastSquares=*/false, nHatf, K);

            scalar worstK = 0, maxN = 0;
            for (label i = 0; i < c.m.nCells(); ++i)
            {
                const vector& C = c.g.C()[i];
                const bool interior = C.x > scalar(1.4)*h && C.x < scalar(6.6)*h
                                   && C.y > scalar(1.4)*h && C.y < scalar(6.6)*h
                                   && C.z > scalar(1.4)*h && C.z < scalar(6.6)*h;
                if (interior) worstK = std::fmax(worstK, std::fabs(K[i]));
            }
            for (scalar s : nHatf.internal) maxN = std::fmax(maxN, std::fabs(s));
            return std::pair<scalar,scalar>{worstK, maxN};
        };

        const auto unit   = flatWorstK(scalar(1));
        const auto coarse = flatWorstK(scalar(4));   // 64x the cell volume -> deltaN 4x smaller
        std::printf("  flat interface: worst |K| = %.3e (h = 1), %.3e (h = 4, deltaN 4x smaller)\n",
                    (double)unit.first, (double)coarse.first);
        std::printf("  ...against 2/R = %.4f for the sphere below, on a comparable mesh\n", (double)(2.0/3.0));

        check("a plane's spurious curvature is at least 5 orders below a real one",
              unit.first < scalar(1e-5) * (scalar(2)/scalar(3)));
        check("...and it SHRINKS with deltaN, which identifies it as the stabiliser's residue",
              coarse.first < scalar(0.5) * unit.first);
        check("...while nHatf itself is non-zero, so the smallness is a cancellation, not an empty field",
              unit.second > scalar(0.5));
    }

    // ---- 2. A SPHERE: THE MAGNITUDE, THE SIGN, AND THE CONVERGENCE --------------------------------
    {
        // alpha1 = 1 inside, 0 outside -- a DROP of phase 1. gradAlpha points inward, nHat points into
        // phase 1, and K = -div(nHat) is POSITIVE. Invert alpha and it must flip.
        auto curvatureAtCentre = [&](label N, bool drop)
        {
            const scalar h = scalar(12) / scalar(N);        // the box is 12 units across either way
            const Case c = makeCase(N, h);
            const vector centre{scalar(6), scalar(6), scalar(6)};
            const scalar R = scalar(3);
            const scalar width = scalar(1.2);

            std::vector<scalar> v(static_cast<std::size_t>(c.m.nCells()));
            for (label i = 0; i < c.m.nCells(); ++i)
            {
                const vector& C = c.g.C()[i];
                const scalar r = std::sqrt((C.x-centre.x)*(C.x-centre.x)
                                         + (C.y-centre.y)*(C.y-centre.y)
                                         + (C.z-centre.z)*(C.z-centre.z));
                const scalar inside = scalar(0.5)*(scalar(1) - std::tanh((r - R)/width));
                v[i] = drop ? inside : (scalar(1) - inside);
            }
            const GeometricField<scalar> a = fieldFrom(c, v);

            InterfaceCoeffs ic; ic.cAlpha = scalar(1);
            SurfaceScalarField nHatf;
            std::vector<scalar> K;
            calculateK(a, ic, c.m, c.g, c.fvp, false, nHatf, K);

            // average K over the cells sitting ON the interface, where |gradAlpha| is largest and the
            // deltaN stabiliser is negligible. Away from it K is damped and means nothing.
            scalar sum = 0; label n = 0;
            for (label i = 0; i < c.m.nCells(); ++i)
            {
                const vector& C = c.g.C()[i];
                const scalar r = std::sqrt((C.x-centre.x)*(C.x-centre.x)
                                         + (C.y-centre.y)*(C.y-centre.y)
                                         + (C.z-centre.z)*(C.z-centre.z));
                if (std::fabs(r - R) < scalar(0.5)*h) { sum += K[i]; ++n; }
            }
            return (n > 0) ? sum/static_cast<scalar>(n) : scalar(0);
        };

        const scalar R = scalar(3), exact = scalar(2)/R;
        const scalar coarse = curvatureAtCentre(16, true);
        const scalar fine   = curvatureAtCentre(24, true);
        const scalar bubble = curvatureAtCentre(24, false);
        std::printf("  sphere R = 3, exact K = 2/R = %.4f\n", (double)exact);
        std::printf("    16^3: K = %+.4f  (error %.1f%%)\n",
                    (double)coarse, (double)(100*std::fabs(coarse-exact)/exact));
        std::printf("    24^3: K = %+.4f  (error %.1f%%)\n",
                    (double)fine,   (double)(100*std::fabs(fine-exact)/exact));
        std::printf("    24^3, phase 1 INVERTED (a bubble): K = %+.4f\n", (double)bubble);

        check("a DROP of phase 1 has POSITIVE curvature", fine > scalar(0));
        check("...and a BUBBLE of it has negative curvature -- the sign that a plane cannot pin",
              bubble < scalar(0));
        check("...equal and opposite to within the discretisation",
              std::fabs(fine + bubble) < scalar(0.1)*exact);
        check("the magnitude is 2/R to better than 20% on the coarse mesh",
              std::fabs(coarse - exact) < scalar(0.20)*exact);
        // THE TOLERANCE ABOVE IS ONLY HONEST IF THE ERROR IS CONVERGING, so that is the real arm.
        check("...and REFINING the mesh reduces the error, so the bound is the scheme's, not this mesh's",
              std::fabs(fine - exact) < std::fabs(coarse - exact));
    }

    // ---- 3. the contact angle: an exact postcondition ---------------------------------------------
    {
        const scalar dN = scalar(1e-8);
        // a wall whose outward normal is +y, and an interface normal coming in at some other angle
        const std::vector<vector> nf{vector{0, 1, 0}, vector{0, 1, 0}, vector{0, 1, 0}};
        std::vector<vector> nHat{
            vector{scalar(0.6), scalar(0.8), 0},
            vector{scalar(-0.5), scalar(0.866025403784), 0},
            vector{scalar(0.3), scalar(0.2), scalar(0.932737905309)}};
        const std::vector<vector> before = nHat;

        for (scalar thetaDeg : {scalar(45), scalar(90), scalar(120)})
        {
            std::vector<vector> n = before;
            const scalar th = thetaDeg * scalar(M_PI) / scalar(180);
            const std::vector<scalar> theta(n.size(), th);
            correctContactAngle(n, nf, theta, dN);

            scalar worst = 0;
            for (std::size_t i = 0; i < n.size(); ++i)
            {
                const scalar dot = n[i].x*nf[i].x + n[i].y*nf[i].y + n[i].z*nf[i].z;
                worst = std::fmax(worst, std::fabs(std::acos(dot) - th));
            }
            std::printf("  theta = %3.0f deg: worst |acos(nHat & nf) - theta| = %.3e rad\n",
                        (double)thetaDeg, (double)worst);
            check("after the correction the interface meets the wall AT theta", worst <= scalar(1e-7));
        }

        // CONTROL: the uncorrected normals do NOT meet the wall at theta, or the arm proves nothing.
        const scalar th45 = scalar(45) * scalar(M_PI) / scalar(180);
        scalar worstBefore = 0;
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            const scalar dot = before[i].x*nf[i].x + before[i].y*nf[i].y + before[i].z*nf[i].z;
            worstBefore = std::fmax(worstBefore, std::fabs(std::acos(dot) - th45));
        }
        check("...which the uncorrected normals do not (control)", worstBefore > scalar(0.1));

        // The correction also writes alpha's own WALL GRADIENT -- it does not only bend the normal
        // used for curvature. A port that stopped at the normal would wet the wall the same way
        // whatever theta says.
        std::vector<vector> n = before;
        const std::vector<scalar> theta(n.size(), th45);
        correctContactAngle(n, nf, theta, dN);
        const std::vector<vector> gradAlphaf(n.size(), vector{0, scalar(3), 0});
        const std::vector<scalar> grad = contactAngleGradient(n, nf, gradAlphaf);
        checkNum("acap.gradient() = (nf & nHat)*mag(gradAlphaf)",
                 grad[0], std::cos(th45) * scalar(3), scalar(1e-6));
        check("...and it depends on theta, so the wall wets differently (control)",
              std::fabs(grad[0] - scalar(3)) > scalar(0.5));
    }

    // ---- 4. smoothing, and the path that is refused ----------------------------------------------
    // ANISOTROPIC CELLS, 2 x 1 x 0.5. fvc::average is AREA-WEIGHTED --
    // surfaceSum(magSf*ssf)/surfaceSum(magSf) -- and on a cube every face has the same area, so the
    // weighting is a no-op and a plain mean passes every arm. Removing the weights was tried on a cube
    // fixture and the gate stayed green. Three distinct face areas is what makes the two differ.
    {
        Case c;
        c.m = boxtest::boxMesh(8, 8, 8, scalar(0), scalar(2), scalar(1), scalar(0.5));
        c.g.build(c.m);
        c.fvp = buildPatches(c.m, c.g);
        // a checkerboard: the noisiest field on this mesh, so a smoother has something to do
        std::vector<scalar> v(static_cast<std::size_t>(c.m.nCells()));
        for (label i = 0; i < c.m.nCells(); ++i)
            v[i] = ((i % 2) == 0) ? scalar(0.4) : scalar(0.6);

        // THE ORACLE AND ITS RIVAL, both one pass, computed here from OpenFOAM's own expression.
        {
            const SurfaceScalarField f = fvc::interpolate(v, c.m, c.g, c.fvp);
            std::vector<scalar> wNum(v.size(), 0), wDen(v.size(), 0);   // area-weighted
            std::vector<scalar> pNum(v.size(), 0), pDen(v.size(), 0);   // plain mean
            for (label fi = 0; fi < c.m.nInternalFaces(); ++fi)
            {
                const scalar A = c.g.magSf()[fi];
                for (label ci : {c.m.owner()[fi], c.m.neighbour()[fi]})
                {
                    wNum[ci] += A*f.internal[fi]; wDen[ci] += A;
                    pNum[ci] += f.internal[fi];   pDen[ci] += scalar(1);
                }
            }
            for (std::size_t pi = 0; pi < c.fvp.size(); ++pi)
                for (label i = 0; i < c.fvp[pi].size; ++i)
                {
                    const label ci = c.fvp[pi].faceCells[i];
                    const scalar A = c.g.magSf()[c.fvp[pi].start + i];
                    wNum[ci] += A*f.boundary[pi][i]; wDen[ci] += A;
                    pNum[ci] += f.boundary[pi][i];   pDen[ci] += scalar(1);
                }
            std::vector<scalar> got = v;
            smoothAlpha(got, 1, c.m, c.g, c.fvp);
            scalar wErr = 0, rivalGap = 0;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                wErr     = std::fmax(wErr,     std::fabs(got[i] - wNum[i]/wDen[i]));
                rivalGap = std::fmax(rivalGap, std::fabs(wNum[i]/wDen[i] - pNum[i]/pDen[i]));
            }
            std::printf("  smoothing: |brae - area-weighted| = %.3e, area-weighted vs plain mean = %.3e\n",
                        (double)wErr, (double)rivalGap);
            check("smoothAlpha is fvc::average -- AREA-WEIGHTED", wErr <= scalar(1e-15));
            check("...and on anisotropic cells the plain mean is a different number (control)",
                  rivalGap > scalar(1e-4));
        }

        auto spread = [](const std::vector<scalar>& x)
        {
            scalar mn = x[0], mx = x[0];
            for (scalar s : x) { mn = std::fmin(mn, s); mx = std::fmax(mx, s); }
            return mx - mn;
        };
        const scalar before = spread(v);
        std::vector<scalar> s1 = v, s3 = v;
        smoothAlpha(s1, 1, c.m, c.g, c.fvp);
        smoothAlpha(s3, 3, c.m, c.g, c.fvp);
        std::printf("  checkerboard spread: %.4f -> %.4f (1 pass) -> %.4f (3 passes)\n",
                    (double)before, (double)spread(s1), (double)spread(s3));
        check("one smoothing pass reduces the spread", spread(s1) < before);
        check("...and three reduce it further", spread(s3) < spread(s1));
        check("zero passes leaves the field untouched",
              [&]{ std::vector<scalar> z = v; smoothAlpha(z, 0, c.m, c.g, c.fvp); return z == v; }());

        // ...and the combination brae will not run: NO OpenFOAM tutorial sets nAlphaSmoothCurvature at
        // all, so there is no case to validate leastSquares-plus-smoothing against.
        const GeometricField<scalar> a = fieldFrom(c, v);
        InterfaceCoeffs ic; ic.cAlpha = scalar(1); ic.nAlphaSmoothCurvature = 2;
        SurfaceScalarField nHatf;
        std::vector<scalar> K;
        bool threw = false;
        try { calculateK(a, ic, c.m, c.g, c.fvp, /*gradLeastSquares=*/true, nHatf, K); }
        catch (const std::exception&) { threw = true; }
        check("smoothing + a leastSquares gradient is refused by name", threw);
        threw = false;
        try { calculateK(a, ic, c.m, c.g, c.fvp, /*gradLeastSquares=*/false, nHatf, K); }
        catch (const std::exception&) { threw = true; }
        check("...while smoothing with the Gauss gradient runs (control)", !threw);
    }

    std::printf("test_interface_curvature_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
