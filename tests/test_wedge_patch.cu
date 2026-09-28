// The `wedge` constraint patch -- OF wedgePolyPatch::calcGeometry + wedgeFvPatchField.
//
// A wedge is how OpenFOAM does AXISYMMETRY: one cell thick azimuthally, bounded by two planes at a small
// angle either side of a coordinate plane, coupled by a rotation about the symmetry axis. The patch type
// IS the physics -- there is no user choice in it -- so every number below comes from the geometry:
//
//   centreNormal  the coordinate direction n leans against, componentwise sign(n_i)*(max(|n_i|,0.5)-0.5)
//   axis          normalise(centreNormal ^ n)
//   faceT         rotationTensor(centreNormal, n)   -- the HALF angle
//   cellT         faceT & faceT                     -- the FULL angle, plane to plane
//
// SCALARS ARE NOT ROTATED, and that is OF's own specialisation rather than a simplification here:
// wedgeFvPatchField<scalar> returns snGrad 0, value = patchInternalField and snGradTransformDiag 0. Leg 3
// pins it, because treating a scalar wedge as anything other than zeroGradient would silently change
// every pressure and turbulence boundary in an axisymmetric case.
//
// LEG 5 IS THE ONE THE DEVICE PATH RESTS ON. brae runs a vector wedge through the MIXED (Robin) slot
// with valueFraction d_k = 0.5*(1 - cellT_kk), which reproduces OF's valueInternalCoeffs (1 - d_k) and
// gradientInternalCoeffs (-deltaCoeffs*d_k) exactly. That is only legitimate if the blend
// d_k*ref_k + (1 - d_k)*u_k can be made to equal the rotated value (faceT & u)_k -- which is what the
// refValue the device kernel computes has to satisfy, including on the axis component where d_k is 0.
#include "wedge_patch.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
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

// A wedge patch whose plane is tilted by the half-angle `th` about the x axis, away from the x-y plane.
// One face is enough for the geometry, which is an average over the patch.
FvPatch wedgePatch(scalar th, const std::string& name = "back")
{
    FvPatch p;
    p.name = name;
    p.type = "wedge";
    p.size = 1;
    p.start = 0;
    p.faceCells = {0};
    p.deltaCoeffs = {2.0};
    p.nf = { vector{0, std::sin(th), std::cos(th)} };
    p.magSf = {1.0};
    p.Cf = { vector{0, 0, 0} };
    return p;
}

vector rotate(const tensor& T, const vector& v)
{
    return vector{T.xx*v.x + T.xy*v.y + T.xz*v.z,
                  T.yx*v.x + T.yy*v.y + T.yz*v.z,
                  T.zx*v.x + T.zy*v.y + T.zz*v.z};
}
} // namespace

int main()
{
    std::printf("== wedge (axisymmetric constraint patch) ==\n");

    const scalar th = 2.5*M_PI/180.0;          // OF's recommended half angle
    const WedgeGeometry w = wedgeGeometry(wedgePatch(th));

    // ---- Leg 1: the geometry OF derives ------------------------------------------------------------
    {
        check(std::fabs(w.centreNormal.z - 1.0) < 1e-14 && std::fabs(w.centreNormal.x) < 1e-14
           && std::fabs(w.centreNormal.y) < 1e-14,
              "centreNormal is the coordinate direction the wedge straddles (+z)");
        check(std::fabs(w.cosAngle - std::cos(th)) < 1e-14, "cosAngle is the cosine of the HALF angle");
        check(std::fabs(std::fabs(w.axis.x) - 1.0) < 1e-12 && std::fabs(w.axis.y) + std::fabs(w.axis.z) < 1e-12,
              "the symmetry axis is x, normal to both the centre plane and the patch");
    }

    // ---- Leg 2: faceT rotates by the half angle, cellT by the full one -----------------------------
    // faceT must take centreNormal exactly onto n -- that is its definition -- and cellT must take it
    // twice as far, onto the OTHER wedge plane. A cellT built as anything but faceT&faceT would still
    // look like a rotation and fail here.
    {
        const vector fn = rotate(w.faceT, w.centreNormal);
        const vector cn = rotate(w.cellT, w.centreNormal);
        const vector wantF{0, std::sin(th), std::cos(th)};
        const vector wantC{0, std::sin(2*th), std::cos(2*th)};
        check(mag(fn - wantF) < 1e-14, "faceT takes the centre normal onto the patch normal (half angle)");
        check(mag(cn - wantC) < 1e-13, "cellT takes it to the opposite plane (full angle)");
        // ...and the axis component is untouched by either
        const vector ax = rotate(w.cellT, vector{1, 0, 0});
        check(mag(ax - vector{1, 0, 0}) < 1e-14, "the rotation leaves the axis direction alone");
    }

    // ---- Leg 3: a SCALAR wedge is exactly zeroGradient ---------------------------------------------
    {
        const FvPatch p = wedgePatch(th);
        WedgePatchField<scalar> f(p, w.faceT, w.cellT);
        const std::vector<scalar> internal{7.5};
        f.evaluate(internal);
        check(std::fabs(f.value()[0] - 7.5) < 1e-15, "a scalar wedge takes the cell value unchanged");
        check(f.bcCategory() == 0, "...and is handed to the device as its zeroGradient category");
        check(!f.fixesValue(), "...and does not fix the value");
    }

    // ---- Leg 4: a VECTOR wedge is the rotated cell value -------------------------------------------
    {
        const FvPatch p = wedgePatch(th);
        WedgePatchField<vector> f(p, w.faceT, w.cellT);
        const vector u{3.0, -1.0, 0.5};
        f.evaluate(std::vector<vector>{u});
        check(mag(f.value()[0] - rotate(w.faceT, u)) < 1e-14, "a vector wedge value is faceT & U_cell");
        check(f.bcCategory() == 5, "...and is handed to the device as a MIXED (Robin) patch");
        // discrimination: the rotation is not the identity on the non-axis components
        check(mag(f.value()[0] - u) > 1e-4, "vacuity guard: the rotation actually changes the value");
    }

    // ---- Leg 5: the mixed mapping reproduces the rotation exactly -----------------------------------
    // d_k = 0.5*(1 - cellT_kk); the device solves ref_k from d*ref + (1-d)*u = (faceT & u)_k, and on the
    // axis component d_k is exactly 0, where the blend is already pure zeroGradient -- so ref_k there is
    // multiplied by zero and must not be divided by it.
    {
        const vector u{3.0, -1.0, 0.5};
        const vector want = rotate(w.faceT, u);
        const scalar cT[3] = {w.cellT.xx, w.cellT.yy, w.cellT.zz};
        const scalar uk[3] = {u.x, u.y, u.z};
        const scalar wk[3] = {want.x, want.y, want.z};
        scalar worst = 0;
        int axisComponents = 0;
        for (int k = 0; k < 3; ++k)
        {
            const scalar d = 0.5*(1.0 - cT[k]);
            const scalar ref = (d > 1e-30) ? (wk[k] - (1.0 - d)*uk[k])/d : 0.0;
            const scalar blended = d*ref + (1.0 - d)*uk[k];
            worst = std::max(worst, std::fabs(blended - wk[k]));
            if (d <= 1e-30) ++axisComponents;
        }
        check(worst < 1e-13, "the mixed blend d*ref + (1-d)*u reproduces (faceT & u) on every component");
        check(axisComponents == 1, "vacuity guard: exactly one component (the axis) has d = 0");
    }

    // ---- Leg 7: a mesh change carries the cached patch internal field -------------------------------
    // OpenFOAM's wedgeFvPatchField holds nothing: its coefficient methods call patchInternalField() on the
    // spot. brae's take no arguments and read a cache filled at evaluate(), so the map has to refill it --
    // which it does by RECOMPUTING from the `internal` it is handed, already the new mapped cell field.
    // Without that, a REFINED wedge is caught by the coefficient guard and aborts with "before it was ever
    // evaluated" -- misleading, since it had been evaluated, on the old mesh. The same applies to
    // symmetryPlane, whose cache has no guard at all: it reads a silent zero past the end instead.
    {
        FvPatch p = wedgePatch(th);
        WedgePatchField<vector> f(p, w.faceT, w.cellT);
        const vector u{3.0, -1.0, 0.5};
        f.evaluate(std::vector<vector>{u});

        FvPatchFieldMapping pm;
        pm.direct = true;
        pm.directAddressing = {0, 0};          // one face refined into two coplanar children
        FvPatch two = wedgePatch(th);
        two.size = 2;
        two.faceCells = {0, 1};
        two.deltaCoeffs = {2.0, 2.0};
        two.nf.push_back(two.nf[0]);
        two.magSf = {0.5, 0.5};
        two.Cf.push_back(two.Cf[0]);
        p = two;                                // the field holds a reference
        const vector u2{-2.0, 4.0, 1.25};
        f.autoMap(pm, std::vector<vector>{u, u2});
        bool threw = false;
        std::vector<vector> gbc;
        try { gbc = f.gradientBoundaryCoeffs(); } catch (const std::exception&) { threw = true; }
        check(!threw, "a refined wedge can still be asked for its boundary coefficients");
        check(!threw && gbc.size() == 2, "...one per new face");
        // the second face's cell is u2, so its coefficient must be built from u2 and not from a zero or
        // from face 0's cell: snGrad - gradientInternalCoeffs*pif, per component
        const vector r2 = rotate(w.cellT, u2);
        const vector sn2{0.5*2.0*(r2.x - u2.x), 0.5*2.0*(r2.y - u2.y), 0.5*2.0*(r2.z - u2.z)};
        const scalar dd[3] = {0.5*(1.0 - w.cellT.xx), 0.5*(1.0 - w.cellT.yy), 0.5*(1.0 - w.cellT.zz)};
        const vector want2{sn2.x + dd[0]*2.0*u2.x, sn2.y + dd[1]*2.0*u2.y, sn2.z + dd[2]*2.0*u2.z};
        check(!threw && gbc.size() == 2 && mag(gbc[1] - want2) < 1e-13,
              "...and the ADDED face's coefficient is built from its OWN cell");
        check(mag(want2) > 1e-4, "vacuity guard: the added face's coefficient is not zero anyway");
    }

    // ---- Leg 8: symmetryPlane's cache under the same change, where the failure is SILENT ------------
    // The wedge aborts on a short cache (leg 7). symmetryPlane reads it through pifAt, which returns a
    // static zero past the end -- so a grown patch would build every ADDED face's coefficients against a
    // cell value of zero and say nothing. Its map recomputes the cache for the same reason.
    // MEASURED without the recompute: the added face's gradientBoundaryCoeffs read (0 0 0).
    {
        FvPatch p;
        p.name = "sym";
        p.type = "symmetryPlane";
        p.size = 1;
        p.faceCells = {0};
        p.deltaCoeffs = {2.0};
        // TILTED on purpose: on an axis-aligned plane the two terms of the coefficient cancel exactly
        // (-dc*n_k*(n.pif) against -(-dc*|n_k|)*pif_k), so an axis-aligned arm reads zero whatever the
        // cache holds and cannot witness anything.
        p.nf = { vector{0, 0.6, 0.8} };
        p.magSf = {1.0};
        p.Cf = { vector{0, 0, 0} };
        SymmetryPlanePatchField<vector> f(p);
        const vector u{3.0, -1.0, 0.5};
        f.evaluate(std::vector<vector>{u});

        FvPatchFieldMapping pm;
        pm.direct = true;
        pm.directAddressing = {0, 0};
        FvPatch two = p;
        two.size = 2;
        two.faceCells = {0, 1};
        two.deltaCoeffs = {2.0, 2.0};
        two.nf.push_back(two.nf[0]);
        two.magSf = {0.5, 0.5};
        two.Cf.push_back(two.Cf[0]);
        p = two;
        const vector u2{-2.0, 4.0, 1.25};
        f.autoMap(pm, std::vector<vector>{u, u2});
        const std::vector<vector> gbc = f.gradientBoundaryCoeffs();
        // -n(n.pif)*deltaCoeffs, minus gradientInternalCoeffs*pif per component
        const vector n2 = p.nf[1];
        const scalar nd = n2.x*u2.x + n2.y*u2.y + n2.z*u2.z;
        const vector sn{-nd*n2.x*2.0, -nd*n2.y*2.0, -nd*n2.z*2.0};
        const vector want2{sn.x + 2.0*std::fabs(n2.x)*u2.x,
                           sn.y + 2.0*std::fabs(n2.y)*u2.y,
                           sn.z + 2.0*std::fabs(n2.z)*u2.z};
        check(gbc.size() == 2, "a refined symmetryPlane gives one coefficient per new face");
        check(gbc.size() == 2 && mag(gbc[1] - want2) < 1e-13,
              "...and the ADDED face's is built from its OWN cell, not a silent zero");
        check(mag(want2) > 1e-4, "vacuity guard: it is not zero anyway");
    }

    // ---- Leg 6: the degeneracies OF refuses --------------------------------------------------------
    {
        bool threw = false;
        try { wedgeGeometry(wedgePatch(0.0)); } catch (const std::exception&) { threw = true; }
        check(threw, "refusal: a patch lying IN a coordinate plane has no wedge angle");
    }
    {
        // a normal at 45 degrees leans against no single coordinate direction: OF refuses this too
        FvPatch p = wedgePatch(0.1);
        p.nf = { vector{0, std::sqrt(0.5), std::sqrt(0.5)} };
        bool threw = false;
        try { wedgeGeometry(p); } catch (const std::exception&) { threw = true; }
        check(threw, "refusal: a wedge that does not straddle a coordinate plane");
    }
    {
        // ...and the negative control: a legitimate small angle is accepted
        bool threw = false;
        try { wedgeGeometry(wedgePatch(5.0*M_PI/180.0)); } catch (const std::exception&) { threw = true; }
        check(!threw, "negative control: an ordinary 5 degree wedge is accepted");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
