// A wedge on the DEVICE: it borrows the mixed (Robin) slot, and must not be swept along with the real
// mixed patches.
//
// brae runs a vector wedge through device category 5 because OF's wedgeFvPatchField coefficients ARE a
// Robin blend: valueInternalCoeffs = 1 - d_k, gradientInternalCoeffs = -deltaCoeffs*d_k, with the
// PER-COMPONENT valueFraction d_k = 0.5*(1 - cellT_kk). On the symmetry axis d_k is exactly 0 -- that
// component is pure zeroGradient and must contribute NOTHING to the matrix.
//
// The trap is that category 5 also means "freestream" to deviceUpdateMixedFreestream, which recomputes
// the valueFraction of every masked face from the sign of the boundary flux. A wedge caught by that
// updater has its (0, 1.9e-3, 1.9e-3) overwritten with a flat 1 on all three components -- and a wedge
// with valueFraction 1 is a fixedValue WALL on both wedge planes, not an axisymmetric constraint.
//
// WHY IT MATTERS. The wedge planes of an axisymmetric mesh sit a distance r*sin(theta) from the cell
// centre, so their deltaCoeffs go as 1/r and their laplacian diagonal nuEff*magSf*deltaCoeffs/V goes as
// 1/r^2. Turning that on is not a small error near the axis: on pimpleFoam/laminar/movingCone it made
// UEqn.A() 17x too large in the first radial row, so rAU -- and with it U = HbyA - rAU*grad(p) -- came
// out SEVEN TIMES short of OpenFOAM's on a pressure field that agreed to 2%. The whole-field error went
// 8.5e-02 -> 3.6e-03 when the wedge left the mixed mask.
//
// Leg 2 is the regression proper and Leg 3 is its discrimination: a genuine freestream patch in the same
// fixture MUST still be updated, so "stop calling the updater" cannot pass this file. That patch is built
// as the reader builds freestreamVelocity and freestreamPressure (freestream = true). Before the device
// kept a freestream mask of its own it was built without the flag, a plain `mixed` by brae's own type
// model, and Leg 3 passed only because the updater swept every mixed face.
//
// Leg 3b: a plain `mixed` (wallYmin) keeps the valueFraction its dictionary stated through the same
// update. mixedFvPatchField::updateCoeffs does not change it; only freestreamVelocity (.C:106) and
// freestreamPressure (.C:119) recompute theirs. Control BRAE_CONTROL_MIXED_VF_FREESTREAM=1 (the device
// sweeps every mixed face again) turns it red.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "wedge_patch.cuh"
#include "device_buffer.cuh"
#include "device_boundary.cuh"
#include "box_mesh.cuh"
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
    std::printf("== wedge coefficients on the device (mixed slot, not a mixed patch) ==\n");

    PrimitiveMesh m = boxtest::boxMesh(4, 4, 3);
    FvGeometry g; g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // OF's recommended 2.5 degree half angle, about x, straddling the x-y plane -- the movingCone
    // geometry. faceT/cellT come from the real wedgeGeometry so d_k is the real thing, not a stand-in.
    const scalar th = 2.5*M_PI/180.0;
    WedgeGeometry w;
    {
        FvPatch probe;
        probe.name = "back"; probe.type = "wedge"; probe.size = 1; probe.start = 0;
        probe.faceCells = {0}; probe.deltaCoeffs = {2.0}; probe.magSf = {1.0};
        probe.nf = { vector{0, std::sin(th), std::cos(th)} };
        probe.Cf = { vector{0, 0, 0} };
        w = wedgeGeometry(probe);
    }
    const scalar dK[3] = { scalar(0.5)*(1 - w.cellT.xx),
                           scalar(0.5)*(1 - w.cellT.yy),
                           scalar(0.5)*(1 - w.cellT.zz) };

    GeometricField<vector> U;  U.internal.resize(nC);
    GeometricField<vector> H;  H.internal.resize(nC);
    GeometricField<scalar> p;  p.internal.resize(nC);
    for (label c = 0; c < nC; ++c)
    {
        U.internal[c] = {1.0 + 0.01*c, 0.5, -0.25};
        H.internal[c] = {100.0 + c, 200.0 + c, 300.0 + c};   // nothing like U: leg 4 needs them distinct
        p.internal[c] = 10.0 + c;
    }

    std::size_t wedgePatch = 0, freePatch = 1, plainPatch = 2;
    const scalar plainVfU = 0.3;
    const scalar plainVfP = 0.6;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& q = fvp[pi];
        if (pi == wedgePatch)
        {
            fvp[pi].type = "wedge";
            U.boundary.push_back(std::make_unique<WedgePatchField<vector>>(q, w.faceT, w.cellT));
            H.boundary.push_back(std::make_unique<WedgePatchField<vector>>(q, w.faceT, w.cellT));
            p.boundary.push_back(std::make_unique<WedgePatchField<scalar>>(q, w.faceT, w.cellT));
        }
        else if (pi == freePatch)
        {
            U.boundary.push_back(std::make_unique<MixedPatchField<vector>>(q, true, vector{9, 9, 9},
                                                                          std::vector<vector>{}, true,
                                                                          /*freestream=*/true));
            H.boundary.push_back(std::make_unique<MixedPatchField<vector>>(q, true, vector{9, 9, 9},
                                                                          std::vector<vector>{}, true,
                                                                          /*freestream=*/true));
            p.boundary.push_back(std::make_unique<MixedPatchField<scalar>>(q, true, scalar(0),
                                                                          std::vector<scalar>{}, false,
                                                                          /*freestream=*/true));
        }
        else if (pi == plainPatch)
        {
            // As the reader builds a plain `mixed`: velocitySign true, the case's valueFraction set.
            auto mu = std::make_unique<MixedPatchField<vector>>(q, true, vector{9, 9, 9},
                                                                std::vector<vector>{}, true);
            mu->setValueFraction(std::vector<scalar>(static_cast<std::size_t>(q.size), plainVfU));
            U.boundary.push_back(std::move(mu));
            auto mh = std::make_unique<MixedPatchField<vector>>(q, true, vector{9, 9, 9},
                                                                std::vector<vector>{}, true);
            mh->setValueFraction(std::vector<scalar>(static_cast<std::size_t>(q.size), plainVfU));
            H.boundary.push_back(std::move(mh));
            auto mp = std::make_unique<MixedPatchField<scalar>>(q, true, scalar(0),
                                                                std::vector<scalar>{}, true);
            mp->setValueFraction(std::vector<scalar>(static_cast<std::size_t>(q.size), plainVfP));
            p.boundary.push_back(std::move(mp));
        }
        else
        {
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
            H.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
            p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        }
    }
    U.evaluateBoundary();
    H.evaluateBoundary();
    p.evaluateBoundary();

    DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    DeviceBoundary       dbP = buildDeviceBoundary(p, fvp, g);

    std::vector<label> patchOf, cellOf;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i) { patchOf.push_back((label)pi); cellOf.push_back(fvp[pi].faceCells[i]); }
    const int nB = (int)patchOf.size();

    DeviceBuffer<scalar> Uc[3], Hc[3];
    for (int k = 0; k < 3; ++k)
    {
        std::vector<scalar> u(nC), h(nC);
        for (label c = 0; c < nC; ++c) { u[c] = component(U.internal[c], k); h[c] = component(H.internal[c], k); }
        Uc[k].copyFrom(u); Hc[k].copyFrom(h);
    }
    auto vfHost = [&](int k) { std::vector<scalar> v; dbU.comp[k].valueFraction.copyTo(v); return v; };

    // ---- Leg 1: the wedge is out of the mixed mask, the freestream patch is in --------------------
    {
        std::vector<label> mask;
        dbU.comp[0].mixedMask.copyTo(mask);
        bool wedgeOut = true, freeIn = true;
        int nWedge = 0, nFree = 0;
        for (int i = 0; i < nB; ++i)
        {
            if (patchOf[i] == (label)wedgePatch) { ++nWedge; wedgeOut = wedgeOut && (mask[i] == 0); }
            if (patchOf[i] == (label)freePatch)  { ++nFree;  freeIn  = freeIn  && (mask[i] == 1); }
        }
        check(nWedge > 0 && nFree > 0, "vacuity guard: the fixture has both a wedge and a freestream patch");
        check(wedgeOut, "a wedge face is NOT in the mixed mask, though it reports category 5");
        check(freeIn,   "...and a real freestream face still is");
    }

    // ---- Leg 2: the geometric valueFraction survives the per-step freestream update ---------------
    // This is the regression. Note d_x is EXACTLY zero (the symmetry axis) -- the component that must
    // contribute no matrix coefficient at all, and the one an overwrite to 1 destroys most violently.
    {
        const std::vector<scalar> was[3] = { vfHost(0), vfHost(1), vfHost(2) };
        DeviceBuffer<scalar> phiB;
        {
            std::vector<scalar> f(nB);
            for (int i = 0; i < nB; ++i) f[i] = (i % 2) ? scalar(-1) : scalar(1);   // both flux signs present
            phiB.copyFrom(f);
        }
        deviceUpdateMixedFreestream(dbU, dbP, phiB, Uc[0], Uc[1], Uc[2]);
        const std::vector<scalar> now[3] = { vfHost(0), vfHost(1), vfHost(2) };

        scalar drift = 0, offAxis = 0;
        bool axisZero = true;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] == (label)wedgePatch)
                for (int k = 0; k < 3; ++k)
                {
                    drift = std::max(drift, std::fabs(now[k][i] - dK[k]));
                    if (k == 0) axisZero = axisZero && (now[k][i] == scalar(0));
                    else        offAxis  = std::max(offAxis, now[k][i]);
                }
        check(drift < 1e-15, "the wedge keeps its geometric valueFraction 0.5*(1 - cellT_kk) after the update");
        check(axisZero, "...with the axis component still EXACTLY zero");
        check(offAxis > 1e-4 && offAxis < 1e-2, "vacuity guard: the off-axis d_k is the small 2.5 deg value (~1.9e-3)");
        (void)was;

        // ---- Leg 3: discrimination -- the real freestream patch WAS updated -----------------------
        // Without this, deleting the deviceUpdateMixedFreestream call entirely would pass leg 2.
        scalar moved = 0;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] == (label)freePatch)
                for (int k = 0; k < 3; ++k) moved = std::max(moved, std::fabs(now[k][i] - was[k][i]));
        check(moved > 0.1, "discrimination: the freestream patch's valueFraction IS recomputed from the flux");

        // ---- Leg 3b: a plain `mixed` keeps the valueFraction its dictionary stated ---------------
        std::vector<scalar> vfP;
        dbP.valueFraction.copyTo(vfP);
        int nPlain = 0;
        scalar plainMoved = 0;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] == (label)plainPatch)
            {
                ++nPlain;
                for (int k = 0; k < 3; ++k) plainMoved = std::max(plainMoved, std::fabs(now[k][i] - plainVfU));
                plainMoved = std::max(plainMoved, std::fabs(vfP[i] - plainVfP));
            }
        check(nPlain > 0, "vacuity guard: the fixture has a plain mixed patch");
        check(plainMoved == 0.0, "a plain mixed U and p keep their stated valueFraction through the update");
    }

    // ---- Leg 4: the resulting laplacian coefficients are per-component, not a wall ----------------
    // iC = -gamma*magSf*deltaCoeffs*d_k. The axis component must be identically 0 while the other two
    // are not: that asymmetry IS the axisymmetric constraint, and a valueFraction of 1 erases it by
    // making all three equal (and larger by ~1/d, i.e. ~500x).
    {
        DeviceBuffer<scalar> gamma;
        gamma.copyFrom(std::vector<scalar>(nC, scalar(1e-5)));
        scalar wx = 0, wy = 0, wz = 0, fixedSpread = 0;
        std::vector<scalar> iCh[3];
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> iC, bC;
            deviceBCLaplacianCoeffs(dbU.comp[k], gamma, iC, bC);
            iC.copyTo(iCh[k]);
        }
        for (int i = 0; i < nB; ++i)
        {
            if (patchOf[i] == (label)wedgePatch)
            {
                wx = std::max(wx, std::fabs(iCh[0][i]));
                wy = std::max(wy, std::fabs(iCh[1][i]));
                wz = std::max(wz, std::fabs(iCh[2][i]));
            }
            else if (patchOf[i] == (label)freePatch)   // a real mixed patch: same coefficient on all three
                fixedSpread = std::max(fixedSpread, std::fabs(iCh[0][i] - iCh[1][i]));
        }
        check(wx == 0.0, "wedge: the axis component contributes NO laplacian diagonal");
        check(wy > 0.0 && std::fabs(wy - wz) < 1e-30*std::max(wy, scalar(1)) + 1e-18,
              "...and the two off-axis components contribute the same small one");
        check(fixedSpread < 1e-30, "discrimination: a real mixed patch is component-independent, as it should be");
    }

    // ---- Leg 5: HbyA's wedge value is HbyA's OWN rotation, not U's -------------------------------
    // OF's fvMatrix::H() inherits psi's BC types and rotates ITSELF; rAU's wedge is the scalar wedge
    // (= zeroGradient), so HbyA_b = faceT & HbyA_cell. Evaluating HbyA through U's refValue instead
    // leaves (faceT & U_c) - (faceT & HbyA_c) on every wedge face -- a spurious flux the pressure
    // equation answers with an equally spurious inflow somewhere else.
    {
        deviceUpdateWedge(dbU, Uc[0], Uc[1], Uc[2]);        // refValue built from U, as the solver does
        DeviceBuffer<scalar> hb[3];
        for (int k = 0; k < 3; ++k) deviceBCValue(dbU.comp[k], Hc[k], hb[k]);
        std::vector<scalar> before[3];
        for (int k = 0; k < 3; ++k) hb[k].copyTo(before[k]);

        auto rotated = [&](label c, int k)
        {
            const vector& v = H.internal[c];
            const scalar t[3][3] = {{w.faceT.xx, w.faceT.xy, w.faceT.xz},
                                    {w.faceT.yx, w.faceT.yy, w.faceT.yz},
                                    {w.faceT.zx, w.faceT.zy, w.faceT.zz}};
            return t[k][0]*v.x + t[k][1]*v.y + t[k][2]*v.z;
        };
        scalar wrong = 0;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] == (label)wedgePatch)
                for (int k = 0; k < 3; ++k) wrong = std::max(wrong, std::fabs(before[k][i] - rotated(cellOf[i], k)));
        check(wrong > 1.0, "negative control: blending HbyA against U's refValue is NOT HbyA's own rotation");

        deviceWedgeFaceValue(dbU, Hc[0], Hc[1], Hc[2], hb[0], hb[1], hb[2]);
        std::vector<scalar> after[3];
        for (int k = 0; k < 3; ++k) hb[k].copyTo(after[k]);

        scalar err = 0, elsewhereMoved = 0;
        for (int i = 0; i < nB; ++i)
            for (int k = 0; k < 3; ++k)
            {
                if (patchOf[i] == (label)wedgePatch) err = std::max(err, std::fabs(after[k][i] - rotated(cellOf[i], k)));
                else elsewhereMoved = std::max(elsewhereMoved, std::fabs(after[k][i] - before[k][i]));
            }
        check(err < 1e-12, "HbyA_b on a wedge face == faceT & HbyA_cell");
        check(elsewhereMoved == 0.0, "...and no other patch is touched");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
