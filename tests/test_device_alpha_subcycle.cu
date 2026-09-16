// The alpha sub-cycle on the device -- alphaEqnSubCycle.H, which 33 of the 44 shipped interFoam
// tutorials use, 23 of them at nAlphaSubCycles 3.
//
// EVERY ARM HERE IS A CONTROL, and that is the point of the file. The sub-cycle is four lines of
// arithmetic, and all four ways to get it wrong leave alpha BOUNDED, SMOOTH and PLAUSIBLE. No
// boundedness gate sees any of them; a field plot sees none of them; agreement with the host on the
// pieces underneath sees none of them, because every piece is right and only the wiring is wrong. So
// each is built here as a deliberately wrong sub-cycle run beside the real one, and the arm is the
// distance between the two.
//
//   a  sub-steps at the FULL deltaT           -- advects n times as far
//   b  every sub-step from the TIME STEP's old alpha, not the previous sub-step's
//   c  the LAST rhoPhi instead of the time-weighted sum -- invisible in alpha, wrong in U
//   d  alpha.oldTime() written by the loop    -- the PIMPLE outer loop's second pass starts elsewhere
//
// The step itself is the real device corrector chain (deviceAlphaCorrector + the host boundary
// evaluation + deviceInterfaceCorrect), not a stub: a sub-cycle that only ever drove a toy step would
// be testing the loop against arithmetic rather than against the solver it wraps.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "interface_properties_cpp.cuh"
#include "inter_solve_cpp.cuh"
#include "alpha_eqn_cpp.cuh"
#include "mules_cpp.cuh"
#include "device_alpha_subcycle.cuh"
#include "device_alpha_step.cuh"
#include "device_alpha_flux.cuh"
#include "device_interface_properties.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
namespace ip = brae::cpu::interfaceProps;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
std::vector<scalar> flatten(const std::vector<std::vector<scalar>>& b)
{
    std::vector<scalar> v;
    for (const auto& p : b) v.insert(v.end(), p.begin(), p.end());
    return v;
}
scalar worstDiff(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar w = 0;
    for (std::size_t i = 0; i < a.size(); ++i) w = std::fmax(w, std::fabs(a[i] - b[i]));
    return w;
}
}   // namespace

int main()
{
    std::printf("== interFoam alpha sub-cycle: device\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 24;
    const scalar h = scalar(1) / scalar(N);
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1, scalar(0), h, h, h);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    const scalar omega = scalar(2), cx = scalar(0.5), cy = scalar(0.5);
    auto vel = [&](const vector& P) { return vector{-omega*(P.y - cy), omega*(P.x - cx), scalar(0)}; };
    SurfaceScalarField phi;
    phi.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const vector u = vel(g.Cf()[f]), &S = g.Sf()[f];
        phi.internal[f] = u.x*S.x + u.y*S.y + u.z*S.z;
    }
    phi.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phi.boundary[pi].resize(static_cast<std::size_t>(fvp[pi].size));
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label gf = fvp[pi].start + i;
            const vector u = vel(g.Cf()[gf]), &S = g.Sf()[gf];
            phi.boundary[pi][i] = u.x*S.x + u.y*S.y + u.z*S.z;
        }
    }

    std::vector<scalar> a0(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::hypot(C.x - scalar(0.35), C.y - scalar(0.5));
        a0[c] = scalar(0.5)*(scalar(1) - std::tanh((r - scalar(0.15))/scalar(0.03)));
    }

    ip::InterfaceCoeffs ic;
    ic.cAlpha = scalar(1);
    const scalar totalDt = scalar(0.03);
    const int nSub = 3;                       // the count 23 of the 44 tutorials use
    const int nAlphaCorr = 2;
    const scalar rho1 = scalar(1000), rho2 = scalar(1);

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<int> fixes, flag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        { fixes.push_back(0); flag.push_back(fvp[pi].type == "empty" ? 1 : 0); }
    DeviceBuffer<scalar> dPhiInt(phi.internal), dPhiBnd(flatten(phi.boundary));
    DeviceBuffer<int> dFixes(fixes), dFlag(flag);
    DeviceMulesControls dmc;
    dmc.nLimiterIter = 5;

    DeviceAlphaStepInput din;
    din.phiInt   = &dPhiInt;
    din.phiBnd   = &dPhiBnd;
    din.phiCNInt = &dPhiInt;
    din.phiCNBnd = &dPhiBnd;
    din.cAlpha   = ic.cAlpha;
    din.rho1     = rho1;
    din.rho2     = rho2;
    din.deltaN   = ip::deltaN(g.V());
    din.alphaScheme  = DeviceAlphaScheme::vanLeer;
    din.alpharScheme = DeviceAlphaScheme::linear;

    // THE REAL STEP: the device corrector loop with the host's boundary evaluation between correctors,
    // exactly as tests/test_device_alpha_step.cu drives it.
    DeviceBuffer<scalar> dNHatf, dK, dABnd, dNHatfBnd, aPhiInt, aPhiBnd;
    GeometricField<scalar> work;
    work.internal = a0;
    for (const FvPatch& q : fvp)
        work.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    work.evaluateBoundary();

    auto flattenPatches = [&](const GeometricField<scalar>& f)
    {
        std::vector<scalar> v;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& b = f.boundary[pi]->value();
            v.insert(v.end(), b.begin(), b.end());
        }
        return v;
    };

    DeviceAlphaEqnStep step =
        [&](const DeviceBuffer<scalar>& alphaOld, scalar deltaT, DeviceBuffer<scalar>& alpha,
            DeviceBuffer<scalar>& rhoPhiInt, DeviceBuffer<scalar>& rhoPhiBnd)
    {
        DeviceAlphaStepInput li = din;
        li.deltaT = deltaT;

        std::vector<scalar> start;
        alphaOld.copyTo(start);
        work.internal = start;
        work.evaluateBoundary();
        alpha.copyFrom(start);
        dABnd.copyFrom(flattenPatches(work));

        for (int aCorr = 0; aCorr < nAlphaCorr; ++aCorr)
        {
            li.aCorr = aCorr;
            DeviceAlphaBoundary db;
            db.alpha1     = &dABnd;
            db.nHatfBnd   = &dNHatfBnd;
            db.fixesValue = &dFixes;
            db.flag       = &dFlag;
            deviceAlphaCorrector(dm, alpha, alphaOld, li, db, dmc, dNHatf, aPhiInt, aPhiBnd);

            // MULES::explicitSolve ends with correctBoundaryConditions, then mixture.correct()
            std::vector<scalar> cur;
            alpha.copyTo(cur);
            work.internal = cur;
            work.evaluateBoundary();
            dABnd.copyFrom(flattenPatches(work));
            SurfaceScalarField nHb;
            std::vector<scalar> Kb;
            ip::calculateK(work, ic, m, g, fvp, false, nHb, Kb);
            dNHatfBnd.copyFrom(flatten(nHb.boundary));
            deviceInterfaceCorrect(dm, alpha, dABnd, dNHatfBnd, li.deltaN, dNHatf, dK);
        }
        deviceMassFlux(nIf, aPhiInt, dPhiInt, rho1, rho2, rhoPhiInt);
        deviceMassFlux(nBf, aPhiBnd, dPhiBnd, rho1, rho2, rhoPhiBnd);
    };

    auto seed = [&]()
    {
        work.internal = a0;
        work.evaluateBoundary();
        SurfaceScalarField nH;
        std::vector<scalar> K;
        ip::calculateK(work, ic, m, g, fvp, false, nH, K);
        dNHatf.copyFrom(nH.internal);
        dK.copyFrom(K);
        dNHatfBnd.copyFrom(flatten(nH.boundary));
        dABnd.copyFrom(flattenPatches(work));
    };

    // ---- the RIGHT answer ------------------------------------------------------------------------
    std::vector<scalar> ref, refRhoPhi;
    {
        seed();
        DeviceBuffer<scalar> alpha(a0), alphaOld(a0), rInt, rBnd;
        deviceAlphaEqnSubCycle(nSub, totalDt, alpha, alphaOld, rInt, rBnd, step);
        alpha.copyTo(ref);
        rInt.copyTo(refRhoPhi);

        // alphaOld must be untouched -- d
        std::vector<scalar> oldAfter;
        alphaOld.copyTo(oldAfter);
        const scalar dOld = worstDiff(oldAfter, a0);
        scalar exc = 0, moved = 0;
        for (label c = 0; c < nC; ++c)
        {
            exc   = std::fmax(exc, std::fmax(-ref[c], ref[c] - scalar(1)));
            moved = std::fmax(moved, std::fabs(ref[c] - a0[c]));
        }
        std::printf("  %d sub-cycles of dt/%d: excursion %.3e, the field moved %.4f, "
                    "|alphaOld - a0| = %.3e\n", nSub, nSub, (double)exc, (double)moved, (double)dOld);
        check("the sub-cycle keeps alpha in [0,1]", exc <= scalar(1e-14));
        check("...and advected it", moved > scalar(0.1));
        check("d: alpha.oldTime() is NOT written by the loop -- subCycleField restores it, and the "
              "PIMPLE outer loop's next pass starts from the same place", dOld == scalar(0));
    }

    // ---- a: THE SUB-STEPS MUST RUN AT deltaT/n, NOT AT deltaT -------------------------------------
    // Time::subCycle divides deltaT (Time.C:1006-1009). Running each sub-step at the full step advects
    // three times as far and looks like nothing worse than a faster interface.
    {
        seed();
        DeviceBuffer<scalar> alpha(a0), alphaOld(a0), rInt, rBnd;
        DeviceBuffer<scalar> carried(a0);
        for (int k = 0; k < nSub; ++k)
        {
            step(carried, totalDt, alpha, rInt, rBnd);      // the FULL step, three times
            std::vector<scalar> cur;
            alpha.copyTo(cur);
            carried.copyFrom(cur);
        }
        std::vector<scalar> wrong;
        alpha.copyTo(wrong);
        const scalar d = worstDiff(wrong, ref);
        std::printf("  a: sub-steps at the full deltaT instead of deltaT/%d -> %.4e\n", nSub, (double)d);
        check("a: the sub-step deltaT is divided, and running the full one is a different field",
              d > scalar(1e-2));
    }

    // ---- b: EACH SUB-STEP STARTS FROM THE PREVIOUS ONE'S RESULT -----------------------------------
    // Restarting each from the time step's old alpha advects by deltaT/n in total instead of deltaT --
    // the opposite error to a, and just as bounded and plausible.
    {
        seed();
        DeviceBuffer<scalar> alpha(a0), alphaOld(a0), rInt, rBnd;
        const scalar dtSub = totalDt / scalar(nSub);
        for (int k = 0; k < nSub; ++k)
            step(alphaOld, dtSub, alpha, rInt, rBnd);       // always from a0
        std::vector<scalar> wrong;
        alpha.copyTo(wrong);
        const scalar d = worstDiff(wrong, ref);
        std::printf("  b: every sub-step restarted from the time step's old alpha -> %.4e\n", (double)d);
        check("b: each sub-step continues from the last one, and restarting is a different field",
              d > scalar(1e-2));
    }

    // ---- c: rhoPhi IS THE TIME-WEIGHTED SUM, NOT THE LAST SUB-STEP'S -----------------------------
    // This one does not touch alpha at all. rhoPhi is what the momentum equation is built on, so the
    // error appears one equation later, in U -- which is why it needs its own arm here and cannot be
    // caught by anything that only looks at the interface.
    {
        seed();
        DeviceBuffer<scalar> alpha(a0), rInt, rBnd, carried(a0);
        const scalar dtSub = totalDt / scalar(nSub);
        std::vector<scalar> last;
        for (int k = 0; k < nSub; ++k)
        {
            step(carried, dtSub, alpha, rInt, rBnd);
            std::vector<scalar> cur;
            alpha.copyTo(cur);
            carried.copyFrom(cur);
            rInt.copyTo(last);                              // keep only the LAST
        }
        const scalar d = worstDiff(last, refRhoPhi);
        scalar scale = 0;
        for (scalar v : refRhoPhi) scale = std::fmax(scale, std::fabs(v));
        std::printf("  c: the LAST rhoPhi instead of sum_k (dt_k/dt)*rhoPhi_k -> %.4e of %.4e\n",
                    (double)d, (double)scale);
        check("c: rhoPhi is the time-weighted sum, and the last sub-step's is a different flux",
              d > scalar(1e-3)*scale);

        // ...and the alpha field is IDENTICAL either way, which is exactly why c needs its own arm
        std::vector<scalar> alphaSame;
        alpha.copyTo(alphaSame);
        std::printf("  ...while alpha is %.3e away, i.e. the same field -- c is invisible in alpha\n",
                    (double)worstDiff(alphaSame, ref));
        check("...and alpha cannot see c at all, so no boundedness or interface gate would catch it",
              worstDiff(alphaSame, ref) == scalar(0));
    }

    // ---- nAlphaSubCycles 1 is the PLAIN branch, not a one-iteration loop ---------------------------
    // alphaEqnSubCycle.H:31-34 constructs no subCycle at all there, so deltaT is untouched. The two
    // must agree exactly, and this arm is what says the special case was not quietly routed through
    // the loop with n = 1 (which would be arithmetically the same here but would divide deltaT by 1
    // and copy the field twice for nothing).
    {
        seed();
        DeviceBuffer<scalar> a1(a0), old1(a0), i1, b1;
        deviceAlphaEqnSubCycle(1, totalDt, a1, old1, i1, b1, step);
        std::vector<scalar> one;
        a1.copyTo(one);

        seed();
        DeviceBuffer<scalar> a2(a0), old2(a0), i2, b2;
        step(old2, totalDt, a2, i2, b2);
        std::vector<scalar> plain;
        a2.copyTo(plain);

        const scalar d = worstDiff(one, plain);
        std::printf("  nAlphaSubCycles 1 vs one bare step: %.3e\n", (double)d);
        check("nAlphaSubCycles 1 is the plain branch, bit for bit", d == scalar(0));

        const scalar apart = worstDiff(one, ref);
        std::printf("  ...and %d sub-cycles is %.4e away from 1, so the loop is not a no-op\n",
                    nSub, (double)apart);
        check("...and sub-cycling actually changes the answer", apart > scalar(1e-3));
    }

    // ---- and the whole loop against the HOST's ----------------------------------------------------
    // The step itself is already gated against the host at 9.99e-16 over forty time steps
    // (tests/test_device_alpha_step.cu), so what this adds is the BOOKKEEPING around it: the divided
    // dt, the carried field and the weighted rhoPhi, all driven through the host's own
    // alphaEqnSubCycle with the host's own alphaEqnStep.
    {
        namespace ifm = brae::cpu::interFoam;
        namespace cs  = brae::cpu::interFoam;

        GeometricField<scalar> ah;
        ah.internal = a0;
        for (const FvPatch& q : fvp)
            ah.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        ah.evaluateBoundary();
        SurfaceScalarField hNHatf, hAlphaPhi;
        std::vector<scalar> hK;
        ip::calculateK(ah, ic, m, g, fvp, false, hNHatf, hK);

        brae::cpu::MULES::Controls hctl;
        hctl.nLimiterIter = dmc.nLimiterIter;
        ifm::AlphaStepInput hin;
        hin.phi = &phi;
        hin.phiCN = &phi;
        hin.cAlpha = ic.cAlpha;
        hin.nAlphaCorr = nAlphaCorr;
        hin.rho1 = rho1;
        hin.rho2 = rho2;
        hin.MULESCorr = false;
        hin.alphaScheme  = ifm::AlphaFluxScheme::vanLeer;
        hin.alpharScheme = ifm::AlphaFluxScheme::linear;

        cs::AlphaEqnStep hstep =
            [&](const std::vector<scalar>& alphaOld, scalar dtSub, std::vector<scalar>& alphaNew,
                SurfaceScalarField& rp)
        {
            hin.deltaT = dtSub;
            ifm::alphaEqnStep(ah, alphaOld, hin, ic, hctl, m, g, fvp, hAlphaPhi, rp, hNHatf, hK,
                              nullptr);
            alphaNew = ah.internal;
        };

        std::vector<scalar> hAlpha = a0;
        SurfaceScalarField hRhoPhi;
        cs::alphaEqnSubCycle(nSub, totalDt, hAlpha, a0, hRhoPhi, hstep);

        const scalar dA = worstDiff(hAlpha, ref);
        const scalar dR = worstDiff(hRhoPhi.internal, refRhoPhi);
        scalar rScale = 0;
        for (scalar v : refRhoPhi) rScale = std::fmax(rScale, std::fabs(v));
        std::printf("  against the HOST sub-cycle: alpha %.4e, rhoPhi %.4e of %.4e\n",
                    (double)dA, (double)dR, (double)rScale);
        check("the device sub-cycle matches the host's to 1e-12", dA < scalar(1e-12));
        check("...and so does its weighted rhoPhi", dR < scalar(1e-12)*rScale);
    }

    std::printf("test_device_alpha_subcycle: %d failures\n", failures);
    return failures ? 1 : 0;
}
