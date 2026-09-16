// CMULES on the device -- the semi-implicit limiter, `MULESCorr yes`, which damBreak and twelve other
// shipped tutorials select.
//
// The host side is gated in tests/test_mules_cpp.cu against boundedness and against an unlimited
// control, and the whole solver it sits in reaches 3.4346e-09 on damBreak against real OpenFOAM. So
// agreement here is agreement with a limiter that has already been shown to limit.
//
// WHAT THIS FILE HAS TO PIN, over and above "it matches the host". CMULES differs from the explicit
// path in four ways (A, B, C and D in mules_cpp.cuh), and three of them are invisible to a boundedness
// gate -- they each leave alpha in [0,1] and merely give the wrong answer:
//
//   A  correct() takes the CURRENT psi and rho, not psi.oldTime() and rho.oldTime(). Arm 4 feeds it a
//      zero correction, where the right answer is "psi is unchanged" and the wrong one is "psi is
//      back at its old time" -- and the fixture makes those two differ by 0.75.
//
//   B  no sumPhiBD in the budget. There is no donor flux in CMULES; it went through the implicit
//      matrix. Arm 1 is what would catch a copied explicit transform, because that term is a per-cell
//      offset and lambda is built from it.
//
//   C  uncoupled boundary faces ARE limited, but outlets only, tested on phi + phiCorr -- the TOTAL.
//      Arm 3 is the host gate's own three-case fixture, and its third case is the discriminator: an
//      INFLOW face whose correction reverses the total must be limited. Testing `phi` alone passes
//      cases 1 and 2 and fails case 3.
//
// D is dictionary reading and has no device side.
//
// BIT-IDENTITY IS NOT GUARANTEED and is not claimed: the budgets are per-cell gathers here and a serial
// face loop there, floating-point addition is not associative, and the explicit limiter does differ on
// five of its 264 faces (by 3.7e-15). On THIS fixture CMULES happens to come out exact on all 264, 32
// of them strictly inside (0,1) so there was somewhere for a difference to show. The bound stays a
// tolerance because the guarantee does not exist; the boundedness that MULES exists for is asserted
// exactly, as it must be.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "mules_cpp.cuh"
#include "device_mules.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
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
std::vector<scalar> flatten(const std::vector<std::vector<scalar>>& b)
{
    std::vector<scalar> v;
    for (const auto& p : b) v.insert(v.end(), p.begin(), p.end());
    return v;
}
}   // namespace

int main()
{
    std::printf("== CMULES: device vs host ==\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 12;
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    // A DIAGONAL flux: every cell then has faces limited by different neighbours at once, which is
    // what makes nLimiterIter bite instead of reaching its fixed point on the first pass. The x
    // component is what arm 3's three cases are built around -- the inlet's outward normal is -x, so
    // its phi is NEGATIVE.
    const scalar kU = 1;
    const vector U{kU, scalar(0.6)*kU, scalar(0)};
    SurfaceScalarField phi;
    phi.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const vector& S = g.Sf()[f];
        phi.internal[f] = U.x*S.x + U.y*S.y + U.z*S.z;
    }
    phi.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phi.boundary[pi].resize(static_cast<std::size_t>(fvp[pi].size));
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const vector& S = g.Sf()[fvp[pi].start + i];
            phi.boundary[pi][i] = U.x*S.x + U.y*S.y + U.z*S.z;
        }
    }

    // A SMOOTH blob, not a step. The first version of this fixture used a step with a four-times
    // correction and lambda came back IDENTICALLY ZERO on all 264 faces -- which is the limiter
    // discarding the case's scheme wholesale and reverting to first-order upwind, and it made arm 1's
    // agreement with the host vacuous (zero equals zero) and arm 2's "psi moved" fail outright. The
    // profile and the multiplier below are chosen so lambda spans the interior of [0,1].
    GeometricField<scalar> a;
    a.internal.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::hypot(C.x - scalar(N)/2, C.y - scalar(N)/2);
        a.internal[c] = scalar(0.5)*(scalar(1) + std::tanh(scalar(N)/5 - r));
    }
    for (const FvPatch& q : fvp)
        a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    a.evaluateBoundary();

    // phiCorr: the antidiffusive part of a central scheme -- central minus upwind, which is exactly
    // what the caller hands CMULES after the implicit upwind solve.
    SurfaceScalarField corr0;
    corr0.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const label o = m.owner()[f], n = m.neighbour()[f];
        const scalar up = (phi.internal[f] >= 0) ? a.internal[o] : a.internal[n];
        const scalar cd = scalar(0.5)*(a.internal[o] + a.internal[n]);
        corr0.internal[f] = phi.internal[f] * (cd - up);
    }
    corr0.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        corr0.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size),
                                  (fvp[pi].name == "outlet") ? scalar(2) : scalar(0));

    const scalar dt = scalar(0.2);
    const scalar rDeltaT = scalar(1)/dt;
    mules::Controls ctl;
    ctl.nLimiterIter = 5;

    // --- the device fixture, built once -------------------------------------------------------------
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<scalar> aBnd;
    std::vector<int> fixes, flagPlain;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& v = a.boundary[pi]->value();
        aBnd.insert(aBnd.end(), v.begin(), v.end());
        const int fx = a.boundary[pi]->fixesValue() ? 1 : 0;
        const int fl = (fvp[pi].type == "empty") ? 1 : ((fvp[pi].type == "wedge") ? 2 : 0);
        for (label i = 0; i < fvp[pi].size; ++i) { fixes.push_back(fx); flagPlain.push_back(fl); }
    }
    DeviceBuffer<scalar> dPsi(a.internal), dPsiBnd(aBnd), dPhiBnd(flatten(phi.boundary));
    DeviceBuffer<int>    dFixes(fixes), dFlag(flagPlain);
    DeviceMulesFields df;                       // rho == 1, Sp == Su == 0, bounds [0,1]
    DeviceMulesControls dc;
    dc.nLimiterIter = ctl.nLimiterIter;

    // ---- 1. lambda against the host --------------------------------------------------------------
    std::vector<scalar> devLam;
    {
        mules::Limiter hLam;
        mules::Fields hf;
        mules::limiterCorr(hLam, rDeltaT, a, phi, corr0, hf, ctl, m, g, fvp);

        DeviceBuffer<scalar> dCorr(corr0.internal), dCorrBnd(flatten(corr0.boundary));
        DeviceBuffer<scalar> lamInt, lamBnd;
        deviceMulesLimiterCorr(dm, (int)nIf, (int)nBf, rDeltaT, dPsi, dPsiBnd, dFixes, dFlag,
                               dPhiBnd, dCorr, dCorrBnd, df, dc, lamInt, lamBnd);
        if (cudaDeviceSynchronize() != cudaSuccess)
        { std::printf("  FAIL: kernels did not complete\n"); return 1; }
        lamInt.copyTo(devLam);

        int nDiff = 0;
        scalar worst = 0;
        for (label f = 0; f < nIf; ++f)
        {
            if (devLam[f] != hLam.internal[f]) ++nDiff;
            worst = std::fmax(worst, std::fabs(devLam[f] - hLam.internal[f]));
        }
        scalar lo = 1, hi = 0;
        int interior = 0;
        for (label f = 0; f < nIf; ++f)
        {
            lo = std::fmin(lo, devLam[f]);
            hi = std::fmax(hi, devLam[f]);
            if (devLam[f] > scalar(1e-9) && devLam[f] < scalar(1) - scalar(1e-9)) ++interior;
        }
        std::printf("  lambda: %d of %ld faces differ at all; worst %.4e; range [%.4f, %.4f]; "
                    "%d faces strictly inside (0,1)\n",
                    nDiff, (long)nIf, (double)worst, (double)lo, (double)hi, interior);
        check("lambda agrees with the host to 1e-12", worst < scalar(1e-12));
        // THE PAIR, because either half alone is trivially satisfiable. lambda == 1 everywhere is no
        // limiter at all; lambda == 0 everywhere is the case's scheme silently discarded for
        // first-order upwind, and it makes a comparison against the host read as perfect agreement.
        check("...and lambda is not identically 1, so the limiter bit", lo < scalar(0.99));
        check("...nor identically 0, which would be upwind wearing the limiter's name",
              hi > scalar(0.01));
        // ...and a lambda that only ever took the values 0 and 1 would compare bit-for-bit whatever
        // the summation order, so the agreement above would say nothing about the arithmetic.
        check("...and lambda takes values strictly inside (0,1), so the comparison has somewhere to "
              "show a difference", interior >= 8);
    }

    // ---- 2. THE PROPERTY: limitCorr then correct keeps psi in [0,1]; unlimited does not -----------
    {
        std::vector<scalar> lim, unlim;
        {
            DeviceBuffer<scalar> dCorr(corr0.internal), dCorrBnd(flatten(corr0.boundary));
            DeviceBuffer<scalar> dOut(a.internal);
            deviceMulesLimitCorr(dm, (int)nIf, (int)nBf, rDeltaT, dOut, dPsiBnd, dFixes, dFlag,
                                 dPhiBnd, dCorr, dCorrBnd, df, dc);
            deviceMulesCorrect(dm, rDeltaT, dCorr, dCorrBnd, df, dOut);
            dOut.copyTo(lim);
        }
        {
            // the control: the same correction applied with no limiter at all
            DeviceBuffer<scalar> dCorr(corr0.internal), dCorrBnd(flatten(corr0.boundary));
            DeviceBuffer<scalar> dOut(a.internal);
            deviceMulesCorrect(dm, rDeltaT, dCorr, dCorrBnd, df, dOut);
            dOut.copyTo(unlim);
        }
        scalar wLim = 0, wUn = 0, moved = 0;
        for (label c = 0; c < nC; ++c)
        {
            wLim  = std::fmax(wLim, std::fmax(-lim[c],   lim[c]   - scalar(1)));
            wUn   = std::fmax(wUn,  std::fmax(-unlim[c], unlim[c] - scalar(1)));
            moved = std::fmax(moved, std::fabs(lim[c] - a.internal[c]));
        }
        std::printf("  worst excursion: limited %.3e, UNLIMITED %.3e; the limited field moved %.4f\n",
                    (double)wLim, (double)wUn, (double)moved);
        check("device CMULES keeps psi in [0,1]", wLim <= scalar(1e-14));
        check("...and the unlimited control does NOT, so the bound is doing work", wUn > scalar(1e-2));
        check("...and psi moved, so it is not satisfied by a zero correction", moved > scalar(1e-2));
    }

    // ---- 3. C: OUTLETS ONLY, AND THE TEST IS ON phi + phiCorr ------------------------------------
    // The host gate's own three-case fixture. Case 3 is the discriminator: written against `phi`
    // alone, cases 1 and 2 still pass and case 3 fails.
    {
        auto inletLambda = [&](scalar inletCorr)
        {
            SurfaceScalarField c2 = corr0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                c2.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size),
                                       (fvp[pi].name == "inlet")  ? inletCorr
                                     : (fvp[pi].name == "outlet") ? scalar(2)
                                                                  : scalar(0));
            DeviceBuffer<scalar> dCorr(c2.internal), dCorrBnd(flatten(c2.boundary));
            DeviceBuffer<scalar> lamInt, lamBnd;
            deviceMulesLimiterCorr(dm, (int)nIf, (int)nBf, rDeltaT, dPsi, dPsiBnd, dFixes, dFlag,
                                   dPhiBnd, dCorr, dCorrBnd, df, dc, lamInt, lamBnd);
            std::vector<scalar> lb;
            lamBnd.copyTo(lb);
            scalar inLam = 1, outLam = 1;
            label off = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    if (fvp[pi].name == "inlet")  inLam  = std::fmin(inLam,  lb[off + i]);
                    if (fvp[pi].name == "outlet") outLam = std::fmin(outLam, lb[off + i]);
                }
                off += fvp[pi].size;
            }
            return std::pair<scalar,scalar>{inLam, outLam};
        };

        const auto inflow = inletLambda(scalar(0.5));      // phi -1, corr +0.5, total -0.5: an inlet
        std::printf("  boundary: inlet phi %+.1f corr %+.1f total %+.1f -> lambda %.6f\n",
                    (double)(-kU), 0.5, (double)(-kU + 0.5), (double)inflow.first);
        check("a boundary face whose TOTAL flux enters the domain is left unlimited",
              inflow.first >= scalar(1) - scalar(1e-12));

        std::printf("  boundary: outlet phi %+.1f corr %+.1f total %+.1f -> lambda %.6f\n",
                    (double)kU, 2.0, (double)(kU + 2.0), (double)inflow.second);
        check("...while the outlet on the same run IS limited (control)",
              inflow.second < scalar(1) - scalar(1e-9));

        const auto reversed = inletLambda(scalar(2));      // phi -1, corr +2, total +1: an OUTLET now
        std::printf("  boundary: inlet phi %+.1f corr %+.1f total %+.1f -> lambda %.6f\n",
                    (double)(-kU), 2.0, (double)(-kU + 2.0), (double)reversed.first);
        check("an INFLOW face whose correction reverses the total IS limited -- the test is on "
              "phi + phiCorr, not on phi", reversed.first < scalar(1) - scalar(1e-9));
    }

    // ---- 4. A: correct() TAKES THE CURRENT psi, NOT psi.oldTime() --------------------------------
    // With a zero correction the right answer is "psi is exactly what came in". An implementation that
    // reached for psi.oldTime() -- which is what the explicit solve takes -- would return the old
    // field instead, and the fixture below makes those differ by 0.75 so the two are not confusable.
    {
        std::vector<scalar> advanced(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) advanced[c] = a.internal[c]*scalar(0.25) + scalar(0.75)*scalar(0.5);
        scalar apart = 0;
        for (label c = 0; c < nC; ++c) apart = std::fmax(apart, std::fabs(advanced[c] - a.internal[c]));

        DeviceBuffer<scalar> dOut(advanced);
        DeviceBuffer<scalar> zero(std::vector<scalar>(static_cast<std::size_t>(nIf), scalar(0)));
        DeviceBuffer<scalar> zeroB(std::vector<scalar>(static_cast<std::size_t>(nBf), scalar(0)));
        deviceMulesCorrect(dm, rDeltaT, zero, zeroB, df, dOut);
        std::vector<scalar> out;
        dOut.copyTo(out);

        scalar dIn = 0, dOld = 0;
        for (label c = 0; c < nC; ++c)
        {
            dIn  = std::fmax(dIn,  std::fabs(out[c] - advanced[c]));
            dOld = std::fmax(dOld, std::fabs(out[c] - a.internal[c]));
        }
        std::printf("  correct with a zero correction: |psi - psi_in| = %.3e, |psi - psi_old| = %.3e "
                    "(the two are %.2f apart)\n", (double)dIn, (double)dOld, (double)apart);
        check("correct() leaves the CURRENT psi alone when the correction is zero", dIn <= scalar(1e-15));
        check("...and psi_old is a different field, so that is not vacuous", dOld > scalar(0.1));
    }

    // ---- 5. an EMPTY patch is in no sum ------------------------------------------------------------
    // emptyFvPatch::size() is 0 in OpenFOAM, so its faces never enter sumPhip or mSumPhim. In brae they
    // exist, and CMULES is where that bites: phiCorr on the boundary comes from the CALLER here, so a
    // 2-D case's front and back would otherwise pour a correction into every cell's budget. The two
    // runs below differ only in what is written on a flagged-empty patch.
    {
        std::vector<int> flagEmpty = flagPlain;
        label off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (fvp[pi].name == "wallZmin" || fvp[pi].name == "wallZmax")
                for (label i = 0; i < fvp[pi].size; ++i) flagEmpty[off + i] = 1;
            off += fvp[pi].size;
        }
        DeviceBuffer<int> dFlagEmpty(flagEmpty);

        auto lamWith = [&](scalar onEmpty)
        {
            SurfaceScalarField c2 = corr0;
            off = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                if (fvp[pi].name == "wallZmin" || fvp[pi].name == "wallZmax")
                    c2.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), onEmpty);
            DeviceBuffer<scalar> dCorr(c2.internal), dCorrBnd(flatten(c2.boundary));
            DeviceBuffer<scalar> lamInt, lamBnd;
            deviceMulesLimiterCorr(dm, (int)nIf, (int)nBf, rDeltaT, dPsi, dPsiBnd, dFixes, dFlagEmpty,
                                   dPhiBnd, dCorr, dCorrBnd, df, dc, lamInt, lamBnd);
            std::vector<scalar> v;
            lamInt.copyTo(v);
            return v;
        };
        const std::vector<scalar> q0 = lamWith(scalar(0));
        const std::vector<scalar> q1 = lamWith(scalar(1e3));
        scalar worst = 0;
        for (label f = 0; f < nIf; ++f) worst = std::fmax(worst, std::fabs(q0[f] - q1[f]));
        std::printf("  a flagged-empty patch carrying 0 vs 1e3: worst change in internal lambda = %.3e\n",
                    (double)worst);
        check("an empty patch enters no budget, so what is written on it cannot move lambda",
              worst == scalar(0));

        // the fail-proof: with the SAME patch not flagged empty, 1e3 moves lambda a great deal. If it
        // did not, arm 5 would be passing because the fixture is inert.
        auto lamOrdinary = [&](scalar onIt)
        {
            SurfaceScalarField c2 = corr0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                if (fvp[pi].name == "wallZmin" || fvp[pi].name == "wallZmax")
                    c2.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), onIt);
            DeviceBuffer<scalar> dCorr(c2.internal), dCorrBnd(flatten(c2.boundary));
            DeviceBuffer<scalar> lamInt, lamBnd;
            deviceMulesLimiterCorr(dm, (int)nIf, (int)nBf, rDeltaT, dPsi, dPsiBnd, dFixes, dFlag,
                                   dPhiBnd, dCorr, dCorrBnd, df, dc, lamInt, lamBnd);
            std::vector<scalar> v;
            lamInt.copyTo(v);
            return v;
        };
        const std::vector<scalar> r0 = lamOrdinary(scalar(0));
        const std::vector<scalar> r1 = lamOrdinary(scalar(1e3));
        scalar moved = 0;
        for (label f = 0; f < nIf; ++f) moved = std::fmax(moved, std::fabs(r0[f] - r1[f]));
        std::printf("  ...and the SAME patch left ordinary: %.3e\n", (double)moved);
        check("...while an ordinary patch carrying it moves lambda, so the fixture is not inert",
              moved > scalar(1e-2));
    }

    std::printf("test_device_mules_corr: %d failures\n", failures);
    return failures ? 1 : 0;
}
