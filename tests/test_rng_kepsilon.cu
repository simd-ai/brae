// RNGkEpsilon -- OF TurbulenceModels/RAS/RNGkEpsilon.
//
// The renormalisation-group k-epsilon differs from the standard model in TWO ways, and a port that
// takes only the second is wrong everywhere, not just where the extra term bites:
//
//   1. EVERY coefficient. Cmu 0.0845 (not 0.09), C1 1.42, C2 1.68, C3 -0.33, sigmak = sigmaEps
//      = 0.71942. Cmu feeds nut = Cmu k^2/eps AND the nut wall functions, so reading an RNG case with
//      kEpsilon's defaults is a ~7% error on the eddy viscosity before any strain is involved.
//
//   2. The epsilon production coefficient becomes (C1 - R):
//          eta = sqrt(|S2|)*k/eps,   R = eta*(1 - eta/eta0)/(1 + beta*eta^3),   eta0 = 4.38, beta = 0.012
//      R changes SIGN at eta = eta0: below it R > 0 and weakens epsilon production; above it R < 0 and
//      strengthens it, which drains k in strongly strained regions. That sign change is the whole model
//      -- it is why RNG does not over-predict separated and swirling flow the way the standard model
//      does -- so Leg 2 pins BOTH sides of eta0 rather than one convenient point.
//
// OF applies (C1 - R) to the G production ALONE; the SuSp(((2/3)C1 - C3)*divU, eps) term keeps the plain
// C1. Leg 3 is that discrimination: a fix that folded R into both terms would pass Leg 2 and fail here.
//
// gByNu in brae IS OF's S2 (= dev(twoSymm(gradU)) && gradU, the contraction G/nut is built from), so eta
// is available without computing anything new -- Leg 2 checks the identity numerically rather than
// trusting the naming.
#include "device_buffer.cuh"
#include "device_kepsilon.cuh"
#include "kepsilon_coeffs.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_dict.cuh"
#include "solver_controls.cuh"
#include "turbulence_setup.cuh"
#include "box_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstdio>
#include <fstream>
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

FoamDict dictFromString(const std::string& body)
{
    const std::string path = "test_rng_kepsilon.tmpdict";
    { std::ofstream f(path); f << body; }
    FoamDict d = readDict(path);
    std::remove(path.c_str());
    return d;
}

// OF RNGkEpsilon.C, transcribed independently of the kernel under test.
scalar ofR(scalar S2, scalar k, scalar eps, scalar eta0, scalar beta)
{
    const scalar eta = std::sqrt(std::fabs(S2)) * k / eps;
    return (eta * (scalar(1) - eta / eta0)) / (beta * eta * eta * eta + scalar(1));
}
} // namespace

int main()
{
    std::printf("== RNGkEpsilon ==\n");

    // ---- Leg 1: the coefficient set, and that it is not kEpsilon's ---------------------------------
    {
        const FoamDict d = dictFromString(
            "simulationType RAS;\n"
            "RAS { RASModel RNGkEpsilon; turbulence on; printCoeffs on; }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        readTurbulenceModel(d, ctl, {"test", true, true});
        const KEpsilonCoeffs& c = ctl.keCoeffs;
        check(c.rng, "parser: RASModel RNGkEpsilon selects the RNG path");
        check(std::fabs(c.Cmu - 0.0845) < 1e-14, "Cmu = 0.0845 (NOT kEpsilon's 0.09 -- it feeds nut and the wall functions)");
        check(std::fabs(c.C1 - 1.42) < 1e-14 && std::fabs(c.C2 - 1.68) < 1e-14, "C1 = 1.42, C2 = 1.68");
        check(std::fabs(c.C3 + 0.33) < 1e-14, "C3 = -0.33, and it is NEGATIVE (kEpsilon's is 0)");
        check(std::fabs(c.sigmaK - 0.71942) < 1e-14 && std::fabs(c.sigmaEps - 0.71942) < 1e-14,
              "sigmak = sigmaEps = 0.71942");
        check(std::fabs(c.eta0 - 4.38) < 1e-14 && std::fabs(c.beta - 0.012) < 1e-14, "eta0 = 4.38, beta = 0.012");
        check(!c.realizable, "vacuity guard: RNG did not accidentally select realizableKE");
    }
    {
        const FoamDict d = dictFromString(
            "simulationType RAS;\n"
            "RAS { RASModel RNGkEpsilon; turbulence on;\n"
            "      RNGkEpsilonCoeffs { Cmu 0.1; eta0 5.5; beta 0.02; sigmak 0.9; } }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        readTurbulenceModel(d, ctl, {"test", true, true});
        const KEpsilonCoeffs& c = ctl.keCoeffs;
        check(std::fabs(c.Cmu - 0.1) < 1e-14 && std::fabs(c.eta0 - 5.5) < 1e-14
           && std::fabs(c.beta - 0.02) < 1e-14 && std::fabs(c.sigmaK - 0.9) < 1e-14,
              "RNGkEpsilonCoeffs overrides are honoured");
        check(std::fabs(c.C2 - 1.68) < 1e-14, "...and an absent key keeps the RNG default, not kEpsilon's");
    }
    {
        // negative control: the standard model must be untouched by any of this
        const FoamDict d = dictFromString(
            "simulationType RAS;\n"
            "RAS { RASModel kEpsilon; turbulence on; }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        readTurbulenceModel(d, ctl, {"test", true, true});
        check(!ctl.keCoeffs.rng && std::fabs(ctl.keCoeffs.Cmu - 0.09) < 1e-14
           && std::fabs(ctl.keCoeffs.C1 - 1.44) < 1e-14,
              "negative control: plain kEpsilon still gets Cmu 0.09 / C1 1.44 and no R term");
    }
    {
        bool threw = false;
        try
        {
            const FoamDict d = dictFromString("simulationType RAS;\nRAS { RASModel RNGkOmega; turbulence on; }\n");
            DeviceSimpleControls ctl; ctl.turbulent = true;
            readTurbulenceModel(d, ctl, {"test", true, true});
        }
        catch (const std::exception&) { threw = true; }
        check(threw, "refusal: an unimplemented RASModel is still rejected, not silently run as kEpsilon");
    }

    // ---- Leg 2: the R term, on BOTH sides of eta0 --------------------------------------------------
    PrimitiveMesh m = boxtest::boxMesh(4, 3, 2);
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells();

    KEpsilonCoeffs rc;
    rc.rng = true; rc.Cmu = 0.0845; rc.C1 = 1.42; rc.C2 = 1.68; rc.C3 = -0.33;
    rc.sigmaK = rc.sigmaEps = 0.71942; rc.eta0 = 4.38; rc.beta = 0.012;

    // eta = sqrt(S2)*k/eps swept across eta0 = 4.38: with k = 1 and eps = 1, eta = sqrt(S2), so S2 from
    // 1 to ~100 spans eta 1 -> 10 and crosses eta0 at S2 = 19.2.
    std::vector<scalar> hS2(nC), hK(nC), hE(nC), hDivU(nC, 0.0);
    for (label c = 0; c < nC; ++c)
    {
        hS2[c] = 1.0 + 4.0*c;      // 1 .. 93  ->  eta 1 .. 9.6
        hK[c]  = 1.0;
        hE[c]  = 1.0;
    }
    DeviceBuffer<scalar> S2, k, eps, divU;
    S2.copyFrom(hS2); k.copyFrom(hK); eps.copyFrom(hE); divU.copyFrom(hDivU);

    auto runEps = [&](const KEpsilonCoeffs& co, std::vector<scalar>& hd, std::vector<scalar>& hs)
    {
        DeviceBuffer<scalar> diag, src;
        diag.copyFrom(std::vector<scalar>(nC, scalar(0)));
        src.copyFrom(std::vector<scalar>(nC, scalar(0)));
        deviceEpsReaction(dm, eps, k, S2, divU, diag, src, co);
        diag.copyTo(hd); src.copyTo(hs);
    };

    std::vector<scalar> dRng, sRng, dStd, sStd;
    runEps(rc, dRng, sRng);
    KEpsilonCoeffs nc = rc; nc.rng = false;      // same coefficients, R term OFF: isolates R alone
    runEps(nc, dStd, sStd);

    const std::vector<scalar>& V = g.V();
    {
        scalar worst = 0;
        int below = 0, above = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar R = ofR(hS2[c], hK[c], hE[c], rc.eta0, rc.beta);
            const scalar want = V[c] * ((rc.C1 - R) * rc.Cmu * hK[c] * hS2[c]);
            worst = std::max(worst, std::fabs(sRng[c] - want)/std::max(std::fabs(want), scalar(1e-30)));
            const scalar eta = std::sqrt(hS2[c]) * hK[c] / hE[c];
            if (eta < rc.eta0) ++below; else ++above;
        }
        check(worst < 1e-13, "epsilon production is (C1 - R)*Cmu*k*S2 with OF's R");
        check(below >= 2 && above >= 2, "vacuity guard: the sweep really straddles eta0 (both signs of R)");
    }
    {
        // The sign change IS the model. Below eta0 R > 0 weakens production; above it R < 0 strengthens it.
        bool weakenedBelow = false, strengthenedAbove = false;
        for (label c = 0; c < nC; ++c)
        {
            const scalar eta = std::sqrt(hS2[c]) * hK[c] / hE[c];
            if (eta < rc.eta0 && sRng[c] < sStd[c] - 1e-12) weakenedBelow = true;
            if (eta > rc.eta0 && sRng[c] > sStd[c] + 1e-12) strengthenedAbove = true;
        }
        check(weakenedBelow,    "below eta0: R > 0, epsilon production is WEAKENED vs the same model without R");
        check(strengthenedAbove, "above eta0: R < 0, epsilon production is STRENGTHENED");
    }

    // ---- Leg 3: R touches the production term only, never the divU SuSp ---------------------------
    // OF: (C1 - R)*G*eps/k  -  fvm::SuSp(((2/3)C1 - C3)*divU, eps). The second keeps the plain C1, so
    // with a COMPRESSIVE divU (which drives the SuSp into the diagonal) the diagonals of the RNG and
    // non-RNG runs must agree exactly -- R is nowhere in them.
    {
        std::vector<scalar> hd2(nC, 0.0);
        for (label c = 0; c < nC; ++c) hd2[c] = 3.0;       // divU > 0 -> sp > 0 -> lands in the diagonal
        DeviceBuffer<scalar> dU2; dU2.copyFrom(hd2);
        auto runWithDivU = [&](const KEpsilonCoeffs& co, std::vector<scalar>& hd, std::vector<scalar>& hs)
        {
            DeviceBuffer<scalar> diag, src;
            diag.copyFrom(std::vector<scalar>(nC, scalar(0)));
            src.copyFrom(std::vector<scalar>(nC, scalar(0)));
            deviceEpsReaction(dm, eps, k, S2, dU2, diag, src, co);
            diag.copyTo(hd); src.copyTo(hs);
        };
        std::vector<scalar> dr, sr, ds, ss;
        runWithDivU(rc, dr, sr);
        runWithDivU(nc, ds, ss);
        scalar diagDiff = 0, srcDiff = 0;
        for (label c = 0; c < nC; ++c)
        {
            diagDiff = std::max(diagDiff, std::fabs(dr[c] - ds[c]));
            srcDiff  = std::max(srcDiff,  std::fabs(sr[c] - ss[c]));
        }
        check(diagDiff == 0.0, "discrimination: R is absent from the diagonal -- the divU SuSp keeps plain C1");
        check(srcDiff > 1e-6, "vacuity guard: R does move the source, so the diagonal check is not trivially true");
    }

    // ---- Leg 4: with rng off, nothing changed for the standard model ------------------------------
    {
        KEpsilonCoeffs std0;    // kEpsilon defaults
        std::vector<scalar> d0, s0;
        runEps(std0, d0, s0);
        scalar worst = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar want = V[c] * (std0.C1 * std0.Cmu * hK[c] * hS2[c]);
            worst = std::max(worst, std::fabs(s0[c] - want)/std::max(std::fabs(want), scalar(1e-30)));
        }
        check(worst < 1e-13, "negative control: the kEpsilon path is bit-for-bit the plain C1*Cmu*k*S2 it was");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
