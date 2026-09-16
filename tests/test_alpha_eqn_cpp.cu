// interFoam's alphaEqn.H -- the flux assembly, without MULES.
//
// The ORACLE is OpenFOAM's own expressions, transcribed from the cited lines and evaluated here. Where
// an arm tests a COMPOSITION rather than a formula (arm 5), the oracle is that composition written out
// literally from the same validated primitive, and the control enumerates the wrong compositions and
// requires each to differ -- the file shows what the mistake looks like instead of asserting that it is
// one.
//
// WHAT EACH ARM EXISTS TO CATCH:
//
// 1. nAlphaCorr and nAlphaSubCycles have NO DEFAULT (get<label>, alphaControls.H). nAlphaSubCycles in
//    particular sets how many sub-steps the alpha field is advanced by inside one PIMPLE step, and 23
//    of the 44 shipped tutorials set it to 3 -- defaulting it to 1 would run a third of the advection.
//
// 2. CrankNicolson + sub-cycling is a FatalError in OpenFOAM (alphaEqn.H:28-34), and a cold start's
//    first step runs Euler whatever the scheme says (:36-45). Two control-flow facts, both invisible in
//    any field.
//
// 3. phic IS ZEROED ON EVERY NON-COUPLED BOUNDARY (alphaEqn.H:79-89). Interface compression is
//    anti-diffusion; leaving it on at an inlet sharpens an interface the boundary does not have.
//
// 4. vanLeer IS NOT CLAMPED TO [0,1]. Its limiter reaches 2, which is the Sweby TVD ceiling. The two
//    exact identities below pin it from both ends: on a LINEAR field it returns central differencing
//    exactly, and across a STEP it degenerates to upwind exactly.
//
// 5. alphaPhiUn's compressive term is a nested flux with TWO minus signs.
//
// 6. rhoPhi's (rho1 - rho2) is an ORDER, and swapping it inverts the mixture while keeping the
//    dimensions and the magnitude.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "alpha_eqn_cpp.cuh"
#include "limitedSchemes_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

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

scalar worst(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar w = 0;
    for (std::size_t i = 0; i < a.size(); ++i) w = std::fmax(w, std::fabs(a[i] - b[i]));
    return w;
}

// A zeroGradient-walled scalar field on the given cell values.
GeometricField<scalar> makeAlpha(const std::vector<scalar>& cells, const std::vector<FvPatch>& fvp)
{
    GeometricField<scalar> a;
    a.internal = cells;
    for (const FvPatch& q : fvp)
        a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    a.evaluateBoundary();
    return a;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== interFoam alphaEqn (flux assembly) ==\n");
    const std::string tut = (argc > 1) ? argv[1] : "";

    // A single row of cells: 6 cells, 5 internal faces, all of them in x. One dimension is all the
    // interface advection needs, and it makes every face flux readable.
    const label N = 6;
    PrimitiveMesh m = boxtest::boxMesh(N, 1, 1);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nIf = m.nInternalFaces();

    // ---- 1. damBreak's own alphaControls -----------------------------------------------------------
    if (!tut.empty() && std::filesystem::exists(tut + "/system/fvSolution"))
    {
        const FoamDict fvSolution = readDict(tut + "/system/fvSolution");
        const AlphaControls c = readAlphaControls(fvSolution, "alpha.water");
        checkNum("damBreak nAlphaCorr",      scalar(c.nAlphaCorr),      scalar(2));
        checkNum("damBreak nAlphaSubCycles", scalar(c.nAlphaSubCycles), scalar(1));
        check   ("damBreak MULESCorr yes",   c.MULESCorr);
        check   ("alphaApplyPrevCorr absent -> false", !c.alphaApplyPrevCorr);
        checkNum("icAlpha absent -> 0", c.icAlpha, scalar(0));
        checkNum("scAlpha absent -> 0", c.scAlpha, scalar(0));
    }
    else
    {
        std::printf("  SKIP: OpenFOAM's damBreak tutorial not found at \"%s\"\n", tut.c_str());
        return 77;
    }

    // ---- 1b. the refusals, and the control ---------------------------------------------------------
    {
        const std::string base = "/tmp/brae_alpha_eqn";
        std::filesystem::remove_all(base);
        auto write = [&](const std::string& dir, const std::string& body)
        {
            std::filesystem::create_directories(base + "/" + dir);
            std::ofstream(base + "/" + dir + "/fvSolution")
                << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
                << "solvers\n{\n    \"alpha.water.*\"\n    {\n" << body << "    }\n}\n";
            return readDict(base + "/" + dir + "/fvSolution");
        };
        auto refuses = [&](const FoamDict& d)
        {
            try { (void)readAlphaControls(d, "alpha.water"); } catch (const std::exception&) { return true; }
            return false;
        };
        check("a missing nAlphaCorr is refused, not defaulted",
              refuses(write("nocorr", "        nAlphaSubCycles 1;\n")));
        check("a missing nAlphaSubCycles is refused, not defaulted to 1",
              refuses(write("nosub",  "        nAlphaCorr 1;\n")));
        check("a MULESCorr that is not a Switch is refused, not treated as off",
              refuses(write("badsw",  "        nAlphaCorr 1; nAlphaSubCycles 1; MULESCorr maybe;\n")));
        // CONTROL: the ordinary form must read, or the three refusals prove nothing.
        const FoamDict good = write("good", "        nAlphaCorr 2; nAlphaSubCycles 3; MULESCorr no;\n");
        check("a well-formed entry still reads", !refuses(good));
        const AlphaControls c = readAlphaControls(good, "alpha.water");
        checkNum("...with nAlphaSubCycles 3", scalar(c.nAlphaSubCycles), scalar(3));
        check   ("...and MULESCorr off", !c.MULESCorr);
    }

    // ---- 2. the off-centring coefficient -----------------------------------------------------------
    {
        checkNum("Euler      -> ocCoeff 0", offCentringCoeff(AlphaDdt::Euler,      3, scalar(0.9), true), scalar(0));
        checkNum("localEuler -> ocCoeff 0", offCentringCoeff(AlphaDdt::localEuler, 3, scalar(0.9), true), scalar(0));
        checkNum("CrankNicolson, warmed up -> the scheme's own",
                 offCentringCoeff(AlphaDdt::CrankNicolson, 1, scalar(0.9), true), scalar(0.9));
        checkNum("CrankNicolson, FIRST step of a cold start -> 0",
                 offCentringCoeff(AlphaDdt::CrankNicolson, 1, scalar(0.9), false), scalar(0));
        bool threw = false;
        try { (void)offCentringCoeff(AlphaDdt::CrankNicolson, 2, scalar(0.9), true); }
        catch (const std::exception&) { threw = true; }
        check("CrankNicolson with nAlphaSubCycles > 1 is refused, as OpenFOAM FatalErrors", threw);
        threw = false;
        try { (void)offCentringCoeff(AlphaDdt::other, 1, scalar(0), true); }
        catch (const std::exception&) { threw = true; }
        check("`backward` ddt(alpha) is refused -- OpenFOAM refuses it too", threw);
        checkNum("cnCoeff = 1/(1+oc); Euler gives 1", blendingCoeff(scalar(0)), scalar(1));
        checkNum("...and 1/1.9 at oc = 0.9",          blendingCoeff(scalar(0.9)), scalar(1)/scalar(1.9));
    }

    // ---- 3. phic, and the boundary that must be zero -----------------------------------------------
    {
        SurfaceScalarField phi;
        phi.internal.assign(static_cast<std::size_t>(nIf), scalar(0));
        for (label f = 0; f < nIf; ++f) phi.internal[f] = scalar(0.3) * scalar(f + 1);
        phi.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            // a LARGE boundary flux on purpose: if the zeroing is missing, phic_b is large too
            phi.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(50));

        const scalar cAlpha = scalar(1);
        SurfaceScalarField phic;
        compressionFlux(cAlpha, phi, g.magSf(), fvp, scalar(0), {}, scalar(0), {}, phic);

        for (label f = 0; f < nIf; ++f)
            checkNum("phic = cAlpha*|phi/magSf|",
                     phic.internal[f], cAlpha * std::fabs(phi.internal[f] / g.magSf()[f]));

        scalar maxB = 0;
        for (const std::vector<scalar>& b : phic.boundary)
            for (scalar s : b) maxB = std::fmax(maxB, std::fabs(s));
        std::printf("  boundary phi is 50 everywhere; worst |phic| on the boundary = %.3g\n", (double)maxB);
        check("phic is ZERO on every non-coupled boundary face", maxB == scalar(0));
        check("...and the control shows it would not be: cAlpha*|50/magSf| is large",
              cAlpha * scalar(50) / fvp[0].magSf[0] > scalar(1));

        // icAlpha blends; scAlpha is ADDED AFTERWARDS and is NOT scaled by (1 - icAlpha).
        const std::vector<scalar> magUf(static_cast<std::size_t>(nIf), scalar(2));
        const std::vector<scalar> shearf(static_cast<std::size_t>(nIf), scalar(7));
        SurfaceScalarField pic;
        compressionFlux(cAlpha, phi, g.magSf(), fvp, scalar(0.25), magUf, scalar(0.5), shearf, pic);
        const scalar base0 = cAlpha * std::fabs(phi.internal[0] / g.magSf()[0]);
        const scalar want0 = base0 * scalar(0.75) + (cAlpha * scalar(0.25)) * scalar(2)
                           + scalar(0.5) * scalar(7);
        checkNum("icAlpha blends, scAlpha adds after the blend", pic.internal[0], want0);
        const scalar wrong0 = (base0 * scalar(0.75) + (cAlpha * scalar(0.25)) * scalar(2)
                             + scalar(0.5) * scalar(7)) * scalar(0.75);
        check("...and scaling the shear term by (1-icAlpha) too would differ (control)",
              std::fabs(want0 - wrong0) > scalar(1e-6));

        bool threw = false;
        try { compressionFlux(cAlpha, phi, g.magSf(), fvp, scalar(0.25), {}, scalar(0), {}, pic); }
        catch (const std::exception&) { threw = true; }
        check("icAlpha > 0 without interpolate(mag(U)) is refused", threw);
    }

    // ---- 4. vanLeer: the two exact identities ------------------------------------------------------
    {
        // (a) a LINEAR alpha field. NVDTVD's r is 1 exactly (the upwind cell's projected gradient
        //     equals the face difference), vanLeer's limiter is (1+1)/(1+1) = 1, and the weights are
        //     therefore the CENTRAL-DIFFERENCE weights. Upwind would give 1 or 0 instead.
        std::vector<scalar> lin(static_cast<std::size_t>(N));
        for (label c = 0; c < N; ++c) lin[c] = scalar(0.1) + scalar(0.2) * g.C()[c].x;
        const GeometricField<scalar> a = makeAlpha(lin, fvp);
        const std::vector<scalar> phiPos(static_cast<std::size_t>(nIf), scalar(1));
        const std::vector<vector> grad = fvc::gaussGrad(a, m, g, fvp);
        const std::vector<scalar> w = brae::cpu::limitedSchemes::vanLeerWeights(phiPos, a, grad, m, g);
        // interior faces only: the first and last internal face see a zeroGradient wall, where the
        // Gauss gradient is one-sided and r is not 1.
        scalar wcd = 0;
        for (label f = 1; f < nIf - 1; ++f) wcd = std::fmax(wcd, std::fabs(w[f] - g.weights()[f]));
        std::printf("  linear alpha: worst |vanLeer weight - CD weight| = %.3e\n", (double)wcd);
        check("on a linear field vanLeer is central differencing, exactly", wcd <= scalar(1e-14));
        check("...which upwind is not (control)", std::fabs(g.weights()[2] - scalar(1)) > scalar(0.1));

        // (b) a STEP. r goes negative, the limiter is 0, and the weights collapse to upwind -- exactly.
        std::vector<scalar> step{1, 1, 1, 0, 0, 0};
        const GeometricField<scalar> as = makeAlpha(step, fvp);
        const std::vector<vector> gs = fvc::gaussGrad(as, m, g, fvp);
        const std::vector<scalar> ws = brae::cpu::limitedSchemes::vanLeerWeights(phiPos, as, gs, m, g);
        checkNum("across a step vanLeer degenerates to upwind", ws[2], scalar(1));
        check("...so the interface face is not centrally differenced",
              std::fabs(ws[2] - g.weights()[2]) > scalar(0.4));

        // (c) THE CEILING, AND IT IS TESTED THROUGH THE REAL FUNCTION.
        //
        //     vanLeer's limiter is (r+|r|)/(1+|r|): it passes through 1 at r = 1 and tends to 2. A port
        //     that reused limitedLinear's clamp01 would cap it at 1. Arms (a) and (b) CANNOT see that
        //     -- a linear field gives r = 1 and a step gives r < 0, and the clamp changes neither -- so
        //     a fixture of those two alone passes with the clamp in place. This was verified by
        //     inserting the clamp: the gate stayed green until this arm existed.
        //
        //     A CONVEX profile is what exercises r > 1: the gradient shrinks downstream, so the upwind
        //     cell's projected gradient exceeds the face difference. On the six interior faces below r
        //     runs 1.5 to 1.8 and the limiter 1.20 to 1.29, which pushes the face weight BELOW the
        //     central-difference weight -- the scheme leaning past central, which is exactly what the
        //     clamp would forbid.
        using brae::cpu::limitedSchemes::detail::vanLeerLimiter;
        checkNum("vanLeer(1) = 1",    vanLeerLimiter(scalar(1)),   scalar(1));
        checkNum("vanLeer(-5) = 0",   vanLeerLimiter(scalar(-5)),  scalar(0));
        checkNum("vanLeer(1e9) -> 2", vanLeerLimiter(scalar(1e9)), scalar(2), scalar(1e-8));
        check("...so it is NOT bounded by 1, and clamping it would be limitedLinear",
              vanLeerLimiter(scalar(3)) > scalar(1));

        const std::vector<scalar> convex{0, scalar(0.40), scalar(0.65), scalar(0.80),
                                         scalar(0.89), scalar(0.94)};
        const GeometricField<scalar> ac = makeAlpha(convex, fvp);
        const std::vector<vector> gc = fvc::gaussGrad(ac, m, g, fvp);
        const std::vector<scalar> wc = brae::cpu::limitedSchemes::vanLeerWeights(phiPos, ac, gc, m, g);
        bool sawAboveOne = false, sawBelowCD = false;
        for (label f = 1; f < nIf - 1; ++f)
        {
            const label P = m.owner()[f], Nn = m.neighbour()[f];
            const vector d{g.C()[Nn].x - g.C()[P].x, g.C()[Nn].y - g.C()[P].y, g.C()[Nn].z - g.C()[P].z};
            const scalar r = brae::cpu::limitedSchemes::detail::rScalar(
                phiPos[f], ac.internal[P], ac.internal[Nn], gc[P], gc[Nn], d);
            if (vanLeerLimiter(r) > scalar(1.0 + 1e-9)) sawAboveOne = true;
            if (wc[f] < g.weights()[f] - scalar(1e-9))  sawBelowCD  = true;
            // and the CONTROL, computed here: the clamped limiter would give exactly the CD weight
            const scalar limClamped = std::fmin(vanLeerLimiter(r), scalar(1));
            const scalar wClamped   = limClamped*g.weights()[f] + (scalar(1) - limClamped)*scalar(1);
            check("a clamped vanLeer would give a DIFFERENT weight on this face",
                  std::fabs(wc[f] - wClamped) > scalar(1e-6));
        }
        check("the convex fixture actually drives the limiter above 1", sawAboveOne);
        check("...and the face weight below central differencing, which a clamp forbids", sawBelowCD);
    }

    // ---- 5. alphaPhiUn: the nested flux, and the three wrong sign combinations ----------------------
    {
        // An INTERFACE, not a pure phase: the compressive term vanishes identically wherever alpha1 is
        // 0 or 1, so a fixture without a partially-filled cell cannot tell any two sign conventions
        // apart. alpha1 crosses over three cells here.
        const std::vector<scalar> a1v{1, 1, scalar(0.75), scalar(0.25), 0, 0};
        std::vector<scalar> a2v(a1v.size());
        for (std::size_t c = 0; c < a1v.size(); ++c) a2v[c] = scalar(1) - a1v[c];
        const GeometricField<scalar> alpha1 = makeAlpha(a1v, fvp);
        const GeometricField<scalar> alpha2 = makeAlpha(a2v, fvp);

        SurfaceScalarField phi, phir;
        phi.internal.assign(static_cast<std::size_t>(nIf), scalar(0.4));
        phir.internal.assign(static_cast<std::size_t>(nIf), scalar(0.9));
        phi.boundary.resize(fvp.size());
        phir.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            phi.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
            phir.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        }

        const AlphaFluxScheme sA = AlphaFluxScheme::vanLeer;   // damBreak: div(phi,alpha) Gauss vanLeer
        const AlphaFluxScheme sR = AlphaFluxScheme::linear;    // damBreak: div(phirb,alpha) Gauss linear

        SurfaceScalarField got;
        alphaPhiUn(phi, phir, alpha1, alpha2, sA, sR, m, g, fvp, got);

        // THE ORACLE: alphaEqn.H:164-176 written out literally, from the same primitive.
        auto neg = [&](const SurfaceScalarField& s)
        {
            SurfaceScalarField r = s;
            for (scalar& v : r.internal) v = -v;
            for (std::vector<scalar>& b : r.boundary) for (scalar& v : b) v = -v;
            return r;
        };
        auto flux = [&](const SurfaceScalarField& psi, const GeometricField<scalar>& vf, AlphaFluxScheme s)
        {
            SurfaceScalarField o;
            fluxWithScheme(psi, vf, s, m, g, fvp, o);
            return o;
        };
        SurfaceScalarField adv   = flux(phi, alpha1, sA);
        SurfaceScalarField inner = neg(flux(neg(phir), alpha2, sR));
        SurfaceScalarField comp  = flux(inner, alpha1, sR);
        std::vector<scalar> want(static_cast<std::size_t>(nIf));
        for (label f = 0; f < nIf; ++f) want[f] = adv.internal[f] + comp.internal[f];
        std::printf("  alphaPhiUn: worst |brae - OpenFOAM's expression| = %.3e\n",
                    (double)worst(got.internal, want));
        check("alphaPhiUn matches OpenFOAM's nested expression", worst(got.internal, want) <= scalar(1e-15));

        // THE CONTROLS: the three other sign combinations, each computed here so the file shows what
        // the wrong answer is rather than asserting that it is wrong.
        //
        // AND THE REASON THIS IS RUN TWICE. Under `Gauss linear` the interpolation weights do not
        // depend on the flux at all, so the two minus signs cancel EXACTLY and dropping both is not an
        // error -- it is the same number. That is not a weakness of the fixture; it is why the mistake
        // is undetectable on the 31 shipped tutorials that name `div(phirb,alpha) Gauss linear`. It
        // becomes an error the moment alpharScheme is flux-dependent, which is the other 11 (`Gauss
        // vanLeer` in 7, `Gauss interfaceCompression` in 4). Both cases are asserted, because the
        // claim worth pinning is not "the signs matter" but "the signs matter HERE and not THERE".
        auto sweep = [&](AlphaFluxScheme s, const char* schemeName, bool bothMustDiffer)
        {
            SurfaceScalarField a  = flux(phi, alpha1, sA);
            SurfaceScalarField ok = flux(neg(flux(neg(phir), alpha2, s)), alpha1, s);
            std::vector<scalar> ref(static_cast<std::size_t>(nIf));
            for (label f = 0; f < nIf; ++f) ref[f] = a.internal[f] + ok.internal[f];

            const std::vector<std::pair<const char*, SurfaceScalarField>> variants{
                {"inner minus dropped",  flux(neg(flux(phir, alpha2, s)), alpha1, s)},
                {"outer minus dropped",  flux(flux(neg(phir), alpha2, s), alpha1, s)},
                {"both minuses dropped", flux(flux(phir, alpha2, s), alpha1, s)}};
            for (std::size_t i = 0; i < variants.size(); ++i)
            {
                std::vector<scalar> alt(static_cast<std::size_t>(nIf));
                for (label f = 0; f < nIf; ++f) alt[f] = a.internal[f] + variants[i].second.internal[f];
                const scalar d = worst(alt, ref);
                std::printf("  %-8s %-22s worst difference %.3e\n", schemeName, variants[i].first, (double)d);
                const bool isBoth = (i == 2);
                const std::string what = std::string("alpharScheme ") + schemeName + ": "
                                       + variants[i].first + (isBoth && !bothMustDiffer
                                         ? " cancels exactly, as linear weights must"
                                         : " differs, so this arm discriminates");
                check(what.c_str(), (isBoth && !bothMustDiffer) ? (d == scalar(0)) : (d > scalar(1e-6)));
            }
        };
        sweep(AlphaFluxScheme::linear,  "linear",  /*bothMustDiffer=*/false);
        sweep(AlphaFluxScheme::vanLeer, "vanLeer", /*bothMustDiffer=*/true);

        // ...and the physical postcondition the whole term exists for: it vanishes in a PURE phase.
        const GeometricField<scalar> one  = makeAlpha(std::vector<scalar>(N, scalar(1)), fvp);
        const GeometricField<scalar> zero = makeAlpha(std::vector<scalar>(N, scalar(0)), fvp);
        SurfaceScalarField pure;
        alphaPhiUn(phi, phir, one, zero, sA, sR, m, g, fvp, pure);
        scalar dmax = 0;
        for (label f = 0; f < nIf; ++f)
            dmax = std::fmax(dmax, std::fabs(pure.internal[f] - phi.internal[f] * scalar(1)));
        check("in pure phase 1 the compressive term vanishes and alphaPhiUn is just phi", dmax <= scalar(1e-15));
    }

    // ---- 6. rhoPhi -------------------------------------------------------------------------------
    {
        const scalar rho1 = scalar(1000), rho2 = scalar(1);
        SurfaceScalarField phiF, aPhi, rhoPhi;
        phiF.internal = {scalar(0.4), scalar(0.4)};
        phiF.boundary.resize(0);
        aPhi.internal = {scalar(0.4), scalar(0)};      // face 0 pure water, face 1 pure air
        aPhi.boundary.resize(0);
        massFlux(aPhi, phiF, rho1, rho2, rhoPhi);
        checkNum("a face carrying only phase 1 gives phi*rho1", rhoPhi.internal[0], scalar(0.4) * rho1);
        checkNum("a face carrying only phase 2 gives phi*rho2", rhoPhi.internal[1], scalar(0.4) * rho2);

        SurfaceScalarField swapped;
        massFlux(aPhi, phiF, rho2, rho1, swapped);     // the order reversed
        std::printf("  rho1-rho2 order: correct %.6g, swapped %.6g\n",
                    (double)rhoPhi.internal[0], (double)swapped.internal[0]);
        check("swapping (rho1 - rho2) inverts the mixture, so the order is pinned",
              std::fabs(swapped.internal[0] - rhoPhi.internal[0]) > scalar(1));
        check("...and the swapped water face carries air's density",
              std::fabs(swapped.internal[0] - scalar(0.4) * rho2) < scalar(1e-12));
    }

    // ---- 7. the scheme that is not ported ---------------------------------------------------------
    {
        const GeometricField<scalar> a = makeAlpha(std::vector<scalar>(N, scalar(0.5)), fvp);
        SurfaceScalarField psi, out;
        psi.internal.assign(static_cast<std::size_t>(nIf), scalar(1));
        psi.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            psi.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        bool threw = false;
        try { fluxWithScheme(psi, a, AlphaFluxScheme::interfaceCompression, m, g, fvp, out); }
        catch (const std::exception&) { threw = true; }
        check("`Gauss interfaceCompression` is refused by name, not run as vanLeer", threw);
        // CONTROL: the two schemes the tutorials actually use must work.
        fluxWithScheme(psi, a, AlphaFluxScheme::vanLeer, m, g, fvp, out);
        checkNum("vanLeer on a uniform alpha gives psi*alpha", out.internal[0], scalar(0.5));
        fluxWithScheme(psi, a, AlphaFluxScheme::linear, m, g, fvp, out);
        checkNum("linear  on a uniform alpha gives psi*alpha", out.internal[0], scalar(0.5));
    }

    std::printf("test_alpha_eqn_cpp: %d failures\n", failures);
    return failures ? 1 : 0;
}
