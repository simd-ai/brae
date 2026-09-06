// limited/blended convection schemes, _cpp reference.
//
// The assertions come from OpenFOAM's SOURCE, not from a stored field, because a weight array has exact
// properties that pin it down:
//   * TVD bound: every weight lies in [0,1] -- limitedSurfaceInterpolationScheme blends CD with upwind;
//   * limiter -> 1 (k -> 0, twoByk -> inf) must give EXACTLY the linear (CD) weights;
//   * limiter -> 0 (k huge) must give EXACTLY the upwind weights;
//   * LUST must equal 0.75*CD + 0.25*upwind to machine precision (LUST.H:113-114);
//   * limitedLinearV must DIFFER from limitedLinear -- the V form shares one limiter across the three
//     components (NVDVTVDV), and a port that limited them independently would still look plausible.
// The last one is the control: without it the V variant could be a copy of the scalar one and pass.
//
// Run: test_limitedschemes_cpp <caseDir> <timeDir>
#include "limitedSchemes_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "fvm.cuh"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu;

static int g_fails = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-60s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}
static scalar maxAbsDiff(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar mx = 0;
    for (std::size_t i = 0; i < a.size(); ++i) mx = std::fmax(mx, std::fabs(a[i]-b[i]));
    return mx;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    U.evaluateBoundary(); p.evaluateBoundary();
    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    const std::vector<scalar>& phi = phiF.internalField;

    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, fvp);
    const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);
    const std::vector<scalar>& cd = g.weights();
    const std::vector<scalar> up = limitedSchemes::upwindWeights(phi);

    std::printf("test_limitedschemes_cpp: nC=%d nInternalFaces=%d\n", (int)nC, (int)nIf);

    // ---- LUST: an exact identity -------------------------------------------------------------
    {
        const std::vector<scalar> w = limitedSchemes::lustWeights(phi, g);
        std::vector<scalar> want(nIf);
        for (label f = 0; f < nIf; ++f) want[f] = 0.75*cd[f] + 0.25*up[f];
        const scalar d = maxAbsDiff(w, want);
        std::printf("  %-60s max|d|=%.3e\n", "LUST == 0.75*linear + 0.25*upwind", d);
        check(d == 0.0, "LUST weights are exactly the LUST.H blend");
        // ...and LUST must not BE either of its parts, or the blend is not doing anything.
        check(maxAbsDiff(w, cd) > 1e-6 && maxAbsDiff(w, up) > 1e-6,
              "LUST differs from both linear and upwind (control)");
    }

    // ---- limitedLinear: the two limiter extremes ---------------------------------------------
    {
        // k -> 0 makes twoByk enormous, so clamp(twoByk*r) saturates at 1 wherever r > 0 -- i.e. the
        // scheme becomes pure linear except where the limiter genuinely switches off (r <= 0).
        const std::vector<scalar> wLin = limitedSchemes::limitedLinearWeights(phi, p, gradP, 1e-12, m, g);
        // k huge makes twoByk ~ 0, so the limiter is 0 everywhere: pure upwind.
        const std::vector<scalar> wUp  = limitedSchemes::limitedLinearWeights(phi, p, gradP, 1e12,  m, g);
        // ASYMPTOTIC, not exact: twoByk = 2/k is 2e-12 here, and r reaches ~1e3 through OpenFOAM's own
        // 1000x guard, so the limiter is ~1e-9 rather than 0. Asserting exact equality was wrong.
        const scalar dUp = maxAbsDiff(wUp, up);
        std::printf("  %-60s max|d|=%.3e\n", "limitedLinear(k=1e12) -> upwind", dUp);
        check(dUp < 1e-8, "limitedLinear collapses to upwind as k -> inf");
        // Where r > 0 the k->0 form must be exactly CD; count how many faces that is, to prove the
        // comparison is not vacuous on this mesh.
        label nCD = 0;
        for (label f = 0; f < nIf; ++f) if (std::fabs(wLin[f] - cd[f]) < 1e-14) ++nCD;
        std::printf("  limitedLinear(k->0) equals linear on %d of %d faces\n", (int)nCD, (int)nIf);
        check(nCD > nIf/10, "limitedLinear reaches the linear limit on a real fraction of faces");
    }

    // ---- TVD bound: every scheme's weights stay in [0,1] -------------------------------------
    {
        const std::vector<scalar> a = limitedSchemes::limitedLinearWeights(phi, p, gradP, 1.0, m, g);
        const std::vector<scalar> b = limitedSchemes::limitedLinearVWeights(phi, U, gradU, 1.0, m, g);
        const std::vector<scalar> c = limitedSchemes::lustWeights(phi, g);
        scalar lo = 1e30, hi = -1e30;
        for (const auto* v : {&a, &b, &c})
            for (scalar x : *v) { lo = std::fmin(lo, x); hi = std::fmax(hi, x); }
        std::printf("  all scheme weights in [%.6f, %.6f]\n", lo, hi);
        check(lo >= 0.0 && hi <= 1.0, "every scheme's weights satisfy the TVD bound [0,1]");

        // CONTROL: the V form couples the components, so it must NOT equal the scalar form applied to a
        // component. Comparing against limitedLinear on U's x-component is the closest wrong answer.
        GeometricField<scalar> Ux = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
        for (label ci = 0; ci < nC; ++ci) Ux.internal[ci] = U.internal[ci].x;
        Ux.evaluateBoundary();
        std::vector<vector> gUx(nC);
        for (label ci = 0; ci < nC; ++ci) gUx[ci] = { gradU[ci].xx, gradU[ci].yx, gradU[ci].zx };
        const std::vector<scalar> sx = limitedSchemes::limitedLinearWeights(phi, Ux, gUx, 1.0, m, g);
        const scalar dv = maxAbsDiff(b, sx);
        std::printf("  %-60s max|d|=%.3e\n", "limitedLinearV vs per-component limitedLinear", dv);
        check(dv > 1e-6, "the V form really couples the components (control)");
    }

    // ---- linearUpwindV: the limiter's exact postcondition ------------------------------------
    // After limiting, every face correction must satisfy either corr == 0 or
    // 0 <= magSqr(corr) <= (corr & maxCorr). That is the limiter restated, so it holds by construction --
    // which is the point: it fails loudly if a branch was transcribed wrongly. The controls are that the
    // limiter must actually FIRE (zero some faces, scale others) and that the result must differ from the
    // unlimited linearUpwind correction, or linearUpwindV would just be linearUpwind under another name.
    {
        namespace lsx = limitedSchemes;
        const std::vector<vector> fc = lsx::linearUpwindVFaceCorrection(phi, U, gradU, m, g);
        const std::vector<scalar>& w = g.weights();
        const std::vector<label>& own = m.owner();
        const std::vector<label>& nei = m.neighbour();
        label nZero = 0, nScaled = 0, nBad = 0;
        for (label f = 0; f < nIf; ++f)
        {
            const label P = own[f], N = nei[f];
            const bool out = (phi[f] > 0.0);
            const scalar a = out ? (1.0 - w[f]) : w[f];
            const vector& vP = U.internal[P]; const vector& vN = U.internal[N];
            const vector mc = out ? vector{a*(vN.x-vP.x), a*(vN.y-vP.y), a*(vN.z-vP.z)}
                                  : vector{a*(vP.x-vN.x), a*(vP.y-vN.y), a*(vP.z-vN.z)};
            const scalar sq = fc[f].x*fc[f].x + fc[f].y*fc[f].y + fc[f].z*fc[f].z;
            const scalar mx = fc[f].x*mc.x + fc[f].y*mc.y + fc[f].z*mc.z;
            if (sq == 0.0) { ++nZero; continue; }
            if (mx < 0.0 || sq > mx*(1.0 + 1e-9)) ++nBad;
            if (sq < mx*(1.0 - 1e-9)) {} else ++nScaled;
        }
        std::printf("  linearUpwindV faces: %d zeroed, %d at the limit, %d violating\n",
                    (int)nZero, (int)nScaled, (int)nBad);
        check(nBad == 0, "every limited face satisfies 0 <= magSqr(corr) <= corr & maxCorr");
        check(nZero > 0, "the limiter zeroes faces where the correction opposes the jump (control)");

        // ...and it must not equal the UNLIMITED linearUpwind correction.
        const std::vector<vector> lu =
            fvm::linearUpwindCorrection<vector, tensor>(phi, gradU, m, g);
        const std::vector<vector> lv = lsx::linearUpwindVCorrection(phi, U, gradU, m, g);
        scalar d = 0, mg = 0;
        for (label c = 0; c < nC; ++c)
        {
            d  = std::fmax(d,  std::fabs(lu[c].x - lv[c].x));
            mg = std::fmax(mg, std::fabs(lu[c].x));
        }
        std::printf("  %-60s rel=%.3e\n", "linearUpwindV vs unlimited linearUpwind", mg > 0 ? d/mg : d);
        check(d > 0.0, "linearUpwindV differs from linearUpwind -- the limiter is live (control)");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
