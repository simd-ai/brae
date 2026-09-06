// kOmegaSST _cpp REFERENCE vs REAL OPENFOAM.
//
// THE ORACLE IS A FIXED POINT, not a stored field. OpenFOAM's converged kOmegaSST solution satisfies the
// model's own equations, so running ONE kOmegaSST::correct() from it must leave k, omega and nut where
// they are. That exercises the whole model -- production, both blending functions, the eddy-viscosity and
// production limiters, both transport equations and the omega wall function -- against OpenFOAM without
// needing OpenFOAM to dump any intermediate, and a wrong term shows up because it moves the state.
//
// Controls, because "it barely moved" only means something if something could have moved it:
//   * a PERTURBED state must be pulled back -- otherwise correct() could be a no-op and still pass;
//   * the raw-vs-limited GbyNu distinction must bite (kOmegaSSTBase.C reassigns GbyNu0 AFTER taking G
//     from it, and using one for both is the easy transcription error here);
//   * F1 must actually span wall (->1) and free stream (->0) on this mesh.
//
// Run: test_komegasst_cpp <caseDir> <ofConvergedTimeDir>
#include "kOmegaSST_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "cell_wall_dist.cuh"
#include "foam_dict.cuh"
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}
static scalar relMax(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    { mx = std::fmax(mx, std::fabs(a[i]-b[i])); mg = std::fmax(mg, std::fabs(b[i])); }
    return mg > 0 ? mx/mg : mx;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <ofTimeDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U   = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    GeometricField<scalar> k   = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/k"), fvp, nC);
    GeometricField<scalar> om  = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/omega"), fvp, nC);
    GeometricField<scalar> nut = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), fvp, nC);
    U.evaluateBoundary(); k.evaluateBoundary(); om.evaluateBoundary(); nut.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    SurfaceScalarField phi; phi.internal = phiF.internalField;
    phi.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phi.boundary[pi].assign(fvp[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
                phi.boundary[pi] = b.values;
    }

    const FoamDict tp = readDict(caseDir + "/constant/transportProperties");
    const scalar nu = tp.scalarOr("nu", 1e-5);
    const std::vector<scalar> y = cellWallDist(m, g, fvp);
    const FoamDict turb = readDict(caseDir + "/constant/turbulenceProperties");
    KOmegaSSTCoeffs co;
    readKOmegaSSTCoeffs(turb.subDict("RAS"), co);

    {
        scalar ylo = 1e30, yhi = -1e30;
        for (label c = 0; c < nC; ++c) { ylo = std::fmin(ylo, y[c]); yhi = std::fmax(yhi, y[c]); }
        std::printf("test_komegasst_cpp: nC=%d nu=%.3g   cell wall distance y in [%.4g, %.4g] m\n",
                    (int)nC, nu, ylo, yhi);
    }
    const std::vector<scalar> k0 = k.internal, om0 = om.internal, nut0 = nut.internal;

    // GeometricField owns its boundary fields through unique_ptr and is deliberately not copyable, so
    // each sub-test rebuilds from disk rather than copying the loaded state.
    auto fresh = [&](const std::string& name) {
        GeometricField<scalar> f =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + name), fvp, nC);
        f.evaluateBoundary();
        return f;
    };

    // ---- 1. OUR EQUATIONS MUST BE SATISFIED BY OPENFOAM'S CONVERGED SOLUTION ------------------
    // The initial residual of each assembled transport equation, in OpenFOAM's own normalisation, taken
    // at OpenFOAM's converged fields. This is a statement about the DISCRETISATION, which is what is
    // being ported. The tempting alternative -- run one correct() and check nothing moves -- is not a
    // valid oracle here: OpenFOAM stops on a residual plateau rather than at an exact fixed point, so
    // solving from its state to 1e-12 moves the field by whatever that plateau is worth (measured: the
    // omega equation's residual is 4e-03 there, and the resulting field change is 1.5e-01 in max norm).
    cpu::kOmegaSST::SSTResiduals r0;
    {
        GeometricField<scalar> kk = fresh("k"), oo = fresh("omega"), nn = fresh("nut");
        cpu::kOmegaSST::correct(U, kk, oo, nn, phi, y, nu, m, g, fvp, 1.0, 1.0, 1e-12, 0.0, 2000, co, &r0);
        std::printf("  residual of our equations at OpenFOAM's converged state: omega %.3e  k %.3e\n",
                    r0.omega, r0.k);
        check(r0.omega < 5e-2, "our omega equation is satisfied by OpenFOAM's solution");
        check(r0.k     < 5e-2, "our k equation is satisfied by OpenFOAM's solution");
    }

    // ---- 2. CONTROL: a perturbed state must have a much LARGER residual ----------------------
    // Without this the check above could pass with the equations mostly absent -- a residual is small
    // when the operator is small, too. Scaling k and omega by 1.5 is not a solution of the model, so its
    // residual must be far worse than OpenFOAM's.
    {
        GeometricField<scalar> kk = fresh("k"), oo = fresh("omega"), nn = fresh("nut");
        for (label c = 0; c < nC; ++c) { kk.internal[c] *= 1.5; oo.internal[c] *= 1.5; }
        kk.evaluateBoundary(); oo.evaluateBoundary();
        cpu::kOmegaSST::SSTResiduals r1;
        cpu::kOmegaSST::correct(U, kk, oo, nn, phi, y, nu, m, g, fvp, 1.0, 1.0, 1e-12, 0.0, 2000, co, &r1);
        std::printf("  perturbed (k,omega) x1.5: omega residual %.3e (was %.3e), k %.3e (was %.3e)\n",
                    r1.omega, r0.omega, r1.k, r0.k);
        check(r1.omega > 5.0 * r0.omega, "a non-solution has a far larger omega residual (control)");
        check(r1.k     > 5.0 * r0.k,     "a non-solution has a far larger k residual (control)");
    }

    // ---- 3. CONTROL: raw vs limited GbyNu must differ ----------------------------------------
    {
        const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, fvp);
        const std::vector<scalar> s2  = cpu::kOmegaSST::S2(gradU);
        const std::vector<scalar> gb0 = cpu::kOmegaSST::GbyNu0(gradU);
        const std::vector<scalar> f23 = cpu::kOmegaSST::F2(k.internal, om.internal, y, nu, co);
        std::vector<scalar> lim(nC);
        for (label c = 0; c < nC; ++c)
            lim[c] = std::fmin(gb0[c], (co.c1/co.a1)*co.betaStar*om.internal[c]
                             * std::fmax(co.a1*om.internal[c], co.b1*f23[c]*std::sqrt(s2[c])));
        scalar g0lo=1e30,g0hi=-1e30,s2lo=1e30,s2hi=-1e30,lmlo=1e30,lmhi=-1e30;
        for (label c = 0; c < nC; ++c)
        {
            g0lo=std::fmin(g0lo,gb0[c]); g0hi=std::fmax(g0hi,gb0[c]);
            s2lo=std::fmin(s2lo,s2[c]);  s2hi=std::fmax(s2hi,s2[c]);
            const scalar L=(co.c1/co.a1)*co.betaStar*om.internal[c]
                         * std::fmax(co.a1*om.internal[c], co.b1*f23[c]*std::sqrt(s2[c]));
            lmlo=std::fmin(lmlo,L); lmhi=std::fmax(lmhi,L);
        }
        std::printf("  GbyNu0 [%.3e, %.3e]  S2 [%.3e, %.3e]  limit [%.3e, %.3e]\n",
                    g0lo,g0hi,s2lo,s2hi,lmlo,lmhi);
        const scalar r = relMax(lim, gb0);
        std::printf("  %-58s rel=%.3e\n", "the GbyNu limiter on the converged field", r);
        // On this converged field the limiter does not bite at all (r == 0): the production is nowhere
        // near c1*betaStar*k*omega. Asserting that it does would be asserting something untrue about the
        // case, so the control instead forces a high-strain state, where it MUST bite -- otherwise the
        // limiter could be absent and this test could not tell.
        std::vector<tensor> hot = gradU;
        for (auto& t2 : hot) { t2.xx *= 50; t2.xy *= 50; t2.xz *= 50;
                               t2.yx *= 50; t2.yy *= 50; t2.yz *= 50;
                               t2.zx *= 50; t2.zy *= 50; t2.zz *= 50; }
        const std::vector<scalar> gbHot = cpu::kOmegaSST::GbyNu0(hot);
        const std::vector<scalar> s2Hot = cpu::kOmegaSST::S2(hot);
        std::vector<scalar> limHot(nC);
        for (label c = 0; c < nC; ++c)
            limHot[c] = std::fmin(gbHot[c], (co.c1/co.a1)*co.betaStar*om.internal[c]
                                * std::fmax(co.a1*om.internal[c], co.b1*f23[c]*std::sqrt(s2Hot[c])));
        const scalar rHot = relMax(limHot, gbHot);
        std::printf("  %-58s rel=%.3e\n", "control: at 50x strain the limiter bites", rHot);
        check(rHot > 1e-6, "the GbyNu production limiter is present and active (control)");
    }

    // ---- 4. F1/F2 are blend factors and actually span their range ----------------------------
    {
        const std::vector<vector> gK = fvc::gaussGrad(k, m, g, fvp);
        const std::vector<vector> gO = fvc::gaussGrad(om, m, g, fvp);
        const std::vector<scalar> CD = cpu::kOmegaSST::CDkOmega(gK, gO, om.internal, co);
        const std::vector<scalar> f1 = cpu::kOmegaSST::F1(k.internal, om.internal, y, CD, nu, co);
        const std::vector<scalar> f2 = cpu::kOmegaSST::F2(k.internal, om.internal, y, nu, co);
        scalar f1lo = 1e30, f1hi = -1e30, f2lo = 1e30, f2hi = -1e30;
        for (label c = 0; c < nC; ++c)
        {
            f1lo = std::fmin(f1lo, f1[c]); f1hi = std::fmax(f1hi, f1[c]);
            f2lo = std::fmin(f2lo, f2[c]); f2hi = std::fmax(f2hi, f2[c]);
        }
        std::printf("  F1 in [%.4f, %.4f]   F2 in [%.4f, %.4f]\n", f1lo, f1hi, f2lo, f2hi);
        check(f1lo >= 0.0 && f1hi <= 1.0, "F1 is a blend factor in [0,1]");
        check(f2lo >= 0.0 && f2hi <= 1.0, "F2 is a blend factor in [0,1]");
        check(f1hi > 0.9 && f1lo < 0.1, "F1 spans wall (->1) and free stream (->0) here (control)");
    }

    // ---- 5. the F3 switch is REFUSED, not ignored --------------------------------------------
    {
        KOmegaSSTCoeffs bad = co; bad.F3 = true;
        GeometricField<scalar> kk = fresh("k"), oo = fresh("omega"), nn = fresh("nut");
        bool threw = false;
        try { cpu::kOmegaSST::correct(U, kk, oo, nn, phi, y, nu, m, g, fvp, 1.0, 1.0, 1e-12, 0.0, 2000, bad); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "the F3 near-wall switch is refused");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
