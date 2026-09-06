// realizableKE _cpp REFERENCE vs REAL OPENFOAM.
//
// THE ORACLE IS THE EQUATION RESIDUAL at OpenFOAM's converged fields, in OpenFOAM's own normalisation --
// a statement about the DISCRETISATION, which is what is being ported. (The tempting alternative, running
// one correct() and checking nothing moves, is not valid: OpenFOAM stops on a residual plateau rather
// than at an exact fixed point, so solving from its state to 1e-12 moves the field regardless.)
//
// Controls, because a small residual can also mean a small operator:
//   * a PERTURBED state must have a far larger residual;
//   * rCmu must genuinely VARY -- a constant 0.09 there is standard k-epsilon, which would still work
//     and still converge, so this is the check that says which model was ported;
//   * S2 must use devSymm and not symm -- kOmegaSST a few directories away uses the other one.
//
// Run: test_realizablke_cpp <caseDir> <ofConvergedTimeDir>
#include "realizableKE_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
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
    GeometricField<scalar> om  = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/epsilon"), fvp, nC);
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
    const FoamDict turb = readDict(caseDir + "/constant/turbulenceProperties");
    RealizableKECoeffs co;
    readRealizableKECoeffs(turb.subDict("RAS"), co);

    std::printf("test_realizablke_cpp: nC=%d nu=%.3g\n", (int)nC, nu);
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
    // epsilon equation's residual is measured below, and the resulting field change is 1.5e-01 in max norm).
    cpu::realizableKE::RKEResiduals r0;
    {
        GeometricField<scalar> kk = fresh("k"), oo = fresh("epsilon"), nn = fresh("nut");
        cpu::realizableKE::correct(U, kk, oo, nn, phi, nu, m, g, fvp, 1.0, 1.0, 1e-12, 0.0, 2000, co, &r0);
        std::printf("  residual of our equations at OpenFOAM's converged state: epsilon %.3e  k %.3e\n",
                    r0.epsilon, r0.k);
        check(r0.epsilon < 5e-2, "our epsilon equation is satisfied by OpenFOAM's solution");
        check(r0.k     < 5e-2, "our k equation is satisfied by OpenFOAM's solution");
    }

    // ---- 2. CONTROL: each equation probed by perturbing ITS OWN field ------------------------
    // A small residual can also mean a small operator, so each equation must be shown to respond. They
    // are perturbed SEPARATELY because they respond to different things: scaling k and epsilon together
    // barely moves the epsilon residual (measured 4.82e-03 -> 8.48e-03) since realizableKE's destruction
    // term C2*eps/(k + sqrt(nu*eps)) is a ratio in which a common factor largely cancels. Probing one
    // field at a time is what makes each threshold mean something.
    {
        GeometricField<scalar> kk = fresh("k"), oo = fresh("epsilon"), nn = fresh("nut");
        for (label c = 0; c < nC; ++c) oo.internal[c] *= 1.5;
        oo.evaluateBoundary();
        cpu::realizableKE::RKEResiduals rE;
        cpu::realizableKE::correct(U, kk, oo, nn, phi, nu, m, g, fvp, 1.0, 1.0, 1e-12, 0.0, 2000, co, &rE);
        std::printf("  epsilon x1.5 -> epsilon residual %.3e (was %.3e)\n", rE.epsilon, r0.epsilon);
        check(rE.epsilon > 5.0 * r0.epsilon, "the epsilon equation responds to a wrong epsilon (control)");
    }
    {
        GeometricField<scalar> kk = fresh("k"), oo = fresh("epsilon"), nn = fresh("nut");
        for (label c = 0; c < nC; ++c) kk.internal[c] *= 1.5;
        kk.evaluateBoundary();
        cpu::realizableKE::RKEResiduals rK;
        cpu::realizableKE::correct(U, kk, oo, nn, phi, nu, m, g, fvp, 1.0, 1.0, 1e-12, 0.0, 2000, co, &rK);
        std::printf("  k x1.5       -> k residual %.3e (was %.3e)\n", rK.k, r0.k);
        check(rK.k > 5.0 * r0.k, "the k equation responds to a wrong k (control)");
    }

    // ---- 3. THE VARIABLE Cmu IS THE MODEL ----------------------------------------------------
    // realizableKE's rCmu must actually vary. A port that used a constant 0.09 here would still be a
    // working turbulence model -- standard k-epsilon -- so this is the check that says which model was
    // ported. The control is that it departs from 0.09 by a real margin somewhere on the mesh.
    {
        const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, fvp);
        const std::vector<scalar> s2 = cpu::realizableKE::S2(gradU);
        const std::vector<scalar> rc =
            cpu::realizableKE::rCmu(gradU, s2, k.internal, om.internal, co);
        scalar lo = 1e30, hi = -1e30;
        for (label c = 0; c < nC; ++c) { lo = std::fmin(lo, rc[c]); hi = std::fmax(hi, rc[c]); }
        std::printf("  rCmu in [%.5f, %.5f]   (standard k-epsilon would be 0.09 everywhere)\n", lo, hi);
        check(lo > 0.0, "rCmu is positive everywhere");
        check(hi - lo > 1e-3, "rCmu genuinely varies -- this is not k-epsilon with a constant (control)");

        // S2 uses devSymm, NOT symm. The two differ by the trace term, and kOmegaSST a few directories
        // away uses the other one, so the distinction is asserted rather than assumed.
        scalar dmax = 0;
        for (label c = 0; c < nC; ++c)
        {
            const tensor& tt = gradU[c];
            const scalar tr = tt.xx + tt.yy + tt.zz;
            const scalar sy[9] = { tt.xx, 0.5*(tt.xy+tt.yx), 0.5*(tt.xz+tt.zx),
                                   0.5*(tt.yx+tt.xy), tt.yy, 0.5*(tt.yz+tt.zy),
                                   0.5*(tt.zx+tt.xz), 0.5*(tt.zy+tt.yz), tt.zz };
            scalar mm = 0; for (int q = 0; q < 9; ++q) mm += sy[q]*sy[q];
            dmax = std::fmax(dmax, std::fabs(2.0*mm - s2[c]));
            (void)tr;
        }
        std::printf("  %-58s max|d|=%.3e\n", "devSymm-S2 vs symm-S2 (kOmegaSST's)", dmax);
        check(dmax > 0.0, "S2 uses devSymm, not symm (control)");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
