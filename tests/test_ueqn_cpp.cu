// The _cpp reference for simpleFoam's UEqn.H, against OpenFOAM's own momentum-matrix dump.
//
// The assembly is validated by DECOMPOSITION rather than as one number, because a single relative error
// covering convection, diffusion, the explicit stress and relaxation cannot tell you which of them is
// wrong -- and "find the first divergent intermediate" is the whole method here:
//
//   1. momentumCore == OpenFOAM's div(phi,U) - laplacian(nu,U), exactly (that is what momentum.dat holds);
//   2. the full assembleUEqn differs from the core ONLY in `source`, and by exactly the explicit dev2
//      term -- so the implicit structure is provably untouched by adding divDevReff;
//   3. relaxation is the identity at alpha = 1 and demonstrably not at alpha < 1;
//   4. MRF and fvOptions are REFUSED, not ignored.
//
// Run: test_ueqn_cpp <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "UEqn_cpp.cuh"
#include "linearViscousStress_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
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

static scalar relS(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(a[i] - b[i]));
        mg = std::fmax(mg, std::fabs(b[i]));
    }
    return mg > 0 ? mx / mg : mx;
}

static scalar relV(const std::vector<vector>& a, const std::vector<vector>& b)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        mx = std::fmax(mx, std::fmax(std::fabs(a[i].x - b[i].x),
                       std::fmax(std::fabs(a[i].y - b[i].y), std::fabs(a[i].z - b[i].z))));
        mg = std::fmax(mg, std::fmax(std::fabs(b[i].x),
                       std::fmax(std::fabs(b[i].y), std::fabs(b[i].z))));
    }
    return mg > 0 ? mx / mg : mx;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, m.nCells());
    U.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    std::vector<std::vector<scalar>> phiBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        phiBnd[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size)
                phiBnd[pi] = b.values;
    }

    // Laminar: nuEff = nu everywhere, cells and boundary, matching what the OpenFOAM dump was built with.
    std::vector<scalar> nuEffC(m.nCells(), nu);
    std::vector<std::vector<scalar>> nuEffB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) nuEffB[pi].assign(patches[pi].size, nu);

    cpu::MomentumInput in;
    in.phi = &phiF.internalField;  in.phiBnd = &phiBnd;
    in.nuEff = &nuEffC;            in.nuEffBnd = &nuEffB;
    in.relaxU = 1.0;

    std::printf("test_ueqn_cpp:\n");

    // --- 1. the convection/diffusion core against OpenFOAM ------------------------------------
    const FvVectorMatrix core = cpu::momentumCore(U, in, m, g, patches);

    std::ifstream f(caseDir + "/momentum.dat");
    if (!f) { std::printf("FAIL no momentum.dat\n"); return 1; }
    label nC, nIf, np; f >> nC >> nIf >> np;
    std::vector<scalar> ofDiag(nC), ofUp(nIf), ofLo(nIf);
    std::vector<vector> ofSrc(nC);
    for (auto& v : ofDiag) f >> v;
    for (auto& v : ofUp)   f >> v;
    for (auto& v : ofLo)   f >> v;
    for (auto& v : ofSrc)  f >> v.x >> v.y >> v.z;

    const scalar tol = 1e-11;
    std::printf("  -- momentumCore vs OpenFOAM (div(phi,U) - laplacian(nu,U))\n");
    check(relS(core.diag,  ofDiag) <= tol, "diag");
    check(relS(core.upper, ofUp)   <= tol, "upper");
    check(relS(core.lower, ofLo)   <= tol, "lower");
    check(relV(core.source, ofSrc) <= tol, "source");

    // --- 2. divDevReff touches the SOURCE only, and by exactly the explicit term ---------------
    const FvVectorMatrix full = cpu::assembleUEqn(U, in, m, g, patches);
    std::printf("  -- assembleUEqn = core + explicit dev2 (implicit structure unchanged)\n");
    check(relS(full.diag,  core.diag)  == 0.0, "diag identical to the core");
    check(relS(full.upper, core.upper) == 0.0, "upper identical to the core");
    check(relS(full.lower, core.lower) == 0.0, "lower identical to the core");

    const std::vector<vector> expl =
        cpu::divDevReffExplicit(U, nuEffC, nuEffB, m, g, patches);
    const std::vector<scalar>& V = g.V();
    std::vector<vector> predicted(core.source.size());
    for (std::size_t c = 0; c < predicted.size(); ++c)
        predicted[c] = {core.source[c].x - expl[c].x * V[c],
                        core.source[c].y - expl[c].y * V[c],
                        core.source[c].z - expl[c].z * V[c]};
    check(relV(full.source, predicted) <= 1e-14, "source = core.source - divDevReffExplicit*V");

    // CONTROL: the term must actually have been ADDED, otherwise the identity above holds trivially and
    // would keep holding if divDevReff were dropped.
    //
    // Note what it is NOT: an assertion that the term is large. For incompressible flow tr(grad U) =
    // div(U) ~ 0, so dev2(T(grad U)) ~ grad(U)^T, and with constant nu its divergence is ~ nu*grad(div U)
    // -- near zero by construction on a converged solution. A first version of this test asserted the
    // opposite and failed; the measured contribution is printed so the number is on the record rather than
    // guessed at. It matters on cases where nu is NOT constant (any turbulent case, where nuEff varies
    // cell to cell), which is exactly why test_divdevreff_cpp validates it against OpenFOAM directly on a
    // k-epsilon field instead of relying on this one.
    const scalar dev2Rel = relV(full.source, core.source);
    std::printf("  %-58s rel=%.3e\n", "dev2 contribution to source (laminar, const nu)", dev2Rel);
    check(dev2Rel > 0.0, "the dev2 term was added at all (control)");

    // --- 3. relaxation ------------------------------------------------------------------------
    std::printf("  -- UEqn.relax()\n");
    cpu::MomentumInput inRelaxed = in;
    inRelaxed.relaxU = 0.7;
    const FvVectorMatrix relaxed = cpu::assembleUEqn(U, inRelaxed, m, g, patches);
    check(relS(relaxed.diag, full.diag) > 1e-3, "alpha=0.7 changes the diagonal");
    check(relS(relaxed.upper, full.upper) == 0.0, "relaxation leaves the off-diagonals alone");
    // OpenFOAM's fvMatrix::relax raises the diagonal; a factor below 1 can only increase |diag|.
    bool raised = true;
    for (std::size_t c = 0; c < relaxed.diag.size(); ++c)
        if (std::fabs(relaxed.diag[c]) + 1e-30 < std::fabs(full.diag[c])) { raised = false; break; }
    check(raised, "relaxation never lowers |diag| (OpenFOAM raises it)");

    // --- 4. MRF and fvOptions are refused, not ignored -----------------------------------------
    std::printf("  -- refusals\n");
    for (int which = 0; which < 2; ++which)
    {
        cpu::MomentumInput bad = in;
        (which == 0 ? bad.hasMRF : bad.hasFvOptions) = true;
        bool threw = false;
        try { cpu::assembleUEqn(U, bad, m, g, patches); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, which == 0 ? "MRF is refused" : "fvOptions is refused");
    }

    std::printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
