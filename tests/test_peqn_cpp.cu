// The _cpp reference for simpleFoam's pEqn.H, stage by stage against OpenFOAM's own dumps.
//
// pEqn.H has seven places it can be wrong -- rAU, HbyA, the flux, adjustPhi, the Laplacian, the reference
// cell, and the flux correction -- and a single relative error on `p` cannot say which. So each stage is
// checked against the tightest thing available:
//
//   rAU, HbyA        ops.dat   (OpenFOAM's fvMatrix::A() and H())
//   pEqn Laplacian   peqn.dat  (OpenFOAM's fvm::laplacian(interpolate(rAU), p))
//   pEqn source      derived   (Laplacian's own source + div(phiHbyA)*V, subtracted back out)
//   setReference     analytic  (fvMatrix.C:1011-1023 is two lines; assert exactly those two)
//   correctFlux      analytic  (at p == 0 the flux is -boundaryCoeffs, exactly)
//   relaxField       analytic  (GeometricField.C:1094)
//   refusals         MRF / fvOptions / fixedFluxPressure must all throw
//
// Run: test_peqn_cpp <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "pEqn_cpp.cuh"

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
    std::printf("  %-56s %s\n", what, ok ? "OK" : "FAIL");
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

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, nC);
    U.evaluateBoundary();
    GeometricField<scalar> p =
        buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, nC);
    p.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    std::vector<std::vector<scalar>> phiBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        phiBnd[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size)
                phiBnd[pi] = b.values;
    }

    // The same UEqn the OpenFOAM dumps were taken from: div(phi,U) - laplacian(nu,U), laminar.
    FvVectorMatrix UEqn = fvm::div(phiF.internalField, phiBnd, U, m, patches);
    addEqual(UEqn, fvm::laplacian(U, 1e-5, m, g, patches), -1.0);

    std::printf("test_peqn_cpp:\n");

    // --- stages 1-3 -----------------------------------------------------------------------
    cpu::PressureInput in;
    in.pRefCell = -1;                       // no reference: adjustPhi must NOT run
    const cpu::PressureStages st = cpu::pressurePredictor(UEqn, U, p, in, m, g, patches);

    std::ifstream ops(caseDir + "/ops.dat");
    if (!ops) { std::printf("FAIL no ops.dat\n"); return 1; }
    { label a, b, c; ops >> a >> b >> c; }
    std::vector<scalar> Aof(nC);
    std::vector<vector> Hof(nC);
    for (auto& v : Aof) ops >> v;
    for (auto& v : Hof) ops >> v.x >> v.y >> v.z;

    std::vector<scalar> rAUof(nC);
    for (label c = 0; c < nC; ++c) rAUof[c] = 1.0 / Aof[c];
    std::printf("  -- stage 1-2: rAU = 1/UEqn.A(), HbyA = rAU*UEqn.H()\n");
    check(relS(st.rAU, rAUof) <= 1e-11, "rAU vs OpenFOAM A()");

    std::vector<scalar> hx, hxOf;
    for (label c = 0; c < nC; ++c)
    {
        hx.push_back(st.HbyA[c].x); hx.push_back(st.HbyA[c].y); hx.push_back(st.HbyA[c].z);
        hxOf.push_back(rAUof[c]*Hof[c].x); hxOf.push_back(rAUof[c]*Hof[c].y); hxOf.push_back(rAUof[c]*Hof[c].z);
    }
    check(relS(hx, hxOf) <= 1e-11, "HbyA (internal) vs rAU*OpenFOAM H()");
    check(!st.phiAdjusted, "adjustPhi did NOT run when p needs no reference");

    // --- stage 4: the pressure Laplacian --------------------------------------------------
    const FvScalarMatrix pEqn = cpu::assemblePEqn(st, p, in, m, g, patches);

    std::ifstream pf(caseDir + "/peqn.dat");
    if (!pf) { std::printf("FAIL no peqn.dat\n"); return 1; }
    label pnC, pnIf, pnp; pf >> pnC >> pnIf >> pnp;
    std::vector<scalar> lDiag(pnC), lUp(pnIf), lLo(pnIf), lSrc(pnC);
    for (auto& v : lDiag) pf >> v;
    for (auto& v : lUp)   pf >> v;
    for (auto& v : lLo)   pf >> v;
    for (auto& v : lSrc)  pf >> v;

    std::printf("  -- stage 4: fvm::laplacian(rAU, p) vs OpenFOAM\n");
    check(relS(pEqn.diag,  lDiag) <= 1e-11, "diag");
    check(relS(pEqn.upper, lUp)   <= 1e-11, "upper");
    check(relS(pEqn.lower, lLo)   <= 1e-11, "lower");
    for (label pp = 0; pp < pnp; ++pp)
    {
        std::string name; label sz; pf >> name >> sz;
        std::vector<scalar> iC(sz), bC(sz);
        for (label i = 0; i < sz; ++i) pf >> iC[i] >> bC[i];
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k) if (patches[k].name == name) { pk = k; break; }
        check(relS(pEqn.internalCoeffs[pk], iC) <= 1e-11, (name + ":internalCoeffs").c_str());
        check(relS(pEqn.boundaryCoeffs[pk], bC) <= 1e-11, (name + ":boundaryCoeffs").c_str());
    }

    // The source must be the Laplacian's own source PLUS div(phiHbyA)*V and nothing else.
    const std::vector<scalar> divPhiHbyA = fvc::div(st.phiHbyA, m, g, patches);
    const std::vector<scalar>& V = g.V();
    std::vector<scalar> back(nC);
    for (label c = 0; c < nC; ++c) back[c] = pEqn.source[c] - divPhiHbyA[c]*V[c];
    std::printf("  -- stage 4: source == laplacian source + div(phiHbyA)*V\n");
    check(relS(back, lSrc) <= 1e-11, "source minus div(phiHbyA)*V vs OpenFOAM");
    // ...and div(phiHbyA)*V must not be zero, or the check above is vacuous.
    scalar mxRhs = 0; for (label c = 0; c < nC; ++c) mxRhs = std::fmax(mxRhs, std::fabs(divPhiHbyA[c]*V[c]));
    check(mxRhs > 0.0, "div(phiHbyA)*V is non-zero (control)");

    // --- stage 5: setReference is exactly OpenFOAM's two lines ----------------------------
    std::printf("  -- stage 5: pEqn.setReference (fvMatrix.C:1011-1023)\n");
    cpu::PressureInput inRef = in;
    inRef.pRefCell = 0; inRef.pRefValue = 3.5;
    const FvScalarMatrix pRef = cpu::assemblePEqn(st, p, inRef, m, g, patches);
    check(std::fabs(pRef.diag[0] - 2.0*pEqn.diag[0]) <= 1e-12*std::fabs(pEqn.diag[0]),
          "diag[refCell] is DOUBLED (not overwritten)");
    check(std::fabs(pRef.source[0] - (pEqn.source[0] + pEqn.diag[0]*3.5))
              <= 1e-12*std::fabs(pEqn.source[0] + pEqn.diag[0]*3.5),
          "source[refCell] += diag[refCell]*refValue");
    bool onlyRefCell = true;
    for (label c = 1; c < nC; ++c)
        if (pRef.diag[c] != pEqn.diag[c] || pRef.source[c] != pEqn.source[c]) { onlyRefCell = false; break; }
    check(onlyRefCell, "no other cell is touched");

    // --- stage 7: correctFlux -------------------------------------------------------------
    // At p == 0, matrixFlux is exactly -boundaryCoeffs on the boundary and 0 internally, so
    // phi == phiHbyA + boundaryCoeffs. Analytic, so it pins the sign and the composition together.
    std::printf("  -- stage 7: phi = phiHbyA - pEqn.flux()\n");
    const std::vector<scalar> pZero(nC, 0.0);
    const SurfaceScalarField phi0 = cpu::correctFlux(st, pEqn, pZero, m, patches);
    bool internalOk = true;
    for (std::size_t i = 0; i < phi0.internal.size(); ++i)
        if (std::fabs(phi0.internal[i] - st.phiHbyA.internal[i]) > 1e-14*std::fabs(st.phiHbyA.internal[i]) + 1e-300)
        { internalOk = false; break; }
    check(internalOk, "at p=0 the internal flux is unchanged");
    bool bndOk = true;
    for (std::size_t pi = 0; pi < patches.size() && bndOk; ++pi)
        for (std::size_t i = 0; i < phi0.boundary[pi].size(); ++i)
        {
            const scalar want = st.phiHbyA.boundary[pi][i] + pEqn.boundaryCoeffs[pi][i];
            if (std::fabs(phi0.boundary[pi][i] - want) > 1e-12*std::fabs(want) + 1e-300)
            { bndOk = false; break; }
        }
    check(bndOk, "at p=0 the boundary flux is phiHbyA + boundaryCoeffs");
    // Control: with the real p the correction must actually change phi.
    const SurfaceScalarField phiReal = cpu::correctFlux(st, pEqn, p.internal, m, patches);
    check(relS(phiReal.internal, st.phiHbyA.internal) > 1e-6,
          "with the real p the flux correction is non-trivial (control)");

    // --- relaxField -----------------------------------------------------------------------
    std::printf("  -- p.relax() (GeometricField.C:1094)\n");
    std::vector<scalar> pv = p.internal, pPrev(nC, 1.0), pAt1 = p.internal;
    cpu::relaxField(pAt1, pPrev, 1.0);
    check(pAt1 == p.internal, "alpha = 1 is the identity");
    std::vector<scalar> pAt = p.internal;
    cpu::relaxField(pAt, pPrev, 0.3);
    bool relOk = true;
    for (label c = 0; c < nC; ++c)
        if (std::fabs(pAt[c] - (pPrev[c] + 0.3*(pv[c] - pPrev[c]))) > 1e-15*std::fabs(pv[c]) + 1e-300)
        { relOk = false; break; }
    check(relOk, "alpha = 0.3 gives pPrev + alpha*(p - pPrev)");

    // --- refusals -------------------------------------------------------------------------
    std::printf("  -- refusals\n");
    // `consistent` and fixedFluxPressure are no longer here: SIMPLEC is implemented, and so is
    // constrainPressure (pEqn.H:21) -- the never-updated fixedFluxPressure case now refuses from the
    // patch itself (FixedFluxPressurePatchField::requireUpdated, gated in test_uniform_function1).
    const char* names[2] = {"MRF", "fvOptions"};
    for (int which = 0; which < 2; ++which)
    {
        cpu::PressureInput bad = in;
        if (which == 0) bad.hasMRF = true;
        else bad.hasFvOptions = true;
        bool threw = false;
        try { cpu::pressurePredictor(UEqn, U, p, bad, m, g, patches); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, (std::string(names[which]) + " is refused").c_str());
    }

    std::printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
