// The _cpp reference for divDevReff, against OpenFOAM's own dump.
//
// This is the first component extracted onto the OF-mirrored architecture, so it is also the template for
// how every later one is proven:
//
//   1. the OpenFOAM oracle is a dump from OpenFOAM itself (validation/kEpsCorrect/divdevreff.dat,
//      produced by dumpDivDevReff), not another brae path -- a self-comparison proves nothing;
//   2. the sign convention is asserted, because the header says that is the one thing a transcription of
//      this function can get wrong without failing loudly;
//   3. a NEGATIVE CONTROL runs the same code with the known-wrong boundary viscosity and asserts it does
//      NOT match. Without it, a test that passes tells you nothing about whether it could ever fail.
//
// Run: test_divdevreff_cpp <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "linearViscousStress_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static scalar relErrV(const std::vector<vector>& a, const std::vector<vector>& b)
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
    GeometricField<scalar> nut =
        buildField<scalar>(readField<scalar>(caseDir + "/282/nut"), patches, m.nCells());

    // nuEff = nu + nut, cell and boundary. The boundary array uses nut's OWN boundary values, which on a
    // wall-function patch are nut_wall -- not the cell value.
    std::vector<scalar> nuEffC(m.nCells());
    for (label c = 0; c < m.nCells(); ++c) nuEffC[c] = nu + nut.internal[c];

    std::vector<std::vector<scalar>> nuEffB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& nv = nut.boundary[pi]->value();
        nuEffB[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i) nuEffB[pi][i] = nu + nv[i];
    }

    // The OpenFOAM oracle. dumpDivDevReff writes fvc::div(nuEff*dev2(T(grad U))) WITHOUT the leading
    // minus that appears in linearViscousStress.C, so the reference must be negated to compare.
    std::ifstream in(caseDir + "/divdevreff.dat");
    if (!in) { std::printf("FAIL no divdevreff.dat\n"); return 1; }
    label nC; in >> nC;
    std::vector<vector> dof(nC);
    for (auto& v : dof) in >> v.x >> v.y >> v.z;

    int fails = 0;

    // --- 1. the extracted _cpp primitive matches OpenFOAM -------------------------------------
    const std::vector<vector> mine = cpu::divDevReffExplicit(U, nuEffC, nuEffB, m, g, patches);

    std::vector<vector> minusOF(nC);
    for (label c = 0; c < nC; ++c) minusOF[c] = {-dof[c].x, -dof[c].y, -dof[c].z};

    const scalar rel = relErrV(mine, minusOF);
    const scalar tol = 1e-6;
    std::printf("  divDevReffExplicit vs -OpenFOAM   n=%d rel=%.3e %s\n",
                nC, rel, rel < tol ? "OK" : "FAIL");
    if (!(rel < tol)) ++fails;

    // --- 2. the SIGN is the assertion, not an accident ----------------------------------------
    // If the minus in linearViscousStress.C were dropped, `mine` would equal +dof instead. Assert the
    // wrong sign is genuinely wrong here, so a future edit that flips it cannot pass quietly.
    const scalar relWrongSign = relErrV(mine, dof);
    std::printf("  wrong-sign control                     rel=%.3e %s\n",
                relWrongSign, relWrongSign > 1e-3 ? "OK (differs)" : "FAIL (sign not pinned)");
    if (!(relWrongSign > 1e-3)) ++fails;

    // --- 3. NEGATIVE CONTROL: the boundary viscosity must matter ------------------------------
    // Re-run with nuEff on the boundary replaced by the OWNER CELL value -- the exact defect the header
    // warns about, and one brae has shipped before. If this still matched, the test would be blind to it.
    std::vector<std::vector<scalar>> nuEffB_wrong(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        nuEffB_wrong[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            nuEffB_wrong[pi][i] = nuEffC[patches[pi].faceCells[i]];
        }
    }
    const std::vector<vector> wrong =
        cpu::divDevReffExplicit(U, nuEffC, nuEffB_wrong, m, g, patches);
    const scalar relWrong = relErrV(wrong, minusOF);
    std::printf("  wall-nuEff negative control            rel=%.3e %s\n",
                relWrong, relWrong > tol ? "OK (detected)" : "FAIL (test is blind)");
    if (!(relWrong > tol)) ++fails;

    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
