#include <algorithm>
// fvOptions framework + explicitPorositySource/DarcyForchheimer, _cpp reference.
//
// The dictionary half is asserted against what the CASE says, and the numerics half against an exact
// identity from OpenFOAM's source. Both matter and they fail differently: a mis-parsed cellZone applies a
// correct resistance to the wrong cells, and a mis-signed resistance applies the wrong one to the right
// cells. Neither shows up as a crash.
//
// Run: test_fvoptions_cpp <caseDir> <timeDir>
#include "fvOptions_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu;

static int g_fails = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    U.evaluateBoundary();

    const fvOptions::OptionList opts = fvOptions::read(caseDir, m);
    std::printf("test_fvoptions_cpp: nC=%d, %d option(s)\n", (int)nC, (int)opts.options.size());

    check(!opts.empty(), "the fvOptions dictionary was found and parsed");
    check(opts.firstUnsupported().empty(), "every active option's type is implemented");

    const fvOptions::Option* po = nullptr;
    for (const auto& o : opts.options) if (o.active && o.unsupported.empty()) { po = &o; break; }
    if (!po) { std::printf("  no usable option -- nothing to test\nFAIL\n"); return 1; }

    std::printf("  option '%s' type '%s' on %d cells (allCells=%d)\n",
                po->name.c_str(), po->type.c_str(), (int)po->cells.size(), (int)po->allCells);
    std::printf("  D diag (%.4g %.4g %.4g)   F diag (%.4g %.4g %.4g)\n",
                po->D.xx, po->D.yy, po->D.zz, po->F.xx, po->F.yy, po->F.zz);

    // ---- the SELECTION must be a real subset, not everything and not nothing ------------------
    check(!po->allCells, "selectionMode cellZone resolved to a zone, not to all cells");
    check(po->cells.size() > 0 && (label)po->cells.size() < nC,
          "the porous zone is a strict subset of the mesh (control)");

    // ---- the COEFFICIENTS, against the dictionary --------------------------------------------
    // simpleCar: d (5e7 -1000 -1000), f (0 0 0), identity coordinate system. The 0.5 lives in F.
    //
    // THIS ASSERTION WAS WRONG, and it is corrected here rather than relaxed. It used to require that
    // `d.y = -1000 survives with its sign`, which contradicts OpenFOAM: porosityModel.C's
    // adjustNegativeResistance -- called from DarcyForchheimer.C:67-68 and fixedCoeff.C:122-123 -- turns
    // a NEGATIVE component into `val*(-maxCmpt)`, i.e. POSITIVE and scaled by the largest component.
    // Here maxCmpt = 5e7, so d.y = -1000*(-5e7) = 5e10. That is the physical intent of a porous baffle:
    // flow passes along x and is blocked across it, and a literal negative resistance would ACCELERATE
    // the flow. All-negative is a fatal error in OpenFOAM, not a clamp.
    //
    // Found by porting fixedCoeff for angledDuctExplicitFixedCoeff, whose `alpha (500 -1000 -1000)` is
    // really (500, 500000, 500000): brae's momentum diagonal read 2.05e-04 against OpenFOAM's 3.15e-02
    // inside the porous zone and was exact everywhere else.
    check(std::fabs(po->D.xx - 5e7) < 1e-6*5e7, "d.x parsed and transformed to D.xx = 5e7");
    check(std::fabs(po->D.yy - 5e10) < 1e-9*5e10,
          "d.y = -1000 becomes +5e10 (adjustNegativeResistance)");
    check(std::fabs(po->D.xy) + std::fabs(po->D.xz) + std::fabs(po->D.yz) < 1e-6*5e7,
          "an identity coordinate system leaves D diagonal");

    // ---- the RESISTANCE, against OpenFOAM's own expression -----------------------------------
    // diag += V*tr(Cd);  source -= V*((Cd - I*tr(Cd)) & U),  Cd = nu*D + magU*F.
    // With f = 0 the whole thing is nu*D, so the identity is exact and checkable by hand.
    {
        FvVectorMatrix M;
        M.diag.assign(nC, 0.0);
        M.source.assign(nC, vector{0,0,0});
        const scalar nu = 1e-5;
        fvOptions::addSup(opts, M, U, nu, g);

        label touched = 0, untouched = 0;
        scalar worstDiag = 0, worstSrc = 0;
        for (label c = 0; c < nC; ++c)
        {
            const bool inZone = std::find(po->cells.begin(), po->cells.end(), c) != po->cells.end();
            if (!inZone) { if (M.diag[c] == 0.0) ++untouched; continue; }
            ++touched;
            const vector& u = U.internal[c];
            const scalar magU = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
            const scalar cdxx = nu*po->D.xx + magU*po->F.xx;
            const scalar cdyy = nu*po->D.yy + magU*po->F.yy;
            const scalar cdzz = nu*po->D.zz + magU*po->F.zz;
            const scalar iso = cdxx + cdyy + cdzz;
            worstDiag = std::fmax(worstDiag, std::fabs(M.diag[c] - g.V()[c]*iso));
            const scalar wantX = -g.V()[c]*((cdxx - iso)*u.x);
            worstSrc = std::fmax(worstSrc, std::fabs(M.source[c].x - wantX));
        }
        std::printf("  %d cells in the zone, %d outside left untouched; max|d| diag %.3e source %.3e\n",
                    (int)touched, (int)untouched, worstDiag, worstSrc);
        check(worstDiag < 1e-9, "diag += V*tr(Cd) matches DarcyForchheimerTemplates.C exactly");
        check(worstSrc  < 1e-9, "source -= V*((Cd - I*tr(Cd)) & U) matches exactly");
        check(untouched == nC - touched, "cells outside the zone get NO resistance (control)");

        // THE SIGN. The porosity must RESIST: a positive diagonal contribution. This is the check that
        // catches the double-negation trap (`eqn -= porosityEqn` inside, `UEqn == fvOptions(U)` outside);
        // getting one of the two wrong gives a porosity that accelerates the flow.
        scalar minDiag = 1e30;
        for (const label c : po->cells) minDiag = std::fmin(minDiag, M.diag[c]);
        std::printf("  %-62s min=%.3e\n", "the resistance is a SINK (positive diagonal)", minDiag);
        check(minDiag > 0.0, "the porosity resists rather than accelerates (sign control)");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
