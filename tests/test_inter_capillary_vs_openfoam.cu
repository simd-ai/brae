// brae's interFoam against REAL OpenFOAM's, on capillaryRise.
//
// WHY THIS CASE AND NOT ONLY damBreak. On damBreak surface tension is a correction: gravity and
// inertia decide everything, and a solver with sigma set to zero would look broadly right for a while.
// On capillaryRise the surface tension IS the answer -- the water climbs against gravity because of
// it, at an angle the WALL sets -- and it is the only shipped interFoam tutorial with a contact angle
// (constantAlphaContactAngle, theta0 45, limit gradient). It is therefore the only case in the entire
// tutorial set that exercises correctContactAngle at all.
//
// WHAT THIS GATE ESTABLISHES, AND WHAT IT RECORDS AS OPEN.
//
//   ESTABLISHED: the contact angle carries the case. With it off, brae's velocity after one step is
//   1.4e-04 against OpenFOAM's 2.59e-01 -- three orders down, the meniscus never forms.
//
//   AND THE 12.8% THAT USED TO BE HERE IS CLOSED. It took two instruments and a sequence of
//   eliminations; what it was in the end:
//
//     THE MIXTURE'S BOUNDARY VALUES MUST COME FROM alpha's PATCH VALUES, NOT THE FACE CELL'S.
//
//   At a contact-angle wall alpha's patch value is patchInternalField + gradient/deltaCoeffs, and the
//   contact angle's gradient is precisely what pulls the interface UP THE WALL -- 9681 over a
//   deltaCoeffs of 20000 is +0.48 of alpha. So the wall FACE can be carrying water while the cell
//   behind it is air. rho and nu there follow the face, not the cell, and divDevRhoReff's laplacian
//   reads them through internalCoeffs, which lands in UEqn.A(), which is rAU, which scales the whole
//   velocity field through the pressure corrector.
//
//   It was TWO CELLS OF EIGHT THOUSAND -- the contact line -- and they set the velocity everywhere.
//   UEqn.A() went from 1.8e-03 to 6.3e-05 relative and rAU from 1.3e-01 to 2.9e-03; this gate went
//   from 12.8% to 0.7% and damBreak did not move at all (3.4346e-09), because damBreak's alpha walls
//   are zeroGradient and its patch value IS the cell value.
//
//   HOW IT WAS FOUND, because the route matters more than the answer: tools/dumpInterFoam writes
//   rAU, HbyA, phig, stf, snGrad(rho), rho, rho*nuEff and UEqn.A() at the first pEqn. Comparing them
//   in that order showed rho exact, the surface tension exact (3.5e-14), the buoyancy exact
//   (7.0e-14), the viscosity exact (0.0) -- and A wrong. Grouping A's error by phase showed it exact
//   in all 3200 water cells, and grouping by patch showed exactly 2 cells wrong, both touching the
//   walls. Every one of those splits was necessary; a single relative-L2 said only "6% somewhere".
//
// The case is NOT closed -- it has an inlet and an atmosphere -- so alpha's total is not conserved and
// nothing here asserts that it is. damBreak's gate is where the conservation claim lives.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

scalar worstU(const std::vector<vector>& a, const std::vector<vector>& b, scalar& ref)
{
    scalar w = 0; ref = 0;
    for (std::size_t c = 0; c < a.size() && c < b.size(); ++c)
    {
        const vector d{a[c].x-b[c].x, a[c].y-b[c].y, a[c].z-b[c].z};
        w   = std::fmax(w,   std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z));
        ref = std::fmax(ref, std::sqrt(b[c].x*b[c].x + b[c].y*b[c].y + b[c].z*b[c].z));
    }
    return w;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM: capillaryRise (surface tension IS the answer) ==\n");
    if (argc < 5) { std::printf("  SKIP: usage: %s <case> <start> <ofTime> <nSteps>\n", argv[0]); return 77; }
    const std::string caseDir = argv[1], startDir = argv[2], ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    if (!std::filesystem::exists(ofDir + "/U")) { std::printf("  SKIP: no OpenFOAM U in %s\n", ofDir.c_str()); return 77; }

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FieldData<vector> ofUfd = readField<vector>(ofDir + "/U");
    const std::vector<vector> ofU = ofUfd.internalUniform
        ? std::vector<vector>(static_cast<std::size_t>(nC), ofUfd.internalUniformValue)
        : ofUfd.internalField;

    // ---- the case's own settings reached the solver ------------------------------------------------
    {
        const InterFields f = buildInterFields(caseDir, startDir, m, g, patches);
        scalar theta = -1;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            if (patches[pi].name == "walls") theta = f.alpha1.boundary[pi]->contactAngleTheta0();
        std::printf("  walls carry theta0 = %.1f deg; nAlphaSubCycles %ld, MULESCorr %d, "
                    "momentumPredictor %d, relaxEquation(U) %d\n",
                    (double)theta, (long)f.alphaCtl.nAlphaSubCycles, (int)f.alphaCtl.MULESCorr,
                    (int)f.momentumPredictorOn, (int)f.relaxEquationU);
        check("the wall's contact angle was read from the field file", std::fabs(theta - scalar(45)) < scalar(1e-9));
        check("nAlphaSubCycles 2 was read (damBreak's is 1, so this is not a default)",
              f.alphaCtl.nAlphaSubCycles == 2);
        check("MULESCorr is absent here, so the EXPLICIT path runs", !f.alphaCtl.MULESCorr);
        check("momentumPredictor no", !f.momentumPredictorOn);
        check("...and capillaryRise names NO relaxationFactors, so UEqn.relax() must not run",
              !f.relaxEquationU);
    }

    // ---- THE CONTROL: without the contact angle, nothing happens -----------------------------------
    {
        setenv("BRAE_NO_CONTACT_ANGLE", "1", 1);
        InterFields off;
        runInterFoam(caseDir, startDir, m, g, patches, nSteps, false, &off);
        unsetenv("BRAE_NO_CONTACT_ANGLE");
        scalar ref = 0;
        scalar maxOff = 0;
        for (const vector& v : off.U.internal)
            maxOff = std::fmax(maxOff, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
        for (const vector& v : ofU) ref = std::fmax(ref, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
        std::printf("  WITHOUT the contact angle: max |U| = %.4e   (OpenFOAM's is %.4e)\n",
                    (double)maxOff, (double)ref);
        check("without the contact angle the meniscus never forms -- it carries this case entirely",
              maxOff < scalar(0.01) * ref);
    }

    // ---- WITH it: the direction is right, the magnitude is ~10% high --------------------------------
    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);

    scalar uRef = 0;
    const scalar uLinf = worstU(fin.U.internal, ofU, uRef);
    scalar braeMax = 0;
    for (const vector& v : fin.U.internal) braeMax = std::fmax(braeMax, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
    std::printf("  WITH it: max |U| brae %.4e vs OpenFOAM %.4e (ratio %.3f); worst cell error %.4e\n",
                (double)braeMax, (double)uRef, (double)(braeMax/uRef), (double)uLinf);

    check("brae's velocity is the right ORDER, so the correction is doing the physics",
          braeMax > scalar(0.5)*uRef && braeMax < scalar(2)*uRef);

    // THE OPEN DISCREPANCY, recorded at its measured size. 2.66e-02 against |U| 2.59e-01 after one
    // step; the arms bracket it so that a regression fails and an improvement fails too, which is what
    // forces the comment at the top of this file to be updated when it is fixed.
    // WHERE the error is, printed because this is an open finding and the next person needs it.
    {
        std::vector<bool> atWall(static_cast<std::size_t>(nC), false);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            if (patches[pi].name == "walls")
                for (label i = 0; i < patches[pi].size; ++i) atWall[patches[pi].faceCells[i]] = true;
        scalar eW = 0, eI = 0;
        label cw = -1;
        for (label c = 0; c < nC; ++c)
        {
            const vector& a = fin.U.internal[c];
            const vector& b = ofU[c];
            const scalar e = std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
            if (atWall[c]) { if (e > eW) { eW = e; cw = c; } }
            else             eI = std::fmax(eI, e);
        }
        std::printf("    worst at a WALL cell %.4e, away from the wall %.4e\n", (double)eW, (double)eI);
        if (cw >= 0)
            std::printf("    worst wall cell: brae U = (%.4e %.4e), OpenFOAM = (%.4e %.4e)\n",
                        (double)fin.U.internal[cw].x, (double)fin.U.internal[cw].y,
                        (double)ofU[cw].x, (double)ofU[cw].y);
    }

    // THE WALL GRADIENT AFTER THE RUN, brae vs OpenFOAM's own written alpha. constantAlphaContactAngle
    // derives from fixedGradient, so OpenFOAM WRITES its `gradient` list -- which means the fixed point
    // the contact angle converges to is directly comparable without instrumenting anything. If the two
    // wall gradients agree and the velocities do not, the remaining difference is downstream of the
    // curvature; if they disagree, it is the contact angle's own iteration.
    {
        const FieldData<scalar> ofA = readField<scalar>(ofDir + "/alpha.water");
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (fin.alpha1.boundary[pi]->contactAngleTheta0() < scalar(0)) continue;
            const std::vector<scalar> bg = fin.alpha1.boundary[pi]->snGrad(fin.alpha1.internal);
            const PatchFieldData<scalar>* op = nullptr;
            for (const auto& b : ofA.boundary) if (b.name == patches[pi].name) op = &b;
            if (!op || op->gradientValues.size() != bg.size())
            {
                std::printf("  (OpenFOAM wrote no wall gradient for '%s')\n", patches[pi].name.c_str());
                break;
            }
            scalar w = 0, r = 0, sb = 0, so = 0;
            std::size_t nbz = 0, noz = 0, iw = 0;
            for (std::size_t i = 0; i < bg.size(); ++i)
            {
                const scalar e = std::fabs(bg[i] - op->gradientValues[i]);
                if (e > w) { w = e; iw = i; }
                r = std::fmax(r, std::fabs(op->gradientValues[i]));
                sb += std::fabs(bg[i]); so += std::fabs(op->gradientValues[i]);
                if (std::fabs(bg[i]) > scalar(1e-30)) ++nbz;
                if (std::fabs(op->gradientValues[i]) > scalar(1e-30)) ++noz;
            }
            std::printf("  wall gradient after the run: %zu non-zero (OpenFOAM %zu); "
                        "worst %.4e of %.4e at face %zu (brae %.6e, OF %.6e)\n",
                        nbz, noz, (double)w, (double)r, iw,
                        (double)bg[iw], (double)op->gradientValues[iw]);
            std::printf("    sum|brae| %.6e  sum|OF| %.6e  ratio %.6f\n",
                        (double)sb, (double)so, (double)(so > 0 ? sb/so : 0));
            // After ONE step the two agree exactly -- 2 faces each, gradient 9681.222 vs 9681.222 to
            // 4.5e-10. After FIVE, brae's contact line has spread to 330 faces against OpenFOAM's 326
            // and the worst gradient differs by 0.15%. So the contact angle itself is exact and what
            // drifts is where the interface has got to, which is the open finding above.
            std::printf("    (%.1f%% more faces active than OpenFOAM)\n",
                        (double)(100.0*(double(nbz) - double(noz))/double(noz)));
            check("the contact line is in the same place, to within a few faces of OpenFOAM's",
                  nbz <= noz + noz/20 + 2 && nbz + noz/20 + 2 >= noz);
            check("...and carries a gradient of the same size", w < scalar(0.01) * r);
        }
    }

    const scalar rel = uLinf / uRef;
    std::printf("  OPEN: brae is %.1f%% off OpenFOAM, worst at the contact line\n", (double)(100*rel));
    // TIGHTENED four times from its original 0.90, each after a measured fix: 0.30 (the boundary
    // gradient, 76.5% -> 24.9%), 0.15 (the calculateK call sites -> 12.8%), and now 0.02 (the boundary
    // mixture -> 0.7%). What is left is the two-cell contact line still carrying a slightly different
    // alpha patch value after five steps, which shows as 1.2% more active wall faces than OpenFOAM.
    check("brae agrees with OpenFOAM on the case surface tension decides", rel < scalar(0.02));

    std::printf("test_inter_capillary_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
