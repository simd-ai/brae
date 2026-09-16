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
//   1.4e-04 against OpenFOAM's 2.59e-01 -- three orders down, the meniscus never forms. With it on,
//   brae gets 2.85e-01. The correction is doing the physics, not decorating it.
//
//   OPEN: brae's curvature is about 7.5% HIGH. After ONE step -- where alpha has not moved at all, so
//   the alpha equation is out of the picture and U comes entirely from the surface-tension force --
//   brae's worst wall cell is U = (-1.902e-02, 2.782e-01) against OpenFOAM's (-1.654e-02, 2.589e-01).
//   Over five steps that reaches 25% at the contact line while the PEAK velocity agrees to 2.1%.
//
//   WHAT HAS BEEN ELIMINATED, each by measurement rather than by reading:
//     * the alpha equation -- at step 1 alpha agrees exactly, because it has not moved
//     * the gradient scheme -- capillaryRise and damBreak both say `default Gauss linear`
//     * theta() -- constantAlphaContactAngle returns theta0 uniformly, which is what brae does
//     * relaxation -- capillaryRise names NO relaxationFactors, and brae now reads that (it was
//       hardcoded on, which was a real defect; the dominance clamp turned out to be a no-op because
//       rho/dt is 1e8, so the number did not move)
//     * deltaN -- 6.3e-05 against a |gradAlpha| of order 6e+03, i.e. six orders below relevance
//     * THE BOUNDARY GRADIENT -- this WAS the bug, and fixing it took the error from 76.5% to 24.9%.
//       fvc::grad runs gaussGrad::correctBoundaryConditions, which replaces the wall-normal component
//       of the boundary gradient with the patch's own snGrad; brae was using the raw cell gradient,
//       so on a contact-angle patch it discarded exactly the quantity the contact angle sets.
//
//     * alphaContactAngle's NON-IDEMPOTENT evaluate() -- it mutates its own gradient under
//       `limit gradient`, so the number of times each solver evaluates the alpha boundary could
//       matter. It does not: the clamp bites on 2 of 800 faces and in 3 of 20 calls, and FORCING
//       idempotence makes the error four times worse (24.9% -> 128%), so the mutation is
//       load-bearing and its call count is not the problem.
//     * A PURE SCALE on the surface-tension force. Scaling sigma by 0.93 makes the peak velocity
//       match EXACTLY (ratio 1.000) and makes the worst-cell error WORSE (1.95e-02 -> 2.37e-02). So
//       the magnitude is not the story: the curvature FIELD has a different shape near the contact
//       line, and a solver tuned to match the peak would be further from OpenFOAM everywhere else.
//
//     * THE CURVATURE FORMULA. tools/dumpInterfaceK now reads OpenFOAM's own K from its UNMODIFIED
//       interfaceProperties, and tests/interfoam_curvature_vs_openfoam.sh compares brae's against it
//       at FOUR calculateK pass counts: agreement is 2.5e-14 relative at every one, i.e. round-off.
//       The formula, the contact-angle rotation, deltaN and the wall gradient are all exactly right.
//
//   THAT LEFT ONE THING, AND IT WAS REAL: calculateK is a FIXED POINT, not a pure function. It reads
//   alpha's wall gradient, which correctContactAngle wrote at the end of its own previous pass, so on
//   capillaryRise the wall gradient runs 7070.5 -> 8659.4 -> 9353.1 -> 9681.2 over four passes. brae
//   was running it in the wrong PLACES: at the top of each alpha corrector instead of the bottom, not
//   at all in createFields, and not at all in the mixture.correct() between the sub-cycle and UEqn.
//   Moving those to interFoam.C's own call sites took this gate from 24.9% to 12.8%.
//
//   WHAT IS LEFT is the remaining call-sequence difference. brae and OpenFOAM now make the same FOUR
//   calculateK passes before the first momentum equation, but the alpha boundary is EVALUATED a
//   different number of times between them -- MULES, the sub-cycle reset and evaluateBoundary each
//   trigger one, and each runs alphaContactAngle's clamp. Counting those on both sides is the next
//   step, and it is now the only candidate left.
//
//   The arms below record the discrepancy at its measured size: they fail if it grows, and they fail
//   if it disappears without this comment being updated.
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

    const scalar rel = uLinf / uRef;
    std::printf("  OPEN: brae is %.1f%% off OpenFOAM, worst at the contact line\n", (double)(100*rel));
    // TIGHTENED twice from its original 0.90: to 0.30 after the boundary-gradient fix (76.5% ->
    // 24.9%), and to 0.15 after the calculateK call sites were moved to interFoam.C's own (-> 12.8%).
    check("the known discrepancy has not GROWN", rel < scalar(0.15));
    check("...and if it has been FIXED, this arm fails so the finding gets closed rather than forgotten",
          rel > scalar(0.01));

    std::printf("test_inter_capillary_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
