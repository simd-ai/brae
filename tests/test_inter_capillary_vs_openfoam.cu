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
//   AND THE 0.7% THAT WAS LEFT IS CLOSED TOO -- 5e-08 now -- and it was THREE defects, all of them
//   at a boundary and none of them visible in a field away from one. Found by running this gate at
//   1, 2, 3, 4 and 5 steps (0.2%, 0.4%, 0.8%, 0.7%, 0.7%, so something was already wrong in step ONE,
//   before the contact line had moved a single face) and then reading the pressure corrector's terms
//   against tools/dumpInterFoam, split wall cells from the rest, as the block below still does:
//
//     1. mixture.correct() IS calcNu() AND THEN interfaceProperties::correct()
//        (immiscibleIncompressibleTwoPhaseMixture.H:78-82). The second call rewrites the contact
//        angle's gradient and so alpha's patch value; brae blended the wall viscosity AFTER it. UEqn.A()
//        was exact in every cell off the wall and 1.26% high in the air cells at it. That was the
//        whole of step one's 0.2%.
//
//     2. pressureInletOutletVelocity, inletOutlet and totalPressure look phi UP when they update.
//        brae's are told, through updateFromFlux, and interFoam never told them -- so the bottom
//        inlet, which DRAWS WATER IN, was treated as an outflow for the whole run and lost its two
//        fixed tangential components: 2/3 muEff magSf deltaCoeffs of UEqn.A() in every cell of the
//        bottom row, 5.33e+05 by that formula and 5.339e+05 measured. It appears at step two because
//        step one starts from rest.
//
//     3. `alpha2 = 1.0 - alpha1` (alphaEqn.H:223) runs ONE LINE ABOVE the corrector's
//        mixture.correct(), so alpha2's patch values are one contact-angle pass older than alpha1's,
//        and `rho == alpha1*rho1 + alpha2*rho2` blends the two. Worth 7.4e-05 of rho*nu at the wall;
//        with it the wall's muEff matches OpenFOAM's to 1e-17.
//
//   The device path had all three and one more of its own -- it never called
//   deviceUpdatePressureInletOutletVelocity -- and it agreed with the host only because both made the
//   same omission. The last arm of this file holds the device driver against OpenFOAM directly.
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
#include "device_gate_finite.cuh"
#include "inter_peqn_cpp.cuh"
#include <cuda_runtime.h>

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
    PressureTaps taps;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, true, &fin,
                                     scalar(1.0e300), &taps);
    check("brae ran the same number of steps", r.steps == nSteps);

    // worstU accumulates with std::fmax, which drops NaN: a non-finite brae U would read 0 error.
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    check("OpenFOAM's U has one value per cell", ofU.size() == fin.U.internal.size());
    scalar uRef = 0;
    const scalar uLinf = worstU(fin.U.internal, ofU, uRef);
    scalar braeMax = 0;
    for (const vector& v : fin.U.internal) braeMax = std::fmax(braeMax, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
    std::printf("  WITH it: max |U| brae %.4e vs OpenFOAM %.4e (ratio %.3f); worst cell error %.4e\n",
                (double)braeMax, (double)uRef, (double)(braeMax/uRef), (double)uLinf);

    check("brae's velocity is the right ORDER, so the correction is doing the physics",
          braeMax > scalar(0.5)*uRef && braeMax < scalar(2)*uRef);

    // alpha AND p_rgh, which OpenFOAM writes at the same instant. U alone cannot say whether a gap
    // came in through the alpha sub-cycle or through the pressure corrector; these two can.
    {
        auto readCells = [&](const std::string& path)
        {
            const FieldData<scalar> fd = readField<scalar>(path);
            if (fd.internalUniform)
            {
                return std::vector<scalar>(static_cast<std::size_t>(nC), fd.internalUniformValue);
            }
            return fd.internalField;
        };
        const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
        const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
        failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
        failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
        check("OpenFOAM's alpha and p_rgh have one value per cell",
              ofAlpha.size() == static_cast<std::size_t>(nC) && ofPrgh.size() == static_cast<std::size_t>(nC));
        scalar wa = 0, wp = 0, sp = 0;
        label ca = -1;
        for (label c = 0; c < nC && ofAlpha.size() == static_cast<std::size_t>(nC); ++c)
        {
            const scalar ea = std::fabs(fin.alpha1.internal[c] - ofAlpha[c]);
            if (ea > wa)
            {
                wa = ea;
                ca = c;
            }
            wp = std::fmax(wp, std::fabs(fin.p_rgh.internal[c] - ofPrgh[c]));
            sp = std::fmax(sp, std::fabs(ofPrgh[c]));
        }
        std::printf("  alpha: worst %.4e", (double)wa);
        if (ca >= 0)
        {
            std::printf(" at cell %ld (brae %.10f, OpenFOAM %.10f)",
                        (long)ca, (double)fin.alpha1.internal[ca], (double)ofAlpha[ca]);
        }
        std::printf(";  p_rgh: worst %.4e of %.4e\n", (double)wp, (double)sp);
        // MEASURED 4.6e-09 and 8.1e-08 relative after five steps; both were 4.6e-05 and 1.3e-04 before
        // the three boundary defects at the top of this file were fixed.
        check("alpha agrees with OpenFOAM's to 1e-7", wa < scalar(1e-7));
        check("p_rgh agrees with OpenFOAM's to 2e-6 relative", wp < scalar(2e-6)*std::fmax(sp, scalar(1e-30)));
    }

    // THE PRESSURE CORRECTOR TERM BY TERM, when the staging script ran tools/dumpInterFoam rather than
    // interFoam. Both sides hold the LAST corrector of the last step. Each line splits cells (or faces)
    // touching the wall from the rest, because the open gap is worst at the contact line.
    if (std::filesystem::exists(ofDir + "/rAU.dump"))
    {
        std::vector<bool> wallCell(static_cast<std::size_t>(nC), false);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].name != "walls") continue;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                wallCell[patches[pi].faceCells[i]] = true;
            }
        }
        const label nIf = m.nInternalFaces();
        auto report = [&](
            const char* name,
            const std::vector<scalar>& brae,
            const std::string& file,
            bool faces)
        {
            if (!std::filesystem::exists(file))
            {
                std::printf("    %-10s (no dump)\n", name);
                return;
            }
            const FieldData<scalar> fd = readField<scalar>(file);
            const std::size_t n = static_cast<std::size_t>(faces ? nIf : nC);
            std::vector<scalar> of = fd.internalUniform
                ? std::vector<scalar>(n, fd.internalUniformValue)
                : fd.internalField;
            if (of.size() != n || brae.size() != n)
            {
                std::printf("    %-10s size mismatch (brae %zu, OpenFOAM %zu, expected %zu)\n",
                            name, brae.size(), of.size(), n);
                ++failures;
                return;
            }
            failures += brae::gatecheck::nonFinite(name, brae);
            scalar wW = 0, wI = 0, sc = 0;
            std::size_t at = 0;
            for (std::size_t k = 0; k < n; ++k)
            {
                const bool wall = faces
                    ? (wallCell[m.owner()[k]] || wallCell[m.neighbour()[k]])
                    : wallCell[k];
                const scalar e = std::fabs(brae[k] - of[k]);
                if (wall && e > wW)
                {
                    wW = e;
                    at = k;
                }
                if (!wall)
                {
                    wI = std::fmax(wI, e);
                }
                sc = std::fmax(sc, std::fabs(of[k]));
            }
            std::printf("    %-10s wall %.3e  rest %.3e  of %.3e  (relative %.2e; worst wall %s %zu: brae %.9e, OF %.9e)\n",
                        name, (double)wW, (double)wI, (double)sc,
                        (double)(std::fmax(wW, wI)/std::fmax(sc, scalar(1e-300))),
                        faces ? "face" : "cell", at, (double)brae[at], (double)of[at]);
        };
        std::printf("  pressure corrector against tools/dumpInterFoam, last corrector of step %ld:\n",
                    (long)nSteps);
        report("UEqn.A", taps.A, ofDir + "/UEqnA.dump", false);
        report("rAU", taps.rAU, ofDir + "/rAU.dump", false);
        report("rAUf", taps.rAUf, ofDir + "/rAUf.dump", true);
        report("stf", taps.stf, ofDir + "/stf.dump", true);
        report("snGradRho", taps.snGradRho, ofDir + "/snGradRho.dump", true);
        report("phig", taps.phig, ofDir + "/phig.dump", true);
        report("phiHbyA", taps.phiHbyA, ofDir + "/phiHbyA.dump", true);
        report("rho", fin.rho, ofDir + "/rho.dump", false);
    }

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
            }
            // ACTIVE means above 1e-6 of the largest gradient on the wall, not "non-zero". The count
            // used to threshold at 1e-30, which counts round-off: with the two codes agreeing to 5e-09
            // of the peak everywhere, brae still read 11.7% "more faces active" after two steps,
            // every one of them a face holding +2e-05 where OpenFOAM holds -3e-05 against a peak of
            // 1e+04. That arm FAILED at STEPS=2 on a contact line that was in the same place.
            for (std::size_t i = 0; i < bg.size(); ++i)
            {
                if (std::fabs(bg[i]) > scalar(1e-6)*r)
                {
                    ++nbz;
                }
                if (std::fabs(op->gradientValues[i]) > scalar(1e-6)*r)
                {
                    ++noz;
                }
            }
            std::printf("  wall gradient after the run: %zu active faces (OpenFOAM %zu); "
                        "worst %.4e of %.4e at face %zu (brae %.6e, OF %.6e)\n",
                        nbz, noz, (double)w, (double)r, iw,
                        (double)bg[iw], (double)op->gradientValues[iw]);
            std::printf("    sum|brae| %.6e  sum|OF| %.6e  ratio %.9f\n",
                        (double)sb, (double)so, (double)(so > 0 ? sb/so : 0));
            check("the contact line covers exactly the faces OpenFOAM's does", nbz == noz && noz > 0);
            // MEASURED 5.2e-09 of the peak after five steps. It was 4.5e-05 of it (0.446 of 9998).
            check("...and carries OpenFOAM's gradient to 1e-7 of its peak", w < scalar(1e-7) * r);
        }
    }

    const scalar rel = uLinf / uRef;
    std::printf("  brae is %.3e of |U| off OpenFOAM\n", (double)rel);
    // TIGHTENED five times from its original 0.90, each after a measured fix: 0.30 (the boundary
    // gradient, 76.5% -> 24.9%), 0.15 (the calculateK call sites -> 12.8%), 0.02 (the boundary mixture
    // -> 0.7%), and now 1e-6 (the three boundary defects at the top of this file -> 5.0e-08; the worst
    // of the five step counts is 1.4e-07, at four).
    check("brae agrees with OpenFOAM on the case surface tension decides, to 1e-6", rel < scalar(1e-6));

    // THE DEVICE DRIVER, against OpenFOAM directly and not through the host. damBreak's device gate
    // compares device with host, and on this case that comparison would have passed while BOTH were
    // 0.7% out: all three defects above were in both paths. And the device had two more of its own that
    // only this case can see:
    //
    //   its alpha-boundary hook ran the host's calculateK on EVERY call, including the sub-step's reset
    //   of alpha1, which is not a mixture.correct(). At a contact-angle wall each pass rewrites the
    //   gradient, so the device took five curvature passes a step where OpenFOAM takes three: K 0.6%
    //   out at the contact line and U 1.9% out after ONE step, with every other term exact.
    //
    //   it never called deviceUpdatePressureInletOutletVelocity, so an inflow face stayed zeroGradient.
    //
    // TWO STATEMENTS, AND THEY NEED TWO OpenFOAM RUNS. With every linear solve tightened on BOTH codes
    // the device is OpenFOAM's to 3.5e-08 -- that is the discretisation, and it is checked. At the
    // case's OWN tolerances (p_rgh relTol 0.05 on the first two correctors) it is 9.2e-04, because the
    // device still runs its own solver on p_rgh where the host runs the case's PCG with DIC, and a
    // different solver at the same relTol stops somewhere else. That is recorded as OPEN, bounded from
    // above so it cannot grow unnoticed; a device DIC is what closes it.
    {
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess)
        {
            cudaGetLastError();
            nDev = 0;
        }
        if (nDev <= 0)
        {
            std::printf("  (no CUDA device: the device arms are skipped)\n");
        }
        else
        {
            InterFields dev;
            const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
            check("the device driver ran the same number of steps", rd.steps == nSteps);
            failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
            scalar dRef = 0;
            const scalar dLinf = worstU(dev.U.internal, ofU, dRef);
            std::printf("  DEVICE at the case's own tolerances: %.4e of %.4e (%.3e)  OPEN -- its p_rgh "
                        "solver is not the case's\n",
                        (double)dLinf, (double)dRef, (double)(dLinf/dRef));
            check("...which leaves the device within 2e-3 of OpenFOAM, and no worse", dLinf < scalar(2e-3)*dRef);

            if (argc > 6)
            {
                const std::string tightCase = argv[5], tightOf = argv[6];
                const FieldData<vector> tfd = readField<vector>(tightOf + "/U");
                const std::vector<vector> tU = tfd.internalUniform
                    ? std::vector<vector>(static_cast<std::size_t>(nC), tfd.internalUniformValue)
                    : tfd.internalField;
                InterFields tdev;
                const RunReport rt = runInterFoamDevice(tightCase, tightCase + "/0", m, g, patches,
                                                        nSteps, false, &tdev);
                check("the device driver ran the tightened case too", rt.steps == nSteps);
                check("OpenFOAM's tightened U has one value per cell",
                      tU.size() == static_cast<std::size_t>(nC));
                failures += brae::gatecheck::nonFinite("device U (tight)", tdev.U.internal);
                scalar tRef = 0;
                const scalar tLinf = worstU(tdev.U.internal, tU, tRef);
                std::printf("  DEVICE with every solve tightened on both codes: %.4e of %.4e (%.3e)\n",
                            (double)tLinf, (double)tRef, (double)(tLinf/tRef));
                check("the DEVICE's discretisation is OpenFOAM's on the contact-angle case, to 1e-6",
                      tLinf < scalar(1e-6)*tRef);
            }
            else
            {
                std::printf("  (no tightened OpenFOAM run given: the device discretisation arm is skipped)\n");
            }
        }
    }

    std::printf("test_inter_capillary_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
