// brae's GAMG against REAL OpenFOAM's, inside interFoam: the p_rgh solves of a wave-tank tutorial
// whose fvSolution names `solver GAMG;`, run as shipped.
//
// THE ORACLE is OpenFOAM's log under its own debug switches, which change no digit of its answer
// (the script proves that on every run) and print two things a field cannot say:
//   `DebugSwitches { GAMGAgglomeration 1; }`  the hierarchy, level by level: cells, faces per cell
//                                             and lduAddressing::band()'s profile -- which moves
//                                             when the NUMBERING does, not only the merging
//   `DebugSwitches { GAMG 1; }`               the coarsest-level solve of EVERY V-cycle, "DICPCG:
//                                             Solving for coarsestLevelCorr": an iteration count and
//                                             a final residual that are functions of the coarsest
//                                             matrix and of the residual restricted all the way down
// ...beside the "GAMG:  Solving for p_rgh" line of every solve, and the written fields.
//
// tests/interfoam_gamg_vs_openfoam.sh says what each profile is for and what it measured.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include "inter_solve_log.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

struct Diff
{
    scalar linf = 0;
    scalar refMax = 0;
    scalar rel() const
    {
        return linf/std::fmax(refMax, scalar(1e-300));
    }
};

Diff compare(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.linf = std::fmax(d.linf, std::fabs(a[i] - b[i]));
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    return d;
}

Diff compare(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const vector e{a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z};
        d.linf = std::fmax(d.linf, mag(e));
        d.refMax = std::fmax(d.refMax, mag(b[i]));
    }
    return d;
}

// One row of GAMGAgglomeration::printLevels, serial: the level, its cells, faces per cell and band
// profile. The last two are printed at precision 4.
struct OfLevel
{
    label level = 0;
    label nCells = 0;
    scalar faceCellRatio = 0;
    scalar profile = 0;
};

std::vector<OfLevel> readOfLevels(const std::string& path)
{
    std::vector<OfLevel> out;
    std::ifstream in(path);
    std::string line;
    bool inTable = false;
    while (std::getline(in, line))
    {
        if (line.rfind("GAMGAgglomeration:", 0) == 0)
        {
            inTable = true;
            continue;
        }
        if (!inTable) continue;
        std::istringstream row(line);
        std::vector<std::string> t;
        std::string w;
        while (row >> w)
        {
            t.push_back(w);
        }
        // the heading's lines carry words or dashes; a level's row is eleven numbers
        if (t.size() != 11 || t[0].find_first_not_of("0123456789") != std::string::npos)
        {
            if (!out.empty()) break;
            continue;
        }
        OfLevel lv;
        lv.level = static_cast<label>(std::atol(t[0].c_str()));
        lv.nCells = static_cast<label>(std::atol(t[2].c_str()));
        lv.faceCellRatio = std::atof(t[4].c_str());
        lv.profile = std::atof(t[10].c_str());
        out.push_back(lv);
    }
    return out;
}

// a number as OpenFOAM's stream prints it at precision 4, read back
scalar atPrecision4(scalar v)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4g", static_cast<double>(v));
    return std::atof(buf);
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: GAMG for p_rgh ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <profile> "
                    "<pcgOfTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string profile = argv[6];
    const std::string pcgDir = argv[7];
    std::printf("  profile: %s\n", profile.c_str());

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/false, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    check("the case names GAMG for p_rghFinal, and brae read it so", fin.pSolveFinal.gamgSolver());

    // THE HIERARCHY
    const std::vector<OfLevel> ofLevels = readOfLevels(logPath);
    check("OpenFOAM printed its hierarchy", ofLevels.size() > 1);
    std::printf("  levels: OpenFOAM %zu, brae %zu\n", ofLevels.size(), r.gamgLevels.size());
    check("brae built as many levels as OpenFOAM", r.gamgLevels.size() == ofLevels.size());
    std::size_t sameCells = 0;
    std::size_t sameRatio = 0;
    std::size_t sameProfile = 0;
    for (std::size_t k = 0; k < ofLevels.size() && k < r.gamgLevels.size(); ++k)
    {
        const RunReport::GamgLevel& mine = r.gamgLevels[k];
        const scalar ratio = atPrecision4(scalar(mine.nFaces)/scalar(mine.nCells));
        const scalar prof = atPrecision4(mine.profile);
        std::printf("    level %2d  cells %8d / %8d   faces per cell %-8.4g / %-8.4g   profile %-10.4g / %-10.4g\n",
                    (int)ofLevels[k].level, (int)ofLevels[k].nCells, (int)mine.nCells,
                    (double)ofLevels[k].faceCellRatio, (double)ratio, (double)ofLevels[k].profile, (double)prof);
        if (mine.nCells == ofLevels[k].nCells)
        {
            ++sameCells;
        }
        if (ratio == ofLevels[k].faceCellRatio)
        {
            ++sameRatio;
        }
        if (prof == ofLevels[k].profile)
        {
            ++sameProfile;
        }
    }
    check("...every level has OpenFOAM's cell count", sameCells == ofLevels.size());
    check("...and OpenFOAM's faces per cell, to the four digits it prints", sameRatio == ofLevels.size());
    check("...and OpenFOAM's band PROFILE, which only the same numbering gives", sameProfile == ofLevels.size());

    // THE SOLVES. Same solver, same operation order: the FINAL residuals are asserted as well.
    // MEASURED over eighteen of the nineteen profiles: initial residuals 5.3e-12 in step one and at most
    // 4.4e-08 over the run, final residuals 4.6e-08 -- the first corrector is PCG at relTol 0.1 and
    // leaves both codes inside that ball. Bounds 1e-10, 1e-6 and 1e-6.
    //
    // `deep` READS 1.24e-05 ON TEN CONSECUTIVE SOLVES AND IT IS NOT THE SOLVER. Steps 15 to 19, both
    // correctors, the SAME relative offset on each, gone again at step 20, with alpha and p_rgh 6e-12
    // from OpenFOAM throughout. The top patch is pressureInletOutletVelocity, which branches on the
    // sign of the face flux, and at x = 11.5 OpenFOAM's flux there reads -8.0e-15, +9.8e-14 and
    // +3.4e-15 at t = 0.14, 0.15 and 0.16 -- a sign set by round-off, three orders under the 1e-11 the
    // two codes agree to. They take different branches on that ONE face; U's diagonal in the cell
    // under it changes, so does rAU and with it that face's coefficient in the p_rgh matrix -- on a
    // fixedValue patch whose value is the solution there (p_rgh = p0 = 0 in the air), so the solution
    // does not move and the matrix's row sum does. normFactor carries sumA*average(p_rgh), and every
    // residual of the step is divided by it. U in the two cells under the face is 1.6e-09 apart
    // (|U| 1e-05 there); all 40 iteration counts and all 125 coarsest-level solves are OpenFOAM's.
    const bool deep = (profile == "deep");
    const scalar runBound = deep ? scalar(1e-4) : scalar(1e-6);
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(1e-10),
                                               runBound, runBound);

    // STEP ONE'S LAST SOLVE, to every digit the log carries. Both codes start step one from the same
    // fields, so this residual is the V-cycles' own arithmetic and nothing else's.
    if (nSteps > 0 && !ofP.empty() && r.pSolves.size() == ofP.size())
    {
        const std::size_t k = ofP.size()/static_cast<std::size_t>(nSteps) - 1;
        const scalar e = brae::gatecheck::residualRelDiff(r.pSolves[k].finalResidual, ofP[k].finalResidual);
        std::printf("  step one's last p_rgh solve: OpenFOAM %.15e in %d, brae %.15e in %d; relative %.3e\n",
                    (double)ofP[k].finalResidual, ofP[k].nIterations, (double)r.pSolves[k].finalResidual,
                    r.pSolves[k].nIterations, (double)e);
        check("...step one's last solve left OpenFOAM's final residual", e < scalar(1e-9));
    }

    // THE COARSEST LEVEL, once per V-cycle. Every one starts from zero, so its initial residual is 1
    // by construction; the iteration count and the final residual are the information. MEASURED:
    // every count equal on every profile (3 to 125 of them), final residuals 4.4e-07 at worst
    // (`coarsest`, whose 459-cell level takes the most iterations). Bound 1e-5. With the coarsest
    // level solved to 1e-12 instead of the entry's own tolerance NOT ONE count is OpenFOAM's -- and
    // the fields do not move a digit, so this arm is the only thing that holds that decision.
    const std::vector<LinearSolveRecord> ofC = brae::gatecheck::readOfSolves(logPath, "coarsestLevelCorr");
    check("OpenFOAM logged a coarsest-level solve per V-cycle", !ofC.empty());
    failures += brae::gatecheck::compareSolves("host", r.gamgCoarsestSolves, ofC, nSteps,
                                               "coarsestLevelCorr", scalar(1e-10), scalar(1e-10),
                                               scalar(1e-5));

    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    auto cells = [&](const FieldData<scalar>& fd)
    {
        std::vector<scalar> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };
    auto vectorCells = [&](const FieldData<vector>& fd)
    {
        std::vector<vector> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };
    const std::vector<scalar> ofAlpha = cells(readField<scalar>(ofDir + "/" + fin.alphaName));
    const std::vector<scalar> ofPrgh = cells(readField<scalar>(ofDir + "/p_rgh"));
    const std::vector<vector> ofU = vectorCells(readField<vector>(ofDir + "/U"));

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    // MEASURED, worst of the nineteen profiles: alpha 4.0e-11 and p_rgh 1.9e-10 (`gaussSeidel`), U
    // 1.9e-09, and 1.4e-08 on `deep` for the reason given above. With PBiCGStab standing in for GAMG,
    // which is what this solver ran before, `shipped` read alpha 2.3e-02 and U 19%. Bounds at about
    // 25x, 25x and 15x; the wave gate's, measured with PCG on both sides, are 5e-9, 2e-8 and 5e-7.
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < scalar(1e-9));
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < scalar(5e-9));
    check("U agrees with OpenFOAM's relatively", dU.rel() < scalar(2e-7));

    // THE CONTROL, on the oracle: OpenFOAM's own answer with PCG and DIC in GAMG's place, at the
    // same tolerance. It is what brae's interFoam ran under a notice before this solver existed.
    const Diff cA = compare(cells(readField<scalar>(pcgDir + "/" + fin.alphaName)), ofAlpha);
    const Diff cU = compare(vectorCells(readField<vector>(pcgDir + "/U")), ofU);
    std::printf("  CONTROL: OpenFOAM with PCG against OpenFOAM with GAMG, alpha %.4e, U relative %.4e\n",
                (double)cA.linf, (double)cU.rel());
    check("swapping the solver moves OpenFOAM's own alpha far more than brae is from it",
          cA.linf > scalar(1000)*std::fmax(dA.linf, scalar(1e-16)));
    check("...and its U", cU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-16)));

    // THE DEVICE LOOP, against OpenFOAM directly -- what `brae_interFoam -device` runs. The hierarchy is
    // the host's own code; what is under test is the V-cycle on the device: the coarse matrices and the
    // restriction by fixed-order gathers, DIC through the level schedule, the scaling's device
    // reductions. The device's GAMG has the DIC smoother only and REFUSES the rest, which
    // tests/interfoam_refusals.sh holds; those three profiles have no device arm.
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    const bool dicSmoother = (fin.pSolveFinal.gamg.smoother == "DIC");
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device arm is skipped)\n");
    }
    else if (!dicSmoother)
    {
        std::printf("  (smoother %s: the device's GAMG refuses it, so there is no device arm)\n",
                    fin.pSolveFinal.gamg.smoother.c_str());
    }
    else
    {
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        check("the device driver ran the same number of steps", rd.steps == nSteps);
        check("...on the same hierarchy", rd.gamgLevels.size() == r.gamgLevels.size());
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        // STEP ONE at 5e-9 where the host is at 1e-10: the residual is a difference of two numbers of
        // order one summed in the device's order -- MEASURED 2.2e-10 at worst, as the wave gate's
        // device arm reads. Over the run 6.4e-09 (`deep` 2.5e-05, for the host's reason) and coarsest
        // final residuals 4.4e-07: the host's bounds hold both.
        failures += brae::gatecheck::compareSolves("device", rd.pSolves, ofP, nSteps, "p_rgh",
                                                   scalar(5e-9), runBound, runBound);
        failures += brae::gatecheck::compareSolves("device", rd.gamgCoarsestSolves, ofC, nSteps,
                                                   "coarsestLevelCorr", scalar(1e-10), scalar(1e-10),
                                                   scalar(1e-5));
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        const Diff hA = compare(dev.alpha1.internal, fin.alpha1.internal);
        const Diff hU = compare(dev.U.internal, fin.U.internal);
        std::printf("  DEVICE vs OpenFOAM: alpha %.4e, p_rgh %.4e, U %.4e\n", (double)eA.linf,
                    (double)eP.rel(), (double)eU.rel());
        std::printf("  DEVICE vs host:     alpha %.4e, U %.4e\n", (double)hA.linf, (double)hU.rel());
        check("the DEVICE's alpha agrees with OpenFOAM's to the host's bound", eA.linf < scalar(1e-9));
        check("...its p_rgh", eP.rel() < scalar(5e-9));
        check("...and its U", eU.rel() < scalar(2e-7));
    }

    std::printf("test_inter_gamg_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
