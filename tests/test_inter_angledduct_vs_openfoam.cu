// brae's interFoam with an fvOption against REAL OpenFOAM's, on RAS/angledDuct as shipped.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_angledduct_vs_openfoam.sh stages it and carries the measurements. The option is an
// explicitPorositySource with DarcyForchheimer on the `porosity` cellZone, its resistance given in a
// frame turned 45 degrees about z. interFoam's momentum equation is force-dimensioned, so the model
// takes the mixture's rho and, finding no `thermo:mu`, rho*nu with the mixture's LAMINAR nu.
//
// THE CONTROL is OpenFOAM's own answer with the option `active no` at the same instant.
//
// THE DEVICE LOOP MUST REFUSE the case, naming the option.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include "inter_solve_log.cuh"
#include "patch_entry_lookup.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

// TWO ARMS, and the bounds differ by four orders for a MEASURED reason.
//
//   inactive   the option `active no`, in both codes. MEASURED: alpha 5.7e-16, p_rgh 3.8e-15, U 4.3e-15,
//              k 7.6e-15, epsilon 5.8e-15, nut 5.2e-15, p_rgh initial residuals within 1.2e-13. This is
//              the arm that holds the massFlowRate inlet, the tilted slip wall, the turbulent inlets and
//              kEpsilon on this mesh, at the floor.
//   porous     as shipped. MEASURED: alpha 1.6e-14, p_rgh 6.3e-11, U 5.1e-11, k 4.6e-12, epsilon 5.3e-12,
//              nut 5.7e-12, p_rgh initial residuals within 1.5e-10 in step one and 1.1e-09 over the run,
//              every iteration count OpenFOAM's. THE FOUR ORDERS ARE ROUND-OFF, MEASURED TWICE. The
//              resistance is 2e8 along the duct and 2e11 across it, and only its isotropic third goes
//              into the diagonal; the rest is explicit, fed back at 0.9985 per corrector. Multiplying
//              ONE entry of D by (1 + 2.2e-16) moves brae's p_rgh agreement from 6.3e-11 to 1.7e-11 --
//              the comparison cannot see below that. And the residuals: the first solve of the run agrees
//              to 1.1e-16; the second corrector's initial residual is 2.2e-05, a difference of numbers
//              1e5 times its size, and with the option off -- where it is 2.5e-02 -- the same line agrees
//              to 1e-13.
// Each bound is about 30x its measurement.
struct Bounds
{
    scalar alpha;
    scalar prgh;
    scalar U;
    scalar k;
    scalar epsilon;
    scalar nut;
    scalar pResidualStepOne;
    scalar pResidualRun;
};
const Bounds INACTIVE{2e-14, 2e-13, 2e-13, 3e-13, 2e-13, 2e-13, 1e-10, 1e-5};
const Bounds POROUS{5e-13, 2e-9, 2e-9, 2e-10, 2e-10, 2e-10, 5e-9, 3e-8};
//   porousWater  the duct started full of water, so rho is 1000 in the porous zone and the model's rho
//              weighting is visible (as shipped rho is exactly 1 there for the whole gated run, and the
//              kinematic form changes NO digit). MEASURED: alpha 6.6e-13, p_rgh 4.6e-12, U 2.8e-12, k
//              9.5e-14, epsilon 1.8e-13, nut 1.1e-13, p_rgh initial residuals within 6.1e-11.
const Bounds POROUS_WATER{2e-11, 2e-10, 1e-10, 3e-12, 6e-12, 3e-12, 1e-9, 2e-9};
//   uncorrected / limited0  the case's laplacianSchemes and snGradSchemes `default` take
//              nonOrthDeltaCoeffs with the correction flux left off. THE FIELD BOUNDS ARE POROUS's,
//              unchanged and met with room: MEASURED host p_rgh 1.20e-11, U 4.56e-11, k 2.55e-12,
//              epsilon 5.36e-12, nut 1.94e-12; device p_rgh 3.93e-11, U 3.84e-11, k 3.63e-12,
//              epsilon 5.69e-12, nut 2.90e-12. Against a control -- OpenFOAM's own `orthogonal` run --
//              of U 2.142e-03 and k 2.998e-02, which is what brae computed under both names.
//
//              ONLY THE p_rgh RESIDUAL BOUND DIFFERS, and its DENOMINATOR is why. compareSolves holds a
//              RELATIVE difference, and this arm's third corrector of step one starts already converged:
//              OpenFOAM's own log reads `Initial residual = 1.84022270527491e-07` there, against
//              1.03816627673642e-05 for the same corrector under `corrected`. The worst relative
//              difference is 1.363e-07 of 1.840e-07, which is 2.5e-14 ABSOLUTE; the `porous` arm's
//              1.472e-10 of 1.038e-05 is 1.5e-15 absolute. The two arms agree in absolute terms to
//              within an order -- both are round-off on p_rgh's normalisation -- and differ 56x in what
//              they are divided by. Every one of the 30 iteration counts is OpenFOAM's on BOTH arms and
//              both codes, which a wrong operator on a 44-degree mesh does not leave intact: the device
//              arm before these lines took 27 of 30.
const Bounds NONORTH{5e-13, 2e-9, 2e-9, 2e-10, 2e-10, 2e-10, 3e-7, 3e-7};

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
    scalar rel() const { return linf/std::fmax(refMax, scalar(1e-300)); }
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
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/angledDuct (explicitPorositySource) ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <otherOfTimeDir> <porous|uncorrected|limited0|inactive>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string inactiveDir = argv[6];
    const bool water = (std::string(argv[7]) == "porousWater");
    // `splitGrad`: grad(k) and grad(epsilon) name different schemes, which `fvc::grad` resolves by the
    // FIELD's name. It is the shipped `porous` case plus those two entries, so it takes porous's bounds.
    const bool splitGrad = (std::string(argv[7]) == "splitGrad");
    // `uncorrected` / `limited0`: the case's laplacianSchemes and snGradSchemes `default`, which take
    // nonOrthDeltaCoeffs WITHOUT the correction flux. Both are the shipped `porous` case with those two
    // entries changed, so both take porous's bounds; `argv[6]` is the `orthogonal` run, which is what
    // brae computed under both of these names.
    const bool uncorrected = (std::string(argv[7]) == "uncorrected");
    const bool limited0 = (std::string(argv[7]) == "limited0");
    const bool nonOrthArm = uncorrected || limited0;
    const bool porous = (std::string(argv[7]) == "porous") || water || splitGrad || nonOrthArm;
    const Bounds& B = water ? POROUS_WATER : (nonOrthArm ? NONORTH : (porous ? POROUS : INACTIVE));
    std::printf("  arm: %s\n", uncorrected ? "uncorrected -- nonOrthDeltaCoeffs, no correction flux"
                : limited0 ? "limited0 -- the same scheme reached through limitedSnGrad at k = 0"
                : splitGrad ? "splitGrad -- grad(epsilon) leastSquares beside grad(k) Gauss linear" : water ? "porousWater -- the duct started full of water, so rho is 1000 in the porous zone" : porous ? "porous -- as shipped" : "inactive -- the option switched off in both codes");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    // THE PATH, not only the answer: a laminar run of this case also reaches the end
    check("brae ran the case turbulent", fin.turbulence.on);
    if (splitGrad)
    {
        // WHAT WAS READ, per equation. Without this the arm would pass on a brae that resolved BOTH names
        // through `default` and never saw the two literal keys -- which is exactly the collapse it holds.
        std::printf("  SPLIT GRAD: k leastSquares %d limitK %.3g   |   epsilon leastSquares %d limitK %.3g\n",
                    (int)fin.turbulence.kGrad.leastSquares, (double)fin.turbulence.kGrad.cellLimitK,
                    (int)fin.turbulence.secondGrad.leastSquares, (double)fin.turbulence.secondGrad.cellLimitK);
        check("brae read grad(k) as the case's own Gauss linear",
              !fin.turbulence.kGrad.leastSquares && fin.turbulence.kGrad.cellLimitK == scalar(0));
        check("...and grad(epsilon) as leastSquares", fin.turbulence.secondGrad.leastSquares);
        check("...so the two differ, which is what this profile exists to run",
              fin.turbulence.kGrad.leastSquares != fin.turbulence.secondGrad.leastSquares);
    }
    if (nonOrthArm)
    {
        // WHAT WAS READ. Without this the arm would pass on a brae that resolved the entry to `corrected`
        // or to `orthogonal` and happened to land inside the bounds -- and `orthogonal` is exactly what it
        // used to resolve to. THE TWO FACTS ARE SEPARATE: nonOrthDeltaCoeffs on the implicit half, and no
        // correction flux. uncorrectedSnGrad.H:113-119 returns nonOrthDeltaCoeffs exactly as
        // correctedSnGrad.H:108-114 does; only orthogonalSnGrad.H:113-119 returns deltaCoeffs.
        std::printf("  SCHEME READ: laplacian `%s` corrected %d nonOrthCoeffs %d limit %.3g   |   "
                    "snGrad `%s` corrected %d nonOrthCoeffs %d limit %.3g\n",
                    fin.laplacianScheme.raw.c_str(), (int)fin.laplacianScheme.corrected,
                    (int)fin.laplacianScheme.nonOrthCoeffs, (double)fin.laplacianScheme.limitCoeff,
                    fin.snGradScheme.raw.c_str(), (int)fin.snGradScheme.corrected,
                    (int)fin.snGradScheme.nonOrthCoeffs, (double)fin.snGradScheme.limitCoeff);
        check("brae reads the laplacian's coefficients as nonOrthDeltaCoeffs",
              fin.laplacianScheme.nonOrthCoeffs);
        check("...and adds NO correction flux to it", !fin.laplacianScheme.corrected);
        check("...and the snGrad's coefficients the same way", fin.snGradScheme.nonOrthCoeffs);
        check("...with no correction there either", !fin.snGradScheme.corrected);
        // `limited 0`'s limiter is identically zero (limitedSnGrad.C:48-58), so it must not survive as a
        // cap the correction path would read -- the two profiles must reach ONE state.
        check("...and no surviving limiter coefficient, so `limited 0` and `uncorrected` are one state",
              fin.laplacianScheme.limitCoeff == scalar(0) && fin.snGradScheme.limitCoeff == scalar(0));
    }
    check("...under kEpsilon, in the uniform lineage",
          fin.turbulence.model == InterRasModel::KEpsilon && !fin.turbulence.variableDensity);
    // THE PATH: one active option, a DarcyForchheimer on a proper part of the mesh
    std::size_t nActive = 0;
    std::size_t nPorous = 0;
    for (const brae::cpu::fvOptions::Option& o : fin.fvOptions.options)
    {
        if (!o.active) continue;
        ++nActive;
        nPorous += o.cells.size();
    }
    std::printf("  fvOptions: %zu active option(s) on %zu of %d cells\n", nActive, nPorous, (int)nC);
    check(porous ? "brae read ONE active option, on a proper part of the mesh"
                 : "brae read the option as INACTIVE",
          porous ? (nActive == 1 && nPorous > 0 && nPorous < static_cast<std::size_t>(nC)) : nActive == 0);

    auto readCells = [&](const std::string& path)
    {
        const FieldData<scalar> fd = readField<scalar>(path);
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
    auto readVectorCells = [&](const std::string& path)
    {
        const FieldData<vector> fd = readField<vector>(path);
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

    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofA = brae::gatecheck::readOfSolves(logPath, fin.alphaName);
    const std::vector<LinearSolveRecord> ofE = brae::gatecheck::readOfSolves(logPath, "epsilon");
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    // the FAIL-PROOF: nothing in compareSolves can pass on an empty parse
    check("OpenFOAM's log gave one epsilon and one k solve per step",
          ofE.size() == static_cast<std::size_t>(nSteps) && ofK.size() == ofE.size());
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", B.pResidualStepOne,
                                               B.pResidualRun);
    if (!water)
    {
        failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                                   scalar(1e-10), scalar(1e-9));
    }
    else
    {
        // ALPHA IS UNIFORM TO ROUND-OFF in this arm, so MULESCorr's implicit solve is a solve of noise:
        // OpenFOAM itself runs it to the 1000-iteration cap every step, and the residuals of noise
        // agree to 1e-06 where the field agrees to 6.6e-13. The counts are asserted, the residuals
        // printed.
        brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                       scalar(1e-10), scalar(1e-9), scalar(-1), nullptr, false);
        bool sameCounts = r.alphaSolves.size() == ofA.size() && !ofA.empty();
        for (std::size_t q = 0; q < ofA.size() && q < r.alphaSolves.size(); ++q)
        {
            sameCounts = sameCounts && r.alphaSolves[q].nIterations == ofA[q].nIterations;
        }
        check("every alpha solve took OpenFOAM's iteration count, the 1000-iteration cap included", sameCounts);
    }
    failures += brae::gatecheck::compareSolves("host", r.epsilonSolves, ofE, nSteps, "epsilon",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));
    failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));

    // BEFORE any fmax: std::fmax drops a NaN -- see tests/device_gate_finite.cuh
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
    failures += brae::gatecheck::nonFinite("brae epsilon", fin.turbulence.epsilon.internal);
    failures += brae::gatecheck::nonFinite("brae nut", fin.turbulence.nut.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const std::vector<scalar> ofKf = readCells(ofDir + "/k");
    const std::vector<scalar> ofEf = readCells(ofDir + "/epsilon");
    const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
    check("OpenFOAM's fields have one value per cell",
          ofAlpha.size() == static_cast<std::size_t>(nC) && ofKf.size() == ofAlpha.size()
       && ofEf.size() == ofAlpha.size() && ofNut.size() == ofAlpha.size());

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    const Diff dK = compare(fin.turbulence.k.internal, ofKf);
    const Diff dE = compare(fin.turbulence.epsilon.internal, ofEf);
    const Diff dN = compare(fin.turbulence.nut.internal, ofNut);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
    std::printf("  epsilon: relative %.4e   (epsilon up to %.4e)\n", (double)dE.rel(), (double)dE.refMax);
    std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dN.rel(), (double)dN.refMax);

    // the wall nut, which the cell comparison cannot see: nutkWallFunction's own values
    const FieldData<scalar> ofNutFd = readField<scalar>(ofDir + "/nut");
    scalar dWallNut = 0;
    scalar wallNutScale = 0;
    std::size_t nWallFaces = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        const PatchFieldData<scalar>* b = findPatchEntry(ofNutFd.boundary, patches[pi]);
        if (!b || b->valueUniform) continue;
        const Diff d = compare(fin.turbulence.nut.boundary[pi]->value(), b->values);
        dWallNut = std::fmax(dWallNut, d.linf);
        wallNutScale = std::fmax(wallNutScale, d.refMax);
        nWallFaces += b->values.size();
    }
    std::printf("  wall nut: Linf %.4e on %zu wall faces (nut_w up to %.4e)\n", (double)dWallNut,
                nWallFaces, (double)wallNutScale);

    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B.alpha);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B.prgh);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B.U);
    check("k agrees with OpenFOAM's relatively", dK.rel() < B.k);
    check("epsilon agrees with OpenFOAM's relatively", dE.rel() < B.epsilon);
    check("nut agrees with OpenFOAM's relatively", dN.rel() < B.nut);
    check("the wall's nut is OpenFOAM's nutkWallFunction value",
          nWallFaces > 0 && dWallNut <= B.nut*std::fmax(wallNutScale, scalar(1e-300)));

    // THE FIXTURE CAN DISCRIMINATE: the model is live on it, in OpenFOAM's own numbers
    scalar nutMax = 0;
    scalar nuRatioMax = 0;
    for (std::size_t c = 0; c < ofNut.size(); ++c)
    {
        nutMax = std::fmax(nutMax, ofNut[c]);
        nuRatioMax = std::fmax(nuRatioMax, ofNut[c]/std::fmax(fin.nu[c], scalar(1e-300)));
    }
    std::printf("  OpenFOAM's nut reaches %.4e, %.1f times the mixture's nu\n", (double)nutMax,
                (double)nuRatioMax);
    check("OpenFOAM's eddy viscosity is above the laminar one somewhere", nuRatioMax > scalar(1));

    // THE CONTROL, on the oracle
    const Diff dOffU = compare(readVectorCells(inactiveDir + "/U"), ofU);
    const Diff dOffP = compare(readCells(inactiveDir + "/p_rgh"), ofPrgh);
    std::printf("  CONTROL: %s, U relative %.4e, p_rgh %.4e\n",
                nonOrthArm ? "OpenFOAM's `orthogonal` against OpenFOAM's own run of this arm -- what brae "
                             "computed under this name against what it should have"
                           : "OpenFOAM with the option inactive against OpenFOAM with it",
                (double)dOffU.rel(), (double)dOffP.rel());
    check(splitGrad ? "the GRADIENT SPLIT moves OpenFOAM's own U far more than brae is from it -- `argv[6]` "
                      "is the `porous` run, which is this same case with one `default` for both gradients, "
                      "so the only difference between them is grad(epsilon)'s scheme. MEASURED: epsilon "
                      "6.795e-03 over 27,870 of 28,000 cells, nut 8.547e-03, U 6.451e-04. On RAS/damBreak "
                      "the same profile reads EXACTLY ZERO -- that case ships `Gauss linear orthogonal` and "
                      "`upwind`, so no turbulence gradient is ever read -- which is why the fixture is here"
          : nonOrthArm ? "THE COEFFICIENT CHOICE moves OpenFOAM's own U far more than brae is from it. "
                         "`argv[6]` is this same case under `Gauss linear orthogonal`/`orthogonal` -- "
                         "deltaCoeffs where this arm takes nonOrthDeltaCoeffs -- and orthogonal is what "
                         "brae ran under BOTH `uncorrected` (behind a mesh refusal) and `limited 0` "
                         "(behind nothing). MEASURED, OpenFOAM against OpenFOAM: alpha 1.261e-05, p_rgh "
                         "4.552e-05, U 2.142e-03, k 2.998e-02, epsilon 6.430e-02, nut 4.805e-02, all "
                         "28,000 of 28,000 cells. The mesh is 44.5185 degrees non-orthogonal on 18,575 "
                         "of them, 1/cos = 1.4025, which is why the fixture is here"
          : porous ? "the porosity moves OpenFOAM's own U far more than brae is from it"
                   : "...and so does switching it back on",
          dOffU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14))
       && dOffU.rel() > (splitGrad ? scalar(1e-5) : scalar(1e-3)));

    // THE DEVICE LOOP RUNS IT, at the same bounds
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device arm is not exercised)\n");
    }
    else if (splitGrad)
    {
        // THE DEVICE REFUSES A GRADIENT SPLIT, by name. Its closure carries ONE pair of gradient flags in
        // the kernels (`sin.co`/`kin.co` are the model coefficients, whose gradKLeastSq/gradKLimitK are
        // k's), so grad(epsilon) would be computed with grad(k)'s scheme. The host honours each.
        InterFields dev;
        bool named = false;
        try
        {
            runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
            std::printf("  FAIL the device arm RAN a case whose two turbulence gradients differ\n");
            ++failures;
        }
        catch (const std::exception& e)
        {
            named = std::string(e.what()).find("one gradient for both equations") != std::string::npos;
            std::printf("  device refusal: %s\n", e.what());
        }
        check("the device arm refuses the gradient split under its own name", named);
    }
    else
    {
        // fvOptions' explicitPorositySource/DarcyForchheimer in the device UEqn, and the massFlowRate
        // inlet rebuilt from the mixture's boundary rho at every assembly -- the two the device loop
        // refused. Both transcribed from the host arm's lines, and held to the host's OWN bounds.
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        check("the device driver ran the same number of steps", rd.steps == nSteps);
        // ...and the DEVICE's own p_rgh solves against OpenFOAM's log, as the host arm above is held:
        // the case names GAMG with a GaussSeidel smoother and PCG with one on pcorr, so a substituted
        // solver shows here before it shows in a field.
        // the nonOrth arms' p_rgh residuals are held to the SAME bound as their host half, for the same
        // reason: a relative difference on an initial residual of 1.84e-07. Every other arm keeps the
        // default 1e-10/1e-5, which they meet.
        failures += nonOrthArm
            ? brae::gatecheck::compareSolves("device", rd.pSolves, ofP, nSteps, "p_rgh",
                                             B.pResidualStepOne, B.pResidualRun)
            : brae::gatecheck::compareSolves("device", rd.pSolves, ofP, nSteps);
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        const Diff eK = compare(dev.turbulence.k.internal, ofKf);
        const Diff eE = compare(dev.turbulence.epsilon.internal, ofEf);
        const Diff eN = compare(dev.turbulence.nut.internal, ofNut);
        std::printf("  DEVICE vs OpenFOAM: alpha %.4e, p_rgh %.4e, U %.4e, k %.4e, epsilon %.4e, "
                    "nut %.4e\n", (double)eA.linf, (double)eP.rel(), (double)eU.rel(), (double)eK.rel(),
                    (double)eE.rel(), (double)eN.rel());
        check("the DEVICE's alpha agrees with OpenFOAM's", eA.linf < B.alpha);
        check("...its p_rgh", eP.rel() < B.prgh);
        check("...its U", eU.rel() < B.U);
        check("...its k", eK.rel() < B.k);
        check("...its epsilon", eE.rel() < B.epsilon);
        check("...and its nut", eN.rel() < B.nut);
        check(nonOrthArm ? "the coefficient choice moves OpenFOAM's own U far more than the DEVICE is "
                           "from it -- the device arm took nonOrthDeltaCoeffs too, and its refusal of "
                           "`uncorrected` on a sheared mesh is lifted by these lines"
                         : "the porosity moves OpenFOAM's own U far more than the DEVICE is from it",
              dOffU.rel() > scalar(1000)*std::fmax(eU.rel(), scalar(1e-14)) && dOffU.rel() > scalar(1e-3));
    }

    std::printf("test_inter_angledduct_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
