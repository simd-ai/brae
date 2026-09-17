// brae's interFoam turbulence against REAL OpenFOAM's, on RAS/damBreak, in BOTH of interFoam's lineages.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, as in
// test_inter_dambreak_vs_openfoam.cu, plus its own solver log for epsilon and k.
//
// THREE PROFILES -- tests/interfoam_ras_dambreak_vs_openfoam.sh says why each exists and carries the
// measurement of what each one can see:
//   variable   the tutorial as shipped (`density variable`): kEpsilon weighted by the mixture rho,
//              convecting with rhoPhi, and NO validate() -- the first UEqn runs on the file's nut = 0
//   uniform    the same case without that line: the ordinary single-phase model, validate() at
//              construction
//   custom     uniform with its own kEpsilonCoeffs, equation relaxation 0.7 and a tolerance at which
//              `minIter 1` is what forces each sweep -- three settings the shipped case cannot see
//
// TWO CONTROLS ON THE ORACLE, each OpenFOAM's own answer at the same instant:
//   laminar     the case with `simulationType laminar`. If the turbulent oracle sat on top of it, a
//               brae that ignored turbulence would pass every field bound here.
//   the other   the run this profile differs from in its one setting: the other lineage for
//               `variable` and `uniform`, the plain uniform run for `custom`.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include "inter_solve_log.cuh"
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
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/damBreak (kEpsilon) ==\n");
    if (argc < 10)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> "
                    "<variable|uniform|custom> <lineage> <laminarOfTimeDir> <otherOfTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string profile = argv[6];
    const bool variable = (std::string(argv[7]) == "variable");
    const std::string laminarDir = argv[8];
    const std::string otherDir = argv[9];
    const bool custom = (profile == "custom");
    std::printf("  profile: %s\n",
                custom ? "custom -- its own coefficients, relaxation 0.7, and minIter forcing each sweep"
              : variable ? "variable -- `density variable`, as the tutorial ships"
                         : "uniform -- the ordinary single-phase lineage");

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
    check("...in the lineage this profile names", fin.turbulence.variableDensity == variable);
    if (custom)
    {
        // what was READ, beside the answer it produces: the oracle control below says the three
        // settings move OpenFOAM, and these say brae picked each of them up from the case
        const InterTurbulence& t = fin.turbulence;
        check("brae read the case's six kEpsilonCoeffs",
              t.coeffs.Cmu == scalar(0.12) && t.coeffs.C1 == scalar(1.5) && t.coeffs.C2 == scalar(1.8)
           && t.coeffs.C3 == scalar(0.2) && t.coeffs.sigmaK == scalar(1.1)
           && t.coeffs.sigmaEps == scalar(1.2));
        check("...the 0.7 relaxation under the names kFinal and epsilonFinal",
              t.kRelaxFinal.on && t.kRelaxFinal.factor == scalar(0.7)
           && t.epsRelaxFinal.on && t.epsRelaxFinal.factor == scalar(0.7));
        check("...and minIter 1 at tolerance 0.5",
              t.kSolveFinal.minIter == 1 && t.kSolveFinal.tol == scalar(0.5)
           && t.epsSolveFinal.minIter == 1);
    }

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

    // the solver logs: p_rgh and alpha as on the laminar case, and the closure's two
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofA = brae::gatecheck::readOfSolves(logPath, fin.alphaName);
    const std::vector<LinearSolveRecord> ofE = brae::gatecheck::readOfSolves(logPath, "epsilon");
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    // the FAIL-PROOF: nothing in compareSolves can pass on an empty parse
    check("OpenFOAM's log gave one epsilon and one k solve per step",
          ofE.size() == static_cast<std::size_t>(nSteps) && ofK.size() == ofE.size());
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps);
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               scalar(1e-10), scalar(1e-9));
    // MEASURED: initial residuals 1.8e-13 (epsilon) and 1.3e-13 (k) from OpenFOAM's over the run, so
    // both bounds are 1e-10. The FINAL residual's worst is 8.8e-07 relative and that is round-off, not
    // the solver: it is step one's k solve, which ends at 4.140e-10, and the two codes are 3.6e-16
    // apart there -- a few units in the last place of a normalised residual. PBiCGStab in the same
    // seat takes 1 of 5 iteration counts and leaves final residuals 100% out.
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
       && ofNut.size() == ofAlpha.size());

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

    // MEASURED, worst of the three profiles: alpha 4.9e-13, p_rgh 7.1e-13, U 1.3e-11, k 2.8e-13,
    // epsilon 2.4e-13, nut 4.8e-13. Each bound is about 30x its measurement. The smallest thing this
    // port could get wrong and still run -- handing inletOutlet rhoPhi where it looks up phi -- is
    // 4.9e-04 of U, six orders outside.
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < scalar(2e-11));
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < scalar(2e-11));
    check("U agrees with OpenFOAM's relatively", dU.rel() < scalar(5e-10));
    check("k agrees with OpenFOAM's relatively", dK.rel() < scalar(1e-11));
    check("epsilon agrees with OpenFOAM's relatively", dE.rel() < scalar(1e-11));
    check("nut agrees with OpenFOAM's relatively", dN.rel() < scalar(2e-11));

    // THE CONTROLS, on the oracle
    const Diff dLamU = compare(readVectorCells(laminarDir + "/U"), ofU);
    const Diff dOtherU = compare(readVectorCells(otherDir + "/U"), ofU);
    const Diff dOtherNut = compare(readCells(otherDir + "/nut"), ofNut);
    std::printf("  CONTROL: OpenFOAM laminar against OpenFOAM turbulent, U relative %.4e\n",
                (double)dLamU.rel());
    std::printf("  CONTROL: OpenFOAM without this profile's setting against OpenFOAM with it, U relative "
                "%.4e, nut %.4e\n", (double)dOtherU.rel(), (double)dOtherNut.rel());
    // MEASURED 1.33 (laminar), 0.29 (the other lineage) and 3.7e-02 (custom against plain uniform),
    // beside a U bound of 5e-10
    check("turbulence moves OpenFOAM's own U by more than 10%", dLamU.rel() > scalar(0.1));
    check(custom ? "...and the custom settings move it by more than 1%"
                 : "...and the lineage moves it by more than 10%, so `density` is live on this fixture",
          dOtherU.rel() > (custom ? scalar(0.01) : scalar(0.1)));

    std::printf("test_inter_ras_dambreak_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
