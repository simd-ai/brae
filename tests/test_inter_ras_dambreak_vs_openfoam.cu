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
// THE DEVICE LOOP RUNS EVERY PROFILE TOO, twice: with the device closure (what `-device` runs), held to
// OpenFOAM at the host's bounds, and with the HOST closure in its place (BRAE_INTER_HOST_CLOSURE), held
// to the first. Against OpenFOAM a disagreement could be the loop's or the closure's; between those two
// it can only be the closure's. Each arm asserts WHICH closure it ran.
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
#include <cuda_runtime.h>
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
    // nutAtmosphere: the uniform lineage with the atmosphere's nut an inletOutlet, which
    // correctBoundaryConditions evaluates after the model's field assignment
    const bool nutAtmosphere = (profile == "nutAtmosphere");
    // `sst`: RAS/damBreak made kOmegaSST, whose second field is omega and whose closure runs on the
    // device through device_inter_turbulence's SST branch
    const bool sst = (profile == "sst");
    const char* secondName = sst ? "omega" : "epsilon";
    std::printf("  profile: %s\n",
                nutAtmosphere ? "nutAtmosphere -- uniform, the atmosphere's nut an inletOutlet"
              : custom ? "custom -- its own coefficients, relaxation 0.7, and minIter forcing each sweep"
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
    const std::vector<LinearSolveRecord> ofE = brae::gatecheck::readOfSolves(logPath, secondName);
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    // the FAIL-PROOF: nothing in compareSolves can pass on an empty parse
    check("OpenFOAM's log gave one solve of the closure's second field and one k solve per step",
          ofE.size() == static_cast<std::size_t>(nSteps) && ofK.size() == ofE.size());
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps);
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               scalar(1e-10), scalar(1e-9));
    // MEASURED: initial residuals 1.8e-13 (epsilon) and 1.3e-13 (k) from OpenFOAM's over the run, so
    // both bounds are 1e-10. The FINAL residual's worst is 8.8e-07 relative and that is round-off, not
    // the solver: it is step one's k solve, which ends at 4.140e-10, and the two codes are 3.6e-16
    // apart there -- a few units in the last place of a normalised residual. PBiCGStab in the same
    // seat takes 1 of 5 iteration counts and leaves final residuals 100% out.
    failures += brae::gatecheck::compareSolves("host", sst ? r.omegaSolves : r.epsilonSolves, ofE, nSteps,
                                               secondName, scalar(1e-10), scalar(1e-10), scalar(1e-5));
    failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));

    // BEFORE any fmax: std::fmax drops a NaN -- see tests/device_gate_finite.cuh
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
    const std::vector<scalar>& braeSecond = sst ? fin.turbulence.omega.internal
                                                : fin.turbulence.epsilon.internal;
    failures += brae::gatecheck::nonFinite("brae second field", braeSecond);
    failures += brae::gatecheck::nonFinite("brae nut", fin.turbulence.nut.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const std::vector<scalar> ofKf = readCells(ofDir + "/k");
    const std::vector<scalar> ofEf = readCells(ofDir + "/" + std::string(secondName));
    const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
    check("OpenFOAM's fields have one value per cell",
          ofAlpha.size() == static_cast<std::size_t>(nC) && ofKf.size() == ofAlpha.size()
       && ofNut.size() == ofAlpha.size());

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    const Diff dK = compare(fin.turbulence.k.internal, ofKf);
    const Diff dE = compare(braeSecond, ofEf);
    const Diff dN = compare(fin.turbulence.nut.internal, ofNut);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
    std::printf("  %-8s relative %.4e   (%s up to %.4e)\n", (std::string(secondName) + ":").c_str(),
                (double)dE.rel(), secondName, (double)dE.refMax);
    std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dN.rel(), (double)dN.refMax);

    // MEASURED, worst of the three profiles: alpha 4.9e-13, p_rgh 7.1e-13, U 1.3e-11, k 2.8e-13,
    // epsilon 2.4e-13, nut 4.8e-13. Each bound is about 30x its measurement. The smallest thing this
    // port could get wrong and still run -- handing inletOutlet rhoPhi where it looks up phi -- is
    // 4.9e-04 of U, six orders outside.
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < scalar(2e-11));
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < scalar(2e-11));
    check("U agrees with OpenFOAM's relatively", dU.rel() < scalar(5e-10));
    check("k agrees with OpenFOAM's relatively", dK.rel() < scalar(1e-11));
    check("the closure's second field agrees with OpenFOAM's relatively", dE.rel() < scalar(1e-11));
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
    // MEASURED for nutAtmosphere against plain uniform: U 2.7e-03, nut 4.7e-02 -- 20 of the atmosphere's
    // 46 faces take air IN at t = 0.005, where the inletValue stands in for the cell's nut
    check(nutAtmosphere ? "...and the atmosphere's inletOutlet nut moves it by more than 1e-3"
          : custom ? "...and the custom settings move it by more than 1%"
                   : "...and the lineage moves it by more than 10%, so `density` is live on this fixture",
          dOtherU.rel() > (nutAtmosphere ? scalar(1e-3) : custom ? scalar(0.01) : scalar(0.1)));

    // THE DEVICE LOOP, against OpenFOAM directly and at the case's own tolerances -- what
    // `brae_interFoam -device` runs on this tutorial.
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device arm is skipped)\n");
    }
    else if (nutAtmosphere)
    {
        // the device closure carries no flux-conditional nut patch: it must refuse, by name
        bool named = false;
        try
        {
            InterFields dev;
            runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        }
        catch (const std::exception& e)
        {
            const std::string w = e.what();
            named = w.find("atmosphere") != std::string::npos && w.find("inletOutlet") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses an inletOutlet nut and names the patch", named);
    }
    else
    {
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        check("the device driver ran the same number of steps", rd.steps == nSteps);
        check("...turbulent, in the lineage this profile names",
              dev.turbulence.on && dev.turbulence.variableDensity == variable);
        // THE PATH: the device loop can carry either closure, and "agrees with OpenFOAM" is true of both
        check("...with the closure ON THE DEVICE", rd.turbulenceOnDevice);
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        failures += brae::gatecheck::nonFinite("device k", dev.turbulence.k.internal);
        const std::vector<scalar>& devSecond = sst ? dev.turbulence.omega.internal
                                                   : dev.turbulence.epsilon.internal;
        failures += brae::gatecheck::nonFinite("device second field", devSecond);
        failures += brae::gatecheck::nonFinite("device nut", dev.turbulence.nut.internal);
        failures += brae::gatecheck::compareSolves("device", rd.pSolves, ofP, nSteps);
        failures += brae::gatecheck::compareSolves("device", rd.alphaSolves, ofA, nSteps,
                                                   dev.alphaName.c_str(), scalar(1e-10), scalar(1e-9));
        // MEASURED: initial residuals 1.8e-13 (epsilon) and 4.0e-13 (k) from OpenFOAM's over the run.
        // THE CONTROL IS ON `custom`: with the wall laplacian coefficient left out of relax() -- what
        // the device closure did until this gate -- epsilon's fields do not move (9.8e-14) and its
        // initial residuals are 1.1e-04 out in every step. 1e-10 is six orders inside that.
        failures += brae::gatecheck::compareSolves("device", sst ? rd.omegaSolves : rd.epsilonSolves, ofE,
                                                   nSteps, secondName, scalar(1e-10), scalar(1e-10),
                                                   scalar(1e-5));
        failures += brae::gatecheck::compareSolves("device", rd.kSolves, ofK, nSteps, "k",
                                                   scalar(1e-10), scalar(1e-10), scalar(1e-5));
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        const Diff eK = compare(dev.turbulence.k.internal, ofKf);
        const Diff eE = compare(devSecond, ofEf);
        const Diff eN = compare(dev.turbulence.nut.internal, ofNut);
        std::printf("  DEVICE vs OpenFOAM: alpha %.4e, p_rgh %.4e, U %.4e, k %.4e, %s %.4e, nut %.4e\n",
                    (double)eA.linf, (double)eP.rel(), (double)eU.rel(), (double)eK.rel(), secondName,
                    (double)eE.rel(), (double)eN.rel());
        // THE HOST'S BOUNDS. MEASURED, worst of the three profiles: alpha 4.9e-13, p_rgh 7.1e-13,
        // U 3.3e-11, k 2.8e-13, epsilon 2.5e-13, nut 4.8e-13. Each device-side decision was broken in
        // turn, once: the patches handed rhoPhi for phi 4.9e-04 of U; the wall nut recomputed instead
        // of read from the stored patch 6.1e-06; rho.oldTime() taken as rho 5.6%; divU from the mass
        // flux 53%; BiCGStab for the case's symGaussSeidel 1 of 5 epsilon iteration counts.
        check("the DEVICE's alpha agrees with OpenFOAM's to the host's bound", eA.linf < scalar(2e-11));
        check("...its p_rgh", eP.rel() < scalar(2e-11));
        check("...its U", eU.rel() < scalar(5e-10));
        // `sst` HAS ITS OWN BOUNDS, and the reason is the device LOOP, not the closure. MEASURED with
        // every solve tightened on both codes -- which does not move them -- device against OpenFOAM:
        // k 5.2e-12, omega 3.1e-11, nut 1.9e-10, U 5.5e-11, where the HOST loop on the same case is
        // 6.7e-15, 1.9e-15, 1.3e-13 and 1.1e-13 and the device closure is the host closure's to 5e-15 /
        // 8e-16 / 3.7e-14 (the arm below). The device loop's U is about 5e-11 from OpenFOAM on the
        // kEpsilon profiles too; under kOmegaSST nut carries it through k/omega and F2, which is the
        // 1.9e-10. Bounds at about 30x the measurement.
        check("...its k", eK.rel() < (sst ? scalar(2e-10) : scalar(1e-11)));
        check("...its second closure field", eE.rel() < (sst ? scalar(1e-9) : scalar(1e-11)));
        check("...and its nut", eN.rel() < (sst ? scalar(5e-9) : scalar(2e-11)));

        // THE SAME DEVICE LOOP WITH THE HOST CLOSURE IN THE DEVICE ONE'S PLACE -- the `_cpp` reference
        // as the in-repo oracle, with everything around it held fixed. Against OpenFOAM a disagreement
        // could be the loop's or the closure's; against this it can only be the closure's.
        setenv("BRAE_INTER_HOST_CLOSURE", "1", 1);
        InterFields mix;
        const RunReport rm = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &mix);
        unsetenv("BRAE_INTER_HOST_CLOSURE");
        check("the mixed run took the HOST closure", !rm.turbulenceOnDevice && rm.steps == nSteps);
        failures += brae::gatecheck::nonFinite("mixed k", mix.turbulence.k.internal);
        const std::vector<scalar>& mixSecond = sst ? mix.turbulence.omega.internal
                                                   : mix.turbulence.epsilon.internal;
        failures += brae::gatecheck::nonFinite("mixed second field", mixSecond);
        failures += brae::gatecheck::nonFinite("mixed nut", mix.turbulence.nut.internal);
        failures += brae::gatecheck::nonFinite("mixed U", mix.U.internal);
        const Diff mU = compare(dev.U.internal, mix.U.internal);
        const Diff mK = compare(dev.turbulence.k.internal, mix.turbulence.k.internal);
        const Diff mE = compare(devSecond, mixSecond);
        const Diff mN = compare(dev.turbulence.nut.internal, mix.turbulence.nut.internal);
        std::printf("  DEVICE closure vs HOST closure, same device loop: U %.4e, k %.4e, %s %.4e, "
                    "nut %.4e\n", (double)mU.rel(), (double)mK.rel(), secondName, (double)mE.rel(),
                    (double)mN.rel());
        // MEASURED, worst of the three profiles: U 3.3e-14, k 1.1e-15, epsilon 1.6e-15, nut 1.9e-15
        // -- round-off between two implementations that share no kernel. Bounds at about 30x.
        // THE MODULE'S OWN CLAIM: the device closure is the host reference's. MEASURED under `sst` at the
        // case's own tolerances: U 8.6e-13, k 1.2e-13, omega 6.6e-14, nut 9.5e-13, and with every solve
        // tightened 1.4e-13 / 5.3e-15 / 7.8e-16 / 3.7e-14 -- round-off between two implementations that
        // share no kernel. kEpsilon's are 3.3e-14 / 1.1e-15 / 1.6e-15 / 1.9e-15. Bounds at about 30x.
        check("the device closure agrees with the host closure in U", mU.rel() < (sst ? scalar(3e-11) : scalar(1e-12)));
        check("...in k", mK.rel() < (sst ? scalar(5e-12) : scalar(5e-14)));
        check("...in the second closure field", mE.rel() < (sst ? scalar(5e-12) : scalar(5e-14)));
        check("...and in nut", mN.rel() < (sst ? scalar(5e-11) : scalar(5e-14)));
        // the two closures took the same sweeps, solve for solve
        std::size_t sameE = 0;
        std::size_t sameK = 0;
        const std::vector<LinearSolveRecord>& devSecondSolves = sst ? rd.omegaSolves : rd.epsilonSolves;
        const std::vector<LinearSolveRecord>& mixSecondSolves = sst ? rm.omegaSolves : rm.epsilonSolves;
        for (std::size_t q = 0; q < devSecondSolves.size() && q < mixSecondSolves.size(); ++q)
        {
            if (devSecondSolves[q].nIterations == mixSecondSolves[q].nIterations)
            {
                ++sameE;
            }
        }
        for (std::size_t q = 0; q < rd.kSolves.size() && q < rm.kSolves.size(); ++q)
        {
            if (rd.kSolves[q].nIterations == rm.kSolves[q].nIterations)
            {
                ++sameK;
            }
        }
        check("...and took the host closure's sweep counts, solve for solve",
              sameE == static_cast<std::size_t>(nSteps) && sameK == static_cast<std::size_t>(nSteps));
    }

    std::printf("test_inter_ras_dambreak_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
