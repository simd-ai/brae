// brae's interFoam under CRANKNICOLSON against REAL OpenFOAM's, on RAS/damBreak with
// `ddtSchemes { default CrankNicolson <oc>; }`, on BOTH arms.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_cn_vs_openfoam.sh stages it and carries the description of the scheme, the profiles
// and the measurements. THE CONTROL is OpenFOAM's own answer under Euler at the same instant: the scheme
// moves OpenFOAM's U by 3.2e-02 here, and both arms must sit inside that by orders.
//
// THE PATH, not only the answer: the momentum equation's ddt0 field must exist and have been advanced
// once per step, which the run report's step count against the field's time index says.
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
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

// MEASURED, per profile and arm, 20 steps of 1e-3; each bound is about 30x its measurement.
//   cn       host: alpha 9.3e-15, p_rgh 8.0e-15, U 1.5e-14, k 8.9e-15, epsilon 6.0e-15, nut 1.2e-14;
//                  60 of 60 p_rgh counts, initial residuals within 2.1e-12
//            device: alpha 8.2e-15, p_rgh 7.5e-15, U 1.9e-12, k 7.6e-15, epsilon 7.3e-15, nut 1.3e-14;
//                  60 of 60 counts, residuals within 5.0e-11
//   cnOuter  host: 1.0e-14, 1.2e-14, 1.0e-14, 6.5e-15, 6.3e-15, 9.6e-15; 120 of 120, 5.3e-11
//            device: 5.8e-15, 7.5e-15, 1.7e-12, 1.3e-14, 1.4e-14, 2.1e-14; 120 of 120, 4.5e-11
//   cnFull   host: 2.4e-14, 9.8e-13, 1.3e-10, 5.3e-12, 4.3e-13, 6.2e-12; 60 of 60, 1.0e-09
//            device: 1.2e-14, 5.8e-13, 7.8e-11, 3.1e-12, 2.6e-13, 3.7e-12; 60 of 60, 5.1e-10
// cnFull is a pure CrankNicolson (oc 1), whose two-step recurrence carries round-off undamped: three
// orders above the off-centred profiles on both arms, and OpenFOAM's own Euler-vs-CN distance is
// 3.2e-02 beside it. The device's U on `cn` is a hundred times the host's, as on every VoF case: the
// MULES gather order times the density ratio.
struct Bounds
{
    scalar alpha;
    scalar prgh;
    scalar U;
    scalar k;
    scalar epsilon;
    scalar nut;
    scalar pResidual;
};
const Bounds H_CN{3e-13, 3e-13, 5e-13, 3e-13, 2e-13, 4e-13, 1e-10};
const Bounds H_OUTER{3e-13, 4e-13, 5e-13, 2e-13, 2e-13, 3e-13, 2e-9};
const Bounds H_FULL{1e-12, 3e-11, 4e-9, 2e-10, 1.5e-11, 2e-10, 3e-8};
const Bounds D_CN{3e-13, 3e-13, 6e-11, 3e-13, 3e-13, 4e-13, 2e-9};
const Bounds D_OUTER{3e-13, 3e-13, 6e-11, 4e-13, 4e-13, 7e-13, 2e-9};
const Bounds D_FULL{1e-12, 2e-11, 3e-9, 1e-10, 1e-11, 1.2e-10, 2e-8};

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

// one arm's fields against the oracle, held to its bounds
void holdArm(
    const char* who,
    const InterFields& f,
    const Bounds& B,
    const std::vector<scalar>& ofAlpha,
    const std::vector<scalar>& ofPrgh,
    const std::vector<vector>& ofU,
    const std::vector<scalar>& ofK,
    const std::vector<scalar>& ofE,
    const std::vector<scalar>& ofNut,
    scalar controlU,
    Diff& dUOut)
{
    failures += brae::gatecheck::nonFinite("alpha", f.alpha1.internal);
    failures += brae::gatecheck::nonFinite("p_rgh", f.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("U", f.U.internal);
    failures += brae::gatecheck::nonFinite("k", f.turbulence.k.internal);
    failures += brae::gatecheck::nonFinite("epsilon", f.turbulence.epsilon.internal);
    failures += brae::gatecheck::nonFinite("nut", f.turbulence.nut.internal);
    const Diff dA = compare(f.alpha1.internal, ofAlpha);
    const Diff dP = compare(f.p_rgh.internal, ofPrgh);
    const Diff dU = compare(f.U.internal, ofU);
    const Diff dK = compare(f.turbulence.k.internal, ofK);
    const Diff dE = compare(f.turbulence.epsilon.internal, ofE);
    const Diff dN = compare(f.turbulence.nut.internal, ofNut);
    std::printf("  %-7s alpha %.4e, p_rgh %.4e, U %.4e, k %.4e, epsilon %.4e, nut %.4e\n", who,
                (double)dA.linf, (double)dP.rel(), (double)dU.rel(), (double)dK.rel(), (double)dE.rel(),
                (double)dN.rel());
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B.alpha);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B.prgh);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B.U);
    check("k agrees with OpenFOAM's relatively", dK.rel() < B.k);
    check("epsilon agrees with OpenFOAM's relatively", dE.rel() < B.epsilon);
    check("nut agrees with OpenFOAM's relatively", dN.rel() < B.nut);
    check("CrankNicolson moves OpenFOAM's own U far more than this arm is from it",
          controlU > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && controlU > scalar(1e-3));
    dUOut = dU;
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/damBreak under CrankNicolson ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <eulerOfTimeDir> <cn|cnOuter|cnFull>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string eulerDir = argv[6];
    const std::string profile = argv[7];
    const bool outer = (profile == "cnOuter");
    const bool full = (profile == "cnFull");
    const Bounds& HB = outer ? H_OUTER : full ? H_FULL : H_CN;
    const Bounds& DB = outer ? D_OUTER : full ? D_FULL : D_CN;
    std::printf("  profile: %s\n", outer ? "cnOuter -- CrankNicolson 0.5 with nOuterCorrectors 2"
                                 : full ? "cnFull -- CrankNicolson 1, the un-off-centred scheme"
                                        : "cn -- CrankNicolson 0.5, the tutorial's PIMPLE");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/false, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    // THE PATH: the scheme was read as itself, with the case's coefficient, for both operand sets
    check("brae read ddtSchemes as CrankNicolson for U and for alpha",
          fin.ddtU == DdtScheme::CrankNicolson && fin.ddtAlpha == AlphaDdt::CrankNicolson);
    check("...with the case's off-centring coefficient",
          std::fabs(fin.ddtOcCoeff - (full ? scalar(1) : scalar(0.5))) == scalar(0)
       && std::fabs(fin.ddtAlphaOcCoeff - (full ? scalar(1) : scalar(0.5))) == scalar(0));
    check("brae ran the case turbulent, under kEpsilon, in the `density variable` lineage",
          fin.turbulence.on && fin.turbulence.model == InterRasModel::KEpsilon && fin.turbulence.variableDensity);
    check("...with nOuterCorrectors as the profile set it", fin.pimple.nOuterCorrectors == (outer ? 2 : 1));
    // the closure's own ddt0 fields exist and were advanced on the last step: created at step one
    // (start index 1) and evaluated once per step since
    check("the closure's k ddt0 field exists, born on step one and advanced on the last",
          fin.turbulence.cn.ddt0K.exists && fin.turbulence.cn.ddt0K.startTimeIndex == 1
       && fin.turbulence.cn.ddt0K.timeIndex == nSteps);

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
    check("...and the profile's number of p_rgh solves",
          ofP.size() == static_cast<std::size_t>(nSteps)*static_cast<std::size_t>(outer ? 6 : 3));
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", HB.pResidual, HB.pResidual);
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               scalar(1e-10), scalar(1e-9));
    failures += brae::gatecheck::compareSolves("host", r.epsilonSolves, ofE, nSteps, "epsilon",
                                               scalar(1e-10), scalar(1e-8), scalar(1e-5));
    failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                               scalar(1e-10), scalar(1e-8), scalar(1e-5));

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const std::vector<scalar> ofKf = readCells(ofDir + "/k");
    const std::vector<scalar> ofEf = readCells(ofDir + "/epsilon");
    const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
    check("OpenFOAM's fields have one value per cell",
          ofAlpha.size() == static_cast<std::size_t>(nC) && ofKf.size() == ofAlpha.size()
       && ofEf.size() == ofAlpha.size() && ofNut.size() == ofAlpha.size());

    // THE CONTROL, on the oracle: OpenFOAM under Euler against OpenFOAM under CrankNicolson
    const Diff dCtlU = compare(readVectorCells(eulerDir + "/U"), ofU);
    const Diff dCtlA = compare(readCells(eulerDir + "/alpha.water"), ofAlpha);
    std::printf("  CONTROL: OpenFOAM under Euler against OpenFOAM under CrankNicolson, U relative %.4e, alpha %.4e\n",
                (double)dCtlU.rel(), (double)dCtlA.linf);

    Diff dUHost;
    holdArm("host:", fin, HB, ofAlpha, ofPrgh, ofU, ofKf, ofEf, ofNut, dCtlU.rel(), dUHost);

    // THE DEVICE LOOP, on the same case from the same start, held to OpenFOAM by its OWN bounds
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
    else
    {
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        check("the device driver ran the same number of steps", rd.steps == nSteps);
        check("...with the closure ON THE DEVICE", rd.turbulenceOnDevice);
        failures += brae::gatecheck::compareSolves("device", rd.pSolves, ofP, nSteps, "p_rgh", DB.pResidual, DB.pResidual);
        // THE SECOND PASS'S alpha PRE-SOLVE starts from the first pass's alpha and its initial residual is
        // 1e-5 of the normFactor -- a difference of nearly equal quantities, so the device's 1e-12 in phi
        // (its VoF floor, the MULES gather order times the density ratio) reads as 2.4e-07 RELATIVE there
        // while the host, at 1e-14 in phi, reproduces OpenFOAM's digits. MEASURED on cnOuter: 2.6e-07,
        // on all twenty second passes alike; every count OpenFOAM's; every first pass at the floor.
        failures += brae::gatecheck::compareSolves("device", rd.alphaSolves, ofA, nSteps, dev.alphaName.c_str(),
                                                   scalar(1e-10), outer ? scalar(1e-5) : scalar(1e-8));
        failures += brae::gatecheck::compareSolves("device", rd.epsilonSolves, ofE, nSteps, "epsilon",
                                                   scalar(1e-9), scalar(1e-8));
        failures += brae::gatecheck::compareSolves("device", rd.kSolves, ofK, nSteps, "k",
                                                   scalar(1e-9), scalar(1e-8));
        Diff dUDev;
        holdArm("DEVICE:", dev, DB, ofAlpha, ofPrgh, ofU, ofKf, ofEf, ofNut, dCtlU.rel(), dUDev);
    }

    std::printf("test_inter_cn_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
