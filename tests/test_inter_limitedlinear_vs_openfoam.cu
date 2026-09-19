// brae's interFoam with `Gauss limitedLinear 0.2` on div(rhoPhi,U) against REAL OpenFOAM's, on
// laminar/vofToLagrangian/eulerianInjection.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps and its solver log --
// tests/interfoam_limitedlinear_vs_openfoam.sh stages it and carries the measurements. TWO CONTROLS, both
// OpenFOAM's own answer at the same instant: the case under `Gauss upwind` (the scheme matters) and under
// `Gauss limitedLinear 1` (its coefficient matters).
//
// THE DEVICE LOOP MUST REFUSE the scheme, naming it.
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
#include <exception>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

// MEASURED, 60 steps of 2e-5 on the 45^3 block (bounds about 30x): alpha 9.2e-14, p_rgh 8.8e-12,
// U 2.7e-12. Controls: upwind 8.0e-01, limitedLinear 1 at 7.8e-01.
const scalar B_ALPHA = 3e-12;
const scalar B_PRGH = 3e-10;
const scalar B_U = 1e-10;

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: eulerianInjection (limitedLinear 0.2 on U) ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <upwindOfTimeDir> <ll1OfTimeDir>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string upwindDir = argv[6];
    const std::string ll1Dir = argv[7];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    check("brae ran div(rhoPhi,U) as limitedLinear 0.2, the case's",
          fin.divRhoPhiU == DivScheme::limitedLinear && fin.divRhoPhiUCoeff == scalar(0.2));
    check("...laminar, with no momentum predictor", !fin.turbulence.on && !fin.momentumPredictorOn);

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
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(1e-9), scalar(1e-8));

    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/" + fin.alphaName);
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B_ALPHA);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B_PRGH);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B_U);

    // THE CONTROLS, on the oracle
    const Diff dUp = compare(readVectorCells(upwindDir + "/U"), ofU);
    const Diff dL1 = compare(readVectorCells(ll1Dir + "/U"), ofU);
    std::printf("  CONTROL: OpenFOAM under upwind against OpenFOAM under limitedLinear 0.2, U relative %.4e\n",
                (double)dUp.rel());
    std::printf("  CONTROL: OpenFOAM under limitedLinear 1 against limitedLinear 0.2, U relative %.4e\n",
                (double)dL1.rel());
    check("the scheme moves OpenFOAM's own U far more than brae is from it",
          dUp.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)));
    check("...and so does its coefficient", dL1.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)));

    // THE DEVICE LOOP REFUSES, by name
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device refusal is not exercised)\n");
    }
    else
    {
        bool named = false;
        try
        {
            InterFields dev;
            runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        }
        catch (const std::exception& e)
        {
            // the shipped case's GAMG smoother is refused first; the scheme is held by the refusals script
            named = std::string(e.what()).find("-device") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses the case by name", named);
    }

    std::printf("test_inter_limitedlinear_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
