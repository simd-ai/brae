// brae's interFoam with LES kEqn against REAL OpenFOAM's, on LES/nozzleFlow2D.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, its solver log, and the
// filter width it writes with writeObjects(delta,geometricDelta) -- tests/interfoam_les_vs_openfoam.sh
// stages all three and carries the measurements. THE CONTROL is OpenFOAM's own answer with the case run
// laminar at the same instant.
//
// THE DEVICE LOOP MUST REFUSE the case, naming the model.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include "inter_solve_log.cuh"
#include "les_delta_cpp.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

// MEASURED, 100 steps of 1e-9 (the bounds are about 30x):
//   alpha 6.6e-12, p_rgh 8.2e-11, U 4.5e-12, k 2.1e-12, nut 1.1e-12; delta 1.1e-15;
//   all 400 p_rgh and 100 k iteration counts OpenFOAM's, p_rgh initial residuals within 9.6e-11
const scalar B_ALPHA = 2e-10;
const scalar B_PRGH = 3e-9;
const scalar B_U = 1.5e-10;
const scalar B_K = 6e-11;
const scalar B_NUT = 3e-11;
const scalar B_DELTA = 1e-14;

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: LES/nozzleFlow2D (LES kEqn, smooth delta, wedge) ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <laminarOfTimeDir> <ofDeltaDir> [<ofDelta3dCase>]\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string laminarDir = argv[6];
    const std::string deltaDir = argv[7];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    check("brae ran the case LES kEqn, in the uniform lineage",
          fin.turbulence.on && fin.turbulence.model == InterRasModel::KEqnLES && !fin.turbulence.variableDensity);
    // THE PATH: the mesh is an axisymmetric wedge -- 2-D to cubeRootVol -- and the smoothing moved the delta
    {
        std::size_t nWedge = 0;
        for (const FvPatch& p : patches)
        {
            nWedge += (p.type == "wedge") ? 1 : 0;
        }
        const cpu::LESdelta::GeometricDirections gd = cpu::LESdelta::geometricDirections(m, patches);
        std::printf("  wedge patches %zu, geometric directions %d, thickness %.6e\n", nWedge, gd.nD, (double)gd.thickness);
        check("the mesh is a wedge, and cubeRootVol takes its 2-D branch", nWedge == 2 && gd.nD == 2 && gd.thickness > 0);
        check("the delta is `smooth` around cubeRootVol", fin.turbulence.deltaSpec.smooth);
        const std::vector<scalar> geo = cpu::LESdelta::cubeRootVol(m, g, patches, fin.turbulence.deltaSpec.deltaCoeff);
        std::size_t raised = 0;
        for (label c = 0; c < nC; ++c)
        {
            raised += (fin.turbulence.delta[c] > geo[c]) ? 1 : 0;
        }
        std::printf("  the smoothing wave raised the delta in %zu of %d cells\n", raised, (int)nC);
        check("...and the smoothing wave raised it somewhere, so the wave is exercised", raised > 0);
        check("the k convection scheme is limitedLinear 1, as the case names",
              fin.turbulence.lesCoeffs.limitedLinear && fin.turbulence.lesCoeffs.limitedLinearCoeff == scalar(1));
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

    // THE FILTER WIDTH against OpenFOAM's own, before AND after smoothing
    {
        const std::vector<scalar> geo = cpu::LESdelta::cubeRootVol(m, g, patches, fin.turbulence.deltaSpec.deltaCoeff);
        const Diff dg = compare(geo, readCells(deltaDir + "/geometricDelta"));
        const Diff dd = compare(fin.turbulence.delta, readCells(deltaDir + "/delta"));
        std::printf("  geometricDelta relative %.4e, delta relative %.4e\n", (double)dg.rel(), (double)dd.rel());
        check("cubeRootVol's delta is OpenFOAM's", dg.rel() < B_DELTA);
        check("...and the smoothed delta is OpenFOAM's", dd.rel() < B_DELTA);
    }
    // ...AND ON THE SAME MESH MADE 3-D, where cubeRootVol takes cbrt(V) and the wave runs on that
    if (argc > 8)
    {
        const std::string c3 = argv[8];
        PrimitiveMesh m3;
        m3.read(c3 + "/constant/polyMesh");
        FvGeometry g3;
        g3.build(m3);
        const std::vector<FvPatch> p3 = buildPatches(m3, g3);
        const cpu::LESdelta::GeometricDirections gd3 = cpu::LESdelta::geometricDirections(m3, p3);
        const std::vector<scalar> geo3 = cpu::LESdelta::cubeRootVol(m3, g3, p3, fin.turbulence.deltaSpec.deltaCoeff);
        const std::vector<scalar> del3 = cpu::LESdelta::compute(fin.turbulence.deltaSpec, m3, g3, p3);
        const FieldData<scalar> og = readField<scalar>(c3 + "/0/geometricDelta");
        const FieldData<scalar> od = readField<scalar>(c3 + "/0/delta");
        const Diff dg3 = compare(geo3, og.internalField);
        const Diff dd3 = compare(del3, od.internalField);
        std::printf("  3-D: geometric directions %d, geometricDelta relative %.4e, delta relative %.4e\n", gd3.nD,
                    (double)dg3.rel(), (double)dd3.rel());
        check("the 3-D mesh is 3-D to cubeRootVol", gd3.nD == 3);
        check("...its cbrt(V) delta is OpenFOAM's, and so is the smoothed one",
              dg3.rel() < B_DELTA && dd3.rel() < B_DELTA && og.internalField.size() == geo3.size());
    }

    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    check("OpenFOAM's log gave one k solve per step", ofK.size() == static_cast<std::size_t>(nSteps));
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(1e-9), scalar(1e-8));
    failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                               scalar(1e-10), scalar(1e-9), scalar(1e-5));

    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
    failures += brae::gatecheck::nonFinite("brae nut", fin.turbulence.nut.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/" + fin.alphaName);
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const std::vector<scalar> ofKf = readCells(ofDir + "/k");
    const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    const Diff dK = compare(fin.turbulence.k.internal, ofKf);
    const Diff dN = compare(fin.turbulence.nut.internal, ofNut);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
    std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dN.rel(), (double)dN.refMax);
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B_ALPHA);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B_PRGH);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B_U);
    check("k agrees with OpenFOAM's relatively", dK.rel() < B_K);
    check("nut agrees with OpenFOAM's relatively", dN.rel() < B_NUT);

    // THE FIXTURE CAN DISCRIMINATE: the eddy viscosity is live, in OpenFOAM's own numbers
    {
        scalar ratio = 0;
        for (std::size_t c = 0; c < ofNut.size(); ++c)
        {
            ratio = std::fmax(ratio, ofNut[c]/std::fmax(fin.nu[c], scalar(1e-300)));
        }
        std::printf("  OpenFOAM's nut reaches %.3f of the mixture's nu\n", (double)ratio);
        check("OpenFOAM's nut is a visible fraction of nu somewhere", ratio > scalar(0.01));
    }

    // THE CONTROL, on the oracle
    const Diff dLamU = compare(readVectorCells(laminarDir + "/U"), ofU);
    std::printf("  CONTROL: OpenFOAM run laminar against OpenFOAM run LES, U relative %.4e\n", (double)dLamU.rel());
    check("the LES model moves OpenFOAM's own U far more than brae is from it",
          dLamU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)));

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
            named = std::string(e.what()).find("LES kEqn") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses the case and names the model", named);
    }

    std::printf("test_inter_les_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
