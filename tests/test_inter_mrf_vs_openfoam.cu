// brae's interFoam with MRF against REAL OpenFOAM's, on laminar/mixerVessel2D as shipped.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_mrf_vs_openfoam.sh stages it and carries the measurements. interFoam reaches MRF in
// four places that do arithmetic (UEqn.H:1, :6; pEqn.H:17, :19): the rotor patch takes the frame
// velocity, UEqn takes rho*(Omega x U) on the zone's cells, the ddtCorr flux is zeroed on the zone's
// faces, and phiHbyA is made relative to the frame.
//
// THE CONTROL is OpenFOAM's own answer with the zone `active no` at the same instant: the fluid then
// never moves, so a brae that ignored MRFProperties -- which it once did, silently -- sits on it.
//
// THE DEVICE LOOP MUST REFUSE the case, naming MRF.
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

// MEASURED, twenty steps of 1e-3: alpha 8.2e-15, p_rgh 4.9e-14, U 1.9e-15; all 60 p_rgh iteration
// counts OpenFOAM's, initial residuals within 1.2e-11. Each bound is about 30x its measurement. The
// script's header has what each of the four MRF terms costs when it is broken.
#define BOUND_ALPHA 3e-13
#define BOUND_PRGH 2e-12
#define BOUND_U 1e-13

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: laminar/mixerVessel2D (MRF) ==\n");
    if (argc < 7)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <inactiveOfTimeDir>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string inactiveDir = argv[6];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    // THE PATH, not only the answer
    check("brae read ONE active MRF zone", fin.mrfZones.size() == 1);
    if (fin.mrfZones.size() == 1)
    {
        const brae::cpu::MRF::Zone& z = fin.mrfZones[0];
        std::size_t nIncluded = 0;
        std::size_t nExcluded = 0;
        for (const std::vector<label>& v : z.includedFaces)
        {
            nIncluded += v.size();
        }
        for (const std::vector<label>& v : z.excludedFaces)
        {
            nExcluded += v.size();
        }
        std::printf("  zone: %zu cells of %d, %zu internal faces, %zu included and %zu excluded boundary "
                    "faces, Omega (%g %g %g)\n", z.cells.size(), (int)nC, z.internalFaces.size(), nIncluded,
                    nExcluded, (double)z.Omega.x, (double)z.Omega.y, (double)z.Omega.z);
        check("...a proper part of the mesh, so the zone has an INTERFACE for makeRelative to act on",
              !z.cells.empty() && z.cells.size() < static_cast<std::size_t>(nC));
        check("...with boundary faces that move with the frame (the rotor)", nIncluded > 0);
        check("...turning at the case's 6.2831853 rad/s about z",
              std::fabs(z.Omega.z - scalar(6.2831853)) < scalar(1e-12) && z.Omega.x == scalar(0)
           && z.Omega.y == scalar(0));
    }
    check("p_rgh needs a reference in this closed vessel, and brae read it so", fin.pRef.needReference);
    check("the case is laminar", !fin.turbulence.on);

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
    check("OpenFOAM's log gave p_rgh solves to compare", !ofP.empty());
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps);
    // no MULESCorr in this case, so alpha is never solved implicitly -- in either code
    check("neither OpenFOAM nor brae solved alpha implicitly", ofA.empty() && r.alphaSolves.empty());

    // BEFORE any fmax: std::fmax drops a NaN -- see tests/device_gate_finite.cuh
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    check("OpenFOAM's fields have one value per cell",
          ofAlpha.size() == static_cast<std::size_t>(nC) && ofU.size() == ofAlpha.size());

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);

    // THE ROTOR'S FRAME VELOCITY has no oracle of its own: OpenFOAM writes a noSlip patch as its type
    // alone, without values. It is held through U -- the script's header has the number for
    // MRF.correctBoundaryVelocity skipped.
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < scalar(BOUND_ALPHA));
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < scalar(BOUND_PRGH));
    check("U agrees with OpenFOAM's relatively", dU.rel() < scalar(BOUND_U));

    // THE CONTROL, on the oracle
    const std::vector<vector> offU = readVectorCells(inactiveDir + "/U");
    const Diff dOffU = compare(offU, ofU);
    scalar offMax = 0;
    for (const vector& v : offU)
    {
        offMax = std::fmax(offMax, mag(v));
    }
    std::printf("  CONTROL: OpenFOAM with the zone inactive against OpenFOAM with it, U relative %.4e "
                "(inactive |U| up to %.4e)\n", (double)dOffU.rel(), (double)offMax);
    check("the zone moves OpenFOAM's own U far more than brae is from it",
          dOffU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dOffU.rel() > scalar(0.1));

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
            named = std::string(e.what()).find("MRF") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses the case and names MRF", named);
    }

    std::printf("test_inter_mrf_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
