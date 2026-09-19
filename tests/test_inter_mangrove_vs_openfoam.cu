// brae's interFoam with the MANGROVE fvOptions against REAL OpenFOAM's, on
// laminar/waves/mangroveInteraction.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_mangrove_vs_openfoam.sh stages it and carries the measurements. The case runs a
// Boussinesq wave paddle against a shallowWaterAbsorption outlet, kEpsilon solved with PBiCG and DILU,
// and two options over the cellZone the seaweed surface marks: multiphaseMangrovesSource (drag and added
// mass on U) and multiphaseMangrovesTurbulenceModel (the vegetation's k and epsilon).
// TWO CONTROLS, both OpenFOAM's own answer at the same instant: both options off, and the turbulence
// option off alone.
//
// THE DEVICE LOOP MUST REFUSE the case, naming the option.
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

// MEASURED, 450 fixed steps of 0.01 on the halved block (bounds about 30x): alpha 4.6e-14, p_rgh
// 3.8e-14, U 3.4e-12, k 1.7e-13, epsilon 1.0e-13, nut 3.0e-13; every p_rgh, k and epsilon count
// OpenFOAM's, and k's and epsilon's final residuals with them.
const scalar B_ALPHA = 1.5e-12;
const scalar B_PRGH = 1.2e-12;
const scalar B_U = 1e-10;
const scalar B_K = 5e-12;
const scalar B_EPSILON = 3e-12;
const scalar B_NUT = 1e-11;

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: waves/mangroveInteraction (mangrove fvOptions) ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <offOfTimeDir> "
                    "<turbulenceOffOfTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string offDir = argv[6];
    const std::string turbOffDir = argv[7];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/false, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    check("brae ran the case turbulent, under kEpsilon",
          fin.turbulence.on && fin.turbulence.model == InterRasModel::KEpsilon);
    check("...its k and epsilon with PBiCG and DILU, as the case names",
          fin.turbulence.kSolveFinal.pbicgDILU() && fin.turbulence.epsSolveFinal.pbicgDILU());

    // THE PATH: both mangrove options read, active, over a non-empty zone
    std::size_t nSource = 0;
    std::size_t nTurb = 0;
    std::size_t zoneCells = 0;
    for (const cpu::fvOptions::Option& o : fin.fvOptions.options)
    {
        if (!o.active || !o.unsupported.empty()) continue;
        if (o.mangroves == cpu::fvOptions::Option::Mangroves::source) ++nSource;
        if (o.mangroves == cpu::fvOptions::Option::Mangroves::turbulence) ++nTurb;
        for (const auto& reg : o.mangroveRegions) zoneCells = std::max(zoneCells, reg.cells.size());
    }
    std::printf("  mangrove options: %zu source, %zu turbulence; the zone holds %zu of %d cells\n",
                nSource, nTurb, zoneCells, (int)nC);
    check("both mangrove options are active, over a zone of cells", nSource == 1 && nTurb == 1 && zoneCells > 0);

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
    const std::vector<LinearSolveRecord> ofE = brae::gatecheck::readOfSolves(logPath, "epsilon");
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    check("OpenFOAM's log gave one epsilon and one k solve per step",
          ofE.size() == static_cast<std::size_t>(nSteps) && ofK.size() == ofE.size());
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(2e-6), scalar(2e-6));
    failures += brae::gatecheck::compareSolves("host", r.epsilonSolves, ofE, nSteps, "epsilon",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));
    failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));

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
    auto worstAt = [&](const char* name, auto diffOf)
    {
        scalar worst = 0;
        label at = -1;
        for (label c = 0; c < nC; ++c)
        {
            const scalar d = diffOf(c);
            if (d > worst)
            {
                worst = d;
                at = c;
            }
        }
        if (at >= 0)
        {
            std::printf("  worst %-7s cell %d at (%.4g %.4g %.4g): %.4e\n", name, (int)at, (double)g.C()[at].x,
                        (double)g.C()[at].y, (double)g.C()[at].z, (double)worst);
        }
    };
    worstAt("U", [&](label c)
    {
        const vector e{fin.U.internal[c].x - ofU[c].x, fin.U.internal[c].y - ofU[c].y, fin.U.internal[c].z - ofU[c].z};
        return mag(e);
    });
    worstAt("k", [&](label c) { return std::fabs(fin.turbulence.k.internal[c] - ofKf[c]); });
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
    std::printf("  epsilon: relative %.4e   (epsilon up to %.4e)\n", (double)dE.rel(), (double)dE.refMax);
    std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dN.rel(), (double)dN.refMax);

    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B_ALPHA);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B_PRGH);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B_U);
    check("k agrees with OpenFOAM's relatively", dK.rel() < B_K);
    check("epsilon agrees with OpenFOAM's relatively", dE.rel() < B_EPSILON);
    check("nut agrees with OpenFOAM's relatively", dN.rel() < B_NUT);

    // THE CONTROLS, on the oracle
    const Diff dOffU = compare(readVectorCells(offDir + "/U"), ofU);
    const Diff dTurbK = compare(readCells(turbOffDir + "/k"), ofKf);
    std::printf("  CONTROL: OpenFOAM with both mangrove options off, U relative %.4e\n", (double)dOffU.rel());
    std::printf("  CONTROL: OpenFOAM with the turbulence option off, k relative %.4e\n", (double)dTurbK.rel());
    check("the mangroves move OpenFOAM's own U far more than brae is from it",
          dOffU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dOffU.rel() > scalar(1e-6));
    check("...and their turbulence source moves its k far more than brae is from it",
          dTurbK.rel() > scalar(1000)*std::fmax(dK.rel(), scalar(1e-14)) && dTurbK.rel() > scalar(1e-6));

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
            named = std::string(e.what()).find("Mangroves") != std::string::npos;
            std::printf("  device: %s\n", std::string(e.what()).substr(0, 200).c_str());
        }
        check("the device loop refuses the case and names the option", named);
    }

    std::printf("test_inter_mangrove_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
