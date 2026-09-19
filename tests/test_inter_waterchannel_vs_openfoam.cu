// brae's interFoam under kOmegaSST against REAL OpenFOAM's, on RAS/waterChannel as shipped.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its own solver
// log for omega and k -- tests/interfoam_waterchannel_vs_openfoam.sh stages it and says what the case
// exercises. kOmegaSST is the ordinary incompressible model here (no `density` line): the mixture's nu
// as a field, the volumetric phi, the cell wall distance for F1 and F2, omegaWallFunction and
// nutkWallFunction on the one `wall` patch, validate() at construction.
//
// THE CONTROL is OpenFOAM's own answer with `simulationType laminar` at the same instant: if the
// turbulent oracle sat on top of it, a brae that ignored the model would pass every field bound.
//
// THE DEVICE LOOP RUNS kOmegaSST NOW (gated on RAS/damBreak made kOmegaSST, tests/
// interfoam_ras_dambreak_vs_openfoam.sh `sst`) and must still refuse THIS case, naming the one thing the
// device closure does not carry: the outlet's inletOutlet nut, which correctBoundaryConditions evaluates
// against the flux.
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

// MEASURED, ten steps of 0.1: alpha 7.9e-13, p_rgh 7.6e-13, U 4.3e-12, k 2.4e-12, omega 3.9e-12, nut
// 3.5e-12, the wall's nut 9.2e-13 of its largest value; all 20 p_rgh, 10 alpha, 10 omega and 10 k
// iteration counts OpenFOAM's, initial residuals within 9.3e-13. Each bound is about 30x its
// measurement. BROKEN ONCE EACH, the script's header has the numbers.
#define BOUND_ALPHA 2e-11
#define BOUND_PRGH 2e-11
#define BOUND_U 2e-10
#define BOUND_K 1e-10
#define BOUND_OMEGA 1e-10
#define BOUND_NUT 1e-10

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/waterChannel (kOmegaSST) ==\n");
    if (argc < 7)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <laminarOfTimeDir>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string laminarDir = argv[6];

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
    check("...under kOmegaSST", fin.turbulence.model == InterRasModel::KOmegaSST);
    check("...in the uniform lineage", !fin.turbulence.variableDensity);
    check("...with a wall distance in every cell",
          fin.turbulence.yCell.size() == static_cast<std::size_t>(nC));

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
    const std::vector<LinearSolveRecord> ofO = brae::gatecheck::readOfSolves(logPath, "omega");
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    // the FAIL-PROOF: nothing in compareSolves can pass on an empty parse
    check("OpenFOAM's log gave one omega and one k solve per step",
          ofO.size() == static_cast<std::size_t>(nSteps) && ofK.size() == ofO.size());
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps);
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               scalar(1e-10), scalar(1e-9));
    failures += brae::gatecheck::compareSolves("host", r.omegaSolves, ofO, nSteps, "omega",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));
    failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));

    // BEFORE any fmax: std::fmax drops a NaN -- see tests/device_gate_finite.cuh
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
    failures += brae::gatecheck::nonFinite("brae omega", fin.turbulence.omega.internal);
    failures += brae::gatecheck::nonFinite("brae nut", fin.turbulence.nut.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const std::vector<scalar> ofKf = readCells(ofDir + "/k");
    const std::vector<scalar> ofOf = readCells(ofDir + "/omega");
    const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
    check("OpenFOAM's fields have one value per cell",
          ofAlpha.size() == static_cast<std::size_t>(nC) && ofKf.size() == ofAlpha.size()
       && ofOf.size() == ofAlpha.size() && ofNut.size() == ofAlpha.size());

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    const Diff dK = compare(fin.turbulence.k.internal, ofKf);
    const Diff dO = compare(fin.turbulence.omega.internal, ofOf);
    const Diff dN = compare(fin.turbulence.nut.internal, ofNut);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
    std::printf("  omega:   relative %.4e   (omega up to %.4e)\n", (double)dO.rel(), (double)dO.refMax);
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

    check("alpha agrees with OpenFOAM's absolutely", dA.linf < scalar(BOUND_ALPHA));
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < scalar(BOUND_PRGH));
    check("U agrees with OpenFOAM's relatively", dU.rel() < scalar(BOUND_U));
    check("k agrees with OpenFOAM's relatively", dK.rel() < scalar(BOUND_K));
    check("omega agrees with OpenFOAM's relatively", dO.rel() < scalar(BOUND_OMEGA));
    check("nut agrees with OpenFOAM's relatively", dN.rel() < scalar(BOUND_NUT));
    check("the wall's nut is OpenFOAM's nutkWallFunction value",
          nWallFaces > 0 && dWallNut <= scalar(BOUND_NUT)*std::fmax(wallNutScale, scalar(1e-300)));

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
    const Diff dLamU = compare(readVectorCells(laminarDir + "/U"), ofU);
    std::printf("  CONTROL: OpenFOAM laminar against OpenFOAM kOmegaSST, U relative %.4e\n",
                (double)dLamU.rel());
    check("the model moves OpenFOAM's own U far more than brae is from it",
          dLamU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dLamU.rel() > scalar(1e-3));
    // ...and for a profile that changes nut's patches, OpenFOAM's own answer with the patches otherwise
    // (argv[7]): they must move OpenFOAM's nut far more than brae is from it, and by at least argv[8]
    // (default 1e-4), the floor that keeps two round-off answers from passing as a control
    if (argc > 7)
    {
        const scalar floor = (argc > 8) ? static_cast<scalar>(std::atof(argv[8])) : scalar(1e-4);
        const Diff dShipN = compare(readCells(std::string(argv[7]) + "/nut"), ofNut);
        std::printf("  CONTROL: OpenFOAM with the nut patches otherwise against this profile, nut relative %.4e "
                    "(floor %.1e)\n", (double)dShipN.rel(), (double)floor);
        check("the profile's nut patches move OpenFOAM's own nut far more than brae is from it",
              dShipN.rel() > scalar(1000)*std::fmax(dN.rel(), scalar(1e-14)) && dShipN.rel() > floor);
    }

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
            const std::string w = e.what();
            // what is left of this case that the device does not carry: U's outlet is a plain
            // inletOutlet, whose valueFraction OpenFOAM sets from the flux sign at every momentum
            // assembly. MEASURED with that refusal lifted and nothing else changed: U 8.9e-02,
            // nut 8.7e-01 against OpenFOAM, where the host loop on this same case is 4.3e-12.
            named = w.find("inletOutlet") != std::string::npos && w.find("U patch") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses the case, naming U's inletOutlet outlet", named);
    }

    std::printf("test_inter_waterchannel_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
