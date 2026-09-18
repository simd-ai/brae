// brae's interFoam with a free-level inlet against REAL OpenFOAM's, on RAS/weirOverflow as shipped.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_weiroverflow_vs_openfoam.sh stages it and carries the measurements. Two conditions:
//   U      variableHeightFlowRateInletVelocity   n*avgU*alpha_p, avgU = -flowRate/gSum(magSf*alpha_p):
//                                                the prescribed volume rate through the WET part of the inlet
//   alpha  variableHeightFlowRate                mixed: on inflow the clipped face-cell value, fixed;
//                                                otherwise zeroGradient
//
// THE CONTROL is OpenFOAM's own answer with the inlet U a fixedValue at the file's (0 0 0) -- what a
// loop that never rebuilt the condition runs. The alpha condition has none on this case; the script says
// why, and tests/test_variable_height_flow_rate.cu holds its per-face logic.
//
// THE DEVICE LOOP MUST REFUSE the case, naming the condition.
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

// MEASURED, ten steps of 1e-3, as shipped: alpha 4.0e-13, p_rgh 8.0e-13, U 1.8e-11, k 5.1e-14, epsilon
// 6.2e-14, nut 8.8e-14, the inlet's velocity 2.7e-13 of its largest value; all 30 p_rgh, 10 epsilon and
// 10 k iteration counts OpenFOAM's. Each bound is about 30x its measurement.
// THE p_rgh RESIDUALS agree to 3e-09 or better except one, 7.2e-08: step one's third corrector, whose
// initial residual is 2.8e-05 after a 58-iteration PCG solve stopped at relTol 0.05. p_rgh itself agrees
// to 8e-13 there, and a residual that small is that difference divided by it. Hence 2e-06, not 1e-10.
struct Bounds
{
    scalar alpha;
    scalar prgh;
    scalar U;
    scalar k;
    scalar epsilon;
    scalar nut;
};
const Bounds B{1e-11, 3e-11, 5e-10, 2e-12, 2e-12, 3e-12};

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/weirOverflow (variableHeightFlowRate) ==\n");
    if (argc < 7)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <frozenOfTimeDir>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string frozenDir = argv[6];

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
    check("...under kEpsilon, in the uniform lineage",
          fin.turbulence.model == InterRasModel::KEpsilon && !fin.turbulence.variableDensity);
    // THE PATH: both conditions were built as themselves, on the same patch
    std::size_t nVhU = 0;
    std::size_t nVhAlpha = 0;
    scalar inletSpeed = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (fin.U.boundary[pi]->isVariableHeightFlowRateInlet())
        {
            ++nVhU;
            for (const vector& v : fin.U.boundary[pi]->value())
            {
                inletSpeed = std::fmax(inletSpeed, mag(v));
            }
        }
        if (fin.alpha1.boundary[pi]->isVariableHeightFlowRate())
        {
            ++nVhAlpha;
        }
    }
    std::printf("  inlet: %zu variableHeightFlowRateInletVelocity, %zu variableHeightFlowRate, |U_b| up to %.4e\n",
                nVhU, nVhAlpha, (double)inletSpeed);
    check("brae built ONE of each condition", nVhU == 1 && nVhAlpha == 1);
    check("...and the inlet is no longer at the file's (0 0 0)", inletSpeed > scalar(0));

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
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(2e-6), scalar(2e-6));
    // no MULESCorr in this case, so alpha is never solved implicitly -- in either code
    check("neither OpenFOAM nor brae solved alpha implicitly", ofA.empty() && r.alphaSolves.empty());
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
    // WHERE the worst cell is, for each field: a gap that starts at one patch names its cause
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
    worstAt("alpha", [&](label c) { return std::fabs(fin.alpha1.internal[c] - ofAlpha[c]); });
    worstAt("p_rgh", [&](label c) { return std::fabs(fin.p_rgh.internal[c] - ofPrgh[c]); });
    worstAt("U", [&](label c)
    {
        const vector e{fin.U.internal[c].x - ofU[c].x, fin.U.internal[c].y - ofU[c].y, fin.U.internal[c].z - ofU[c].z};
        return mag(e);
    });
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

    // THE INLET ITSELF, which OpenFOAM writes (a fixedValue's `value`): brae's patch against it
    const FieldData<vector> ofUFd = readField<vector>(ofDir + "/U");
    scalar dInlet = 0;
    scalar inletScale = 0;
    std::size_t nInlet = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!fin.U.boundary[pi]->isVariableHeightFlowRateInlet()) continue;
        const PatchFieldData<vector>* b = findPatchEntry(ofUFd.boundary, patches[pi]);
        if (!b || b->valueUniform) continue;
        const Diff d = compare(fin.U.boundary[pi]->value(), b->values);
        dInlet = std::fmax(dInlet, d.linf);
        inletScale = std::fmax(inletScale, d.refMax);
        nInlet += b->values.size();
    }
    std::printf("  inlet velocity: Linf %.4e on %zu faces (|U_b| up to %.4e)\n", (double)dInlet, nInlet,
                (double)inletScale);
    check("the inlet carries OpenFOAM's written velocity, face by face",
          nInlet > 0 && inletScale > scalar(0) && dInlet <= B.U*inletScale);

    // THE CONTROL, on the oracle
    const Diff dFrozenU = compare(readVectorCells(frozenDir + "/U"), ofU);
    std::printf("  CONTROL: OpenFOAM with the inlet FROZEN at the file's value against OpenFOAM, U relative %.4e\n",
                (double)dFrozenU.rel());
    check("the velocity condition moves OpenFOAM's own U far more than brae is from it",
          dFrozenU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dFrozenU.rel() > scalar(1e-2));

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
            named = std::string(e.what()).find("variableHeightFlowRate") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses the case and names the condition", named);
    }

    std::printf("test_inter_weiroverflow_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
