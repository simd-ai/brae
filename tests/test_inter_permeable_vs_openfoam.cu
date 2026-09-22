// brae's interFoam with a permeable wall against REAL OpenFOAM's, on laminar/damBreakPermeable.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_permeable_vs_openfoam.sh stages it and carries the measurements. Two conditions on
// `rightWall`, each switching FACE BY FACE on the phase fraction at the patch against alphaMin:
//   U      permeableAlphaPressureInletOutletVelocity   wet, or any face the flux enters: U = 0;
//                                                      dry and leaving: zeroGradient
//   p_rgh  prghPermeableAlphaTotalPressure             wet: the flux-consistent gradient a wall takes;
//                                                      dry: the total pressure, in p_rgh's terms
//
// THE CONTROL is OpenFOAM's own answer with that wall CLOSED -- noSlip and fixedFluxPressure -- at the
// same instant. TWO PROFILES: `shipped`, where the wall is dry for the whole gated run, and `wetWall`,
// the water column staged against it, where it is wet and faces go dry as the column falls.
//
// THE DEVICE LOOP runs the same case from the same start and is held to OpenFOAM by its own bounds,
// to OpenFOAM's p_rgh solves one by one, and to the wall's own written values.
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

// MEASURED, per profile; each bound is about 30x its measurement.
//   shipped   20 steps: alpha 8.5e-15, p_rgh 8.0e-15, U 2.7e-15, k 2.6e-15, epsilon 6.6e-15, nut 3.7e-15;
//             every p_rgh residual OpenFOAM's to the last printed digit; the wall's U_b 5.3e-16, p_rgh_b 4.9e-15
//   wetWall   140 steps: alpha 4.0e-13, p_rgh 5.0e-14, U 1.4e-14, k 3.0e-14, epsilon 3.1e-14, nut 1.2e-14;
//             all 420 p_rgh counts OpenFOAM's, residuals within 1.8e-09; 27 faces wet at the start, 25 at the end
struct Bounds
{
    scalar alpha;
    scalar prgh;
    scalar U;
    scalar k;
    scalar epsilon;
    scalar nut;
};
const Bounds SHIPPED{3e-13, 3e-13, 1e-13, 1e-13, 2e-13, 1e-13};
const Bounds WETWALL{1e-11, 2e-12, 5e-13, 1e-12, 1e-12, 4e-13};

// THE DEVICE ARM'S OWN BOUNDS, each about 30x its measurement. The arm is deterministic: two runs of
// either profile print the same digits.
//   shipped   alpha 6.9e-15, p_rgh 4.7e-15, U 4.3e-13, k 6.8e-15, epsilon 6.4e-15, nut 2.2e-14; all 60
//             p_rgh counts OpenFOAM's, initial residuals within 5.8e-13 in step one and 3.0e-11 over the run
//   wetWall   alpha 1.6e-13, p_rgh 4.3e-14, U 9.9e-14, k 1.1e-13, epsilon 6.2e-14, nut 1.1e-13; all 420
//             counts, initial residuals within 1.1e-09; the same 25 faces wet at the end as the host loop
// shipped's U is a hundred times the host arm's 3.9e-15 and is not a defect of this wall: it is the
// device arm's floor on a VoF case, where MULES gathers in another order and the density ratio is 1000.
struct DeviceBounds
{
    Bounds f;
    scalar pStepOne;
    scalar pRun;
};
const DeviceBounds DEV_SHIPPED{{2e-13, 1.5e-13, 1.5e-11, 2e-13, 2e-13, 7e-13}, 2e-11, 1e-9};
const DeviceBounds DEV_WETWALL{{5e-12, 1.5e-12, 3e-12, 3.5e-12, 2e-12, 3.5e-12}, 2e-11, 4e-8};

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: laminar/damBreakPermeable (permeable wall) ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <closedOfTimeDir> <shipped|wetWall>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string closedDir = argv[6];
    const bool wetWall = (std::string(argv[7]) == "wetWall");
    const Bounds& B = wetWall ? WETWALL : SHIPPED;
    const DeviceBounds& DB = wetWall ? DEV_WETWALL : DEV_SHIPPED;
    std::printf("  profile: %s\n", wetWall ? "wetWall -- the water column staged against the permeable wall" : "shipped -- the wall stays dry for the gated run");

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
    check("...under kEpsilon, in the `density variable` lineage the case ships",
          fin.turbulence.model == InterRasModel::KEpsilon && fin.turbulence.variableDensity);
    // THE PATH: both conditions were built as themselves, on one patch, and which faces of it are wet
    std::size_t nPermU = 0;
    std::size_t nPermP = 0;
    std::size_t wetStart = 0;
    std::size_t wetEnd = 0;
    std::size_t nPermFaces = 0;
    {
        const FieldData<scalar> a0 = readField<scalar>(startDir + "/alpha.water");
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            nPermU += fin.U.boundary[pi]->needsAlphaPatchValues() ? 1 : 0;
            if (!fin.p_rgh.boundary[pi]->isPrghPermeableAlphaTotalPressure()) continue;
            ++nPermP;
            nPermFaces = static_cast<std::size_t>(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const scalar aStart = a0.internalUniform ? a0.internalUniformValue : a0.internalField[c];
                wetStart += (aStart > scalar(0.01)) ? 1 : 0;
                wetEnd += (fin.alpha1.boundary[pi]->value()[static_cast<std::size_t>(i)] > scalar(0.01)) ? 1 : 0;
            }
        }
    }
    std::printf("  permeable wall: %zu faces, %zu wet at the start and %zu at the end (alphaMin 0.01)\n",
                nPermFaces, wetStart, wetEnd);
    check("brae built ONE of each condition", nPermU == 1 && nPermP == 1);
    if (wetWall)
    {
        check("the wall starts partly WET and partly dry", wetStart > 0 && wetStart < nPermFaces);
        check("...and faces SWITCHED during the run, so the per-face switch is exercised", wetEnd != wetStart);
    }
    else
    {
        check("the wall is dry from start to end, the open branch alone", wetStart == 0 && wetEnd == 0);
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

    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofA = brae::gatecheck::readOfSolves(logPath, fin.alphaName);
    const std::vector<LinearSolveRecord> ofE = brae::gatecheck::readOfSolves(logPath, "epsilon");
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    // the FAIL-PROOF: nothing in compareSolves can pass on an empty parse
    check("OpenFOAM's log gave one epsilon and one k solve per step",
          ofE.size() == static_cast<std::size_t>(nSteps) && ofK.size() == ofE.size());
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(2e-6), scalar(2e-6));
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               scalar(1e-10), scalar(1e-9));
    // EPSILON'S RESIDUALS ARE PRINTED, NOT ASSERTED, and the reason is structural. rightWall is a `wall`
    // with an epsilonWallFunction, and here it carries a FLUX. OpenFOAM's epsilonWallFunction derives
    // from fixedValue and brae's is a zeroGradient patch beside constrained cells, so the convection
    // term leaves a different diagonal in the wall cells before setValues overwrites their rows. The
    // SOLUTION is the same -- epsilon agrees to 7e-15 below -- but the residual's normalisation sums
    // those rows: MEASURED, 1.8e-06 per step and growing with the wall's flux, against 0.0 on the same
    // case with the wall closed. The iteration counts are asserted.
    brae::gatecheck::compareSolves("host", r.epsilonSolves, ofE, nSteps, "epsilon", scalar(1e-10),
                                   scalar(1e-10), scalar(-1), nullptr, false);
    {
        bool sameCounts = r.epsilonSolves.size() == ofE.size() && !ofE.empty();
        for (std::size_t q = 0; q < ofE.size() && q < r.epsilonSolves.size(); ++q)
        {
            sameCounts = sameCounts && r.epsilonSolves[q].nIterations == ofE[q].nIterations;
        }
        check("every epsilon solve took OpenFOAM's iteration count", sameCounts);
    }
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

    // THE WALL ITSELF: OpenFOAM writes both mixed conditions' values, so brae's patches are held to them
    const FieldData<vector> ofUFd = readField<vector>(ofDir + "/U");
    const FieldData<scalar> ofPFd = readField<scalar>(ofDir + "/p_rgh");
    scalar dWallU = 0;
    scalar wallUScale = 0;
    scalar dWallP = 0;
    scalar wallPScale = 0;
    std::size_t nCompared = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!fin.p_rgh.boundary[pi]->isPrghPermeableAlphaTotalPressure()) continue;
        const PatchFieldData<vector>* bu = findPatchEntry(ofUFd.boundary, patches[pi]);
        const PatchFieldData<scalar>* bp = findPatchEntry(ofPFd.boundary, patches[pi]);
        if (!bu || !bp || !bu->hasValue || !bp->hasValue) continue;
        // OpenFOAM writes a patch whose faces all hold one value as `uniform`: the wet wall's U is
        // (0 0 0) on every face, wet or sucked-through
        const std::size_t nf = static_cast<std::size_t>(patches[pi].size);
        const std::vector<vector> ofUb = bu->valueUniform ? std::vector<vector>(nf, bu->uniformValue) : bu->values;
        const std::vector<scalar> ofPb = bp->valueUniform ? std::vector<scalar>(nf, bp->uniformValue) : bp->values;
        const Diff du = compare(fin.U.boundary[pi]->value(), ofUb);
        const Diff dp = compare(fin.p_rgh.boundary[pi]->value(), ofPb);
        dWallU = du.linf;
        wallUScale = du.refMax;
        dWallP = dp.linf;
        wallPScale = dp.refMax;
        nCompared = nf;
    }
    std::printf("  on the wall: U_b Linf %.4e (|U_b| up to %.4e), p_rgh_b Linf %.4e (|p_rgh_b| up to %.4e), %zu faces\n",
                (double)dWallU, (double)wallUScale, (double)dWallP, (double)wallPScale, nCompared);
    check("the wall's U is OpenFOAM's written value, face by face",
          nCompared > 0 && dWallU <= B.U*std::fmax(dU.refMax, scalar(1e-300)));
    check("...and its p_rgh",
          nCompared > 0 && dWallP <= B.prgh*std::fmax(dP.refMax, scalar(1e-300)));

    // THE CONTROL, on the oracle
    const Diff dClosedU = compare(readVectorCells(closedDir + "/U"), ofU);
    const Diff dClosedP = compare(readCells(closedDir + "/p_rgh"), ofPrgh);
    std::printf("  CONTROL: OpenFOAM with the wall CLOSED against OpenFOAM with it permeable, U relative %.4e, "
                "p_rgh %.4e\n", (double)dClosedU.rel(), (double)dClosedP.rel());
    check("the permeable wall moves OpenFOAM's own U far more than brae is from it",
          dClosedU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dClosedU.rel() > scalar(1e-3));

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
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        failures += brae::gatecheck::nonFinite("device k", dev.turbulence.k.internal);
        failures += brae::gatecheck::nonFinite("device epsilon", dev.turbulence.epsilon.internal);
        failures += brae::gatecheck::nonFinite("device nut", dev.turbulence.nut.internal);
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        const Diff eK = compare(dev.turbulence.k.internal, ofKf);
        const Diff eE = compare(dev.turbulence.epsilon.internal, ofEf);
        const Diff eN = compare(dev.turbulence.nut.internal, ofNut);
        std::printf("  DEVICE:  alpha %.4e, p_rgh %.4e, U %.4e, k %.4e, epsilon %.4e, nut %.4e\n",
                    (double)eA.linf, (double)eP.rel(), (double)eU.rel(), (double)eK.rel(),
                    (double)eE.rel(), (double)eN.rel());
        std::printf("  host:    alpha %.4e, p_rgh %.4e, U %.4e, k %.4e, epsilon %.4e, nut %.4e\n",
                    (double)dA.linf, (double)dP.rel(), (double)dU.rel(), (double)dK.rel(),
                    (double)dE.rel(), (double)dN.rel());
        // every p_rgh solve of the run, against OpenFOAM's log
        failures += brae::gatecheck::compareSolves("device", rd.pSolves, ofP, nSteps, "p_rgh", DB.pStepOne,
                                                   DB.pRun);
        // THE WALL ITSELF on the device arm: the same faces wet, and both conditions' written values
        std::size_t wetEndDev = 0;
        scalar eWallU = 0;
        scalar eWallP = 0;
        std::size_t nComparedDev = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!dev.p_rgh.boundary[pi]->isPrghPermeableAlphaTotalPressure()) continue;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                wetEndDev += (dev.alpha1.boundary[pi]->value()[static_cast<std::size_t>(i)] > scalar(0.01)) ? 1 : 0;
            }
            const PatchFieldData<vector>* bu = findPatchEntry(ofUFd.boundary, patches[pi]);
            const PatchFieldData<scalar>* bp = findPatchEntry(ofPFd.boundary, patches[pi]);
            if (!bu || !bp || !bu->hasValue || !bp->hasValue) continue;
            const std::size_t nf = static_cast<std::size_t>(patches[pi].size);
            const std::vector<vector> ofUb = bu->valueUniform ? std::vector<vector>(nf, bu->uniformValue) : bu->values;
            const std::vector<scalar> ofPb = bp->valueUniform ? std::vector<scalar>(nf, bp->uniformValue) : bp->values;
            eWallU = compare(dev.U.boundary[pi]->value(), ofUb).linf;
            eWallP = compare(dev.p_rgh.boundary[pi]->value(), ofPb).linf;
            nComparedDev = nf;
        }
        std::printf("  DEVICE on the wall: %zu faces wet at the end (host %zu), U_b Linf %.4e, p_rgh_b Linf %.4e\n",
                    wetEndDev, wetEnd, (double)eWallU, (double)eWallP);
        check("the device left the faces wet that the host loop left wet", wetEndDev == wetEnd);
        // held to the device arm's OWN bounds (DEV_*, at the top of this file, with the measurement)
        check("the device's alpha agrees with OpenFOAM's absolutely", eA.linf < DB.f.alpha);
        check("...its p_rgh", eP.rel() < DB.f.prgh);
        check("...its U", eU.rel() < DB.f.U);
        check("...its k", eK.rel() < DB.f.k);
        check("...its epsilon", eE.rel() < DB.f.epsilon);
        check("...its nut", eN.rel() < DB.f.nut);
        // the wall's own values are held to the HOST arm's bounds: measured 9.0e-16 and 4.9e-15 on
        // shipped, exactly 0 and 1.3e-11 (of 2.6e+03) on wetWall
        check("...the wall's U, face by face",
              nComparedDev > 0 && eWallU <= B.U*std::fmax(dU.refMax, scalar(1e-300)));
        check("...and the wall's p_rgh",
              nComparedDev > 0 && eWallP <= B.prgh*std::fmax(dP.refMax, scalar(1e-300)));
        check("the permeable wall moves OpenFOAM's own U far more than the device is from it",
              dClosedU.rel() > scalar(1000)*std::fmax(eU.rel(), scalar(1e-14)));
    }

    std::printf("test_inter_permeable_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
