// brae's interFoam WAVE boundary conditions against REAL OpenFOAM's, on the nine laminar/waves
// tutorials -- one per wave generation model.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, and its LOG, which for
// these conditions says three things a field cannot:
//   the block under "Wave model: patch <p>"     every constant the model derives, label for label:
//                                               reference depth, wave length, StokesV's Lambda,
//                                               cnoidal's m parameter, a solitary wave's x0
//   "Updating <model> wave model for patch <p>" the ORDER the models update in -- once per time INDEX,
//                                               and a sub-cycle is its own index
//   "Selecting waveModel <model>"               WHEN each model is created, which decides what alpha a
//                                               case naming no reference depth takes it from
// ...and the time directory carries the patch VALUES the two conditions last assigned, which is the
// wave model's own output, face by face.
//
// tests/interfoam_waves_vs_openfoam.sh says what each profile is for and what it measured.
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
#include <fstream>
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

// OpenFOAM's log, as far as the wave models speak in it
struct OfWaveLog
{
    // "Updating <type> wave model for patch <name>" -> name, in order
    std::vector<std::string> updates;
    // one block per "Wave model: patch <name>": the indented "label : value" lines under it, as text
    std::vector<std::string> patch;
    std::vector<std::vector<std::pair<std::string, std::string>>> entries;
};

std::string trimmed(const std::string& t)
{
    const std::size_t b = t.find_first_not_of(" \t");
    const std::size_t e = t.find_last_not_of(" \t");
    return (b == std::string::npos) ? std::string() : t.substr(b, e - b + 1);
}

OfWaveLog readOfWaveLog(const std::string& path)
{
    OfWaveLog w;
    std::ifstream in(path);
    std::string line;
    const std::string kU = " wave model for patch ";
    const std::string kP = "Wave model: patch ";
    bool inBlock = false;
    while (std::getline(in, line))
    {
        std::size_t p = line.find(kU);
        if (line.rfind("Updating ", 0) == 0 && p != std::string::npos)
        {
            w.updates.push_back(line.substr(p + kU.size()));
            inBlock = false;
            continue;
        }
        if (line.rfind(kP, 0) == 0)
        {
            w.patch.push_back(line.substr(kP.size()));
            w.entries.emplace_back();
            inBlock = true;
            continue;
        }
        // the block is the run of indented lines under its heading
        if (!inBlock) continue;
        if (line.rfind("    ", 0) != 0)
        {
            inBlock = false;
            continue;
        }
        p = line.find(':');
        if (p == std::string::npos) continue;
        w.entries.back().push_back({trimmed(line.substr(0, p)), trimmed(line.substr(p + 1))});
    }
    return w;
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: waves (waveAlpha, waveVelocity) ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <profile> "
                    "<noWaveOfTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string profile = argv[6];
    const std::string stillDir = argv[7];
    // `tight`: both codes' p_rgh solves at 1e-13 with no relTol, which takes the stopping point out of
    // the comparison. Every other profile runs the case's own tolerances.
    const bool tight = (profile == "tight");
    std::printf("  profile: %s\n", profile.c_str());

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    check("brae found wave boundary conditions on the case", fin.waves.any);

    // THE MODELS' CONSTANTS AND THEIR UPDATE ORDER, against OpenFOAM's log
    const OfWaveLog ofw = readOfWaveLog(logPath);
    check("OpenFOAM's log describes at least one wave model and its updates",
          !ofw.patch.empty() && !ofw.updates.empty());
    for (std::size_t q = 0; q < ofw.patch.size(); ++q)
    {
        std::size_t pi = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k)
        {
            if (patches[k].name == ofw.patch[q])
            {
                pi = k;
            }
        }
        const bool have = pi < patches.size() && fin.waves.model[pi];
        check("brae built a model for the patch OpenFOAM built one for", have);
        if (!have) continue;
        const brae::cpu::waveModels::WaveModel& wm = *fin.waves.model[pi];
        std::string ofType;
        for (const auto& e : ofw.entries[q])
        {
            if (e.first == "Type")
            {
                ofType = e.second;
            }
        }
        std::printf("  patch %-8s OpenFOAM %s, brae %s\n", ofw.patch[q].c_str(), ofType.c_str(),
                    wm.type().c_str());
        check("...of the same type", ofType == wm.type());
        // EVERY NUMBER THE MODEL DERIVES, label for label against the block OpenFOAM printed when it
        // created the model: the reference depth, the wave length, StokesV's lambda, cnoidal's m, a
        // solitary wave's x0. The log carries 15 significant digits, so 1e-14 relative is its last.
        std::size_t matched = 0;
        scalar worst = 0;
        std::string worstLabel;
        for (const auto& mine : wm.info())
        {
            for (const auto& e : ofw.entries[q])
            {
                if (e.first != mine.first) continue;
                const scalar ofv = std::atof(e.second.c_str());
                const scalar err = std::fabs(mine.second - ofv)/std::fmax(std::fabs(ofv), scalar(1e-300));
                const scalar rel = (mine.second == ofv) ? scalar(0) : err;
                if (rel >= worst)
                {
                    worst = rel;
                    worstLabel = mine.first;
                }
                ++matched;
            }
        }
        std::printf("    %zu of brae's %zu constants found in OpenFOAM's block; worst %.3e (%s)\n",
                    matched, wm.info().size(), (double)worst, worstLabel.c_str());
        check("...every constant brae reports is one OpenFOAM printed", matched == wm.info().size());
        check("...and equal to it to the log's last digit", worst < scalar(1e-14));
    }
    {
        std::vector<std::string> mine;
        for (const std::string& u : fin.waves.updateLog)
        {
            mine.push_back(u.substr(0, u.find('@')));
        }
        std::printf("  updates: OpenFOAM %zu, brae %zu; first step:", ofw.updates.size(), mine.size());
        for (std::size_t q = 0; q < mine.size() && q < 8; ++q)
        {
            std::printf(" %s", fin.waves.updateLog[q].c_str());
        }
        std::printf("\n");
        check("the wave models updated in OpenFOAM's ORDER, update for update", mine == ofw.updates);
    }

    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    auto cells = [&](const FieldData<scalar>& fd)
    {
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
    auto vectorCells = [&](const FieldData<vector>& fd)
    {
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

    const FieldData<scalar> ofAlphaFd = readField<scalar>(ofDir + "/" + fin.alphaName);
    const FieldData<vector> ofUFd = readField<vector>(ofDir + "/U");
    const std::vector<scalar> ofAlpha = cells(ofAlphaFd);
    const std::vector<scalar> ofPrgh = cells(readField<scalar>(ofDir + "/p_rgh"));
    const std::vector<vector> ofU = vectorCells(ofUFd);

    scalar uScale = 0;
    for (const vector& u : ofU)
    {
        uScale = std::fmax(uScale, mag(u));
    }
    // MEASURED at the case's own tolerances (p_rgh 1e-6 relTol 0.1, p_rghFinal 1e-7), worst of
    // `shipped`, `trough` and `crest`: alpha 1.4e-10, p_rgh 6.2e-10, and U 1.8e-08 at full amplitude
    // and 1.2e-07 on `shipped`, where |U| is 4.7e-03 and the same absolute difference is a larger
    // fraction. That is the tolerance ball: under `tight` the same fixture reads alpha 7.7e-12, p_rgh
    // 7.7e-12 and U 6.3e-10 -- and p_rgh's is alpha's times rho*g*h, the weight of the water column
    // the alpha difference stands for. Bounds at about 30x, 4x for U on `shipped`. With
    // constrainPressure taking phi_b for Sf & U_b the same arms read alpha 3.0e-03 and U 260%.
    //
    // THOSE WERE MEASURED WITH PCG STAGED IN GAMG's PLACE. With p_rghFinal run as the tutorials name
    // it -- GAMG, on both loops -- the worst of the fourteen profiles reads alpha 2.9e-11, p_rgh
    // 5.2e-11 and U 9.3e-09 (`shipped`, on the device; 1.9e-09 elsewhere), so the bounds came down
    // from 5e-9, 2e-8 and 5e-7. `tight` is unchanged at 7.8e-12, 7.8e-12 and 5.7e-10.
    const scalar alphaBound = tight ? scalar(3e-10) : scalar(1e-9);
    const scalar pBound = tight ? scalar(3e-10) : scalar(2e-9);
    const scalar uBound = tight ? scalar(2e-8) : scalar(2e-7);

    // THE PATCH VALUES the two conditions last assigned -- the wave model's output, face by face
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::size_t n = static_cast<std::size_t>(patches[pi].size);
        if (fin.waves.alphaPatch[pi])
        {
            for (const auto& b : ofAlphaFd.boundary)
            {
                if (b.name != patches[pi].name) continue;
                const std::vector<scalar> ofb = b.valueUniform ? std::vector<scalar>(n, b.uniformValue)
                                                               : b.values;
                const Diff d = compare(fin.alpha1.boundary[pi]->value(), ofb);
                std::printf("  waveAlpha    on %-8s patch values Linf %.4e (up to %.4e)\n",
                            patches[pi].name.c_str(), (double)d.linf, (double)d.refMax);
                check("...the alpha patch values are OpenFOAM's", d.linf < scalar(1e-12));
            }
        }
        if (fin.waves.UPatch[pi])
        {
            for (const auto& b : ofUFd.boundary)
            {
                if (b.name != patches[pi].name) continue;
                const std::vector<vector> ofb = b.valueUniform ? std::vector<vector>(n, b.uniformValue)
                                                               : b.values;
                const Diff d = compare(fin.U.boundary[pi]->value(), ofb);
                std::printf("  waveVelocity on %-8s patch values Linf %.4e (|U_b| up to %.4e)\n",
                            patches[pi].name.c_str(), (double)d.linf, (double)d.refMax);
                // AGAINST THE FIELD'S SCALE AND AT THE FIELD'S BOUND. Not the patch's own scale: under
                // tight solves the outlet's velocity is 1e-12 -- the wave has not arrived -- and a
                // ratio of two such numbers is noise. And not tighter than U's: both models read
                // alpha's water level, and the outlet's is NOTHING BUT that reading times
                // sqrt(g/h), so it carries whatever alpha carries. MEASURED, of the field's |U|: inlet
                // 2e-10, outlet 9.6e-09 on `shipped` (where that velocity is itself the tolerance
                // ball, 3.8e-04 before any wave arrives) and 1.6e-11 under `tight`.
                check("...the velocity patch values are OpenFOAM's", d.linf < uBound*uScale);
            }
        }
    }

    // Every iteration count is asserted on every profile. The initial RESIDUALS are asserted at the
    // case's own tolerances only: under `tight` the second corrector starts from a residual of 1e-11,
    // where two correct solves differ in the fourth digit (measured 1.5e-04 relative) and mean nothing.
    // MEASURED over the run WITH PCG STAGED FOR p_rghFinal: at most 5.5e-08 on eleven of the thirteen
    // profiles, bound 1e-5. The two 3-D solitary cases read 4.6e-06 and 4.2e-05, every bit of it on the
    // SECOND corrector, which starts from the 1.8e-07 the first one's one-iteration solve left behind
    // -- 1e-12 of absolute difference is 5e-06 of that -- while their first correctors agree to 1e-08
    // and all 60 iteration counts are OpenFOAM's. Those two were bounded at 1e-3.
    //
    // THAT EXCEPTION IS GONE. It belonged to the PCG staged in GAMG's place: run as the tutorials name
    // it, the two 3-D cases read 1.7e-09 and 6.4e-09 over the run and the worst of all thirteen
    // untightened profiles is 6.4e-09, host and device, so one bound holds them all and it is 1e-6.
    const scalar runBound = tight ? scalar(1e300) : scalar(1e-6);
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(1e-10),
                                               runBound);

    // `mompred`: the momentum predictor's own solves. This case is 2-D in x and z, so OpenFOAM logs
    // Ux and Uz and no Uy -- the empty direction is skipped, not solved to zero.
    if (profile == "mompred")
    {
        const std::vector<LinearSolveRecord> ofUx = brae::gatecheck::readOfSolves(logPath, "Ux");
        const std::vector<LinearSolveRecord> ofUy = brae::gatecheck::readOfSolves(logPath, "Uy");
        const std::vector<LinearSolveRecord> ofUz = brae::gatecheck::readOfSolves(logPath, "Uz");
        check("OpenFOAM logged a Ux and a Uz solve for every step, and no Uy",
              ofUx.size() == static_cast<std::size_t>(nSteps) && ofUz.size() == ofUx.size() && ofUy.empty());
        check("...and brae did not solve Uy either", r.uSolves[1].empty());
        failures += brae::gatecheck::compareSolves("host", r.uSolves[0], ofUx, nSteps, "Ux",
                                                   scalar(1e-10), scalar(1e-6));
        failures += brae::gatecheck::compareSolves("host", r.uSolves[2], ofUz, nSteps, "Uz",
                                                   scalar(1e-10), scalar(1e-6));
    }

    // `mulescorr`: the implicit alpha pre-solve, one per sub-cycle per step. Its matrix is assembled on
    // the wave patch's NEW value -- the update's second call site -- and updating after the pre-solve
    // instead reads alpha 8.9e-07 from OpenFOAM against 1.1e-10.
    const bool mulesCorr = (profile == "mulescorr");
    const std::vector<LinearSolveRecord> ofAlphaSolves =
        mulesCorr ? brae::gatecheck::readOfSolves(logPath, fin.alphaName) : std::vector<LinearSolveRecord>();
    if (mulesCorr)
    {
        check("OpenFOAM logged an alpha solve for every sub-cycle of every step",
              ofAlphaSolves.size() == static_cast<std::size_t>(nSteps)
                                     * static_cast<std::size_t>(fin.alphaCtl.nAlphaSubCycles));
        failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofAlphaSolves, nSteps,
                                                   fin.alphaName.c_str(), scalar(1e-10), scalar(1e-6));
    }

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    // the bounds are stated where they are first used, above the patch values
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < alphaBound);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < pBound);
    check("U agrees with OpenFOAM's relatively", dU.rel() < uBound);

    // THE CONTROL, on the oracle: OpenFOAM's own answer for the same tank with NO wave
    const Diff dStillU = compare(vectorCells(readField<vector>(stillDir + "/U")), ofU);
    const Diff dStillA = compare(cells(readField<scalar>(stillDir + "/" + fin.alphaName)), ofAlpha);
    std::printf("  CONTROL: OpenFOAM with no wave against OpenFOAM with it, U relative %.4e, alpha %.4e\n",
                (double)dStillU.rel(), (double)dStillA.linf);
    check("the wave moves OpenFOAM's own U far more than brae is from it",
          dStillU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)));

    // THE DEVICE LOOP, against OpenFOAM directly and at the case's own tolerances -- what
    // `brae_interFoam -device` runs. The wave model is the HOST's on both paths (a boundary condition's
    // per-patch arithmetic, reached through the alpha and velocity hooks); what is under test is that
    // the device loop calls it at OpenFOAM's clock, in OpenFOAM's order, and feeds what it returns into
    // the alpha flux, the limiter, the momentum matrix and constrainPressure.
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
    else
    {
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        check("the device driver ran the same number of steps", rd.steps == nSteps);
        check("...and found the wave boundary conditions", dev.waves.any);
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        {
            std::vector<std::string> mine;
            for (const std::string& u : dev.waves.updateLog)
            {
                mine.push_back(u.substr(0, u.find('@')));
            }
            check("the DEVICE loop updated the wave models in OpenFOAM's ORDER, update for update",
                  mine == ofw.updates);
        }
        // STEP ONE's initial residuals: the host reads 1.4e-11 at worst and is bounded at 1e-10; the
        // device reads 2.2e-10, on the second corrector's 6e-05 -- a difference of 1e-14 in a residual
        // that is itself the difference of two numbers of order one, summed in the GPU's order. 5e-9.
        failures += brae::gatecheck::compareSolves("device", rd.pSolves, ofP, nSteps, "p_rgh",
                                                   scalar(5e-9), runBound);
        if (mulesCorr)
        {
            failures += brae::gatecheck::compareSolves("device", rd.alphaSolves, ofAlphaSolves, nSteps,
                                                       dev.alphaName.c_str(), scalar(5e-9), scalar(1e-6));
        }
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        std::printf("  DEVICE vs OpenFOAM: alpha %.4e, p_rgh %.4e, U %.4e\n", (double)eA.linf,
                    (double)eP.rel(), (double)eU.rel());
        // THE HOST'S BOUNDS. MEASURED, worst of the thirteen profiles at the case's tolerances: alpha
        // 1.3e-10, p_rgh 6.3e-10, U 1.8e-08 (1.3e-07 on `shipped`); under `tight` 7.8e-12, 7.8e-12 and
        // 7.5e-10. ONE CASE READS VERY DIFFERENTLY ON THE TWO PATHS AND IT IS NOT A DEFECT: on
        // solitaryMcCowan the host is 1.6e-10 of alpha from OpenFOAM and the device 2.3e-12. With every
        // p_rgh solve tightened on all three codes they read 1.9e-12 and 2.0e-12 -- the same -- so the
        // host's figure is where that case's relTol 0.1 happened to leave it.
        check("the DEVICE's alpha agrees with OpenFOAM's to the host's bound", eA.linf < alphaBound);
        check("...its p_rgh", eP.rel() < pBound);
        check("...and its U", eU.rel() < uBound);
    }

    std::printf("test_inter_waves_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
