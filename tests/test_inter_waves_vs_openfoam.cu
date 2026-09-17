// brae's interFoam WAVE boundary conditions against REAL OpenFOAM's, on laminar/waves/stokesI.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, and its LOG, which for
// these conditions says three things a field cannot:
//   "Reference water depth" and "Wave length"   the model's two derived constants, per patch
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
    // per "Wave model: patch <name>" block
    std::vector<std::string> patch;
    std::vector<scalar> waterDepthRef;
    std::vector<scalar> waveLength;
};

OfWaveLog readOfWaveLog(const std::string& path)
{
    OfWaveLog w;
    std::ifstream in(path);
    std::string line;
    const std::string kU = " wave model for patch ";
    const std::string kP = "Wave model: patch ";
    const std::string kD = "Reference water depth : ";
    const std::string kL = "Wave length : ";
    while (std::getline(in, line))
    {
        std::size_t p = line.find(kU);
        if (line.rfind("Updating ", 0) == 0 && p != std::string::npos)
        {
            w.updates.push_back(line.substr(p + kU.size()));
            continue;
        }
        p = line.find(kP);
        if (p != std::string::npos)
        {
            w.patch.push_back(line.substr(p + kP.size()));
            w.waterDepthRef.push_back(scalar(-1));
            w.waveLength.push_back(scalar(0));
            continue;
        }
        if (w.patch.empty()) continue;
        p = line.find(kD);
        if (p != std::string::npos)
        {
            w.waterDepthRef.back() = std::atof(line.c_str() + p + kD.size());
        }
        p = line.find(kL);
        if (p != std::string::npos)
        {
            w.waveLength.back() = std::atof(line.c_str() + p + kL.size());
        }
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
        std::printf("  patch %-8s OpenFOAM: depth %.15g  length %.15g\n", ofw.patch[q].c_str(),
                    (double)ofw.waterDepthRef[q], (double)ofw.waveLength[q]);
        check("...brae built a model for the patch OpenFOAM built one for", have);
        if (!have) continue;
        const brae::cpu::waveModels::WaveModel& wm = *fin.waves.model[pi];
        std::printf("  patch %-8s brae:     depth %.15g  length %.15g  (%s)\n", ofw.patch[q].c_str(),
                    (double)wm.waterDepthRef(), (double)wm.waveLength(), wm.type().c_str());
        // the log prints 15 significant digits
        check("...with OpenFOAM's reference water depth",
              std::fabs(wm.waterDepthRef() - ofw.waterDepthRef[q]) < scalar(2e-15)*ofw.waterDepthRef[q]);
        check("...and its wave length",
              std::fabs(wm.waveLength() - ofw.waveLength[q])
            < scalar(2e-15)*std::fmax(ofw.waveLength[q], scalar(1)));
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
    const scalar alphaBound = tight ? scalar(3e-10) : scalar(5e-9);
    const scalar pBound = tight ? scalar(3e-10) : scalar(2e-8);
    const scalar uBound = tight ? scalar(2e-8) : scalar(5e-7);

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
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(1e-10),
                                               tight ? scalar(1e300) : scalar(1e-5));

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

    std::printf("test_inter_waves_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
