// brae's interFoam under localEuler (LTS) against REAL OpenFOAM's, on RAS/DTCHull: the local time step
// setRDeltaT.H forms, step by step, and the fields it advances. The script
// (tests/interfoam_dtchull_vs_openfoam.sh) says what is staged and why.
//
// usage: test_inter_dtchull_vs_openfoam <caseDir> <ofCaseDir> <nSteps> <ofLog> [measure]
//   caseDir     the staged case, meshed, with its 0 directory
//   ofCaseDir   the same case after OpenFOAM ran it, one time directory per step
//   ofLog       OpenFOAM's log, for setRDeltaT.H's Info lines and the solves
//   measure     print every number and assert nothing but the path -- for setting bounds
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "inter_solve_log.cuh"
#include "device_gate_finite.cuh"
#include "patch_entry_lookup.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {

int failures = 0;
bool measureOnly = false;

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

// a bound, unless the harness only measures
void bound(
    const std::string& what,
    scalar value,
    scalar limit)
{
    if (measureOnly)
    {
        std::printf("  meas: %-52s %.4e\n", what.c_str(), (double)value);
        return;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%-52s %.4e (bound %.1e)", what.c_str(), (double)value, (double)limit);
    check(buf, value <= limit);
}

// THE BOUNDS, from ten laminar steps: rDeltaT 3.4e-12 at worst over the steps, the time-scale lines
// 3.4e-12, alpha 2.6e-10, p_rgh 6.1e-10, U 6.4e-13, the p_rgh initial residuals 4.5e-11 -- each bound about
// three times its measurement. OpenFOAM against itself with one interface cell's alpha moved by one ulp
// reads alpha 8.8e-11, p_rgh 1.2e-10 and U 3.7e-13 after the same ten steps; see the script.
// ...and the `ras` profile's, ten kOmegaSST steps under the tutorial's `linearUpwind limitedGrad`: k
// 3.4e-12, omega 9.0e-12, nut 2.8e-11, U 1.5e-12 (the laminar profile's U bound stays its own). OpenFOAM
// against itself with one ulp reads k 2.8e-13, omega 4.7e-13, nut 6.2e-12, U 1.2e-13 there; at the FIRST
// step, before anything amplifies, U is 7.9e-13 on the laminar profile as well, so that gap is not the
// closure's, and omega and nut follow it (4e-13). Staged upwind, the same fields read within 10% of these.
constexpr scalar BOUND_K = 1e-11;
constexpr scalar BOUND_OMEGA = 3e-11;
constexpr scalar BOUND_NUT = 1e-10;
constexpr scalar BOUND_U_LAMINAR = 3e-12;
constexpr scalar BOUND_U_RAS = 5e-12;
// ...and the hull's wall nut, face by face, under nutkRoughWallFunction: 6.2e-11 after ten steps (1.7e-12 at
// the first), OpenFOAM's one-ulp floor 7.9e-12. Its controls -- the smooth wall function, the history
// dropped -- read 1.0e+00 and 4.6e-01.
constexpr scalar BOUND_NUT_WALL = 2e-10;
constexpr scalar BOUND_U_OUTLET = 2e-11;

struct Diff
{
    scalar linf = 0;
    scalar refMax = 0;
    long nOff = 0;           // cells more than 1e-12 of refMax apart
    scalar rel() const { return linf/std::fmax(refMax, scalar(1e-300)); }
};

Diff compare(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const scalar e = std::fabs(a[i] - b[i]);
        d.linf = std::fmax(d.linf, e);
        if (e > scalar(1e-12)*d.refMax) ++d.nOff;
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
        d.refMax = std::fmax(d.refMax, mag(b[i]));
    }
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const vector e{a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z};
        d.linf = std::fmax(d.linf, mag(e));
        if (mag(e) > scalar(1e-12)*d.refMax) ++d.nOff;
    }
    return d;
}

template<class T>
std::vector<T> readCells(
    const std::string& path,
    label nC)
{
    const FieldData<T> fd = readField<T>(path);
    std::vector<T> v;
    if (fd.internalUniform)
    {
        v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
    }
    else
    {
        v = fd.internalField;
    }
    return v;
}

// setRDeltaT.H's Info lines, in order: {kind, min, max} with kind 0 flow, 1 smoothed, 2 damped
struct TimeScaleLine
{
    int kind = 0;
    double lo = 0;
    double hi = 0;
};

std::vector<TimeScaleLine> readTimeScaleLines(const std::string& logPath)
{
    std::vector<TimeScaleLine> out;
    std::ifstream in(logPath);
    const std::regex re("^(Flow|Smoothed flow|Damped flow) time scale min/max = ([^,]+), (.+)$");
    std::string line;
    while (std::getline(in, line))
    {
        std::smatch mt;
        if (!std::regex_match(line, mt, re)) continue;
        TimeScaleLine t;
        t.kind = (mt[1] == "Flow") ? 0 : (mt[1] == "Smoothed flow") ? 1 : 2;
        t.lo = std::strtod(mt[2].str().c_str(), nullptr);
        t.hi = std::strtod(mt[3].str().c_str(), nullptr);
        out.push_back(t);
    }
    return out;
}

scalar relDiff(
    double a,
    double b)
{
    return std::fabs(a - b)/std::fmax(std::fabs(b), 1e-300);
}

} // namespace


int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/DTCHull under localEuler ==\n");
    if (argc < 5)
    {
        std::printf("  SKIP: usage: %s <caseDir> <ofCaseDir> <nSteps> <ofLog> [measure]\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string ofCase = argv[2];
    const label nSteps = static_cast<label>(std::atol(argv[3]));
    const std::string logPath = argv[4];
    measureOnly = (argc > 5 && std::string(argv[5]) == "measure");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();
    std::printf("  mesh: %d cells\n", (int)nC);

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, caseDir + "/0", m, g, patches, nSteps, /*verbose=*/true, &fin);

    // THE PATH
    check("brae ran the same number of steps", r.steps == nSteps);
    check("brae ran the case under localEuler", fin.lts);
    check("...and formed one local time step per step",
          r.rDeltaTPerStep.size() == static_cast<std::size_t>(nSteps)
       && r.ltsLog.size() == static_cast<std::size_t>(nSteps));
    // damping from the third step: timeIndex > startTimeIndex + 1, tested before ++runTime
    bool dampOk = true;
    for (label k = 0; k < nSteps && k < static_cast<label>(r.ltsLog.size()); ++k)
    {
        dampOk = dampOk && (r.ltsLog[static_cast<std::size_t>(k)].damped == (k >= 2));
    }
    check("...damped from the third step on and never before", dampOk);

    // setRDeltaT.H's Info lines against OpenFOAM's, step by step
    const std::vector<TimeScaleLine> ofLines = readTimeScaleLines(logPath);
    std::vector<TimeScaleLine> braeLines;
    for (const SetRDeltaTReport& lr : r.ltsLog)
    {
        braeLines.push_back({0, lr.flowMin, lr.flowMax});
        braeLines.push_back({1, lr.smoothedMin, lr.smoothedMax});
        if (lr.damped)
        {
            braeLines.push_back({2, lr.dampedMin, lr.dampedMax});
        }
    }
    // OpenFOAM's run may be LONGER than brae's -- a control runs fewer steps against the same oracle --
    // so the lines are compared over brae's, and OpenFOAM must have at least as many
    check("OpenFOAM's log gave setRDeltaT's lines, at least as many as brae's",
          !ofLines.empty() && ofLines.size() >= braeLines.size());
    scalar dLines = 0;
    for (std::size_t i = 0; i < ofLines.size() && i < braeLines.size(); ++i)
    {
        if (ofLines[i].kind != braeLines[i].kind)
        {
            dLines = 1;
            continue;
        }
        dLines = std::fmax(dLines, relDiff(braeLines[i].lo, ofLines[i].lo));
        dLines = std::fmax(dLines, relDiff(braeLines[i].hi, ofLines[i].hi));
    }
    bound("time scale min/max lines, largest relative", dLines, 1e-11);

    // the local time step each step ran with, against the rDeltaT OpenFOAM writes at that step's time
    for (label k = 1; k <= nSteps; ++k)
    {
        if (static_cast<std::size_t>(k) > r.rDeltaTPerStep.size()) break;
        const std::string dir = ofCase + "/" + std::to_string(k);
        const std::vector<scalar> ofR = readCells<scalar>(dir + "/rDeltaT", nC);
        const std::vector<scalar>& br = r.rDeltaTPerStep[static_cast<std::size_t>(k - 1)];
        const Diff d = compare(br, ofR);
        long above = 0;
        for (const scalar v : ofR)
        {
            if (v > scalar(1)) ++above;
        }
        std::printf("  step %2d: rDeltaT %ld of %d cells above the floor 1/maxDeltaT, %ld differ\n",
                    (int)k, above, (int)nC, d.nOff);
        bound("rDeltaT at step " + std::to_string(k) + ", relative", d.rel(), 1e-11);
    }

    // BEFORE any fmax: std::fmax drops a NaN, so a non-finite field would read as a match
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    failures += brae::gatecheck::nonFinite("brae rDeltaT", fin.rDeltaT);

    // the fields at the last step
    const std::string last = ofCase + "/" + std::to_string(nSteps);
    const Diff dA = compare(fin.alpha1.internal, readCells<scalar>(last + "/alpha.water", nC));
    const Diff dP = compare(fin.p_rgh.internal, readCells<scalar>(last + "/p_rgh", nC));
    const Diff dU = compare(fin.U.internal, readCells<vector>(last + "/U", nC));
    std::printf("  cells more than 1e-12 apart: alpha %ld, p_rgh %ld, U %ld\n", dA.nOff, dP.nOff, dU.nOff);
    bound("alpha, relative to its largest value", dA.rel(), 1e-9);
    bound("p_rgh, relative", dP.rel(), 2e-9);
    bound("U, relative", dU.rel(), fin.turbulence.on ? BOUND_U_RAS : BOUND_U_LAMINAR);

    // THE CLOSURE, on the `ras` profile: kOmegaSST's fvm::ddt(omega) and fvm::ddt(k) under the local step
    if (fin.turbulence.on)
    {
        check("...under kOmegaSST, in the uniform lineage",
              fin.turbulence.model == InterRasModel::KOmegaSST && !fin.turbulence.variableDensity);
        failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
        failures += brae::gatecheck::nonFinite("brae omega", fin.turbulence.omega.internal);
        failures += brae::gatecheck::nonFinite("brae nut", fin.turbulence.nut.internal);
        const Diff dK = compare(fin.turbulence.k.internal, readCells<scalar>(last + "/k", nC));
        const Diff dO = compare(fin.turbulence.omega.internal, readCells<scalar>(last + "/omega", nC));
        const Diff dN = compare(fin.turbulence.nut.internal, readCells<scalar>(last + "/nut", nC));
        std::printf("  cells more than 1e-12 apart: k %ld, omega %ld, nut %ld\n", dK.nOff, dO.nOff, dN.nOff);
        bound("k, relative", dK.rel(), BOUND_K);
        bound("omega, relative", dO.rel(), BOUND_OMEGA);
        bound("nut, relative", dN.rel(), BOUND_NUT);
        // THE WALL nut, face by face. nutkRoughWallFunction carries history -- a limiter against the patch's
        // previous value -- so the patch is its own witness, and OpenFOAM writes it at every step
        {
            const FieldData<scalar> ofNut = readField<scalar>(last + "/nut");
            scalar dWall = 0;
            scalar wallMax = 0;
            int nRoughOf = 0;
            int nRoughBrae = 0;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                if (patches[pi].type != "wall") continue;
                const PatchFieldData<scalar>* b = findPatchEntry(ofNut.boundary, patches[pi]);
                if (!b) continue;
                nRoughOf += (b->type == "nutkRoughWallFunction") ? 1 : 0;
                nRoughBrae += fin.turbulence.nut.boundary[pi]->nutkRoughKs() ? 1 : 0;
                const std::size_t n = static_cast<std::size_t>(patches[pi].size);
                const std::vector<scalar> ofv = b->valueUniform ? std::vector<scalar>(n, b->uniformValue) : b->values;
                const std::vector<scalar>& bv = fin.turbulence.nut.boundary[pi]->value();
                failures += brae::gatecheck::nonFinite(("brae wall nut on " + patches[pi].name).c_str(), bv);
                for (std::size_t i = 0; i < n && i < ofv.size() && i < bv.size(); ++i)
                {
                    dWall = std::fmax(dWall, std::fabs(bv[i] - ofv[i]));
                    wallMax = std::fmax(wallMax, std::fabs(ofv[i]));
                }
            }
            std::printf("  wall nut: %d nutkRoughWallFunction patch(es) in OpenFOAM's file, %d in brae's\n",
                        nRoughOf, nRoughBrae);
            check("...brae built a rough wall wherever OpenFOAM has one", nRoughOf == nRoughBrae);
            bound("wall nut, face by face, relative to its largest", dWall/std::fmax(wallMax, scalar(1e-300)),
                  BOUND_NUT_WALL);
        }
        const std::vector<LinearSolveRecord> ofO = brae::gatecheck::readOfSolves(logPath, "omega");
        const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
        check("OpenFOAM's log gave the omega and k solves",
              ofO.size() >= static_cast<std::size_t>(nSteps) && ofK.size() >= static_cast<std::size_t>(nSteps));
        failures += brae::gatecheck::compareSolves("host", r.omegaSolves, ofO, nSteps, "omega",
                                                   scalar(1e-10), scalar(1e-10), scalar(1e-5), nullptr,
                                                   !measureOnly);
        failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                                   scalar(1e-10), scalar(1e-10), scalar(1e-5), nullptr,
                                                   !measureOnly);
    }

    // THE OUTLET's U, face by face, where OpenFOAM's file names outletPhaseMeanVelocity: its value is the
    // blend its updateCoeffs set, so the patch is its own witness
    {
        const FieldData<vector> ofU = readField<vector>(last + "/U");
        scalar dOut = 0;
        scalar outMax = 0;
        int nOf = 0;
        int nBrae = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const PatchFieldData<vector>* b = findPatchEntry(ofU.boundary, patches[pi]);
            if (!b || b->type != "outletPhaseMeanVelocity") continue;
            ++nOf;
            nBrae += fin.U.boundary[pi]->isOutletPhaseMeanVelocity() ? 1 : 0;
            const std::size_t n = static_cast<std::size_t>(patches[pi].size);
            const std::vector<vector> ofv = b->valueUniform ? std::vector<vector>(n, b->uniformValue) : b->values;
            const std::vector<vector>& bv = fin.U.boundary[pi]->value();
            failures += brae::gatecheck::nonFinite(("brae U on " + patches[pi].name).c_str(), bv);
            for (std::size_t i = 0; i < n && i < ofv.size() && i < bv.size(); ++i)
            {
                const vector e{bv[i].x - ofv[i].x, bv[i].y - ofv[i].y, bv[i].z - ofv[i].z};
                dOut = std::fmax(dOut, mag(e));
                outMax = std::fmax(outMax, mag(ofv[i]));
            }
        }
        if (nOf > 0)
        {
            check("brae built outletPhaseMeanVelocity wherever OpenFOAM's file has it", nOf == nBrae);
            bound("outlet U, face by face, relative to its largest", dOut/std::fmax(outMax, scalar(1e-300)),
                  BOUND_U_OUTLET);
        }
    }

    // the solves
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofA = brae::gatecheck::readOfSolves(logPath, fin.alphaName);
    check("OpenFOAM's log gave the p_rgh and alpha solves",
          !ofP.empty() && ofA.size() >= static_cast<std::size_t>(nSteps));
    // p_rgh's initial residuals over the run. `ras` runs the tutorial's own outletPhaseMeanVelocity, and
    // that profile is the more sensitive one: OpenFOAM against ITSELF with one interface cell's alpha moved
    // by one ulp reads 4.9e-11 there, and brae 3.0e-10, 6x its floor -- the same ratio its fields sit at
    // from the third step on, all of it growing from a U difference of 6e-13 at the FIRST step that the
    // laminar profile has too (the outlet itself is 5.8e-12 face by face). It was 2e-10 while the profile
    // staged the outlet as inletOutlet; laminar keeps that.
    const scalar pRunBound = fin.turbulence.on ? scalar(6e-10) : scalar(2e-10);
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh",
                                               scalar(1e-10), pRunBound, scalar(-1), nullptr,
                                               !measureOnly);
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5), nullptr,
                                               !measureOnly);

    std::printf("test_inter_dtchull_vs_openfoam: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
