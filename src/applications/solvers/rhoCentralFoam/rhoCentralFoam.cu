// rhoCentralFoam.cu -- the driver: rhoCentralFoam.C's main() for both arms. See the header.
#include "rhoCentralFoam.cuh"
#include "rhoCentralCreateFields_cpp.cuh"
#include "rhoCentralStep.cuh"
#include "rhoCentralStep_cpp.cuh"

#include "brae_time.cuh"
#include "run_clock.cuh"
#include "time_controls.cuh"
#include "foam_dict.cuh"
#include "foam_field_writer.cuh"
#include "io_precision.cuh"
#include "write_control.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace brae {
namespace rhoCentral {

namespace {

void printIntegrals(const char* tag, const Integrals& I)
{
    std::printf("diag %s mass %.17Le momentum %.17Le %.17Le %.17Le energy %.17Le\n", tag, I.mass, I.mom[0], I.mom[1],
                I.mom[2], I.energy);
    std::printf("diag %s min rho %.9e p %.9e T %.9e e %.9e max rho %.9e p %.9e T %.9e |U| %.9e\n", tag, I.minRho,
                I.minP, I.minT, I.minE, I.maxRho, I.maxP, I.maxT, I.maxMagU);
}

} // namespace


Integrals integrals(const Case& cs, const State& s)
{
    const std::vector<scalar>& V = cs.g.V();
    Integrals r;
    r.minRho = r.minP = r.minT = r.minE = 1e300;
    r.maxRho = r.maxP = r.maxT = r.maxMagU = -1e300;
    for (label i = 0; i < cs.nC; ++i)
    {
        r.mass += static_cast<long double>(V[i])*s.rho[i];
        r.energy += static_cast<long double>(V[i])*s.rhoE[i];
        r.mom[0] += static_cast<long double>(V[i])*s.rhoU[i].x;
        r.mom[1] += static_cast<long double>(V[i])*s.rhoU[i].y;
        r.mom[2] += static_cast<long double>(V[i])*s.rhoU[i].z;
        r.minRho = std::min(r.minRho, s.rho[i]);
        r.maxRho = std::max(r.maxRho, s.rho[i]);
        r.minP = std::min(r.minP, s.p[i]);
        r.maxP = std::max(r.maxP, s.p[i]);
        r.minT = std::min(r.minT, s.T[i]);
        r.maxT = std::max(r.maxT, s.T[i]);
        r.minE = std::min(r.minE, s.e[i]);
        r.maxMagU = std::max(r.maxMagU, mag(s.U[i]));
    }
    return r;
}


void writeTime(const Case& cs, const State& s, const std::string& timeName)
{
    const std::string src = cs.caseDir + "/" + cs.startName + "/";
    const std::string out = cs.caseDir + "/" + timeName;
    std::filesystem::create_directories(out);
    const int prec = cs.writePrecision;
    // every patch value, in patch order (writeVolField's flat layout; no coupled patch is accepted here)
    writeVolField(src + "U", out + "/U", s.U, cs.patches, prec, s.UB);
    writeVolField(src + "p", out + "/p", s.p, cs.patches, prec, s.pB);
    writeVolField(src + "T", out + "/T", s.T, cs.patches, prec, s.TB);
    // rho is NO_READ, AUTO_WRITE with calculated patches (createFields.H:24-35)
    static const DerivedFieldSpec rhoSpec{"rho", "dimensions      [1 -3 0 0 0 0 0];"};
    writeVolField(src + "T", out + "/rho", s.rho, cs.patches, prec, s.rhoB, &rhoSpec);
}


int runRhoCentralFoam(const std::string& caseDir, bool onDevice, int control)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    setIOPrecision(controlDict.intOr("writePrecision", 6));
    const Case cs = readRhoCentralCase(caseDir);
    // brae's Time reads controlDict's `functions` and says which it does not run (none is registered here); the
    // loop runs on OpenFOAM's accumulating clock instead, whose deltaT their write times still trim (WriteCadence)
    Time time(caseDir, controlDict);
    RunClock clock = *cs.clock;
    const TimeControls& tc = cs.timeControls;
    // Info's precision is IOstream::defaultPrecision(), which controlDict's writePrecision sets (TimeIO.C:377-389)
    const int logPrec = cs.writePrecision;

    std::printf("brae rhoCentralFoam (%s arm): %ld cells, %ld internal faces, start %s, endTime %.*g, deltaT %.*g, "
                "adjustTimeStep %s",
                onDevice ? "cuda" : "host reference", (long)cs.nC, (long)cs.nIF, cs.startName.c_str(), logPrec,
                (double)clock.endTime(), logPrec, (double)clock.deltaT(), tc.adjustTimeStep ? "yes" : "no");
    if (tc.adjustTimeStep) std::printf(" (maxCo %.*g, maxDeltaT %.*g)", logPrec, (double)tc.maxCo, logPrec,
                                       (double)tc.maxDeltaT);
    std::printf("\nfluxScheme: %s\ninviscid: every mu is 0\nsolution directions: (%d %d %d)\n",
                cs.tadmor ? "Tadmor" : "Kurganov", cs.sd.d[0], cs.sd.d[1], cs.sd.d[2]);

    PatchFields pf;
    State s = cpu::rhoCentral::createFields(cs, pf);
    const Integrals I0 = integrals(cs, s);
    printIntegrals("start", I0);

    const bool clockControl = control == controlUntrimmed || control == controlLaggedCourant;
    if (control != controlNone && !clockControl && !onDevice)
        throw std::runtime_error("BRAE_CONTROL_RCF upwind and noPressureJump are device-arm controls; the host "
                                 "reference has none.");
    if (clockControl && !tc.adjustTimeStep)
        throw std::runtime_error("BRAE_CONTROL_RCF untrimmed and laggedCourant are controls of adjustTimeStep; this "
                                 "case runs a fixed deltaT.");

    std::unique_ptr<Arm> arm;
    if (onDevice)
    {
        arm = gpu::rhoCentral::makeDeviceArm(cs, s, pf, clockControl ? controlNone : control);
        std::printf("diag device-memory %.3f MiB\n", arm->deviceBytes()/1048576.0);
    }
    else
    {
        arm = cpu::rhoCentral::makeHostArm(cs, s, pf);
    }

    // rhoCentralFoam.C:80-81: CoNum is 0 until the first step's centralCourantNo.H, so the first setDeltaT.H grows
    // deltaT by the full 1.2
    scalar CoNum = 0;
    double stepSeconds = 0;
    long steps = 0;
    // the log of the steps the arm has not reported yet: their deltaT and time name (the device arm reports a window
    // of steps at once at a fixed deltaT)
    std::vector<std::pair<scalar, std::string>> unlogged;
    std::printf("\nStarting time loop\n\n");
    // the log, OpenFOAM's lines a step: setDeltaT.H's "deltaT = " under adjustTimeStep, centralCourantNo.H's, Time
    auto flushArm = [&]()
    {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<StepReport> reps = arm->flush();
        stepSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (reps.size() != unlogged.size())
            throw std::logic_error("brae rhoCentralFoam: the arm reported a different number of steps than it ran.");
        for (std::size_t i = 0; i < reps.size(); ++i)
        {
            const StepReport& rep = reps[i];
            const std::string& tn = unlogged[i].second;
            if (tc.adjustTimeStep) std::printf("deltaT = %.*g\n", logPrec, (double)unlogged[i].first);
            std::printf("Mean and max Courant Numbers = %.*g %.*g\nTime = %s\n\n", logPrec, (double)rep.meanCoNum,
                        logPrec, (double)rep.CoNum, tn.c_str());
            if (rep.status)
                throw std::runtime_error(
                    "step to time " + tn + " left an unphysical state (status " + std::to_string(rep.status)
                    + ": 1 negative temperature seed, 2 e->T inversion not converged in 100 iterations -- both "
                      "FatalErrors in OpenFOAM's thermo -- 4 rho, p or T not positive, or e not finite).");
            CoNum = rep.CoNum;
            // the control: the Courant number as it would be at the step before's deltaT
            if (control == controlLaggedCourant) CoNum = rep.CoNum/unlogged[i].first*clock.deltaT0();
        }
        unlogged.clear();
    };
    // rhoCentralFoam.C:85-97 and 288-294, !LTS: Time::run, setDeltaT.H, ++runTime, the step, runTime.write()
    while (clock.run())
    {
        if (tc.adjustTimeStep)
        {
            const scalar dt = setDeltaT(clock.deltaT(), CoNum, tc);   // setDeltaT.H
            if (control == controlUntrimmed) clock.setDeltaTNoAdjust(dt);
            else clock.setDeltaT(dt);                                  // Time::setDeltaT: adjustDeltaT
        }
        ++clock;
        unlogged.emplace_back(clock.deltaT(), clock.timeName());
        const auto t0 = std::chrono::steady_clock::now();
        arm->step(clock.deltaT());
        stepSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++steps;
        const bool write = clock.writeTime();
        if (tc.adjustTimeStep || write || arm->pending() >= arm->window() || !clock.run()) flushArm();
        if (write)
        {
            arm->download(s);
            writeTime(cs, s, clock.timeName());
            std::printf("written %s\n", clock.timeName().c_str());
        }
    }
    flushArm();
    time.end();
    // no write here: rhoCentralFoam.C writes only through runTime.write(), on write steps
    arm->download(s);
    scalar bndMass = 0, bndEnergy = 0;
    vector bndMom{0, 0, 0};
    arm->boundaryOutflow(bndMass, bndMom, bndEnergy);
    std::printf("diag end time %s value %.17g steps %ld\n", clock.timeName().c_str(), (double)clock.value(), steps);

    // ---- diagnostics ------------------------------------------------------------------------------------------------
    const Integrals I1 = integrals(cs, s);
    printIntegrals("end", I1);
    // conservation: total(end) - total(start) + integral of the boundary outflow; zero to rounding for a conservative
    // scheme, whatever the boundary does
    const long double eM = (I1.mass - I0.mass) + bndMass;
    const long double eE = (I1.energy - I0.energy) + bndEnergy;
    const long double eU[3] = {(I1.mom[0] - I0.mom[0]) + bndMom.x, (I1.mom[1] - I0.mom[1]) + bndMom.y,
                               (I1.mom[2] - I0.mom[2]) + bndMom.z};
    std::printf("diag boundary-outflow mass %.17e momentum %.17e %.17e %.17e energy %.17e\n", (double)bndMass,
                (double)bndMom.x, (double)bndMom.y, (double)bndMom.z, (double)bndEnergy);
    std::printf("diag balance mass %.6Le energy %.6Le momentum %.6Le %.6Le %.6Le "
                "(relative to the start totals: mass %.3Le energy %.3Le)\n",
                eM, eE, eU[0], eU[1], eU[2], fabsl(eM)/fabsl(I0.mass), fabsl(eE)/fabsl(I0.energy));
    std::printf("diag steps %ld step-seconds %.6f per-step-us %.3f cell-steps-per-second %.6e\n", steps, stepSeconds,
                1e6*stepSeconds/(steps ? steps : 1), stepSeconds > 0 ? (double)cs.nC*steps/stepSeconds : 0.0);
    std::printf("End\n");
    return 0;
}

} // namespace rhoCentral
} // namespace brae
