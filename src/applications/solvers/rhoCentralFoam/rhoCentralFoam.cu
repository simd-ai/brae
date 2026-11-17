// rhoCentralFoam.cu -- the driver: rhoCentralFoam.C's main() for both arms. See the header.
#include "rhoCentralFoam.cuh"
#include "rhoCentralCreateFields_cpp.cuh"
#include "rhoCentralStep.cuh"
#include "rhoCentralStep_cpp.cuh"

#include "brae_time.cuh"
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

    Time time(caseDir, controlDict);
    WriteControl& wc = time.writeControl();
    time.setSteps(static_cast<int>(cs.nSteps));

    std::printf("brae rhoCentralFoam (%s arm): %ld cells, %ld internal faces, start %s, deltaT %g, %ld steps\n",
                onDevice ? "cuda" : "host reference", (long)cs.nC, (long)cs.nIF, cs.startName.c_str(),
                (double)cs.deltaT, cs.nSteps);
    std::printf("fluxScheme: %s\ninviscid: every mu is 0\nsolution directions: (%d %d %d)\n",
                cs.tadmor ? "Tadmor" : "Kurganov", cs.sd.d[0], cs.sd.d[1], cs.sd.d[2]);

    PatchFields pf;
    State s = cpu::rhoCentral::createFields(cs, pf);
    const Integrals I0 = integrals(cs, s);
    printIntegrals("start", I0);

    if (control != controlNone && !onDevice)
        throw std::runtime_error("BRAE_CONTROL_RCF is a device-arm control; the host reference has none.");

    std::unique_ptr<Arm> arm;
    if (onDevice)
    {
        arm = gpu::rhoCentral::makeDeviceArm(cs, s, pf, control);
        std::printf("diag device-memory %.3f MiB\n", arm->deviceBytes()/1048576.0);
    }
    else
    {
        arm = cpu::rhoCentral::makeHostArm(cs, s, pf);
    }

    double stepSeconds = 0;
    long steps = 0;
    long logged = 0;
    std::printf("\nStarting time loop\n\n");
    // the log, OpenFOAM's two lines a step; reports arrive in the arm's windows (one step on the host)
    auto flushArm = [&]()
    {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<StepReport> reps = arm->flush();
        stepSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (const StepReport& rep : reps)
        {
            const std::string tn = WriteControl::timeName(wc.timeValue(static_cast<int>(++logged)));
            std::printf("Mean and max Courant Numbers = %.6g %.6g\nTime = %s\n\n", (double)rep.meanCoNum,
                        (double)rep.CoNum, tn.c_str());
            if (rep.status)
                throw std::runtime_error(
                    "step to time " + tn + " left an unphysical state (status " + std::to_string(rep.status)
                    + ": 1 negative temperature seed, 2 e->T inversion not converged in 100 iterations -- both "
                      "FatalErrors in OpenFOAM's thermo -- 4 rho, p or T not positive, or e not finite).");
        }
    };
    while (time.loop())
    {
        const auto t0 = std::chrono::steady_clock::now();
        arm->step();
        stepSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++steps;
        const bool write = time.writeTime();
        if (write || arm->pending() >= arm->window() || steps == cs.nSteps) flushArm();
        if (write)
        {
            const std::string tname = WriteControl::timeName(wc.timeValue(time.timeIndex()));
            arm->download(s);
            writeTime(cs, s, tname);
            std::printf("written %s\n", tname.c_str());
        }
    }
    flushArm();
    time.end();
    const std::string endName = WriteControl::timeName(wc.timeValue(static_cast<int>(steps)));
    arm->download(s);
    scalar bndMass = 0, bndEnergy = 0;
    vector bndMom{0, 0, 0};
    arm->boundaryOutflow(bndMass, bndMom, bndEnergy);
    writeTime(cs, s, endName);
    std::printf("written %s\n", endName.c_str());

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
