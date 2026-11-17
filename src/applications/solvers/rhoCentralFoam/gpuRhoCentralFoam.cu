// brae_rhoCentralFoam -- OpenFOAM v2412's rhoCentralFoam (density-based, central-upwind), first stage.
//
//   brae_rhoCentralFoam [-case] <caseDir>
//
// Scope of this stage (everything else is refused at start-up by name, see readRhoCentralCase):
//   single GPU, static mesh, hePsiThermo/perfectGas/hConst/const, laminar AND inviscid (mu = 0), fixed deltaT,
//   Euler, fluxScheme Kurganov or Tadmor, reconstruct(rho|T) vanLeer + reconstruct(U) vanLeerV, Gauss linear
//   gradients, zeroGradient/fixedValue/empty patches on U, p and T.
//
// Two arms share the case, createFields, the loop and the writer; only the step moves:
//   BRAE_RHOCENTRALFOAM_ARM=cuda  (default) the device step, rhoCentralFoam_device.cu
//   BRAE_RHOCENTRALFOAM_ARM=cpu   the host reference, rhoCentralFoam_cpp.cu (a transcription of rhoCentralFoam.C)
//
// At the end it prints `diag` lines: the conservation balance (the change of the totals of V*rho, V*rhoU, V*rhoE
// against the time integral of what crossed the boundary), the positivity extrema, the step wall time and the
// device memory. tests/rho_central_sod_vs_openfoam.sh reads them.
#include "rhoCentralFoam_cpp.cuh"
#include "rhoCentralFoam_device.cuh"

#include "brae_time.cuh"
#include "foam_dict.cuh"
#include "io_precision.cuh"
#include "start_time.cuh"
#include "write_control.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

using namespace brae;
using namespace brae::rhoCentral;

namespace {

void printIntegrals(const char* tag, const Integrals& I)
{
    std::printf("diag %s mass %.17Le momentum %.17Le %.17Le %.17Le energy %.17Le\n", tag, I.mass, I.mom[0], I.mom[1],
                I.mom[2], I.energy);
    std::printf("diag %s min rho %.9e p %.9e T %.9e e %.9e max rho %.9e p %.9e T %.9e |U| %.9e\n", tag, I.minRho,
                I.minP, I.minT, I.minE, I.maxRho, I.maxP, I.maxT, I.maxMagU);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string caseDir = ".";
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "-case" && i + 1 < argc) caseDir = argv[++i];
            else if (!a.empty() && a[0] != '-') caseDir = a;
            else if (a == "-device") {}   // the launcher's hand-over flag; the device arm is the default
            else throw std::runtime_error("unknown option `" + a + "`");
        }
        setCaseEnvironment(caseDir);

        const char* armEnv = std::getenv("BRAE_RHOCENTRALFOAM_ARM");
        const std::string arm = armEnv ? armEnv : "cuda";
        if (arm != "cuda" && arm != "cpu")
            throw std::runtime_error("BRAE_RHOCENTRALFOAM_ARM is `" + arm + "`; it takes `cuda` (default) or `cpu`.");
        const bool onDevice = (arm == "cuda");

        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        setIOPrecision(controlDict.intOr("writePrecision", 6));
        Case cs = readRhoCentralCase(caseDir);

        Time time(caseDir, controlDict);
        WriteControl& wc = time.writeControl();
        time.setSteps(static_cast<int>(cs.nSteps));

        std::printf("brae rhoCentralFoam (%s arm): %ld cells, %ld internal faces, start %s, deltaT %g, %ld steps\n",
                    onDevice ? "cuda" : "host reference", (long)cs.nC, (long)cs.nIF, cs.startName.c_str(),
                    (double)cs.deltaT, cs.nSteps);
        std::printf("fluxScheme: %s\ninviscid: every mu is 0\nsolution directions: (%d %d %d)\n",
                    cs.tadmor ? "Tadmor" : "Kurganov", cs.sd.d[0], cs.sd.d[1], cs.sd.d[2]);

        State s = createFields(cs);
        const Integrals I0 = integrals(cs, s);
        printIntegrals("start", I0);

        int control = controlNone;
        if (const char* c = std::getenv("BRAE_CONTROL_RCF"))
        {
            const std::string v(c);
            if (v == "upwind") control = controlUpwind;
            else if (v == "noPressureJump") control = controlNoPressureJump;
            else if (!v.empty())
                throw std::runtime_error("BRAE_CONTROL_RCF is `" + v + "`; it takes upwind or noPressureJump.");
            if (control != controlNone && !onDevice)
                throw std::runtime_error("BRAE_CONTROL_RCF is a device-arm control; the host reference has none.");
            if (control != controlNone)
                std::fprintf(stderr, "brae rhoCentralFoam: NEGATIVE CONTROL `%s` -- deliberately wrong numerics\n", c);
        }

        std::unique_ptr<DeviceSolver> dev;
        if (onDevice)
        {
            dev = std::make_unique<DeviceSolver>(cs, s);
            dev->setControl(control);
            std::printf("diag device-memory %.3f MiB\n", dev->deviceBytes()/1048576.0);
        }

        scalar bndMass = 0, bndEnergy = 0;
        vector bndMom{0, 0, 0};
        double stepSeconds = 0;
        long steps = 0;
        std::printf("\nStarting time loop\n\n");
        // the log, OpenFOAM's two lines a step; the device arm's reports arrive in windows (DeviceSolver::flush)
        auto logStep = [&](const StepReport& rep, long index)
        {
            const std::string tn = WriteControl::timeName(wc.timeValue(static_cast<int>(index)));
            std::printf("Mean and max Courant Numbers = %.6g %.6g\nTime = %s\n\n", (double)rep.meanCoNum,
                        (double)rep.CoNum, tn.c_str());
            if (rep.status)
                throw std::runtime_error(
                    "step to time " + tn + " left an unphysical state (status " + std::to_string(rep.status)
                    + ": 1 negative temperature seed, 2 e->T inversion not converged in 100 iterations -- both "
                      "FatalErrors in OpenFOAM's thermo -- 4 rho, p or T not positive, or e not finite).");
        };
        long logged = 0;
        auto flushDevice = [&]()
        {
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<StepReport> reps = dev->flush();
            stepSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            for (const StepReport& r : reps) logStep(r, ++logged);
        };
        while (time.loop())
        {
            const auto t0 = std::chrono::steady_clock::now();
            StepReport rep;
            if (onDevice) dev->step();
            else          rep = stepCpp(cs, s);
            stepSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            ++steps;
            const bool write = time.writeTime();
            if (!onDevice)
            {
                bndMass += cs.deltaT*rep.bndMass;
                bndMom += cs.deltaT*rep.bndMom;
                bndEnergy += cs.deltaT*rep.bndEnergy;
                logStep(rep, ++logged);
            }
            else if (write || dev->pending() == DeviceSolver::kReportWindow || steps == cs.nSteps)
            {
                flushDevice();
            }
            if (write)
            {
                const std::string tname = WriteControl::timeName(wc.timeValue(time.timeIndex()));
                if (onDevice) dev->download(s);
                writeTime(cs, s, tname);
                std::printf("written %s\n", tname.c_str());
            }
        }
        if (onDevice) flushDevice();
        time.end();
        const std::string endName = WriteControl::timeName(wc.timeValue(static_cast<int>(steps)));
        if (onDevice)
        {
            dev->download(s);
            dev->boundaryIntegrals(bndMass, bndMom, bndEnergy);
        }
        writeTime(cs, s, endName);
        std::printf("written %s\n", endName.c_str());

        // ---- diagnostics ------------------------------------------------------------------------------------------
        const Integrals I1 = integrals(cs, s);
        printIntegrals("end", I1);
        // conservation: total(end) - total(start) + integral of the boundary outflow; zero to rounding for a
        // conservative scheme, whatever the boundary does
        const long double eM = (I1.mass - I0.mass) + bndMass;
        const long double eE = (I1.energy - I0.energy) + bndEnergy;
        const long double eU[3] = {(I1.mom[0] - I0.mom[0]) + bndMom.x, (I1.mom[1] - I0.mom[1]) + bndMom.y,
                                   (I1.mom[2] - I0.mom[2]) + bndMom.z};
        std::printf("diag boundary-outflow mass %.17e momentum %.17e %.17e %.17e energy %.17e\n", (double)bndMass,
                    (double)bndMom.x, (double)bndMom.y, (double)bndMom.z, (double)bndEnergy);
        std::printf("diag balance mass %.6Le energy %.6Le momentum %.6Le %.6Le %.6Le "
                    "(relative to the start totals: mass %.3Le energy %.3Le)\n",
                    eM, eE, eU[0], eU[1], eU[2], fabsl(eM)/fabsl(I0.mass), fabsl(eE)/fabsl(I0.energy));
        std::printf("diag steps %ld step-seconds %.6f per-step-us %.3f cell-steps-per-second %.6e\n", steps,
                    stepSeconds, 1e6*stepSeconds/(steps ? steps : 1),
                    stepSeconds > 0 ? (double)cs.nC*steps/stepSeconds : 0.0);
        std::printf("End\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "\nbrae rhoCentralFoam: ERROR: %s\n", e.what());
        return 1;
    }
}
