// brae_rhoCentralFoam -- OpenFOAM v2412's rhoCentralFoam (density-based, central-upwind). EXPERIMENTAL.
//
//   brae_rhoCentralFoam [-case] <caseDir>
//
// The solver's scope and its refusals: rhoCentralControls.cuh. Its layout: ARCHITECTURE.md. Two arms share the
// case, createFields, the loop and the writer (rhoCentralFoam.cu); only the step differs:
//   BRAE_RHOCENTRALFOAM_ARM=cuda  (default) the device arm, rhoCentralStep.cu
//   BRAE_RHOCENTRALFOAM_ARM=cpu   the host reference, rhoCentralStep_cpp.cu
// BRAE_CONTROL_RCF selects a gate's negative control (rhoCentralState.cuh).
//
// At the end it prints `diag` lines: the conservation balance, the positivity extrema, the step wall time and the
// device memory. tests/rho_central_sod_vs_openfoam.sh reads them.
#include "rhoCentralFoam.cuh"

#include "start_time.cuh"   // setCaseEnvironment

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    using namespace brae::rhoCentral;
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
        brae::setCaseEnvironment(caseDir);
        // The solver is labelled experimental (ARCHITECTURE.md sec. 6). On stderr, so the log keeps OpenFOAM's layout.
        std::fprintf(stderr, "brae rhoCentralFoam: EXPERIMENTAL, single GPU. What is validated against OpenFOAM v2412 "
                             "and what is refused by name: src/applications/solvers/rhoCentralFoam/ARCHITECTURE.md "
                             "sec. 6 and rhoCentralControls.cuh.\n");

        const char* armEnv = std::getenv("BRAE_RHOCENTRALFOAM_ARM");
        const std::string arm = armEnv ? armEnv : "cuda";
        if (arm != "cuda" && arm != "cpu")
            throw std::runtime_error("BRAE_RHOCENTRALFOAM_ARM is `" + arm + "`; it takes `cuda` (default) or `cpu`.");

        int control = controlNone;
        if (const char* c = std::getenv("BRAE_CONTROL_RCF"))
        {
            const std::string v(c);
            if (v == "upwind") control = controlUpwind;
            else if (v == "noPressureJump") control = controlNoPressureJump;
            else if (v == "untrimmed") control = controlUntrimmed;
            else if (v == "laggedCourant") control = controlLaggedCourant;
            else if (v == "noTransform") control = controlNoTransform;
            else if (v == "laggedRhoB") control = controlLaggedRhoB;
            else if (v == "inletOutletInverted") control = controlInletOutletInverted;
            else if (v == "laggedPhi") control = controlLaggedPhi;
            else if (v == "inletOutletNewT") control = controlInletOutletNewT;
            else if (v == "noPurge") control = controlNoPurge;
            else if (v == "restartIndexZero") control = controlRestartIndexZero;
            else if (v == "restartGridFromZero") control = controlRestartGridFromZero;
            else if (v == "restartIoExtrapolate") control = controlRestartIoExtrapolate;
            else if (v == "noUniformTime") control = controlNoUniformTime;
            else if (v == "limiterVanLeer") control = controlLimiterVanLeer;
            else if (v == "limiterScalarOnV") control = controlLimiterScalarOnV;
            else if (v == "noTauMC") control = controlNoTauMC;
            else if (v == "noSigmaDotU") control = controlNoSigmaDotU;
            else if (v == "noDynamicHead") control = controlNoDynamicHead;
            else if (v == "nozzleFlowInverted") control = controlNozzleFlowInverted;
            else if (v == "waveNoRelax") control = controlWaveNoRelax;
            else if (v == "waveVolumeFlux") control = controlWaveVolumeFlux;
            else if (v == "totalTemperatureOnce") control = controlTotalTemperatureOnce;
            else if (v == "residualUnfused") control = controlResidualUnfused;
            else if (v == "ltsUnsmoothed") control = controlLtsUnsmoothed;
            else if (v == "janafLowOnly") control = controlJanafLowOnly;
            else if (v == "alphaheOldT") control = controlAlphaheOldT;
            else if (v == "maxwellNoCreep") control = controlMaxwellNoCreep;
            else if (v == "maxwellNoCurvature") control = controlMaxwellNoCurvature;
            else if (v == "smoluchowskiNoJump") control = controlSmoluchowskiNoJump;
            else if (v == "rasLaminarMuEff") control = controlRasLaminarMuEff;
            else if (v == "rasNoAlphat") control = controlRasNoAlphat;
            else if (v == "rasCorrectFirst") control = controlRasCorrectFirst;
            else if (v == "rasPhiUndivided") control = controlRasPhiUndivided;
            else if (v == "motionFluxAbsolute") control = controlMotionFluxAbsolute;
            else if (v == "motionDdtStatic") control = controlMotionDdtStatic;
            else if (v == "motionWallStatic") control = controlMotionWallStatic;
            else if (v == "motionPiovStatic") control = controlMotionPiovStatic;
            else if (!v.empty())
                throw std::runtime_error("BRAE_CONTROL_RCF is `" + v + "`; it takes upwind, noPressureJump, untrimmed, "
                                         "laggedCourant, noTransform, laggedRhoB, inletOutletInverted, laggedPhi, "
                                         "inletOutletNewT, noPurge, restartIndexZero, restartGridFromZero, "
                                         "restartIoExtrapolate, noUniformTime, limiterVanLeer, limiterScalarOnV, noTauMC, "
                                         "noSigmaDotU, noDynamicHead, nozzleFlowInverted, waveNoRelax, waveVolumeFlux, "
                                         "totalTemperatureOnce, residualUnfused, ltsUnsmoothed, janafLowOnly, alphaheOldT, "
                                         "maxwellNoCreep, maxwellNoCurvature, smoluchowskiNoJump, rasLaminarMuEff, "
                                         "rasNoAlphat, rasCorrectFirst, rasPhiUndivided, motionFluxAbsolute, motionDdtStatic, "
                                         "motionWallStatic or motionPiovStatic.");
            if (control != controlNone)
                std::fprintf(stderr, "brae rhoCentralFoam: NEGATIVE CONTROL `%s` -- deliberately wrong numerics\n", c);
        }
        return runRhoCentralFoam(caseDir, arm == "cuda", control);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "\nbrae rhoCentralFoam: ERROR: %s\n", e.what());
        return 1;
    }
}
