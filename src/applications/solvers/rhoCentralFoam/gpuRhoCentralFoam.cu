// brae_rhoCentralFoam -- OpenFOAM v2412's rhoCentralFoam (density-based, central-upwind). EXPERIMENTAL.
//
//   brae_rhoCentralFoam [-case] <caseDir>
//
// The solver's scope and its refusals: rhoCentralControls.cuh. Its layout: ARCHITECTURE.md. Two arms share the
// case, createFields, the loop and the writer (rhoCentralFoam.cu); only the step differs:
//   BRAE_RHOCENTRALFOAM_ARM=cuda  (default) the device arm, rhoCentralStep.cu
//   BRAE_RHOCENTRALFOAM_ARM=cpu   the host reference, rhoCentralStep_cpp.cu
// BRAE_CONTROL_RCF selects a gate's negative control (rhoCentralState.cuh), device arm only.
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
            else if (!v.empty())
                throw std::runtime_error("BRAE_CONTROL_RCF is `" + v + "`; it takes upwind or noPressureJump.");
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
