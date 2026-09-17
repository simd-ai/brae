// brae_interFoam -- the OF-mirror interFoam as a runnable solver.
//
// Everything under this directory was, until now, reachable only from tests/: the components are gated
// individually and the whole solver is gated against real OpenFOAM on damBreak (alpha 3.4e-09) and
// capillaryRise (0.7%), and no user could run it. `brae -case <dir>` on an `application interFoam`
// case reported that brae has no such solver.
//
// The case-to-fields translation and the time loop are the SAME ones the gates call --
// buildInterFields and runInterFoam -- so what ships is what is measured. A private copy here would be
// the defect this project keeps finding one level up: the gate proves the step, the driver feeds it
// something else, and nothing compares the two.
//
// WHAT IT RUNS TURBULENT: RAS kEpsilon, host and `-device`, in both of interFoam's lineages -- the
// ordinary single-phase model, and under `density variable` the rho-weighted one
// (inter_turbulence_cpp.cuh; device_inter_turbulence.cuh for what the device closure is handed).
// An earlier version of this header said interFoam's turbulence "is a MIXTURE model, not the
// single-phase one brae has". That was written from memory and is wrong for 15 of the 17 turbulent
// tutorials: incompressibleInterPhaseTransportModel.C:99-106 constructs the ordinary one by default.
//
// WHAT IT WILL NOT RUN. Every refusal the components carry is in force: LES and every RASModel but
// kEpsilon, `vanLeerV` or `limitedLinear` on div(rhoPhi,U),
// `interfaceCompression` on div(phirb,alpha), any ddtSchemes default but Euler, a `corrected` or
// `limited` laplacian or snGrad on a mesh that is not orthogonal, MRF, fvOptions, and a case that
// omits nAlphaCorr, nAlphaSubCycles, cAlpha, maxAlphaCo or -- under MULESCorr -- nLimiterIter. Each
// throws by name.
//
// THAT LIST WAS NOT TRUE when it was written: MRF and fvOptions were named here and refused nowhere,
// and a moving or refining mesh was not even named. brae was run over all 44 shipped tutorials and the
// ones that reached `End:` were counted -- two did that should not have, both on `dynamicRefineFvMesh`.
// Also refused now: any `dynamicFvMesh` but staticFvMesh (19 tutorials), a dictionary-form `sigma` and a
// missing one (both used to become ZERO surface tension), a setTimeStep function object, and on
// `-device` nOuterCorrectors above 1 and nNonOrthogonalCorrectors above 0, which that loop does not run.
// tests/interfoam_refusals.sh holds every one of them, each beside the form OpenFOAM treats as nothing
// -- `staticFvMesh`, `active no` -- which must still RUN.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "inter_driver_cpp.cuh"
#include "foam_dict.cuh"
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

int main(int argc, char** argv)
{
    bool onDevice = false;
    std::string caseDir = ".";
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-case") == 0 && i + 1 < argc) caseDir = argv[++i];
        else if (std::strcmp(argv[i], "-device") == 0) onDevice = true;
        else if (std::strcmp(argv[i], "-help") == 0)
        {
            std::printf("brae_interFoam: OpenFOAM's interFoam, re-ported.\n"
                        "  usage: brae_interFoam -case <dir> [-device]\n"
                        "    -device  run the time loop on the GPU (runInterFoamDevice). The same\n"
                        "             components and the same case translation; the boundary\n"
                        "             conditions stay on the host. Gated against the host loop on\n"
                        "             damBreak's own mesh: alpha 7.3e-11, U 2.6e-09 relative,\n"
                        "             p_rgh 6.9e-11 over five steps.\n");
            return 0;
        }
    }

    try
    {
        using namespace brae;
        using namespace brae::cpu::interFoam;

        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        const scalar endTime  = controlDict.scalarOr("endTime",  scalar(0));
        const scalar deltaT0  = controlDict.scalarOr("deltaT",   scalar(1e-3));
        const std::string startFrom = controlDict.wordOr("startFrom", "startTime");
        const scalar startTime = (startFrom == "startTime")
                               ? controlDict.scalarOr("startTime", scalar(0)) : scalar(0);

        PrimitiveMesh m;
        m.read(caseDir + "/constant/polyMesh");
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> patches = buildPatches(m, g);

        // The start directory OpenFOAM would use. `0` is written as `0` and not `0.000000`, which is
        // what every tutorial ships.
        char buf[64];
        std::snprintf(buf, sizeof buf, "%g", (double)startTime);
        const std::string startDir = caseDir + "/" + buf;

        // An upper bound on the steps, and endTime below is the REAL bound. A fixed-step case needs
        // exactly (end-start)/dt; an adjustTimeStep case cannot be counted ahead of time, because
        // deltaT is chosen from a Courant number that does not exist yet. This count used to be the
        // only bound and the comment here claimed otherwise: damBreak at `endTime 0.004` ran its 40
        // steps out to t = 0.054, thirteen times past the end of the case. The 4x allows the step to
        // grow at setDeltaT's 1.2 cap for eight steps before endTime has to stop the loop.
        const label nSteps = (deltaT0 > scalar(0))
            ? static_cast<label>(scalar(4)*(endTime - startTime) / deltaT0 + scalar(0.5))
            : label(0);
        if (nSteps < 1)
            throw std::runtime_error(
                "brae interFoam: controlDict gives no steps to take (endTime " +
                std::to_string((double)endTime) + ", deltaT " + std::to_string((double)deltaT0) + ").");

        std::printf("brae interFoam (OF-mirror): %ld cells, start %s, endTime %g\n",
                    (long)m.nCells(), buf, (double)endTime);

        // ONE case translation and ONE time loop per path, both the ones the gates call. A private
        // copy here would be the defect this file's header names.
        const RunReport r = onDevice
            ? runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true,
                                 /*fieldsOut=*/nullptr, endTime)
            : runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true,
                           /*fieldsOut=*/nullptr, endTime);

        std::printf("End: t = %.6g, alpha in [%.3e, %.8f], max|U| %.4g m/s, worst |div(phi)| %.3e\n",
                    (double)r.time, (double)r.alphaMin, (double)r.alphaMax,
                    (double)r.maxU, (double)r.worstDivPhi);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "\nbrae interFoam: %s\n\n", e.what());
        return 1;
    }
}
