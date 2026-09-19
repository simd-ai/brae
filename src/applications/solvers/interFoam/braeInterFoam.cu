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
// And RAS kOmegaSST, on the HOST and in the ordinary lineage only: RAS/waterChannel, gated in
// tests/interfoam_waterchannel_vs_openfoam.sh. `-device` refuses it by name.
// An earlier version of this header said interFoam's turbulence "is a MIXTURE model, not the
// single-phase one brae has". That was written from memory and is wrong for 15 of the 17 turbulent
// tutorials: incompressibleInterPhaseTransportModel.C:99-106 constructs the ordinary one by default.
//
// AND WAVES, host and `-device`: waveAlpha and waveVelocity over all ten of OpenFOAM's wave models
// (inter_waves_cpp.cuh, src/waveModels). Each flux-conditional condition is handed the flux its own
// `phi` entry NAMES -- phi or rhoPhi; on `-device` a VELOCITY condition naming rhoPhi is refused,
// because that one switch runs on the device and reads phi.
//
// AND p_rgh's LINEAR SOLVER AS THE CASE NAMES IT, host and `-device`: PCG with DIC, or GAMG --
// OpenFOAM's own faceAreaPair hierarchy, smoother and V-cycle (src/matrices/lduMatrix/solvers/GAMG),
// which eight of the nine wave tutorials name for p_rghFinal -- and, on the host, PCG with a GAMG
// PRECONDITIONER, which six of the seven solid-body tutorials name. Every GAMG control whose branch is not
// ported is refused by name: mergeLevels above 1, another agglomerator, updateInterval,
// cacheAgglomeration no, interpolateCorrection, directSolveCoarsest, a coarsestLevelCorr dictionary,
// a processorAgglomerator, and any smoother but DIC, DICGaussSeidel, GaussSeidel and symGaussSeidel
// -- on `-device`, any smoother but DIC. Any OTHER solver for p_rgh still substitutes, under a notice.
//
// AND A MOVING MESH, on the host: dynamicMotionSolverFvMesh with the solidBody solver moving the
// whole mesh under any of OpenFOAM's motion functions but drivenLinearMotion
// (src/dynamicFvMesh, src/meshTools/solidBodyMotionFunctions), or with displacementLaplacian DEFORMING
// it from waveMaker paddles (src/fvMotionSolver, src/waveModels/derivedPointPatchFields), with
// movingWallVelocity walls, the relative flux, Uf and the old volumes where interFoam.C and pEqn.H
// read them; CorrectPhi after every update under `correctPhi` (the default on a moving mesh) and at
// the start of EVERY case (initCorrectPhi.H); a wave absorber on a moving mesh; and a CLOSED tank's
// pressure reference (pRefCell or pRefPoint, pRefValue, adjustPhi). `-device` refuses a moving mesh
// and a closed tank. Still refused by name: a cellZone or cellSet, every other motionSolver, a
// turbulent case on a moving mesh, points0, and a restart of a moved mesh. testTubeMixer, the five
// sloshing tanks and the five waveMakers run -- waveMakerPiston and waveMakerFlap gated with their
// pressure solves converged, and the two multi-paddle ones approximating p_rgh under a notice, since
// their `p_rgh { $pcorr; }` names a pattern-keyed entry brae's dictionary expansion does not resolve.
//
// AND `Gauss interfaceCompression` on the alpha fluxes, on the host: the PhiScheme four waveMakers name
// for div(phirb,alpha). `-device` refuses it.
//
// AND THE CASE'S NON-ORTHOGONAL CORRECTIONS, on the host: `corrected` and `limited` laplacians and
// snGrads on a mesh that is not orthogonal (the tanks' 44 degrees), through the pressure equation,
// the viscous term and the three snGrads. `uncorrected` on such a mesh is refused, and so is every
// gradSchemes entry but `Gauss linear` -- gradSchemes were not read at all before. `-device` refuses a
// correction that is not zero.
//
// AND div(rhoPhi,U) AS THE CASE NAMES IT, on both paths: upwind, linear, linearUpwind, linearUpwindV,
// limitedLinearV, LUST and vanLeerV -- the last the V-limited vanLeer the closed-tank tutorials use.
// `Gauss linear` on the device fell through a switch's default to upwind until this was wired.
//
// AND A CYCLIC PAIR, on the host: a translational `cyclic` -- a baffle pair included -- coupled through
// every operator, matrix and linear solver on the path (PCG/DIC and smoothSolver carry the interface
// coefficients), and on it p_rgh's porousBafflePressure, the cyclic with a jump rebuilt at every
// pressure assembly. tests/interfoam_baffle_vs_openfoam.sh holds RAS/damBreakPorousBaffle against
// OpenFOAM. Refused by name across a cyclic: GAMG and PBiCGStab, a momentum predictor, every
// div(rhoPhi,U) scheme but upwind and linearUpwind, interfaceCompression, a moving mesh, MRF, fvOptions,
// waves, kOmegaSST, a pair that is rotational or not orthogonal, and cyclicAMI/ACMI/processor patches.
// `-device` refuses any cyclic, naming the patch.
//
// WHAT IT WILL NOT RUN. Every refusal the components carry is in force: LES and every RASModel but
// kEpsilon and kOmegaSST, `limitedLinear` on div(rhoPhi,U),
// any ddtSchemes default but Euler, every fvOption but explicitPorositySource/DarcyForchheimer (host
// only: tests/interfoam_angledduct_vs_openfoam.sh), MRF on the device or beside a moving mesh, RAS or a
// fixedFluxPressure patch (the host runs it otherwise: tests/interfoam_mrf_vs_openfoam.sh), and a case that
// omits nAlphaCorr, nAlphaSubCycles, cAlpha, maxAlphaCo or -- under MULESCorr -- nLimiterIter. Each
// throws by name.
//
// THAT LIST WAS NOT TRUE when it was written: MRF and fvOptions were named here and refused nowhere,
// and a moving or refining mesh was not even named. brae was run over all 44 shipped tutorials and the
// ones that reached `End:` were counted -- two did that should not have, both on `dynamicRefineFvMesh`.
// Also refused now: any `dynamicFvMesh` but staticFvMesh and the solid-body motion above, a dictionary-form `sigma` and a
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
        std::vector<FvPatch> patches = buildPatches(m, g);
        if (!onDevice)
        {
            // THE HOST LOOP COUPLES A CYCLIC: its operators branch on FvPatch::coupled. The device loop
            // does not, and is handed the mesh as it was -- where a cyclic is refused by name.
            attachCyclicCoupling(patches, m, g);
        }
        // ...handed to the host loop mutable as well, for a case whose mesh moves (MutableMesh)
        MutableMesh mutableMesh;
        mutableMesh.m = &m;
        mutableMesh.g = &g;
        mutableMesh.patches = &patches;

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
                           /*fieldsOut=*/nullptr, endTime, /*pressureTaps=*/nullptr, &mutableMesh);

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
